"""Video-only K-step future latent sampling contracts.

Phase B supplies a deterministic CPU mock velocity model.  Phase C can bind
the same sampler loop to the frozen upstream Video DiT without changing cache
identities or scheduler semantics.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor

from fastwam_ood_eval.thought3.schemas import (
    NATIVE_FUTURE_SHAPE,
    SamplerSchedule,
    build_sampler_schedule,
    sha256_bytes,
)


class FutureSamplingError(RuntimeError):
    """Raised when a video-only sampling contract is violated."""


VelocityModel = Callable[[Tensor, Tensor, Mapping[str, object]], Tensor]


@dataclass(frozen=True)
class FutureSample:
    future_latent: Tensor
    full_state: Tensor
    schedule: SamplerSchedule
    initial_state_sha256: tuple[str, ...]
    latency_ms: float


def tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return sha256_bytes(value.numpy().tobytes())


def make_initial_video_state(
    current_latent: Tensor,
    initial_noise_seeds: Sequence[int],
    *,
    total_latent_frames: int = 3,
    rand_device: str = "cpu",
) -> tuple[Tensor, tuple[str, ...]]:
    """Generate paired float32 CPU noise, then pin slice zero to current.

    Each sample receives an independent generator so rank/batch ordering cannot
    change its noise.  K is intentionally absent from this function.
    """

    if current_latent.ndim != 5 or current_latent.shape[2] != 1:
        raise FutureSamplingError(
            "current_latent must have shape [B,C,1,H,W]"
        )
    if not current_latent.is_floating_point():
        raise FutureSamplingError("current_latent must be floating point")
    if total_latent_frames <= 1:
        raise FutureSamplingError("total_latent_frames must include current and future")
    if rand_device != "cpu":
        raise FutureSamplingError("first protocol requires rand_device='cpu'")
    batch, channels, _, height, width = current_latent.shape
    if len(initial_noise_seeds) != batch:
        raise FutureSamplingError(
            "one initial noise seed is required for every batch sample"
        )
    states: list[Tensor] = []
    hashes: list[str] = []
    for seed in initial_noise_seeds:
        if int(seed) < 0:
            raise FutureSamplingError("initial noise seeds must be non-negative")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        noise = torch.randn(
            channels,
            total_latent_frames,
            height,
            width,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
        states.append(noise)
        hashes.append(tensor_sha256(noise))
    state = torch.stack(states).to(
        device=current_latent.device,
        dtype=current_latent.dtype,
    )
    state = torch.cat((current_latent, state[:, :, 1:]), dim=2)
    return state, tuple(hashes)


class VideoOnlyFutureSampler:
    """Integrate a Video DiT velocity callback over the complete K schedule."""

    def __init__(
        self,
        velocity_model: VelocityModel,
        *,
        shift: float = 5.0,
        num_train_timesteps: int = 1000,
        rand_device: str = "cpu",
    ) -> None:
        self.velocity_model = velocity_model
        self.shift = float(shift)
        self.num_train_timesteps = int(num_train_timesteps)
        self.rand_device = rand_device
        signature = inspect.signature(velocity_model)
        forbidden = {"action", "target_action", "future_frames", "success"}
        overlap = forbidden & set(signature.parameters)
        if overlap:
            raise FutureSamplingError(
                f"video-only velocity callback exposes forbidden inputs: {sorted(overlap)}"
            )

    def sample(
        self,
        current_latent: Tensor,
        *,
        initial_noise_seeds: Sequence[int],
        k: int,
        conditions: Mapping[str, object] | None = None,
    ) -> FutureSample:
        started = time.perf_counter()
        schedule = build_sampler_schedule(
            k,
            shift=self.shift,
            num_train_timesteps=self.num_train_timesteps,
        )
        state, initial_hashes = make_initial_video_state(
            current_latent,
            initial_noise_seeds,
            rand_device=self.rand_device,
        )
        fixed_current = current_latent
        condition_values = dict(conditions or {})
        batch = current_latent.shape[0]
        for timestep, delta in zip(schedule.timesteps, schedule.deltas):
            timestep_tensor = torch.full(
                (batch,),
                float(timestep),
                dtype=torch.float32,
                device=state.device,
            )
            velocity = self.velocity_model(
                state,
                timestep_tensor,
                condition_values,
            )
            if not isinstance(velocity, Tensor) or velocity.shape != state.shape:
                raise FutureSamplingError(
                    "video velocity callback must return a Tensor matching the state"
                )
            if not torch.isfinite(velocity).all():
                raise FutureSamplingError("video velocity callback produced NaN/Inf")
            state = state + float(delta) * velocity.to(
                device=state.device,
                dtype=state.dtype,
            )
            # Current observation is a hard condition, never a generated frame.
            state = torch.cat((fixed_current, state[:, :, 1:]), dim=2)
        future = state[:, :, 1:]
        expected = (
            current_latent.shape[0],
            NATIVE_FUTURE_SHAPE[0],
            *NATIVE_FUTURE_SHAPE[1:],
        )
        if (
            current_latent.shape[1:] == (48, 1, 14, 28)
            and tuple(future.shape) != expected
        ):
            raise FutureSamplingError(
                f"native future output must be {expected}, got {tuple(future.shape)}"
            )
        return FutureSample(
            future_latent=future,
            full_state=state,
            schedule=schedule,
            initial_state_sha256=initial_hashes,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )


class MockVideoVelocity:
    """Deterministic nonlinear velocity used only by CPU Phase B tests."""

    def __call__(
        self,
        state: Tensor,
        timestep: Tensor,
        conditions: Mapping[str, object],
    ) -> Tensor:
        del timestep
        current = state[:, :, 0:1]
        future_frames = state.shape[2] - 1
        ramp = torch.linspace(
            0.05,
            0.10,
            future_frames,
            dtype=state.dtype,
            device=state.device,
        ).view(1, 1, future_frames, 1, 1)
        condition_bias = float(conditions.get("mock_condition_bias", 0.0))
        target_future = current.expand(-1, -1, future_frames, -1, -1) + ramp
        target_future = target_future + condition_bias
        target = torch.cat((current, target_future), dim=2)
        # Non-linearity deliberately makes the K=1/2/4 approximations differ.
        return torch.tanh(state - target) + 0.05 * (state - target)


def make_mock_future_sampler() -> VideoOnlyFutureSampler:
    return VideoOnlyFutureSampler(MockVideoVelocity())
