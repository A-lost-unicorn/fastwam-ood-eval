"""Online-only future conditioning boundary used by Phase B mock evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn

from fastwam_ood_eval.thought3.future_sampler import FutureSample
from fastwam_ood_eval.thought3.latency import PolicyLatency
from fastwam_ood_eval.thought3.model_wrapper import AdapterConditionedModel


class OnlineEvaluationError(RuntimeError):
    """Raised when online evaluation would cross the cache boundary."""


class OnlineSampler(Protocol):
    def sample(
        self,
        current_latent: Tensor,
        *,
        initial_noise_seeds: list[int],
        k: int,
        conditions: dict[str, object] | None = None,
    ) -> FutureSample: ...


@dataclass(frozen=True)
class OnlineActionResult:
    action_chunk: Tensor
    future_latent: Tensor | None
    latency: PolicyLatency
    k: int


class OnlineFutureActionEvaluator:
    """Generate future from the current observation and immediately act.

    There is intentionally no cache path/reader argument.  Offline cache is a
    training-only type and cannot be injected into this runtime boundary.
    """

    def __init__(
        self,
        *,
        backbone: nn.Module,
        conditioned_model: AdapterConditionedModel | None,
        sampler: OnlineSampler | None,
        action_denoise_steps: int = 20,
    ) -> None:
        if action_denoise_steps != 20:
            raise OnlineEvaluationError(
                "Thought3 comparisons freeze action denoising at 20 steps"
            )
        if conditioned_model is None and sampler is not None:
            raise OnlineEvaluationError(
                "B0 cannot generate an unused online future"
            )
        self.backbone = backbone
        self.conditioned_model = conditioned_model
        self.sampler = sampler
        self.action_denoise_steps = action_denoise_steps

    def predict(
        self,
        current_latent: Tensor,
        *,
        initial_noise_seed: int,
        action_noise_seed: int,
        k: int,
        null_future: bool = False,
        shuffled_donor_current: Tensor | None = None,
    ) -> OnlineActionResult:
        total_started = time.perf_counter()
        preprocessing_started = time.perf_counter()
        if (
            current_latent.ndim != 5
            or current_latent.shape[2] != 1
            or current_latent.shape[0] != 1
        ):
            raise OnlineEvaluationError(
                "online mock current latent must be [1,C,1,H,W]"
            )
        current_latent = current_latent.float().contiguous()
        preprocessing_ms = (time.perf_counter() - preprocessing_started) * 1000

        current_started = time.perf_counter()
        # Phase C replaces this identity with the frozen current-frame VAE/DiT
        # path and keeps the same timing boundary.
        current_representation = current_latent[:, :, :1].clone()
        current_ms = (time.perf_counter() - current_started) * 1000

        future_started = time.perf_counter()
        future: Tensor | None
        if self.conditioned_model is None:
            if k != 0 or null_future:
                raise OnlineEvaluationError("B0 requires K=0 and no future")
            future = None
        elif null_future:
            if k != 0:
                raise OnlineEvaluationError("A0 null future requires K=0")
            future = torch.zeros(
                current_latent.shape[0],
                current_latent.shape[1],
                2,
                current_latent.shape[3],
                current_latent.shape[4],
                dtype=current_latent.dtype,
                device=current_latent.device,
            )
        else:
            if k not in {1, 2, 4} or self.sampler is None:
                raise OnlineEvaluationError(
                    "A1/A2/A4/A-shuffle require online K=1/2/4 sampler"
                )
            source_current = (
                shuffled_donor_current
                if shuffled_donor_current is not None
                else current_representation
            )
            if source_current.shape != current_representation.shape:
                raise OnlineEvaluationError(
                    "shuffled donor current shape differs from recipient"
                )
            future = self.sampler.sample(
                source_current,
                initial_noise_seeds=[initial_noise_seed],
                k=k,
                conditions=None,
            ).future_latent
        future_ms = (time.perf_counter() - future_started) * 1000

        generator = torch.Generator(device="cpu")
        generator.manual_seed(action_noise_seed)
        action = torch.randn(
            (1, 8, 7),
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        ).to(current_latent.device)
        adapter_elapsed = 0.0
        adapter_started: list[float] = []
        handles: list[torch.utils.hooks.RemovableHandle] = []
        if self.conditioned_model is not None:
            def adapter_pre_hook(module: nn.Module, inputs: tuple[object, ...]) -> None:
                del module, inputs
                adapter_started.append(time.perf_counter())

            def adapter_post_hook(
                module: nn.Module,
                inputs: tuple[object, ...],
                output: object,
            ) -> None:
                nonlocal adapter_elapsed
                del module, inputs, output
                if not adapter_started:
                    raise OnlineEvaluationError("Adapter timing hook is unbalanced")
                adapter_elapsed += time.perf_counter() - adapter_started.pop()

            handles = [
                self.conditioned_model.adapter.register_forward_pre_hook(
                    adapter_pre_hook
                ),
                self.conditioned_model.adapter.register_forward_hook(
                    adapter_post_hook
                ),
            ]

        action_started = time.perf_counter()
        try:
            for _ in range(self.action_denoise_steps):
                if self.conditioned_model is None:
                    velocity = self.backbone(action)
                else:
                    assert future is not None
                    velocity = self.conditioned_model(
                        action,
                        future_latent=future,
                    )
                action = action - velocity * (1.0 / self.action_denoise_steps)
        finally:
            for handle in handles:
                handle.remove()
        action_ms = (time.perf_counter() - action_started) * 1000
        adapter_ms = adapter_elapsed * 1000.0
        action_without_adapter_ms = max(0.0, action_ms - adapter_ms)
        total_ms = (time.perf_counter() - total_started) * 1000
        return OnlineActionResult(
            action_chunk=action,
            future_latent=future,
            latency=PolicyLatency(
                preprocessing_ms=preprocessing_ms,
                current_state_encoding_ms=current_ms,
                future_sampling_ms=future_ms,
                adapter_ms=adapter_ms,
                action_denoising_ms=action_without_adapter_ms,
                total_policy_ms=total_ms,
                future_decoded_to_video=False,
            ),
            k=k,
        )
