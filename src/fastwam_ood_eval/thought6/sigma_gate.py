"""One frozen sigma gate shared by offline objectives and online inference."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from fastwam_ood_eval.thought6 import ACTION_DENOISE_STEPS, SIGMA_THRESHOLD
from fastwam_ood_eval.thought6.schemas import Thought6Error, object_sha256


@dataclass(frozen=True)
class InferenceSigmaStep:
    step_index: int
    raw_sigma_fp32: float
    scheduler_timestep_bf16: float
    effective_sigma: float
    delta_bf16: float
    gate: int


def shifted_sigma(raw_uniform: float, *, shift: float = 5.0) -> float:
    if not math.isfinite(raw_uniform) or not 0.0 <= raw_uniform <= 1.0:
        raise Thought6Error("raw flow uniform must be finite and in [0,1]")
    if not math.isfinite(shift) or shift <= 0:
        raise Thought6Error("flow shift must be finite and positive")
    return shift * raw_uniform / (1.0 + (shift - 1.0) * raw_uniform)


def sigma_gate(effective_sigma: float) -> int:
    if not math.isfinite(effective_sigma) or not 0.0 <= effective_sigma <= 1.0:
        raise Thought6Error("effective sigma must be finite and in [0,1]")
    return int(effective_sigma >= SIGMA_THRESHOLD)


def offline_sigma_from_seed(
    action_timestep_seed: int,
    *,
    shift: float = 5.0,
    num_train_timesteps: int = 1000,
) -> dict[str, Any]:
    """Reproduce the exact CPU float32 -> BF16 training-style sigma path."""

    import torch

    if isinstance(action_timestep_seed, bool) or action_timestep_seed < 0:
        raise Thought6Error("action timestep seed must be a non-negative integer")
    if num_train_timesteps != 1000 or shift != 5.0:
        raise Thought6Error("Phase 6 freezes shift=5 and 1000 training timesteps")
    generator = torch.Generator(device="cpu").manual_seed(int(action_timestep_seed))
    raw_uniform = torch.rand((1,), generator=generator, dtype=torch.float32)
    raw_sigma = 5.0 * raw_uniform / (1.0 + 4.0 * raw_uniform)
    raw_timestep = raw_sigma * 1000.0
    timestep_bf16 = raw_timestep.to(dtype=torch.bfloat16)
    effective_bf16 = timestep_bf16 / 1000.0
    result = {
        "sampling_process": "training_style_flow_objective",
        "action_timestep_seed": int(action_timestep_seed),
        "raw_uniform_fp32": float(raw_uniform.item()),
        "raw_sigma_fp32": float(raw_sigma.item()),
        "raw_timestep_fp32": float(raw_timestep.item()),
        "sampled_timestep_bf16": float(timestep_bf16.float().item()),
        "effective_sigma_bf16": float(effective_bf16.float().item()),
    }
    result["gate"] = sigma_gate(result["effective_sigma_bf16"])
    return result


def build_inference_sigma_schedule(
    *,
    num_inference_steps: int = ACTION_DENOISE_STEPS,
    shift: float = 5.0,
    num_train_timesteps: int = 1000,
) -> tuple[InferenceSigmaStep, ...]:
    """Reproduce the pinned scheduler, including its actual BF16 values."""

    import torch

    if num_inference_steps != ACTION_DENOISE_STEPS:
        raise Thought6Error("Phase 6 freezes exactly 20 action denoising steps")
    if shift != 5.0 or num_train_timesteps != 1000:
        raise Thought6Error("Phase 6 freezes shift=5 and 1000 timesteps")
    raw_u = torch.linspace(1.0, 0.0, num_inference_steps + 1, dtype=torch.float32)
    raw_sigma = 5.0 * raw_u / (1.0 + 4.0 * raw_u)
    timesteps = (raw_sigma[:-1] * 1000.0).to(dtype=torch.bfloat16)
    deltas = (raw_sigma[1:] - raw_sigma[:-1]).to(dtype=torch.bfloat16)
    rows = []
    for index in range(num_inference_steps):
        timestep = float(timesteps[index].float().item())
        # The scheduler exposes BF16 timesteps; FastWAM turns them into sigma
        # through Python's num_train_timesteps scalar. Preserve the observed
        # timestep exactly and report the effective value as t / 1000.
        effective = timestep / float(num_train_timesteps)
        rows.append(
            InferenceSigmaStep(
                step_index=index,
                raw_sigma_fp32=float(raw_sigma[index].item()),
                scheduler_timestep_bf16=timestep,
                effective_sigma=effective,
                delta_bf16=float(deltas[index].float().item()),
                gate=sigma_gate(effective),
            )
        )
    return tuple(rows)


def validate_runtime_schedule(
    timesteps: Sequence[Any], deltas: Sequence[Any]
) -> tuple[InferenceSigmaStep, ...]:
    """Read actual scheduler tensors and fail if they differ from the frozen math."""

    if len(timesteps) != ACTION_DENOISE_STEPS or len(deltas) != ACTION_DENOISE_STEPS:
        raise Thought6Error("runtime scheduler must expose exactly 20 steps")
    expected = build_inference_sigma_schedule()
    observed: list[InferenceSigmaStep] = []
    for index, (timestep_value, delta_value) in enumerate(zip(timesteps, deltas)):
        timestep = float(timestep_value.detach().float().cpu().reshape(()).item())
        delta = float(delta_value.detach().float().cpu().reshape(()).item())
        effective = timestep / 1000.0
        row = InferenceSigmaStep(
            step_index=index,
            raw_sigma_fp32=expected[index].raw_sigma_fp32,
            scheduler_timestep_bf16=timestep,
            effective_sigma=effective,
            delta_bf16=delta,
            gate=sigma_gate(effective),
        )
        if (
            row.scheduler_timestep_bf16 != expected[index].scheduler_timestep_bf16
            or row.delta_bf16 != expected[index].delta_bf16
            or row.gate != expected[index].gate
        ):
            raise Thought6Error(f"runtime scheduler differs at step {index}")
        observed.append(row)
    return tuple(observed)


def schedule_manifest(rows: Sequence[InferenceSigmaStep]) -> dict[str, Any]:
    payload = {
        "schema_version": "thought6.inference_sigma_schedule.v1",
        "sigma_threshold": SIGMA_THRESHOLD,
        "gate_source": "actual_scheduler_sigma_not_step_index",
        "steps": [asdict(row) for row in rows],
    }
    payload["schedule_sha256"] = object_sha256(payload)
    return payload
