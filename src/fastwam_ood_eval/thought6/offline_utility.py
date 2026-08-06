"""Frozen training-style flow objective and Phase 6B utility statistics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from fastwam_ood_eval.thought6.future_modes import FuturePayload, FusionMode, decide_future_fusion
from fastwam_ood_eval.thought6.gate_decision import MetricInterval, decide_phase6b
from fastwam_ood_eval.thought6.paired_noise import OfflineNoiseIdentity, validate_arm_pairing
from fastwam_ood_eval.thought6.schemas import Thought6Error, object_sha256, tensor_sha256
from fastwam_ood_eval.thought6.sigma_gate import offline_sigma_from_seed
from fastwam_ood_eval.thought6.statistics import BootstrapInterval, hierarchical_bootstrap


OFFLINE_ARMS = (
    "B0_null",
    "F0_correct",
    "F0_shuffle",
    "Fsigma_correct",
    "Fsigma_shuffle",
    "Fsigma_null",
)


@dataclass(frozen=True)
class OfflineObjectiveInput:
    row_identity: OfflineNoiseIdentity
    condition: str
    current_latent: Any
    context: Any
    context_mask: Any
    target_action: Any
    action_is_pad: Any | None
    correct_future: Any
    shuffled_future: Any
    initial_state_sha256: str
    scheduler_sha256: str


def sigma_bucket(value: float) -> str:
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise Thought6Error("sigma bucket input must be finite and in [0,1]")
    if value < 0.25:
        return "[0,0.25)"
    if value < 0.50:
        return "[0.25,0.50)"
    if value < 0.75:
        return "[0.50,0.75)"
    return "[0.75,1.00]"


def _arm_spec(arm: str) -> tuple[FusionMode, FuturePayload | None]:
    return {
        "B0_null": (FusionMode.B0, FuturePayload.NULL),
        "F0_correct": (FusionMode.F0, FuturePayload.CORRECT),
        "F0_shuffle": (FusionMode.F0, FuturePayload.SHUFFLE),
        "Fsigma_correct": (FusionMode.FSIGMA, FuturePayload.CORRECT),
        "Fsigma_shuffle": (FusionMode.FSIGMA, FuturePayload.SHUFFLE),
        "Fsigma_null": (FusionMode.FSIGMA, FuturePayload.NULL),
    }[arm]


def evaluate_offline_objective(
    model: Any,
    adapter: Any,
    sample: OfflineObjectiveInput,
    *,
    loss_lambda_action: float = 1.0,
) -> list[dict[str, Any]]:
    """Evaluate all six paired arms for one held-out objective identity."""

    import torch
    from fastwam_ood_eval.thought3.phase_c_smoke import (
        _action_from_video_cache,
        _prepare_video_cache,
        compute_upstream_action_loss,
    )
    from fastwam_ood_eval.thought6.rollout_policy import SigmaAwareFutureInjector, freeze_for_phase6

    freeze_for_phase6(model, adapter)
    identity = sample.row_identity
    sigma = offline_sigma_from_seed(identity.action_timestep_seed)
    device = str(next(model.parameters()).device)
    dtype = model.torch_dtype
    target = sample.target_action.to(device=device, dtype=dtype)
    if target.ndim == 2:
        target = target.unsqueeze(0)
    generator = torch.Generator(device="cpu").manual_seed(identity.action_noise_seed)
    noise = torch.randn(target.shape, generator=generator, dtype=torch.float32).to(
        device=device, dtype=dtype
    )
    timestep = torch.tensor(
        [sigma["sampled_timestep_bf16"]], device=device, dtype=dtype
    )
    noisy = model.train_action_scheduler.add_noise(target, noise, timestep)
    velocity = model.train_action_scheduler.training_target(target, noise, timestep)
    weight = model.train_action_scheduler.training_weight(timestep)
    current = sample.current_latent.to(device=device, dtype=dtype)
    context = sample.context.to(device=device, dtype=dtype)
    context_mask = sample.context_mask.to(device=device, dtype=torch.bool)
    correct = sample.correct_future.to(device=device, dtype=dtype)
    shuffled = sample.shuffled_future.to(device=device, dtype=dtype)
    future_mask = torch.ones((1, 2, 14, 28), dtype=torch.bool, device=device)
    with torch.inference_mode():
        cache, attention_mask, video_seq_len = _prepare_video_cache(
            model, current, context, context_mask, action_seq_len=target.shape[1]
        )
    rows: list[dict[str, Any]] = []
    with SigmaAwareFutureInjector(model.action_expert.action_encoder, adapter) as injector:
        for arm in OFFLINE_ARMS:
            mode, override = _arm_spec(arm)
            decision = decide_future_fusion(
                mode,
                condition=sample.condition,
                effective_sigma=sigma["effective_sigma_bf16"],
                payload_override=override,
            )
            selected = None
            if decision.payload == "correct":
                selected = correct
            elif decision.payload == "shuffle":
                selected = shuffled
            with torch.inference_mode(), injector.activate_step(
                decision,
                future_latent=selected,
                future_mask=(future_mask if selected is not None else None),
            ) as scope:
                prediction = _action_from_video_cache(
                    model,
                    noisy,
                    timestep,
                    context,
                    context_mask,
                    cache,
                    attention_mask,
                    video_seq_len,
                )
                loss = compute_upstream_action_loss(
                    prediction,
                    velocity,
                    (
                        None
                        if sample.action_is_pad is None
                        else sample.action_is_pad.to(device=device, dtype=torch.bool)
                    ),
                    weight,
                    loss_lambda_action=loss_lambda_action,
                )
            value = float(loss.detach().cpu())
            if not np.isfinite(value):
                raise Thought6Error("offline objective produced NaN/Inf")
            diagnostic = dict(scope.diagnostic or {})
            rows.append(
                {
                    "schema_version": "thought6.phase6b.objective_row.v1",
                    "row_id": identity.row_id,
                    "suite": identity.suite,
                    "task_id": identity.task_id,
                    "task_key": f"{identity.suite}/{identity.task_id}",
                    "episode_id": identity.episode_id,
                    "flow_slot": identity.flow_slot,
                    "flow_slot_is_not_denoising_step": True,
                    "condition": sample.condition.lower(),
                    "arm": arm,
                    "future_mode": decision.payload,
                    "raw_sigma": sigma["raw_sigma_fp32"],
                    "sampled_timestep": sigma["sampled_timestep_bf16"],
                    "effective_sigma_bf16": sigma["effective_sigma_bf16"],
                    "sigma_bucket": sigma_bucket(sigma["effective_sigma_bf16"]),
                    "gate": decision.external_gate,
                    "adapter_called": decision.adapter_called,
                    "adapter_output_rms": diagnostic["adapter_output_rms"],
                    "action_noise_seed": identity.action_noise_seed,
                    "action_timestep_seed": identity.action_timestep_seed,
                    "future_noise_seed": identity.future_noise_seed,
                    "initial_state_sha256": sample.initial_state_sha256,
                    "scheduler_sha256": sample.scheduler_sha256,
                    "action_noise_sha256": tensor_sha256(noise),
                    "action_target_sha256": tensor_sha256(target),
                    "objective_reduction": "official_valid_token_weighted_velocity_MSE",
                    "dtype": str(dtype),
                    "loss": value,
                }
            )
    validate_arm_pairing(rows)
    return rows


def _paired_metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["row_id"]), {})[str(row["arm"])] = row
    output: list[dict[str, Any]] = []
    for row_id, arms in sorted(groups.items()):
        if set(arms) != set(OFFLINE_ARMS):
            raise Thought6Error(f"objective arm set incomplete for {row_id}")
        base = arms["B0_null"]
        losses = {arm: float(value["loss"]) for arm, value in arms.items()}
        output.append(
            {
                "row_id": row_id,
                "suite": base["suite"],
                "task_key": base["task_key"],
                "episode_id": base["episode_id"],
                "condition": base["condition"],
                "sigma_bucket": base["sigma_bucket"],
                "u_f0": losses["B0_null"] - losses["F0_correct"],
                "u_fsigma": losses["Fsigma_null"] - losses["Fsigma_correct"],
                "shuffle_specificity": losses["Fsigma_shuffle"] - losses["Fsigma_correct"],
                "timing_gain": (
                    losses["Fsigma_null"] - losses["Fsigma_correct"]
                    - losses["B0_null"]
                    + losses["F0_correct"]
                ),
                "null_minus_b0": losses["Fsigma_null"] - losses["B0_null"],
                "null_b0_bitwise": (
                    losses["Fsigma_null"] == losses["B0_null"]
                    and arms["Fsigma_null"]["gate"] == 0
                ),
            }
        )
    return output


def _cluster_values(rows: Iterable[Mapping[str, Any]], metric: str) -> dict[str, dict[str, list[float]]]:
    values: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        values.setdefault(str(row["task_key"]), {}).setdefault(
            str(row["episode_id"]), []
        ).append(float(row[metric]))
    return values


def _interval(
    rows: Sequence[Mapping[str, Any]], metric: str, *, replicates: int, seed: int
) -> BootstrapInterval:
    return hierarchical_bootstrap(
        _cluster_values(rows, metric), replicates=replicates, seed=seed
    )


def aggregate_phase6b(
    raw_rows: Sequence[Mapping[str, Any]], *, replicates: int = 10_000, seed: int = 6607
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build frozen aggregate/slice statistics and evaluate the five gates."""

    validate_arm_pairing(raw_rows)
    paired = _paired_metrics(raw_rows)
    metrics = ("u_f0", "u_fsigma", "shuffle_specificity", "timing_gain", "null_minus_b0")
    slices: dict[str, Any] = {}
    filters: dict[str, list[dict[str, Any]]] = {
        "aggregate": paired,
        "condition/clean": [row for row in paired if row["condition"] == "clean"],
        "condition/camera": [row for row in paired if row["condition"] == "camera"],
    }
    for suite in sorted({str(row["suite"]) for row in paired}):
        filters[f"suite/{suite}"] = [row for row in paired if row["suite"] == suite]
    for bucket in ("[0,0.25)", "[0.25,0.50)", "[0.50,0.75)", "[0.75,1.00]"):
        filters[f"sigma/{bucket}"] = [row for row in paired if row["sigma_bucket"] == bucket]
    for name, subset in filters.items():
        task_count = len({str(row["task_key"]) for row in subset})
        if not subset:
            slices[name] = {"status": "empty"}
        elif task_count < 2:
            slices[name] = {
                "status": "descriptive_only_single_task_cluster",
                "count": len(subset),
                "metrics": {
                    metric: {"mean": float(np.mean([row[metric] for row in subset]))}
                    for metric in metrics
                },
            }
        else:
            slices[name] = {
                "status": "complete",
                "count": len(subset),
                "task_count": task_count,
                "metrics": {
                    metric: _interval(
                        subset, metric, replicates=replicates, seed=seed
                    ).to_dict()
                    for metric in metrics
                },
            }
    per_task = {}
    for task in sorted({str(row["task_key"]) for row in paired}):
        subset = [row for row in paired if row["task_key"] == task]
        per_task[task] = {
            metric: float(np.mean([row[metric] for row in subset])) for metric in metrics
        }
    clean = slices["condition/clean"]["metrics"]["u_fsigma"]
    camera = slices["condition/camera"]["metrics"]
    parity = all(bool(row["null_b0_bitwise"]) for row in paired)
    gate = decide_phase6b(
        {
            "fsigma_clean_utility": MetricInterval(clean["mean"], clean["lower"], clean["upper"]),
            "fsigma_camera_utility": MetricInterval(
                camera["u_fsigma"]["mean"], camera["u_fsigma"]["lower"], camera["u_fsigma"]["upper"]
            ),
            "fsigma_shuffle_specificity": MetricInterval(
                camera["shuffle_specificity"]["mean"],
                camera["shuffle_specificity"]["lower"],
                camera["shuffle_specificity"]["upper"],
            ),
            "fsigma_minus_f0_utility": MetricInterval(
                camera["timing_gain"]["mean"], camera["timing_gain"]["lower"], camera["timing_gain"]["upper"]
            ),
            "null_b0_bitwise_parity": parity,
        }
    )
    result = {
        "schema_version": "thought6.phase6b.utility_result.v1",
        "status": "complete",
        "scientific_result": True,
        "objective_row_count": len(raw_rows),
        "paired_objective_count": len(paired),
        "arms": list(OFFLINE_ARMS),
        "statistics": {"method": "task_episode_hierarchical_bootstrap", "replicates": replicates, "seed": seed},
        "slices": slices,
        "per_task": per_task,
        "null_b0_bitwise_parity": parity,
        "raw_rows_sha256": object_sha256(list(raw_rows)),
    }
    result["result_sha256"] = object_sha256(result)
    return result, gate


def load_completed_phase6a(path: str | Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("status") != "complete" or value.get("phase6b_unlocked") is not True:
        raise Thought6Error("Phase 6B remains locked until Phase 6A passes")
    return value
