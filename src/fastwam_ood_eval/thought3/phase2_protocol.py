"""Pure protocol/schema helpers for Thought3 Phase 2 full 28/4 training.

This module intentionally does not import torch.  It is safe to import from a
CLI ``--dry-run`` and freezes the one permitted post-Phase-1 recipe:

* 28 training and 4 development samples from the Phase D cache;
* matched A0/A1 tracks only;
* one LR, structure, seed, sample order, and action-flow schedule;
* inverse-initial-loss, unit-mean sample weights;
* a fixed step-200 endpoint with no checkpoint selection.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from fastwam_ood_eval.thought3.schemas import canonical_json


PHASE2_CONFIG_SCHEMA = "thought3.phase2.full_28_4.config.v1"
PHASE2_CALIBRATION_SCHEMA = "thought3.phase2.full_28_4.calibration.v1"
PHASE2_TRACK_SCHEMA = "thought3.phase2.full_28_4.track.v1"
PHASE2_RESULT_SCHEMA = "thought3.phase2.full_28_4.result.v1"
PHASE2_ARTIFACT_SCHEMA = "thought3.phase2.full_28_4.artifacts.v1"
PHASE2_VARIANTS = ("A0", "A1")
PHASE2_CONFIRMATION_ENV = "CONFIRM_THOUGHT3_PHASE2_FULL"


class Phase2ProtocolError(RuntimeError):
    """Raised when the frozen Phase 2 protocol is changed or violated."""


@dataclass(frozen=True)
class Phase2FullTrainingConfig:
    """Strict, standalone Phase 2 configuration."""

    source_path: Path
    raw: Mapping[str, Any]
    experiment_name: str
    output_dir: Path
    experiment_seed: int
    thought3_base_config_path: Path
    thought3_base_config_sha256: str
    phase_d_gate_path: Path
    phase_d_gate_sha256: str
    phase1_aggregate_path: Path
    phase1_aggregate_sha256: str
    phase1_artifact_manifest_path: Path
    phase1_artifact_manifest_sha256: str
    e9_audit_path: Path
    e9_audit_sha256: str
    cache_fingerprint: str
    split_fingerprint: str
    expected_phase1_classification: str
    variants: tuple[str, ...]
    train_count: int
    development_count: int
    optimizer: str
    optimizer_updates: int
    objectives_per_update: int
    learning_rate: float
    weight_decay: float
    train_seed: int
    checkpoint_interval: int
    training_flow_slot_offset: int
    calibration_flow_steps: tuple[int, ...]
    development_flow_steps: tuple[int, ...]
    sample_weight_recipe: str
    calibration_variant: str
    primary_checkpoint_step: int
    primary_checkpoint_rule: str
    development_direction_rule: str
    catastrophic_loss_multiplier: float
    recipe_selection_disclosure: str
    device: str
    visible_gpu_count: int
    parallel_track_count: int
    max_gpu_memory_gb: float
    deterministic_algorithms: bool
    cublas_workspace_config: str
    scope: Mapping[str, bool]

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            canonical_json(dict(self.raw)).encode("utf-8")
        ).hexdigest()

    @property
    def training_flow_start(self) -> int:
        return self.training_flow_slot_offset + 1

    @property
    def training_flow_end(self) -> int:
        return (
            self.training_flow_slot_offset
            + self.optimizer_updates * self.objectives_per_update
        )

    @property
    def track_root(self) -> Path:
        return self.output_dir / "tracks"

    def track_output_dir(self, variant: str) -> Path:
        if variant not in PHASE2_VARIANTS:
            raise Phase2ProtocolError(f"unsupported Phase 2 variant: {variant}")
        return self.track_root / variant.lower()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Phase2ProtocolError(f"{name} must be a mapping")
    return value


def _strict_keys(
    value: Mapping[str, Any],
    expected: set[str],
    name: str,
) -> None:
    changed = set(value) ^ expected
    if changed:
        raise Phase2ProtocolError(
            f"{name} keys changed: {sorted(changed)}"
        )


def _sha256(value: object, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef"
        for character in normalized
    ):
        raise Phase2ProtocolError(
            f"{name} must be a 64-character SHA-256"
        )
    return normalized


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise Phase2ProtocolError(f"{name} must be a positive integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise Phase2ProtocolError(
            f"{name} must be a positive integer"
        ) from exc
    if normalized <= 0:
        raise Phase2ProtocolError(f"{name} must be a positive integer")
    return normalized


def _positive_unique_ints(
    value: object,
    name: str,
) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise Phase2ProtocolError(
            f"{name} must be a nonempty ordered integer list"
        )
    normalized = tuple(
        _positive_int(item, f"{name}[]") for item in value
    )
    if len(set(normalized)) != len(normalized):
        raise Phase2ProtocolError(f"{name} must contain unique values")
    return normalized


def load_phase2_full_training_config(
    path: str | Path,
) -> Phase2FullTrainingConfig:
    """Load the strict Phase 2 preregistration without importing torch."""

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    raw_value = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    root = _mapping(raw_value, "Phase 2 config")
    if root.get("schema_version") != PHASE2_CONFIG_SCHEMA:
        raise Phase2ProtocolError(
            f"schema_version must be {PHASE2_CONFIG_SCHEMA}"
        )
    _strict_keys(
        root,
        {
            "schema_version",
            "experiment",
            "source",
            "recipe",
            "runtime",
            "scope",
        },
        "config",
    )
    experiment = _mapping(root["experiment"], "experiment")
    source = _mapping(root["source"], "source")
    recipe = _mapping(root["recipe"], "recipe")
    runtime = _mapping(root["runtime"], "runtime")
    scope = _mapping(root["scope"], "scope")
    _strict_keys(
        experiment,
        {"name", "output_dir", "seed"},
        "experiment",
    )
    _strict_keys(
        source,
        {
            "thought3_base_config_path",
            "thought3_base_config_sha256",
            "phase_d_gate_path",
            "phase_d_gate_sha256",
            "phase1_aggregate_path",
            "phase1_aggregate_sha256",
            "phase1_artifact_manifest_path",
            "phase1_artifact_manifest_sha256",
            "e9_audit_path",
            "e9_audit_sha256",
            "cache_fingerprint",
            "split_fingerprint",
            "expected_phase1_classification",
        },
        "source",
    )
    _strict_keys(
        recipe,
        {
            "variants",
            "train_count",
            "development_count",
            "optimizer",
            "optimizer_updates",
            "objectives_per_update",
            "learning_rate",
            "weight_decay",
            "train_seed",
            "checkpoint_interval",
            "training_flow_slot_offset",
            "calibration_flow_steps",
            "development_flow_steps",
            "sample_weight_recipe",
            "calibration_variant",
            "primary_checkpoint_step",
            "primary_checkpoint_rule",
            "development_direction_rule",
            "catastrophic_loss_multiplier",
            "recipe_selection_disclosure",
        },
        "recipe",
    )
    _strict_keys(
        runtime,
        {
            "device",
            "visible_gpu_count",
            "parallel_track_count",
            "max_gpu_memory_gb",
            "deterministic_algorithms",
            "cublas_workspace_config",
        },
        "runtime",
    )
    expected_scope_keys = {
        "read_training_action_target",
        "read_training_future_cache",
        "read_development",
        "read_future_rgb",
        "read_ood",
        "read_rollout_success",
        "start_rollout",
        "train_a2_or_a4",
        "select_checkpoint_from_development",
        "tune_from_phase2_outcome",
    }
    _strict_keys(scope, expected_scope_keys, "scope")

    variants = tuple(str(value) for value in recipe["variants"])
    calibration_flows = _positive_unique_ints(
        recipe["calibration_flow_steps"],
        "recipe.calibration_flow_steps",
    )
    development_flows = _positive_unique_ints(
        recipe["development_flow_steps"],
        "recipe.development_flow_steps",
    )
    cfg = Phase2FullTrainingConfig(
        source_path=source_path.resolve(),
        raw=root,
        experiment_name=str(experiment["name"]),
        output_dir=Path(str(experiment["output_dir"])),
        experiment_seed=int(experiment["seed"]),
        thought3_base_config_path=Path(
            str(source["thought3_base_config_path"])
        ),
        thought3_base_config_sha256=_sha256(
            source["thought3_base_config_sha256"],
            "source.thought3_base_config_sha256",
        ),
        phase_d_gate_path=Path(str(source["phase_d_gate_path"])),
        phase_d_gate_sha256=_sha256(
            source["phase_d_gate_sha256"],
            "source.phase_d_gate_sha256",
        ),
        phase1_aggregate_path=Path(
            str(source["phase1_aggregate_path"])
        ),
        phase1_aggregate_sha256=_sha256(
            source["phase1_aggregate_sha256"],
            "source.phase1_aggregate_sha256",
        ),
        phase1_artifact_manifest_path=Path(
            str(source["phase1_artifact_manifest_path"])
        ),
        phase1_artifact_manifest_sha256=_sha256(
            source["phase1_artifact_manifest_sha256"],
            "source.phase1_artifact_manifest_sha256",
        ),
        e9_audit_path=Path(str(source["e9_audit_path"])),
        e9_audit_sha256=_sha256(
            source["e9_audit_sha256"],
            "source.e9_audit_sha256",
        ),
        cache_fingerprint=_sha256(
            source["cache_fingerprint"],
            "source.cache_fingerprint",
        ),
        split_fingerprint=_sha256(
            source["split_fingerprint"],
            "source.split_fingerprint",
        ),
        expected_phase1_classification=str(
            source["expected_phase1_classification"]
        ),
        variants=variants,
        train_count=_positive_int(
            recipe["train_count"], "recipe.train_count"
        ),
        development_count=_positive_int(
            recipe["development_count"],
            "recipe.development_count",
        ),
        optimizer=str(recipe["optimizer"]),
        optimizer_updates=_positive_int(
            recipe["optimizer_updates"],
            "recipe.optimizer_updates",
        ),
        objectives_per_update=_positive_int(
            recipe["objectives_per_update"],
            "recipe.objectives_per_update",
        ),
        learning_rate=float(recipe["learning_rate"]),
        weight_decay=float(recipe["weight_decay"]),
        train_seed=int(recipe["train_seed"]),
        checkpoint_interval=_positive_int(
            recipe["checkpoint_interval"],
            "recipe.checkpoint_interval",
        ),
        training_flow_slot_offset=_positive_int(
            recipe["training_flow_slot_offset"],
            "recipe.training_flow_slot_offset",
        ),
        calibration_flow_steps=calibration_flows,
        development_flow_steps=development_flows,
        sample_weight_recipe=str(recipe["sample_weight_recipe"]),
        calibration_variant=str(recipe["calibration_variant"]),
        primary_checkpoint_step=_positive_int(
            recipe["primary_checkpoint_step"],
            "recipe.primary_checkpoint_step",
        ),
        primary_checkpoint_rule=str(
            recipe["primary_checkpoint_rule"]
        ),
        development_direction_rule=str(
            recipe["development_direction_rule"]
        ),
        catastrophic_loss_multiplier=float(
            recipe["catastrophic_loss_multiplier"]
        ),
        recipe_selection_disclosure=str(
            recipe["recipe_selection_disclosure"]
        ),
        device=str(runtime["device"]),
        visible_gpu_count=_positive_int(
            runtime["visible_gpu_count"],
            "runtime.visible_gpu_count",
        ),
        parallel_track_count=_positive_int(
            runtime["parallel_track_count"],
            "runtime.parallel_track_count",
        ),
        max_gpu_memory_gb=float(runtime["max_gpu_memory_gb"]),
        deterministic_algorithms=bool(
            runtime["deterministic_algorithms"]
        ),
        cublas_workspace_config=str(
            runtime["cublas_workspace_config"]
        ),
        scope={str(key): bool(value) for key, value in scope.items()},
    )
    expected_scope = {
        "read_training_action_target": True,
        "read_training_future_cache": True,
        "read_development": True,
        "read_future_rgb": False,
        "read_ood": False,
        "read_rollout_success": False,
        "start_rollout": False,
        "train_a2_or_a4": False,
        "select_checkpoint_from_development": False,
        "tune_from_phase2_outcome": False,
    }
    training_flow_set = set(
        range(cfg.training_flow_start, cfg.training_flow_end + 1)
    )
    if (
        cfg.experiment_name != "thought3_phase2_full_28_4_a0_a1_v1"
        or cfg.output_dir
        != Path("outputs/thought3/phase2_full_28_4_a0_a1_v1")
        or cfg.experiment_seed != 3407
        or cfg.expected_phase1_classification
        != "future_content_sensitivity_observed"
        or cfg.variants != PHASE2_VARIANTS
        or cfg.train_count != 28
        or cfg.development_count != 4
        or cfg.optimizer != "AdamW"
        or cfg.optimizer_updates != 200
        or cfg.objectives_per_update != cfg.train_count
        or cfg.learning_rate != 3e-4
        or cfg.weight_decay != 1e-2
        or cfg.train_seed != 3407
        or cfg.checkpoint_interval != 50
        or cfg.primary_checkpoint_step != cfg.optimizer_updates
        or cfg.optimizer_updates % cfg.checkpoint_interval
        or cfg.sample_weight_recipe
        != "inverse_initial_loss_unit_mean_v1"
        or cfg.calibration_variant != "A0"
        or cfg.primary_checkpoint_rule
        != "fixed_step_200_no_selection_no_fallback"
        or cfg.development_direction_rule
        != "a1_final_mean_loss_lt_a0_final_mean_loss_and_a1_reduction_positive"
        or cfg.catastrophic_loss_multiplier != 10.0
        or cfg.calibration_flow_steps != tuple(range(139, 171))
        or cfg.development_flow_steps != tuple(range(171, 203))
        or set(cfg.calibration_flow_steps)
        & set(cfg.development_flow_steps)
        or set(cfg.calibration_flow_steps) & training_flow_set
        or set(cfg.development_flow_steps) & training_flow_set
        or set(range(75, 139))
        & (
            set(cfg.calibration_flow_steps)
            | set(cfg.development_flow_steps)
            | training_flow_set
        )
        or cfg.device != "cuda:0"
        or cfg.visible_gpu_count != 1
        or cfg.parallel_track_count != 2
        or cfg.max_gpu_memory_gb != 23.8
        or cfg.deterministic_algorithms is not True
        or cfg.cublas_workspace_config != ":4096:8"
        or cfg.scope != expected_scope
        or cfg.learning_rate <= 0
        or cfg.weight_decay < 0
        or cfg.train_seed < 0
        or not math.isfinite(cfg.catastrophic_loss_multiplier)
        or cfg.catastrophic_loss_multiplier <= 1
        or not cfg.recipe_selection_disclosure.strip()
    ):
        raise Phase2ProtocolError(
            "Phase 2 frozen 28/4 single-recipe protocol changed"
        )
    return cfg


def _stable_seed(*values: object) -> int:
    digest = hashlib.sha256(
        "\0".join(str(value) for value in values).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def phase2_flow_objective_identity(
    *,
    base_sample_id: str,
    train_seed: int,
    flow_step: int,
) -> dict[str, Any]:
    """Pure reproduction of the frozen real-training action-flow identity."""

    if (
        not base_sample_id
        or isinstance(train_seed, bool)
        or not isinstance(train_seed, int)
        or train_seed < 0
        or isinstance(flow_step, bool)
        or not isinstance(flow_step, int)
        or flow_step < 1
    ):
        raise Phase2ProtocolError("invalid action-flow identity input")
    noise_seed = _stable_seed(
        "thought3-real-action-noise-v1",
        train_seed,
        flow_step,
        base_sample_id,
    )
    timestep_seed = _stable_seed(
        "thought3-real-action-time-v1",
        train_seed,
        flow_step,
        base_sample_id,
    )
    digest = hashlib.sha256(
        (
            f"{base_sample_id}\0{train_seed}\0{flow_step}\0"
            f"{noise_seed}\0{timestep_seed}"
        ).encode("utf-8")
    ).hexdigest()
    return {
        "action_noise_seed": noise_seed,
        "action_timestep_seed": timestep_seed,
        "flow_objective_sha256": digest,
        "flow_step": flow_step,
    }


def phase2_training_flow_slot(
    optimizer_update: int,
    micro_index: int,
    *,
    optimizer_updates: int = 200,
    objectives_per_update: int = 28,
    flow_slot_offset: int = 50_000,
) -> int:
    """Map one matched Phase 2 update/sample position to a unique flow slot."""

    values = (
        optimizer_update,
        micro_index,
        optimizer_updates,
        objectives_per_update,
        flow_slot_offset,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise Phase2ProtocolError("Phase 2 flow-slot inputs must be integers")
    if (
        not 1 <= optimizer_update <= optimizer_updates
        or not 1 <= micro_index <= objectives_per_update
        or optimizer_updates < 1
        or objectives_per_update < 1
        or flow_slot_offset < 1
    ):
        raise Phase2ProtocolError("Phase 2 flow-slot input is out of range")
    return (
        flow_slot_offset
        + (optimizer_update - 1) * objectives_per_update
        + micro_index
    )


def phase2_identity_schedule_sha256(
    sample_ids: Sequence[str],
    *,
    train_seed: int,
    optimizer_updates: int = 200,
    flow_slot_offset: int = 50_000,
) -> str:
    """Hash the full sample/noise/timestep schedule before model execution."""

    normalized = tuple(str(value) for value in sample_ids)
    if not normalized or len(set(normalized)) != len(normalized):
        raise Phase2ProtocolError(
            "Phase 2 schedule requires unique ordered sample IDs"
        )
    payload: list[str] = []
    for update in range(1, optimizer_updates + 1):
        for micro, sample_id in enumerate(normalized, start=1):
            slot = phase2_training_flow_slot(
                update,
                micro,
                optimizer_updates=optimizer_updates,
                objectives_per_update=len(normalized),
                flow_slot_offset=flow_slot_offset,
            )
            identity = phase2_flow_objective_identity(
                base_sample_id=sample_id,
                train_seed=train_seed,
                flow_step=slot,
            )
            payload.append(
                "\0".join(
                    (
                        str(update),
                        str(micro),
                        sample_id,
                        str(slot),
                        str(identity["action_noise_seed"]),
                        str(identity["action_timestep_seed"]),
                        str(identity["flow_objective_sha256"]),
                    )
                )
            )
    return hashlib.sha256("\n".join(payload).encode("utf-8")).hexdigest()


def phase2_sample_loss_weights_sha256(
    sample_ids: Sequence[str],
    weights: Mapping[str, float],
) -> str:
    """Hash one ordered, positive, unit-mean weight vector of any size."""

    normalized = tuple(str(value) for value in sample_ids)
    if (
        not normalized
        or len(set(normalized)) != len(normalized)
        or set(weights) != set(normalized)
    ):
        raise Phase2ProtocolError(
            "sample weights must match the unique ordered cohort"
        )
    values = tuple(float(weights[sample_id]) for sample_id in normalized)
    if (
        any(not math.isfinite(value) or value <= 0 for value in values)
        or not math.isclose(
            sum(values),
            float(len(values)),
            rel_tol=0.0,
            abs_tol=1e-10,
        )
    ):
        raise Phase2ProtocolError(
            "sample weights must be finite, positive, and unit mean"
        )
    payload = "\n".join(
        "\0".join((sample_id, repr(weight)))
        for sample_id, weight in zip(normalized, values)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def inverse_initial_loss_unit_mean_weights(
    sample_ids: Sequence[str],
    initial_losses: Mapping[str, float],
) -> tuple[dict[str, float], str]:
    """Apply the preregistered E9-derived normalization to a full cohort."""

    normalized = tuple(str(value) for value in sample_ids)
    if (
        not normalized
        or len(set(normalized)) != len(normalized)
        or set(initial_losses) != set(normalized)
    ):
        raise Phase2ProtocolError(
            "initial losses must match the ordered calibration cohort"
        )
    losses = tuple(float(initial_losses[sample_id]) for sample_id in normalized)
    if any(not math.isfinite(value) or value <= 0 for value in losses):
        raise Phase2ProtocolError(
            "initial calibration losses must be finite and positive"
        )
    inverse = tuple(1.0 / value for value in losses)
    scale = float(len(inverse)) / sum(inverse)
    weights = {
        sample_id: value * scale
        for sample_id, value in zip(normalized, inverse)
    }
    digest = phase2_sample_loss_weights_sha256(normalized, weights)
    return weights, digest


def metric_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash complete ordered JSON metric rows."""

    return hashlib.sha256(
        "\n".join(canonical_json(dict(row)) for row in rows).encode("utf-8")
    ).hexdigest()
