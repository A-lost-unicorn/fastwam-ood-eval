"""Gate E.8: larger-flow read-only replication of A0 stability.

This sequential diagnostic evaluates only the frozen E.6 A0 step-100 and
step-200 checkpoints.  It uses 64 action-flow slots that were not used by
E.2--E.7 and splits them into two pre-registered 32-flow blocks.  No optimizer,
backward pass, checkpoint write, development outcome, OOD outcome, rollout, or
future RGB is allowed.
"""

from __future__ import annotations

import gc
import hashlib
import math
import os
import statistics
import time
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from fastwam_ood_eval.evaluation.evaluator import git_commit, git_dirty
from fastwam_ood_eval.thought3.checkpointing import (
    adapter_state_sha256,
    load_adapter_checkpoint,
)
from fastwam_ood_eval.thought3.config import (
    Thought3Config,
    load_thought3_config,
)
from fastwam_ood_eval.thought3.injection import (
    ActionEncoderFutureInjector,
)
from fastwam_ood_eval.thought3.io_utils import (
    atomic_write_json,
    load_json,
    sha256_file,
)
from fastwam_ood_eval.thought3.phase_c_smoke import _load_upstream_model
from fastwam_ood_eval.thought3.phase_e6_fresh_cohort_replication import (
    PHASE_E6_CONFIG,
    PHASE_E6_ROOT,
    derive_e6_track_config,
    replication_performance_checks,
    verify_frozen_fresh_cohort,
    verify_frozen_phase_e5,
)
from fastwam_ood_eval.thought3.phase_e7_checkpoint_trajectory import (
    PHASE_E6_CHECKPOINT_FILE_SHA256,
    PHASE_E6_SAMPLE_PAYLOAD_SHA256,
    PHASE_E7_CONFIG,
    PHASE_E7_FASTWAM_COMMIT,
    PHASE_E7_ROOT,
    _checkpoint_hashes,
    _checkpoint_path,
    verify_frozen_phase_e6,
)
from fastwam_ood_eval.thought3.phase_e_training_smoke import (
    _verify_phase_d_gate,
)
from fastwam_ood_eval.thought3.real_training import (
    _checkpoint_expected,
    _flow_objective_identity,
    _flow_timestep_and_weight_scalars,
    _loss_for_real_sample,
    build_real_adapter,
    fixed_subset_outcome,
    prepare_real_training_data,
)
from fastwam_ood_eval.thought3.safety import ensure_thought3_output_path


PHASE_E8_SCHEMA = "thought3.phase_e8.a0_flow_variance_replication.v1"
PHASE_E8_CONFIG = Path(
    "configs/thought3/phase_e8_a0_flow_variance_replication.yaml"
)
PHASE_E8_EXPERIMENT_NAME = (
    "thought3_phase_e8_a0_flow_variance_replication"
)
PHASE_E8_ROOT = Path(
    "outputs/thought3/phase_e8_a0_flow_variance_replication_v1"
)
PHASE_E8_CONFIG_FINGERPRINT = (
    "ed587c61cec3e386e5b44af11fca646dab527acbe46cce34d6badfd34ff09f7f"
)
PHASE_E8_CHECKPOINT_STEPS = (100, 200)
PHASE_E8_FLOW_BLOCK_A = tuple(range(11, 43))
PHASE_E8_FLOW_BLOCK_B = tuple(range(43, 75))
PHASE_E8_FLOW_STEPS = PHASE_E8_FLOW_BLOCK_A + PHASE_E8_FLOW_BLOCK_B
PHASE_E8_IDENTITY_SCHEDULE_SHA256 = (
    "710b809614aeb502c944275c4c43759d2383b00e52fd9d5216898fb949b5772a"
)
PHASE_E8_ZERO_WEIGHT_POSITIONS = (
    (2, 47),
    (2, 59),
    (2, 68),
    (4, 69),
    (5, 58),
    (6, 70),
    (7, 11),
)
PHASE_E8_E7_TARGET_SAMPLE_IDS = (
    "75359438f810e6921754de327beda8bd974343f5e89fb54d7ac8852f79c89c9b",
    "5f82a5db9be7a61f969fd32f5bca19dbb19a65106fb49d5357705be2d03def44",
    "81363feff988d3f3faaeeb66191e7ff9c4fd40c85d7b3b7cd0bda84cd41e3b9b",
)
PHASE_E8_BOOTSTRAP_REPLICATES = 20_000
PHASE_E8_BOOTSTRAP_SEED = 20_260_729_080
PHASE_E8_FIVE_FLOW_RESAMPLES = 20_000
PHASE_E8_FIVE_FLOW_SEED = 20_260_729_081
PHASE_E8_FAMILYWISE_ALPHA = 0.05
PHASE_E8_FAMILYWISE_COMPARISONS = 16
PHASE_E8_MIN_CONFIRMED_TARGET_SAMPLES = 2
PHASE_E8_PROBE_OBJECTIVES = 1_536
PHASE_E7_FROZEN_ARTIFACTS = {
    "gate_e7_result.json": (
        "9b242a3a38638cf2f67c31dd343af0e0d1ec39941d3e784dcd3e167bf14baa4b"
    ),
    "run_status.json": (
        "207dc70a5a83bd67787f038559a4262708b9fb4e355f628cbc6cca90a162e125"
    ),
    "pre_validation_result.json": (
        "cbe4bf697c07307bca3f9708fefd235160ccb6bcf355920c85913ac979616b5f"
    ),
    "data_preparation.json": (
        "f6635c8d0e80d052ad06ce5848bbd2d2ee14635fd0594d44095ccc3461a57fc4"
    ),
    "logs/phase_e7.log": (
        "e32a9bbbd74582f39d4593f851235e29c6145dd01b6c4cd3188f77ac8a78d899"
    ),
}


class PhaseE8GateError(RuntimeError):
    """Raised when the frozen read-only E.8 protocol is violated."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress(
    stage: str,
    values: Mapping[str, Any] | None = None,
    **extra: Any,
) -> None:
    import json

    payload = dict(values or {})
    payload.update(extra)
    print(
        json.dumps(
            {
                "phase": "E.8",
                "stage": stage,
                "time": _utc_now(),
                **payload,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _assert_phase_e8_scope(cfg: Thought3Config) -> None:
    e7 = load_thought3_config(PHASE_E7_CONFIG)
    expected = replace(
        e7,
        experiment=replace(
            e7.experiment,
            name=PHASE_E8_EXPERIMENT_NAME,
            output_dir=PHASE_E8_ROOT,
        ),
    )
    observed_payload = cfg.to_dict()
    expected_payload = expected.to_dict()
    observed_payload.pop("source_path")
    expected_payload.pop("source_path")
    if observed_payload != expected_payload:
        raise PhaseE8GateError(
            "Gate E.8 changes more than experiment name/output"
        )
    if (
        cfg.fingerprint != PHASE_E8_CONFIG_FINGERPRINT
        or cfg.experiment.name != PHASE_E8_EXPERIMENT_NAME
        or cfg.experiment.output_dir != PHASE_E8_ROOT
    ):
        raise PhaseE8GateError("Gate E.8 frozen config identity changed")


def _require_phase_e8_confirmation() -> None:
    if os.environ.get("CONFIRM_THOUGHT3_PHASE_E8") != "YES":
        raise PhaseE8GateError(
            "set CONFIRM_THOUGHT3_PHASE_E8=YES for real read-only probes"
        )


def _verify_execution_repository() -> dict[str, Any]:
    project_root = Path.cwd()
    fastwam_root = Path("third_party/FastWAM")
    provenance = {
        "fastwam_commit": git_commit(fastwam_root),
        "fastwam_dirty": git_dirty(fastwam_root),
        "project_commit": git_commit(project_root),
        "project_dirty": git_dirty(project_root),
    }
    if (
        provenance["project_commit"] is None
        or provenance["project_dirty"] is not False
        or provenance["fastwam_commit"] != PHASE_E7_FASTWAM_COMMIT
        or provenance["fastwam_dirty"] is not False
    ):
        raise PhaseE8GateError(
            "Gate E.8 requires clean project/FastWAM repositories and "
            "the frozen FastWAM commit"
        )
    return provenance


def probe_identity_schedule_sha256(
    sample_ids: Sequence[str],
    *,
    train_seed: int,
    flow_steps: Sequence[int],
) -> str:
    """Hash every E.8 RNG identity knowable before model loading."""

    normalized_ids = tuple(str(value) for value in sample_ids)
    normalized_steps = tuple(int(value) for value in flow_steps)
    if (
        len(normalized_ids) != 8
        or len(set(normalized_ids)) != 8
        or not normalized_steps
        or len(set(normalized_steps)) != len(normalized_steps)
        or any(value < 1 for value in normalized_steps)
    ):
        raise PhaseE8GateError(
            "Gate E.8 identity requires 8 unique samples and unique "
            "positive flow steps"
        )
    rows: list[str] = []
    for sample_index, base_sample_id in enumerate(
        normalized_ids,
        start=1,
    ):
        for flow_step in normalized_steps:
            identity = _flow_objective_identity(
                base_sample_id=base_sample_id,
                train_seed=train_seed,
                flow_step=flow_step,
            )
            rows.append(
                "\0".join(
                    (
                        str(sample_index),
                        base_sample_id,
                        str(flow_step),
                        str(identity["action_noise_seed"]),
                        str(identity["action_timestep_seed"]),
                        str(identity["flow_objective_sha256"]),
                    )
                )
            )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _assert_frozen_probe_design(
    sample_ids: Sequence[str],
    *,
    train_seed: int,
) -> str:
    if (
        PHASE_E8_CHECKPOINT_STEPS != (100, 200)
        or PHASE_E8_FLOW_BLOCK_A != tuple(range(11, 43))
        or PHASE_E8_FLOW_BLOCK_B != tuple(range(43, 75))
        or PHASE_E8_FLOW_STEPS
        != tuple(range(11, 75))
        or set(PHASE_E8_FLOW_BLOCK_A) & set(PHASE_E8_FLOW_BLOCK_B)
        or set(PHASE_E8_FLOW_STEPS) & set(range(0, 11))
        or set(PHASE_E8_FLOW_STEPS)
        & (
            set(range(10_001, 10_201))
            | set(range(20_001, 21_601))
            | set(range(31_001, 32_601))
        )
        or PHASE_E8_BOOTSTRAP_REPLICATES != 20_000
        or PHASE_E8_FIVE_FLOW_RESAMPLES != 20_000
        or PHASE_E8_FAMILYWISE_COMPARISONS != 16
        or PHASE_E8_MIN_CONFIRMED_TARGET_SAMPLES != 2
        or PHASE_E8_PROBE_OBJECTIVES != 1_536
    ):
        raise PhaseE8GateError("Gate E.8 frozen probe design changed")
    identity = probe_identity_schedule_sha256(
        sample_ids,
        train_seed=train_seed,
        flow_steps=PHASE_E8_FLOW_STEPS,
    )
    if identity != PHASE_E8_IDENTITY_SCHEDULE_SHA256:
        raise PhaseE8GateError(
            "Gate E.8 frozen probe RNG identity changed"
        )
    return identity


def verify_frozen_phase_e7() -> dict[str, Any]:
    """Validate the complete E.7 result and its E.6 checkpoint parent."""

    artifact_sha256: dict[str, str] = {}
    for name, expected in PHASE_E7_FROZEN_ARTIFACTS.items():
        path = PHASE_E7_ROOT / name
        if not path.is_file() or sha256_file(path) != expected:
            raise PhaseE8GateError(
                f"frozen Gate E.7 artifact changed/missing: {path}"
            )
        artifact_sha256[str(path)] = expected
    result = load_json(PHASE_E7_ROOT / "gate_e7_result.json")
    status = load_json(PHASE_E7_ROOT / "run_status.json")
    target_ids = tuple(
        str(row["base_sample_id"])
        for row in result["tracks"]["A0"]["200"]["outcome"][
            "per_sample"
        ]
        if bool(row["non_worsened"]) is False
    )
    if (
        result.get("schema_version")
        != "thought3.phase_e7.checkpoint_trajectory.v1"
        or result.get("status") != "complete"
        or result.get("engineering_passed") is not True
        or result.get("gate_e7_passed") is not True
        or status.get("status") != "complete"
        or status.get("gate_e7_passed") is not True
        or result["diagnostic_classification"]["classification"]
        != "not_supported_no_material_late_degradation"
        or result["diagnostic_candidate"][
            "diagnostic_candidate_steps"
        ]
        != []
        or result["primary_panel"]["flow_steps"] != [6, 7, 8, 9, 10]
        or result["scope"]["probe_objectives"] != 800
        or result["scope"]["backward_calls"] != 0
        or result["scope"]["optimizer_steps"] != 0
        or target_ids != PHASE_E8_E7_TARGET_SAMPLE_IDS
        or not all(
            result["tracks"]["A0"]["100"][
                "performance_checks"
            ].values()
        )
        or all(
            result["tracks"]["A0"]["200"][
                "performance_checks"
            ].values()
        )
    ):
        raise PhaseE8GateError(
            "Gate E.7 is not the frozen completed diagnostic parent"
        )
    e6 = verify_frozen_phase_e6()
    if (
        list(result["fresh_cohort_frozen"]["sample_ids"])
        != list(e6["sample_ids"])
    ):
        raise PhaseE8GateError("E.7/E.6 sample identity mismatch")
    return {
        "artifact_sha256": artifact_sha256,
        "e6": e6,
        "e7_classification": result[
            "diagnostic_classification"
        ]["classification"],
        "e7_primary_step100": result["tracks"]["A0"]["100"][
            "outcome"
        ],
        "e7_primary_step200": result["tracks"]["A0"]["200"][
            "outcome"
        ],
        "e7_target_sample_ids": list(target_ids),
        "known_before_e8": {
            "e7_all_results_read": True,
            "step100_selected_post_e7_as_early_comparator": True,
            "step200_is_target_endpoint": True,
            "target_samples_selected_from_e7_primary_step200": True,
        },
        "root": str(PHASE_E7_ROOT),
        "sample_ids": list(e6["sample_ids"]),
    }


def _aggregate_probe_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_ids: Sequence[str],
    flow_steps: Sequence[int],
) -> dict[str, Any]:
    """Aggregate a complete 8-sample × arbitrary-flow A0 grid."""

    normalized_ids = tuple(str(value) for value in sample_ids)
    normalized_steps = tuple(int(value) for value in flow_steps)
    if (
        len(normalized_ids) != 8
        or len(set(normalized_ids)) != 8
        or not normalized_steps
        or len(set(normalized_steps)) != len(normalized_steps)
        or any(value < 1 for value in normalized_steps)
    ):
        raise PhaseE8GateError("invalid E.8 probe grid identity")
    expected_pairs = {
        (base_sample_id, flow_step)
        for base_sample_id in normalized_ids
        for flow_step in normalized_steps
    }
    keyed: dict[tuple[str, int], Mapping[str, Any]] = {}
    fields = (
        "action_hidden_norm",
        "action_loss",
        "action_weight",
        "attention_residual_norm",
        "gated_delta_nonzero_fraction",
        "gated_delta_norm",
        "gated_delta_to_action_hidden_ratio",
        "latency_ms",
        "peak_memory_mib",
        "timestep",
    )
    for row in rows:
        key = (str(row["base_sample_id"]), int(row["flow_step"]))
        if key in keyed:
            raise PhaseE8GateError(f"duplicate E.8 objective: {key}")
        values = [float(row[field]) for field in fields]
        if (
            any(not math.isfinite(value) for value in values)
            or float(row["action_hidden_norm"]) <= 0
            or float(row["action_loss"]) < 0
            or float(row["action_weight"]) < 0
            or float(row["gated_delta_nonzero_fraction"]) < 0
            or float(row["gated_delta_nonzero_fraction"]) > 1
            or (
                float(row["action_weight"]) == 0
                and float(row["action_loss"]) != 0
            )
        ):
            raise PhaseE8GateError(f"invalid E.8 objective: {key}")
        keyed[key] = row
    if set(keyed) != expected_pairs:
        raise PhaseE8GateError("E.8 objective grid is incomplete")
    ordered_rows = [
        dict(keyed[(base_sample_id, flow_step)])
        for base_sample_id in normalized_ids
        for flow_step in normalized_steps
    ]
    per_sample: list[dict[str, Any]] = []
    for base_sample_id in normalized_ids:
        sample_rows = [
            keyed[(base_sample_id, flow_step)]
            for flow_step in normalized_steps
        ]

        def mean(field: str) -> float:
            return statistics.fmean(
                float(row[field]) for row in sample_rows
            )

        per_sample.append(
            {
                "action_hidden_norm": mean("action_hidden_norm"),
                "action_loss": mean("action_loss"),
                "action_weight": mean("action_weight"),
                "attention_residual_norm": mean(
                    "attention_residual_norm"
                ),
                "base_sample_id": base_sample_id,
                "flow_objective_count": len(normalized_steps),
                "flow_steps": list(normalized_steps),
                "gated_delta_nonzero_fraction": mean(
                    "gated_delta_nonzero_fraction"
                ),
                "gated_delta_norm": mean("gated_delta_norm"),
                "gated_delta_to_action_hidden_ratio": mean(
                    "gated_delta_to_action_hidden_ratio"
                ),
                "max_objective_gated_delta_to_action_hidden_ratio": max(
                    float(
                        row[
                            "gated_delta_to_action_hidden_ratio"
                        ]
                    )
                    for row in sample_rows
                ),
                "zero_action_loss_objective_count": sum(
                    float(row["action_loss"]) == 0
                    for row in sample_rows
                ),
                "zero_weight_objective_count": sum(
                    float(row["action_weight"]) == 0
                    for row in sample_rows
                ),
            }
        )
    sample_losses = [
        float(row["action_loss"]) for row in per_sample
    ]
    sample_ratios = [
        float(row["gated_delta_to_action_hidden_ratio"])
        for row in per_sample
    ]
    objective_ratios = [
        float(row["gated_delta_to_action_hidden_ratio"])
        for row in ordered_rows
    ]
    return {
        "flow_objective_count": len(ordered_rows),
        "flow_steps": list(normalized_steps),
        "max_gated_delta_to_action_hidden_ratio": max(sample_ratios),
        "max_objective_gated_delta_to_action_hidden_ratio": max(
            objective_ratios
        ),
        "mean_action_loss": statistics.fmean(sample_losses),
        "median_gated_delta_to_action_hidden_ratio": statistics.median(
            sample_ratios
        ),
        "per_objective": ordered_rows,
        "per_sample": per_sample,
        "sample_count": 8,
        "sample_ids": list(normalized_ids),
        "uses_ground_truth_future_input": False,
        "variant": "A0",
        "zero_action_loss_objective_count": sum(
            float(row["action_loss"]) == 0 for row in ordered_rows
        ),
        "zero_weight_objective_count": sum(
            float(row["action_weight"]) == 0 for row in ordered_rows
        ),
    }


def _subset_probe(
    probe: Mapping[str, Any],
    *,
    flow_steps: Sequence[int],
) -> dict[str, Any]:
    selected = set(int(value) for value in flow_steps)
    rows = [
        row
        for row in probe["per_objective"]
        if int(row["flow_step"]) in selected
    ]
    result = _aggregate_probe_rows(
        rows,
        sample_ids=probe["sample_ids"],
        flow_steps=flow_steps,
    )
    result.update(
        {
            "gate_raw": float(probe["gate_raw"]),
            "max_objective_peak_memory_mib": max(
                float(row["peak_memory_mib"]) for row in rows
            ),
            "mean_objective_latency_ms": statistics.fmean(
                float(row["latency_ms"]) for row in rows
            ),
        }
    )
    return result


def _probe_outcome(
    initial_probe: Mapping[str, Any],
    final_probe: Mapping[str, Any],
    *,
    flow_steps: Sequence[int],
) -> dict[str, Any]:
    expected = list(int(value) for value in flow_steps)
    expected_count = 8 * len(expected)
    if (
        initial_probe["flow_steps"] != expected
        or final_probe["flow_steps"] != expected
        or int(initial_probe["flow_objective_count"]) != expected_count
        or int(final_probe["flow_objective_count"]) != expected_count
        or initial_probe["sample_ids"] != final_probe["sample_ids"]
    ):
        raise PhaseE8GateError("E.8 initial/final grid mismatch")
    initial_rows = {
        (str(row["base_sample_id"]), int(row["flow_step"])): row
        for row in initial_probe["per_objective"]
    }
    final_rows = {
        (str(row["base_sample_id"]), int(row["flow_step"])): row
        for row in final_probe["per_objective"]
    }
    if (
        len(initial_rows) != expected_count
        or set(initial_rows) != set(final_rows)
    ):
        raise PhaseE8GateError("E.8 initial/final objective mismatch")
    ratios: list[float] = []
    zero_initial = 0
    positive_final_from_zero = 0
    for key, initial_row in initial_rows.items():
        final_row = final_rows[key]
        if (
            float(initial_row["timestep"])
            != float(final_row["timestep"])
            or float(initial_row["action_weight"])
            != float(final_row["action_weight"])
        ):
            raise PhaseE8GateError("E.8 paired flow input changed")
        initial_loss = float(initial_row["action_loss"])
        final_loss = float(final_row["action_loss"])
        if initial_loss == 0:
            zero_initial += 1
            positive_final_from_zero += final_loss > 0
        else:
            ratios.append(final_loss / initial_loss)
    if not ratios:
        raise PhaseE8GateError("E.8 has no positive initial loss")
    outcome = fixed_subset_outcome(initial_probe, final_probe)
    return {
        **outcome,
        "flow_objective_count": expected_count,
        "flow_steps": expected,
        "max_objective_gated_delta_to_action_hidden_ratio": float(
            final_probe[
                "max_objective_gated_delta_to_action_hidden_ratio"
            ]
        ),
        "max_objective_loss_ratio": max(ratios),
        "objective_loss_ratio_count": len(ratios),
        "positive_final_from_zero_initial_loss_count": (
            positive_final_from_zero
        ),
        "zero_initial_loss_objective_count": zero_initial,
        "zero_weight_objective_count": int(
            initial_probe["zero_weight_objective_count"]
        ),
    }


def _probe_checks(
    probe: Mapping[str, Any],
    *,
    sample_ids: Sequence[str],
    train_seed: int,
) -> dict[str, bool]:
    rows = list(probe["per_objective"])
    normalized_ids = [str(value) for value in sample_ids]
    observed_zero_positions = [
        (
            normalized_ids.index(str(row["base_sample_id"])) + 1,
            int(row["flow_step"]),
        )
        for row in rows
        if float(row["action_weight"]) == 0
    ]
    expected_grid = {
        (sample_id, flow_step)
        for sample_id in normalized_ids
        for flow_step in PHASE_E8_FLOW_STEPS
    }
    identity_exact = all(
        all(
            row.get(key)
            == _flow_objective_identity(
                base_sample_id=str(row["base_sample_id"]),
                train_seed=train_seed,
                flow_step=int(row["flow_step"]),
            )[key]
            for key in (
                "action_noise_seed",
                "action_timestep_seed",
                "flow_objective_sha256",
            )
        )
        for row in rows
    )
    return {
        "complete_probe_grid": (
            probe["flow_steps"] == list(PHASE_E8_FLOW_STEPS)
            and int(probe["flow_objective_count"]) == 512
            and len(rows) == 512
            and {
                (str(row["base_sample_id"]), int(row["flow_step"]))
                for row in rows
            }
            == expected_grid
        ),
        "finite_probe": all(
            all(
                math.isfinite(float(row[field]))
                for field in (
                    "action_hidden_norm",
                    "action_loss",
                    "action_weight",
                    "attention_residual_norm",
                    "gated_delta_nonzero_fraction",
                    "gated_delta_norm",
                    "gated_delta_to_action_hidden_ratio",
                    "latency_ms",
                    "peak_memory_mib",
                    "timestep",
                )
            )
            for row in rows
        ),
        "memory_below_43_gib": (
            float(probe["max_objective_peak_memory_mib"]) < 43 * 1024
        ),
        "no_ground_truth_future": (
            probe.get("uses_ground_truth_future_input") is False
        ),
        "probe_rng_identity_exact": (
            identity_exact
            and int(probe.get("train_seed", -1)) == train_seed
            and probe.get("identity_schedule_sha256")
            == PHASE_E8_IDENTITY_SCHEDULE_SHA256
        ),
        "zero_weight_loss_exact": all(
            float(row["action_weight"]) != 0
            or float(row["action_loss"]) == 0
            for row in rows
        ),
        "zero_weight_positions_exact": (
            tuple(observed_zero_positions)
            == PHASE_E8_ZERO_WEIGHT_POSITIONS
            and int(probe["zero_weight_objective_count"])
            == len(PHASE_E8_ZERO_WEIGHT_POSITIONS)
        ),
    }


def paired_flow_bootstrap(
    initial_probe: Mapping[str, Any],
    final_probe: Mapping[str, Any],
    *,
    checkpoint_step: int,
    bootstrap_replicates: int = PHASE_E8_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = PHASE_E8_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Estimate sample-level paired flow uncertainty with frozen FWER."""

    import numpy as np

    if (
        checkpoint_step not in PHASE_E8_CHECKPOINT_STEPS
        or bootstrap_replicates != PHASE_E8_BOOTSTRAP_REPLICATES
    ):
        raise PhaseE8GateError("E.8 bootstrap design changed")
    sample_ids = [str(value) for value in initial_probe["sample_ids"]]
    if sample_ids != [str(value) for value in final_probe["sample_ids"]]:
        raise PhaseE8GateError("E.8 bootstrap sample order changed")
    initial_rows = {
        (str(row["base_sample_id"]), int(row["flow_step"])): float(
            row["action_loss"]
        )
        for row in initial_probe["per_objective"]
    }
    final_rows = {
        (str(row["base_sample_id"]), int(row["flow_step"])): float(
            row["action_loss"]
        )
        for row in final_probe["per_objective"]
    }
    if set(initial_rows) != set(final_rows):
        raise PhaseE8GateError("E.8 bootstrap objective pairing changed")
    one_sided_quantile = (
        PHASE_E8_FAMILYWISE_ALPHA
        / PHASE_E8_FAMILYWISE_COMPARISONS
    )
    per_sample: list[dict[str, Any]] = []
    for sample_index, sample_id in enumerate(sample_ids):
        initial = np.asarray(
            [
                initial_rows[(sample_id, flow)]
                for flow in PHASE_E8_FLOW_STEPS
            ],
            dtype=np.float64,
        )
        final = np.asarray(
            [
                final_rows[(sample_id, flow)]
                for flow in PHASE_E8_FLOW_STEPS
            ],
            dtype=np.float64,
        )
        if float(initial.mean()) <= 0:
            raise PhaseE8GateError(
                "E.8 bootstrap initial mean must be positive"
            )
        rng = np.random.default_rng(
            bootstrap_seed + checkpoint_step * 1_009 + sample_index
        )
        indices = rng.integers(
            0,
            len(PHASE_E8_FLOW_STEPS),
            size=(bootstrap_replicates, len(PHASE_E8_FLOW_STEPS)),
        )
        boot_initial = initial[indices].mean(axis=1)
        boot_final = final[indices].mean(axis=1)
        if bool(np.any(boot_initial <= 0)):
            raise PhaseE8GateError(
                "E.8 bootstrap produced non-positive initial mean"
            )
        boot_change = (boot_final - boot_initial) / boot_initial
        full_change = float(
            (final.mean() - initial.mean()) / initial.mean()
        )
        block_a_change = float(
            (
                final[: len(PHASE_E8_FLOW_BLOCK_A)].mean()
                - initial[: len(PHASE_E8_FLOW_BLOCK_A)].mean()
            )
            / initial[: len(PHASE_E8_FLOW_BLOCK_A)].mean()
        )
        block_b_change = float(
            (
                final[len(PHASE_E8_FLOW_BLOCK_A) :].mean()
                - initial[len(PHASE_E8_FLOW_BLOCK_A) :].mean()
            )
            / initial[len(PHASE_E8_FLOW_BLOCK_A) :].mean()
        )
        lower = float(
            np.quantile(
                boot_change,
                one_sided_quantile,
                method="linear",
            )
        )
        upper = float(
            np.quantile(
                boot_change,
                1.0 - one_sided_quantile,
                method="linear",
            )
        )
        confirmed_worsened = (
            full_change > 0
            and block_a_change > 0
            and block_b_change > 0
            and lower > 0
        )
        per_sample.append(
            {
                "base_sample_id": sample_id,
                "block_a_relative_change": block_a_change,
                "block_b_relative_change": block_b_change,
                "bonferroni_one_sided_lower": lower,
                "bonferroni_one_sided_upper": upper,
                "confirmed_improved": (
                    full_change < 0
                    and block_a_change < 0
                    and block_b_change < 0
                    and upper < 0
                ),
                "confirmed_worsened": confirmed_worsened,
                "e7_target_sample": (
                    sample_id in PHASE_E8_E7_TARGET_SAMPLE_IDS
                ),
                "flow_worsened_fraction": float(
                    np.mean(final > initial)
                ),
                "full_relative_change": full_change,
                "material_two_percent_worsening": (
                    confirmed_worsened and full_change >= 0.02
                ),
            }
        )
    confirmed = [
        row["base_sample_id"]
        for row in per_sample
        if row["confirmed_worsened"]
    ]
    confirmed_targets = [
        sample_id
        for sample_id in confirmed
        if sample_id in PHASE_E8_E7_TARGET_SAMPLE_IDS
    ]
    return {
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "checkpoint_step": checkpoint_step,
        "confirmed_target_sample_count": len(confirmed_targets),
        "confirmed_target_sample_ids": confirmed_targets,
        "confirmed_worsened_sample_count": len(confirmed),
        "confirmed_worsened_sample_ids": confirmed,
        "familywise_alpha": PHASE_E8_FAMILYWISE_ALPHA,
        "familywise_comparisons": PHASE_E8_FAMILYWISE_COMPARISONS,
        "one_sided_quantile": one_sided_quantile,
        "per_sample": per_sample,
        "resampling_unit": "paired_flow_within_sample",
    }


def five_flow_resampling_sensitivity(
    initial_probe: Mapping[str, Any],
    final_probe: Mapping[str, Any],
    *,
    checkpoint_step: int,
    resamples: int = PHASE_E8_FIVE_FLOW_RESAMPLES,
    seed: int = PHASE_E8_FIVE_FLOW_SEED,
) -> dict[str, Any]:
    """Describe how often a five-flow panel would pass the original A0 gate."""

    import numpy as np

    if (
        checkpoint_step not in PHASE_E8_CHECKPOINT_STEPS
        or resamples != PHASE_E8_FIVE_FLOW_RESAMPLES
    ):
        raise PhaseE8GateError("E.8 five-flow resampling design changed")
    sample_ids = [str(value) for value in initial_probe["sample_ids"]]
    initial_rows = {
        (str(row["base_sample_id"]), int(row["flow_step"])): float(
            row["action_loss"]
        )
        for row in initial_probe["per_objective"]
    }
    final_rows = {
        (str(row["base_sample_id"]), int(row["flow_step"])): row
        for row in final_probe["per_objective"]
    }
    initial = np.asarray(
        [
            [
                initial_rows[(sample_id, flow)]
                for flow in PHASE_E8_FLOW_STEPS
            ]
            for sample_id in sample_ids
        ],
        dtype=np.float64,
    )
    final = np.asarray(
        [
            [
                float(final_rows[(sample_id, flow)]["action_loss"])
                for flow in PHASE_E8_FLOW_STEPS
            ]
            for sample_id in sample_ids
        ],
        dtype=np.float64,
    )
    ratios = np.asarray(
        [
            [
                float(
                    final_rows[(sample_id, flow)][
                        "gated_delta_to_action_hidden_ratio"
                    ]
                )
                for flow in PHASE_E8_FLOW_STEPS
            ]
            for sample_id in sample_ids
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed + checkpoint_step * 1_009)
    random_scores = rng.random(
        (resamples, len(PHASE_E8_FLOW_STEPS))
    )
    selected = np.argpartition(random_scores, kth=4, axis=1)[:, :5]
    initial_means = initial[:, selected].mean(axis=2).T
    final_means = final[:, selected].mean(axis=2).T
    per_sample_ratios = ratios[:, selected].mean(axis=2).T
    objective_ratios = ratios[:, selected]
    initial_pooled = initial_means.mean(axis=1)
    final_pooled = final_means.mean(axis=1)
    reductions = (initial_pooled - final_pooled) / initial_pooled
    non_worsened = (final_means <= initial_means).sum(axis=1)
    catastrophic = (final_means > 2.0 * initial_means).sum(axis=1)
    median_ratios = np.median(per_sample_ratios, axis=1)
    max_objective_ratios = objective_ratios.max(axis=(0, 2))
    gate_pass = (
        (reductions >= 0)
        & (non_worsened >= 6)
        & (catastrophic == 0)
        & (median_ratios <= 0.5)
        & (max_objective_ratios <= 1.0)
    )
    count_distribution = {
        str(value): int(np.sum(non_worsened == value))
        for value in range(9)
    }
    return {
        "checkpoint_step": checkpoint_step,
        "five_flow_gate_fail_rate": float(np.mean(~gate_pass)),
        "five_flow_gate_pass_rate": float(np.mean(gate_pass)),
        "five_flow_sample_stability_fail_rate": float(
            np.mean(non_worsened < 6)
        ),
        "non_worsened_count_distribution": count_distribution,
        "pooled_mean_worsening_rate": float(np.mean(reductions < 0)),
        "reduction_fraction_quantiles": {
            "p05": float(np.quantile(reductions, 0.05)),
            "p50": float(np.quantile(reductions, 0.50)),
            "p95": float(np.quantile(reductions, 0.95)),
        },
        "resamples": resamples,
        "sampling": "five_of_64_without_replacement_same_slots_all_samples",
        "seed": seed,
    }


def classify_a0_flow_variance(
    step_results: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen tail-risk-versus-five-flow-variance decision."""

    if set(step_results) != set(PHASE_E8_CHECKPOINT_STEPS):
        raise PhaseE8GateError("E.8 checkpoint set changed")
    endpoint = step_results[200]
    comparator = step_results[100]
    endpoint_stats = endpoint["paired_bootstrap"]
    comparator_stats = comparator["paired_bootstrap"]
    endpoint_targets = set(
        endpoint_stats["confirmed_target_sample_ids"]
    )
    comparator_targets = set(
        comparator_stats["confirmed_target_sample_ids"]
    )
    endpoint_confirmed = set(
        endpoint_stats["confirmed_worsened_sample_ids"]
    )
    all_endpoint_panels_stable = all(
        all(endpoint["panels"][panel]["performance_checks"].values())
        for panel in ("full", "block_a", "block_b")
    )
    if len(endpoint_targets) >= PHASE_E8_MIN_CONFIRMED_TARGET_SAMPLES:
        classification = "persistent_target_tail_risk_supported"
        binary_answer = "tail_risk"
    elif (
        not endpoint_confirmed
        and all_endpoint_panels_stable
    ):
        classification = "five_flow_panel_variance_supported"
        binary_answer = "five_flow_variance"
    else:
        classification = "mixed_or_inconclusive"
        binary_answer = "inconclusive"
    overlap = endpoint_targets & comparator_targets
    if not endpoint_targets:
        onset = "not_applicable_no_confirmed_endpoint_target"
    elif not overlap:
        onset = "late_emergent_after_step100"
    elif overlap == endpoint_targets:
        onset = "already_present_by_step100"
    else:
        onset = "partly_present_by_step100"
    return {
        "all_step200_panels_pass_original_a0_gate": (
            all_endpoint_panels_stable
        ),
        "binary_answer": binary_answer,
        "classification": classification,
        "confirmed_step100_target_sample_ids": sorted(
            comparator_targets
        ),
        "confirmed_step200_sample_ids": sorted(endpoint_confirmed),
        "confirmed_step200_target_sample_ids": sorted(
            endpoint_targets
        ),
        "e7_target_sample_ids": list(PHASE_E8_E7_TARGET_SAMPLE_IDS),
        "min_confirmed_target_samples": (
            PHASE_E8_MIN_CONFIRMED_TARGET_SAMPLES
        ),
        "onset_subclassification": onset,
    }


def _evaluate_panel(
    cfg: Thought3Config,
    *,
    model: Any,
    adapter: Any,
    samples: Sequence[Any],
    device: str,
) -> dict[str, Any]:
    import torch

    if any(parameter.grad is not None for parameter in adapter.parameters()):
        raise PhaseE8GateError(
            "Gate E.8 Adapter unexpectedly has gradients before probe"
        )
    injector = ActionEncoderFutureInjector(
        model.action_expert.action_encoder,
        adapter,
    )
    was_training = adapter.training
    adapter.eval()
    objective_rows: list[dict[str, Any]] = []
    try:
        with torch.inference_mode():
            for sample in samples:
                for flow_step in PHASE_E8_FLOW_STEPS:
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    started = time.perf_counter()
                    loss = _loss_for_real_sample(
                        cfg,
                        model,
                        adapter,
                        injector,
                        sample,
                        step=flow_step,
                        device=device,
                    )
                    torch.cuda.synchronize(device)
                    diagnostics = adapter.last_diagnostics
                    if (
                        diagnostics is None
                        or diagnostics.action_hidden_norm <= 0
                    ):
                        raise PhaseE8GateError(
                            "Gate E.8 action diagnostics are invalid"
                        )
                    timestep, weight = _flow_timestep_and_weight_scalars(
                        model,
                        sample,
                        train_seed=cfg.training.train_seed,
                        step=flow_step,
                        device=device,
                    )
                    ratio = (
                        diagnostics.gated_delta_norm
                        / diagnostics.action_hidden_norm
                    )
                    row = {
                        "action_hidden_norm": (
                            diagnostics.action_hidden_norm
                        ),
                        "action_loss": float(
                            loss.detach().float().cpu()
                        ),
                        "action_weight": weight,
                        "attention_residual_norm": (
                            diagnostics.attention_residual_norm
                        ),
                        "base_sample_id": sample.base_sample_id,
                        "flow_step": flow_step,
                        "gated_delta_nonzero_fraction": (
                            diagnostics.gated_delta_nonzero_fraction
                        ),
                        "gated_delta_norm": diagnostics.gated_delta_norm,
                        "gated_delta_to_action_hidden_ratio": ratio,
                        "latency_ms": (
                            time.perf_counter() - started
                        )
                        * 1000.0,
                        "peak_memory_mib": (
                            int(
                                torch.cuda.max_memory_allocated(device)
                            )
                            / 2**20
                        ),
                        "timestep": timestep,
                    }
                    row.update(
                        _flow_objective_identity(
                            base_sample_id=sample.base_sample_id,
                            train_seed=cfg.training.train_seed,
                            flow_step=flow_step,
                        )
                    )
                    objective_rows.append(row)
                    del loss
    finally:
        adapter.train(was_training)
        injector.close()
    if any(parameter.grad is not None for parameter in adapter.parameters()):
        raise PhaseE8GateError(
            "Gate E.8 read-only probe produced Adapter gradients"
        )
    result = _aggregate_probe_rows(
        objective_rows,
        sample_ids=[sample.base_sample_id for sample in samples],
        flow_steps=PHASE_E8_FLOW_STEPS,
    )
    result.update(
        {
            "gate_raw": float(adapter.gate.detach().float().cpu()),
            "identity_schedule_sha256": (
                probe_identity_schedule_sha256(
                    result["sample_ids"],
                    train_seed=cfg.training.train_seed,
                    flow_steps=PHASE_E8_FLOW_STEPS,
                )
            ),
            "max_objective_peak_memory_mib": max(
                float(row["peak_memory_mib"])
                for row in objective_rows
            ),
            "mean_objective_latency_ms": statistics.fmean(
                float(row["latency_ms"]) for row in objective_rows
            ),
            "train_seed": cfg.training.train_seed,
        }
    )
    return result


def _panel_results(
    initial_probe: Mapping[str, Any],
    final_probe: Mapping[str, Any],
) -> dict[str, Any]:
    panels: dict[str, Any] = {}
    for name, flow_steps in (
        ("full", PHASE_E8_FLOW_STEPS),
        ("block_a", PHASE_E8_FLOW_BLOCK_A),
        ("block_b", PHASE_E8_FLOW_BLOCK_B),
    ):
        initial = (
            initial_probe
            if name == "full"
            else _subset_probe(initial_probe, flow_steps=flow_steps)
        )
        final = (
            final_probe
            if name == "full"
            else _subset_probe(final_probe, flow_steps=flow_steps)
        )
        outcome = _probe_outcome(
            initial,
            final,
            flow_steps=flow_steps,
        )
        panels[name] = {
            "flow_steps": list(flow_steps),
            "outcome": outcome,
            "performance_checks": replication_performance_checks(
                "A0",
                {"outcome": outcome},
            ),
        }
    return panels


def _run_phase_e8(
    cfg: Thought3Config,
    *,
    execution_repository: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np
    import torch

    from fastwam_ood_eval.thought3.model_wrapper import (
        parameter_state_sha256,
    )

    _assert_phase_e8_scope(cfg)
    _require_phase_e8_confirmation()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise PhaseE8GateError(
            "Gate E.8 requires exactly one CUDA-visible GPU"
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise PhaseE8GateError(
            "Gate E.8 requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    torch.cuda.set_device("cuda:0")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    np.random.seed(cfg.experiment.seed)
    torch.manual_seed(cfg.experiment.seed)
    torch.cuda.manual_seed_all(cfg.experiment.seed)

    output = ensure_thought3_output_path(cfg.experiment.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    e7 = verify_frozen_phase_e7()
    identity_sha256 = _assert_frozen_probe_design(
        e7["sample_ids"],
        train_seed=cfg.training.train_seed,
    )
    e5 = verify_frozen_phase_e5()
    cohort = verify_frozen_fresh_cohort(
        cfg,
        e5_sample_ids=e5["sample_ids"],
    )
    if (
        list(cohort["sample_ids"]) != list(e7["sample_ids"])
        or tuple(PHASE_E8_E7_TARGET_SAMPLE_IDS)
        != tuple(e7["e7_target_sample_ids"])
    ):
        raise PhaseE8GateError(
            "Gate E.8 sample/target identity differs from frozen E.7"
        )
    phase_d = _verify_phase_d_gate(cfg)
    e6_cfg = load_thought3_config(PHASE_E6_CONFIG)
    track_cfg = derive_e6_track_config(e6_cfg, variant="A0")
    checkpoint_hashes_before = _checkpoint_hashes()
    e7_hashes_before = {
        name: sha256_file(PHASE_E7_ROOT / name)
        for name in PHASE_E7_FROZEN_ARTIFACTS
    }
    _progress(
        "frozen_inputs_verified",
        checkpoint_steps=list(PHASE_E8_CHECKPOINT_STEPS),
        flow_count=len(PHASE_E8_FLOW_STEPS),
        identity_schedule_sha256=identity_sha256,
        source_gate_e7_sha256=PHASE_E7_FROZEN_ARTIFACTS[
            "gate_e7_result.json"
        ],
    )

    _progress("model_load_started", device="cuda:0")
    model, upstream_cfg, model_report = _load_upstream_model(cfg)
    torch.cuda.synchronize("cuda:0")
    model_report["load_peak_mib"] = (
        int(torch.cuda.max_memory_allocated("cuda:0")) / 2**20
    )
    _progress(
        "model_loaded",
        load_peak_mib=model_report["load_peak_mib"],
    )
    prepared = prepare_real_training_data(
        cfg,
        model=model,
        upstream_cfg=upstream_cfg,
        device="cuda:0",
        progress=_progress,
        train_only_limit=8,
        train_only_offset=8,
    )
    data_report = dict(prepared.report)
    atomic_write_json(output / "data_preparation.json", data_report)
    source = data_report["current_source"]
    samples_by_id = {
        sample.base_sample_id: sample for sample in prepared.samples
    }
    if (
        source["actual_future_read"] is not False
        or int(source["future_rgb_frames_decoded"]) != 0
        or int(source["action_target_rows_read"]) != 256
        or int(source["current_camera_frames_decoded"]) != 16
        or int(source["state_rows_read"]) != 8
        or data_report["future_rgb_used_as_input"] is not False
        or data_report["split_counts"] != {"train": 8, "development": 0}
        or data_report["available_split_counts"]
        != {"train": 28, "development": 4}
        or data_report["selection_mode"] != "ordered_train_window"
        or int(data_report["train_only_offset"]) != 8
        or data_report["sample_payload_sha256"]
        != PHASE_E6_SAMPLE_PAYLOAD_SHA256
        or set(samples_by_id) != set(e7["sample_ids"])
    ):
        raise PhaseE8GateError("Gate E.8 data-access audit failed")
    samples = tuple(
        samples_by_id[sample_id] for sample_id in e7["sample_ids"]
    )
    _progress("probe_data_ready", samples=len(samples))

    frozen_before = parameter_state_sha256(
        iter(model.named_parameters())
    )
    adapter = build_real_adapter(track_cfg, device="cuda:0")
    initial_adapter_sha = adapter_state_sha256(adapter.state_dict())
    execution_error: BaseException | None = None
    execution_traceback: str | None = None
    initial_probe: dict[str, Any] = {}
    checkpoint_results: dict[int, dict[str, Any]] = {}
    try:
        initial_probe = _evaluate_panel(
            track_cfg,
            model=model,
            adapter=adapter,
            samples=samples,
            device="cuda:0",
        )
        _progress(
            "initial_probe_complete",
            mean_action_loss=initial_probe["mean_action_loss"],
        )
        del adapter
        torch.cuda.empty_cache()
        for step in PHASE_E8_CHECKPOINT_STEPS:
            adapter = build_real_adapter(track_cfg, device="cuda:0")
            if (
                adapter_state_sha256(adapter.state_dict())
                != initial_adapter_sha
            ):
                raise PhaseE8GateError(
                    "Gate E.8 initial Adapter state drifted"
                )
            checkpoint = _checkpoint_path("A0", step)
            manifest = load_adapter_checkpoint(
                checkpoint,
                adapter=adapter,
                expected=_checkpoint_expected(
                    track_cfg,
                    prepared,
                    frozen_parameter_sha256=frozen_before,
                ),
            )
            if (
                int(manifest.global_step) != step
                or int(manifest.sample_cursor) != step * 8
                or manifest.extra.get(
                    "gate_e6_fresh_cohort_replication"
                )
                is not True
            ):
                raise PhaseE8GateError(
                    "Gate E.8 checkpoint provenance changed"
                )
            final_probe = _evaluate_panel(
                track_cfg,
                model=model,
                adapter=adapter,
                samples=samples,
                device="cuda:0",
            )
            panels = _panel_results(initial_probe, final_probe)
            paired_bootstrap = paired_flow_bootstrap(
                initial_probe,
                final_probe,
                checkpoint_step=step,
            )
            five_flow = five_flow_resampling_sensitivity(
                initial_probe,
                final_probe,
                checkpoint_step=step,
            )
            checkpoint_results[step] = {
                "checkpoint": str(checkpoint),
                "checkpoint_adapter_state_sha256": (
                    manifest.extra["adapter_state_sha256"]
                ),
                "final_probe": final_probe,
                "five_flow_resampling": five_flow,
                "paired_bootstrap": paired_bootstrap,
                "panels": panels,
                "step": step,
                "variant": "A0",
            }
            _progress(
                "checkpoint_probe_complete",
                confirmed_targets=paired_bootstrap[
                    "confirmed_target_sample_count"
                ],
                full_loss_reduction=panels["full"]["outcome"][
                    "loss_reduction_fraction"
                ],
                full_non_worsened=panels["full"]["outcome"][
                    "non_worsened_sample_count"
                ],
                step=step,
            )
            del adapter
            torch.cuda.empty_cache()
    except BaseException as exc:
        execution_error = exc
        execution_traceback = traceback.format_exc()

    frozen_after = parameter_state_sha256(
        iter(model.named_parameters())
    )
    checkpoint_hashes_after = _checkpoint_hashes()
    e7_hashes_after = {
        name: sha256_file(PHASE_E7_ROOT / name)
        for name in PHASE_E7_FROZEN_ARTIFACTS
    }
    prevalidation = {
        "captured_at": _utc_now(),
        "checkpoint_file_sha256_after": checkpoint_hashes_after,
        "checkpoint_file_sha256_before": checkpoint_hashes_before,
        "checkpoint_results": checkpoint_results,
        "data_preparation": data_report,
        "execution_error": (
            None
            if execution_error is None
            else f"{type(execution_error).__name__}: {execution_error}"
        ),
        "execution_traceback": execution_traceback,
        "frozen_parameter_sha256_after": frozen_after,
        "frozen_parameter_sha256_before": frozen_before,
        "initial_adapter_sha256": initial_adapter_sha,
        "initial_probe": initial_probe,
        "model_load": model_report,
        "phase_d_frozen": phase_d,
        "phase_e7_frozen": e7,
        "probe_identity_schedule_sha256": identity_sha256,
        "repository": dict(execution_repository),
        "schema_version": PHASE_E8_SCHEMA,
    }
    atomic_write_json(output / "pre_validation_result.json", prevalidation)
    _progress("frozen_hash_after", sha256=frozen_after)
    if execution_error is not None:
        del prepared, upstream_cfg, model
        gc.collect()
        torch.cuda.empty_cache()
        raise PhaseE8GateError(
            "Gate E.8 probe execution failed after frozen hash capture"
        ) from execution_error

    initial_checks = _probe_checks(
        initial_probe,
        sample_ids=e7["sample_ids"],
        train_seed=cfg.training.train_seed,
    )
    initial_checks["initial_zero_gate_exact"] = all(
        float(row["gated_delta_norm"]) == 0
        for row in initial_probe["per_objective"]
    )
    probe_checks = {
        str(step): _probe_checks(
            checkpoint_results[step]["final_probe"],
            sample_ids=e7["sample_ids"],
            train_seed=cfg.training.train_seed,
        )
        for step in PHASE_E8_CHECKPOINT_STEPS
    }
    classification = classify_a0_flow_variance(checkpoint_results)
    repository_after = _verify_execution_repository()
    cross_checks = {
        "all_initial_checks_passed": all(initial_checks.values()),
        "all_probe_checks_passed": all(
            all(checks.values()) for checks in probe_checks.values()
        ),
        "checkpoint_files_unchanged": (
            checkpoint_hashes_before
            == checkpoint_hashes_after
            == PHASE_E6_CHECKPOINT_FILE_SHA256
        ),
        "e7_artifacts_unchanged": (
            e7_hashes_before
            == e7_hashes_after
            == PHASE_E7_FROZEN_ARTIFACTS
        ),
        "frozen_fastwam_has_no_grad": all(
            parameter.grad is None for parameter in model.parameters()
        ),
        "frozen_fastwam_not_trainable": not any(
            parameter.requires_grad for parameter in model.parameters()
        ),
        "frozen_fastwam_unchanged": frozen_before == frozen_after,
        "no_backward_called": True,
        "no_checkpoint_write": True,
        "no_optimizer_created": True,
        "repository_provenance_unchanged": (
            repository_after == dict(execution_repository)
        ),
    }
    engineering_passed = all(cross_checks.values())
    result = {
        "checkpoint_steps": list(PHASE_E8_CHECKPOINT_STEPS),
        "completed_at": _utc_now(),
        "config": cfg.to_dict(),
        "config_fingerprint": cfg.fingerprint,
        "cross_checks": cross_checks,
        "data_preparation": data_report,
        "diagnostic_classification": classification,
        "engineering_passed": engineering_passed,
        "flow_design": {
            "block_a": list(PHASE_E8_FLOW_BLOCK_A),
            "block_b": list(PHASE_E8_FLOW_BLOCK_B),
            "flow_count": len(PHASE_E8_FLOW_STEPS),
            "flow_steps": list(PHASE_E8_FLOW_STEPS),
            "identity_schedule_sha256": identity_sha256,
            "zero_weight_positions": [
                list(value)
                for value in PHASE_E8_ZERO_WEIGHT_POSITIONS
            ],
        },
        "frozen_parameter_sha256_after": frozen_after,
        "frozen_parameter_sha256_before": frozen_before,
        "gate_e8_passed": engineering_passed,
        "initial_adapter_sha256": initial_adapter_sha,
        "initial_checks": initial_checks,
        "initial_probe": initial_probe,
        "model_load": model_report,
        "phase_d_frozen": phase_d,
        "phase_e7_frozen": e7,
        "preregistered_analysis": {
            "bootstrap_replicates": PHASE_E8_BOOTSTRAP_REPLICATES,
            "bootstrap_seed": PHASE_E8_BOOTSTRAP_SEED,
            "classification_rule": (
                "tail risk if >=2 of 3 E.7 target samples are confirmed "
                "worsened; five-flow variance if no sample is confirmed "
                "worsened and full/both 32-flow blocks pass original A0 "
                "stability; otherwise mixed/inconclusive"
            ),
            "familywise_alpha": PHASE_E8_FAMILYWISE_ALPHA,
            "familywise_comparisons": (
                PHASE_E8_FAMILYWISE_COMPARISONS
            ),
            "five_flow_resamples": PHASE_E8_FIVE_FLOW_RESAMPLES,
            "five_flow_seed": PHASE_E8_FIVE_FLOW_SEED,
            "target_sample_ids": list(
                PHASE_E8_E7_TARGET_SAMPLE_IDS
            ),
        },
        "probe_checks": probe_checks,
        "repository_after": repository_after,
        "repository_before": dict(execution_repository),
        "schema_version": PHASE_E8_SCHEMA,
        "scope": {
            "backward_calls": 0,
            "checkpoint_count": 2,
            "checkpoint_probe_objectives": 1_024,
            "development_outcomes_read": False,
            "future_rgb_frames_read": 0,
            "initial_probe_objectives": 512,
            "new_training_samples_consumed": 0,
            "ood_outcomes_read": False,
            "optimizer_steps": 0,
            "probe_objectives": PHASE_E8_PROBE_OBJECTIVES,
            "rollout_started": False,
            "success_outcomes_read": False,
            "training_objectives": 0,
            "training_samples_read_for_probe": 8,
            "uses_ground_truth_future": False,
        },
        "status": "complete" if engineering_passed else "invalid",
        "steps": {
            str(step): checkpoint_results[step]
            for step in PHASE_E8_CHECKPOINT_STEPS
        },
    }
    del prepared, upstream_cfg, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_phase_e8_a0_flow_variance_replication(
    cfg: Thought3Config,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Run E.8 once; valid scientific classifications return success."""

    _assert_phase_e8_scope(cfg)
    _require_phase_e8_confirmation()
    output = ensure_thought3_output_path(cfg.experiment.output_dir)
    result_path = output / "gate_e8_result.json"
    status_path = output / "run_status.json"
    if result_path.is_file():
        existing = load_json(result_path)
        if resume and existing.get("engineering_passed") is True:
            return existing
        if resume:
            raise PhaseE8GateError(
                "existing Gate E.8 result is invalid; preserve this Run ID"
            )
        raise FileExistsError(
            f"Gate E.8 result exists; pass --resume: {result_path}"
        )
    if status_path.is_file():
        raise PhaseE8GateError(
            "partial Gate E.8 evidence must be preserved under this Run ID"
        )
    execution_repository = _verify_execution_repository()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        status_path,
        {
            "schema_version": PHASE_E8_SCHEMA,
            "started_at": _utc_now(),
            "status": "running",
        },
    )
    started = time.perf_counter()
    try:
        result = _run_phase_e8(
            cfg,
            execution_repository=execution_repository,
        )
        result["gate_wall_s"] = time.perf_counter() - started
        atomic_write_json(result_path, result)
        if result["engineering_passed"] is not True:
            raise PhaseE8GateError(
                "Gate E.8 engineering checks failed; inspect result"
            )
    except BaseException as exc:
        atomic_write_json(
            status_path,
            {
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at": _utc_now(),
                "gate_e8_passed": False,
                "result": (
                    str(result_path.resolve())
                    if result_path.is_file()
                    else None
                ),
                "schema_version": PHASE_E8_SCHEMA,
                "status": "failed",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    atomic_write_json(
        status_path,
        {
            "diagnostic_classification": result[
                "diagnostic_classification"
            ]["classification"],
            "finished_at": _utc_now(),
            "gate_e8_passed": True,
            "result": str(result_path.resolve()),
            "schema_version": PHASE_E8_SCHEMA,
            "status": "complete",
        },
    )
    return result
