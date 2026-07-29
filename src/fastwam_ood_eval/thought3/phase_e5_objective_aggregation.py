"""Gate E.5: paired full-cohort objective-aggregation diagnostic.

E.4 established that unique train-flow objectives improve held-out direction
but do not reach the frozen stability gate.  E.5 keeps the same 200 AdamW
updates, samples, variants, LR grid, held-out probe and thresholds.  Its sole
training change is to average one unique objective from each of the eight
samples before every optimizer update.
"""

from __future__ import annotations

import gc
import math
import os
import statistics
import time
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from fastwam_ood_eval.thought3.config import (
    Thought3Config,
    load_thought3_config,
    validate_config,
)
from fastwam_ood_eval.thought3.io_utils import (
    atomic_write_json,
    load_json,
    load_jsonl,
    sha256_file,
)
from fastwam_ood_eval.thought3.objective_aggregation_training import (
    OBJECTIVE_AGGREGATION_EXPECTED_ZERO_WEIGHT_SLOTS,
    OBJECTIVE_AGGREGATION_FLOW_SLOT_OFFSET,
    OBJECTIVE_AGGREGATION_HELDOUT_FLOW_STEPS,
    OBJECTIVE_AGGREGATION_UPDATES,
    OBJECTIVES_PER_UPDATE,
    ObjectiveAggregationProtocol,
    PHASE_E5_OBJECTIVE_AGGREGATION_PROTOCOL,
    objective_aggregation_flow_slot,
    objective_aggregation_identity_schedule_sha256,
    objective_aggregation_schedule_sha256,
    run_full_cohort_objective_aggregation,
)
from fastwam_ood_eval.thought3.phase_c_smoke import (
    _load_upstream_model,
)
from fastwam_ood_eval.thought3.phase_e2_eight_sample import (
    PHASE_E2_LR_GRID,
    _assert_phase_e2_scope,
    _matched_recipe_payload,
    performance_checks,
    select_smallest_eligible_lr,
)
from fastwam_ood_eval.thought3.phase_e3_multiflow import (
    PHASE_E2_SAMPLE_PAYLOAD_SHA256,
)
from fastwam_ood_eval.thought3.phase_e4_diversified_flow import (
    PHASE_E4_CONFIG_FINGERPRINT,
    PHASE_E4_ROOT,
)
from fastwam_ood_eval.thought3.phase_e_training_smoke import (
    _verify_phase_d_gate,
)
from fastwam_ood_eval.thought3.real_training import (
    _flow_objective_identity,
    multiflow_subset_outcome,
    prepare_real_training_data,
)
from fastwam_ood_eval.thought3.safety import (
    ensure_thought3_output_path,
)


PHASE_E5_SCHEMA = "thought3.phase_e5.objective_aggregation.v1"
PHASE_E5_EXPERIMENT_NAME = (
    "thought3_phase_e5_objective_aggregation_diagnostic"
)
PHASE_E5_ROOT = Path(
    "outputs/thought3/phase_e5_objective_aggregation_v1"
)
PHASE_E5_CONFIG_FINGERPRINT = (
    "c4c681534cf4c143a1675c24ded719b7bf0a4c2964b2384704b4338e147122fc"
)
PHASE_E5_FROZEN_IDENTITY_SCHEDULE_SHA256 = (
    "b6f9778d303a6ad2c4bef781f4a6027a800d013814110daa47eb7cb1d13af86d"
)
PHASE_E4_CONFIG = Path(
    "configs/thought3/phase_e4_diversified_flow_diagnostic.yaml"
)
PHASE_E4_FROZEN_ARTIFACTS = {
    "gate_e4_result.json": (
        "48314003c146327c93e3c5ecb173762cde09c27afb1b38124e741a222e974240"
    ),
    "run_status.json": (
        "8c092f6aedbb67054e6853a49e35ec14f4cd3221b7867df6c72d6ff89a0acc43"
    ),
    "pre_validation_result.json": (
        "4a74f33aa3af211854f86873c933530f904466c776c9ac97c969d7ef99cf8223"
    ),
    "data_preparation.json": (
        "5cb61c57ab52feb93b395e3e3f379411e481f936839251b48048aa492c33a699"
    ),
    "logs/phase_e4.log": (
        "6412697e39c55d5ba2c3232615d03007e69d517dff4b81701a12196814480886"
    ),
}


class PhaseE5GateError(RuntimeError):
    """Raised when Gate E.5 violates its frozen engineering protocol."""


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
                "phase": "E.5",
                "stage": stage,
                "time": _utc_now(),
                **payload,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _assert_phase_e5_scope(cfg: Thought3Config) -> None:
    """Reject every recipe change except the frozen accumulation factor."""

    normalized = replace(
        cfg,
        training=replace(
            cfg.training,
            gradient_accumulation_steps=1,
        ),
    )
    _assert_phase_e2_scope(normalized)
    if cfg.training.gradient_accumulation_steps != OBJECTIVES_PER_UPDATE:
        raise PhaseE5GateError(
            "Gate E.5 gradient accumulation must remain exactly 8"
        )
    if cfg.experiment.name != PHASE_E5_EXPERIMENT_NAME:
        raise PhaseE5GateError("Gate E.5 experiment name changed")
    if cfg.experiment.output_dir != PHASE_E5_ROOT:
        raise PhaseE5GateError("Gate E.5 output directory changed")
    if cfg.fingerprint != PHASE_E5_CONFIG_FINGERPRINT:
        raise PhaseE5GateError("Gate E.5 config fingerprint changed")


def _require_phase_e5_confirmation() -> None:
    if os.environ.get("CONFIRM_THOUGHT3_PHASE_E5") != "YES":
        raise PhaseE5GateError(
            "set CONFIRM_THOUGHT3_PHASE_E5=YES for real full-cohort "
            "objective-aggregation training"
        )


def verify_frozen_phase_e4() -> dict[str, Any]:
    """Validate the exact valid-negative E.4 result before model loading."""

    artifact_sha256: dict[str, str] = {}
    for name, expected_sha in PHASE_E4_FROZEN_ARTIFACTS.items():
        path = PHASE_E4_ROOT / name
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise PhaseE5GateError(
                f"frozen Gate E.4 artifact changed/missing: {path}"
            )
        artifact_sha256[str(path)] = expected_sha
    cfg = load_thought3_config(PHASE_E4_CONFIG)
    result = load_json(PHASE_E4_ROOT / "gate_e4_result.json")
    status = load_json(PHASE_E4_ROOT / "run_status.json")
    prevalidation = load_json(
        PHASE_E4_ROOT / "pre_validation_result.json"
    )
    if (
        cfg.fingerprint != PHASE_E4_CONFIG_FINGERPRINT
        or result.get("schema_version")
        != "thought3.phase_e4.diversified_flow.v1"
        or result.get("config_fingerprint")
        != PHASE_E4_CONFIG_FINGERPRINT
        or result.get("status") != "failed"
        or result.get("gate_e4_passed") is not False
        or result.get("selected_lr_slug") is not None
        or result.get("selected_learning_rate") is not None
        or status.get("status") != "failed"
        or status.get("gate_e4_passed") is not False
        or prevalidation.get("execution_error") is not None
        or prevalidation.get("execution_traceback") is not None
    ):
        raise PhaseE5GateError(
            "Gate E.4 is not the frozen valid-negative diagnostic"
        )
    if (
        any(bool(value) for value in result["eligibility"].values())
        or not all(bool(value) for value in result["cross_checks"].values())
        or not all(
            bool(value)
            for checks in result["paired_checks"].values()
            for value in checks.values()
        )
        or any(
            not all(
                bool(value)
                for value in result["tracks"][lr_slug][variant][
                    "execution_checks"
                ].values()
            )
            for lr_slug, _ in PHASE_E2_LR_GRID
            for variant in ("A0", "A1")
        )
        or result["scope"]
        != {
            "development_outcomes_read": False,
            "future_rgb_frames_read": 0,
            "heldout_probe_objectives": 480,
            "learning_rate_count": 3,
            "ood_outcomes_read": False,
            "optimizer_steps": 1200,
            "rollout_started": False,
            "sample_count": 8,
            "single_gpu": True,
            "success_outcomes_read": False,
            "task_count": 1,
            "track_count": 6,
            "uses_ground_truth_future": False,
        }
    ):
        raise PhaseE5GateError(
            "Gate E.4 integrity/execution evidence is incomplete"
        )
    sample_ids = list(
        result["tracks"]["lr_1e_04"]["A0"]["result"]["sample_ids"]
    )
    if any(
        list(
            result["tracks"][lr_slug][variant]["result"]["sample_ids"]
        )
        != sample_ids
        for lr_slug, _ in PHASE_E2_LR_GRID
        for variant in ("A0", "A1")
    ):
        raise PhaseE5GateError("Gate E.4 track sample IDs differ")
    identity_sha256 = objective_aggregation_identity_schedule_sha256(
        sample_ids,
        train_seed=3407,
    )
    if identity_sha256 != PHASE_E5_FROZEN_IDENTITY_SCHEDULE_SHA256:
        raise PhaseE5GateError(
            "Gate E.5 preregistered identity schedule changed"
        )
    return {
        "artifact_sha256": artifact_sha256,
        "config_fingerprint": PHASE_E4_CONFIG_FINGERPRINT,
        "gate_e4_passed": False,
        "identity_schedule_sha256": identity_sha256,
        "root": str(PHASE_E4_ROOT),
        "sample_ids": sample_ids,
    }


def derive_e5_track_config(
    cfg: Thought3Config,
    *,
    variant: str,
    lr_slug: str,
    learning_rate: float,
) -> Thought3Config:
    """Derive one of six matched E.5 A0/A1 × LR tracks."""

    expected = dict(PHASE_E2_LR_GRID)
    if (
        variant not in {"A0", "A1"}
        or lr_slug not in expected
        or learning_rate != expected[lr_slug]
    ):
        raise PhaseE5GateError(
            f"unsupported Gate E.5 track: {variant}/{lr_slug}/{learning_rate}"
        )
    derived = replace(
        cfg,
        variant=variant,
        experiment=replace(
            cfg.experiment,
            name=f"thought3_phase_e5_{variant.lower()}_{lr_slug}",
            output_dir=(
                cfg.experiment.output_dir
                / "tracks"
                / lr_slug
                / variant.lower()
            ),
        ),
        sampler=replace(
            cfg.sampler,
            active_k=0 if variant == "A0" else 1,
        ),
        training=replace(
            cfg.training,
            learning_rate=learning_rate,
        ),
    )
    validate_config(derived)
    return derived


def _derive_tracks(
    cfg: Thought3Config,
) -> dict[str, dict[str, Thought3Config]]:
    tracks = {
        lr_slug: {
            variant: derive_e5_track_config(
                cfg,
                variant=variant,
                lr_slug=lr_slug,
                learning_rate=learning_rate,
            )
            for variant in ("A0", "A1")
        }
        for lr_slug, learning_rate in PHASE_E2_LR_GRID
    }
    for lr_slug, variants in tracks.items():
        if (
            _matched_recipe_payload(variants["A0"])
            != _matched_recipe_payload(variants["A1"])
        ):
            raise PhaseE5GateError(
                f"Gate E.5 A0/A1 recipe differs at {lr_slug}"
            )
    return tracks


def _initial_probe_signature(
    result: Mapping[str, Any],
) -> tuple[tuple[str, int, float, float, float, float], ...]:
    return tuple(
        (
            str(row["base_sample_id"]),
            int(row["flow_step"]),
            float(row["timestep"]),
            float(row["action_weight"]),
            float(row["action_loss"]),
            float(row["gated_delta_norm"]),
        )
        for row in result["initial_probe"]["per_objective"]
    )


def _objective_schedule_matches(
    cfg: Thought3Config,
    rows: Sequence[Mapping[str, Any]],
    sample_ids: Sequence[str],
    *,
    protocol: ObjectiveAggregationProtocol = (
        PHASE_E5_OBJECTIVE_AGGREGATION_PROTOCOL
    ),
) -> bool:
    if (
        len(rows)
        != OBJECTIVE_AGGREGATION_UPDATES * OBJECTIVES_PER_UPDATE
    ):
        return False
    for objective_index, row in enumerate(rows, start=1):
        optimizer_update = (
            (objective_index - 1) // OBJECTIVES_PER_UPDATE + 1
        )
        micro_index = (
            (objective_index - 1) % OBJECTIVES_PER_UPDATE + 1
        )
        base_sample_id = str(sample_ids[micro_index - 1])
        flow_slot = objective_aggregation_flow_slot(
            optimizer_update,
            micro_index,
            flow_slot_offset=protocol.flow_slot_offset,
        )
        expected = _flow_objective_identity(
            base_sample_id=base_sample_id,
            train_seed=cfg.training.train_seed,
            flow_step=flow_slot,
        )
        if (
            int(row["objective_index"]) != objective_index
            or int(row["optimizer_update"]) != optimizer_update
            or int(row["micro_index"]) != micro_index
            or int(row["cohort_sample_index"]) != micro_index - 1
            or int(row["sample_cursor"]) != objective_index
            or int(row["training_flow_slot"]) != flow_slot
            or int(row["flow_step"]) != flow_slot
            or str(row["base_sample_id"]) != base_sample_id
            or any(
                row[field] != expected[field]
                for field in (
                    "action_noise_seed",
                    "action_timestep_seed",
                    "flow_objective_sha256",
                )
            )
        ):
            return False
    return True


def _update_aggregation_matches(
    objective_rows: Sequence[Mapping[str, Any]],
    update_rows: Sequence[Mapping[str, Any]],
) -> bool:
    if len(update_rows) != OBJECTIVE_AGGREGATION_UPDATES:
        return False
    for optimizer_update, update_row in enumerate(
        update_rows,
        start=1,
    ):
        cohort = objective_rows[
            (optimizer_update - 1) * OBJECTIVES_PER_UPDATE:
            optimizer_update * OBJECTIVES_PER_UPDATE
        ]
        raw_losses = [float(row["action_loss"]) for row in cohort]
        action_weights = [float(row["action_weight"]) for row in cohort]
        contributions = [
            float(row["gate_gradient_contribution_mean_scaled"])
            for row in cohort
        ]
        gate_gradient = float(update_row["gate_gradient"])
        absolute_sum = sum(abs(value) for value in contributions)
        expected_cancellation = (
            abs(gate_gradient) / absolute_sum if absolute_sum else 0.0
        )
        running_contribution = 0.0
        contribution_trace_matches = True
        for row, contribution in zip(cohort, contributions):
            running_contribution += contribution
            contribution_trace_matches = (
                contribution_trace_matches
                and math.isclose(
                    float(row["gate_gradient_cumulative"]),
                    running_contribution,
                    rel_tol=1e-5,
                    abs_tol=1e-8,
                )
                and math.isclose(
                    float(
                        row[
                            "gate_gradient_contribution_unscaled"
                        ]
                    ),
                    contribution * OBJECTIVES_PER_UPDATE,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                and int(row["gate_gradient_contribution_sign"])
                == (
                    1
                    if contribution > 0
                    else -1
                    if contribution < 0
                    else 0
                )
            )
        if (
            int(update_row["optimizer_update"]) != optimizer_update
            or int(update_row["objective_count"])
            != OBJECTIVES_PER_UPDATE
            or int(update_row["objective_index_start"])
            != (optimizer_update - 1) * OBJECTIVES_PER_UPDATE + 1
            or int(update_row["objective_index_end"])
            != optimizer_update * OBJECTIVES_PER_UPDATE
            or int(update_row["sample_cursor"])
            != optimizer_update * OBJECTIVES_PER_UPDATE
            or update_row["gradient_reduction"] != "arithmetic_mean"
            or not contribution_trace_matches
            or int(update_row["gate_gradient_sign"])
            != (
                1
                if gate_gradient > 0
                else -1
                if gate_gradient < 0
                else 0
            )
            or not math.isclose(
                float(update_row["mean_action_loss"]),
                statistics.fmean(raw_losses),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(update_row["summed_action_loss"]),
                sum(raw_losses),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(update_row["action_weight_mean"]),
                statistics.fmean(action_weights),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(update_row["action_weight_sum"]),
                sum(action_weights),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or int(update_row["zero_weight_objective_count"])
            != sum(value == 0 for value in action_weights)
            or not math.isclose(
                sum(contributions),
                gate_gradient,
                rel_tol=1e-5,
                abs_tol=1e-8,
            )
            or not math.isclose(
                float(
                    update_row[
                        "gate_gradient_absolute_contribution_sum"
                    ]
                ),
                absolute_sum,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(
                    update_row["gate_gradient_cancellation_ratio"]
                ),
                expected_cancellation,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or any(
                row["gradient_reduction"] != "arithmetic_mean"
                or not math.isclose(
                    float(row["mean_scaled_backward_loss"]),
                    float(row["action_loss"]) / OBJECTIVES_PER_UPDATE,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or float(row["gate_raw_before_update"])
                != float(update_row["gate_raw_before_update"])
                or float(row["gate_raw_after_update"])
                != float(update_row["gate_raw_after_update"])
                or float(row["optimizer_update_peak_memory_mib"])
                != float(update_row["peak_memory_mib"])
                or float(row["optimizer_update_time_ms"])
                != float(update_row["update_time_ms"])
                for row in cohort
            )
        ):
            return False
    return True


def _track_checks(
    cfg: Thought3Config,
    result: Mapping[str, Any],
    *,
    protocol: ObjectiveAggregationProtocol = (
        PHASE_E5_OBJECTIVE_AGGREGATION_PROTOCOL
    ),
    frozen_identity_schedule_sha256: str = (
        PHASE_E5_FROZEN_IDENTITY_SCHEDULE_SHA256
    ),
    forbidden_training_slot_ranges: tuple[
        tuple[int, int], ...
    ] = ((10_001, 10_200),),
) -> tuple[dict[str, bool], dict[str, Any]]:
    objective_path = Path(str(result["objective_metrics"]))
    update_path = Path(str(result["update_metrics"]))
    probe_path = Path(str(result["probe_metrics"]))
    objective_rows = load_jsonl(objective_path)
    update_rows = load_jsonl(update_path)
    probe_rows = load_jsonl(probe_path)
    sample_ids = list(result["sample_ids"])
    zero_slots = tuple(
        (
            int(row["optimizer_update"]),
            int(row["micro_index"]),
            int(row["training_flow_slot"]),
        )
        for row in objective_rows
        if float(row["action_weight"]) == 0
    )
    recomputed_outcome = multiflow_subset_outcome(
        result["initial_probe"],
        result["final_probe"],
    )
    schedule_matches = _objective_schedule_matches(
        cfg,
        objective_rows,
        sample_ids,
        protocol=protocol,
    )
    schedule_sha_matches = False
    if schedule_matches:
        schedule_sha_matches = (
            result["train_flow_schedule_sha256"]
            == objective_aggregation_schedule_sha256(
                objective_rows,
                flow_slot_offset=protocol.flow_slot_offset,
            )
        )
    expected_identity_sha = (
        objective_aggregation_identity_schedule_sha256(
            sample_ids,
            train_seed=cfg.training.train_seed,
            flow_slot_offset=protocol.flow_slot_offset,
        )
    )
    finite_trace = (
        all(
            not bool(row["nan_or_inf"])
            and math.isfinite(float(row["action_loss"]))
            and math.isfinite(float(row["action_weight"]))
            and math.isfinite(float(row["timestep"]))
            and math.isfinite(float(row["action_hidden_norm"]))
            and math.isfinite(float(row["attention_residual_norm"]))
            and math.isfinite(float(row["future_token_norm"]))
            and math.isfinite(float(row["gated_delta_norm"]))
            and math.isfinite(
                float(row["gated_delta_to_action_hidden_ratio"])
            )
            and math.isfinite(float(row["mean_scaled_backward_loss"]))
            and math.isfinite(
                float(
                    row[
                        "gate_gradient_contribution_mean_scaled"
                    ]
                )
            )
            for row in objective_rows
        )
        and all(
            not bool(row["nan_or_inf"])
            and math.isfinite(float(row["mean_action_loss"]))
            and math.isfinite(float(row["gate_gradient"]))
            and math.isfinite(
                float(row["gate_gradient_cancellation_ratio"])
            )
            and 0
            <= float(row["gate_gradient_cancellation_ratio"])
            <= 1 + 1e-6
            and math.isfinite(float(row["peak_memory_mib"]))
            and math.isfinite(float(row["update_time_ms"]))
            and all(
                bool(group["finite"])
                for group in row["gradient_groups"].values()
            )
            for row in update_rows
        )
        and all(
            math.isfinite(float(row["mean_action_loss"]))
            for row in probe_rows
        )
    )
    execution = {
        "complete_200_updates_1600_objectives": (
            result.get("status") == "complete"
            and int(result.get("completed_steps", -1))
            == OBJECTIVE_AGGREGATION_UPDATES
            and int(result.get("completed_objectives", -1))
            == OBJECTIVE_AGGREGATION_UPDATES
            * OBJECTIVES_PER_UPDATE
            and len(update_rows) == OBJECTIVE_AGGREGATION_UPDATES
            and len(objective_rows)
            == OBJECTIVE_AGGREGATION_UPDATES
            * OBJECTIVES_PER_UPDATE
        ),
        "full_cohort_each_update": (
            int(result.get("sample_count", -1))
            == OBJECTIVES_PER_UPDATE
            and len(set(sample_ids)) == OBJECTIVES_PER_UPDATE
            and schedule_matches
        ),
        "unique_disjoint_training_flow_slots": (
            [int(row["training_flow_slot"]) for row in objective_rows]
            == list(
                range(
                    protocol.flow_slot_offset + 1,
                    protocol.flow_slot_offset
                    + OBJECTIVE_AGGREGATION_UPDATES
                    * OBJECTIVES_PER_UPDATE
                    + 1,
                )
            )
            and not {
                int(row["training_flow_slot"])
                for row in objective_rows
            }
            & {0, *OBJECTIVE_AGGREGATION_HELDOUT_FLOW_STEPS}
            and all(
                not {
                    int(row["training_flow_slot"])
                    for row in objective_rows
                }
                & set(range(start, end + 1))
                for start, end in forbidden_training_slot_ranges
            )
        ),
        "objective_seed_identity_exact": schedule_matches,
        "identity_schedule_sha_exact": (
            expected_identity_sha
            == frozen_identity_schedule_sha256
            == result["identity_schedule_sha256"]
        ),
        "observed_schedule_sha_exact": schedule_sha_matches,
        "mean_aggregation_recomputes_exactly": (
            _update_aggregation_matches(
                objective_rows,
                update_rows,
            )
        ),
        "frozen_zero_weight_slots_exact": (
            zero_slots
            == protocol.expected_zero_weight_slots
            and int(result["zero_weight_objective_count"])
            == len(
                protocol.expected_zero_weight_slots
            )
            and tuple(
                tuple(int(value) for value in row)
                for row in result["zero_weight_slots"]
            )
            == protocol.expected_zero_weight_slots
            and all(
                (float(row["action_weight"]) == 0)
                == bool(row["zero_weight_objective"])
                and (
                    float(row["action_weight"]) != 0
                    or float(row["action_loss"]) == 0
                )
                for row in objective_rows
            )
        ),
        "heldout_probe_schedule": (
            [int(row["global_step"]) for row in probe_rows]
            == [0, OBJECTIVE_AGGREGATION_UPDATES]
            and all(
                list(row["flow_steps"])
                == list(OBJECTIVE_AGGREGATION_HELDOUT_FLOW_STEPS)
                and int(row["flow_objective_count"]) == 40
                and int(row["sample_count"]) == 8
                and list(row["sample_ids"]) == sample_ids
                for row in probe_rows
            )
        ),
        "outcome_recomputes_exactly": (
            dict(result["outcome"]) == recomputed_outcome
        ),
        "first_update_gate_only": (
            float(update_rows[0]["gradient_groups"]["gate"]["l2"])
            > 0
            and int(
                update_rows[0]["gradient_groups"]["non_gate"][
                    "nonzero_element_count"
                ]
            )
            == 0
        ),
        "second_update_projector_gradient": (
            int(
                update_rows[1]["gradient_groups"][
                    "future_projector"
                ]["nonzero_element_count"]
            )
            > 0
        ),
        "second_update_attention_gradient": (
            int(
                update_rows[1]["gradient_groups"]["attention"][
                    "nonzero_element_count"
                ]
            )
            > 0
        ),
        "first_non_gate_paths_at_update_2": (
            int(
                result[
                    "first_non_gate_nonzero_gradient_update"
                ]
            )
            == 2
            and int(
                result[
                    "first_projector_nonzero_gradient_update"
                ]
            )
            == 2
            and int(
                result[
                    "first_attention_nonzero_gradient_update"
                ]
            )
            == 2
        ),
        "finite_trace": finite_trace,
        "adapter_only_optimizer": (
            result.get("optimizer_parameter_scope") == "adapter_only"
        ),
        "no_development_or_ood_outcomes": (
            result.get("uses_development_outcomes") is False
            and result.get("uses_ood_or_success_outcomes") is False
        ),
        "no_ground_truth_future_rgb": (
            result.get("uses_ground_truth_future_input") is False
        ),
        "memory_below_43_gib": (
            float(result["max_peak_memory_mib"]) < 43 * 1024
            and all(
                float(row["peak_memory_mib"]) < 43 * 1024
                for row in update_rows
            )
        ),
        "checkpoint_roundtrip": (
            result["checkpoint_roundtrip"].get("state_equal") is True
            and int(result["checkpoint_roundtrip"]["global_step"])
            == OBJECTIVE_AGGREGATION_UPDATES
        ),
    }
    artifacts = {
        "files_sha256": {
            "manifest": sha256_file(
                cfg.experiment.output_dir / "training_manifest.json"
            ),
            "objective_metrics": sha256_file(objective_path),
            "probe_metrics": sha256_file(probe_path),
            "state": sha256_file(
                cfg.experiment.output_dir / "training_state.json"
            ),
            "update_metrics": sha256_file(update_path),
        },
        "identity_schedule_sha256": expected_identity_sha,
        "observed_schedule_sha256": result[
            "train_flow_schedule_sha256"
        ],
        "output_dir": str(cfg.experiment.output_dir),
    }
    return execution, artifacts


def _run_phase_e5(
    cfg: Thought3Config,
    *,
    resume: bool,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from fastwam_ood_eval.thought3.model_wrapper import (
        parameter_state_sha256,
    )

    _assert_phase_e5_scope(cfg)
    _require_phase_e5_confirmation()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise PhaseE5GateError(
            "Gate E.5 requires exactly one CUDA-visible GPU"
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise PhaseE5GateError(
            "Gate E.5 requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
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
    phase_e4 = verify_frozen_phase_e4()
    phase_d = _verify_phase_d_gate(cfg)
    tracks = _derive_tracks(cfg)
    _progress(
        "frozen_inputs_verified",
        gate_e4_sha256=PHASE_E4_FROZEN_ARTIFACTS[
            "gate_e4_result.json"
        ],
        identity_schedule_sha256=(
            PHASE_E5_FROZEN_IDENTITY_SCHEDULE_SHA256
        ),
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
    )
    data_report = dict(prepared.report)
    atomic_write_json(output / "data_preparation.json", data_report)
    source = data_report["current_source"]
    prepared_ids = {
        sample.base_sample_id for sample in prepared.samples
    }
    if (
        source["actual_future_read"] is not False
        or int(source["future_rgb_frames_decoded"]) != 0
        or int(source["action_target_rows_read"]) != 256
        or int(source["current_camera_frames_decoded"]) != 16
        or int(source["state_rows_read"]) != 8
        or data_report["future_rgb_used_as_input"] is not False
        or data_report["split_counts"]
        != {"train": 8, "development": 0}
        or data_report["available_split_counts"]
        != {"train": 28, "development": 4}
        or data_report["selection_mode"] != "ordered_train_only"
        or data_report["sample_payload_sha256"]
        != PHASE_E2_SAMPLE_PAYLOAD_SHA256
        or prepared_ids != set(phase_e4["sample_ids"])
    ):
        raise PhaseE5GateError("Gate E.5 data-access audit failed")

    frozen_before = parameter_state_sha256(
        iter(model.named_parameters())
    )
    results: dict[str, dict[str, Mapping[str, Any]]] = {}
    execution_error: BaseException | None = None
    execution_traceback: str | None = None
    try:
        for lr_slug, learning_rate in PHASE_E2_LR_GRID:
            results[lr_slug] = {}
            for variant in ("A0", "A1"):
                _progress(
                    "track_started",
                    learning_rate=learning_rate,
                    lr_slug=lr_slug,
                    variant=variant,
                )
                track_result = (
                    run_full_cohort_objective_aggregation(
                        tracks[lr_slug][variant],
                        model=model,
                        prepared=prepared,
                        frozen_parameter_sha256=frozen_before,
                        resume=resume,
                        device="cuda:0",
                        progress=_progress,
                    )
                )
                results[lr_slug][variant] = track_result
                _progress(
                    "track_complete",
                    learning_rate=learning_rate,
                    loss_reduction_fraction=track_result["outcome"][
                        "loss_reduction_fraction"
                    ],
                    lr_slug=lr_slug,
                    non_worsened=track_result["outcome"][
                        "non_worsened_sample_count"
                    ],
                    variant=variant,
                )
    except BaseException as exc:
        execution_error = exc
        execution_traceback = traceback.format_exc()

    frozen_after = parameter_state_sha256(
        iter(model.named_parameters())
    )
    prevalidation = {
        "captured_at": _utc_now(),
        "data_preparation": data_report,
        "execution_error": (
            None
            if execution_error is None
            else f"{type(execution_error).__name__}: {execution_error}"
        ),
        "execution_traceback": execution_traceback,
        "frozen_parameter_sha256_after": frozen_after,
        "frozen_parameter_sha256_before": frozen_before,
        "model_load": model_report,
        "phase_d_frozen": phase_d,
        "phase_e4_frozen": phase_e4,
        "schema_version": PHASE_E5_SCHEMA,
        "tracks": {
            lr_slug: {
                variant: dict(value)
                for variant, value in variants.items()
            }
            for lr_slug, variants in results.items()
        },
    }
    atomic_write_json(
        output / "pre_validation_result.json",
        prevalidation,
    )
    _progress("frozen_hash_after", sha256=frozen_after)
    if execution_error is not None:
        del prepared, upstream_cfg, model
        gc.collect()
        torch.cuda.empty_cache()
        raise PhaseE5GateError(
            "Gate E.5 track execution failed after frozen hash capture"
        ) from execution_error

    execution_checks: dict[str, dict[str, dict[str, bool]]] = {}
    artifacts: dict[str, dict[str, dict[str, Any]]] = {}
    performance: dict[str, dict[str, dict[str, bool]]] = {}
    paired_checks: dict[str, dict[str, bool]] = {}
    eligibility: dict[str, bool] = {}
    for lr_slug, _ in PHASE_E2_LR_GRID:
        execution_checks[lr_slug] = {}
        artifacts[lr_slug] = {}
        performance[lr_slug] = {}
        for variant in ("A0", "A1"):
            checks, track_artifacts = _track_checks(
                tracks[lr_slug][variant],
                results[lr_slug][variant],
            )
            execution_checks[lr_slug][variant] = checks
            artifacts[lr_slug][variant] = track_artifacts
            performance[lr_slug][variant] = performance_checks(
                results[lr_slug][variant]
            )
        a0 = results[lr_slug]["A0"]
        a1 = results[lr_slug]["A1"]
        paired_checks[lr_slug] = {
            "same_sample_ids": a0["sample_ids"] == a1["sample_ids"],
            "same_initial_adapter": (
                a0["initial_adapter_sha256"]
                == a1["initial_adapter_sha256"]
            ),
            "same_initial_multiflow_probe": (
                _initial_probe_signature(a0)
                == _initial_probe_signature(a1)
            ),
            "same_identity_schedule": (
                artifacts[lr_slug]["A0"][
                    "identity_schedule_sha256"
                ]
                == artifacts[lr_slug]["A1"][
                    "identity_schedule_sha256"
                ]
            ),
            "same_observed_objective_schedule": (
                artifacts[lr_slug]["A0"][
                    "observed_schedule_sha256"
                ]
                == artifacts[lr_slug]["A1"][
                    "observed_schedule_sha256"
                ]
            ),
            "same_parameter_count": (
                a0["trainable_parameter_count"]
                == a1["trainable_parameter_count"]
            ),
            "same_training_budget": (
                a0["completed_steps"]
                == a1["completed_steps"]
                == OBJECTIVE_AGGREGATION_UPDATES
                and a0["completed_objectives"]
                == a1["completed_objectives"]
                == OBJECTIVE_AGGREGATION_UPDATES
                * OBJECTIVES_PER_UPDATE
            ),
        }
        eligibility[lr_slug] = (
            all(execution_checks[lr_slug]["A0"].values())
            and all(execution_checks[lr_slug]["A1"].values())
            and all(performance[lr_slug]["A0"].values())
            and all(performance[lr_slug]["A1"].values())
            and all(paired_checks[lr_slug].values())
        )

    selected_slug = select_smallest_eligible_lr(eligibility)
    all_results = [
        result
        for variants in results.values()
        for result in variants.values()
    ]
    cross_checks = {
        "all_initial_adapter_sha_equal": (
            len(
                {
                    str(result["initial_adapter_sha256"])
                    for result in all_results
                }
            )
            == 1
        ),
        "all_initial_multiflow_probes_equal": (
            len(
                {
                    _initial_probe_signature(result)
                    for result in all_results
                }
            )
            == 1
        ),
        "all_identity_schedules_equal_frozen": (
            {
                str(result["identity_schedule_sha256"])
                for result in all_results
            }
            == {PHASE_E5_FROZEN_IDENTITY_SCHEDULE_SHA256}
        ),
        "all_observed_objective_schedules_equal": (
            len(
                {
                    str(result["train_flow_schedule_sha256"])
                    for result in all_results
                }
            )
            == 1
        ),
        "all_sample_ids_equal": (
            len(
                {
                    tuple(result["sample_ids"])
                    for result in all_results
                }
            )
            == 1
        ),
        "frozen_fastwam_unchanged": frozen_before == frozen_after,
        "phase_e4_artifacts_unchanged": all(
            sha256_file(PHASE_E4_ROOT / name) == expected
            for name, expected in PHASE_E4_FROZEN_ARTIFACTS.items()
        ),
    }
    gate_passed = (
        all(
            all(checks.values())
            for variants in execution_checks.values()
            for checks in variants.values()
        )
        and all(
            all(checks.values()) for checks in paired_checks.values()
        )
        and all(cross_checks.values())
        and selected_slug is not None
    )
    learning_rates = dict(PHASE_E2_LR_GRID)
    result = {
        "completed_at": _utc_now(),
        "config": cfg.to_dict(),
        "config_fingerprint": cfg.fingerprint,
        "cross_checks": cross_checks,
        "data_preparation": data_report,
        "determinism": {
            "cublas_workspace_config": ":4096:8",
            "deterministic_algorithms": True,
            "flash_sdp": False,
            "math_sdp": True,
            "mem_efficient_sdp": False,
            "tf32": False,
        },
        "eligibility": eligibility,
        "frozen_parameter_sha256_after": frozen_after,
        "frozen_parameter_sha256_before": frozen_before,
        "gate_e5_passed": gate_passed,
        "model_load": model_report,
        "paired_checks": paired_checks,
        "phase_d_frozen": phase_d,
        "phase_e4_frozen": phase_e4,
        "preregistered_gate": {
            "budget_matching": "optimizer_updates",
            "gradient_reduction": "arithmetic_mean",
            "heldout_flow_steps": list(
                OBJECTIVE_AGGREGATION_HELDOUT_FLOW_STEPS
            ),
            "identity_schedule_sha256": (
                PHASE_E5_FROZEN_IDENTITY_SCHEDULE_SHA256
            ),
            "learning_rates": [
                value for _, value in PHASE_E2_LR_GRID
            ],
            "max_catastrophic_samples": 0,
            "max_median_delta_hidden_ratio": 0.5,
            "max_sample_delta_hidden_ratio": 1.0,
            "min_loss_reduction_fraction": 0.1,
            "min_non_worsened_samples": 6,
            "objectives_per_update": OBJECTIVES_PER_UPDATE,
            "optimizer_updates_per_track": (
                OBJECTIVE_AGGREGATION_UPDATES
            ),
            "sample_count": 8,
            "selection_rule": (
                "smallest learning rate eligible for both A0 and A1"
            ),
            "training_flow_slot_end": (
                objective_aggregation_flow_slot(200, 8)
            ),
            "training_flow_slot_start": (
                objective_aggregation_flow_slot(1, 1)
            ),
            "training_flow_slot_offset": (
                OBJECTIVE_AGGREGATION_FLOW_SLOT_OFFSET
            ),
            "variants": ["A0", "A1"],
        },
        "schema_version": PHASE_E5_SCHEMA,
        "scope": {
            "development_outcomes_read": False,
            "future_rgb_frames_read": 0,
            "heldout_probe_objectives": 480,
            "learning_rate_count": 3,
            "matched_optimizer_update_budget": True,
            "ood_outcomes_read": False,
            "optimizer_updates": 1200,
            "rollout_started": False,
            "sample_count": 8,
            "single_gpu": True,
            "success_outcomes_read": False,
            "task_count": 1,
            "track_count": 6,
            "training_objectives": 9600,
            "uses_ground_truth_future": False,
        },
        "selected_learning_rate": (
            learning_rates[selected_slug]
            if selected_slug is not None
            else None
        ),
        "selected_lr_slug": selected_slug,
        "status": "passed" if gate_passed else "failed",
        "tracks": {
            lr_slug: {
                variant: {
                    "artifacts": artifacts[lr_slug][variant],
                    "execution_checks": (
                        execution_checks[lr_slug][variant]
                    ),
                    "performance_checks": performance[lr_slug][variant],
                    "result": dict(results[lr_slug][variant]),
                }
                for variant in ("A0", "A1")
            }
            for lr_slug, _ in PHASE_E2_LR_GRID
        },
    }
    del prepared, upstream_cfg, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_phase_e5_objective_aggregation(
    cfg: Thought3Config,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Run Gate E.5 while preserving valid positive or negative outcomes."""

    _assert_phase_e5_scope(cfg)
    _require_phase_e5_confirmation()
    output = ensure_thought3_output_path(cfg.experiment.output_dir)
    result_path = output / "gate_e5_result.json"
    status_path = output / "run_status.json"
    if result_path.is_file():
        existing = load_json(result_path)
        if resume and existing.get("gate_e5_passed") is True:
            return existing
        if resume:
            raise PhaseE5GateError(
                "existing Gate E.5 result failed; preserve this Run ID"
            )
        raise FileExistsError(
            f"Gate E.5 result exists; pass --resume: {result_path}"
        )
    if status_path.is_file() and not resume:
        raise PhaseE5GateError(
            "existing partial Gate E.5 requires --resume or a new Run ID"
        )
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        status_path,
        {
            "schema_version": PHASE_E5_SCHEMA,
            "started_at": _utc_now(),
            "status": "running",
        },
    )
    started = time.perf_counter()
    try:
        result = _run_phase_e5(cfg, resume=resume)
        result["gate_wall_s"] = time.perf_counter() - started
        atomic_write_json(result_path, result)
        if result["gate_e5_passed"] is not True:
            raise PhaseE5GateError(
                "Gate E.5 hard checks failed; inspect gate_e5_result.json"
            )
    except BaseException as exc:
        atomic_write_json(
            status_path,
            {
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at": _utc_now(),
                "gate_e5_passed": False,
                "result": (
                    str(result_path.resolve())
                    if result_path.is_file()
                    else None
                ),
                "schema_version": PHASE_E5_SCHEMA,
                "status": "failed",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    atomic_write_json(
        status_path,
        {
            "finished_at": _utc_now(),
            "gate_e5_passed": True,
            "result": str(result_path.resolve()),
            "schema_version": PHASE_E5_SCHEMA,
            "status": "passed",
        },
    )
    return result
