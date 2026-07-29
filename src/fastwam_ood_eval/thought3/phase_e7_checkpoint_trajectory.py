"""Gate E.7: read-only trajectory diagnosis of frozen E.6 checkpoints.

No optimizer or backward pass is allowed.  The primary panel uses previously
unused action-flow draws 6..10.  The already observed E.6 draws 1..5 are a
continuity panel only and cannot determine the diagnostic classification.
"""

from __future__ import annotations

import gc
import hashlib
import math
import os
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
from fastwam_ood_eval.thought3.phase_e5_objective_aggregation import (
    _initial_probe_signature,
)
from fastwam_ood_eval.thought3.phase_e6_fresh_cohort_replication import (
    PHASE_E6_CONFIG,
    PHASE_E6_CONFIG_FINGERPRINT,
    PHASE_E6_FROZEN_COHORT_SHA256,
    PHASE_E6_FROZEN_IDENTITY_SCHEDULE_SHA256,
    PHASE_E6_LEARNING_RATE,
    PHASE_E6_ROOT,
    derive_e6_track_config,
    paired_superiority_checks,
    replication_performance_checks,
    verify_frozen_fresh_cohort,
    verify_frozen_phase_e5,
)
from fastwam_ood_eval.thought3.phase_e_training_smoke import (
    _verify_phase_d_gate,
)
from fastwam_ood_eval.thought3.real_training import (
    _checkpoint_expected,
    _flow_objective_identity,
    build_real_adapter,
    evaluate_multiflow_probe_grid,
    multiflow_probe_grid_outcome,
    prepare_real_training_data,
)
from fastwam_ood_eval.thought3.safety import ensure_thought3_output_path


PHASE_E7_SCHEMA = "thought3.phase_e7.checkpoint_trajectory.v1"
PHASE_E7_CONFIG = Path(
    "configs/thought3/phase_e7_checkpoint_trajectory.yaml"
)
PHASE_E7_EXPERIMENT_NAME = "thought3_phase_e7_checkpoint_trajectory"
PHASE_E7_ROOT = Path(
    "outputs/thought3/phase_e7_checkpoint_trajectory_v1"
)
PHASE_E7_CONFIG_FINGERPRINT = (
    "3823a3403e2d94c4690cf210209e1b530388722446fe64220a79560c18209af2"
)
PHASE_E7_CHECKPOINT_STEPS = (50, 100, 150, 200)
PHASE_E7_EARLY_STEPS = (50, 100, 150)
PHASE_E7_PRIMARY_FLOW_STEPS = (6, 7, 8, 9, 10)
PHASE_E7_CONTINUITY_FLOW_STEPS = (1, 2, 3, 4, 5)
PHASE_E7_PRIMARY_ZERO_WEIGHT_POSITIONS: tuple[
    tuple[int, int], ...
] = ()
PHASE_E7_CONTINUITY_ZERO_WEIGHT_POSITIONS = ((8, 5),)
PHASE_E7_PRIMARY_IDENTITY_SCHEDULE_SHA256 = (
    "3361f17069cb79bea7a330181fc97ecc3adfa9f3473d55b60640ad4249752f68"
)
PHASE_E7_CONTINUITY_IDENTITY_SCHEDULE_SHA256 = (
    "94f54e530b7cf9ea4a8f178f8fa47afe3cab8769e652d65a5c0a25dcf085d739"
)
PHASE_E7_MIN_LATE_SAMPLE_COUNT_DROP = 2
PHASE_E7_E6_PREREG_COMMIT = (
    "cb6f311fe1154722eaaeaf1f02f26cfde4922d56"
)
PHASE_E7_E6_RESULT_COMMIT = (
    "e5eeb3b6763a100bb371f1f78a548e11f1e1205a"
)
PHASE_E7_FASTWAM_COMMIT = (
    "45d8e1458921d83f8ad6cf9ce993d371208dabd0"
)
PHASE_E6_FROZEN_ARTIFACTS = {
    "gate_e6_result.json": (
        "464d9d3e02c52c2b1f2838ce59fe71a9b35716884d4d1da4b3d0e2ad78b42af6"
    ),
    "run_status.json": (
        "b6dd1edf41375e4ecd5d6495976298b6246307eafe16946be6662a99cb3b9adc"
    ),
    "pre_validation_result.json": (
        "3639032aa3d8faed5fd20d9f5da313ee51fb7605cf81aac95465a78390d83ec2"
    ),
    "data_preparation.json": (
        "4f8c6d02c06a4f6a80bc01ec54e88c13a39d4bab4be9f04b7cb547347af552df"
    ),
    "logs/phase_e6.log": (
        "b888d48f3b45dedc7577f616a6910400d950d38aa38be75ebbacf4f8d90eb81d"
    ),
}
PHASE_E6_SAMPLE_PAYLOAD_SHA256 = (
    "f5e61fd99d68244d7fa3cca6cc1ff59aabc12317840e4832ff2595f9ff78252f"
)
PHASE_E6_CHECKPOINT_FILE_SHA256 = {
    "A0": {
        50: {
            "adapter.safetensors": (
                "36ec9e9e0394d2115af749358f6608fe0215c13c752200a419f4916c5584ba3a"
            ),
            "manifest.json": (
                "c2f6d4793038c9292f8774e3c72c336260dd7b989bcc1dcbbcf1547863f24c58"
            ),
            "optimizer.pt": (
                "2c60f67b18cf55618e0f559b666cd52b49534b35aed4b7097e14f5ace3cf4a69"
            ),
        },
        100: {
            "adapter.safetensors": (
                "b42e683e5463b6b28f9b15c400decca97ee3472aaaa43000351f932c4d770cf6"
            ),
            "manifest.json": (
                "d8730f48c926bc78f3bfec6745908bd49b9887fd8ee52335b7a12cdc905292c5"
            ),
            "optimizer.pt": (
                "4278f9691f10639df66d6b7919def64243f8ee14a0bf1dd1b7a98b4580690869"
            ),
        },
        150: {
            "adapter.safetensors": (
                "64efdff45f3ff20b68a197d2ed64af7ffc3d25b2d8f351ab647d28e9fc50393f"
            ),
            "manifest.json": (
                "e052ccb80f10a77bbd664c80809cdd7f25fdde5ca99eb7a87f8a4d215d69bc06"
            ),
            "optimizer.pt": (
                "082e5167cd8971a90dc20e72584065c7728a3c82dd9c2481ad92dcbc1d5d17d0"
            ),
        },
        200: {
            "adapter.safetensors": (
                "c8cdef567f0b41570db9e44f8afbaa182661be48c363c7af7d678c1dc8a9a292"
            ),
            "manifest.json": (
                "1d233ae7720aff8f06a12fc7d9b9dae134f47be5bcb6b74805f7d6d3345c8dcf"
            ),
            "optimizer.pt": (
                "87c8680ac795c8fbe6766bdabf6d3320c00b16bd0ffe03fa7514a93c6621049a"
            ),
        },
    },
    "A1": {
        50: {
            "adapter.safetensors": (
                "62437b947a0b29a4c4c608fc691e7da48c6fe7ccf2b4ae181eb7bc8d4ac40b32"
            ),
            "manifest.json": (
                "0af7df8555884ed20312c26b59b2f7f3378b2a48a516190774fb3ab2aa8d3b80"
            ),
            "optimizer.pt": (
                "3393fe9837fca55a384516b81eb6307d5c25cb918358d6decc71e853c7d9229a"
            ),
        },
        100: {
            "adapter.safetensors": (
                "18021db7419b9be4f182844bf9e78c9331dbd5f6d96456b8c66cabe6af12ca7f"
            ),
            "manifest.json": (
                "7c3e01e3ddf5755d0a0451f8648f1871586e52dd9000cfe75c88901a8121f9a3"
            ),
            "optimizer.pt": (
                "73689d942bffc7c9004d4eb46df577f291cfc44e2042c03e527efcaf5ec588ed"
            ),
        },
        150: {
            "adapter.safetensors": (
                "04f3af7c7be3166d14dd4e89ad872ff0a07d1e33f776dc96c50124180f81e743"
            ),
            "manifest.json": (
                "e8cd038be429fc80e1beca770ab629b06c17b8571e5f16cd70f826779da310f0"
            ),
            "optimizer.pt": (
                "66f8771d80c71b257e320c5fe0eaa29a1518a295d7d85f3da2aa0707d50bdcff"
            ),
        },
        200: {
            "adapter.safetensors": (
                "aa55622c03aafea05c1bfedcb8548df398b0912dcecba397741c190c6b01b78f"
            ),
            "manifest.json": (
                "82cfa32891dc7bfc252835e45b34ad6d3da026329ef8870733a9c8f9f69448f3"
            ),
            "optimizer.pt": (
                "72daa2fc60bac643f47fe3a1101bc548b0de2fef7c589bf1b3e3e7161b434fe1"
            ),
        },
    },
}


class PhaseE7GateError(RuntimeError):
    """Raised when the frozen read-only E.7 protocol is violated."""


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
                "phase": "E.7",
                "stage": stage,
                "time": _utc_now(),
                **payload,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _assert_phase_e7_scope(cfg: Thought3Config) -> None:
    e6 = load_thought3_config(PHASE_E6_CONFIG)
    expected = replace(
        e6,
        experiment=replace(
            e6.experiment,
            name=PHASE_E7_EXPERIMENT_NAME,
            output_dir=PHASE_E7_ROOT,
        ),
    )
    observed_payload = cfg.to_dict()
    expected_payload = expected.to_dict()
    observed_payload.pop("source_path")
    expected_payload.pop("source_path")
    if observed_payload != expected_payload:
        raise PhaseE7GateError(
            "Gate E.7 changes more than experiment name/output"
        )
    if (
        cfg.fingerprint != PHASE_E7_CONFIG_FINGERPRINT
        or cfg.experiment.name != PHASE_E7_EXPERIMENT_NAME
        or cfg.experiment.output_dir != PHASE_E7_ROOT
    ):
        raise PhaseE7GateError("Gate E.7 frozen config identity changed")


def _require_phase_e7_confirmation() -> None:
    if os.environ.get("CONFIRM_THOUGHT3_PHASE_E7") != "YES":
        raise PhaseE7GateError(
            "set CONFIRM_THOUGHT3_PHASE_E7=YES for real read-only probes"
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
        raise PhaseE7GateError(
            "Gate E.7 requires clean project/FastWAM repositories and "
            "the frozen FastWAM commit"
        )
    return provenance


def probe_identity_schedule_sha256(
    sample_ids: Sequence[str],
    *,
    train_seed: int,
    flow_steps: Sequence[int],
) -> str:
    """Hash every probe RNG identity knowable before model loading."""

    normalized_ids = tuple(str(value) for value in sample_ids)
    normalized_steps = tuple(int(value) for value in flow_steps)
    if (
        len(normalized_ids) != 8
        or len(set(normalized_ids)) != 8
        or len(normalized_steps) != 5
        or len(set(normalized_steps)) != 5
        or any(value < 1 for value in normalized_steps)
    ):
        raise PhaseE7GateError(
            "Gate E.7 probe identity requires 8 samples and 5 flows"
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
) -> dict[str, str]:
    if (
        PHASE_E7_CHECKPOINT_STEPS != (50, 100, 150, 200)
        or PHASE_E7_EARLY_STEPS != (50, 100, 150)
        or PHASE_E7_PRIMARY_FLOW_STEPS != (6, 7, 8, 9, 10)
        or PHASE_E7_CONTINUITY_FLOW_STEPS != (1, 2, 3, 4, 5)
        or set(PHASE_E7_PRIMARY_FLOW_STEPS)
        & set(PHASE_E7_CONTINUITY_FLOW_STEPS)
        or PHASE_E7_PRIMARY_ZERO_WEIGHT_POSITIONS
        or PHASE_E7_CONTINUITY_ZERO_WEIGHT_POSITIONS != ((8, 5),)
    ):
        raise PhaseE7GateError("Gate E.7 frozen probe design changed")
    identity_sha256 = {
        "primary": probe_identity_schedule_sha256(
            sample_ids,
            train_seed=train_seed,
            flow_steps=PHASE_E7_PRIMARY_FLOW_STEPS,
        ),
        "continuity": probe_identity_schedule_sha256(
            sample_ids,
            train_seed=train_seed,
            flow_steps=PHASE_E7_CONTINUITY_FLOW_STEPS,
        ),
    }
    if identity_sha256 != {
        "primary": PHASE_E7_PRIMARY_IDENTITY_SCHEDULE_SHA256,
        "continuity": (
            PHASE_E7_CONTINUITY_IDENTITY_SCHEDULE_SHA256
        ),
    }:
        raise PhaseE7GateError(
            "Gate E.7 frozen probe RNG identity changed"
        )
    return identity_sha256


def _checkpoint_path(variant: str, step: int) -> Path:
    return (
        PHASE_E6_ROOT
        / "tracks"
        / variant.lower()
        / "checkpoints"
        / f"step_{step:08d}"
    )


def _checkpoint_hashes() -> dict[str, dict[int, dict[str, str]]]:
    result: dict[str, dict[int, dict[str, str]]] = {}
    for variant, by_step in PHASE_E6_CHECKPOINT_FILE_SHA256.items():
        result[variant] = {}
        for step, expected in by_step.items():
            checkpoint = _checkpoint_path(variant, step)
            observed_names = {
                path.name for path in checkpoint.iterdir()
            }
            if observed_names != set(expected):
                raise PhaseE7GateError(
                    "Gate E.6 checkpoint directory contents changed: "
                    f"{variant}/step-{step}"
                )
            result[variant][step] = {
                name: sha256_file(checkpoint / name)
                for name in expected
            }
    return result


def _false_boolean_paths(
    value: Mapping[str, Any],
    *,
    prefix: str,
) -> list[str]:
    paths: list[str] = []

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                visit(nested, f"{path}.{key}")
        elif item is False:
            paths.append(path)

    visit(value, prefix)
    return paths


def verify_frozen_phase_e6() -> dict[str, Any]:
    """Validate E.6 result and all eight checkpoint directories."""

    artifact_sha256: dict[str, str] = {}
    for name, expected in PHASE_E6_FROZEN_ARTIFACTS.items():
        path = PHASE_E6_ROOT / name
        if not path.is_file() or sha256_file(path) != expected:
            raise PhaseE7GateError(
                f"frozen Gate E.6 artifact changed/missing: {path}"
            )
        artifact_sha256[str(path)] = expected
    result = load_json(PHASE_E6_ROOT / "gate_e6_result.json")
    status = load_json(PHASE_E6_ROOT / "run_status.json")
    false_paths: list[str] = []
    for key in (
        "cross_checks",
        "paired_checks",
        "paired_superiority_checks",
    ):
        false_paths.extend(
            _false_boolean_paths(result[key], prefix=key)
        )
    for variant in ("A0", "A1"):
        false_paths.extend(
            _false_boolean_paths(
                result["tracks"][variant]["execution_checks"],
                prefix=f"{variant}.execution",
            )
        )
        false_paths.extend(
            _false_boolean_paths(
                result["tracks"][variant]["performance_checks"],
                prefix=f"{variant}.performance",
            )
        )
    expected_false = [
        (
            "A0.performance."
            "at_least_6_of_8_samples_non_worsened"
        )
    ]
    if (
        result.get("schema_version")
        != "thought3.phase_e6.fresh_cohort_replication.v1"
        or result.get("config_fingerprint")
        != PHASE_E6_CONFIG_FINGERPRINT
        or result.get("status") != "failed"
        or result.get("gate_e6_passed") is not False
        or status.get("status") != "failed"
        or status.get("gate_e6_passed") is not False
        or false_paths != expected_false
        or result["fresh_cohort"]["cohort_sha256"]
        != PHASE_E6_FROZEN_COHORT_SHA256
        or result["fresh_cohort"]["identity_schedule_sha256"]
        != PHASE_E6_FROZEN_IDENTITY_SCHEDULE_SHA256
        or float(result["tested_learning_rate"])
        != PHASE_E6_LEARNING_RATE
    ):
        raise PhaseE7GateError(
            "Gate E.6 is not the frozen valid-negative parent"
        )

    checkpoint_hashes = _checkpoint_hashes()
    if checkpoint_hashes != PHASE_E6_CHECKPOINT_FILE_SHA256:
        raise PhaseE7GateError("Gate E.6 checkpoint file hash changed")
    checkpoint_evidence: dict[str, dict[str, Any]] = {}
    for variant in ("A0", "A1"):
        checkpoint_evidence[variant] = {}
        for step in PHASE_E7_CHECKPOINT_STEPS:
            checkpoint = _checkpoint_path(variant, step)
            manifest = load_json(checkpoint / "manifest.json")
            extra = manifest["extra"]
            if (
                manifest["variant"] != variant
                or int(manifest["global_step"]) != step
                or int(manifest["sample_cursor"]) != step * 8
                or extra.get("gate_e6_fresh_cohort_replication")
                is not True
                or int(extra["objective_count"]) != step * 8
                or int(extra["training_flow_slot_offset"]) != 31_000
                or extra["identity_schedule_sha256"]
                != PHASE_E6_FROZEN_IDENTITY_SCHEDULE_SHA256
                or bool(extra["uses_ground_truth_future_input"])
            ):
                raise PhaseE7GateError(
                    f"Gate E.6 checkpoint provenance changed: "
                    f"{variant}/step-{step}"
                )
            checkpoint_evidence[variant][str(step)] = {
                "files_sha256": checkpoint_hashes[variant][step],
                "path": str(checkpoint),
            }
    return {
        "artifact_sha256": artifact_sha256,
        "checkpoint_evidence": checkpoint_evidence,
        "checkpoint_file_sha256": checkpoint_hashes,
        "gate_e6_passed": False,
        "known_before_e7": {
            "a0_step200_loss_reduction_fraction": (
                result["tracks"]["A0"]["result"]["outcome"][
                    "loss_reduction_fraction"
                ]
            ),
            "a0_step200_non_worsened_sample_count": 4,
            "a1_step200_loss_reduction_fraction": (
                result["tracks"]["A1"]["result"]["outcome"][
                    "loss_reduction_fraction"
                ]
            ),
            "a1_step200_non_worsened_sample_count": 7,
            "continuity_flow_steps": [1, 2, 3, 4, 5],
            "intermediate_checkpoint_outcomes_read": False,
        },
        "preregister_commit": PHASE_E7_E6_PREREG_COMMIT,
        "result_commit": PHASE_E7_E6_RESULT_COMMIT,
        "root": str(PHASE_E6_ROOT),
        "sample_ids": list(result["fresh_cohort"]["sample_ids"]),
        "step200_outcomes": {
            variant: dict(
                result["tracks"][variant]["result"]["outcome"]
            )
            for variant in ("A0", "A1")
        },
    }


def _probe_checks(
    probe: Mapping[str, Any],
    *,
    flow_steps: Sequence[int],
    expected_zero_weight_positions: Sequence[tuple[int, int]],
    train_seed: int,
) -> dict[str, bool]:
    rows = list(probe["per_objective"])
    sample_ids = list(probe["sample_ids"])
    observed_zero_positions = [
        (
            sample_ids.index(str(row["base_sample_id"])) + 1,
            int(row["flow_step"]),
        )
        for row in rows
        if float(row["action_weight"]) == 0
    ]
    expected_grid = {
        (str(sample_id), int(flow_step))
        for sample_id in sample_ids
        for flow_step in flow_steps
    }
    identity_exact = True
    for row in rows:
        expected_identity = _flow_objective_identity(
            base_sample_id=str(row["base_sample_id"]),
            train_seed=train_seed,
            flow_step=int(row["flow_step"]),
        )
        identity_exact = identity_exact and all(
            row.get(key) == expected_identity[key]
            for key in (
                "action_noise_seed",
                "action_timestep_seed",
                "flow_objective_sha256",
            )
        )
    return {
        "complete_probe_grid": (
            list(probe["flow_steps"]) == list(flow_steps)
            and int(probe["flow_objective_count"])
            == len(sample_ids) * len(flow_steps)
            and len(rows) == len(sample_ids) * len(flow_steps)
            and {
                (
                    str(row["base_sample_id"]),
                    int(row["flow_step"]),
                )
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
        "probe_rng_identity_exact": (
            identity_exact
            and int(probe.get("train_seed", -1)) == train_seed
            and probe.get("identity_schedule_sha256")
            == probe_identity_schedule_sha256(
                sample_ids,
                train_seed=train_seed,
                flow_steps=flow_steps,
            )
        ),
        "no_ground_truth_future": (
            probe.get("uses_ground_truth_future_input") is False
        ),
        "zero_weight_positions_exact": (
            observed_zero_positions
            == list(expected_zero_weight_positions)
            and int(probe["zero_weight_objective_count"])
            == len(expected_zero_weight_positions)
        ),
        "zero_weight_loss_exact": all(
            float(row["action_weight"]) != 0
            or float(row["action_loss"]) == 0
            for row in rows
        ),
    }


def classify_a0_trajectory(
    checkpoints: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen late-over-training classification rule."""

    if set(checkpoints) != set(PHASE_E7_CHECKPOINT_STEPS):
        raise PhaseE7GateError("E.7 A0 trajectory checkpoint set changed")
    early_eligible = [
        step
        for step in PHASE_E7_EARLY_STEPS
        if all(checkpoints[step]["performance_checks"].values())
    ]
    earliest = min(early_eligible) if early_eligible else None
    endpoint_eligible = all(
        checkpoints[200]["performance_checks"].values()
    )
    comparison: dict[str, Any] | None = None
    if earliest is not None:
        early_outcome = checkpoints[earliest]["outcome"]
        endpoint_outcome = checkpoints[200]["outcome"]
        comparison = {
            "earliest_stable_step": earliest,
            "endpoint_minus_early_mean_action_loss": (
                float(endpoint_outcome["final_mean_action_loss"])
                - float(early_outcome["final_mean_action_loss"])
            ),
            "mean_loss_increased": (
                float(endpoint_outcome["final_mean_action_loss"])
                > float(early_outcome["final_mean_action_loss"])
            ),
            "non_worsened_sample_count_drop": (
                int(early_outcome["non_worsened_sample_count"])
                - int(
                    endpoint_outcome[
                        "non_worsened_sample_count"
                    ]
                )
            ),
        }
    if endpoint_eligible:
        classification = "not_supported_endpoint_stable"
        material_degradation = False
    elif earliest is None:
        classification = "not_supported_no_earlier_stable_checkpoint"
        material_degradation = False
    else:
        assert comparison is not None
        material_degradation = (
            int(comparison["non_worsened_sample_count_drop"])
            >= PHASE_E7_MIN_LATE_SAMPLE_COUNT_DROP
            and bool(comparison["mean_loss_increased"])
        )
        classification = (
            "late_overtraining_supported"
            if material_degradation
            else "not_supported_no_material_late_degradation"
        )
    return {
        "classification": classification,
        "comparison_to_earliest_stable": comparison,
        "earliest_a0_stable_step": earliest,
        "early_a0_stable_steps": early_eligible,
        "endpoint_a0_stable": endpoint_eligible,
        "late_material_degradation": material_degradation,
        "min_late_sample_count_drop": (
            PHASE_E7_MIN_LATE_SAMPLE_COUNT_DROP
        ),
    }


def diagnostic_joint_candidate_steps(
    tracks: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Return primary-panel candidates under the frozen joint rule."""

    if set(tracks) != {"A0", "A1"}:
        raise PhaseE7GateError("E.7 candidate tracks changed")
    candidates: list[int] = []
    paired_by_step: dict[str, dict[str, Any]] = {}
    for step in PHASE_E7_CHECKPOINT_STEPS:
        a0 = tracks["A0"][step]
        a1 = tracks["A1"][step]
        paired_checks, paired_values = paired_superiority_checks(
            {"outcome": a0["outcome"]},
            {"outcome": a1["outcome"]},
        )
        paired_by_step[str(step)] = {
            "checks": paired_checks,
            "values": paired_values,
        }
        if (
            all(a0["performance_checks"].values())
            and all(a1["performance_checks"].values())
            and all(paired_checks.values())
        ):
            candidates.append(step)
    return {
        "diagnostic_candidate_steps": candidates,
        "earliest_diagnostic_candidate_step": (
            min(candidates) if candidates else None
        ),
        "paired_by_step": paired_by_step,
        "selection_status": (
            "post_run_diagnostic_candidate_only"
            if candidates
            else "no_joint_diagnostic_candidate"
        ),
    }


def _evaluate_panel(
    cfg: Thought3Config,
    *,
    model: Any,
    adapter: Any,
    samples: Sequence[Any],
    flow_steps: Sequence[int],
    device: str,
) -> dict[str, Any]:
    if any(parameter.grad is not None for parameter in adapter.parameters()):
        raise PhaseE7GateError(
            "Gate E.7 Adapter unexpectedly has gradients before probe"
        )
    injector = ActionEncoderFutureInjector(
        model.action_expert.action_encoder,
        adapter,
    )
    try:
        import torch

        with torch.inference_mode():
            result = evaluate_multiflow_probe_grid(
                cfg,
                model,
                adapter,
                injector,
                samples,
                flow_steps=flow_steps,
                device=device,
            )
    finally:
        injector.close()
    if any(parameter.grad is not None for parameter in adapter.parameters()):
        raise PhaseE7GateError(
            "Gate E.7 read-only probe produced Adapter gradients"
        )
    for row in result["per_objective"]:
        identity = _flow_objective_identity(
            base_sample_id=str(row["base_sample_id"]),
            train_seed=cfg.training.train_seed,
            flow_step=int(row["flow_step"]),
        )
        row.update(identity)
    result["identity_schedule_sha256"] = (
        probe_identity_schedule_sha256(
            result["sample_ids"],
            train_seed=cfg.training.train_seed,
            flow_steps=flow_steps,
        )
    )
    result["train_seed"] = cfg.training.train_seed
    return result


def _run_phase_e7(
    cfg: Thought3Config,
    *,
    execution_repository: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np
    import torch

    from fastwam_ood_eval.thought3.model_wrapper import (
        parameter_state_sha256,
    )

    _assert_phase_e7_scope(cfg)
    _require_phase_e7_confirmation()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise PhaseE7GateError(
            "Gate E.7 requires exactly one CUDA-visible GPU"
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise PhaseE7GateError(
            "Gate E.7 requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
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
    e6 = verify_frozen_phase_e6()
    probe_identity_sha256 = _assert_frozen_probe_design(
        e6["sample_ids"],
        train_seed=cfg.training.train_seed,
    )
    e5 = verify_frozen_phase_e5()
    cohort = verify_frozen_fresh_cohort(
        cfg,
        e5_sample_ids=e5["sample_ids"],
    )
    if list(cohort["sample_ids"]) != list(e6["sample_ids"]):
        raise PhaseE7GateError(
            "Gate E.7 fresh-cohort identity differs from frozen E.6"
        )
    phase_d = _verify_phase_d_gate(cfg)
    e6_cfg = load_thought3_config(PHASE_E6_CONFIG)
    track_cfgs = {
        variant: derive_e6_track_config(e6_cfg, variant=variant)
        for variant in ("A0", "A1")
    }
    checkpoint_hashes_before = _checkpoint_hashes()
    _progress(
        "frozen_inputs_verified",
        checkpoint_count=8,
        primary_flow_steps=list(PHASE_E7_PRIMARY_FLOW_STEPS),
        primary_identity_schedule_sha256=(
            probe_identity_sha256["primary"]
        ),
        source_gate_e6_sha256=PHASE_E6_FROZEN_ARTIFACTS[
            "gate_e6_result.json"
        ],
    )

    _progress("model_load_started", device="cuda:0")
    model, upstream_cfg, model_report = _load_upstream_model(cfg)
    torch.cuda.synchronize("cuda:0")
    model_report["load_peak_mib"] = (
        int(torch.cuda.max_memory_allocated("cuda:0")) / 2**20
    )
    _progress("model_loaded", load_peak_mib=model_report["load_peak_mib"])
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
        or set(samples_by_id) != set(e6["sample_ids"])
    ):
        raise PhaseE7GateError("Gate E.7 data-access audit failed")
    samples = tuple(
        samples_by_id[sample_id] for sample_id in e6["sample_ids"]
    )
    _progress("probe_data_ready", samples=len(samples))

    frozen_before = parameter_state_sha256(
        iter(model.named_parameters())
    )
    initial_adapter_sha: dict[str, str] = {}
    initial_probes: dict[str, dict[str, Any]] = {}
    checkpoint_results: dict[
        str, dict[int, dict[str, Any]]
    ] = {"A0": {}, "A1": {}}
    execution_error: BaseException | None = None
    execution_traceback: str | None = None
    try:
        for variant in ("A0", "A1"):
            adapter = build_real_adapter(
                track_cfgs[variant],
                device="cuda:0",
            )
            initial_adapter_sha[variant] = adapter_state_sha256(
                adapter.state_dict()
            )
            try:
                initial_probes[variant] = {
                    "primary": _evaluate_panel(
                        track_cfgs[variant],
                        model=model,
                        adapter=adapter,
                        samples=samples,
                        flow_steps=PHASE_E7_PRIMARY_FLOW_STEPS,
                        device="cuda:0",
                    ),
                    "continuity": _evaluate_panel(
                        track_cfgs[variant],
                        model=model,
                        adapter=adapter,
                        samples=samples,
                        flow_steps=PHASE_E7_CONTINUITY_FLOW_STEPS,
                        device="cuda:0",
                    ),
                }
            finally:
                del adapter
                torch.cuda.empty_cache()
            _progress("initial_probe_complete", variant=variant)

        for variant in ("A0", "A1"):
            for step in PHASE_E7_CHECKPOINT_STEPS:
                adapter = build_real_adapter(
                    track_cfgs[variant],
                    device="cuda:0",
                )
                if (
                    adapter_state_sha256(adapter.state_dict())
                    != initial_adapter_sha[variant]
                ):
                    raise PhaseE7GateError(
                        "Gate E.7 initial Adapter state drifted"
                    )
                checkpoint = _checkpoint_path(variant, step)
                manifest = load_adapter_checkpoint(
                    checkpoint,
                    adapter=adapter,
                    expected=_checkpoint_expected(
                        track_cfgs[variant],
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
                    raise PhaseE7GateError(
                        "Gate E.7 checkpoint semantic provenance changed"
                    )
                try:
                    primary_probe = _evaluate_panel(
                        track_cfgs[variant],
                        model=model,
                        adapter=adapter,
                        samples=samples,
                        flow_steps=PHASE_E7_PRIMARY_FLOW_STEPS,
                        device="cuda:0",
                    )
                    continuity_probe = _evaluate_panel(
                        track_cfgs[variant],
                        model=model,
                        adapter=adapter,
                        samples=samples,
                        flow_steps=PHASE_E7_CONTINUITY_FLOW_STEPS,
                        device="cuda:0",
                    )
                finally:
                    del adapter
                    torch.cuda.empty_cache()
                primary_outcome = multiflow_probe_grid_outcome(
                    initial_probes[variant]["primary"],
                    primary_probe,
                    expected_flow_steps=PHASE_E7_PRIMARY_FLOW_STEPS,
                )
                continuity_outcome = multiflow_probe_grid_outcome(
                    initial_probes[variant]["continuity"],
                    continuity_probe,
                    expected_flow_steps=(
                        PHASE_E7_CONTINUITY_FLOW_STEPS
                    ),
                )
                checkpoint_results[variant][step] = {
                    "checkpoint": str(checkpoint),
                    "checkpoint_adapter_state_sha256": (
                        manifest.extra["adapter_state_sha256"]
                    ),
                    "continuity_outcome": continuity_outcome,
                    "continuity_probe": continuity_probe,
                    "primary_outcome": primary_outcome,
                    "primary_probe": primary_probe,
                    "step": step,
                    "variant": variant,
                }
                _progress(
                    "checkpoint_probe_complete",
                    primary_loss_reduction_fraction=(
                        primary_outcome["loss_reduction_fraction"]
                    ),
                    primary_non_worsened=(
                        primary_outcome[
                            "non_worsened_sample_count"
                        ]
                    ),
                    step=step,
                    variant=variant,
                )
    except BaseException as exc:
        execution_error = exc
        execution_traceback = traceback.format_exc()

    frozen_after = parameter_state_sha256(
        iter(model.named_parameters())
    )
    checkpoint_hashes_after = _checkpoint_hashes()
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
        "fresh_cohort_frozen": cohort,
        "initial_adapter_sha256": initial_adapter_sha,
        "initial_probes": initial_probes,
        "model_load": model_report,
        "phase_d_frozen": phase_d,
        "phase_e6_frozen": e6,
        "probe_identity_schedule_sha256": probe_identity_sha256,
        "repository": dict(execution_repository),
        "schema_version": PHASE_E7_SCHEMA,
    }
    atomic_write_json(output / "pre_validation_result.json", prevalidation)
    _progress("frozen_hash_after", sha256=frozen_after)
    if execution_error is not None:
        del prepared, upstream_cfg, model
        gc.collect()
        torch.cuda.empty_cache()
        raise PhaseE7GateError(
            "Gate E.7 probe execution failed after frozen hash capture"
        ) from execution_error

    initial_checks = {
        "a0_a1_continuity_initial_exact": (
            _initial_probe_signature(
                {"initial_probe": initial_probes["A0"]["continuity"]}
            )
            == _initial_probe_signature(
                {"initial_probe": initial_probes["A1"]["continuity"]}
            )
        ),
        "a0_a1_primary_initial_exact": (
            _initial_probe_signature(
                {"initial_probe": initial_probes["A0"]["primary"]}
            )
            == _initial_probe_signature(
                {"initial_probe": initial_probes["A1"]["primary"]}
            )
        ),
        "initial_adapter_sha_equal": (
            initial_adapter_sha["A0"] == initial_adapter_sha["A1"]
        ),
        "initial_continuity_probes_valid": all(
            all(
                _probe_checks(
                    initial_probes[variant]["continuity"],
                    flow_steps=PHASE_E7_CONTINUITY_FLOW_STEPS,
                    expected_zero_weight_positions=(
                        PHASE_E7_CONTINUITY_ZERO_WEIGHT_POSITIONS
                    ),
                    train_seed=cfg.training.train_seed,
                ).values()
            )
            for variant in ("A0", "A1")
        ),
        "initial_primary_probes_valid": all(
            all(
                _probe_checks(
                    initial_probes[variant]["primary"],
                    flow_steps=PHASE_E7_PRIMARY_FLOW_STEPS,
                    expected_zero_weight_positions=(
                        PHASE_E7_PRIMARY_ZERO_WEIGHT_POSITIONS
                    ),
                    train_seed=cfg.training.train_seed,
                ).values()
            )
            for variant in ("A0", "A1")
        ),
        "initial_zero_gate_exact": all(
            float(row["gated_delta_norm"]) == 0
            for variant in ("A0", "A1")
            for panel in ("primary", "continuity")
            for row in initial_probes[variant][panel]["per_objective"]
        ),
    }
    analysis_tracks: dict[str, dict[int, dict[str, Any]]] = {
        "A0": {},
        "A1": {},
    }
    all_probe_checks_pass = True
    for variant in ("A0", "A1"):
        for step in PHASE_E7_CHECKPOINT_STEPS:
            record = checkpoint_results[variant][step]
            primary_checks = _probe_checks(
                record["primary_probe"],
                flow_steps=PHASE_E7_PRIMARY_FLOW_STEPS,
                expected_zero_weight_positions=(
                    PHASE_E7_PRIMARY_ZERO_WEIGHT_POSITIONS
                ),
                train_seed=cfg.training.train_seed,
            )
            continuity_checks = _probe_checks(
                record["continuity_probe"],
                flow_steps=PHASE_E7_CONTINUITY_FLOW_STEPS,
                expected_zero_weight_positions=(
                    PHASE_E7_CONTINUITY_ZERO_WEIGHT_POSITIONS
                ),
                train_seed=cfg.training.train_seed,
            )
            all_probe_checks_pass = (
                all_probe_checks_pass
                and all(primary_checks.values())
                and all(continuity_checks.values())
            )
            analysis_tracks[variant][step] = {
                **record,
                "continuity_probe_checks": continuity_checks,
                "performance_checks": (
                    replication_performance_checks(
                        variant,
                        {"outcome": record["primary_outcome"]},
                    )
                ),
                "primary_probe_checks": primary_checks,
                "outcome": record["primary_outcome"],
            }
    trajectory = classify_a0_trajectory(analysis_tracks["A0"])
    candidate = diagnostic_joint_candidate_steps(analysis_tracks)
    continuity_analysis_tracks: dict[
        str, dict[int, dict[str, Any]]
    ] = {"A0": {}, "A1": {}}
    for variant in ("A0", "A1"):
        for step in PHASE_E7_CHECKPOINT_STEPS:
            record = analysis_tracks[variant][step]
            continuity_analysis_tracks[variant][step] = {
                "outcome": record["continuity_outcome"],
                "performance_checks": replication_performance_checks(
                    variant,
                    {"outcome": record["continuity_outcome"]},
                ),
            }
    continuity_trajectory = classify_a0_trajectory(
        continuity_analysis_tracks["A0"]
    )
    continuity_checks = {
        f"{variant.lower()}_step200_outcome_exact_e6": (
            analysis_tracks[variant][200]["continuity_outcome"]
            == e6["step200_outcomes"][variant]
        )
        for variant in ("A0", "A1")
    }
    repository_after = _verify_execution_repository()
    cross_checks = {
        "all_initial_checks_passed": all(initial_checks.values()),
        "all_probe_checks_passed": all_probe_checks_pass,
        "checkpoint_files_unchanged": (
            checkpoint_hashes_before
            == checkpoint_hashes_after
            == PHASE_E6_CHECKPOINT_FILE_SHA256
        ),
        "continuity_step200_reproduces_e6": all(
            continuity_checks.values()
        ),
        "frozen_fastwam_has_no_grad": all(
            parameter.grad is None for parameter in model.parameters()
        ),
        "frozen_fastwam_not_trainable": not any(
            parameter.requires_grad for parameter in model.parameters()
        ),
        "frozen_fastwam_unchanged": frozen_before == frozen_after,
        "no_backward_called": True,
        "no_optimizer_created": True,
        "phase_e6_root_artifacts_unchanged": all(
            sha256_file(PHASE_E6_ROOT / name) == expected
            for name, expected in PHASE_E6_FROZEN_ARTIFACTS.items()
        ),
        "repository_provenance_unchanged": (
            repository_after == dict(execution_repository)
        ),
    }
    engineering_passed = all(cross_checks.values())
    result = {
        "checkpoint_steps": list(PHASE_E7_CHECKPOINT_STEPS),
        "completed_at": _utc_now(),
        "config": cfg.to_dict(),
        "config_fingerprint": cfg.fingerprint,
        "continuity_checks": continuity_checks,
        "continuity_panel": {
            "descriptive_a0_trajectory": continuity_trajectory,
            "flow_steps": list(PHASE_E7_CONTINUITY_FLOW_STEPS),
            "known_step200_outcome_before_e7": True,
            "role": "descriptive_reproduction_only",
            "used_for_classification": False,
        },
        "cross_checks": cross_checks,
        "data_preparation": data_report,
        "diagnostic_candidate": candidate,
        "diagnostic_classification": trajectory,
        "engineering_passed": engineering_passed,
        "frozen_parameter_sha256_after": frozen_after,
        "frozen_parameter_sha256_before": frozen_before,
        "fresh_cohort_frozen": cohort,
        "gate_e7_passed": engineering_passed,
        "initial_adapter_sha256": initial_adapter_sha,
        "initial_checks": initial_checks,
        "initial_probes": initial_probes,
        "model_load": model_report,
        "phase_d_frozen": phase_d,
        "phase_e6_frozen": e6,
        "probe_identity_schedule_sha256": probe_identity_sha256,
        "repository_after": repository_after,
        "repository_before": dict(execution_repository),
        "primary_panel": {
            "flow_steps": list(PHASE_E7_PRIMARY_FLOW_STEPS),
            "intermediate_outcomes_known_before_e7": False,
            "role": "frozen_primary_diagnostic",
            "used_for_classification": True,
            "zero_weight_positions": [],
        },
        "preregistered_diagnostic": {
            "a0_early_steps": list(PHASE_E7_EARLY_STEPS),
            "a0_stability_rule": {
                "max_catastrophic_samples": 0,
                "max_median_delta_hidden_ratio": 0.5,
                "max_objective_delta_hidden_ratio": 1.0,
                "min_loss_reduction_fraction": 0.0,
                "min_non_worsened_samples": 6,
            },
            "a1_absolute_rule": {
                "max_catastrophic_samples": 0,
                "max_median_delta_hidden_ratio": 0.5,
                "max_objective_delta_hidden_ratio": 1.0,
                "min_loss_reduction_fraction": 0.1,
                "min_non_worsened_samples": 6,
            },
            "joint_candidate_rule": (
                "earliest step satisfying A0 stability, A1 absolute, "
                "and A1-vs-A0 >=10% mean / >=6-of-8 samples"
            ),
            "late_overtraining_rule": (
                "an early step passes A0 stability; step 200 fails; "
                "non-worsened count drops by >=2; step-200 mean loss "
                "is higher than the earliest stable checkpoint"
            ),
            "primary_flow_steps": list(
                PHASE_E7_PRIMARY_FLOW_STEPS
            ),
        },
        "schema_version": PHASE_E7_SCHEMA,
        "scope": {
            "backward_calls": 0,
            "checkpoint_count": 8,
            "checkpoint_probe_panels": 16,
            "development_outcomes_read": False,
            "future_rgb_frames_read": 0,
            "initial_probe_panels": 4,
            "ood_outcomes_read": False,
            "optimizer_steps": 0,
            "probe_objectives": 800,
            "rollout_started": False,
            "success_outcomes_read": False,
            "new_training_samples_consumed": 0,
            "training_objectives": 0,
            "training_samples_read_for_probe": 8,
            "uses_ground_truth_future": False,
        },
        "status": "complete" if engineering_passed else "invalid",
        "tracks": {
            variant: {
                str(step): value
                for step, value in analysis_tracks[variant].items()
            }
            for variant in ("A0", "A1")
        },
    }
    del prepared, upstream_cfg, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_phase_e7_checkpoint_trajectory(
    cfg: Thought3Config,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Run E.7 once; valid scientific classifications return success."""

    _assert_phase_e7_scope(cfg)
    _require_phase_e7_confirmation()
    output = ensure_thought3_output_path(cfg.experiment.output_dir)
    result_path = output / "gate_e7_result.json"
    status_path = output / "run_status.json"
    if result_path.is_file():
        existing = load_json(result_path)
        if resume and existing.get("engineering_passed") is True:
            return existing
        if resume:
            raise PhaseE7GateError(
                "existing Gate E.7 result is invalid; preserve this Run ID"
            )
        raise FileExistsError(
            f"Gate E.7 result exists; pass --resume: {result_path}"
        )
    if status_path.is_file():
        raise PhaseE7GateError(
            "partial Gate E.7 evidence must be preserved under this Run ID"
        )
    execution_repository = _verify_execution_repository()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        status_path,
        {
            "schema_version": PHASE_E7_SCHEMA,
            "started_at": _utc_now(),
            "status": "running",
        },
    )
    started = time.perf_counter()
    try:
        result = _run_phase_e7(
            cfg,
            execution_repository=execution_repository,
        )
        result["gate_wall_s"] = time.perf_counter() - started
        atomic_write_json(result_path, result)
        if result["engineering_passed"] is not True:
            raise PhaseE7GateError(
                "Gate E.7 engineering checks failed; inspect result"
            )
    except BaseException as exc:
        atomic_write_json(
            status_path,
            {
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at": _utc_now(),
                "gate_e7_passed": False,
                "result": (
                    str(result_path.resolve())
                    if result_path.is_file()
                    else None
                ),
                "schema_version": PHASE_E7_SCHEMA,
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
            "gate_e7_passed": True,
            "result": str(result_path.resolve()),
            "schema_version": PHASE_E7_SCHEMA,
            "status": "complete",
        },
    )
    return result
