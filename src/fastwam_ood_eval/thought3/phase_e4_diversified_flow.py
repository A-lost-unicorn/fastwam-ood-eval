"""Gate E.4: paired diversified-flow training on the frozen E.3 cohort.

This sequential engineering diagnostic changes exactly one optimizer input:
instead of reusing flow_step=0 for every visit, each of 200 visits receives a
unique deterministic action-flow slot.  A0/A1 and all learning rates share the
same sample/noise/timestep schedule.  Evaluation remains the frozen E.3
held-out flow grid 1..5 and never reads development, OOD, success, or rollout.
"""

from __future__ import annotations

import gc
import math
import os
import time
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

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
    PHASE_E2_FROZEN_ARTIFACTS,
    PHASE_E2_ROOT,
    PHASE_E2_SAMPLE_PAYLOAD_SHA256,
)
from fastwam_ood_eval.thought3.phase_e_training_smoke import (
    _verify_phase_d_gate,
)
from fastwam_ood_eval.thought3.real_training import (
    DIVERSIFIED_HELDOUT_FLOW_STEPS,
    DIVERSIFIED_TRAIN_FLOW_SLOT_OFFSET,
    _flow_objective_identity,
    diversified_flow_schedule_sha256,
    diversified_training_flow_slot,
    multiflow_subset_outcome,
    prepare_real_training_data,
    run_diversified_flow_training,
)
from fastwam_ood_eval.thought3.safety import (
    ensure_thought3_output_path,
)


PHASE_E4_SCHEMA = "thought3.phase_e4.diversified_flow.v1"
PHASE_E4_EXPERIMENT_NAME = (
    "thought3_phase_e4_diversified_flow_diagnostic"
)
PHASE_E4_ROOT = Path(
    "outputs/thought3/phase_e4_diversified_flow_v1"
)
PHASE_E4_CONFIG_FINGERPRINT = (
    "e8c67a088c2c78e85e86c0cc0fac011e23303c59559d98c44dbc7051bdf578d1"
)
PHASE_E3_V2_ROOT = Path(
    "outputs/thought3/phase_e3_multiflow_v2"
)
PHASE_E3_V2_CONFIG = Path(
    "configs/thought3/phase_e3_multiflow_diagnostic_v2.yaml"
)
PHASE_E3_V2_CONFIG_FINGERPRINT = (
    "eeab3e38c1fd7ce15afc0852c1cac1007455a5551758c37d068ad6ea470b392e"
)
PHASE_E3_V2_FROZEN_ARTIFACTS = {
    "gate_e3_result.json": (
        "517c1e0cfc198f0bc44ab03d0d59349f20131d5c00efd958dd10f67aee1defe3"
    ),
    "run_status.json": (
        "f1bfa70b18df2a9494a88dea52501659cfd10f7f368bf4531d7da12582dc70c3"
    ),
    "pre_validation_result.json": (
        "68b7af97b5e17473ddb76472fe22c95abf5e1ec06e54ed7baeff324a2918ec14"
    ),
    "data_preparation.json": (
        "0b505d9764cbf97e45fdebb9d95c68cbb4e3cd88bed2e0d73cebe95b1ce14ae6"
    ),
    "logs/phase_e3.log": (
        "861c4bc58ac2bd3d3729d30e72aba3886908d996e01eb3e8f14858007191becc"
    ),
}
PHASE_E4_EXPECTED_ZERO_WEIGHT_STEPS = (49, 142)


class PhaseE4GateError(RuntimeError):
    """Raised when Gate E.4 violates its frozen engineering protocol."""


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
                "phase": "E.4",
                "stage": stage,
                "time": _utc_now(),
                **payload,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _assert_phase_e4_scope(cfg: Thought3Config) -> None:
    """Reject any change beyond diversified optimizer flow slots."""

    _assert_phase_e2_scope(cfg)
    if cfg.experiment.name != PHASE_E4_EXPERIMENT_NAME:
        raise PhaseE4GateError("Gate E.4 experiment name changed")
    if cfg.experiment.output_dir != PHASE_E4_ROOT:
        raise PhaseE4GateError("Gate E.4 output directory changed")
    if cfg.fingerprint != PHASE_E4_CONFIG_FINGERPRINT:
        raise PhaseE4GateError("Gate E.4 config fingerprint changed")


def verify_frozen_phase_e3_v2() -> dict[str, Any]:
    """Validate the exact valid-negative E.3 result before model loading."""

    artifact_sha256: dict[str, str] = {}
    for name, expected_sha in PHASE_E3_V2_FROZEN_ARTIFACTS.items():
        path = PHASE_E3_V2_ROOT / name
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise PhaseE4GateError(
                f"frozen Gate E.3 v2 artifact changed/missing: {path}"
            )
        artifact_sha256[str(path)] = expected_sha
    for name, expected_sha in PHASE_E2_FROZEN_ARTIFACTS.items():
        if sha256_file(PHASE_E2_ROOT / name) != expected_sha:
            raise PhaseE4GateError(
                f"frozen Gate E.2 artifact changed: {name}"
            )

    cfg = load_thought3_config(PHASE_E3_V2_CONFIG)
    result = load_json(PHASE_E3_V2_ROOT / "gate_e3_result.json")
    status = load_json(PHASE_E3_V2_ROOT / "run_status.json")
    prevalidation = load_json(
        PHASE_E3_V2_ROOT / "pre_validation_result.json"
    )
    if (
        cfg.fingerprint != PHASE_E3_V2_CONFIG_FINGERPRINT
        or result.get("schema_version")
        != "thought3.phase_e3.multiflow.v2"
        or result.get("config_fingerprint")
        != PHASE_E3_V2_CONFIG_FINGERPRINT
        or result.get("status") != "failed"
        or result.get("gate_e3_passed") is not False
        or result.get("selected_lr_slug") is not None
        or result.get("selected_learning_rate") is not None
        or status.get("status") != "failed"
        or status.get("gate_e3_passed") is not False
        or prevalidation.get("execution_error") is not None
        or prevalidation.get("execution_traceback") is not None
    ):
        raise PhaseE4GateError(
            "Gate E.3 v2 is not the frozen valid-negative diagnostic"
        )
    if (
        any(bool(value) for value in result["eligibility"].values())
        or not all(bool(value) for value in result["initial_checks"].values())
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
                    "probe_checks"
                ].values()
            )
            for lr_slug, _ in PHASE_E2_LR_GRID
            for variant in ("A0", "A1")
        )
    ):
        raise PhaseE4GateError(
            "Gate E.3 v2 integrity checks are not all valid"
        )
    return {
        "artifact_sha256": artifact_sha256,
        "config_fingerprint": PHASE_E3_V2_CONFIG_FINGERPRINT,
        "gate_e3_passed": False,
        "root": str(PHASE_E3_V2_ROOT),
        "sample_ids": list(
            result["initial_probes"]["A0"]["sample_ids"]
        ),
    }


def derive_e4_track_config(
    cfg: Thought3Config,
    *,
    variant: str,
    lr_slug: str,
    learning_rate: float,
) -> Thought3Config:
    """Derive one of six matched E.4 A0/A1 × LR tracks."""

    expected = dict(PHASE_E2_LR_GRID)
    if (
        variant not in {"A0", "A1"}
        or lr_slug not in expected
        or learning_rate != expected[lr_slug]
    ):
        raise PhaseE4GateError(
            f"unsupported Gate E.4 track: {variant}/{lr_slug}"
        )
    derived = replace(
        cfg,
        variant=variant,
        experiment=replace(
            cfg.experiment,
            name=f"thought3_phase_e4_{variant.lower()}_{lr_slug}",
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
            variant: derive_e4_track_config(
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
            raise PhaseE4GateError(
                f"Gate E.4 A0/A1 recipe differs at {lr_slug}"
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


def _objective_schedule_signature(
    metrics: list[Mapping[str, Any]],
) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            int(row["global_step"]),
            str(row["base_sample_id"]),
            int(row["training_flow_slot"]),
            int(row["action_noise_seed"]),
            int(row["action_timestep_seed"]),
            str(row["flow_objective_sha256"]),
            float(row["timestep"]),
            float(row["action_weight"]),
        )
        for row in metrics
    )


def _track_checks(
    cfg: Thought3Config,
    result: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    metrics_path = Path(str(result["metrics"]))
    probe_path = Path(str(result["probe_metrics"]))
    metrics = load_jsonl(metrics_path)
    probes = load_jsonl(probe_path)
    expected_slots = [
        diversified_training_flow_slot(step)
        for step in range(1, 201)
    ]
    expected_zero_steps = [
        int(row["global_step"])
        for row in metrics
        if float(row["action_weight"]) == 0
    ]
    identities_match = all(
        all(
            row[field] == expected[field]
            for field in (
                "action_noise_seed",
                "action_timestep_seed",
                "flow_objective_sha256",
            )
        )
        for row in metrics
        for expected in (
            _flow_objective_identity(
                base_sample_id=str(row["base_sample_id"]),
                train_seed=cfg.training.train_seed,
                flow_step=int(row["training_flow_slot"]),
            ),
        )
    )
    recomputed_outcome = multiflow_subset_outcome(
        result["initial_probe"],
        result["final_probe"],
    )
    execution = {
        "complete_200_steps": (
            result.get("status") == "complete"
            and int(result.get("completed_steps", -1)) == 200
            and [int(row["global_step"]) for row in metrics]
            == list(range(1, 201))
        ),
        "paired_eight_sample_round_robin": (
            int(result.get("sample_count", -1)) == 8
            and len(set(result["sample_ids"])) == 8
            and all(
                str(row["base_sample_id"])
                == str(
                    result["sample_ids"][
                        (int(row["global_step"]) - 1) % 8
                    ]
                )
                for row in metrics
            )
        ),
        "unique_disjoint_training_flow_slots": (
            [int(row["training_flow_slot"]) for row in metrics]
            == expected_slots
            and len(set(expected_slots)) == 200
            and not set(expected_slots)
            & set(DIVERSIFIED_HELDOUT_FLOW_STEPS)
            and 0 not in expected_slots
        ),
        "objective_seed_identity_exact": identities_match,
        "schedule_sha_exact": (
            result["train_flow_schedule_sha256"]
            == diversified_flow_schedule_sha256(metrics)
        ),
        "frozen_zero_weight_steps_exact": (
            tuple(expected_zero_steps)
            == PHASE_E4_EXPECTED_ZERO_WEIGHT_STEPS
            and int(result["zero_weight_step_count"])
            == len(PHASE_E4_EXPECTED_ZERO_WEIGHT_STEPS)
            and all(
                float(row["action_weight"]) != 0
                or float(row["loss"]) == 0
                for row in metrics
            )
        ),
        "heldout_probe_schedule": (
            [int(row["global_step"]) for row in probes] == [0, 200]
            and all(
                list(row["flow_steps"])
                == list(DIVERSIFIED_HELDOUT_FLOW_STEPS)
                and int(row["flow_objective_count"]) == 40
                and int(row["sample_count"]) == 8
                and list(row["sample_ids"])
                == list(result["sample_ids"])
                for row in probes
            )
        ),
        "outcome_recomputes_exactly": (
            dict(result["outcome"]) == recomputed_outcome
        ),
        "first_step_gate_only": (
            float(metrics[0]["gradient_groups"]["gate"]["l2"]) > 0
            and int(
                metrics[0]["gradient_groups"]["non_gate"][
                    "nonzero_element_count"
                ]
            )
            == 0
        ),
        "second_step_projector_gradient": (
            int(
                metrics[1]["gradient_groups"]["future_projector"][
                    "nonzero_element_count"
                ]
            )
            > 0
        ),
        "second_step_attention_gradient": (
            int(
                metrics[1]["gradient_groups"]["attention"][
                    "nonzero_element_count"
                ]
            )
            > 0
        ),
        "first_non_gate_paths_at_step_2": (
            int(result["first_non_gate_nonzero_gradient_step"]) == 2
            and int(
                result["first_projector_nonzero_gradient_step"]
            )
            == 2
            and int(
                result["first_attention_nonzero_gradient_step"]
            )
            == 2
        ),
        "finite_trace": (
            all(
                not bool(row["nan_or_inf"])
                and math.isfinite(float(row["loss"]))
                and math.isfinite(float(row["action_weight"]))
                and math.isfinite(float(row["timestep"]))
                and all(
                    bool(group["finite"])
                    for group in row["gradient_groups"].values()
                )
                for row in metrics
            )
            and all(
                math.isfinite(float(row["mean_action_loss"]))
                for row in probes
            )
        ),
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
        ),
        "checkpoint_roundtrip": (
            result["checkpoint_roundtrip"].get("state_equal") is True
            and int(result["checkpoint_roundtrip"]["global_step"]) == 200
        ),
    }
    artifacts = {
        "files_sha256": {
            "manifest": sha256_file(
                cfg.experiment.output_dir / "training_manifest.json"
            ),
            "metrics": sha256_file(metrics_path),
            "probe_metrics": sha256_file(probe_path),
            "state": sha256_file(
                cfg.experiment.output_dir / "training_state.json"
            ),
        },
        "objective_schedule_signature": _objective_schedule_signature(
            metrics
        ),
        "output_dir": str(cfg.experiment.output_dir),
    }
    return execution, artifacts


def _run_phase_e4(
    cfg: Thought3Config,
    *,
    resume: bool,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from fastwam_ood_eval.thought3.model_wrapper import (
        parameter_state_sha256,
    )

    _assert_phase_e4_scope(cfg)
    if os.environ.get("CONFIRM_THOUGHT3_PHASE_E4") != "YES":
        raise PhaseE4GateError(
            "set CONFIRM_THOUGHT3_PHASE_E4=YES for real diversified-flow "
            "training"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise PhaseE4GateError(
            "Gate E.4 requires exactly one CUDA-visible GPU"
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise PhaseE4GateError(
            "Gate E.4 requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
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
    phase_e3 = verify_frozen_phase_e3_v2()
    phase_d = _verify_phase_d_gate(cfg)
    tracks = _derive_tracks(cfg)
    _progress(
        "frozen_inputs_verified",
        gate_e3_sha256=PHASE_E3_V2_FROZEN_ARTIFACTS[
            "gate_e3_result.json"
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
        or prepared_ids != set(phase_e3["sample_ids"])
    ):
        raise PhaseE4GateError("Gate E.4 data-access audit failed")

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
                track_result = run_diversified_flow_training(
                    tracks[lr_slug][variant],
                    model=model,
                    prepared=prepared,
                    frozen_parameter_sha256=frozen_before,
                    resume=resume,
                    device="cuda:0",
                    progress=_progress,
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
        "phase_e3_frozen": phase_e3,
        "schema_version": PHASE_E4_SCHEMA,
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
        raise PhaseE4GateError(
            "Gate E.4 track execution failed after frozen hash capture"
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
            "same_objective_schedule": (
                artifacts[lr_slug]["A0"][
                    "objective_schedule_signature"
                ]
                == artifacts[lr_slug]["A1"][
                    "objective_schedule_signature"
                ]
            ),
            "same_parameter_count": (
                a0["trainable_parameter_count"]
                == a1["trainable_parameter_count"]
            ),
            "same_training_budget": (
                a0["completed_steps"] == a1["completed_steps"] == 200
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
        "all_objective_schedules_equal": (
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
        "phase_e2_artifacts_unchanged": all(
            sha256_file(PHASE_E2_ROOT / name) == expected
            for name, expected in PHASE_E2_FROZEN_ARTIFACTS.items()
        ),
        "phase_e3_artifacts_unchanged": all(
            sha256_file(PHASE_E3_V2_ROOT / name) == expected
            for name, expected in PHASE_E3_V2_FROZEN_ARTIFACTS.items()
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
        "gate_e4_passed": gate_passed,
        "model_load": model_report,
        "paired_checks": paired_checks,
        "phase_d_frozen": phase_d,
        "phase_e3_frozen": phase_e3,
        "preregistered_gate": {
            "heldout_flow_steps": list(
                DIVERSIFIED_HELDOUT_FLOW_STEPS
            ),
            "learning_rates": [
                value for _, value in PHASE_E2_LR_GRID
            ],
            "max_catastrophic_samples": 0,
            "max_median_delta_hidden_ratio": 0.5,
            "max_sample_delta_hidden_ratio": 1.0,
            "min_loss_reduction_fraction": 0.1,
            "min_non_worsened_samples": 6,
            "sample_count": 8,
            "selection_rule": (
                "smallest learning rate eligible for both A0 and A1"
            ),
            "steps_per_track": 200,
            "training_flow_slot_end": (
                diversified_training_flow_slot(200)
            ),
            "training_flow_slot_start": (
                diversified_training_flow_slot(1)
            ),
            "training_flow_slot_offset": (
                DIVERSIFIED_TRAIN_FLOW_SLOT_OFFSET
            ),
            "variants": ["A0", "A1"],
        },
        "schema_version": PHASE_E4_SCHEMA,
        "scope": {
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


def run_phase_e4_diversified_flow(
    cfg: Thought3Config,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Run Gate E.4 while preserving valid pass/fail outcomes."""

    _assert_phase_e4_scope(cfg)
    output = ensure_thought3_output_path(cfg.experiment.output_dir)
    result_path = output / "gate_e4_result.json"
    status_path = output / "run_status.json"
    if result_path.is_file():
        existing = load_json(result_path)
        if resume and existing.get("gate_e4_passed") is True:
            return existing
        if resume:
            raise PhaseE4GateError(
                "existing Gate E.4 result failed; preserve this Run ID"
            )
        raise FileExistsError(
            f"Gate E.4 result exists; pass --resume: {result_path}"
        )
    if status_path.is_file() and not resume:
        raise PhaseE4GateError(
            "existing partial Gate E.4 requires --resume or a new Run ID"
        )
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        status_path,
        {
            "schema_version": PHASE_E4_SCHEMA,
            "started_at": _utc_now(),
            "status": "running",
        },
    )
    started = time.perf_counter()
    try:
        result = _run_phase_e4(cfg, resume=resume)
        result["gate_wall_s"] = time.perf_counter() - started
        atomic_write_json(result_path, result)
        if result["gate_e4_passed"] is not True:
            raise PhaseE4GateError(
                "Gate E.4 hard checks failed; inspect gate_e4_result.json"
            )
    except BaseException as exc:
        atomic_write_json(
            status_path,
            {
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at": _utc_now(),
                "gate_e4_passed": False,
                "result": (
                    str(result_path.resolve())
                    if result_path.is_file()
                    else None
                ),
                "schema_version": PHASE_E4_SCHEMA,
                "status": "failed",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    atomic_write_json(
        status_path,
        {
            "finished_at": _utc_now(),
            "gate_e4_passed": True,
            "result": str(result_path.resolve()),
            "schema_version": PHASE_E4_SCHEMA,
            "status": "passed",
        },
    )
    return result
