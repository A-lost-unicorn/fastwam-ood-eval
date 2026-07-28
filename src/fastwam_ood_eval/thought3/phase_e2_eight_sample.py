"""Gate E.2: eight-sample, train-only LR and hidden-scale diagnostic.

The gate reuses the frozen Phase D cache and never reads development, OOD, or
success outcomes.  It evaluates A0/A1 at three preregistered learning rates on
the same eight standard LIBERO train samples with fixed per-sample
noise/timesteps.  A candidate must improve fixed loss without producing an
unbounded BF16 correction to the frozen action hidden state.
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

from fastwam_ood_eval.thought3.config import Thought3Config, validate_config
from fastwam_ood_eval.thought3.io_utils import (
    atomic_write_json,
    load_json,
    load_jsonl,
    sha256_file,
)
from fastwam_ood_eval.thought3.phase_c_smoke import _load_upstream_model
from fastwam_ood_eval.thought3.phase_e_training_smoke import (
    OFFICIAL_LIBERO_REVISION,
    _verify_phase_d_gate,
)
from fastwam_ood_eval.thought3.real_training import (
    prepare_real_training_data,
    run_fixed_subset_training,
)
from fastwam_ood_eval.thought3.safety import ensure_thought3_output_path


PHASE_E2_SCHEMA = "thought3.phase_e2.eight_sample.v1"
PHASE_E2_LR_GRID: tuple[tuple[str, float], ...] = (
    ("lr_1e_04", 1e-4),
    ("lr_3e_04", 3e-4),
    ("lr_1e_03", 1e-3),
)
PHASE_E2_MIN_LOSS_REDUCTION_FRACTION = 0.10
PHASE_E2_MIN_NON_WORSENED_SAMPLES = 6
PHASE_E2_MAX_CATASTROPHIC_SAMPLES = 0
PHASE_E2_MAX_MEDIAN_DELTA_HIDDEN_RATIO = 0.50
PHASE_E2_MAX_SAMPLE_DELTA_HIDDEN_RATIO = 1.00


class PhaseE2GateError(RuntimeError):
    """Raised when Gate E.2 violates its frozen engineering protocol."""


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
                "phase": "E.2",
                "stage": stage,
                "time": _utc_now(),
                **payload,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _assert_phase_e2_scope(cfg: Thought3Config) -> None:
    """Reject silent expansion beyond the eight-sample diagnostic."""

    if cfg.runtime.backend != "fastwam" or cfg.runtime.device != "cuda:0":
        raise PhaseE2GateError(
            "Gate E.2 requires backend=fastwam and logical cuda:0"
        )
    if cfg.variant != "A1" or cfg.sampler.active_k != 1:
        raise PhaseE2GateError(
            "Gate E.2 orchestration config must be A1/active_k=1"
        )
    if tuple(cfg.sampler.cache_k) != (1, 2, 4):
        raise PhaseE2GateError("Gate E.2 requires the paired Phase D cache")
    if (
        cfg.cache.root
        != Path("outputs/thought3/cache/phase_d_libero_goal_task0_v1")
        or cfg.cache.pilot_limit != 32
        or cfg.cache.shard_size != 8
    ):
        raise PhaseE2GateError(
            "Gate E.2 is frozen to the 32-sample Phase D cache"
        )
    if len(cfg.data.dataset_roots) != 1:
        raise PhaseE2GateError("Gate E.2 accepts one standard LIBERO root")
    if cfg.data.dataset_revision != OFFICIAL_LIBERO_REVISION:
        raise PhaseE2GateError("Gate E.2 LIBERO revision mismatch")
    if cfg.data.inventory_path != Path(
        "outputs/thought3/phase_d_cache_smoke_v1/inventory.jsonl"
    ):
        raise PhaseE2GateError(
            "Gate E.2 inventory is not the frozen Phase D inventory"
        )
    if (
        cfg.data.split_seed != 3407
        or cfg.experiment.seed != 3407
        or cfg.training.train_seed != 3407
    ):
        raise PhaseE2GateError("Gate E.2 seeds must remain 3407")
    if (
        cfg.training.max_steps != 200
        or cfg.training.microbatch_size != 1
        or cfg.training.gradient_accumulation_steps != 1
        or cfg.training.checkpoint_interval != 50
    ):
        raise PhaseE2GateError(
            "Gate E.2 budget is 200 steps per track, batch 1, checkpoint 50"
        )
    if (
        cfg.training.learning_rate != 1e-4
        or cfg.training.weight_decay != 1e-2
        or cfg.training.gradient_checkpointing
        or cfg.training.gate_l2 != 0
    ):
        raise PhaseE2GateError(
            "Gate E.2 orchestration optimizer recipe changed"
        )
    if cfg.runtime.online_use_cache:
        raise PhaseE2GateError(
            "Gate E.2 forbids online training-cache generation"
        )


def derive_e2_track_config(
    cfg: Thought3Config,
    *,
    variant: str,
    lr_slug: str,
    learning_rate: float,
) -> Thought3Config:
    """Derive one of six matched A0/A1 × LR tracks."""

    expected = dict(PHASE_E2_LR_GRID)
    if (
        variant not in {"A0", "A1"}
        or lr_slug not in expected
        or learning_rate != expected[lr_slug]
    ):
        raise PhaseE2GateError(
            f"unsupported Gate E.2 track: {variant}/{lr_slug}/{learning_rate}"
        )
    derived = replace(
        cfg,
        variant=variant,
        experiment=replace(
            cfg.experiment,
            name=(
                f"thought3_phase_e2_{variant.lower()}_{lr_slug}"
            ),
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


def _matched_recipe_payload(cfg: Thought3Config) -> dict[str, Any]:
    payload = cfg.to_dict()
    payload.pop("source_path")
    payload.pop("variant")
    payload["experiment"].pop("name")
    payload["experiment"].pop("output_dir")
    payload["sampler"].pop("active_k")
    return payload


def _derive_tracks(
    cfg: Thought3Config,
) -> dict[str, dict[str, Thought3Config]]:
    tracks = {
        lr_slug: {
            variant: derive_e2_track_config(
                cfg,
                variant=variant,
                lr_slug=lr_slug,
                learning_rate=learning_rate,
            )
            for variant in ("A0", "A1")
        }
        for lr_slug, learning_rate in PHASE_E2_LR_GRID
    }
    for lr_slug, by_variant in tracks.items():
        if (
            _matched_recipe_payload(by_variant["A0"])
            != _matched_recipe_payload(by_variant["A1"])
        ):
            raise PhaseE2GateError(
                f"Gate E.2 A0/A1 recipe differs at {lr_slug}"
            )
    return tracks


def _initial_probe_signature(
    result: Mapping[str, Any],
) -> tuple[tuple[str, float, float, float], ...]:
    """Compare only zero-gate action-equivalent fields, not latent residuals."""

    return tuple(
        (
            str(row["base_sample_id"]),
            float(row["action_loss"]),
            float(row["gated_delta_norm"]),
            float(row["gated_delta_nonzero_fraction"]),
        )
        for row in result["initial_probe"]["per_sample"]
    )


def performance_checks(result: Mapping[str, Any]) -> dict[str, bool]:
    """Apply preregistered train-only loss and hidden-scale thresholds."""

    outcome = result["outcome"]
    return {
        "mean_loss_reduction_at_least_10_percent": (
            float(outcome["loss_reduction_fraction"])
            >= PHASE_E2_MIN_LOSS_REDUCTION_FRACTION
        ),
        "at_least_6_of_8_samples_non_worsened": (
            int(outcome["non_worsened_sample_count"])
            >= PHASE_E2_MIN_NON_WORSENED_SAMPLES
        ),
        "no_sample_loss_above_2x_initial": (
            int(outcome["catastrophic_sample_count"])
            <= PHASE_E2_MAX_CATASTROPHIC_SAMPLES
        ),
        "median_delta_hidden_at_most_0_5": (
            float(
                outcome[
                    "median_gated_delta_to_action_hidden_ratio"
                ]
            )
            <= PHASE_E2_MAX_MEDIAN_DELTA_HIDDEN_RATIO
        ),
        "max_delta_hidden_at_most_1_0": (
            float(
                outcome["max_gated_delta_to_action_hidden_ratio"]
            )
            <= PHASE_E2_MAX_SAMPLE_DELTA_HIDDEN_RATIO
        ),
    }


def select_smallest_eligible_lr(
    eligibility: Mapping[str, bool],
) -> str | None:
    """Conservatively select the first eligible LR in frozen ascending order."""

    expected = {slug for slug, _ in PHASE_E2_LR_GRID}
    if set(eligibility) != expected:
        raise PhaseE2GateError(
            "Gate E.2 eligibility does not cover the frozen LR grid"
        )
    for lr_slug, _ in PHASE_E2_LR_GRID:
        if bool(eligibility[lr_slug]):
            return lr_slug
    return None


def _track_checks(
    cfg: Thought3Config,
    result: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    metrics_path = Path(str(result["metrics"]))
    probe_path = Path(str(result["probe_metrics"]))
    metrics = load_jsonl(metrics_path)
    probes = load_jsonl(probe_path)
    first, second = metrics[0], metrics[1]
    execution = {
        "complete_200_steps": (
            result.get("status") == "complete"
            and int(result.get("completed_steps", -1)) == 200
            and [int(row["global_step"]) for row in metrics]
            == list(range(1, 201))
        ),
        "fixed_eight_sample_round_robin": (
            int(result.get("sample_count", -1)) == 8
            and len(set(result["sample_ids"])) == 8
            and all(
                str(row["base_sample_id"])
                == result["sample_ids"][
                    (int(row["global_step"]) - 1) % 8
                ]
                for row in metrics
            )
            and all(
                int(row["fixed_action_flow_step"]) == 0
                for row in metrics
            )
        ),
        "fixed_probe_schedule": (
            [int(row["global_step"]) for row in probes]
            == [0, 50, 100, 150, 200]
            and all(
                int(row["sample_count"]) == 8
                and list(row["sample_ids"]) == list(result["sample_ids"])
                and len(row["per_sample"]) == 8
                and all(
                    str(sample["base_sample_id"])
                    == str(result["sample_ids"][index])
                    for index, sample in enumerate(row["per_sample"])
                )
                for row in probes
            )
        ),
        "first_step_gate_only": (
            float(first["gradient_groups"]["gate"]["l2"]) > 0
            and int(
                first["gradient_groups"]["non_gate"][
                    "nonzero_element_count"
                ]
            )
            == 0
        ),
        "second_step_projector_gradient": int(
            second["gradient_groups"]["future_projector"][
                "nonzero_element_count"
            ]
        )
        > 0,
        "second_step_attention_gradient": int(
            second["gradient_groups"]["attention"][
                "nonzero_element_count"
            ]
        )
        > 0,
        "first_non_gate_paths_at_step_2": (
            int(
                result.get(
                    "first_non_gate_nonzero_gradient_step",
                    -1,
                )
            )
            == 2
            and int(
                result.get(
                    "first_projector_nonzero_gradient_step",
                    -1,
                )
            )
            == 2
            and int(
                result.get(
                    "first_attention_nonzero_gradient_step",
                    -1,
                )
            )
            == 2
        ),
        "finite_trace": (
            all(
                not bool(row["nan_or_inf"])
                and math.isfinite(float(row["loss"]))
                and all(
                    bool(group["finite"])
                    for group in row["gradient_groups"].values()
                )
                for row in metrics
            )
            and all(
                math.isfinite(float(row["mean_action_loss"]))
                and math.isfinite(
                    float(
                        row[
                            "max_gated_delta_to_action_hidden_ratio"
                        ]
                    )
                )
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
            isinstance(result.get("checkpoint_roundtrip"), Mapping)
            and result["checkpoint_roundtrip"].get("state_equal") is True
            and len(
                str(
                    result["checkpoint_roundtrip"].get(
                        "adapter_state_sha256",
                        "",
                    )
                )
            )
            == 64
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
        "output_dir": str(cfg.experiment.output_dir),
    }
    return execution, artifacts


def _run_phase_e2(
    cfg: Thought3Config,
    *,
    resume: bool,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from fastwam_ood_eval.thought3.model_wrapper import (
        parameter_state_sha256,
    )

    _assert_phase_e2_scope(cfg)
    if os.environ.get("CONFIRM_THOUGHT3_PHASE_E2") != "YES":
        raise PhaseE2GateError(
            "set CONFIRM_THOUGHT3_PHASE_E2=YES for real eight-sample training"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise PhaseE2GateError(
            "Gate E.2 requires exactly one CUDA-visible GPU"
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise PhaseE2GateError(
            "Gate E.2 requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
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
    phase_d = _verify_phase_d_gate(cfg)
    tracks = _derive_tracks(cfg)
    _progress(
        "phase_d_verified",
        cache_fingerprint=phase_d["cache_fingerprint"],
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
        or int(data_report["sample_count"]) != 8
    ):
        raise PhaseE2GateError("Gate E.2 data-access audit failed")
    _progress(
        "training_data_ready",
        samples=data_report["sample_count"],
        split_counts=data_report["split_counts"],
    )

    frozen_before = parameter_state_sha256(
        iter(model.named_parameters())
    )
    _progress("frozen_hash_before", sha256=frozen_before)
    results: dict[str, dict[str, Mapping[str, Any]]] = {}
    track_execution_error: BaseException | None = None
    track_execution_traceback: str | None = None
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
                track_result = run_fixed_subset_training(
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
                    max_delta_hidden_ratio=track_result["outcome"][
                        "max_gated_delta_to_action_hidden_ratio"
                    ],
                    variant=variant,
                )
    except BaseException as exc:
        track_execution_error = exc
        track_execution_traceback = traceback.format_exc()
    frozen_after = parameter_state_sha256(
        iter(model.named_parameters())
    )
    prevalidation = {
        "captured_at": _utc_now(),
        "data_preparation": data_report,
        "frozen_parameter_sha256_after": frozen_after,
        "frozen_parameter_sha256_before": frozen_before,
        "model_load": model_report,
        "phase_d_frozen": phase_d,
        "schema_version": PHASE_E2_SCHEMA,
        "track_execution_error": (
            None
            if track_execution_error is None
            else (
                f"{type(track_execution_error).__name__}: "
                f"{track_execution_error}"
            )
        ),
        "track_execution_traceback": track_execution_traceback,
        "tracks": {
            lr_slug: {
                variant: dict(value)
                for variant, value in by_variant.items()
            }
            for lr_slug, by_variant in results.items()
        },
    }
    atomic_write_json(
        output / "pre_validation_result.json",
        prevalidation,
    )
    _progress("frozen_hash_after", sha256=frozen_after)
    if track_execution_error is not None:
        del prepared, upstream_cfg, model
        gc.collect()
        torch.cuda.empty_cache()
        raise PhaseE2GateError(
            "Gate E.2 track execution failed after frozen hash capture"
        ) from track_execution_error

    execution_checks: dict[str, dict[str, dict[str, bool]]] = {}
    track_artifacts: dict[str, dict[str, dict[str, Any]]] = {}
    track_performance: dict[str, dict[str, dict[str, bool]]] = {}
    paired_checks: dict[str, dict[str, bool]] = {}
    eligibility: dict[str, bool] = {}
    for lr_slug, _ in PHASE_E2_LR_GRID:
        execution_checks[lr_slug] = {}
        track_artifacts[lr_slug] = {}
        track_performance[lr_slug] = {}
        for variant in ("A0", "A1"):
            checks, artifacts = _track_checks(
                tracks[lr_slug][variant],
                results[lr_slug][variant],
            )
            execution_checks[lr_slug][variant] = checks
            track_artifacts[lr_slug][variant] = artifacts
            track_performance[lr_slug][variant] = performance_checks(
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
            "same_initial_fixed_probe": (
                _initial_probe_signature(a0)
                == _initial_probe_signature(a1)
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
            and all(track_performance[lr_slug]["A0"].values())
            and all(track_performance[lr_slug]["A1"].values())
            and all(paired_checks[lr_slug].values())
        )
    selected_slug = select_smallest_eligible_lr(eligibility)
    learning_rates = dict(PHASE_E2_LR_GRID)
    cross_track_checks = {
        "all_initial_adapter_sha_equal": len(
            {
                str(result["initial_adapter_sha256"])
                for by_variant in results.values()
                for result in by_variant.values()
            }
        )
        == 1,
        "all_sample_ids_equal": len(
            {
                tuple(result["sample_ids"])
                for by_variant in results.values()
                for result in by_variant.values()
            }
        )
        == 1,
        "all_zero_gate_initial_probes_equal": len(
            {
                _initial_probe_signature(result)
                for by_variant in results.values()
                for result in by_variant.values()
            }
        )
        == 1,
        "frozen_fastwam_unchanged": frozen_before == frozen_after,
    }
    all_execution_passed = all(
        all(checks.values())
        for by_variant in execution_checks.values()
        for checks in by_variant.values()
    )
    all_paired_checks_passed = all(
        all(checks.values()) for checks in paired_checks.values()
    )
    gate_passed = (
        all_execution_passed
        and all_paired_checks_passed
        and all(cross_track_checks.values())
        and selected_slug is not None
    )
    result = {
        "completed_at": _utc_now(),
        "config": cfg.to_dict(),
        "config_fingerprint": cfg.fingerprint,
        "cross_track_checks": cross_track_checks,
        "data_preparation": data_report,
        "determinism": {
            "cublas_workspace_config": ":4096:8",
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "deterministic_algorithms": True,
            "flash_sdp": False,
            "math_sdp": True,
            "mem_efficient_sdp": False,
            "tf32": False,
        },
        "eligibility": eligibility,
        "frozen_parameter_sha256_after": frozen_after,
        "frozen_parameter_sha256_before": frozen_before,
        "gate_e2_passed": gate_passed,
        "model_load": model_report,
        "paired_checks": paired_checks,
        "phase_d_frozen": phase_d,
        "preregistered_gate": {
            "catastrophic_loss_ratio": 2.0,
            "learning_rates": [
                value for _, value in PHASE_E2_LR_GRID
            ],
            "max_catastrophic_samples": (
                PHASE_E2_MAX_CATASTROPHIC_SAMPLES
            ),
            "max_median_delta_hidden_ratio": (
                PHASE_E2_MAX_MEDIAN_DELTA_HIDDEN_RATIO
            ),
            "max_sample_delta_hidden_ratio": (
                PHASE_E2_MAX_SAMPLE_DELTA_HIDDEN_RATIO
            ),
            "min_loss_reduction_fraction": (
                PHASE_E2_MIN_LOSS_REDUCTION_FRACTION
            ),
            "min_non_worsened_samples": (
                PHASE_E2_MIN_NON_WORSENED_SAMPLES
            ),
            "sample_count": 8,
            "selection_rule": (
                "smallest learning rate eligible for both A0 and A1"
            ),
            "steps_per_track": 200,
            "variants": ["A0", "A1"],
        },
        "schema_version": PHASE_E2_SCHEMA,
        "scope": {
            "development_outcomes_read": False,
            "future_rgb_frames_read": 0,
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
                    "artifacts": track_artifacts[lr_slug][variant],
                    "execution_checks": (
                        execution_checks[lr_slug][variant]
                    ),
                    "performance_checks": (
                        track_performance[lr_slug][variant]
                    ),
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


def run_phase_e2_eight_sample(
    cfg: Thought3Config,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Run Gate E.2 and preserve both passing and failed outcomes."""

    output = ensure_thought3_output_path(cfg.experiment.output_dir)
    result_path = output / "gate_e2_result.json"
    status_path = output / "run_status.json"
    if result_path.is_file():
        existing = load_json(result_path)
        if not resume:
            raise FileExistsError(
                f"Gate E.2 result exists; pass --resume: {result_path}"
            )
        if existing.get("gate_e2_passed") is True:
            return existing
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        status_path,
        {
            "schema_version": PHASE_E2_SCHEMA,
            "started_at": _utc_now(),
            "status": "running",
        },
    )
    started = time.perf_counter()
    try:
        result = _run_phase_e2(cfg, resume=resume)
        result["gate_wall_s"] = time.perf_counter() - started
        atomic_write_json(result_path, result)
        if result["gate_e2_passed"] is not True:
            raise PhaseE2GateError(
                "Gate E.2 hard checks failed; inspect gate_e2_result.json"
            )
    except BaseException as exc:
        atomic_write_json(
            status_path,
            {
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at": _utc_now(),
                "gate_e2_passed": False,
                "result": (
                    str(result_path.resolve())
                    if result_path.is_file()
                    else None
                ),
                "schema_version": PHASE_E2_SCHEMA,
                "status": "failed",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    atomic_write_json(
        status_path,
        {
            "finished_at": _utc_now(),
            "gate_e2_passed": True,
            "result": str(result_path.resolve()),
            "schema_version": PHASE_E2_SCHEMA,
            "status": "passed",
        },
    )
    return result
