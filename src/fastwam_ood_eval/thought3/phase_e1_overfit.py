"""Gate E.1: fixed-sample, fixed-noise real Fast-WAM overfit diagnostic.

This gate is intentionally narrower than Gate E.  It uses one standard
LIBERO training sample and repeats one exact flow-matching objective for A0
and A1.  It answers only whether the frozen-model injection graph and
Adapter-only optimizer can overfit that exact objective.  It does not select
a model, evaluate success, access OOD outcomes, or support a future-benefit
claim.
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
    run_fixed_sample_overfit,
)
from fastwam_ood_eval.thought3.safety import ensure_thought3_output_path


PHASE_E1_SCHEMA = "thought3.phase_e1.fixed_overfit.v1"
PHASE_E1_MIN_LOSS_REDUCTION_FRACTION = 0.50


class PhaseE1GateError(RuntimeError):
    """Raised when the fixed-overfit diagnostic violates its protocol."""


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
                "phase": "E.1",
                "stage": stage,
                "time": _utc_now(),
                **payload,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _assert_phase_e1_scope(cfg: Thought3Config) -> None:
    """Reject expansion beyond the preregistered one-sample diagnostic."""

    if cfg.runtime.backend != "fastwam" or cfg.runtime.device != "cuda:0":
        raise PhaseE1GateError(
            "Gate E.1 requires backend=fastwam and logical cuda:0"
        )
    if cfg.variant != "A1" or cfg.sampler.active_k != 1:
        raise PhaseE1GateError(
            "Gate E.1 orchestration config must be A1/active_k=1"
        )
    if tuple(cfg.sampler.cache_k) != (1, 2, 4):
        raise PhaseE1GateError("Gate E.1 requires the paired Phase D cache")
    if (
        cfg.cache.root
        != Path("outputs/thought3/cache/phase_d_libero_goal_task0_v1")
        or cfg.cache.pilot_limit != 32
        or cfg.cache.shard_size != 8
    ):
        raise PhaseE1GateError(
            "Gate E.1 is frozen to the 32-sample Phase D cache"
        )
    if len(cfg.data.dataset_roots) != 1:
        raise PhaseE1GateError("Gate E.1 accepts one standard LIBERO root")
    if cfg.data.dataset_revision != OFFICIAL_LIBERO_REVISION:
        raise PhaseE1GateError("Gate E.1 LIBERO revision mismatch")
    if cfg.data.inventory_path != Path(
        "outputs/thought3/phase_d_cache_smoke_v1/inventory.jsonl"
    ):
        raise PhaseE1GateError(
            "Gate E.1 inventory is not the frozen Phase D inventory"
        )
    if (
        cfg.data.split_seed != 3407
        or cfg.experiment.seed != 3407
        or cfg.training.train_seed != 3407
    ):
        raise PhaseE1GateError("Gate E.1 seeds must remain 3407")
    if (
        cfg.training.max_steps != 200
        or cfg.training.microbatch_size != 1
        or cfg.training.gradient_accumulation_steps != 1
        or cfg.training.checkpoint_interval != 50
    ):
        raise PhaseE1GateError(
            "Gate E.1 budget is 200 steps per variant, batch 1, checkpoint 50"
        )
    if (
        cfg.training.learning_rate != 1e-3
        or cfg.training.weight_decay != 1e-2
        or cfg.training.gradient_checkpointing
        or cfg.training.gate_l2 != 0
    ):
        raise PhaseE1GateError("Gate E.1 optimizer recipe changed")
    if cfg.runtime.online_use_cache:
        raise PhaseE1GateError(
            "Gate E.1 forbids online training-cache generation"
        )


def derive_overfit_variant_config(
    cfg: Thought3Config,
    *,
    variant: str,
) -> Thought3Config:
    """Derive matched A0/A1 tracks without changing the source YAML."""

    if variant not in {"A0", "A1"}:
        raise PhaseE1GateError(
            f"unsupported Gate E.1 variant: {variant}"
        )
    derived = replace(
        cfg,
        variant=variant,
        experiment=replace(
            cfg.experiment,
            name=f"thought3_phase_e1_{variant.lower()}_fixed_overfit",
            output_dir=(
                cfg.experiment.output_dir
                / "variants"
                / variant.lower()
            ),
        ),
        sampler=replace(
            cfg.sampler,
            active_k=0 if variant == "A0" else 1,
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


def _derive_matched_configs(
    cfg: Thought3Config,
) -> dict[str, Thought3Config]:
    variants = {
        variant: derive_overfit_variant_config(cfg, variant=variant)
        for variant in ("A0", "A1")
    }
    if (
        _matched_recipe_payload(variants["A0"])
        != _matched_recipe_payload(variants["A1"])
    ):
        raise PhaseE1GateError("A0/A1 Gate E.1 recipes are not matched")
    return variants


def _track_checks(
    cfg: Thought3Config,
    result: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    metrics_path = Path(str(result["metrics"]))
    metrics = load_jsonl(metrics_path)
    steps = [int(row["global_step"]) for row in metrics]
    first = metrics[0]
    second = metrics[1]
    checks = {
        "complete_200_steps": (
            result.get("status") == "complete"
            and int(result.get("completed_steps", -1)) == 200
            and steps == list(range(1, 201))
        ),
        "fixed_single_sample": (
            len({str(row["base_sample_id"]) for row in metrics}) == 1
            and all(
                int(row["fixed_action_flow_step"]) == 0
                for row in metrics
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
        "first_non_gate_gradient_at_step_2": int(
            result.get("first_non_gate_nonzero_gradient_step", -1)
        )
        == 2,
        "finite_trace": (
            all(not bool(row["nan_or_inf"]) for row in metrics)
            and all(
                math.isfinite(float(row["loss"])) for row in metrics
            )
            and all(
                all(
                    bool(group["finite"])
                    for group in row["gradient_groups"].values()
                )
                for row in metrics
            )
        ),
        "actual_hidden_delta_nonzero": (
            float(result["final_gated_delta_norm"]) > 0
            and float(result["final_gated_delta_nonzero_fraction"]) > 0
        ),
        "loss_reduction_at_least_50_percent": (
            math.isfinite(float(result["initial_action_loss"]))
            and math.isfinite(float(result["final_action_loss"]))
            and float(result["loss_reduction_fraction"])
            >= PHASE_E1_MIN_LOSS_REDUCTION_FRACTION
        ),
        "adapter_only_optimizer": (
            result.get("optimizer_parameter_scope") == "adapter_only"
        ),
        "no_ground_truth_future_rgb": (
            result.get("uses_ground_truth_future_input") is False
        ),
        "memory_below_43_gib": (
            float(result["max_peak_memory_mib"]) < 43 * 1024
        ),
        "checkpoint_roundtrip": (
            isinstance(result.get("checkpoint_roundtrip"), Mapping)
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
                cfg.experiment.output_dir / "overfit_manifest.json"
            ),
            "metrics": sha256_file(metrics_path),
            "state": sha256_file(
                cfg.experiment.output_dir / "overfit_state.json"
            ),
        },
        "first_gate_gradient": float(first["gate_gradient"]),
        "first_gate_gradient_sign": int(first["gate_gradient_sign"]),
        "last_gate_gradient": float(metrics[-1]["gate_gradient"]),
        "last_gate_gradient_sign": int(
            metrics[-1]["gate_gradient_sign"]
        ),
        "last_gradient_groups": metrics[-1]["gradient_groups"],
        "output_dir": str(cfg.experiment.output_dir),
    }
    return checks, artifacts


def _run_phase_e1(
    cfg: Thought3Config,
    *,
    resume: bool,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from fastwam_ood_eval.thought3.model_wrapper import (
        parameter_state_sha256,
    )

    _assert_phase_e1_scope(cfg)
    if os.environ.get("CONFIRM_THOUGHT3_PHASE_E1") != "YES":
        raise PhaseE1GateError(
            "set CONFIRM_THOUGHT3_PHASE_E1=YES for real overfit diagnosis"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise PhaseE1GateError(
            "Gate E.1 requires exactly one CUDA-visible GPU"
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise PhaseE1GateError(
            "Gate E.1 requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
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
    variants = _derive_matched_configs(cfg)
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
    )
    data_report = dict(prepared.report)
    atomic_write_json(output / "data_preparation.json", data_report)
    source = data_report["current_source"]
    if (
        source["actual_future_read"] is not False
        or int(source["future_rgb_frames_decoded"]) != 0
        or int(source["action_target_rows_read"]) != 1024
        or data_report["future_rgb_used_as_input"] is not False
        or data_report["split_counts"]
        != {"train": 28, "development": 4}
    ):
        raise PhaseE1GateError("Gate E.1 data-access audit failed")
    _progress(
        "training_data_ready",
        samples=data_report["sample_count"],
        split_counts=data_report["split_counts"],
    )

    frozen_before = parameter_state_sha256(
        iter(model.named_parameters())
    )
    _progress("frozen_hash_before", sha256=frozen_before)
    tracks: dict[str, Mapping[str, Any]] = {}
    for variant in ("A0", "A1"):
        _progress("overfit_started", variant=variant)
        tracks[variant] = run_fixed_sample_overfit(
            variants[variant],
            model=model,
            prepared=prepared,
            frozen_parameter_sha256=frozen_before,
            resume=resume,
            device="cuda:0",
            progress=_progress,
        )
        _progress(
            "overfit_complete",
            final_loss=tracks[variant]["final_action_loss"],
            loss_reduction_fraction=tracks[variant][
                "loss_reduction_fraction"
            ],
            variant=variant,
        )

    # Freeze evidence is captured before any outcome validation so a failed
    # loss gate cannot prevent the frozen-before/after audit from closing.
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
        "schema_version": PHASE_E1_SCHEMA,
        "tracks": {key: dict(value) for key, value in tracks.items()},
    }
    atomic_write_json(
        output / "pre_validation_result.json",
        prevalidation,
    )
    _progress("frozen_hash_after", sha256=frozen_after)

    track_checks: dict[str, dict[str, bool]] = {}
    track_artifacts: dict[str, dict[str, Any]] = {}
    for variant in ("A0", "A1"):
        checks, artifacts = _track_checks(
            variants[variant],
            tracks[variant],
        )
        track_checks[variant] = checks
        track_artifacts[variant] = artifacts

    paired_checks = {
        "same_base_sample": (
            tracks["A0"]["base_sample_id"]
            == tracks["A1"]["base_sample_id"]
        ),
        "same_fixed_noise_and_timestep": (
            int(tracks["A0"]["fixed_action_flow_step"]) == 0
            and int(tracks["A1"]["fixed_action_flow_step"]) == 0
            and variants["A0"].training.train_seed
            == variants["A1"].training.train_seed
        ),
        "same_initial_adapter": (
            tracks["A0"]["initial_adapter_sha256"]
            == tracks["A1"]["initial_adapter_sha256"]
        ),
        "same_initial_loss_under_zero_gate": (
            tracks["A0"]["initial_action_loss"]
            == tracks["A1"]["initial_action_loss"]
        ),
        "same_parameter_count": (
            tracks["A0"]["trainable_parameter_count"]
            == tracks["A1"]["trainable_parameter_count"]
        ),
        "frozen_fastwam_unchanged": frozen_before == frozen_after,
    }
    gate_passed = (
        all(paired_checks.values())
        and all(
            all(checks.values()) for checks in track_checks.values()
        )
    )
    result = {
        "completed_at": _utc_now(),
        "config": cfg.to_dict(),
        "config_fingerprint": cfg.fingerprint,
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
        "frozen_parameter_sha256_after": frozen_after,
        "frozen_parameter_sha256_before": frozen_before,
        "gate_e1_passed": gate_passed,
        "model_load": model_report,
        "paired_checks": paired_checks,
        "phase_d_frozen": phase_d,
        "preregistered_gate": {
            "max_steps_per_variant": 200,
            "minimum_loss_reduction_fraction": (
                PHASE_E1_MIN_LOSS_REDUCTION_FRACTION
            ),
            "sample_count": 1,
            "variants": ["A0", "A1"],
        },
        "schema_version": PHASE_E1_SCHEMA,
        "scope": {
            "development_outcomes_read": False,
            "future_rgb_frames_read": 0,
            "ood_outcomes_read": False,
            "optimizer_steps": 400,
            "rollout_started": False,
            "single_gpu": True,
            "success_outcomes_read": False,
            "task_count": 1,
            "uses_ground_truth_future": False,
        },
        "status": "passed" if gate_passed else "failed",
        "tracks": {
            variant: {
                "artifacts": track_artifacts[variant],
                "checks": track_checks[variant],
                "result": dict(tracks[variant]),
            }
            for variant in ("A0", "A1")
        },
    }
    del prepared, upstream_cfg, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_phase_e1_overfit(
    cfg: Thought3Config,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Run Gate E.1 and record both passing and failed diagnostic outcomes."""

    output = ensure_thought3_output_path(cfg.experiment.output_dir)
    result_path = output / "gate_e1_result.json"
    status_path = output / "run_status.json"
    if result_path.is_file():
        existing = load_json(result_path)
        if not resume:
            raise FileExistsError(
                f"Gate E.1 result exists; pass --resume: {result_path}"
            )
        if existing.get("gate_e1_passed") is True:
            return existing
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        status_path,
        {
            "schema_version": PHASE_E1_SCHEMA,
            "started_at": _utc_now(),
            "status": "running",
        },
    )
    started = time.perf_counter()
    try:
        result = _run_phase_e1(cfg, resume=resume)
        result["gate_wall_s"] = time.perf_counter() - started
        atomic_write_json(result_path, result)
        if result["gate_e1_passed"] is not True:
            raise PhaseE1GateError(
                "Gate E.1 hard checks failed; inspect gate_e1_result.json"
            )
    except BaseException as exc:
        atomic_write_json(
            status_path,
            {
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at": _utc_now(),
                "gate_e1_passed": False,
                "result": (
                    str(result_path.resolve())
                    if result_path.is_file()
                    else None
                ),
                "schema_version": PHASE_E1_SCHEMA,
                "status": "failed",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    atomic_write_json(
        status_path,
        {
            "finished_at": _utc_now(),
            "gate_e1_passed": True,
            "result": str(result_path.resolve()),
            "schema_version": PHASE_E1_SCHEMA,
            "status": "passed",
        },
    )
    return result
