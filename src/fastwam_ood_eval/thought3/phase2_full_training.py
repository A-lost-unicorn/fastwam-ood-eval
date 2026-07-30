"""Executable Thought3 Phase 2 full 28/4 matched A0/A1 training.

The public runner has four explicit stages:

``calibrate``
    Load the frozen Fast-WAM once, prepare the complete Phase D 28/4 subset,
    and freeze one inverse-initial-loss weight vector from train-only flows.

``A0`` / ``A1``
    Train one Adapter-only track.  The two stages are designed to run in
    separate processes on separate physical GPUs while each sees logical
    ``cuda:0``.

``finalize``
    CPU-only aggregation of the two completed tracks.  It never declares
    Phase 2 complete-for-rollout: the trained A1 checkpoint still requires the
    preregistered online correct/null/shuffle sensitivity recheck.
"""

from __future__ import annotations

import gc
import hashlib
import math
import os
import statistics
import subprocess
import time
import traceback
import uuid
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
    atomic_write_jsonl,
    atomic_write_text,
    load_json,
    load_jsonl,
    sha256_file,
)
from fastwam_ood_eval.thought3.phase2_protocol import (
    PHASE2_ARTIFACT_SCHEMA,
    PHASE2_CALIBRATION_SCHEMA,
    PHASE2_CONFIRMATION_ENV,
    PHASE2_RESULT_SCHEMA,
    PHASE2_TRACK_SCHEMA,
    PHASE2_VARIANTS,
    Phase2FullTrainingConfig,
    Phase2ProtocolError,
    inverse_initial_loss_unit_mean_weights,
    metric_rows_sha256,
    phase2_flow_objective_identity,
    phase2_identity_schedule_sha256,
    phase2_sample_loss_weights_sha256,
    phase2_training_flow_slot,
)
from fastwam_ood_eval.thought3.safety import ensure_thought3_output_path


class Phase2ExecutionError(Phase2ProtocolError):
    """Raised when a real Phase 2 execution violates a frozen invariant."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress(stage: str, **values: Any) -> None:
    import json

    print(
        json.dumps(
            {
                "phase": "Thought3-Phase2-full-28-4",
                "stage": stage,
                "time": _utc_now(),
                **values,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def phase2_dry_run_payload(
    cfg: Phase2FullTrainingConfig,
    *,
    stage: str,
) -> dict[str, Any]:
    """Describe one stage without importing torch or writing artifacts."""

    if stage not in {"calibrate", "A0", "A1", "finalize"}:
        raise Phase2ProtocolError(f"invalid Phase 2 stage: {stage}")
    return {
        "calibration_flow_steps": list(cfg.calibration_flow_steps),
        "command": "thought3-train-phase2-full",
        "config_fingerprint": cfg.fingerprint,
        "development_count": cfg.development_count,
        "development_flow_steps": list(cfg.development_flow_steps),
        "dry_run": True,
        "fixed_primary_checkpoint_step": cfg.primary_checkpoint_step,
        "objectives_per_track": (
            cfg.optimizer_updates * cfg.objectives_per_update
        ),
        "output_dir": str(cfg.output_dir),
        "parallel_track_count": cfg.parallel_track_count,
        "phase1_required_classification": (
            cfg.expected_phase1_classification
        ),
        "sample_weight_recipe": cfg.sample_weight_recipe,
        "stage": stage,
        "train_count": cfg.train_count,
        "training_flow_range": [
            cfg.training_flow_start,
            cfg.training_flow_end,
        ],
        "variants": list(cfg.variants),
        "would_load_checkpoint": False,
        "would_load_fastwam": False,
        "would_write": False,
    }


def _git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_status(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _verify_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise Phase2ExecutionError(f"{label} is missing: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise Phase2ExecutionError(
            f"{label} SHA changed: {observed} != {expected_sha256}"
        )


def _verify_phase1_artifact_tree(cfg: Phase2FullTrainingConfig) -> None:
    manifest = load_json(cfg.phase1_artifact_manifest_path)
    root = cfg.phase1_artifact_manifest_path.parent
    files = manifest.get("files")
    if (
        not isinstance(files, Mapping)
        or int(manifest.get("file_count", -1)) != len(files)
        or len(files) != 62
    ):
        raise Phase2ExecutionError(
            "Phase 1 artifact manifest count/shape changed"
        )
    for relative, descriptor_value in files.items():
        descriptor = (
            descriptor_value
            if isinstance(descriptor_value, Mapping)
            else {}
        )
        path = root / str(relative)
        if (
            not path.is_file()
            or int(descriptor.get("bytes", -1)) != path.stat().st_size
            or descriptor.get("sha256") != sha256_file(path)
        ):
            raise Phase2ExecutionError(
                f"Phase 1 artifact changed: {relative}"
            )


def verify_phase2_prerequisites(
    cfg: Phase2FullTrainingConfig,
    *,
    require_clean: bool,
) -> tuple[Thought3Config, dict[str, Any]]:
    """Verify Phase D, valid E9 audit, and positive Phase 1 branch."""

    expected_files = (
        (
            cfg.thought3_base_config_path,
            cfg.thought3_base_config_sha256,
            "Thought3 base config",
        ),
        (
            cfg.phase_d_gate_path,
            cfg.phase_d_gate_sha256,
            "Phase D result",
        ),
        (
            cfg.phase1_aggregate_path,
            cfg.phase1_aggregate_sha256,
            "Phase 1 aggregate",
        ),
        (
            cfg.phase1_artifact_manifest_path,
            cfg.phase1_artifact_manifest_sha256,
            "Phase 1 artifact manifest",
        ),
        (
            cfg.e9_audit_path,
            cfg.e9_audit_sha256,
            "E9a-v2.1 audit result",
        ),
    )
    for path, digest, label in expected_files:
        _verify_file(path, digest, label)
    _verify_phase1_artifact_tree(cfg)

    base_cfg = load_thought3_config(cfg.thought3_base_config_path)
    phase_d = load_json(cfg.phase_d_gate_path)
    phase1 = load_json(cfg.phase1_aggregate_path)
    e9_audit = load_json(cfg.e9_audit_path)
    if (
        phase_d.get("status") != "passed"
        or phase_d.get("cache_validation", {}).get("status") != "valid"
        or phase_d.get("cache_validation", {}).get("cache_fingerprint")
        != cfg.cache_fingerprint
        or phase_d.get("plan", {}).get("split_fingerprint")
        != cfg.split_fingerprint
        or base_cfg.cache.root
        != Path("outputs/thought3/cache/phase_d_libero_goal_task0_v1")
        or phase1.get("decision", {}).get("classification")
        != cfg.expected_phase1_classification
        or phase1.get("decision", {}).get("next_branch") != "A"
        or int(phase1.get("sample_count", -1)) != 8
        or e9_audit.get("audit_valid") is not True
        or e9_audit.get("outcome")
        != "audit_valid_scientific_failed"
        or e9_audit.get("scientific_result", {}).get("e9b_locked")
        is not True
        or e9_audit.get("scientific_result", {}).get(
            "sample_tail_mitigation_classification"
        )
        != "sample_tail_mitigation_not_supported"
    ):
        raise Phase2ExecutionError(
            "Phase 2 prerequisite classification/provenance changed"
        )
    project = Path(".").resolve()
    fastwam = Path("third_party/FastWAM").resolve()
    project_commit = _git_head(project)
    fastwam_commit = _git_head(fastwam)
    project_status = _git_status(project)
    fastwam_status = _git_status(fastwam)
    if require_clean and (project_status or fastwam_status):
        raise Phase2ExecutionError(
            "Phase 2 requires clean project and Fast-WAM worktrees"
        )
    if fastwam_commit != base_cfg.backbone.fastwam_commit:
        raise Phase2ExecutionError("Fast-WAM commit differs from config")
    return base_cfg, {
        "cache_fingerprint": cfg.cache_fingerprint,
        "checked_at": _utc_now(),
        "e9_audit_outcome": e9_audit["outcome"],
        "fastwam_commit": fastwam_commit,
        "fastwam_worktree_clean": not bool(fastwam_status),
        "phase1_classification": phase1["decision"]["classification"],
        "phase1_next_branch": phase1["decision"]["next_branch"],
        "project_commit": project_commit,
        "project_worktree_clean": not bool(project_status),
        "recipe_selection_disclosure": (
            cfg.recipe_selection_disclosure
        ),
        "split_fingerprint": cfg.split_fingerprint,
    }


def derive_phase2_thought3_config(
    cfg: Phase2FullTrainingConfig,
    base_cfg: Thought3Config,
    *,
    variant: str,
    output_dir: Path,
) -> Thought3Config:
    """Derive one immutable A0/A1 training config from the frozen base."""

    if variant not in PHASE2_VARIANTS:
        raise Phase2ProtocolError(f"unsupported Phase 2 variant: {variant}")
    derived = replace(
        base_cfg,
        variant=variant,
        experiment=replace(
            base_cfg.experiment,
            name=f"{cfg.experiment_name}_{variant.lower()}",
            output_dir=output_dir,
            seed=cfg.experiment_seed,
        ),
        runtime=replace(
            base_cfg.runtime,
            device=cfg.device,
            max_gpu_memory_gb=cfg.max_gpu_memory_gb,
        ),
        sampler=replace(
            base_cfg.sampler,
            active_k=0 if variant == "A0" else 1,
        ),
        training=replace(
            base_cfg.training,
            max_steps=cfg.optimizer_updates,
            microbatch_size=1,
            gradient_accumulation_steps=cfg.objectives_per_update,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            train_seed=cfg.train_seed,
            gradient_checkpointing=False,
            gate_l2=0.0,
            checkpoint_interval=cfg.checkpoint_interval,
        ),
    )
    validate_config(derived)
    return derived


def _require_confirmation() -> None:
    if os.environ.get(PHASE2_CONFIRMATION_ENV) != "YES":
        raise Phase2ExecutionError(
            f"set {PHASE2_CONFIRMATION_ENV}=YES for real Phase 2 work"
        )


def _configure_cuda(cfg: Phase2FullTrainingConfig) -> None:
    import numpy as np
    import torch

    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != cfg.visible_gpu_count
    ):
        raise Phase2ExecutionError(
            "each Phase 2 GPU stage requires exactly one CUDA-visible GPU"
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != (
        cfg.cublas_workspace_config
    ):
        raise Phase2ExecutionError(
            "CUBLAS_WORKSPACE_CONFIG differs from Phase 2 protocol"
        )
    torch.cuda.set_device(cfg.device)
    torch.use_deterministic_algorithms(cfg.deterministic_algorithms)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(
        torch.backends.cuda.matmul,
        "allow_bf16_reduced_precision_reduction",
    ):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            False
        )
    torch.set_float32_matmul_precision("highest")
    np.random.seed(cfg.experiment_seed)
    torch.manual_seed(cfg.experiment_seed)
    torch.cuda.manual_seed_all(cfg.experiment_seed)


def _load_model_and_data(
    cfg: Phase2FullTrainingConfig,
    thought3_cfg: Thought3Config,
) -> tuple[Any, Any, Any, dict[str, Any], str]:
    import torch

    from fastwam_ood_eval.thought3.model_wrapper import (
        parameter_state_sha256,
    )
    from fastwam_ood_eval.thought3.phase_c_smoke import (
        _load_upstream_model,
    )
    from fastwam_ood_eval.thought3.real_training import (
        prepare_real_training_data,
    )

    torch.cuda.reset_peak_memory_stats(cfg.device)
    _progress("model_load_started", device=cfg.device)
    model, upstream_cfg, model_report = _load_upstream_model(thought3_cfg)
    torch.cuda.synchronize(cfg.device)
    properties = torch.cuda.get_device_properties(cfg.device)
    model_report = {
        **model_report,
        "cuda_version": torch.version.cuda,
        "gpu_name": properties.name,
        "gpu_total_memory_mib": properties.total_memory / 2**20,
        "logical_device": cfg.device,
        "physical_gpu_id": os.environ.get(
            "THOUGHT3_PHYSICAL_GPU_ID"
        ),
        "visible_device_count": torch.cuda.device_count(),
        "load_peak_allocated_mib": (
            int(torch.cuda.max_memory_allocated(cfg.device)) / 2**20
        ),
        "load_peak_reserved_mib": (
            int(torch.cuda.max_memory_reserved(cfg.device)) / 2**20
        ),
    }
    if max(
        model_report["load_peak_allocated_mib"],
        model_report["load_peak_reserved_mib"],
    ) >= (cfg.max_gpu_memory_gb * 1024):
        raise Phase2ExecutionError(
            "Phase 2 model load exceeded the frozen memory ceiling"
        )
    _progress("model_loaded", **model_report)
    prepared = prepare_real_training_data(
        thought3_cfg,
        model=model,
        upstream_cfg=upstream_cfg,
        device=cfg.device,
        progress=lambda stage, values: _progress(stage, **dict(values)),
    )
    report = dict(prepared.report)
    source = report["current_source"]
    if (
        report["split_counts"]
        != {
            "train": cfg.train_count,
            "development": cfg.development_count,
        }
        or report["available_split_counts"]
        != {
            "train": cfg.train_count,
            "development": cfg.development_count,
        }
        or report["selection_mode"] != "complete_phase_d_subset"
        or report["cache_fingerprint"] != cfg.cache_fingerprint
        or report["split_fingerprint"] != cfg.split_fingerprint
        or report["future_rgb_used_as_input"] is not False
        or source["actual_future_read"] is not False
        or int(source["future_rgb_frames_decoded"]) != 0
        or int(source["action_target_rows_read"])
        != (cfg.train_count + cfg.development_count) * 32
        or int(source["current_camera_frames_decoded"])
        != (cfg.train_count + cfg.development_count) * 2
        or int(source["state_rows_read"])
        != cfg.train_count + cfg.development_count
        or any(parameter.requires_grad for parameter in model.parameters())
    ):
        raise Phase2ExecutionError(
            "Phase 2 complete 28/4 data-access audit failed"
        )
    frozen = parameter_state_sha256(iter(model.named_parameters()))
    return model, upstream_cfg, prepared, model_report, frozen


def _split_samples(
    prepared: Any,
    *,
    seed: int,
) -> tuple[list[Any], list[Any]]:
    from fastwam_ood_eval.thought3.real_training import _ordered_samples

    train = _ordered_samples(
        (
            sample
            for sample in prepared.samples
            if sample.split == "train"
        ),
        seed=seed,
    )
    development = _ordered_samples(
        (
            sample
            for sample in prepared.samples
            if sample.split == "development"
        ),
        seed=seed,
    )
    return train, development


def _validate_grid_prefix(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_ids: Sequence[str],
    flow_steps: Sequence[int],
    train_seed: int,
    variant: str,
) -> None:
    expected_pairs = [
        (sample_id, int(flow_step))
        for sample_id in sample_ids
        for flow_step in flow_steps
    ]
    if len(rows) > len(expected_pairs):
        raise Phase2ExecutionError("grid metric prefix is too long")
    for index, row in enumerate(rows):
        sample_id, flow_step = expected_pairs[index]
        numeric_values = (
            float(row.get("action_loss", float("nan"))),
            float(row.get("action_weight", float("nan"))),
            float(row.get("action_hidden_norm", float("nan"))),
            float(
                row.get(
                    "gated_delta_to_action_hidden_ratio",
                    float("nan"),
                )
            ),
            float(row.get("timestep", float("nan"))),
        )
        identity = phase2_flow_objective_identity(
            base_sample_id=sample_id,
            train_seed=train_seed,
            flow_step=flow_step,
        )
        if (
            str(row.get("base_sample_id")) != sample_id
            or int(row.get("flow_step", -1)) != flow_step
            or int(row.get("objective_index", -1)) != index + 1
            or str(row.get("variant")) != variant
            or int(row.get("train_seed", -1)) != train_seed
            or row.get("flow_objective_sha256")
            != identity["flow_objective_sha256"]
            or int(row.get("action_noise_seed", -1))
            != identity["action_noise_seed"]
            or int(row.get("action_timestep_seed", -1))
            != identity["action_timestep_seed"]
            or any(not math.isfinite(value) for value in numeric_values)
            or numeric_values[0] < 0
            or numeric_values[1] < 0
            or numeric_values[2] <= 0
            or numeric_values[3] < 0
            or numeric_values[4] < 0
            or (numeric_values[1] == 0 and numeric_values[0] != 0)
        ):
            raise Phase2ExecutionError(
                f"grid metric prefix identity mismatch at {index + 1}"
            )


def _evaluate_grid_resumable(
    thought3_cfg: Thought3Config,
    phase2_cfg: Phase2FullTrainingConfig,
    *,
    model: Any,
    adapter: Any,
    injector: Any,
    samples: Sequence[Any],
    flow_steps: Sequence[int],
    metrics_path: Path,
    resume: bool,
    progress_stage: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate and checkpoint a sample-major loss grid after each sample."""

    import torch

    from fastwam_ood_eval.thought3.real_training import (
        _flow_timestep_and_weight_scalars,
        _loss_for_real_sample,
    )

    sample_ids = [sample.base_sample_id for sample in samples]
    existing = load_jsonl(metrics_path) if metrics_path.is_file() else []
    if existing and not resume:
        raise FileExistsError(metrics_path)
    _validate_grid_prefix(
        existing,
        sample_ids=sample_ids,
        flow_steps=flow_steps,
        train_seed=thought3_cfg.training.train_seed,
        variant=thought3_cfg.variant,
    )
    if len(existing) % len(flow_steps):
        raise Phase2ExecutionError(
            "grid resume prefix must end at a complete sample"
        )
    rows = [dict(row) for row in existing]
    completed_samples = len(rows) // len(flow_steps)
    was_training = adapter.training
    adapter.eval()
    try:
        with torch.no_grad():
            for sample_index in range(completed_samples, len(samples)):
                sample = samples[sample_index]
                for flow_index, flow_step in enumerate(flow_steps):
                    torch.cuda.synchronize(phase2_cfg.device)
                    torch.cuda.reset_peak_memory_stats(phase2_cfg.device)
                    started = time.perf_counter()
                    loss = _loss_for_real_sample(
                        thought3_cfg,
                        model,
                        adapter,
                        injector,
                        sample,
                        step=int(flow_step),
                        device=phase2_cfg.device,
                    )
                    torch.cuda.synchronize(phase2_cfg.device)
                    diagnostics = adapter.last_diagnostics
                    if (
                        diagnostics is None
                        or diagnostics.action_hidden_norm <= 0
                        or not bool(torch.isfinite(loss).item())
                    ):
                        raise Phase2ExecutionError(
                            "Phase 2 grid produced invalid diagnostics/loss"
                        )
                    timestep, action_weight = (
                        _flow_timestep_and_weight_scalars(
                            model,
                            sample,
                            train_seed=thought3_cfg.training.train_seed,
                            step=int(flow_step),
                            device=phase2_cfg.device,
                        )
                    )
                    identity = phase2_flow_objective_identity(
                        base_sample_id=sample.base_sample_id,
                        train_seed=thought3_cfg.training.train_seed,
                        flow_step=int(flow_step),
                    )
                    loss_value = float(loss.detach().float().cpu())
                    if (
                        not math.isfinite(loss_value)
                        or loss_value < 0
                        or action_weight < 0
                        or (action_weight == 0 and loss_value != 0)
                    ):
                        raise Phase2ExecutionError(
                            "Phase 2 grid loss/weight is invalid"
                        )
                    rows.append(
                        {
                            **identity,
                            "action_hidden_norm": (
                                diagnostics.action_hidden_norm
                            ),
                            "action_loss": loss_value,
                            "action_weight": action_weight,
                            "attention_residual_norm": (
                                diagnostics.attention_residual_norm
                            ),
                            "base_sample_id": sample.base_sample_id,
                            "flow_index": flow_index,
                            "gated_delta_nonzero_fraction": (
                                diagnostics.gated_delta_nonzero_fraction
                            ),
                            "gated_delta_norm": (
                                diagnostics.gated_delta_norm
                            ),
                            "gated_delta_to_action_hidden_ratio": (
                                diagnostics.gated_delta_norm
                                / diagnostics.action_hidden_norm
                            ),
                            "latency_ms": (
                                time.perf_counter() - started
                            )
                            * 1000.0,
                            "objective_index": len(rows) + 1,
                            "peak_memory_mib": (
                                int(
                                    torch.cuda.max_memory_allocated(
                                        phase2_cfg.device
                                    )
                                )
                                / 2**20
                            ),
                            "sample_index": sample_index,
                            "timestep": timestep,
                            "train_seed": (
                                thought3_cfg.training.train_seed
                            ),
                            "variant": thought3_cfg.variant,
                        }
                    )
                    del loss
                atomic_write_jsonl(metrics_path, rows)
                _progress(
                    progress_stage,
                    completed=sample_index + 1,
                    total=len(samples),
                )
    finally:
        adapter.train(was_training)
    _validate_grid_prefix(
        rows,
        sample_ids=sample_ids,
        flow_steps=flow_steps,
        train_seed=thought3_cfg.training.train_seed,
        variant=thought3_cfg.variant,
    )
    if len(rows) != len(samples) * len(flow_steps):
        raise Phase2ExecutionError("Phase 2 grid did not complete")
    per_sample: list[dict[str, Any]] = []
    for sample_index, sample_id in enumerate(sample_ids):
        sample_rows = rows[
            sample_index * len(flow_steps):
            (sample_index + 1) * len(flow_steps)
        ]
        per_sample.append(
            {
                "action_loss": statistics.fmean(
                    float(row["action_loss"]) for row in sample_rows
                ),
                "base_sample_id": sample_id,
                "flow_count": len(flow_steps),
                "max_gated_delta_to_action_hidden_ratio": max(
                    float(row["gated_delta_to_action_hidden_ratio"])
                    for row in sample_rows
                ),
                "zero_weight_objective_count": sum(
                    float(row["action_weight"]) == 0
                    for row in sample_rows
                ),
            }
        )
    losses = [float(row["action_loss"]) for row in per_sample]
    aggregate = {
        "flow_count": len(flow_steps),
        "flow_steps": list(flow_steps),
        "mean_action_loss": statistics.fmean(losses),
        "median_action_loss": statistics.median(losses),
        "metric_rows_sha256": metric_rows_sha256(rows),
        "objective_count": len(rows),
        "per_sample": per_sample,
        "sample_count": len(samples),
        "sample_ids": sample_ids,
        "variant": thought3_cfg.variant,
        "zero_weight_objective_count": sum(
            float(row["action_weight"]) == 0 for row in rows
        ),
    }
    return rows, aggregate


def _artifact_descriptor(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _calibration_paths(cfg: Phase2FullTrainingConfig) -> dict[str, Path]:
    root = cfg.output_dir / "calibration"
    return {
        "root": root,
        "status": root / "run_status.json",
        "train_metrics": root / "train_calibration_objectives.jsonl",
        "development_metrics": (
            root / "development_initial_objectives.jsonl"
        ),
        "result": root / "calibration.json",
        "artifacts": root / "artifact_manifest.json",
        "data": root / "data_preparation.json",
        "preflight": root / "preflight.json",
    }


def _load_calibration(
    cfg: Phase2FullTrainingConfig,
) -> tuple[dict[str, Any], str]:
    paths = _calibration_paths(cfg)
    result = load_json(paths["result"])
    manifest = load_json(paths["artifacts"])
    if (
        result.get("schema_version") != PHASE2_CALIBRATION_SCHEMA
        or result.get("status") != "complete"
        or result.get("phase2_config_fingerprint") != cfg.fingerprint
        or manifest.get("schema_version") != PHASE2_ARTIFACT_SCHEMA
    ):
        raise Phase2ExecutionError("Phase 2 calibration is incomplete")
    for relative, descriptor in manifest["files"].items():
        path = paths["root"] / relative
        if (
            not path.is_file()
            or int(descriptor["bytes"]) != path.stat().st_size
            or descriptor["sha256"] != sha256_file(path)
        ):
            raise Phase2ExecutionError(
                f"Phase 2 calibration artifact changed: {relative}"
            )
    digest = sha256_file(paths["result"])
    if digest != manifest["calibration_sha256"]:
        raise Phase2ExecutionError("calibration result SHA mismatch")
    weights = {
        str(key): float(value)
        for key, value in result["sample_loss_weights"].items()
    }
    if (
        phase2_sample_loss_weights_sha256(
            result["train_sample_ids"],
            weights,
        )
        != result["sample_loss_weights_sha256"]
    ):
        raise Phase2ExecutionError("calibration weight identity changed")
    return result, digest


def run_phase2_calibration(
    cfg: Phase2FullTrainingConfig,
    *,
    resume: bool,
) -> dict[str, Any]:
    """Generate the one shared train-only normalization vector."""

    _require_confirmation()
    import torch

    from fastwam_ood_eval.thought3.checkpointing import (
        adapter_state_sha256,
    )
    from fastwam_ood_eval.thought3.injection import (
        ActionEncoderFutureInjector,
    )
    from fastwam_ood_eval.thought3.model_wrapper import (
        parameter_state_sha256,
    )
    from fastwam_ood_eval.thought3.real_training import build_real_adapter

    _configure_cuda(cfg)
    base_cfg, preflight = verify_phase2_prerequisites(
        cfg,
        require_clean=True,
    )
    paths = _calibration_paths(cfg)
    root = ensure_thought3_output_path(paths["root"])
    if (
        paths["result"].is_file()
        and paths["artifacts"].is_file()
        and resume
    ):
        result, digest = _load_calibration(cfg)
        return {
            **result,
            "calibration_sha256": digest,
            "resumed": True,
        }
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError(
            f"Phase 2 calibration output exists; use --resume: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(paths["preflight"], preflight)
    atomic_write_json(
        paths["status"],
        {
            "schema_version": PHASE2_CALIBRATION_SCHEMA,
            "started_at": _utc_now(),
            "status": "running",
        },
    )
    thought3_cfg = derive_phase2_thought3_config(
        cfg,
        base_cfg,
        variant=cfg.calibration_variant,
        output_dir=root,
    )
    model = upstream_cfg = prepared = adapter = injector = None
    try:
        (
            model,
            upstream_cfg,
            prepared,
            model_report,
            frozen_before,
        ) = _load_model_and_data(cfg, thought3_cfg)
        atomic_write_json(paths["data"], dict(prepared.report))
        train_samples, development_samples = _split_samples(
            prepared,
            seed=cfg.train_seed,
        )
        if (
            len(train_samples) != cfg.train_count
            or len(development_samples) != cfg.development_count
        ):
            raise Phase2ExecutionError(
                "Phase 2 calibration split count changed"
            )
        adapter = build_real_adapter(
            thought3_cfg,
            device=cfg.device,
        )
        initial_adapter_sha256 = adapter_state_sha256(
            adapter.state_dict()
        )
        injector = ActionEncoderFutureInjector(
            model.action_expert.action_encoder,
            adapter,
        )
        train_rows, train_grid = _evaluate_grid_resumable(
            thought3_cfg,
            cfg,
            model=model,
            adapter=adapter,
            injector=injector,
            samples=train_samples,
            flow_steps=cfg.calibration_flow_steps,
            metrics_path=paths["train_metrics"],
            resume=resume,
            progress_stage="calibration_train_sample_complete",
        )
        development_rows, development_grid = (
            _evaluate_grid_resumable(
                thought3_cfg,
                cfg,
                model=model,
                adapter=adapter,
                injector=injector,
                samples=development_samples,
                flow_steps=cfg.development_flow_steps,
                metrics_path=paths["development_metrics"],
                resume=resume,
                progress_stage=(
                    "calibration_development_sample_complete"
                ),
            )
        )
        if (
            float(adapter.gate.detach().float().cpu()) != 0.0
            or any(
                float(row["gated_delta_to_action_hidden_ratio"]) != 0.0
                for row in (*train_rows, *development_rows)
            )
        ):
            raise Phase2ExecutionError(
                "calibration zero-gate path is not exact identity"
            )
        initial_losses = {
            str(row["base_sample_id"]): float(row["action_loss"])
            for row in train_grid["per_sample"]
        }
        train_ids = [sample.base_sample_id for sample in train_samples]
        weights, weight_sha = inverse_initial_loss_unit_mean_weights(
            train_ids,
            initial_losses,
        )
        frozen_after = parameter_state_sha256(
            iter(model.named_parameters())
        )
        if frozen_before != frozen_after:
            raise Phase2ExecutionError(
                "frozen Fast-WAM changed during calibration"
            )
        result = {
            "calibration_flow_steps": list(
                cfg.calibration_flow_steps
            ),
            "completed_at": _utc_now(),
            "data_preparation": dict(prepared.report),
            "development_initial": development_grid,
            "development_sample_ids": [
                sample.base_sample_id for sample in development_samples
            ],
            "fastwam_frozen_sha256_after": frozen_after,
            "fastwam_frozen_sha256_before": frozen_before,
            "initial_adapter_sha256": initial_adapter_sha256,
            "initial_sample_losses": initial_losses,
            "model_load": model_report,
            "phase2_config_fingerprint": cfg.fingerprint,
            "preflight": preflight,
            "recipe_selection_disclosure": (
                cfg.recipe_selection_disclosure
            ),
            "sample_loss_weights": weights,
            "sample_loss_weights_sha256": weight_sha,
            "sample_payload_sha256": prepared.report[
                "sample_payload_sha256"
            ],
            "schema_version": PHASE2_CALIBRATION_SCHEMA,
            "status": "complete",
            "train_calibration": train_grid,
            "train_sample_ids": train_ids,
            "uses_development_for_checkpoint_selection": False,
            "uses_future_rgb": False,
            "uses_ood_or_success": False,
        }
        atomic_write_json(paths["result"], result)
        manifest_files = {}
        for key in (
            "result",
            "train_metrics",
            "development_metrics",
            "data",
            "preflight",
        ):
            path = paths[key]
            manifest_files[str(path.relative_to(root))] = (
                _artifact_descriptor(path)
            )
        calibration_sha = sha256_file(paths["result"])
        atomic_write_json(
            paths["artifacts"],
            {
                "calibration_sha256": calibration_sha,
                "file_count": len(manifest_files),
                "files": manifest_files,
                "schema_version": PHASE2_ARTIFACT_SCHEMA,
            },
        )
        atomic_write_json(
            paths["status"],
            {
                "calibration_sha256": calibration_sha,
                "completed_at": _utc_now(),
                "schema_version": PHASE2_CALIBRATION_SCHEMA,
                "status": "complete",
            },
        )
        _progress(
            "calibration_complete",
            calibration_sha256=calibration_sha,
            sample_loss_weights_sha256=weight_sha,
        )
        return {**result, "calibration_sha256": calibration_sha}
    except BaseException as exc:
        atomic_write_json(
            paths["status"],
            {
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at": _utc_now(),
                "schema_version": PHASE2_CALIBRATION_SCHEMA,
                "status": "failed",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        if injector is not None:
            injector.close()
        del injector, adapter, prepared, upstream_cfg, model
        gc.collect()
        torch.cuda.empty_cache()


def _training_schedule_sha256(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    payload = []
    for row in rows:
        payload.append(
            "\0".join(
                (
                    str(int(row["optimizer_update"])),
                    str(int(row["micro_index"])),
                    str(row["base_sample_id"]),
                    str(int(row["training_flow_slot"])),
                    str(int(row["action_noise_seed"])),
                    str(int(row["action_timestep_seed"])),
                    str(row["flow_objective_sha256"]),
                    repr(float(row["timestep"])),
                    repr(float(row["action_weight"])),
                )
            )
        )
    return hashlib.sha256("\n".join(payload).encode("utf-8")).hexdigest()


def _validate_training_prefix(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_ids: Sequence[str],
    cfg: Phase2FullTrainingConfig,
) -> None:
    cohort = len(sample_ids)
    if (
        len(rows) > cfg.optimizer_updates * cohort
        or len(rows) % cohort
    ):
        raise Phase2ExecutionError(
            "Phase 2 train metrics are not a full-update prefix"
        )
    for objective_index, row in enumerate(rows, start=1):
        update = (objective_index - 1) // cohort + 1
        micro = (objective_index - 1) % cohort + 1
        sample_id = sample_ids[micro - 1]
        slot = phase2_training_flow_slot(
            update,
            micro,
            optimizer_updates=cfg.optimizer_updates,
            objectives_per_update=cohort,
            flow_slot_offset=cfg.training_flow_slot_offset,
        )
        identity = phase2_flow_objective_identity(
            base_sample_id=sample_id,
            train_seed=cfg.train_seed,
            flow_step=slot,
        )
        if (
            int(row.get("objective_index", -1)) != objective_index
            or int(row.get("optimizer_update", -1)) != update
            or int(row.get("micro_index", -1)) != micro
            or str(row.get("base_sample_id")) != sample_id
            or int(row.get("training_flow_slot", -1)) != slot
            or int(row.get("flow_step", -1)) != slot
            or row.get("flow_objective_sha256")
            != identity["flow_objective_sha256"]
            or int(row.get("action_noise_seed", -1))
            != identity["action_noise_seed"]
            or int(row.get("action_timestep_seed", -1))
            != identity["action_timestep_seed"]
        ):
            raise Phase2ExecutionError(
                f"Phase 2 train schedule mismatch at {objective_index}"
            )


def _validate_update_prefix(
    rows: Sequence[Mapping[str, Any]],
    *,
    completed_updates: int,
) -> None:
    if len(rows) < completed_updates:
        raise Phase2ExecutionError(
            "checkpoint exceeds committed update metrics"
        )
    for index, row in enumerate(rows[:completed_updates], start=1):
        if int(row.get("optimizer_update", -1)) != index:
            raise Phase2ExecutionError(
                "Phase 2 update metrics are not contiguous"
            )


def _track_paths(
    cfg: Phase2FullTrainingConfig,
    variant: str,
) -> dict[str, Path]:
    root = cfg.track_output_dir(variant)
    return {
        "root": root,
        "status": root / "run_status.json",
        "objective_metrics": root / "train_objective_metrics.jsonl",
        "update_metrics": root / "train_update_metrics.jsonl",
        "development_metrics": (
            root / "development_final_objectives.jsonl"
        ),
        "state": root / "training_state.json",
        "manifest": root / "training_manifest.json",
        "result": root / "track_result.json",
        "data": root / "data_preparation.json",
        "preflight": root / "preflight.json",
        "checkpoints": root / "checkpoints",
        "artifacts": root / "artifact_manifest.json",
    }


def _checkpoint_extra_matches(
    extra: Mapping[str, Any],
    *,
    cfg: Phase2FullTrainingConfig,
    calibration_sha256: str,
    weight_sha256: str,
    identity_schedule_sha256: str,
    objective_rows: Sequence[Mapping[str, Any]],
    update_rows: Sequence[Mapping[str, Any]],
) -> bool:
    return (
        extra.get("phase2_full_28_4") is True
        and extra.get("phase2_exploratory_full_training") is True
        and extra.get("gate_e_smoke") is False
        and extra.get("calibration_sha256") == calibration_sha256
        and extra.get("sample_loss_weights_sha256") == weight_sha256
        and extra.get("identity_schedule_sha256")
        == identity_schedule_sha256
        and int(extra.get("objectives_per_update", -1))
        == cfg.objectives_per_update
        and int(extra.get("objective_count", -1))
        == len(objective_rows)
        and int(extra.get("training_flow_slot_offset", -1))
        == cfg.training_flow_slot_offset
        and extra.get("objective_metrics_prefix_sha256")
        == metric_rows_sha256(objective_rows)
        and extra.get("update_metrics_prefix_sha256")
        == metric_rows_sha256(update_rows)
        and extra.get("train_flow_schedule_sha256")
        == _training_schedule_sha256(objective_rows)
        and extra.get("primary_checkpoint_rule")
        == cfg.primary_checkpoint_rule
    )


def _train_phase2_track(
    cfg: Phase2FullTrainingConfig,
    thought3_cfg: Thought3Config,
    *,
    model: Any,
    prepared: Any,
    frozen_parameter_sha256: str,
    calibration: Mapping[str, Any],
    calibration_sha256: str,
    resume: bool,
) -> dict[str, Any]:
    import torch

    from fastwam_ood_eval.thought3.checkpointing import (
        adapter_state_sha256,
        find_latest_checkpoint,
        load_adapter_checkpoint,
        save_adapter_checkpoint,
    )
    from fastwam_ood_eval.thought3.injection import (
        ActionEncoderFutureInjector,
    )
    from fastwam_ood_eval.thought3.real_training import (
        _checkpoint_expected,
        _checkpoint_manifest,
        _checkpoint_roundtrip,
        _flow_timestep_and_weight_scalars,
        _loss_for_real_sample,
        adapter_gradient_groups,
        build_real_adapter,
    )

    variant = thought3_cfg.variant
    paths = _track_paths(cfg, variant)
    root = ensure_thought3_output_path(paths["root"])
    root.mkdir(parents=True, exist_ok=True)
    invocation_id = uuid.uuid4().hex
    atomic_write_json(
        paths["status"],
        {
            "invocation_id": invocation_id,
            "schema_version": PHASE2_TRACK_SCHEMA,
            "started_at": _utc_now(),
            "status": "running",
            "variant": variant,
        },
    )
    train_samples, development_samples = _split_samples(
        prepared,
        seed=cfg.train_seed,
    )
    sample_ids = [sample.base_sample_id for sample in train_samples]
    development_ids = [
        sample.base_sample_id for sample in development_samples
    ]
    if (
        sample_ids != list(calibration["train_sample_ids"])
        or development_ids
        != list(calibration["development_sample_ids"])
        or prepared.report["sample_payload_sha256"]
        != calibration["sample_payload_sha256"]
    ):
        raise Phase2ExecutionError(
            f"Phase 2 {variant} data identity differs from calibration"
        )
    weights = {
        str(key): float(value)
        for key, value in calibration["sample_loss_weights"].items()
    }
    weight_sha = phase2_sample_loss_weights_sha256(
        sample_ids,
        weights,
    )
    if weight_sha != calibration["sample_loss_weights_sha256"]:
        raise Phase2ExecutionError("Phase 2 sample weights changed")
    identity_schedule_sha = phase2_identity_schedule_sha256(
        sample_ids,
        train_seed=cfg.train_seed,
        optimizer_updates=cfg.optimizer_updates,
        flow_slot_offset=cfg.training_flow_slot_offset,
    )
    adapter = build_real_adapter(thought3_cfg, device=cfg.device)
    initial_adapter_sha = adapter_state_sha256(adapter.state_dict())
    if initial_adapter_sha != calibration["initial_adapter_sha256"]:
        raise Phase2ExecutionError(
            "A0/A1 Adapter initialization differs from calibration"
        )
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    if {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    } != {id(parameter) for parameter in adapter.parameters()}:
        raise Phase2ExecutionError(
            "Phase 2 optimizer contains non-Adapter parameters"
        )
    injector = ActionEncoderFutureInjector(
        model.action_expert.action_encoder,
        adapter,
    )
    start_update = 0
    latest = (
        find_latest_checkpoint(paths["checkpoints"]) if resume else None
    )
    loaded_manifest = None
    if latest is not None:
        loaded_manifest = load_adapter_checkpoint(
            latest,
            adapter=adapter,
            optimizer=optimizer,
            expected=_checkpoint_expected(
                thought3_cfg,
                prepared,
                frozen_parameter_sha256=frozen_parameter_sha256,
            ),
        )
        start_update = int(loaded_manifest.global_step)
        if (
            start_update < 1
            or start_update > cfg.optimizer_updates
            or int(loaded_manifest.sample_cursor)
            != start_update * cfg.objectives_per_update
        ):
            raise Phase2ExecutionError(
                "Phase 2 checkpoint cursor/update is invalid"
            )
    objective_rows = (
        load_jsonl(paths["objective_metrics"])
        if paths["objective_metrics"].is_file()
        else []
    )
    update_rows = (
        load_jsonl(paths["update_metrics"])
        if paths["update_metrics"].is_file()
        else []
    )
    required_objectives = start_update * cfg.objectives_per_update
    if len(objective_rows) < required_objectives:
        raise Phase2ExecutionError(
            "Phase 2 checkpoint exceeds objective metrics"
        )
    objective_rows = [
        dict(row) for row in objective_rows[:required_objectives]
    ]
    _validate_training_prefix(
        objective_rows,
        sample_ids=sample_ids,
        cfg=cfg,
    )
    _validate_update_prefix(
        update_rows,
        completed_updates=start_update,
    )
    update_rows = [dict(row) for row in update_rows[:start_update]]
    if loaded_manifest is not None and not _checkpoint_extra_matches(
        loaded_manifest.extra,
        cfg=cfg,
        calibration_sha256=calibration_sha256,
        weight_sha256=weight_sha,
        identity_schedule_sha256=identity_schedule_sha,
        objective_rows=objective_rows,
        update_rows=update_rows,
    ):
        raise Phase2ExecutionError(
            "Phase 2 checkpoint/metric provenance mismatch"
        )
    state_payload = {
        "calibration_sha256": calibration_sha256,
        "development_sample_ids": development_ids,
        "identity_schedule_sha256": identity_schedule_sha,
        "initial_adapter_sha256": initial_adapter_sha,
        "phase2_config_fingerprint": cfg.fingerprint,
        "sample_ids": sample_ids,
        "sample_loss_weights_sha256": weight_sha,
        "thought3_config": thought3_cfg.to_dict(),
        "thought3_config_fingerprint": thought3_cfg.fingerprint,
        "training_flow_slot_offset": cfg.training_flow_slot_offset,
        "variant": variant,
    }
    if paths["state"].is_file():
        if load_json(paths["state"]) != state_payload:
            raise Phase2ExecutionError(
                "Phase 2 training state provenance changed"
            )
    else:
        if start_update:
            raise Phase2ExecutionError(
                "checkpoint exists without Phase 2 training state"
            )
        atomic_write_json(paths["state"], state_payload)

    first_non_gate = None
    first_projector = None
    first_attention = None
    for row in update_rows:
        groups = row["gradient_groups"]
        update = int(row["optimizer_update"])
        if (
            first_non_gate is None
            and int(groups["non_gate"]["nonzero_element_count"]) > 0
        ):
            first_non_gate = update
        if (
            first_projector is None
            and int(
                groups["future_projector"]["nonzero_element_count"]
            )
            > 0
        ):
            first_projector = update
        if (
            first_attention is None
            and int(groups["attention"]["nonzero_element_count"]) > 0
        ):
            first_attention = update
    started = time.perf_counter()
    try:
        for update in range(start_update + 1, cfg.optimizer_updates + 1):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(cfg.device)
            torch.cuda.reset_peak_memory_stats(cfg.device)
            update_started = time.perf_counter()
            gate_before = float(adapter.gate.detach().float().cpu())
            losses: list[float] = []
            weighted_losses: list[float] = []
            action_weights: list[float] = []
            current_objective_rows: list[dict[str, Any]] = []
            for micro, sample in enumerate(train_samples, start=1):
                slot = phase2_training_flow_slot(
                    update,
                    micro,
                    optimizer_updates=cfg.optimizer_updates,
                    objectives_per_update=cfg.objectives_per_update,
                    flow_slot_offset=cfg.training_flow_slot_offset,
                )
                loss = _loss_for_real_sample(
                    thought3_cfg,
                    model,
                    adapter,
                    injector,
                    sample,
                    step=slot,
                    device=cfg.device,
                )
                if not bool(torch.isfinite(loss).item()):
                    raise Phase2ExecutionError(
                        f"Phase 2 {variant} loss is NaN/Inf"
                    )
                timestep, action_weight = (
                    _flow_timestep_and_weight_scalars(
                        model,
                        sample,
                        train_seed=cfg.train_seed,
                        step=slot,
                        device=cfg.device,
                    )
                )
                raw_loss = float(loss.detach().float().cpu())
                sample_weight = weights[sample.base_sample_id]
                diagnostics = adapter.last_diagnostics
                if (
                    diagnostics is None
                    or diagnostics.action_hidden_norm <= 0
                    or action_weight < 0
                    or (action_weight == 0 and raw_loss != 0)
                ):
                    raise Phase2ExecutionError(
                        f"Phase 2 {variant} objective diagnostics invalid"
                    )
                (
                    loss
                    * sample_weight
                    / cfg.objectives_per_update
                ).backward()
                identity = phase2_flow_objective_identity(
                    base_sample_id=sample.base_sample_id,
                    train_seed=cfg.train_seed,
                    flow_step=slot,
                )
                objective_index = (
                    (update - 1) * cfg.objectives_per_update + micro
                )
                current_objective_rows.append(
                    {
                        **identity,
                        "action_hidden_norm": (
                            diagnostics.action_hidden_norm
                        ),
                        "action_loss": raw_loss,
                        "action_weight": action_weight,
                        "base_sample_id": sample.base_sample_id,
                        "gated_delta_to_action_hidden_ratio": (
                            diagnostics.gated_delta_norm
                            / diagnostics.action_hidden_norm
                        ),
                        "mean_scaled_backward_loss": (
                            raw_loss
                            * sample_weight
                            / cfg.objectives_per_update
                        ),
                        "micro_index": micro,
                        "objective_index": objective_index,
                        "optimizer_update": update,
                        "sample_loss_weight": sample_weight,
                        "timestep": timestep,
                        "training_flow_slot": slot,
                        "variant": variant,
                        "zero_weight_objective": (
                            action_weight == 0
                        ),
                    }
                )
                losses.append(raw_loss)
                weighted_losses.append(raw_loss * sample_weight)
                action_weights.append(action_weight)
                del loss
            groups = adapter_gradient_groups(adapter)
            if not all(bool(value["finite"]) for value in groups.values()):
                raise Phase2ExecutionError(
                    f"Phase 2 {variant} gradient is NaN/Inf"
                )
            if update == 1 and (
                float(groups["gate"]["l2"]) <= 0
                or int(
                    groups["non_gate"]["nonzero_element_count"]
                )
                != 0
            ):
                raise Phase2ExecutionError(
                    f"Phase 2 {variant} first update is not gate-only"
                )
            if update == 2 and (
                int(
                    groups["future_projector"][
                        "nonzero_element_count"
                    ]
                )
                <= 0
                or int(
                    groups["attention"]["nonzero_element_count"]
                )
                <= 0
            ):
                raise Phase2ExecutionError(
                    f"Phase 2 {variant} second update did not open paths"
                )
            if first_non_gate is None and int(
                groups["non_gate"]["nonzero_element_count"]
            ) > 0:
                first_non_gate = update
            if first_projector is None and int(
                groups["future_projector"]["nonzero_element_count"]
            ) > 0:
                first_projector = update
            if first_attention is None and int(
                groups["attention"]["nonzero_element_count"]
            ) > 0:
                first_attention = update
            if any(
                parameter.grad is not None
                for parameter in model.parameters()
            ):
                raise Phase2ExecutionError(
                    "frozen Fast-WAM received a gradient"
                )
            optimizer.step()
            torch.cuda.synchronize(cfg.device)
            peak_memory = (
                int(torch.cuda.max_memory_allocated(cfg.device)) / 2**20
            )
            peak_reserved = (
                int(torch.cuda.max_memory_reserved(cfg.device)) / 2**20
            )
            if max(peak_memory, peak_reserved) >= (
                cfg.max_gpu_memory_gb * 1024
            ):
                raise Phase2ExecutionError(
                    "Phase 2 training exceeded memory ceiling"
                )
            update_time_ms = (
                time.perf_counter() - update_started
            ) * 1000.0
            gate_after = float(adapter.gate.detach().float().cpu())
            objective_rows.extend(current_objective_rows)
            update_rows.append(
                {
                    "action_weight_mean": statistics.fmean(
                        action_weights
                    ),
                    "gate_raw_after_update": gate_after,
                    "gate_raw_before_update": gate_before,
                    "gradient_groups": groups,
                    "mean_action_loss": statistics.fmean(losses),
                    "mean_weighted_action_loss": statistics.fmean(
                        weighted_losses
                    ),
                    "objective_count": cfg.objectives_per_update,
                    "optimizer_update": update,
                    "peak_memory_mib": peak_memory,
                    "peak_reserved_memory_mib": peak_reserved,
                    "update_time_ms": update_time_ms,
                    "variant": variant,
                    "zero_weight_objective_count": sum(
                        value == 0 for value in action_weights
                    ),
                }
            )
            should_checkpoint = (
                update % cfg.checkpoint_interval == 0
                or update == cfg.optimizer_updates
            )
            if should_checkpoint:
                _validate_training_prefix(
                    objective_rows,
                    sample_ids=sample_ids,
                    cfg=cfg,
                )
                atomic_write_jsonl(
                    paths["objective_metrics"],
                    objective_rows,
                )
                atomic_write_jsonl(paths["update_metrics"], update_rows)
                checkpoint = (
                    paths["checkpoints"] / f"step_{update:08d}"
                )
                save_adapter_checkpoint(
                    checkpoint,
                    adapter=adapter,
                    optimizer=optimizer,
                    manifest=_checkpoint_manifest(
                        thought3_cfg,
                        adapter,
                        split_fingerprint=prepared.split_fingerprint,
                        cache_fingerprint=prepared.cache_fingerprint,
                        frozen_parameter_sha256=(
                            frozen_parameter_sha256
                        ),
                        global_step=update,
                        sample_cursor=(
                            update * cfg.objectives_per_update
                        ),
                        train_sample_count=cfg.objectives_per_update,
                        extra={
                            "calibration_sha256": calibration_sha256,
                            "gate_e_smoke": False,
                            "identity_schedule_sha256": (
                                identity_schedule_sha
                            ),
                            "objective_count": len(objective_rows),
                            "objective_metrics_prefix_sha256": (
                                metric_rows_sha256(objective_rows)
                            ),
                            "objectives_per_update": (
                                cfg.objectives_per_update
                            ),
                            "phase2_full_28_4": True,
                            "phase2_exploratory_full_training": True,
                            "primary_checkpoint_rule": (
                                cfg.primary_checkpoint_rule
                            ),
                            "sample_loss_weights_sha256": weight_sha,
                            "train_flow_schedule_sha256": (
                                _training_schedule_sha256(
                                    objective_rows
                                )
                            ),
                            "training_flow_slot_offset": (
                                cfg.training_flow_slot_offset
                            ),
                            "update_metrics_prefix_sha256": (
                                metric_rows_sha256(update_rows)
                            ),
                        },
                    ),
                )
                _progress(
                    "track_checkpoint",
                    objective_count=len(objective_rows),
                    optimizer_update=update,
                    variant=variant,
                )
            torch.cuda.empty_cache()
        if (
            len(objective_rows)
            != cfg.optimizer_updates * cfg.objectives_per_update
            or len(update_rows) != cfg.optimizer_updates
        ):
            raise Phase2ExecutionError(
                f"Phase 2 {variant} training did not complete 200x28"
            )
        development_rows, final_development = (
            _evaluate_grid_resumable(
                thought3_cfg,
                cfg,
                model=model,
                adapter=adapter,
                injector=injector,
                samples=development_samples,
                flow_steps=cfg.development_flow_steps,
                metrics_path=paths["development_metrics"],
                resume=resume,
                progress_stage=(
                    f"track_{variant.lower()}_development_sample_complete"
                ),
            )
        )
        del development_rows
        checkpoint = find_latest_checkpoint(paths["checkpoints"])
        if checkpoint is None:
            raise Phase2ExecutionError("Phase 2 wrote no checkpoint")
        roundtrip = _checkpoint_roundtrip(
            thought3_cfg,
            adapter,
            optimizer,
            checkpoint,
            prepared=prepared,
            frozen_parameter_sha256=frozen_parameter_sha256,
            device=cfg.device,
        )
        initial_development = calibration["development_initial"]
        initial_mean = float(initial_development["mean_action_loss"])
        final_mean = float(final_development["mean_action_loss"])
        result = {
            "adapter_fingerprint": (
                thought3_cfg.adapter_structural_fingerprint
            ),
            "adapter_state_sha256": adapter_state_sha256(
                adapter.state_dict()
            ),
            "calibration_sha256": calibration_sha256,
            "checkpoint": str(checkpoint),
            "checkpoint_roundtrip": roundtrip,
            "completed_at": _utc_now(),
            "completed_objectives": len(objective_rows),
            "completed_updates": len(update_rows),
            "development_final": final_development,
            "development_initial": initial_development,
            "development_loss_reduction_fraction": (
                (initial_mean - final_mean) / initial_mean
            ),
            "development_sample_ids": development_ids,
            "first_attention_nonzero_gradient_update": (
                first_attention
            ),
            "first_non_gate_nonzero_gradient_update": first_non_gate,
            "first_projector_nonzero_gradient_update": first_projector,
            "identity_schedule_sha256": identity_schedule_sha,
            "initial_adapter_sha256": initial_adapter_sha,
            "max_peak_memory_mib": max(
                float(row["peak_memory_mib"]) for row in update_rows
            ),
            "max_peak_reserved_memory_mib": max(
                float(row["peak_reserved_memory_mib"])
                for row in update_rows
            ),
            "mean_optimizer_update_time_ms": statistics.fmean(
                float(row["update_time_ms"]) for row in update_rows
            ),
            "objective_metrics_sha256": metric_rows_sha256(
                objective_rows
            ),
            "optimizer_parameter_scope": "adapter_only",
            "phase2_config_fingerprint": cfg.fingerprint,
            "primary_checkpoint_step": cfg.primary_checkpoint_step,
            "primary_checkpoint_rule": cfg.primary_checkpoint_rule,
            "sample_ids": sample_ids,
            "sample_loss_weights_sha256": weight_sha,
            "schema_version": PHASE2_TRACK_SCHEMA,
            "status": "complete",
            "thought3_config": thought3_cfg.to_dict(),
            "thought3_config_fingerprint": thought3_cfg.fingerprint,
            "train_flow_schedule_sha256": (
                _training_schedule_sha256(objective_rows)
            ),
            "train_flow_slot_end": cfg.training_flow_end,
            "train_flow_slot_start": cfg.training_flow_start,
            "update_metrics_sha256": metric_rows_sha256(update_rows),
            "uses_development_for_checkpoint_selection": False,
            "uses_future_rgb": False,
            "uses_ood_or_success": False,
            "variant": variant,
            "wall_s_this_invocation": time.perf_counter() - started,
            "zero_weight_objective_count": sum(
                bool(row["zero_weight_objective"])
                for row in objective_rows
            ),
        }
        atomic_write_json(paths["manifest"], result)
        return result
    finally:
        injector.close()
        del optimizer, adapter
        torch.cuda.empty_cache()


def run_phase2_track(
    cfg: Phase2FullTrainingConfig,
    *,
    variant: str,
    resume: bool,
) -> dict[str, Any]:
    """Run one real track on its sole visible physical GPU."""

    _require_confirmation()
    import torch

    from fastwam_ood_eval.thought3.model_wrapper import (
        parameter_state_sha256,
    )

    if variant not in PHASE2_VARIANTS:
        raise Phase2ProtocolError(f"invalid Phase 2 track: {variant}")
    _configure_cuda(cfg)
    base_cfg, preflight = verify_phase2_prerequisites(
        cfg,
        require_clean=True,
    )
    calibration, calibration_sha = _load_calibration(cfg)
    paths = _track_paths(cfg, variant)
    if (
        paths["result"].is_file()
        and paths["artifacts"].is_file()
        and resume
    ):
        existing, _ = _load_track_result(cfg, variant)
        if existing.get("calibration_sha256") != calibration_sha:
            raise Phase2ExecutionError(
                "completed track calibration provenance changed"
            )
        return existing
    if (
        paths["root"].exists()
        and any(paths["root"].iterdir())
        and not resume
    ):
        raise FileExistsError(
            f"Phase 2 {variant} output exists; use --resume: "
            f"{paths['root']}"
        )
    paths["root"].mkdir(parents=True, exist_ok=True)
    atomic_write_json(paths["preflight"], preflight)
    thought3_cfg = derive_phase2_thought3_config(
        cfg,
        base_cfg,
        variant=variant,
        output_dir=paths["root"],
    )
    model = upstream_cfg = prepared = None
    try:
        (
            model,
            upstream_cfg,
            prepared,
            model_report,
            frozen_before,
        ) = _load_model_and_data(cfg, thought3_cfg)
        atomic_write_json(paths["data"], dict(prepared.report))
        result = _train_phase2_track(
            cfg,
            thought3_cfg,
            model=model,
            prepared=prepared,
            frozen_parameter_sha256=frozen_before,
            calibration=calibration,
            calibration_sha256=calibration_sha,
            resume=resume,
        )
        frozen_after = parameter_state_sha256(
            iter(model.named_parameters())
        )
        if frozen_after != frozen_before:
            raise Phase2ExecutionError(
                f"frozen Fast-WAM changed in {variant}"
            )
        result = {
            **result,
            "fastwam_frozen_sha256_after": frozen_after,
            "fastwam_frozen_sha256_before": frozen_before,
            "model_load": model_report,
        }
        atomic_write_json(paths["result"], result)
        artifact_files = {}
        for key in (
            "result",
            "manifest",
            "objective_metrics",
            "update_metrics",
            "development_metrics",
            "state",
            "data",
            "preflight",
        ):
            path = paths[key]
            artifact_files[str(path.relative_to(paths["root"]))] = (
                _artifact_descriptor(path)
            )
        checkpoint_root = Path(result["checkpoint"])
        for checkpoint_file in (
            checkpoint_root / "adapter.safetensors",
            checkpoint_root / "optimizer.pt",
            checkpoint_root / "manifest.json",
        ):
            artifact_files[
                str(checkpoint_file.relative_to(paths["root"]))
            ] = _artifact_descriptor(checkpoint_file)
        atomic_write_json(
            paths["artifacts"],
            {
                "file_count": len(artifact_files),
                "files": artifact_files,
                "schema_version": PHASE2_ARTIFACT_SCHEMA,
                "track_result_sha256": sha256_file(paths["result"]),
                "variant": variant,
            },
        )
        atomic_write_json(
            paths["status"],
            {
                "completed_at": _utc_now(),
                "result_sha256": sha256_file(paths["result"]),
                "schema_version": PHASE2_TRACK_SCHEMA,
                "status": "complete",
                "variant": variant,
            },
        )
        _progress(
            "track_complete",
            development_loss_reduction_fraction=result[
                "development_loss_reduction_fraction"
            ],
            variant=variant,
        )
        return result
    except BaseException as exc:
        paths["root"].mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            paths["status"],
            {
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at": _utc_now(),
                "schema_version": PHASE2_TRACK_SCHEMA,
                "status": "failed",
                "traceback": traceback.format_exc(),
                "variant": variant,
            },
        )
        raise
    finally:
        del prepared, upstream_cfg, model
        gc.collect()
        torch.cuda.empty_cache()


def _load_track_result(
    cfg: Phase2FullTrainingConfig,
    variant: str,
) -> tuple[dict[str, Any], str]:
    paths = _track_paths(cfg, variant)
    result = load_json(paths["result"])
    artifacts = load_json(paths["artifacts"])
    if (
        result.get("schema_version") != PHASE2_TRACK_SCHEMA
        or result.get("status") != "complete"
        or result.get("variant") != variant
        or result.get("phase2_config_fingerprint") != cfg.fingerprint
        or artifacts.get("schema_version") != PHASE2_ARTIFACT_SCHEMA
        or artifacts.get("variant") != variant
    ):
        raise Phase2ExecutionError(
            f"Phase 2 {variant} track is incomplete"
        )
    for relative, descriptor in artifacts["files"].items():
        path = paths["root"] / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(descriptor["bytes"])
            or sha256_file(path) != descriptor["sha256"]
        ):
            raise Phase2ExecutionError(
                f"Phase 2 {variant} artifact changed: {relative}"
            )
    digest = sha256_file(paths["result"])
    if digest != artifacts["track_result_sha256"]:
        raise Phase2ExecutionError(
            f"Phase 2 {variant} result SHA changed"
        )
    checkpoint_root = Path(str(result["checkpoint"]))
    checkpoint_manifest = load_json(checkpoint_root / "manifest.json")
    checkpoint_extra = checkpoint_manifest.get("extra", {})
    checkpoint_files = checkpoint_extra.get("files_sha256", {})
    if (
        int(checkpoint_manifest.get("global_step", -1)) != 200
        or checkpoint_manifest.get("variant") != variant
        or checkpoint_extra.get("checkpoint_kind") != "adapter_only"
        or checkpoint_extra.get("contains_backbone") is not False
        or checkpoint_extra.get("phase2_full_28_4") is not True
        or checkpoint_extra.get(
            "phase2_exploratory_full_training"
        )
        is not True
        or checkpoint_extra.get("gate_e_smoke") is not False
        or checkpoint_extra.get("adapter_state_sha256")
        != result["adapter_state_sha256"]
        or checkpoint_extra.get("sample_loss_weights_sha256")
        != result["sample_loss_weights_sha256"]
        or checkpoint_extra.get("identity_schedule_sha256")
        != result["identity_schedule_sha256"]
        or not isinstance(checkpoint_files, Mapping)
        or set(checkpoint_files)
        != {"adapter.safetensors", "optimizer.pt"}
        or any(
            not (checkpoint_root / str(name)).is_file()
            or sha256_file(checkpoint_root / str(name)) != digest
            for name, digest in checkpoint_files.items()
        )
    ):
        raise Phase2ExecutionError(
            f"Phase 2 {variant} final checkpoint provenance changed"
        )
    return result, digest


def _phase2_report_text(result: Mapping[str, Any]) -> str:
    a0 = result["tracks"]["A0"]
    a1 = result["tracks"]["A1"]
    return (
        "# Thought3 Phase 2：完整 28/4 A0/A1 训练结果\n\n"
        f"状态：`{result['classification']}`\n\n"
        "这是 exploratory full-data training，不是 Clean/OOD rollout，"
        "也不是正式多 seed 论文结果。\n\n"
        "## 固定协议\n\n"
        "- train/development：28/4，单个 libero_goal task；\n"
        "- A0/A1：相同初始化、200 updates、每 update 28 objectives；\n"
        "- LR：3e-4；sample weights：inverse-initial-loss unit mean；\n"
        "- 主 checkpoint：固定 step 200，无 dev checkpoint 选择或回退；\n"
        "- OOD/success/future RGB：均未读取。\n\n"
        "## Development 方向\n\n"
        f"- 初始 mean loss：{result['development_initial_mean_loss']:.9f}\n"
        f"- A0 final / reduction：{a0['development_final_mean_loss']:.9f} / "
        f"{a0['development_loss_reduction_fraction']:.3%}\n"
        f"- A1 final / reduction：{a1['development_final_mean_loss']:.9f} / "
        f"{a1['development_loss_reduction_fraction']:.3%}\n"
        f"- A1 final - A0 final：{result['a1_minus_a0_final_mean_loss']:.9f}\n\n"
        "## 下一步\n\n"
        f"`{result['next_required_stage']}`。在完整 A1 checkpoint 上复验"
        " correct/null/shuffle action sensitivity 后，才可决定是否进入最小"
        " Clean/OOD pilot；本汇总不会自动解锁 Phase 3。\n"
    )


def classify_phase2_training_direction(
    *,
    hard_checks: Mapping[str, bool],
    initial_mean: float,
    a0_final_mean: float,
    a1_final_mean: float,
) -> tuple[str, str, bool]:
    """Apply the frozen Phase 2 endpoint rule without a tunable threshold."""

    values = (initial_mean, a0_final_mean, a1_final_mean)
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in values):
        raise Phase2ExecutionError(
            "Phase 2 classification requires finite nonnegative losses"
        )
    if not hard_checks or not all(
        isinstance(value, bool) for value in hard_checks.values()
    ):
        raise Phase2ExecutionError(
            "Phase 2 classification requires explicit boolean hard checks"
        )
    if not all(hard_checks.values()):
        return (
            "phase2_engineering_invalid",
            "repair_only_the_failed_phase2_hard_invariant",
            False,
        )
    direction = (
        float(a1_final_mean) < float(a0_final_mean)
        and float(a1_final_mean) < float(initial_mean)
    )
    if direction:
        return (
            "training_valid_pending_full_checkpoint_online_sensitivity",
            "phase2_full_checkpoint_online_correct_null_shuffle_recheck",
            True,
        )
    return (
        "training_valid_dev_direction_not_observed",
        "stop_before_phase3_and_register_negative_direction",
        False,
    )


def finalize_phase2_training(
    cfg: Phase2FullTrainingConfig,
    *,
    resume: bool,
) -> dict[str, Any]:
    """CPU-only matched-track aggregation and Phase 3 lock decision."""

    _require_confirmation()
    _, preflight = verify_phase2_prerequisites(
        cfg,
        require_clean=True,
    )
    root = ensure_thought3_output_path(cfg.output_dir)
    result_path = root / "phase2_training_result.json"
    report_path = root / "phase2_training_report.md"
    artifacts_path = root / "artifact_manifest.json"
    status_path = root / "run_status.json"
    if result_path.is_file() and not resume:
        raise FileExistsError(
            f"Phase 2 final result exists; use --resume: {result_path}"
        )
    if result_path.is_file() and artifacts_path.is_file() and resume:
        existing = load_json(result_path)
        artifacts = load_json(artifacts_path)
        if (
            existing.get("schema_version") != PHASE2_RESULT_SCHEMA
            or existing.get("phase2_config_fingerprint")
            != cfg.fingerprint
            or artifacts.get("schema_version")
            != PHASE2_ARTIFACT_SCHEMA
        ):
            raise Phase2ExecutionError(
                "completed Phase 2 result provenance changed"
            )
        for relative, descriptor in artifacts["files"].items():
            path = root / str(relative)
            if (
                not path.is_file()
                or path.stat().st_size != int(descriptor["bytes"])
                or sha256_file(path) != descriptor["sha256"]
            ):
                raise Phase2ExecutionError(
                    f"completed Phase 2 artifact changed: {relative}"
                )
        return existing
    calibration, calibration_sha = _load_calibration(cfg)
    tracks: dict[str, dict[str, Any]] = {}
    track_shas: dict[str, str] = {}
    for variant in PHASE2_VARIANTS:
        tracks[variant], track_shas[variant] = _load_track_result(
            cfg,
            variant,
        )
    a0 = tracks["A0"]
    a1 = tracks["A1"]
    initial_mean = float(
        calibration["development_initial"]["mean_action_loss"]
    )
    a0_final = float(a0["development_final"]["mean_action_loss"])
    a1_final = float(a1["development_final"]["mean_action_loss"])
    hard_checks = {
        "both_tracks_complete_200x28": all(
            int(track["completed_updates"]) == cfg.optimizer_updates
            and int(track["completed_objectives"])
            == cfg.optimizer_updates * cfg.objectives_per_update
            for track in tracks.values()
        ),
        "matched_sample_identity": (
            a0["sample_ids"] == a1["sample_ids"]
            == calibration["train_sample_ids"]
            and a0["development_sample_ids"]
            == a1["development_sample_ids"]
            == calibration["development_sample_ids"]
        ),
        "matched_sample_weights": (
            a0["sample_loss_weights_sha256"]
            == a1["sample_loss_weights_sha256"]
            == calibration["sample_loss_weights_sha256"]
        ),
        "matched_training_schedule": (
            a0["identity_schedule_sha256"]
            == a1["identity_schedule_sha256"]
            and a0["train_flow_schedule_sha256"]
            == a1["train_flow_schedule_sha256"]
        ),
        "fixed_step_200_primary": all(
            int(track["primary_checkpoint_step"]) == 200
            and track["primary_checkpoint_rule"]
            == cfg.primary_checkpoint_rule
            and int(track["checkpoint_roundtrip"]["global_step"]) == 200
            and track["checkpoint_roundtrip"]["state_equal"] is True
            for track in tracks.values()
        ),
        "finite_train_and_development": all(
            all(
                math.isfinite(float(value))
                for value in (
                    track["development_final"]["mean_action_loss"],
                    track["development_initial"]["mean_action_loss"],
                    track["development_loss_reduction_fraction"],
                    track["max_peak_memory_mib"],
                )
            )
            for track in tracks.values()
        ),
        "no_catastrophic_loss": all(
            float(track["development_final"]["mean_action_loss"])
            <= initial_mean * cfg.catastrophic_loss_multiplier
            for track in tracks.values()
        ),
        "frozen_fastwam_unchanged": all(
            track["fastwam_frozen_sha256_before"]
            == track["fastwam_frozen_sha256_after"]
            == calibration["fastwam_frozen_sha256_before"]
            for track in tracks.values()
        ),
        "adapter_only_checkpoint_roundtrip": all(
            track["optimizer_parameter_scope"] == "adapter_only"
            and track["checkpoint_roundtrip"]["state_equal"] is True
            for track in tracks.values()
        ),
        "first_two_update_gradient_contract": all(
            int(track["first_non_gate_nonzero_gradient_update"]) == 2
            and int(track["first_projector_nonzero_gradient_update"]) == 2
            and int(track["first_attention_nonzero_gradient_update"]) == 2
            for track in tracks.values()
        ),
        "memory_usable_on_4090": all(
            max(
                float(track["max_peak_memory_mib"]),
                float(track["max_peak_reserved_memory_mib"]),
                float(track["model_load"]["load_peak_allocated_mib"]),
                float(track["model_load"]["load_peak_reserved_mib"]),
            )
            < cfg.max_gpu_memory_gb * 1024
            for track in tracks.values()
        ),
        "scope_preserved": all(
            track["uses_development_for_checkpoint_selection"] is False
            and track["uses_future_rgb"] is False
            and track["uses_ood_or_success"] is False
            for track in tracks.values()
        ),
    }
    classification, next_stage, direction_observed = (
        classify_phase2_training_direction(
            hard_checks=hard_checks,
            initial_mean=initial_mean,
            a0_final_mean=a0_final,
            a1_final_mean=a1_final,
        )
    )
    result = {
        "a1_minus_a0_final_mean_loss": a1_final - a0_final,
        "calibration_sha256": calibration_sha,
        "classification": classification,
        "completed_at": _utc_now(),
        "development_direction_observed": direction_observed,
        "development_initial_mean_loss": initial_mean,
        "hard_checks": hard_checks,
        "next_required_stage": next_stage,
        "phase2_config_fingerprint": cfg.fingerprint,
        "phase3_unlocked": False,
        "preflight": preflight,
        "recipe_selection_disclosure": (
            cfg.recipe_selection_disclosure
        ),
        "schema_version": PHASE2_RESULT_SCHEMA,
        "scope": dict(cfg.scope),
        "status": "complete",
        "track_result_sha256": track_shas,
        "tracks": {
            variant: {
                "adapter_state_sha256": track[
                    "adapter_state_sha256"
                ],
                "checkpoint": track["checkpoint"],
                "development_final_mean_loss": float(
                    track["development_final"]["mean_action_loss"]
                ),
                "development_loss_reduction_fraction": float(
                    track["development_loss_reduction_fraction"]
                ),
                "max_peak_memory_mib": float(
                    track["max_peak_memory_mib"]
                ),
                "max_peak_reserved_memory_mib": float(
                    track["max_peak_reserved_memory_mib"]
                ),
                "mean_optimizer_update_time_ms": float(
                    track["mean_optimizer_update_time_ms"]
                ),
            }
            for variant, track in tracks.items()
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(result_path, result)
    atomic_write_text(report_path, _phase2_report_text(result))
    artifact_files = {
        "phase2_training_result.json": _artifact_descriptor(result_path),
        "phase2_training_report.md": _artifact_descriptor(report_path),
        "calibration/calibration.json": _artifact_descriptor(
            _calibration_paths(cfg)["result"]
        ),
        "tracks/a0/track_result.json": _artifact_descriptor(
            _track_paths(cfg, "A0")["result"]
        ),
        "tracks/a1/track_result.json": _artifact_descriptor(
            _track_paths(cfg, "A1")["result"]
        ),
    }
    atomic_write_json(
        artifacts_path,
        {
            "file_count": len(artifact_files),
            "files": artifact_files,
            "schema_version": PHASE2_ARTIFACT_SCHEMA,
        },
    )
    atomic_write_json(
        status_path,
        {
            "classification": classification,
            "completed_at": _utc_now(),
            "next_required_stage": next_stage,
            "phase3_unlocked": False,
            "schema_version": PHASE2_RESULT_SCHEMA,
            "status": "complete",
        },
    )
    _progress(
        "finalize_complete",
        classification=classification,
        next_required_stage=next_stage,
        phase3_unlocked=False,
    )
    return result


def run_phase2_stage(
    cfg: Phase2FullTrainingConfig,
    *,
    stage: str,
    resume: bool,
) -> dict[str, Any]:
    """Dispatch one explicit stage without silently starting another."""

    if stage == "calibrate":
        return run_phase2_calibration(cfg, resume=resume)
    if stage in PHASE2_VARIANTS:
        return run_phase2_track(cfg, variant=stage, resume=resume)
    if stage == "finalize":
        return finalize_phase2_training(cfg, resume=resume)
    raise Phase2ProtocolError(f"invalid Phase 2 stage: {stage}")
