"""Gate E: real A0/A1 Adapter training and deterministic-resume smoke.

Gate E is deliberately an engineering gate.  It uses the frozen 32-sample
Phase D cache, one standard LIBERO task, one visible GPU, and no simulator or
OOD outcomes.  Each A0/A1 primary track is interrupted at step 50 and resumed
to step 100.  A separately initialized uninterrupted 100-step reference proves
that checkpoint restore preserves exact Adapter semantics.
"""

from __future__ import annotations

import gc
import os
import time
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from fastwam_ood_eval.thought3.cache_validator import validate_cache
from fastwam_ood_eval.thought3.config import (
    Thought3Config,
    validate_config,
)
from fastwam_ood_eval.thought3.io_utils import (
    atomic_write_json,
    load_json,
    load_jsonl,
    sha256_file,
)
from fastwam_ood_eval.thought3.phase_c_smoke import _load_upstream_model
from fastwam_ood_eval.thought3.real_training import (
    PreparedRealTrainingData,
    prepare_real_training_data,
    run_real_variant_training,
)
from fastwam_ood_eval.thought3.safety import ensure_thought3_output_path


PHASE_E_SCHEMA = "thought3.phase_e.gate.v1"
PHASE_E_INTERRUPTION_STEP = 50
OFFICIAL_LIBERO_REVISION = "117413dc0ca99c7cd64036c4eaa4a316c537d692"
PHASE_D_CACHE_FINGERPRINT = (
    "63a70e1af38f68bc894fc11d03c84f212e6c6328a5051256c9d045741156d9c5"
)
PHASE_D_SPLIT_FINGERPRINT = (
    "ea5402955023ccd48d790d821a73f98549b31d1ace8af035a90ceae2ad3951eb"
)
PHASE_D_FROZEN: Mapping[str, str] = {
    "outputs/thought3/phase_d_cache_smoke_v1/run_status.json": (
        "d302cd63d3fd18161775f92ac3aa9d18e84842ee97b3316fe0f427df2e819baa"
    ),
    "outputs/thought3/phase_d_cache_smoke_v1/gate_d_result.json": (
        "a636d649491ad9df67a1ea2cb91d8e9bf708784a410ba7b8304248f33ed1882d"
    ),
    "outputs/thought3/phase_d_cache_smoke_v1/logs/phase_d.log": (
        "97cdb718877a2c58a0a11352102d874b4c1b670b38ed090e90d60f91e5412d84"
    ),
    (
        "outputs/thought3/phase_d_cache_smoke_v1/"
        "phase_d_inventory_manifest.json"
    ): "a53735d5ac62738284a22a5b6422beb7edb5290d04d92f4ea7057986a6c01b9a",
    (
        "outputs/thought3/cache/phase_d_libero_goal_task0_v1/"
        "cache_plan_manifest.json"
    ): "c4ab6c4e3b4c205f5366de034ee5f6c202a420d07a058bad1d0d3eb86731eac9",
    (
        "outputs/thought3/cache/phase_d_libero_goal_task0_v1/"
        "cache_manifest.json"
    ): "1a0d73b1e4e6a4b12ac367b50f3a49f04a81a4ed6f692d957129d3b9d2f75816",
    (
        "outputs/thought3/cache/phase_d_libero_goal_task0_v1/"
        "real_cache_build_report.json"
    ): "221f32039df792a3b4d64dbe35bcedf7d99741f70bb6381e374fac903027f8c5",
}


class PhaseEGateError(RuntimeError):
    """Raised when real training violates the frozen Gate E protocol."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress(stage: str, **values: Any) -> None:
    import json

    print(
        json.dumps(
            {
                "phase": "E",
                "stage": stage,
                "time": _utc_now(),
                **values,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _assert_phase_e_scope(cfg: Thought3Config) -> None:
    """Reject any silent expansion beyond the registered one-task smoke."""

    if cfg.runtime.backend != "fastwam" or cfg.runtime.device != "cuda:0":
        raise PhaseEGateError(
            "Phase E requires backend=fastwam and logical cuda:0"
        )
    if cfg.variant != "A1" or cfg.sampler.active_k != 1:
        raise PhaseEGateError(
            "Phase E orchestration config must be A1/active_k=1"
        )
    if tuple(cfg.sampler.cache_k) != (1, 2, 4):
        raise PhaseEGateError("Phase E requires the paired Phase D cache")
    if (
        cfg.cache.root
        != Path("outputs/thought3/cache/phase_d_libero_goal_task0_v1")
        or cfg.cache.pilot_limit != 32
        or cfg.cache.shard_size != 8
    ):
        raise PhaseEGateError(
            "Phase E is frozen to the 32-sample Phase D cache"
        )
    if len(cfg.data.dataset_roots) != 1:
        raise PhaseEGateError("Phase E accepts one standard LIBERO root")
    if cfg.data.dataset_revision != OFFICIAL_LIBERO_REVISION:
        raise PhaseEGateError("Phase E LIBERO revision mismatch")
    if cfg.data.inventory_path != Path(
        "outputs/thought3/phase_d_cache_smoke_v1/inventory.jsonl"
    ):
        raise PhaseEGateError("Phase E inventory is not the frozen Phase D one")
    if (
        cfg.data.split_seed != 3407
        or cfg.experiment.seed != 3407
        or cfg.training.train_seed != 3407
    ):
        raise PhaseEGateError("Phase E seeds must remain 3407")
    if (
        cfg.training.max_steps != 100
        or cfg.training.microbatch_size != 1
        or cfg.training.gradient_accumulation_steps != 1
        or cfg.training.checkpoint_interval != 25
    ):
        raise PhaseEGateError(
            "Phase E budget is frozen to 100 steps, batch 1, checkpoint 25"
        )
    if (
        cfg.training.learning_rate != 1e-3
        or cfg.training.weight_decay != 1e-2
        or cfg.training.gradient_checkpointing
        or cfg.training.gate_l2 != 0
    ):
        raise PhaseEGateError("Phase E optimizer recipe changed")
    if cfg.runtime.online_use_cache:
        raise PhaseEGateError("Phase E forbids online training-cache access")


def _verify_phase_d_gate(cfg: Thought3Config) -> dict[str, Any]:
    observed: dict[str, str] = {}
    for raw_path, expected in PHASE_D_FROZEN.items():
        path = Path(raw_path)
        if not path.is_file():
            raise PhaseEGateError(f"frozen Phase D artifact missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise PhaseEGateError(
                f"frozen Phase D artifact changed: {path}; "
                f"expected={expected}, actual={actual}"
            )
        observed[raw_path] = actual
    status = load_json(
        "outputs/thought3/phase_d_cache_smoke_v1/run_status.json"
    )
    result = load_json(
        "outputs/thought3/phase_d_cache_smoke_v1/gate_d_result.json"
    )
    if (
        status.get("gate_d_passed") is not True
        or result.get("gate_d_passed") is not True
    ):
        raise PhaseEGateError("frozen Phase D gate does not report pass")
    cache = validate_cache(cfg.cache.root)
    split_fingerprint = result.get("plan", {}).get(
        "split_fingerprint"
    )
    if (
        cache.get("cache_fingerprint") != PHASE_D_CACHE_FINGERPRINT
        or split_fingerprint != PHASE_D_SPLIT_FINGERPRINT
        or int(cache.get("entry_count", -1)) != 96
        or int(cache.get("shard_count", -1)) != 12
        or cache.get("uses_ground_truth_future") is not False
    ):
        raise PhaseEGateError(f"Phase D cache validation changed: {cache}")
    return {
        "artifact_sha256": observed,
        "cache_fingerprint": cache["cache_fingerprint"],
        "entry_count": cache["entry_count"],
        "shard_count": cache["shard_count"],
        "split_fingerprint": split_fingerprint,
    }


def derive_variant_config(
    cfg: Thought3Config,
    *,
    variant: str,
    track: str,
) -> Thought3Config:
    """Derive a matched A0/A1 config without changing the source YAML."""

    if variant not in {"A0", "A1"}:
        raise PhaseEGateError(f"unsupported Phase E variant: {variant}")
    if track not in {"resumed", "uninterrupted"}:
        raise PhaseEGateError(f"unsupported Phase E track: {track}")
    active_k = 0 if variant == "A0" else 1
    name = f"thought3_phase_e_{variant.lower()}_{track}"
    output = (
        cfg.experiment.output_dir
        / "variants"
        / variant.lower()
        / track
    )
    derived = replace(
        cfg,
        variant=variant,
        experiment=replace(
            cfg.experiment,
            name=name,
            output_dir=output,
        ),
        sampler=replace(cfg.sampler, active_k=active_k),
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


def _derive_all_configs(
    cfg: Thought3Config,
) -> dict[str, dict[str, Thought3Config]]:
    values = {
        variant: {
            track: derive_variant_config(
                cfg,
                variant=variant,
                track=track,
            )
            for track in ("resumed", "uninterrupted")
        }
        for variant in ("A0", "A1")
    }
    recipes = {
        str(_matched_recipe_payload(track_cfg))
        for by_track in values.values()
        for track_cfg in by_track.values()
    }
    if len(recipes) != 1:
        raise PhaseEGateError("A0/A1 real training recipes are not matched")
    return values


def _latest_step(cfg: Thought3Config) -> int:
    from fastwam_ood_eval.thought3.checkpointing import (
        find_latest_checkpoint,
    )
    from fastwam_ood_eval.thought3.schemas import AdapterCheckpointManifest

    checkpoint = find_latest_checkpoint(
        cfg.experiment.output_dir / "checkpoints"
    )
    if checkpoint is None:
        return 0
    manifest = AdapterCheckpointManifest.from_dict(
        load_json(checkpoint / "manifest.json")
    )
    return manifest.global_step


def _run_resumed_track(
    cfg: Thought3Config,
    *,
    model: Any,
    prepared: PreparedRealTrainingData,
    frozen_parameter_sha256: str,
    resume: bool,
) -> dict[str, Any]:
    step = _latest_step(cfg)
    if step == 0:
        run_real_variant_training(
            cfg,
            model=model,
            prepared=prepared,
            frozen_parameter_sha256=frozen_parameter_sha256,
            resume=False,
            device="cuda:0",
            stop_after_steps=PHASE_E_INTERRUPTION_STEP,
            progress=_progress,
        )
        step = _latest_step(cfg)
    elif not resume:
        raise FileExistsError(
            f"Phase E resumed track already exists: {cfg.experiment.output_dir}"
        )
    if step < PHASE_E_INTERRUPTION_STEP:
        run_real_variant_training(
            cfg,
            model=model,
            prepared=prepared,
            frozen_parameter_sha256=frozen_parameter_sha256,
            resume=True,
            device="cuda:0",
            stop_after_steps=PHASE_E_INTERRUPTION_STEP,
            progress=_progress,
        )
        step = _latest_step(cfg)
    if step != PHASE_E_INTERRUPTION_STEP and step != cfg.training.max_steps:
        raise PhaseEGateError(
            f"planned interruption must commit step 50, observed step {step}"
        )
    result = run_real_variant_training(
        cfg,
        model=model,
        prepared=prepared,
        frozen_parameter_sha256=frozen_parameter_sha256,
        resume=True,
        device="cuda:0",
        progress=_progress,
    )
    if int(result["completed_steps"]) != cfg.training.max_steps:
        raise PhaseEGateError(f"{cfg.variant} resumed track did not finish")
    return result


def _run_uninterrupted_track(
    cfg: Thought3Config,
    *,
    model: Any,
    prepared: PreparedRealTrainingData,
    frozen_parameter_sha256: str,
    resume: bool,
) -> dict[str, Any]:
    manifest_path = cfg.experiment.output_dir / "training_manifest.json"
    if manifest_path.is_file():
        existing = load_json(manifest_path)
        if (
            resume
            and existing.get("status") == "complete"
            and int(existing.get("completed_steps", -1))
            == cfg.training.max_steps
        ):
            return existing
        raise PhaseEGateError(
            "the uninterrupted reference was interrupted or reused; "
            "use a new Phase E run ID for a valid exact-resume proof"
        )
    if cfg.experiment.output_dir.exists() and _latest_step(cfg):
        raise PhaseEGateError(
            "the uninterrupted reference contains a partial checkpoint"
        )
    return run_real_variant_training(
        cfg,
        model=model,
        prepared=prepared,
        frozen_parameter_sha256=frozen_parameter_sha256,
        resume=False,
        device="cuda:0",
        progress=_progress,
    )


def _semantic_adapter_sha256(result: Mapping[str, Any]) -> str:
    roundtrip = result.get("checkpoint_roundtrip")
    if not isinstance(roundtrip, Mapping):
        raise PhaseEGateError("training result has no checkpoint round-trip")
    value = str(roundtrip.get("adapter_state_sha256", ""))
    if len(value) != 64:
        raise PhaseEGateError("training result has no Adapter semantic hash")
    return value


def _track_artifacts(
    cfg: Thought3Config,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    paths = {
        "development_metrics": cfg.experiment.output_dir
        / "development_metrics.jsonl",
        "metrics": cfg.experiment.output_dir / "train_metrics.jsonl",
        "status": cfg.experiment.output_dir / "run_status.json",
        "training_manifest": cfg.experiment.output_dir
        / "training_manifest.json",
        "training_state": cfg.experiment.output_dir / "training_state.json",
    }
    checkpoint = Path(str(result["checkpoint"]))
    paths.update(
        {
            "adapter": checkpoint / "adapter.safetensors",
            "checkpoint_manifest": checkpoint / "manifest.json",
            "optimizer": checkpoint / "optimizer.pt",
        }
    )
    for path in paths.values():
        if not path.is_file():
            raise PhaseEGateError(f"Phase E artifact missing: {path}")
    return {
        "files_sha256": {
            name: sha256_file(path) for name, path in paths.items()
        },
        "output_dir": str(cfg.experiment.output_dir),
    }


def _validate_track(
    cfg: Thought3Config,
    result: Mapping[str, Any],
    *,
    expected_resume_step: int,
) -> dict[str, Any]:
    if (
        result.get("status") != "complete"
        or int(result.get("completed_steps", -1)) != 100
        or int(result.get("resumed_from_step", -1))
        != expected_resume_step
        or result.get("optimizer_parameter_scope") != "adapter_only"
        or result.get("uses_ground_truth_future_input") is not False
        or result.get("selection_split") != "development"
        or result.get("selection_metric") != "development_action_loss"
    ):
        raise PhaseEGateError(
            f"{cfg.variant}/{cfg.experiment.name} manifest failed: {result}"
        )
    if (
        result.get("loss_decreased") is not True
        or float(result["final_validation_action_loss"])
        >= float(result["initial_validation_action_loss"])
    ):
        raise PhaseEGateError(
            f"{cfg.variant} development loss did not decrease"
        )
    for key in (
        "first_non_gate_nonzero_gradient_step",
        "first_projector_nonzero_gradient_step",
        "first_attention_nonzero_gradient_step",
    ):
        if int(result.get(key, -1)) != 2:
            raise PhaseEGateError(
                f"{cfg.variant} expected {key}=2, got {result.get(key)}"
            )
    if float(result["max_peak_memory_mib"]) >= 43 * 1024:
        raise PhaseEGateError(f"{cfg.variant} exceeded 43 GiB")

    metrics = load_jsonl(cfg.experiment.output_dir / "train_metrics.jsonl")
    if [int(row["global_step"]) for row in metrics] != list(range(1, 101)):
        raise PhaseEGateError(f"{cfg.variant} metrics are not 1..100")
    first, second = metrics[0], metrics[1]
    if (
        float(first["gradient_groups"]["gate"]["l2"]) <= 0
        or int(
            first["gradient_groups"]["non_gate"][
                "nonzero_element_count"
            ]
        )
        != 0
        or int(
            second["gradient_groups"]["future_projector"][
                "nonzero_element_count"
            ]
        )
        <= 0
        or int(
            second["gradient_groups"]["attention"][
                "nonzero_element_count"
            ]
        )
        <= 0
        or any(bool(row["nan_or_inf"]) for row in metrics)
    ):
        raise PhaseEGateError(
            f"{cfg.variant} zero-gate/two-step gradient trace failed"
        )
    development = load_jsonl(
        cfg.experiment.output_dir / "development_metrics.jsonl"
    )
    if [int(row["global_step"]) for row in development] != [
        0,
        25,
        50,
        75,
        100,
    ]:
        raise PhaseEGateError(
            f"{cfg.variant} development checkpoint schedule changed"
        )
    return {
        "artifacts": _track_artifacts(cfg, result),
        "base_sample_order": [
            str(row["base_sample_id"]) for row in metrics
        ],
        "first_step_loss": float(first["loss"]),
        "semantic_adapter_sha256": _semantic_adapter_sha256(result),
    }


def _run_phase_e(
    cfg: Thought3Config,
    *,
    resume: bool,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from fastwam_ood_eval.thought3.model_wrapper import (
        parameter_state_sha256,
    )

    _assert_phase_e_scope(cfg)
    if os.environ.get("CONFIRM_THOUGHT3_PHASE_E") != "YES":
        raise PhaseEGateError(
            "set CONFIRM_THOUGHT3_PHASE_E=YES for real Adapter training"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise PhaseEGateError(
            "Phase E requires exactly one CUDA-visible GPU"
        )
    torch.cuda.set_device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(cfg.experiment.seed)
    torch.manual_seed(cfg.experiment.seed)
    torch.cuda.manual_seed_all(cfg.experiment.seed)

    phase_d = _verify_phase_d_gate(cfg)
    variants = _derive_all_configs(cfg)
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
    if (
        prepared.report["current_source"]["actual_future_read"] is not False
        or prepared.report["current_source"]["future_rgb_frames_decoded"] != 0
        or prepared.report["current_source"]["action_target_rows_read"] != 1024
        or prepared.report["future_rgb_used_as_input"] is not False
    ):
        raise PhaseEGateError("real training data access audit failed")
    _progress(
        "training_data_ready",
        samples=prepared.report["sample_count"],
        split_counts=prepared.report["split_counts"],
    )

    frozen_before = parameter_state_sha256(
        iter(model.named_parameters())
    )
    _progress("frozen_hash_before", sha256=frozen_before)
    results: dict[str, dict[str, Mapping[str, Any]]] = {}
    validations: dict[str, dict[str, Mapping[str, Any]]] = {}
    for variant in ("A0", "A1"):
        resumed_cfg = variants[variant]["resumed"]
        uninterrupted_cfg = variants[variant]["uninterrupted"]
        _progress("variant_resumed_started", variant=variant)
        resumed_result = _run_resumed_track(
            resumed_cfg,
            model=model,
            prepared=prepared,
            frozen_parameter_sha256=frozen_before,
            resume=resume,
        )
        _progress("variant_uninterrupted_started", variant=variant)
        uninterrupted_result = _run_uninterrupted_track(
            uninterrupted_cfg,
            model=model,
            prepared=prepared,
            frozen_parameter_sha256=frozen_before,
            resume=resume,
        )
        results[variant] = {
            "resumed": resumed_result,
            "uninterrupted": uninterrupted_result,
        }
        validations[variant] = {
            "resumed": _validate_track(
                resumed_cfg,
                resumed_result,
                expected_resume_step=PHASE_E_INTERRUPTION_STEP,
            ),
            "uninterrupted": _validate_track(
                uninterrupted_cfg,
                uninterrupted_result,
                expected_resume_step=0,
            ),
        }
        if (
            validations[variant]["resumed"][
                "semantic_adapter_sha256"
            ]
            != validations[variant]["uninterrupted"][
                "semantic_adapter_sha256"
            ]
        ):
            raise PhaseEGateError(
                f"{variant} resumed/uninterrupted Adapter hash differs"
            )
        _progress(
            "variant_complete",
            adapter_sha256=validations[variant]["resumed"][
                "semantic_adapter_sha256"
            ],
            variant=variant,
        )

    a0 = results["A0"]["resumed"]
    a1 = results["A1"]["resumed"]
    if (
        a0["initial_adapter_sha256"] != a1["initial_adapter_sha256"]
        or a0["trainable_parameter_count"]
        != a1["trainable_parameter_count"]
        or a0["adapter_fingerprint"] != a1["adapter_fingerprint"]
        or validations["A0"]["resumed"]["base_sample_order"]
        != validations["A1"]["resumed"]["base_sample_order"]
        or validations["A0"]["resumed"]["first_step_loss"]
        != validations["A1"]["resumed"]["first_step_loss"]
        or a0["initial_validation_action_loss"]
        != a1["initial_validation_action_loss"]
    ):
        raise PhaseEGateError("A0/A1 paired-recipe invariants failed")

    frozen_after = parameter_state_sha256(
        iter(model.named_parameters())
    )
    if frozen_after != frozen_before:
        raise PhaseEGateError("frozen Fast-WAM parameter hash changed")
    _progress("frozen_hash_after", sha256=frozen_after)

    result: dict[str, Any] = {
        "completed_at": _utc_now(),
        "config": cfg.to_dict(),
        "config_fingerprint": cfg.fingerprint,
        "data_preparation": dict(prepared.report),
        "determinism": {
            "a0_resumed_equals_uninterrupted": True,
            "a1_resumed_equals_uninterrupted": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
        },
        "frozen_parameter_sha256_after": frozen_after,
        "frozen_parameter_sha256_before": frozen_before,
        "gate_e_passed": True,
        "model_load": model_report,
        "paired_recipe": {
            "adapter_fingerprint": a0["adapter_fingerprint"],
            "initial_adapter_sha256": a0[
                "initial_adapter_sha256"
            ],
            "initial_action_loss_equal": True,
            "initial_validation_action_loss_equal": True,
            "sample_order_equal": True,
            "trainable_parameter_count": a0[
                "trainable_parameter_count"
            ],
        },
        "phase_d_frozen": phase_d,
        "schema_version": PHASE_E_SCHEMA,
        "scope": {
            "future_rgb_frames_read": 0,
            "long_training_started": False,
            "ood_evaluation_started": False,
            "primary_optimizer_steps": 200,
            "reference_optimizer_steps": 200,
            "rollout_started": False,
            "single_gpu": True,
            "suite_count": 1,
            "task_count": 1,
            "uses_ground_truth_future": False,
        },
        "status": "passed",
        "tracks": {
            variant: {
                track: {
                    "artifacts": validations[variant][track][
                        "artifacts"
                    ],
                    "result": dict(results[variant][track]),
                    "semantic_adapter_sha256": validations[variant][
                        track
                    ]["semantic_adapter_sha256"],
                }
                for track in ("resumed", "uninterrupted")
            }
            for variant in ("A0", "A1")
        },
    }
    del prepared, upstream_cfg, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_phase_e_training_smoke(
    cfg: Thought3Config,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Run Gate E and atomically record its final pass/fail state."""

    output = ensure_thought3_output_path(cfg.experiment.output_dir)
    result_path = output / "gate_e_result.json"
    status_path = output / "run_status.json"
    if result_path.exists():
        if not resume:
            raise FileExistsError(
                f"Phase E result exists; pass --resume: {result_path}"
            )
        existing = load_json(result_path)
        if existing.get("gate_e_passed") is True:
            return existing
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        status_path,
        {
            "schema_version": PHASE_E_SCHEMA,
            "started_at": _utc_now(),
            "status": "running",
        },
    )
    started = time.perf_counter()
    try:
        result = _run_phase_e(cfg, resume=resume)
    except BaseException as exc:
        atomic_write_json(
            status_path,
            {
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at": _utc_now(),
                "gate_e_passed": False,
                "schema_version": PHASE_E_SCHEMA,
                "status": "failed",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    result["gate_wall_s"] = time.perf_counter() - started
    atomic_write_json(result_path, result)
    atomic_write_json(
        status_path,
        {
            "finished_at": _utc_now(),
            "gate_e_passed": True,
            "result": str(result_path.resolve()),
            "schema_version": PHASE_E_SCHEMA,
            "status": "passed",
        },
    )
    return result
