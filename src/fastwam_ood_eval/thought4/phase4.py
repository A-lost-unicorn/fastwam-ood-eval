"""Top-level dry-run, real smoke and formal Phase 4 execution."""

from __future__ import annotations

import gc
import json
import math
import os
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from fastwam_ood_eval.thought4.audit import (
    require_project_clean,
    runtime_model_audit,
    static_audit,
)
from fastwam_ood_eval.thought4.cohort import (
    materialized_cohort_manifest,
    plan_base_states,
    planned_cohort_manifest,
)
from fastwam_ood_eval.thought4.config import Thought4Config, config_to_dict
from fastwam_ood_eval.thought4.io_utils import (
    Thought4ArtifactError,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    ensure_run_mutable,
    write_or_verify_json,
    write_or_verify_jsonl,
    write_or_verify_text,
)
from fastwam_ood_eval.thought4.schemas import sha256_canonical


class Phase4ExecutionError(RuntimeError):
    """Raised when a Phase 4 gate/scope/integrity check fails."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress(stage: str, **fields: Any) -> None:
    print(
        json.dumps(
            {"phase": "4", "stage": stage, "time": _utc_now(), **fields},
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _json_artifact(
    path: Path,
    payload: Mapping[str, Any],
    *,
    resume: bool,
) -> Path:
    return (
        write_or_verify_json(path, payload)
        if resume
        else atomic_write_json(path, payload)
    )


def _text_artifact(path: Path, value: str, *, resume: bool) -> Path:
    return (
        write_or_verify_text(path, value)
        if resume
        else atomic_write_text(path, value)
    )


def dry_run_payload(cfg: Thought4Config, *, stage: str) -> dict[str, Any]:
    """Strictly read-only: no Torch import, CUDA/model load or artifact write."""

    if stage not in {"smoke", "formal"}:
        raise Phase4ExecutionError("stage must be smoke or formal")
    audit = static_audit(cfg)
    plans = plan_base_states(cfg.cohort, horizon=cfg.probe.horizon)
    planned = planned_cohort_manifest(
        plans, config_fingerprint=cfg.fingerprint
    )
    return {
        "schema_version": "thought4.phase4.dry_run.v1",
        "stage": stage,
        "dry_run": True,
        "config_fingerprint": cfg.fingerprint,
        "output_dir": str(cfg.experiment.output_dir),
        "planned_base_states": len(plans),
        "execution_base_states": min(2, len(plans)) if stage == "smoke" else len(plans),
        "condition_count_per_base_state": len(cfg.cohort.conditions),
        "video_layers": list(cfg.backbone.video_layers),
        "action_hooks": list(cfg.backbone.action_hooks),
        "planned_split_counts": {
            split: sum(value.split == split for value in plans)
            for split in ("train", "development", "test")
        },
        "planned_cohort_sha256": planned["manifest_sha256"],
        "static_audit_sha256": audit["audit_sha256"],
        "would_load_torch": False,
        "would_load_gpu_model": False,
        "would_construct_simulator": False,
        "would_write": False,
        "would_read_future_rgb": False,
        "would_read_success_outcome": False,
        "formal_requires_completed_smoke": stage == "formal",
        "confirmation_required": (
            "CONFIRM_THOUGHT4_PHASE4_SMOKE=YES"
            if stage == "smoke"
            else "CONFIRM_THOUGHT4_PHASE4_FORMAL=YES"
        ),
    }


def _verify_formal_smoke_gate(
    cfg: Thought4Config,
    *,
    expected_project_commit: str | None = None,
    smoke_config_path: str | Path = (
        "configs/thought4/phase4_geometry_action_smoke.yaml"
    ),
) -> dict[str, Any]:
    from fastwam_ood_eval.thought4.config import load_thought4_config

    smoke_cfg = load_thought4_config(smoke_config_path)
    smoke_root = smoke_cfg.experiment.output_dir
    status_path = smoke_root / "run_status.json"
    result_path = smoke_root / "smoke_result.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise Phase4ExecutionError(
            "formal is locked until the frozen real Thought4 smoke completes"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase4ExecutionError(
            "formal smoke prerequisite artifacts are invalid"
        ) from exc
    if not isinstance(status, dict) or not isinstance(result, dict):
        raise Phase4ExecutionError(
            "formal smoke prerequisite artifacts must be JSON objects"
        )
    checks = {
        "status_complete": status.get("status") == "complete",
        "status_stage_smoke": status.get("stage") == "smoke",
        "status_config_matches": (
            status.get("config_fingerprint") == smoke_cfg.fingerprint
        ),
        "result_passed": result.get("status") == "passed",
        "result_not_scientific": result.get("scientific_result") is False,
        "result_formal_unlocked": result.get("formal_unlocked") is True,
        "result_config_matches": (
            result.get("config_fingerprint") == smoke_cfg.fingerprint
        ),
        "backbone_before_matches": (
            result.get("backbone_parameter_sha256_before")
            == cfg.backbone.frozen_parameter_sha256
        ),
        "backbone_after_matches": (
            result.get("backbone_parameter_sha256_after")
            == cfg.backbone.frozen_parameter_sha256
        ),
        "identity_replacement_passed": (
            result.get("identity_replacement", {}).get("passed") is True
        ),
        "identity_replacement_boundary": (
            result.get("identity_replacement", {}).get("module_path")
            == "mot.video_kv_cache.15.v"
            and result.get("identity_replacement", {}).get("hook_location")
            == "forward_action_with_video_cache argument"
        ),
        "future_rgb_not_read": result.get("future_rgb_read") is False,
        "success_outcome_not_read": result.get("success_outcome_read") is False,
    }
    if expected_project_commit is not None:
        checks["same_project_commit"] = (
            result.get("project_commit") == expected_project_commit
        )
    supplied_sha = result.get("result_sha256")
    unhashed = dict(result)
    unhashed.pop("result_sha256", None)
    checks["result_sha_valid"] = supplied_sha == sha256_canonical(unhashed)
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise Phase4ExecutionError(
            f"formal smoke prerequisite hard checks failed: {failed}"
        )
    return {
        "schema_version": "thought4.phase4.formal_smoke_gate.v1",
        "smoke_config_fingerprint": smoke_cfg.fingerprint,
        "smoke_result_sha256": supplied_sha,
        "checks": checks,
        "passed": True,
    }


def _require_confirmation(stage: str) -> None:
    variable = (
        "CONFIRM_THOUGHT4_PHASE4_SMOKE"
        if stage == "smoke"
        else "CONFIRM_THOUGHT4_PHASE4_FORMAL"
    )
    if os.environ.get(variable) != "YES":
        raise Phase4ExecutionError(f"{stage} requires {variable}=YES")
    physical = os.environ.get("THOUGHT4_GPU_ID", "")
    if not physical.isdigit():
        raise Phase4ExecutionError(
            "Thought4 v1 requires exactly one physical THOUGHT4_GPU_ID"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != physical:
        raise Phase4ExecutionError(
            "CUDA_VISIBLE_DEVICES does not match THOUGHT4_GPU_ID"
        )


def _prepare_run(cfg: Thought4Config, stage: str, *, resume: bool) -> Path:
    output = ensure_run_mutable(cfg.experiment.output_dir)
    status = output / "run_status.json"
    prior: dict[str, Any] | None = None
    if status.exists():
        if not resume:
            raise Thought4ArtifactError(
                f"partial run exists; inspect it and pass --resume: {output}"
            )
        try:
            prior = json.loads(status.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Thought4ArtifactError(
                f"invalid partial run status: {status}"
            ) from exc
        if prior.get("stage") != stage:
            raise Thought4ArtifactError("resume stage differs from partial run")
        if prior.get("config_fingerprint") != cfg.fingerprint:
            raise Thought4ArtifactError(
                "resume config fingerprint differs from partial run"
            )
    output.mkdir(parents=True, exist_ok=True)
    now = _utc_now()
    atomic_write_json(
        status,
        {
            "schema_version": "thought4.phase4.run_status.v1",
            "status": "running",
            "stage": stage,
            "started_at": (
                prior.get("started_at", now) if prior is not None else now
            ),
            "last_resumed_at": now if prior is not None else None,
            "config_fingerprint": cfg.fingerprint,
            "physical_gpu_id": os.environ.get("THOUGHT4_GPU_ID"),
        },
        overwrite=status.exists(),
    )
    return output


def _finish_status(
    output: Path,
    *,
    stage: str,
    status: str,
    started_at: str,
    error: str | None = None,
) -> None:
    status_path = output / "run_status.json"
    prior: dict[str, Any] = {}
    if status_path.is_file():
        try:
            value = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                prior = value
        except (OSError, json.JSONDecodeError):
            prior = {}
    atomic_write_json(
        status_path,
        {
            **prior,
            "schema_version": "thought4.phase4.run_status.v1",
            "status": status,
            "stage": stage,
            "started_at": prior.get("started_at", started_at),
            "finished_at": _utc_now(),
            "error": error,
        },
        overwrite=True,
    )


def _write_common_render_artifacts(
    output: Path,
    cfg: Thought4Config,
    plans: Sequence[Any],
    samples: Sequence[Any],
    states: Mapping[str, Any],
    *,
    resume: bool,
) -> tuple[dict[str, Any], list[Path]]:
    cohort = materialized_cohort_manifest(
        plans,
        states,
        config_fingerprint=cfg.fingerprint,
    )
    cohort_path = (
        write_or_verify_json(output / "cohort_manifest.json", cohort)
        if resume
        else atomic_write_json(output / "cohort_manifest.json", cohort)
    )
    render_rows = [sample.rendered.record.to_dict() for sample in samples]
    render_path = (
        write_or_verify_jsonl(
            output / "paired_render_manifest.jsonl", render_rows
        )
        if resume
        else atomic_write_jsonl(
            output / "paired_render_manifest.jsonl", render_rows
        )
    )
    label_rows = []
    from fastwam_ood_eval.thought4.paired_rendering import array_sha256

    for sample in samples:
        label_rows.append(
            {
                "schema_version": "thought4.phase4.label_manifest.v1",
                "sample_id": sample.plan.identity.sample_id,
                "condition": sample.condition,
                "input_time": f"t={sample.plan.frame_index}",
                "label_time": "per-label; see label_time_by_name",
                "label_time_by_name": {
                    key: (
                        f"t+1...t+{cfg.probe.horizon}"
                        if key.startswith("action_")
                        else "t"
                    )
                    for key in sorted(sample.labels)
                },
                "trajectory_label_source": sample.trajectory_label_source,
                "initial_object_layout_sha256": (
                    sample.initial_object_layout_sha256
                ),
                "initial_object_layout_matches_clean": (
                    sample.initial_object_layout_matches_clean
                ),
                "initial_robot_state_sha256": (
                    sample.initial_robot_state_sha256
                ),
                "initial_robot_state_matches_clean": (
                    sample.initial_robot_state_matches_clean
                ),
                "demonstration_state_alignment": dict(
                    sample.demonstration_state_alignment
                ),
                "future_rgb_read": False,
                "labels": {
                    key: {
                        "shape": list(value.shape),
                        "sha256": array_sha256(value),
                    }
                    for key, value in sorted(sample.labels.items())
                },
                "masks": {
                    key: {
                        "shape": list(value.shape),
                        "sha256": array_sha256(value.astype("uint8")),
                    }
                    for key, value in sorted(sample.masks.items())
                },
            }
        )
    label_path = (
        write_or_verify_jsonl(output / "label_manifest.jsonl", label_rows)
        if resume
        else atomic_write_jsonl(output / "label_manifest.jsonl", label_rows)
    )
    return cohort, [cohort_path, render_path, label_path]


def _write_feature_shards(
    output: Path,
    cfg: Thought4Config,
    examples: Sequence[Any],
    samples: Sequence[Any],
    *,
    resume: bool,
) -> tuple[list[dict[str, Any]], list[Path]]:
    from fastwam_ood_eval.thought4.video_feature_extractor import (
        ExtractedFeature,
        FeatureShardWriter,
    )

    identity_by_id = {
        sample.plan.identity.sample_id: sample.plan.identity for sample in samples
    }
    manifest_rows: list[dict[str, Any]] = []
    artifact_paths: list[Path] = []
    by_source = {
        source: [value for value in examples if value.source == source]
        for source in ("A", "B")
    }
    for source, values in by_source.items():
        for shard_index, start in enumerate(
            range(0, len(values), cfg.runtime.shard_size)
        ):
            chunk = values[start : start + cfg.runtime.shard_size]
            writer = FeatureShardWriter(
                output,
                source=source,
                shard_index=shard_index,
                resume=resume,
            )
            features = [
                ExtractedFeature(
                    identity=identity_by_id[value.sample_id],
                    condition=value.condition,
                    source=value.source,
                    module_path=value.module_path,
                    layer_index=value.layer_index,
                    denoise_step_index=value.denoise_step_index,
                    pooling=value.pooling,
                    tensor=value.feature,
                )
                for value in chunk
            ]
            records = writer.write(features)
            manifest_rows.extend(record.to_dict() for record in records)
            artifact_paths.extend((writer.path, writer.checksum_path))
    manifest_path = (
        write_or_verify_jsonl(
            output / "feature_manifest.jsonl", manifest_rows
        )
        if resume
        else atomic_write_jsonl(
            output / "feature_manifest.jsonl", manifest_rows
        )
    )
    artifact_paths.append(manifest_path)
    return manifest_rows, artifact_paths


def _probe_backward_smoke(examples: Sequence[Any]) -> dict[str, Any]:
    import torch

    from fastwam_ood_eval.thought4.probe_models import build_probe

    source_a = [value for value in examples if value.source == "A"]
    source_b = [value for value in examples if value.source == "B"]
    if not source_a or not source_b:
        raise Phase4ExecutionError("real smoke did not capture both feature sources")
    feature = torch.stack(
        [torch.as_tensor(value.feature).detach().float() for value in source_a[:2]]
    )
    if feature.shape[0] == 1:
        feature = feature.repeat(2, 1)
    target = torch.stack(
        [
            torch.as_tensor(value.labels["eef_object_translation_camera"])
            .detach()
            .float()
            for value in source_a[:2]
        ]
    )
    if target.shape[0] == 1:
        target = target.repeat(2, 1)
    probe = build_probe(
        "linear", input_dim=feature.shape[1], output_dim=target.shape[1]
    )
    prediction = probe(feature.detach())
    loss = (prediction - target).square().mean()
    loss.backward()
    gradients = [
        float(parameter.grad.abs().sum())
        for parameter in probe.parameters()
        if parameter.grad is not None
    ]
    if not gradients or max(gradients) <= 0:
        raise Phase4ExecutionError("probe backward produced no gradient")
    return {
        "loss": float(loss.detach()),
        "probe_gradient_l1": sum(gradients),
        "feature_requires_grad": bool(feature.requires_grad),
        "source_a_count": len(source_a),
        "source_b_count": len(source_b),
        "only_probe_trainable": True,
    }


def run_real_smoke(cfg: Thought4Config, *, resume: bool = False) -> dict[str, Any]:
    _require_confirmation("smoke")
    if cfg.experiment.mode != "smoke":
        raise Phase4ExecutionError("smoke command requires experiment.mode=smoke")
    project_commit = require_project_clean()
    started_at = _utc_now()
    output = _prepare_run(cfg, "smoke", resume=resume)
    try:
        _progress("static_prevalidation")
        audit = static_audit(cfg)
        prevalidation_path = _json_artifact(
            output / "pre_validation_result.json",
            {
                **audit,
                "stage": "smoke",
                "config": config_to_dict(cfg),
            },
            resume=resume,
        )
        all_plans = plan_base_states(cfg.cohort, horizon=cfg.probe.horizon)
        # Smoke validates only the chain, never produces a scientific estimate.
        plans = tuple(all_plans[:2])
        _progress("paired_render_started", base_states=len(plans))
        from fastwam_ood_eval.thought4.real_runtime import (
            extract_probe_examples,
            load_frozen_fastwam,
            release_fastwam,
            render_probe_samples,
        )

        samples, states = render_probe_samples(cfg, plans)
        gc.collect()
        _cohort, common_paths = _write_common_render_artifacts(
            output, cfg, plans, samples, states, resume=resume
        )
        _progress("model_load_started")
        runtime = load_frozen_fastwam(cfg)
        try:
            from fastwam_ood_eval.thought4.feature_hooks import (
                assert_backbone_frozen,
                parameter_state_sha256,
            )

            assert_backbone_frozen(runtime.model)
            sha_before = parameter_state_sha256(runtime.model)
            if sha_before != cfg.backbone.frozen_parameter_sha256:
                raise Phase4ExecutionError(
                    "loaded Fast-WAM parameter SHA differs from frozen Thought3 SHA"
                )
            runtime_audit = runtime_model_audit(runtime.model, cfg)
            examples, inference_rows = extract_probe_examples(
                cfg, runtime, samples
            )
            backward = _probe_backward_smoke(examples)
            from fastwam_ood_eval.thought4.intervention_runtime import (
                run_identity_replacement_smoke,
            )

            clean_sample = next(
                (
                    sample
                    for sample in samples
                    if sample.condition == "clean"
                ),
                None,
            )
            if clean_sample is None:
                raise Phase4ExecutionError(
                    "real smoke has no Clean sample for replacement validation"
                )
            identity_replacement = run_identity_replacement_smoke(
                cfg, runtime, clean_sample
            )
            sha_after = parameter_state_sha256(runtime.model)
            if sha_before != sha_after:
                raise Phase4ExecutionError("Fast-WAM parameter SHA changed in smoke")
        finally:
            release_fastwam(runtime)
        feature_rows, feature_paths = _write_feature_shards(
            output, cfg, examples, samples, resume=resume
        )
        smoke_payload = {
            "schema_version": "thought4.phase4.real_smoke.v1",
            "status": "passed",
            "scientific_result": False,
            "project_commit": project_commit,
            "config_fingerprint": cfg.fingerprint,
            "base_state_count": len(plans),
            "condition_count": len(samples),
            "runtime_model_audit": runtime_audit,
            "model_load_latency_s": runtime.load_latency_s,
            "backbone_parameter_sha256_before": sha_before,
            "backbone_parameter_sha256_after": sha_after,
            "feature_record_count": len(feature_rows),
            "probe_backward": backward,
            "identity_replacement": identity_replacement,
            "inference_rows": inference_rows,
            "future_rgb_read": False,
            "success_outcome_read": False,
            "formal_unlocked": True,
        }
        smoke_payload["result_sha256"] = sha256_canonical(smoke_payload)
        smoke_path = _json_artifact(
            output / "smoke_result.json", smoke_payload, resume=resume
        )
        _finish_status(
            output,
            stage="smoke",
            status="complete",
            started_at=started_at,
        )
        _progress("smoke_complete", result=str(smoke_path))
        return {
            "status": "complete",
            "result": str(smoke_path),
            "formal_unlocked": True,
            "scientific_result": False,
        }
    except BaseException as exc:
        _finish_status(
            output,
            stage="smoke",
            status="error",
            started_at=started_at,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def _write_probe_bundle(
    path: Path,
    examples: Sequence[Any],
    *,
    resume: bool,
) -> tuple[Path, Path]:
    """Persist tensor/label inputs for checksum-valid read-only reuse."""

    import torch

    payload = [
        {
            "sample_id": value.sample_id,
            "episode_id": value.episode_id,
            "split": value.split,
            "condition": value.condition,
            "source": value.source,
            "module_path": value.module_path,
            "layer_index": value.layer_index,
            "denoise_step_index": value.denoise_step_index,
            "pooling": value.pooling,
            "feature": torch.as_tensor(value.feature).detach().cpu(),
            "labels": {
                key: torch.as_tensor(label).detach().cpu()
                for key, label in value.labels.items()
            },
            "masks": {
                key: torch.as_tensor(mask).detach().cpu()
                for key, mask in value.masks.items()
            },
        }
        for value in examples
    ]
    from fastwam_ood_eval.thought4.schemas import sha256_file
    from fastwam_ood_eval.thought4.video_feature_extractor import tensor_sha256

    path.parent.mkdir(parents=True, exist_ok=True)
    checksum_path = path.with_suffix(".sha256")
    if path.exists() or checksum_path.exists():
        if not resume or not path.is_file() or not checksum_path.is_file():
            raise Thought4ArtifactError(
                f"refusing incomplete/overwriting probe bundle: {path}"
            )
        expected_sha = checksum_path.read_text(encoding="utf-8").strip()
        if len(expected_sha) != 64 or sha256_file(path) != expected_sha:
            raise Thought4ArtifactError("probe bundle checksum mismatch on resume")
        try:
            existing = torch.load(
                path, map_location="cpu", weights_only=True
            )
        except Exception as exc:
            raise Thought4ArtifactError(
                "probe bundle cannot be loaded safely on resume"
            ) from exc
        if not isinstance(existing, list) or len(existing) != len(payload):
            raise Thought4ArtifactError(
                "probe bundle row count differs on resume"
            )
        for index, (first, second) in enumerate(zip(existing, payload)):
            metadata_keys = (
                "sample_id",
                "episode_id",
                "split",
                "condition",
                "source",
                "module_path",
                "layer_index",
                "denoise_step_index",
                "pooling",
            )
            if any(first.get(key) != second.get(key) for key in metadata_keys):
                raise Thought4ArtifactError(
                    f"probe bundle metadata differs at row {index}"
                )
            tensor_groups = (
                ({"feature": first["feature"]}, {"feature": second["feature"]}),
                (first["labels"], second["labels"]),
                (first["masks"], second["masks"]),
            )
            for observed_group, expected_group in tensor_groups:
                if set(observed_group) != set(expected_group):
                    raise Thought4ArtifactError(
                        f"probe bundle tensor keys differ at row {index}"
                    )
                for key in observed_group:
                    if tensor_sha256(observed_group[key]) != tensor_sha256(
                        expected_group[key]
                    ):
                        raise Thought4ArtifactError(
                            f"probe bundle tensor differs at row {index}:{key}"
                        )
        return path, checksum_path
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    checksum = sha256_file(path)
    atomic_write_text(checksum_path, checksum + "\n")
    return path, checksum_path


def run_formal_diagnosis(
    cfg: Thought4Config, *, resume: bool = False
) -> dict[str, Any]:
    _require_confirmation("formal")
    if cfg.experiment.mode != "formal":
        raise Phase4ExecutionError("formal command requires experiment.mode=formal")
    project_commit = require_project_clean()
    smoke_gate = _verify_formal_smoke_gate(
        cfg, expected_project_commit=project_commit
    )
    started_at = _utc_now()
    output = _prepare_run(cfg, "formal", resume=resume)
    try:
        audit = static_audit(cfg)
        prevalidation_path = _json_artifact(
            output / "pre_validation_result.json",
            {
                **audit,
                "stage": "formal",
                "config": config_to_dict(cfg),
                "smoke_gate": smoke_gate,
            },
            resume=resume,
        )
        plans = plan_base_states(cfg.cohort, horizon=cfg.probe.horizon)
        planned_path = _json_artifact(
            output / "planned_cohort_manifest.json",
            planned_cohort_manifest(
                plans, config_fingerprint=cfg.fingerprint
            ),
            resume=resume,
        )
        from fastwam_ood_eval.thought4.real_runtime import (
            extract_probe_examples,
            load_frozen_fastwam,
            release_fastwam,
            render_probe_samples,
        )

        _progress("formal_paired_render_started", base_states=len(plans))
        samples, states = render_probe_samples(cfg, plans)
        gc.collect()
        cohort, common_paths = _write_common_render_artifacts(
            output, cfg, plans, samples, states, resume=resume
        )
        _progress("formal_model_load_started")
        runtime = load_frozen_fastwam(cfg)
        try:
            from fastwam_ood_eval.thought4.feature_hooks import (
                assert_backbone_frozen,
                parameter_state_sha256,
            )
            from fastwam_ood_eval.thought4.intervention_runtime import (
                run_geometry_subspace_intervention,
            )
            from fastwam_ood_eval.thought4.pipeline import (
                run_probe_panel,
                select_intervention_feature,
            )

            assert_backbone_frozen(runtime.model)
            sha_before = parameter_state_sha256(runtime.model)
            if sha_before != cfg.backbone.frozen_parameter_sha256:
                raise Phase4ExecutionError(
                    "loaded Fast-WAM parameter SHA differs from frozen Thought3 SHA"
                )
            runtime_audit = runtime_model_audit(runtime.model, cfg)
            examples, inference_rows = extract_probe_examples(
                cfg, runtime, samples
            )
            feature_rows, feature_paths = _write_feature_shards(
                output, cfg, examples, samples, resume=resume
            )
            bundle_path, bundle_checksum_path = _write_probe_bundle(
                output / "probe_inputs.pt", examples, resume=resume
            )
            panel_kwargs = {
                "probe_models": cfg.probe.models,
                "seeds": cfg.probe.seeds,
                "hidden_dim": cfg.probe.mlp_hidden_dim,
                "learning_rate": cfg.probe.learning_rate,
                "weight_decay": cfg.probe.weight_decay,
                "max_epochs": cfg.probe.max_epochs,
                "patience": cfg.probe.patience,
                "batch_size": cfg.probe.batch_size,
                "bootstrap_replicates": cfg.probe.bootstrap_replicates,
                "bootstrap_seed": cfg.probe.bootstrap_seed,
                "device": "cpu",
            }
            _progress("video_probe_panel_started")
            video_panel = run_probe_panel(
                examples, source="A", **panel_kwargs
            )
            _progress("action_probe_panel_started")
            action_panel = run_probe_panel(
                examples, source="B", **panel_kwargs
            )
            selection = select_intervention_feature(
                video_panel.result,
                target=cfg.intervention.target_label,
                seed=cfg.probe.seeds[0],
            )
            probe_key = (
                selection["feature_key"],
                selection["target"],
                cfg.probe.seeds[0],
            )
            if probe_key not in video_panel.linear_models:
                raise Phase4ExecutionError(
                    "selected linear probe is missing from in-memory registry"
                )
            _progress(
                "geometry_subspace_intervention_started",
                module_path=selection["module_path"],
            )
            intervention = run_geometry_subspace_intervention(
                cfg,
                runtime,
                samples,
                selection=selection,
                linear_probe=video_panel.linear_models[probe_key],
                probe_examples=examples,
            )
            sha_after = parameter_state_sha256(runtime.model)
        finally:
            release_fastwam(runtime)
        from fastwam_ood_eval.thought4.decision import (
            derive_diagnostic_evidence,
            select_method,
        )
        from fastwam_ood_eval.thought4.report import (
            build_artifact_manifest,
            build_layer_summary,
            diagnostic_report_markdown,
            execution_integrity,
        )

        video_path = _json_artifact(
            output / "video_probe_results.json",
            video_panel.result,
            resume=resume,
        )
        action_path = _json_artifact(
            output / "action_probe_results.json",
            action_panel.result,
            resume=resume,
        )
        intervention_path = _json_artifact(
            output / "intervention_results.json",
            intervention,
            resume=resume,
        )
        layer_summary = build_layer_summary(
            video_panel.result, action_panel.result
        )
        layer_path = _json_artifact(
            output / "layer_summary.json", layer_summary, resume=resume
        )
        evidence, evidence_payload = derive_diagnostic_evidence(
            video_panel.result,
            action_panel.result,
            intervention,
        )
        evidence_path = _json_artifact(
            output / "diagnostic_evidence.json",
            evidence_payload,
            resume=resume,
        )
        method = select_method(evidence)
        method_path = _json_artifact(
            output / "method_selection.json", method, resume=resume
        )
        integrity = execution_integrity(
            config_fingerprint=cfg.fingerprint,
            backbone_sha_before=sha_before,
            backbone_sha_after=sha_after,
            checkpoint_sha256=cfg.backbone.checkpoint_sha256,
            cohort_sha256=cohort["manifest_sha256"],
            future_rgb_read=False,
            success_outcome_read=False,
        )
        integrity["runtime_model_audit"] = runtime_audit
        integrity["inference_rows"] = inference_rows
        integrity["smoke_gate"] = smoke_gate
        integrity_path = _json_artifact(
            output / "execution_integrity.json", integrity, resume=resume
        )
        report_path = _text_artifact(
            output / "report.md",
            diagnostic_report_markdown(
                method_selection=method,
                evidence=evidence_payload,
                layer_summary=layer_summary,
                intervention=intervention,
            ),
            resume=resume,
        )
        manifest_inputs = [
            prevalidation_path,
            planned_path,
            *common_paths,
            *feature_paths,
            bundle_path,
            bundle_checksum_path,
            video_path,
            action_path,
            intervention_path,
            layer_path,
            evidence_path,
            method_path,
            integrity_path,
            report_path,
        ]
        artifact_manifest = build_artifact_manifest(output, manifest_inputs)
        artifact_path = _json_artifact(
            output / "artifact_manifest.json",
            artifact_manifest,
            resume=resume,
        )
        _finish_status(
            output,
            stage="formal",
            status="complete",
            started_at=started_at,
        )
        _progress(
            "formal_complete",
            classification=method["classification"],
            recommendation=method["recommendation"],
        )
        return {
            "status": "complete",
            "method_selection": str(method_path),
            "classification": method["classification"],
            "recommendation": method["recommendation"],
            "report": str(report_path),
            "artifact_manifest": str(artifact_path),
        }
    except BaseException as exc:
        _finish_status(
            output,
            stage="formal",
            status="error",
            started_at=started_at,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
