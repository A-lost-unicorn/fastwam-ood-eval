"""Guarded multi-GPU pilot/formal orchestration for Phase 5.

The expensive stages use process-per-track model replicas.  Rendering is
materialized once before workers start, each worker owns exactly one visible
GPU, and the CPU finalizer refuses to classify a partially collected panel.
"""

from __future__ import annotations

import gc
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from fastwam_ood_eval.thought5.artifacts import (
    build_artifact_manifest,
    execution_integrity,
    validate_artifact_manifest,
    write_report_transition,
    write_status_transition,
)
from fastwam_ood_eval.thought5.checkpointing import (
    frozen_parameter_sha256,
    geoeq_state_dict,
    load_geoeq_checkpoint,
    restore_geoeq_state,
    save_geoeq_checkpoint,
    tensor_state_sha256,
)
from fastwam_ood_eval.thought5.config import Thought5Config
from fastwam_ood_eval.thought5.geo_equiv_model import GeoEqAttachment
from fastwam_ood_eval.thought5.geo_targets import shuffled_target_indices
from fastwam_ood_eval.thought5.paired_geometry_data import (
    assert_formal_exclusion,
    cohort_manifest,
    load_condition_catalog,
)
from fastwam_ood_eval.thought5.schemas import (
    clean_project_commit,
    file_sha256,
    object_sha256,
    write_json_once,
)
from fastwam_ood_eval.thought5.trainer import (
    gradient_report,
    matched_optimizer,
    paired_training_loss,
    weights_for_variant,
)


class Phase5PanelError(RuntimeError):
    pass


def _progress(stage: str, **values: Any) -> None:
    from datetime import datetime, timezone

    print(
        json.dumps(
            {
                "phase": "5",
                "stage": stage,
                "time": datetime.now(timezone.utc).isoformat(),
                **values,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def parallel_schedule(
    stage: str, physical_gpu_ids: Sequence[str]
) -> tuple[tuple[tuple[str, str], ...], ...]:
    """Return immutable process waves as ``(variant, physical_gpu)`` pairs."""

    ids = tuple(str(value) for value in physical_gpu_ids)
    if stage == "pilot":
        if len(ids) not in {2, 3}:
            raise Phase5PanelError("pilot requires two or three physical GPUs")
        if len(ids) == 2:
            return (
                (("B1", ids[0]), ("G3", ids[1])),
                (("G4", ids[0]),),
            )
        return ((("B1", ids[0]), ("G3", ids[1]), ("G4", ids[2])),)
    if stage == "formal":
        if len(ids) not in {3, 4}:
            raise Phase5PanelError("formal requires three or four physical GPUs")
        if len(ids) == 3:
            return (
                (("B1", ids[0]), ("G1", ids[1]), ("G2", ids[2])),
                (("G3", ids[0]), ("B0", ids[1])),
            )
        return (
            (
                ("B1", ids[0]),
                ("G1", ids[1]),
                ("G2", ids[2]),
                ("G3", ids[3]),
            ),
            (("B0", ids[0]),),
        )
    raise Phase5PanelError(f"unsupported panel stage: {stage}")


def _visible_physical_ids(stage: str) -> tuple[str, ...]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values or len(set(values)) != len(values) or any(not value.isdigit() for value in values):
        raise Phase5PanelError("CUDA_VISIBLE_DEVICES must contain distinct physical GPU IDs")
    parallel_schedule(stage, values)
    return values


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _planned_state(row: Mapping[str, Any]) -> Any:
    from fastwam_ood_eval.thought4.cohort import PlannedBaseState

    split = "test" if row["split"] == "formal" else str(row["split"])
    frame = int(row["frame_index"])
    return PlannedBaseState(
        task_id=str(row["task_index"]),
        task_index=int(row["task_index"]),
        episode_id=f"episode_{int(row['episode_index']):06d}",
        episode_index=int(row["episode_index"]),
        task_local_episode_index=int(row["task_local_episode_index"]),
        frame_index=frame,
        split=split,
        timestamp=frame / 20.0,
        replay_action_count=frame,
    )


def _task_render_config(
    cfg: Thought5Config,
    *,
    task_index: int,
    task_name: str,
    catalog: Mapping[str, Mapping[str, Sequence[Any]]],
) -> Any:
    from fastwam_ood_eval.thought5.real_runtime import _thought4_smoke_config

    base = _thought4_smoke_config(cfg)
    condition_ids = tuple(
        sorted(
            (
                condition,
                tuple(int(value.classification_id) for value in catalog[task_name][condition]),
            )
            for condition in cfg.cohort.conditions
        )
    )
    return replace(
        base,
        experiment=replace(
            base.experiment,
            name=f"{cfg.experiment.name}_render_task_{task_index}",
            output_dir=(
                cfg.experiment.output_dir
                / "runtime"
                / "paired_render"
                / f"task_{task_index}"
            ),
            seed=cfg.experiment.seed,
        ),
        cohort=replace(
            base.cohort,
            task_ids=(task_index,),
            condition_task_ids=condition_ids,
            target_object_name=cfg.cohort.target_object_by_task[task_index],
            split_seed=cfg.cohort.split_seed,
            conditions=cfg.cohort.conditions,
        ),
        probe=replace(base.probe, horizon=cfg.cohort.horizon),
    )


def prepare_render_cache(cfg: Thought5Config, *, resume: bool) -> dict[str, Any]:
    """Materialize exact-state rendering once, without loading Fast-WAM."""

    root = cfg.experiment.output_dir / "runtime" / "render_cache"
    result_path = root / "render_cache_manifest.json"
    if result_path.is_file():
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        if prior.get("status") == "complete":
            if prior.get("config_fingerprint") != cfg.fingerprint:
                raise Phase5PanelError("render cache belongs to another config")
            for row in prior["shards"]:
                path = root / row["relative_path"]
                if not path.is_file() or file_sha256(path) != row["sha256"]:
                    raise Phase5PanelError("render cache shard checksum mismatch")
            return prior
        if not resume:
            raise Phase5PanelError("partial render cache exists; pass --resume after audit")
    manifest = cohort_manifest(cfg.cohort)
    assert_formal_exclusion(manifest)
    write_json_once(
        cfg.experiment.output_dir / "cohort_manifest.json",
        manifest,
        allow_identical=True,
    )
    rows = manifest["rows"]
    catalog = load_condition_catalog(
        cfg.cohort.classification_path,
        sorted({str(row["task_name"]) for row in rows}),
    )
    shards: list[dict[str, Any]] = []
    for task_index in sorted({int(row["task_index"]) for row in rows}):
        task_rows = [row for row in rows if int(row["task_index"]) == task_index]
        task_name = str(task_rows[0]["task_name"])
        plans = tuple(_planned_state(row) for row in task_rows)
        t4_cfg = _task_render_config(
            cfg,
            task_index=task_index,
            task_name=task_name,
            catalog=catalog,
        )
        from fastwam_ood_eval.thought4.real_runtime import render_probe_samples

        _progress("panel_render_task_started", task=task_index, base_states=len(plans))
        samples, _states = render_probe_samples(t4_cfg, plans)
        expected = len(plans) * len(cfg.cohort.conditions)
        if len(samples) != expected:
            raise Phase5PanelError(
                f"task {task_index} rendered {len(samples)} samples, expected {expected}"
            )
        shard = root / f"task_{task_index:02d}.pkl"
        if shard.exists():
            if not resume:
                raise Phase5PanelError(f"render shard already exists: {shard}")
        else:
            from fastwam_ood_eval.thought5.future_runtime import (
                render_future_target_sequences,
            )

            _progress(
                "panel_future_target_render_started",
                task=task_index,
                condition_samples=len(samples),
            )
            future_targets = render_future_target_sequences(cfg, samples)
            _atomic_pickle(
                shard,
                {
                    "schema_version": "thought5.phase5.render_shard.v1",
                    "samples": samples,
                    "future_targets": future_targets,
                },
            )
        shards.append(
            {
                "task_index": task_index,
                "base_states": len(plans),
                "samples": len(samples),
                "relative_path": shard.name,
                "bytes": shard.stat().st_size,
                "sha256": file_sha256(shard),
            }
        )
        del samples
        gc.collect()
    result = {
        "schema_version": "thought5.phase5.render_cache.v1",
        "status": "complete",
        "config_fingerprint": cfg.fingerprint,
        "cohort_manifest_sha256": manifest["manifest_sha256"],
        "conditions": list(cfg.cohort.conditions),
        "shards": shards,
        "future_rgb_read": False,
        "success_outcome_read": False,
    }
    result["render_cache_sha256"] = object_sha256(result)
    write_status_transition(result_path, result)
    return result


def _load_rendered_samples(cfg: Thought5Config) -> list[Any]:
    root = cfg.experiment.output_dir / "runtime" / "render_cache"
    manifest = json.loads((root / "render_cache_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("config_fingerprint") != cfg.fingerprint:
        raise Phase5PanelError("render cache is incomplete or belongs to another config")
    samples: list[Any] = []
    for row in manifest["shards"]:
        path = root / row["relative_path"]
        if file_sha256(path) != row["sha256"]:
            raise Phase5PanelError("render cache changed before worker load")
        with path.open("rb") as handle:
            values = pickle.load(handle)
        if not isinstance(values, Mapping) or values.get("schema_version") != (
            "thought5.phase5.render_shard.v1"
        ):
            raise Phase5PanelError("render shard schema changed")
        samples.extend(values["samples"])
    return samples


def _load_future_targets(cfg: Thought5Config) -> dict[str, Any]:
    root = cfg.experiment.output_dir / "runtime" / "render_cache"
    manifest = json.loads(
        (root / "render_cache_manifest.json").read_text(encoding="utf-8")
    )
    targets: dict[str, Any] = {}
    for row in manifest["shards"]:
        path = root / row["relative_path"]
        if file_sha256(path) != row["sha256"]:
            raise Phase5PanelError("render cache changed before target load")
        with path.open("rb") as handle:
            values = pickle.load(handle)
        for key, value in values["future_targets"].items():
            if key in targets:
                raise Phase5PanelError("duplicate future target cache key")
            targets[key] = value
    return targets


def _map_value(value: Any, *, device: str) -> Any:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device)
    if isinstance(value, Mapping):
        return {key: _map_value(item, device=device) for key, item in value.items()}
    return value


def _move_batch(batch: Any, *, device: str) -> Any:
    values = {
        field.name: _map_value(getattr(batch, field.name), device=device)
        for field in fields(batch)
    }
    return type(batch)(**values)


def _prepare_batches(
    cfg: Thought5Config,
    runtime: Any,
    samples: Sequence[Any],
    *,
    split: str,
) -> list[Any]:
    from fastwam_ood_eval.thought5.real_runtime import _build_smoke_batch

    source_split = "test" if split == "formal" else split
    groups: dict[str, list[Any]] = {}
    for sample in samples:
        if sample.plan.identity.split == source_split:
            groups.setdefault(sample.plan.identity.sample_id, []).append(sample)
    if not groups:
        raise Phase5PanelError(f"render cache contains no {split} samples")
    batches: list[Any] = []
    # _build_smoke_batch only uses cfg/runtime/sample payloads; its t4 argument
    # is retained for compatibility with the smoke implementation.
    for index, (_pair_id, values) in enumerate(sorted(groups.items())):
        batch = _build_smoke_batch(
            cfg,
            None,
            runtime,
            values,
            noise_seed_offset=index * 1009,
        )
        batches.append(_move_batch(batch, device="cpu"))
        del batch
    permutation = shuffled_target_indices(
        [str(batch.pair_ids[0]) for batch in batches],
        seed=cfg.training.shuffled_geometry_seed,
    )
    for index, donor_index in enumerate(permutation):
        batches[index].shuffled_clean_geometry_target = batches[
            donor_index
        ].clean_geometry_target
        batches[index].shuffled_camera_geometry_target = batches[
            donor_index
        ].camera_geometry_target
        batches[index].shuffled_pair_ids = batches[donor_index].pair_ids
    return batches


def _mean_development_loss(
    cfg: Thought5Config,
    attachment: GeoEqAttachment,
    batches: Sequence[Any],
    *,
    variant: str,
) -> dict[str, float]:
    import torch

    weights = weights_for_variant(cfg.training.lambda_by_variant[variant])
    totals: dict[str, list[float]] = {}
    attachment.model.eval()
    attachment.geo_projector.eval()
    attachment.ray_pose_encoder.eval()
    with torch.no_grad():
        for cpu_batch in batches:
            batch = _move_batch(cpu_batch, device=cfg.runtime.device)
            loss, components = paired_training_loss(
                attachment,
                batch,
                variant=variant,
                weights=weights,
            )
            totals.setdefault("selection_objective", []).append(float(loss.cpu()))
            for key, value in components.items():
                totals.setdefault(key, []).append(float(value.detach().cpu()))
            del batch, loss, components
    attachment.model.train(False)
    attachment.geo_projector.train()
    attachment.ray_pose_encoder.train()
    return {key: sum(values) / len(values) for key, values in totals.items()}


def _candidate_steps(cfg: Thought5Config) -> tuple[int, ...]:
    if cfg.experiment.stage == "formal":
        return (cfg.training.max_steps,)
    stride = max(1, cfg.training.max_steps // 4)
    return tuple(
        sorted(set(range(stride, cfg.training.max_steps + 1, stride)) | {cfg.training.max_steps})
    )


def run_track_worker(cfg: Thought5Config, *, variant: str, resume: bool) -> dict[str, Any]:
    """Train one independent matched track and freeze its development choice."""

    import torch
    from fastwam_ood_eval.thought4.real_runtime import load_frozen_fastwam, release_fastwam
    from fastwam_ood_eval.thought5.real_runtime import _require_single_visible_cuda, _thought4_smoke_config

    if variant not in cfg.training.variants:
        raise Phase5PanelError(f"variant {variant} is outside the frozen stage")
    project_commit = clean_project_commit()
    _require_single_visible_cuda(cfg)
    output = cfg.experiment.output_dir / "tracks" / variant.lower()
    status_path = output / "run_status.json"
    if status_path.is_file():
        prior = json.loads(status_path.read_text(encoding="utf-8"))
        if prior.get("status") == "complete":
            return json.loads((output / "track_result.json").read_text(encoding="utf-8"))
        if prior.get("project_commit") not in {None, project_commit}:
            raise Phase5PanelError(
                f"partial {variant} track belongs to another project commit"
            )
        if not resume:
            raise Phase5PanelError(f"partial {variant} track exists; pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    write_status_transition(
        status_path,
        {
            "schema_version": "thought5.phase5.track_status.v1",
            "status": "running",
            "variant": variant,
            "config_fingerprint": cfg.fingerprint,
            "project_commit": project_commit,
        },
    )
    runtime = None
    attachment = None
    started = time.perf_counter()
    try:
        samples = _load_rendered_samples(cfg)
        t4_cfg = _thought4_smoke_config(cfg)
        torch.manual_seed(cfg.experiment.seed)
        torch.cuda.manual_seed_all(cfg.experiment.seed)
        torch.cuda.reset_peak_memory_stats(cfg.runtime.device)
        runtime = load_frozen_fastwam(t4_cfg)
        release_sha = frozen_parameter_sha256(runtime.model.named_parameters())
        if release_sha != cfg.backbone.frozen_parameter_sha256:
            raise Phase5PanelError("release model SHA mismatch before track attachment")
        if variant == "B0":
            result = {
                "schema_version": "thought5.phase5.track_result.v1",
                "status": "complete",
                "variant": variant,
                "training_steps": 0,
                "checkpoint": str(cfg.backbone.checkpoint_path),
                "checkpoint_kind": "official_release_read_only",
                "release_parameter_sha256": release_sha,
                "config_fingerprint": cfg.fingerprint,
                "project_commit": project_commit,
                "scientific_result": False,
                "elapsed_s": time.perf_counter() - started,
            }
        else:
            torch.manual_seed(cfg.experiment.seed + 5001)
            torch.cuda.manual_seed_all(cfg.experiment.seed + 5001)
            attachment = GeoEqAttachment(
                runtime.model,
                lora_targets=cfg.method.lora_targets,
                lora_rank=cfg.method.lora_rank,
                lora_alpha=cfg.method.lora_alpha,
                lora_dropout=cfg.method.lora_dropout,
                projector_hidden_dim=cfg.method.geo_projector_hidden_dim,
                ray_pose_hidden_dim=cfg.method.ray_pose_hidden_dim,
            )
            frozen_attached_before = frozen_parameter_sha256(
                runtime.model.named_parameters()
            )
            initial_state = {
                name: value.detach().cpu().clone()
                for name, value in geoeq_state_dict(attachment).items()
            }
            initial_state_sha = tensor_state_sha256(initial_state)
            train_batches = _prepare_batches(cfg, runtime, samples, split="train")
            development_batches = _prepare_batches(
                cfg, runtime, samples, split="development"
            )
            restore_geoeq_state(attachment, initial_state)
            optimizer = matched_optimizer(
                attachment,
                learning_rate=cfg.training.learning_rate,
                weight_decay=cfg.training.weight_decay,
            )
            weights = weights_for_variant(cfg.training.lambda_by_variant[variant])
            candidates = set(_candidate_steps(cfg))
            candidate_states: dict[int, dict[str, Any]] = {}
            development_rows: list[dict[str, Any]] = []
            training_rows: list[dict[str, Any]] = []
            for step in range(1, cfg.training.max_steps + 1):
                optimizer.zero_grad(set_to_none=True)
                aggregate_components: dict[str, list[float]] = {}
                for offset in range(cfg.training.gradient_accumulation):
                    index = (
                        (step - 1) * cfg.training.gradient_accumulation + offset
                    ) % len(train_batches)
                    batch = _move_batch(train_batches[index], device=cfg.runtime.device)
                    loss, components = paired_training_loss(
                        attachment,
                        batch,
                        variant=variant,
                        weights=weights,
                    )
                    if not bool(torch.isfinite(loss)):
                        raise Phase5PanelError(f"{variant} produced non-finite loss")
                    (loss / cfg.training.gradient_accumulation).backward()
                    for key, value in components.items():
                        aggregate_components.setdefault(key, []).append(
                            float(value.detach().cpu())
                        )
                    del batch, loss, components
                gradients = gradient_report(attachment)
                if not all(value["finite"] for value in gradients.values()):
                    raise Phase5PanelError(f"{variant} produced non-finite gradients")
                optimizer.step()
                training_rows.append(
                    {
                        "step": step,
                        "components": {
                            key: sum(values) / len(values)
                            for key, values in aggregate_components.items()
                        },
                        "gradients": gradients if step <= 2 else None,
                    }
                )
                if step in candidates:
                    development = _mean_development_loss(
                        cfg,
                        attachment,
                        development_batches,
                        variant=variant,
                    )
                    development_rows.append({"step": step, **development})
                    candidate_states[step] = {
                        name: value.detach().cpu().clone()
                        for name, value in geoeq_state_dict(attachment).items()
                    }
                    _progress(
                        "panel_track_checkpoint",
                        variant=variant,
                        step=step,
                        development_objective=development["selection_objective"],
                    )
            if cfg.experiment.stage == "formal":
                selected_step = cfg.training.max_steps
            else:
                selected_step = min(
                    development_rows,
                    key=lambda row: (row["selection_objective"], -row["step"]),
                )["step"]
            selected_state = candidate_states[int(selected_step)]
            restore_geoeq_state(attachment, selected_state)
            frozen_after = frozen_parameter_sha256(runtime.model.named_parameters())
            if frozen_after != frozen_attached_before:
                raise Phase5PanelError(
                    f"{variant} changed a frozen Fast-WAM parameter"
                )
            checkpoint = save_geoeq_checkpoint(
                output / "checkpoints" / f"step_{int(selected_step):08d}",
                attachment=attachment,
                variant=variant,
                global_step=int(selected_step),
                config_fingerprint=cfg.fingerprint,
                cohort_fingerprint=json.loads(
                    (cfg.experiment.output_dir / "cohort_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )["manifest_sha256"],
                backbone_checkpoint_sha256=cfg.backbone.checkpoint_sha256,
                frozen_before_sha256=frozen_attached_before,
                frozen_after_sha256=frozen_after,
            )
            result = {
                "schema_version": "thought5.phase5.track_result.v1",
                "status": "complete",
                "variant": variant,
                "config_fingerprint": cfg.fingerprint,
                "project_commit": project_commit,
                "initial_state_sha256": initial_state_sha,
                "selected_step": int(selected_step),
                "checkpoint": str(checkpoint),
                "checkpoint_kind": "geoeq_adapter_only",
                "training_rows": training_rows,
                "development_rows": development_rows,
                "parameter_manifest": attachment.parameter_manifest(),
                "release_parameter_sha256": release_sha,
                "frozen_attached_sha256_before": frozen_attached_before,
                "frozen_attached_sha256": frozen_after,
                "peak_memory_mib": torch.cuda.max_memory_allocated(
                    cfg.runtime.device
                )
                / 2**20,
                "elapsed_s": time.perf_counter() - started,
                # A training track is never a scientific result on its own.
                "scientific_result": False,
            }
        from fastwam_ood_eval.thought5.representation_runtime import (
            collect_representation_bundle,
        )

        representation_bundle = collect_representation_bundle(
            cfg,
            runtime,
            attachment,
            variant=variant,
            samples=samples,
            output_path=output / "representation_bundle.pkl",
        )
        result["representation_bundle"] = representation_bundle
        if variant in {"B1", "G3", "G4"}:
            from fastwam_ood_eval.thought5.future_runtime import (
                collect_future_bundle,
            )

            future_bundle = collect_future_bundle(
                cfg,
                runtime,
                attachment,
                variant=variant,
                samples=samples,
                future_targets=_load_future_targets(cfg),
                output_path=output / "future_bundle.pkl",
            )
            result["future_bundle"] = future_bundle
        write_status_transition(output / "track_result.json", result)
        write_status_transition(
            status_path,
            {
                "schema_version": "thought5.phase5.track_status.v1",
                "status": "complete",
                "variant": variant,
                "config_fingerprint": cfg.fingerprint,
                "project_commit": project_commit,
                "result": str(output / "track_result.json"),
            },
        )
        return result
    except BaseException as exc:
        write_status_transition(
            status_path,
            {
                "schema_version": "thought5.phase5.track_status.v1",
                "status": "error",
                "variant": variant,
                "config_fingerprint": cfg.fingerprint,
                "project_commit": project_commit,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    finally:
        if attachment is not None:
            attachment.close()
        if runtime is not None:
            release_fastwam(runtime)
        gc.collect()


def _load_variant_runtime(
    cfg: Thought5Config, *, variant: str
) -> tuple[Any, GeoEqAttachment | None, dict[str, Any]]:
    """Load the official release plus one already-frozen track checkpoint."""

    import torch
    from fastwam_ood_eval.thought4.real_runtime import load_frozen_fastwam
    from fastwam_ood_eval.thought5.real_runtime import _thought4_smoke_config

    runtime = load_frozen_fastwam(_thought4_smoke_config(cfg))
    release_sha = frozen_parameter_sha256(runtime.model.named_parameters())
    if release_sha != cfg.backbone.frozen_parameter_sha256:
        raise Phase5PanelError("release model SHA mismatch in evaluation worker")
    if variant == "B0":
        return runtime, None, {"release_parameter_sha256": release_sha}
    track_path = (
        cfg.experiment.output_dir
        / "tracks"
        / variant.lower()
        / "track_result.json"
    )
    if not track_path.is_file():
        raise Phase5PanelError(f"{variant} track result is absent")
    track = json.loads(track_path.read_text(encoding="utf-8"))
    if (
        track.get("status") != "complete"
        or track.get("variant") != variant
        or track.get("config_fingerprint") != cfg.fingerprint
    ):
        raise Phase5PanelError(f"{variant} track result provenance differs")
    torch.manual_seed(cfg.experiment.seed + 5001)
    torch.cuda.manual_seed_all(cfg.experiment.seed + 5001)
    attachment = GeoEqAttachment(
        runtime.model,
        lora_targets=cfg.method.lora_targets,
        lora_rank=cfg.method.lora_rank,
        lora_alpha=cfg.method.lora_alpha,
        lora_dropout=cfg.method.lora_dropout,
        projector_hidden_dim=cfg.method.geo_projector_hidden_dim,
        ray_pose_hidden_dim=cfg.method.ray_pose_hidden_dim,
    )
    manifest = load_geoeq_checkpoint(
        track["checkpoint"],
        attachment=attachment,
        expected={
            "variant": variant,
            "config_fingerprint": cfg.fingerprint,
            "backbone_checkpoint_sha256": cfg.backbone.checkpoint_sha256,
        },
    )
    runtime.model.eval()
    attachment.geo_projector.eval()
    attachment.ray_pose_encoder.eval()
    return runtime, attachment, {
        "release_parameter_sha256": release_sha,
        "track_result_sha256": file_sha256(track_path),
        "geoeq_checkpoint": str(track["checkpoint"]),
        "geoeq_checkpoint_manifest_sha256": manifest["manifest_sha256"],
    }


def _freeze_evaluation_graph(
    runtime: Any, attachment: GeoEqAttachment | None
) -> dict[str, str]:
    """Freeze every non-Adapter tensor while preserving input gradients."""

    runtime.model.requires_grad_(False)
    if attachment is not None:
        attachment.geo_projector.requires_grad_(False)
        attachment.ray_pose_encoder.requires_grad_(False)
    return {
        "model": tensor_state_sha256(runtime.model.state_dict()),
        "geoeq": (
            tensor_state_sha256(geoeq_state_dict(attachment))
            if attachment is not None
            else "not_applicable_B0"
        ),
    }


def _assert_evaluation_graph_unchanged(
    runtime: Any,
    attachment: GeoEqAttachment | None,
    before: Mapping[str, str],
) -> None:
    after = _freeze_evaluation_graph(runtime, attachment)
    if dict(before) != after:
        raise Phase5PanelError("evaluation changed the frozen policy graph")


def _validated_calibration(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Phase5PanelError("B1 future-utility calibration is absent")
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored = payload.get("calibration_sha256")
    unsigned = dict(payload)
    unsigned.pop("calibration_sha256", None)
    if payload.get("status") != "complete" or stored != object_sha256(unsigned):
        raise Phase5PanelError("future-utility calibration checksum differs")
    return payload


def run_future_calibration_worker(
    cfg: Thought5Config, *, variant: str, resume: bool
) -> dict[str, Any]:
    """Freeze the shared B1 train-only weights before parallel H2 workers."""

    import torch
    from fastwam_ood_eval.thought4.real_runtime import release_fastwam
    from fastwam_ood_eval.thought5.future_runtime import (
        calibrate_future_sample_weights,
        load_future_bundle,
    )
    from fastwam_ood_eval.thought5.real_runtime import _require_single_visible_cuda

    if variant != "B1":
        raise Phase5PanelError("future calibration is defined only on B1")
    _require_single_visible_cuda(cfg)
    calibration_path = cfg.experiment.output_dir / "future_utility_calibration.json"
    status_path = cfg.experiment.output_dir / "future_utility_calibration_status.json"
    if calibration_path.is_file():
        calibration = _validated_calibration(calibration_path)
        return {
            "status": "complete",
            "calibration": str(calibration_path),
            "calibration_sha256": calibration["calibration_sha256"],
            "idempotent_reuse": True,
        }
    if status_path.is_file() and not resume:
        raise Phase5PanelError("partial future calibration exists; pass --resume")
    write_status_transition(
        status_path,
        {
            "schema_version": "thought5.phase5.utility_calibration_status.v1",
            "status": "running",
            "config_fingerprint": cfg.fingerprint,
        },
    )
    runtime = None
    attachment = None
    try:
        runtime, attachment, _provenance = _load_variant_runtime(cfg, variant="B1")
        if attachment is None:
            raise Phase5PanelError("B1 calibration attachment is absent")
        frozen_before = _freeze_evaluation_graph(runtime, attachment)
        track = json.loads(
            (
                cfg.experiment.output_dir / "tracks" / "b1" / "track_result.json"
            ).read_text(encoding="utf-8")
        )
        future = load_future_bundle(
            Path(track["future_bundle"]["path"]),
            variant="B1",
            fingerprint=cfg.fingerprint,
        )
        calibration = calibrate_future_sample_weights(
            cfg, attachment, future["adapter_entries"]
        )
        _assert_evaluation_graph_unchanged(runtime, attachment, frozen_before)
        write_status_transition(calibration_path, calibration)
        final = {
            "schema_version": "thought5.phase5.utility_calibration_status.v1",
            "status": "complete",
            "config_fingerprint": cfg.fingerprint,
            "calibration": str(calibration_path),
            "calibration_sha256": calibration["calibration_sha256"],
        }
        write_status_transition(status_path, final)
        return final
    except BaseException as exc:
        write_status_transition(
            status_path,
            {
                "schema_version": "thought5.phase5.utility_calibration_status.v1",
                "status": "error",
                "config_fingerprint": cfg.fingerprint,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    finally:
        if attachment is not None:
            attachment.close()
        if runtime is not None:
            release_fastwam(runtime)
        torch.cuda.empty_cache()
        gc.collect()


def run_future_utility_worker(
    cfg: Thought5Config, *, variant: str, resume: bool
) -> dict[str, Any]:
    """Train/evaluate A0/A1/AS on one frozen B1/G3/G4 backbone."""

    import torch
    from fastwam_ood_eval.thought4.real_runtime import release_fastwam
    from fastwam_ood_eval.thought5.future_runtime import (
        calibrate_future_sample_weights,
        load_future_bundle,
        train_and_evaluate_future_adapters,
    )
    from fastwam_ood_eval.thought5.real_runtime import _require_single_visible_cuda

    if variant not in {"B1", "G3", "G4"} or variant not in cfg.training.variants:
        raise Phase5PanelError(f"{variant} is outside the future-utility panel")
    _require_single_visible_cuda(cfg)
    root = cfg.experiment.output_dir / "tracks" / variant.lower() / "utility"
    status_path = root / "run_status.json"
    result_path = root / "utility_result.json"
    if status_path.is_file():
        prior = json.loads(status_path.read_text(encoding="utf-8"))
        if prior.get("status") == "complete":
            return json.loads(result_path.read_text(encoding="utf-8"))
        if not resume:
            raise Phase5PanelError(f"partial {variant} utility exists; pass --resume")
    root.mkdir(parents=True, exist_ok=True)
    write_status_transition(
        status_path,
        {
            "schema_version": "thought5.phase5.utility_status.v1",
            "status": "running",
            "variant": variant,
            "config_fingerprint": cfg.fingerprint,
        },
    )
    runtime = None
    attachment = None
    try:
        runtime, attachment, provenance = _load_variant_runtime(
            cfg, variant=variant
        )
        if attachment is None:
            raise Phase5PanelError("future utility requires a GeoEq attachment")
        frozen_before = _freeze_evaluation_graph(runtime, attachment)
        track = json.loads(
            (
                cfg.experiment.output_dir
                / "tracks"
                / variant.lower()
                / "track_result.json"
            ).read_text(encoding="utf-8")
        )
        future = load_future_bundle(
            Path(track["future_bundle"]["path"]),
            variant=variant,
            fingerprint=cfg.fingerprint,
        )
        calibration_path = (
            cfg.experiment.output_dir / "future_utility_calibration.json"
        )
        if variant == "B1":
            if calibration_path.is_file():
                calibration = _validated_calibration(calibration_path)
            else:
                calibration = calibrate_future_sample_weights(
                    cfg, attachment, future["adapter_entries"]
                )
                write_status_transition(calibration_path, calibration)
        else:
            calibration = _validated_calibration(calibration_path)
        descriptor = train_and_evaluate_future_adapters(
            cfg,
            attachment,
            backbone_variant=variant,
            entries=future["adapter_entries"],
            calibration=calibration,
            output_path=root / "future_utility_bundle.pkl",
        )
        _assert_evaluation_graph_unchanged(runtime, attachment, frozen_before)
        result = {
            "schema_version": "thought5.phase5.utility_worker_result.v1",
            "status": "complete",
            "variant": variant,
            "config_fingerprint": cfg.fingerprint,
            "bundle": descriptor,
            "calibration_sha256": calibration["calibration_sha256"],
            "frozen_graph_sha256_before": frozen_before,
            "frozen_graph_unchanged": True,
            "backbone_gradient_count": sum(
                parameter.grad is not None for parameter in runtime.model.parameters()
            ),
            "success_outcome_read": False,
            **provenance,
        }
        if result["backbone_gradient_count"] != 0:
            raise Phase5PanelError("future utility produced a backbone gradient")
        write_status_transition(result_path, result)
        write_status_transition(
            status_path,
            {
                "schema_version": "thought5.phase5.utility_status.v1",
                "status": "complete",
                "variant": variant,
                "config_fingerprint": cfg.fingerprint,
                "result": str(result_path),
            },
        )
        return result
    except BaseException as exc:
        write_status_transition(
            status_path,
            {
                "schema_version": "thought5.phase5.utility_status.v1",
                "status": "error",
                "variant": variant,
                "config_fingerprint": cfg.fingerprint,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    finally:
        if attachment is not None:
            attachment.close()
        if runtime is not None:
            release_fastwam(runtime)
        torch.cuda.empty_cache()
        gc.collect()


def run_rollout_worker(
    cfg: Thought5Config, *, variant: str, resume: bool
) -> dict[str, Any]:
    """Collect one model's rollout bundle with episode-level resume."""

    import torch
    from fastwam_ood_eval.thought4.real_runtime import release_fastwam
    from fastwam_ood_eval.thought5.real_runtime import _require_single_visible_cuda
    from fastwam_ood_eval.thought5.rollout_runtime import collect_rollout_bundle

    if variant not in cfg.training.variants:
        raise Phase5PanelError(f"{variant} is outside the rollout panel")
    _require_single_visible_cuda(cfg)
    root = cfg.experiment.output_dir / "tracks" / variant.lower() / "rollout"
    status_path = root / "run_status.json"
    result_path = root / "rollout_result.json"
    if status_path.is_file():
        prior = json.loads(status_path.read_text(encoding="utf-8"))
        if prior.get("status") == "complete":
            return json.loads(result_path.read_text(encoding="utf-8"))
        if not resume:
            raise Phase5PanelError(f"partial {variant} rollout exists; pass --resume")
    root.mkdir(parents=True, exist_ok=True)
    write_status_transition(
        status_path,
        {
            "schema_version": "thought5.phase5.rollout_status.v1",
            "status": "running",
            "variant": variant,
            "config_fingerprint": cfg.fingerprint,
        },
    )
    runtime = None
    attachment = None
    try:
        runtime, attachment, provenance = _load_variant_runtime(
            cfg, variant=variant
        )
        frozen_before = _freeze_evaluation_graph(runtime, attachment)
        descriptor = collect_rollout_bundle(
            cfg,
            runtime,
            attachment,
            variant=variant,
            samples=_load_rendered_samples(cfg),
            output_path=root / "rollout_bundle.pkl",
        )
        _assert_evaluation_graph_unchanged(runtime, attachment, frozen_before)
        result = {
            "schema_version": "thought5.phase5.rollout_worker_result.v1",
            "status": "complete",
            "variant": variant,
            "config_fingerprint": cfg.fingerprint,
            "bundle": descriptor,
            "frozen_graph_sha256_before": frozen_before,
            "frozen_graph_unchanged": True,
            "checkpoint_selection_read_rollout": False,
            **provenance,
        }
        write_status_transition(result_path, result)
        write_status_transition(
            status_path,
            {
                "schema_version": "thought5.phase5.rollout_status.v1",
                "status": "complete",
                "variant": variant,
                "config_fingerprint": cfg.fingerprint,
                "result": str(result_path),
            },
        )
        return result
    except BaseException as exc:
        write_status_transition(
            status_path,
            {
                "schema_version": "thought5.phase5.rollout_status.v1",
                "status": "error",
                "variant": variant,
                "config_fingerprint": cfg.fingerprint,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    finally:
        if attachment is not None:
            attachment.close()
        if runtime is not None:
            release_fastwam(runtime)
        torch.cuda.empty_cache()
        gc.collect()


def _spawn_wave(
    cfg: Thought5Config,
    wave: Sequence[tuple[str, str]],
    *,
    resume: bool,
    mode: str = "track",
) -> None:
    if mode not in {"track", "calibration", "utility", "rollout"}:
        raise Phase5PanelError(f"unknown panel worker mode: {mode}")
    processes: list[tuple[str, subprocess.Popen[Any], Any]] = []
    project_commit = clean_project_commit()
    log_root = cfg.experiment.output_dir / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    for variant, physical_gpu in wave:
        log = (log_root / f"{mode}_{variant.lower()}.log").open("a", encoding="utf-8")
        environment = dict(os.environ)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": physical_gpu,
                "MUJOCO_EGL_DEVICE_ID": physical_gpu,
                "THOUGHT5_PANEL_WORKER_VARIANT": variant,
                "THOUGHT5_PANEL_WORKER_MODE": mode,
                "THOUGHT5_PROJECT_COMMIT": project_commit,
            }
        )
        command = [
            sys.executable,
            "-m",
            "fastwam_ood_eval.cli",
            f"thought5-{cfg.experiment.stage}",
            "--config",
            str(_config_path_for_stage(cfg.experiment.stage)),
            "--device",
            "cuda:0",
        ]
        if resume:
            command.append("--resume")
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((variant, process, log))
        _progress(
            "panel_worker_spawned",
            mode=mode,
            variant=variant,
            physical_gpu=physical_gpu,
        )
    failures: dict[str, int] = {}
    for variant, process, log in processes:
        code = process.wait()
        log.close()
        if code:
            failures[variant] = code
    if failures:
        raise Phase5PanelError(f"panel {mode} worker failures: {failures}")


def _config_path_for_stage(stage: str) -> Path:
    return Path(
        {
            "pilot": "configs/thought5/phase5_pilot_v3.yaml",
            "formal": "configs/thought5/phase5_formal_v2.yaml",
        }[stage]
    )


def _collect_track_results(cfg: Thought5Config) -> dict[str, Any]:
    results: dict[str, Any] = {}
    initial_hashes: set[str] = set()
    trainable_counts: set[int] = set()
    project_commits: set[str] = set()
    for variant in cfg.training.variants:
        path = cfg.experiment.output_dir / "tracks" / variant.lower() / "track_result.json"
        if not path.is_file():
            raise Phase5PanelError(f"missing completed track result: {variant}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") != "complete" or row.get("config_fingerprint") != cfg.fingerprint:
            raise Phase5PanelError(f"invalid track result: {variant}")
        results[variant] = row
        project_commits.add(str(row.get("project_commit")))
        if variant != "B0":
            initial_hashes.add(str(row["initial_state_sha256"]))
            trainable_counts.add(int(row["parameter_manifest"]["trainable_parameter_count"]))
    if len(initial_hashes) > 1 or len(trainable_counts) > 1:
        raise Phase5PanelError("matched track initialization or parameter budget differs")
    if project_commits != {clean_project_commit()}:
        raise Phase5PanelError("track project commits are not exactly matched")
    return results


def _evaluate_representation(
    cfg: Thought5Config, tracks: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    from fastwam_ood_eval.thought5.representation_runtime import (
        evaluate_representation_bundles,
    )

    paths = {
        variant: Path(track["representation_bundle"]["path"])
        for variant, track in tracks.items()
    }
    result = evaluate_representation_bundles(cfg, paths)
    write_status_transition(
        cfg.experiment.output_dir / "representation_results.json", result
    )
    return result


def _evaluate_future_geometry(
    cfg: Thought5Config, tracks: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    from fastwam_ood_eval.thought5.future_runtime import evaluate_future_bundles

    paths = {
        variant: Path(track["future_bundle"]["path"])
        for variant, track in tracks.items()
        if "future_bundle" in track
    }
    result = evaluate_future_bundles(cfg, paths)
    write_status_transition(
        cfg.experiment.output_dir / "future_geometry_results.json", result
    )
    return result


def _parallel_waves(
    variants: Sequence[str], physical_gpu_ids: Sequence[str]
) -> tuple[tuple[tuple[str, str], ...], ...]:
    if not variants or not physical_gpu_ids:
        raise Phase5PanelError("worker wave cannot be empty")
    ids = tuple(physical_gpu_ids)
    waves = []
    for start in range(0, len(variants), len(ids)):
        chunk = variants[start : start + len(ids)]
        waves.append(
            tuple((variant, ids[index]) for index, variant in enumerate(chunk))
        )
    return tuple(waves)


def _run_future_utility_panel(
    cfg: Thought5Config,
    tracks: Mapping[str, Mapping[str, Any]],
    physical_gpu_ids: Sequence[str],
    *,
    resume: bool,
) -> dict[str, Any]:
    variants = tuple(
        variant for variant in ("B1", "G3", "G4") if variant in tracks
    )
    if not {"B1", "G3"}.issubset(variants):
        raise Phase5PanelError("H2 requires B1/G3 tracks")
    # Calibration alone runs first.  The long B1/G3 Adapter tracks can then
    # occupy separate GPUs without racing the shared normalization artifact.
    _spawn_wave(
        cfg,
        (("B1", str(physical_gpu_ids[0])),),
        resume=resume,
        mode="calibration",
    )
    for wave in _parallel_waves(variants, physical_gpu_ids):
        _spawn_wave(cfg, wave, resume=resume, mode="utility")
    descriptors: dict[str, Any] = {}
    bundle_paths: dict[str, Path] = {}
    for variant in variants:
        result_path = (
            cfg.experiment.output_dir
            / "tracks"
            / variant.lower()
            / "utility"
            / "utility_result.json"
        )
        result = _complete_result(result_path, label=f"{variant} future utility")
        descriptors[variant] = result
        bundle_paths[variant] = Path(result["bundle"]["path"])
    from fastwam_ood_eval.thought5.future_runtime import evaluate_utility_bundles

    aggregate = evaluate_utility_bundles(cfg, bundle_paths)
    aggregate["worker_results"] = descriptors
    aggregate["result_sha256"] = object_sha256(
        {key: value for key, value in aggregate.items() if key != "result_sha256"}
    )
    write_status_transition(
        cfg.experiment.output_dir / "future_utility_results.json", aggregate
    )
    return aggregate


def _run_rollout_panel(
    cfg: Thought5Config,
    tracks: Mapping[str, Mapping[str, Any]],
    physical_gpu_ids: Sequence[str],
    *,
    resume: bool,
) -> dict[str, Any]:
    variants = tuple(variant for variant in cfg.training.variants if variant in tracks)
    required = {"B1", "G3"} | ({"B0"} if cfg.experiment.stage == "formal" else set())
    if not required.issubset(variants):
        raise Phase5PanelError("H3 rollout variants are incomplete")
    for wave in _parallel_waves(variants, physical_gpu_ids):
        _spawn_wave(cfg, wave, resume=resume, mode="rollout")
    descriptors: dict[str, Any] = {}
    bundle_paths: dict[str, Path] = {}
    for variant in variants:
        result_path = (
            cfg.experiment.output_dir
            / "tracks"
            / variant.lower()
            / "rollout"
            / "rollout_result.json"
        )
        result = _complete_result(result_path, label=f"{variant} rollout")
        descriptors[variant] = result
        bundle_paths[variant] = Path(result["bundle"]["path"])
    # The pilot has no B0 track.  H3 itself only needs B1/G3/G4; the formal
    # collector additionally enforces B0 for the original-checkpoint table.
    from fastwam_ood_eval.thought5.rollout_runtime import evaluate_rollout_bundles

    if cfg.experiment.stage == "pilot":
        # The aggregator's B0 requirement is a formal provenance constraint;
        # pilot remains explicitly non-scientific and uses only B1/G3/G4.
        from fastwam_ood_eval.thought5.rollout_eval import evaluate_rollouts

        import pickle as _pickle

        records = []
        for path in bundle_paths.values():
            with path.open("rb") as handle:
                records.extend(_pickle.load(handle)["records"])
        aggregate = evaluate_rollouts(
            records,
            bootstrap_replicates=cfg.evaluation.bootstrap_replicates,
            bootstrap_seed=cfg.evaluation.bootstrap_seed + 70,
            clean_noninferiority_margin=cfg.evaluation.clean_noninferiority_margin,
            g4_equivalence_fraction=cfg.evaluation.g4_equivalence_fraction,
        )
        aggregate.update(
            {
                "bundle_descriptors": {
                    key: {"path": str(path), "sha256": file_sha256(path)}
                    for key, path in bundle_paths.items()
                },
                "exact_state_initial_pairing_verified": True,
                "checkpoint_selection_read_rollout": False,
                "inference_uses_gt_depth": False,
                "test_time_future_imagination": False,
            }
        )
    else:
        aggregate = evaluate_rollout_bundles(cfg, bundle_paths)
    aggregate["worker_results"] = descriptors
    aggregate["result_sha256"] = object_sha256(
        {key: value for key, value in aggregate.items() if key != "result_sha256"}
    )
    write_status_transition(cfg.experiment.output_dir / "rollout_results.json", aggregate)
    return aggregate


def _complete_result(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise Phase5PanelError(f"{label} collector has not committed {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise Phase5PanelError(f"{label} result is not complete")
    return payload


def _validated_execution_schedule(cfg: Thought5Config) -> dict[str, Any]:
    path = cfg.experiment.output_dir / "execution_schedule.json"
    if not path.is_file():
        raise Phase5PanelError("frozen execution schedule is absent")
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored = payload.get("schedule_sha256")
    unsigned = dict(payload)
    unsigned.pop("schedule_sha256", None)
    if (
        payload.get("status") != "frozen"
        or payload.get("execution_only") is not True
        or payload.get("config_fingerprint") != cfg.fingerprint
        or payload.get("stage") != cfg.experiment.stage
        or stored != object_sha256(unsigned)
    ):
        raise Phase5PanelError("frozen execution schedule is invalid")
    return payload


def _selected_development_score(track: Mapping[str, Any]) -> float:
    selected = int(track["selected_step"])
    matches = [
        row
        for row in track["development_rows"]
        if int(row["step"]) == selected
    ]
    if len(matches) != 1:
        raise Phase5PanelError("selected checkpoint lacks one development row")
    return float(matches[0]["selection_objective"])


def _pilot_direction_and_freeze(
    cfg: Thought5Config, tracks: Mapping[str, Any]
) -> dict[str, Any]:
    """Freeze formal only after every registered pilot endpoint is complete.

    Training/development loss is one input to the gate; it can never unlock
    formal by itself.  The three scientific collectors stay explicitly
    non-inferential in a single-task pilot but must report their preregistered
    directional checks and the G4 specificity control.
    """

    scores = {
        variant: _selected_development_score(track)
        for variant, track in tracks.items()
        if variant != "B0"
    }
    root = cfg.experiment.output_dir
    representation = _complete_result(
        root / "representation_results.json", label="representation"
    )
    future_geometry = _complete_result(
        root / "future_geometry_results.json", label="future geometry"
    )
    future_utility = _complete_result(
        root / "future_utility_results.json", label="future utility"
    )
    rollout = _complete_result(root / "rollout_results.json", label="rollout")
    collector_direction = {
        "representation": bool(representation.get("pilot_direction_observed")),
        "future_geometry": bool(future_geometry.get("pilot_direction_observed")),
        "future_utility": bool(future_utility.get("pilot_direction_observed")),
        "rollout": bool(rollout.get("pilot_direction_observed")),
    }
    shuffled_matches = any(
        value.get("shuffled_control_matches_gain") is not False
        for value in (representation, future_geometry, future_utility, rollout)
    )
    training_direction = (
        scores["G3"] < scores["B1"] and scores["G3"] < scores["G4"]
    )
    direction = bool(
        training_direction
        and all(collector_direction.values())
        and not shuffled_matches
    )
    result = {
        "schema_version": "thought5.phase5.pilot_direction.v1",
        "status": "complete",
        "scientific_result": False,
        "single_task_noninferential": True,
        "scores": scores,
        "training_direction_observed": training_direction,
        "collector_direction": collector_direction,
        "shuffled_control_matches_gain": shuffled_matches,
        "g3_direction_observed": direction,
        "formal_unlocked": direction,
        "formal_outcomes_read": False,
    }
    write_status_transition(cfg.experiment.output_dir / "pilot_direction.json", result)
    if direction:
        pilot_schedule = _validated_execution_schedule(cfg)
        formal_cfg_path = Path("configs/thought5/phase5_formal_v2.yaml")
        from fastwam_ood_eval.thought5.config import load_thought5_config

        formal_cfg = load_thought5_config(formal_cfg_path)
        formal_manifest_path = formal_cfg.experiment.output_dir / "cohort_manifest.json"
        if not formal_manifest_path.is_file():
            raise Phase5PanelError(
                "run the Phase5 audit/dry-run before freezing formal"
            )
        formal_manifest = json.loads(
            formal_manifest_path.read_text(encoding="utf-8")
        )
        if formal_manifest.get("manifest_sha256") is None:
            raise Phase5PanelError("formal candidate cohort is not sealed")
        freeze = {
            "schema_version": "thought5.phase5.formal_protocol_freeze.v1",
            "status": "frozen",
            "source_pilot_config_fingerprint": cfg.fingerprint,
            "formal_config_fingerprint": formal_cfg.fingerprint,
            "project_commit": clean_project_commit(),
            "formal_config_sha256": file_sha256(formal_cfg_path),
            "formal_cohort_manifest_sha256": formal_manifest[
                "manifest_sha256"
            ],
            "pilot_selected_steps_observed": {
                variant: int(tracks[variant]["selected_step"])
                for variant in ("B1", "G3")
            },
            "formal_training_steps": formal_cfg.training.max_steps,
            "formal_checkpoint_rule": formal_cfg.training.checkpoint_rule,
            "lambda_g3": dict(cfg.training.lambda_by_variant["G3"]),
            "layer": 15,
            "lora_targets": list(cfg.method.lora_targets),
            "checkpoint_selection_read_rollout": False,
            "formal_recipe_mutation_allowed": False,
            "pilot_evidence_sha256": {
                name: file_sha256(root / name)
                for name in (
                    "representation_results.json",
                    "future_geometry_results.json",
                    "future_utility_results.json",
                    "rollout_results.json",
                )
            },
            "pilot_execution_schedule_sha256": pilot_schedule[
                "schedule_sha256"
            ],
            "pilot_execution_schedule_file_sha256": file_sha256(
                root / "execution_schedule.json"
            ),
        }
        freeze["freeze_sha256"] = object_sha256(freeze)
        write_status_transition(
            cfg.experiment.output_dir / "formal_protocol_frozen.json", freeze
        )
    return result


def _validate_frozen_pilot_schedule(freeze: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(
        "outputs/thought5/phase5_camera_equivariant_geo_repa_pilot_v3"
    )
    path = root / "execution_schedule.json"
    if (
        not path.is_file()
        or file_sha256(path)
        != freeze.get("pilot_execution_schedule_file_sha256")
    ):
        raise Phase5PanelError("pilot execution schedule changed before formal")
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored = payload.get("schedule_sha256")
    unsigned = dict(payload)
    unsigned.pop("schedule_sha256", None)
    if (
        stored != freeze.get("pilot_execution_schedule_sha256")
        or stored != object_sha256(unsigned)
    ):
        raise Phase5PanelError("pilot execution schedule seal is invalid")
    return payload


def _pilot_specificity_for_formal(freeze: Mapping[str, Any]) -> bool:
    root = Path(
        "outputs/thought5/phase5_camera_equivariant_geo_repa_pilot_v3"
    )
    _validate_frozen_pilot_schedule(freeze)
    values = []
    for name in (
        "representation_results.json",
        "future_geometry_results.json",
        "future_utility_results.json",
        "rollout_results.json",
    ):
        path = root / name
        expected = freeze["pilot_evidence_sha256"].get(name)
        if not path.is_file() or file_sha256(path) != expected:
            raise Phase5PanelError(f"pilot evidence changed before formal: {name}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = payload.get("shuffled_control_matches_gain")
        if value is None:
            raise Phase5PanelError(f"pilot G4 result is absent in {name}")
        values.append(bool(value))
    return any(values)


def _exploratory_mediation(
    representation: Mapping[str, Any],
    utility: Mapping[str, Any],
    rollout: Mapping[str, Any],
) -> dict[str, Any]:
    """Task-level descriptive chain, never a causal mediation estimate."""

    import numpy as np

    representation_by_task = {
        str(task): -float(value["mean_g3_minus_b1_camera_gap"])
        for task, value in representation.get("per_task", {}).items()
    }
    utility_values: dict[str, list[float]] = {}
    for row in utility.get("rows", []):
        if row["condition"] != "camera":
            continue
        utility_values.setdefault(str(row["task_id"]), []).append(
            float(row["utility"])
        )
    # Rows hold utility within each backbone; use the explicit G3-B1 pairing.
    by_identity: dict[tuple[str, str, int], dict[str, float]] = {}
    for row in utility.get("rows", []):
        if row["condition"] != "camera":
            continue
        key = (str(row["task_id"]), str(row["episode_id"]), int(row["flow_slot"]))
        by_identity.setdefault(key, {})[str(row["backbone"])] = float(row["utility"])
    utility_by_task: dict[str, list[float]] = {}
    for (task, _episode, _flow), values in by_identity.items():
        if {"B1", "G3"}.issubset(values):
            utility_by_task.setdefault(task, []).append(values["G3"] - values["B1"])
    utility_mean = {
        task: float(np.mean(values)) for task, values in utility_by_task.items()
    }
    rollout_rows: dict[tuple[str, int], dict[str, bool]] = {}
    for descriptor in rollout.get("bundle_descriptors", {}).values():
        with Path(descriptor["path"]).open("rb") as handle:
            bundle = pickle.load(handle)
        for row in bundle["records"]:
            if row.condition == "camera" and row.variant in {"B1", "G3"}:
                rollout_rows.setdefault(
                    (str(row.task_id), int(row.episode_seed)), {}
                )[row.variant] = bool(row.success)
    success_by_task: dict[str, list[float]] = {}
    for (task, _seed), values in rollout_rows.items():
        if {"B1", "G3"}.issubset(values):
            success_by_task.setdefault(task, []).append(
                float(values["G3"]) - float(values["B1"])
            )
    success_mean = {
        task: float(np.mean(values)) for task, values in success_by_task.items()
    }
    tasks = sorted(set(representation_by_task) & set(utility_mean) & set(success_mean))
    correlations: dict[str, float | None] = {
        "representation_vs_utility": None,
        "utility_vs_success": None,
        "representation_vs_success": None,
    }
    if len(tasks) >= 3:
        arrays = {
            "representation": np.asarray([representation_by_task[t] for t in tasks]),
            "utility": np.asarray([utility_mean[t] for t in tasks]),
            "success": np.asarray([success_mean[t] for t in tasks]),
        }
        for name, left, right in (
            ("representation_vs_utility", "representation", "utility"),
            ("utility_vs_success", "utility", "success"),
            ("representation_vs_success", "representation", "success"),
        ):
            correlations[name] = (
                float(np.corrcoef(arrays[left], arrays[right])[0, 1])
                if np.std(arrays[left]) > 0 and np.std(arrays[right]) > 0
                else None
            )
    return {
        "status": "exploratory_noncausal",
        "task_count": len(tasks),
        "per_task": {
            task: {
                "representation_gain": representation_by_task[task],
                "future_utility_gain": utility_mean[task],
                "camera_success_gain": success_mean[task],
            }
            for task in tasks
        },
        "pearson_correlations": correlations,
        "inference_allowed": False,
        "note": "Task-level association is not a causal mediation estimate.",
    }


def _formal_report(
    classification: Mapping[str, Any],
    representation: Mapping[str, Any],
    future_geometry: Mapping[str, Any],
    utility: Mapping[str, Any],
    rollout: Mapping[str, Any],
) -> str:
    import numpy as np

    label = str(classification["classification"])
    evidence = classification["evidence"]

    def fmt(value: Any) -> str:
        return "NA" if value is None else f"{float(value):.6f}"

    def rollout_value(variant: str, condition: str, key: str) -> Any:
        return rollout.get("summaries", {}).get(
            f"{variant}:{condition}", {}
        ).get(key)

    def linear_probe_rmse(
        variant: str, panel: str, target: str, condition: str
    ) -> float | None:
        rows = (
            representation.get("all_registered_probe_panels", {})
            .get(variant, {})
            .get(panel, {})
            .get("rows", [])
        )
        values = [
            float(
                row["condition_metrics"][condition]["rmse_grouped_bootstrap"][
                    "estimate"
                ]
            )
            for row in rows
            if row.get("probe_kind") == "linear"
            and row.get("target") == target
            and condition in row.get("condition_metrics", {})
        ]
        return float(np.mean(values)) if values else None

    action_current = {
        variant: linear_probe_rmse(
            variant, "action", "eef_object_translation_camera", "camera"
        )
        for variant in ("B1", "G3")
    }
    action_future = {
        variant: linear_probe_rmse(
            variant, "action", "action_se3_trajectory", "camera"
        )
        for variant in ("B1", "G3")
    }
    utility_summary: dict[str, dict[str, float]] = {}
    for backbone in ("B1", "G3"):
        rows = [
            row for row in utility.get("rows", []) if row["backbone"] == backbone
        ]
        utility_summary[backbone] = {
            variant: float(np.mean([row["losses"][variant] for row in rows]))
            for variant in ("A0", "A1", "AS")
        }
        utility_summary[backbone]["utility"] = float(
            np.mean([row["utility"] for row in rows])
        )
        utility_summary[backbone]["specificity"] = float(
            np.mean([row["specificity"] for row in rows])
        )
    component_rows = []
    main_summary = representation.get("variant_main_endpoint_summary", {})
    for variant in ("B0", "B1", "G1", "G2", "G3"):
        component_rows.append(
            "| {variant} | {gap} | {clean} | {camera} |".format(
                variant=variant,
                gap=fmt(main_summary.get(variant, {}).get("camera_gap")),
                clean=fmt(rollout_value(variant, "clean", "success_rate")),
                camera=fmt(rollout_value(variant, "camera", "success_rate")),
            )
        )
    support_level = (
        "full support"
        if label == "full_mechanism_support"
        else (
            "not supported"
            if label == "mechanism_not_supported"
            else "partial support"
        )
    )
    if all(
        bool(classification[key])
        for key in ("h1_supported", "h2_supported", "h3_supported")
    ):
        mediation_answer = (
            "三层证据方向一致，支持 future utility 是 Camera geometry repair "
            "连接到闭环收益的一条重要路径；任务级相关仍不是正式因果中介估计。"
        )
    elif classification["h1_supported"] and not classification["h2_supported"]:
        mediation_answer = (
            "只支持当前/未来表征修复，未证明该修复经 future utility 中介到达动作与成功率。"
        )
    else:
        mediation_answer = "证据链不完整，不能主张 future utility 中介。"
    return f"""# Phase 5 — Camera-Equivariant Geometry Alignment

## Frozen evidence chain

Thought1 found Camera OOD failure; Thought2 found degraded future consistency;
Thought3 found future-content action sensitivity without held-out utility;
Thought4 localized an action-consumed `camera_equivariance_gap`. Thought5
intervened only with the preregistered Geo-REPA + relative pose/camera-ray path.

## Formal result and preregistered gates

- Mechanism classification: `{label}` ({support_level})
- H1 / H2 / H3: `{classification['h1_supported']}` / `{classification['h2_supported']}` / `{classification['h3_supported']}`
- Camera geometry gap reduction vs B1: `{representation['gap_reduction_fraction']:.6f}`
- H1 episode/task upper CI: `{fmt(representation['g3_minus_b1_camera_grouped_bootstrap']['upper'])}` / `{fmt(representation['g3_minus_b1_camera_task_cluster_bootstrap']['upper'])}`
- H2 episode/task lower CI: `{fmt(utility['utility_g3_minus_b1_grouped_bootstrap']['lower'])}` / `{fmt(utility['utility_g3_minus_b1_task_cluster_bootstrap']['lower'])}`
- H3 episode/task lower CI: `{fmt(rollout['g3_minus_b1_paired_intervals']['camera']['lower'])}` / `{fmt(rollout['g3_minus_b1_task_cluster_intervals']['camera']['lower'])}`

## Fifteen required answers

1. **Camera geometry gap:** Geo-REPA + Pose/Ray 的相对缩减为 `{representation['gap_reduction_fraction']:.6f}`；H1=`{classification['h1_supported']}`。
2. **Video layer:** 干预和主 probe 均固定在 layer 15 的 `mot.video_kv_cache.15.v`，没有按 formal 结果重选层。
3. **Action probes:** Camera 下 current-geometry RMSE B1/G3=`{fmt(action_current['B1'])}`/`{fmt(action_current['G3'])}`；future-SE(3) RMSE=`{fmt(action_future['B1'])}`/`{fmt(action_future['G3'])}`。数值仅表示冻结 probe 可读性，不等同 success。
4. **K=1 future geometry:** Camera RMSE B1/G3=`{future_geometry['main_camera_error']['B1']:.6f}`/`{future_geometry['main_camera_error']['G3']:.6f}`；Clean=`{future_geometry['main_clean_error']['B1']:.6f}`/`{future_geometry['main_clean_error']['G3']:.6f}`；probe 只在 train 拟合、development 选 alpha、formal 只读。
5. **Action-sensitive → action-useful:** H2=`{classification['h2_supported']}`；G3 mean utility=`{utility_summary['G3']['utility']:.6f}`，B1=`{utility_summary['B1']['utility']:.6f}`。
6. **Correct/null/shuffle:** G3 A0/A1/AS loss=`{utility_summary['G3']['A0']:.6f}`/`{utility_summary['G3']['A1']:.6f}`/`{utility_summary['G3']['AS']:.6f}`；correct-null 与 correct-shuffle lower CI=`{fmt(utility['g3_correct_minus_null_utility_grouped_bootstrap']['lower'])}`/`{fmt(utility['g3_correct_minus_shuffle_specificity_grouped_bootstrap']['lower'])}`。
7. **Camera OOD success:** B1/G3=`{fmt(rollout_value('B1', 'camera', 'success_rate'))}`/`{fmt(rollout_value('G3', 'camera', 'success_rate'))}`；H3=`{classification['h3_supported']}`。
8. **Clean performance:** B1/G3=`{fmt(rollout_value('B1', 'clean', 'success_rate'))}`/`{fmt(rollout_value('G3', 'clean', 'success_rate'))}`；non-inferiority=`{rollout['clean_noninferior']}`。
9. **Lighting / Robot-init:** B1→G3 success 为 Lighting `{fmt(rollout_value('B1', 'lighting', 'success_rate'))}`→`{fmt(rollout_value('G3', 'lighting', 'success_rate'))}`，Robot-init `{fmt(rollout_value('B1', 'robot_init', 'success_rate'))}`→`{fmt(rollout_value('G3', 'robot_init', 'success_rate'))}`；Camera specificity=`{rollout['camera_specific']}`。
10. **Matched fine-tuning control:** B0/B1 Camera success=`{fmt(rollout_value('B0', 'camera', 'success_rate'))}`/`{fmt(rollout_value('B1', 'camera', 'success_rate'))}`；B1 explains ≥80% gain=`{evidence['matched_control_explains_gain']}`。
11. **G1/G2 component contribution:** 以下表同时给出 layer-15 Camera gap 与闭环 success；它用于分离 Geo-REPA 和 Pose/Ray，不把单一数值事后指定为主终点。

| Variant | Camera geometry gap | Clean success | Camera success |
|---|---:|---:|---:|
{chr(10).join(component_rows)}

12. **Shuffled geometry:** pilot G4 是否达到 G3 的 ≥80% 收益=`{evidence['shuffled_control_matches_gain']}`；若为 true，机制分类按预注册规则降为 not supported。
13. **总体判定:** `{support_level}`，机器可读标签为 `{label}`。
14. **可写/不可写:** 只有 `full_mechanism_support` 允许写“Camera Equivariance Gap 是连接 future utility 缺失与 Camera OOD failure 的重要机制之一”。即使 full，也不能写成唯一、充分原因，不能外推到所有 world-action models，也不能把 probe 改善等同于 success 改善。
15. **Future utility 中介:** {mediation_answer}

## Interpretation boundary

Probe、technical action sensitivity、held-out utility 与 closed-loop success 是
四类独立终点。`exploratory_mediation` 只提供 task-level 描述性相关，不构成正式
causal mediation analysis；所有结论均受预注册 cohort、Fast-WAM checkpoint、
LIBERO Goal 与四种扰动范围限制。
"""


def _finalize_formal(
    cfg: Thought5Config,
    tracks: Mapping[str, Mapping[str, Any]],
    freeze: Mapping[str, Any],
) -> dict[str, Any]:
    from dataclasses import asdict

    from fastwam_ood_eval.thought5.mechanism_decision import (
        MechanismEvidence,
        classify_mechanism,
    )

    root = cfg.experiment.output_dir
    representation = _complete_result(
        root / "representation_results.json", label="representation"
    )
    future_geometry = _complete_result(
        root / "future_geometry_results.json", label="future geometry"
    )
    utility = _complete_result(root / "future_utility_results.json", label="utility")
    rollout = _complete_result(root / "rollout_results.json", label="rollout")
    b0_camera = float(rollout["summaries"]["B0:camera"]["success_rate"])
    b1_camera = float(rollout["summaries"]["B1:camera"]["success_rate"])
    g3_camera = float(rollout["summaries"]["G3:camera"]["success_rate"])
    total_gain = g3_camera - b0_camera
    matched_gain = b1_camera - b0_camera
    matched_explains = bool(
        total_gain <= 0
        or matched_gain >= cfg.evaluation.g4_equivalence_fraction * total_gain
    )
    utility_task = utility["utility_g3_minus_b1_task_cluster_bootstrap"]
    rollout_task = rollout["g3_minus_b1_task_cluster_intervals"]["camera"]
    representation_task = representation[
        "g3_minus_b1_camera_task_cluster_bootstrap"
    ]
    evidence = MechanismEvidence(
        h1_camera_gap_reduction_fraction=float(
            representation["gap_reduction_fraction"]
        ),
        h1_paired_ci_upper_below_zero=float(
            representation["g3_minus_b1_camera_grouped_bootstrap"]["upper"]
        )
        < 0,
        h1_task_ci_upper_below_zero=float(representation_task["upper"]) < 0,
        h1_clean_non_degraded=bool(representation["clean_non_degraded"]),
        h1_lighting_specific=bool(representation["lighting_specific"]),
        h2_a1_better_a0=float(
            utility["g3_correct_minus_null_utility_grouped_bootstrap"]["lower"]
        )
        > 0,
        h2_a1_better_shuffle=float(
            utility[
                "g3_correct_minus_shuffle_specificity_grouped_bootstrap"
            ]["lower"]
        )
        > 0,
        h2_utility_gain_grouped_ci_lower_above_zero=float(
            utility["utility_g3_minus_b1_grouped_bootstrap"]["lower"]
        )
        > 0,
        h2_utility_gain_task_ci_lower_above_zero=float(utility_task["lower"])
        > 0,
        h2_a0_not_abnormally_worse=bool(
            utility["a0_g3_not_abnormally_worse"]
        ),
        h3_camera_gain_grouped_ci_lower_above_zero=float(
            rollout["g3_minus_b1_paired_intervals"]["camera"]["lower"]
        )
        > 0,
        h3_camera_gain_task_ci_lower_above_zero=float(rollout_task["lower"])
        > 0,
        h3_clean_noninferior=bool(rollout["clean_noninferior"]),
        h3_camera_specific=bool(rollout["camera_specific"]),
        matched_control_explains_gain=matched_explains,
        shuffled_control_matches_gain=_pilot_specificity_for_formal(freeze),
    )
    evaluator_decisions = {
        "H1": bool(representation["h1_supported"]),
        "H2": bool(utility["h2_supported"]),
        "H3": bool(rollout["h3_supported"]),
    }
    reconstructed_decisions = {
        "H1": evidence.h1,
        "H2": evidence.h2,
        "H3": evidence.h3,
    }
    if evaluator_decisions != reconstructed_decisions:
        raise Phase5PanelError(
            "mechanism evidence reconstruction differs from endpoint gates"
        )
    exploratory = _exploratory_mediation(representation, utility, rollout)
    evidence_payload = {
        "schema_version": "thought5.phase5.mechanism_evidence.v1",
        "status": "complete",
        "H1": evidence.h1,
        "H2": evidence.h2,
        "H3": evidence.h3,
        "evidence": asdict(evidence),
        "matched_control_rule": {
            "B0_camera_success": b0_camera,
            "B1_camera_success": b1_camera,
            "G3_camera_success": g3_camera,
            "equivalence_fraction": cfg.evaluation.g4_equivalence_fraction,
        },
        "shuffled_control_source": "frozen single-task pilot G4",
        "exploratory_mediation": exploratory,
    }
    evidence_payload["evidence_sha256"] = object_sha256(evidence_payload)
    write_status_transition(root / "mechanism_evidence.json", evidence_payload)
    classification = classify_mechanism(evidence)
    classification.update(
        {
            "status": "complete",
            "evidence_sha256": evidence_payload["evidence_sha256"],
        }
    )
    classification["classification_sha256"] = object_sha256(classification)
    write_status_transition(root / "mechanism_classification.json", classification)
    write_report_transition(
        root / "report.md",
        _formal_report(
            classification,
            representation,
            future_geometry,
            utility,
            rollout,
        ),
    )
    cohort = json.loads((root / "cohort_manifest.json").read_text(encoding="utf-8"))
    execution_schedule = _validated_execution_schedule(cfg)
    checkpoint_hashes = {
        variant: file_sha256(Path(track["checkpoint"]) / "manifest.json")
        for variant, track in tracks.items()
        if variant != "B0"
    }
    integrity = execution_integrity(
        config_fingerprint=cfg.fingerprint,
        cohort_sha256=cohort["manifest_sha256"],
        stage_status={
            "training": "complete",
            "representation": "complete",
            "future_geometry": "complete",
            "future_utility": "complete",
            "rollout": "complete",
            "classification": "complete",
        },
        checkpoints=checkpoint_hashes,
        immutable_inputs={
            "backbone_checkpoint_sha256": cfg.backbone.checkpoint_sha256,
            "dataset_stats_sha256": cfg.backbone.dataset_stats_sha256,
            "fastwam_commit": cfg.backbone.fastwam_commit,
            "formal_freeze_sha256": str(freeze["freeze_sha256"]),
            "execution_schedule_sha256": execution_schedule[
                "schedule_sha256"
            ],
            "project_commit": clean_project_commit(),
        },
        status="complete",
    )
    write_status_transition(root / "execution_integrity.json", integrity)
    final = {
        "schema_version": "thought5.phase5.run_status.v1",
        "status": "complete",
        "stage": "formal",
        "config_fingerprint": cfg.fingerprint,
        "project_commit": clean_project_commit(),
        "scientific_result": True,
        "classification": classification["classification"],
        "H1": evidence.h1,
        "H2": evidence.h2,
        "H3": evidence.h3,
        "execution_schedule_sha256": execution_schedule["schedule_sha256"],
    }
    write_status_transition(root / "run_status.json", final)
    names = sorted(
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    manifest = build_artifact_manifest(root, names=names, status="complete")
    write_status_transition(root / "artifact_manifest.json", manifest)
    validate_artifact_manifest(root, manifest)
    return final


def _run_panel(cfg: Thought5Config, *, resume: bool) -> dict[str, Any]:
    project_commit = clean_project_commit()
    expected_commit = os.environ.get("THOUGHT5_PROJECT_COMMIT")
    if expected_commit is not None and expected_commit != project_commit:
        raise Phase5PanelError("panel worker project commit changed after spawn")
    worker_variant = os.environ.get("THOUGHT5_PANEL_WORKER_VARIANT")
    if worker_variant:
        worker_mode = os.environ.get("THOUGHT5_PANEL_WORKER_MODE", "track")
        if worker_mode == "track":
            return run_track_worker(cfg, variant=worker_variant, resume=resume)
        if worker_mode == "calibration":
            return run_future_calibration_worker(
                cfg, variant=worker_variant, resume=resume
            )
        if worker_mode == "utility":
            return run_future_utility_worker(
                cfg, variant=worker_variant, resume=resume
            )
        if worker_mode == "rollout":
            return run_rollout_worker(cfg, variant=worker_variant, resume=resume)
        raise Phase5PanelError(f"unknown worker mode: {worker_mode}")
    physical = _visible_physical_ids(cfg.experiment.stage)
    output = cfg.experiment.output_dir
    status_path = output / "run_status.json"
    if status_path.is_file():
        prior = json.loads(status_path.read_text(encoding="utf-8"))
        if prior.get("status") == "complete":
            raise Phase5PanelError("completed panel output is immutable")
        if prior.get("status") not in {"NOT RUN", "error", "running"}:
            raise Phase5PanelError("panel status is not resumable")
        if prior.get("project_commit") not in {None, project_commit}:
            raise Phase5PanelError("partial panel belongs to another project commit")
        if prior.get("status") != "NOT RUN" and not resume:
            raise Phase5PanelError("partial panel exists; inspect and pass --resume")
    utility_variants = tuple(
        variant
        for variant in ("B1", "G3", "G4")
        if variant in cfg.training.variants
    )
    schedule = {
        "schema_version": "thought5.phase5.execution_schedule.v1",
        "status": "frozen",
        "execution_only": True,
        "stage": cfg.experiment.stage,
        "config_fingerprint": cfg.fingerprint,
        "project_commit": project_commit,
        "physical_gpu_ids": list(physical),
        "worker_contract": "one independent process and model per physical GPU",
        "distributed_training": False,
        "track_waves": parallel_schedule(cfg.experiment.stage, physical),
        "future_calibration_waves": ((("B1", physical[0]),),),
        "future_utility_waves": _parallel_waves(utility_variants, physical),
        "rollout_waves": _parallel_waves(cfg.training.variants, physical),
    }
    schedule["schedule_sha256"] = object_sha256(schedule)
    write_json_once(
        output / "execution_schedule.json",
        schedule,
        allow_identical=True,
    )
    _validated_execution_schedule(cfg)
    write_status_transition(
        status_path,
        {
            "schema_version": "thought5.phase5.run_status.v1",
            "status": "running",
            "stage": cfg.experiment.stage,
            "config_fingerprint": cfg.fingerprint,
            "project_commit": project_commit,
            "scientific_result": False,
            "execution_schedule_sha256": schedule["schedule_sha256"],
        },
    )
    try:
        cache = prepare_render_cache(cfg, resume=resume)
        for wave in parallel_schedule(cfg.experiment.stage, physical):
            _spawn_wave(cfg, wave, resume=resume)
        tracks = _collect_track_results(cfg)
        training = {
            "schema_version": "thought5.phase5.training_results.v1",
            "status": "complete",
            "stage": cfg.experiment.stage,
            "project_commit": project_commit,
            "matched_parameter_budget": True,
            "render_cache_sha256": cache["render_cache_sha256"],
            "execution_schedule_sha256": schedule["schedule_sha256"],
            "tracks": tracks,
        }
        write_status_transition(output / "training_results.json", training)
        _progress("representation_evaluation_started")
        _evaluate_representation(cfg, tracks)
        _progress("representation_evaluation_complete")
        _progress("future_geometry_evaluation_started")
        _evaluate_future_geometry(cfg, tracks)
        _progress("future_geometry_evaluation_complete")
        _progress("future_utility_panel_started")
        _run_future_utility_panel(
            cfg, tracks, physical, resume=resume
        )
        _progress("future_utility_panel_complete")
        _progress("paired_rollout_panel_started")
        _run_rollout_panel(cfg, tracks, physical, resume=resume)
        _progress("paired_rollout_panel_complete")
        if cfg.experiment.stage == "pilot":
            direction = _pilot_direction_and_freeze(cfg, tracks)
            final = {
                "schema_version": "thought5.phase5.run_status.v1",
                "status": "complete",
                "stage": "pilot",
                "config_fingerprint": cfg.fingerprint,
                "project_commit": project_commit,
                "scientific_result": False,
                "formal_unlocked": direction["formal_unlocked"],
                "execution_schedule_sha256": schedule["schedule_sha256"],
                "note": "single-task directional pilot; not a formal result",
            }
            write_status_transition(status_path, final)
            return final
        freeze_path = Path(
            "outputs/thought5/phase5_camera_equivariant_geo_repa_pilot_v3/"
            "formal_protocol_frozen.json"
        )
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        return _finalize_formal(cfg, tracks, freeze)
    except BaseException as exc:
        write_status_transition(
            status_path,
            {
                "schema_version": "thought5.phase5.run_status.v1",
                "status": "error",
                "stage": cfg.experiment.stage,
                "config_fingerprint": cfg.fingerprint,
                "project_commit": project_commit,
                "error": f"{type(exc).__name__}: {exc}",
                "scientific_result": False,
                "execution_schedule_sha256": schedule["schedule_sha256"],
            },
        )
        raise


def run_pilot(cfg: Thought5Config, *, resume: bool = False) -> dict[str, Any]:
    if cfg.experiment.stage != "pilot":
        raise Phase5PanelError("pilot runner received a non-pilot config")
    smoke = Path(
        "outputs/thought5/phase5_camera_equivariant_geo_repa_smoke_v4/smoke_result.json"
    )
    if not smoke.is_file():
        raise Phase5PanelError("pilot remains locked until real smoke completes")
    payload = json.loads(smoke.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or not payload.get("pilot_unlocked"):
        raise Phase5PanelError("real smoke did not unlock pilot")
    if payload.get("project_commit") != clean_project_commit():
        raise Phase5PanelError("pilot code commit differs from the smoke commit")
    return _run_panel(cfg, resume=resume)


def run_formal(cfg: Thought5Config, *, resume: bool = False) -> dict[str, Any]:
    if cfg.experiment.stage != "formal":
        raise Phase5PanelError("formal runner received a non-formal config")
    freeze = Path(
        "outputs/thought5/phase5_camera_equivariant_geo_repa_pilot_v3/"
        "formal_protocol_frozen.json"
    )
    if not freeze.is_file():
        raise Phase5PanelError("formal remains locked until pilot freezes its recipe")
    payload = json.loads(freeze.read_text(encoding="utf-8"))
    stored = payload.get("freeze_sha256")
    unsigned = dict(payload)
    unsigned.pop("freeze_sha256", None)
    if payload.get("status") != "frozen" or stored != object_sha256(unsigned):
        raise Phase5PanelError("formal protocol freeze is invalid")
    if payload.get("formal_config_fingerprint") != cfg.fingerprint:
        raise Phase5PanelError("formal config differs from the pilot-frozen candidate")
    if payload.get("project_commit") != clean_project_commit():
        raise Phase5PanelError("formal code commit differs from the pilot freeze")
    _validate_frozen_pilot_schedule(payload)
    manifest_path = cfg.experiment.output_dir / "cohort_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("formal_cohort_manifest_sha256") != manifest.get(
        "manifest_sha256"
    ):
        raise Phase5PanelError("formal cohort differs from the pilot-frozen manifest")
    return _run_panel(cfg, resume=resume)
