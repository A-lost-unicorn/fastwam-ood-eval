"""Real feature collection and frozen-probe evaluation for Phase 5-A.

This module reuses Thought4's probe implementation but never its formal rows or
model-selection outcome.  Layer 15, pooling, targets, probe capacity, seeds and
test-blind fitting rules are fixed before any Phase5 rollout is read.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fastwam_ood_eval.thought5.camera_rays import two_camera_token_rays
from fastwam_ood_eval.thought5.pose_transforms import (
    pose_embedding_12,
    relative_clean_to_camera,
)
from fastwam_ood_eval.thought5.paired_geometry_data import cohort_manifest
from fastwam_ood_eval.thought5.representation_eval import (
    RepresentationRecord,
    evaluate_h1,
)
from fastwam_ood_eval.thought5.schemas import file_sha256, object_sha256
from fastwam_ood_eval.thought5.trainer import VARIANT_FLAGS


PROBE_SEEDS = (4407, 4408, 4409)
VIDEO_FEATURE_KEY = (
    "A|mot.video_kv_cache.15.v|layer=15|denoise=none|pool=spatial_mean"
)
class RepresentationRuntimeError(RuntimeError):
    pass


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


def _clean_extrinsics(samples: Sequence[Any]) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for sample in samples:
        if sample.condition != "clean":
            continue
        pair_id = sample.plan.identity.sample_id
        values[pair_id] = np.asarray(
            sample.rendered.record.camera.extrinsic_camera_to_world,
            dtype=np.float64,
        )
    expected = {sample.plan.identity.sample_id for sample in samples}
    if set(values) != expected:
        raise RepresentationRuntimeError("a condition group lacks its Clean camera")
    return values


def _conditioning_for_sample(sample: Any, clean_extrinsic: np.ndarray) -> tuple[Any, Any]:
    import torch

    camera = sample.rendered.record.camera
    grid = two_camera_token_rays(
        camera.intrinsic,
        image_height=224,
        camera_width=224,
        token_height=7,
        tokens_per_camera_width=7,
    )
    relative = relative_clean_to_camera(
        clean_extrinsic,
        np.asarray(camera.extrinsic_camera_to_world, dtype=np.float64),
    )
    rays = torch.from_numpy(grid.rays_camera).unsqueeze(0)
    pose = torch.from_numpy(pose_embedding_12(relative)).unsqueeze(0)
    return rays, pose


def _relative_depth(depth: Any) -> Any:
    value = np.asarray(depth, dtype=np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise RepresentationRuntimeError("probe depth target is invalid")
    return value - float(np.mean(value))


def _filter_example(example: Any) -> Any | None:
    labels: dict[str, Any] = {}
    if (
        example.source == "A"
        and example.module_path == "mot.video_kv_cache.15.v"
        and example.pooling == "spatial_mean"
    ):
        labels = {
            key: example.labels[key]
            for key in (
                "eef_object_translation_camera",
                "eef_object_translation_world",
            )
        }
    elif (
        example.source == "A"
        and example.module_path == "mot.video_kv_cache.15.v"
        and example.pooling == "foreground_mean"
    ):
        labels = {
            "depth": example.labels["depth"],
            "relative_depth": _relative_depth(example.labels["depth"]),
        }
    elif (
        example.source == "B"
        and example.module_path == "action_expert.blocks.15.norm1"
    ):
        labels = {
            key: example.labels[key]
            for key in (
                "eef_object_translation_camera",
                "eef_object_translation_world",
                "action_se3_trajectory",
            )
        }
    if not labels:
        return None
    return replace(
        example,
        labels=labels,
        masks={key: value for key, value in example.masks.items() if key in labels},
    )


def collect_representation_bundle(
    cfg: Any,
    runtime: Any,
    attachment: Any | None,
    *,
    variant: str,
    samples: Sequence[Any],
    output_path: Path,
) -> dict[str, Any]:
    """Collect frozen layer/action features with per-sample camera conditioning."""

    from fastwam_ood_eval.thought4.real_runtime import extract_probe_examples
    from fastwam_ood_eval.thought5.real_runtime import _thought4_smoke_config

    if output_path.is_file():
        prior = _load_bundle(output_path)
        if (
            prior.get("variant") != variant
            or prior.get("config_fingerprint") != cfg.fingerprint
        ):
            raise RepresentationRuntimeError(
                "existing representation bundle has different provenance"
            )
        return {
            "schema_version": (
                "thought5.phase5.representation_bundle_descriptor.v1"
            ),
            "status": "complete",
            "variant": variant,
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": file_sha256(output_path),
            "example_count": len(prior["examples"]),
            "inference_count": len(prior["inference_rows"]),
            "idempotent_reuse": True,
        }
    if variant != "B0" and variant not in VARIANT_FLAGS:
        raise RepresentationRuntimeError(f"unsupported variant: {variant}")
    t4_cfg = _thought4_smoke_config(cfg)
    clean = _clean_extrinsics(samples)
    examples: list[Any] = []
    inference_rows: list[dict[str, Any]] = []
    ray_enabled = bool(
        variant != "B0" and VARIANT_FLAGS[variant]["ray_pose"]
    )
    for sample in sorted(
        samples,
        key=lambda value: (
            value.plan.identity.sample_id,
            value.condition,
        ),
    ):
        rays, pose = _conditioning_for_sample(
            sample, clean[sample.plan.identity.sample_id]
        )
        scope = (
            attachment.conditioning(
                rays=rays,
                camera_pose_12=pose,
                enable_ray_pose=ray_enabled,
            )
            if attachment is not None
            else nullcontext()
        )
        with scope:
            current, rows = extract_probe_examples(t4_cfg, runtime, [sample])
        for example in current:
            selected = _filter_example(example)
            if selected is not None:
                examples.append(selected)
        inference_rows.extend(rows)
    if not examples or not inference_rows:
        raise RepresentationRuntimeError("real feature collection is empty")
    cohort_seeds = {
        (
            int(row["task_index"]),
            int(row["episode_index"]),
            int(row["frame_index"]),
        ): int(row["seed"])
        for row in cohort_manifest(cfg.cohort)["rows"]
    }
    sample_metadata = {
        sample.plan.identity.sample_id: {
            "task_id": str(sample.plan.task_index),
            "episode_id": sample.plan.identity.episode_id,
            "split": sample.plan.identity.split,
            "cohort_seed": cohort_seeds[
                (
                    int(sample.plan.task_index),
                    int(sample.plan.episode_index),
                    int(sample.plan.frame_index),
                )
            ],
        }
        for sample in samples
    }
    bundle = {
        "schema_version": "thought5.phase5.representation_bundle.v1",
        "status": "complete",
        "variant": variant,
        "config_fingerprint": cfg.fingerprint,
        "examples": examples,
        "inference_rows": inference_rows,
        "sample_metadata": sample_metadata,
        "future_rgb_read": False,
        "success_outcome_read": False,
    }
    _atomic_pickle(output_path, bundle)
    descriptor = {
        "schema_version": "thought5.phase5.representation_bundle_descriptor.v1",
        "status": "complete",
        "variant": variant,
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": file_sha256(output_path),
        "example_count": len(examples),
        "inference_count": len(inference_rows),
    }
    return descriptor


def _load_bundle(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if value.get("status") != "complete":
        raise RepresentationRuntimeError("representation bundle is incomplete")
    return value


def _probe_kwargs(cfg: Any) -> dict[str, Any]:
    return {
        "probe_models": ("linear", "mlp"),
        "seeds": PROBE_SEEDS,
        "hidden_dim": 256,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "max_epochs": 100,
        "patience": 12,
        "batch_size": 32,
        "bootstrap_replicates": cfg.evaluation.bootstrap_replicates,
        "bootstrap_seed": cfg.evaluation.bootstrap_seed,
        "device": "cpu",
    }


def _main_records(
    *,
    variant: str,
    bundle: Mapping[str, Any],
    video_panel: Any,
) -> list[RepresentationRecord]:
    from fastwam_ood_eval.thought4.pipeline import (
        _per_sample_rmse,
        _predict,
        _stack,
    )

    examples = [
        value
        for value in bundle["examples"]
        if value.feature_key == VIDEO_FEATURE_KEY
        and "eef_object_translation_camera" in value.labels
        and value.split == "test"
    ]
    metadata = bundle["sample_metadata"]
    records: list[RepresentationRecord] = []
    for seed in PROBE_SEEDS:
        key = (VIDEO_FEATURE_KEY, "eef_object_translation_camera", seed)
        if key not in video_panel.linear_models:
            raise RepresentationRuntimeError("frozen main linear probe is absent")
        model = video_panel.linear_models[key]
        for condition in ("clean", "camera", "lighting", "robot_init"):
            selected = [value for value in examples if value.condition == condition]
            dataset = _stack(
                selected,
                "eef_object_translation_camera",
            )
            prediction = _predict(model, dataset)
            errors = _per_sample_rmse(
                prediction, dataset.targets, dataset.valid_mask
            )
            for example, error in zip(selected, errors, strict=True):
                identity = metadata[example.sample_id]
                records.append(
                    RepresentationRecord(
                        variant=variant,
                        task_id=str(identity["task_id"]),
                        episode_id=str(identity["episode_id"]),
                        seed=int(seed),
                        condition=condition,
                        endpoint="video_eef_object_translation_camera",
                        error=float(error),
                    )
                )
    return records


def evaluate_representation_bundles(
    cfg: Any,
    bundle_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Fit frozen probes and evaluate H1 without reading rollout outcomes."""

    from fastwam_ood_eval.thought4.pipeline import run_probe_panel

    required = {"B1", "G3"}
    if not required.issubset(bundle_paths):
        raise RepresentationRuntimeError("H1 requires B1 and G3 bundles")
    probe_results: dict[str, Any] = {}
    records: list[RepresentationRecord] = []
    descriptors: dict[str, Any] = {}
    for variant, path in sorted(bundle_paths.items()):
        bundle = _load_bundle(path)
        if bundle.get("variant") != variant:
            raise RepresentationRuntimeError("bundle variant mismatch")
        examples = list(bundle["examples"])
        video = run_probe_panel(
            examples, source="A", **_probe_kwargs(cfg)
        )
        action = run_probe_panel(
            examples, source="B", **_probe_kwargs(cfg)
        )
        records.extend(
            _main_records(variant=variant, bundle=bundle, video_panel=video)
        )
        probe_results[variant] = {
            "video": video.result,
            "action": action.result,
        }
        descriptors[variant] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }
    h1 = evaluate_h1(
        records,
        bootstrap_replicates=cfg.evaluation.bootstrap_replicates,
        bootstrap_seed=cfg.evaluation.bootstrap_seed,
        task_bootstrap_seed=cfg.evaluation.task_bootstrap_seed,
        clean_noninferiority_fraction=cfg.evaluation.clean_noninferiority_margin,
        g4_equivalence_fraction=cfg.evaluation.g4_equivalence_fraction,
    )
    main_summary: dict[str, Any] = {}
    for variant in sorted({row.variant for row in records}):
        condition_error = {
            condition: float(
                np.mean(
                    [
                        row.error
                        for row in records
                        if row.variant == variant
                        and row.condition == condition
                        and row.endpoint
                        == "video_eef_object_translation_camera"
                    ]
                )
            )
            for condition in ("clean", "camera", "lighting", "robot_init")
        }
        main_summary[variant] = {
            "condition_error": condition_error,
            "camera_gap": condition_error["camera"] - condition_error["clean"],
            "lighting_gap": condition_error["lighting"]
            - condition_error["clean"],
            "robot_init_gap": condition_error["robot_init"]
            - condition_error["clean"],
        }
    result = {
        **h1,
        "probe_rule": {
            "source": "mot.video_kv_cache.15.v",
            "pooling": "spatial_mean",
            "target": "eef_object_translation_camera",
            "models": ["linear", "mlp"],
            "seeds": list(PROBE_SEEDS),
            "selection_reads_formal": False,
        },
        "all_registered_probe_panels": probe_results,
        "variant_main_endpoint_summary": main_summary,
        "bundle_descriptors": descriptors,
        "record_count": len(records),
        "success_outcome_read": False,
        "result_sha256": None,
    }
    result["result_sha256"] = object_sha256(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )
    return result
