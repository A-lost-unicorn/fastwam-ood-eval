"""Real K=1 future collection and simulator-replay geometry targets.

The target renderer runs before Fast-WAM is loaded.  It restores each frozen
time-t simulator state, applies the same demonstration actions, and captures
the +4/+8 states corresponding to the two future latent frames.  Success is
neither queried nor used for selection.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fastwam_ood_eval.thought5.camera_rays import two_camera_token_rays
from fastwam_ood_eval.thought5.geo_targets import build_geometry_targets
from fastwam_ood_eval.thought5.paired_geometry_data import (
    cohort_manifest,
    load_condition_catalog,
)
from fastwam_ood_eval.thought5.pose_transforms import (
    pose_embedding_12,
    relative_clean_to_camera,
)


FUTURE_CONTROL_OFFSETS = (4, 8)
FUTURE_PROBE_INPUT_DIM = 3072
FUTURE_PROBE_OUTPUT_DIM = 7


class FutureRuntimeError(RuntimeError):
    pass


@dataclass
class FutureTargetFrame:
    control_offset: int
    observation: Mapping[str, Any]
    depth: Any
    camera_to_world: Any
    geometry_target: Any
    simulator_state_sha256: str


@dataclass
class FutureTargetSequence:
    sample_id: str
    condition: str
    frames: tuple[FutureTargetFrame, ...]
    future_rgb_read_for_evaluation: bool = True
    success_outcome_read: bool = False


@dataclass
class FutureAdapterEntry:
    sample_id: str
    task_id: str
    episode_id: str
    cohort_seed: int
    split: str
    condition: str
    current_latent: Any
    context: Any
    context_mask: Any
    target_action: Any
    action_is_pad: Any
    rays: Any
    pose_12: Any
    correct_future: Any
    future_mask: Any
    initial_noise_seed: int
    denoise_schedule_sha256: str


@dataclass
class FutureProbeEntry:
    """Train/dev/formal row for the test-blind Phase 5-B probe.

    The stored feature is a frozen random projection of each action-consumed
    layer-15 future token.  Keeping only this projection makes the bundle
    tractable while preserving identical probe capacity for every backbone.
    """

    sample_id: str
    task_id: str
    episode_id: str
    cohort_seed: int
    split: str
    condition: str
    projected_hidden: Any
    actual_depth_relation: Any
    actual_eef_object: Any
    actual_camera_geometry: Any
    predicted_embeddings: Any | None
    actual_embeddings: Any | None


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


def _classification_lookup(
    cfg: Any, samples: Sequence[Any]
) -> dict[tuple[str, str], int]:
    # Planned task_id is the numeric Phase5 task string.  Catalog lookup uses
    # the normalized dataset task name retained by the cohort config rows.
    manifest_by_task = {
        int(row["task_index"]): str(row["task_name"])
        for row in cohort_manifest(cfg.cohort)["rows"]
    }
    catalog = load_condition_catalog(
        cfg.cohort.classification_path,
        sorted(set(manifest_by_task.values())),
    )
    lookup: dict[tuple[str, str], int] = {}
    for task_name, by_condition in catalog.items():
        for condition, variants in by_condition.items():
            for variant in variants:
                key = (condition, variant.task_name)
                if key in lookup and lookup[key] != variant.classification_id:
                    raise FutureRuntimeError("condition task name is ambiguous")
                lookup[key] = variant.classification_id
    return lookup


def _future_job(cfg: Any, sample: Any, classification_id: int) -> Any:
    from fastwam_ood_eval.evaluation.jobs import EvaluationJob

    payload = (
        f"thought5-future-target\0{sample.plan.identity.sample_id}\0"
        f"{sample.condition}\0{classification_id}"
    )
    return EvaluationJob(
        experiment_id=cfg.experiment.name,
        job_id=hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24],
        suite=cfg.cohort.suite,
        task_id=int(sample.plan.task_index),
        task_name=str(sample.plan.task_id),
        upstream_task_id=int(classification_id) - 1,
        upstream_task_name=str(sample.rendered.record.condition_variant),
        episode_index=int(sample.plan.task_local_episode_index),
        episode_seed=int(cfg.experiment.seed + sample.plan.task_local_episode_index),
        initial_state_index=int(sample.plan.task_local_episode_index),
        condition=sample.condition,
        perturbation_category=(
            None if sample.condition == "clean" else sample.condition
        ),
        perturbation_level=(
            None if sample.condition == "clean" else "thought5_frozen_v1"
        ),
        perturbation_parameters={"classification_id": classification_id},
        policy_variant="fastwam_geoeq_target_renderer",
        test_time_future_imagination=False,
        comparison_group="thought5_future_exact_state",
    )


def render_future_target_sequences(
    cfg: Any, samples: Sequence[Any]
) -> dict[str, FutureTargetSequence]:
    """Replay +4/+8 targets for every rendered condition sample."""

    from fastwam_ood_eval.envs.libero_plus_adapter import LiberoPlusAdapter
    from fastwam_ood_eval.thought4.paired_rendering import (
        camera_metadata,
        render_rgb_depth,
        simulator_state_sha256,
    )
    from fastwam_ood_eval.thought4.real_runtime import (
        _geometry_state,
        _observation_for_state,
        load_demonstration_episode,
    )

    lookup = _classification_lookup(cfg, samples)
    adapters = {
        condition: LiberoPlusAdapter(
            image_size=(224, 224),
            root=Path("third_party/LIBERO-plus"),
            config_dir=(
                cfg.experiment.output_dir
                / "runtime"
                / "future_targets"
                / condition
            ),
        )
        for condition in cfg.cohort.conditions
    }
    results: dict[str, FutureTargetSequence] = {}
    try:
        for sample in sorted(
            samples,
            key=lambda value: (
                value.plan.identity.sample_id,
                value.condition,
            ),
        ):
            condition = str(sample.condition)
            variant_name = str(sample.rendered.record.condition_variant)
            classification_id = lookup.get((condition, variant_name))
            if classification_id is None:
                raise FutureRuntimeError(
                    f"classification ID missing for {condition}/{variant_name}"
                )
            adapter = adapters[condition]
            adapter.reset(_future_job(cfg, sample, classification_id))
            state_t = np.asarray(sample.rendered.simulator_state).copy()
            _observation_for_state(adapter, state_t)
            episode = load_demonstration_episode(
                cfg.cohort.dataset_root, sample.plan.episode_index
            )
            if sample.plan.frame_index + FUTURE_CONTROL_OFFSETS[-1] > len(
                episode.actions
            ):
                raise FutureRuntimeError("future target crosses episode boundary")
            frames: list[FutureTargetFrame] = []
            for offset in range(1, FUTURE_CONTROL_OFFSETS[-1] + 1):
                action_index = sample.plan.frame_index + offset - 1
                action = np.asarray(
                    episode.actions[action_index], dtype=np.float64
                ).copy()
                action[-1] = 1.0 - 2.0 * float(action[-1])
                observation, _reward, _done, _info = adapter.env.step(action)
                if offset not in FUTURE_CONTROL_OFFSETS:
                    continue
                rgb, depth = render_rgb_depth(
                    adapter.env,
                    camera_name="agentview",
                    height=224,
                    width=224,
                )
                camera = camera_metadata(
                    adapter.env,
                    camera_name="agentview",
                    height=224,
                    width=224,
                )
                geometry = _geometry_state(
                    adapter,
                    observation,
                    target_object_name=cfg.cohort.target_object_by_task[
                        int(sample.plan.task_index)
                    ],
                )
                grid = two_camera_token_rays(
                    camera.intrinsic,
                    image_height=224,
                    camera_width=224,
                    token_height=7,
                    tokens_per_camera_width=7,
                )
                target = build_geometry_targets(
                    depth_map=depth,
                    ray_grid=grid,
                    camera_to_world=np.asarray(
                        camera.extrinsic_camera_to_world, dtype=np.float64
                    ),
                    eef_position_world=geometry["eef_position_world"],
                    object_position_world=geometry["object_position_world"],
                )
                # Use the observation's two camera RGB streams for the same
                # official preprocessing path as the current frame.  The
                # direct render above exists solely to obtain metric depth.
                del rgb
                frames.append(
                    FutureTargetFrame(
                        control_offset=offset,
                        observation=deepcopy(dict(observation)),
                        depth=np.asarray(depth).copy(),
                        camera_to_world=np.asarray(
                            camera.extrinsic_camera_to_world,
                            dtype=np.float64,
                        ).copy(),
                        geometry_target=target,
                        simulator_state_sha256=simulator_state_sha256(
                            adapter.env.get_sim_state()
                        ),
                    )
                )
            if tuple(frame.control_offset for frame in frames) != FUTURE_CONTROL_OFFSETS:
                raise FutureRuntimeError("future target offset panel is incomplete")
            key = f"{sample.plan.identity.sample_id}:{condition}"
            if key in results:
                raise FutureRuntimeError("duplicate future target sequence")
            results[key] = FutureTargetSequence(
                sample_id=sample.plan.identity.sample_id,
                condition=condition,
                frames=tuple(frames),
            )
    finally:
        for adapter in adapters.values():
            adapter.close()
    # Exact-state conditions must remain physically paired after replaying the
    # same controls.  Robot-init is intentionally outside this assertion.
    by_sample: dict[str, dict[str, FutureTargetSequence]] = {}
    for value in results.values():
        by_sample.setdefault(value.sample_id, {})[value.condition] = value
    for sample_id, conditions in by_sample.items():
        for index, _offset in enumerate(FUTURE_CONTROL_OFFSETS):
            hashes = {
                conditions[condition].frames[index].simulator_state_sha256
                for condition in ("clean", "camera", "lighting")
            }
            if len(hashes) != 1:
                raise FutureRuntimeError(
                    f"future exact-state replay diverged for {sample_id}"
                )
    return results


def current_conditioning(sample: Any, clean_sample: Any) -> tuple[Any, Any]:
    """Return one batched ray field and Clean-to-condition pose tensor."""

    import torch

    camera = sample.rendered.record.camera
    clean_camera = clean_sample.rendered.record.camera
    grid = two_camera_token_rays(
        camera.intrinsic,
        image_height=224,
        camera_width=224,
        token_height=7,
        tokens_per_camera_width=7,
    )
    relative = relative_clean_to_camera(
        np.asarray(clean_camera.extrinsic_camera_to_world, dtype=np.float64),
        np.asarray(camera.extrinsic_camera_to_world, dtype=np.float64),
    )
    return (
        torch.from_numpy(grid.rays_camera).unsqueeze(0),
        torch.from_numpy(pose_embedding_12(relative)).unsqueeze(0),
    )


class _AttachedVideoVelocity:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.calls = 0

    def __call__(
        self, state: Any, timestep: Any, conditions: Mapping[str, object]
    ) -> Any:
        self.calls += 1
        return self.model.video_expert(
            x=state,
            timestep=timestep.to(device=state.device, dtype=state.dtype),
            context=conditions["context"],
            context_mask=conditions["context_mask"],
            action=None,
            fuse_vae_embedding_in_latents=True,
        )


def _cohort_seed_by_identity(cfg: Any) -> dict[tuple[int, int, int], int]:
    return {
        (
            int(row["task_index"]),
            int(row["episode_index"]),
            int(row["frame_index"]),
        ): int(row["seed"])
        for row in cohort_manifest(cfg.cohort)["rows"]
    }


def _future_noise_seed(cfg: Any, sample_id: str, condition: str) -> int:
    digest = hashlib.sha256(
        f"thought5-k1-noise-v1\0{cfg.experiment.seed}\0{sample_id}\0{condition}".encode(
            "utf-8"
        )
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _frame_embeddings(value: Any) -> Any:
    # [1,C,F,H,W] -> [F,C*H*W], retaining current at index zero.
    return value[0].permute(1, 0, 2, 3).detach().float().cpu().reshape(
        value.shape[2], -1
    )


def _target_arrays(sequence: FutureTargetSequence) -> dict[str, Any]:
    import torch

    depth_relation = []
    relation = []
    camera_points = []
    for frame in sequence.frames:
        target = frame.geometry_target
        depth_relation.append(torch.from_numpy(target.depth_relation).reshape(98, 1))
        relation.append(
            torch.from_numpy(
                np.repeat(
                    target.eef_object_translation.reshape(1, 3), 98, axis=0
                )
            )
        )
        camera_points.append(torch.from_numpy(target.points_camera))
    return {
        "depth_relation": torch.stack(depth_relation).float(),
        "eef_object_world": torch.stack(relation).float(),
        "point_camera": torch.stack(camera_points).float(),
    }


def _future_probe_projection(
    *, input_dim: int, output_dim: int, seed: int
) -> np.ndarray:
    """Return the preregistered signed projection shared by all variants."""

    if input_dim != FUTURE_PROBE_INPUT_DIM or output_dim <= 0:
        raise FutureRuntimeError("future probe projection dimensions changed")
    generator = np.random.default_rng(seed)
    signs = generator.integers(0, 2, size=(input_dim, output_dim), dtype=np.int8)
    return ((signs.astype(np.float32) * 2.0) - 1.0) / np.sqrt(output_dim)


def _project_future_hidden(cfg: Any, hidden: Any) -> Any:
    import torch

    if tuple(hidden.shape) != (1, 294, FUTURE_PROBE_INPUT_DIM):
        raise FutureRuntimeError(
            f"K=1 layer-15 capture shape changed: {tuple(hidden.shape)}"
        )
    projection = torch.from_numpy(
        _future_probe_projection(
            input_dim=FUTURE_PROBE_INPUT_DIM,
            output_dim=cfg.evaluation.future_probe_projection_dim,
            seed=cfg.evaluation.future_probe_projection_seed,
        )
    ).to(device=hidden.device, dtype=torch.float32)
    future = hidden[:, 98:].reshape(2, 98, FUTURE_PROBE_INPUT_DIM).float()
    return torch.matmul(future, projection).detach().cpu().contiguous()


def _probe_arrays(
    entries: Sequence[FutureProbeEntry], *, split: str
) -> tuple[np.ndarray, np.ndarray]:
    selected = [entry for entry in entries if entry.split == split]
    if not selected:
        raise FutureRuntimeError(f"future probe split is empty: {split}")
    features = []
    targets = []
    for entry in selected:
        feature = np.asarray(entry.projected_hidden, dtype=np.float32)
        depth = np.asarray(entry.actual_depth_relation, dtype=np.float32)
        relation = np.asarray(entry.actual_eef_object, dtype=np.float32)
        camera = np.asarray(entry.actual_camera_geometry, dtype=np.float32)
        if feature.ndim != 3 or feature.shape[:2] != (2, 98):
            raise FutureRuntimeError("future probe feature shape changed")
        if depth.shape != (2, 98, 1) or relation.shape != (2, 98, 3):
            raise FutureRuntimeError("future probe target shape changed")
        if camera.shape != (2, 98, 3):
            raise FutureRuntimeError("future camera target shape changed")
        features.append(feature.reshape(-1, feature.shape[-1]))
        targets.append(
            np.concatenate((depth, relation, camera), axis=-1).reshape(
                -1, FUTURE_PROBE_OUTPUT_DIM
            )
        )
    x = np.concatenate(features, axis=0)
    y = np.concatenate(targets, axis=0)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise FutureRuntimeError("future probe arrays contain NaN/Inf")
    return x, y


def _fit_future_probe(
    entries: Sequence[FutureProbeEntry], *, alphas: Sequence[float]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit train-only linear ridge and select alpha on development only."""

    x_train, y_train = _probe_arrays(entries, split="train")
    x_dev, y_dev = _probe_arrays(entries, split="development")
    train_ids = {entry.sample_id for entry in entries if entry.split == "train"}
    development_ids = {
        entry.sample_id for entry in entries if entry.split == "development"
    }
    formal_ids = {entry.sample_id for entry in entries if entry.split == "formal"}
    if train_ids & development_ids or (train_ids | development_ids) & formal_ids:
        raise FutureRuntimeError("future probe split identities overlap")
    x_mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    x_scale = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    x_scale = np.maximum(x_scale, np.float32(1e-6))
    y_mean = y_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    y_scale = y_train.std(axis=0, dtype=np.float64).astype(np.float32)
    y_scale = np.maximum(y_scale, np.float32(1e-6))
    train_x = (x_train - x_mean) / x_scale
    train_y = (y_train - y_mean) / y_scale
    dev_x = (x_dev - x_mean) / x_scale
    dev_y = (y_dev - y_mean) / y_scale
    gram = np.asarray(train_x.T @ train_x, dtype=np.float64)
    cross = np.asarray(train_x.T @ train_y, dtype=np.float64)
    identity = np.eye(gram.shape[0], dtype=np.float64)
    candidates: list[tuple[float, float, np.ndarray]] = []
    for raw_alpha in alphas:
        alpha = float(raw_alpha)
        if not np.isfinite(alpha) or alpha <= 0:
            raise FutureRuntimeError("future probe ridge alpha must be positive")
        weight = np.linalg.solve(gram + alpha * identity, cross).astype(
            np.float32
        )
        normalized_prediction = dev_x @ weight
        score = float(np.mean((normalized_prediction - dev_y) ** 2))
        candidates.append((score, alpha, weight))
    score, alpha, weight = min(candidates, key=lambda value: (value[0], value[1]))
    model = {
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
        "weight": weight,
    }
    digest = hashlib.sha256()
    for key in ("x_mean", "x_scale", "y_mean", "y_scale", "weight"):
        digest.update(np.asarray(model[key]).tobytes(order="C"))
    metadata = {
        "model": "linear_ridge",
        "projection_dim": int(x_train.shape[1]),
        "output_dim": FUTURE_PROBE_OUTPUT_DIM,
        "train_token_rows": int(x_train.shape[0]),
        "development_token_rows": int(x_dev.shape[0]),
        "formal_sample_count": len(formal_ids),
        "ridge_candidates": [float(value) for value in alphas],
        "selected_alpha": alpha,
        "development_normalized_mse": score,
        "selection_reads_formal": False,
        "probe_sha256": digest.hexdigest(),
    }
    return model, metadata


def _predict_future_probe(model: Mapping[str, Any], feature: Any) -> np.ndarray:
    value = np.asarray(feature, dtype=np.float32)
    normalized = (value - model["x_mean"]) / model["x_scale"]
    prediction = normalized @ model["weight"]
    return prediction * model["y_scale"] + model["y_mean"]


def _future_probe_records(
    *, variant: str, entries: Sequence[FutureProbeEntry], model: Mapping[str, Any]
) -> list[Any]:
    from fastwam_ood_eval.thought5.future_geometry_eval import (
        FutureGeometryRecord,
    )

    records = []
    for entry in entries:
        if entry.split != "formal":
            continue
        if entry.predicted_embeddings is None or entry.actual_embeddings is None:
            raise FutureRuntimeError("formal future embeddings are absent")
        prediction = _predict_future_probe(model, entry.projected_hidden)
        if prediction.shape != (2, 98, FUTURE_PROBE_OUTPUT_DIM):
            raise FutureRuntimeError("future probe prediction shape changed")
        records.append(
            FutureGeometryRecord(
                variant=variant,
                task_id=entry.task_id,
                episode_id=entry.episode_id,
                seed=entry.cohort_seed,
                condition=entry.condition,
                predicted_embeddings=entry.predicted_embeddings,
                actual_embeddings=entry.actual_embeddings,
                predicted_depth_relation=prediction[..., 0:1],
                actual_depth_relation=entry.actual_depth_relation,
                predicted_eef_object=prediction[..., 1:4],
                actual_eef_object=entry.actual_eef_object,
                predicted_camera_geometry=prediction[..., 4:7],
                actual_camera_geometry=entry.actual_camera_geometry,
            )
        )
    if not records:
        raise FutureRuntimeError("formal future probe records are empty")
    return records


def _prepare_current(
    cfg: Any, runtime: Any, sample: Any
) -> tuple[Any, Any, Any, Any, Any, Any]:
    import torch
    from fastwam_ood_eval.thought3.real_training import (
        preprocess_current_action_target,
    )
    from fastwam_ood_eval.thought4.real_runtime import (
        load_demonstration_episode,
    )
    from fastwam_ood_eval.thought5.real_runtime import _padded_actions

    image, _raw_proprio, _images = runtime.official._obs_to_model_input(
        dict(sample.observation),
        cfg=runtime.upstream_cfg,
        processor=runtime.processor,
        width=runtime.input_width,
        height=runtime.input_height,
        device=cfg.runtime.device,
        dtype=runtime.model.torch_dtype,
    )
    episode = load_demonstration_episode(
        cfg.cohort.dataset_root, sample.plan.episode_index
    )
    raw_action, raw_pad = _padded_actions(episode, sample.plan.frame_index)
    raw_state = torch.from_numpy(
        runtime.official._extract_sim_state(dict(sample.observation))
    )
    target_action, proprio, action_pad = preprocess_current_action_target(
        raw_action, raw_state, raw_pad, processor=runtime.processor
    )
    prompt = runtime.official.DEFAULT_PROMPT.format(task=sample.task_description)
    with torch.no_grad():
        context, context_mask = runtime.model.encode_prompt(prompt)
        context, context_mask = runtime.model._append_proprio_to_context(
            context,
            context_mask,
            proprio.to(
                device=cfg.runtime.device,
                dtype=runtime.model.torch_dtype,
            ),
        )
        current = runtime.model._encode_input_image_latents_tensor(image)
    if tuple(current.shape) != (1, 48, 1, 14, 28):
        raise FutureRuntimeError("current latent shape changed")
    return current, context, context_mask, target_action, action_pad, episode


def _encode_actual_future(
    cfg: Any, runtime: Any, sequence: FutureTargetSequence
) -> Any:
    import torch

    values = []
    with torch.no_grad():
        for frame in sequence.frames:
            image, _proprio, _images = runtime.official._obs_to_model_input(
                dict(frame.observation),
                cfg=runtime.upstream_cfg,
                processor=runtime.processor,
                width=runtime.input_width,
                height=runtime.input_height,
                device=cfg.runtime.device,
                dtype=runtime.model.torch_dtype,
            )
            latent = runtime.model._encode_input_image_latents_tensor(image)
            if tuple(latent.shape) != (1, 48, 1, 14, 28):
                raise FutureRuntimeError("actual future latent shape changed")
            values.append(latent)
    return torch.cat(values, dim=2)


def collect_future_bundle(
    cfg: Any,
    runtime: Any,
    attachment: Any,
    *,
    variant: str,
    samples: Sequence[Any],
    future_targets: Mapping[str, FutureTargetSequence],
    output_path: Path,
) -> dict[str, Any]:
    """Generate paired K=1 latents and retain adapter-training inputs."""

    import torch
    from fastwam_ood_eval.thought3.future_sampler import VideoOnlyFutureSampler
    from fastwam_ood_eval.thought5.schemas import file_sha256, object_sha256
    from fastwam_ood_eval.thought5.trainer import VARIANT_FLAGS

    if output_path.is_file():
        with output_path.open("rb") as handle:
            prior = pickle.load(handle)
        if (
            prior.get("status") != "complete"
            or prior.get("variant") != variant
            or prior.get("config_fingerprint") != cfg.fingerprint
            or prior.get("schema_version")
            != "thought5.phase5.future_bundle.v2"
        ):
            raise FutureRuntimeError("existing future bundle provenance differs")
        return {
            "status": "complete",
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": file_sha256(output_path),
            "entry_count": len(prior["adapter_entries"]),
            "future_probe_entry_count": len(prior["future_probe_entries"]),
            "geometry_record_count": sum(
                entry.split == "formal"
                for entry in prior["future_probe_entries"]
            ),
            "idempotent_reuse": True,
        }
    if variant not in {"B1", "G3", "G4"}:
        raise FutureRuntimeError("future bundle is limited to B1/G3/G4")
    by_pair: dict[str, dict[str, Any]] = {}
    for sample in samples:
        by_pair.setdefault(sample.plan.identity.sample_id, {})[
            sample.condition
        ] = sample
    cohort_seeds = _cohort_seed_by_identity(cfg)
    adapter_entries: list[FutureAdapterEntry] = []
    future_probe_entries: list[FutureProbeEntry] = []
    for sample in sorted(
        samples,
        key=lambda value: (
            value.plan.identity.sample_id,
            value.condition,
        ),
    ):
        if sample.condition not in {"clean", "camera", "lighting"}:
            continue
        pair_id = sample.plan.identity.sample_id
        cohort_seed = cohort_seeds[
            (
                int(sample.plan.task_index),
                int(sample.plan.episode_index),
                int(sample.plan.frame_index),
            )
        ]
        clean_sample = by_pair[pair_id]["clean"]
        target_key = f"{pair_id}:{sample.condition}"
        if target_key not in future_targets:
            raise FutureRuntimeError("future target sequence is absent")
        target_sequence = future_targets[target_key]
        current, context, context_mask, target_action, action_pad, _episode = (
            _prepare_current(cfg, runtime, sample)
        )
        rays, pose = current_conditioning(sample, clean_sample)
        rays = rays.to(device=cfg.runtime.device, dtype=torch.float32)
        pose = pose.to(device=cfg.runtime.device, dtype=torch.float32)
        noise_seed = _future_noise_seed(cfg, pair_id, sample.condition)
        velocity = _AttachedVideoVelocity(runtime.model)
        sampler = VideoOnlyFutureSampler(
            velocity,
            shift=5.0,
            num_train_timesteps=1000,
            rand_device="cpu",
        )
        with torch.inference_mode(), attachment.conditioning(
            rays=rays,
            camera_pose_12=pose,
            enable_ray_pose=bool(VARIANT_FLAGS[variant]["ray_pose"]),
        ):
            generated = sampler.sample(
                current,
                initial_noise_seeds=(noise_seed,),
                k=1,
                conditions={"context": context, "context_mask": context_mask},
            )
        if velocity.calls != 1 or tuple(generated.future_latent.shape) != (
            1,
            48,
            2,
            14,
            28,
        ):
            raise FutureRuntimeError("K=1 sampler contract failed")
        projected_hidden = _project_future_hidden(cfg, attachment.captured_value)
        actual_geometry = _target_arrays(target_sequence)
        split = (
            "formal"
            if sample.plan.identity.split == "test"
            else sample.plan.identity.split
        )
        predicted_embeddings = None
        actual_embeddings = None
        actual_future = None
        if split == "formal":
            actual_future = _encode_actual_future(cfg, runtime, target_sequence)
            predicted_embeddings = _frame_embeddings(
                torch.cat((current, generated.future_latent), dim=2)
            )
            actual_embeddings = _frame_embeddings(
                torch.cat((current, actual_future), dim=2)
            )
        future_probe_entries.append(
            FutureProbeEntry(
                sample_id=f"{pair_id}:{sample.condition}",
                task_id=str(sample.plan.task_index),
                episode_id=sample.plan.identity.episode_id,
                cohort_seed=cohort_seed,
                split=split,
                condition=sample.condition,
                projected_hidden=projected_hidden,
                actual_depth_relation=actual_geometry[
                    "depth_relation"
                ].contiguous(),
                actual_eef_object=actual_geometry[
                    "eef_object_world"
                ].contiguous(),
                actual_camera_geometry=actual_geometry[
                    "point_camera"
                ].contiguous(),
                predicted_embeddings=predicted_embeddings,
                actual_embeddings=actual_embeddings,
            )
        )
        adapter_entries.append(
            FutureAdapterEntry(
                sample_id=f"{pair_id}:{sample.condition}",
                task_id=str(sample.plan.task_index),
                episode_id=sample.plan.identity.episode_id,
                cohort_seed=cohort_seed,
                split=split,
                condition=sample.condition,
                current_latent=current.detach().cpu().contiguous(),
                context=context.detach().cpu().contiguous(),
                context_mask=context_mask.detach().cpu().bool().contiguous(),
                target_action=target_action.detach().cpu().contiguous(),
                action_is_pad=action_pad.detach().cpu().bool().contiguous(),
                rays=rays.detach().cpu().contiguous(),
                pose_12=pose.detach().cpu().contiguous(),
                correct_future=generated.future_latent.detach()
                .cpu()
                .contiguous(),
                future_mask=torch.ones((1, 2, 14, 28), dtype=torch.bool),
                initial_noise_seed=noise_seed,
                denoise_schedule_sha256=object_sha256(
                    generated.schedule.to_dict()
                ),
            )
        )
        del (
            current,
            context,
            context_mask,
            actual_future,
            generated,
        )
        torch.cuda.empty_cache()
    bundle = {
        "schema_version": "thought5.phase5.future_bundle.v2",
        "status": "complete",
        "variant": variant,
        "config_fingerprint": cfg.fingerprint,
        "future_k": 1,
        "future_control_offsets": list(FUTURE_CONTROL_OFFSETS),
        "paired_noise_namespace": "thought5-k1-noise-v1",
        "adapter_entries": adapter_entries,
        "future_probe_entries": future_probe_entries,
        "future_probe_protocol": {
            "model": cfg.evaluation.future_probe_model,
            "source": "mot.video_kv_cache.15.v K=1 future tokens",
            "projection_dim": cfg.evaluation.future_probe_projection_dim,
            "projection_seed": cfg.evaluation.future_probe_projection_seed,
            "ridge_alphas": list(cfg.evaluation.future_probe_ridge_alphas),
            "fit_split": "train",
            "selection_split": "development",
            "formal_read_for_selection": False,
            "same_capacity_all_backbones": True,
        },
        "actual_future_rgb_read_for_geometry_evaluation": True,
        "actual_future_rgb_read_for_adapter_training": False,
        "success_outcome_read": False,
    }
    _atomic_pickle(output_path, bundle)
    return {
        "status": "complete",
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": file_sha256(output_path),
        "entry_count": len(adapter_entries),
        "future_probe_entry_count": len(future_probe_entries),
        "geometry_record_count": sum(
            entry.split == "formal" for entry in future_probe_entries
        ),
    }


def evaluate_future_bundles(
    cfg: Any, bundle_paths: Mapping[str, Path]
) -> dict[str, Any]:
    from fastwam_ood_eval.thought5.future_geometry_eval import (
        evaluate_future_geometry,
    )
    from fastwam_ood_eval.thought5.schemas import file_sha256, object_sha256

    if not {"B1", "G3"}.issubset(bundle_paths):
        raise FutureRuntimeError("future geometry requires B1/G3 bundles")
    records: list[Any] = []
    descriptors: dict[str, Any] = {}
    probe_metadata: dict[str, Any] = {}
    split_identities: dict[str, dict[str, set[str]]] = {}
    for variant, path in sorted(bundle_paths.items()):
        with path.open("rb") as handle:
            bundle = pickle.load(handle)
        if bundle.get("status") != "complete" or bundle.get("variant") != variant:
            raise FutureRuntimeError("future bundle identity differs")
        if bundle.get("schema_version") != "thought5.phase5.future_bundle.v2":
            raise FutureRuntimeError("future bundle predates frozen probe protocol")
        entries = list(bundle["future_probe_entries"])
        split_identities[variant] = {
            split: {
                entry.sample_id for entry in entries if entry.split == split
            }
            for split in ("train", "development", "formal")
        }
        model, metadata = _fit_future_probe(
            entries,
            alphas=cfg.evaluation.future_probe_ridge_alphas,
        )
        records.extend(
            _future_probe_records(variant=variant, entries=entries, model=model)
        )
        probe_metadata[variant] = metadata
        descriptors[variant] = {"path": str(path), "sha256": file_sha256(path)}
    reference = split_identities["B1"]
    for variant, identities in split_identities.items():
        if identities != reference:
            raise FutureRuntimeError(
                f"future probe split identities are not matched: {variant}"
            )
    result = evaluate_future_geometry(
        records,
        bootstrap_replicates=cfg.evaluation.bootstrap_replicates,
        bootstrap_seed=cfg.evaluation.bootstrap_seed + 50,
        g4_equivalence_fraction=cfg.evaluation.g4_equivalence_fraction,
    )
    result.update(
        {
            "bundle_descriptors": descriptors,
            "probe_protocol": {
                "model": cfg.evaluation.future_probe_model,
                "source": "mot.video_kv_cache.15.v K=1 future tokens",
                "projection_dim": cfg.evaluation.future_probe_projection_dim,
                "projection_seed": cfg.evaluation.future_probe_projection_seed,
                "ridge_candidates": list(
                    cfg.evaluation.future_probe_ridge_alphas
                ),
                "fit_split": "train",
                "selection_split": "development",
                "formal_read_for_selection": False,
                "same_capacity_and_rule_all_backbones": True,
            },
            "probe_fits": probe_metadata,
            "actual_future_rgb_read_for_geometry_evaluation": True,
            "actual_future_rgb_read_for_adapter_training": False,
            "success_outcome_read": False,
        }
    )
    result["result_sha256"] = object_sha256(result)
    return result


def load_future_bundle(path: Path, *, variant: str, fingerprint: str) -> Any:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if (
        value.get("status") != "complete"
        or value.get("variant") != variant
        or value.get("config_fingerprint") != fingerprint
        or value.get("schema_version") != "thought5.phase5.future_bundle.v2"
    ):
        raise FutureRuntimeError("future bundle provenance differs")
    return value


def _device_entry(entry: FutureAdapterEntry, device: str, dtype: Any) -> FutureAdapterEntry:
    import torch

    def move(value: Any, *, boolean: bool = False) -> Any:
        return value.to(
            device=device,
            dtype=torch.bool if boolean else dtype,
        )

    return FutureAdapterEntry(
        sample_id=entry.sample_id,
        task_id=entry.task_id,
        episode_id=entry.episode_id,
        cohort_seed=entry.cohort_seed,
        split=entry.split,
        condition=entry.condition,
        current_latent=move(entry.current_latent),
        context=move(entry.context),
        context_mask=move(entry.context_mask, boolean=True),
        target_action=move(entry.target_action),
        action_is_pad=move(entry.action_is_pad, boolean=True),
        rays=entry.rays.to(device=device, dtype=torch.float32),
        pose_12=entry.pose_12.to(device=device, dtype=torch.float32),
        correct_future=move(entry.correct_future),
        future_mask=move(entry.future_mask, boolean=True),
        initial_noise_seed=entry.initial_noise_seed,
        denoise_schedule_sha256=entry.denoise_schedule_sha256,
    )


def _future_for_adapter(
    entry: FutureAdapterEntry,
    adapter_variant: str,
    *,
    donor: FutureAdapterEntry | None,
) -> tuple[Any, Any]:
    import torch

    if adapter_variant == "A0":
        return torch.zeros_like(entry.correct_future), entry.future_mask
    if adapter_variant == "A1":
        return entry.correct_future, entry.future_mask
    if adapter_variant == "AS":
        if donor is None or donor.sample_id == entry.sample_id:
            raise FutureRuntimeError("AS requires a true shuffled donor")
        return donor.correct_future, donor.future_mask
    raise FutureRuntimeError(f"unknown future Adapter variant: {adapter_variant}")


def _adapter_objective(
    attachment: Any,
    adapter: Any,
    injector: Any,
    entry: FutureAdapterEntry,
    *,
    adapter_variant: str,
    donor: FutureAdapterEntry | None,
    flow_slot: int,
    train_seed: int,
    ray_pose_enabled: bool,
) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from fastwam_ood_eval.thought3.phase2_protocol import (
        phase2_flow_objective_identity,
    )
    from fastwam_ood_eval.thought3.phase_c_smoke import (
        _action_from_video_cache,
        _prepare_video_cache,
        _sample_training_t_on_cpu,
        compute_upstream_action_loss,
    )

    model = attachment.model
    target = entry.target_action
    if target.ndim == 2:
        target = target.unsqueeze(0)
    padding = entry.action_is_pad
    if padding.ndim == 1:
        padding = padding.unsqueeze(0)
    identity = phase2_flow_objective_identity(
        base_sample_id=entry.sample_id,
        train_seed=train_seed,
        flow_step=flow_slot,
    )
    generator = torch.Generator(device="cpu").manual_seed(
        int(identity["action_noise_seed"])
    )
    noise = torch.randn(
        tuple(target.shape), generator=generator, dtype=torch.float32, device="cpu"
    ).to(device=target.device, dtype=target.dtype)
    timestep = _sample_training_t_on_cpu(
        model.train_action_scheduler,
        int(identity["action_timestep_seed"]),
        str(target.device),
        target.dtype,
    )
    noisy = model.train_action_scheduler.add_noise(target, noise, timestep)
    velocity = model.train_action_scheduler.training_target(target, noise, timestep)
    action_weight = model.train_action_scheduler.training_weight(timestep)
    future, future_mask = _future_for_adapter(
        entry, adapter_variant, donor=donor
    )
    with attachment.conditioning(
        rays=entry.rays,
        camera_pose_12=entry.pose_12,
        enable_ray_pose=ray_pose_enabled,
    ):
        video_cache, attention_mask, video_seq_len = _prepare_video_cache(
            model,
            entry.current_latent,
            entry.context,
            entry.context_mask,
            action_seq_len=int(target.shape[1]),
        )
        with injector.activate(future, future_mask, expected_calls=1):
            prediction = _action_from_video_cache(
                model,
                noisy,
                timestep,
                entry.context,
                entry.context_mask,
                video_cache,
                attention_mask,
                video_seq_len,
            )
    loss = compute_upstream_action_loss(
        prediction,
        velocity,
        padding,
        action_weight,
        loss_lambda_action=model.loss_lambda_action,
    )
    return loss, prediction, identity


def _ordered_clean_training_entries(
    entries: Sequence[FutureAdapterEntry], *, limit: int = 28
) -> list[FutureAdapterEntry]:
    values = sorted(
        [
            entry
            for entry in entries
            if entry.split == "train" and entry.condition == "clean"
        ],
        key=lambda entry: hashlib.sha256(
            f"thought5-adapter-train-v1\0{entry.sample_id}".encode("utf-8")
        ).hexdigest(),
    )
    if not values:
        raise FutureRuntimeError("future Adapter training set is empty")
    return values[: min(limit, len(values))]


def _donor_lookup(
    entries: Sequence[FutureAdapterEntry], *, seed: int
) -> dict[str, FutureAdapterEntry]:
    from fastwam_ood_eval.thought5.geo_targets import shuffled_target_indices

    result: dict[str, FutureAdapterEntry] = {}
    by_condition: dict[str, list[FutureAdapterEntry]] = {}
    for entry in entries:
        by_condition.setdefault(entry.condition, []).append(entry)
    for condition_index, (condition, values) in enumerate(
        sorted(by_condition.items())
    ):
        ordered = sorted(values, key=lambda entry: entry.sample_id)
        permutation = shuffled_target_indices(
            [entry.sample_id for entry in ordered],
            seed=seed + condition_index * 1009,
        )
        for entry, donor in zip(ordered, permutation, strict=True):
            selected = ordered[donor]
            if selected.condition != condition or selected.sample_id == entry.sample_id:
                raise FutureRuntimeError("AS donor derangement contract failed")
            result[entry.sample_id] = selected
    if set(result) != {entry.sample_id for entry in entries}:
        raise FutureRuntimeError("AS donor lookup is incomplete")
    return result


def calibrate_future_sample_weights(
    cfg: Any,
    attachment: Any,
    entries: Sequence[FutureAdapterEntry],
) -> dict[str, Any]:
    """Reproduce Phase2 inverse-initial-loss unit-mean weights on B1."""

    import torch
    from fastwam_ood_eval.thought3.checkpointing import adapter_state_sha256
    from fastwam_ood_eval.thought3.config import load_thought3_config
    from fastwam_ood_eval.thought3.injection import ActionEncoderFutureInjector
    from fastwam_ood_eval.thought3.phase2_protocol import (
        load_phase2_full_training_config,
    )
    from fastwam_ood_eval.thought3.real_training import build_real_adapter
    from fastwam_ood_eval.thought5.schemas import object_sha256

    thought3 = load_thought3_config(
        "configs/thought3/phase_e6_fresh_cohort_replication.yaml"
    )
    phase2 = load_phase2_full_training_config(
        "configs/thought3/phase2_full_28_4_a0_a1.yaml"
    )
    selected = _ordered_clean_training_entries(entries)
    torch.manual_seed(cfg.evaluation.thought3_adapter_seed)
    torch.cuda.manual_seed_all(cfg.evaluation.thought3_adapter_seed)
    adapter = build_real_adapter(thought3, device=cfg.runtime.device)
    initial_sha = adapter_state_sha256(adapter.state_dict())
    injector = ActionEncoderFutureInjector(
        attachment.model.action_expert.action_encoder, adapter
    )
    losses: dict[str, list[float]] = {entry.sample_id: [] for entry in selected}
    try:
        adapter.eval()
        with torch.no_grad():
            for entry_cpu in selected:
                entry = _device_entry(
                    entry_cpu, cfg.runtime.device, attachment.model.torch_dtype
                )
                for flow_slot in phase2.calibration_flow_steps:
                    loss, _prediction, _identity = _adapter_objective(
                        attachment,
                        adapter,
                        injector,
                        entry,
                        adapter_variant="A0",
                        donor=None,
                        flow_slot=int(flow_slot),
                        train_seed=cfg.evaluation.thought3_adapter_seed,
                        ray_pose_enabled=False,
                    )
                    losses[entry.sample_id].append(float(loss.detach().cpu()))
    finally:
        injector.close()
    means = {
        key: float(np.mean(values)) for key, values in losses.items()
    }
    if any(not np.isfinite(value) or value <= 0 for value in means.values()):
        raise FutureRuntimeError("future Adapter calibration loss is invalid")
    inverse = {key: 1.0 / value for key, value in means.items()}
    normalizer = float(np.mean(list(inverse.values())))
    weights = {key: value / normalizer for key, value in inverse.items()}
    result = {
        "schema_version": "thought5.phase5.future_utility_calibration.v1",
        "status": "complete",
        "source_backbone": "B1",
        "sample_ids": [entry.sample_id for entry in selected],
        "calibration_flow_steps": list(phase2.calibration_flow_steps),
        "sample_initial_mean_loss": means,
        "sample_weights": weights,
        "sample_weight_recipe": "inverse_initial_loss_unit_mean_v1",
        "initial_adapter_sha256": initial_sha,
        "future_variant": "A0",
        "ray_pose_enabled": False,
        "success_outcome_read": False,
    }
    result["calibration_sha256"] = object_sha256(result)
    return result


def _prediction_stats(prediction: Any) -> dict[str, Any]:
    from fastwam_ood_eval.thought3.future_sampler import tensor_sha256

    value = prediction.detach().float().cpu().contiguous()
    return {
        "sha256": tensor_sha256(value),
        "rms": float(value.square().mean().sqrt()),
        "translation_rms": float(value[..., :3].square().mean().sqrt()),
        "rotation_rms": float(value[..., 3:6].square().mean().sqrt()),
        "gripper_rms": float(value[..., 6:7].square().mean().sqrt()),
    }


def _action_chunk_seed(cfg: Any, sample_id: str) -> int:
    digest = hashlib.sha256(
        (
            f"thought5-h2-action-chunk-v1\0"
            f"{cfg.evaluation.thought3_adapter_seed}\0{sample_id}"
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _denoise_action_chunk(
    cfg: Any,
    attachment: Any,
    adapter: Any,
    injector: Any,
    entry: FutureAdapterEntry,
    *,
    adapter_variant: str,
    donor: FutureAdapterEntry | None,
    ray_pose_enabled: bool,
) -> tuple[Any, int, str]:
    """Run the fixed official 20-step Action DiT for Phase1-style checks."""

    import torch
    from fastwam_ood_eval.thought3.phase_c_smoke import (
        _action_from_video_cache,
        _prepare_video_cache,
    )
    from fastwam_ood_eval.thought5.schemas import object_sha256

    model = attachment.model
    action_seed = _action_chunk_seed(cfg, entry.sample_id)
    generator = torch.Generator(device="cpu").manual_seed(action_seed)
    action = torch.randn(
        (1, 32, 7), generator=generator, dtype=torch.float32, device="cpu"
    ).to(device=entry.current_latent.device, dtype=model.torch_dtype)
    future, future_mask = _future_for_adapter(
        entry, adapter_variant, donor=donor
    )
    with attachment.conditioning(
        rays=entry.rays,
        camera_pose_12=entry.pose_12,
        enable_ray_pose=ray_pose_enabled,
    ):
        video_cache, attention_mask, video_seq_len = _prepare_video_cache(
            model,
            entry.current_latent,
            entry.context,
            entry.context_mask,
            action_seq_len=32,
        )
        timesteps, deltas = model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=cfg.runtime.action_denoise_steps,
            device=cfg.runtime.device,
            dtype=action.dtype,
            shift_override=None,
        )
        with injector.activate(
            future,
            future_mask,
            expected_calls=cfg.runtime.action_denoise_steps,
        ):
            for step_t, step_delta in zip(timesteps, deltas, strict=True):
                timestep = step_t.unsqueeze(0).to(
                    device=cfg.runtime.device, dtype=action.dtype
                )
                prediction = _action_from_video_cache(
                    model,
                    action,
                    timestep,
                    entry.context,
                    entry.context_mask,
                    video_cache,
                    attention_mask,
                    video_seq_len,
                )
                action = model.infer_action_scheduler.step(
                    prediction, step_delta, action
                )
    schedule_sha = object_sha256(
        {
            "timesteps": timesteps.detach().float().cpu().tolist(),
            "deltas": deltas.detach().float().cpu().tolist(),
            "steps": cfg.runtime.action_denoise_steps,
        }
    )
    value = action[0].detach().float().cpu().contiguous()
    if tuple(value.shape) != (32, 7) or not bool(torch.isfinite(value).all()):
        raise FutureRuntimeError("H2 technical action chunk is invalid")
    return value, action_seed, schedule_sha


def _technical_action_sensitivity(
    action_chunks: Mapping[tuple[str, str], Any],
    replay_chunks: Mapping[str, Any],
    *,
    action_seeds: Mapping[str, int],
    schedule_sha256: str,
) -> dict[str, Any]:
    from fastwam_ood_eval.thought3.online_counterfactual import (
        action_pair_metrics,
        delta_direction_cosine,
    )

    sample_ids = sorted(
        sample_id
        for variant, sample_id in action_chunks
        if variant == "A1"
    )
    replay_rows = [
        action_pair_metrics(action_chunks[("A1", sample_id)], replay_chunks[sample_id])
        for sample_id in sample_ids
    ]
    if not replay_rows:
        raise FutureRuntimeError("H2 action replay panel is empty")
    replay_l2 = [float(row["l2"]) for row in replay_rows]
    replay_linf = [float(row["linf"]) for row in replay_rows]
    replay_p95 = float(np.quantile(replay_l2, 0.95))
    threshold = max(1e-7, 10.0 * replay_p95)
    hard_pass = all(row["finite"] for row in replay_rows) and max(replay_linf) <= 1e-5
    if not hard_pass:
        raise FutureRuntimeError("H2 action replay floor failed")
    rows = []
    for sample_id in sample_ids:
        null = action_chunks[("A0", sample_id)]
        correct = action_chunks[("A1", sample_id)]
        shuffle = action_chunks[("AS", sample_id)]
        correct_null = action_pair_metrics(correct, null)
        correct_shuffle = action_pair_metrics(correct, shuffle)
        rows.append(
            {
                "sample_id": sample_id,
                "action_seed": action_seeds[sample_id],
                "denoise_schedule_sha256": schedule_sha256,
                "correct_null": correct_null,
                "correct_shuffle": correct_shuffle,
                "null_shuffle": action_pair_metrics(null, shuffle),
                "correct_null_vs_correct_shuffle_delta_cosine": (
                    delta_direction_cosine(
                        correct=correct, null=null, shuffle=shuffle
                    )
                ),
                "correct_replay": replay_rows[len(rows)],
            }
        )
    return {
        "schema_version": "thought5.phase5.h2_action_sensitivity.v1",
        "status": "complete",
        "tensor_semantics": "fully_denoised_normalized_action_chunk_32x7",
        "action_denoise_steps": 20,
        "paired_action_seed": True,
        "paired_denoise_schedule": True,
        "replay_floor": {
            "rule_source": "Thought3 Phase1: max(1e-7, 10*p95 replay L2)",
            "hard_max_linf": 1e-5,
            "hard_passed": hard_pass,
            "p95_l2": replay_p95,
            "material_l2_threshold": threshold,
        },
        "correct_null_exceeds_replay_floor": sum(
            float(row["correct_null"]["l2"]) > threshold for row in rows
        ),
        "correct_shuffle_exceeds_replay_floor": sum(
            float(row["correct_shuffle"]["l2"]) > threshold for row in rows
        ),
        "sample_count": len(rows),
        "rows": rows,
        "scientific_role": (
            "technical sensitivity check only; H2 is decided by held-out action loss"
        ),
    }


def _save_future_adapter_checkpoint(
    directory: Path,
    *,
    adapter: Any,
    backbone_variant: str,
    adapter_variant: str,
    cfg: Any,
    calibration_sha256: str,
) -> dict[str, Any]:
    """Commit a small Adapter-only checkpoint with full provenance."""

    import torch
    from fastwam_ood_eval.thought3.checkpointing import adapter_state_sha256
    from fastwam_ood_eval.thought5.schemas import file_sha256, object_sha256

    state = {
        name: value.detach().cpu().contiguous()
        for name, value in adapter.state_dict().items()
    }
    semantic_sha = adapter_state_sha256(state)
    if directory.exists():
        manifest_path = directory / "manifest.json"
        weights_path = directory / "adapter_state.pt"
        if not manifest_path.is_file() or not weights_path.is_file():
            raise FutureRuntimeError("partial future Adapter checkpoint exists")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored = manifest.pop("manifest_sha256", None)
        if stored != object_sha256(manifest):
            raise FutureRuntimeError("future Adapter manifest checksum differs")
        if (
            manifest.get("adapter_state_sha256") != semantic_sha
            or manifest.get("file_sha256") != file_sha256(weights_path)
            or manifest.get("config_fingerprint") != cfg.fingerprint
            or manifest.get("backbone_variant") != backbone_variant
            or manifest.get("adapter_variant") != adapter_variant
        ):
            raise FutureRuntimeError("future Adapter checkpoint provenance differs")
        manifest["manifest_sha256"] = stored
        return manifest
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{directory.name}.", suffix=".tmp", dir=directory.parent
        )
    )
    try:
        weights_path = temporary / "adapter_state.pt"
        torch.save(state, weights_path)
        manifest = {
            "schema_version": "thought5.phase5.future_adapter_checkpoint.v1",
            "checkpoint_kind": "future_adapter_only",
            "contains_backbone": False,
            "backbone_variant": backbone_variant,
            "adapter_variant": adapter_variant,
            "global_step": cfg.evaluation.thought3_adapter_steps,
            "config_fingerprint": cfg.fingerprint,
            "backbone_checkpoint_sha256": cfg.backbone.checkpoint_sha256,
            "calibration_sha256": calibration_sha256,
            "trainable_parameter_names": [
                name for name, parameter in adapter.named_parameters()
                if parameter.requires_grad
            ],
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in adapter.parameters()
                if parameter.requires_grad
            ),
            "adapter_state_sha256": semantic_sha,
            "file_sha256": file_sha256(weights_path),
        }
        manifest["manifest_sha256"] = object_sha256(manifest)
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, directory)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest


def train_and_evaluate_future_adapters(
    cfg: Any,
    attachment: Any,
    *,
    backbone_variant: str,
    entries: Sequence[FutureAdapterEntry],
    calibration: Mapping[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Train matched A0/A1/AS adapters and evaluate expanded held-out flows."""

    import torch
    from fastwam_ood_eval.thought3.checkpointing import adapter_state_sha256
    from fastwam_ood_eval.thought3.config import load_thought3_config
    from fastwam_ood_eval.thought3.injection import ActionEncoderFutureInjector
    from fastwam_ood_eval.thought3.phase2_protocol import (
        load_phase2_full_training_config,
        phase2_training_flow_slot,
    )
    from fastwam_ood_eval.thought3.real_training import (
        adapter_gradient_groups,
        build_real_adapter,
    )
    from fastwam_ood_eval.thought5.future_utility_eval import FutureUtilityRecord
    from fastwam_ood_eval.thought5.schemas import file_sha256, object_sha256

    if output_path.is_file():
        with output_path.open("rb") as handle:
            prior = pickle.load(handle)
        if (
            prior.get("status") != "complete"
            or prior.get("backbone") != backbone_variant
            or prior.get("config_fingerprint") != cfg.fingerprint
        ):
            raise FutureRuntimeError("existing utility bundle provenance differs")
        return {
            "status": "complete",
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": file_sha256(output_path),
            "record_count": len(prior["records"]),
            "idempotent_reuse": True,
        }
    if calibration.get("status") != "complete" or calibration.get(
        "source_backbone"
    ) != "B1":
        raise FutureRuntimeError("utility calibration is not the frozen B1 result")
    if any(parameter.requires_grad for _name, parameter in attachment.named_parameters()):
        raise FutureRuntimeError(
            "GeoEq backbone/attachment must be frozen during future Adapter training"
        )
    thought3 = load_thought3_config(
        "configs/thought3/phase_e6_fresh_cohort_replication.yaml"
    )
    phase2 = load_phase2_full_training_config(
        "configs/thought3/phase2_full_28_4_a0_a1.yaml"
    )
    train = _ordered_clean_training_entries(entries)
    if [entry.sample_id for entry in train] != list(calibration["sample_ids"]):
        raise FutureRuntimeError("B1/G3 utility training identities differ")
    weights = {str(k): float(v) for k, v in calibration["sample_weights"].items()}
    formal = sorted(
        [
            entry
            for entry in entries
            if entry.split == "formal" and entry.condition in {"clean", "camera"}
        ],
        key=lambda entry: entry.sample_id,
    )
    if not formal:
        raise FutureRuntimeError("expanded held-out utility set is empty")
    train_donors = _donor_lookup(train, seed=cfg.training.shuffled_geometry_seed + 91)
    formal_donors = _donor_lookup(
        formal, seed=cfg.training.shuffled_geometry_seed + 92
    )
    schedule_sha = object_sha256(
        {
            "objective": "official_fastwam_flow_matching_velocity_mse",
            "action_denoise_steps": cfg.runtime.action_denoise_steps,
            "development_flow_steps": list(phase2.development_flow_steps),
        }
    )
    records: list[FutureUtilityRecord] = []
    track_rows: dict[str, Any] = {}
    initial_hashes: set[str] = set()
    action_chunks: dict[tuple[str, str], Any] = {}
    replay_chunks: dict[str, Any] = {}
    action_seeds: dict[str, int] = {}
    action_schedule_hashes: set[str] = set()
    ray_enabled = backbone_variant in {"G3", "G4"}
    for adapter_variant in ("A0", "A1", "AS"):
        torch.manual_seed(cfg.evaluation.thought3_adapter_seed)
        torch.cuda.manual_seed_all(cfg.evaluation.thought3_adapter_seed)
        adapter = build_real_adapter(thought3, device=cfg.runtime.device)
        initial_hash = adapter_state_sha256(adapter.state_dict())
        initial_hashes.add(initial_hash)
        optimizer = torch.optim.AdamW(
            adapter.parameters(),
            lr=cfg.evaluation.thought3_adapter_lr,
            weight_decay=phase2.weight_decay,
        )
        injector = ActionEncoderFutureInjector(
            attachment.model.action_expert.action_encoder, adapter
        )
        update_rows: list[dict[str, Any]] = []
        try:
            adapter.train()
            for update in range(1, cfg.evaluation.thought3_adapter_steps + 1):
                optimizer.zero_grad(set_to_none=True)
                losses: list[float] = []
                for micro, entry_cpu in enumerate(train, start=1):
                    entry = _device_entry(
                        entry_cpu,
                        cfg.runtime.device,
                        attachment.model.torch_dtype,
                    )
                    donor = (
                        _device_entry(
                            train_donors[entry_cpu.sample_id],
                            cfg.runtime.device,
                            attachment.model.torch_dtype,
                        )
                        if adapter_variant == "AS"
                        else None
                    )
                    slot = phase2_training_flow_slot(
                        update,
                        micro,
                        optimizer_updates=cfg.evaluation.thought3_adapter_steps,
                        objectives_per_update=len(train),
                        flow_slot_offset=phase2.training_flow_slot_offset,
                    )
                    loss, _prediction, _identity = _adapter_objective(
                        attachment,
                        adapter,
                        injector,
                        entry,
                        adapter_variant=adapter_variant,
                        donor=donor,
                        flow_slot=slot,
                        train_seed=cfg.evaluation.thought3_adapter_seed,
                        ray_pose_enabled=ray_enabled,
                    )
                    if not bool(torch.isfinite(loss)):
                        raise FutureRuntimeError("future Adapter loss is non-finite")
                    (loss * weights[entry_cpu.sample_id] / len(train)).backward()
                    losses.append(float(loss.detach().cpu()))
                groups = adapter_gradient_groups(adapter)
                if not all(bool(value["finite"]) for value in groups.values()):
                    raise FutureRuntimeError("future Adapter gradient is non-finite")
                if update == 1 and (
                    float(groups["gate"]["l2"]) <= 0
                    or int(groups["non_gate"]["nonzero_element_count"]) != 0
                ):
                    raise FutureRuntimeError("first Adapter update is not gate-only")
                if update == 2 and any(
                    int(groups[name]["nonzero_element_count"]) <= 0
                    for name in ("future_projector", "attention")
                ):
                    raise FutureRuntimeError("second Adapter update did not open paths")
                optimizer.step()
                if update <= 2 or update % 50 == 0:
                    update_rows.append(
                        {
                            "update": update,
                            "mean_loss": float(np.mean(losses)),
                            "gradient_groups": groups,
                        }
                    )
            adapter.eval()
            with torch.no_grad():
                for entry_cpu in formal:
                    entry = _device_entry(
                        entry_cpu,
                        cfg.runtime.device,
                        attachment.model.torch_dtype,
                    )
                    donor = (
                        _device_entry(
                            formal_donors[entry_cpu.sample_id],
                            cfg.runtime.device,
                            attachment.model.torch_dtype,
                        )
                        if adapter_variant == "AS"
                        else None
                    )
                    for flow_slot in phase2.development_flow_steps:
                        loss, prediction, identity = _adapter_objective(
                            attachment,
                            adapter,
                            injector,
                            entry,
                            adapter_variant=adapter_variant,
                            donor=donor,
                            flow_slot=int(flow_slot),
                            train_seed=cfg.evaluation.thought3_adapter_seed,
                            ray_pose_enabled=ray_enabled,
                        )
                        stats = _prediction_stats(prediction)
                        records.append(
                            FutureUtilityRecord(
                                backbone=backbone_variant,
                                adapter_variant=adapter_variant,
                                task_id=entry.task_id,
                                episode_id=entry.episode_id,
                                condition=entry.condition,
                                flow_slot=int(flow_slot),
                                action_noise_seed=int(identity["action_noise_seed"]),
                                action_timestep_seed=int(
                                    identity["action_timestep_seed"]
                                ),
                                denoise_schedule_sha256=schedule_sha,
                                loss=float(loss.detach().cpu()),
                                action_sha256=str(stats["sha256"]),
                                action_rms=float(stats["rms"]),
                                translation_rms=float(stats["translation_rms"]),
                                rotation_rms=float(stats["rotation_rms"]),
                                gripper_rms=float(stats["gripper_rms"]),
                            )
                        )
                    chunk, action_seed, action_schedule_sha = _denoise_action_chunk(
                        cfg,
                        attachment,
                        adapter,
                        injector,
                        entry,
                        adapter_variant=adapter_variant,
                        donor=donor,
                        ray_pose_enabled=ray_enabled,
                    )
                    action_chunks[(adapter_variant, entry_cpu.sample_id)] = chunk
                    action_seeds[entry_cpu.sample_id] = action_seed
                    action_schedule_hashes.add(action_schedule_sha)
                    if adapter_variant == "A1":
                        replay, replay_seed, replay_schedule_sha = (
                            _denoise_action_chunk(
                                cfg,
                                attachment,
                                adapter,
                                injector,
                                entry,
                                adapter_variant=adapter_variant,
                                donor=None,
                                ray_pose_enabled=ray_enabled,
                            )
                        )
                        if (
                            replay_seed != action_seed
                            or replay_schedule_sha != action_schedule_sha
                        ):
                            raise FutureRuntimeError(
                                "H2 action replay identity changed"
                            )
                        replay_chunks[entry_cpu.sample_id] = replay
            checkpoint = _save_future_adapter_checkpoint(
                output_path.parent
                / "adapter_checkpoints"
                / adapter_variant.lower(),
                adapter=adapter,
                backbone_variant=backbone_variant,
                adapter_variant=adapter_variant,
                cfg=cfg,
                calibration_sha256=str(calibration["calibration_sha256"]),
            )
            track_rows[adapter_variant] = {
                "initial_adapter_sha256": initial_hash,
                "final_adapter_sha256": adapter_state_sha256(
                    adapter.state_dict()
                ),
                "fixed_checkpoint_step": cfg.evaluation.thought3_adapter_steps,
                "checkpoint": str(
                    output_path.parent
                    / "adapter_checkpoints"
                    / adapter_variant.lower()
                ),
                "checkpoint_manifest_sha256": checkpoint["manifest_sha256"],
                "update_rows": update_rows,
            }
        finally:
            injector.close()
            del optimizer, adapter
            torch.cuda.empty_cache()
    if len(initial_hashes) != 1:
        raise FutureRuntimeError("A0/A1/AS initial Adapter states differ")
    if len(action_schedule_hashes) != 1:
        raise FutureRuntimeError("A0/A1/AS action denoise schedules differ")
    technical_sensitivity = _technical_action_sensitivity(
        action_chunks,
        replay_chunks,
        action_seeds=action_seeds,
        schedule_sha256=next(iter(action_schedule_hashes)),
    )
    bundle = {
        "schema_version": "thought5.phase5.future_utility_bundle.v1",
        "status": "complete",
        "backbone": backbone_variant,
        "config_fingerprint": cfg.fingerprint,
        "calibration_sha256": calibration["calibration_sha256"],
        "adapter_recipe": {
            "source": "Thought3 Phase2 full 28/4",
            "optimizer": "AdamW",
            "steps": cfg.evaluation.thought3_adapter_steps,
            "learning_rate": cfg.evaluation.thought3_adapter_lr,
            "weight_decay": phase2.weight_decay,
            "train_seed": cfg.evaluation.thought3_adapter_seed,
            "training_sample_count": len(train),
            "sample_weight_recipe": calibration["sample_weight_recipe"],
            "checkpoint_rule": "fixed_step_200_no_selection_no_fallback",
        },
        "tracks": track_rows,
        "technical_action_sensitivity": technical_sensitivity,
        "records": records,
        "formal_sample_count": len(formal),
        "formal_flow_steps": list(phase2.development_flow_steps),
        "actual_future_rgb_read": False,
        "success_outcome_read": False,
    }
    _atomic_pickle(output_path, bundle)
    return {
        "status": "complete",
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": file_sha256(output_path),
        "record_count": len(records),
    }


def evaluate_utility_bundles(
    cfg: Any, bundle_paths: Mapping[str, Path]
) -> dict[str, Any]:
    from fastwam_ood_eval.thought5.future_utility_eval import (
        evaluate_future_utility,
    )
    from fastwam_ood_eval.thought5.schemas import file_sha256, object_sha256

    if not {"B1", "G3"}.issubset(bundle_paths):
        raise FutureRuntimeError("future utility requires B1/G3 bundles")
    records = []
    descriptors = {}
    technical: dict[str, Any] = {}
    for variant, path in sorted(bundle_paths.items()):
        with path.open("rb") as handle:
            value = pickle.load(handle)
        if value.get("status") != "complete" or value.get("backbone") != variant:
            raise FutureRuntimeError("utility bundle identity differs")
        records.extend(value["records"])
        descriptors[variant] = {"path": str(path), "sha256": file_sha256(path)}
        checks = value.get("technical_action_sensitivity")
        if not isinstance(checks, Mapping) or checks.get("status") != "complete":
            raise FutureRuntimeError("technical action-sensitivity panel is absent")
        if not bool(checks["replay_floor"]["hard_passed"]):
            raise FutureRuntimeError("technical action replay floor did not pass")
        technical[variant] = checks
    reference_ids = {
        row["sample_id"] for row in technical["B1"]["rows"]
    }
    for variant, checks in technical.items():
        if {row["sample_id"] for row in checks["rows"]} != reference_ids:
            raise FutureRuntimeError(
                f"{variant} technical action panel is not identity-matched"
            )
    result = evaluate_future_utility(
        records,
        bootstrap_replicates=cfg.evaluation.bootstrap_replicates,
        bootstrap_seed=cfg.evaluation.bootstrap_seed + 60,
        g4_equivalence_fraction=cfg.evaluation.g4_equivalence_fraction,
    )
    result["bundle_descriptors"] = descriptors
    result["technical_action_sensitivity"] = technical
    result["actual_future_rgb_read"] = False
    result["success_outcome_read"] = False
    result["result_sha256"] = object_sha256(result)
    return result
