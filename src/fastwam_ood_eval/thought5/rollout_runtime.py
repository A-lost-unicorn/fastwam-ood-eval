"""Real paired Phase 5 rollout collection.

The collector deliberately reuses the project's standard LIBERO episode
runner and the official Fast-WAM action helper.  It adds only the frozen
GeoEq checkpoint and the camera metadata required by RayPoseEncoder.  No
simulator depth, geometry target, future RGB, or formal outcome is available
to the policy.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import pickle
import tempfile
import time
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


class RolloutRuntimeError(RuntimeError):
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


def _rollout_eval_config(cfg: Any, *, variant: str, output_dir: Path) -> Any:
    """Build the standard runner config without exposing its job planner."""

    from fastwam_ood_eval.config import load_config

    base = load_config("configs/eval_ood_full.yaml")
    return replace(
        base,
        experiment=replace(
            base.experiment,
            name=f"{cfg.experiment.name}_rollout_{variant.lower()}",
            output_dir=output_dir,
            seed=cfg.experiment.seed,
            overwrite=False,
            resume=True,
            save_video=cfg.evaluation.rollout_save_failure_videos,
            save_failure_video_only=True,
        ),
        hardware=replace(base.hardware, devices=(0,), precision="bf16"),
        checkpoint=replace(
            base.checkpoint,
            path=cfg.backbone.checkpoint_path,
            dataset_stats_path=cfg.backbone.dataset_stats_path,
        ),
        policy=replace(
            base.policy,
            variant="fastwam",
            test_time_future_imagination=False,
            comparison_group="thought5_paired_camera_geoeq",
            training_recipe_id=variant,
        ),
        benchmark=replace(
            base.benchmark,
            backend="libero_plus",
            suite=cfg.cohort.suite,
            suite_config=Path("configs/suites/libero_goal.yaml"),
            tasks=tuple(cfg.cohort.formal_tasks),
            episodes_per_task=cfg.cohort.formal_episodes_per_task,
            max_steps=cfg.evaluation.rollout_max_steps,
            num_steps_wait=cfg.evaluation.rollout_wait_steps,
            control_horizon=cfg.evaluation.rollout_control_horizon,
            image_size=cfg.evaluation.rollout_image_size,
        ),
        recording=replace(
            base.recording,
            save_observations=False,
            save_actions=False,
            save_robot_state=False,
        ),
    )


def _formal_samples(samples: Sequence[Any]) -> list[Any]:
    values = [
        sample for sample in samples if sample.plan.identity.split == "test"
    ]
    if not values:
        raise RolloutRuntimeError("formal rollout sample panel is empty")
    return values


def _clean_extrinsic_by_task(samples: Sequence[Any]) -> dict[int, np.ndarray]:
    values: dict[int, np.ndarray] = {}
    for sample in _formal_samples(samples):
        if sample.condition != "clean":
            continue
        task = int(sample.plan.task_index)
        current = np.asarray(
            sample.rendered.record.camera.extrinsic_camera_to_world,
            dtype=np.float64,
        )
        if task in values and not np.allclose(values[task], current, atol=1e-10):
            raise RolloutRuntimeError("Clean camera extrinsic changed within a task")
        values[task] = current
    expected = {int(sample.plan.task_index) for sample in _formal_samples(samples)}
    if set(values) != expected:
        raise RolloutRuntimeError("Clean camera reference is missing for a task")
    return values


def build_rollout_jobs(cfg: Any, samples: Sequence[Any], *, variant: str) -> list[Any]:
    """Use the frozen rendered variants and untouched cohort seeds verbatim."""

    from fastwam_ood_eval.evaluation.jobs import EvaluationJob
    from fastwam_ood_eval.thought5.future_runtime import _classification_lookup
    from fastwam_ood_eval.thought5.paired_geometry_data import (
        UPSTREAM_GOAL_ORDER,
        cohort_manifest,
    )

    selected = _formal_samples(samples)
    classification = _classification_lookup(cfg, selected)
    seed_by_identity = {
        (
            int(row["task_index"]),
            int(row["episode_index"]),
            int(row["frame_index"]),
        ): int(row["seed"])
        for row in cohort_manifest(cfg.cohort)["rows"]
        if row["split"] == "formal"
    }
    jobs: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for sample in sorted(
        selected,
        key=lambda value: (
            int(value.plan.task_index),
            int(value.plan.task_local_episode_index),
            str(value.condition),
        ),
    ):
        pair_id = str(sample.plan.identity.sample_id)
        key = (pair_id, str(sample.condition))
        if key in seen:
            raise RolloutRuntimeError("duplicate rollout pair/condition")
        seen.add(key)
        task = int(sample.plan.task_index)
        variant_name = str(sample.rendered.record.condition_variant)
        classification_id = classification.get((sample.condition, variant_name))
        if classification_id is None:
            raise RolloutRuntimeError("rollout condition classification is absent")
        cohort_seed = seed_by_identity[
            (
                task,
                int(sample.plan.episode_index),
                int(sample.plan.frame_index),
            )
        ]
        identity = {
            "phase": "thought5-rollout-v1",
            "config_fingerprint": cfg.fingerprint,
            "variant": variant,
            "pair_id": pair_id,
            "condition": sample.condition,
            "classification_id": classification_id,
            "episode_seed": cohort_seed,
        }
        job_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()[:24]
        jobs.append(
            EvaluationJob(
                experiment_id=cfg.experiment.name,
                job_id=job_id,
                suite=cfg.cohort.suite,
                task_id=task,
                task_name=UPSTREAM_GOAL_ORDER[task],
                upstream_task_id=int(classification_id) - 1,
                upstream_task_name=variant_name,
                episode_index=int(sample.plan.task_local_episode_index),
                episode_seed=cohort_seed,
                initial_state_index=int(sample.plan.task_local_episode_index),
                condition=str(sample.condition),
                perturbation_category=(
                    None if sample.condition == "clean" else str(sample.condition)
                ),
                perturbation_level=(
                    None if sample.condition == "clean" else "thought5_frozen_v1"
                ),
                perturbation_parameters={
                    "classification_id": int(classification_id),
                    "condition_variant": variant_name,
                    "paired_sample_id": pair_id,
                    "exact_state_pair": sample.condition
                    in cfg.cohort.exact_state_conditions,
                },
                policy_variant=f"fastwam_geoeq_{variant}",
                test_time_future_imagination=False,
                comparison_group="thought5_paired_camera_geoeq",
                training_recipe_id=variant,
            )
        )
    expected = (
        len(cfg.cohort.formal_tasks)
        * cfg.cohort.formal_episodes_per_task
        * len(cfg.cohort.conditions)
    )
    if len(jobs) != expected:
        raise RolloutRuntimeError(
            f"rollout plan has {len(jobs)} jobs, expected {expected}"
        )
    return jobs


class StateTrackingLiberoPlusAdapter:
    """Thin delegating wrapper that records the post-reset physical state."""

    def __init__(self, cfg: Any, output_dir: Path) -> None:
        from fastwam_ood_eval.envs.libero_plus_adapter import LiberoPlusAdapter

        self._inner = LiberoPlusAdapter(
            cfg.evaluation.rollout_image_size,
            root=Path("third_party/LIBERO-plus"),
            config_dir=output_dir / "libero_plus",
        )
        self.last_initial_state_sha256: str | None = None

    @property
    def env(self) -> Any:
        return self._inner.env

    @property
    def task_description(self) -> str:
        return self._inner.task_description

    def reset(self, job: Any) -> dict[str, Any]:
        from fastwam_ood_eval.thought4.paired_rendering import (
            get_simulator_state,
            simulator_state_sha256,
        )

        observation = self._inner.reset(job)
        self.last_initial_state_sha256 = simulator_state_sha256(
            get_simulator_state(self._inner.env)
        )
        return observation

    def step(self, action: Any) -> Any:
        return self._inner.step(action)

    def is_success(self) -> bool:
        return self._inner.is_success()

    def close(self) -> None:
        self._inner.close()


class GeoEqRolloutPolicy:
    """Official Fast-WAM policy plus the frozen, depth-free Ray/Pose input."""

    def __init__(
        self,
        cfg: Any,
        runtime: Any,
        attachment: Any | None,
        *,
        variant: str,
        environment: StateTrackingLiberoPlusAdapter,
        clean_extrinsics: Mapping[int, np.ndarray],
    ) -> None:
        self.cfg = cfg
        self.runtime = runtime
        self.attachment = attachment
        self.variant = variant
        self.environment = environment
        self.clean_extrinsics = clean_extrinsics
        self.task_index: int | None = None
        self.task_description = ""
        self._first = True

    def set_task_index(self, task_index: int) -> None:
        if task_index not in self.clean_extrinsics:
            raise RolloutRuntimeError("rollout task lacks Clean camera reference")
        self.task_index = int(task_index)

    def reset(self, task_description: str, *, seed: int | None = None) -> None:
        import torch

        self.task_description = task_description
        self.runtime.upstream_cfg.seed = seed
        self._first = True
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.cfg.runtime.device)

    def _conditioning(self) -> Any:
        import torch
        from fastwam_ood_eval.thought4.paired_rendering import camera_metadata

        if self.attachment is None:
            return nullcontext()
        if self.task_index is None:
            raise RolloutRuntimeError("rollout task context was not set")
        metadata = camera_metadata(
            self.environment.env,
            camera_name="agentview",
            height=224,
            width=224,
        )
        grid = two_camera_token_rays(
            metadata.intrinsic,
            image_height=224,
            camera_width=224,
            token_height=7,
            tokens_per_camera_width=7,
        )
        relative = relative_clean_to_camera(
            self.clean_extrinsics[self.task_index],
            np.asarray(metadata.extrinsic_camera_to_world, dtype=np.float64),
        )
        rays = torch.from_numpy(grid.rays_camera).unsqueeze(0)
        pose = torch.from_numpy(pose_embedding_12(relative)).unsqueeze(0)
        return self.attachment.conditioning(
            rays=rays,
            camera_pose_12=pose,
            enable_ray_pose=self.variant in {"G2", "G3", "G4"},
        )

    def act(self, observation: dict[str, Any]) -> Any:
        import torch
        from fastwam_ood_eval.policy.base import PolicyOutput

        started = time.perf_counter()
        with torch.inference_mode(), self._conditioning():
            actions, images, future_frames = self.runtime.official._predict_action_chunk(
                obs=observation,
                task_description=self.task_description,
                model=self.runtime.model,
                processor=self.runtime.processor,
                cfg=self.runtime.upstream_cfg,
                action_horizon=self.runtime.action_horizon,
                input_w=self.runtime.input_width,
                input_h=self.runtime.input_height,
                model_device=self.cfg.runtime.device,
            )
            if future_frames is not None:
                raise RolloutRuntimeError(
                    "Fast-WAM rollout unexpectedly generated future RGB"
                )
        torch.cuda.synchronize(self.cfg.runtime.device)
        latency = (time.perf_counter() - started) * 1000.0
        warmup = latency if self._first else None
        self._first = False
        primary = images.get("image")
        return PolicyOutput(
            actions=actions,
            latency_ms=latency,
            warmup_latency_ms=warmup,
            action_chunk_shape=list(actions.shape),
            observation_image_shape=(
                list(primary.shape) if hasattr(primary, "shape") else None
            ),
            gpu_memory_allocated_mb=torch.cuda.memory_allocated(
                self.cfg.runtime.device
            )
            / 2**20,
            gpu_memory_reserved_mb=torch.cuda.memory_reserved(
                self.cfg.runtime.device
            )
            / 2**20,
            extra={
                "gt_depth_read": False,
                "future_rgb_read": False,
                "ray_pose_enabled": self.variant in {"G2", "G3", "G4"},
            },
        )

    def peak_memory_mb(self) -> float:
        import torch

        return float(
            torch.cuda.max_memory_allocated(self.cfg.runtime.device) / 2**20
        )

    def close(self) -> None:
        return None


def _read_latest_episode_results(path: Path) -> dict[str, Any]:
    from fastwam_ood_eval.schemas.episode_result import EpisodeResult

    values: dict[str, Any] = {}
    if not path.is_file():
        return values
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            result = EpisodeResult.from_dict(json.loads(line))
        except Exception as exc:  # noqa: BLE001
            raise RolloutRuntimeError(
                f"invalid rollout JSONL line {line_number}: {exc}"
            ) from exc
        values[result.job_id] = result
    return values


def _validate_initial_state_pairing(results: Sequence[Any]) -> None:
    groups: dict[tuple[int, int], dict[str, str]] = {}
    for result in results:
        state_hash = result.extra.get("initial_state_sha256")
        if not isinstance(state_hash, str) or len(state_hash) != 64:
            raise RolloutRuntimeError("rollout initial-state hash is absent")
        groups.setdefault((int(result.task_id), int(result.episode_seed)), {})[
            result.condition
        ] = state_hash
    for identity, conditions in groups.items():
        exact = {conditions.get(name) for name in ("clean", "camera", "lighting")}
        if None in exact or len(exact) != 1:
            raise RolloutRuntimeError(
                f"rollout exact-state initial conditions differ for {identity}"
            )
        if "robot_init" not in conditions:
            raise RolloutRuntimeError("rollout Robot-init specificity row is absent")


def collect_rollout_bundle(
    cfg: Any,
    runtime: Any,
    attachment: Any | None,
    *,
    variant: str,
    samples: Sequence[Any],
    output_path: Path,
) -> dict[str, Any]:
    """Run or resume one model's frozen paired rollout panel."""

    from fastwam_ood_eval.evaluation.episode_runner import run_episode
    from fastwam_ood_eval.schemas.episode_result import append_result
    from fastwam_ood_eval.thought5.rollout_eval import RolloutRecord
    from fastwam_ood_eval.thought5.schemas import file_sha256

    if output_path.is_file():
        with output_path.open("rb") as handle:
            prior = pickle.load(handle)
        if (
            prior.get("status") != "complete"
            or prior.get("variant") != variant
            or prior.get("config_fingerprint") != cfg.fingerprint
        ):
            raise RolloutRuntimeError("existing rollout bundle provenance differs")
        return {
            "status": "complete",
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": file_sha256(output_path),
            "record_count": len(prior["records"]),
            "idempotent_reuse": True,
        }
    jobs = build_rollout_jobs(cfg, samples, variant=variant)
    worker_root = output_path.parent / "rollout_runtime"
    eval_cfg = _rollout_eval_config(cfg, variant=variant, output_dir=worker_root)
    environment = StateTrackingLiberoPlusAdapter(cfg, worker_root / "runtime")
    policy = GeoEqRolloutPolicy(
        cfg,
        runtime,
        attachment,
        variant=variant,
        environment=environment,
        clean_extrinsics=_clean_extrinsic_by_task(samples),
    )
    result_path = worker_root / "episode_results.jsonl"
    latest = _read_latest_episode_results(result_path)
    expected_job_ids = {job.job_id for job in jobs}
    if set(latest) - expected_job_ids:
        raise RolloutRuntimeError("rollout resume JSONL contains foreign jobs")
    provenance = {
        "checkpoint": str(cfg.backbone.checkpoint_path),
        "checkpoint_hash": cfg.backbone.checkpoint_sha256,
        "fastwam_commit": cfg.backbone.fastwam_commit,
    }
    completed: dict[str, Any] = {
        job_id: result
        for job_id, result in latest.items()
        if result.error is None
        and result.status == "completed"
        and result.extra.get("config_fingerprint") == cfg.fingerprint
        and result.extra.get("variant") == variant
    }
    try:
        for job in jobs:
            if job.job_id in completed:
                continue
            policy.set_task_index(job.task_id)
            result = run_episode(
                cfg=eval_cfg,
                job=job,
                policy=policy,
                environment=environment,
                worker_rank=0,
                provenance=provenance,
                worker_dir=worker_root,
            )
            result.extra.update(
                {
                    "config_fingerprint": cfg.fingerprint,
                    "variant": variant,
                    "initial_state_sha256": environment.last_initial_state_sha256,
                    "exact_state_pair": job.condition
                    in cfg.cohort.exact_state_conditions,
                    "gt_depth_read": False,
                    "future_rgb_read": False,
                    "checkpoint_selection_read_rollout": False,
                }
            )
            append_result(result_path, result)
            if result.error is not None or result.status != "completed":
                raise RolloutRuntimeError(
                    f"rollout job {job.job_id} failed: {result.error}"
                )
            completed[job.job_id] = result
    finally:
        policy.close()
        environment.close()
    ordered = [completed[job.job_id] for job in jobs]
    if len(ordered) != len(jobs):
        raise RolloutRuntimeError("rollout panel did not complete every job")
    _validate_initial_state_pairing(ordered)
    records = [
        RolloutRecord(
            variant=variant,
            task_id=str(result.task_id),
            episode_seed=int(result.episode_seed),
            condition=str(result.condition),
            success=bool(result.success),
            latency_ms=float(result.policy_latency_mean_ms),
            peak_memory_mib=float(result.gpu_peak_memory_mb),
        )
        for result in ordered
    ]
    bundle = {
        "schema_version": "thought5.phase5.rollout_bundle.v1",
        "status": "complete",
        "variant": variant,
        "config_fingerprint": cfg.fingerprint,
        "records": records,
        "job_count": len(jobs),
        "episode_result_jsonl": str(result_path),
        "exact_state_initial_pairing_verified": True,
        "paired_seed_policy": "Phase5 untouched cohort seed reused across conditions/models",
        "rollout_outcomes_read": True,
        "checkpoint_selection_read_rollout": False,
        "inference_uses_gt_depth": False,
        "test_time_future_imagination": False,
    }
    _atomic_pickle(output_path, bundle)
    gc.collect()
    return {
        "status": "complete",
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": file_sha256(output_path),
        "record_count": len(records),
    }


def evaluate_rollout_bundles(
    cfg: Any, bundle_paths: Mapping[str, Path]
) -> dict[str, Any]:
    from fastwam_ood_eval.thought5.rollout_eval import evaluate_rollouts
    from fastwam_ood_eval.thought5.schemas import file_sha256, object_sha256

    if not {"B0", "B1", "G3"}.issubset(bundle_paths):
        raise RolloutRuntimeError("rollout aggregation requires B0/B1/G3")
    records = []
    descriptors = {}
    for variant, path in sorted(bundle_paths.items()):
        with path.open("rb") as handle:
            value = pickle.load(handle)
        if (
            value.get("status") != "complete"
            or value.get("variant") != variant
            or value.get("config_fingerprint") != cfg.fingerprint
        ):
            raise RolloutRuntimeError("rollout bundle identity differs")
        records.extend(value["records"])
        descriptors[variant] = {"path": str(path), "sha256": file_sha256(path)}
    result = evaluate_rollouts(
        records,
        bootstrap_replicates=cfg.evaluation.bootstrap_replicates,
        bootstrap_seed=cfg.evaluation.bootstrap_seed + 70,
        clean_noninferiority_margin=cfg.evaluation.clean_noninferiority_margin,
        g4_equivalence_fraction=cfg.evaluation.g4_equivalence_fraction,
    )
    result.update(
        {
            "bundle_descriptors": descriptors,
            "exact_state_initial_pairing_verified": True,
            "rollout_outcomes_read": True,
            "checkpoint_selection_read_rollout": False,
            "inference_uses_gt_depth": False,
            "test_time_future_imagination": False,
        }
    )
    result["result_sha256"] = object_sha256(result)
    return result
