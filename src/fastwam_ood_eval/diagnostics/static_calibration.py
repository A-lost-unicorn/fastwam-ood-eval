"""Independent no-op/static calibration for Thought 2 motion metrics.

This protocol never samples a policy action and never reads diagnostic success
labels.  It estimates the operational null distribution of the approximate
frame-wise VAE embedding used by :mod:`fastwam_ood_eval.diagnostics.metrics`:

* repeated encodes of one identical model-space frame measure encoder noise;
* standard LIBERO no-op steps measure residual simulator/render motion;
* the full-horizon threshold candidate is the larger pre-registered quantile.

Raw samples live in a namespace that is deliberately incompatible with both
Thought 1 episode results and Thought 2 future-diagnostic records.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from fastwam_ood_eval.config import EvalConfig
from fastwam_ood_eval.diagnostics.artifact_writer import (
    _append_jsonl,
    _atomic_json_write,
    _to_rgb_uint8,
    action_chunk_hash,
    load_all_completed_jobs,
)
from fastwam_ood_eval.diagnostics.future_probe import (
    APPROXIMATE_REENCODED_EMBEDDING,
)
from fastwam_ood_eval.diagnostics.rng_isolation import RngIsolation
from fastwam_ood_eval.envs.base import BaseBenchmarkEnv
from fastwam_ood_eval.evaluation.jobs import (
    EvaluationJob,
    plan_jobs,
    shard_jobs,
    write_jobs,
)
from fastwam_ood_eval.evaluation.success_checker import is_episode_success
from fastwam_ood_eval.policy.base import BasePolicy

LOGGER = logging.getLogger(__name__)

STATIC_CALIBRATION_SCHEMA = "thought2-static-null-calibration-v1"
STANDARD_NOOP_ACTION = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
CALIBRATION_SEED_OFFSET = 2_000_000


class SupportsStaticCalibrationEncoder(Protocol):
    """Small surface shared by the real Fast-WAM probe and CPU test doubles."""

    def observation_to_model_frame(self, observation: dict[str, Any]) -> Any:
        """Convert an observation through the official model preprocessing."""

    def encode_frame_embeddings(self, frames: Sequence[Any]) -> Any:
        """Encode frames independently with the frozen first-frame VAE path."""


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if "bfloat16" in str(getattr(value, "dtype", "")) and hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "tolist") and callable(value.tolist):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def resolved_settle_steps(cfg: EvalConfig) -> int:
    value = cfg.static_calibration.settle_steps
    return cfg.benchmark.num_steps_wait if value is None else int(value)


def _sha256_file_if_present(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    except OSError:
        return None


def _implementation_hashes() -> dict[str, str | None]:
    root = Path(__file__).resolve().parents[1]
    relative = (
        "config.py",
        "diagnostics/static_calibration.py",
        "diagnostics/future_probe.py",
        "diagnostics/rng_isolation.py",
        "policy/fastwam_future_probe.py",
    )
    return {
        name: _sha256_file_if_present(root / name)
        for name in relative
    }


def calibration_compatibility_payload(
    cfg: EvalConfig,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return semantics that must match before calibration cohorts are pooled."""

    prov = dict(provenance or {})
    calibration = _jsonable(cfg.static_calibration)
    calibration["settle_steps"] = resolved_settle_steps(cfg)
    return {
        "schema": STATIC_CALIBRATION_SCHEMA,
        "calibration": calibration,
        "standard_noop_action": list(STANDARD_NOOP_ACTION),
        "standard_noop_action_sha256": action_chunk_hash(STANDARD_NOOP_ACTION),
        "frame_embedding_semantics": APPROXIMATE_REENCODED_EMBEDDING,
        "checkpoint": {
            "path": str(cfg.checkpoint.path) if cfg.checkpoint.path is not None else None,
            "model_name": cfg.checkpoint.model_name,
            "config_path": (
                str(cfg.checkpoint.config_path)
                if cfg.checkpoint.config_path is not None
                else None
            ),
            "config_sha256": (
                _sha256_file_if_present(cfg.checkpoint.config_path)
                if cfg.checkpoint.config_path is not None
                else None
            ),
            "dataset_stats_path": (
                str(cfg.checkpoint.dataset_stats_path)
                if cfg.checkpoint.dataset_stats_path is not None
                else None
            ),
            "dataset_stats_sha256": (
                _sha256_file_if_present(cfg.checkpoint.dataset_stats_path)
                if cfg.checkpoint.dataset_stats_path is not None
                else None
            ),
            "checkpoint_sha256": prov.get("checkpoint_hash"),
        },
        "policy": _jsonable(cfg.policy),
        "model_input": {
            "suite": cfg.benchmark.suite,
            "image_size": list(cfg.benchmark.image_size),
        },
        "upstream": {
            "fastwam_commit": prov.get("fastwam_commit"),
            "fastwam_dirty": prov.get("fastwam_dirty"),
        },
        "implementation_files_sha256": _implementation_hashes(),
    }


def calibration_compatibility_fingerprint(
    cfg: EvalConfig,
    provenance: Mapping[str, Any] | None = None,
) -> str:
    payload = calibration_compatibility_payload(cfg, provenance)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def static_calibration_protocol_fingerprint(
    cfg: EvalConfig,
    jobs: Sequence[EvaluationJob],
    provenance: Mapping[str, Any] | None = None,
) -> str:
    """Fingerprint both shared calibration semantics and this exact cohort."""

    job_rows = [job.to_dict() for job in jobs]
    payload = {
        "schema": STATIC_CALIBRATION_SCHEMA,
        "compatibility": calibration_compatibility_payload(cfg, provenance),
        "cohort": {
            "experiment_name": cfg.experiment.name,
            "experiment_seed": cfg.experiment.seed,
            "benchmark": {
                "backend": cfg.benchmark.backend,
                "suite": cfg.benchmark.suite,
                "tasks": (
                    list(cfg.benchmark.tasks)
                    if cfg.benchmark.tasks is not None
                    else "all"
                ),
                "episodes_per_task": cfg.benchmark.episodes_per_task,
                "max_steps": cfg.benchmark.max_steps,
            },
            "perturbation": _jsonable(cfg.perturbation),
            "job_manifest_sha256": hashlib.sha256(
                json.dumps(
                    job_rows,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        },
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def static_calibration_id(job_id: str, protocol_fingerprint: str) -> str:
    return hashlib.sha256(
        f"{job_id}\x1f{protocol_fingerprint}".encode("utf-8")
    ).hexdigest()[:24]


def preflight_static_calibration_output(output_dir: Path) -> None:
    """Reject any existing Thought 1 or future-diagnostic output namespace."""

    output = Path(output_dir)
    conflicts: list[Path] = []
    resolved = output.resolve()
    for ancestor in (resolved, *resolved.parents):
        for name in ("experiment_manifest.json", "diagnostic_manifest.json"):
            path = ancestor / name
            if path.is_file():
                conflicts.append(path)
        if ancestor == Path.cwd().resolve():
            break
    for name in ("experiment_manifest.json", "diagnostic_manifest.json"):
        path = output / name
        if path.is_file():
            conflicts.append(path)
    conflicts.extend(output.glob("workers/rank_*/episode_results.jsonl"))
    conflicts.extend(output.glob("workers/rank_*/diagnostics.jsonl"))
    if output.is_dir():
        conflicts.extend(output.rglob("experiment_manifest.json"))
        conflicts.extend(output.rglob("diagnostic_manifest.json"))
    if conflicts:
        raise RuntimeError(
            "Refusing to use a Thought 1/future-diagnostic output as a static "
            f"calibration output: {output}; conflict={conflicts[0]}"
        )


class StaticCalibrationWriter:
    """Durable per-rank calibration sample and completion writer."""

    def __init__(self, output_dir: Path, worker_rank: int = 0) -> None:
        self.output_dir = Path(output_dir)
        self.worker_rank = int(worker_rank)
        self.worker_dir = self.output_dir / "workers" / f"rank_{self.worker_rank}"
        self.samples_path = self.worker_dir / "static_calibration_samples.jsonl"
        self.completed_jobs_path = self.worker_dir / "completed_jobs.jsonl"
        self.artifact_root = self.worker_dir / "artifacts"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "calibration_manifest.json"

    def prepare_manifest(
        self,
        *,
        cfg: EvalConfig,
        jobs: Sequence[EvaluationJob],
        protocol_fingerprint: str,
        compatibility_fingerprint: str,
        provenance: Mapping[str, Any],
        write: bool,
    ) -> dict[str, Any]:
        preflight_static_calibration_output(self.output_dir)
        existing: dict[str, Any] | None = None
        if self.manifest_path.is_file():
            try:
                value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid static calibration manifest: {self.manifest_path}"
                ) from exc
            if not isinstance(value, dict):
                raise RuntimeError(
                    f"Invalid static calibration manifest object: {self.manifest_path}"
                )
            existing = value
            previous = str(existing.get("protocol_fingerprint", ""))
            if (
                previous != protocol_fingerprint
                and cfg.experiment.resume
                and not cfg.experiment.overwrite
            ):
                raise RuntimeError(
                    "Static calibration protocol changed while resume is enabled; "
                    "choose a fresh output directory or explicitly set overwrite=true. "
                    f"previous={previous or 'missing'}, current={protocol_fingerprint}"
                )
            if previous == protocol_fingerprint:
                return existing

        payload = {
            "schema_version": 1,
            "kind": "static_motion_calibration",
            "schema": STATIC_CALIBRATION_SCHEMA,
            "experiment_id": cfg.experiment.name,
            "protocol_fingerprint": protocol_fingerprint,
            "compatibility_fingerprint": compatibility_fingerprint,
            "calibration_compatibility": calibration_compatibility_payload(
                cfg, provenance
            ),
            "planned_job_count": len(jobs),
            "job_manifest": str(
                self.output_dir / "static_calibration_job_manifest.jsonl"
            ),
            "config_source": str(cfg.source_path),
            "config": cfg.to_dict(),
            "provenance": _jsonable(provenance),
            "status": "calibration_worker_outputs_pending",
        }
        if write:
            _atomic_json_write(self.manifest_path, payload)
            write_jobs(
                self.output_dir / "static_calibration_job_manifest.jsonl",
                jobs,
            )
        return payload

    def append_sample(self, sample: Mapping[str, Any]) -> None:
        payload = dict(_jsonable(sample))
        payload.setdefault("recorded_at_ns", time.time_ns())
        _append_jsonl(self.samples_path, payload)

    def mark_job_complete(
        self,
        *,
        job_id: str,
        protocol_fingerprint: str,
        status: str,
        sample_id: str,
        error: str | None,
        attempt_id: str,
        attempt_started_ns: int,
    ) -> None:
        if status not in {"completed", "skipped", "error", "exception"}:
            raise ValueError(f"Unsupported static calibration status: {status}")
        _append_jsonl(
            self.completed_jobs_path,
            {
                "job_id": job_id,
                "sample_id": sample_id,
                "protocol_fingerprint": protocol_fingerprint,
                "status": status,
                "error": error,
                "attempt_id": attempt_id,
                "attempt_started_ns": attempt_started_ns,
                "recorded_at_ns": time.time_ns(),
            },
        )

    def write_frames(
        self,
        *,
        job_id: str,
        frames_by_offset: Mapping[int, Any],
    ) -> dict[str, str]:
        from PIL import Image

        paths: dict[str, str] = {}
        root = self.artifact_root / job_id
        for offset, frame in sorted(frames_by_offset.items()):
            path = root / f"offset_{int(offset):03d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.stem + ".tmp" + path.suffix)
            Image.fromarray(_to_rgb_uint8(frame)).save(temporary)
            temporary.replace(path)
            paths[str(offset)] = str(
                path.resolve().relative_to(self.output_dir.resolve())
            )
        return paths


def _finite_array(value: Any, *, name: str) -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - real runtime provides NumPy
        raise RuntimeError("Static calibration requires numpy") from exc

    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if "bfloat16" in str(getattr(value, "dtype", "")) and hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty frame sequence")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _pair_motion_energy(left: Any, right: Any) -> float:
    import numpy as np

    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        raise ValueError(
            f"Embedding shape mismatch: {left_array.shape} versus {right_array.shape}"
        )
    return float(np.mean(np.abs(right_array - left_array)))


def _pixel_mae(left: Any, right: Any) -> float:
    import numpy as np

    left_array = np.asarray(_to_rgb_uint8(left), dtype=np.float64)
    right_array = np.asarray(_to_rgb_uint8(right), dtype=np.float64)
    if left_array.shape != right_array.shape:
        raise ValueError(
            f"Frame shape mismatch: {left_array.shape} versus {right_array.shape}"
        )
    return float(np.mean(np.abs(right_array - left_array)) / 255.0)


def compute_static_calibration_metrics(
    repeated_same_frame_embeddings: Any,
    trajectory_embeddings: Any,
    *,
    capture_offsets: Sequence[int],
    frames_by_offset: Mapping[int, Any] | None = None,
) -> dict[str, Any]:
    """Compute per-sample null energies without choosing a threshold."""

    repeated = _finite_array(
        repeated_same_frame_embeddings,
        name="repeated_same_frame_embeddings",
    )
    trajectory = _finite_array(
        trajectory_embeddings,
        name="trajectory_embeddings",
    )
    offsets = tuple(int(value) for value in capture_offsets)
    if repeated.shape[0] < 2:
        raise ValueError("At least two repeated same-frame embeddings are required")
    if trajectory.shape[0] != len(offsets):
        raise ValueError(
            "Trajectory embedding count must match capture_offsets: "
            f"{trajectory.shape[0]} versus {len(offsets)}"
        )
    if repeated.shape[1:] != trajectory.shape[1:]:
        raise ValueError(
            "Repeated and trajectory embedding shapes differ: "
            f"{repeated.shape[1:]} versus {trajectory.shape[1:]}"
        )

    pairwise = [
        _pair_motion_energy(repeated[left], repeated[right])
        for left in range(repeated.shape[0])
        for right in range(left + 1, repeated.shape[0])
    ]
    by_offset = {
        str(offset): _pair_motion_energy(trajectory[0], trajectory[index])
        for index, offset in enumerate(offsets)
        if index > 0
    }
    pixel_by_offset: dict[str, float] = {}
    if frames_by_offset is not None:
        current = frames_by_offset[offsets[0]]
        pixel_by_offset = {
            str(offset): _pixel_mae(current, frames_by_offset[offset])
            for offset in offsets[1:]
        }
    full_offset = str(offsets[-1])
    return {
        "same_frame_pairwise_motion_energy": pairwise,
        "same_frame_max_motion_energy": max(pairwise),
        "noop_motion_energy_by_offset": by_offset,
        "noop_full_horizon_motion_energy": by_offset[full_offset],
        "noop_full_horizon_offset": offsets[-1],
        "pixel_mae_by_offset": pixel_by_offset,
        "pixel_full_horizon_mae": pixel_by_offset.get(full_offset),
        "repeated_embedding_shape": list(repeated.shape),
        "trajectory_embedding_shape": list(trajectory.shape),
        "motion_energy_unit": (
            "mean absolute difference in independently re-encoded first-frame "
            "VAE embedding units"
        ),
    }


def validate_static_calibration_encoder(
    cfg: EvalConfig,
    encoder: SupportsStaticCalibrationEncoder,
) -> None:
    for name in ("observation_to_model_frame", "encode_frame_embeddings"):
        if not callable(getattr(encoder, name, None)):
            raise RuntimeError(f"Static calibration encoder lacks {name}()")
    validator = getattr(encoder, "validate_capability", None)
    if callable(validator):
        validator()
    if cfg.benchmark.backend != "mock":
        verification = getattr(encoder, "checkpoint_verification", None)
        if not isinstance(verification, Mapping) or (
            verification.get("unconditional_video_architecture_verified") is not True
        ):
            raise RuntimeError(
                "Real static calibration requires the release-compatible "
                "unconditional Fast-WAM checkpoint verification"
            )


def _runtime_control_frequency(environment: BaseBenchmarkEnv) -> float | None:
    values: list[float] = []
    adapter_env = getattr(environment, "env", None)
    for candidate in (
        environment,
        adapter_env,
        getattr(adapter_env, "env", None),
    ):
        value = getattr(candidate, "control_freq", None)
        try:
            frequency = float(value)
        except (TypeError, ValueError):
            continue
        if frequency > 0 and frequency not in values:
            values.append(frequency)
    if len(values) > 1 and any(
        not math.isclose(values[0], item) for item in values[1:]
    ):
        raise RuntimeError(
            f"Conflicting runtime environment control frequencies: {values}"
        )
    return values[0] if values else None


def _base_sample(
    *,
    cfg: EvalConfig,
    job: EvaluationJob,
    worker_rank: int,
    protocol_fingerprint: str,
    compatibility_fingerprint: str,
    provenance: Mapping[str, Any],
    sample_id: str,
    attempt_id: str,
    attempt_started_ns: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "static_motion_calibration_sample",
        "sample_id": sample_id,
        "experiment_id": cfg.experiment.name,
        "job_id": job.job_id,
        "worker_rank": worker_rank,
        "protocol_fingerprint": protocol_fingerprint,
        "compatibility_fingerprint": compatibility_fingerprint,
        "suite": job.suite,
        "task_id": job.task_id,
        "task_name": job.task_name,
        "upstream_task_id": job.upstream_task_id,
        "upstream_task_name": job.upstream_task_name,
        "episode_index": job.episode_index,
        "episode_seed": job.episode_seed,
        "initial_state_index": job.initial_state_index,
        "condition": job.condition,
        "perturbation_category": job.perturbation_category,
        "perturbation_level": job.perturbation_level,
        "perturbation_parameters": _jsonable(job.perturbation_parameters),
        "policy_action_sampled": False,
        "standard_noop_action": list(STANDARD_NOOP_ACTION),
        "standard_noop_action_sha256": action_chunk_hash(STANDARD_NOOP_ACTION),
        "settle_steps": resolved_settle_steps(cfg),
        "capture_offsets": list(cfg.static_calibration.capture_offsets),
        "repeated_same_frame_encodes": (
            cfg.static_calibration.repeated_same_frame_encodes
        ),
        "frame_embedding_semantics": APPROXIMATE_REENCODED_EMBEDDING,
        "provenance": _jsonable(provenance),
        "attempt_id": attempt_id,
        "attempt_started_ns": attempt_started_ns,
    }


def run_static_calibration_episode(
    *,
    cfg: EvalConfig,
    job: EvaluationJob,
    policy: BasePolicy,
    environment: BaseBenchmarkEnv,
    encoder: SupportsStaticCalibrationEncoder,
    writer: StaticCalibrationWriter,
    worker_rank: int,
    provenance: Mapping[str, Any],
    protocol_fingerprint: str,
    compatibility_fingerprint: str,
) -> dict[str, Any]:
    """Collect exactly one durable calibration record for an assigned job."""

    attempt_started_ns = time.time_ns()
    attempt_id = hashlib.sha256(
        f"{job.job_id}:{worker_rank}:{attempt_started_ns}".encode("utf-8")
    ).hexdigest()[:24]
    sample_id = static_calibration_id(job.job_id, protocol_fingerprint)
    sample = _base_sample(
        cfg=cfg,
        job=job,
        worker_rank=worker_rank,
        protocol_fingerprint=protocol_fingerprint,
        compatibility_fingerprint=compatibility_fingerprint,
        provenance=provenance,
        sample_id=sample_id,
        attempt_id=attempt_id,
        attempt_started_ns=attempt_started_ns,
    )
    if job.skip_reason:
        sample.update(
            {
                "status": "skipped",
                "eligible_for_threshold": False,
                "exclusion_reason": job.skip_reason,
                "metrics": {},
                "artifact_paths": {},
                "error": job.skip_reason,
            }
        )
        writer.append_sample(sample)
        writer.mark_job_complete(
            job_id=job.job_id,
            protocol_fingerprint=protocol_fingerprint,
            status="skipped",
            sample_id=sample_id,
            error=job.skip_reason,
            attempt_id=attempt_id,
            attempt_started_ns=attempt_started_ns,
        )
        return {"job_id": job.job_id, "status": "skipped", "eligible": False}

    frames_by_offset: dict[int, Any] = {}
    metrics: dict[str, Any] = {}
    artifacts: dict[str, str] = {}
    error: str | None = None
    exclusion_reason: str | None = None
    completed_status = "completed"
    settle_executed = 0
    capture_steps_executed = 0
    same_frame_encoding_latency_ms: float | None = None
    trajectory_encoding_latency_ms: float | None = None
    try:
        observation = environment.reset(job)
        sample["runtime_control_frequency_hz"] = _runtime_control_frequency(
            environment
        )
        policy.reset(environment.task_description, seed=job.episode_seed)
        for _ in range(resolved_settle_steps(cfg)):
            step_result = environment.step(list(STANDARD_NOOP_ACTION))
            observation = step_result.observation
            settle_executed += 1
            if is_episode_success(environment, step_result):
                exclusion_reason = "success_during_settle_noop"
                break
            if step_result.done:
                exclusion_reason = "environment_done_during_settle_noop"
                break

        if exclusion_reason is None:
            frames_by_offset[0] = encoder.observation_to_model_frame(observation)
            sample["model_frame_shape"] = list(
                getattr(frames_by_offset[0], "shape", ())
            )
            capture_set = set(cfg.static_calibration.capture_offsets[1:])
            maximum_offset = cfg.static_calibration.capture_offsets[-1]
            for offset in range(1, maximum_offset + 1):
                step_result = environment.step(list(STANDARD_NOOP_ACTION))
                observation = step_result.observation
                capture_steps_executed += 1
                if offset in capture_set:
                    frames_by_offset[offset] = encoder.observation_to_model_frame(
                        observation
                    )
                if is_episode_success(environment, step_result):
                    exclusion_reason = "success_during_measurement_noop"
                    break
                if step_result.done:
                    exclusion_reason = "environment_done_during_measurement_noop"
                    break

        missing_offsets = [
            offset
            for offset in cfg.static_calibration.capture_offsets
            if offset not in frames_by_offset
        ]
        if exclusion_reason is None and missing_offsets:
            exclusion_reason = f"missing_capture_offsets:{missing_offsets}"

        if exclusion_reason is None:
            current = frames_by_offset[0]
            repeated_frames = [
                current
                for _ in range(
                    cfg.static_calibration.repeated_same_frame_encodes
                )
            ]
            calibration_seed = job.episode_seed + CALIBRATION_SEED_OFFSET
            started = time.perf_counter()
            with RngIsolation(calibration_seed):
                repeated_embeddings = encoder.encode_frame_embeddings(
                    repeated_frames
                )
            same_frame_encoding_latency_ms = (
                time.perf_counter() - started
            ) * 1000.0

            trajectory_frames = [
                frames_by_offset[offset]
                for offset in cfg.static_calibration.capture_offsets
            ]
            started = time.perf_counter()
            with RngIsolation(calibration_seed + 1):
                trajectory_embeddings = encoder.encode_frame_embeddings(
                    trajectory_frames
                )
            trajectory_encoding_latency_ms = (
                time.perf_counter() - started
            ) * 1000.0
            metrics = compute_static_calibration_metrics(
                repeated_embeddings,
                trajectory_embeddings,
                capture_offsets=cfg.static_calibration.capture_offsets,
                frames_by_offset=frames_by_offset,
            )
            metrics.update(
                {
                    "same_frame_encoding_latency_ms": (
                        same_frame_encoding_latency_ms
                    ),
                    "trajectory_encoding_latency_ms": (
                        trajectory_encoding_latency_ms
                    ),
                    "same_frame_encoding_seed": calibration_seed,
                    "trajectory_encoding_seed": calibration_seed + 1,
                }
            )
            if cfg.static_calibration.save_frames:
                artifacts = writer.write_frames(
                    job_id=job.job_id,
                    frames_by_offset=frames_by_offset,
                )
    except Exception as exc:  # calibration failures must be durable and retryable
        completed_status = "exception"
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=12)}"
        LOGGER.exception("Static calibration job %s failed", job.job_id)

    eligible = completed_status == "completed" and exclusion_reason is None
    sample.update(
        {
            "status": (
                "completed"
                if eligible
                else ("excluded" if completed_status == "completed" else "exception")
            ),
            "eligible_for_threshold": eligible,
            "exclusion_reason": exclusion_reason,
            "settle_steps_executed": settle_executed,
            "capture_steps_executed": capture_steps_executed,
            "metrics": metrics,
            "artifact_paths": artifacts,
            "error": error,
        }
    )
    writer.append_sample(sample)
    writer.mark_job_complete(
        job_id=job.job_id,
        protocol_fingerprint=protocol_fingerprint,
        status=completed_status,
        sample_id=sample_id,
        error=error or exclusion_reason,
        attempt_id=attempt_id,
        attempt_started_ns=attempt_started_ns,
    )
    return {
        "job_id": job.job_id,
        "status": sample["status"],
        "eligible": eligible,
    }


def run_static_calibration_worker(
    cfg: EvalConfig,
    *,
    policy: BasePolicy,
    environment: BaseBenchmarkEnv,
    encoder: SupportsStaticCalibrationEncoder,
    jobs: Iterable[EvaluationJob] | None = None,
    rank: int = 0,
    world_size: int = 1,
    provenance: Mapping[str, Any] | None = None,
    writer: StaticCalibrationWriter | None = None,
    close_resources: bool = True,
    rerun: str = "incomplete",
) -> dict[str, int]:
    """Run one deterministic calibration shard with global per-job resume."""

    if not cfg.static_calibration.enabled:
        raise ValueError("Static calibration is disabled in this configuration")
    if cfg.diagnostics.enabled:
        raise ValueError("Static calibration and future diagnostics cannot run together")
    if rerun not in {"incomplete", "failed", "all"}:
        raise ValueError("rerun must be incomplete, failed, or all")

    all_jobs = list(jobs) if jobs is not None else plan_jobs(cfg)
    assigned = shard_jobs(all_jobs, rank, world_size)
    artifact_writer = writer or StaticCalibrationWriter(
        cfg.experiment.output_dir, rank
    )
    prov = dict(provenance or {})
    compatibility_fingerprint = calibration_compatibility_fingerprint(cfg, prov)
    protocol_fingerprint = static_calibration_protocol_fingerprint(
        cfg, all_jobs, prov
    )
    completed_count = 0
    eligible_count = 0
    pending_count = 0
    try:
        # The semantic gate precedes every environment reset.
        validate_static_calibration_encoder(cfg, encoder)
        artifact_writer.prepare_manifest(
            cfg=cfg,
            jobs=all_jobs,
            protocol_fingerprint=protocol_fingerprint,
            compatibility_fingerprint=compatibility_fingerprint,
            provenance=prov,
            write=rank == 0,
        )
        completed = (
            load_all_completed_jobs(artifact_writer.output_dir)
            if cfg.experiment.resume
            else {}
        )

        def should_run(job: EvaluationJob) -> bool:
            if cfg.experiment.overwrite or rerun == "all":
                return True
            previous = completed.get((job.job_id, protocol_fingerprint))
            if previous is None:
                return True
            if rerun == "failed":
                return previous.get("status") in {"error", "exception"}
            return previous.get("status") not in {"completed", "skipped"}

        pending = [job for job in assigned if should_run(job)]
        pending_count = len(pending)
        for job in pending:
            outcome = run_static_calibration_episode(
                cfg=cfg,
                job=job,
                policy=policy,
                environment=environment,
                encoder=encoder,
                writer=artifact_writer,
                worker_rank=rank,
                provenance=prov,
                protocol_fingerprint=protocol_fingerprint,
                compatibility_fingerprint=compatibility_fingerprint,
            )
            completed_count += 1
            eligible_count += int(bool(outcome["eligible"]))
    finally:
        if close_resources:
            environment.close()
            policy.close()
    return {
        "assigned": len(assigned),
        "pending": pending_count,
        "completed": completed_count,
        "eligible_samples": eligible_count,
        "skipped_by_resume": len(assigned) - pending_count,
    }


__all__ = [
    "CALIBRATION_SEED_OFFSET",
    "STANDARD_NOOP_ACTION",
    "STATIC_CALIBRATION_SCHEMA",
    "StaticCalibrationWriter",
    "SupportsStaticCalibrationEncoder",
    "calibration_compatibility_fingerprint",
    "calibration_compatibility_payload",
    "compute_static_calibration_metrics",
    "preflight_static_calibration_output",
    "resolved_settle_steps",
    "run_static_calibration_episode",
    "run_static_calibration_worker",
    "static_calibration_protocol_fingerprint",
    "validate_static_calibration_encoder",
]
