"""Durable one-row-per-probe schema for shadow future diagnostics."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping


def _json_value(value: Any, *, path: str) -> Any:
    """Convert small metadata while refusing tensors/arrays and non-finite values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item, path=f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item, path=f"{path}[]") for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if is_dataclass(value):
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return _json_value(value.to_dict(), path=path)
        return _json_value(asdict(value), path=path)
    # Large frame/latent tensors must be written through the artifact layer.
    if (
        hasattr(value, "detach")
        or hasattr(value, "numpy")
        or hasattr(value, "__array_interface__")
        or hasattr(value, "shape")
    ):
        raise TypeError(
            f"{path} contains an array/tensor; write video or latent data as an artifact and store its path"
        )
    raise TypeError(f"{path} contains unsupported type {type(value).__name__}")


def _scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, float | int | bool | None]:
    result: dict[str, float | int | bool | None] = {}
    for key, value in metrics.items():
        if value is None or isinstance(value, (bool, int)):
            result[str(key)] = value
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"metrics.{key} must be finite or null")
            result[str(key)] = value
        else:
            raise TypeError(
                f"metrics.{key} must be a numeric/bool/null scalar, got {type(value).__name__}"
            )
    return result


@dataclass(kw_only=True)
class FutureDiagnosticResult:
    """One durable record for one selected replanning probe.

    Media and latent arrays are intentionally absent.  Their relative artifact
    paths are stored instead, keeping JSONL compact, safe, and resumable.
    """

    schema_version: int
    diagnostic_id: str
    experiment_id: str
    source_experiment_id: str
    job_id: str
    replan_index: int
    origin_env_step: int

    # Exact study-facing names are stored alongside legacy/internal aliases so
    # downstream notebooks never have to guess their semantics.
    probe_id: str | None = None
    environment_step: int | None = None

    timestamp: str | None = None
    attempt_id: str | None = None
    attempt_started_ns: int | None = None
    recorded_at_ns: int | None = None
    worker_rank: int = 0
    suite: str | None = None
    task_id: int | None = None
    task_name: str | None = None
    episode_index: int | None = None
    episode_seed: int | None = None
    initial_state_index: int | None = None
    condition: str | None = None
    perturbation_category: str | None = None
    perturbation_level: str | None = None
    perturbation_parameters: dict[str, Any] = field(default_factory=dict)

    checkpoint: str | None = None
    checkpoint_hash: str | None = None
    fastwam_commit: str | None = None
    mode: str = "action_conditioned_future"
    action_conditioned_verified: bool = False
    causal_interpretation_allowed: bool = False

    inference_seed: int | None = None
    diagnostic_seed: int | None = None
    num_video_frames: int | None = None
    num_inference_steps: int | None = None
    action_hash: str | None = None
    action_hash_before: str | None = None
    action_hash_after: str | None = None
    action_unchanged: bool | None = None
    action_chunk_shape: list[int] | None = None
    # Baseline action chunk returned before the shadow probe; this is never an
    # action re-predicted from the generated future.
    predicted_actions: list[Any] | None = None
    executed_actions: list[Any] | None = None
    executed_action_count: int = 0

    success: bool | None = None
    episode_success: bool | None = None
    termination_reason: str = "unknown"
    alignment: dict[str, Any] = field(default_factory=dict)
    approximate_alignment: bool | None = None
    metrics: dict[str, float | int | bool | None] = field(default_factory=dict)
    static_future_flag: bool | None = None
    metric_metadata: dict[str, Any] = field(default_factory=dict)

    current_frame_path: str | None = None
    predicted_video_path: str | None = None
    actual_video_path: str | None = None
    side_by_side_video_path: str | None = None
    latent_path: str | None = None
    action_artifact_path: str | None = None
    artifact_paths: dict[str, str | None] = field(default_factory=dict)

    generation_latency_ms: float | None = None
    generation_peak_memory_mb: float | None = None
    status: str = "completed"
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.probe_id is None:
            self.probe_id = self.diagnostic_id
        elif self.probe_id != self.diagnostic_id:
            raise ValueError("probe_id and diagnostic_id must identify the same probe")
        if self.environment_step is None:
            self.environment_step = self.origin_env_step
        elif self.environment_step != self.origin_env_step:
            raise ValueError("environment_step and origin_env_step must match")
        if self.action_hash is None:
            self.action_hash = self.action_hash_before or self.action_hash_after
        if self.success is None and self.episode_success is None:
            self.success = False
            self.episode_success = False
        elif self.success is None:
            self.success = self.episode_success
        elif self.episode_success is None:
            self.episode_success = self.success
        elif self.success != self.episode_success:
            raise ValueError("success and episode_success must match")
        if self.approximate_alignment is None:
            alignment = (
                self.alignment.to_dict()
                if hasattr(self.alignment, "to_dict") and callable(self.alignment.to_dict)
                else self.alignment
            )
            timestamp_status = (
                alignment.get("timestamp_status") if isinstance(alignment, Mapping) else None
            )
            exact_step_mapping = (
                bool(alignment.get("exact_step_mapping", False))
                if isinstance(alignment, Mapping)
                else False
            )
            action_coverage = (
                alignment.get("action_conditioning_coverage_complete")
                if isinstance(alignment, Mapping)
                else None
            )
            self.approximate_alignment = (
                not exact_step_mapping
                or timestamp_status != "exact"
                or action_coverage is False
            )
        metric_static_flag = self.metrics.get(
            "static_future_flag",
            self.metrics.get("predicted_static"),
        )
        if self.static_future_flag is None:
            self.static_future_flag = metric_static_flag
        elif metric_static_flag is not None and self.static_future_flag != metric_static_flag:
            raise ValueError("static_future_flag must match metrics.static_future_flag")
        if self.causal_interpretation_allowed:
            raise ValueError(
                "offline future diagnostics are associational; causal_interpretation_allowed must be false"
            )
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        if self.replan_index < 0 or self.origin_env_step < 0:
            raise ValueError("replan_index and origin_env_step must be non-negative")
        if self.attempt_started_ns is not None and self.attempt_started_ns < 0:
            raise ValueError("attempt_started_ns must be non-negative")
        if self.recorded_at_ns is not None and self.recorded_at_ns < 0:
            raise ValueError("recorded_at_ns must be non-negative")
        if self.executed_action_count < 0:
            raise ValueError("executed_action_count must be non-negative")
        if not self.diagnostic_id or not self.experiment_id or not self.job_id:
            raise ValueError("diagnostic_id, experiment_id, and job_id must be non-empty")
        if not self.source_experiment_id:
            raise ValueError("source_experiment_id must be non-empty")
        if self.status not in {"completed", "unavailable", "skipped", "error", "exception"}:
            raise ValueError(f"unsupported diagnostic status: {self.status}")
        _scalar_metrics(self.metrics)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metrics"] = _scalar_metrics(self.metrics)
        # asdict has already recursively converted TemporalAlignment when one was
        # supplied despite the mapping annotation.
        return _json_value(payload, path="future_diagnostic_result")

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FutureDiagnosticResult":
        known = set(cls.__dataclass_fields__)
        payload = {key: value for key, value in data.items() if key in known}
        return cls(**payload)


def append_future_diagnostic_result(path: Path, result: FutureDiagnosticResult) -> None:
    """Append and fsync one durable probe result."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(result.to_json() + "\n")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass


# Familiar alias for callers mirroring schemas.episode_result.
append_result = append_future_diagnostic_result


__all__ = [
    "FutureDiagnosticResult",
    "append_future_diagnostic_result",
    "append_result",
]
