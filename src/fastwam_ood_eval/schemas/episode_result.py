"""JSON-serializable episode result schema."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EpisodeResult:
    experiment_id: str
    job_id: str
    timestamp: str
    git_commit: str | None
    fastwam_commit: str | None
    libero_commit: str | None
    libero_plus_commit: str | None
    checkpoint: str | None
    checkpoint_hash: str | None
    suite: str
    task_id: int
    task_name: str
    episode_index: int
    episode_seed: int
    condition: str
    perturbation_category: str | None
    perturbation_level: str | None
    perturbation_parameters: dict[str, Any]
    success: bool
    steps: int
    termination_reason: str
    policy_latency_mean_ms: float
    policy_latency_p50_ms: float
    policy_latency_p95_ms: float
    warmup_latency_ms: float | None
    action_chunk_shape: list[int] | None
    observation_image_shape: list[int] | None
    episode_duration_s: float
    gpu_peak_memory_mb: float
    gpu_memory_allocated_mb: float
    gpu_memory_reserved_mb: float
    video_path: str | None
    error: str | None
    worker_rank: int = 0
    status: str = "completed"
    failure_category: str | None = None
    failure_notes: str | None = None
    policy_variant: str = "unspecified"
    test_time_future_imagination: bool = False
    comparison_group: str | None = None
    training_recipe_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodeResult":
        known = set(cls.__dataclass_fields__)
        payload = {key: value for key, value in data.items() if key in known}
        return cls(**payload)


def append_result(path: Path, result: EpisodeResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(result.to_json() + "\n")
        handle.flush()
        try:
            import os

            os.fsync(handle.fileno())
        except OSError:
            pass
