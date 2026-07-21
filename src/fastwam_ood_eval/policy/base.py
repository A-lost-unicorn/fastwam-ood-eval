"""Policy interface shared by real and mock evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PolicyOutput:
    actions: Any
    latency_ms: float
    warmup_latency_ms: float | None = None
    action_chunk_shape: list[int] | None = None
    observation_image_shape: list[int] | None = None
    gpu_memory_allocated_mb: float = 0.0
    gpu_memory_reserved_mb: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class BasePolicy(ABC):
    @abstractmethod
    def reset(self, task_description: str, *, seed: int | None = None) -> None:
        """Reset all episode-local policy state."""

    @abstractmethod
    def act(self, observation: dict[str, Any]) -> PolicyOutput:
        """Return an action chunk."""

    def close(self) -> None:
        """Release optional policy resources."""

