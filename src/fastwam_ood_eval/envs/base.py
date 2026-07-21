"""Unified benchmark environment interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from fastwam_ood_eval.evaluation.jobs import EvaluationJob


@dataclass
class StepResult:
    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class BaseBenchmarkEnv(ABC):
    task_description: str

    @abstractmethod
    def reset(self, job: EvaluationJob) -> dict[str, Any]:
        """Reset the selected task and episode seed."""

    @abstractmethod
    def step(self, action: Any) -> StepResult:
        """Execute one simulator action."""

    @abstractmethod
    def is_success(self) -> bool:
        """Return the environment's official success signal."""

    @abstractmethod
    def close(self) -> None:
        """Release simulator resources."""

