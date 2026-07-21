"""Deterministic lightweight benchmark environment."""

from __future__ import annotations

from typing import Any

from fastwam_ood_eval.envs.base import BaseBenchmarkEnv, StepResult
from fastwam_ood_eval.evaluation.jobs import EvaluationJob
from fastwam_ood_eval.reproducibility import stable_int


class MockBenchmarkEnv(BaseBenchmarkEnv):
    def __init__(self, max_steps: int = 8) -> None:
        self.max_steps = max_steps
        self.steps = 0
        self.target = max_steps + 1
        self.task_description = "mock task"
        self.closed = False

    def _observation(self) -> dict[str, Any]:
        return {
            "agentview_image": [[[self.steps, 0, 0]]],
            "robot0_eye_in_hand_image": [[[self.steps, 0, 0]]],
            "robot_state": [float(self.steps)],
        }

    def reset(self, job: EvaluationJob) -> dict[str, Any]:
        self.steps = 0
        self.task_description = job.task_name.replace("_", " ")
        difficulty = int(job.perturbation_parameters.get("official_difficulty", 0) or 0)
        base = 1 + stable_int(job.episode_seed, job.task_name, bits=8) % max(1, self.max_steps - 1)
        self.target = base + difficulty
        return self._observation()

    def step(self, action: Any) -> StepResult:
        self.steps += 1
        done = self.is_success()
        return StepResult(self._observation(), float(done), done, {"mock": True})

    def is_success(self) -> bool:
        return self.steps >= self.target and self.target <= self.max_steps

    def close(self) -> None:
        self.closed = True

