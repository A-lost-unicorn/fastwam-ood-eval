"""Central success policy: always defer to the benchmark's official signal."""

from __future__ import annotations

from fastwam_ood_eval.envs.base import BaseBenchmarkEnv, StepResult


def is_episode_success(environment: BaseBenchmarkEnv, step_result: StepResult) -> bool:
    return bool(step_result.done or environment.is_success())

