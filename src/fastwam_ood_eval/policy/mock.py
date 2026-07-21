"""Small deterministic policy used by CPU tests."""

from __future__ import annotations

import random
import time
from typing import Any

from fastwam_ood_eval.policy.base import BasePolicy, PolicyOutput


class MockPolicy(BasePolicy):
    def __init__(self, control_horizon: int = 2) -> None:
        self.control_horizon = control_horizon
        self._rng = random.Random(0)
        self._first = True

    def reset(self, task_description: str, *, seed: int | None = None) -> None:
        self._rng.seed(seed if seed is not None else 0)
        self._first = True

    def act(self, observation: dict[str, Any]) -> PolicyOutput:
        started = time.perf_counter()
        actions = [[self._rng.uniform(-0.01, 0.01)] * 6 + [-1.0] for _ in range(self.control_horizon)]
        latency = max((time.perf_counter() - started) * 1000.0, 0.001)
        image = observation.get("agentview_image", [])
        shape = list(getattr(image, "shape", ())) or None
        warmup = latency if self._first else None
        self._first = False
        return PolicyOutput(
            actions=actions,
            latency_ms=latency,
            warmup_latency_ms=warmup,
            action_chunk_shape=[self.control_horizon, 7],
            observation_image_shape=shape,
        )

