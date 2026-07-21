"""Deterministic non-parametric bootstrap confidence intervals."""

from __future__ import annotations

import random


def bootstrap_mean_ci(
    values: list[float],
    *,
    confidence: float = 0.95,
    samples: int = 2000,
    seed: int = 0,
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = random.Random(seed)
    size = len(values)
    estimates = sorted(sum(rng.choice(values) for _ in range(size)) / size for _ in range(samples))
    alpha = (1.0 - confidence) / 2.0
    lower_index = max(0, min(samples - 1, int(alpha * samples)))
    upper_index = max(0, min(samples - 1, int((1.0 - alpha) * samples) - 1))
    return estimates[lower_index], estimates[upper_index]

