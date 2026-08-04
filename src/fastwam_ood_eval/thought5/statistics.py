"""Frozen episode-grouped and task-cluster bootstrap utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float
    replicates: int
    seed: int


def _interval(values: np.ndarray, estimates: np.ndarray, seed: int) -> Interval:
    return Interval(
        estimate=float(np.mean(values)),
        lower=float(np.quantile(estimates, 0.025)),
        upper=float(np.quantile(estimates, 0.975)),
        replicates=int(len(estimates)),
        seed=seed,
    )


def grouped_bootstrap_mean(
    values_by_group: Mapping[str, Sequence[float]],
    *,
    replicates: int = 2000,
    seed: int = 5507,
) -> Interval:
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    groups = sorted(values_by_group)
    if not groups or any(not values_by_group[group] for group in groups):
        raise ValueError("bootstrap groups must be non-empty")
    flat = np.asarray(
        [value for group in groups for value in values_by_group[group]],
        dtype=np.float64,
    )
    if not np.isfinite(flat).all():
        raise ValueError("bootstrap values must be finite")
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = rng.choice(groups, size=len(groups), replace=True)
        draws[index] = np.mean(
            [value for group in selected for value in values_by_group[str(group)]]
        )
    return _interval(flat, draws, seed)


def task_cluster_bootstrap(
    values_by_task_episode: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    replicates: int = 2000,
    seed: int = 5517,
) -> Interval:
    tasks = sorted(values_by_task_episode)
    if len(tasks) < 2:
        raise ValueError("task-cluster bootstrap requires at least two tasks")
    task_means = np.asarray(
        [
            np.mean(
                [
                    value
                    for values in values_by_task_episode[task].values()
                    for value in values
                ]
            )
            for task in tasks
        ],
        dtype=np.float64,
    )
    if not np.isfinite(task_means).all():
        raise ValueError("task bootstrap values must be finite")
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = rng.choice(len(tasks), size=len(tasks), replace=True)
        draws[index] = float(np.mean(task_means[selected]))
    return _interval(task_means, draws, seed)


def paired_differences(
    left: Iterable[float], right: Iterable[float]
) -> np.ndarray:
    lhs = np.asarray(list(left), dtype=np.float64)
    rhs = np.asarray(list(right), dtype=np.float64)
    if lhs.shape != rhs.shape or not lhs.size:
        raise ValueError("paired arrays must have the same non-empty shape")
    result = lhs - rhs
    if not np.isfinite(result).all():
        raise ValueError("paired differences must be finite")
    return result
