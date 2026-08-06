"""Preregistered hierarchical and paired bootstrap statistics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from fastwam_ood_eval.thought6 import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED


@dataclass(frozen=True)
class BootstrapInterval:
    mean: float
    median: float
    lower: float
    upper: float
    replicates: int
    seed: int
    task_values: Mapping[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def hierarchical_bootstrap(
    values: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> BootstrapInterval:
    if replicates < 100:
        raise ValueError("hierarchical bootstrap requires at least 100 replicates")
    tasks = sorted(values)
    if len(tasks) < 2:
        raise ValueError("hierarchical bootstrap requires at least two task clusters")
    episode_means: dict[str, dict[str, float]] = {}
    for task in tasks:
        episodes = values[task]
        if not episodes:
            raise ValueError(f"task {task} has no episodes")
        episode_means[task] = {}
        for episode, rows in sorted(episodes.items()):
            array = np.asarray(rows, dtype=np.float64)
            if not len(array) or not np.isfinite(array).all():
                raise ValueError(f"task/episode {task}/{episode} is empty or non-finite")
            episode_means[task][episode] = float(np.mean(array))
    task_values = {
        task: float(np.mean(list(episode_means[task].values()))) for task in tasks
    }
    flat = np.asarray(
        [value for task in tasks for value in episode_means[task].values()],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected_task_indices = rng.integers(0, len(tasks), size=len(tasks))
        selected_task_means: list[float] = []
        for task_index in selected_task_indices:
            task = tasks[int(task_index)]
            episode_values = np.asarray(
                list(episode_means[task].values()), dtype=np.float64
            )
            selected_episodes = rng.integers(
                0, len(episode_values), size=len(episode_values)
            )
            selected_task_means.append(float(np.mean(episode_values[selected_episodes])))
        draws[replicate] = float(np.mean(selected_task_means))
    return BootstrapInterval(
        mean=float(np.mean(list(task_values.values()))),
        median=float(np.median(flat)),
        lower=float(np.quantile(draws, 0.025)),
        upper=float(np.quantile(draws, 0.975)),
        replicates=replicates,
        seed=seed,
        task_values=task_values,
    )


def paired_success_bootstrap(
    pairs: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> BootstrapInterval:
    """Task-cluster + initial-state paired resampling of success differences."""

    return hierarchical_bootstrap(pairs, replicates=replicates, seed=seed)
