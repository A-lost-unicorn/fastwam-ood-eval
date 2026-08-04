"""Phase 5-D paired Clean/OOD rollout aggregation and H3 decision."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from fastwam_ood_eval.thought5.statistics import (
    grouped_bootstrap_mean,
    task_cluster_bootstrap,
)


@dataclass(frozen=True)
class RolloutRecord:
    variant: str
    task_id: str
    episode_seed: int
    condition: str
    success: bool
    latency_ms: float
    peak_memory_mib: float

    @property
    def key(self) -> tuple[str, int]:
        return self.task_id, self.episode_seed


def evaluate_rollouts(
    records: Sequence[RolloutRecord],
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 5587,
    clean_noninferiority_margin: float = 0.05,
    g4_equivalence_fraction: float = 0.8,
) -> dict[str, Any]:
    if not records:
        raise ValueError("rollout panel is empty")
    by: dict[tuple[str, str], dict[tuple[str, int], RolloutRecord]] = defaultdict(dict)
    for row in records:
        if row.variant not in {"B0", "B1", "G1", "G2", "G3", "G4"}:
            raise ValueError("unknown rollout variant")
        if row.condition not in {"clean", "camera", "lighting", "robot_init"}:
            raise ValueError("unknown rollout condition")
        if min(row.latency_ms, row.peak_memory_mib) < 0 or not np.isfinite(
            [row.latency_ms, row.peak_memory_mib]
        ).all():
            raise ValueError("invalid rollout telemetry")
        slot = by[(row.variant, row.condition)]
        if row.key in slot:
            raise ValueError("duplicate rollout task/seed")
        slot[row.key] = row
    for condition in ("clean", "camera", "lighting", "robot_init"):
        if set(by[("B1", condition)]) != set(by[("G3", condition)]):
            raise ValueError("B1/G3 rollout seeds are not exactly matched")
    summaries: dict[str, Any] = {}
    intervals: dict[str, Any] = {}
    task_intervals: dict[str, Any] = {}
    for variant_condition, values in sorted(by.items()):
        variant, condition = variant_condition
        success = [float(row.success) for row in values.values()]
        summaries[f"{variant}:{condition}"] = {
            "episodes": len(values),
            "success_rate": float(np.mean(success)),
            "latency_ms_mean": float(np.mean([row.latency_ms for row in values.values()])),
            "peak_memory_mib_max": float(max(row.peak_memory_mib for row in values.values())),
        }
    for index, condition in enumerate(("clean", "camera", "lighting", "robot_init")):
        b1 = by[("B1", condition)]
        g3 = by[("G3", condition)]
        differences = {
            f"{task}/{seed}": [float(g3[key].success) - float(b1[key].success)]
            for key in b1
            for task, seed in [key]
        }
        intervals[condition] = asdict(
            grouped_bootstrap_mean(
                differences,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + index,
            )
        )
        by_task: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for (task, seed), b1_row in b1.items():
            by_task[task][str(seed)].append(
                float(g3[(task, seed)].success) - float(b1_row.success)
            )
        if len(by_task) >= 2:
            task_intervals[condition] = asdict(
                task_cluster_bootstrap(
                    by_task,
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed + 100 + index,
                )
            )
        else:
            task_intervals[condition] = {
                "status": "PILOT_SINGLE_TASK_NOT_INFERENTIAL",
                "task_count": len(by_task),
                "estimate": intervals[condition]["estimate"],
                "lower": None,
                "upper": None,
            }
    b1_clean = summaries["B1:clean"]["success_rate"]
    g3_clean = summaries["G3:clean"]["success_rate"]
    clean_noninferior = g3_clean >= b1_clean - clean_noninferiority_margin
    camera_specific = intervals["camera"]["estimate"] > intervals["lighting"]["estimate"]
    formal_multitask = all(
        row.get("status") != "PILOT_SINGLE_TASK_NOT_INFERENTIAL"
        for row in task_intervals.values()
    )
    g4_matches: bool | None = None
    if ("G4", "camera") in by:
        if set(by[("G4", "camera")]) != set(by[("B1", "camera")]):
            raise ValueError("B1/G4 rollout seeds are not exactly matched")
        g3_gain = intervals["camera"]["estimate"]
        g4_gain = float(
            np.mean(
                [
                    float(by[("G4", "camera")][key].success)
                    - float(by[("B1", "camera")][key].success)
                    for key in by[("B1", "camera")]
                ]
            )
        )
        g4_matches = bool(
            g3_gain <= 0
            or g4_gain >= g4_equivalence_fraction * g3_gain
        )
    pilot_direction = bool(
        intervals["camera"]["lower"] > 0
        and clean_noninferior
        and camera_specific
        and g4_matches is not True
    )
    h3 = bool(
        formal_multitask
        and pilot_direction
        and float(task_intervals["camera"]["lower"]) > 0
    )
    return {
        "schema_version": "thought5.phase5.rollout_results.v1",
        "status": "complete",
        "paired_rollout_seeds": True,
        "summaries": summaries,
        "g3_minus_b1_paired_intervals": intervals,
        "g3_minus_b1_task_cluster_intervals": task_intervals,
        "clean_noninferior": clean_noninferior,
        "camera_specific": camera_specific,
        "pilot_direction_observed": pilot_direction,
        "shuffled_control_matches_gain": g4_matches,
        "g4_equivalence_fraction": g4_equivalence_fraction,
        "formal_multitask_inference": formal_multitask,
        "h3_supported": h3,
    }


def not_run_rollout_result() -> dict[str, Any]:
    return {
        "schema_version": "thought5.phase5.rollout_results.v1",
        "status": "NOT RUN",
        "paired_rollout_seeds": True,
        "h3_supported": None,
    }
