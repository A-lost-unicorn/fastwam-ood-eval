"""Frozen Thought4-style representation endpoints for Phase 5-A."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from fastwam_ood_eval.thought5.statistics import (
    grouped_bootstrap_mean,
    task_cluster_bootstrap,
)


FROZEN_REPRESENTATION_ENDPOINTS = (
    "video_eef_object_translation_camera",
    "video_eef_object_translation_world",
    "video_depth",
    "video_relative_depth",
    "camera_geometry_gap",
    "lighting_geometry_gap",
    "rank3_geometry_subspace_coordinate_shift",
    "camera_minus_lighting_paired_difference",
    "action_current_geometry",
    "action_future_se3",
)


@dataclass(frozen=True)
class RepresentationRecord:
    variant: str
    task_id: str
    episode_id: str
    seed: int
    condition: str
    endpoint: str
    error: float

    @property
    def pair_key(self) -> tuple[str, str, int, str]:
        return self.variant, self.task_id, self.seed, self.episode_id


def _finite_records(records: Sequence[RepresentationRecord]) -> None:
    if not records:
        raise ValueError("representation evaluation is empty")
    for row in records:
        if row.endpoint not in FROZEN_REPRESENTATION_ENDPOINTS:
            raise ValueError(f"unregistered endpoint: {row.endpoint}")
        if row.condition not in {"clean", "camera", "lighting", "robot_init"}:
            raise ValueError(f"unknown condition: {row.condition}")
        if not np.isfinite(row.error) or row.error < 0:
            raise ValueError("probe errors must be finite and non-negative")


def exact_state_gaps(
    records: Sequence[RepresentationRecord], *, endpoint: str
) -> dict[str, dict[tuple[str, str, int], float]]:
    _finite_records(records)
    selected = [row for row in records if row.endpoint == endpoint]
    by_pair: dict[tuple[str, str, int, str], dict[str, float]] = defaultdict(dict)
    for row in selected:
        if row.condition in {"clean", "camera", "lighting"}:
            if row.condition in by_pair[row.pair_key]:
                raise ValueError("duplicate representation condition row")
            by_pair[row.pair_key][row.condition] = row.error
    result: dict[str, dict[tuple[str, str, int], float]] = {
        "camera": {},
        "lighting": {},
    }
    for (variant, task, seed, episode), values in by_pair.items():
        if set(values) != {"clean", "camera", "lighting"}:
            raise ValueError("exact-state representation triplet is incomplete")
        key = (task, episode, seed)
        for condition in ("camera", "lighting"):
            result[condition][(variant, *key)] = values[condition] - values["clean"]
    return result


def evaluate_h1(
    records: Sequence[RepresentationRecord],
    *,
    endpoint: str = "video_eef_object_translation_camera",
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 5517,
    task_bootstrap_seed: int = 5527,
    clean_noninferiority_fraction: float = 0.05,
    g4_equivalence_fraction: float = 0.8,
) -> dict[str, Any]:
    gaps = exact_state_gaps(records, endpoint=endpoint)
    by_condition_variant: dict[str, dict[str, dict[tuple[str, str, int], float]]] = {
        condition: defaultdict(dict) for condition in ("camera", "lighting")
    }
    for condition, rows in gaps.items():
        for (variant, task, episode, seed), value in rows.items():
            by_condition_variant[condition][variant][(task, episode, seed)] = value
    for condition in ("camera", "lighting"):
        if set(by_condition_variant[condition]) < {"B1", "G3"}:
            raise ValueError("H1 requires matched B1 and G3 records")
        if set(by_condition_variant[condition]["B1"]) != set(
            by_condition_variant[condition]["G3"]
        ):
            raise ValueError("B1/G3 representation pairs are not matched")
    differences: dict[str, dict[tuple[str, str, int], float]] = {}
    for condition in ("camera", "lighting"):
        differences[condition] = {
            key: by_condition_variant[condition]["G3"][key]
            - by_condition_variant[condition]["B1"][key]
            for key in by_condition_variant[condition]["B1"]
        }
    grouped = {
        condition: grouped_bootstrap_mean(
            {
                f"{task}/{episode}/{seed}": [value]
                for (task, episode, seed), value in values.items()
            },
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + index,
        )
        for index, (condition, values) in enumerate(differences.items())
    }
    task_values = {
        task: defaultdict(list)
        for task, _episode, _seed in differences["camera"]
    }
    for (task, episode, _seed), value in differences["camera"].items():
        task_values[task][episode].append(value)
    formal_multitask = len(task_values) >= 2
    if formal_multitask:
        task_interval: dict[str, Any] = asdict(
            task_cluster_bootstrap(
                task_values,
                replicates=bootstrap_replicates,
                seed=task_bootstrap_seed,
            )
        )
    else:
        task_interval = {
            "status": "PILOT_SINGLE_TASK_NOT_INFERENTIAL",
            "task_count": len(task_values),
            "estimate": float(np.mean(list(differences["camera"].values()))),
            "lower": None,
            "upper": None,
        }
    b1_camera = float(np.mean(list(by_condition_variant["camera"]["B1"].values())))
    g3_camera = float(np.mean(list(by_condition_variant["camera"]["G3"].values())))
    gap_reduction = (b1_camera - g3_camera) / max(abs(b1_camera), 1e-12)
    clean = defaultdict(dict)
    for row in records:
        if row.endpoint == endpoint and row.condition == "clean" and row.variant in {"B1", "G3"}:
            clean[row.variant][(row.task_id, row.episode_id, row.seed)] = row.error
    if set(clean) != {"B1", "G3"} or set(clean["B1"]) != set(clean["G3"]):
        raise ValueError("clean B1/G3 records are incomplete")
    clean_b1 = float(np.mean(list(clean["B1"].values())))
    clean_g3 = float(np.mean(list(clean["G3"].values())))
    clean_non_degraded = clean_g3 <= clean_b1 * (1 + clean_noninferiority_fraction)
    lighting_reduction = -grouped["lighting"].estimate
    camera_reduction = -grouped["camera"].estimate
    lighting_specific = camera_reduction > lighting_reduction
    g4_matches: bool | None = None
    g4_camera = None
    if "G4" in by_condition_variant["camera"]:
        if set(by_condition_variant["camera"]["G4"]) != set(
            by_condition_variant["camera"]["B1"]
        ):
            raise ValueError("B1/G4 representation pairs are not matched")
        g4_camera = float(
            np.mean(list(by_condition_variant["camera"]["G4"].values()))
        )
        g3_benefit = b1_camera - g3_camera
        g4_benefit = b1_camera - g4_camera
        g4_matches = bool(
            g3_benefit <= 0
            or g4_benefit >= g4_equivalence_fraction * g3_benefit
        )
    pilot_direction = bool(
        gap_reduction >= 0.25
        and grouped["camera"].upper < 0
        and clean_non_degraded
        and lighting_specific
        and g4_matches is not True
    )
    h1 = bool(
        formal_multitask
        and pilot_direction
        and float(task_interval["upper"]) < 0
    )
    return {
        "schema_version": "thought5.phase5.representation_results.v1",
        "status": "complete",
        "endpoint": endpoint,
        "fixed_layer": 15,
        "gap_reduction_fraction": gap_reduction,
        "b1_camera_gap": b1_camera,
        "g3_camera_gap": g3_camera,
        "g4_camera_gap": g4_camera,
        "g3_minus_b1_camera_grouped_bootstrap": asdict(grouped["camera"]),
        "g3_minus_b1_lighting_grouped_bootstrap": asdict(grouped["lighting"]),
        "g3_minus_b1_camera_task_cluster_bootstrap": task_interval,
        "clean_b1_error": clean_b1,
        "clean_g3_error": clean_g3,
        "clean_non_degraded": clean_non_degraded,
        "lighting_specific": lighting_specific,
        "pilot_direction_observed": bool(pilot_direction),
        "shuffled_control_matches_gain": g4_matches,
        "g4_equivalence_fraction": g4_equivalence_fraction,
        "formal_multitask_inference": formal_multitask,
        "h1_supported": h1,
        "per_task": {
            task: {
                "mean_g3_minus_b1_camera_gap": float(
                    np.mean(
                        [
                            value
                            for (row_task, _episode, _seed), value in differences[
                                "camera"
                            ].items()
                            if row_task == task
                        ]
                    )
                )
            }
            for task in sorted(task_values)
        },
    }


def not_run_representation_result() -> dict[str, Any]:
    return {
        "schema_version": "thought5.phase5.representation_results.v1",
        "status": "NOT RUN",
        "fixed_layer": 15,
        "endpoints": list(FROZEN_REPRESENTATION_ENDPOINTS),
        "h1_supported": None,
    }
