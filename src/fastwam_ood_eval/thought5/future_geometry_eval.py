"""Phase 5-B future-latent and future-geometry evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from fastwam_ood_eval.diagnostics.metrics import compute_future_metrics
from fastwam_ood_eval.thought5.statistics import (
    grouped_bootstrap_mean,
    task_cluster_bootstrap,
)


@dataclass(frozen=True)
class FutureGeometryRecord:
    variant: str
    task_id: str
    episode_id: str
    seed: int
    condition: str
    predicted_embeddings: Any
    actual_embeddings: Any
    predicted_depth_relation: Any
    actual_depth_relation: Any
    predicted_eef_object: Any
    actual_eef_object: Any
    predicted_camera_geometry: Any
    actual_camera_geometry: Any


def _rmse(left: Any, right: Any) -> float:
    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    if lhs.shape != rhs.shape or not lhs.size:
        raise ValueError("future geometry arrays must be matched and non-empty")
    difference = lhs - rhs
    if not np.isfinite(difference).all():
        raise ValueError("future geometry contains NaN/Inf")
    return float(np.sqrt(np.mean(difference**2)))


def score_future_geometry(record: FutureGeometryRecord) -> dict[str, Any]:
    metrics = compute_future_metrics(
        record.predicted_embeddings,
        record.actual_embeddings,
        static_motion_threshold=1e-4,
    )
    return {
        "variant": record.variant,
        "task_id": record.task_id,
        "episode_id": record.episode_id,
        "seed": record.seed,
        "condition": record.condition,
        **metrics,
        "future_depth_relation_rmse": _rmse(
            record.predicted_depth_relation, record.actual_depth_relation
        ),
        "future_eef_object_position_rmse": _rmse(
            record.predicted_eef_object, record.actual_eef_object
        ),
        "future_camera_geometry_rmse": _rmse(
            record.predicted_camera_geometry, record.actual_camera_geometry
        ),
    }


def evaluate_future_geometry(
    records: Sequence[FutureGeometryRecord],
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 5567,
    g4_equivalence_fraction: float = 0.8,
) -> dict[str, Any]:
    rows = [score_future_geometry(row) for row in records]
    if not rows:
        raise ValueError("future geometry evaluation is empty")
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault(row["variant"], {}).setdefault(row["condition"], []).append(row)
    for variant in ("B1", "G3"):
        if set(grouped.get(variant, {})) < {"clean", "camera"}:
            raise ValueError("future geometry requires B1/G3 clean and camera")
    comparisons: dict[str, Any] = {}
    task_comparisons: dict[str, Any] = {}
    for metric in (
        "future_latent_l1",
        "motion_direction_cosine",
        "future_depth_relation_rmse",
        "future_eef_object_position_rmse",
        "future_camera_geometry_rmse",
    ):
        for condition in ("clean", "camera"):
            left = {
                (row["task_id"], row["episode_id"], row["seed"]): row[metric]
                for row in grouped["B1"][condition]
            }
            right = {
                (row["task_id"], row["episode_id"], row["seed"]): row[metric]
                for row in grouped["G3"][condition]
            }
            if set(left) != set(right) or any(
                left[key] is None or right[key] is None for key in left
            ):
                raise ValueError("B1/G3 future rows are not exactly matched")
            # For errors, negative G3-B1 is improvement; cosine uses B1-G3.
            sign = -1.0 if metric == "motion_direction_cosine" else 1.0
            values = {
                "/".join(map(str, key)): [sign * (right[key] - left[key])]
                for key in left
            }
            comparisons[f"{metric}:{condition}"] = asdict(
                grouped_bootstrap_mean(
                    values,
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed + len(comparisons),
                )
            )
            by_task: dict[str, dict[str, list[float]]] = {}
            for (task, episode, seed), value in (
                (
                    key,
                    sign * (right[key] - left[key]),
                )
                for key in left
            ):
                by_task.setdefault(task, {}).setdefault(
                    f"{episode}/{seed}", []
                ).append(value)
            if len(by_task) >= 2:
                task_comparisons[f"{metric}:{condition}"] = asdict(
                    task_cluster_bootstrap(
                        by_task,
                        replicates=bootstrap_replicates,
                        seed=bootstrap_seed + 100 + len(task_comparisons),
                    )
                )
            else:
                task_comparisons[f"{metric}:{condition}"] = {
                    "status": "PILOT_SINGLE_TASK_NOT_INFERENTIAL",
                    "task_count": len(by_task),
                }
    main_metric = "future_camera_geometry_rmse"
    b1_clean = float(
        np.mean([row[main_metric] for row in grouped["B1"]["clean"]])
    )
    g3_clean = float(
        np.mean([row[main_metric] for row in grouped["G3"]["clean"]])
    )
    b1_camera_rows = grouped["B1"]["camera"]
    g3_camera_rows = grouped["G3"]["camera"]
    b1_camera = float(np.mean([row[main_metric] for row in b1_camera_rows]))
    g3_camera = float(np.mean([row[main_metric] for row in g3_camera_rows]))
    g4_camera = None
    g4_matches: bool | None = None
    if "G4" in grouped and "camera" in grouped["G4"]:
        g4_by_key = {
            (row["task_id"], row["episode_id"], row["seed"]): row
            for row in grouped["G4"]["camera"]
        }
        b1_by_key = {
            (row["task_id"], row["episode_id"], row["seed"]): row
            for row in b1_camera_rows
        }
        if set(g4_by_key) != set(b1_by_key):
            raise ValueError("B1/G4 future-geometry rows are not matched")
        g4_camera = float(
            np.mean([row[main_metric] for row in g4_by_key.values()])
        )
        g3_benefit = b1_camera - g3_camera
        g4_benefit = b1_camera - g4_camera
        g4_matches = bool(
            g3_benefit <= 0
            or g4_benefit >= g4_equivalence_fraction * g3_benefit
        )
    required_camera = (
        "future_latent_l1:camera",
        "future_depth_relation_rmse:camera",
        "future_eef_object_position_rmse:camera",
        "future_camera_geometry_rmse:camera",
    )
    camera_improves = all(
        float(comparisons[key]["estimate"]) < 0 for key in required_camera
    )
    camera_specific = (
        float(comparisons[f"{main_metric}:camera"]["estimate"])
        < float(comparisons[f"{main_metric}:clean"]["estimate"])
    )
    pilot_direction = bool(
        camera_improves and camera_specific and g4_matches is not True
    )
    return {
        "schema_version": "thought5.phase5.future_geometry_results.v1",
        "status": "complete",
        "sampler_k": 1,
        "paired_noise": True,
        "rows": rows,
        "g3_minus_b1_error_intervals": comparisons,
        "g3_minus_b1_task_cluster_intervals": task_comparisons,
        "main_camera_error": {
            "B1": b1_camera,
            "G3": g3_camera,
            "G4": g4_camera,
        },
        "main_clean_error": {"B1": b1_clean, "G3": g3_clean},
        "clean_to_camera_gap": {
            "B1": b1_camera - b1_clean,
            "G3": g3_camera - g3_clean,
        },
        "camera_specific": camera_specific,
        "pilot_direction_observed": pilot_direction,
        "shuffled_control_matches_gain": g4_matches,
        "g4_equivalence_fraction": g4_equivalence_fraction,
    }


def not_run_future_geometry_result() -> dict[str, Any]:
    return {
        "schema_version": "thought5.phase5.future_geometry_results.v1",
        "status": "NOT RUN",
        "sampler_k": 1,
        "paired_noise": True,
    }
