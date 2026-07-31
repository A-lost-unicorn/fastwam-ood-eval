"""Metrics, exact-state gaps and episode-grouped bootstrap for probes."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence


class ProbeEvaluationError(ValueError):
    """Raised for invalid predictions, masks or grouping."""


def _numpy(value: Any) -> Any:
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if not np.isfinite(result).all():
        raise ProbeEvaluationError("metric input contains NaN/Inf")
    return result


def _valid_flat(prediction: Any, target: Any, mask: Any | None) -> tuple[Any, Any]:
    import numpy as np

    pred = _numpy(prediction)
    truth = _numpy(target)
    if pred.shape != truth.shape:
        raise ProbeEvaluationError(
            f"prediction/target shapes differ: {pred.shape} vs {truth.shape}"
        )
    if mask is None:
        return pred.reshape(-1), truth.reshape(-1)
    valid = np.asarray(mask, dtype=bool)
    while valid.ndim < truth.ndim:
        valid = np.expand_dims(valid, -1)
    valid = np.broadcast_to(valid, truth.shape)
    if not valid.any():
        raise ProbeEvaluationError("metric mask has no valid elements")
    return pred[valid], truth[valid]


def regression_metrics(
    prediction: Any, target: Any, mask: Any | None = None
) -> dict[str, float]:
    import numpy as np

    pred, truth = _valid_flat(prediction, target, mask)
    delta = pred - truth
    return {
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta**2))),
    }


def depth_metrics(prediction: Any, target: Any) -> dict[str, float]:
    import numpy as np

    pred = _numpy(prediction).astype(float).reshape(-1)
    truth = _numpy(target).astype(float).reshape(-1)
    if pred.shape != truth.shape or bool((pred <= 0).any()) or bool((truth <= 0).any()):
        raise ProbeEvaluationError("depth values must be matching and positive")
    abs_rel = np.mean(np.abs(pred - truth) / truth)
    rmse = np.sqrt(np.mean((pred - truth) ** 2))
    ratio = np.maximum(pred / truth, truth / pred)
    # Spearman correlation without scipy; tied ranks receive stable ordinal ranks.
    pred_rank = np.empty_like(pred, dtype=float)
    truth_rank = np.empty_like(truth, dtype=float)
    pred_rank[np.argsort(pred, kind="mergesort")] = np.arange(len(pred))
    truth_rank[np.argsort(truth, kind="mergesort")] = np.arange(len(truth))
    rank_correlation = (
        float(np.corrcoef(pred_rank, truth_rank)[0, 1])
        if len(pred) > 1 and pred_rank.std() > 0 and truth_rank.std() > 0
        else 0.0
    )
    return {
        "abs_rel": float(abs_rel),
        "rmse": float(rmse),
        "delta1": float(np.mean(ratio < 1.25)),
        "relative_depth_rank_correlation": rank_correlation,
    }


def rotation_metrics_6d(
    prediction: Any, target: Any, mask: Any | None = None
) -> dict[str, float]:
    import numpy as np

    from fastwam_ood_eval.thought4.geometry_labels import (
        rotation_6d_to_matrix,
        rotation_geodesic_degrees,
    )

    pred = _numpy(prediction)
    truth = _numpy(target)
    if pred.shape != truth.shape or pred.shape[-1] != 6:
        raise ProbeEvaluationError("rotation tensors must match [...,6]")
    valid = (
        np.ones(pred.shape[:-1], dtype=bool)
        if mask is None
        else np.broadcast_to(np.asarray(mask, dtype=bool), pred.shape[:-1])
    )
    errors: list[float] = []
    for pred_row, truth_row, keep in zip(
        pred.reshape(-1, 6), truth.reshape(-1, 6), valid.reshape(-1)
    ):
        if keep:
            try:
                errors.append(
                    rotation_geodesic_degrees(
                        rotation_6d_to_matrix(pred_row),
                        rotation_6d_to_matrix(truth_row),
                    )
                )
            except ValueError:
                errors.append(180.0)
    if not errors:
        raise ProbeEvaluationError("rotation mask has no valid elements")
    return {
        "rotation_geodesic_degrees": float(np.mean(errors)),
        "rotation_geodesic_median_degrees": float(np.median(errors)),
    }


def trajectory_metrics(
    translation_prediction: Any,
    translation_target: Any,
    valid_mask: Any,
    *,
    rotation_prediction: Any | None = None,
    rotation_target: Any | None = None,
    gripper_prediction: Any | None = None,
    gripper_target: Any | None = None,
) -> dict[str, float]:
    import numpy as np

    pred = _numpy(translation_prediction)
    truth = _numpy(translation_target)
    mask = np.asarray(valid_mask, dtype=bool)
    if pred.shape != truth.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ProbeEvaluationError("translation trajectories must be [N,H,3]")
    if mask.shape != pred.shape[:2]:
        raise ProbeEvaluationError("trajectory valid mask must be [N,H]")
    distances = np.linalg.norm(pred - truth, axis=-1)
    if not mask.any():
        raise ProbeEvaluationError("trajectory mask has no valid steps")
    final_errors: list[float] = []
    for row, valid in zip(distances, mask):
        indices = np.flatnonzero(valid)
        if len(indices):
            final_errors.append(float(row[indices[-1]]))
    result = {
        "translation_rmse": float(
            np.sqrt(np.mean((pred[mask] - truth[mask]) ** 2))
        ),
        "trajectory_ade": float(distances[mask].mean()),
        "trajectory_fde": float(np.mean(final_errors)),
    }
    if rotation_prediction is not None or rotation_target is not None:
        if rotation_prediction is None or rotation_target is None:
            raise ProbeEvaluationError("both rotation prediction/target are required")
        result.update(rotation_metrics_6d(rotation_prediction, rotation_target, mask))
    if gripper_prediction is not None or gripper_target is not None:
        if gripper_prediction is None or gripper_target is None:
            raise ProbeEvaluationError("both gripper prediction/target are required")
        grip_pred = _numpy(gripper_prediction)
        grip_truth = _numpy(gripper_target)
        if grip_pred.shape != grip_truth.shape:
            raise ProbeEvaluationError("gripper prediction/target shapes differ")
        grip_valid = np.broadcast_to(mask[..., None], grip_truth.shape)
        grip_binary = grip_pred >= 0.5
        truth_binary = grip_truth >= 0.5
        true_positive = np.logical_and(grip_binary, truth_binary)[grip_valid].sum()
        false_positive = np.logical_and(grip_binary, ~truth_binary)[grip_valid].sum()
        false_negative = np.logical_and(~grip_binary, truth_binary)[grip_valid].sum()
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        result.update(
            {
                "gripper_mae": float(
                    np.abs(grip_pred[grip_valid] - grip_truth[grip_valid]).mean()
                ),
                "gripper_accuracy": float(
                    (grip_binary[grip_valid] == truth_binary[grip_valid]).mean()
                ),
                "gripper_f1": float(
                    2 * precision * recall / max(1e-12, precision + recall)
                ),
            }
        )
    # Frozen scale-free composite: lower is better, all components disclosed.
    result["se3_trajectory_composite"] = float(
        result["trajectory_ade"]
        + result.get("rotation_geodesic_degrees", 0.0) / 180.0
        + result.get("gripper_mae", 0.0)
    )
    return result


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    replicates: int
    seed: int
    group_count: int


def episode_grouped_bootstrap(
    values: Sequence[float],
    episode_ids: Sequence[str],
    *,
    replicates: int,
    seed: int,
    statistic: Callable[[Sequence[float]], float] | None = None,
) -> BootstrapInterval:
    import numpy as np

    if len(values) != len(episode_ids) or not values:
        raise ProbeEvaluationError("bootstrap values/episode IDs are empty or differ")
    if replicates <= 0 or seed < 0:
        raise ProbeEvaluationError("bootstrap replicates/seed are invalid")
    numeric = [float(value) for value in values]
    if not all(math.isfinite(value) for value in numeric):
        raise ProbeEvaluationError("bootstrap values contain NaN/Inf")
    reducer = statistic or (lambda sample: float(np.mean(sample)))
    groups: dict[str, list[float]] = {}
    for value, episode_id in zip(numeric, episode_ids):
        groups.setdefault(str(episode_id), []).append(value)
    keys = sorted(groups)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(replicates):
        selected = [keys[rng.randrange(len(keys))] for _ in keys]
        flattened = [value for key in selected for value in groups[key]]
        samples.append(float(reducer(flattened)))
    return BootstrapInterval(
        estimate=float(reducer(numeric)),
        lower=float(np.quantile(samples, 0.025)),
        upper=float(np.quantile(samples, 0.975)),
        replicates=replicates,
        seed=seed,
        group_count=len(keys),
    )


def paired_condition_gap(
    clean_values: Mapping[str, float],
    condition_values: Mapping[str, float],
    *,
    episode_by_pair: Mapping[str, str],
    replicates: int,
    seed: int,
) -> BootstrapInterval:
    keys = sorted(set(clean_values) & set(condition_values))
    if set(keys) != set(clean_values) or set(keys) != set(condition_values):
        raise ProbeEvaluationError("paired condition identities do not match exactly")
    if set(keys) != set(episode_by_pair):
        raise ProbeEvaluationError("paired episode mapping identities differ")
    gaps = [float(condition_values[key]) - float(clean_values[key]) for key in keys]
    return episode_grouped_bootstrap(
        gaps,
        [episode_by_pair[key] for key in keys],
        replicates=replicates,
        seed=seed,
    )

