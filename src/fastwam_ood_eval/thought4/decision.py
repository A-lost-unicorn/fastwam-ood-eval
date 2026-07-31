"""Frozen evidence-to-method decision rule for Thought4."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from fastwam_ood_eval.thought4 import ALLOWED_METHOD_CLASSIFICATIONS
from fastwam_ood_eval.thought4.schemas import sha256_canonical


class MethodSelectionError(ValueError):
    """Raised when evidence is incomplete or a method class is invalid."""


@dataclass(frozen=True)
class DiagnosticEvidence:
    clean_video_geometry_readable: bool
    camera_video_gap_significant: bool
    camera_gap_larger_than_lighting: bool
    action_geometry_readable: bool
    action_motion_readable: bool
    geometry_subspace_action_sensitive: bool
    robot_init_pattern_distinct_from_camera: bool
    evidence_artifact_sha256: str


RECOMMENDATIONS = {
    "video_geometry_representation_gap": "Geo-REPA",
    "world_action_interface_gap": "SE(3)-Align",
    "camera_equivariance_gap": (
        "Geo-REPA + relative pose / camera-ray equivariance"
    ),
    "geometry_hypothesis_not_supported": "geometry hypothesis not supported",
}


def select_method(evidence: DiagnosticEvidence) -> dict[str, Any]:
    """Apply the preregistered priority rule and emit exactly one class."""

    if not evidence.clean_video_geometry_readable:
        classification = "video_geometry_representation_gap"
        rationale = (
            "Clean Video-side geometry was not reliably readable; the failure "
            "precedes the Video-to-Action interface."
        )
    elif (
        evidence.camera_video_gap_significant
        and evidence.camera_gap_larger_than_lighting
    ):
        classification = "camera_equivariance_gap"
        rationale = (
            "Geometry was readable in Clean but degraded specifically under "
            "exact-state Camera shifts more than Lighting."
        )
    elif (
        (
            not evidence.action_geometry_readable
            or not evidence.action_motion_readable
        )
        and evidence.geometry_subspace_action_sensitive
    ):
        classification = "world_action_interface_gap"
        rationale = (
            "Video geometry was readable and action-sensitive, while Action-side "
            "current geometry and/or future SE(3) structure was not reliably "
            "readable."
        )
    else:
        classification = "geometry_hypothesis_not_supported"
        rationale = (
            "The preregistered representation/interface/equivariance signatures "
            "were not jointly supported by the diagnostic evidence."
        )
    if classification not in ALLOWED_METHOD_CLASSIFICATIONS:
        raise MethodSelectionError("internal invalid method classification")
    payload: dict[str, Any] = {
        "schema_version": "thought4.phase4.method_selection.v1",
        "classification": classification,
        "recommendation": RECOMMENDATIONS[classification],
        "rationale": rationale,
        "evidence": asdict(evidence),
        "decision_rule_version": "thought4.method_rule.v1",
        "claim_boundary": {
            "can_claim": [
                "geometry readability in frozen hidden representations",
                "condition-specific paired probe gaps",
                "action sensitivity to a probe-defined geometry subspace",
            ],
            "cannot_claim": [
                "policy success improvement",
                "causal sufficiency of geometry for Camera OOD",
                "effectiveness of the recommended repair before training/evaluation",
            ],
        },
        "next_branch_only": RECOMMENDATIONS[classification],
    }
    payload["selection_sha256"] = sha256_canonical(payload)
    return payload


def validate_method_selection(payload: Mapping[str, Any]) -> None:
    classification = payload.get("classification")
    if classification not in ALLOWED_METHOD_CLASSIFICATIONS:
        raise MethodSelectionError(
            f"classification must be exactly one of {ALLOWED_METHOD_CLASSIFICATIONS}"
        )
    if payload.get("recommendation") != RECOMMENDATIONS[classification]:
        raise MethodSelectionError("recommendation does not match classification")
    supplied = payload.get("selection_sha256")
    unhashed = dict(payload)
    unhashed.pop("selection_sha256", None)
    if supplied != sha256_canonical(unhashed):
        raise MethodSelectionError("method selection SHA mismatch")


EVIDENCE_THRESHOLDS = {
    "probe_relative_improvement_over_mean": 0.05,
    "probe_relative_improvement_over_shuffle": 0.05,
    "intervention_fraction_above_replay_floor": 0.75,
}


def _feature_proximity(row: Mapping[str, Any]) -> int:
    path = str(row.get("module_path", ""))
    if path.endswith(".head"):
        return 10_000
    layer = row.get("layer_index")
    if layer is not None:
        return int(layer)
    if path.endswith(".action_encoder"):
        return -1
    return -1


def _best_linear_rows(
    payload: Mapping[str, Any],
    *,
    target: str,
) -> tuple[Mapping[str, Any], ...]:
    candidates = [
        row
        for row in payload.get("rows", [])
        if row.get("probe_kind") == "linear" and row.get("target") == target
    ]
    if not candidates:
        raise MethodSelectionError(f"missing linear rows for target={target}")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in candidates:
        grouped.setdefault(str(row["feature_key"]), []).append(row)
    expected_seeds = {int(row["seed"]) for row in candidates}
    summaries: list[
        tuple[float, int, str, str, tuple[Mapping[str, Any], ...]]
    ] = []
    for feature_key, values in grouped.items():
        seeds = [int(row["seed"]) for row in values]
        if len(seeds) != len(set(seeds)) or set(seeds) != expected_seeds:
            raise MethodSelectionError(
                f"incomplete or duplicate seed panel for {feature_key}"
            )
        ordered = tuple(sorted(values, key=lambda row: int(row["seed"])))
        mean_loss = sum(float(row["development_loss"]) for row in ordered) / len(
            ordered
        )
        summaries.append(
            (
                mean_loss,
                -_feature_proximity(ordered[0]),
                str(ordered[0]["module_path"]),
                feature_key,
                ordered,
            )
        )
    # Development-only layer/model selection; test conditions are read only
    # after a complete multi-seed feature group has been chosen.
    return min(summaries, key=lambda value: value[:4])[4]


def _metric_error(metrics: Mapping[str, Any]) -> float:
    def error(metrics: Mapping[str, Any]) -> float:
        for key in (
            "se3_trajectory_composite",
            "rmse",
            "trajectory_ade",
            "rotation_geodesic_degrees",
            "gripper_mae",
            "abs_rel",
        ):
            if key in metrics:
                return float(metrics[key])
        raise MethodSelectionError("probe metrics expose no frozen error scalar")

    return error(metrics)


def _probe_readability(rows: tuple[Mapping[str, Any], ...]) -> dict[str, Any]:
    if not rows:
        raise MethodSelectionError("probe readability needs at least one seed row")
    clean_rows = [row["condition_metrics"]["clean"] for row in rows]
    probe_rmse = sum(_metric_error(row["metrics"]) for row in clean_rows) / len(
        clean_rows
    )
    mean_rmse = sum(
        _metric_error(row["baselines"]["target_mean"]) for row in clean_rows
    ) / len(clean_rows)
    shuffle_rmse = sum(
        _metric_error(row["shuffled_label_control"]) for row in clean_rows
    ) / len(clean_rows)

    threshold = EVIDENCE_THRESHOLDS[
        "probe_relative_improvement_over_mean"
    ]
    shuffle_threshold = EVIDENCE_THRESHOLDS[
        "probe_relative_improvement_over_shuffle"
    ]
    readable = (
        probe_rmse <= mean_rmse * (1.0 - threshold)
        and probe_rmse <= shuffle_rmse * (1.0 - shuffle_threshold)
    )
    return {
        "readable": readable,
        "seed_count": len(rows),
        "probe_error_mean": probe_rmse,
        "target_mean_error_mean": mean_rmse,
        "shuffled_error_mean": shuffle_rmse,
    }


def _paired_gap_summary(
    rows: tuple[Mapping[str, Any], ...], condition: str
) -> dict[str, Any]:
    values = [row["exact_state_paired_rmse_gaps"][condition] for row in rows]
    return {
        "seed_count": len(values),
        "estimate_mean": sum(float(value["estimate"]) for value in values)
        / len(values),
        "lower_min": min(float(value["lower"]) for value in values),
        "upper_max": max(float(value["upper"]) for value in values),
        "all_seed_lower_above_zero": all(
            float(value["lower"]) > 0.0 for value in values
        ),
        "by_seed": [
            {"seed": int(row["seed"]), **dict(value)}
            for row, value in zip(rows, values)
        ],
    }


def derive_diagnostic_evidence(
    video_probe_result: Mapping[str, Any],
    action_probe_result: Mapping[str, Any],
    intervention_result: Mapping[str, Any],
) -> tuple[DiagnosticEvidence, dict[str, Any]]:
    """Derive booleans with frozen thresholds and development-only row selection."""

    video_rows = _best_linear_rows(
        video_probe_result,
        target="eef_object_translation_camera",
    )
    action_rows = _best_linear_rows(
        action_probe_result,
        target="action_se3_trajectory",
    )
    action_geometry_rows = _best_linear_rows(
        action_probe_result,
        target="eef_object_translation_camera",
    )
    camera = _paired_gap_summary(video_rows, "camera")
    lighting = _paired_gap_summary(video_rows, "lighting")
    camera_significant = bool(camera["all_seed_lower_above_zero"])
    camera_larger_by_seed = [
        float(row["exact_state_paired_rmse_gaps"]["camera"]["estimate"])
        > float(row["exact_state_paired_rmse_gaps"]["lighting"]["estimate"])
        for row in video_rows
    ]
    camera_larger = all(camera_larger_by_seed)
    comparisons = int(intervention_result.get("comparison_count", 0))
    above = int(
        intervention_result.get("correct_shuffle_above_floor_count", 0)
    )
    if comparisons <= 0 or above < 0 or above > comparisons:
        raise MethodSelectionError(
            "intervention result has no valid matched comparisons"
        )
    sensitivity_fraction = above / comparisons
    coordinate_shift = intervention_result.get(
        "geometry_coordinate_condition_shift"
    )
    if not isinstance(coordinate_shift, Mapping):
        raise MethodSelectionError(
            "intervention result lacks geometry coordinate condition shift"
        )
    coordinate_unhashed = dict(coordinate_shift)
    coordinate_sha = coordinate_unhashed.pop("result_sha256", None)
    if coordinate_sha != sha256_canonical(coordinate_unhashed):
        raise MethodSelectionError("geometry coordinate-shift SHA mismatch")
    video_robot_gap = sum(
        float(row["gaps_vs_clean_rmse"]["robot_init"]) for row in video_rows
    ) / len(video_rows)
    video_camera_gap = sum(
        float(row["gaps_vs_clean_rmse"]["camera"]) for row in video_rows
    ) / len(video_rows)
    robot_distinct_by_seed = []
    for row in video_rows:
        robot_interval = row["condition_metrics"]["robot_init"][
            "rmse_grouped_bootstrap"
        ]
        camera_interval = row["condition_metrics"]["camera"][
            "rmse_grouped_bootstrap"
        ]
        robot_distinct_by_seed.append(
            float(robot_interval["lower"]) > float(camera_interval["upper"])
            or float(camera_interval["lower"]) > float(robot_interval["upper"])
        )
    video_readability = _probe_readability(video_rows)
    action_readability = _probe_readability(action_rows)
    action_geometry_readability = _probe_readability(action_geometry_rows)
    evidence_payload = {
        "schema_version": "thought4.phase4.evidence_derivation.v1",
        "thresholds": dict(EVIDENCE_THRESHOLDS),
        "selected_video_row_sha256_by_seed": [
            row["row_sha256"] for row in video_rows
        ],
        "selected_action_row_sha256_by_seed": [
            row["row_sha256"] for row in action_rows
        ],
        "selected_action_geometry_row_sha256_by_seed": [
            row["row_sha256"] for row in action_geometry_rows
        ],
        "video_readability": video_readability,
        "action_geometry_readability": action_geometry_readability,
        "action_motion_readability": action_readability,
        "video_clean_readable": video_readability["readable"],
        "action_geometry_clean_readable": action_geometry_readability[
            "readable"
        ],
        "action_clean_readable": action_readability["readable"],
        "camera_paired_gap": camera,
        "lighting_paired_gap": lighting,
        "camera_gap_larger_than_lighting_by_seed": camera_larger_by_seed,
        "intervention_fraction_above_floor": sensitivity_fraction,
        "geometry_coordinate_condition_shift_sha256": coordinate_sha,
        "geometry_coordinate_camera_minus_lighting": coordinate_shift[
            "camera_minus_lighting_paired_grouped_bootstrap"
        ],
        "robot_init_minus_camera_gap": video_robot_gap - video_camera_gap,
        "robot_init_pattern_distinct_by_seed": robot_distinct_by_seed,
        "seed_aggregation_rule": "development_mean_then_metric_mean",
        "row_selection_used_development_only": True,
    }
    evidence_payload["evidence_sha256"] = sha256_canonical(evidence_payload)
    evidence = DiagnosticEvidence(
        clean_video_geometry_readable=bool(
            evidence_payload["video_clean_readable"]
        ),
        camera_video_gap_significant=camera_significant,
        camera_gap_larger_than_lighting=camera_larger,
        action_geometry_readable=bool(
            evidence_payload["action_geometry_clean_readable"]
        ),
        action_motion_readable=bool(evidence_payload["action_clean_readable"]),
        geometry_subspace_action_sensitive=(
            sensitivity_fraction
            >= EVIDENCE_THRESHOLDS[
                "intervention_fraction_above_replay_floor"
            ]
        ),
        robot_init_pattern_distinct_from_camera=all(robot_distinct_by_seed),
        evidence_artifact_sha256=evidence_payload["evidence_sha256"],
    )
    return evidence, evidence_payload
