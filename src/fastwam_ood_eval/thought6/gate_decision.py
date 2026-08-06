"""Frozen Phase 6B/6C gate order and final mechanism classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from fastwam_ood_eval.thought6 import MECHANISM_CLASSIFICATIONS


@dataclass(frozen=True)
class MetricInterval:
    estimate: float
    lower: float
    upper: float


def decide_phase6b(metrics: Mapping[str, MetricInterval | bool]) -> dict[str, object]:
    clean = metrics["fsigma_clean_utility"]
    camera = metrics["fsigma_camera_utility"]
    specificity = metrics["fsigma_shuffle_specificity"]
    timing = metrics["fsigma_minus_f0_utility"]
    if not all(isinstance(value, MetricInterval) for value in (clean, camera, specificity, timing)):
        raise ValueError("Phase 6B metric intervals are incomplete")
    gates = {
        "gate_1_clean_noninferiority": clean.estimate >= 0 and clean.lower > -0.002,
        "gate_2_camera_positive_utility": camera.estimate > 0 and camera.lower > 0,
        "gate_3_correct_content_specificity": (
            specificity.estimate > 0 and specificity.lower > 0
        ),
        "gate_4_timing_benefit": timing.estimate > 0 and timing.lower > 0,
        "gate_5_no_artificial_degradation": bool(metrics["null_b0_bitwise_parity"]),
    }
    unlocked = all(gates.values())
    return {
        "schema_version": "thought6.phase6b_gate_decision.v1",
        "status": "complete",
        "gate_order": list(gates),
        "gates": gates,
        "phase6c_unlocked": unlocked,
        "current_recipe": "active" if unlocked else "stopped",
        "metrics": {
            key: asdict(value) if isinstance(value, MetricInterval) else bool(value)
            for key, value in metrics.items()
        },
    }


def decide_stage2(
    *,
    camera_difference: MetricInterval,
    clean_noninferiority_passed: bool,
    fsigma_better_f0_direction: bool,
) -> dict[str, object]:
    unlock = (
        camera_difference.estimate > 0
        and camera_difference.lower <= 0
        and clean_noninferiority_passed
        and fsigma_better_f0_direction
    )
    return {
        "stage2_unlocked": unlock,
        "reason": (
            "positive_camera_direction_ci_inconclusive_and_other_frozen_conditions_pass"
            if unlock
            else "preregistered_expansion_rule_not_met"
        ),
        "no_automatic_start": True,
    }


def classify_mechanism(
    *,
    phase6b_all_gates: bool,
    camera_significant: bool,
    camera_positive_direction: bool,
    clean_noninferiority: bool,
    fsigma_better_f0: bool,
    correct_better_null: bool,
    correct_better_shuffle: bool,
    oracle_only: bool = False,
) -> dict[str, object]:
    if oracle_only and not phase6b_all_gates:
        classification = "oracle_only_support"
    elif camera_significant and not (
        phase6b_all_gates and correct_better_null and correct_better_shuffle
    ):
        classification = "performance_without_utility_mediation"
    elif phase6b_all_gates and camera_significant and clean_noninferiority and fsigma_better_f0:
        classification = "full_support"
    elif phase6b_all_gates:
        classification = "utility_only_support"
    else:
        classification = "not_supported"
    if not camera_positive_direction and classification == "utility_only_support":
        classification = "utility_only_support"
    if classification not in MECHANISM_CLASSIFICATIONS:
        raise AssertionError("invalid Thought6 mechanism classification")
    return {
        "schema_version": "thought6.mechanism_classification.v1",
        "classification": classification,
        "phase6b_all_gates": phase6b_all_gates,
        "camera_significant": camera_significant,
        "camera_positive_direction": camera_positive_direction,
        "clean_noninferiority": clean_noninferiority,
        "fsigma_better_f0": fsigma_better_f0,
        "correct_better_null": correct_better_null,
        "correct_better_shuffle": correct_better_shuffle,
        "label_oracle_is_diagnostic_only": True,
    }
