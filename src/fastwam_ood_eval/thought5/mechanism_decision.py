"""Preregistered Phase 5 H1/H2/H3 and final classification logic."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from fastwam_ood_eval.thought5 import MECHANISM_CLASSIFICATIONS


@dataclass(frozen=True)
class MechanismEvidence:
    h1_camera_gap_reduction_fraction: float
    h1_paired_ci_upper_below_zero: bool
    h1_task_ci_upper_below_zero: bool
    h1_clean_non_degraded: bool
    h1_lighting_specific: bool
    h2_a1_better_a0: bool
    h2_a1_better_shuffle: bool
    h2_utility_gain_grouped_ci_lower_above_zero: bool
    h2_utility_gain_task_ci_lower_above_zero: bool
    h2_a0_not_abnormally_worse: bool
    h3_camera_gain_grouped_ci_lower_above_zero: bool
    h3_camera_gain_task_ci_lower_above_zero: bool
    h3_clean_noninferior: bool
    h3_camera_specific: bool
    matched_control_explains_gain: bool
    shuffled_control_matches_gain: bool

    @property
    def h1(self) -> bool:
        return (
            self.h1_camera_gap_reduction_fraction >= 0.25
            and self.h1_paired_ci_upper_below_zero
            and self.h1_task_ci_upper_below_zero
            and self.h1_clean_non_degraded
            and self.h1_lighting_specific
        )

    @property
    def h2(self) -> bool:
        return (
            self.h2_a1_better_a0
            and self.h2_a1_better_shuffle
            and self.h2_utility_gain_grouped_ci_lower_above_zero
            and self.h2_utility_gain_task_ci_lower_above_zero
            and self.h2_a0_not_abnormally_worse
        )

    @property
    def h3(self) -> bool:
        return (
            self.h3_camera_gain_grouped_ci_lower_above_zero
            and self.h3_camera_gain_task_ci_lower_above_zero
            and self.h3_clean_noninferior
            and self.h3_camera_specific
        )


def classify_mechanism(evidence: MechanismEvidence) -> dict[str, object]:
    if (
        not evidence.h1
        or evidence.shuffled_control_matches_gain
        or evidence.matched_control_explains_gain
    ):
        classification = "mechanism_not_supported"
    elif evidence.h2 and evidence.h3:
        classification = "full_mechanism_support"
    elif evidence.h2:
        classification = "utility_without_closed_loop_support"
    elif evidence.h3:
        classification = "closed_loop_without_future_mediation"
    else:
        classification = "representation_only_support"
    if classification not in MECHANISM_CLASSIFICATIONS:
        raise AssertionError("invalid mechanism classification")
    return {
        "schema_version": "thought5.phase5.mechanism_classification.v1",
        "classification": classification,
        "h1_supported": evidence.h1,
        "h2_supported": evidence.h2,
        "h3_supported": evidence.h3,
        "evidence": asdict(evidence),
        "causal_scope": (
            "important_mechanism_one_of_multiple_possible_causes"
            if classification == "full_mechanism_support"
            else "no_full_mechanism_claim"
        ),
    }
