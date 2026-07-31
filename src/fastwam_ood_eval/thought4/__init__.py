"""Thought4: frozen Geometry--Action Gap diagnosis.

This namespace is deliberately independent from Thought1/2/3 artifacts.  It
contains diagnostic feature extraction, lightweight probes and one scoped
geometry-subspace intervention; it does not contain a policy-training method.
"""

from __future__ import annotations

THOUGHT4_CONFIG_SCHEMA = "thought4.phase4.config.v1"
THOUGHT4_COHORT_SCHEMA = "thought4.phase4.cohort.v1"
THOUGHT4_RENDER_SCHEMA = "thought4.phase4.paired_render.v1"
THOUGHT4_FEATURE_SCHEMA = "thought4.phase4.feature.v1"
THOUGHT4_RESULT_SCHEMA = "thought4.phase4.result.v1"

ALLOWED_CONDITIONS = ("clean", "camera", "lighting", "robot_init")
ALLOWED_SPLITS = ("train", "development", "test")
ALLOWED_METHOD_CLASSIFICATIONS = (
    "video_geometry_representation_gap",
    "world_action_interface_gap",
    "camera_equivariance_gap",
    "geometry_hypothesis_not_supported",
)

