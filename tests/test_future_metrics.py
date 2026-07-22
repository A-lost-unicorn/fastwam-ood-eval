from __future__ import annotations

import math

import numpy as np
import pytest

from fastwam_ood_eval.diagnostics.metrics import (
    build_metric_metadata,
    compute_future_latent_metrics,
    compute_future_metrics,
    compute_motion_metrics,
    compute_resource_metrics,
    future_latent_cosine_distance,
    future_latent_l1,
    motion_energy,
)


def test_complete_metrics_match_documented_formulas() -> None:
    embeddings = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])

    metrics = compute_future_metrics(
        embeddings,
        embeddings.copy(),
        static_motion_threshold=1.0,
        generation_latency_ms=12.5,
        generation_peak_memory_mb=256.0,
    )

    assert metrics == {
        "future_latent_l1": 0.0,
        "future_latent_cosine_distance": pytest.approx(0.0),
        "predicted_motion_energy": 3.0,
        "actual_motion_energy": 3.0,
        "motion_energy_ratio": 1.0,
        "motion_direction_cosine": pytest.approx(1.0),
        "static_future_flag": False,
        "predicted_static": False,
        "actual_static": False,
        "diagnostic_latency_ms": 12.5,
        "diagnostic_peak_memory_mb": 256.0,
        "future_generation_latency_ms": 12.5,
        "future_generation_peak_memory_mb": 256.0,
    }
    assert motion_energy(embeddings) == 3.0


def test_latent_metrics_exclude_conditioning_frame_zero() -> None:
    predicted = np.array([[999.0, -999.0], [1.0, 2.0], [3.0, 4.0]])
    actual = np.array([[-999.0, 999.0], [1.0, 2.0], [3.0, 4.0]])

    assert future_latent_l1(predicted, actual) == 0.0
    assert future_latent_cosine_distance(predicted, actual) == pytest.approx(0.0)


def test_fewer_than_two_aligned_frames_is_unavailable() -> None:
    one_frame = np.array([[1.0, 2.0]])

    assert compute_future_latent_metrics(one_frame, one_frame) == {
        "future_latent_l1": None,
        "future_latent_cosine_distance": None,
    }
    assert compute_motion_metrics(
        one_frame,
        one_frame,
        static_motion_threshold=1.0,
    ) == {
        "predicted_motion_energy": None,
        "actual_motion_energy": None,
        "motion_energy_ratio": None,
        "motion_direction_cosine": None,
        "static_future_flag": None,
        "predicted_static": None,
        "actual_static": None,
    }
    assert motion_energy(one_frame) is None


def test_zero_norm_cosines_and_ratios_are_null_not_nan() -> None:
    zeros = np.zeros((3, 4), dtype=np.float32)
    metrics = compute_future_metrics(zeros, zeros, static_motion_threshold=1.0)

    assert metrics["future_latent_l1"] == 0.0
    assert metrics["future_latent_cosine_distance"] is None
    assert metrics["motion_energy_ratio"] is None
    assert metrics["motion_direction_cosine"] is None
    assert metrics["static_future_flag"] is True
    assert metrics["predicted_static"] is True
    assert metrics["actual_static"] is True
    assert not any(
        isinstance(value, float) and math.isnan(value) for value in metrics.values()
    )


def test_motion_direction_can_capture_opposite_change() -> None:
    predicted = np.array([[0.0, 0.0], [1.0, 2.0]])
    actual = np.array([[0.0, 0.0], [-1.0, -2.0]])

    metrics = compute_motion_metrics(
        predicted,
        actual,
        static_motion_threshold=0.0,
    )

    assert metrics["motion_energy_ratio"] == 1.0
    assert metrics["motion_direction_cosine"] == pytest.approx(-1.0)


@pytest.mark.parametrize(
    ("predicted", "actual"),
    [
        (np.zeros((2, 3)), np.zeros((2, 4))),
        (np.array([[0.0], [np.inf]]), np.zeros((2, 1))),
        (np.array(1.0), np.array(1.0)),
    ],
)
def test_invalid_metric_inputs_are_rejected(predicted: np.ndarray, actual: np.ndarray) -> None:
    with pytest.raises(ValueError):
        compute_future_metrics(predicted, actual)


@pytest.mark.parametrize("value", [-1.0, math.inf, math.nan])
def test_invalid_resource_values_are_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        compute_resource_metrics(
            generation_latency_ms=value,
            generation_peak_memory_mb=0.0,
        )


def test_metric_metadata_marks_latent_proxy_as_approximate() -> None:
    metadata = build_metric_metadata(
        static_motion_threshold=1.0,
        motion_value_unit="uint8_pixel_value",
    )

    assert metadata["future_latent"]["status"] == "approximate"
    assert (
        metadata["future_latent"]["reason"]
        == "reencoded_frame_embedding_without_temporal_context"
    )
    assert metadata["future_latent"]["index_zero_excluded"] is True
    assert metadata["motion"]["formula"] == "mean(abs(last_frame - first_frame))"
    assert metadata["motion"]["unit"] == "uint8_pixel_value"
    assert metadata["motion"]["static_motion_threshold"] == 1.0
    assert (
        metadata["motion"]["static_future_flag_definition"]
        == "predicted_motion_energy <= static_motion_threshold"
    )
