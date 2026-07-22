from __future__ import annotations

import json

import numpy as np
import pytest

from fastwam_ood_eval.diagnostics.metrics import build_metric_metadata
from fastwam_ood_eval.diagnostics.protocol import FutureProbeOutput, SupportsFutureProbe
from fastwam_ood_eval.diagnostics.temporal_alignment import build_temporal_alignment
from fastwam_ood_eval.schemas.future_diagnostic_result import (
    FutureDiagnosticResult,
    append_future_diagnostic_result,
)


def _result(**overrides: object) -> FutureDiagnosticResult:
    values: dict[str, object] = {
        "schema_version": 1,
        "diagnostic_id": "diag-0",
        "experiment_id": "phase2",
        "source_experiment_id": "phase1",
        "job_id": "job-0",
        "replan_index": 2,
        "origin_env_step": 20,
        "episode_success": False,
        "termination_reason": "max_steps",
        "condition": "ood",
        "perturbation_category": "camera_viewpoints",
        "perturbation_level": "hard",
        "checkpoint_hash": "abc123",
        "mode": "action_conditioned_future",
        "action_conditioned_verified": True,
        "causal_interpretation_allowed": False,
        "inference_seed": 17,
        "diagnostic_seed": 10017,
        "action_hash_before": "before",
        "action_hash_after": "before",
        "action_unchanged": True,
        "executed_action_count": 10,
        "alignment": build_temporal_alignment(
            origin_env_step=20,
            action_video_freq_ratio=4,
            predicted_frame_count=9,
            executed_action_count=10,
        ),
        "metrics": {
            "future_latent_l1": 0.25,
            "future_latent_cosine_distance": None,
            "static_future_flag": False,
            "predicted_static": False,
        },
        "metric_metadata": build_metric_metadata(static_motion_threshold=1.0),
        "predicted_video_path": "diagnostics/diag-0/predicted.mp4",
        "actual_video_path": "diagnostics/diag-0/actual.mp4",
        "side_by_side_video_path": "diagnostics/diag-0/comparison.mp4",
        "latent_path": "diagnostics/diag-0/latents.npz",
        "generation_latency_ms": 42.0,
        "generation_peak_memory_mb": 512.0,
        "status": "completed",
    }
    values.update(overrides)
    return FutureDiagnosticResult(**values)  # type: ignore[arg-type]


def test_future_diagnostic_result_round_trip() -> None:
    result = _result()

    payload = result.to_dict()
    restored = FutureDiagnosticResult.from_dict(payload)

    assert restored.to_dict() == payload
    assert payload["alignment"]["frames"][1]["actual_env_step_offset"] == 4
    assert payload["causal_interpretation_allowed"] is False
    assert payload["action_conditioned_verified"] is True
    assert payload["probe_id"] == payload["diagnostic_id"] == "diag-0"
    assert payload["environment_step"] == payload["origin_env_step"] == 20
    assert payload["action_hash"] == payload["action_hash_before"] == "before"
    assert payload["success"] == payload["episode_success"] is False
    assert payload["approximate_alignment"] is True
    assert payload["static_future_flag"] is False
    assert "predicted_frames" not in payload
    assert "NaN" not in result.to_json()


def test_append_writes_one_json_object_per_probe(tmp_path) -> None:
    output = tmp_path / "future_diagnostics.jsonl"

    append_future_diagnostic_result(output, _result())
    append_future_diagnostic_result(output, _result(diagnostic_id="diag-1", replan_index=3))

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["diagnostic_id"] for row in rows] == ["diag-0", "diag-1"]
    assert all(row["causal_interpretation_allowed"] is False for row in rows)


def test_unknown_fields_are_ignored_for_forward_compatible_reads() -> None:
    payload = _result().to_dict()
    payload["future_schema_field"] = "newer-writer"

    restored = FutureDiagnosticResult.from_dict(payload)

    assert restored.diagnostic_id == "diag-0"


def test_approximate_alignment_requires_both_exact_steps_and_exact_timestamps() -> None:
    exact = _result(
        alignment={"exact_step_mapping": True, "timestamp_status": "exact"},
    )
    inexact_steps = _result(
        alignment={"exact_step_mapping": False, "timestamp_status": "exact"},
    )
    incomplete_action_coverage = _result(
        alignment={
            "exact_step_mapping": True,
            "timestamp_status": "exact",
            "action_conditioning_coverage_complete": False,
        },
    )

    assert exact.approximate_alignment is False
    assert inexact_steps.approximate_alignment is True
    assert incomplete_action_coverage.approximate_alignment is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"causal_interpretation_allowed": True},
        {"metrics": {"bad": float("nan")}},
        {"metrics": {"bad": [1.0]}},
        {"extra": {"raw_video": np.zeros((2, 8, 8, 3), dtype=np.uint8)}},
    ],
)
def test_invalid_or_large_json_values_are_rejected(overrides: dict[str, object]) -> None:
    if "extra" in overrides:
        with pytest.raises(TypeError, match="artifact"):
            _result(**overrides).to_dict()
    else:
        with pytest.raises((TypeError, ValueError)):
            _result(**overrides)


def test_future_probe_protocol_is_structural_and_keeps_tensors_in_memory() -> None:
    class Probe:
        def predict_action_conditioned_future(
            self,
            observation: dict,
            actions: object,
            *,
            diagnostic_seed: int,
            num_video_frames: int | None,
            num_inference_steps: int,
        ) -> FutureProbeOutput:
            return FutureProbeOutput(
                predicted_frames=np.zeros((3, 8, 8, 3), dtype=np.uint8),
                latency_ms=3.0,
                metadata={"diagnostic_seed": diagnostic_seed},
            )

    probe = Probe()
    assert isinstance(probe, SupportsFutureProbe)
    output = probe.predict_action_conditioned_future(
        {},
        np.zeros((10, 7)),
        diagnostic_seed=9,
        num_video_frames=3,
        num_inference_steps=4,
    )
    assert output.predicted_frames.shape == (3, 8, 8, 3)
    assert output.metadata == {"diagnostic_seed": 9}
