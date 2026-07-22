from __future__ import annotations

import pytest

from fastwam_ood_eval.diagnostics.temporal_alignment import (
    TemporalAlignment,
    build_temporal_alignment,
)


def test_alignment_uses_exact_action_offsets_and_reports_tail() -> None:
    alignment = build_temporal_alignment(
        origin_env_step=30,
        action_video_freq_ratio=4,
        predicted_frame_count=9,
        executed_action_count=10,
    )

    assert [frame.predicted_frame_index for frame in alignment.frames] == [0, 1, 2]
    assert [frame.actual_env_step_offset for frame in alignment.frames] == [0, 4, 8]
    assert [frame.actual_env_step for frame in alignment.frames] == [30, 34, 38]
    assert alignment.aligned_frame_count == 3
    assert alignment.unaligned_tail_action_steps == 2
    assert alignment.truncated is True
    assert alignment.exact_step_mapping is True
    assert alignment.has_aligned_future is True
    assert alignment.timestamp_status == "unavailable"
    assert all(frame.relative_time_seconds is None for frame in alignment.frames)


@pytest.mark.parametrize(
    ("verified", "status", "approximate"),
    [(False, "approximate", True), (True, "exact", False)],
)
def test_timestamp_certainty_is_separate_from_exact_step_mapping(
    verified: bool,
    status: str,
    approximate: bool,
) -> None:
    alignment = build_temporal_alignment(
        origin_env_step=5,
        action_video_freq_ratio=4,
        predicted_frame_count=3,
        executed_action_count=8,
        control_frequency_hz=20.0,
        control_frequency_verified=verified,
    )

    assert [frame.relative_time_seconds for frame in alignment.frames] == [0.0, 0.2, 0.4]
    assert [frame.relative_time_is_approximate for frame in alignment.frames] == [
        approximate,
        approximate,
        approximate,
    ]
    assert alignment.timestamp_status == status
    assert alignment.exact_step_mapping is True


def test_early_termination_truncates_without_fabricating_future_frames() -> None:
    before_first_video_stride = build_temporal_alignment(
        origin_env_step=7,
        action_video_freq_ratio=4,
        predicted_frame_count=3,
        executed_action_count=3,
    )
    at_first_video_stride = build_temporal_alignment(
        origin_env_step=7,
        action_video_freq_ratio=4,
        predicted_frame_count=3,
        executed_action_count=4,
    )

    assert [frame.actual_env_step_offset for frame in before_first_video_stride.frames] == [0]
    assert before_first_video_stride.has_aligned_future is False
    assert before_first_video_stride.unaligned_tail_action_steps == 3
    assert [frame.actual_env_step_offset for frame in at_first_video_stride.frames] == [0, 4]
    assert at_first_video_stride.has_aligned_future is True


def test_alignment_round_trip_is_machine_readable() -> None:
    alignment = TemporalAlignment.build(
        origin_env_step=11,
        action_video_freq_ratio=4,
        predicted_frame_count=4,
        executed_action_count=9,
        control_frequency_hz=10.0,
    )

    restored = TemporalAlignment.from_dict(alignment.to_dict())

    assert restored == alignment
    assert restored.to_dict()["aligned_frame_count"] == 3
    assert restored.to_dict()["has_aligned_future"] is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"origin_env_step": -1},
        {"action_video_freq_ratio": 0},
        {"action_video_freq_ratio": True},
        {"predicted_frame_count": 0},
        {"executed_action_count": -1},
        {"control_frequency_hz": 0.0},
        {"control_frequency_verified": True},
    ],
)
def test_invalid_alignment_inputs_are_rejected(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "origin_env_step": 0,
        "action_video_freq_ratio": 4,
        "predicted_frame_count": 3,
        "executed_action_count": 8,
    }
    values.update(kwargs)
    if kwargs == {"control_frequency_verified": True}:
        values["control_frequency_hz"] = None

    with pytest.raises(ValueError):
        build_temporal_alignment(**values)  # type: ignore[arg-type]
