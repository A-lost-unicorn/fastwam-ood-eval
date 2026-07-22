"""Temporal alignment between predicted video frames and environment steps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class AlignedFutureFrame:
    """One exact step mapping plus an optional time approximation."""

    predicted_frame_index: int
    actual_env_step_offset: int
    actual_env_step: int
    relative_time_seconds: float | None
    relative_time_is_approximate: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AlignedFutureFrame":
        return cls(
            predicted_frame_index=int(data["predicted_frame_index"]),
            actual_env_step_offset=int(data["actual_env_step_offset"]),
            actual_env_step=int(data["actual_env_step"]),
            relative_time_seconds=(
                None
                if data.get("relative_time_seconds") is None
                else float(data["relative_time_seconds"])
            ),
            relative_time_is_approximate=(
                None
                if data.get("relative_time_is_approximate") is None
                else bool(data["relative_time_is_approximate"])
            ),
        )


@dataclass(frozen=True)
class TemporalAlignment:
    """Machine-readable alignment for one replanning probe.

    ``actual_env_step_offset`` is exact and is derived only from the declared
    action/video ratio.  Seconds are kept separate because a configured control
    frequency is not necessarily a runtime-verified simulator frequency.
    """

    origin_env_step: int
    action_video_freq_ratio: int
    predicted_frame_count: int
    executed_action_count: int
    frames: tuple[AlignedFutureFrame, ...]
    truncated: bool
    unaligned_tail_action_steps: int
    exact_step_mapping: bool
    timestamp_status: str
    control_frequency_hz: float | None = None

    @property
    def aligned_frame_count(self) -> int:
        return len(self.frames)

    @property
    def has_aligned_future(self) -> bool:
        return any(frame.predicted_frame_index > 0 for frame in self.frames)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["frames"] = [frame.to_dict() for frame in self.frames]
        payload["aligned_frame_count"] = self.aligned_frame_count
        payload["has_aligned_future"] = self.has_aligned_future
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TemporalAlignment":
        frames = tuple(AlignedFutureFrame.from_dict(item) for item in data.get("frames", ()))
        return cls(
            origin_env_step=int(data["origin_env_step"]),
            action_video_freq_ratio=int(data["action_video_freq_ratio"]),
            predicted_frame_count=int(data["predicted_frame_count"]),
            executed_action_count=int(data["executed_action_count"]),
            frames=frames,
            truncated=bool(data["truncated"]),
            unaligned_tail_action_steps=int(data["unaligned_tail_action_steps"]),
            exact_step_mapping=bool(data.get("exact_step_mapping", True)),
            timestamp_status=str(data.get("timestamp_status", "unavailable")),
            control_frequency_hz=(
                None
                if data.get("control_frequency_hz") is None
                else float(data["control_frequency_hz"])
            ),
        )

    @classmethod
    def build(
        cls,
        *,
        origin_env_step: int,
        action_video_freq_ratio: int,
        predicted_frame_count: int,
        executed_action_count: int,
        control_frequency_hz: float | None = None,
        control_frequency_verified: bool = False,
    ) -> "TemporalAlignment":
        return build_temporal_alignment(
            origin_env_step=origin_env_step,
            action_video_freq_ratio=action_video_freq_ratio,
            predicted_frame_count=predicted_frame_count,
            executed_action_count=executed_action_count,
            control_frequency_hz=control_frequency_hz,
            control_frequency_verified=control_frequency_verified,
        )


def build_temporal_alignment(
    *,
    origin_env_step: int,
    action_video_freq_ratio: int,
    predicted_frame_count: int,
    executed_action_count: int,
    control_frequency_hz: float | None = None,
    control_frequency_verified: bool = False,
) -> TemporalAlignment:
    """Align available predicted frames to observations after executed actions.

    Frame zero is the observation at ``origin_env_step``.  Frame ``i`` maps to
    the observation after exactly ``i * action_video_freq_ratio`` actions.  A
    terminated chunk is truncated rather than padded with fabricated targets.
    """

    if isinstance(origin_env_step, bool) or int(origin_env_step) != origin_env_step:
        raise ValueError("origin_env_step must be an integer")
    if origin_env_step < 0:
        raise ValueError("origin_env_step must be non-negative")
    if isinstance(action_video_freq_ratio, bool) or int(action_video_freq_ratio) != action_video_freq_ratio:
        raise ValueError("action_video_freq_ratio must be an integer")
    if action_video_freq_ratio <= 0:
        raise ValueError("action_video_freq_ratio must be positive")
    if isinstance(predicted_frame_count, bool) or int(predicted_frame_count) != predicted_frame_count:
        raise ValueError("predicted_frame_count must be an integer")
    if predicted_frame_count <= 0:
        raise ValueError("predicted_frame_count must be positive and include frame zero")
    if isinstance(executed_action_count, bool) or int(executed_action_count) != executed_action_count:
        raise ValueError("executed_action_count must be an integer")
    if executed_action_count < 0:
        raise ValueError("executed_action_count must be non-negative")
    if control_frequency_hz is not None and control_frequency_hz <= 0:
        raise ValueError("control_frequency_hz must be positive when provided")
    if control_frequency_verified and control_frequency_hz is None:
        raise ValueError("a verified timestamp requires control_frequency_hz")

    origin = int(origin_env_step)
    ratio = int(action_video_freq_ratio)
    frame_count = int(predicted_frame_count)
    executed = int(executed_action_count)
    timestamp_status = (
        "unavailable"
        if control_frequency_hz is None
        else ("exact" if control_frequency_verified else "approximate")
    )
    frames: list[AlignedFutureFrame] = []
    for prediction_index in range(frame_count):
        offset = prediction_index * ratio
        if offset > executed:
            break
        relative_time = None if control_frequency_hz is None else offset / control_frequency_hz
        frames.append(
            AlignedFutureFrame(
                predicted_frame_index=prediction_index,
                actual_env_step_offset=offset,
                actual_env_step=origin + offset,
                relative_time_seconds=relative_time,
                relative_time_is_approximate=(
                    None if relative_time is None else not control_frequency_verified
                ),
            )
        )

    last_offset = frames[-1].actual_env_step_offset if frames else 0
    return TemporalAlignment(
        origin_env_step=origin,
        action_video_freq_ratio=ratio,
        predicted_frame_count=frame_count,
        executed_action_count=executed,
        frames=tuple(frames),
        truncated=len(frames) < frame_count,
        unaligned_tail_action_steps=max(0, executed - last_offset),
        exact_step_mapping=True,
        timestamp_status=timestamp_status,
        control_frequency_hz=(None if control_frequency_hz is None else float(control_frequency_hz)),
    )


# Concise alias for callers that treat alignment as an operation rather than a constructor.
align_future_frames = build_temporal_alignment

