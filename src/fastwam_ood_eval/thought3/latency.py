"""Stage-separated latency records for Thought3 online evaluation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class PolicyLatency:
    preprocessing_ms: float
    current_state_encoding_ms: float
    future_sampling_ms: float
    adapter_ms: float
    action_denoising_ms: float
    total_policy_ms: float
    future_decoded_to_video: bool = False

    def __post_init__(self) -> None:
        values = (
            self.preprocessing_ms,
            self.current_state_encoding_ms,
            self.future_sampling_ms,
            self.adapter_ms,
            self.action_denoising_ms,
            self.total_policy_ms,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("latency values must be finite and non-negative")
        if self.future_decoded_to_video:
            raise ValueError(
                "Thought3 online latency protocol keeps future in latent space"
            )

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot summarize empty latency records")
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_latency(
    records: Iterable[PolicyLatency],
) -> dict[str, dict[str, float]]:
    values = list(records)
    if not values:
        raise ValueError("cannot summarize empty latency records")
    fields = (
        "preprocessing_ms",
        "current_state_encoding_ms",
        "future_sampling_ms",
        "adapter_ms",
        "action_denoising_ms",
        "total_policy_ms",
    )
    return {
        field: {
            "mean": sum(getattr(record, field) for record in values) / len(values),
            "p50": _percentile(
                [getattr(record, field) for record in values], 0.50
            ),
            "p95": _percentile(
                [getattr(record, field) for record in values], 0.95
            ),
        }
        for field in fields
    }
