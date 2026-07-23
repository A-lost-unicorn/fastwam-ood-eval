"""Model-agnostic protocols for opt-in future diagnostics.

The probe is deliberately separate from :class:`BasePolicy`: callers must
continue to execute the action chunk produced by the ordinary policy path and
use this interface only for shadow diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class FutureProbeOutput:
    """Non-serialized output of one shadow future probe.

    Tensor/image values intentionally remain in memory.  Durable result schemas
    store only scalar metrics, metadata, and paths written by the artifact
    layer; they must never JSON-encode these potentially large values.
    """

    predicted_frames: Any
    model_space_input: Any | None = None
    model_space_predicted_frames: Any | None = None
    predicted_latents: Any | None = None
    latency_ms: float = 0.0
    gpu_peak_memory_mb: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SupportsFutureProbe(Protocol):
    """Legacy structural interface for an action-conditioned predictor."""

    def predict_action_conditioned_future(
        self,
        observation: dict,
        actions: Any,
        *,
        diagnostic_seed: int,
        num_video_frames: int | None,
        num_inference_steps: int,
    ) -> FutureProbeOutput:
        """Predict a future without changing the action chunk to be executed."""


@runtime_checkable
class SupportsUnconditionalFutureProbe(Protocol):
    """Structural interface for a current-observation-conditioned predictor."""

    def predict_unconditional_future(
        self,
        observation: dict,
        actions: Any,
        *,
        diagnostic_seed: int,
        num_video_frames: int | None,
        num_inference_steps: int,
    ) -> FutureProbeOutput:
        """Predict without feeding the protected action chunk into the video branch."""


__all__ = [
    "FutureProbeOutput",
    "SupportsFutureProbe",
    "SupportsUnconditionalFutureProbe",
]
