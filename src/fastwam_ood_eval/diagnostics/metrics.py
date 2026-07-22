"""Pure numerical metrics for future diagnostics.

This module deliberately does not import Fast-WAM.  Callers must encode every
aligned frame with the same frozen VAE before passing the resulting per-frame
embeddings here.  With the released checkpoint those are re-encoded frame
embeddings, so their scientific interpretation remains an approximate proxy.
"""

from __future__ import annotations

import math
from typing import Any


def _as_finite_array(value: Any, *, name: str) -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - the Fast-WAM runtime provides NumPy
        raise RuntimeError("Future diagnostic metrics require numpy") from exc

    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    # NumPy cannot directly materialize a torch.bfloat16 tensor.
    if "bfloat16" in str(getattr(value, "dtype", "")) and hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        raise ValueError(f"{name} must have a leading frame dimension")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _paired_arrays(predicted: Any, actual: Any) -> tuple[Any, Any]:
    predicted_array = _as_finite_array(predicted, name="predicted")
    actual_array = _as_finite_array(actual, name="actual")
    if predicted_array.shape != actual_array.shape:
        raise ValueError(
            "predicted and actual values must have identical shapes, got "
            f"{predicted_array.shape} and {actual_array.shape}"
        )
    return predicted_array, actual_array


def _cosine(left: Any, right: Any, *, epsilon: float) -> float | None:
    import numpy as np

    left_flat = np.asarray(left, dtype=np.float64).reshape(-1)
    right_flat = np.asarray(right, dtype=np.float64).reshape(-1)
    left_norm = float(np.linalg.norm(left_flat))
    right_norm = float(np.linalg.norm(right_flat))
    if left_norm <= epsilon or right_norm <= epsilon:
        return None
    value = float(np.dot(left_flat, right_flat) / (left_norm * right_norm))
    if not math.isfinite(value):
        return None
    return max(-1.0, min(1.0, value))


def _validate_epsilon(epsilon: float) -> float:
    value = float(epsilon)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("epsilon must be a finite positive number")
    return value


def future_latent_l1(predicted_embeddings: Any, actual_embeddings: Any) -> float | None:
    """Mean absolute error over aligned *future* embeddings.

    Index zero is the conditioning/current frame and is always excluded.  The
    remaining frame and feature dimensions are flattened into one paired set.
    ``None`` means that no future frame was available.
    """

    import numpy as np

    predicted, actual = _paired_arrays(predicted_embeddings, actual_embeddings)
    if predicted.shape[0] < 2:
        return None
    return float(np.mean(np.abs(predicted[1:] - actual[1:])))


def future_latent_cosine_distance(
    predicted_embeddings: Any,
    actual_embeddings: Any,
    *,
    epsilon: float = 1e-8,
) -> float | None:
    """Return ``1 - cosine`` over aligned future embeddings, excluding index 0.

    A zero-norm side has no defined cosine and returns ``None`` rather than NaN.
    """

    epsilon = _validate_epsilon(epsilon)
    predicted, actual = _paired_arrays(predicted_embeddings, actual_embeddings)
    if predicted.shape[0] < 2:
        return None
    similarity = _cosine(predicted[1:], actual[1:], epsilon=epsilon)
    return None if similarity is None else 1.0 - similarity


def compute_future_latent_metrics(
    predicted_embeddings: Any,
    actual_embeddings: Any,
    *,
    epsilon: float = 1e-8,
) -> dict[str, float | None]:
    """Compute the two named future-latent proxy metrics."""

    return {
        "future_latent_l1": future_latent_l1(predicted_embeddings, actual_embeddings),
        "future_latent_cosine_distance": future_latent_cosine_distance(
            predicted_embeddings,
            actual_embeddings,
            epsilon=epsilon,
        ),
    }


def motion_energy(frame_embeddings: Any) -> float | None:
    """Return ``mean(abs(last_frame - first_frame))`` in the input unit."""

    import numpy as np

    values = _as_finite_array(frame_embeddings, name="frame_embeddings")
    if values.shape[0] < 2:
        return None
    return float(np.mean(np.abs(values[-1] - values[0])))


def compute_motion_metrics(
    predicted_frame_embeddings: Any,
    actual_frame_embeddings: Any,
    *,
    static_motion_threshold: float,
    epsilon: float = 1e-8,
) -> dict[str, float | bool | None]:
    """Compare end-to-start predicted and actual representation motion.

    Energy is ``mean(abs(last-first))`` and therefore retains the caller's input
    unit.  Direction cosine is computed between the flattened ``last-first``
    deltas.  Ratios and directions with a zero actual/norm denominator are
    returned as ``None``.
    """

    epsilon = _validate_epsilon(epsilon)
    threshold = float(static_motion_threshold)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("static_motion_threshold must be finite and non-negative")
    predicted, actual = _paired_arrays(predicted_frame_embeddings, actual_frame_embeddings)
    if predicted.shape[0] < 2:
        return {
            "predicted_motion_energy": None,
            "actual_motion_energy": None,
            "motion_energy_ratio": None,
            "motion_direction_cosine": None,
            "static_future_flag": None,
            "predicted_static": None,
            "actual_static": None,
        }

    import numpy as np

    predicted_delta = predicted[-1] - predicted[0]
    actual_delta = actual[-1] - actual[0]
    predicted_energy = float(np.mean(np.abs(predicted_delta)))
    actual_energy = float(np.mean(np.abs(actual_delta)))
    ratio = None if actual_energy <= epsilon else predicted_energy / actual_energy
    direction = _cosine(predicted_delta, actual_delta, epsilon=epsilon)
    return {
        "predicted_motion_energy": predicted_energy,
        "actual_motion_energy": actual_energy,
        "motion_energy_ratio": ratio,
        "motion_direction_cosine": direction,
        "static_future_flag": predicted_energy <= threshold,
        "predicted_static": predicted_energy <= threshold,
        "actual_static": actual_energy <= threshold,
    }


def compute_resource_metrics(
    *,
    generation_latency_ms: float | None,
    generation_peak_memory_mb: float | None,
) -> dict[str, float | None]:
    """Validate and expose probe-only latency and memory telemetry."""

    def checked(value: float | None, name: str) -> float | None:
        if value is None:
            return None
        result = float(value)
        if not math.isfinite(result) or result < 0:
            raise ValueError(f"{name} must be finite and non-negative")
        return result

    return {
        "diagnostic_latency_ms": checked(generation_latency_ms, "generation_latency_ms"),
        "diagnostic_peak_memory_mb": checked(
            generation_peak_memory_mb,
            "generation_peak_memory_mb",
        ),
        "future_generation_latency_ms": checked(generation_latency_ms, "generation_latency_ms"),
        "future_generation_peak_memory_mb": checked(
            generation_peak_memory_mb,
            "generation_peak_memory_mb",
        ),
    }


def compute_future_metrics(
    predicted_embeddings: Any,
    actual_embeddings: Any,
    *,
    predicted_motion_embeddings: Any | None = None,
    actual_motion_embeddings: Any | None = None,
    static_motion_threshold: float = 1.0,
    epsilon: float = 1e-8,
    generation_latency_ms: float | None = None,
    generation_peak_memory_mb: float | None = None,
) -> dict[str, float | bool | None]:
    """Return the complete scalar metric record for one aligned probe."""

    motion_predicted = (
        predicted_embeddings if predicted_motion_embeddings is None else predicted_motion_embeddings
    )
    motion_actual = actual_embeddings if actual_motion_embeddings is None else actual_motion_embeddings
    metrics: dict[str, float | bool | None] = {}
    metrics.update(compute_future_latent_metrics(predicted_embeddings, actual_embeddings, epsilon=epsilon))
    metrics.update(
        compute_motion_metrics(
            motion_predicted,
            motion_actual,
            static_motion_threshold=static_motion_threshold,
            epsilon=epsilon,
        )
    )
    metrics.update(
        compute_resource_metrics(
            generation_latency_ms=generation_latency_ms,
            generation_peak_memory_mb=generation_peak_memory_mb,
        )
    )
    return metrics


def build_metric_metadata(
    *,
    static_motion_threshold: float,
    motion_value_unit: str = "input_unit",
) -> dict[str, Any]:
    """Describe formulas and the deliberately limited latent interpretation."""

    threshold = float(static_motion_threshold)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("static_motion_threshold must be finite and non-negative")
    return {
        "future_latent": {
            "status": "approximate",
            "reason": "reencoded_frame_embedding_without_temporal_context",
            "index_zero_excluded": True,
            "l1_formula": "mean(abs(predicted[1:] - actual[1:]))",
            "cosine_distance_formula": "1 - cosine(predicted[1:], actual[1:])",
        },
        "motion": {
            "formula": "mean(abs(last_frame - first_frame))",
            "direction_formula": "cosine(predicted_last-first, actual_last-first)",
            "unit": str(motion_value_unit),
            "static_motion_threshold": threshold,
            "static_future_flag_definition": "predicted_motion_energy <= static_motion_threshold",
            "zero_denominator_result": None,
        },
        "resources": {
            "latency_scope": "shadow_future_generation_only",
            "memory_scope": "shadow_future_generation_peak_only",
        },
    }
