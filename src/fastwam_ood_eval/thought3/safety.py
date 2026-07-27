"""Filesystem and information-boundary guards for Thought 3."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping


class Thought3SafetyError(ValueError):
    """Raised when a path or batch would cross a frozen experiment boundary."""


FORBIDDEN_TRAINING_KEYS = frozenset(
    {
        "actual_future",
        "actual_future_frames",
        "environment_future",
        "future_frames",
        "future_observation",
        "future_observations",
        "gt_future",
        "gt_future_latent",
        "next_image",
        "next_observation",
        "next_observations",
        "success",
        "termination_reason",
    }
)

ALLOWED_TRAINING_KEYS = frozenset(
    {
        "sample_id",
        "base_sample_id",
        "current_rgb",
        "current_proprio",
        "context",
        "context_mask",
        "target_action",
        "action_is_pad",
        "future_latent",
        "future_mask",
        "metadata",
    }
)


def _parts_lower(path: Path) -> tuple[str, ...]:
    return tuple(part.casefold() for part in path.resolve().parts)


def _contains_sequence(parts: tuple[str, ...], sequence: tuple[str, ...]) -> bool:
    size = len(sequence)
    return any(parts[index : index + size] == sequence for index in range(len(parts) - size + 1))


def ensure_thought3_output_path(path: str | Path) -> Path:
    """Require an isolated Thought 3 path and reject frozen/upstream trees.

    Requiring a literal ``thought3`` path component also keeps temporary test
    outputs honest while allowing tests to run outside the repository.
    """

    resolved = Path(path).resolve()
    parts = _parts_lower(resolved)
    forbidden = (
        ("outputs", "thought1"),
        ("outputs", "thought2"),
        ("third_party",),
    )
    for sequence in forbidden:
        if _contains_sequence(parts, sequence):
            raise Thought3SafetyError(
                f"Thought3 refuses to write inside frozen/upstream path: {resolved}"
            )
    if "thought3" not in parts:
        raise Thought3SafetyError(
            f"Thought3 output paths require a literal 'thought3' component: {resolved}"
        )
    return resolved


def ensure_standard_training_source(path: str | Path) -> Path:
    """Reject test rollouts, LIBERO-Plus and frozen result trees as training data."""

    resolved = Path(path).resolve()
    parts = _parts_lower(resolved)
    forbidden = (
        ("outputs", "thought1"),
        ("outputs", "thought2"),
        ("outputs", "thought3"),
        ("libero-plus",),
        ("libero_plus",),
    )
    for sequence in forbidden:
        if _contains_sequence(parts, sequence):
            raise Thought3SafetyError(
                f"Thought3 training data must be standard LIBERO demonstrations: {resolved}"
            )
    return resolved


def validate_training_batch_keys(
    batch: Mapping[str, object],
    *,
    require_future: bool = True,
) -> None:
    """Enforce an allowlist so real future observations cannot reach the model."""

    keys = {str(key) for key in batch}
    forbidden = sorted(keys & FORBIDDEN_TRAINING_KEYS)
    unknown = sorted(keys - ALLOWED_TRAINING_KEYS)
    if forbidden:
        raise Thought3SafetyError(
            f"Training batch contains forbidden post-execution/future fields: {forbidden}"
        )
    if unknown:
        raise Thought3SafetyError(
            f"Training batch contains fields outside the Thought3 allowlist: {unknown}"
        )
    required = {
        "current_rgb",
        "current_proprio",
        "context",
        "context_mask",
        "target_action",
        "action_is_pad",
    }
    if require_future:
        required.update({"future_latent", "future_mask"})
    missing = sorted(required - keys)
    if missing:
        raise Thought3SafetyError(f"Training batch is missing required fields: {missing}")


def validate_no_forbidden_sources(paths: Iterable[str | Path]) -> tuple[Path, ...]:
    return tuple(ensure_standard_training_source(path) for path in paths)
