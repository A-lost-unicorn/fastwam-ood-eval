"""Versioned, fail-closed schemas used by the Thought4 diagnosis."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from fastwam_ood_eval.thought4 import (
    ALLOWED_CONDITIONS,
    ALLOWED_SPLITS,
    THOUGHT4_COHORT_SCHEMA,
    THOUGHT4_FEATURE_SCHEMA,
    THOUGHT4_RENDER_SCHEMA,
)


class Thought4SchemaError(ValueError):
    """Raised when a Thought4 artifact violates its frozen schema."""


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_canonical(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def require_finite(values: Any, name: str) -> None:
    """Fail closed for nested numeric payloads without importing NumPy/Torch."""

    if hasattr(values, "isfinite"):
        result = values.isfinite()
        ok = bool(result.all().item()) if hasattr(result, "all") else bool(result)
        if not ok:
            raise Thought4SchemaError(f"{name} contains NaN/Inf")
        return
    if isinstance(values, Mapping):
        for key, value in values.items():
            require_finite(value, f"{name}.{key}")
        return
    if isinstance(values, (list, tuple)):
        for index, value in enumerate(values):
            require_finite(value, f"{name}[{index}]")
        return
    if isinstance(values, float) and not math.isfinite(values):
        raise Thought4SchemaError(f"{name} contains NaN/Inf")


def _nonempty(value: str, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise Thought4SchemaError(f"{name} must not be empty")
    return normalized


def _digest(value: str, name: str) -> str:
    normalized = _nonempty(value, name).lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise Thought4SchemaError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _matrix(
    value: Sequence[Sequence[float]],
    shape: tuple[int, int],
    name: str,
) -> tuple[tuple[float, ...], ...]:
    rows = tuple(tuple(float(item) for item in row) for row in value)
    if len(rows) != shape[0] or any(len(row) != shape[1] for row in rows):
        raise Thought4SchemaError(f"{name} must have shape {shape}")
    require_finite(rows, name)
    return rows


@dataclass(frozen=True)
class SampleIdentity:
    """Identity at input time ``t``; future labels must never alter this key."""

    task_id: str
    episode_id: str
    frame_index: int
    split: str
    timestamp: float
    label_identity: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _nonempty(self.task_id, "task_id"))
        object.__setattr__(self, "episode_id", _nonempty(self.episode_id, "episode_id"))
        object.__setattr__(
            self, "label_identity", _nonempty(self.label_identity, "label_identity")
        )
        if self.frame_index < 0:
            raise Thought4SchemaError("frame_index must be non-negative")
        if self.split not in ALLOWED_SPLITS:
            raise Thought4SchemaError(f"invalid split: {self.split}")
        if not math.isfinite(float(self.timestamp)):
            raise Thought4SchemaError("timestamp must be finite")

    @property
    def sample_id(self) -> str:
        return sha256_canonical(asdict(self))


@dataclass(frozen=True)
class CohortRecord:
    identity: SampleIdentity
    simulator_state_sha256: str
    simulator_state_locator: str
    episode_seed_namespace: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "simulator_state_sha256",
            _digest(self.simulator_state_sha256, "simulator_state_sha256"),
        )
        _nonempty(self.simulator_state_locator, "simulator_state_locator")
        _nonempty(self.episode_seed_namespace, "episode_seed_namespace")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": THOUGHT4_COHORT_SCHEMA,
            "identity": asdict(self.identity),
            "sample_id": self.identity.sample_id,
            "simulator_state_sha256": self.simulator_state_sha256,
            "simulator_state_locator": self.simulator_state_locator,
            "episode_seed_namespace": self.episode_seed_namespace,
        }


@dataclass(frozen=True)
class CameraMetadata:
    camera_name: str
    intrinsic: tuple[tuple[float, ...], ...]
    extrinsic_camera_to_world: tuple[tuple[float, ...], ...]

    @classmethod
    def from_values(
        cls,
        camera_name: str,
        intrinsic: Sequence[Sequence[float]],
        extrinsic_camera_to_world: Sequence[Sequence[float]],
    ) -> "CameraMetadata":
        return cls(
            camera_name=_nonempty(camera_name, "camera_name"),
            intrinsic=_matrix(intrinsic, (3, 3), "camera.intrinsic"),
            extrinsic_camera_to_world=_matrix(
                extrinsic_camera_to_world,
                (4, 4),
                "camera.extrinsic_camera_to_world",
            ),
        )

    @property
    def identity_sha256(self) -> str:
        return sha256_canonical(asdict(self))


@dataclass(frozen=True)
class PairedRenderRecord:
    """One rendered condition with explicit exact-state semantics."""

    identity: SampleIdentity
    condition: str
    condition_variant: str
    exact_state_pair: bool
    clean_reference_sample_id: str
    clean_reference_state_sha256: str
    simulator_state_sha256: str
    object_eef_state_sha256: str
    rgb_sha256: str
    depth_sha256: str
    camera: CameraMetadata
    lighting_config_sha256: str

    def __post_init__(self) -> None:
        if self.condition not in ALLOWED_CONDITIONS:
            raise Thought4SchemaError(f"invalid condition: {self.condition}")
        for name in (
            "clean_reference_sample_id",
            "clean_reference_state_sha256",
            "simulator_state_sha256",
            "object_eef_state_sha256",
            "rgb_sha256",
            "depth_sha256",
            "lighting_config_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        _nonempty(self.condition_variant, "condition_variant")
        if self.condition == "robot_init" and self.exact_state_pair:
            raise Thought4SchemaError(
                "robot_init changes physical state and cannot be an exact-state pair"
            )
        if self.condition in {"clean", "camera", "lighting"}:
            if not self.exact_state_pair:
                raise Thought4SchemaError(
                    f"{self.condition} must use exact-state paired rendering"
                )
            if self.simulator_state_sha256 != self.clean_reference_state_sha256:
                raise Thought4SchemaError(
                    f"{self.condition} state hash differs from clean reference"
                )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": THOUGHT4_RENDER_SCHEMA,
            "identity": asdict(self.identity),
            "sample_id": self.identity.sample_id,
            "condition": self.condition,
            "condition_variant": self.condition_variant,
            "exact_state_pair": self.exact_state_pair,
            "clean_reference_sample_id": self.clean_reference_sample_id,
            "clean_reference_state_sha256": self.clean_reference_state_sha256,
            "simulator_state_sha256": self.simulator_state_sha256,
            "object_eef_state_sha256": self.object_eef_state_sha256,
            "rgb_sha256": self.rgb_sha256,
            "depth_sha256": self.depth_sha256,
            "camera": {
                **asdict(self.camera),
                "identity_sha256": self.camera.identity_sha256,
            },
            "lighting_config_sha256": self.lighting_config_sha256,
        }
        payload["record_sha256"] = sha256_canonical(payload)
        return payload


@dataclass(frozen=True)
class FeatureRecord:
    identity: SampleIdentity
    condition: str
    source: str
    module_path: str
    layer_index: int | None
    denoise_step_index: int | None
    pooling: str
    tensor_shape: tuple[int, ...]
    tensor_dtype: str
    feature_sha256: str
    shard_path: str
    shard_sha256: str

    def __post_init__(self) -> None:
        if self.condition not in ALLOWED_CONDITIONS:
            raise Thought4SchemaError(f"invalid condition: {self.condition}")
        if self.source not in {"A", "B"}:
            raise Thought4SchemaError("feature source must be A or B")
        _nonempty(self.module_path, "module_path")
        _nonempty(self.pooling, "pooling")
        _nonempty(self.tensor_dtype, "tensor_dtype")
        if not self.tensor_shape or any(int(value) <= 0 for value in self.tensor_shape):
            raise Thought4SchemaError("tensor_shape must contain positive dimensions")
        if self.layer_index is not None and self.layer_index < 0:
            raise Thought4SchemaError("layer_index must be non-negative")
        if self.denoise_step_index is not None and self.denoise_step_index < 0:
            raise Thought4SchemaError("denoise_step_index must be non-negative")
        if self.source == "A" and self.denoise_step_index is not None:
            raise Thought4SchemaError(
                "Video source A must not have an action denoise-step index"
            )
        if self.source == "B" and self.denoise_step_index is None:
            raise Thought4SchemaError(
                "Action source B must record its denoise-step index"
            )
        object.__setattr__(
            self, "feature_sha256", _digest(self.feature_sha256, "feature_sha256")
        )
        object.__setattr__(
            self, "shard_sha256", _digest(self.shard_sha256, "shard_sha256")
        )
        _nonempty(self.shard_path, "shard_path")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": THOUGHT4_FEATURE_SCHEMA,
            "identity": asdict(self.identity),
            "sample_id": self.identity.sample_id,
            **{
                key: value
                for key, value in asdict(self).items()
                if key != "identity"
            },
        }
        payload["record_sha256"] = sha256_canonical(payload)
        return payload


def validate_episode_split(records: Iterable[SampleIdentity]) -> None:
    """Ensure no episode contributes frames to more than one split."""

    seen: dict[tuple[str, str], str] = {}
    count = 0
    for record in records:
        count += 1
        key = (record.task_id, record.episode_id)
        prior = seen.setdefault(key, record.split)
        if prior != record.split:
            raise Thought4SchemaError(
                f"episode split leakage for task/episode={key}: {prior} vs {record.split}"
            )
    if count == 0:
        raise Thought4SchemaError("cohort must not be empty")


def deterministic_episode_split(
    episode_ids: Iterable[tuple[str, str]],
    *,
    seed: int,
    train_count: int,
    development_count: int,
    test_count: int,
) -> dict[tuple[str, str], str]:
    """Outcome-blind, stable episode split based only on identity and seed."""

    if seed < 0 or min(train_count, development_count, test_count) < 0:
        raise Thought4SchemaError("split seed/counts must be non-negative")
    unique = sorted(set(episode_ids))
    required = train_count + development_count + test_count
    if len(unique) < required:
        raise Thought4SchemaError(
            f"need {required} unique episodes, found {len(unique)}"
        )
    ordered = sorted(
        unique,
        key=lambda value: hashlib.sha256(
            f"thought4-episode-split-v1\0{seed}\0{value[0]}\0{value[1]}".encode()
        ).hexdigest(),
    )[:required]
    result: dict[tuple[str, str], str] = {}
    boundaries = (train_count, train_count + development_count)
    for index, key in enumerate(ordered):
        result[key] = (
            "train"
            if index < boundaries[0]
            else "development"
            if index < boundaries[1]
            else "test"
        )
    return result
