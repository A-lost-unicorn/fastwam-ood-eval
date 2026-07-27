"""Versioned, canonical schemas for Thought 3 identities and manifests."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from fastwam_ood_eval.thought3 import (
    THOUGHT3_CACHE_SCHEMA,
    THOUGHT3_CHECKPOINT_SCHEMA,
    THOUGHT3_SPLIT_SCHEMA,
)


class Thought3SchemaError(ValueError):
    """Raised when an artifact does not satisfy its versioned schema."""


ALLOWED_K = (1, 2, 4)
FUTURE_SOURCE_KIND = "model_sampled_from_current"
NATIVE_FUTURE_LAYOUT = "CTHW"
NATIVE_FUTURE_SHAPE = (48, 2, 14, 28)


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


def _nonempty(value: str, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise Thought3SchemaError(f"{name} must not be empty")
    return normalized


def _hex_digest(value: str, name: str) -> str:
    normalized = _nonempty(value, name).lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef"
        for character in normalized
    ):
        raise Thought3SchemaError(f"{name} must be a 64-character SHA-256 digest")
    return normalized


@dataclass(frozen=True)
class SamplerSchedule:
    """A complete shifted-flow schedule from sigma=1 to sigma=0."""

    k: int
    shift: float
    num_train_timesteps: int
    sigma_nodes: tuple[float, ...]
    timesteps: tuple[float, ...]
    deltas: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.k not in ALLOWED_K:
            raise Thought3SchemaError(f"k must be one of {ALLOWED_K}, got {self.k}")
        if self.shift <= 0 or self.num_train_timesteps <= 0:
            raise Thought3SchemaError("shift and num_train_timesteps must be positive")
        if len(self.sigma_nodes) != self.k + 1:
            raise Thought3SchemaError("sigma_nodes must contain k+1 values")
        if len(self.timesteps) != self.k or len(self.deltas) != self.k:
            raise Thought3SchemaError("timesteps and deltas must contain k values")
        if not math.isclose(self.sigma_nodes[0], 1.0, abs_tol=1e-8):
            raise Thought3SchemaError("schedule must start at sigma=1")
        if not math.isclose(self.sigma_nodes[-1], 0.0, abs_tol=1e-8):
            raise Thought3SchemaError("schedule must finish at sigma=0")
        if any(
            value < 0.0 or value > 1.0
            for value in self.sigma_nodes
        ):
            raise Thought3SchemaError("sigma nodes must remain within [0,1]")
        if any(
            right >= left
            for left, right in zip(self.sigma_nodes, self.sigma_nodes[1:])
        ):
            raise Thought3SchemaError("sigma nodes must be strictly decreasing")
        if any(delta >= 0 for delta in self.deltas):
            raise Thought3SchemaError("all scheduler deltas must be negative")
        expected_deltas = tuple(
            self.sigma_nodes[index + 1] - self.sigma_nodes[index]
            for index in range(self.k)
        )
        if any(
            not math.isclose(observed, expected, abs_tol=1e-8)
            for observed, expected in zip(self.deltas, expected_deltas)
        ):
            raise Thought3SchemaError("scheduler deltas do not match sigma nodes")
        expected_timesteps = tuple(
            value * self.num_train_timesteps
            for value in self.sigma_nodes[:-1]
        )
        if any(
            not math.isclose(observed, expected, abs_tol=1e-6)
            for observed, expected in zip(self.timesteps, expected_timesteps)
        ):
            raise Thought3SchemaError("timesteps do not match sigma nodes")

    @property
    def fingerprint(self) -> str:
        return sha256_canonical(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SamplerSchedule":
        data = dict(payload)
        data["sigma_nodes"] = tuple(float(value) for value in data["sigma_nodes"])
        data["timesteps"] = tuple(float(value) for value in data["timesteps"])
        data["deltas"] = tuple(float(value) for value in data["deltas"])
        return cls(**data)


def build_sampler_schedule(
    k: int,
    *,
    shift: float = 5.0,
    num_train_timesteps: int = 1000,
) -> SamplerSchedule:
    if k not in ALLOWED_K:
        raise Thought3SchemaError(f"k must be one of {ALLOWED_K}, got {k}")
    if shift <= 0:
        raise Thought3SchemaError("shift must be positive")
    u_nodes = tuple((k - index) / k for index in range(k + 1))
    sigma_nodes = tuple(
        (shift * value) / (1.0 + (shift - 1.0) * value)
        for value in u_nodes
    )
    timesteps = tuple(value * float(num_train_timesteps) for value in sigma_nodes[:-1])
    deltas = tuple(
        sigma_nodes[index + 1] - sigma_nodes[index]
        for index in range(k)
    )
    return SamplerSchedule(
        k=k,
        shift=float(shift),
        num_train_timesteps=int(num_train_timesteps),
        sigma_nodes=sigma_nodes,
        timesteps=timesteps,
        deltas=deltas,
    )


@dataclass(frozen=True)
class BaseSampleIdentity:
    dataset_revision: str
    suite: str
    task_id: str
    task_name: str
    demonstration_id: str
    episode_index: int
    frame_index: int
    timestamp_ns: int
    camera_keys: tuple[str, ...]
    language: str
    checkpoint_sha256: str
    stats_sha256: str
    sampler_config_sha256: str
    preprocessing_sha256: str
    split_manifest_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "dataset_revision",
            "suite",
            "task_id",
            "task_name",
            "demonstration_id",
            "language",
        ):
            _nonempty(getattr(self, name), name)
        if self.episode_index < 0 or self.frame_index < 0 or self.timestamp_ns < 0:
            raise Thought3SchemaError("episode/frame/timestamp values must be non-negative")
        if not self.camera_keys or any(not str(key).strip() for key in self.camera_keys):
            raise Thought3SchemaError("camera_keys must contain non-empty camera names")
        if len(set(self.camera_keys)) != len(self.camera_keys):
            raise Thought3SchemaError("camera_keys must not contain duplicates")
        for name in (
            "checkpoint_sha256",
            "stats_sha256",
            "sampler_config_sha256",
            "preprocessing_sha256",
            "split_manifest_sha256",
        ):
            _hex_digest(getattr(self, name), name)

    @property
    def base_sample_id(self) -> str:
        return sha256_canonical(self.to_dict())

    @property
    def episode_id(self) -> str:
        return sha256_canonical(
            {
                "suite": self.suite,
                "task_id": self.task_id,
                "demonstration_id": self.demonstration_id,
                "episode_index": self.episode_index,
            }
        )

    @property
    def language_sha256(self) -> str:
        return hashlib.sha256(self.language.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["camera_keys"] = list(self.camera_keys)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BaseSampleIdentity":
        data = dict(payload)
        data["camera_keys"] = tuple(str(value) for value in data["camera_keys"])
        return cls(**data)


def derive_initial_noise_seed(
    base_sample_id: str,
    global_cache_seed: int,
    *,
    namespace: str = "thought3-noise-v1",
) -> int:
    if global_cache_seed < 0:
        raise Thought3SchemaError("global_cache_seed must be non-negative")
    _hex_digest(base_sample_id, "base_sample_id")
    digest = hashlib.sha256(
        f"{namespace}\0{global_cache_seed}\0{base_sample_id}".encode("utf-8")
    ).digest()
    # Torch accepts a signed 64-bit seed.  Keep the high bit clear.
    return int.from_bytes(digest[:8], "big", signed=False) & ((1 << 63) - 1)


def cache_sample_id(
    *,
    base_sample_id: str,
    k: int,
    initial_noise_seed: int,
    cache_schema: str = THOUGHT3_CACHE_SCHEMA,
) -> str:
    _hex_digest(base_sample_id, "base_sample_id")
    if k not in ALLOWED_K:
        raise Thought3SchemaError(f"k must be one of {ALLOWED_K}, got {k}")
    if initial_noise_seed < 0:
        raise Thought3SchemaError("initial_noise_seed must be non-negative")
    return sha256_canonical(
        {
            "base_sample_id": base_sample_id,
            "cache_schema": cache_schema,
            "initial_noise_seed": int(initial_noise_seed),
            "k": int(k),
        }
    )


@dataclass(frozen=True)
class FutureLatentRecord:
    base_sample_id: str
    cache_sample_id: str
    split: str
    k: int
    initial_noise_seed: int
    schedule: SamplerSchedule
    checkpoint_sha256: str
    stats_sha256: str
    cache_fingerprint: str
    initial_state_sha256: str
    latent_shape: tuple[int, int, int, int] = NATIVE_FUTURE_SHAPE
    latent_layout: str = NATIVE_FUTURE_LAYOUT
    latent_dtype: str = "bfloat16"
    latent_sha256: str | None = None
    generation_latency_ms: float | None = None
    generation_peak_memory_mb: float | None = None
    source_kind: str = FUTURE_SOURCE_KIND
    uses_ground_truth_future: bool = False
    schema_version: str = THOUGHT3_CACHE_SCHEMA

    def __post_init__(self) -> None:
        _hex_digest(self.base_sample_id, "base_sample_id")
        _hex_digest(self.cache_sample_id, "cache_sample_id")
        _hex_digest(self.checkpoint_sha256, "checkpoint_sha256")
        _hex_digest(self.stats_sha256, "stats_sha256")
        _hex_digest(self.cache_fingerprint, "cache_fingerprint")
        _hex_digest(self.initial_state_sha256, "initial_state_sha256")
        if self.latent_sha256 is not None:
            _hex_digest(self.latent_sha256, "latent_sha256")
        if self.initial_noise_seed < 0:
            raise Thought3SchemaError("initial_noise_seed must be non-negative")
        if self.k not in ALLOWED_K or self.schedule.k != self.k:
            raise Thought3SchemaError("record K does not match its scheduler")
        expected_cache_id = cache_sample_id(
            base_sample_id=self.base_sample_id,
            k=self.k,
            initial_noise_seed=self.initial_noise_seed,
            cache_schema=self.schema_version,
        )
        if self.cache_sample_id != expected_cache_id:
            raise Thought3SchemaError("cache_sample_id does not match base ID, K and seed")
        if tuple(self.latent_shape) != NATIVE_FUTURE_SHAPE:
            raise Thought3SchemaError(
                "native future latent shape must be "
                f"{NATIVE_FUTURE_SHAPE}, got {self.latent_shape}"
            )
        if self.latent_layout != NATIVE_FUTURE_LAYOUT:
            raise Thought3SchemaError("native per-sample future latent layout must be CTHW")
        if self.source_kind != FUTURE_SOURCE_KIND or self.uses_ground_truth_future:
            raise Thought3SchemaError(
                "future latent provenance indicates forbidden real-future input"
            )
        if self.latent_dtype not in {"bfloat16", "float32"}:
            raise Thought3SchemaError("cache latent dtype must be bfloat16 or float32")
        for name, value in (
            ("generation_latency_ms", self.generation_latency_ms),
            ("generation_peak_memory_mb", self.generation_peak_memory_mb),
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise Thought3SchemaError(f"{name} must be finite and non-negative")
        if self.schema_version != THOUGHT3_CACHE_SCHEMA:
            raise Thought3SchemaError(f"unsupported cache schema: {self.schema_version}")
        if self.split not in {"train", "development"}:
            raise Thought3SchemaError("cache split must be train or development")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schedule"] = self.schedule.to_dict()
        payload["latent_shape"] = list(self.latent_shape)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FutureLatentRecord":
        data = dict(payload)
        data["schedule"] = SamplerSchedule.from_dict(data["schedule"])
        data["latent_shape"] = tuple(int(value) for value in data["latent_shape"])
        return cls(**data)


@dataclass(frozen=True)
class CachePlanEntry:
    identity: BaseSampleIdentity
    split: str
    k: int
    global_cache_seed: int
    initial_noise_seed: int
    cache_sample_id: str
    shard_index: int

    def __post_init__(self) -> None:
        if self.split not in {"train", "development"}:
            raise Thought3SchemaError("plan split must be train or development")
        if self.k not in ALLOWED_K:
            raise Thought3SchemaError(f"k must be one of {ALLOWED_K}")
        if self.shard_index < 0:
            raise Thought3SchemaError("shard_index must be non-negative")
        if self.global_cache_seed < 0:
            raise Thought3SchemaError("global_cache_seed must be non-negative")
        if self.initial_noise_seed != derive_initial_noise_seed(
            self.identity.base_sample_id,
            self.global_cache_seed,
        ):
            raise Thought3SchemaError("initial_noise_seed does not match the base sample")
        expected_cache_id = cache_sample_id(
            base_sample_id=self.identity.base_sample_id,
            k=self.k,
            initial_noise_seed=self.initial_noise_seed,
        )
        if self.cache_sample_id != expected_cache_id:
            raise Thought3SchemaError("cache_sample_id does not match base sample, K and seed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "base_sample_id": self.identity.base_sample_id,
            "split": self.split,
            "k": self.k,
            "global_cache_seed": self.global_cache_seed,
            "initial_noise_seed": self.initial_noise_seed,
            "cache_sample_id": self.cache_sample_id,
            "shard_index": self.shard_index,
        }

    @classmethod
    def create(
        cls,
        *,
        identity: BaseSampleIdentity,
        split: str,
        k: int,
        global_cache_seed: int,
        shard_index: int,
    ) -> "CachePlanEntry":
        seed = derive_initial_noise_seed(identity.base_sample_id, global_cache_seed)
        return cls(
            identity=identity,
            split=split,
            k=k,
            global_cache_seed=global_cache_seed,
            initial_noise_seed=seed,
            cache_sample_id=cache_sample_id(
                base_sample_id=identity.base_sample_id,
                k=k,
                initial_noise_seed=seed,
            ),
            shard_index=shard_index,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CachePlanEntry":
        data = dict(payload)
        data.pop("base_sample_id", None)
        data["identity"] = BaseSampleIdentity.from_dict(data["identity"])
        return cls(**data)


@dataclass(frozen=True)
class EpisodeDescriptor:
    suite: str
    task_id: str
    task_name: str
    demonstration_id: str
    episode_index: int

    @property
    def episode_id(self) -> str:
        return sha256_canonical(
            {
                "suite": self.suite,
                "task_id": self.task_id,
                "demonstration_id": self.demonstration_id,
                "episode_index": self.episode_index,
            }
        )


@dataclass(frozen=True)
class EpisodeSplitManifest:
    dataset_revision: str
    seed: int
    development_fraction: float
    train_episode_ids: tuple[str, ...]
    development_episode_ids: tuple[str, ...]
    strata: Mapping[str, Mapping[str, int]]
    schema_version: str = THOUGHT3_SPLIT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != THOUGHT3_SPLIT_SCHEMA:
            raise Thought3SchemaError("unsupported split schema")
        if not 0 < self.development_fraction < 1:
            raise Thought3SchemaError("development_fraction must be between 0 and 1")
        if self.seed < 0:
            raise Thought3SchemaError("split seed must be non-negative")
        train = set(self.train_episode_ids)
        development = set(self.development_episode_ids)
        if len(train) != len(self.train_episode_ids) or len(development) != len(
            self.development_episode_ids
        ):
            raise Thought3SchemaError("split episode IDs must be unique")
        if train & development:
            raise Thought3SchemaError("train and development episodes overlap")

    @property
    def fingerprint(self) -> str:
        return sha256_canonical(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_revision": self.dataset_revision,
            "development_episode_ids": list(self.development_episode_ids),
            "development_fraction": self.development_fraction,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "strata": {
                str(key): dict(value)
                for key, value in sorted(self.strata.items())
            },
            "train_episode_ids": list(self.train_episode_ids),
        }


@dataclass(frozen=True)
class AdapterCheckpointManifest:
    backbone_checkpoint_sha256: str
    dataset_stats_sha256: str
    fastwam_commit: str
    adapter_fingerprint: str
    config_fingerprint: str
    split_fingerprint: str
    cache_fingerprint: str
    variant: str
    k: int
    train_seed: int
    global_step: int
    epoch: int
    sample_cursor: int
    trainable_parameter_count: int
    trainable_parameter_names: tuple[str, ...]
    frozen_parameter_sha256: str
    world_size: int
    schema_version: str = THOUGHT3_CHECKPOINT_SCHEMA
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "backbone_checkpoint_sha256",
            "dataset_stats_sha256",
            "adapter_fingerprint",
            "config_fingerprint",
            "split_fingerprint",
            "cache_fingerprint",
            "frozen_parameter_sha256",
        ):
            _hex_digest(getattr(self, name), name)
        if self.schema_version != THOUGHT3_CHECKPOINT_SCHEMA:
            raise Thought3SchemaError("unsupported checkpoint schema")
        if len(self.fastwam_commit) != 40 or any(
            character not in "0123456789abcdef"
            for character in self.fastwam_commit.lower()
        ):
            raise Thought3SchemaError(
                "fastwam_commit must be a 40-character hexadecimal commit"
            )
        if self.variant not in {"A0", "A1", "A2", "A4", "A-shuffle"}:
            raise Thought3SchemaError("adapter checkpoints cannot use B0/unknown variants")
        expected_k = {"A0": 0, "A1": 1, "A2": 2, "A4": 4}
        if self.variant in expected_k and self.k != expected_k[self.variant]:
            raise Thought3SchemaError("checkpoint variant/K mismatch")
        if self.variant == "A-shuffle" and self.k not in ALLOWED_K:
            raise Thought3SchemaError("A-shuffle K must be 1, 2 or 4")
        if min(
            self.train_seed,
            self.global_step,
            self.epoch,
            self.sample_cursor,
            self.trainable_parameter_count,
        ) < 0:
            raise Thought3SchemaError("checkpoint counters/counts must be non-negative")
        if self.world_size <= 0:
            raise Thought3SchemaError("world_size must be positive")
        if len(set(self.trainable_parameter_names)) != len(self.trainable_parameter_names):
            raise Thought3SchemaError("trainable parameter names must be unique")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["trainable_parameter_names"] = list(self.trainable_parameter_names)
        payload["extra"] = dict(self.extra)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AdapterCheckpointManifest":
        data = dict(payload)
        data["trainable_parameter_names"] = tuple(data["trainable_parameter_names"])
        return cls(**data)


def validate_paired_cache_entries(
    entries: Iterable[Mapping[str, Any]],
    *,
    required_k: Sequence[int] = ALLOWED_K,
) -> None:
    grouped: dict[str, dict[int, Mapping[str, Any]]] = {}
    for entry in entries:
        base_id = str(entry["base_sample_id"])
        k = int(entry["k"])
        if k in grouped.setdefault(base_id, {}):
            raise Thought3SchemaError(f"duplicate cache entry for base={base_id}, K={k}")
        grouped[base_id][k] = entry
    required = set(int(value) for value in required_k)
    for base_id, by_k in grouped.items():
        if set(by_k) != required:
            raise Thought3SchemaError(
                f"K pairing mismatch for {base_id}: "
                f"expected {sorted(required)}, got {sorted(by_k)}"
            )
        seeds = {int(value["initial_noise_seed"]) for value in by_k.values()}
        if len(seeds) != 1:
            raise Thought3SchemaError(f"initial noise differs across K for {base_id}")
