"""Leak-resistant sample inventory, episode split, and future-cache plan."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from fastwam_ood_eval.thought3 import (
    THOUGHT3_CACHE_PLAN_SCHEMA,
    THOUGHT3_CACHE_SCHEMA,
)
from fastwam_ood_eval.thought3.config import Thought3Config
from fastwam_ood_eval.thought3.io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    load_json,
    load_jsonl,
    sha256_file,
)
from fastwam_ood_eval.thought3.safety import ensure_thought3_output_path
from fastwam_ood_eval.thought3.schemas import (
    ALLOWED_K,
    FUTURE_SOURCE_KIND,
    BaseSampleIdentity,
    CachePlanEntry,
    EpisodeDescriptor,
    EpisodeSplitManifest,
    sha256_canonical,
)
from fastwam_ood_eval.thought3.training_dataset import (
    build_episode_split,
    episode_inventory_fingerprint,
    split_for_episode,
)


PLAN_FILENAME = "cache_plan.jsonl"
PLAN_MANIFEST_FILENAME = "cache_plan_manifest.json"
SPLIT_FILENAME = "split_manifest.json"


@dataclass(frozen=True)
class InventorySample:
    suite: str
    task_id: str
    task_name: str
    demonstration_id: str
    episode_index: int
    frame_index: int
    timestamp_ns: int
    camera_keys: tuple[str, ...]
    language: str

    def __post_init__(self) -> None:
        if self.episode_index < 0 or self.frame_index < 0 or self.timestamp_ns < 0:
            raise ValueError("inventory indices/timestamp must be non-negative")
        if not self.suite or not self.task_id or not self.demonstration_id:
            raise ValueError("inventory suite/task/demonstration must be non-empty")
        if len(self.camera_keys) != 2 or len(set(self.camera_keys)) != 2:
            raise ValueError("inventory requires two distinct camera keys")

    @property
    def episode(self) -> EpisodeDescriptor:
        return EpisodeDescriptor(
            suite=self.suite,
            task_id=self.task_id,
            task_name=self.task_name,
            demonstration_id=self.demonstration_id,
            episode_index=self.episode_index,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["camera_keys"] = list(self.camera_keys)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InventorySample":
        data = dict(payload)
        data["camera_keys"] = tuple(str(value) for value in data["camera_keys"])
        return cls(**data)


def make_mock_inventory(
    sample_count: int,
    *,
    camera_keys: Sequence[str],
) -> list[InventorySample]:
    if sample_count < 4:
        raise ValueError(
            "mock inventory needs at least four samples (two episodes per task)"
        )
    values: list[InventorySample] = []
    for index in range(sample_count):
        task = index % 2
        episode_index = index // 2
        values.append(
            InventorySample(
                suite="libero_mock",
                task_id=f"task_{task}",
                task_name=f"mock task {task}",
                demonstration_id=f"demo_task{task}_{episode_index:05d}",
                episode_index=episode_index,
                frame_index=0,
                timestamp_ns=index * 100_000_000,
                camera_keys=tuple(str(value) for value in camera_keys),
                language=f"perform mock task {task}",
            )
        )
    return values


def load_inventory(cfg: Thought3Config) -> list[InventorySample]:
    if cfg.runtime.backend == "mock":
        return make_mock_inventory(
            cfg.data.mock_sample_count,
            camera_keys=cfg.data.camera_keys,
        )
    if cfg.data.inventory_path is None:
        raise ValueError("fastwam cache planning requires data.inventory_path")
    rows = load_jsonl(cfg.data.inventory_path)
    return [InventorySample.from_dict(row) for row in rows]


def _unique_episodes(
    samples: Sequence[InventorySample],
) -> list[EpisodeDescriptor]:
    by_id: dict[str, EpisodeDescriptor] = {}
    for sample in samples:
        episode = sample.episode
        by_id.setdefault(episode.episode_id, episode)
    return list(by_id.values())


def _identity_for_sample(
    sample: InventorySample,
    *,
    cfg: Thought3Config,
    split_manifest: EpisodeSplitManifest,
) -> BaseSampleIdentity:
    sampler_hash = sha256_canonical(
        {
            "cache_schema": THOUGHT3_CACHE_SCHEMA,
            "cache_k": list(ALLOWED_K),
            "global_cache_seed": cfg.sampler.global_cache_seed,
            "num_train_timesteps": cfg.sampler.num_train_timesteps,
            "rand_device": cfg.sampler.rand_device,
            "shift": cfg.sampler.shift,
        }
    )
    preprocessing_hash = sha256_canonical(
        {
            "camera_keys": list(sample.camera_keys),
            "image_range": [-1, 1],
            "image_size": [
                cfg.backbone.image_height,
                cfg.backbone.image_width,
            ],
            "input_temporality": "current_observation_only",
            "layout": "BCHW_two_cameras_horizontal",
        }
    )
    return BaseSampleIdentity(
        dataset_revision=cfg.data.dataset_revision,
        suite=sample.suite,
        task_id=sample.task_id,
        task_name=sample.task_name,
        demonstration_id=sample.demonstration_id,
        episode_index=sample.episode_index,
        frame_index=sample.frame_index,
        timestamp_ns=sample.timestamp_ns,
        camera_keys=sample.camera_keys,
        language=sample.language,
        checkpoint_sha256=cfg.backbone.checkpoint_sha256,
        stats_sha256=cfg.backbone.dataset_stats_sha256,
        sampler_config_sha256=sampler_hash,
        preprocessing_sha256=preprocessing_hash,
        split_manifest_sha256=split_manifest.fingerprint,
    )


def create_cache_plan(
    cfg: Thought3Config,
) -> tuple[
    list[CachePlanEntry],
    EpisodeSplitManifest,
    dict[str, Any],
]:
    inventory = load_inventory(cfg)
    episodes = _unique_episodes(inventory)
    split = build_episode_split(
        episodes,
        dataset_revision=cfg.data.dataset_revision,
        seed=cfg.data.split_seed,
        development_fraction=cfg.data.development_fraction,
    )
    identities: list[tuple[BaseSampleIdentity, str]] = []
    for sample in inventory:
        identity = _identity_for_sample(
            sample,
            cfg=cfg,
            split_manifest=split,
        )
        identities.append(
            (identity, split_for_episode(sample.episode.episode_id, split))
        )
    identities.sort(key=lambda value: value[0].base_sample_id)
    if cfg.cache.pilot_limit is not None:
        identities = identities[: cfg.cache.pilot_limit]
    entries: list[CachePlanEntry] = []
    shards_per_k: dict[str, int] = {}
    for k in cfg.sampler.cache_k:
        for index, (identity, sample_split) in enumerate(identities):
            entries.append(
                CachePlanEntry.create(
                    identity=identity,
                    split=sample_split,
                    k=k,
                    global_cache_seed=cfg.sampler.global_cache_seed,
                    shard_index=index // cfg.cache.shard_size,
                )
            )
        shards_per_k[str(k)] = math.ceil(
            len(identities) / cfg.cache.shard_size
        )
    cache_fingerprint = sha256_canonical(
        {
            "cache_schema": THOUGHT3_CACHE_SCHEMA,
            "entries": [entry.to_dict() for entry in entries],
        }
    )
    payload_bytes = (
        len(identities)
        * len(cfg.sampler.cache_k)
        * math.prod(cfg.sampler.latent_shape)
        * (2 if cfg.sampler.cache_dtype == "bfloat16" else 4)
    )
    manifest: dict[str, Any] = {
        "schema_version": THOUGHT3_CACHE_PLAN_SCHEMA,
        "cache_schema_version": THOUGHT3_CACHE_SCHEMA,
        "cache_fingerprint": cache_fingerprint,
        "config": cfg.to_dict(),
        "config_fingerprint": cfg.fingerprint,
        "dataset_revision": cfg.data.dataset_revision,
        "entry_count": len(entries),
        "estimated_latent_payload_bytes": payload_bytes,
        "estimated_latent_payload_gib": payload_bytes / (1024**3),
        "future_source_kind": FUTURE_SOURCE_KIND,
        "inventory_episode_count": len(episodes),
        "inventory_fingerprint": episode_inventory_fingerprint(episodes),
        "k_values": list(cfg.sampler.cache_k),
        "sample_count": len(identities),
        "shard_size": cfg.cache.shard_size,
        "shards_per_k": shards_per_k,
        "split_fingerprint": split.fingerprint,
        "uses_ground_truth_future": False,
    }
    return entries, split, manifest


def write_cache_plan(
    cfg: Thought3Config,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    root = ensure_thought3_output_path(cfg.cache.root)
    entries, split, manifest = create_cache_plan(cfg)
    committed_manifest = root / PLAN_MANIFEST_FILENAME
    if committed_manifest.exists():
        existing_entries, existing = load_cache_plan(root)
        if not resume:
            raise FileExistsError(
                f"cache plan already exists; pass --resume: {committed_manifest}"
            )
        if (
            existing.get("cache_fingerprint") != manifest["cache_fingerprint"]
            or [entry.to_dict() for entry in existing_entries]
            != [entry.to_dict() for entry in entries]
        ):
            raise ValueError(
                "existing cache plan is incompatible with current config"
            )
        return existing
    partial = [
        root / PLAN_FILENAME,
        root / SPLIT_FILENAME,
    ]
    if any(path.exists() for path in partial):
        raise FileExistsError(
            "uncommitted cache plan artifacts exist; inspect them before retrying"
        )
    root.mkdir(parents=True, exist_ok=True)
    split_path = atomic_write_json(root / SPLIT_FILENAME, split.to_dict())
    plan_path = atomic_write_jsonl(
        root / PLAN_FILENAME,
        (entry.to_dict() for entry in entries),
    )
    manifest = {
        **manifest,
        "plan_file": PLAN_FILENAME,
        "plan_sha256": sha256_file(plan_path),
        "split_file": SPLIT_FILENAME,
        "split_sha256": sha256_file(split_path),
    }
    atomic_write_json(root / PLAN_MANIFEST_FILENAME, manifest)
    return manifest


def load_cache_plan(
    root: str | Path,
) -> tuple[list[CachePlanEntry], dict[str, Any]]:
    cache_root = ensure_thought3_output_path(root)
    manifest = load_json(cache_root / PLAN_MANIFEST_FILENAME)
    if manifest.get("schema_version") != THOUGHT3_CACHE_PLAN_SCHEMA:
        raise ValueError("unsupported cache plan schema")
    plan_path = cache_root / str(manifest["plan_file"])
    split_path = cache_root / str(manifest["split_file"])
    if sha256_file(plan_path) != manifest.get("plan_sha256"):
        raise ValueError("cache plan checksum mismatch")
    if not split_path.is_file() or sha256_file(split_path) != manifest.get(
        "split_sha256"
    ):
        raise ValueError("cache split manifest checksum mismatch")
    entries = [
        CachePlanEntry.from_dict(row) for row in load_jsonl(plan_path)
    ]
    if len(entries) != int(manifest["entry_count"]):
        raise ValueError("cache plan entry count mismatch")
    actual_fingerprint = sha256_canonical(
        {
            "cache_schema": THOUGHT3_CACHE_SCHEMA,
            "entries": [entry.to_dict() for entry in entries],
        }
    )
    if actual_fingerprint != manifest.get("cache_fingerprint"):
        raise ValueError("cache plan fingerprint mismatch")
    return entries, manifest
