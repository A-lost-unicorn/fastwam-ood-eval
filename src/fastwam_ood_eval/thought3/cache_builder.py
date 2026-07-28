"""Resumable, rank-sharded future-cache construction."""

from __future__ import annotations

import hashlib
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from fastwam_ood_eval.thought3 import (
    THOUGHT3_CACHE_SCHEMA,
    THOUGHT3_CACHE_SHARD_SCHEMA,
)
from fastwam_ood_eval.thought3.cache_planner import load_cache_plan
from fastwam_ood_eval.thought3.config import Thought3Config
from fastwam_ood_eval.thought3.future_cache import (
    CacheValidationError,
    atomic_save_safetensors,
    shard_paths,
    validate_cache_shard,
)
from fastwam_ood_eval.thought3.future_sampler import (
    make_mock_future_sampler,
    tensor_sha256,
)
from fastwam_ood_eval.thought3.io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    sha256_file,
)
from fastwam_ood_eval.thought3.safety import ensure_thought3_output_path
from fastwam_ood_eval.thought3.schemas import (
    FUTURE_SOURCE_KIND,
    BaseSampleIdentity,
    CachePlanEntry,
    FutureLatentRecord,
)
from fastwam_ood_eval.thought3.sharding import (
    shard_by_index,
    validate_rank_world_size,
)


def mock_signal(base_sample_id: str) -> float:
    """Stable learnable signal used by mock cache/training, never real data."""

    digest = hashlib.sha256(
        f"thought3-mock-signal-v1\0{base_sample_id}".encode("utf-8")
    ).digest()
    integer = int.from_bytes(digest[:4], "big", signed=False)
    return (integer / ((1 << 32) - 1)) * 1.6 - 0.8


def make_mock_current_latent(identity: BaseSampleIdentity) -> torch.Tensor:
    signal = mock_signal(identity.base_sample_id)
    height = torch.linspace(-0.02, 0.02, 14).view(1, 1, 1, 14, 1)
    width = torch.linspace(-0.02, 0.02, 28).view(1, 1, 1, 1, 28)
    channels = torch.linspace(-0.01, 0.01, 48).view(1, 48, 1, 1, 1)
    return torch.full((1, 48, 1, 14, 28), signal) + height + width + channels


def _nearest_existing_parent(path: Path) -> Path:
    value = path
    while not value.exists():
        if value.parent == value:
            return value
        value = value.parent
    return value


def _check_disk_budget(
    root: Path,
    *,
    estimated_bytes: int,
    reserve_fraction: float,
) -> None:
    usage = shutil.disk_usage(_nearest_existing_parent(root))
    usable = usage.free * (1.0 - reserve_fraction)
    if estimated_bytes > usable:
        raise RuntimeError(
            "insufficient cache disk budget: "
            f"need~{estimated_bytes} bytes, usable={int(usable)} bytes "
            f"after reserve_fraction={reserve_fraction}"
        )


def _group_shards(
    entries: Iterable[CachePlanEntry],
) -> dict[tuple[int, int], list[CachePlanEntry]]:
    grouped: dict[tuple[int, int], list[CachePlanEntry]] = defaultdict(list)
    for entry in entries:
        grouped[(entry.k, entry.shard_index)].append(entry)
    for values in grouped.values():
        values.sort(key=lambda entry: entry.identity.base_sample_id)
    return dict(grouped)


def _build_mock_shard(
    cfg: Thought3Config,
    entries: list[CachePlanEntry],
    *,
    cache_fingerprint: str,
) -> dict[str, Any]:
    if not entries:
        raise ValueError("cannot build an empty cache shard")
    k = entries[0].k
    shard_index = entries[0].shard_index
    if any(entry.k != k or entry.shard_index != shard_index for entry in entries):
        raise ValueError("cache shard entries disagree on K/shard index")
    paths = shard_paths(cfg.cache.root, k, shard_index)
    sampler = make_mock_future_sampler()
    latents: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    dtype = (
        torch.bfloat16
        if cfg.sampler.cache_dtype == "bfloat16"
        else torch.float32
    )
    for tensor_index, entry in enumerate(entries):
        current = make_mock_current_latent(entry.identity)
        sample = sampler.sample(
            current,
            initial_noise_seeds=[entry.initial_noise_seed],
            k=entry.k,
        )
        latent = sample.future_latent[0].to(dtype=dtype).contiguous()
        mask = torch.ones(
            latent.shape[1:],
            dtype=torch.bool,
        )
        record = FutureLatentRecord(
            base_sample_id=entry.identity.base_sample_id,
            cache_sample_id=entry.cache_sample_id,
            split=entry.split,
            k=entry.k,
            initial_noise_seed=entry.initial_noise_seed,
            schedule=sample.schedule,
            checkpoint_sha256=cfg.backbone.checkpoint_sha256,
            stats_sha256=cfg.backbone.dataset_stats_sha256,
            cache_fingerprint=cache_fingerprint,
            initial_state_sha256=sample.initial_state_sha256[0],
            latent_dtype=cfg.sampler.cache_dtype,
            latent_sha256=tensor_sha256(latent),
            generation_latency_ms=sample.latency_ms,
            source_kind=FUTURE_SOURCE_KIND,
            uses_ground_truth_future=False,
        )
        latents.append(latent)
        masks.append(mask)
        rows.append(
            {
                "episode_id": entry.identity.episode_id,
                "identity": entry.identity.to_dict(),
                "record": record.to_dict(),
                "task_id": entry.identity.task_id,
                "tensor_index": tensor_index,
            }
        )

    latent_tensor = torch.stack(latents)
    mask_tensor = torch.stack(masks)
    paths.tensor.parent.mkdir(parents=True, exist_ok=True)
    # Tensor and metadata are staged independently.  The manifest is written
    # last and is the only commit marker recognized by resume/readers.
    atomic_save_safetensors(
        paths.tensor,
        {
            "future_latents": latent_tensor,
            "future_masks": mask_tensor,
        },
        metadata={
            "cache_fingerprint": cache_fingerprint,
            "schema_version": THOUGHT3_CACHE_SCHEMA,
            "source_kind": FUTURE_SOURCE_KIND,
        },
    )
    atomic_write_jsonl(paths.metadata, rows)
    manifest: dict[str, Any] = {
        "schema_version": THOUGHT3_CACHE_SHARD_SCHEMA,
        "cache_schema_version": THOUGHT3_CACHE_SCHEMA,
        "cache_fingerprint": cache_fingerprint,
        "cache_sample_ids": [
            row["record"]["cache_sample_id"] for row in rows
        ],
        "future_source_kind": FUTURE_SOURCE_KIND,
        "k": k,
        "metadata_file": paths.metadata.name,
        "metadata_file_sha256": sha256_file(paths.metadata),
        "sample_count": len(entries),
        "shard_index": shard_index,
        "tensor_file": paths.tensor.name,
        "tensor_file_sha256": sha256_file(paths.tensor),
        "tensor_sha256": {
            "future_latents": tensor_sha256(latent_tensor),
            "future_masks": tensor_sha256(mask_tensor),
        },
        "uses_ground_truth_future": False,
    }
    atomic_write_json(paths.manifest, manifest)
    validate_cache_shard(
        paths,
        expected_cache_fingerprint=cache_fingerprint,
    )
    return manifest


def build_cache(
    cfg: Thought3Config,
    *,
    resume: bool,
    rank: int = 0,
    world_size: int = 1,
    device: str = "cpu",
) -> dict[str, Any]:
    """Build only this rank's whole shards; never split one shard across ranks."""

    validate_rank_world_size(rank, world_size)
    if cfg.runtime.backend == "fastwam":
        from fastwam_ood_eval.thought3.real_cache_builder import (
            build_real_cache,
        )

        return build_real_cache(
            cfg,
            resume=resume,
            rank=rank,
            world_size=world_size,
            device=device,
        )
    if cfg.runtime.backend != "mock":
        raise RuntimeError(
            f"unsupported Thought3 cache backend: {cfg.runtime.backend}"
        )
    if device != "cpu":
        raise RuntimeError("Phase B mock cache build is CPU-only")
    root = ensure_thought3_output_path(cfg.cache.root)
    entries, plan_manifest = load_cache_plan(root)
    if plan_manifest["config_fingerprint"] != cfg.fingerprint:
        raise RuntimeError("cache plan/config fingerprint mismatch")
    grouped = _group_shards(entries)
    all_keys = sorted(grouped)
    assigned_keys = shard_by_index(all_keys, rank, world_size)
    bytes_per_element = 2 if cfg.sampler.cache_dtype == "bfloat16" else 4
    assigned_samples = sum(len(grouped[key]) for key in assigned_keys)
    estimate = (
        assigned_samples
        * math.prod(cfg.sampler.latent_shape)
        * bytes_per_element
    )
    _check_disk_budget(
        root,
        estimated_bytes=estimate,
        reserve_fraction=cfg.cache.required_free_space_fraction,
    )
    built = 0
    skipped = 0
    for key in assigned_keys:
        k, shard_index = key
        paths = shard_paths(root, k, shard_index)
        if paths.manifest.exists():
            try:
                validate_cache_shard(
                    paths,
                    expected_cache_fingerprint=plan_manifest[
                        "cache_fingerprint"
                    ],
                )
            except CacheValidationError as exc:
                raise CacheValidationError(
                    f"resume found corrupt committed shard {paths.manifest}: {exc}"
                ) from exc
            if not resume:
                raise FileExistsError(
                    f"cache shard already exists; pass --resume: {paths.manifest}"
                )
            skipped += 1
            continue
        _build_mock_shard(
            cfg,
            grouped[key],
            cache_fingerprint=plan_manifest["cache_fingerprint"],
        )
        built += 1

    committed = sum(
        shard_paths(root, k, shard_index).manifest.is_file()
        for k, shard_index in all_keys
    )
    result = {
        "assigned_shards": len(assigned_keys),
        "built_shards": built,
        "cache_fingerprint": plan_manifest["cache_fingerprint"],
        "committed_shards": committed,
        "complete": committed == len(all_keys),
        "rank": rank,
        "skipped_valid_shards": skipped,
        "total_shards": len(all_keys),
        "world_size": world_size,
    }
    if result["complete"]:
        atomic_write_json(
            root / "cache_manifest.json",
            {
                "cache_fingerprint": plan_manifest["cache_fingerprint"],
                "cache_schema_version": THOUGHT3_CACHE_SCHEMA,
                "complete": True,
                "config": cfg.to_dict(),
                "config_fingerprint": cfg.fingerprint,
                "plan_manifest_sha256": sha256_file(
                    root / "cache_plan_manifest.json"
                ),
                "total_shards": len(all_keys),
                "world_size": world_size,
            },
        )
    return result
