"""Whole-cache validation, K pairing checks, and corruption detection."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from fastwam_ood_eval.thought3.cache_planner import load_cache_plan
from fastwam_ood_eval.thought3.future_cache import (
    CacheValidationError,
    shard_paths,
    validate_cache_shard,
)
from fastwam_ood_eval.thought3.io_utils import load_jsonl
from fastwam_ood_eval.thought3.safety import ensure_thought3_output_path
from fastwam_ood_eval.thought3.schemas import (
    FutureLatentRecord,
    validate_paired_cache_entries,
)


def validate_cache(
    root: str | Path,
    *,
    verify_tensors: bool = True,
) -> dict[str, Any]:
    cache_root = ensure_thought3_output_path(root)
    entries, plan_manifest = load_cache_plan(cache_root)
    expected: dict[tuple[int, int], list[str]] = defaultdict(list)
    for entry in entries:
        expected[(entry.k, entry.shard_index)].append(entry.cache_sample_id)
    expected_manifest_paths = {
        shard_paths(cache_root, k, shard_index).manifest.resolve()
        for k, shard_index in expected
    }
    observed_manifest_paths = {
        path.resolve()
        for path in cache_root.glob("k*/shard_*.manifest.json")
    }
    if observed_manifest_paths != expected_manifest_paths:
        missing = sorted(
            str(path)
            for path in expected_manifest_paths - observed_manifest_paths
        )
        extra = sorted(
            str(path)
            for path in observed_manifest_paths - expected_manifest_paths
        )
        raise CacheValidationError(
            f"cache shard manifest set differs from plan: missing={missing}, extra={extra}"
        )
    records: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    shard_count = 0
    for (k, shard_index), expected_ids in sorted(expected.items()):
        paths = shard_paths(cache_root, k, shard_index)
        manifest = validate_cache_shard(
            paths,
            expected_cache_fingerprint=plan_manifest["cache_fingerprint"],
            load_tensors=verify_tensors,
        )
        if int(manifest["k"]) != k or int(manifest["shard_index"]) != shard_index:
            raise CacheValidationError("shard key disagrees with path/plan")
        if sorted(manifest["cache_sample_ids"]) != sorted(expected_ids):
            raise CacheValidationError("shard membership differs from cache plan")
        for row in load_jsonl(paths.metadata):
            record = FutureLatentRecord.from_dict(row["record"])
            if record.cache_sample_id in observed_ids:
                raise CacheValidationError(
                    f"duplicate cache sample ID: {record.cache_sample_id}"
                )
            observed_ids.add(record.cache_sample_id)
            records.append(
                {
                    "base_sample_id": record.base_sample_id,
                    "initial_noise_seed": record.initial_noise_seed,
                    "initial_state_sha256": record.initial_state_sha256,
                    "k": record.k,
                }
            )
        shard_count += 1
    expected_ids = {entry.cache_sample_id for entry in entries}
    if observed_ids != expected_ids:
        raise CacheValidationError("cache sample set differs from plan")
    validate_paired_cache_entries(records)
    initial_hashes: dict[str, set[str]] = defaultdict(set)
    for row in records:
        initial_hashes[str(row["base_sample_id"])].add(
            str(row["initial_state_sha256"])
        )
    mismatched = sorted(
        base_id for base_id, hashes in initial_hashes.items() if len(hashes) != 1
    )
    if mismatched:
        raise CacheValidationError(
            f"initial noisy state differs across K: {mismatched[:5]}"
        )
    return {
        "cache_fingerprint": plan_manifest["cache_fingerprint"],
        "entry_count": len(records),
        "paired_k_valid": True,
        "sample_count": int(plan_manifest["sample_count"]),
        "shard_count": shard_count,
        "status": "valid",
        "tensor_checksums_verified": bool(verify_tensors),
        "uses_ground_truth_future": False,
    }
