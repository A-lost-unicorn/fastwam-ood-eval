"""Sharded safetensors cache format and random-access reader."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from fastwam_ood_eval.thought3 import THOUGHT3_CACHE_SHARD_SCHEMA
from fastwam_ood_eval.thought3.future_sampler import tensor_sha256
from fastwam_ood_eval.thought3.io_utils import (
    load_json,
    load_jsonl,
    sha256_file,
)
from fastwam_ood_eval.thought3.safety import ensure_thought3_output_path
from fastwam_ood_eval.thought3.schemas import (
    BaseSampleIdentity,
    FutureLatentRecord,
    NATIVE_FUTURE_SHAPE,
)


class CacheValidationError(RuntimeError):
    """Raised when a committed cache artifact is missing or corrupt."""


@dataclass(frozen=True)
class ShardPaths:
    tensor: Path
    metadata: Path
    manifest: Path


def shard_paths(root: str | Path, k: int, shard_index: int) -> ShardPaths:
    cache_root = ensure_thought3_output_path(root)
    directory = cache_root / f"k{k}"
    stem = f"shard_{shard_index:06d}"
    return ShardPaths(
        tensor=directory / f"{stem}.safetensors",
        metadata=directory / f"{stem}.metadata.jsonl",
        manifest=directory / f"{stem}.manifest.json",
    )


def atomic_save_safetensors(
    path: str | Path,
    tensors: Mapping[str, Tensor],
    *,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    target = Path(path)
    ensure_thought3_output_path(target.parent)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(
            {
                str(name): tensor.detach().cpu().contiguous()
                for name, tensor in tensors.items()
            },
            str(temporary),
            metadata=dict(metadata or {}),
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _load_and_check_rows(
    paths: ShardPaths,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = load_jsonl(paths.metadata)
    if len(rows) != int(manifest["sample_count"]):
        raise CacheValidationError("metadata sample count mismatch")
    expected_ids = [str(value) for value in manifest["cache_sample_ids"]]
    observed_ids: list[str] = []
    for expected_index, row in enumerate(rows):
        if int(row.get("tensor_index", -1)) != expected_index:
            raise CacheValidationError("metadata tensor_index is not contiguous")
        try:
            identity = BaseSampleIdentity.from_dict(row["identity"])
            record = FutureLatentRecord.from_dict(row["record"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CacheValidationError(
                f"invalid cache metadata row {expected_index}: {exc}"
            ) from exc
        if identity.base_sample_id != record.base_sample_id:
            raise CacheValidationError("record/identity base_sample_id mismatch")
        if int(record.k) != int(manifest["k"]):
            raise CacheValidationError("record K differs from shard manifest")
        if record.cache_fingerprint != manifest["cache_fingerprint"]:
            raise CacheValidationError(
                "record cache fingerprint differs from shard manifest"
            )
        if row.get("episode_id") != identity.episode_id:
            raise CacheValidationError("metadata episode_id mismatch")
        if row.get("task_id") != identity.task_id:
            raise CacheValidationError("metadata task_id mismatch")
        observed_ids.append(record.cache_sample_id)
    if observed_ids != expected_ids:
        raise CacheValidationError("metadata cache_sample_id ordering mismatch")
    return rows


def validate_cache_shard(
    paths: ShardPaths,
    *,
    expected_cache_fingerprint: str | None = None,
    load_tensors: bool = True,
) -> dict[str, Any]:
    if not paths.manifest.is_file():
        raise CacheValidationError(f"missing shard commit manifest: {paths.manifest}")
    manifest = load_json(paths.manifest)
    if manifest.get("schema_version") != THOUGHT3_CACHE_SHARD_SCHEMA:
        raise CacheValidationError("unsupported cache shard schema")
    if paths.tensor.name != manifest.get("tensor_file"):
        raise CacheValidationError("tensor filename differs from shard manifest")
    if paths.metadata.name != manifest.get("metadata_file"):
        raise CacheValidationError("metadata filename differs from shard manifest")
    for artifact, expected, label in (
        (paths.tensor, manifest.get("tensor_file_sha256"), "tensor file"),
        (paths.metadata, manifest.get("metadata_file_sha256"), "metadata file"),
    ):
        if not artifact.is_file():
            raise CacheValidationError(f"missing {label}: {artifact}")
        if sha256_file(artifact) != expected:
            raise CacheValidationError(f"{label} checksum mismatch: {artifact}")
    if (
        expected_cache_fingerprint is not None
        and manifest.get("cache_fingerprint") != expected_cache_fingerprint
    ):
        raise CacheValidationError("shard cache fingerprint mismatch")
    rows = _load_and_check_rows(paths, manifest)
    if load_tensors:
        try:
            tensors = load_file(str(paths.tensor), device="cpu")
        except Exception as exc:
            raise CacheValidationError(
                f"cannot decode safetensors shard: {paths.tensor}: {exc}"
            ) from exc
        if set(tensors) != {"future_latents", "future_masks"}:
            raise CacheValidationError("unexpected tensor names in cache shard")
        sample_count = int(manifest["sample_count"])
        expected_latent_shape = (sample_count, *NATIVE_FUTURE_SHAPE)
        if tuple(tensors["future_latents"].shape) != expected_latent_shape:
            raise CacheValidationError(
                "future_latents shape mismatch: "
                f"expected {expected_latent_shape}, "
                f"got {tuple(tensors['future_latents'].shape)}"
            )
        expected_mask_shape = (
            sample_count,
            NATIVE_FUTURE_SHAPE[1],
            NATIVE_FUTURE_SHAPE[2],
            NATIVE_FUTURE_SHAPE[3],
        )
        if tuple(tensors["future_masks"].shape) != expected_mask_shape:
            raise CacheValidationError("future_masks shape mismatch")
        for name, tensor in tensors.items():
            if tensor_sha256(tensor) != manifest["tensor_sha256"][name]:
                raise CacheValidationError(f"{name} tensor checksum mismatch")
        for index, row in enumerate(rows):
            record = FutureLatentRecord.from_dict(row["record"])
            if tensor_sha256(tensors["future_latents"][index]) != record.latent_sha256:
                raise CacheValidationError(
                    f"per-sample latent checksum mismatch at index {index}"
                )
            if not tensors["future_masks"][index].all():
                raise CacheValidationError("native mock cache mask must be fully valid")
    return manifest


class FutureCacheReader:
    """Validated cache reader keyed by stable base sample ID and K."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_cache_fingerprint: str | None = None,
        validate: bool = True,
    ) -> None:
        self.root = ensure_thought3_output_path(root)
        self.expected_cache_fingerprint = expected_cache_fingerprint
        self._index: dict[tuple[str, int], tuple[ShardPaths, int, dict[str, Any]]] = {}
        self._loaded_path: Path | None = None
        self._loaded_tensors: dict[str, Tensor] | None = None
        manifests = sorted(self.root.glob("k*/shard_*.manifest.json"))
        if not manifests:
            raise CacheValidationError(f"no committed cache shards under {self.root}")
        for manifest_path in manifests:
            manifest = load_json(manifest_path)
            paths = ShardPaths(
                tensor=manifest_path.parent / str(manifest["tensor_file"]),
                metadata=manifest_path.parent / str(manifest["metadata_file"]),
                manifest=manifest_path,
            )
            if validate:
                validate_cache_shard(
                    paths,
                    expected_cache_fingerprint=expected_cache_fingerprint,
                )
            rows = load_jsonl(paths.metadata)
            for index, row in enumerate(rows):
                record = FutureLatentRecord.from_dict(row["record"])
                key = (record.base_sample_id, record.k)
                if key in self._index:
                    raise CacheValidationError(
                        f"duplicate cached base/K entry: {key}"
                    )
                self._index[key] = (paths, index, row)

    @property
    def keys(self) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(self._index))

    def metadata(self, base_sample_id: str, k: int) -> dict[str, Any]:
        key = (base_sample_id, int(k))
        if key not in self._index:
            available = sorted(
                observed_k
                for observed_base, observed_k in self._index
                if observed_base == base_sample_id
            )
            raise CacheValidationError(
                f"cache K mismatch/missing entry for base={base_sample_id}, "
                f"requested K={k}, available={available}"
            )
        return self._index[key][2]

    def get(
        self,
        base_sample_id: str,
        k: int,
    ) -> tuple[Tensor, Tensor, dict[str, Any]]:
        row = self.metadata(base_sample_id, k)
        paths, tensor_index, _ = self._index[(base_sample_id, int(k))]
        if self._loaded_path != paths.tensor:
            self._loaded_tensors = load_file(str(paths.tensor), device="cpu")
            self._loaded_path = paths.tensor
        assert self._loaded_tensors is not None
        return (
            self._loaded_tensors["future_latents"][tensor_index].clone(),
            self._loaded_tensors["future_masks"][tensor_index].clone(),
            row,
        )
