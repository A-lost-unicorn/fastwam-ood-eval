"""Atomic Adapter-only checkpoints with strict provenance compatibility."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file

from fastwam_ood_eval.thought3.adapter import FutureToActionAdapter
from fastwam_ood_eval.thought3.io_utils import (
    atomic_write_json,
    load_json,
    sha256_file,
)
from fastwam_ood_eval.thought3.safety import ensure_thought3_output_path
from fastwam_ood_eval.thought3.schemas import AdapterCheckpointManifest


class AdapterCheckpointError(RuntimeError):
    """Raised when an Adapter checkpoint is unsafe or incompatible."""


ADAPTER_FILENAME = "adapter.safetensors"
OPTIMIZER_FILENAME = "optimizer.pt"
MANIFEST_FILENAME = "manifest.json"


def _expected_trainable_names(
    adapter: FutureToActionAdapter,
) -> tuple[str, ...]:
    return tuple(
        f"adapter.{name}"
        for name, parameter in adapter.named_parameters()
        if parameter.requires_grad
    )


def adapter_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor semantics independent of safetensors metadata ordering."""

    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def save_adapter_checkpoint(
    directory: str | Path,
    *,
    adapter: FutureToActionAdapter,
    manifest: AdapterCheckpointManifest,
    optimizer: torch.optim.Optimizer | None = None,
) -> Path:
    target = ensure_thought3_output_path(directory)
    if target.exists():
        raise FileExistsError(f"checkpoint already exists: {target}")
    if manifest.trainable_parameter_count != adapter.trainable_parameter_count:
        raise AdapterCheckpointError(
            "manifest trainable parameter count differs from Adapter"
        )
    expected_names = _expected_trainable_names(adapter)
    if tuple(manifest.trainable_parameter_names) != expected_names:
        raise AdapterCheckpointError(
            "manifest trainable parameter names differ from Adapter"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
    )
    try:
        adapter_path = temporary / ADAPTER_FILENAME
        state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in adapter.state_dict().items()
        }
        save_file(
            state,
            str(adapter_path),
            metadata={
                "adapter_fingerprint": manifest.adapter_fingerprint,
                "schema_version": manifest.schema_version,
            },
        )
        file_hashes = {ADAPTER_FILENAME: sha256_file(adapter_path)}
        if optimizer is not None:
            optimizer_path = temporary / OPTIMIZER_FILENAME
            torch.save(optimizer.state_dict(), optimizer_path)
            file_hashes[OPTIMIZER_FILENAME] = sha256_file(optimizer_path)
        enriched = replace(
            manifest,
            extra={
                **dict(manifest.extra),
                "files_sha256": file_hashes,
                "adapter_state_sha256": adapter_state_sha256(state),
                "contains_backbone": False,
                "checkpoint_kind": "adapter_only",
            },
        )
        atomic_write_json(temporary / MANIFEST_FILENAME, enriched.to_dict())
        os.rename(temporary, target)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return target


def _validate_compatibility(
    manifest: AdapterCheckpointManifest,
    expected: Mapping[str, Any] | None,
) -> None:
    if not expected:
        return
    allowed = {
        "adapter_fingerprint",
        "backbone_checkpoint_sha256",
        "cache_fingerprint",
        "config_fingerprint",
        "dataset_stats_sha256",
        "fastwam_commit",
        "frozen_parameter_sha256",
        "k",
        "split_fingerprint",
        "variant",
    }
    unknown = set(expected) - allowed
    if unknown:
        raise AdapterCheckpointError(
            f"unknown checkpoint compatibility keys: {sorted(unknown)}"
        )
    mismatches = {
        key: (getattr(manifest, key), value)
        for key, value in expected.items()
        if getattr(manifest, key) != value
    }
    if mismatches:
        raise AdapterCheckpointError(
            f"checkpoint provenance mismatch: {mismatches}"
        )


def load_adapter_checkpoint(
    directory: str | Path,
    *,
    adapter: FutureToActionAdapter,
    optimizer: torch.optim.Optimizer | None = None,
    expected: Mapping[str, Any] | None = None,
) -> AdapterCheckpointManifest:
    root = ensure_thought3_output_path(directory)
    try:
        manifest = AdapterCheckpointManifest.from_dict(
            load_json(root / MANIFEST_FILENAME)
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        raise AdapterCheckpointError(f"invalid checkpoint manifest: {exc}") from exc
    if manifest.extra.get("contains_backbone") is not False:
        raise AdapterCheckpointError("checkpoint does not prove backbone exclusion")
    if manifest.extra.get("checkpoint_kind") != "adapter_only":
        raise AdapterCheckpointError("checkpoint kind is not adapter_only")
    _validate_compatibility(manifest, expected)
    hashes = manifest.extra.get("files_sha256")
    if not isinstance(hashes, Mapping) or ADAPTER_FILENAME not in hashes:
        raise AdapterCheckpointError("checkpoint file hashes are absent")
    for name, expected_hash in hashes.items():
        path = root / str(name)
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise AdapterCheckpointError(
                f"checkpoint file checksum mismatch: {path}"
            )
    state = load_file(str(root / ADAPTER_FILENAME), device="cpu")
    if adapter_state_sha256(state) != manifest.extra.get(
        "adapter_state_sha256"
    ):
        raise AdapterCheckpointError("Adapter semantic state checksum mismatch")
    expected_names = set(adapter.state_dict())
    if set(state) != expected_names:
        raise AdapterCheckpointError(
            "Adapter state keys differ from checkpoint structure"
        )
    try:
        adapter.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise AdapterCheckpointError(
            f"Adapter tensor shape mismatch: {exc}"
        ) from exc
    if tuple(manifest.trainable_parameter_names) != _expected_trainable_names(adapter):
        raise AdapterCheckpointError(
            "checkpoint trainable allowlist differs from current Adapter"
        )
    if optimizer is not None:
        optimizer_path = root / OPTIMIZER_FILENAME
        if not optimizer_path.is_file():
            raise AdapterCheckpointError("checkpoint has no optimizer state")
        try:
            optimizer.load_state_dict(
                torch.load(
                    optimizer_path,
                    map_location="cpu",
                    weights_only=True,
                )
            )
        except Exception as exc:
            raise AdapterCheckpointError(
                f"cannot restore optimizer state: {exc}"
            ) from exc
    return manifest


def find_latest_checkpoint(root: str | Path) -> Path | None:
    checkpoint_root = ensure_thought3_output_path(root)
    candidates: list[tuple[int, Path]] = []
    if not checkpoint_root.exists():
        return None
    for manifest_path in checkpoint_root.glob("*/manifest.json"):
        try:
            manifest = AdapterCheckpointManifest.from_dict(
                load_json(manifest_path)
            )
        except (TypeError, ValueError, FileNotFoundError):
            continue
        candidates.append((manifest.global_step, manifest_path.parent))
    if not candidates:
        return None
    return max(candidates, key=lambda value: (value[0], str(value[1])))[1]
