"""Adapter-only Fast-WAM-GeoEq checkpoints and semantic tensor hashes."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from fastwam_ood_eval.thought5.schemas import (
    Thought5ArtifactError,
    file_sha256,
    object_sha256,
    write_json_once,
)


class GeoEqCheckpointError(Thought5ArtifactError):
    pass


def tensor_state_sha256(state: Mapping[str, Any]) -> str:
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor):
            raise GeoEqCheckpointError(f"state entry is not a tensor: {name}")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def frozen_parameter_sha256(named_parameters: Iterable[tuple[str, Any]]) -> str:
    return tensor_state_sha256(
        {
            name: parameter
            for name, parameter in named_parameters
            if not parameter.requires_grad
        }
    )


def geoeq_state_dict(attachment: Any) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name, value in attachment.model.state_dict().items():
        if ".lora_A" in name or ".lora_B" in name:
            state[f"backbone.{name}"] = value.detach().cpu().contiguous()
    for prefix, module in (
        ("geo_projector", attachment.geo_projector),
        ("ray_pose_encoder", attachment.ray_pose_encoder),
    ):
        for name, value in module.state_dict().items():
            state[f"{prefix}.{name}"] = value.detach().cpu().contiguous()
    if not state:
        raise GeoEqCheckpointError("GeoEq state dict is empty")
    return state


def save_geoeq_checkpoint(
    directory: str | Path,
    *,
    attachment: Any,
    variant: str,
    global_step: int,
    config_fingerprint: str,
    cohort_fingerprint: str,
    backbone_checkpoint_sha256: str,
    frozen_before_sha256: str,
    frozen_after_sha256: str,
    optimizer: Any | None = None,
) -> Path:
    import torch

    target = Path(directory)
    if target.exists():
        raise GeoEqCheckpointError(f"checkpoint already exists: {target}")
    if frozen_before_sha256 != frozen_after_sha256:
        raise GeoEqCheckpointError("frozen backbone changed during training")
    state = geoeq_state_dict(attachment)
    manifest = attachment.parameter_manifest()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    )
    try:
        weights = temporary / "geoeq_state.pt"
        torch.save(state, weights)
        files = {"geoeq_state.pt": file_sha256(weights)}
        if optimizer is not None:
            optimizer_path = temporary / "optimizer.pt"
            torch.save(optimizer.state_dict(), optimizer_path)
            files["optimizer.pt"] = file_sha256(optimizer_path)
        payload = {
            "schema_version": "thought5.phase5.checkpoint.v1",
            "checkpoint_kind": "geoeq_adapter_only",
            "contains_backbone": False,
            "variant": variant,
            "global_step": int(global_step),
            "config_fingerprint": config_fingerprint,
            "cohort_fingerprint": cohort_fingerprint,
            "backbone_checkpoint_sha256": backbone_checkpoint_sha256,
            "frozen_parameter_sha256": frozen_before_sha256,
            "trainable_parameter_manifest": manifest,
            "state_sha256": tensor_state_sha256(state),
            "files_sha256": files,
        }
        payload["manifest_sha256"] = object_sha256(payload)
        write_json_once(temporary / "manifest.json", payload)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(temporary, target)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return target


def restore_geoeq_state(attachment: Any, state: Mapping[str, Any]) -> None:
    expected = set(geoeq_state_dict(attachment))
    if set(state) != expected:
        raise GeoEqCheckpointError("in-memory GeoEq state keys differ")
    backbone = {
        name.removeprefix("backbone."): value
        for name, value in state.items()
        if name.startswith("backbone.")
    }
    attachment.model.load_state_dict(backbone, strict=False)
    for prefix, module in (
        ("geo_projector.", attachment.geo_projector),
        ("ray_pose_encoder.", attachment.ray_pose_encoder),
    ):
        module.load_state_dict(
            {
                name.removeprefix(prefix): value
                for name, value in state.items()
                if name.startswith(prefix)
            },
            strict=True,
        )


def load_geoeq_checkpoint(
    directory: str | Path,
    *,
    attachment: Any,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    import json
    import torch

    root = Path(directory)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise GeoEqCheckpointError("checkpoint manifest is absent")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored_manifest_sha = manifest.pop("manifest_sha256", None)
    if stored_manifest_sha != object_sha256(manifest):
        raise GeoEqCheckpointError("checkpoint manifest checksum mismatch")
    manifest["manifest_sha256"] = stored_manifest_sha
    if manifest.get("contains_backbone") is not False:
        raise GeoEqCheckpointError("checkpoint may contain frozen backbone")
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise GeoEqCheckpointError(f"checkpoint provenance mismatch: {mismatches}")
    for name, digest in manifest["files_sha256"].items():
        if file_sha256(root / name) != digest:
            raise GeoEqCheckpointError(f"checkpoint file checksum mismatch: {name}")
    state = torch.load(root / "geoeq_state.pt", map_location="cpu", weights_only=True)
    if tensor_state_sha256(state) != manifest["state_sha256"]:
        raise GeoEqCheckpointError("checkpoint semantic state checksum mismatch")
    expected_keys = set(geoeq_state_dict(attachment))
    if set(state) != expected_keys:
        raise GeoEqCheckpointError("checkpoint keys differ from current architecture")
    backbone = {
        name.removeprefix("backbone."): value
        for name, value in state.items()
        if name.startswith("backbone.")
    }
    attachment.model.load_state_dict(backbone, strict=False)
    for prefix, module in (
        ("geo_projector.", attachment.geo_projector),
        ("ray_pose_encoder.", attachment.ray_pose_encoder),
    ):
        substate = {
            name.removeprefix(prefix): value
            for name, value in state.items()
            if name.startswith(prefix)
        }
        module.load_state_dict(substate, strict=True)
    return manifest
