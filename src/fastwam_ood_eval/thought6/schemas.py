"""Canonical schemas, immutable artifacts, and integrity seals for Thought6."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


class Thought6Error(RuntimeError):
    """Base fail-closed error for the Phase 6 protocol."""


class Thought6ArtifactError(Thought6Error):
    """Raised when an immutable artifact would be overwritten or is invalid."""


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot encode {type(value).__name__} as canonical JSON")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: Any) -> str:
    import torch

    value = tensor.detach().cpu().contiguous()
    metadata = f"{value.dtype}|{tuple(value.shape)}".encode("utf-8")
    # NumPy cannot represent torch.bfloat16. Hash the exact storage bytes so
    # the same helper covers release-model BF16 tensors and CPU test tensors.
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(metadata + raw).hexdigest()


def validate_finite(value: Any, *, path: str = "root") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise Thought6ArtifactError(f"non-finite value at {path}")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            validate_finite(nested, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            validate_finite(nested, path=f"{path}[{index}]")


def seal_full_object(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop("full_object_sha256", None)
    validate_finite(payload)
    payload["full_object_sha256"] = object_sha256(payload)
    return payload


def validate_full_object_seal(value: Mapping[str, Any]) -> bool:
    stored = value.get("full_object_sha256")
    payload = dict(value)
    payload.pop("full_object_sha256", None)
    return isinstance(stored, str) and stored == object_sha256(payload)


def _atomic_write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def write_once_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    target = Path(path)
    validate_finite(value)
    payload = canonical_json_bytes(value) + b"\n"
    if target.exists():
        if target.read_bytes() == payload:
            return target
        raise Thought6ArtifactError(f"refusing to overwrite artifact: {target}")
    return _atomic_write(target, payload)


def write_stage_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    """Allow only NOT RUN/error -> finalized transition; finalized is immutable."""

    target = Path(path)
    validate_finite(value)
    payload = canonical_json_bytes(value) + b"\n"
    if target.exists():
        if target.read_bytes() == payload:
            return target
        previous = json.loads(target.read_text(encoding="utf-8"))
        if previous.get("status") not in {"NOT RUN", "running", "error"}:
            raise Thought6ArtifactError(
                f"refusing to mutate finalized artifact: {target}"
            )
    return _atomic_write(target, payload)


def write_once_text(path: str | Path, value: str) -> Path:
    target = Path(path)
    payload = value.encode("utf-8")
    if target.exists():
        if target.read_bytes() == payload:
            return target
        raise Thought6ArtifactError(f"refusing to overwrite report: {target}")
    return _atomic_write(target, payload)


def write_report_transition(path: str | Path, value: str) -> Path:
    target = Path(path)
    payload = value.encode("utf-8")
    if target.exists():
        if target.read_bytes() == payload:
            return target
        if "**NOT RUN**" not in target.read_text(encoding="utf-8"):
            raise Thought6ArtifactError(f"refusing to mutate report: {target}")
    return _atomic_write(target, payload)


def build_artifact_manifest(
    root: str | Path,
    *,
    names: Iterable[str],
    status: str,
) -> dict[str, Any]:
    directory = Path(root)
    rows: list[dict[str, Any]] = []
    for name in sorted(set(names)):
        path = directory / name
        if not path.is_file():
            raise Thought6ArtifactError(f"artifact is absent: {path}")
        rows.append(
            {
                "relative_path": name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return seal_full_object(
        {
            "schema_version": "thought6.artifact_manifest.v1",
            "status": status,
            "root": str(directory),
            "artifacts": rows,
        }
    )


def validate_artifact_manifest(root: str | Path, value: Mapping[str, Any]) -> None:
    if not validate_full_object_seal(value):
        raise Thought6ArtifactError("artifact manifest full-object hash mismatch")
    directory = Path(root)
    for row in value.get("artifacts", []):
        path = directory / str(row["relative_path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or file_sha256(path) != str(row["sha256"])
        ):
            raise Thought6ArtifactError(f"artifact manifest mismatch: {path}")
