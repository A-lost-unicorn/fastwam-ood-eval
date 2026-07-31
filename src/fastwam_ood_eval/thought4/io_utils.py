"""Atomic, isolated artifact helpers for Thought4."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from fastwam_ood_eval.thought4.schemas import canonical_json


class Thought4ArtifactError(RuntimeError):
    """Raised when an artifact write could overwrite or escape Thought4."""


def ensure_thought4_output_path(path: str | Path) -> Path:
    target = Path(path).resolve()
    root = Path("outputs/thought4").resolve()
    if target == root or root not in target.parents:
        raise Thought4ArtifactError(
            f"artifact must be below outputs/thought4/: {target}"
        )
    return target


def ensure_run_mutable(run_dir: str | Path) -> Path:
    root = ensure_thought4_output_path(run_dir)
    status_path = root / "run_status.json"
    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Thought4ArtifactError(f"invalid run status: {status_path}") from exc
        if isinstance(status, Mapping) and status.get("status") == "complete":
            raise Thought4ArtifactError(
                f"completed Thought4 output is immutable: {root}"
            )
    return root


def _atomic_replace(path: Path, payload: bytes, *, overwrite: bool) -> Path:
    ensure_thought4_output_path(path.parent)
    if path.exists() and not overwrite:
        raise Thought4ArtifactError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not overwrite:
            raise Thought4ArtifactError(f"refusing to overwrite artifact: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def atomic_write_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    encoded = (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return _atomic_replace(Path(path), encoded, overwrite=overwrite)


def atomic_write_jsonl(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    overwrite: bool = False,
) -> Path:
    encoded = "".join(canonical_json(dict(row)) + "\n" for row in rows).encode(
        "utf-8"
    )
    return _atomic_replace(Path(path), encoded, overwrite=overwrite)


def atomic_write_text(
    path: str | Path,
    value: str,
    *,
    overwrite: bool = False,
) -> Path:
    return _atomic_replace(Path(path), value.encode("utf-8"), overwrite=overwrite)


def write_or_verify_json(
    path: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    """Write once, or validate an existing deterministic JSON artifact."""

    target = Path(path)
    if not target.exists():
        return atomic_write_json(target, payload)
    try:
        existing = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Thought4ArtifactError(f"invalid existing JSON artifact: {target}") from exc
    if existing != dict(payload):
        raise Thought4ArtifactError(
            f"existing JSON artifact differs during resume: {target}"
        )
    return target


def write_or_verify_jsonl(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    """Write once, or validate every row of an existing JSONL artifact."""

    target = Path(path)
    materialized = [dict(row) for row in rows]
    if not target.exists():
        return atomic_write_jsonl(target, materialized)
    try:
        existing = [
            json.loads(line)
            for line in target.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise Thought4ArtifactError(
            f"invalid existing JSONL artifact: {target}"
        ) from exc
    if existing != materialized:
        raise Thought4ArtifactError(
            f"existing JSONL artifact differs during resume: {target}"
        )
    return target


def write_or_verify_text(path: str | Path, value: str) -> Path:
    """Write once, or require byte-identical text during resume."""

    target = Path(path)
    if not target.exists():
        return atomic_write_text(target, value)
    try:
        existing = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise Thought4ArtifactError(
            f"invalid existing text artifact: {target}"
        ) from exc
    if existing != value:
        raise Thought4ArtifactError(
            f"existing text artifact differs during resume: {target}"
        )
    return target
