"""Small atomic artifact helpers shared by isolated Thought3 writers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from fastwam_ood_eval.thought3.safety import ensure_thought3_output_path
from fastwam_ood_eval.thought3.schemas import canonical_json


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    ensure_thought3_output_path(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
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
    _atomic_replace_bytes(target, encoded)
    return target


def atomic_write_jsonl(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    target = Path(path)
    encoded = "".join(canonical_json(dict(row)) + "\n" for row in rows).encode(
        "utf-8"
    )
    _atomic_replace_bytes(target, encoded)
    return target


def atomic_write_text(path: str | Path, text: str) -> Path:
    target = Path(path)
    _atomic_replace_bytes(target, text.encode("utf-8"))
    return target


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(
                    f"JSONL row {line_number} must be an object: {path}"
                )
            rows.append(value)
    return rows
