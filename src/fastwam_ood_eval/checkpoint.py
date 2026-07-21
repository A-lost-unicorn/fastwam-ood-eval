"""Checkpoint provenance utilities."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path | None, chunk_size: int = 8 * 1024 * 1024) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def cached_sha256_file(path: Path | None, cache_path: Path) -> str | None:
    """Hash once across workers, guarded by an advisory file lock."""
    if path is None or not path.is_file():
        return None
    import fcntl

    resolved = path.resolve()
    stat = resolved.stat()
    identity = {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                cached = {}
            if all(cached.get(key) == value for key, value in identity.items()) and cached.get("sha256"):
                return str(cached["sha256"])
        digest = sha256_file(resolved)
        payload = {**identity, "sha256": digest}
        temporary = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(cache_path)
        return digest
