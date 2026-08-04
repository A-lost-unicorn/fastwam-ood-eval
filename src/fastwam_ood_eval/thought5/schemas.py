"""Strict schemas, canonical hashing, and fail-closed artifact writers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping


class Thought5ArtifactError(RuntimeError):
    """Raised when an artifact violates the frozen Phase 5 contract."""


def clean_project_commit(root: str | Path = ".") -> str:
    """Return HEAD only for a clean, committed real-experiment snapshot."""

    repository = Path(root)
    head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise Thought5ArtifactError(
            "real Thought5 stages require a clean committed project snapshot"
        )
    if len(head) != 40 or any(
        character not in "0123456789abcdef" for character in head
    ):
        raise Thought5ArtifactError("project HEAD is not a full Git commit SHA")
    return head


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return UTF-8 canonical JSON after rejecting NaN and infinity."""

    plain = _plain(value)
    require_finite(plain)
    return json.dumps(
        plain,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def require_finite(value: Any, *, path: str = "root") -> None:
    """Recursively fail closed on non-finite numerical artifacts."""

    if isinstance(value, float) and not math.isfinite(value):
        raise Thought5ArtifactError(f"non-finite value at {path}: {value!r}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            require_finite(item, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            require_finite(item, path=f"{path}[{index}]")


def _self_hash_payload(value: Mapping[str, Any], hash_field: str) -> dict[str, Any]:
    payload = dict(value)
    payload.pop(hash_field, None)
    return payload


def seal_full_object(
    value: Mapping[str, Any], *, hash_field: str = "full_object_sha256"
) -> dict[str, Any]:
    """Seal all completed fields using the sole self-hash exclusion convention.

    The hash is computed only after every business field exists.  Validation
    removes only the self-referential hash field, so adding or editing any other
    field invalidates the seal.  This fixes the partial-object writer pattern
    disclosed for Thought4 v6 without mutating that frozen artifact.
    """

    sealed = dict(value)
    sealed[hash_field] = object_sha256(_self_hash_payload(sealed, hash_field))
    return sealed


def validate_full_object_seal(
    value: Mapping[str, Any], *, hash_field: str = "full_object_sha256"
) -> bool:
    expected = value.get(hash_field)
    return isinstance(expected, str) and expected == object_sha256(
        _self_hash_payload(value, hash_field)
    )


def write_json_once(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    allow_identical: bool = False,
) -> Path:
    """Atomically write JSON and reject overwrite of differing artifacts."""

    target = Path(path)
    payload = canonical_json_bytes(value) + b"\n"
    if target.exists():
        existing = target.read_bytes()
        if allow_identical and existing == payload:
            return target
        raise Thought5ArtifactError(f"refusing to overwrite artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def write_text_once(
    path: str | Path, text: str, *, allow_identical: bool = False
) -> Path:
    target = Path(path)
    payload = text.encode("utf-8")
    if target.exists():
        if allow_identical and target.read_bytes() == payload:
            return target
        raise Thought5ArtifactError(f"refusing to overwrite artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


@dataclass(frozen=True)
class EpisodeKey:
    task_index: int
    episode_index: int
    seed: int

    def __post_init__(self) -> None:
        if min(self.task_index, self.episode_index, self.seed) < 0:
            raise ValueError("task, episode, and seed must be non-negative")

    @property
    def identity(self) -> str:
        return f"task={self.task_index}/episode={self.episode_index}/seed={self.seed}"


@dataclass(frozen=True)
class PairIdentity:
    pair_id: str
    clean: EpisodeKey
    condition: str
    clean_state_sha256: str
    condition_state_sha256: str
    exact_state: bool

    def validate(self) -> None:
        if self.condition not in {"camera", "lighting", "robot_init"}:
            raise ValueError(f"unknown paired condition: {self.condition}")
        hashes_equal = self.clean_state_sha256 == self.condition_state_sha256
        if self.exact_state and not hashes_equal:
            raise ValueError("exact-state pair has unequal simulator state hashes")
        if self.condition in {"camera", "lighting"} and not self.exact_state:
            raise ValueError(f"{self.condition} must be exact-state paired")
        if self.condition == "robot_init" and self.exact_state:
            raise ValueError("robot-init must not be labelled exact-state")


@dataclass(frozen=True)
class CohortRow:
    split: str
    task_index: int
    task_name: str
    episode_index: int
    task_local_episode_index: int
    seed: int
    frame_index: int
    source: str = "lerobot"

    @property
    def group_key(self) -> str:
        return f"task={self.task_index}/episode={self.episode_index}/seed={self.seed}"

    @property
    def sample_id(self) -> str:
        return object_sha256(asdict(self))
