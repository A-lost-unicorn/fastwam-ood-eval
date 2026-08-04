"""Phase 5 artifact transitions, manifests, and full-object integrity seals."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from fastwam_ood_eval.thought5.schemas import (
    Thought5ArtifactError,
    canonical_json_bytes,
    file_sha256,
    object_sha256,
    seal_full_object,
    validate_full_object_seal,
)


FORMAL_RESULT_FILES = (
    "training_results.json",
    "representation_results.json",
    "future_geometry_results.json",
    "future_utility_results.json",
    "rollout_results.json",
    "mechanism_evidence.json",
    "mechanism_classification.json",
)


TRANSITIONAL_STATUSES = frozenset(
    {
        "NOT RUN",
        "candidate_not_frozen",
        "mock_shape_verified_real_model_NOT_RUN",
        "running",
        "error",
    }
)


def write_status_transition(path: str | Path, value: Mapping[str, Any]) -> Path:
    """Allow NOT RUN -> running/complete, but never mutate completed artifacts."""

    target = Path(path)
    payload = canonical_json_bytes(value) + b"\n"
    if target.exists():
        # A resume/finalizer may reach an already committed stage.  Treat an
        # exact byte-for-byte replay as verification, not mutation.  Any
        # changed field still fails closed below.
        if target.read_bytes() == payload:
            return target
        previous = json.loads(target.read_text(encoding="utf-8"))
        status = previous.get("status")
        if status not in TRANSITIONAL_STATUSES:
            raise Thought5ArtifactError(
                f"refusing to mutate finalized artifact ({status}): {target}"
            )
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


def write_report_transition(path: str | Path, value: str) -> Path:
    """Replace only the explicit NOT-RUN report scaffold, then freeze it.

    JSON artifacts carry a machine-readable status.  The human report instead
    uses a conspicuous ``**NOT RUN**`` marker.  This helper gives it the same
    one-way state transition without allowing an already completed report to
    be rewritten after formal outcomes are known.
    """

    target = Path(path)
    payload = value.encode("utf-8")
    if target.exists():
        previous = target.read_text(encoding="utf-8")
        if target.read_bytes() == payload:
            return target
        if "**NOT RUN**" not in previous:
            raise Thought5ArtifactError(
                f"refusing to mutate finalized report: {target}"
            )
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


def execution_integrity(
    *,
    config_fingerprint: str,
    cohort_sha256: str,
    stage_status: Mapping[str, str],
    checkpoints: Mapping[str, str],
    immutable_inputs: Mapping[str, str],
    status: str | None = None,
) -> dict[str, Any]:
    if status is None:
        status = (
            "NOT RUN"
            if any(value == "NOT RUN" for value in stage_status.values())
            else "complete"
        )
    payload = {
        "schema_version": "thought5.phase5.execution_integrity.v1",
        "status": status,
        "config_fingerprint": config_fingerprint,
        "cohort_sha256": cohort_sha256,
        "stage_status": dict(sorted(stage_status.items())),
        "checkpoints": dict(sorted(checkpoints.items())),
        "immutable_inputs": dict(sorted(immutable_inputs.items())),
        "all_fields_final_before_hash": True,
        "self_hash_convention": "canonical_object_excluding_only_full_object_sha256",
    }
    sealed = seal_full_object(payload)
    if not validate_full_object_seal(sealed):
        raise Thought5ArtifactError("execution integrity self-check failed")
    return sealed


def build_artifact_manifest(
    root: str | Path,
    *,
    names: Iterable[str] | None = None,
    status: str = "complete",
) -> dict[str, Any]:
    directory = Path(root)
    selected = sorted(names or [p.name for p in directory.iterdir() if p.is_file()])
    rows: list[dict[str, Any]] = []
    for name in selected:
        path = directory / name
        if not path.is_file():
            raise Thought5ArtifactError(f"artifact is absent: {path}")
        rows.append(
            {
                "relative_path": name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    payload = {
        "schema_version": "thought5.phase5.artifact_manifest.v1",
        "status": status,
        "root": str(directory),
        "artifacts": rows,
    }
    payload["manifest_sha256"] = object_sha256(payload)
    return payload


def validate_artifact_manifest(root: str | Path, manifest: Mapping[str, Any]) -> None:
    directory = Path(root)
    stored = manifest.get("manifest_sha256")
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    if stored != object_sha256(payload):
        raise Thought5ArtifactError("artifact manifest object hash mismatch")
    for row in manifest["artifacts"]:
        path = directory / row["relative_path"]
        if (
            not path.is_file()
            or path.stat().st_size != row["bytes"]
            or file_sha256(path) != row["sha256"]
        ):
            raise Thought5ArtifactError(f"artifact manifest mismatch: {path}")
