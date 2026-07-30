"""CPU-only, post-run artifact audit for Thought3 E.9a-v2.

The parent run is immutable.  This module never imports or instantiates
Fast-WAM, never opens checkpoint tensor payloads, and writes only to a new
audit directory.  Missing held-out RNG fields are reconstructed as *derived
evidence* from the frozen run code path; they are never represented as fields
that were observed in the original JSONL.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from fastwam_ood_eval.thought3.io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    load_json,
    load_jsonl,
    sha256_file,
)
from fastwam_ood_eval.thought3.safety import ensure_thought3_output_path
from fastwam_ood_eval.thought3.schemas import canonical_json


AUDIT_CONFIG_SCHEMA = "thought3.phase_e9_v2_1.audit_config.v1"
AUDIT_RESULT_SCHEMA = "thought3.phase_e9_v2_1.audit_result.v1"
DERIVED_IDENTITY_SCHEMA = "thought3.phase_e9_v2_1.derived_identity.v1"
VALID_OUTCOME = "audit_valid_scientific_failed"
INVALID_OUTCOME = "audit_invalid_identity_unrecoverable"
TRACK_KEYS = (
    "raw/A0",
    "raw/A1",
    "normalized/A0",
    "normalized/A1",
)
MISSING_PARENT_IDENTITY_FIELDS = (
    "action_noise_seed",
    "action_timestep_seed",
    "flow_objective_sha256",
)
SCHEDULE_FIELDS = (
    "objective_index",
    "optimizer_update",
    "cohort_sample_index",
    "base_sample_id",
    "flow_step",
    "training_flow_slot",
    "action_noise_seed",
    "action_timestep_seed",
    "flow_objective_sha256",
    "micro_index",
    "sample_cursor",
    "zero_weight_objective",
    "timestep",
    "action_weight",
)


class E9V21AuditError(RuntimeError):
    """Raised when the read-only audit contract cannot be executed safely."""


@dataclass(frozen=True)
class E9V21AuditConfig:
    source_path: Path
    raw: Mapping[str, Any]
    parent_root: Path
    output_root: Path
    expected_file_count: int
    gate_sha256: str
    parent_config_fingerprint: str
    original_schema: str
    original_status: str
    core_artifacts: Mapping[str, str]
    project_run_commit: str
    fastwam_commit: str
    source_blobs: Mapping[str, str]
    train_seed: int
    noise_namespace: str
    timestep_namespace: str
    heldout_flows: tuple[int, ...]
    sample_ids: tuple[str, ...]
    expected_zero_weight_positions: tuple[tuple[int, int], ...]
    expected_probe_rows: int
    expected_probes_per_track: int
    weight_abs_tolerance: float
    paired_advantage_threshold: float
    expected_classification: str
    expected_independent_replication_candidate: bool
    expected_raw_confirmed_harm_count: int
    expected_normalized_confirmed_harm_count: int

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            canonical_json(dict(self.raw)).encode("utf-8")
        ).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise E9V21AuditError(f"{name} must be a mapping")
    return value


def load_e9_v21_audit_config(path: str | Path) -> E9V21AuditConfig:
    """Load the standalone audit schema without changing Thought3 v1 configs."""

    source = Path(path)
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(value, "audit config")
    if root.get("schema_version") != AUDIT_CONFIG_SCHEMA:
        raise E9V21AuditError(
            f"schema_version must be {AUDIT_CONFIG_SCHEMA}"
        )
    expected_top = {
        "schema_version",
        "parent",
        "provenance",
        "identity",
        "scientific",
        "output",
    }
    if set(root) != expected_top:
        raise E9V21AuditError(
            "audit config top-level keys changed: "
            f"{sorted(set(root) ^ expected_top)}"
        )
    parent = _mapping(root["parent"], "parent")
    provenance = _mapping(root["provenance"], "provenance")
    identity = _mapping(root["identity"], "identity")
    scientific = _mapping(root["scientific"], "scientific")
    output = _mapping(root["output"], "output")
    core = {
        str(name): str(digest)
        for name, digest in _mapping(
            parent.get("core_artifacts"), "parent.core_artifacts"
        ).items()
    }
    blobs = {
        str(name): str(digest)
        for name, digest in _mapping(
            provenance.get("source_blobs"), "provenance.source_blobs"
        ).items()
    }
    digest_values = (
        [parent.get("gate_sha256"), parent.get("config_fingerprint")]
        + list(core.values())
    )
    # Config fingerprints are SHA-256; Git blobs and commits are SHA-1 here.
    if not all(_is_sha256(value) for value in digest_values):
        raise E9V21AuditError("audit config contains an invalid SHA-256")
    for label, value in {
        "project_run_commit": provenance.get("project_run_commit"),
        "fastwam_commit": provenance.get("fastwam_commit"),
        **blobs,
    }.items():
        text = str(value)
        if len(text) != 40 or any(c not in "0123456789abcdef" for c in text):
            raise E9V21AuditError(f"{label} must be a 40-character Git SHA")
    flows = tuple(int(value) for value in identity.get("heldout_flows", ()))
    samples = tuple(str(value) for value in identity.get("sample_ids", ()))
    zero_positions = tuple(
        tuple(int(item) for item in pair)
        for pair in identity.get("expected_zero_weight_positions", ())
    )
    if (
        flows != tuple(range(75, 107))
        or len(samples) != 8
        or len(set(samples)) != 8
        or any(not _is_sha256(sample_id) for sample_id in samples)
        or zero_positions != ((1, 80), (7, 93))
    ):
        raise E9V21AuditError("frozen held-out identity design changed")
    cfg = E9V21AuditConfig(
        source_path=source,
        raw=root,
        parent_root=Path(str(parent["root"])),
        output_root=Path(str(output["root"])),
        expected_file_count=int(parent["expected_file_count"]),
        gate_sha256=str(parent["gate_sha256"]),
        parent_config_fingerprint=str(parent["config_fingerprint"]),
        original_schema=str(parent["original_schema"]),
        original_status=str(parent["original_status"]),
        core_artifacts=core,
        project_run_commit=str(provenance["project_run_commit"]),
        fastwam_commit=str(provenance["fastwam_commit"]),
        source_blobs=blobs,
        train_seed=int(identity["train_seed"]),
        noise_namespace=str(identity["noise_namespace"]),
        timestep_namespace=str(identity["timestep_namespace"]),
        heldout_flows=flows,
        sample_ids=samples,
        expected_zero_weight_positions=zero_positions,
        expected_probe_rows=int(identity["expected_probe_rows"]),
        expected_probes_per_track=int(
            identity["expected_probes_per_track"]
        ),
        weight_abs_tolerance=float(
            identity["weight_cpu_gpu_abs_tolerance"]
        ),
        paired_advantage_threshold=float(
            scientific["paired_advantage_threshold"]
        ),
        expected_classification=str(
            scientific["expected_classification"]
        ),
        expected_independent_replication_candidate=bool(
            scientific["expected_independent_replication_candidate"]
        ),
        expected_raw_confirmed_harm_count=int(
            scientific["expected_raw_confirmed_harm_count"]
        ),
        expected_normalized_confirmed_harm_count=int(
            scientific["expected_normalized_confirmed_harm_count"]
        ),
    )
    if (
        cfg.parent_root.resolve() == cfg.output_root.resolve()
        or cfg.parent_root.resolve() in cfg.output_root.resolve().parents
        or cfg.expected_file_count != 77
        or cfg.expected_probe_rows != 256
        or cfg.expected_probes_per_track != 2
        or cfg.train_seed != 3407
        or cfg.weight_abs_tolerance != 1e-6
        or cfg.paired_advantage_threshold != 0.10
    ):
        raise E9V21AuditError("frozen audit contract changed")
    ensure_thought3_output_path(cfg.parent_root)
    ensure_thought3_output_path(cfg.output_root)
    return cfg


def audit_dry_run_payload(cfg: E9V21AuditConfig) -> dict[str, Any]:
    """Describe the formal audit without reading parent artifacts or torch."""

    return {
        "command": "thought3-audit-e9-v2-artifacts",
        "config_fingerprint": cfg.fingerprint,
        "dry_run": True,
        "parent_root": str(cfg.parent_root),
        "output_dir": str(cfg.output_root),
        "scope": {
            "backward": False,
            "checkpoint_tensor_load": False,
            "forward": False,
            "gpu_model_load": False,
            "optimizer": False,
            "parent_write": False,
            "reserved_cohort_read": False,
        },
        "would_load_checkpoint": False,
        "would_load_fastwam": False,
        "would_write": False,
    }


def _tree_snapshot(root: Path) -> dict[str, dict[str, Any]]:
    if not root.is_dir():
        raise E9V21AuditError(f"parent directory does not exist: {root}")
    snapshot: dict[str, dict[str, Any]] = {}
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        stat = path.stat()
        snapshot[relative] = {
            "bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": sha256_file(path),
        }
    return snapshot


def _git_blob(commit: str, path: str) -> str | None:
    completed = subprocess.run(
        ["git", "ls-tree", commit, "--", path],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    fields = completed.stdout.strip().split()
    return fields[2] if len(fields) >= 4 else None


def _git_head(path: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _reflog_window(commit: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "git",
            "reflog",
            "--date=iso-strict",
            "--format=%H%x09%gD",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    matching = [
        line for line in completed.stdout.splitlines()
        if line.startswith(f"{commit}\t")
    ]
    return {
        "available": completed.returncode == 0,
        "matching_entries": matching,
        "run_commit_present": bool(matching),
    }


def _stable_seed(*values: object) -> int:
    digest = hashlib.sha256(
        "\0".join(str(value) for value in values).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def derive_flow_identity(
    *,
    base_sample_id: str,
    train_seed: int,
    flow_step: int,
    noise_namespace: str,
    timestep_namespace: str,
) -> dict[str, Any]:
    """Reproduce the exact legacy E.9 run identity function."""

    noise_seed = _stable_seed(
        noise_namespace,
        train_seed,
        flow_step,
        base_sample_id,
    )
    timestep_seed = _stable_seed(
        timestep_namespace,
        train_seed,
        flow_step,
        base_sample_id,
    )
    digest = hashlib.sha256(
        (
            f"{base_sample_id}\0{train_seed}\0{flow_step}\0"
            f"{noise_seed}\0{timestep_seed}"
        ).encode("utf-8")
    ).hexdigest()
    return {
        "action_noise_seed": noise_seed,
        "action_timestep_seed": timestep_seed,
        "flow_objective_sha256": digest,
        "flow_step": flow_step,
    }


class _CpuSchedulerReconstruction:
    """Minimal copy of the frozen official timestep/weight math."""

    def __init__(self) -> None:
        import torch

        self.torch = torch
        self.steps = 1000
        self.shift = 5.0
        grid = torch.linspace(
            1.0, 0.0, self.steps + 1, dtype=torch.float64
        )[:-1]
        timestep = self._phi(grid) * float(self.steps)
        y = torch.exp(
            -2.0
            * ((timestep - (self.steps / 2.0)) / self.steps) ** 2
        )
        self.y_min = float(y.min().item())
        self.norm = float((y - self.y_min).mean().item())

    def _phi(self, value: Any) -> Any:
        return self.shift * value / (
            1.0 + (self.shift - 1.0) * value
        )

    def timestep_and_weight(self, seed: int) -> tuple[float, float]:
        torch = self.torch
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        uniform = torch.rand(
            (1,),
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
        timestep = (
            self._phi(uniform) * float(self.steps)
        ).to(dtype=torch.bfloat16)
        value = timestep.to(dtype=torch.float32)
        y = torch.exp(
            -2.0 * ((value - (self.steps / 2.0)) / self.steps) ** 2
        )
        weight = (y - self.y_min) / (self.norm + 1e-10)
        return (
            float(value.reshape(())),
            float(weight.reshape(())),
        )


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(
        float(left),
        float(right),
        rel_tol=0.0,
        abs_tol=tolerance,
    )


def _canonical_sha(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _schedule_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    rows = load_jsonl(path)
    projected = [
        {field: row.get(field) for field in SCHEDULE_FIELDS}
        for row in rows
    ]
    return rows, _canonical_sha(projected)


def _normalize_track_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(dict(payload)))
    value["variant"] = "<VARIANT>"
    value["sampler"]["active_k"] = "<ACTIVE_K>"
    value["experiment"]["name"] = "<TRACK_NAME>"
    value["experiment"]["output_dir"] = "<TRACK_OUTPUT>"
    value.pop("source_path", None)
    return value


def _track_paths(parent: Path, track_key: str) -> dict[str, Path]:
    recipe, variant = track_key.split("/")
    root = parent / "tracks" / recipe / variant.lower()
    return {
        "root": root,
        "heldout": root / "heldout_multiflow_metrics.jsonl",
        "objectives": root / "train_objective_metrics.jsonl",
        "manifest": root / "training_manifest.json",
        "status": root / "run_status.json",
    }


def _audit_probe_grid(
    cfg: E9V21AuditConfig,
    *,
    track_key: str,
    probes: Sequence[Mapping[str, Any]],
    scheduler: _CpuSchedulerReconstruction,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    checks: dict[str, bool] = {}
    checks["two_initial_final_probes"] = (
        len(probes) == cfg.expected_probes_per_track
        and [int(probe.get("global_step", -1)) for probe in probes]
        == [0, 200]
    )
    derived_by_position: list[dict[str, Any]] = []
    timestep_mismatches = 0
    weight_mismatches = 0
    max_weight_abs_difference = 0.0
    missing_fields_exact = True
    zero_positions_by_probe: list[list[tuple[int, int]]] = []
    grid_exact = True
    aggregate_exact = True
    for probe_index, probe in enumerate(probes):
        rows = list(probe.get("per_objective", ()))
        if (
            len(rows) != cfg.expected_probe_rows
            or list(probe.get("sample_ids", ())) != list(cfg.sample_ids)
            or list(probe.get("flow_steps", ())) != list(cfg.heldout_flows)
            or int(probe.get("flow_objective_count", -1))
            != cfg.expected_probe_rows
        ):
            grid_exact = False
        expected_pairs = [
            (sample_id, flow)
            for sample_id in cfg.sample_ids
            for flow in cfg.heldout_flows
        ]
        observed_pairs = [
            (str(row.get("base_sample_id")), int(row.get("flow_step", -1)))
            for row in rows
        ]
        if observed_pairs != expected_pairs:
            grid_exact = False
        zero_positions: list[tuple[int, int]] = []
        for objective_position, row in enumerate(rows):
            sample_index = objective_position // len(cfg.heldout_flows)
            flow_index = objective_position % len(cfg.heldout_flows)
            sample_id, flow_step = expected_pairs[objective_position]
            missing_fields_exact = missing_fields_exact and all(
                field not in row for field in MISSING_PARENT_IDENTITY_FIELDS
            )
            identity = derive_flow_identity(
                base_sample_id=sample_id,
                train_seed=cfg.train_seed,
                flow_step=flow_step,
                noise_namespace=cfg.noise_namespace,
                timestep_namespace=cfg.timestep_namespace,
            )
            reconstructed_timestep, reconstructed_weight = (
                scheduler.timestep_and_weight(
                    int(identity["action_timestep_seed"])
                )
            )
            stored_timestep = float(row.get("timestep", math.nan))
            stored_weight = float(row.get("action_weight", math.nan))
            if stored_timestep != reconstructed_timestep:
                timestep_mismatches += 1
            weight_difference = abs(stored_weight - reconstructed_weight)
            max_weight_abs_difference = max(
                max_weight_abs_difference, weight_difference
            )
            if weight_difference > cfg.weight_abs_tolerance:
                weight_mismatches += 1
            if stored_weight == 0.0:
                zero_positions.append((sample_index, flow_step))
            if probe_index == 0:
                derived_by_position.append(
                    {
                        "schema_version": DERIVED_IDENTITY_SCHEMA,
                        "identity_evidence_kind": (
                            "deterministically_reconstructed_from_"
                            "frozen_run_code_and_stored_grid"
                        ),
                        "original_parent_fields_observed": False,
                        "parent_config_fingerprint": (
                            cfg.parent_config_fingerprint
                        ),
                        "sample_id": sample_id,
                        "sample_index": sample_index,
                        "flow_step": flow_step,
                        "flow_index": flow_index,
                        "objective_position": objective_position,
                        "noise_namespace": cfg.noise_namespace,
                        "timestep_namespace": cfg.timestep_namespace,
                        **identity,
                        "reconstructed_bfloat16_timestep": (
                            reconstructed_timestep
                        ),
                        "reconstructed_cpu_action_weight": (
                            reconstructed_weight
                        ),
                    }
                )
        zero_positions_by_probe.append(zero_positions)
        if rows:
            per_sample_means = []
            for sample_index in range(len(cfg.sample_ids)):
                start = sample_index * len(cfg.heldout_flows)
                stop = start + len(cfg.heldout_flows)
                per_sample_means.append(
                    statistics.fmean(
                        float(row["action_loss"])
                        for row in rows[start:stop]
                    )
                )
            recomputed_mean = statistics.fmean(per_sample_means)
            aggregate_exact = aggregate_exact and _close(
                recomputed_mean,
                float(probe.get("mean_action_loss", math.nan)),
            )
    checks["grid_8x32_exact"] = grid_exact
    checks["parent_identity_fields_absent_as_documented"] = (
        missing_fields_exact
    )
    checks["all_512_timestep_values_exact"] = timestep_mismatches == 0
    checks["all_512_weights_match_cpu_within_frozen_tolerance"] = (
        weight_mismatches == 0
    )
    checks["zero_weight_positions_exact"] = bool(zero_positions_by_probe) and all(
        tuple(value) == cfg.expected_zero_weight_positions
        for value in zero_positions_by_probe
    )
    checks["probe_aggregates_recompute"] = aggregate_exact
    checks["initial_final_identity_pairing_exact"] = (
        len(probes) == 2
        and [
            (
                str(row.get("base_sample_id")),
                int(row.get("flow_step", -1)),
            )
            for row in probes[0].get("per_objective", ())
        ]
        == [
            (
                str(row.get("base_sample_id")),
                int(row.get("flow_step", -1)),
            )
            for row in probes[1].get("per_objective", ())
        ]
    )
    return {
        "checks": checks,
        "max_cpu_gpu_weight_abs_difference": max_weight_abs_difference,
        "probe_count": len(probes),
        "stored_objective_rows": sum(
            len(probe.get("per_objective", ())) for probe in probes
        ),
        "timestep_mismatch_count": timestep_mismatches,
        "track_key": track_key,
        "weight_mismatch_count": weight_mismatches,
        "zero_weight_positions": [
            [list(value) for value in positions]
            for positions in zero_positions_by_probe
        ],
    }, derived_by_position


def _recompute_pair_values(
    track_probes: Mapping[str, Sequence[Mapping[str, Any]]],
    recipe: str,
) -> dict[str, Any]:
    a0 = track_probes[f"{recipe}/A0"][1]
    a1 = track_probes[f"{recipe}/A1"][1]
    a0_mean = float(a0["mean_action_loss"])
    a1_mean = float(a1["mean_action_loss"])
    a0_samples = {
        str(row["base_sample_id"]): float(row["action_loss"])
        for row in a0["per_sample"]
    }
    a1_samples = {
        str(row["base_sample_id"]): float(row["action_loss"])
        for row in a1["per_sample"]
    }
    return {
        "a0_final_mean_action_loss": a0_mean,
        "a1_final_mean_action_loss": a1_mean,
        "a1_non_higher_sample_count": sum(
            a1_samples[sample_id] <= a0_samples[sample_id]
            for sample_id in a0_samples
        ),
        "a1_relative_mean_improvement_over_a0": (
            (a0_mean - a1_mean) / a0_mean
        ),
    }


def _dict_float_equal(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    tolerance: float = 1e-12,
) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        first, second = left[key], right[key]
        if isinstance(first, (float, int)) and isinstance(
            second, (float, int)
        ):
            if isinstance(first, bool) or isinstance(second, bool):
                if first is not second:
                    return False
            elif not _close(float(first), float(second), tolerance):
                return False
        elif first != second:
            return False
    return True


def _all_true(values: Mapping[str, Any]) -> bool:
    return all(bool(value) for value in values.values())


def _audit_parent(
    cfg: E9V21AuditConfig,
    before: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parent = cfg.parent_root
    gate = load_json(parent / "gate_e9a_result.json")
    checks: dict[str, bool] = {}
    checks["parent_file_count_exact"] = len(before) == cfg.expected_file_count
    checks["all_frozen_core_artifact_hashes_exact"] = all(
        name in before and before[name]["sha256"] == expected
        for name, expected in cfg.core_artifacts.items()
    )
    checks["parent_gate_hash_exact"] = (
        before.get("gate_e9a_result.json", {}).get("sha256")
        == cfg.gate_sha256
    )
    checks["parent_original_invalid_status_preserved"] = (
        gate.get("schema_version") == cfg.original_schema
        and gate.get("status") == cfg.original_status
        and gate.get("engineering_passed") is False
        and gate.get("gate_e9a_passed") is False
        and gate.get("config_fingerprint")
        == cfg.parent_config_fingerprint
    )
    blobs = {
        path: _git_blob(cfg.project_run_commit, path)
        for path in cfg.source_blobs
    }
    checks["frozen_run_source_blobs_exact"] = blobs == dict(cfg.source_blobs)
    fastwam_head = _git_head(Path("third_party/FastWAM"))
    checks["fastwam_commit_exact"] = fastwam_head == cfg.fastwam_commit
    reflog = _reflog_window(cfg.project_run_commit)
    checks["run_commit_present_in_local_reflog"] = bool(
        reflog["run_commit_present"]
    )

    scheduler = _CpuSchedulerReconstruction()
    track_probes: dict[str, list[dict[str, Any]]] = {}
    track_audits: dict[str, Any] = {}
    all_derived: list[dict[str, Any]] = []
    schedule_shas: dict[str, str] = {}
    normalized_configs: dict[str, dict[str, Any]] = {}
    known_gap_only = True
    track_complete = True
    all_probe_checks = True
    all_same_initial_adapter = True
    initial_adapter_shas: set[str] = set()
    for track_key in TRACK_KEYS:
        paths = _track_paths(parent, track_key)
        probes = load_jsonl(paths["heldout"])
        manifest = load_json(paths["manifest"])
        status = load_json(paths["status"])
        track_probes[track_key] = probes
        probe_audit, derived = _audit_probe_grid(
            cfg,
            track_key=track_key,
            probes=probes,
            scheduler=scheduler,
        )
        all_derived.extend(
            [{**row, "reference_track": track_key} for row in derived]
            if track_key == "raw/A0"
            else []
        )
        all_probe_checks = all_probe_checks and _all_true(
            probe_audit["checks"]
        )
        parent_track = gate["tracks"][track_key]
        false_execution = sorted(
            key
            for key, value in parent_track["execution_checks"].items()
            if not bool(value)
        )
        known_gap_only = known_gap_only and false_execution == [
            "heldout_rng_and_zero_weight_identity_exact"
        ]
        track_complete = track_complete and (
            manifest.get("status") == "complete"
            and int(manifest.get("completed_steps", -1)) == 200
            and int(manifest.get("completed_objectives", -1)) == 1600
            and int(manifest.get("sample_count", -1)) == 8
            and status.get("status") == "complete"
        )
        initial_adapter_shas.add(str(manifest.get("initial_adapter_sha256")))
        _, schedule_sha = _schedule_rows(paths["objectives"])
        schedule_shas[track_key] = schedule_sha
        normalized_configs[track_key] = _normalize_track_config(
            manifest["config"]
        )
        track_audits[track_key] = {
            **probe_audit,
            "parent_false_execution_checks": false_execution,
            "training_complete_200x8": (
                manifest.get("status") == "complete"
                and int(manifest.get("completed_steps", -1)) == 200
                and int(manifest.get("completed_objectives", -1)) == 1600
            ),
            "training_schedule_sha256_recomputed": schedule_sha,
        }
    checks["known_telemetry_gap_is_only_track_execution_failure"] = (
        known_gap_only
    )
    checks["all_tracks_complete_200_updates_1600_objectives"] = (
        track_complete
    )
    checks["all_track_probe_audits_pass"] = all_probe_checks
    checks["all_four_training_schedules_exact"] = (
        len(set(schedule_shas.values())) == 1
        and bool(schedule_shas)
    )
    checks["all_four_track_configs_match_except_frozen_arm_metadata"] = (
        len({_canonical_sha(value) for value in normalized_configs.values()})
        == 1
    )
    all_same_initial_adapter = len(initial_adapter_shas) == 1
    checks["all_four_same_initial_adapter"] = all_same_initial_adapter
    checks["parent_paired_contracts_all_true"] = _all_true(
        gate["paired_checks"]
    )

    # Re-run the frozen tail bootstrap/classifier from the pinned source blob.
    from fastwam_ood_eval.thought3.phase_e9_sample_tail_mitigation import (
        classify_sample_tail_mitigation,
        paired_tail_bootstrap,
    )

    tail = {
        track_key: paired_tail_bootstrap(
            track_probes[track_key][0],
            track_probes[track_key][1],
            track_key=track_key,
        )
        for track_key in TRACK_KEYS
    }
    checks["tail_bootstrap_recomputes_exactly"] = (
        _canonical_sha(tail) == _canonical_sha(gate["tail_bootstrap"])
    )
    classification = classify_sample_tail_mitigation(
        {
            key: gate["tracks"][key]["performance_checks"]
            for key in TRACK_KEYS
        },
        gate["normalized_paired_checks"],
        tail,
    )
    checks["scientific_classification_recomputes_exactly"] = (
        classification == gate["mitigation_classification"]
        and classification["classification"]
        == cfg.expected_classification
        and classification["independent_replication_candidate"]
        is cfg.expected_independent_replication_candidate
        and int(classification["raw_confirmed_harm_count"])
        == cfg.expected_raw_confirmed_harm_count
        and int(classification["normalized_confirmed_harm_count"])
        == cfg.expected_normalized_confirmed_harm_count
    )
    paired_values = {
        recipe: _recompute_pair_values(track_probes, recipe)
        for recipe in ("raw", "normalized")
    }
    checks["raw_paired_values_recompute"] = _dict_float_equal(
        paired_values["raw"], gate["raw_paired_values"]
    )
    checks["normalized_paired_values_recompute"] = _dict_float_equal(
        paired_values["normalized"], gate["normalized_paired_values"]
    )
    checks["normalized_paired_advantage_remains_below_frozen_threshold"] = (
        paired_values["normalized"][
            "a1_relative_mean_improvement_over_a0"
        ]
        < cfg.paired_advantage_threshold
        and gate["normalized_paired_checks"][
            "a1_final_mean_at_least_10_percent_below_a0"
        ]
        is False
    )
    checks["frozen_fastwam_sha_unchanged"] = (
        gate["frozen_parameter_sha256_before"]
        == gate["frozen_parameter_sha256_after"]
        == "ac0dd59de78495dc8ed5c5601e24951b78f3fd6ad40843d76b83106eccceb4f8"
        and gate["cross_checks"]["frozen_fastwam_unchanged"] is True
    )
    scope = gate["scope"]
    reserved = gate["reserved_e9b_replication"]
    checks["reserve_17_28_not_consumed"] = (
        int(scope["reserved_replication_samples_decoded"]) == 0
        and int(scope["new_training_samples_consumed"]) == 0
        and reserved["decoded_or_trained_by_e9a"] is False
        and gate["cross_checks"]["reserved_cohort_not_decoded_or_trained"]
        is True
    )
    checks["development_ood_success_rollout_not_read"] = (
        scope["development_outcomes_read"] is False
        and scope["ood_outcomes_read"] is False
        and scope["success_outcomes_read"] is False
        and scope["rollout_started"] is False
    )
    checks["future_rgb_and_ground_truth_future_not_read"] = (
        int(scope["future_rgb_frames_read"]) == 0
        and scope["uses_ground_truth_future"] is False
        and gate["data_preparation"]["future_rgb_used_as_input"] is False
    )
    checks["parent_scope_counts_exact"] = (
        int(scope["track_count"]) == 4
        and int(scope["optimizer_steps"]) == 800
        and int(scope["backward_calls"]) == 6400
        and int(scope["training_objectives"]) == 6400
        and int(scope["heldout_probe_objectives"]) == 2048
    )
    checks["derived_identity_manifest_has_256_unique_rows"] = (
        len(all_derived) == 256
        and len(
            {
                (
                    row["sample_id"],
                    row["flow_step"],
                    row["flow_objective_sha256"],
                )
                for row in all_derived
            }
        )
        == 256
    )
    return {
        "checks": checks,
        "classification_recomputed": classification,
        "derived_identity": {
            "field_status": {
                field: (
                    "not_observed_in_parent;"
                    "deterministically_reconstructed"
                )
                for field in MISSING_PARENT_IDENTITY_FIELDS
            },
            "formula_dependency": {
                "seed_fields": [
                    "frozen_namespace",
                    "train_seed",
                    "flow_step",
                    "sample_id",
                ],
                "contextual_mapping_fields_not_encoded_in_seed": [
                    "config_fingerprint",
                    "sample_index",
                    "objective_position",
                ],
                "mapping_is_one_to_one_on_frozen_grid": (
                    checks[
                        "derived_identity_manifest_has_256_unique_rows"
                    ]
                ),
            },
            "unique_identity_count": len(all_derived),
        },
        "paired_values_recomputed": paired_values,
        "parent_original_status": gate["status"],
        "provenance": {
            "fastwam_head": fastwam_head,
            "git_source_blobs": blobs,
            "project_run_commit": cfg.project_run_commit,
            "reflog": reflog,
        },
        "scientific_result": {
            "e9b_locked": True,
            "independent_replication_candidate": False,
            "normalization_tail_stabilization_signal": (
                classification["raw_confirmed_harm_count"] == 2
                and classification["normalized_confirmed_harm_count"] == 0
            ),
            "normalized_a1_absolute_reduction": gate["tracks"][
                "normalized/A1"
            ]["result"]["outcome"]["loss_reduction_fraction"],
            "normalized_a1_vs_a0_paired_advantage": paired_values[
                "normalized"
            ]["a1_relative_mean_improvement_over_a0"],
            "paired_advantage_threshold": cfg.paired_advantage_threshold,
            "sample_tail_mitigation_classification": classification[
                "classification"
            ],
        },
        "track_audits": track_audits,
    }, all_derived


def _artifact_manifest(root: Path, names: Iterable[str]) -> dict[str, Any]:
    files = {}
    for name in names:
        path = root / name
        files[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "schema_version": "thought3.phase_e9_v2_1.audit_artifacts.v1",
        "files": files,
    }


def run_e9_v21_readonly_audit(
    cfg: E9V21AuditConfig,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the formal CPU-only audit and preserve the parent byte-for-byte."""

    if os.environ.get("CONFIRM_THOUGHT3_E9_V21_AUDIT") != "YES":
        raise E9V21AuditError(
            "set CONFIRM_THOUGHT3_E9_V21_AUDIT=YES for the formal audit"
        )
    output = ensure_thought3_output_path(cfg.output_root)
    result_path = output / "audit_result.json"
    if output.exists():
        if resume and result_path.is_file():
            return load_json(result_path)
        raise FileExistsError(
            f"audit output already exists; use --resume only to read it: {output}"
        )
    # Snapshot before creating any output.  Parent is a disjoint directory.
    before = _tree_snapshot(cfg.parent_root)
    started_at = _utc_now()
    torch_cuda_initialized_before: bool | None = None
    try:
        import torch

        torch_cuda_initialized_before = bool(torch.cuda.is_initialized())
    except ImportError:
        pass
    if torch_cuda_initialized_before:
        raise E9V21AuditError("CPU-only audit found CUDA already initialized")

    audit, derived_rows = _audit_parent(cfg, before)
    after = _tree_snapshot(cfg.parent_root)
    parent_immutable = before == after
    audit["checks"]["parent_tree_unchanged_before_after"] = parent_immutable
    torch_cuda_initialized_after: bool | None = None
    try:
        import torch

        torch_cuda_initialized_after = bool(torch.cuda.is_initialized())
    except ImportError:
        pass
    audit["checks"]["cuda_never_initialized"] = (
        torch_cuda_initialized_before is False
        and torch_cuda_initialized_after is False
    )
    audit_valid = _all_true(audit["checks"])
    outcome = VALID_OUTCOME if audit_valid else INVALID_OUTCOME
    result = {
        "schema_version": AUDIT_RESULT_SCHEMA,
        "audit_completed_at": _utc_now(),
        "audit_started_at": started_at,
        "audit_valid": audit_valid,
        "checks": audit["checks"],
        "config_fingerprint": cfg.fingerprint,
        "derived_identity": audit["derived_identity"],
        "outcome": outcome,
        "parent": {
            "artifact_count": len(before),
            "gate_sha256": before["gate_e9a_result.json"]["sha256"],
            "original_schema": cfg.original_schema,
            "original_status": audit["parent_original_status"],
            "root": str(cfg.parent_root),
            "tree_snapshot_sha256": _canonical_sha(before),
            "v2_artifacts_modified": not parent_immutable,
        },
        "provenance": audit["provenance"],
        "scope": {
            "backward_calls": 0,
            "checkpoint_tensor_loads": 0,
            "forward_calls": 0,
            "gpu_model_loads": 0,
            "optimizer_steps": 0,
            "parent_files_written": 0,
            "reserved_cohort_rows_read": 0,
            "torch_cuda_initialized_after": torch_cuda_initialized_after,
            "torch_cuda_initialized_before": torch_cuda_initialized_before,
        },
        "scientific_result": audit["scientific_result"],
        "track_audits": audit["track_audits"],
    }
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_jsonl(output / "derived_identity_manifest.jsonl", derived_rows)
    atomic_write_json(
        output / "parent_artifact_manifest.json",
        {
            "schema_version": (
                "thought3.phase_e9_v2_1.parent_artifacts.v1"
            ),
            "parent_root": str(cfg.parent_root),
            "files": before,
            "snapshot_sha256": _canonical_sha(before),
        },
    )
    atomic_write_json(result_path, result)
    atomic_write_json(
        output / "run_status.json",
        {
            "schema_version": (
                "thought3.phase_e9_v2_1.audit_run_status.v1"
            ),
            "finished_at": _utc_now(),
            "outcome": outcome,
            "parent_modified": not parent_immutable,
            "status": "completed" if audit_valid else "failed",
        },
    )
    manifest = _artifact_manifest(
        output,
        (
            "audit_result.json",
            "derived_identity_manifest.jsonl",
            "parent_artifact_manifest.json",
            "run_status.json",
        ),
    )
    atomic_write_json(output / "artifact_manifest.json", manifest)
    return result

