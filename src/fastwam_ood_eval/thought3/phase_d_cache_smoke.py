"""Phase D: one-task, 32-sample real future-latent cache gate."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from fastwam_ood_eval.thought3.cache_builder import build_cache
from fastwam_ood_eval.thought3.cache_planner import (
    InventorySample,
    load_cache_plan,
    write_cache_plan,
)
from fastwam_ood_eval.thought3.cache_validator import validate_cache
from fastwam_ood_eval.thought3.config import Thought3Config
from fastwam_ood_eval.thought3.future_cache import (
    CacheValidationError,
    ShardPaths,
    validate_cache_shard,
)
from fastwam_ood_eval.thought3.io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    load_json,
    load_jsonl,
    sha256_file,
)
from fastwam_ood_eval.thought3.real_cache_builder import (
    load_real_build_report,
)
from fastwam_ood_eval.thought3.safety import (
    ensure_standard_training_source,
    ensure_thought3_output_path,
)
from fastwam_ood_eval.thought3.schemas import sha256_canonical


PHASE_D_SCHEMA = "thought3.phase_d.gate.v1"
OFFICIAL_LIBERO_REVISION = "117413dc0ca99c7cd64036c4eaa4a316c537d692"
OFFICIAL_ARCHIVE_SHA256 = (
    "a21ae10171535585fb43e6405d9efa09ff38ef34689e4176428ca005af3a39ea"
)
PHASE_D_TASK_INDEX = 0
PHASE_D_TASK_NAME = "open the middle drawer of the cabinet"
PHASE_D_INVENTORY_MANIFEST = "phase_d_inventory_manifest.json"
PHASE_C_FROZEN = {
    "outputs/thought3/phase_c_single_sample_v1/run_status.json": (
        "581de5813e11fd19c8d7a1433c511c1a32e896900f62677ac3e47330d3f3bc33"
    ),
    "outputs/thought3/phase_c_single_sample_v1/gate_c_result.json": (
        "ccac9ac39fd7920dc89726313b89a3ae16ab71b5494b072d0b6c6ba6778d3f02"
    ),
    "outputs/thought3/phase_c_single_sample_v1/logs/phase_c.log": (
        "f09670e9e5bd8bdb9ddd51653d71f7f5759c8f51cc3cb079a1c95993c5e648d2"
    ),
}


class PhaseDGateError(RuntimeError):
    """Raised when a Phase D hard gate fails."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress(stage: str, **fields: Any) -> None:
    print(
        json.dumps(
            {"phase": "D", "stage": stage, **fields},
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _assert_phase_d_scope(cfg: Thought3Config) -> None:
    if cfg.runtime.backend != "fastwam":
        raise PhaseDGateError("Phase D requires runtime.backend=fastwam")
    if cfg.runtime.device != "cuda:0":
        raise PhaseDGateError("Phase D requires logical runtime.device=cuda:0")
    if cfg.variant != "A1" or cfg.sampler.active_k != 1:
        raise PhaseDGateError(
            "Phase D orchestration config is frozen to variant=A1/active_k=1"
        )
    if tuple(cfg.sampler.cache_k) != (1, 2, 4):
        raise PhaseDGateError("Phase D must cache exactly K=1/2/4")
    if cfg.cache.pilot_limit != 32 or cfg.cache.shard_size != 8:
        raise PhaseDGateError(
            "Phase D is frozen to pilot_limit=32 and shard_size=8"
        )
    if len(cfg.data.dataset_roots) != 1:
        raise PhaseDGateError("Phase D accepts one standard LIBERO root")
    if cfg.data.dataset_revision != OFFICIAL_LIBERO_REVISION:
        raise PhaseDGateError("Phase D dataset revision mismatch")
    if cfg.data.inventory_path is None:
        raise PhaseDGateError("Phase D requires data.inventory_path")
    if cfg.training.max_steps != 1:
        raise PhaseDGateError(
            "Phase D training.max_steps is a scope sentinel and must remain 1"
        )
    if cfg.runtime.online_use_cache:
        raise PhaseDGateError("Phase D forbids online cache reads")


def _verify_phase_c_gate() -> dict[str, str]:
    observed: dict[str, str] = {}
    for raw_path, expected in PHASE_C_FROZEN.items():
        path = Path(raw_path)
        if not path.is_file():
            raise PhaseDGateError(f"frozen Phase C artifact missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise PhaseDGateError(
                f"frozen Phase C artifact changed: {path}; "
                f"expected={expected}, actual={actual}"
            )
        observed[raw_path] = actual
    status = load_json(
        "outputs/thought3/phase_c_single_sample_v1/run_status.json"
    )
    result = load_json(
        "outputs/thought3/phase_c_single_sample_v1/gate_c_result.json"
    )
    if (
        status.get("gate_c_passed") is not True
        or result.get("gate_c_passed") is not True
    ):
        raise PhaseDGateError("frozen Phase C artifacts do not report pass")
    return observed


def _read_tasks(path: Path) -> dict[int, str]:
    tasks: dict[int, str] = {}
    for row in load_jsonl(path):
        index = int(row["task_index"])
        task = str(row["task"])
        if index in tasks:
            raise PhaseDGateError(f"duplicate task index in {path}: {index}")
        tasks[index] = task
    return tasks


def _inventory_rows(
    dataset_root: Path,
    *,
    camera_keys: Sequence[str],
) -> tuple[list[InventorySample], dict[str, Any]]:
    import pyarrow.parquet as pq

    meta = dataset_root / "meta"
    info_path = meta / "info.json"
    tasks_path = meta / "tasks.jsonl"
    episodes_path = meta / "episodes.jsonl"
    info = load_json(info_path)
    tasks = _read_tasks(tasks_path)
    if tasks.get(PHASE_D_TASK_INDEX) != PHASE_D_TASK_NAME:
        raise PhaseDGateError(
            "pinned Phase D task text differs from dataset metadata"
        )
    chunk_size = int(info["chunks_size"])
    selected_episodes = [
        row
        for row in load_jsonl(episodes_path)
        if list(row.get("tasks", [])) == [PHASE_D_TASK_NAME]
    ]
    if len(selected_episodes) < 2:
        raise PhaseDGateError("Phase D task has fewer than two episodes")

    values: list[InventorySample] = []
    parquet_hashes: dict[str, str] = {}
    columns = [
        "timestamp",
        "frame_index",
        "episode_index",
        "task_index",
    ]
    for episode in selected_episodes:
        episode_index = int(episode["episode_index"])
        relative = Path(
            str(info["data_path"]).format(
                episode_chunk=episode_index // chunk_size,
                episode_index=episode_index,
            )
        )
        parquet = dataset_root / relative
        if not parquet.is_file():
            raise FileNotFoundError(parquet)
        table = pq.read_table(parquet, columns=columns).slice(0, 1)
        if table.num_rows != 1:
            raise PhaseDGateError(f"empty episode parquet: {parquet}")
        row = table.to_pydict()
        observed_episode = int(row["episode_index"][0])
        frame_index = int(row["frame_index"][0])
        task_index = int(row["task_index"][0])
        timestamp = float(row["timestamp"][0])
        if (
            observed_episode != episode_index
            or frame_index != 0
            or task_index != PHASE_D_TASK_INDEX
            or timestamp < 0
        ):
            raise PhaseDGateError(
                f"unexpected first row in {parquet}: "
                f"episode={observed_episode}, frame={frame_index}, "
                f"task={task_index}, timestamp={timestamp}"
            )
        values.append(
            InventorySample(
                suite="libero_goal",
                task_id=f"task_{PHASE_D_TASK_INDEX}",
                task_name=PHASE_D_TASK_NAME,
                demonstration_id=f"episode_{episode_index:06d}",
                episode_index=episode_index,
                frame_index=frame_index,
                timestamp_ns=int(round(timestamp * 1_000_000_000)),
                camera_keys=tuple(str(key) for key in camera_keys),
                language=PHASE_D_TASK_NAME,
            )
        )
        parquet_hashes[str(relative)] = sha256_file(parquet)
    values.sort(key=lambda sample: sample.episode_index)
    inventory = {
        "camera_keys": list(camera_keys),
        "current_frames_per_episode": 1,
        "dataset_info_sha256": sha256_file(info_path),
        "dataset_root": str(dataset_root),
        "episode_count": len(values),
        "episodes_metadata_sha256": sha256_file(episodes_path),
        "future_rgb_requested": False,
        "parquet_sha256": parquet_hashes,
        "selection": "first_frame_per_episode",
        "task_id": f"task_{PHASE_D_TASK_INDEX}",
        "task_index": PHASE_D_TASK_INDEX,
        "task_name": PHASE_D_TASK_NAME,
        "tasks_metadata_sha256": sha256_file(tasks_path),
    }
    inventory["source_fingerprint"] = sha256_canonical(inventory)
    return values, inventory


def prepare_phase_d_inventory(
    cfg: Thought3Config,
    *,
    resume: bool,
) -> dict[str, Any]:
    """Create a deterministic one-current-frame-per-episode inventory."""

    _assert_phase_d_scope(cfg)
    root = ensure_standard_training_source(cfg.data.dataset_roots[0])
    archive = root.parent / f"{root.name}.tar.gz"
    if not archive.is_file():
        raise FileNotFoundError(archive)
    archive_hash = sha256_file(archive)
    if archive_hash != OFFICIAL_ARCHIVE_SHA256:
        raise PhaseDGateError(
            "official Phase D dataset archive SHA-256 mismatch"
        )
    samples, source = _inventory_rows(
        root,
        camera_keys=cfg.data.camera_keys,
    )
    inventory_path = cfg.data.inventory_path
    assert inventory_path is not None
    inventory_path = ensure_thought3_output_path(inventory_path)
    manifest_path = inventory_path.parent / PHASE_D_INVENTORY_MANIFEST
    rows = [sample.to_dict() for sample in samples]

    if inventory_path.exists() or manifest_path.exists():
        if not (inventory_path.is_file() and manifest_path.is_file()):
            raise FileExistsError(
                "partial Phase D inventory artifacts require manual audit"
            )
        if not resume:
            raise FileExistsError(
                f"Phase D inventory exists; pass --resume: {inventory_path}"
            )
        existing_rows = load_jsonl(inventory_path)
        existing_manifest = load_json(manifest_path)
        if existing_rows != rows:
            raise PhaseDGateError(
                "existing Phase D inventory differs from deterministic replay"
            )
        if (
            existing_manifest.get("inventory_sha256")
            != sha256_file(inventory_path)
            or existing_manifest.get("source_fingerprint")
            != source["source_fingerprint"]
        ):
            raise PhaseDGateError(
                "existing Phase D inventory manifest/checksum mismatch"
            )
        return existing_manifest

    atomic_write_jsonl(inventory_path, rows)
    manifest = {
        **source,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_hash,
        "dataset_revision": cfg.data.dataset_revision,
        "inventory_file": inventory_path.name,
        "inventory_sha256": sha256_file(inventory_path),
        "schema_version": "thought3.phase_d.inventory.v1",
        "uses_ground_truth_future": False,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def _validate_plan_scope(
    cfg: Thought3Config,
) -> dict[str, Any]:
    entries, manifest = load_cache_plan(cfg.cache.root)
    base_entries = {
        entry.identity.base_sample_id: entry
        for entry in entries
        if entry.k == 1
    }
    if len(base_entries) != 32 or len(entries) != 96:
        raise PhaseDGateError("Phase D plan must contain 32×3 entries")
    tasks = {
        (entry.identity.suite, entry.identity.task_id, entry.identity.task_name)
        for entry in base_entries.values()
    }
    if tasks != {
        ("libero_goal", "task_0", PHASE_D_TASK_NAME)
    }:
        raise PhaseDGateError(f"Phase D plan is not one-task: {tasks}")
    episode_ids = {
        entry.identity.episode_id for entry in base_entries.values()
    }
    if len(episode_ids) != 32:
        raise PhaseDGateError(
            "Phase D pilot must use one current frame from 32 distinct episodes"
        )
    split_counts = Counter(
        entry.split for entry in base_entries.values()
    )
    if not split_counts["train"] or not split_counts["development"]:
        raise PhaseDGateError(
            "selected Phase D cache must retain both train and development"
        )
    split_manifest = load_json(
        ensure_thought3_output_path(cfg.cache.root)
        / str(manifest["split_file"])
    )
    stratum = split_manifest["strata"].get("libero_goal/task_0")
    if (
        not isinstance(stratum, Mapping)
        or int(stratum["total"]) != 42
        or int(stratum["train"]) != 37
        or int(stratum["development"]) != 5
    ):
        raise PhaseDGateError(
            f"unexpected full episode split: {stratum}"
        )
    return {
        "cache_fingerprint": manifest["cache_fingerprint"],
        "entry_count": len(entries),
        "full_episode_split": dict(stratum),
        "sample_count": len(base_entries),
        "selected_episode_count": len(episode_ids),
        "selected_split": dict(split_counts),
        "split_fingerprint": manifest["split_fingerprint"],
        "task": PHASE_D_TASK_NAME,
    }


def _active_corruption_probe(
    cfg: Thought3Config,
    *,
    cache_fingerprint: str,
) -> dict[str, Any]:
    """Corrupt only a temporary copy and require checksum rejection."""

    manifest_path = sorted(
        ensure_thought3_output_path(cfg.cache.root).glob(
            "k1/shard_*.manifest.json"
        )
    )[0]
    manifest = load_json(manifest_path)
    source_paths = ShardPaths(
        tensor=manifest_path.parent / str(manifest["tensor_file"]),
        metadata=manifest_path.parent / str(manifest["metadata_file"]),
        manifest=manifest_path,
    )
    primary_before = sha256_file(source_paths.tensor)
    temp_parent = Path("/tmp/thought3")
    temp_parent.mkdir(parents=True, exist_ok=True)
    detected = False
    message = ""
    with tempfile.TemporaryDirectory(
        prefix="phase_d_corruption_",
        dir=temp_parent,
    ) as raw_temp:
        temp = Path(raw_temp)
        copied = ShardPaths(
            tensor=temp / source_paths.tensor.name,
            metadata=temp / source_paths.metadata.name,
            manifest=temp / source_paths.manifest.name,
        )
        shutil.copy2(source_paths.tensor, copied.tensor)
        shutil.copy2(source_paths.metadata, copied.metadata)
        shutil.copy2(source_paths.manifest, copied.manifest)
        with copied.tensor.open("r+b") as handle:
            handle.seek(-1, os.SEEK_END)
            byte = handle.read(1)
            handle.seek(-1, os.SEEK_END)
            handle.write(bytes([byte[0] ^ 0x01]))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            validate_cache_shard(
                copied,
                expected_cache_fingerprint=cache_fingerprint,
            )
        except CacheValidationError as exc:
            detected = "checksum mismatch" in str(exc)
            message = str(exc)
    primary_after = sha256_file(source_paths.tensor)
    if not detected or primary_before != primary_after:
        raise PhaseDGateError(
            "temporary corruption was not detected or primary cache changed"
        )
    return {
        "detected": True,
        "primary_unchanged": True,
        "rejection": message,
        "temporary_copy_only": True,
    }


def _cache_artifacts(cfg: Thought3Config) -> dict[str, Any]:
    root = ensure_thought3_output_path(cfg.cache.root)
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    return {
        "file_count": len(paths),
        "files": {
            str(path.relative_to(root)): sha256_file(path)
            for path in paths
        },
        "total_bytes": sum(path.stat().st_size for path in paths),
    }


def _run_phase_d(cfg: Thought3Config, *, resume: bool) -> dict[str, Any]:
    import torch

    _assert_phase_d_scope(cfg)
    if os.environ.get("CONFIRM_THOUGHT3_PHASE_D") != "YES":
        raise PhaseDGateError(
            "set CONFIRM_THOUGHT3_PHASE_D=YES for the real cache gate"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise PhaseDGateError(
            "Phase D requires exactly one CUDA-visible GPU"
        )
    if cfg.runtime.device != "cuda:0":
        raise PhaseDGateError("Phase D requires logical cuda:0")
    phase_c_hashes = _verify_phase_c_gate()
    _progress("phase_c_verified")

    inventory = prepare_phase_d_inventory(cfg, resume=resume)
    _progress(
        "inventory_ready",
        episodes=inventory["episode_count"],
        task=inventory["task_name"],
    )
    plan = write_cache_plan(cfg, resume=resume)
    plan_scope = _validate_plan_scope(cfg)
    _progress(
        "cache_plan_ready",
        cache_fingerprint=plan["cache_fingerprint"],
        entries=plan["entry_count"],
        samples=plan["sample_count"],
    )

    first_build = build_cache(
        cfg,
        resume=resume,
        rank=0,
        world_size=1,
        device=cfg.runtime.device,
    )
    build_report = (
        first_build
        if first_build.get("model_loaded") is True
        else load_real_build_report(cfg.cache.root)
    )
    _progress(
        "cache_built",
        built_shards=first_build["built_shards"],
        model_loaded=first_build["model_loaded"],
    )
    validation = validate_cache(cfg.cache.root)
    _progress(
        "cache_validated",
        entries=validation["entry_count"],
        shards=validation["shard_count"],
    )

    resume_proof = build_cache(
        cfg,
        resume=True,
        rank=0,
        world_size=1,
        device=cfg.runtime.device,
    )
    if (
        resume_proof.get("built_shards") != 0
        or resume_proof.get("skipped_valid_shards")
        != resume_proof.get("total_shards")
        or resume_proof.get("model_loaded") is not False
    ):
        raise PhaseDGateError(
            f"real cache resume proof failed: {resume_proof}"
        )
    validation_after_resume = validate_cache(cfg.cache.root)
    if validation_after_resume != validation:
        raise PhaseDGateError("cache validation changed after no-op resume")
    corruption = _active_corruption_probe(
        cfg,
        cache_fingerprint=str(plan["cache_fingerprint"]),
    )
    _progress("resume_and_corruption_verified")

    source = build_report["current_source"]
    no_leakage = {
        "action_target_read": source["action_target_read"],
        "actual_future_read": source["actual_future_read"],
        "current_camera_frames_decoded": source[
            "current_camera_frames_decoded"
        ],
        "future_rgb_frames_decoded": source[
            "future_rgb_frames_decoded"
        ],
        "inventory_future_rgb_requested": inventory[
            "future_rgb_requested"
        ],
        "uses_ground_truth_future": validation[
            "uses_ground_truth_future"
        ],
    }
    if no_leakage != {
        "action_target_read": False,
        "actual_future_read": False,
        "current_camera_frames_decoded": 64,
        "future_rgb_frames_decoded": 0,
        "inventory_future_rgb_requested": False,
        "uses_ground_truth_future": False,
    }:
        raise PhaseDGateError(
            f"Phase D current-only access audit failed: {no_leakage}"
        )
    artifacts = _cache_artifacts(cfg)
    result: dict[str, Any] = {
        "artifacts": artifacts,
        "build": build_report,
        "cache_validation": validation,
        "completed_at": _utc_now(),
        "config": cfg.to_dict(),
        "config_fingerprint": cfg.fingerprint,
        "corruption_detection": corruption,
        "gate_d_passed": True,
        "inventory": inventory,
        "no_future_rgb_leakage": no_leakage,
        "phase_c_frozen_sha256": phase_c_hashes,
        "plan": plan_scope,
        "resume_proof": resume_proof,
        "schema_version": PHASE_D_SCHEMA,
        "scope": {
            "adapter_training_started": False,
            "cache_base_samples": 32,
            "cache_entries": 96,
            "cache_k": [1, 2, 4],
            "long_training_started": False,
            "optimizer_created": False,
            "optimizer_steps": 0,
            "single_gpu": True,
            "suite_count": 1,
            "task_count": 1,
        },
        "status": "passed",
    }
    return result


def run_phase_d_cache_smoke(
    cfg: Thought3Config,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Run Gate D and atomically record pass/fail under outputs/thought3."""

    output = ensure_thought3_output_path(cfg.experiment.output_dir)
    result_path = output / "gate_d_result.json"
    status_path = output / "run_status.json"
    if result_path.exists() and not resume:
        raise FileExistsError(
            f"Phase D result exists; pass --resume: {result_path}"
        )
    if result_path.exists() and resume:
        value = load_json(result_path)
        if value.get("gate_d_passed") is True:
            return value
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        status_path,
        {
            "schema_version": PHASE_D_SCHEMA,
            "started_at": _utc_now(),
            "status": "running",
        },
    )
    started = time.perf_counter()
    try:
        result = _run_phase_d(cfg, resume=resume)
    except BaseException as exc:
        atomic_write_json(
            status_path,
            {
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at": _utc_now(),
                "gate_d_passed": False,
                "schema_version": PHASE_D_SCHEMA,
                "status": "failed",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    result["gate_wall_s"] = time.perf_counter() - started
    atomic_write_json(result_path, result)
    atomic_write_json(
        status_path,
        {
            "finished_at": _utc_now(),
            "gate_d_passed": True,
            "result": str(result_path.resolve()),
            "schema_version": PHASE_D_SCHEMA,
            "status": "passed",
        },
    )
    return result
