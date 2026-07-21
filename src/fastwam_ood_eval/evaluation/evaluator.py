"""Single-worker evaluator and experiment provenance."""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from fastwam_ood_eval.checkpoint import cached_sha256_file
from fastwam_ood_eval.config import EvalConfig, validate_runtime_paths
from fastwam_ood_eval.envs.base import BaseBenchmarkEnv
from fastwam_ood_eval.envs.libero_adapter import LiberoAdapter
from fastwam_ood_eval.envs.libero_plus_adapter import LiberoPlusAdapter
from fastwam_ood_eval.envs.mock import MockBenchmarkEnv
from fastwam_ood_eval.evaluation.episode_runner import run_episode
from fastwam_ood_eval.evaluation.jobs import EvaluationJob, plan_jobs, read_jobs, shard_jobs, write_jobs
from fastwam_ood_eval.evaluation.resume import filter_jobs_for_resume, load_result_records
from fastwam_ood_eval.policy.base import BasePolicy
from fastwam_ood_eval.policy.fastwam_adapter import FastWAMAdapter
from fastwam_ood_eval.policy.mock import MockPolicy
from fastwam_ood_eval.reproducibility import seed_everything
from fastwam_ood_eval.schemas.episode_result import append_result
from fastwam_ood_eval.schemas.experiment_manifest import write_manifest

LOGGER = logging.getLogger(__name__)


def git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def gpu_environment() -> dict[str, Any]:
    report: dict[str, Any] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
    }
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        report["nvidia_smi"] = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError) as exc:
        report["nvidia_smi_error"] = str(exc)
    try:
        import torch

        report.update(
            {
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
            }
        )
    except ImportError:
        report["torch_error"] = "not installed"
    return report


def provenance(cfg: EvalConfig, *, hash_checkpoint: bool) -> dict[str, Any]:
    return {
        "git_commit": git_commit(Path.cwd()),
        "fastwam_commit": git_commit(Path("third_party/FastWAM")),
        "libero_commit": git_commit(Path("third_party/LIBERO")),
        "libero_plus_commit": git_commit(Path("third_party/LIBERO-plus")),
        "checkpoint": str(cfg.checkpoint.path) if cfg.checkpoint.path else None,
        "checkpoint_hash": (
            cached_sha256_file(cfg.checkpoint.path, cfg.experiment.output_dir / "checkpoint_hash.json")
            if hash_checkpoint
            else None
        ),
    }


def plan_experiment(cfg: EvalConfig) -> tuple[Path, list[EvaluationJob]]:
    jobs = plan_jobs(cfg)
    output_dir = cfg.experiment.output_dir
    manifest_path = output_dir / "job_manifest.jsonl"
    write_jobs(manifest_path, jobs)
    manifest = {
        "experiment_id": cfg.experiment.name,
        "status": "planned_not_run",
        "config_source": str(cfg.source_path),
        "config": cfg.to_dict(),
        "job_count": len(jobs),
        "job_manifest": str(manifest_path),
        "provenance": provenance(cfg, hash_checkpoint=False),
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "gpu_environment": gpu_environment(),
        "limitations": [
            "Planning does not prove that the checkpoint, assets, simulator, or CUDA runtime works.",
            "LIBERO-Plus parameters are official pre-generated variant metadata, not synthesized changes.",
        ],
    }
    write_manifest(output_dir / "experiment_manifest.json", manifest)
    return manifest_path, jobs


def _make_environment(cfg: EvalConfig) -> BaseBenchmarkEnv:
    if cfg.benchmark.backend == "mock":
        return MockBenchmarkEnv(max_steps=cfg.benchmark.max_steps)
    if cfg.benchmark.backend == "libero":
        return LiberoAdapter(
            cfg.benchmark.image_size,
            config_dir=cfg.experiment.output_dir / "runtime" / "libero",
        )
    if cfg.benchmark.backend == "libero_plus":
        return LiberoPlusAdapter(
            cfg.benchmark.image_size,
            config_dir=cfg.experiment.output_dir / "runtime" / "libero_plus",
        )
    raise ValueError(f"Unsupported backend: {cfg.benchmark.backend}")


def _make_policy(cfg: EvalConfig, device: str) -> BasePolicy:
    if cfg.benchmark.backend == "mock":
        return MockPolicy(control_horizon=cfg.benchmark.control_horizon)
    return FastWAMAdapter(cfg, device=device)


def evaluate_worker(
    cfg: EvalConfig,
    *,
    rank: int = 0,
    world_size: int = 1,
    device: str | None = None,
    dry_run: bool = False,
    rerun: str = "incomplete",
) -> dict[str, int]:
    validate_runtime_paths(cfg, require_checkpoint=not dry_run)
    manifest_path = cfg.experiment.output_dir / "job_manifest.jsonl"
    if manifest_path.is_file():
        jobs = read_jobs(manifest_path)
    else:
        manifest_path, jobs = plan_experiment(cfg)
    assigned = shard_jobs(jobs, rank, world_size)
    all_result_paths = list((cfg.experiment.output_dir / "workers").glob("rank_*/episode_results.jsonl"))
    previous = load_result_records(all_result_paths) if cfg.experiment.resume else {}
    pending = filter_jobs_for_resume(
        assigned,
        previous,
        overwrite=cfg.experiment.overwrite,
        rerun=rerun,
    )
    LOGGER.info(
        "rank=%d/%d assigned=%d pending=%d skipped_by_resume=%d manifest=%s",
        rank,
        world_size,
        len(assigned),
        len(pending),
        len(assigned) - len(pending),
        manifest_path,
    )
    if dry_run:
        return {"assigned": len(assigned), "pending": len(pending), "completed": 0}

    selected_device = device or (f"cuda:{os.environ.get('LOCAL_RANK', rank)}")
    if cfg.benchmark.backend == "mock":
        selected_device = "cpu"
    else:
        if world_size > len(cfg.hardware.devices):
            raise RuntimeError(
                f"world_size={world_size} exceeds configured hardware.devices={cfg.hardware.devices}"
            )
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is not installed in the active environment") from exc
        if selected_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested {selected_device}, but torch.cuda.is_available() is false")
        if selected_device.startswith("cuda"):
            try:
                device_index = int(selected_device.split(":", 1)[1])
            except (IndexError, ValueError):
                device_index = 0
            if device_index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"Requested {selected_device}, but only {torch.cuda.device_count()} CUDA devices are visible"
                )
    seed_everything(cfg.experiment.seed + rank)
    worker_dir = cfg.experiment.output_dir / "workers" / f"rank_{rank}"
    result_path = worker_dir / "episode_results.jsonl"
    environment = _make_environment(cfg)
    policy = _make_policy(cfg, selected_device)
    prov = provenance(cfg, hash_checkpoint=True)
    if rank == 0:
        manifest_file = cfg.experiment.output_dir / "experiment_manifest.json"
        try:
            manifest_payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest_payload = {"experiment_id": cfg.experiment.name, "config": cfg.to_dict()}
        manifest_payload.update(
            {
                "status": "worker_outputs_present",
                "provenance": prov,
                "execution": {
                    "world_size": world_size,
                    "device": selected_device,
                    "rank_zero_started": True,
                    "environment": (
                        environment.runtime_config()
                        if callable(getattr(environment, "runtime_config", None))
                        else {"backend": cfg.benchmark.backend}
                    ),
                },
            }
        )
        write_manifest(manifest_file, manifest_payload)
    completed = 0
    try:
        for index, job in enumerate(pending, start=1):
            LOGGER.info("rank=%d job=%d/%d id=%s", rank, index, len(pending), job.job_id)
            result = run_episode(
                cfg=cfg,
                job=job,
                policy=policy,
                environment=environment,
                worker_rank=rank,
                provenance=prov,
                worker_dir=worker_dir,
            )
            append_result(result_path, result)
            completed += 1
    finally:
        environment.close()
        policy.close()
    return {"assigned": len(assigned), "pending": len(pending), "completed": completed}
