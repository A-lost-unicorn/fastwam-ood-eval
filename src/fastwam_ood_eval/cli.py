"""Command-line interface for planning, evaluation, aggregation and review."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from fastwam_ood_eval.analysis.aggregate import aggregate_experiment
from fastwam_ood_eval.analysis.report import generate_report
from fastwam_ood_eval.analysis.review import generate_failure_review
from fastwam_ood_eval.config import (
    ConfigError,
    EvalConfig,
    load_config,
    validate_hardware_inventory,
    validate_runtime_paths,
)
from fastwam_ood_eval.diagnostics.aggregate import aggregate_diagnostics
from fastwam_ood_eval.diagnostics.artifact_writer import (
    ensure_isolated_output,
    load_all_completed_jobs,
)
from fastwam_ood_eval.diagnostics.diagnostic_runner import (
    load_source_jobs,
    run_diagnostic_worker,
    validate_source_provenance,
)
from fastwam_ood_eval.diagnostics.report import generate_diagnostic_report
from fastwam_ood_eval.evaluation.distributed_launcher import distributed_evaluate
from fastwam_ood_eval.evaluation.evaluator import (
    _make_environment,
    _make_policy,
    evaluate_worker,
    git_commit,
    gpu_environment,
    plan_experiment,
    provenance,
)
from fastwam_ood_eval.evaluation.jobs import shard_jobs
from fastwam_ood_eval.logging_utils import configure_logging
from fastwam_ood_eval.policy.fastwam_future_probe import FastWAMFutureProbe
from fastwam_ood_eval.reproducibility import seed_everything


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path, help="Evaluation YAML file")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted YAML setting; repeat as needed",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fastwam-ood", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check Python, CUDA, dependencies, upstreams and checkpoint")
    doctor.add_argument("--config", type=Path, help="Optional config whose runtime paths should be checked")
    doctor.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")

    subparsers.add_parser("fetch-upstreams", help="Clone or report the three pinned-by-checkout upstream repositories")

    plan = subparsers.add_parser("plan", help="Write a deterministic JSONL job manifest without running a model")
    _add_config_arguments(plan)

    for name, help_text in (
        ("evaluate", "Evaluate one worker (or run a CPU mock backend)"),
        ("distributed-evaluate", "Evaluate the rank assigned by torchrun environment variables"),
    ):
        evaluate = subparsers.add_parser(name, help=help_text)
        _add_config_arguments(evaluate)
        evaluate.add_argument("--device", help="Explicit device, e.g. cuda:0 or cpu")
        evaluate.add_argument("--dry-run", action="store_true", help="Plan/filter jobs but load no model or environment")
        evaluate.add_argument(
            "--rerun",
            choices=("incomplete", "failed", "all"),
            default="incomplete",
            help="Which existing job records may be run again",
        )
        evaluate.add_argument("--overwrite", action="store_true", help="Run assigned jobs even if results already exist")

    for name, help_text in (
        ("diagnose-future", "Run one explicit shadow future-diagnostic worker"),
        (
            "distributed-diagnose-future",
            "Run the shadow future-diagnostic shard assigned by torchrun",
        ),
    ):
        diagnose = subparsers.add_parser(name, help=help_text)
        _add_config_arguments(diagnose)
        diagnose.add_argument("--device", help="Explicit device, e.g. cuda:0")
        diagnose.add_argument(
            "--dry-run",
            action="store_true",
            help="Read/filter/shard source jobs without loading a model or environment",
        )
        diagnose.add_argument(
            "--rerun",
            choices=("incomplete", "failed", "all"),
            default="incomplete",
            help="Which existing diagnostic job records may be run again",
        )
        diagnose.add_argument(
            "--overwrite",
            action="store_true",
            help="Run assigned jobs even if matching diagnostic records already exist",
        )

    for name, help_text in (
        ("aggregate", "Aggregate worker JSONL files into CSV/JSON summaries"),
        ("report", "Generate report.md from aggregated metrics"),
        ("review-failures", "Generate a standalone static failure review page"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--experiment-dir", required=True, type=Path)
        if name == "aggregate":
            command.add_argument(
                "--input-dir",
                action="append",
                default=[],
                type=Path,
                help="Additional experiment directory whose worker JSONL should be included",
            )

    aggregate_diagnostic = subparsers.add_parser(
        "aggregate-diagnostics",
        help="Aggregate only shadow diagnostic JSONL into diagnostic CSV summaries",
    )
    aggregate_diagnostic.add_argument("--experiment-dir", required=True, type=Path)
    aggregate_diagnostic.add_argument(
        "--input-dir",
        action="append",
        default=[],
        type=Path,
        help="Additional diagnostic experiment directory to include",
    )
    report_diagnostic = subparsers.add_parser(
        "report-diagnostics",
        help="Generate the thirteen-section Thought 2 diagnostic report",
    )
    report_diagnostic.add_argument("--experiment-dir", required=True, type=Path)
    return parser


def _load(args: argparse.Namespace) -> EvalConfig:
    overrides = list(args.set)
    if getattr(args, "overwrite", False):
        overrides.append("experiment.overwrite=true")
    cfg = load_config(args.config, overrides)
    configure_logging(cfg.experiment.log_level)
    return cfg


def _doctor(args: argparse.Namespace) -> int:
    report: dict[str, object] = {
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "packages": {},
        "upstreams": {},
        "gpu": gpu_environment(),
        "checks": [],
    }
    packages = report["packages"]
    assert isinstance(packages, dict)
    for name in (
        "yaml",
        "torch",
        "torchvision",
        "hydra",
        "imageio",
        "fastwam",
        "mujoco",
        "robosuite",
        "bddl",
        "robomimic",
    ):
        packages[name] = "available" if importlib.util.find_spec(name) is not None else "missing"
    # The two upstreams publish the same top-level ``libero`` package. Installing
    # either one globally makes backend selection order-dependent, so the
    # evaluator loads the selected checkout through its adapter instead.
    checkout_packages = {
        "LIBERO": Path("third_party/LIBERO/libero/libero/__init__.py"),
        "LIBERO-plus": Path("third_party/LIBERO-plus/libero/libero/__init__.py"),
    }
    available_checkouts = [name for name, path in checkout_packages.items() if path.is_file()]
    packages["libero"] = (
        f"available via checkout adapter ({', '.join(available_checkouts)})"
        if available_checkouts
        else "missing"
    )
    upstreams = report["upstreams"]
    assert isinstance(upstreams, dict)
    for name, path in (
        ("FastWAM", Path("third_party/FastWAM")),
        ("LIBERO", Path("third_party/LIBERO")),
        ("LIBERO-plus", Path("third_party/LIBERO-plus")),
    ):
        upstreams[name] = {"exists": path.is_dir(), "commit": git_commit(path)}
    status = 0
    if args.config:
        errors: list[str] = []
        try:
            cfg = load_config(args.config, args.set)
        except ConfigError as exc:
            errors.append(str(exc))
            cfg = None
        if cfg is not None:
            try:
                validate_runtime_paths(cfg, require_checkpoint=True)
            except ConfigError as exc:
                errors.append(str(exc))
            gpu = report["gpu"]
            assert isinstance(gpu, dict)
            try:
                validate_hardware_inventory(
                    cfg,
                    cuda_available=bool(gpu.get("cuda_available", False)),
                    device_memory_gb=[
                        float(device["total_memory_gb"])
                        for device in gpu.get("torch_devices", [])
                        if isinstance(device, dict) and "total_memory_gb" in device
                    ],
                    cuda_visible_devices=(
                        str(gpu["cuda_visible_devices"])
                        if gpu.get("cuda_visible_devices") not in (None, "")
                        else None
                    ),
                )
            except ConfigError as exc:
                errors.append(str(exc))
        if errors:
            report["checks"] = [f"ERROR: {error}" for error in errors]
            status = 1
        else:
            report["checks"] = [
                "configuration valid",
                "runtime paths present",
                "configured CUDA inventory available",
            ]
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return status


def _fetch_upstreams() -> int:
    script = Path("scripts/fetch_upstreams.sh")
    if not script.is_file():
        raise FileNotFoundError(script)
    return subprocess.run(["bash", str(script)], check=False).returncode


def _reject_diagnostics_for_thought1(cfg: EvalConfig, command: str) -> None:
    """Prevent an enabled shadow protocol from entering the Thought 1 paths."""

    if not cfg.diagnostics.enabled:
        return
    replacement = (
        "distributed-diagnose-future"
        if command == "distributed-evaluate"
        else "diagnose-future"
    )
    raise ConfigError(
        f"`fastwam-ood {command}` does not accept diagnostics.enabled=true. "
        f"Use the explicit shadow command `fastwam-ood {replacement}`; "
        "Thought 1 planning and evaluation remain unchanged."
    )


def _validate_diagnostic_config(cfg: EvalConfig) -> None:
    if not cfg.diagnostics.enabled:
        raise ConfigError(
            "Future diagnostics are disabled in this configuration; set "
            "diagnostics.enabled=true and use a separate diagnostic output directory."
        )


def _dry_run_diagnostics(
    cfg: EvalConfig,
    *,
    rank: int,
    world_size: int,
    rerun: str,
) -> dict[str, int]:
    """Perform a strictly read-only diagnostic selection/resume preview."""

    jobs = load_source_jobs(cfg)
    assigned = shard_jobs(jobs, rank, world_size)
    if cfg.experiment.overwrite or rerun == "all":
        pending = assigned
    else:
        # A real run fingerprints the checkpoint content and stores that exact
        # fingerprint in its manifest.  Reuse it when present; computing and
        # caching a checkpoint hash here would violate dry-run's no-write rule.
        manifest_path = cfg.experiment.output_dir / "diagnostic_manifest.json"
        fingerprint: str | None = None
        if manifest_path.is_file():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid diagnostic manifest: {manifest_path}") from exc
            if isinstance(payload, dict) and payload.get("protocol_fingerprint"):
                fingerprint = str(payload["protocol_fingerprint"])

        completed = load_all_completed_jobs(cfg.experiment.output_dir) if fingerprint else {}

        def should_run(job: object) -> bool:
            if fingerprint is None:
                return True
            previous = completed.get((str(getattr(job, "job_id")), fingerprint))
            if previous is None:
                return True
            if rerun == "failed":
                return previous.get("status") in {"error", "exception"} or previous.get(
                    "termination_reason"
                ) in {"exception", "max_steps"}
            return previous.get("status") not in {"completed", "skipped"}

        pending = [job for job in assigned if should_run(job)]
    return {
        "assigned": len(assigned),
        "pending": len(pending),
        "completed": 0,
        "probes": 0,
        "skipped_by_resume": len(assigned) - len(pending),
    }


def _select_diagnostic_device(
    cfg: EvalConfig,
    *,
    rank: int,
    world_size: int,
    device: str | None,
) -> str:
    selected = device or f"cuda:{os.environ.get('LOCAL_RANK', rank)}"
    if cfg.benchmark.backend == "mock":
        return "cpu"
    if world_size > len(cfg.hardware.devices):
        raise RuntimeError(
            f"world_size={world_size} exceeds configured hardware.devices={cfg.hardware.devices}"
        )
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed in the active environment") from exc
    validate_hardware_inventory(
        cfg,
        cuda_available=torch.cuda.is_available(),
        device_memory_gb=[
            torch.cuda.get_device_properties(index).total_memory / 2**30
            for index in range(torch.cuda.device_count())
        ],
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    if selected.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {selected}, but torch.cuda.is_available() is false")
    if selected.startswith("cuda"):
        try:
            device_index = int(selected.split(":", 1)[1])
        except (IndexError, ValueError):
            device_index = 0
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {selected}, but only {torch.cuda.device_count()} CUDA devices are visible"
            )
    return selected


def _preflight_diagnostic_output(cfg: EvalConfig) -> None:
    """Reject a Thought 1 namespace before model load or provenance writes."""

    output_dir = cfg.experiment.output_dir
    ensure_isolated_output(output_dir, cfg.diagnostics.source_output_dir)
    thought1_manifest = output_dir / "experiment_manifest.json"
    thought1_results = list(
        (output_dir / "workers").glob("rank_*/episode_results.jsonl")
    )
    if thought1_manifest.is_file() or thought1_results:
        raise RuntimeError(
            "Refusing to use a Thought 1 evaluation output as a diagnostic output: "
            f"{output_dir}"
        )


def _diagnose_future_worker(
    cfg: EvalConfig,
    *,
    rank: int = 0,
    world_size: int = 1,
    device: str | None = None,
    dry_run: bool = False,
    rerun: str = "incomplete",
) -> dict[str, int]:
    """CLI composition layer for the isolated diagnostic runner."""

    _validate_diagnostic_config(cfg)
    validate_runtime_paths(cfg, require_checkpoint=not dry_run)
    if dry_run:
        return _dry_run_diagnostics(cfg, rank=rank, world_size=world_size, rerun=rerun)

    jobs = load_source_jobs(cfg)
    _preflight_diagnostic_output(cfg)
    selected_device = _select_diagnostic_device(
        cfg,
        rank=rank,
        world_size=world_size,
        device=device,
    )
    seed_everything(cfg.experiment.seed + rank)

    # Ordering is intentional: the probe validates action-conditioned video
    # support before any simulator is constructed, reset, or policy action is
    # sampled.  The released unconditioned checkpoint therefore fails fast.
    policy = _make_policy(cfg, selected_device)
    try:
        probe = FastWAMFutureProbe(policy)
    except AttributeError as exc:
        policy.close()
        raise RuntimeError(
            "Non-dry future diagnostics require a loaded FastWAMAdapter; "
            f"got {type(policy).__name__}."
        ) from exc
    except Exception:
        policy.close()
        raise
    try:
        prov = provenance(cfg, hash_checkpoint=True)
        validate_source_provenance(cfg, prov)
        environment = _make_environment(cfg)
    except Exception:
        policy.close()
        raise
    try:
        return run_diagnostic_worker(
            cfg,
            policy=policy,
            environment=environment,
            probe=probe,
            jobs=jobs,
            rank=rank,
            world_size=world_size,
            provenance=prov,
            close_resources=False,
            rerun=rerun,
        )
    finally:
        environment.close()
        policy.close()


def _distributed_diagnose_future(
    cfg: EvalConfig,
    *,
    device: str | None = None,
    dry_run: bool = False,
    rerun: str = "incomplete",
) -> dict[str, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if os.environ.get("MUJOCO_GL", "").lower() == "egl":
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", str(local_rank))
    selected = device or (
        "cpu" if cfg.benchmark.backend == "mock" else f"cuda:{local_rank}"
    )
    return _diagnose_future_worker(
        cfg,
        rank=rank,
        world_size=world_size,
        device=selected,
        dry_run=dry_run,
        rerun=rerun,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return _doctor(args)
        if args.command == "fetch-upstreams":
            return _fetch_upstreams()
        if args.command == "plan":
            cfg = _load(args)
            _reject_diagnostics_for_thought1(cfg, args.command)
            path, jobs = plan_experiment(cfg)
            print(json.dumps({"job_manifest": str(path), "jobs": len(jobs)}, ensure_ascii=False))
            return 0
        if args.command == "evaluate":
            cfg = _load(args)
            _reject_diagnostics_for_thought1(cfg, args.command)
            result = evaluate_worker(
                cfg,
                device=args.device,
                dry_run=args.dry_run,
                rerun=args.rerun,
            )
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "distributed-evaluate":
            cfg = _load(args)
            _reject_diagnostics_for_thought1(cfg, args.command)
            result = distributed_evaluate(
                cfg,
                device=args.device,
                dry_run=args.dry_run,
                rerun=args.rerun,
            )
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "diagnose-future":
            cfg = _load(args)
            result = _diagnose_future_worker(
                cfg,
                device=args.device,
                dry_run=args.dry_run,
                rerun=args.rerun,
            )
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "distributed-diagnose-future":
            cfg = _load(args)
            result = _distributed_diagnose_future(
                cfg,
                device=args.device,
                dry_run=args.dry_run,
                rerun=args.rerun,
            )
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "aggregate":
            metrics = aggregate_experiment(args.experiment_dir, args.input_dir)
            report_path = generate_report(args.experiment_dir, metrics)
            print(json.dumps({"report": str(report_path), "episodes": metrics["all"]["episodes"]}))
            return 0
        if args.command == "report":
            print(generate_report(args.experiment_dir))
            return 0
        if args.command == "review-failures":
            print(generate_failure_review(args.experiment_dir))
            return 0
        if args.command == "aggregate-diagnostics":
            metrics = aggregate_diagnostics(args.experiment_dir, args.input_dir)
            print(
                json.dumps(
                    {
                        "summary": str(args.experiment_dir / "summary"),
                        "episodes": metrics["episodes"],
                        "clips": metrics["clips"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "report-diagnostics":
            print(generate_diagnostic_report(args.experiment_dir))
            return 0
    except (ConfigError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
