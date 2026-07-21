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
from fastwam_ood_eval.config import ConfigError, EvalConfig, load_config, validate_runtime_paths
from fastwam_ood_eval.evaluation.distributed_launcher import distributed_evaluate
from fastwam_ood_eval.evaluation.evaluator import evaluate_worker, git_commit, gpu_environment, plan_experiment
from fastwam_ood_eval.logging_utils import configure_logging


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
    for name in ("yaml", "torch", "torchvision", "hydra", "libero", "imageio"):
        packages[name] = "available" if importlib.util.find_spec(name) is not None else "missing"
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
        try:
            cfg = load_config(args.config, args.set)
            validate_runtime_paths(cfg, require_checkpoint=True)
            report["checks"] = ["configuration valid", "runtime paths present"]
        except ConfigError as exc:
            report["checks"] = [f"ERROR: {exc}"]
            status = 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return status


def _fetch_upstreams() -> int:
    script = Path("scripts/fetch_upstreams.sh")
    if not script.is_file():
        raise FileNotFoundError(script)
    return subprocess.run(["bash", str(script)], check=False).returncode


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
            path, jobs = plan_experiment(cfg)
            print(json.dumps({"job_manifest": str(path), "jobs": len(jobs)}, ensure_ascii=False))
            return 0
        if args.command == "evaluate":
            cfg = _load(args)
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
            result = distributed_evaluate(
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
    except (ConfigError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
