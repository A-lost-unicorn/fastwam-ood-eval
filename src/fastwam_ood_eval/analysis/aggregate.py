"""Aggregate durable worker JSONL files into experiment summaries."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from fastwam_ood_eval.analysis.confidence_intervals import bootstrap_mean_ci
from fastwam_ood_eval.analysis.robustness_metrics import (
    absolute_drop,
    paired_outcomes,
    paired_policy_comparisons,
    relative_drop,
)
from fastwam_ood_eval.evaluation.episode_runner import percentile
from fastwam_ood_eval.evaluation.resume import load_result_records


def discover_result_files(experiment_dir: Path, input_dirs: Iterable[Path] = ()) -> list[Path]:
    roots = [experiment_dir, *input_dirs]
    paths = {path.resolve() for root in roots for path in (root / "workers").glob("rank_*/episode_results.jsonl")}
    return sorted(paths)


def load_results(experiment_dir: Path, input_dirs: Iterable[Path] = ()) -> list[dict[str, Any]]:
    paths = discover_result_files(experiment_dir, input_dirs)
    combined = experiment_dir / "summary" / "episode_results.jsonl"
    if not paths and combined.is_file():
        paths = [combined]
    records = load_result_records(paths)
    return sorted(
        records.values(),
        key=lambda row: (
            str(row.get("suite")),
            str(row.get("policy_variant", "unspecified")),
            int(row.get("task_id", -1)),
            str(row.get("condition")),
            str(row.get("perturbation_category")),
            str(row.get("perturbation_level")),
            int(row.get("episode_index", -1)),
        ),
    )


def _attempted(row: dict[str, Any]) -> bool:
    return row.get("termination_reason") != "skipped"


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = [row for row in rows if _attempted(row)]
    successes = sum(bool(row.get("success")) for row in attempted)
    success_values = [float(bool(row.get("success"))) for row in attempted]
    low, high = bootstrap_mean_ci(success_values)
    latencies = [float(row.get("policy_latency_mean_ms", 0.0)) for row in attempted]
    steps = [float(row.get("steps", 0)) for row in attempted]
    memory = [float(row.get("gpu_peak_memory_mb", 0.0)) for row in attempted]
    return {
        "episodes": len(rows),
        "attempted": len(attempted),
        "successes": successes,
        "failures": len(attempted) - successes,
        "exceptions": sum(row.get("termination_reason") == "exception" for row in rows),
        "skipped": sum(row.get("termination_reason") == "skipped" for row in rows),
        "success_rate": successes / len(attempted) if attempted else None,
        "success_ci95_low": low,
        "success_ci95_high": high,
        "mean_steps": statistics.fmean(steps) if steps else None,
        "mean_inference_latency_ms": statistics.fmean(latencies) if latencies else None,
        "p50_inference_latency_ms": percentile(latencies, 0.50) if latencies else None,
        "p95_inference_latency_ms": percentile(latencies, 0.95) if latencies else None,
        "gpu_peak_memory_mb": max(memory) if memory else None,
    }


def _group_summary(
    rows: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], tuple[Any, ...]],
    key_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    result: list[dict[str, Any]] = []
    for keys in sorted(grouped, key=lambda item: tuple("" if value is None else str(value) for value in item)):
        result.append({**dict(zip(key_names, keys)), **summarize_rows(grouped[keys])})
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    else:
        fields = list(fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            encoded = {
                key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            }
            writer.writerow(encoded)


def aggregate_experiment(experiment_dir: Path, input_dirs: Iterable[Path] = ()) -> dict[str, Any]:
    sources = [Path(path) for path in input_dirs]
    rows = load_results(experiment_dir, sources)
    summary_dir = experiment_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    combined_jsonl = summary_dir / "episode_results.jsonl"
    with combined_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    _write_csv(summary_dir / "episode_results.csv", rows)

    by_task = _group_summary(
        rows,
        lambda row: (
            row.get("policy_variant", "unspecified"),
            row.get("suite"),
            row.get("task_id"),
            row.get("task_name"),
            row.get("condition"),
        ),
        ("policy_variant", "suite", "task_id", "task_name", "condition"),
    )
    by_perturbation = _group_summary(
        rows,
        lambda row: (
            row.get("policy_variant", "unspecified"),
            row.get("condition"),
            row.get("perturbation_category") or "clean",
        ),
        ("policy_variant", "condition", "perturbation_category"),
    )
    by_level = _group_summary(
        rows,
        lambda row: (
            row.get("policy_variant", "unspecified"),
            row.get("condition"),
            row.get("perturbation_level") or "clean",
        ),
        ("policy_variant", "condition", "perturbation_level"),
    )
    by_policy = _group_summary(
        rows,
        lambda row: (
            row.get("policy_variant", "unspecified"),
            bool(row.get("test_time_future_imagination", False)),
        ),
        ("policy_variant", "test_time_future_imagination"),
    )
    _write_csv(summary_dir / "summary_by_task.csv", by_task)
    _write_csv(summary_dir / "summary_by_perturbation.csv", by_perturbation)
    _write_csv(summary_dir / "summary_by_level.csv", by_level)
    _write_csv(summary_dir / "summary_by_policy.csv", by_policy)
    failures = [row for row in rows if not row.get("success") and row.get("termination_reason") != "skipped"]
    _write_csv(summary_dir / "failures.csv", failures)

    clean_rows = [row for row in rows if row.get("condition") == "clean"]
    ood_rows = [row for row in rows if row.get("condition") == "ood"]
    variants = sorted({str(row.get("policy_variant", "unspecified")) for row in rows})
    robustness_by_policy: list[dict[str, Any]] = []
    checkpoint_hashes_by_policy: dict[str, dict[str, list[str]]] = {}
    for variant in variants:
        variant_rows = [row for row in rows if str(row.get("policy_variant", "unspecified")) == variant]
        variant_clean = [row for row in variant_rows if row.get("condition") == "clean"]
        variant_ood = [row for row in variant_rows if row.get("condition") == "ood"]
        variant_clean_hashes = {
            str(row["checkpoint_hash"]) for row in variant_clean if row.get("checkpoint_hash")
        }
        variant_ood_hashes = {
            str(row["checkpoint_hash"]) for row in variant_ood if row.get("checkpoint_hash")
        }
        if variant_clean_hashes and variant_ood_hashes and variant_clean_hashes != variant_ood_hashes:
            raise ValueError(
                "Clean and OOD checkpoint hashes differ within policy variant; refusing to compute a "
                f"robustness comparison: variant={variant}, clean={sorted(variant_clean_hashes)}, "
                f"ood={sorted(variant_ood_hashes)}"
            )
        variant_clean_summary = summarize_rows(variant_clean)
        variant_ood_summary = summarize_rows(variant_ood)
        robustness_by_policy.append(
            {
                "policy_variant": variant,
                "test_time_future_imagination": any(
                    bool(row.get("test_time_future_imagination", False)) for row in variant_rows
                ),
                "clean": variant_clean_summary,
                "ood": variant_ood_summary,
                "absolute_success_drop": absolute_drop(
                    variant_clean_summary["success_rate"], variant_ood_summary["success_rate"]
                ),
                "relative_success_drop": relative_drop(
                    variant_clean_summary["success_rate"], variant_ood_summary["success_rate"]
                ),
                "paired": paired_outcomes(variant_rows),
            }
        )
        checkpoint_hashes_by_policy[variant] = {
            "clean": sorted(variant_clean_hashes),
            "ood": sorted(variant_ood_hashes),
        }
    clean_summary = summarize_rows(clean_rows)
    ood_summary = summarize_rows(ood_rows)
    single_policy = len(variants) <= 1
    clean_hashes = {
        checkpoint_hash
        for hashes in checkpoint_hashes_by_policy.values()
        for checkpoint_hash in hashes["clean"]
    }
    ood_hashes = {
        checkpoint_hash
        for hashes in checkpoint_hashes_by_policy.values()
        for checkpoint_hash in hashes["ood"]
    }
    metrics = {
        "all": summarize_rows(rows),
        "clean": clean_summary,
        "ood": ood_summary,
        "absolute_success_drop": (
            absolute_drop(clean_summary["success_rate"], ood_summary["success_rate"])
            if single_policy
            else None
        ),
        "relative_success_drop": (
            relative_drop(clean_summary["success_rate"], ood_summary["success_rate"])
            if single_policy
            else None
        ),
        "paired": paired_outcomes(rows) if single_policy else {},
        "mixed_policy_variants": not single_policy,
        "robustness_by_policy": robustness_by_policy,
        "future_imagination_comparisons": paired_policy_comparisons(rows),
        "checkpoint_hashes": {"clean": sorted(clean_hashes), "ood": sorted(ood_hashes)},
        "checkpoint_hashes_by_policy": checkpoint_hashes_by_policy,
        "by_task": by_task,
        "by_perturbation": by_perturbation,
        "by_level": by_level,
        "by_policy": by_policy,
    }
    (summary_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    (summary_dir / "analysis_sources.json").write_text(
        json.dumps(
            {"experiment_dir": str(experiment_dir), "input_dirs": [str(path) for path in sources]},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest_path = experiment_dir / "experiment_manifest.json"
    try:
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            source_manifests = []
            for source in sources:
                source_manifest = source / "experiment_manifest.json"
                if source_manifest.is_file():
                    source_manifests.append(json.loads(source_manifest.read_text(encoding="utf-8")))
            manifest = {
                "experiment_id": experiment_dir.name,
                "source_manifests": source_manifests,
            }
            if source_manifests:
                manifest["config"] = source_manifests[0].get("config", {})
                manifest["provenance"] = source_manifests[0].get("provenance", {})
                manifest["gpu_environment"] = source_manifests[0].get("gpu_environment", {})
        manifest["status"] = "aggregated"
        manifest["analysis_sources"] = [str(experiment_dir), *[str(path) for path in sources]]
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(manifest_path)
    except (OSError, json.JSONDecodeError):
        pass
    return metrics
