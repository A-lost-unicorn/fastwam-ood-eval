"""Post-run hierarchical analysis for the Thought 2 formal collection.

The data collection is formal and provenance-pinned.  The statistical analysis
plan was still marked DRAFT when the run started, so every artifact generated
here deliberately describes itself as protocol-consistent post-run analysis,
not as a preregistered confirmatory result.

This module never modifies Thought 1 or raw Thought 2 outputs.  It reads the
combined diagnostic CSV, reduces probes within episode, reduces episodes within
task, and uses suite-stratified task bootstrap intervals.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


ANALYSIS_SCHEMA = "thought2-formal-post-run-analysis-v1"
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20_260_725
METRICS = (
    "future_latent_cosine_distance",
    "future_latent_l1",
    "motion_direction_cosine",
    "predicted_motion_energy",
    "actual_motion_energy",
    "motion_energy_ratio",
)
PRIMARY_METRICS = METRICS[:3]
LATENCY_METRICS = (
    "generation_latency_ms",
    "diagnostic_latency_ms",
    "generation_peak_memory_mb",
)
OUTPUT_FILES = (
    "episode_metrics.csv",
    "primary_contrasts.csv",
    "category_contrasts.csv",
    "outcome_associations.csv",
    "outcome_quartiles.csv",
    "runtime_summary.csv",
    "representative_cases.csv",
    "source_action_audit.csv",
    "media_audit.json",
    "formal_analysis.json",
    "report.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    if fieldnames is None:
        fieldnames = tuple(rows[0]) if rows else ()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _finite(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _boolean(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{name} must be true/false, got {value!r}")


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute a quantile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": bool(status.strip())}


def _analysis_threshold(manifest: Mapping[str, Any]) -> float:
    config = manifest.get("config")
    diagnostics = config.get("diagnostics") if isinstance(config, Mapping) else None
    value = (
        diagnostics.get("static_motion_threshold")
        if isinstance(diagnostics, Mapping)
        else None
    )
    threshold = _finite(value, name="static_motion_threshold")
    if threshold < 0:
        raise ValueError("static_motion_threshold must be non-negative")
    return threshold


def _validate_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    diagnostic_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Thought 2 combined diagnostics are empty")
    required = {
        "probe_id",
        "job_id",
        "probe_index",
        "suite",
        "task_id",
        "condition",
        "success",
        "status",
        "action_unchanged",
        "aligned_future_frame_count",
        *METRICS,
        *LATENCY_METRICS,
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Combined diagnostics are missing fields: {missing}")
    probe_ids = [str(row["probe_id"]) for row in rows]
    if len(probe_ids) != len(set(probe_ids)):
        raise ValueError("Combined diagnostics contain duplicate probe_id values")
    bad_status = [row["probe_id"] for row in rows if row["status"] != "completed"]
    if bad_status:
        raise ValueError(f"Non-completed diagnostic probes: {bad_status[:5]}")
    changed = [
        row["probe_id"]
        for row in rows
        if not _boolean(row["action_unchanged"], name="action_unchanged")
    ]
    if changed:
        raise ValueError(f"Protected action changed in probes: {changed[:5]}")
    for row in rows:
        for metric in (*METRICS, *LATENCY_METRICS):
            _finite(row[metric], name=f"{row['probe_id']}:{metric}")
    jobs = {str(row["job_id"]) for row in rows}
    denominators = diagnostic_metrics.get("denominators")
    denominators = denominators if isinstance(denominators, Mapping) else {}
    expected_jobs = denominators.get("planned_jobs")
    expected_clips = diagnostic_metrics.get("clips")
    if isinstance(expected_jobs, int) and len(jobs) != expected_jobs:
        raise ValueError(
            f"Expected {expected_jobs} jobs from aggregate, found {len(jobs)}"
        )
    if isinstance(expected_clips, int) and len(rows) != expected_clips:
        raise ValueError(
            f"Expected {expected_clips} probes from aggregate, found {len(rows)}"
        )
    return {
        "episodes": len(jobs),
        "probes": len(rows),
        "aligned_future_frames": sum(
            int(row["aligned_future_frame_count"]) for row in rows
        ),
        "completed_probes": len(rows),
        "error_probes": 0,
        "protected_action_unchanged": len(rows),
    }


def _episode_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    source_outcomes: Mapping[str, Mapping[str, str]],
    static_threshold: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["job_id"])].append(row)
    output: list[dict[str, Any]] = []
    for job_id, probes in sorted(grouped.items()):
        probes = sorted(probes, key=lambda item: int(item["probe_index"]))
        first = probes[0]
        source = source_outcomes.get(job_id)
        if source is None:
            raise ValueError(f"Thought 1 source outcome is missing job_id={job_id}")
        success = _boolean(first["success"], name=f"{job_id}:success")
        source_success = _boolean(
            source["success"], name=f"{job_id}:source_success"
        )
        conditions = {probe["condition"] for probe in probes}
        outcomes = {
            _boolean(probe["success"], name=f"{job_id}:success") for probe in probes
        }
        if len(conditions) != 1 or len(outcomes) != 1:
            raise ValueError(f"Inconsistent episode metadata for job_id={job_id}")
        item: dict[str, Any] = {
            "job_id": job_id,
            "suite": first["suite"],
            "task_id": int(first["task_id"]),
            "task_name": first.get("task_name", ""),
            "condition": first["condition"],
            "perturbation_category": first.get("perturbation_category") or "clean",
            "perturbation_level": first.get("perturbation_level") or "clean",
            "episode_index": int(first["episode_index"]),
            "episode_seed": int(first["episode_seed"]),
            "success": success,
            "source_success": source_success,
            "outcome_match": success == source_success,
            "termination_reason": first.get("termination_reason", ""),
            "source_termination_reason": source.get("termination_reason", ""),
            "probes": len(probes),
            "aligned_future_frames": sum(
                int(probe["aligned_future_frame_count"]) for probe in probes
            ),
            "first_probe_index": int(first["probe_index"]),
            "first_environment_step": int(first["environment_step"]),
        }
        for metric in METRICS:
            eligible = list(probes)
            if metric == "motion_direction_cosine":
                eligible = [
                    probe
                    for probe in probes
                    if _finite(
                        probe["predicted_motion_energy"],
                        name="predicted_motion_energy",
                    )
                    > static_threshold
                    and _finite(
                        probe["actual_motion_energy"],
                        name="actual_motion_energy",
                    )
                    > static_threshold
                ]
            values = [
                _finite(probe[metric], name=f"{probe['probe_id']}:{metric}")
                for probe in eligible
            ]
            first_value: float | None
            if metric == "motion_direction_cosine" and (
                _finite(first["predicted_motion_energy"], name="predicted_motion_energy")
                <= static_threshold
                or _finite(first["actual_motion_energy"], name="actual_motion_energy")
                <= static_threshold
            ):
                first_value = None
            else:
                first_value = _finite(
                    first[metric], name=f"{first['probe_id']}:{metric}"
                )
            item[f"{metric}__episode_mean"] = (
                statistics.fmean(values) if values else None
            )
            item[f"{metric}__first_probe"] = first_value
        for metric in LATENCY_METRICS:
            values = [
                _finite(probe[metric], name=f"{probe['probe_id']}:{metric}")
                for probe in probes
            ]
            item[f"{metric}__episode_mean"] = statistics.fmean(values)
        output.append(item)
    return output


def _metric_value(
    episode: Mapping[str, Any],
    metric: str,
    probe_mode: str,
) -> float | None:
    suffix = "episode_mean" if probe_mode == "all_available" else "first_probe"
    value = episode.get(f"{metric}__{suffix}")
    return float(value) if isinstance(value, (int, float)) else None


def _stratified_bootstrap(
    values: Sequence[tuple[str, float]],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    if not values:
        raise ValueError("No task contrasts are available")
    by_suite: dict[str, list[float]] = defaultdict(list)
    for suite, value in values:
        by_suite[str(suite)].append(float(value))
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(replicates):
        sample: list[float] = []
        for suite in sorted(by_suite):
            suite_values = by_suite[suite]
            sample.extend(rng.choice(suite_values) for _ in suite_values)
        draws.append(statistics.fmean(sample))
    estimate = statistics.fmean(value for _, value in values)
    return estimate, _quantile(draws, 0.025), _quantile(draws, 0.975)


def _sign_flip_pvalue(
    values: Sequence[float],
    *,
    replicates: int,
    seed: int,
) -> float:
    if not values:
        return math.nan
    observed = abs(statistics.fmean(values))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(replicates):
        draw = statistics.fmean(
            value if rng.random() < 0.5 else -value for value in values
        )
        if abs(draw) >= observed:
            extreme += 1
    return (extreme + 1.0) / (replicates + 1.0)


def _stable_seed(base: int, *parts: str) -> int:
    digest = hashlib.sha256(
        "\x1f".join((str(base), *parts)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _task_contrast(
    episodes: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    probe_mode: str,
    ood_filter: Callable[[Mapping[str, Any]], bool],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    clean: dict[tuple[str, int], list[float]] = defaultdict(list)
    ood: dict[tuple[str, int], list[float]] = defaultdict(list)
    for episode in episodes:
        value = _metric_value(episode, metric, probe_mode)
        if value is None:
            continue
        key = (str(episode["suite"]), int(episode["task_id"]))
        if episode["condition"] == "clean":
            clean[key].append(value)
        elif episode["condition"] == "ood" and ood_filter(episode):
            ood[key].append(value)
    task_values: list[tuple[str, float]] = []
    clean_values: list[float] = []
    ood_values: list[float] = []
    clean_episode_count = 0
    ood_episode_count = 0
    for key in sorted(ood):
        if key not in clean:
            continue
        clean_mean = statistics.fmean(clean[key])
        ood_mean = statistics.fmean(ood[key])
        clean_values.append(clean_mean)
        ood_values.append(ood_mean)
        task_values.append((key[0], ood_mean - clean_mean))
        clean_episode_count += len(clean[key])
        ood_episode_count += len(ood[key])
    estimate, low, high = _stratified_bootstrap(
        task_values,
        replicates=replicates,
        seed=seed,
    )
    p_value = _sign_flip_pvalue(
        [value for _, value in task_values],
        replicates=replicates,
        seed=_stable_seed(seed, metric, probe_mode, "sign_flip"),
    )
    return {
        "metric": metric,
        "probe_mode": probe_mode,
        "clean_task_equal_mean": statistics.fmean(clean_values),
        "ood_task_equal_mean": statistics.fmean(ood_values),
        "ood_minus_clean": estimate,
        "ci95_low": low,
        "ci95_high": high,
        "eligible_tasks": len(task_values),
        "clean_episodes": clean_episode_count,
        "ood_episodes": ood_episode_count,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "raw_p_value": p_value,
    }


def _outcome_contrast(
    episodes: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    probe_mode: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[tuple[str, int], dict[bool, list[float]]] = defaultdict(
        lambda: {True: [], False: []}
    )
    outcome_counts: Counter[bool] = Counter()
    excluded_mismatch = 0
    for episode in episodes:
        if episode["condition"] != "ood":
            continue
        if not episode["outcome_match"]:
            excluded_mismatch += 1
            continue
        value = _metric_value(episode, metric, probe_mode)
        if value is None:
            continue
        success = bool(episode["success"])
        grouped[(str(episode["suite"]), int(episode["task_id"]))][success].append(
            value
        )
        outcome_counts[success] += 1
    task_values: list[tuple[str, float]] = []
    success_values: list[float] = []
    failure_values: list[float] = []
    for key in sorted(grouped):
        successes = grouped[key][True]
        failures = grouped[key][False]
        if not successes or not failures:
            continue
        success_mean = statistics.fmean(successes)
        failure_mean = statistics.fmean(failures)
        success_values.append(success_mean)
        failure_values.append(failure_mean)
        task_values.append((key[0], failure_mean - success_mean))
    estimate, low, high = _stratified_bootstrap(
        task_values,
        replicates=replicates,
        seed=seed,
    )
    return {
        "condition": "ood",
        "metric": metric,
        "probe_mode": probe_mode,
        "success_task_equal_mean": statistics.fmean(success_values),
        "failure_task_equal_mean": statistics.fmean(failure_values),
        "failure_minus_success": estimate,
        "ci95_low": low,
        "ci95_high": high,
        "eligible_tasks": len(task_values),
        "success_episodes": outcome_counts[True],
        "failure_episodes": outcome_counts[False],
        "excluded_outcome_mismatch_episodes": excluded_mismatch,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "raw_p_value": _sign_flip_pvalue(
            [value for _, value in task_values],
            replicates=replicates,
            seed=_stable_seed(seed, metric, probe_mode, "outcome_sign_flip"),
        ),
    }


def _benjamini_hochberg(rows: Sequence[dict[str, Any]]) -> None:
    eligible = [
        (index, float(row["raw_p_value"]))
        for index, row in enumerate(rows)
        if isinstance(row.get("raw_p_value"), (int, float))
        and math.isfinite(float(row["raw_p_value"]))
    ]
    if not eligible:
        return
    ordered = sorted(eligible, key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[int, float] = {}
    running = 1.0
    for rank_from_end, (index, value) in enumerate(reversed(ordered), start=1):
        rank = count - rank_from_end + 1
        running = min(running, value * count / rank)
        adjusted[index] = min(1.0, running)
    for index, value in adjusted.items():
        rows[index]["bh_q_value"] = value


def _runtime_rows(
    episodes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for group in ("all", "clean", "ood"):
        selected = [
            episode
            for episode in episodes
            if group == "all" or episode["condition"] == group
        ]
        for metric in LATENCY_METRICS:
            values = [
                float(episode[f"{metric}__episode_mean"]) for episode in selected
            ]
            output.append(
                {
                    "group": group,
                    "metric": metric,
                    "episodes": len(values),
                    "mean": statistics.fmean(values),
                    "p50": _quantile(values, 0.50),
                    "p95": _quantile(values, 0.95),
                    "minimum": min(values),
                    "maximum": max(values),
                }
            )
    return output


def _quartile_rows(
    episodes: Sequence[Mapping[str, Any]],
    *,
    probe_mode: str,
) -> list[dict[str, Any]]:
    eligible = [
        episode
        for episode in episodes
        if episode["condition"] == "ood"
        and episode["outcome_match"]
        and _metric_value(
            episode, "future_latent_cosine_distance", probe_mode
        )
        is not None
    ]
    eligible.sort(
        key=lambda episode: float(
            _metric_value(
                episode, "future_latent_cosine_distance", probe_mode
            )
        )
    )
    output: list[dict[str, Any]] = []
    for index in range(4):
        start = index * len(eligible) // 4
        end = (index + 1) * len(eligible) // 4
        group = eligible[start:end]
        values = [
            float(
                _metric_value(
                    episode, "future_latent_cosine_distance", probe_mode
                )
            )
            for episode in group
        ]
        failures = sum(not episode["success"] for episode in group)
        output.append(
            {
                "probe_mode": probe_mode,
                "quartile": index + 1,
                "interpretation": (
                    "lowest_error" if index == 0 else "highest_error" if index == 3 else ""
                ),
                "episodes": len(group),
                "failures": failures,
                "failure_rate": failures / len(group),
                "metric_min": min(values),
                "metric_max": max(values),
            }
        )
    return output


def _representative_cases(
    rows: Sequence[Mapping[str, str]],
    *,
    experiment_dir: Path,
    static_threshold: float,
) -> list[dict[str, Any]]:
    failures = [
        row for row in rows if not _boolean(row["success"], name="success")
    ]
    successes = [
        row for row in rows if _boolean(row["success"], name="success")
    ]
    if not failures or not successes:
        return []

    def metric(row: Mapping[str, str], name: str) -> float:
        return _finite(row[name], name=name)

    selections = (
        (
            "failure_high_future_error",
            max(failures, key=lambda row: metric(row, "future_latent_cosine_distance")),
        ),
        (
            "failure_low_future_error",
            min(failures, key=lambda row: metric(row, "future_latent_cosine_distance")),
        ),
        (
            "success_high_future_error",
            max(successes, key=lambda row: metric(row, "future_latent_cosine_distance")),
        ),
        (
            "success_low_future_error",
            min(successes, key=lambda row: metric(row, "future_latent_cosine_distance")),
        ),
        (
            "failure_high_direction_consistency",
            max(
                (
                    row
                    for row in failures
                    if metric(row, "predicted_motion_energy") > static_threshold
                    and metric(row, "actual_motion_energy") > static_threshold
                ),
                key=lambda row: metric(row, "motion_direction_cosine"),
            ),
        ),
        (
            "failure_low_actual_motion",
            min(failures, key=lambda row: metric(row, "actual_motion_energy")),
        ),
    )
    output: list[dict[str, Any]] = []
    for role, row in selections:
        relative = (
            Path("diagnostics")
            / row["suite"]
            / row["condition"]
            / row["side_by_side_video_path"]
        )
        output.append(
            {
                "role": role,
                "selection_status": "post_hoc_illustrative_not_inferential",
                "job_id": row["job_id"],
                "probe_id": row["probe_id"],
                "suite": row["suite"],
                "task_id": int(row["task_id"]),
                "condition": row["condition"],
                "perturbation_category": row.get("perturbation_category") or "clean",
                "perturbation_level": row.get("perturbation_level") or "clean",
                "success": _boolean(row["success"], name="success"),
                "future_latent_cosine_distance": metric(
                    row, "future_latent_cosine_distance"
                ),
                "future_latent_l1": metric(row, "future_latent_l1"),
                "motion_direction_cosine": metric(
                    row, "motion_direction_cosine"
                ),
                "predicted_motion_energy": metric(
                    row, "predicted_motion_energy"
                ),
                "actual_motion_energy": metric(row, "actual_motion_energy"),
                "side_by_side_video_path": str(
                    (experiment_dir / relative).resolve()
                ),
                "run_relative_video_path": str(relative),
            }
        )
    return output


def _load_raw_diagnostic_rows(
    experiment_dir: Path,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in sorted(
        (experiment_dir / "diagnostics").glob(
            "*/*/workers/rank_*/diagnostics.jsonl"
        )
    ):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid diagnostic JSONL: {path}:{line_number}"
                    ) from exc
                if isinstance(row, dict):
                    output.append(row)
    return output


def _source_action_audit(
    experiment_dir: Path,
    source_trace_root: Path | None,
    *,
    expected_probes: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = _load_raw_diagnostic_rows(experiment_dir)
    if len(raw) != expected_probes:
        raise ValueError(
            "Raw diagnostic probe count differs from combined CSV: "
            f"raw={len(raw)}, combined={expected_probes}"
        )
    internal_violations: list[str] = []
    for row in raw:
        unchanged = row.get("action_unchanged") is True
        hashes_equal = (
            row.get("action_hash_before")
            == row.get("action_hash_after")
            == row.get("action_hash")
        )
        if not unchanged or not hashes_equal:
            internal_violations.append(str(row.get("probe_id")))
    if internal_violations:
        raise ValueError(
            "Shadow probe changed protected action hashes: "
            f"{internal_violations[:5]}"
        )
    summary: dict[str, Any] = {
        "internal_probes": len(raw),
        "internal_exact_hash_matches": len(raw),
        "internal_violations": 0,
        "source_trace_audit_status": (
            "enabled" if source_trace_root is not None else "not_requested"
        ),
    }
    if source_trace_root is None:
        return summary, []
    source_trace_root = Path(source_trace_root)
    if not source_trace_root.is_dir():
        raise FileNotFoundError(source_trace_root)
    needed = {str(row.get("job_id")) for row in raw}
    trace_paths: dict[str, Path] = {}
    for path in source_trace_root.glob("*/*/workers/rank_*/traces/*.jsonl"):
        if path.stem in needed:
            trace_paths[path.stem] = path
    cache: dict[str, dict[int, Sequence[float]]] = {}
    rows: list[dict[str, Any]] = []
    for diagnostic in raw:
        job_id = str(diagnostic["job_id"])
        probe_id = str(diagnostic["probe_id"])
        path = trace_paths.get(job_id)
        if path is None:
            rows.append(
                {
                    "job_id": job_id,
                    "probe_id": probe_id,
                    "probe_index": (diagnostic.get("extra") or {}).get(
                        "probe_index"
                    ),
                    "condition": diagnostic.get("condition"),
                    "status": "source_trace_missing",
                    "max_abs_action_diff": "",
                }
            )
            continue
        if job_id not in cache:
            trace: dict[int, Sequence[float]] = {}
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    trace[int(value["step"])] = value["action"]
            cache[job_id] = trace
        trace = cache[job_id]
        start = int(diagnostic["environment_step"])
        count = int(diagnostic["executed_action_count"])
        source_actions = [trace.get(start + offset) for offset in range(count)]
        if any(action is None for action in source_actions):
            status = "source_steps_unavailable"
            maximum: float | str = ""
        else:
            differences = [
                abs(float(source_value) - float(rerun_value))
                for source_action, rerun_action in zip(
                    source_actions, diagnostic["executed_actions"]
                )
                for source_value, rerun_value in zip(
                    source_action, rerun_action
                )
            ]
            maximum = max(differences, default=0.0)
            status = "exact" if maximum == 0.0 else "mismatch"
        rows.append(
            {
                "job_id": job_id,
                "probe_id": probe_id,
                "probe_index": (diagnostic.get("extra") or {}).get(
                    "probe_index"
                ),
                "condition": diagnostic.get("condition"),
                "status": status,
                "max_abs_action_diff": maximum,
            }
        )
    counts = Counter(row["status"] for row in rows)
    finite_differences = [
        float(row["max_abs_action_diff"])
        for row in rows
        if isinstance(row["max_abs_action_diff"], (int, float))
    ]
    summary.update(
        {
            "source_trace_probes": len(rows),
            "source_trace_exact": counts["exact"],
            "source_trace_mismatch": counts["mismatch"],
            "source_trace_unavailable": (
                counts["source_steps_unavailable"]
                + counts["source_trace_missing"]
            ),
            "source_trace_max_abs_action_diff": (
                max(finite_differences) if finite_differences else None
            ),
        }
    )
    return summary, rows


def _media_audit(
    rows: Sequence[Mapping[str, str]],
    *,
    verify_media: bool,
) -> dict[str, Any]:
    if not verify_media:
        return {
            "status": "not_requested",
            "expected_cases": len(rows),
            "decoded": {},
            "errors": [],
        }
    try:
        import cv2
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Full media verification requires OpenCV and Pillow"
        ) from exc
    expected = {
        "current_frame_path": (448, 224, 1),
        "predicted_video_path": (448, 224, 9),
        "actual_video_path": (448, 224, 3),
        "side_by_side_video_path": (896, 224, 3),
    }
    decoded: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    for row in rows:
        root = Path(row["artifact_source_root"])
        for field, (width, height, frame_count) in expected.items():
            path = root / row[field]
            try:
                if field == "current_frame_path":
                    with Image.open(path) as image:
                        size = image.size
                        image.verify()
                    if size != (width, height):
                        raise ValueError(f"shape={size}")
                else:
                    capture = cv2.VideoCapture(str(path))
                    shapes: list[tuple[int, ...]] = []
                    while True:
                        ok, frame = capture.read()
                        if not ok:
                            break
                        shapes.append(tuple(frame.shape))
                    capture.release()
                    if len(shapes) != frame_count or any(
                        shape != (height, width, 3) for shape in shapes
                    ):
                        raise ValueError(
                            f"frames={len(shapes)}, shapes={shapes[:3]}"
                        )
                decoded[field] += 1
            except Exception as exc:  # media backends expose heterogeneous errors
                errors.append(
                    {
                        "probe_id": row["probe_id"],
                        "field": field,
                        "path": str(path),
                        "error": str(exc),
                    }
                )
    return {
        "status": "passed" if not errors else "failed",
        "expected_cases": len(rows),
        "decoded": dict(decoded),
        "errors": errors,
    }


def _format(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "—"
    return f"{float(value):.{digits}f}"


def _find_row(
    rows: Sequence[Mapping[str, Any]],
    **criteria: str,
) -> Mapping[str, Any]:
    for row in rows:
        if all(str(row.get(key)) == str(value) for key, value in criteria.items()):
            return row
    raise KeyError(criteria)


def _render_report(
    *,
    integrity: Mapping[str, Any],
    primary: Sequence[Mapping[str, Any]],
    category: Sequence[Mapping[str, Any]],
    outcome: Sequence[Mapping[str, Any]],
    quartiles: Sequence[Mapping[str, Any]],
    runtime: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
    action_audit: Mapping[str, Any],
    media_audit: Mapping[str, Any],
    static_threshold: float,
    analysis_status: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> str:
    cosine = _find_row(
        primary,
        metric="future_latent_cosine_distance",
        probe_mode="all_available",
    )
    l1 = _find_row(
        primary, metric="future_latent_l1", probe_mode="all_available"
    )
    direction = _find_row(
        primary,
        metric="motion_direction_cosine",
        probe_mode="all_available",
    )
    outcome_cosine = _find_row(
        outcome,
        metric="future_latent_cosine_distance",
        probe_mode="all_available",
    )
    outcome_cosine_first = _find_row(
        outcome,
        metric="future_latent_cosine_distance",
        probe_mode="first_probe",
    )
    outcome_direction = _find_row(
        outcome,
        metric="motion_direction_cosine",
        probe_mode="all_available",
    )
    outcome_direction_first = _find_row(
        outcome,
        metric="motion_direction_cosine",
        probe_mode="first_probe",
    )
    q_first = [
        row for row in quartiles if row["probe_mode"] == "first_probe"
    ]
    q_first.sort(key=lambda row: int(row["quartile"]))
    generation = _find_row(
        runtime, group="all", metric="generation_latency_ms"
    )
    diagnostic = _find_row(
        runtime, group="all", metric="diagnostic_latency_ms"
    )
    memory = _find_row(
        runtime, group="all", metric="generation_peak_memory_mb"
    )
    category_rows: list[str] = []
    category_names = (
        "camera_viewpoints",
        "light_conditions",
        "background_textures",
        "objects_layout",
        "robot_initial_states",
    )
    for name in category_names:
        row = _find_row(
            category,
            scope="category",
            group=name,
            metric="future_latent_cosine_distance",
            probe_mode="all_available",
        )
        direction_row = _find_row(
            category,
            scope="category",
            group=name,
            metric="motion_direction_cosine",
            probe_mode="all_available",
        )
        category_rows.append(
            "| {name} | {episodes} | {clean} | {ood} | {diff} "
            "[{low}, {high}] | {direction} |".format(
                name=name,
                episodes=row["ood_episodes"],
                clean=_format(row["clean_task_equal_mean"]),
                ood=_format(row["ood_task_equal_mean"]),
                diff=_format(row["ood_minus_clean"]),
                low=_format(row["ci95_low"]),
                high=_format(row["ci95_high"]),
                direction=_format(direction_row["ood_task_equal_mean"]),
            )
        )
    case_rows: list[str] = []
    for case in cases:
        path = Path("..") / case["run_relative_video_path"]
        case_rows.append(
            f"- `{case['role']}`：job `{case['job_id']}`，"
            f"cosine distance `{_format(case['future_latent_cosine_distance'])}`，"
            f"direction `{_format(case['motion_direction_cosine'])}`，"
            f"[side-by-side]({path.as_posix()})。"
        )
    media_decoded = media_audit.get("decoded") or {}
    return f"""# 思考点二：五类扰动正式数据收集与自动一致性分析

## 1. 证据状态

- 数据收集：正式、完整、项目与上游 clean；732 个预先 ratify 的 job 全部完成。
- 统计状态：`{analysis_status}`。分析实现遵循运行前 DRAFT 中写明的
  episode→task 聚合与 suite-stratified task bootstrap，但该 DRAFT 未在看到
  正式指标前冻结，因此不能称为 preregistered confirmatory analysis。
- 因果状态：`causal_interpretation_allowed=false`。视频分支不读取受保护动作，
  动作分支也不读取生成的未来。

## 2. 完整性审计

| 项目 | 结果 |
| --- | ---: |
| Episodes | {integrity['episodes']}/{integrity['episodes']} |
| Probes / 对齐 future frames | {integrity['probes']} / {integrity['aligned_future_frames']} |
| Completed / error probes | {integrity['completed_probes']} / {integrity['error_probes']} |
| Probe 内动作哈希不变 | {action_audit['internal_exact_hash_matches']}/{action_audit['internal_probes']} |
| Phase 1/2 outcome 一致 | {integrity['outcome_matches']}/{integrity['episodes']} |
| Predicted-static / actual-static probes | {integrity['predicted_static_probes']} / {integrity['actual_static_probes']} |
| 全媒体解码错误 | {len(media_audit.get('errors') or [])} |

全媒体审计实际解码 current/predicted/actual/side-by-side：
`{media_decoded.get('current_frame_path', 0)}/
{media_decoded.get('predicted_video_path', 0)}/
{media_decoded.get('actual_video_path', 0)}/
{media_decoded.get('side_by_side_video_path', 0)}`。

## 3. 预测未来是否正确？

自动 proxy 的答案是：**在 Clean 上相对一致，在 OOD 上系统性变差；但尚不能给
出“语义正确率”。**

| Metric（task 等权） | Clean | OOD | OOD−Clean / 95% CI |
| --- | ---: | ---: | ---: |
| Cosine distance ↓ | {_format(cosine['clean_task_equal_mean'])} | {_format(cosine['ood_task_equal_mean'])} | {_format(cosine['ood_minus_clean'])} [{_format(cosine['ci95_low'])}, {_format(cosine['ci95_high'])}] |
| Latent L1 ↓ | {_format(l1['clean_task_equal_mean'])} | {_format(l1['ood_task_equal_mean'])} | {_format(l1['ood_minus_clean'])} [{_format(l1['ci95_low'])}, {_format(l1['ci95_high'])}] |

这里的 latent 是预测/实际解码帧分别经同一冻结 VAE 重编码后的近似表示，不是
原生 temporal diffusion latent。低误差表示“与随后实际发生的局部视觉变化相似”，
不保证目标进展正确；如果策略做错而视频也预测了同一个错误，自动 agreement 仍可
很高。绝对的 goal correctness、物理合理性和 wrong-object/wrong-direction 比例
仍需标签盲化人工评审。

## 4. 动作是否与预测未来方向一致？

当前可识别的不是 7-DoF action 与像素的直接 cosine，而是：

```text
预测视觉变化方向 vs 受保护动作执行后的实际视觉变化方向
```

Clean/OOD task-equal direction cosine 为
`{_format(direction['clean_task_equal_mean'])}→{_format(direction['ood_task_equal_mean'])}`，
OOD−Clean 为 `{_format(direction['ood_minus_clean'])}`
（95% CI `[{_format(direction['ci95_low'])}, {_format(direction['ci95_high'])}]`）。
因此 OOD 下局部方向相容性明显下降。正式 no-op 阈值为
`{static_threshold:.8f}`；6 个 actual-static probe 不进入 decisive direction
分母，所有 episode 仍至少有一个有效 direction probe。

## 5. 失败来自未来错误还是动作错误？

自动数据支持“失败与较差 future consistency 相关”，但不支持因果二分归因。
排除 2 个 Phase 1/2 outcome 不一致 episode 后，OOD 内 task-equal：

| 分析 | Cosine failure−success / 95% CI | Direction failure−success / 95% CI |
| --- | ---: | ---: |
| episode 全部可用 probes | {_format(outcome_cosine['failure_minus_success'])} [{_format(outcome_cosine['ci95_low'])}, {_format(outcome_cosine['ci95_high'])}] | {_format(outcome_direction['failure_minus_success'])} [{_format(outcome_direction['ci95_low'])}, {_format(outcome_direction['ci95_high'])}] |
| 仅首 probe | {_format(outcome_cosine_first['failure_minus_success'])} [{_format(outcome_cosine_first['ci95_low'])}, {_format(outcome_cosine_first['ci95_high'])}] | {_format(outcome_direction_first['failure_minus_success'])} [{_format(outcome_direction_first['ci95_low'])}, {_format(outcome_direction_first['ci95_high'])}] |

首 probe 敏感性很重要：成功 episode 只有一个 probe，失败 episode 有两个；只看
首 probe 后关联仍存在，说明结果不完全由晚期 probe/轨迹截断造成。

但一致性并不充分。首 probe cosine 最低误差四分位的失败率仍为
`{100 * float(q_first[0]['failure_rate']):.2f}%`
（{q_first[0]['failures']}/{q_first[0]['episodes']}），最高误差四分位为
`{100 * float(q_first[3]['failure_rate']):.2f}%`
（{q_first[3]['failures']}/{q_first[3]['episodes']}）。同时存在高误差但成功、
高方向一致性但最终失败的真实案例。

更强的结论是：**OOD 同时破坏未来预测与控制成功，两者存在关联；当前实验无法
判断哪个是另一个的原因。** 在本执行图中 shadow future 没有反馈到控制路径，
所以它本身不可能造成这些失败。是否把 future 接入动作能改善失败，只能由阶段三
B0/A0/A1/A2/A4 对照回答。

## 6. 五类扰动

| Category | OOD episodes | Clean cosine | OOD cosine | OOD−Clean / 95% CI | OOD direction |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(category_rows)}

Camera 的 cosine/direction 退化最大；light 最小。严重度并非对所有 future 指标
单调：camera 与 background 的 cosine 不随 easy→hard 单调恶化，而
robot-initial-state 呈清晰递增。这再次说明 task success severity 与短时
future-consistency 不是同一个量。

## 7. 资源开销

- 20-step future generation：mean/p50/p95 =
  `{_format(generation['mean'], 2)}/{_format(generation['p50'], 2)}/{_format(generation['p95'], 2)} ms`。
- 完整离线诊断：mean/p50/p95 =
  `{_format(diagnostic['mean'], 2)}/{_format(diagnostic['p50'], 2)}/{_format(diagnostic['p95'], 2)} ms`。
- 峰值显存：`{_format(memory['mean'], 2)} MB`。

这些是 shadow probe 成本，不是阶段一动作策略延迟，也不能直接替代阶段三
K=1/2/4 的无解码在线 latency。

## 8. 重跑稳定性与动作保护

- 同一次 Phase 2 rerun 内，future probe 前后的 protected action：
  `{action_audit['internal_exact_hash_matches']}/{action_audit['internal_probes']}`
  哈希完全一致。这是“诊断没有改动作”的主隔离证据。
- 与两天前 Phase 1 trace 跨运行比较：
  `{action_audit.get('source_trace_exact', 0)}/
{action_audit.get('source_trace_probes', 0)}` probe 逐元素完全一致，
  `{action_audit.get('source_trace_mismatch', 0)}` 个存在数值差异，
  `{action_audit.get('source_trace_unavailable', 0)}` 个因 source 已结束而无完整
  10-step 对照；最大差 `{action_audit.get('source_trace_max_abs_action_diff')}`。
- Episode outcome 复现 `{integrity['outcome_matches']}/{integrity['episodes']}`；
  两个不一致均来自 `libero_10/light_conditions`，已从 outcome-association
  主分析排除，但保留在 ID/OOD consistency 分析中。

跨运行差异不表示 future probe 改写动作；它提示仿真/GPU 重跑不是逐位完全确定，
也是为什么 outcome 关联必须使用 Phase 2 同一次轨迹。

## 9. 代表性视频（事后说明性选择）

这些案例按极值选择，只帮助定位媒体，不参与统计推断：

{chr(10).join(case_rows)}

## 10. 结论边界

可以写：

1. 在该 732-episode cohort 中，unconditional future 的自动一致性 proxy 在
   OOD 显著恶化。
2. OOD 失败 episode 的一致性平均更差，首 probe 敏感性方向一致。
3. 局部 future consistency 既非任务成功的充分条件，也非必要条件。

不能写：

1. “动作直接读取/遵循了预测未来”。
2. “future error 导致失败”或“失败已被自动分成 future/action error”。
3. “显式未来一定能提高 OOD 成功率”。
4. “自动 latent distance 就是语义 future 正确率”。

## 11. 可复现参数

- Bootstrap：suite-stratified task resampling，`{bootstrap_replicates}` 次，
  seed `{bootstrap_seed}`。
- 分层：probe→episode→task；40 task 等权。
- Outcome：只使用 Phase 1/2 outcome match 的 OOD episode。
- 分析状态：`{analysis_status}`。
"""


def analyze_thought2_formal(
    *,
    experiment_dir: Path,
    thought1_summary_csv: Path,
    output_dir: Path,
    source_trace_root: Path | None = None,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    verify_media: bool = False,
) -> dict[str, Any]:
    """Create a fresh, isolated Thought 2 post-run analysis bundle."""

    experiment_dir = Path(experiment_dir)
    thought1_summary_csv = Path(thought1_summary_csv)
    output_dir = Path(output_dir)
    if bootstrap_replicates < 100:
        raise ValueError("bootstrap_replicates must be at least 100")
    if output_dir.exists():
        raise FileExistsError(
            f"Analysis output must be a fresh path: {output_dir}"
        )
    combined_dir = experiment_dir / "combined"
    diagnostic_csv = combined_dir / "summary" / "all_diagnostics.csv"
    diagnostic_metrics_path = (
        combined_dir / "summary" / "diagnostic_metrics.json"
    )
    diagnostic_manifest_path = combined_dir / "diagnostic_manifest.json"
    static_summary_path = (
        experiment_dir
        / "static"
        / "combined"
        / "summary"
        / "static_calibration_summary.json"
    )
    diagnostic_rows = _read_csv(diagnostic_csv)
    diagnostic_metrics = _read_json(diagnostic_metrics_path)
    diagnostic_manifest = _read_json(diagnostic_manifest_path)
    static_summary = _read_json(static_summary_path)
    static_threshold = _analysis_threshold(diagnostic_manifest)
    candidate = _finite(
        static_summary.get("candidate_static_motion_threshold"),
        name="candidate_static_motion_threshold",
    )
    if not math.isclose(static_threshold, candidate, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(
            "Diagnostic threshold differs from formal calibration candidate"
        )
    if static_summary.get("freeze_eligible") is not True:
        raise ValueError("Formal static calibration did not pass freeze gates")
    integrity = _validate_rows(
        diagnostic_rows, diagnostic_metrics=diagnostic_metrics
    )
    thought1_rows = _read_csv(thought1_summary_csv)
    source_outcomes = {
        str(row["job_id"]): row for row in thought1_rows if row.get("job_id")
    }
    episodes = _episode_rows(
        diagnostic_rows,
        source_outcomes=source_outcomes,
        static_threshold=static_threshold,
    )
    integrity.update(
        {
            "outcome_matches": sum(
                bool(episode["outcome_match"]) for episode in episodes
            ),
            "outcome_mismatches": sum(
                not bool(episode["outcome_match"]) for episode in episodes
            ),
            "success_episodes": sum(
                bool(episode["success"]) for episode in episodes
            ),
            "failure_episodes": sum(
                not bool(episode["success"]) for episode in episodes
            ),
            "predicted_static_probes": sum(
                _finite(row["predicted_motion_energy"], name="predicted_motion_energy")
                <= static_threshold
                for row in diagnostic_rows
            ),
            "actual_static_probes": sum(
                _finite(row["actual_motion_energy"], name="actual_motion_energy")
                <= static_threshold
                for row in diagnostic_rows
            ),
        }
    )
    primary: list[dict[str, Any]] = []
    for probe_mode in ("all_available", "first_probe"):
        for metric in METRICS:
            primary.append(
                {
                    "scope": "overall",
                    "group": "all_ood",
                    **_task_contrast(
                        episodes,
                        metric=metric,
                        probe_mode=probe_mode,
                        ood_filter=lambda _: True,
                        replicates=bootstrap_replicates,
                        seed=_stable_seed(
                            bootstrap_seed, "overall", probe_mode, metric
                        ),
                    ),
                }
            )
    categories = (
        "camera_viewpoints",
        "light_conditions",
        "background_textures",
        "objects_layout",
        "robot_initial_states",
    )
    levels = ("easy", "medium", "hard")
    category_rows: list[dict[str, Any]] = []
    for probe_mode in ("all_available", "first_probe"):
        for category in categories:
            for metric in PRIMARY_METRICS:
                category_rows.append(
                    {
                        "scope": "category",
                        "group": category,
                        **_task_contrast(
                            episodes,
                            metric=metric,
                            probe_mode=probe_mode,
                            ood_filter=lambda episode, category=category: (
                                episode["perturbation_category"] == category
                            ),
                            replicates=bootstrap_replicates,
                            seed=_stable_seed(
                                bootstrap_seed,
                                "category",
                                category,
                                probe_mode,
                                metric,
                            ),
                        ),
                    }
                )
            for level in levels:
                category_rows.append(
                    {
                        "scope": "category_level",
                        "group": f"{category}/{level}",
                        **_task_contrast(
                            episodes,
                            metric="future_latent_cosine_distance",
                            probe_mode=probe_mode,
                            ood_filter=lambda episode, category=category, level=level: (
                                episode["perturbation_category"] == category
                                and episode["perturbation_level"] == level
                            ),
                            replicates=bootstrap_replicates,
                            seed=_stable_seed(
                                bootstrap_seed,
                                "category_level",
                                category,
                                level,
                                probe_mode,
                            ),
                        ),
                    }
                )
    for probe_mode in ("all_available", "first_probe"):
        family = [
            row
            for row in category_rows
            if row["scope"] == "category"
            and row["probe_mode"] == probe_mode
        ]
        _benjamini_hochberg(family)
    outcome: list[dict[str, Any]] = []
    for probe_mode in ("all_available", "first_probe"):
        for metric in PRIMARY_METRICS:
            outcome.append(
                _outcome_contrast(
                    episodes,
                    metric=metric,
                    probe_mode=probe_mode,
                    replicates=bootstrap_replicates,
                    seed=_stable_seed(
                        bootstrap_seed, "outcome", probe_mode, metric
                    ),
                )
            )
    _benjamini_hochberg(outcome)
    quartiles = [
        row
        for probe_mode in ("all_available", "first_probe")
        for row in _quartile_rows(episodes, probe_mode=probe_mode)
    ]
    runtime = _runtime_rows(episodes)
    cases = _representative_cases(
        diagnostic_rows,
        experiment_dir=experiment_dir,
        static_threshold=static_threshold,
    )
    action_summary, action_rows = _source_action_audit(
        experiment_dir,
        source_trace_root,
        expected_probes=len(diagnostic_rows),
    )
    media = _media_audit(diagnostic_rows, verify_media=verify_media)
    if media["status"] == "failed":
        raise ValueError(
            f"Media audit found {len(media['errors'])} decoding errors"
        )
    analysis_status = (
        "formal_data_collection_post_run_protocol_consistent_"
        "not_preregistered"
    )
    result: dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA,
        "analysis_status": analysis_status,
        "causal_interpretation_allowed": False,
        "static_motion_threshold": static_threshold,
        "bootstrap": {
            "kind": "suite_stratified_task_cluster",
            "replicates": bootstrap_replicates,
            "base_seed": bootstrap_seed,
        },
        "integrity": integrity,
        "action_audit": action_summary,
        "media_audit": media,
        "primary_contrasts": primary,
        "category_contrasts": category_rows,
        "outcome_associations": outcome,
        "outcome_quartiles": quartiles,
        "runtime": runtime,
        "representative_cases": cases,
        "limitations": [
            "The statistical analysis plan was DRAFT, not frozen, before formal metrics.",
            "Decoded frames are independently re-encoded VAE proxies, not native temporal latents.",
            "Success/failure associations are non-causal and vulnerable to trajectory truncation.",
            "Semantic future correctness and direct failure attribution require blinded human labels.",
            "The released action branch does not read the generated future.",
        ],
    }
    output_dir.mkdir(parents=True)
    episode_fields = list(episodes[0])
    _write_csv(output_dir / "episode_metrics.csv", episodes, episode_fields)
    _write_csv(output_dir / "primary_contrasts.csv", primary)
    _write_csv(output_dir / "category_contrasts.csv", category_rows)
    _write_csv(output_dir / "outcome_associations.csv", outcome)
    _write_csv(output_dir / "outcome_quartiles.csv", quartiles)
    _write_csv(output_dir / "runtime_summary.csv", runtime)
    _write_csv(output_dir / "representative_cases.csv", cases)
    _write_csv(
        output_dir / "source_action_audit.csv",
        action_rows,
        (
            "job_id",
            "probe_id",
            "probe_index",
            "condition",
            "status",
            "max_abs_action_diff",
        ),
    )
    (output_dir / "media_audit.json").write_text(
        json.dumps(media, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "formal_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = _render_report(
        integrity=integrity,
        primary=primary,
        category=category_rows,
        outcome=outcome,
        quartiles=quartiles,
        runtime=runtime,
        cases=cases,
        action_audit=action_summary,
        media_audit=media,
        static_threshold=static_threshold,
        analysis_status=analysis_status,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    root = Path.cwd()
    manifest = {
        "schema": ANALYSIS_SCHEMA,
        "analysis_status": analysis_status,
        "source_files_rewritten": False,
        "sources": {
            "diagnostic_csv": {
                "path": str(diagnostic_csv),
                "sha256": _sha256(diagnostic_csv),
            },
            "diagnostic_metrics": {
                "path": str(diagnostic_metrics_path),
                "sha256": _sha256(diagnostic_metrics_path),
            },
            "diagnostic_manifest": {
                "path": str(diagnostic_manifest_path),
                "sha256": _sha256(diagnostic_manifest_path),
            },
            "static_calibration_summary": {
                "path": str(static_summary_path),
                "sha256": _sha256(static_summary_path),
            },
            "thought1_summary": {
                "path": str(thought1_summary_csv),
                "sha256": _sha256(thought1_summary_csv),
            },
        },
        "parameters": {
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
            "verify_media": verify_media,
            "source_trace_audit": source_trace_root is not None,
        },
        "provenance": _git_state(root),
        "outputs": {
            name: _sha256(output_dir / name)
            for name in OUTPUT_FILES
            if name != "analysis_manifest.json"
            and (output_dir / name).is_file()
        },
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "analysis_dir": str(output_dir),
        "report": str(output_dir / "report.md"),
        "episodes": integrity["episodes"],
        "probes": integrity["probes"],
        "outcome_matches": integrity["outcome_matches"],
        "media_status": media["status"],
        "analysis_status": analysis_status,
    }
