"""Episode-weighted aggregation for shadow future diagnostics."""

from __future__ import annotations

import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from fastwam_ood_eval.diagnostics.artifact_writer import _atomic_json_write, _record_order


KNOWN_METRICS = (
    "future_latent_l1",
    "future_latent_cosine_distance",
    "predicted_motion_energy",
    "actual_motion_energy",
    "motion_energy_ratio",
    "motion_direction_cosine",
    "diagnostic_latency_ms",
    "diagnostic_peak_memory_mb",
)

ALL_FIELDS = (
    "experiment_id", "source_experiment_id", "diagnostic_id", "probe_id", "job_id",
    "attempt_id", "attempt_started_ns", "recorded_at_ns", "artifact_source_root",
    "probe_index", "replan_index", "environment_step", "diagnostic_seed", "suite", "task_id",
    "task_name", "episode_index", "episode_seed", "condition", "perturbation_category",
    "perturbation_level", "success", "termination_reason", "status", "num_video_frames",
    "num_inference_steps", "checkpoint", "checkpoint_hash", "fastwam_commit",
    "mode", "action_conditioned_verified", "action_hash", "action_unchanged",
    "executed_action_count",
    "alignment", "approximate_alignment",
    "aligned_future_frame_count", "static_future_flag", *KNOWN_METRICS,
    "current_frame_path", "predicted_video_path", "actual_video_path",
    "side_by_side_video_path", "latent_path",
    "error", "protocol_fingerprint",
)

GROUP_FIELDS = (
    "group", "metric", "episodes", "clips", "episodes_with_metric",
    "episode_weighted_mean", "episode_weighted_median", "episode_weighted_worst_mean",
    "clip_weighted_mean_diagnostic", "episode_static_fraction_mean",
)


def _thought1_markers(root: Path) -> list[Path]:
    markers: list[Path] = []
    manifest = root / "experiment_manifest.json"
    if manifest.is_file():
        markers.append(manifest)
    markers.extend(sorted((root / "workers").glob("rank_*/episode_results.jsonl")))
    return markers


def _read_diagnostic_manifest(root: Path, *, required: bool) -> dict[str, Any] | None:
    path = root / "diagnostic_manifest.json"
    if not path.is_file():
        if required:
            raise ValueError(
                f"Diagnostic input lacks diagnostic_manifest.json and cannot be verified: {root}"
            )
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid diagnostic manifest: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid diagnostic manifest object: {path}")
    if manifest.get("kind") != "future_shadow_diagnostics":
        raise ValueError(f"Manifest is not a future diagnostic manifest: {path}")
    if manifest.get("protocol_fingerprint") in (None, ""):
        raise ValueError(f"Diagnostic manifest lacks protocol_fingerprint: {path}")
    config = manifest.get("config")
    provenance = manifest.get("provenance")
    if not isinstance(config, Mapping) or not all(
        isinstance(config.get(section), Mapping)
        for section in ("checkpoint", "benchmark", "diagnostics")
    ):
        raise ValueError(f"Diagnostic manifest lacks compatibility config sections: {path}")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"Diagnostic manifest lacks provenance: {path}")
    benchmark = config["benchmark"]
    if benchmark.get("backend") != "mock" and any(
        provenance.get(key) in (None, "") for key in ("checkpoint_hash", "fastwam_commit")
    ):
        raise ValueError(f"Real diagnostic manifest lacks checkpoint/upstream provenance: {path}")
    return manifest


def _comparison_signature(manifest: Mapping[str, Any]) -> str:
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    diagnostics = config.get("diagnostics") if isinstance(config.get("diagnostics"), dict) else {}
    benchmark = config.get("benchmark") if isinstance(config.get("benchmark"), dict) else {}
    checkpoint = config.get("checkpoint") if isinstance(config.get("checkpoint"), dict) else {}
    provenance = manifest.get("provenance") if isinstance(manifest.get("provenance"), dict) else {}
    signature = {
        "checkpoint": {
            "path": checkpoint.get("path"), "model_name": checkpoint.get("model_name"),
            "hash": provenance.get("checkpoint_hash"),
        },
        "fastwam_commit": provenance.get("fastwam_commit"),
        "mode": diagnostics.get("mode"),
        "num_video_frames": diagnostics.get("num_video_frames"),
        "num_inference_steps": diagnostics.get("num_inference_steps"),
        "static_motion_threshold": diagnostics.get("static_motion_threshold"),
        "motion_epsilon": diagnostics.get("motion_epsilon"),
        "probe_strategy": diagnostics.get("probe_strategy"),
        "max_probes_per_episode": diagnostics.get("max_probes_per_episode"),
        "explicit_replan_indices": diagnostics.get("explicit_replan_indices"),
        "benchmark": {
            "suite": benchmark.get("suite"), "control_horizon": benchmark.get("control_horizon"),
            "image_size": benchmark.get("image_size"),
        },
    }
    return json.dumps(signature, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def discover_diagnostic_files(
    experiment_dir: Path, input_dirs: Iterable[Path] = ()
) -> list[tuple[Path, str, Path]]:
    discovered: list[tuple[Path, str, Path]] = []
    seen: set[Path] = set()
    input_roots = [Path(value).resolve() for value in input_dirs]
    roots = [Path(experiment_dir).resolve(), *input_roots]
    signatures: dict[str, list[Path]] = defaultdict(list)
    for root_index, root in enumerate(roots):
        markers = _thought1_markers(root)
        if markers:
            raise ValueError(
                "Refusing to treat a Thought 1/source experiment as diagnostic input: "
                + ", ".join(str(path) for path in markers)
            )
        files = sorted((root / "workers").glob("rank_*/diagnostics.jsonl"))
        # An empty primary directory is a valid comparison output/empty
        # aggregate. Every explicit input and every root with rows must carry a
        # valid manifest so checkpoint/protocol filtering never fails open.
        manifest = _read_diagnostic_manifest(
            root,
            required=bool(files) or root_index > 0,
        )
        if manifest is None:
            continue
        signature = _comparison_signature(manifest)
        signatures[signature].append(root)
        fingerprint = str(manifest["protocol_fingerprint"])
        for path in files:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                discovered.append((resolved, fingerprint, root))
    if len(signatures) > 1:
        details = ", ".join(
            str(root) for roots_for_signature in signatures.values() for root in roots_for_signature
        )
        raise ValueError(
            "Diagnostic inputs have incompatible checkpoint/protocol/metric signatures: " + details
        )
    return discovered


def load_diagnostics(
    experiment_dir: Path, input_dirs: Iterable[Path] = ()
) -> list[dict[str, Any]]:
    """Load the last valid row per probe and ignore stale protocol rows."""

    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    order_by_key: dict[tuple[str, str, str], tuple[int, int, int, str]] = {}
    for path, current_fingerprint, source_root in discover_diagnostic_files(
        experiment_dir, input_dirs
    ):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_index, line in enumerate(handle):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
                fingerprint = extra.get("protocol_fingerprint")
                if current_fingerprint is not None and fingerprint != current_fingerprint:
                    continue
                probe_id = row.get("diagnostic_id") or row.get("probe_id")
                if probe_id in (None, ""):
                    continue
                key = (str(row.get("experiment_id", "")), str(probe_id), str(fingerprint or ""))
                row["_diagnostic_root"] = str(source_root)
                order = _record_order(row, path=path, line_index=line_index)
                if key not in records or order > order_by_key[key]:
                    records[key] = row
                    order_by_key[key] = order
    return sorted(
        records.values(),
        key=lambda row: (
            str(row.get("condition", "")), str(row.get("suite", "")),
            int(row.get("task_id", -1)), int(row.get("episode_index", -1)),
            int(row.get("replan_index", -1)),
        ),
    )


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number and abs(number) != float("inf") else None


def _flatten(row: Mapping[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), Mapping) else {}
    extra = row.get("extra") if isinstance(row.get("extra"), Mapping) else {}
    flat = {field: row.get(field) for field in ALL_FIELDS}
    flat["artifact_source_root"] = row.get("_diagnostic_root")
    flat["probe_index"] = row.get("probe_index", extra.get("probe_index"))
    flat["aligned_future_frame_count"] = extra.get("aligned_future_frame_count", 0)
    flat["protocol_fingerprint"] = extra.get("protocol_fingerprint")
    for name in KNOWN_METRICS:
        flat[name] = metrics.get(name)
    return flat


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            })


def _worst(metric: str, values: Sequence[float]) -> float:
    lower_is_worse = "direction_cosine" in metric
    return min(values) if lower_is_worse else max(values)


def _eligible_metric_value(clip: Mapping[str, Any], metric: str) -> float | None:
    """Exclude failed probes even if they happened to emit partial scalars."""

    if clip.get("status") in {"error", "exception", "skipped"}:
        return None
    metrics = clip.get("metrics") if isinstance(clip.get("metrics"), Mapping) else {}
    return _finite_number(metrics.get(metric))


def _episode_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("_diagnostic_root", "")),
                str(row.get("experiment_id", "")),
                str(row.get("job_id", "")),
            )
        ].append(row)
    episodes: list[dict[str, Any]] = []
    for (source_root, experiment_id, job_id), clips in grouped.items():
        first = clips[0]
        item: dict[str, Any] = {
            "job_id": job_id,
            "experiment_id": experiment_id,
            "artifact_source_root": source_root,
            "suite": first.get("suite"), "task_id": first.get("task_id"),
            "episode_index": first.get("episode_index"), "episode_seed": first.get("episode_seed"),
            "condition": first.get("condition"),
            "perturbation_category": first.get("perturbation_category") or "clean",
            "perturbation_level": first.get("perturbation_level") or "clean",
            "perturbation_group": "{}/{}".format(
                first.get("perturbation_category") or "clean",
                first.get("perturbation_level") or "clean",
            ),
            "outcome": "success" if bool(first.get("success", first.get("episode_success"))) else "failure",
            "success": bool(first.get("success", first.get("episode_success"))),
            "clips": len(clips),
        }
        flags = [
            clip.get("static_future_flag")
            for clip in clips
            if clip.get("status") not in {"error", "exception", "skipped"}
        ]
        flags = [bool(flag) for flag in flags if isinstance(flag, bool)]
        item["static_fraction"] = statistics.fmean(flags) if flags else None
        for metric in KNOWN_METRICS:
            values = [
                value
                for value in (_eligible_metric_value(clip, metric) for clip in clips)
                if value is not None
            ]
            item[f"{metric}__mean"] = statistics.fmean(values) if values else None
            item[f"{metric}__median"] = statistics.median(values) if values else None
            item[f"{metric}__worst"] = _worst(metric, values) if values else None
        episodes.append(item)
    return episodes


def _group_summary(
    episodes: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]], key: str
) -> list[dict[str, Any]]:
    episode_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    clip_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        episode_groups[str(episode.get(key) or "unknown")].append(episode)
    for row in rows:
        value = row.get(key)
        if key == "outcome":
            value = "success" if bool(row.get("success", row.get("episode_success"))) else "failure"
        if key == "perturbation_category":
            value = value or "clean"
        if key == "perturbation_group":
            value = "{}/{}".format(
                row.get("perturbation_category") or "clean",
                row.get("perturbation_level") or "clean",
            )
        clip_groups[str(value or "unknown")].append(row)
    output: list[dict[str, Any]] = []
    for group in sorted(episode_groups):
        episode_group = episode_groups[group]
        clip_group = clip_groups[group]
        static_values = [
            float(value) for value in (episode.get("static_fraction") for episode in episode_group)
            if value is not None
        ]
        for metric in KNOWN_METRICS:
            means = [
                float(value) for value in (episode.get(f"{metric}__mean") for episode in episode_group)
                if value is not None
            ]
            medians = [
                float(value) for value in (episode.get(f"{metric}__median") for episode in episode_group)
                if value is not None
            ]
            worst = [
                float(value) for value in (episode.get(f"{metric}__worst") for episode in episode_group)
                if value is not None
            ]
            clip_values = [
                value
                for value in (_eligible_metric_value(clip, metric) for clip in clip_group)
                if value is not None
            ]
            output.append({
                "group": group, "metric": metric, "episodes": len(episode_group),
                "clips": len(clip_group), "episodes_with_metric": len(means),
                "episode_weighted_mean": statistics.fmean(means) if means else None,
                "episode_weighted_median": statistics.median(medians) if medians else None,
                "episode_weighted_worst_mean": statistics.fmean(worst) if worst else None,
                "clip_weighted_mean_diagnostic": statistics.fmean(clip_values) if clip_values else None,
                "episode_static_fraction_mean": statistics.fmean(static_values) if static_values else None,
            })
    return output


def _paired_differences(episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    clean: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for episode in episodes:
        if episode.get("condition") == "clean":
            clean[(episode.get("suite"), episode.get("task_id"), episode.get("episode_seed"))] = episode
    pairs: list[tuple[tuple[Any, Any, Any], dict[str, Any], dict[str, Any]]] = []
    for episode in episodes:
        if episode.get("condition") != "ood":
            continue
        key = (episode.get("suite"), episode.get("task_id"), episode.get("episode_seed"))
        match = clean.get(key)
        if match is not None:
            pairs.append((key, match, episode))
    metrics: dict[str, Any] = {}
    rng = random.Random(0)
    for metric in KNOWN_METRICS:
        differences_by_cluster: dict[tuple[Any, Any, Any], list[float]] = defaultdict(list)
        for key, left, right in pairs:
            left_value = left.get(f"{metric}__mean")
            right_value = right.get(f"{metric}__mean")
            if left_value is not None and right_value is not None:
                differences_by_cluster[key].append(float(right_value) - float(left_value))
        differences = [statistics.fmean(values) for values in differences_by_cluster.values()]
        if not differences:
            continue
        draws = [
            statistics.fmean(rng.choice(differences) for _ in differences)
            for _ in range(2000)
        ]
        draws.sort()
        metrics[metric] = {
            "paired_clusters": len(differences),
            "ood_minus_id_mean": statistics.fmean(differences),
            "cluster_bootstrap_ci95_low": draws[int(0.025 * (len(draws) - 1))],
            "cluster_bootstrap_ci95_high": draws[int(0.975 * (len(draws) - 1))],
        }
    return {"eligible_episode_pairs": len(pairs), "metrics": metrics}


def _coverage_plan(
    experiment_dir: Path,
    input_dirs: Sequence[Path],
    rows: Sequence[dict[str, Any]],
) -> tuple[int, int | None]:
    roots = [Path(experiment_dir).resolve(), *[Path(value).resolve() for value in input_dirs]]
    planned_jobs = 0
    planned_clips = 0
    manifests_found = 0
    for index, root in enumerate(roots):
        files = list((root / "workers").glob("rank_*/diagnostics.jsonl"))
        manifest = _read_diagnostic_manifest(root, required=bool(files) or index > 0)
        if manifest is None:
            continue
        manifests_found += 1
        job_count = manifest.get("planned_job_count")
        if not isinstance(job_count, int) or job_count < 0:
            raise ValueError(f"Diagnostic manifest has invalid planned_job_count: {root}")
        diagnostics = (
            (manifest.get("config") or {}).get("diagnostics")
            if isinstance(manifest.get("config"), Mapping)
            else None
        )
        probes = diagnostics.get("max_probes_per_episode") if isinstance(diagnostics, Mapping) else None
        if not isinstance(probes, int) or probes <= 0:
            raise ValueError(f"Diagnostic manifest has invalid max_probes_per_episode: {root}")
        planned_jobs += job_count
        planned_clips += job_count * probes
    if manifests_found:
        return planned_jobs, planned_clips
    observed_jobs = {
        (
            str(row.get("_diagnostic_root", "")),
            str(row.get("experiment_id", "")),
            str(row.get("job_id", "")),
        )
        for row in rows
        if row.get("job_id")
    }
    return len(observed_jobs), None


def aggregate_diagnostics(
    experiment_dir: Path, input_dirs: Iterable[Path] = ()
) -> dict[str, Any]:
    """Create exactly five diagnostic CSVs plus a machine-readable summary."""

    experiment_dir = Path(experiment_dir)
    markers = _thought1_markers(experiment_dir)
    if markers:
        raise ValueError(
            "Refusing to write diagnostic summaries into a Thought 1/source experiment: "
            + ", ".join(str(path) for path in markers)
        )
    input_paths = [Path(value) for value in input_dirs]
    rows = load_diagnostics(experiment_dir, input_paths)
    flat = [_flatten(row) for row in rows]
    episodes = _episode_rows(rows)
    overall = _group_summary(
        [{**episode, "overall": "all"} for episode in episodes],
        [{**row, "overall": "all"} for row in rows],
        "overall",
    )
    outcome = _group_summary(episodes, rows, "outcome")
    condition = _group_summary(episodes, rows, "condition")
    perturbation = _group_summary(episodes, rows, "perturbation_group")
    static_cases = [
        item
        for item in flat
        if item.get("static_future_flag") is True
        and item.get("status") not in {"error", "exception", "skipped"}
    ]
    summary_dir = experiment_dir / "summary"
    _write_csv(summary_dir / "all_diagnostics.csv", flat, ALL_FIELDS)
    _write_csv(summary_dir / "consistency_by_outcome.csv", outcome, GROUP_FIELDS)
    _write_csv(summary_dir / "consistency_by_condition.csv", condition, GROUP_FIELDS)
    _write_csv(summary_dir / "consistency_by_perturbation.csv", perturbation, GROUP_FIELDS)
    _write_csv(summary_dir / "static_future_cases.csv", static_cases, ALL_FIELDS)

    status_counts = {name: 0 for name in ("exact", "approximate", "unavailable", "error")}
    for row in rows:
        if row.get("status") in {"error", "exception"}:
            bucket = "error"
        elif (
            row.get("status") in {"unavailable", "skipped"}
            or not row.get("num_video_frames")
            or int((row.get("extra") or {}).get("aligned_future_frame_count", 0)) <= 0
        ):
            bucket = "unavailable"
        elif bool(row.get("approximate_alignment")):
            bucket = "approximate"
        else:
            bucket = "exact"
        status_counts[bucket] += 1
    planned_jobs, planned_clips_maximum = _coverage_plan(
        experiment_dir, input_paths, rows
    )
    completed_jobs = len(
        {
            (
                str(row.get("_diagnostic_root", "")),
                str(row.get("experiment_id", "")),
                str(row.get("job_id", "")),
            )
            for row in rows
            if row.get("job_id")
        }
    )
    metrics = {
        "causal_interpretation_allowed": False,
        "aggregation_primary": "episode_weighted",
        "clip_weighted_role": "diagnostic_only",
        "denominators": {
            "planned_jobs": planned_jobs,
            "planned_clips_maximum": planned_clips_maximum,
            "completed_jobs_with_probe_rows": completed_jobs,
            "generated_clips": sum(bool(row.get("num_video_frames")) for row in rows),
            "exact_clips": status_counts["exact"],
            "approximate_clips": status_counts["approximate"],
            "unavailable_clips": status_counts["unavailable"],
            "error_clips": status_counts["error"],
            "aligned_future_frames": sum(
                int((row.get("extra") or {}).get("aligned_future_frame_count", 0)) for row in rows
            ),
        },
        "episodes": len(episodes), "clips": len(rows), "overall": overall,
        "by_outcome": outcome, "by_condition": condition, "by_perturbation": perturbation,
        "paired_id_ood": _paired_differences(episodes),
        "limitations": [
            "Clips from one episode are not independent; primary summaries first aggregate by job_id.",
            "Success/failure comparisons are associations, not causal effects.",
            "The released Fast-WAM action does not read the generated future in this shadow protocol.",
        ],
    }
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "diagnostic_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = experiment_dir / "diagnostic_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest, dict):
            manifest["status"] = "aggregated"
            manifest["aggregation"] = {
                "episodes": len(episodes),
                "clips": len(rows),
                "error_clips": status_counts["error"],
                "summary_dir": str(summary_dir),
            }
            _atomic_json_write(manifest_path, manifest)
    return metrics


aggregate_experiment_diagnostics = aggregate_diagnostics

__all__ = [
    "aggregate_diagnostics", "aggregate_experiment_diagnostics",
    "discover_diagnostic_files", "load_diagnostics",
]
