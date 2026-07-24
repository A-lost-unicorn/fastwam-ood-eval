"""Aggregation, threshold candidacy, and read-only pilot sensitivity analysis."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from fastwam_ood_eval.diagnostics.artifact_writer import _atomic_json_write
from fastwam_ood_eval.evaluation.evaluator import git_commit, git_dirty


def linear_quantile(values: Sequence[float], quantile: float) -> float | None:
    """NumPy-compatible linear quantile without adding a runtime dependency."""

    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    q = float(quantile)
    if not 0.0 <= q <= 1.0:
        raise ValueError("quantile must be between zero and one")
    position = (len(finite) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return finite[lower]
    weight = position - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def higher_quantile(values: Sequence[float], quantile: float) -> float | None:
    """Conservative empirical quantile using the observed value above q."""

    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    q = float(quantile)
    if not 0.0 <= q <= 1.0:
        raise ValueError("quantile must be between zero and one")
    return finite[int(math.ceil((len(finite) - 1) * q))]


def _distribution(
    values: Sequence[float],
    threshold_quantile: float,
    threshold_quantile_method: str,
) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "mean": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "configured_quantile": threshold_quantile,
            "configured_quantile_method": threshold_quantile_method,
            "configured_quantile_value": None,
            "maximum": None,
        }
    if threshold_quantile_method != "higher":
        raise ValueError(
            "Only the protocol-pinned conservative higher quantile is supported"
        )
    return {
        "count": len(finite),
        "minimum": min(finite),
        "median": linear_quantile(finite, 0.5),
        "mean": sum(finite) / len(finite),
        "p90": linear_quantile(finite, 0.9),
        "p95": linear_quantile(finite, 0.95),
        "p99": linear_quantile(finite, 0.99),
        "configured_quantile": threshold_quantile,
        "configured_quantile_method": threshold_quantile_method,
        "configured_quantile_value": higher_quantile(
            finite, threshold_quantile
        ),
        "maximum": max(finite),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid static calibration manifest: {path}") from exc
    if not isinstance(payload, dict) or payload.get("kind") != "static_motion_calibration":
        raise RuntimeError(f"Not a static calibration manifest: {path}")
    if not payload.get("compatibility_fingerprint"):
        raise RuntimeError(f"Calibration manifest lacks compatibility fingerprint: {path}")
    return payload


def _record_order(
    row: Mapping[str, Any],
    path: Path,
    line_index: int,
) -> tuple[int, int, int, str]:
    def integer(name: str) -> int:
        try:
            value = row.get(name, 0)
            return 0 if isinstance(value, bool) else int(value)
        except (TypeError, ValueError):
            return 0

    try:
        mtime = int(path.stat().st_mtime_ns)
    except OSError:
        mtime = 0
    return (
        integer("attempt_started_ns"),
        integer("recorded_at_ns") or mtime,
        line_index,
        str(path),
    )


def _latest_samples(source_dir: Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    orders: dict[str, tuple[int, int, int, str]] = {}
    for path in sorted(
        source_dir.glob("workers/rank_*/static_calibration_samples.jsonl")
    ):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_index, line in enumerate(handle):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                sample_id = row.get("sample_id")
                if sample_id in (None, ""):
                    continue
                key = str(sample_id)
                order = _record_order(row, path, line_index)
                if key not in latest or order > orders[key]:
                    latest[key] = row
                    orders[key] = order
    return [latest[key] for key in sorted(latest)]


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = Path(path).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(Path(path))
    return unique


def _atomic_text_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sample_csv(samples: Sequence[Mapping[str, Any]]) -> str:
    columns = (
        "source_dir",
        "sample_id",
        "job_id",
        "status",
        "eligible_for_threshold",
        "condition",
        "task_id",
        "episode_index",
        "perturbation_category",
        "perturbation_level",
        "same_frame_max_motion_energy",
        "noop_full_horizon_motion_energy",
        "noop_full_horizon_offset",
        "pixel_full_horizon_mae",
        "exclusion_reason",
        "error",
    )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for sample in samples:
        metrics = sample.get("metrics")
        metrics = metrics if isinstance(metrics, Mapping) else {}
        writer.writerow(
            {
                "source_dir": sample.get("_source_dir"),
                "sample_id": sample.get("sample_id"),
                "job_id": sample.get("job_id"),
                "status": sample.get("status"),
                "eligible_for_threshold": sample.get(
                    "eligible_for_threshold"
                ),
                "condition": sample.get("condition"),
                "task_id": sample.get("task_id"),
                "episode_index": sample.get("episode_index"),
                "perturbation_category": sample.get(
                    "perturbation_category"
                ),
                "perturbation_level": sample.get("perturbation_level"),
                "same_frame_max_motion_energy": metrics.get(
                    "same_frame_max_motion_energy"
                ),
                "noop_full_horizon_motion_energy": metrics.get(
                    "noop_full_horizon_motion_energy"
                ),
                "noop_full_horizon_offset": metrics.get(
                    "noop_full_horizon_offset"
                ),
                "pixel_full_horizon_mae": metrics.get(
                    "pixel_full_horizon_mae"
                ),
                "exclusion_reason": sample.get("exclusion_reason"),
                "error": sample.get("error"),
            }
        )
    return buffer.getvalue()


def _numeric_metric(sample: Mapping[str, Any], name: str) -> float | None:
    metrics = sample.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    value = metrics.get(name)
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _check(
    *,
    observed: Any,
    required: Any,
    passed: bool,
    description: str,
) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "observed": observed,
        "required": required,
        "description": description,
    }


def _freeze_checks(
    *,
    calibration: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    valid_samples: Sequence[Mapping[str, Any]],
    planned_job_count: int,
    source_manifests: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    condition_counts = Counter(str(row.get("condition")) for row in valid_samples)
    ood_category_counts = Counter(
        str(row.get("perturbation_category"))
        for row in valid_samples
        if row.get("condition") == "ood"
        and row.get("perturbation_category") not in (None, "")
    )
    checks: dict[str, dict[str, Any]] = {}
    minimum_total = int(calibration.get("minimum_samples_for_freeze", 0))
    checks["minimum_total_samples"] = _check(
        observed=len(valid_samples),
        required=minimum_total,
        passed=len(valid_samples) >= minimum_total,
        description="independent eligible null samples",
    )
    minimum_condition = int(
        calibration.get("minimum_samples_per_condition_for_freeze", 0)
    )
    for condition in calibration.get("required_conditions", []):
        count = condition_counts[str(condition)]
        checks[f"condition:{condition}"] = _check(
            observed=count,
            required=minimum_condition,
            passed=count >= minimum_condition,
            description=f"eligible {condition} null samples",
        )
    minimum_category = int(
        calibration.get(
            "minimum_samples_per_ood_category_for_freeze", 0
        )
    )
    for category in calibration.get("required_ood_categories", []):
        count = ood_category_counts[str(category)]
        checks[f"ood_category:{category}"] = _check(
            observed=count,
            required=minimum_category,
            passed=count >= minimum_category,
            description=f"eligible OOD null samples for {category}",
        )
    complete_coverage = len(samples) == planned_job_count
    checks["all_planned_jobs_recorded"] = _check(
        observed=len(samples),
        required=planned_job_count,
        passed=complete_coverage,
        description="latest durable sample record per planned calibration job",
    )
    ineligible_count = len(samples) - len(valid_samples)
    checks["no_excluded_or_failed_samples"] = _check(
        observed=ineligible_count,
        required=0,
        passed=ineligible_count == 0,
        description="excluded, skipped, malformed, or failed samples",
    )
    required_dirty_fields = (
        "git_dirty",
        "fastwam_dirty",
        "libero_dirty",
        "libero_plus_dirty",
    )
    clean_sources = 0
    for source in source_manifests:
        provenance = source.get("provenance")
        if isinstance(provenance, Mapping) and all(
            provenance.get(field) is False for field in required_dirty_fields
        ):
            clean_sources += 1
    checks["all_source_trees_explicitly_clean"] = _check(
        observed=f"{clean_sources}/{len(source_manifests)}",
        required=f"{len(source_manifests)}/{len(source_manifests)}",
        passed=clean_sources == len(source_manifests),
        description=(
            "every source manifest explicitly records clean project, Fast-WAM, "
            "LIBERO, and LIBERO-Plus trees"
        ),
    )

    frequencies: list[float] = []
    for row in valid_samples:
        value = row.get("runtime_control_frequency_hz")
        try:
            frequency = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(frequency) and frequency > 0:
            frequencies.append(frequency)
    frequency_consistent = (
        len(frequencies) == len(valid_samples)
        and bool(frequencies)
        and all(
            math.isclose(
                frequencies[0],
                frequency,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for frequency in frequencies[1:]
        )
    )
    checks["runtime_control_frequency_consistent"] = _check(
        observed=(
            sorted(set(frequencies))
            if len(frequencies) == len(valid_samples)
            else {
                "recorded": len(frequencies),
                "eligible_samples": len(valid_samples),
                "values_hz": sorted(set(frequencies)),
            }
        ),
        required="one explicit positive Hz value shared by every eligible sample",
        passed=frequency_consistent,
        description="the no-op horizon has one consistent runtime time scale",
    )

    frame_shapes: list[tuple[int, ...]] = []
    for row in valid_samples:
        value = row.get("model_frame_shape")
        if (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and len(value) == 3
        ):
            try:
                shape = tuple(int(dimension) for dimension in value)
            except (TypeError, ValueError):
                continue
            if all(dimension > 0 for dimension in shape):
                frame_shapes.append(shape)
    shape_consistent = (
        len(frame_shapes) == len(valid_samples)
        and bool(frame_shapes)
        and all(shape == frame_shapes[0] for shape in frame_shapes[1:])
    )
    checks["model_frame_shape_consistent"] = _check(
        observed=(
            [list(shape) for shape in sorted(set(frame_shapes))]
            if len(frame_shapes) == len(valid_samples)
            else {
                "recorded": len(frame_shapes),
                "eligible_samples": len(valid_samples),
                "values": [
                    list(shape) for shape in sorted(set(frame_shapes))
                ],
            }
        ),
        required="one explicit H×W×3 shape shared by every eligible sample",
        passed=shape_consistent,
        description="all null energies use the same model-frame geometry",
    )
    return checks


def aggregate_static_calibration(
    experiment_dir: Path,
    input_dirs: Sequence[Path] = (),
    diagnostic_dirs: Sequence[Path] = (),
) -> dict[str, Any]:
    """Pool compatible null cohorts and write a non-destructive threshold summary."""

    destination = Path(experiment_dir)
    candidates = _unique_paths([destination, *input_dirs])
    sources: list[tuple[Path, Path, dict[str, Any]]] = []
    for directory in candidates:
        manifest_path = directory / "calibration_manifest.json"
        if manifest_path.is_file():
            sources.append((directory, manifest_path, _load_manifest(manifest_path)))
    if not sources:
        raise FileNotFoundError(
            "No calibration_manifest.json found in the experiment/input directories"
        )
    destination_resolved = destination.resolve()
    for directory, _, _ in sources:
        source_resolved = directory.resolve()
        if destination_resolved != source_resolved and (
            source_resolved in destination_resolved.parents
            or destination_resolved in source_resolved.parents
        ):
            raise ValueError(
                "Calibration aggregation destination and source must be disjoint "
                f"unless they are identical: destination={destination_resolved}, "
                f"source={source_resolved}"
            )
    for diagnostic_dir in _unique_paths(diagnostic_dirs):
        diagnostic_resolved = diagnostic_dir.resolve()
        if (
            diagnostic_resolved == destination_resolved
            or diagnostic_resolved in destination_resolved.parents
            or destination_resolved in diagnostic_resolved.parents
        ):
            raise ValueError(
                "Threshold sensitivity output must be disjoint from every "
                f"read-only diagnostic source: destination={destination_resolved}, "
                f"source={diagnostic_resolved}"
            )

    fingerprints = {
        str(manifest["compatibility_fingerprint"])
        for _, _, manifest in sources
    }
    if len(fingerprints) != 1:
        details = {
            str(directory): manifest.get("compatibility_fingerprint")
            for directory, _, manifest in sources
        }
        raise RuntimeError(
            "Static calibration inputs have incompatible encoder/protocol semantics: "
            f"{details}"
        )
    compatibility_fingerprint = next(iter(fingerprints))
    compatibility = sources[0][2].get("calibration_compatibility")
    if not isinstance(compatibility, Mapping):
        raise RuntimeError("Calibration manifest lacks compatibility payload")
    calibration = compatibility.get("calibration")
    if not isinstance(calibration, Mapping):
        raise RuntimeError("Calibration compatibility lacks calibration settings")
    threshold_quantile = float(calibration["threshold_quantile"])
    quantile_method_source_pinned = (
        calibration.get("threshold_quantile_method") is not None
    )
    threshold_quantile_method = str(
        calibration.get("threshold_quantile_method", "higher")
    )

    samples: list[dict[str, Any]] = []
    source_manifests: list[dict[str, Any]] = []
    planned_job_count = 0
    for directory, manifest_path, manifest in sources:
        planned_job_count += int(manifest.get("planned_job_count", 0))
        sample_files = sorted(
            directory.glob(
                "workers/rank_*/static_calibration_samples.jsonl"
            )
        )
        job_manifest_path = (
            directory / "static_calibration_job_manifest.jsonl"
        )
        source_manifests.append(
            {
                "directory": str(directory),
                "manifest_path": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "job_manifest_path": str(job_manifest_path),
                "job_manifest_sha256": (
                    _sha256(job_manifest_path)
                    if job_manifest_path.is_file()
                    else None
                ),
                "sample_files": [
                    {
                        "path": str(path),
                        "sha256": _sha256(path),
                    }
                    for path in sample_files
                ],
                "experiment_id": manifest.get("experiment_id"),
                "protocol_fingerprint": manifest.get(
                    "protocol_fingerprint"
                ),
                "planned_job_count": manifest.get("planned_job_count"),
                "provenance": manifest.get("provenance", {}),
            }
        )
        for row in _latest_samples(directory):
            copied = dict(row)
            copied["_source_dir"] = str(directory)
            copied["_source_protocol_fingerprint"] = manifest.get(
                "protocol_fingerprint"
            )
            copied["_source_compatibility_fingerprint"] = manifest.get(
                "compatibility_fingerprint"
            )
            samples.append(copied)

    expected_offsets = [
        int(value) for value in calibration.get("capture_offsets", [])
    ]
    expected_settle_steps = int(calibration.get("settle_steps", -1))
    expected_full_offset = expected_offsets[-1] if expected_offsets else None
    expected_noop_hash = compatibility.get("standard_noop_action_sha256")
    expected_embedding_semantics = compatibility.get(
        "frame_embedding_semantics"
    )

    def sample_protocol_complete(row: Mapping[str, Any]) -> bool:
        try:
            capture_offsets = [
                int(value) for value in row.get("capture_offsets", [])
            ]
            settle_steps = int(row.get("settle_steps", -1))
            settle_executed = int(row.get("settle_steps_executed", -1))
            capture_executed = int(
                row.get("capture_steps_executed", -1)
            )
        except (TypeError, ValueError):
            return False
        return bool(
            row.get("protocol_fingerprint")
            == row.get("_source_protocol_fingerprint")
            and row.get("compatibility_fingerprint")
            == row.get("_source_compatibility_fingerprint")
            and capture_offsets == expected_offsets
            and settle_steps == expected_settle_steps
            and settle_executed == expected_settle_steps
            and expected_full_offset is not None
            and capture_executed == expected_full_offset
            and row.get("policy_action_sampled") is False
            and row.get("standard_noop_action_sha256")
            == expected_noop_hash
            and row.get("frame_embedding_semantics")
            == expected_embedding_semantics
        )

    valid_samples = [
        row
        for row in samples
        if row.get("status") == "completed"
        and row.get("eligible_for_threshold") is True
        and sample_protocol_complete(row)
        and _numeric_metric(row, "same_frame_max_motion_energy") is not None
        and _numeric_metric(row, "noop_full_horizon_motion_energy") is not None
    ]
    same_frame_values = [
        value
        for row in valid_samples
        if (
            value := _numeric_metric(row, "same_frame_max_motion_energy")
        )
        is not None
    ]
    noop_values = [
        value
        for row in valid_samples
        if (
            value := _numeric_metric(
                row, "noop_full_horizon_motion_energy"
            )
        )
        is not None
    ]
    pixel_values = [
        value
        for row in valid_samples
        if (
            value := _numeric_metric(row, "pixel_full_horizon_mae")
        )
        is not None
    ]
    offset_values: dict[str, list[float]] = {}
    for row in valid_samples:
        metrics = row.get("metrics")
        by_offset = (
            metrics.get("noop_motion_energy_by_offset")
            if isinstance(metrics, Mapping)
            else None
        )
        if not isinstance(by_offset, Mapping):
            continue
        for offset, value in by_offset.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                offset_values.setdefault(str(offset), []).append(number)
    condition_values: dict[str, list[float]] = {}
    category_values: dict[str, list[float]] = {}
    for row in valid_samples:
        value = _numeric_metric(row, "noop_full_horizon_motion_energy")
        if value is None:
            continue
        condition_values.setdefault(str(row.get("condition")), []).append(value)
        if (
            row.get("condition") == "ood"
            and row.get("perturbation_category") not in (None, "")
        ):
            category_values.setdefault(
                str(row.get("perturbation_category")), []
            ).append(value)
    same_frame_distribution = _distribution(
        same_frame_values,
        threshold_quantile,
        threshold_quantile_method,
    )
    noop_distribution = _distribution(
        noop_values,
        threshold_quantile,
        threshold_quantile_method,
    )
    pixel_distribution = _distribution(
        pixel_values,
        threshold_quantile,
        threshold_quantile_method,
    )
    noop_by_offset = {
        offset: _distribution(
            values,
            threshold_quantile,
            threshold_quantile_method,
        )
        for offset, values in sorted(
            offset_values.items(), key=lambda item: int(item[0])
        )
    }
    noop_by_condition = {
        condition: _distribution(
            values,
            threshold_quantile,
            threshold_quantile_method,
        )
        for condition, values in sorted(condition_values.items())
    }
    noop_by_ood_category = {
        category: _distribution(
            values,
            threshold_quantile,
            threshold_quantile_method,
        )
        for category, values in sorted(category_values.items())
    }
    same_frame_q = same_frame_distribution["configured_quantile_value"]
    noop_q = noop_distribution["configured_quantile_value"]
    candidate_threshold = (
        max(float(same_frame_q), float(noop_q))
        if same_frame_q is not None and noop_q is not None
        else None
    )
    checks = _freeze_checks(
        calibration=calibration,
        samples=samples,
        valid_samples=valid_samples,
        planned_job_count=planned_job_count,
        source_manifests=source_manifests,
    )
    checks["quantile_method_source_pinned"] = _check(
        observed=quantile_method_source_pinned,
        required=True,
        passed=quantile_method_source_pinned,
        description=(
            "threshold quantile interpolation method recorded before sample collection"
        ),
    )
    freeze_eligible = bool(checks) and all(
        bool(item["passed"]) for item in checks.values()
    )
    if candidate_threshold is None:
        threshold_status = "unavailable"
    elif freeze_eligible:
        threshold_status = "eligible_for_manual_freeze"
    else:
        threshold_status = "candidate_only"

    status_counts = Counter(str(row.get("status", "missing")) for row in samples)
    condition_counts = Counter(
        str(row.get("condition")) for row in valid_samples
    )
    ood_category_counts = Counter(
        str(row.get("perturbation_category"))
        for row in valid_samples
        if row.get("condition") == "ood"
        and row.get("perturbation_category") not in (None, "")
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "static_motion_calibration_aggregate",
        "compatibility_fingerprint": compatibility_fingerprint,
        "source_manifests": source_manifests,
        "planned_job_count": planned_job_count,
        "latest_sample_count": len(samples),
        "eligible_sample_count": len(valid_samples),
        "status_counts": dict(sorted(status_counts.items())),
        "eligible_condition_counts": dict(sorted(condition_counts.items())),
        "eligible_ood_category_counts": dict(
            sorted(ood_category_counts.items())
        ),
        "capture_offsets": calibration.get("capture_offsets"),
        "full_horizon_offset": (
            calibration.get("capture_offsets", [None])[-1]
        ),
        "threshold_quantile": threshold_quantile,
        "threshold_quantile_method": threshold_quantile_method,
        "threshold_quantile_method_source_pinned": (
            quantile_method_source_pinned
        ),
        "same_frame_noise_distribution": same_frame_distribution,
        "noop_full_horizon_distribution": noop_distribution,
        "noop_motion_distribution_by_offset": noop_by_offset,
        "noop_full_horizon_by_condition": noop_by_condition,
        "noop_full_horizon_by_ood_category": noop_by_ood_category,
        "pixel_full_horizon_distribution": pixel_distribution,
        "candidate_static_motion_threshold": candidate_threshold,
        "candidate_formula": (
            "max(q_same_frame_max_pairwise, q_noop_0_to_full_horizon)"
        ),
        "threshold_status": threshold_status,
        "freeze_eligible": freeze_eligible,
        "freeze_requires_manual_approval": True,
        "freeze_checks": checks,
        "metric_semantics": compatibility.get(
            "frame_embedding_semantics"
        ),
        "standard_noop_action": compatibility.get(
            "standard_noop_action"
        ),
        "limitations": [
            "The metric is an approximate independently re-encoded frame embedding, not a native temporal future latent.",
            "A candidate threshold never rewrites source diagnostic JSONL or becomes frozen automatically.",
            "Success/OOD pilot labels are not used to estimate the null threshold.",
        ],
    }

    summary_dir = destination / "summary"
    summary_path = summary_dir / "static_calibration_summary.json"
    _atomic_json_write(summary_path, summary)
    _atomic_text_write(
        summary_dir / "static_calibration_samples.csv",
        _sample_csv(samples),
    )
    aggregation_manifest = {
        "schema_version": 1,
        "kind": "static_motion_calibration_aggregation",
        "destination": str(destination),
        "compatibility_fingerprint": compatibility_fingerprint,
        "source_manifests": source_manifests,
        "summary_path": str(summary_path),
        "aggregation_provenance": {
            "git_commit": git_commit(Path.cwd()),
            "git_dirty": git_dirty(Path.cwd()),
            "implementation_sha256": _sha256(Path(__file__)),
        },
    }
    _atomic_json_write(
        summary_dir / "static_calibration_aggregation_manifest.json",
        aggregation_manifest,
    )

    if diagnostic_dirs:
        sensitivity = static_threshold_sensitivity(
            candidate_threshold=candidate_threshold,
            threshold_status=threshold_status,
            calibration_summary_path=summary_path,
            diagnostic_dirs=diagnostic_dirs,
            output_dir=summary_dir,
        )
        # Expose the derived result to the immediate caller, but keep the
        # threshold summary file byte-stable.  The sensitivity artifact pins
        # the SHA-256 of that threshold-only summary.
        summary["diagnostic_sensitivity"] = sensitivity
    return summary


def _latest_diagnostic_rows(directory: Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    orders: dict[str, tuple[int, int, int, str]] = {}
    for path in sorted(directory.glob("workers/rank_*/diagnostics.jsonl")):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_index, line in enumerate(handle):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                identity = row.get("diagnostic_id")
                if identity in (None, ""):
                    identity = (
                        f"{row.get('job_id')}:{row.get('replan_index')}:"
                        f"{row.get('extra', {}).get('protocol_fingerprint')}"
                    )
                key = str(identity)
                order = _record_order(row, path, line_index)
                if key not in latest or order > orders[key]:
                    latest[key] = row
                    orders[key] = order
    return [latest[key] for key in sorted(latest)]


def _sensitivity_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "predicted_static_count": sum(
            bool(row.get("_candidate_predicted_static")) for row in rows
        ),
        "predicted_static_rate": (
            sum(bool(row.get("_candidate_predicted_static")) for row in rows)
            / len(rows)
            if rows
            else None
        ),
        "actual_static_count": sum(
            bool(row.get("_candidate_actual_static")) for row in rows
            if row.get("_candidate_actual_static") is not None
        ),
        "actual_static_rate": (
            sum(bool(row.get("_candidate_actual_static")) for row in rows)
            / sum(
                row.get("_candidate_actual_static") is not None for row in rows
            )
            if any(
                row.get("_candidate_actual_static") is not None for row in rows
            )
            else None
        ),
        "legacy_predicted_static_count": sum(
            row.get("_legacy_predicted_static") is True for row in rows
        ),
    }


def _episode_success_label(row: Mapping[str, Any]) -> bool | None:
    """Return only an explicit boolean outcome; never coerce missing to failure."""

    value = row.get("episode_success", row.get("success"))
    return value if isinstance(value, bool) else None


def static_threshold_sensitivity(
    *,
    candidate_threshold: float | None,
    threshold_status: str,
    calibration_summary_path: Path,
    diagnostic_dirs: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Reclassify pilot energies in a derived file; source rows stay immutable."""

    sources = _unique_paths(diagnostic_dirs)
    source_manifests: list[dict[str, Any]] = []
    classified: list[dict[str, Any]] = []
    missing_energy = 0
    for directory in sources:
        manifest_path = directory / "diagnostic_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Diagnostic manifest does not exist: {manifest_path}"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid diagnostic manifest: {manifest_path}"
            ) from exc
        source_manifests.append(
            {
                "directory": str(directory),
                "manifest_path": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "diagnostic_files": [
                    {
                        "path": str(path),
                        "sha256": _sha256(path),
                    }
                    for path in sorted(
                        directory.glob("workers/rank_*/diagnostics.jsonl")
                    )
                ],
                "experiment_id": manifest.get("experiment_id"),
                "protocol_fingerprint": manifest.get(
                    "protocol_fingerprint"
                ),
                "legacy_static_motion_threshold": (
                    manifest.get("config", {})
                    .get("diagnostics", {})
                    .get("static_motion_threshold")
                ),
            }
        )
        for row in _latest_diagnostic_rows(directory):
            metrics = row.get("metrics")
            if not isinstance(metrics, Mapping):
                missing_energy += 1
                continue
            predicted = metrics.get("predicted_motion_energy")
            actual = metrics.get("actual_motion_energy")
            try:
                predicted_value = float(predicted)
            except (TypeError, ValueError):
                missing_energy += 1
                continue
            if not math.isfinite(predicted_value):
                missing_energy += 1
                continue
            actual_value: float | None
            try:
                actual_value = float(actual)
                if not math.isfinite(actual_value):
                    actual_value = None
            except (TypeError, ValueError):
                actual_value = None
            copied = dict(row)
            copied["_source_dir"] = str(directory)
            copied["_predicted_motion_energy"] = predicted_value
            copied["_actual_motion_energy"] = actual_value
            copied["_legacy_predicted_static"] = metrics.get(
                "predicted_static",
                metrics.get("static_future_flag"),
            )
            copied["_candidate_predicted_static"] = (
                predicted_value <= candidate_threshold
                if candidate_threshold is not None
                else None
            )
            copied["_candidate_actual_static"] = (
                actual_value <= candidate_threshold
                if candidate_threshold is not None and actual_value is not None
                else None
            )
            classified.append(copied)

    grouped: dict[str, dict[str, Any]] = {
        "all": _sensitivity_group(classified)
    }
    for condition in sorted(
        {str(row.get("condition")) for row in classified}
    ):
        grouped[f"condition:{condition}"] = _sensitivity_group(
            [row for row in classified if str(row.get("condition")) == condition]
        )
    for success in (True, False):
        grouped[f"episode_success:{str(success).lower()}"] = _sensitivity_group(
            [
                row
                for row in classified
                if _episode_success_label(row) is success
            ]
        )
    missing_success_labels = sum(
        _episode_success_label(row) is None for row in classified
    )

    sensitivity = {
        "schema_version": 1,
        "kind": "derived_static_threshold_sensitivity",
        "candidate_static_motion_threshold": candidate_threshold,
        "threshold_status": threshold_status,
        "interpretation_allowed": threshold_status
        == "eligible_for_manual_freeze",
        "calibration_summary_path": str(calibration_summary_path),
        "calibration_summary_sha256": _sha256(calibration_summary_path),
        "diagnostic_sources": source_manifests,
        "classified_rows": len(classified),
        "missing_motion_energy_rows": missing_energy,
        "missing_episode_success_labels": missing_success_labels,
        "groups": grouped,
        "source_rows_rewritten": False,
        "warning": (
            "candidate_only classifications are sensitivity analysis, not a "
            "frozen paper result"
        ),
    }
    _atomic_json_write(
        Path(output_dir) / "static_threshold_sensitivity.json",
        sensitivity,
    )

    columns = (
        "source_dir",
        "diagnostic_id",
        "job_id",
        "condition",
        "episode_success",
        "predicted_motion_energy",
        "actual_motion_energy",
        "legacy_predicted_static",
        "candidate_predicted_static",
        "candidate_actual_static",
    )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for row in classified:
        writer.writerow(
            {
                "source_dir": row.get("_source_dir"),
                "diagnostic_id": row.get("diagnostic_id"),
                "job_id": row.get("job_id"),
                "condition": row.get("condition"),
                "episode_success": row.get(
                    "episode_success", row.get("success")
                ),
                "predicted_motion_energy": row.get(
                    "_predicted_motion_energy"
                ),
                "actual_motion_energy": row.get("_actual_motion_energy"),
                "legacy_predicted_static": row.get(
                    "_legacy_predicted_static"
                ),
                "candidate_predicted_static": row.get(
                    "_candidate_predicted_static"
                ),
                "candidate_actual_static": row.get(
                    "_candidate_actual_static"
                ),
            }
        )
    _atomic_text_write(
        Path(output_dir) / "static_threshold_sensitivity.csv",
        buffer.getvalue(),
    )
    return sensitivity


def generate_static_calibration_report(experiment_dir: Path) -> Path:
    """Generate a compact paper-facing report from the immutable summary."""

    root = Path(experiment_dir)
    summary_path = root / "summary" / "static_calibration_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Static calibration summary does not exist: {summary_path}"
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid static calibration summary: {summary_path}"
        ) from exc
    checks = summary.get("freeze_checks", {})
    check_lines = [
        "| 冻结条件 | 观测 | 要求 | 通过 |",
        "|---|---:|---:|:---:|",
    ]
    for name, item in checks.items():
        check_lines.append(
            f"| `{name}` | {item.get('observed')} | {item.get('required')} | "
            f"{'是' if item.get('passed') else '否'} |"
        )
    candidate = summary.get("candidate_static_motion_threshold")
    candidate_text = "不可用" if candidate is None else f"{float(candidate):.8f}"
    noise = summary.get("same_frame_noise_distribution", {})
    noop = summary.get("noop_full_horizon_distribution", {})
    condition_lines = [
        "| Condition | n | 中位数 | 均值 | 最大值 |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition, distribution in summary.get(
        "noop_full_horizon_by_condition", {}
    ).items():
        condition_lines.append(
            f"| {condition} | {distribution.get('count')} | "
            f"{distribution.get('median')} | {distribution.get('mean')} | "
            f"{distribution.get('maximum')} |"
        )
    sensitivity_path = root / "summary" / "static_threshold_sensitivity.json"
    sensitivity: Mapping[str, Any] | None = None
    if sensitivity_path.is_file():
        try:
            candidate_sensitivity = json.loads(
                sensitivity_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid static threshold sensitivity: {sensitivity_path}"
            ) from exc
        if isinstance(candidate_sensitivity, Mapping):
            sensitivity = candidate_sensitivity
    sensitivity_text = "未运行派生敏感性分析。"
    if isinstance(sensitivity, Mapping):
        all_group = sensitivity.get("groups", {}).get("all", {})
        sensitivity_text = (
            f"只读重分类 {sensitivity.get('classified_rows', 0)} 个 pilot probe："
            f"候选阈值下 predicted-static "
            f"{all_group.get('predicted_static_count', 0)}/{all_group.get('rows', 0)}。"
            "原始 diagnostics JSONL 未改写。"
        )
    report = f"""# Thought 2 静态/无动作校准报告

## 结论状态

- 阈值状态：`{summary.get('threshold_status')}`
- 候选 `static_motion_threshold`：`{candidate_text}`
- 估计式：`{summary.get('candidate_formula')}`
- 有效样本：{summary.get('eligible_sample_count')}/{summary.get('planned_job_count')}
- 全时域：0→{summary.get('full_horizon_offset')} 个控制步

当前结果{"已达到人工冻结的样本门槛，但仍需人工确认" if summary.get("freeze_eligible") else "仅为候选阈值；不得作为冻结后的 paper 结果"}。

## Null 分布

| 分布 | n | 中位数 | 配置分位数 | 最大值 |
|---|---:|---:|---:|---:|
| 同帧重复编码的每样本最大噪声 | {noise.get('count')} | {noise.get('median')} | {noise.get('configured_quantile_value')} | {noise.get('maximum')} |
| 标准 no-op 0→全时域变化 | {noop.get('count')} | {noop.get('median')} | {noop.get('configured_quantile_value')} | {noop.get('maximum')} |

### 按 condition 描述

{chr(10).join(condition_lines)}

## 冻结门槛

{chr(10).join(check_lines)}

## Pilot 敏感性

{sensitivity_text}

## 解释边界

- 阈值只对应“逐帧独立重编码的 first-frame VAE embedding”，不是原生 temporal future latent。
- 阈值估计不读取 future pilot 的成功/失败标签；pilot 只在阈值产生后做派生重分类。
- `candidate_only` 不能回写原始结果，也不能据此声称未来预测静止或动态。
- 即使状态为 `eligible_for_manual_freeze`，也必须先审核样本覆盖、异常与可视化，再形成冻结版本。
"""
    report_path = root / "summary" / "static_calibration_report.md"
    _atomic_text_write(report_path, report)
    return report_path


__all__ = [
    "aggregate_static_calibration",
    "generate_static_calibration_report",
    "higher_quantile",
    "linear_quantile",
    "static_threshold_sensitivity",
]
