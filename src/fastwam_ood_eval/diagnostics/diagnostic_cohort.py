"""Outcome-blind cohort manifests for formal Thought 2 diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from fastwam_ood_eval.evaluation.evaluator import git_commit, git_dirty
from fastwam_ood_eval.evaluation.jobs import EvaluationJob, read_jobs


COHORT_SCHEMA = "thought2-outcome-blind-diagnostic-cohort-v1"
ALLOWED_STRATUM_FIELDS = {
    "suite",
    "task_id",
    "condition",
    "perturbation_category",
    "perturbation_level",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_paths(source_dir: Path) -> tuple[Path, Path]:
    source_dir = Path(source_dir)
    experiment_manifest = source_dir / "experiment_manifest.json"
    job_manifest = source_dir / "job_manifest.jsonl"
    if not experiment_manifest.is_file():
        raise FileNotFoundError(
            f"Source experiment manifest does not exist: {experiment_manifest}"
        )
    if not job_manifest.is_file():
        raise FileNotFoundError(
            f"Source job manifest does not exist: {job_manifest}"
        )
    return experiment_manifest, job_manifest


def _source_experiment(
    experiment_manifest: Path,
) -> dict[str, Any]:
    try:
        payload = json.loads(
            experiment_manifest.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid source experiment manifest: {experiment_manifest}"
        ) from exc
    if not isinstance(payload, dict) or not payload.get("experiment_id"):
        raise RuntimeError(
            f"Source manifest lacks experiment_id: {experiment_manifest}"
        )
    return payload


def _stratum(job: EvaluationJob, fields: Sequence[str]) -> tuple[Any, ...]:
    return tuple(getattr(job, field) for field in fields)


def _stratum_dict(
    fields: Sequence[str],
    values: Sequence[Any],
) -> dict[str, Any]:
    return {
        str(field): value
        for field, value in zip(fields, values)
    }


def _job_selection_key(
    job: EvaluationJob,
    *,
    seed: int,
    source_job_manifest_sha256: str,
) -> str:
    payload = (
        f"{int(seed)}\x1f{source_job_manifest_sha256}\x1f{job.job_id}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _matches_filters(
    job: EvaluationJob,
    *,
    task_ids: set[int],
    categories: set[str],
    levels: set[str],
) -> bool:
    return bool(
        (not task_ids or job.task_id in task_ids)
        and (
            not categories
            or str(job.perturbation_category) in categories
        )
        and (
            not levels
            or str(job.perturbation_level) in levels
        )
    )


def _select(
    jobs: Sequence[EvaluationJob],
    *,
    seed: int,
    per_stratum: int,
    stratum_fields: Sequence[str],
    source_job_manifest_sha256: str,
    task_ids: Sequence[int],
    categories: Sequence[str],
    levels: Sequence[str],
    anchor_episode_indices: Sequence[int],
) -> tuple[
    list[EvaluationJob],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    task_filter = {int(value) for value in task_ids}
    category_filter = {str(value) for value in categories}
    level_filter = {str(value) for value in levels}
    anchors = tuple(sorted({int(value) for value in anchor_episode_indices}))
    if len(anchors) > per_stratum:
        raise ValueError(
            "anchor_episode_indices cannot exceed per_stratum"
        )
    filtered = [
        job
        for job in jobs
        if _matches_filters(
            job,
            task_ids=task_filter,
            categories=category_filter,
            levels=level_filter,
        )
    ]
    runnable_groups: dict[tuple[Any, ...], list[EvaluationJob]] = (
        defaultdict(list)
    )
    skipped_groups: dict[tuple[Any, ...], list[EvaluationJob]] = defaultdict(
        list
    )
    for job in filtered:
        target = skipped_groups if job.skip_reason else runnable_groups
        target[_stratum(job, stratum_fields)].append(job)

    selected: list[EvaluationJob] = []
    strata: list[dict[str, Any]] = []
    for values in sorted(
        runnable_groups,
        key=lambda item: tuple(str(value) for value in item),
    ):
        candidates = sorted(
            runnable_groups[values],
            key=lambda job: (
                _job_selection_key(
                    job,
                    seed=seed,
                    source_job_manifest_sha256=(
                        source_job_manifest_sha256
                    ),
                ),
                job.job_id,
            ),
        )
        anchored: list[EvaluationJob] = []
        missing_anchors: list[int] = []
        for episode_index in anchors:
            matches = [
                job
                for job in candidates
                if job.episode_index == episode_index
            ]
            if matches:
                anchored.append(matches[0])
            else:
                missing_anchors.append(episode_index)
        anchored_ids = {job.job_id for job in anchored}
        remaining = [
            job for job in candidates if job.job_id not in anchored_ids
        ]
        chosen = [
            *anchored,
            *remaining[: max(0, per_stratum - len(anchored))],
        ]
        selected.extend(chosen)
        shortfall = max(
            max(0, per_stratum - len(chosen)),
            len(missing_anchors),
        )
        strata.append(
            {
                "stratum": _stratum_dict(stratum_fields, values),
                "runnable_candidates": len(candidates),
                "skipped_jobs": len(skipped_groups.get(values, [])),
                "skip_reasons": sorted(
                    {
                        str(job.skip_reason)
                        for job in skipped_groups.get(values, [])
                        if job.skip_reason
                    }
                ),
                "selected": len(chosen),
                "required": per_stratum,
                "anchor_episode_indices": list(anchors),
                "selected_anchor_episode_indices": [
                    job.episode_index for job in anchored
                ],
                "missing_anchor_episode_indices": missing_anchors,
                "shortfall": shortfall,
                "selected_job_ids": [job.job_id for job in chosen],
            }
        )
    unsupported = [
        {
            "stratum": _stratum_dict(stratum_fields, values),
            "skipped_jobs": len(group),
            "skip_reasons": sorted(
                {
                    str(job.skip_reason)
                    for job in group
                    if job.skip_reason
                }
            ),
        }
        for values, group in sorted(
            skipped_groups.items(),
            key=lambda item: tuple(str(value) for value in item[0]),
        )
        if values not in runnable_groups
    ]
    return selected, strata, unsupported


def _outcome_files(source_dir: Path) -> list[str]:
    candidates = [
        *source_dir.glob("workers/rank_*/episode_results.jsonl"),
        source_dir / "summary" / "episode_results.jsonl",
    ]
    return [
        str(path)
        for path in candidates
        if path.is_file() and path.stat().st_size > 0
    ]


def plan_diagnostic_cohort(
    *,
    source_dir: Path,
    output_path: Path,
    seed: int,
    per_stratum: int,
    stratum_fields: Sequence[str],
    task_ids: Sequence[int] = (),
    categories: Sequence[str] = (),
    levels: Sequence[str] = (),
    anchor_episode_indices: Sequence[int] = (),
    allow_short_strata: bool = False,
    freeze: bool = False,
) -> dict[str, Any]:
    """Select jobs using manifest metadata only and write a fresh cohort."""

    source_dir = Path(source_dir)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(
            f"Diagnostic cohort already exists; use a fresh path: {output_path}"
        )
    if source_dir.resolve() in output_path.resolve().parents:
        raise ValueError(
            "Diagnostic cohort output must be outside the source experiment"
        )
    if per_stratum <= 0:
        raise ValueError("per_stratum must be positive")
    fields = tuple(str(value) for value in stratum_fields)
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("stratum_fields must be non-empty and unique")
    unknown = set(fields) - ALLOWED_STRATUM_FIELDS
    if unknown:
        raise ValueError(f"Unsupported stratum fields: {sorted(unknown)}")

    experiment_path, job_path = _source_paths(source_dir)
    source_manifest = _source_experiment(experiment_path)
    jobs = read_jobs(job_path)
    job_hash = _sha256(job_path)
    selected, strata, unsupported = _select(
        jobs,
        seed=seed,
        per_stratum=per_stratum,
        stratum_fields=fields,
        source_job_manifest_sha256=job_hash,
        task_ids=task_ids,
        categories=categories,
        levels=levels,
        anchor_episode_indices=anchor_episode_indices,
    )
    if not selected:
        raise RuntimeError("Diagnostic cohort selection produced zero jobs")
    shortfalls = [
        row for row in strata if int(row["shortfall"]) > 0
    ]
    if shortfalls and not allow_short_strata:
        raise RuntimeError(
            "Diagnostic cohort has undersized runnable strata; adjust the "
            "design or explicitly allow a draft with short strata"
        )

    outcome_files = _outcome_files(source_dir)
    project_dirty = git_dirty(Path.cwd())
    if freeze:
        errors: list[str] = []
        if shortfalls:
            errors.append("one or more supported strata have a shortfall")
        if outcome_files:
            errors.append(
                "source outcome JSONL already exists, so pre-outcome freezing "
                "cannot be certified"
            )
        if project_dirty is not False:
            errors.append(
                "planner project tree is not explicitly clean"
            )
        if errors:
            raise RuntimeError(
                "Cannot freeze outcome-blind diagnostic cohort:\n- "
                + "\n- ".join(errors)
            )

    selection_spec = {
        "seed": int(seed),
        "per_stratum": int(per_stratum),
        "stratum_fields": list(fields),
        "filters": {
            "task_ids": sorted({int(value) for value in task_ids}),
            "categories": sorted({str(value) for value in categories}),
            "levels": sorted({str(value) for value in levels}),
        },
        "anchor_episode_indices": sorted(
            {int(value) for value in anchor_episode_indices}
        ),
        "allow_short_strata": bool(allow_short_strata),
    }
    identity_payload = {
        "schema": COHORT_SCHEMA,
        "source_experiment_id": source_manifest["experiment_id"],
        "source_job_manifest_sha256": job_hash,
        "selection_spec": selection_spec,
        "selected_job_ids": [job.job_id for job in selected],
    }
    cohort_id = _canonical_sha256(identity_payload)
    payload = {
        "schema_version": 1,
        "kind": "outcome_blind_diagnostic_cohort",
        "schema": COHORT_SCHEMA,
        "cohort_id": cohort_id,
        "status": (
            "frozen_before_source_outcomes"
            if freeze
            else "draft_not_frozen"
        ),
        "frozen": bool(freeze),
        "source": {
            "directory": str(source_dir),
            "experiment_id": source_manifest["experiment_id"],
            "experiment_manifest_path": str(experiment_path),
            "experiment_manifest_sha256_at_selection": _sha256(
                experiment_path
            ),
            "job_manifest_path": str(job_path),
            "job_manifest_sha256": job_hash,
            "outcome_files_present_at_selection": outcome_files,
        },
        "selection": {
            **selection_spec,
            "input_job_count": len(jobs),
            "runnable_strata": len(strata),
            "selected_job_count": len(selected),
            "short_strata": len(shortfalls),
            "unsupported_skipped_only_strata": len(unsupported),
            "outcome_fields_read": False,
            "episode_result_files_read": False,
            "selection_variables": [
                "source job_id",
                *fields,
                "configured task/category/level filters",
                "episode_index anchors",
                "skip_reason eligibility",
            ],
        },
        "strata": strata,
        "unsupported_strata": unsupported,
        "selected_jobs": [
            {
                "selection_order": index,
                "job_id": job.job_id,
                "selection_key": _job_selection_key(
                    job,
                    seed=seed,
                    source_job_manifest_sha256=job_hash,
                ),
                "stratum": _stratum_dict(
                    fields, _stratum(job, fields)
                ),
            }
            for index, job in enumerate(selected)
        ],
        "planner_provenance": {
            "git_commit": git_commit(Path.cwd()),
            "git_dirty": project_dirty,
            "implementation_sha256": _sha256(Path(__file__)),
        },
        "interpretation": (
            "Selection is independent of source episode outcomes. A draft must "
            "not be cited as pre-registered until regenerated with frozen=true."
        ),
    }
    _atomic_json(output_path, payload)
    return validate_diagnostic_cohort(output_path, source_dir)


def validate_diagnostic_cohort(
    manifest_path: Path,
    source_dir: Path,
) -> dict[str, Any]:
    """Verify source hash, deterministic selection, and exact job identities."""

    manifest_path = Path(manifest_path)
    source_dir = Path(source_dir)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Diagnostic cohort manifest does not exist: {manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid diagnostic cohort manifest: {manifest_path}"
        ) from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("kind") != "outcome_blind_diagnostic_cohort"
        or payload.get("schema") != COHORT_SCHEMA
    ):
        raise RuntimeError(
            f"Invalid diagnostic cohort identity: {manifest_path}"
        )
    experiment_path, job_path = _source_paths(source_dir)
    source_manifest = _source_experiment(experiment_path)
    source = payload.get("source")
    selection = payload.get("selection")
    if not isinstance(source, Mapping) or not isinstance(
        selection, Mapping
    ):
        raise RuntimeError("Diagnostic cohort lacks source/selection metadata")
    current_job_hash = _sha256(job_path)
    if (
        source.get("job_manifest_sha256") != current_job_hash
        or source.get("experiment_id") != source_manifest.get("experiment_id")
    ):
        raise RuntimeError(
            "Diagnostic cohort no longer matches the source job manifest"
        )
    if (
        selection.get("outcome_fields_read") is not False
        or selection.get("episode_result_files_read") is not False
    ):
        raise RuntimeError(
            "Diagnostic cohort does not certify outcome-blind selection"
        )
    filters = selection.get("filters")
    if not isinstance(filters, Mapping):
        raise RuntimeError("Diagnostic cohort filters are invalid")
    fields = tuple(str(value) for value in selection.get("stratum_fields", ()))
    if (
        not fields
        or len(set(fields)) != len(fields)
        or set(fields) - ALLOWED_STRATUM_FIELDS
    ):
        raise RuntimeError("Diagnostic cohort stratum fields are invalid")
    canonical_filters = {
        "task_ids": sorted(
            {int(value) for value in filters.get("task_ids", ())}
        ),
        "categories": sorted(
            {str(value) for value in filters.get("categories", ())}
        ),
        "levels": sorted(
            {str(value) for value in filters.get("levels", ())}
        ),
    }
    if {
        "task_ids": list(filters.get("task_ids", ())),
        "categories": list(filters.get("categories", ())),
        "levels": list(filters.get("levels", ())),
    } != canonical_filters:
        raise RuntimeError(
            "Diagnostic cohort filters must be sorted and unique"
        )
    anchors = sorted(
        {
            int(value)
            for value in selection.get("anchor_episode_indices", ())
        }
    )
    if list(selection.get("anchor_episode_indices", ())) != anchors:
        raise RuntimeError(
            "Diagnostic cohort anchors must be sorted and unique"
        )
    jobs = read_jobs(job_path)
    selected, strata, unsupported = _select(
        jobs,
        seed=int(selection["seed"]),
        per_stratum=int(selection["per_stratum"]),
        stratum_fields=fields,
        source_job_manifest_sha256=current_job_hash,
        task_ids=tuple(canonical_filters["task_ids"]),
        categories=tuple(canonical_filters["categories"]),
        levels=tuple(canonical_filters["levels"]),
        anchor_episode_indices=tuple(anchors),
    )
    expected_ids = [job.job_id for job in selected]
    recorded_jobs = payload.get("selected_jobs")
    if not isinstance(recorded_jobs, list):
        raise RuntimeError("Diagnostic cohort selected_jobs must be a list")
    recorded_ids = [str(row.get("job_id")) for row in recorded_jobs]
    if recorded_ids != expected_ids or len(set(recorded_ids)) != len(
        recorded_ids
    ):
        raise RuntimeError(
            "Diagnostic cohort selection does not reproduce exactly"
        )
    expected_rows = [
        {
            "selection_order": index,
            "job_id": job.job_id,
            "selection_key": _job_selection_key(
                job,
                seed=int(selection["seed"]),
                source_job_manifest_sha256=current_job_hash,
            ),
            "stratum": _stratum_dict(fields, _stratum(job, fields)),
        }
        for index, job in enumerate(selected)
    ]
    if recorded_jobs != expected_rows:
        raise RuntimeError(
            "Diagnostic cohort job ordering/key audit has changed"
        )
    if payload.get("strata") != strata or payload.get(
        "unsupported_strata"
    ) != unsupported:
        raise RuntimeError("Diagnostic cohort stratum audit has changed")
    expected_summary = {
        "input_job_count": len(jobs),
        "runnable_strata": len(strata),
        "selected_job_count": len(selected),
        "short_strata": sum(
            int(row.get("shortfall", 0)) > 0 for row in strata
        ),
        "unsupported_skipped_only_strata": len(unsupported),
    }
    for key, expected in expected_summary.items():
        if selection.get(key) != expected:
            raise RuntimeError(
                f"Diagnostic cohort selection summary is invalid: {key}"
            )
    if payload.get("frozen") is True:
        if (
            payload.get("status") != "frozen_before_source_outcomes"
            or source.get("outcome_files_present_at_selection") != []
            or payload.get("planner_provenance", {}).get("git_dirty")
            is not False
        ):
            raise RuntimeError("Frozen diagnostic cohort lacks freeze evidence")
    elif (
        payload.get("frozen") is not False
        or payload.get("status") != "draft_not_frozen"
    ):
        raise RuntimeError("Draft diagnostic cohort status is invalid")
    identity_payload = {
        "schema": COHORT_SCHEMA,
        "source_experiment_id": source_manifest["experiment_id"],
        "source_job_manifest_sha256": current_job_hash,
        "selection_spec": {
            "seed": int(selection["seed"]),
            "per_stratum": int(selection["per_stratum"]),
            "stratum_fields": list(fields),
            "filters": canonical_filters,
            "anchor_episode_indices": anchors,
            "allow_short_strata": bool(
                selection.get("allow_short_strata", False)
            ),
        },
        "selected_job_ids": expected_ids,
    }
    if payload.get("cohort_id") != _canonical_sha256(identity_payload):
        raise RuntimeError("Diagnostic cohort identity hash is invalid")
    return {
        "cohort_id": payload.get("cohort_id"),
        "status": payload.get("status"),
        "frozen": payload.get("frozen"),
        "selected_jobs": len(recorded_ids),
        "runnable_strata": len(strata),
        "short_strata": sum(
            int(row.get("shortfall", 0)) > 0 for row in strata
        ),
        "unsupported_strata": len(unsupported),
        "selected_job_ids": recorded_ids,
    }


__all__ = [
    "ALLOWED_STRATUM_FIELDS",
    "COHORT_SCHEMA",
    "plan_diagnostic_cohort",
    "validate_diagnostic_cohort",
]
