"""Result recovery and resume filtering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from fastwam_ood_eval.evaluation.jobs import EvaluationJob


def load_result_records(paths: Iterable[Path]) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A crash can leave one partial final line; earlier durable records remain usable.
                    continue
                job_id = record.get("job_id")
                if job_id:
                    records[str(job_id)] = record
    return records


def filter_jobs_for_resume(
    jobs: Iterable[EvaluationJob],
    records: dict[str, dict],
    *,
    overwrite: bool = False,
    rerun: str = "incomplete",
) -> list[EvaluationJob]:
    if overwrite:
        return list(jobs)
    if rerun not in {"incomplete", "failed", "all"}:
        raise ValueError("rerun must be incomplete, failed, or all")
    selected: list[EvaluationJob] = []
    for job in jobs:
        previous = records.get(job.job_id)
        if previous is None:
            selected.append(job)
            continue
        if rerun == "all":
            selected.append(job)
        elif rerun == "failed" and previous.get("termination_reason") in {"exception", "max_steps"}:
            selected.append(job)
    return selected

