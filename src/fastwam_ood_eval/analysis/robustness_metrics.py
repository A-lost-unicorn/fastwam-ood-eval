"""Robustness drop and paired-outcome metrics."""

from __future__ import annotations

from typing import Any


def absolute_drop(clean_success_rate: float | None, ood_success_rate: float | None) -> float | None:
    if clean_success_rate is None or ood_success_rate is None:
        return None
    return clean_success_rate - ood_success_rate


def relative_drop(
    clean_success_rate: float | None,
    ood_success_rate: float | None,
    *,
    epsilon: float = 1e-12,
) -> float | None:
    if clean_success_rate is None or ood_success_rate is None:
        return None
    return (clean_success_rate - ood_success_rate) / max(clean_success_rate, epsilon)


def paired_outcomes(records: list[dict[str, Any]]) -> dict[str, int]:
    clean: dict[tuple[str, str, int], bool] = {}
    for row in records:
        if row.get("condition") == "clean" and row.get("termination_reason") != "skipped":
            clean[(str(row.get("suite")), str(row.get("task_name")), int(row.get("episode_seed", 0)))] = bool(
                row.get("success")
            )
    counts = {
        "clean_success_ood_failure": 0,
        "clean_failure_ood_success": 0,
        "both_success": 0,
        "both_failure": 0,
        "paired_comparisons": 0,
    }
    for row in records:
        if row.get("condition") != "ood" or row.get("termination_reason") == "skipped":
            continue
        key = (str(row.get("suite")), str(row.get("task_name")), int(row.get("episode_seed", 0)))
        if key not in clean:
            continue
        clean_ok, ood_ok = clean[key], bool(row.get("success"))
        counts["paired_comparisons"] += 1
        if clean_ok and not ood_ok:
            counts["clean_success_ood_failure"] += 1
        elif not clean_ok and ood_ok:
            counts["clean_failure_ood_success"] += 1
        elif clean_ok and ood_ok:
            counts["both_success"] += 1
        else:
            counts["both_failure"] += 1
    return counts

