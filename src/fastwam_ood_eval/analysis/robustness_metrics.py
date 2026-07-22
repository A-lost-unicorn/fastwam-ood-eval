"""Robustness drop and paired-outcome metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations
from typing import Any

from fastwam_ood_eval.analysis.confidence_intervals import bootstrap_mean_ci


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


def exact_mcnemar_p_value(discordant_a: int, discordant_b: int) -> float | None:
    """Two-sided exact McNemar p-value for paired binary outcomes."""
    if discordant_a < 0 or discordant_b < 0:
        raise ValueError("discordant counts must be non-negative")
    total = discordant_a + discordant_b
    if total == 0:
        return None
    lower = min(discordant_a, discordant_b)
    tail = sum(math.comb(total, k) for k in range(lower + 1)) / (2**total)
    return min(1.0, 2.0 * tail)


def _policy_pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Match policies on exactly the same benchmark episode and perturbation."""
    return (
        str(row.get("suite")),
        str(row.get("task_name")),
        int(row.get("episode_seed", 0)),
        str(row.get("condition")),
        row.get("perturbation_category"),
        row.get("perturbation_level"),
        (row.get("perturbation_parameters") or {}).get("classification_id"),
    )


def paired_policy_comparisons(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare future/no-future policies only inside explicit comparison groups.

    The statistic is paired by suite/task/seed/condition/official Plus variant.
    A causal interpretation is permitted only when both configurations declare
    the same non-empty training recipe ID; otherwise the result is labelled as
    an observational comparison between different checkpoints.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        group = row.get("comparison_group")
        if group and _attempted_for_pairing(row):
            grouped[str(group)].append(row)

    comparisons: list[dict[str, Any]] = []
    for group, rows in sorted(grouped.items()):
        variants = sorted({str(row.get("policy_variant", "unspecified")) for row in rows})
        for variant_a, variant_b in combinations(variants, 2):
            rows_a = [row for row in rows if str(row.get("policy_variant")) == variant_a]
            rows_b = [row for row in rows if str(row.get("policy_variant")) == variant_b]
            future_a = {bool(row.get("test_time_future_imagination", False)) for row in rows_a}
            future_b = {bool(row.get("test_time_future_imagination", False)) for row in rows_b}
            if len(future_a) != 1 or len(future_b) != 1 or future_a == future_b:
                continue
            no_future_variant, future_variant = (
                (variant_a, variant_b) if not next(iter(future_a)) else (variant_b, variant_a)
            )
            by_variant = {
                variant_a: {_policy_pair_key(row): row for row in rows_a},
                variant_b: {_policy_pair_key(row): row for row in rows_b},
            }
            shared_keys = sorted(
                set(by_variant[no_future_variant]) & set(by_variant[future_variant]),
                key=str,
            )
            differences: list[float] = []
            future_wins = no_future_wins = both_success = both_failure = 0
            for key in shared_keys:
                no_future_ok = bool(by_variant[no_future_variant][key].get("success"))
                future_ok = bool(by_variant[future_variant][key].get("success"))
                differences.append(float(future_ok) - float(no_future_ok))
                if future_ok and not no_future_ok:
                    future_wins += 1
                elif no_future_ok and not future_ok:
                    no_future_wins += 1
                elif future_ok:
                    both_success += 1
                else:
                    both_failure += 1
            ci_low, ci_high = bootstrap_mean_ci(differences)
            recipe_a = {row.get("training_recipe_id") for row in rows_a if row.get("training_recipe_id")}
            recipe_b = {row.get("training_recipe_id") for row in rows_b if row.get("training_recipe_id")}
            matched_recipe = len(recipe_a) == 1 and recipe_a == recipe_b
            comparisons.append(
                {
                    "comparison_group": group,
                    "no_future_variant": no_future_variant,
                    "future_variant": future_variant,
                    "paired_episodes": len(shared_keys),
                    "future_success_no_future_failure": future_wins,
                    "future_failure_no_future_success": no_future_wins,
                    "both_success": both_success,
                    "both_failure": both_failure,
                    "paired_success_rate_difference": (
                        sum(differences) / len(differences) if differences else None
                    ),
                    "paired_difference_ci95_low": ci_low,
                    "paired_difference_ci95_high": ci_high,
                    "mcnemar_exact_p_value": exact_mcnemar_p_value(future_wins, no_future_wins),
                    "training_recipe_id": next(iter(recipe_a)) if matched_recipe else None,
                    "causal_interpretation_allowed": matched_recipe,
                    "interpretation": (
                        "paired architecture ablation with declared matched training recipe"
                        if matched_recipe
                        else "associational only: training recipe parity is not established"
                    ),
                }
            )
    return comparisons


def _attempted_for_pairing(row: dict[str, Any]) -> bool:
    return row.get("termination_reason") != "skipped"
