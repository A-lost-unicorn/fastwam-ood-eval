from __future__ import annotations

import json

import pytest

from fastwam_ood_eval.analysis.aggregate import aggregate_experiment, summarize_rows
from fastwam_ood_eval.analysis.confidence_intervals import bootstrap_mean_ci
from fastwam_ood_eval.analysis.robustness_metrics import absolute_drop, paired_outcomes, relative_drop


def _row(job_id, condition, success, seed=1, category=None, termination="success"):
    return {
        "job_id": job_id,
        "suite": "s",
        "task_id": 0,
        "task_name": "t",
        "episode_index": seed,
        "episode_seed": seed,
        "condition": condition,
        "perturbation_category": category,
        "perturbation_level": "easy" if category else None,
        "success": success,
        "steps": 3,
        "termination_reason": termination,
        "policy_latency_mean_ms": 2.0,
        "gpu_peak_memory_mb": 10.0,
    }


def test_success_rate_and_drops():
    summary = summarize_rows([_row("a", "clean", True), _row("b", "clean", False)])
    assert summary["success_rate"] == 0.5
    assert absolute_drop(0.8, 0.5) == pytest.approx(0.3)
    assert relative_drop(0.8, 0.4) == 0.5
    assert relative_drop(0.0, 0.0) == 0.0


def test_confidence_interval_range_and_empty():
    low, high = bootstrap_mean_ci([0.0, 1.0, 1.0, 0.0], samples=200)
    assert 0.0 <= low <= high <= 1.0
    assert bootstrap_mean_ci([]) == (None, None)


def test_paired_counts():
    rows = [
        _row("c1", "clean", True, seed=1),
        _row("o1", "ood", False, seed=1, category="camera"),
        _row("c2", "clean", False, seed=2),
        _row("o2", "ood", True, seed=2, category="camera"),
    ]
    result = paired_outcomes(rows)
    assert result["clean_success_ood_failure"] == 1
    assert result["clean_failure_ood_success"] == 1


def test_aggregate_writes_required_outputs(tmp_path):
    worker = tmp_path / "workers" / "rank_0"
    worker.mkdir(parents=True)
    rows = [_row("a", "clean", True), _row("b", "ood", False, category="camera")]
    (worker / "episode_results.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    metrics = aggregate_experiment(tmp_path)
    assert metrics["clean"]["success_rate"] == 1.0
    assert (tmp_path / "summary" / "summary_by_perturbation.csv").is_file()
    assert (tmp_path / "summary" / "episode_results.jsonl").is_file()


def test_empty_aggregation_is_valid(tmp_path):
    metrics = aggregate_experiment(tmp_path)
    assert metrics["all"]["episodes"] == 0
    assert metrics["all"]["success_rate"] is None


def test_mismatched_checkpoint_hashes_are_rejected(tmp_path):
    worker = tmp_path / "workers" / "rank_0"
    worker.mkdir(parents=True)
    clean = _row("a", "clean", True)
    clean["checkpoint_hash"] = "clean-hash"
    ood = _row("b", "ood", False, category="camera")
    ood["checkpoint_hash"] = "ood-hash"
    (worker / "episode_results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in (clean, ood)) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="checkpoint hashes differ"):
        aggregate_experiment(tmp_path)
