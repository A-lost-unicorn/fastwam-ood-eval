from __future__ import annotations

import pytest

from conftest import write_config
from fastwam_ood_eval.config import load_config
from fastwam_ood_eval.evaluation import evaluator


def test_dry_run_does_not_create_model_or_environment(tmp_path, monkeypatch):
    cfg = load_config(write_config(tmp_path, episodes=2))

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run loaded runtime object")

    monkeypatch.setattr(evaluator, "_make_environment", forbidden)
    monkeypatch.setattr(evaluator, "_make_policy", forbidden)
    result = evaluator.evaluate_worker(cfg, dry_run=True)
    assert result["completed"] == 0
    assert result["pending"] > 0


def test_mock_single_and_three_workers(tmp_path):
    cfg = load_config(write_config(tmp_path, perturbation=True, episodes=3))
    evaluator.plan_experiment(cfg)
    total = 0
    for rank in range(3):
        total += evaluator.evaluate_worker(cfg, rank=rank, world_size=3)["completed"]
    assert total > 0
    records = list((cfg.experiment.output_dir / "workers").glob("rank_*/episode_results.jsonl"))
    assert len(records) == 3


def test_distributed_requires_preplanned_manifest(tmp_path):
    cfg = load_config(write_config(tmp_path, episodes=2))
    with pytest.raises(RuntimeError, match="requires a precomputed manifest"):
        evaluator.evaluate_worker(cfg, rank=0, world_size=3, dry_run=True)
