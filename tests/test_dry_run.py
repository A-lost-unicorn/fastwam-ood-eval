from __future__ import annotations

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


def test_mock_single_and_four_workers(tmp_path):
    cfg = load_config(write_config(tmp_path, perturbation=True, episodes=3))
    total = 0
    for rank in range(4):
        total += evaluator.evaluate_worker(cfg, rank=rank, world_size=4)["completed"]
    assert total > 0
    records = list((cfg.experiment.output_dir / "workers").glob("rank_*/episode_results.jsonl"))
    assert len(records) == 4

