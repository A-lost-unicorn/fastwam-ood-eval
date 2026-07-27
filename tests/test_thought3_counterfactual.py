from __future__ import annotations

import torch

from fastwam_ood_eval.thought3.cache_builder import build_cache
from fastwam_ood_eval.thought3.cache_planner import write_cache_plan
from fastwam_ood_eval.thought3.config import load_thought3_config
from fastwam_ood_eval.thought3.counterfactuals import (
    ShuffleCandidate,
    build_shuffle_pairs,
    run_action_counterfactuals,
    validate_shuffle_pairs,
)
from fastwam_ood_eval.thought3.future_cache import FutureCacheReader
from thought3_test_utils import write_thought3_config


def test_shuffle_never_uses_same_episode_or_task(tmp_path):
    cfg = load_thought3_config(
        write_thought3_config(tmp_path, sample_count=8)
    )
    plan = write_cache_plan(cfg)
    build_cache(cfg, resume=False)
    reader = FutureCacheReader(
        cfg.cache.root,
        expected_cache_fingerprint=plan["cache_fingerprint"],
    )
    candidates = [
        ShuffleCandidate.from_cache_metadata(reader.metadata(base_id, k))
        for base_id, k in reader.keys
    ]
    pairs = build_shuffle_pairs(candidates, seed=77)
    validate_shuffle_pairs(pairs)
    assert all(pair.recipient_base_sample_id != pair.donor_base_sample_id for pair in pairs)
    assert all(pair.recipient_episode_id != pair.donor_episode_id for pair in pairs)
    assert all(pair.recipient_task_id != pair.donor_task_id for pair in pairs)
    assert pairs == build_shuffle_pairs(candidates, seed=77)


def test_fixed_action_seed_isolates_future_intervention():
    correct = torch.full((1, 4, 2, 3, 3), 0.5)
    shuffled = torch.full_like(correct, -0.25)

    def action_function(future, seed):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        noise = torch.randn((1, 5, 3), generator=generator)
        return noise + future.mean() * torch.ones_like(noise)

    first = run_action_counterfactuals(
        action_function,
        correct_future=correct,
        shuffled_future=shuffled,
        action_seed=123,
    )
    second = run_action_counterfactuals(
        action_function,
        correct_future=correct,
        shuffled_future=shuffled,
        action_seed=123,
    )
    assert first == second
    assert first["interventions"]["shuffle"]["action_l1"] > 0
    assert first["interventions"]["null"]["action_l1"] > 0
