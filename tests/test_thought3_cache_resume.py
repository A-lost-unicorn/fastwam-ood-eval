from __future__ import annotations

from fastwam_ood_eval.thought3.cache_builder import build_cache
from fastwam_ood_eval.thought3.cache_planner import write_cache_plan
from fastwam_ood_eval.thought3.config import load_thought3_config
from thought3_test_utils import write_thought3_config


def test_cache_resume_skips_every_valid_shard(tmp_path):
    cfg = load_thought3_config(write_thought3_config(tmp_path))
    plan = write_cache_plan(cfg)
    first = build_cache(cfg, resume=False)
    second = build_cache(cfg, resume=True)
    expected_shards = sum(plan["shards_per_k"].values())
    assert first["built_shards"] == expected_shards
    assert first["complete"]
    assert second["built_shards"] == 0
    assert second["skipped_valid_shards"] == expected_shards
