from __future__ import annotations

from fastwam_ood_eval.thought3.cache_planner import create_cache_plan
from fastwam_ood_eval.thought3.config import load_thought3_config
from fastwam_ood_eval.thought3.schemas import (
    cache_sample_id,
    derive_initial_noise_seed,
    validate_paired_cache_entries,
)
from thought3_test_utils import write_thought3_config


def test_cache_sample_identity_is_stable_and_k_specific(tmp_path):
    cfg = load_thought3_config(write_thought3_config(tmp_path))
    entries, split, manifest = create_cache_plan(cfg)
    entries_again, split_again, manifest_again = create_cache_plan(cfg)
    assert [entry.to_dict() for entry in entries] == [
        entry.to_dict() for entry in entries_again
    ]
    assert split.fingerprint == split_again.fingerprint
    assert manifest["cache_fingerprint"] == manifest_again["cache_fingerprint"]
    by_base = {}
    for entry in entries:
        by_base.setdefault(entry.identity.base_sample_id, []).append(entry)
        assert entry.initial_noise_seed == derive_initial_noise_seed(
            entry.identity.base_sample_id,
            cfg.sampler.global_cache_seed,
        )
        assert entry.cache_sample_id == cache_sample_id(
            base_sample_id=entry.identity.base_sample_id,
            k=entry.k,
            initial_noise_seed=entry.initial_noise_seed,
        )
    assert all({value.k for value in values} == {1, 2, 4} for values in by_base.values())
    assert all(
        len({value.initial_noise_seed for value in values}) == 1
        for values in by_base.values()
    )
    validate_paired_cache_entries(
        [
            {
                "base_sample_id": entry.identity.base_sample_id,
                "initial_noise_seed": entry.initial_noise_seed,
                "k": entry.k,
            }
            for entry in entries
        ]
    )


def test_episode_split_has_no_overlap_and_is_stratified(tmp_path):
    cfg = load_thought3_config(
        write_thought3_config(tmp_path, sample_count=10)
    )
    _, split, _ = create_cache_plan(cfg)
    assert not set(split.train_episode_ids) & set(split.development_episode_ids)
    assert set(split.strata) == {"libero_mock/task_0", "libero_mock/task_1"}
    assert all(values["train"] and values["development"] for values in split.strata.values())
