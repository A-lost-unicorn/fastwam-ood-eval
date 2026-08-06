from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from fastwam_ood_eval.thought6.cli import _phase6c_preflight
from fastwam_ood_eval.thought6.gate_decision import MetricInterval, decide_phase6b, decide_stage2
from fastwam_ood_eval.thought6.offline_utility import sigma_bucket
from fastwam_ood_eval.thought6.paired_noise import (
    build_noise_pairing_manifest,
    offline_noise_identity,
    rollout_noise_identity,
    validate_camera_only_pair,
    validate_arm_pairing,
)
from fastwam_ood_eval.thought6.schemas import Thought6Error
from fastwam_ood_eval.thought6.statistics import hierarchical_bootstrap


def test_21_offline_seed_identity_is_reproducible() -> None:
    args = dict(suite="libero_goal", task_id=1, episode_id="episode_1", flow_slot=3, seed=6607)
    assert offline_noise_identity(**args) == offline_noise_identity(**args)


def test_22_flow_slot_changes_objective_seed_not_denoising_contract() -> None:
    a = offline_noise_identity(suite="s", task_id=1, episode_id="e", flow_slot=1, seed=2)
    b = offline_noise_identity(suite="s", task_id=1, episode_id="e", flow_slot=2, seed=2)
    assert a.action_timestep_seed != b.action_timestep_seed and a.flow_slot == 1


def test_23_correct_null_shuffle_action_noise_must_match() -> None:
    base = {"row_id": "r", "action_noise_seed": 1, "action_timestep_seed": 2, "future_noise_seed": 3, "initial_state_sha256": "a", "scheduler_sha256": "b"}
    validate_arm_pairing([{**base, "arm": arm} for arm in ("correct", "null", "shuffle")])


def test_24_pairing_rejects_changed_future_noise() -> None:
    base = {"row_id": "r", "action_noise_seed": 1, "action_timestep_seed": 2, "initial_state_sha256": "a", "scheduler_sha256": "b"}
    with pytest.raises(Thought6Error):
        validate_arm_pairing([{**base, "future_noise_seed": 3}, {**base, "future_noise_seed": 4}])


def test_25_rollout_clean_camera_seed_identity_can_be_shared() -> None:
    row = rollout_noise_identity(stage=1, suite="libero_goal", task_id=1, initial_state_index=0, seed=6607)
    manifest = build_noise_pairing_manifest([], [row])
    assert manifest["clean_camera_share_rollout_pair_identity"] is True


def test_26_stage2_uses_new_initial_state_seed() -> None:
    a = rollout_noise_identity(stage=1, suite="s", task_id=1, initial_state_index=9, seed=1)
    b = rollout_noise_identity(stage=2, suite="s", task_id=1, initial_state_index=10, seed=1)
    assert a.pair_id != b.pair_id and a.environment_seed != b.environment_seed


def test_27_sigma_buckets_are_frozen() -> None:
    assert [sigma_bucket(v) for v in (0, .25, .5, .75, 1)] == ["[0,0.25)", "[0.25,0.50)", "[0.50,0.75)", "[0.75,1.00]", "[0.75,1.00]"]


def test_28_hierarchical_bootstrap_is_reproducible() -> None:
    values = {"t0": {"e0": [1, 2], "e1": [3]}, "t1": {"e2": [4], "e3": [5, 6]}}
    assert hierarchical_bootstrap(values, replicates=100, seed=6607) == hierarchical_bootstrap(values, replicates=100, seed=6607)


def test_29_phase6b_all_gates_unlock() -> None:
    good = MetricInterval(.01, .001, .02)
    result = decide_phase6b({"fsigma_clean_utility": good, "fsigma_camera_utility": good, "fsigma_shuffle_specificity": good, "fsigma_minus_f0_utility": good, "null_b0_bitwise_parity": True})
    assert result["phase6c_unlocked"] is True


def test_30_phase6b_one_gate_failure_stops() -> None:
    good = MetricInterval(.01, .001, .02)
    bad = MetricInterval(-.01, -.02, 0)
    result = decide_phase6b({"fsigma_clean_utility": good, "fsigma_camera_utility": bad, "fsigma_shuffle_specificity": good, "fsigma_minus_f0_utility": good, "null_b0_bitwise_parity": True})
    assert result["current_recipe"] == "stopped"


def test_31_stage2_only_unlocks_inconclusive_positive() -> None:
    assert decide_stage2(camera_difference=MetricInterval(.02, -.01, .05), clean_noninferiority_passed=True, fsigma_better_f0_direction=True)["stage2_unlocked"]
    assert not decide_stage2(camera_difference=MetricInterval(0, -.01, .02), clean_noninferiority_passed=True, fsigma_better_f0_direction=True)["stage2_unlocked"]


def test_32_phase6c_launcher_refuses_unpassed_gate(tmp_path) -> None:
    (tmp_path / "phase6b_gate_decision.json").write_text(json.dumps({"status": "complete", "phase6c_unlocked": False}))
    with pytest.raises(Thought6Error):
        _phase6c_preflight(SimpleNamespace(output_dir=tmp_path), stage=1)


def test_32b_camera_condition_changes_only_camera() -> None:
    validate_camera_only_pair(
        clean_physical_state_sha256="physical",
        camera_physical_state_sha256="physical",
        clean_camera_sha256="camera_clean",
        camera_camera_sha256="camera_shifted",
    )


def test_32c_camera_pair_rejects_physics_change() -> None:
    with pytest.raises(Thought6Error):
        validate_camera_only_pair(
            clean_physical_state_sha256="a",
            camera_physical_state_sha256="b",
            clean_camera_sha256="c",
            camera_camera_sha256="d",
        )
