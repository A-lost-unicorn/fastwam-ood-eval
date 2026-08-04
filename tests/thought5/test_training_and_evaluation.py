from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from fastwam_ood_eval.thought5.future_utility_eval import (
    FutureUtilityRecord,
    evaluate_future_utility,
    validate_matched_counterfactuals,
)
from fastwam_ood_eval.thought5.future_runtime import (
    FutureProbeEntry,
    _donor_lookup,
    _fit_future_probe,
    _future_probe_records,
    _technical_action_sensitivity,
)
from fastwam_ood_eval.thought5.rollout_runtime import (
    RolloutRuntimeError,
    _validate_initial_state_pairing,
)
from fastwam_ood_eval.thought5.representation_eval import (
    RepresentationRecord,
    evaluate_h1,
)
from fastwam_ood_eval.thought5.rollout_eval import RolloutRecord, evaluate_rollouts
from fastwam_ood_eval.thought5.geo_targets import assert_no_future_crossing
from fastwam_ood_eval.thought5.losses import LossWeights, total_loss
from fastwam_ood_eval.thought5.statistics import grouped_bootstrap_mean


def utility_panel():
    rows = []
    for backbone, losses in {
        "B1": {"A0": 1.0, "A1": 1.05, "AS": 1.06},
        "G3": {"A0": 1.0, "A1": 0.8, "AS": 1.1},
    }.items():
        for task in ("8", "9"):
            for variant, loss in losses.items():
                rows.append(
                    FutureUtilityRecord(
                        backbone=backbone,
                        adapter_variant=variant,
                        task_id=task,
                        episode_id=f"{task}-e",
                        condition="camera",
                        flow_slot=1,
                        action_noise_seed=10,
                        action_timestep_seed=11,
                        denoise_schedule_sha256="schedule",
                        loss=loss,
                        action_sha256=f"{variant}-{task}",
                        action_rms={"A0": 1.0, "A1": 1.1, "AS": 1.2}[variant],
                    )
                )
    return rows


def test_lambda_zero_auxiliary_has_no_gradient() -> None:
    original_parameter = torch.tensor(2.0, requires_grad=True)
    auxiliary_parameter = torch.tensor(3.0, requires_grad=True)
    original = original_parameter.square()
    auxiliary = auxiliary_parameter.square()
    total, components = total_loss(
        original_fastwam_loss=original,
        weights=LossWeights(0, 0, 0),
        repa=auxiliary,
        equiv=auxiliary,
        pose_aux=auxiliary,
    )
    total.backward()
    assert original_parameter.grad is not None
    assert auxiliary_parameter.grad is None
    assert components["geo_repa"].item() == 0


def test_a0_a1_as_share_noise_timestep_and_schedule() -> None:
    validate_matched_counterfactuals(utility_panel())


def test_counterfactual_seed_mismatch_is_rejected() -> None:
    rows = utility_panel()
    rows[1] = replace(rows[1], action_noise_seed=999)
    with pytest.raises(ValueError, match="noise"):
        validate_matched_counterfactuals(rows)


def test_future_utility_is_deterministic_and_positive_for_g3() -> None:
    first = evaluate_future_utility(utility_panel(), bootstrap_replicates=100)
    second = evaluate_future_utility(utility_panel(), bootstrap_replicates=100)
    assert first == second
    assert first["h2_supported"] is True


def test_single_task_utility_is_directional_not_formal_inference() -> None:
    rows = [row for row in utility_panel() if row.task_id == "8"]
    for variant, loss in {"A0": 1.0, "A1": 1.04, "AS": 1.05}.items():
        rows.append(
            FutureUtilityRecord(
                backbone="G4",
                adapter_variant=variant,
                task_id="8",
                episode_id="8-e",
                condition="camera",
                flow_slot=1,
                action_noise_seed=10,
                action_timestep_seed=11,
                denoise_schedule_sha256="schedule",
                loss=loss,
                action_sha256=f"g4-{variant}",
                action_rms=1.0,
            )
        )
    result = evaluate_future_utility(rows, bootstrap_replicates=100)
    assert result["pilot_direction_observed"] is True
    assert result["formal_multitask_inference"] is False
    assert result["h2_supported"] is False
    assert result["shuffled_control_matches_gain"] is False


def test_single_task_representation_and_rollout_stay_noninferential() -> None:
    representation = []
    errors = {
        "B1": {"clean": 0.10, "camera": 0.50, "lighting": 0.20, "robot_init": 0.30},
        "G3": {"clean": 0.10, "camera": 0.20, "lighting": 0.18, "robot_init": 0.30},
        "G4": {"clean": 0.10, "camera": 0.48, "lighting": 0.20, "robot_init": 0.30},
    }
    for variant, by_condition in errors.items():
        for condition, error in by_condition.items():
            representation.append(
                RepresentationRecord(
                    variant=variant,
                    task_id="0",
                    episode_id="e0",
                    seed=1,
                    condition=condition,
                    endpoint="video_eef_object_translation_camera",
                    error=error,
                )
            )
    h1 = evaluate_h1(representation, bootstrap_replicates=100)
    assert h1["pilot_direction_observed"] is True
    assert h1["formal_multitask_inference"] is False
    assert h1["h1_supported"] is False

    rollout = []
    for seed in range(4):
        for condition in ("clean", "camera", "lighting", "robot_init"):
            for variant in ("B1", "G3", "G4"):
                success = condition != "camera" or variant == "G3"
                rollout.append(
                    RolloutRecord(
                        variant=variant,
                        task_id="0",
                        episode_seed=seed,
                        condition=condition,
                        success=success,
                        latency_ms=1.0,
                        peak_memory_mib=2.0,
                    )
                )
    h3 = evaluate_rollouts(rollout, bootstrap_replicates=100)
    assert h3["pilot_direction_observed"] is True
    assert h3["formal_multitask_inference"] is False
    assert h3["h3_supported"] is False


def test_no_cross_episode_future_label() -> None:
    assert_no_future_crossing(4, 8, [(4, 9), (4, 10)])


def test_grouped_bootstrap_is_seed_reproducible() -> None:
    values = {"a": [1.0], "b": [2.0], "c": [3.0]}
    assert grouped_bootstrap_mean(values, replicates=100, seed=1) == grouped_bootstrap_mean(
        values, replicates=100, seed=1
    )


def test_shuffled_future_donors_are_deranged_within_condition() -> None:
    entries = [
        SimpleNamespace(sample_id=f"{condition}-{index}", condition=condition)
        for condition in ("clean", "camera")
        for index in range(3)
    ]
    donors = _donor_lookup(entries, seed=5547)
    assert set(donors) == {entry.sample_id for entry in entries}
    for entry in entries:
        assert donors[entry.sample_id].condition == entry.condition
        assert donors[entry.sample_id].sample_id != entry.sample_id


def test_h2_action_chunk_sensitivity_uses_bitwise_replay_floor() -> None:
    chunks = {}
    replay = {}
    seeds = {}
    for index in range(2):
        sample_id = f"sample-{index}"
        chunks[("A0", sample_id)] = torch.zeros(32, 7)
        chunks[("A1", sample_id)] = torch.ones(32, 7)
        chunks[("AS", sample_id)] = -torch.ones(32, 7)
        replay[sample_id] = chunks[("A1", sample_id)].clone()
        seeds[sample_id] = 100 + index
    result = _technical_action_sensitivity(
        chunks,
        replay,
        action_seeds=seeds,
        schedule_sha256="schedule",
    )
    assert result["replay_floor"]["hard_passed"] is True
    assert result["replay_floor"]["material_l2_threshold"] == 1e-7
    assert result["correct_null_exceeds_replay_floor"] == 2
    assert result["correct_shuffle_exceeds_replay_floor"] == 2


def _future_probe_entry(sample_id: str, split: str, offset: float) -> FutureProbeEntry:
    generator = torch.Generator().manual_seed(sum(map(ord, sample_id)))
    feature = torch.randn(2, 98, 8, generator=generator)
    target = torch.stack(
        (
            feature[..., 0] + offset,
            feature[..., 1],
            feature[..., 2],
            feature[..., 3],
            feature[..., 4],
            feature[..., 5],
            feature[..., 6],
        ),
        dim=-1,
    )
    return FutureProbeEntry(
        sample_id=sample_id,
        task_id="8",
        episode_id=sample_id,
        cohort_seed=1,
        split=split,
        condition="camera",
        projected_hidden=feature,
        actual_depth_relation=target[..., 0:1],
        actual_eef_object=target[..., 1:4],
        actual_camera_geometry=target[..., 4:7],
        predicted_embeddings=torch.zeros(3, 4) if split == "formal" else None,
        actual_embeddings=torch.zeros(3, 4) if split == "formal" else None,
    )


def test_future_geometry_probe_is_train_dev_only_and_same_capacity() -> None:
    entries = [
        _future_probe_entry("train-a", "train", 0.0),
        _future_probe_entry("train-b", "train", 0.0),
        _future_probe_entry("dev-a", "development", 0.0),
        _future_probe_entry("formal-a", "formal", 0.0),
    ]
    first_model, first_meta = _fit_future_probe(entries, alphas=(0.01, 1.0))
    changed_formal = replace(
        entries[-1], actual_camera_geometry=torch.full((2, 98, 3), 999.0)
    )
    second_model, second_meta = _fit_future_probe(
        [*entries[:-1], changed_formal], alphas=(0.01, 1.0)
    )
    assert first_meta == second_meta
    assert first_meta["selection_reads_formal"] is False
    assert first_meta["projection_dim"] == 8
    records = _future_probe_records(
        variant="G3", entries=entries, model=first_model
    )
    assert len(records) == 1
    assert records[0].predicted_camera_geometry.shape == (2, 98, 3)


def test_rollout_exact_state_hash_contract() -> None:
    rows = []
    for condition in ("clean", "camera", "lighting", "robot_init"):
        rows.append(
            SimpleNamespace(
                task_id=8,
                episode_seed=1,
                condition=condition,
                extra={
                    "initial_state_sha256": (
                        "a" * 64 if condition != "robot_init" else "b" * 64
                    )
                },
            )
        )
    _validate_initial_state_pairing(rows)
    rows[1].extra["initial_state_sha256"] = "c" * 64
    with pytest.raises(RolloutRuntimeError, match="exact-state"):
        _validate_initial_state_pairing(rows)
