from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from fastwam_ood_eval.thought4.action_intervention import (
    ActionInterventionError,
    ActionSeedIdentity,
    DonorCandidate,
    build_deterministic_derangement,
    compare_action_chunks,
    replay_floor,
    validate_seed_identity,
)
from fastwam_ood_eval.thought4.decision import (
    DiagnosticEvidence,
    derive_diagnostic_evidence,
    select_method,
    validate_method_selection,
)
from fastwam_ood_eval.thought4.geometry_subspace import (
    correct_reconstruction,
    geometry_coordinates,
    replace_geometry_coordinates,
    subspace_from_linear_weight,
)
from fastwam_ood_eval.thought4.intervention_runtime import (
    InterventionRuntimeError,
    _bitwise_correct_reconstruction,
    geometry_coordinate_condition_shift,
)
from fastwam_ood_eval.thought4.probe_models import linear_weight
from fastwam_ood_eval.thought4.pipeline import (
    ProbeExample,
    select_intervention_feature,
)
from fastwam_ood_eval.thought4.probe_training import (
    ProbeDataset,
    ProbeTrainingError,
    ProbeTrainingSpec,
    shuffled_targets,
    target_baselines,
    train_probe,
)
from fastwam_ood_eval.thought4.schemas import sha256_canonical


def dataset(prefix: str, episodes: list[str], count: int) -> ProbeDataset:
    x = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3) / 10
    y = (x[:, :1] * 2 + 0.5).detach()
    return ProbeDataset(
        x.detach(),
        y,
        None,
        tuple(episodes),
        tuple(f"{prefix}{i}" for i in range(count)),
    )


def test_probe_input_detached_controls_and_training() -> None:
    train = dataset("t", ["e0", "e0", "e1", "e1"], 4)
    development = dataset("d", ["e2", "e2"], 2)
    spec = ProbeTrainingSpec("linear", "regression", 8, 0.05, 0.0, 40, 8, 2, 3)
    result = train_probe(train, development, spec)
    assert result.parameter_count == 4
    assert math.isfinite(result.best_development_loss)
    assert result.feature_mean.shape == (3,)
    assert result.target_mean.shape == (1,)
    first, second = train.features[:2]
    with torch.no_grad():
        first_output = result.model(
            ((first - result.feature_mean) / result.feature_std).unsqueeze(0)
        )
        second_output = result.model(
            ((second - result.feature_mean) / result.feature_std).unsqueeze(0)
        )
    observed_delta = (first_output - second_output) * result.target_std
    expected_delta = (first - second) @ linear_weight(result).T
    assert torch.allclose(observed_delta.reshape(-1), expected_delta.reshape(-1))
    shuffled = shuffled_targets(train.targets, seed=4)
    assert not torch.equal(shuffled, train.targets)
    assert all(
        not torch.equal(shuffled[index], train.targets[index])
        for index in range(len(train.targets))
    )
    baselines = target_baselines(train.targets, development.targets)
    assert set(baselines) == {"constant_zero", "target_mean"}
    leaking = ProbeDataset(
        train.features.requires_grad_(),
        train.targets,
        None,
        train.episode_ids,
        train.sample_ids,
    )
    with pytest.raises(ProbeTrainingError, match="detached"):
        leaking.validate()


def test_probe_episode_leakage_fails_closed() -> None:
    train = dataset("t", ["same", "same"], 2)
    development = dataset("d", ["same", "same"], 2)
    spec = ProbeTrainingSpec("linear", "regression", 8, 0.01, 0.0, 2, 1, 2, 3)
    with pytest.raises(ProbeTrainingError, match="leaks"):
        train_probe(train, development, spec)


def test_target_mean_baseline_uses_only_valid_train_labels() -> None:
    train_targets = torch.tensor(
        [[[1.0], [1000.0]], [[3.0], [5.0]]], dtype=torch.float32
    )
    train_mask = torch.tensor([[True, False], [True, True]])
    evaluation_targets = torch.zeros((1, 2, 1), dtype=torch.float32)
    result = target_baselines(
        train_targets,
        evaluation_targets,
        train_mask=train_mask,
    )
    assert torch.equal(
        result["target_mean"], torch.tensor([[[2.0], [5.0]]])
    )


def test_geometry_subspace_is_orthogonal_and_correct_reconstructs() -> None:
    torch.manual_seed(5)
    weight = torch.randn(4, 8)
    subspace = subspace_from_linear_weight(weight, max_rank=3)
    gram = subspace.basis.T @ subspace.basis
    assert torch.allclose(gram, torch.eye(subspace.rank), atol=1e-5)
    hidden = torch.randn(2, 5, 8)
    correct = correct_reconstruction(hidden, subspace)
    assert torch.allclose(correct.output, hidden, atol=1e-5)
    assert correct.residual_reconstruction_error < 1e-5


def test_bf16_correct_reconstruction_uses_fp32_and_is_bitwise_equal() -> None:
    torch.manual_seed(51)
    subspace = subspace_from_linear_weight(
        torch.randn(3, 64, dtype=torch.float32), max_rank=3
    )
    hidden = (torch.randn(2, 7, 64) * 3.0).to(torch.bfloat16)
    old_basis = subspace.basis.to(torch.bfloat16)
    old_projection = (hidden @ old_basis) @ old_basis.T
    old_reconstruction = (hidden - old_projection) + old_projection
    assert not torch.equal(old_reconstruction, hidden)
    assert float(
        (old_reconstruction.float() - hidden.float()).abs().max()
    ) > 0.0

    correct = correct_reconstruction(hidden, subspace)
    assert correct.original_coordinates.dtype == torch.float32
    assert correct.residual.dtype == torch.float32
    assert correct.output.dtype == torch.bfloat16
    assert torch.equal(correct.output, hidden)
    assert correct.residual_reconstruction_error == 0.0

    donor = (torch.randn(2, 7, 64) * 3.0).to(torch.bfloat16)
    replacement = replace_geometry_coordinates(
        hidden,
        geometry_coordinates(donor, subspace),
        subspace,
        norm_match=True,
    )
    assert replacement.output.dtype == torch.bfloat16
    assert torch.isfinite(replacement.output).all()


def test_runtime_correct_control_freezes_bitwise_contract() -> None:
    torch.manual_seed(52)
    subspace = subspace_from_linear_weight(
        torch.randn(3, 64, dtype=torch.float32), max_rank=3
    )
    hidden = (torch.randn(1, 11, 64) * 4.0).to(torch.bfloat16)
    correct, contract = _bitwise_correct_reconstruction(hidden, subspace)
    assert torch.equal(correct.output, hidden)
    assert contract["compute_dtype"] == "torch.float32"
    assert contract["input_dtype"] == "torch.bfloat16"
    assert contract["output_dtype"] == "torch.bfloat16"
    assert contract["single_output_cast"] is True
    assert contract["residual_reconstruction_max_abs"] == 0.0
    assert contract["input_sha256"] == contract["output_sha256"]
    assert contract["bitwise_equal_after_output_cast"] is True
    with pytest.raises(InterventionRuntimeError, match="must be BF16"):
        _bitwise_correct_reconstruction(hidden.float(), subspace)


def test_shuffle_replaces_only_geometry_coordinates() -> None:
    torch.manual_seed(6)
    subspace = subspace_from_linear_weight(torch.randn(3, 7), max_rank=2)
    target = torch.randn(2, 4, 7)
    donor = torch.randn(2, 4, 7)
    donor_coordinates = geometry_coordinates(donor, subspace)
    result = replace_geometry_coordinates(
        target, donor_coordinates, subspace, norm_match=False
    )
    basis = subspace.basis
    target_residual = target - geometry_coordinates(target, subspace) @ basis.T
    output_residual = result.output - geometry_coordinates(result.output, subspace) @ basis.T
    assert torch.allclose(target_residual, output_residual, atol=1e-5)
    assert torch.allclose(
        geometry_coordinates(result.output, subspace), donor_coordinates, atol=1e-5
    )


def test_intervention_selection_uses_only_action_consumed_video_kv() -> None:
    def row(
        path: str, layer: int, loss: float, seed: int = 11
    ) -> dict[str, object]:
        return {
            "source": "A",
            "feature_key": path,
            "module_path": path,
            "layer_index": layer,
            "pooling": "spatial_mean",
            "target": "eef_object_translation_camera",
            "probe_kind": "linear",
            "seed": seed,
            "development_loss": loss,
        }

    result = select_intervention_feature(
        {
            "rows": [
                row("video_expert.blocks.29.norm1", 29, 0.01),
                row("mot.video_kv_cache.7.k", 7, 0.01),
                row("mot.video_kv_cache.7.k", 7, 0.39, 12),
                row("mot.video_kv_cache.15.v", 15, 0.10),
                row("mot.video_kv_cache.15.v", 15, 0.10, 12),
            ]
        },
        target="eef_object_translation_camera",
        seed=11,
    )
    assert result["module_path"] == "mot.video_kv_cache.15.v"
    assert result["candidate_scope"] == "action_consumed_video_kv_only"
    assert result["selection_seeds"] == [11, 12]
    assert result["development_loss_mean_across_seeds"] == pytest.approx(0.10)


def test_geometry_coordinate_shift_uses_exact_state_pairs_and_grouped_ci() -> None:
    examples = []
    values = {
        "clean": torch.tensor([1.0, 0.0]),
        "camera": torch.tensor([2.0, 0.0]),
        "lighting": torch.tensor([1.1, 0.0]),
        "robot_init": torch.tensor([3.0, 0.0]),
    }
    for sample_index in range(2):
        for condition, feature in values.items():
            examples.append(
                ProbeExample(
                    sample_id=f"s{sample_index}",
                    episode_id=f"e{sample_index}",
                    split="test",
                    condition=condition,
                    source="A",
                    module_path="mot.video_kv_cache.15.v",
                    layer_index=15,
                    denoise_step_index=None,
                    pooling="spatial_mean",
                    feature=feature,
                    labels={},
                    masks={},
                )
            )
    feature_key = examples[0].feature_key
    result = geometry_coordinate_condition_shift(
        examples,
        selection={"feature_key": feature_key},
        subspace=subspace_from_linear_weight(torch.tensor([[1.0, 0.0]])),
        bootstrap_replicates=20,
        bootstrap_seed=9,
    )
    summaries = result["condition_summaries"]
    assert summaries["camera"]["exact_state_pair"] is True
    assert summaries["robot_init"]["exact_state_pair"] is False
    assert summaries["camera"]["coordinate_l2_grouped_bootstrap"]["estimate"] == pytest.approx(1.0)
    assert result["camera_minus_lighting_paired_grouped_bootstrap"]["estimate"] == pytest.approx(0.9)


def test_donor_derangement_has_no_fixed_point_or_same_episode() -> None:
    candidates = [
        DonorCandidate(f"s{i}", "task", f"e{i}", i % 2) for i in range(4)
    ]
    first = build_deterministic_derangement(candidates, seed=7)
    second = build_deterministic_derangement(reversed(candidates), seed=7)
    assert first == second
    assert all(pair.target_sample_id != pair.donor_sample_id for pair in first)
    assert all(pair.target_episode_id != pair.donor_episode_id for pair in first)


def test_action_seed_contract_replay_floor_and_metrics() -> None:
    identity = ActionSeedIdentity(1, "a" * 64, "b" * 64, "c" * 64, "d" * 64)
    validate_seed_identity(identity, identity)
    with pytest.raises(ActionInterventionError):
        validate_seed_identity(
            identity,
            ActionSeedIdentity(2, "a" * 64, "b" * 64, "c" * 64, "d" * 64),
        )

    def action(seed: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(1, 4, 7, generator=generator)

    floor = replay_floor(action, seeds=(1, 2), repeats=2)
    assert floor["max_action_l2"] == 0.0
    changed = compare_action_chunks(action(1), action(2))
    assert changed["action_l2"] > 0
    assert len(changed["timestep_l2"]) == 4


@pytest.mark.parametrize(
    ("evidence", "classification"),
    [
        (
            DiagnosticEvidence(
                False, False, False, False, False, False, False, "a" * 64
            ),
            "video_geometry_representation_gap",
        ),
        (
            DiagnosticEvidence(
                True, True, True, True, True, True, True, "a" * 64
            ),
            "camera_equivariance_gap",
        ),
        (
            DiagnosticEvidence(
                True, False, False, False, False, True, True, "a" * 64
            ),
            "world_action_interface_gap",
        ),
        (
            DiagnosticEvidence(
                True, False, False, True, True, False, False, "a" * 64
            ),
            "geometry_hypothesis_not_supported",
        ),
    ],
)
def test_method_selection_emits_exactly_one_class(
    evidence: DiagnosticEvidence, classification: str
) -> None:
    result = select_method(evidence)
    assert result["classification"] == classification
    validate_method_selection(result)


def test_evidence_derivation_selects_feature_by_multi_seed_development_mean() -> None:
    def row(
        feature: str,
        target: str,
        seed: int,
        development_loss: float,
        *,
        readable: bool,
        camera_significant: bool,
    ) -> dict[str, object]:
        metric_key = (
            "se3_trajectory_composite"
            if target == "action_se3_trajectory"
            else "rmse"
        )
        probe_error = 0.5 if readable else 1.0
        control_error = 1.0

        def condition(lower: float, upper: float) -> dict[str, object]:
            return {
                "metrics": {metric_key: probe_error},
                "baselines": {"target_mean": {metric_key: control_error}},
                "shuffled_label_control": {metric_key: control_error},
                "rmse_grouped_bootstrap": {
                    "estimate": (lower + upper) / 2,
                    "lower": lower,
                    "upper": upper,
                },
            }

        camera_lower = 0.2 if camera_significant else -0.2
        return {
            "source": "A" if feature.startswith("video") else "B",
            "feature_key": feature,
            "module_path": feature,
            "layer_index": 15,
            "denoise_step_index": None,
            "pooling": "spatial_mean",
            "target": target,
            "probe_kind": "linear",
            "seed": seed,
            "development_loss": development_loss,
            "row_sha256": f"{feature}:{target}:{seed}",
            "condition_metrics": {
                "clean": condition(0.4, 0.6),
                "camera": condition(0.7, 0.9),
                "lighting": condition(0.45, 0.65),
                "robot_init": condition(1.2, 1.4),
            },
            "exact_state_paired_rmse_gaps": {
                "camera": {
                    "estimate": 0.3,
                    "lower": camera_lower,
                    "upper": 0.4,
                },
                "lighting": {
                    "estimate": 0.05,
                    "lower": -0.05,
                    "upper": 0.1,
                },
            },
            "gaps_vs_clean_rmse": {
                "camera": 0.3,
                "lighting": 0.05,
                "robot_init": 0.8,
            },
        }

    video_rows = [
        row("video_lucky", "eef_object_translation_camera", 1, 0.01, readable=False, camera_significant=False),
        row("video_lucky", "eef_object_translation_camera", 2, 0.99, readable=False, camera_significant=False),
        row("video_stable", "eef_object_translation_camera", 1, 0.20, readable=True, camera_significant=True),
        row("video_stable", "eef_object_translation_camera", 2, 0.20, readable=True, camera_significant=True),
    ]
    action_rows = []
    for seed in (1, 2):
        action_rows.extend(
            (
                row("action_motion", "action_se3_trajectory", seed, 0.1, readable=True, camera_significant=False),
                row("action_geometry", "eef_object_translation_camera", seed, 0.1, readable=True, camera_significant=False),
            )
        )
    coordinate_shift = {
        "camera_minus_lighting_paired_grouped_bootstrap": {
            "estimate": 0.2,
            "lower": 0.1,
            "upper": 0.3,
        }
    }
    coordinate_shift["result_sha256"] = sha256_canonical(coordinate_shift)
    evidence, payload = derive_diagnostic_evidence(
        {"rows": video_rows},
        {"rows": action_rows},
        {
            "comparison_count": 4,
            "correct_shuffle_above_floor_count": 4,
            "geometry_coordinate_condition_shift": coordinate_shift,
        },
    )
    assert evidence.clean_video_geometry_readable is True
    assert evidence.camera_video_gap_significant is True
    assert evidence.camera_gap_larger_than_lighting is True
    assert all("video_stable" in value for value in payload["selected_video_row_sha256_by_seed"])
