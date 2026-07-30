from __future__ import annotations

import math
from pathlib import Path

import pytest

from fastwam_ood_eval.thought3.config import load_thought3_config
from fastwam_ood_eval.thought3.phase2_full_training import (
    Phase2ExecutionError,
    _artifact_relative_key,
    _validate_training_prefix,
    classify_phase2_training_direction,
    derive_phase2_thought3_config,
    run_phase2_calibration,
)
from fastwam_ood_eval.thought3.phase2_protocol import (
    PHASE2_VARIANTS,
    Phase2ProtocolError,
    inverse_initial_loss_unit_mean_weights,
    load_phase2_full_training_config,
    phase2_flow_objective_identity,
    phase2_identity_schedule_sha256,
    phase2_sample_loss_weights_sha256,
    phase2_training_flow_slot,
)
from fastwam_ood_eval.thought3.real_training import (
    _flow_objective_identity,
)


CONFIG = Path("configs/thought3/phase2_full_28_4_a0_a1.yaml")


def test_phase2_artifact_key_normalizes_relative_path_against_absolute_root(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    relative_root = Path("outputs/thought3/phase2/calibration")
    relative_artifact = relative_root / "calibration.json"

    assert (
        _artifact_relative_key(relative_artifact, relative_root.resolve())
        == "calibration.json"
    )
    assert (
        _artifact_relative_key(
            relative_root / "nested" / "metrics.jsonl",
            relative_root.resolve(),
        )
        == "nested/metrics.jsonl"
    )
    with pytest.raises(Phase2ExecutionError, match="outside"):
        _artifact_relative_key(
            Path("outputs/thought3/outside.json"),
            relative_root.resolve(),
        )


def test_phase2_config_freezes_one_28_4_recipe() -> None:
    cfg = load_phase2_full_training_config(CONFIG)
    assert (
        cfg.fingerprint
        == "fabb96a97b7e137ca39a5477c2090deab1844909887b10cd22fa92ebbee66468"
    )
    assert cfg.variants == PHASE2_VARIANTS == ("A0", "A1")
    assert (cfg.train_count, cfg.development_count) == (28, 4)
    assert cfg.optimizer_updates == 200
    assert cfg.objectives_per_update == 28
    assert cfg.learning_rate == 3e-4
    assert cfg.primary_checkpoint_rule == (
        "fixed_step_200_no_selection_no_fallback"
    )
    assert cfg.training_flow_start == 50_001
    assert cfg.training_flow_end == 55_600
    assert cfg.calibration_flow_steps == tuple(range(139, 171))
    assert cfg.development_flow_steps == tuple(range(171, 203))
    used = (
        set(cfg.calibration_flow_steps)
        | set(cfg.development_flow_steps)
        | set(range(cfg.training_flow_start, cfg.training_flow_end + 1))
    )
    assert not used & set(range(75, 139))
    assert cfg.scope["read_development"] is True
    assert cfg.scope["read_ood"] is False
    assert cfg.scope["read_rollout_success"] is False
    assert cfg.scope["train_a2_or_a4"] is False
    assert cfg.scope["select_checkpoint_from_development"] is False


def test_phase2_derived_track_configs_are_matched_except_treatment() -> None:
    cfg = load_phase2_full_training_config(CONFIG)
    base = load_thought3_config(cfg.thought3_base_config_path)
    a0 = derive_phase2_thought3_config(
        cfg,
        base,
        variant="A0",
        output_dir=cfg.track_output_dir("A0"),
    )
    a1 = derive_phase2_thought3_config(
        cfg,
        base,
        variant="A1",
        output_dir=cfg.track_output_dir("A1"),
    )
    assert a0.sampler.active_k == 0
    assert a1.sampler.active_k == 1
    assert a0.training == a1.training
    assert a0.training.gradient_accumulation_steps == 28
    assert a0.training.max_steps == 200
    assert a0.adapter == a1.adapter
    assert a0.backbone == a1.backbone
    assert a0.data == a1.data
    assert a0.cache == a1.cache
    with pytest.raises(Phase2ProtocolError, match="unsupported"):
        derive_phase2_thought3_config(
            cfg,
            base,
            variant="A2",
            output_dir=Path("outputs/thought3/forbidden"),
        )


def test_phase2_pure_flow_identity_matches_real_training_identity() -> None:
    for sample_id, flow_step in (
        ("sample-a", 139),
        ("sample-b", 50_001),
        ("sample-c", 55_600),
    ):
        assert phase2_flow_objective_identity(
            base_sample_id=sample_id,
            train_seed=3407,
            flow_step=flow_step,
        ) == _flow_objective_identity(
            base_sample_id=sample_id,
            train_seed=3407,
            flow_step=flow_step,
        )


def test_phase2_full_schedule_is_unique_and_matched() -> None:
    sample_ids = [f"sample-{index:02d}" for index in range(28)]
    slots = [
        phase2_training_flow_slot(update, micro)
        for update in range(1, 201)
        for micro in range(1, 29)
    ]
    assert slots == list(range(50_001, 55_601))
    assert len(set(slots)) == 5_600
    digest = phase2_identity_schedule_sha256(
        sample_ids,
        train_seed=3407,
    )
    assert len(digest) == 64
    swapped = list(sample_ids)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert phase2_identity_schedule_sha256(
        swapped,
        train_seed=3407,
    ) != digest


def test_inverse_initial_loss_weights_are_positive_unit_mean() -> None:
    sample_ids = ("a", "b", "c", "d")
    losses = {"a": 1.0, "b": 2.0, "c": 4.0, "d": 8.0}
    weights, digest = inverse_initial_loss_unit_mean_weights(
        sample_ids,
        losses,
    )
    assert math.isclose(sum(weights.values()), 4.0, abs_tol=1e-10)
    assert weights["a"] > weights["b"] > weights["c"] > weights["d"]
    assert phase2_sample_loss_weights_sha256(
        sample_ids,
        weights,
    ) == digest
    with pytest.raises(Phase2ProtocolError, match="positive"):
        inverse_initial_loss_unit_mean_weights(
            sample_ids,
            {**losses, "a": 0.0},
        )


def test_training_prefix_rejects_sample_or_flow_drift() -> None:
    cfg = load_phase2_full_training_config(CONFIG)
    sample_ids = [f"sample-{index:02d}" for index in range(28)]
    rows = []
    for micro, sample_id in enumerate(sample_ids, start=1):
        slot = phase2_training_flow_slot(1, micro)
        identity = phase2_flow_objective_identity(
            base_sample_id=sample_id,
            train_seed=cfg.train_seed,
            flow_step=slot,
        )
        rows.append(
            {
                **identity,
                "base_sample_id": sample_id,
                "micro_index": micro,
                "objective_index": micro,
                "optimizer_update": 1,
                "training_flow_slot": slot,
            }
        )
    _validate_training_prefix(rows, sample_ids=sample_ids, cfg=cfg)
    corrupt = [dict(row) for row in rows]
    corrupt[4]["base_sample_id"] = "wrong"
    with pytest.raises(Phase2ExecutionError, match="schedule mismatch"):
        _validate_training_prefix(
            corrupt,
            sample_ids=sample_ids,
            cfg=cfg,
        )


def test_phase2_direction_rule_never_unlocks_phase3_directly() -> None:
    checks = {"finite": True, "frozen": True}
    classification, next_stage, direction = (
        classify_phase2_training_direction(
            hard_checks=checks,
            initial_mean=1.0,
            a0_final_mean=0.9,
            a1_final_mean=0.8,
        )
    )
    assert direction is True
    assert classification == (
        "training_valid_pending_full_checkpoint_online_sensitivity"
    )
    assert next_stage == (
        "phase2_full_checkpoint_online_correct_null_shuffle_recheck"
    )
    classification, _, direction = classify_phase2_training_direction(
        hard_checks=checks,
        initial_mean=1.0,
        a0_final_mean=0.8,
        a1_final_mean=0.9,
    )
    assert direction is False
    assert classification == "training_valid_dev_direction_not_observed"
    classification, _, direction = classify_phase2_training_direction(
        hard_checks={"finite": False},
        initial_mean=1.0,
        a0_final_mean=0.8,
        a1_final_mean=0.7,
    )
    assert direction is False
    assert classification == "phase2_engineering_invalid"


def test_phase2_real_calibration_refuses_before_cuda_or_model(
    monkeypatch,
) -> None:
    cfg = load_phase2_full_training_config(CONFIG)
    monkeypatch.delenv("CONFIRM_THOUGHT3_PHASE2_FULL", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Phase 2 touched CUDA before confirmation")

    monkeypatch.setattr(
        "fastwam_ood_eval.thought3.phase2_full_training._configure_cuda",
        forbidden,
    )
    with pytest.raises(
        Phase2ExecutionError,
        match="CONFIRM_THOUGHT3_PHASE2_FULL",
    ):
        run_phase2_calibration(cfg, resume=False)


def test_phase2_config_rejects_recipe_drift(tmp_path) -> None:
    text = CONFIG.read_text(encoding="utf-8").replace(
        "learning_rate: 0.0003",
        "learning_rate: 0.0001",
    )
    path = tmp_path / "changed.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(Phase2ProtocolError, match="protocol changed"):
        load_phase2_full_training_config(path)
