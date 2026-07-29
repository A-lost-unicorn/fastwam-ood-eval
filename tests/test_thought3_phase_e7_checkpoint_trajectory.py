from __future__ import annotations

import inspect
from dataclasses import replace

import pytest
import torch

from fastwam_ood_eval.thought3.config import load_thought3_config
from fastwam_ood_eval.thought3.phase_e6_fresh_cohort_replication import (
    replication_performance_checks,
)
from fastwam_ood_eval.thought3.phase_e7_checkpoint_trajectory import (
    PHASE_E6_CHECKPOINT_FILE_SHA256,
    PHASE_E7_CHECKPOINT_STEPS,
    PHASE_E7_CONFIG_FINGERPRINT,
    PHASE_E7_CONTINUITY_FLOW_STEPS,
    PHASE_E7_CONTINUITY_IDENTITY_SCHEDULE_SHA256,
    PHASE_E7_CONTINUITY_ZERO_WEIGHT_POSITIONS,
    PHASE_E7_PRIMARY_FLOW_STEPS,
    PHASE_E7_PRIMARY_IDENTITY_SCHEDULE_SHA256,
    PHASE_E7_PRIMARY_ZERO_WEIGHT_POSITIONS,
    PHASE_E7_ROOT,
    PHASE_E7_SCHEMA,
    _assert_frozen_probe_design,
    _assert_phase_e7_scope,
    _probe_checks,
    _run_phase_e7,
    classify_a0_trajectory,
    diagnostic_joint_candidate_steps,
    probe_identity_schedule_sha256,
    run_phase_e7_checkpoint_trajectory,
    verify_frozen_phase_e6,
)
from fastwam_ood_eval.thought3.real_training import (
    _flow_objective_identity,
    aggregate_multiflow_probe_grid_rows,
    multiflow_probe_grid_outcome,
)


CONFIG = "configs/thought3/phase_e7_checkpoint_trajectory.yaml"
SAMPLE_IDS = tuple(f"sample-{index}" for index in range(8))


@pytest.fixture(scope="module")
def frozen_parent() -> dict:
    return verify_frozen_phase_e6()


def test_phase_e7_scope_and_checkpoint_parent_are_frozen(
    frozen_parent,
) -> None:
    cfg = load_thought3_config(CONFIG)
    _assert_phase_e7_scope(cfg)
    assert cfg.fingerprint == PHASE_E7_CONFIG_FINGERPRINT
    assert cfg.experiment.output_dir == PHASE_E7_ROOT
    assert PHASE_E7_SCHEMA == "thought3.phase_e7.checkpoint_trajectory.v1"
    assert PHASE_E7_CHECKPOINT_STEPS == (50, 100, 150, 200)
    assert frozen_parent["gate_e6_passed"] is False
    assert (
        frozen_parent["known_before_e7"][
            "intermediate_checkpoint_outcomes_read"
        ]
        is False
    )
    assert (
        frozen_parent["checkpoint_file_sha256"]
        == PHASE_E6_CHECKPOINT_FILE_SHA256
    )
    assert {
        (variant, step)
        for variant in ("A0", "A1")
        for step in frozen_parent["checkpoint_evidence"][variant]
    } == {
        (variant, str(step))
        for variant in ("A0", "A1")
        for step in PHASE_E7_CHECKPOINT_STEPS
    }


def test_phase_e7_probe_rng_identity_and_zero_slots_are_frozen(
    frozen_parent,
) -> None:
    cfg = load_thought3_config(CONFIG)
    identities = _assert_frozen_probe_design(
        frozen_parent["sample_ids"],
        train_seed=cfg.training.train_seed,
    )
    assert identities == {
        "primary": PHASE_E7_PRIMARY_IDENTITY_SCHEDULE_SHA256,
        "continuity": (
            PHASE_E7_CONTINUITY_IDENTITY_SCHEDULE_SHA256
        ),
    }
    assert PHASE_E7_PRIMARY_FLOW_STEPS == (6, 7, 8, 9, 10)
    assert PHASE_E7_CONTINUITY_FLOW_STEPS == (1, 2, 3, 4, 5)
    assert not (
        set(PHASE_E7_PRIMARY_FLOW_STEPS)
        & set(PHASE_E7_CONTINUITY_FLOW_STEPS)
    )
    assert PHASE_E7_PRIMARY_ZERO_WEIGHT_POSITIONS == ()
    assert PHASE_E7_CONTINUITY_ZERO_WEIGHT_POSITIONS == ((8, 5),)

    observed: dict[str, list[tuple[int, int]]] = {
        "primary": [],
        "continuity": [],
    }
    for panel, flow_steps in (
        ("primary", PHASE_E7_PRIMARY_FLOW_STEPS),
        ("continuity", PHASE_E7_CONTINUITY_FLOW_STEPS),
    ):
        for sample_index, base_sample_id in enumerate(
            frozen_parent["sample_ids"],
            start=1,
        ):
            for flow_step in flow_steps:
                identity = _flow_objective_identity(
                    base_sample_id=base_sample_id,
                    train_seed=cfg.training.train_seed,
                    flow_step=flow_step,
                )
                generator = torch.Generator(
                    device="cpu"
                ).manual_seed(identity["action_timestep_seed"])
                u = torch.rand(
                    (1,),
                    generator=generator,
                    dtype=torch.float32,
                )
                sigma = 5.0 * u / (1.0 + 4.0 * u)
                timestep = (sigma * 1000.0).to(
                    dtype=torch.bfloat16
                )
                if float(timestep.float()) == 1000.0:
                    observed[panel].append(
                        (sample_index, flow_step)
                    )
    assert tuple(observed["primary"]) == (
        PHASE_E7_PRIMARY_ZERO_WEIGHT_POSITIONS
    )
    assert tuple(observed["continuity"]) == (
        PHASE_E7_CONTINUITY_ZERO_WEIGHT_POSITIONS
    )


def _probe(
    sample_losses: list[float],
    *,
    ratio: float,
    flow_steps: tuple[int, ...],
    variant: str,
) -> dict:
    rows = [
        {
            "action_hidden_norm": 10.0,
            "action_loss": sample_losses[sample_index],
            "action_weight": 1.0,
            "attention_residual_norm": 2.0,
            "base_sample_id": base_sample_id,
            "flow_step": flow_step,
            "gated_delta_nonzero_fraction": (
                0.0 if ratio == 0 else 1.0
            ),
            "gated_delta_norm": ratio * 10.0,
            "gated_delta_to_action_hidden_ratio": ratio,
            "latency_ms": 5.0,
            "peak_memory_mib": 100.0,
            "timestep": 100.0 + flow_step,
        }
        for sample_index, base_sample_id in enumerate(SAMPLE_IDS)
        for flow_step in flow_steps
    ]
    return aggregate_multiflow_probe_grid_rows(
        rows,
        sample_ids=SAMPLE_IDS,
        flow_steps=flow_steps,
        variant=variant,
    )


def test_primary_multiflow_grid_requires_explicit_frozen_identity() -> None:
    initial = _probe(
        [1.0] * 8,
        ratio=0.0,
        flow_steps=PHASE_E7_PRIMARY_FLOW_STEPS,
        variant="A0",
    )
    final = _probe(
        [0.8] * 6 + [1.05] * 2,
        ratio=0.4,
        flow_steps=PHASE_E7_PRIMARY_FLOW_STEPS,
        variant="A0",
    )
    outcome = multiflow_probe_grid_outcome(
        initial,
        final,
        expected_flow_steps=PHASE_E7_PRIMARY_FLOW_STEPS,
    )
    assert outcome["flow_steps"] == list(PHASE_E7_PRIMARY_FLOW_STEPS)
    assert outcome["flow_objective_count"] == 40
    assert outcome["non_worsened_sample_count"] == 6
    with pytest.raises(RuntimeError, match="expected flow grid"):
        multiflow_probe_grid_outcome(
            initial,
            final,
            expected_flow_steps=PHASE_E7_CONTINUITY_FLOW_STEPS,
        )


def test_probe_checks_bind_every_objective_rng_identity() -> None:
    probe = _probe(
        [1.0] * 8,
        ratio=0.0,
        flow_steps=PHASE_E7_PRIMARY_FLOW_STEPS,
        variant="A0",
    )
    for row in probe["per_objective"]:
        row.update(
            _flow_objective_identity(
                base_sample_id=row["base_sample_id"],
                train_seed=3407,
                flow_step=row["flow_step"],
            )
        )
    probe["identity_schedule_sha256"] = (
        probe_identity_schedule_sha256(
            SAMPLE_IDS,
            train_seed=3407,
            flow_steps=PHASE_E7_PRIMARY_FLOW_STEPS,
        )
    )
    probe["max_objective_peak_memory_mib"] = 100.0
    probe["train_seed"] = 3407
    assert all(
        _probe_checks(
            probe,
            flow_steps=PHASE_E7_PRIMARY_FLOW_STEPS,
            expected_zero_weight_positions=(),
            train_seed=3407,
        ).values()
    )
    probe["per_objective"][0]["action_noise_seed"] += 1
    assert not _probe_checks(
        probe,
        flow_steps=PHASE_E7_PRIMARY_FLOW_STEPS,
        expected_zero_weight_positions=(),
        train_seed=3407,
    )["probe_rng_identity_exact"]


def _trajectory_checkpoint(
    *,
    stable: bool,
    final_mean: float,
    non_worsened: int,
) -> dict:
    return {
        "outcome": {
            "final_mean_action_loss": final_mean,
            "non_worsened_sample_count": non_worsened,
        },
        "performance_checks": {"frozen_a0_stability": stable},
    }


@pytest.mark.parametrize(
    ("early_stable", "endpoint_stable", "endpoint_count", "endpoint_mean",
     "expected"),
    (
        (
            True,
            False,
            4,
            1.2,
            "late_overtraining_supported",
        ),
        (
            True,
            True,
            7,
            0.9,
            "not_supported_endpoint_stable",
        ),
        (
            False,
            False,
            4,
            1.2,
            "not_supported_no_earlier_stable_checkpoint",
        ),
        (
            True,
            False,
            6,
            1.2,
            "not_supported_no_material_late_degradation",
        ),
    ),
)
def test_a0_trajectory_classification_is_frozen(
    early_stable,
    endpoint_stable,
    endpoint_count,
    endpoint_mean,
    expected,
) -> None:
    checkpoints = {
        step: _trajectory_checkpoint(
            stable=(early_stable if step < 200 else endpoint_stable),
            final_mean=(1.0 if step < 200 else endpoint_mean),
            non_worsened=(7 if step < 200 else endpoint_count),
        )
        for step in PHASE_E7_CHECKPOINT_STEPS
    }
    result = classify_a0_trajectory(checkpoints)
    assert result["classification"] == expected
    assert result["late_material_degradation"] == (
        expected == "late_overtraining_supported"
    )


def _candidate_checkpoint(
    variant: str,
    *,
    reduction: float,
    final_mean: float,
    final_losses: list[float],
) -> dict:
    outcome = {
        "catastrophic_sample_count": 0,
        "final_mean_action_loss": final_mean,
        "loss_reduction_fraction": reduction,
        "max_objective_gated_delta_to_action_hidden_ratio": 0.2,
        "median_gated_delta_to_action_hidden_ratio": 0.1,
        "non_worsened_sample_count": 8,
        "per_sample": [
            {
                "base_sample_id": sample_id,
                "final_action_loss": loss,
            }
            for sample_id, loss in zip(
                SAMPLE_IDS,
                final_losses,
                strict=True,
            )
        ],
    }
    return {
        "outcome": outcome,
        "performance_checks": replication_performance_checks(
            variant,
            {"outcome": outcome},
        ),
    }


def test_joint_candidate_is_earliest_and_post_run_only() -> None:
    tracks = {"A0": {}, "A1": {}}
    for step in PHASE_E7_CHECKPOINT_STEPS:
        tracks["A0"][step] = _candidate_checkpoint(
            "A0",
            reduction=0.01,
            final_mean=1.0,
            final_losses=[1.0] * 8,
        )
        tracks["A1"][step] = _candidate_checkpoint(
            "A1",
            reduction=(0.05 if step == 50 else 0.2),
            final_mean=(0.95 if step == 50 else 0.8),
            final_losses=(
                [0.95] * 8 if step == 50 else [0.8] * 8
            ),
        )
    result = diagnostic_joint_candidate_steps(tracks)
    assert result["diagnostic_candidate_steps"] == [100, 150, 200]
    assert result["earliest_diagnostic_candidate_step"] == 100
    assert (
        result["selection_status"]
        == "post_run_diagnostic_candidate_only"
    )


def test_phase_e7_refuses_before_any_output_write(monkeypatch) -> None:
    cfg = load_thought3_config(CONFIG)
    monkeypatch.delenv("CONFIRM_THOUGHT3_PHASE_E7", raising=False)
    tracked = [
        PHASE_E7_ROOT / "run_status.json",
        PHASE_E7_ROOT / "gate_e7_result.json",
    ]
    before = {
        path: path.read_bytes() if path.is_file() else None
        for path in tracked
    }
    with pytest.raises(RuntimeError, match="CONFIRM_THOUGHT3_PHASE_E7"):
        run_phase_e7_checkpoint_trajectory(cfg)
    after = {
        path: path.read_bytes() if path.is_file() else None
        for path in tracked
    }
    assert after == before


def test_phase_e7_recipe_drift_and_partial_checkpoint_set_fail_closed(
    frozen_parent,
) -> None:
    cfg = load_thought3_config(CONFIG)
    with pytest.raises(RuntimeError, match="changes more"):
        _assert_phase_e7_scope(
            replace(
                cfg,
                training=replace(cfg.training, learning_rate=1e-4),
            )
        )
    with pytest.raises(RuntimeError, match="checkpoint set changed"):
        classify_a0_trajectory(
            {
                50: _trajectory_checkpoint(
                    stable=True,
                    final_mean=1.0,
                    non_worsened=8,
                )
            }
        )
    swapped = list(frozen_parent["sample_ids"])
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert (
        probe_identity_schedule_sha256(
            swapped,
            train_seed=3407,
            flow_steps=PHASE_E7_PRIMARY_FLOW_STEPS,
        )
        != PHASE_E7_PRIMARY_IDENTITY_SCHEDULE_SHA256
    )


def test_phase_e7_orchestrator_has_no_training_or_outcome_api() -> None:
    source = inspect.getsource(_run_phase_e7)
    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert "save_adapter_checkpoint" not in source
    signature = inspect.signature(run_phase_e7_checkpoint_trajectory)
    assert set(signature.parameters) == {"cfg", "resume"}
