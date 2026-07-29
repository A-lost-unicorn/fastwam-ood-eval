from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fastwam_ood_eval.thought3.config import load_thought3_config
from fastwam_ood_eval.thought3.objective_aggregation_training import (
    OBJECTIVE_AGGREGATION_UPDATES,
    OBJECTIVES_PER_UPDATE,
    _validate_resume_metric_provenance,
    objective_aggregation_flow_slot,
    objective_aggregation_identity_schedule_sha256,
)
from fastwam_ood_eval.thought3.phase_e2_eight_sample import (
    _matched_recipe_payload,
)
from fastwam_ood_eval.thought3.phase_e6_fresh_cohort_replication import (
    PHASE_E6_CONFIG_FINGERPRINT,
    PHASE_E6_EXPECTED_ZERO_WEIGHT_SLOTS,
    PHASE_E6_FLOW_SLOT_OFFSET,
    PHASE_E6_FROZEN_COHORT_SHA256,
    PHASE_E6_FROZEN_IDENTITY_SCHEDULE_SHA256,
    PHASE_E6_LEARNING_RATE,
    PHASE_E6_PROTOCOL,
    PHASE_E6_ROOT,
    PHASE_E6_SCHEMA,
    _assert_phase_e6_scope,
    derive_e6_track_config,
    paired_superiority_checks,
    replication_performance_checks,
    run_phase_e6_fresh_cohort_replication,
    verify_frozen_fresh_cohort,
    verify_frozen_phase_e5,
)
from fastwam_ood_eval.thought3.real_training import (
    _flow_objective_identity,
)


CONFIG = "configs/thought3/phase_e6_fresh_cohort_replication.yaml"


def test_phase_e6_scope_and_post_selected_parent_are_frozen() -> None:
    cfg = load_thought3_config(CONFIG)
    _assert_phase_e6_scope(cfg)
    assert cfg.fingerprint == PHASE_E6_CONFIG_FINGERPRINT
    assert cfg.training.learning_rate == PHASE_E6_LEARNING_RATE
    assert cfg.experiment.output_dir == PHASE_E6_ROOT
    assert PHASE_E6_SCHEMA == "thought3.phase_e6.fresh_cohort_replication.v1"
    parent = verify_frozen_phase_e5()
    assert parent["gate_e5_passed"] is False
    assert parent["post_selection"] is True
    assert parent["selected_by_e5_gate"] is False
    assert "after inspecting Gate E.5" in parent["learning_rate_source"]


def test_phase_e6_fresh_cohort_is_exact_and_disjoint() -> None:
    cfg = load_thought3_config(CONFIG)
    parent = verify_frozen_phase_e5()
    cohort = verify_frozen_fresh_cohort(
        cfg,
        e5_sample_ids=parent["sample_ids"],
    )
    assert cohort["cohort_sha256"] == PHASE_E6_FROZEN_COHORT_SHA256
    assert cohort["e5_overlap_count"] == 0
    assert cohort["development_overlap_count"] == 0
    assert len(cohort["sample_ids"]) == 8
    assert len({row["demonstration_id"] for row in cohort["samples"]}) == 8
    assert (
        objective_aggregation_identity_schedule_sha256(
            cohort["sample_ids"],
            train_seed=3407,
            flow_slot_offset=PHASE_E6_FLOW_SLOT_OFFSET,
        )
        == PHASE_E6_FROZEN_IDENTITY_SCHEDULE_SHA256
    )


def test_phase_e6_uses_only_matched_a0_a1_at_3e4() -> None:
    cfg = load_thought3_config(CONFIG)
    a0 = derive_e6_track_config(cfg, variant="A0")
    a1 = derive_e6_track_config(cfg, variant="A1")
    assert (a0.variant, a0.sampler.active_k) == ("A0", 0)
    assert (a1.variant, a1.sampler.active_k) == ("A1", 1)
    assert a0.training.learning_rate == a1.training.learning_rate == 3e-4
    assert _matched_recipe_payload(a0) == _matched_recipe_payload(a1)
    with pytest.raises(RuntimeError, match="unsupported"):
        derive_e6_track_config(cfg, variant="A2")


def test_phase_e6_flow_namespace_and_zero_slots_are_preregistered() -> None:
    slots = [
        objective_aggregation_flow_slot(
            update,
            micro,
            flow_slot_offset=PHASE_E6_FLOW_SLOT_OFFSET,
        )
        for update in range(1, OBJECTIVE_AGGREGATION_UPDATES + 1)
        for micro in range(1, OBJECTIVES_PER_UPDATE + 1)
    ]
    assert slots == list(range(31_001, 32_601))
    assert not set(slots) & set(range(10_001, 10_201))
    assert not set(slots) & set(range(20_001, 21_601))
    assert not set(slots) & {0, 1, 2, 3, 4, 5}
    cohort = verify_frozen_fresh_cohort(
        load_thought3_config(CONFIG),
        e5_sample_ids=verify_frozen_phase_e5()["sample_ids"],
    )
    observed = []
    for update in range(1, OBJECTIVE_AGGREGATION_UPDATES + 1):
        for micro, base_sample_id in enumerate(
            cohort["sample_ids"],
            start=1,
        ):
            slot = objective_aggregation_flow_slot(
                update,
                micro,
                flow_slot_offset=PHASE_E6_FLOW_SLOT_OFFSET,
            )
            identity = _flow_objective_identity(
                base_sample_id=base_sample_id,
                train_seed=3407,
                flow_step=slot,
            )
            generator = torch.Generator(device="cpu").manual_seed(
                identity["action_timestep_seed"]
            )
            u = torch.rand((1,), generator=generator, dtype=torch.float32)
            sigma = 5.0 * u / (1.0 + 4.0 * u)
            timestep = (sigma * 1000.0).to(dtype=torch.bfloat16)
            if float(timestep.float()) == 1000.0:
                observed.append((update, micro, slot))
    assert tuple(observed) == PHASE_E6_EXPECTED_ZERO_WEIGHT_SLOTS
    assert all(update > 2 for update, _, _ in observed)


def _result(
    *,
    reduction: float,
    final_mean: float,
    final_losses: list[float],
    non_worsened: int = 8,
) -> dict:
    return {
        "outcome": {
            "catastrophic_sample_count": 0,
            "final_mean_action_loss": final_mean,
            "loss_reduction_fraction": reduction,
            "max_objective_gated_delta_to_action_hidden_ratio": 0.2,
            "median_gated_delta_to_action_hidden_ratio": 0.1,
            "non_worsened_sample_count": non_worsened,
            "per_sample": [
                {
                    "base_sample_id": f"sample-{index}",
                    "final_action_loss": loss,
                }
                for index, loss in enumerate(final_losses)
            ],
        }
    }


def test_phase_e6_gate_separates_a0_a1_and_paired_superiority() -> None:
    a0 = _result(
        reduction=0.01,
        final_mean=1.0,
        final_losses=[1.0] * 8,
    )
    a1 = _result(
        reduction=0.12,
        final_mean=0.85,
        final_losses=[0.8] * 6 + [1.1, 1.1],
    )
    assert all(replication_performance_checks("A0", a0).values())
    assert all(replication_performance_checks("A1", a1).values())
    checks, values = paired_superiority_checks(a0, a1)
    assert all(checks.values())
    assert values["a1_non_higher_sample_count"] == 6
    failed_a0 = _result(
        reduction=-0.001,
        final_mean=1.0,
        final_losses=[1.0] * 8,
    )
    assert not replication_performance_checks("A0", failed_a0)[
        "a0_mean_loss_does_not_worsen"
    ]


def test_phase_e6_refuses_before_any_run_state_write(
    monkeypatch,
) -> None:
    cfg = load_thought3_config(CONFIG)
    monkeypatch.delenv("CONFIRM_THOUGHT3_PHASE_E6", raising=False)
    result_path = PHASE_E6_ROOT / "gate_e6_result.json"
    result_before = (
        result_path.read_bytes() if result_path.is_file() else None
    )
    with pytest.raises(RuntimeError, match="CONFIRM_THOUGHT3_PHASE_E6"):
        run_phase_e6_fresh_cohort_replication(cfg)
    result_after = (
        result_path.read_bytes() if result_path.is_file() else None
    )
    assert result_after == result_before


def test_phase_e6_resume_marker_is_protocol_specific() -> None:
    with pytest.raises(RuntimeError, match="metric-prefix provenance"):
        _validate_resume_metric_provenance(
            global_step=0,
            sample_cursor=0,
            extra={
                "gate_e5_objective_aggregation": True,
                "gradient_reduction": "arithmetic_mean",
                "heldout_flow_steps": [1, 2, 3, 4, 5],
                "identity_schedule_sha256": (
                    PHASE_E6_FROZEN_IDENTITY_SCHEDULE_SHA256
                ),
                "objective_count": 0,
                "objective_metrics_prefix_sha256": "",
                "objectives_per_update": 8,
                "train_flow_schedule_sha256": "",
                "training_flow_slot_offset": 31_000,
                "update_metrics_prefix_sha256": "",
            },
            objective_rows=[],
            update_rows=[],
            identity_schedule_sha256=(
                PHASE_E6_FROZEN_IDENTITY_SCHEDULE_SHA256
            ),
            protocol=PHASE_E6_PROTOCOL,
        )


def test_phase_e6_scope_rejects_recipe_drift() -> None:
    cfg = load_thought3_config(CONFIG)
    with pytest.raises(RuntimeError, match="changes more"):
        _assert_phase_e6_scope(
            replace(
                cfg,
                training=replace(cfg.training, learning_rate=1e-4),
            )
        )
