from __future__ import annotations

import inspect
import json
import statistics
from dataclasses import replace

import pytest
import torch

from fastwam_ood_eval.thought3.config import load_thought3_config
from fastwam_ood_eval.thought3.objective_aggregation_training import (
    OBJECTIVE_AGGREGATION_EXPECTED_ZERO_WEIGHT_SLOTS,
    OBJECTIVE_AGGREGATION_FLOW_SLOT_OFFSET,
    OBJECTIVE_AGGREGATION_HELDOUT_FLOW_STEPS,
    OBJECTIVE_AGGREGATION_UPDATES,
    OBJECTIVES_PER_UPDATE,
    _backward_mean_objective,
    _validate_resume_metric_provenance,
    objective_aggregation_flow_slot,
    objective_aggregation_identity_schedule_sha256,
    objective_aggregation_metric_rows_sha256,
    objective_aggregation_schedule_sha256,
    run_full_cohort_objective_aggregation,
)
from fastwam_ood_eval.thought3.phase_e2_eight_sample import (
    PHASE_E2_LR_GRID,
    _matched_recipe_payload,
)
from fastwam_ood_eval.thought3.phase_e5_objective_aggregation import (
    PHASE_E4_FROZEN_ARTIFACTS,
    PHASE_E5_CONFIG_FINGERPRINT,
    PHASE_E5_FROZEN_IDENTITY_SCHEDULE_SHA256,
    PHASE_E5_SCHEMA,
    _assert_phase_e5_scope,
    _run_phase_e5,
    _track_checks,
    derive_e5_track_config,
    run_phase_e5_objective_aggregation,
    verify_frozen_phase_e4,
)
from fastwam_ood_eval.thought3.real_training import (
    _flow_objective_identity,
    aggregate_multiflow_probe_rows,
    multiflow_subset_outcome,
)


CONFIG = "configs/thought3/phase_e5_objective_aggregation_diagnostic.yaml"


def test_phase_e5_scope_and_frozen_parent_evidence() -> None:
    cfg = load_thought3_config(CONFIG)
    _assert_phase_e5_scope(cfg)
    assert cfg.fingerprint == PHASE_E5_CONFIG_FINGERPRINT
    assert cfg.training.gradient_accumulation_steps == 8
    assert PHASE_E5_SCHEMA == "thought3.phase_e5.objective_aggregation.v1"
    assert PHASE_E4_FROZEN_ARTIFACTS == {
        "gate_e4_result.json": (
            "48314003c146327c93e3c5ecb173762cde09c27afb1b38124e741a222e974240"
        ),
        "run_status.json": (
            "8c092f6aedbb67054e6853a49e35ec14f4cd3221b7867df6c72d6ff89a0acc43"
        ),
        "pre_validation_result.json": (
            "4a74f33aa3af211854f86873c933530f904466c776c9ac97c969d7ef99cf8223"
        ),
        "data_preparation.json": (
            "5cb61c57ab52feb93b395e3e3f379411e481f936839251b48048aa492c33a699"
        ),
        "logs/phase_e4.log": (
            "6412697e39c55d5ba2c3232615d03007e69d517dff4b81701a12196814480886"
        ),
    }
    evidence = verify_frozen_phase_e4()
    assert evidence["gate_e4_passed"] is False
    assert len(evidence["sample_ids"]) == 8
    assert (
        evidence["identity_schedule_sha256"]
        == PHASE_E5_FROZEN_IDENTITY_SCHEDULE_SHA256
    )


def test_phase_e5_tracks_change_only_variant_k_and_frozen_lr() -> None:
    cfg = load_thought3_config(CONFIG)
    for lr_slug, learning_rate in PHASE_E2_LR_GRID:
        a0 = derive_e5_track_config(
            cfg,
            variant="A0",
            lr_slug=lr_slug,
            learning_rate=learning_rate,
        )
        a1 = derive_e5_track_config(
            cfg,
            variant="A1",
            lr_slug=lr_slug,
            learning_rate=learning_rate,
        )
        assert _matched_recipe_payload(a0) == _matched_recipe_payload(a1)
        assert (a0.variant, a0.sampler.active_k) == ("A0", 0)
        assert (a1.variant, a1.sampler.active_k) == ("A1", 1)
        assert a0.training.gradient_accumulation_steps == 8
        assert a1.training.gradient_accumulation_steps == 8


def test_objective_aggregation_slots_are_unique_and_disjoint() -> None:
    slots = [
        objective_aggregation_flow_slot(update, micro)
        for update in range(1, OBJECTIVE_AGGREGATION_UPDATES + 1)
        for micro in range(1, OBJECTIVES_PER_UPDATE + 1)
    ]
    assert OBJECTIVE_AGGREGATION_FLOW_SLOT_OFFSET == 20_000
    assert OBJECTIVE_AGGREGATION_HELDOUT_FLOW_STEPS == (1, 2, 3, 4, 5)
    assert slots == list(range(20_001, 21_601))
    assert len(set(slots)) == 1600
    assert not set(slots) & {0, *OBJECTIVE_AGGREGATION_HELDOUT_FLOW_STEPS}
    assert not set(slots) & set(range(10_001, 10_201))
    with pytest.raises(RuntimeError, match="1..200"):
        objective_aggregation_flow_slot(0, 1)
    with pytest.raises(RuntimeError, match="1..8"):
        objective_aggregation_flow_slot(1, 9)


def test_identity_schedule_is_preregistered_and_stable() -> None:
    sample_ids = verify_frozen_phase_e4()["sample_ids"]
    assert (
        objective_aggregation_identity_schedule_sha256(
            sample_ids,
            train_seed=3407,
        )
        == PHASE_E5_FROZEN_IDENTITY_SCHEDULE_SHA256
        == "b6f9778d303a6ad2c4bef781f4a6027a800d013814110daa47eb7cb1d13af86d"
    )
    swapped = list(sample_ids)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert (
        objective_aggregation_identity_schedule_sha256(
            swapped,
            train_seed=3407,
        )
        != PHASE_E5_FROZEN_IDENTITY_SCHEDULE_SHA256
    )


def test_frozen_schedule_zero_weight_slots_are_known_before_run() -> None:
    sample_ids = verify_frozen_phase_e4()["sample_ids"]
    zero_slots = []
    for update in range(1, OBJECTIVE_AGGREGATION_UPDATES + 1):
        for micro, base_sample_id in enumerate(sample_ids, start=1):
            flow_slot = objective_aggregation_flow_slot(update, micro)
            identity = _flow_objective_identity(
                base_sample_id=base_sample_id,
                train_seed=3407,
                flow_step=flow_slot,
            )
            generator = torch.Generator(device="cpu").manual_seed(
                identity["action_timestep_seed"]
            )
            u = torch.rand((1,), generator=generator, dtype=torch.float32)
            sigma = 5.0 * u / (1.0 + 4.0 * u)
            timestep = (sigma * 1000.0).to(dtype=torch.bfloat16)
            if float(timestep.float()) == 1000.0:
                zero_slots.append((update, micro, flow_slot))
    assert tuple(zero_slots) == (
        OBJECTIVE_AGGREGATION_EXPECTED_ZERO_WEIGHT_SLOTS
    )
    assert len(zero_slots) == 24


def test_mean_objective_backward_is_an_arithmetic_mean() -> None:
    parameter = torch.tensor(2.0, requires_grad=True)
    for coefficient in range(1, OBJECTIVES_PER_UPDATE + 1):
        _backward_mean_objective(
            parameter * coefficient,
            accumulation_factor=OBJECTIVES_PER_UPDATE,
        )
    assert parameter.grad is not None
    assert float(parameter.grad) == statistics.fmean(range(1, 9))
    with pytest.raises(RuntimeError, match="scalar loss"):
        _backward_mean_objective(
            torch.ones(2, requires_grad=True),
            accumulation_factor=8,
        )


def test_phase_e5_refuses_without_confirmation_before_model_load(
    monkeypatch,
) -> None:
    cfg = load_thought3_config(CONFIG)
    monkeypatch.delenv("CONFIRM_THOUGHT3_PHASE_E5", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Gate E.5 loaded Fast-WAM before confirmation")

    monkeypatch.setattr(
        "fastwam_ood_eval.thought3.phase_e5_objective_aggregation."
        "_load_upstream_model",
        forbidden,
    )
    with pytest.raises(RuntimeError, match="CONFIRM_THOUGHT3_PHASE_E5"):
        _run_phase_e5(cfg, resume=False)


def test_public_phase_e5_refuses_before_run_state_write(
    monkeypatch,
) -> None:
    cfg = load_thought3_config(CONFIG)
    monkeypatch.delenv("CONFIRM_THOUGHT3_PHASE_E5", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Gate E.5 entered root execution without consent")

    monkeypatch.setattr(
        "fastwam_ood_eval.thought3.phase_e5_objective_aggregation."
        "_run_phase_e5",
        forbidden,
    )
    with pytest.raises(RuntimeError, match="CONFIRM_THOUGHT3_PHASE_E5"):
        run_phase_e5_objective_aggregation(cfg, resume=False)


def _fake_probe(
    sample_ids: list[str],
    *,
    loss: float,
    ratio: float,
    variant: str,
) -> dict:
    rows = []
    for sample_index, base_sample_id in enumerate(sample_ids):
        for flow_step in OBJECTIVE_AGGREGATION_HELDOUT_FLOW_STEPS:
            endpoint = sample_index == 7 and flow_step == 5
            rows.append(
                {
                    "action_hidden_norm": 10.0,
                    "action_loss": 0.0 if endpoint else loss,
                    "action_weight": 0.0 if endpoint else 1.0,
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
                    "timestep": (
                        1000.0 if endpoint else 100.0 + flow_step
                    ),
                }
            )
    return aggregate_multiflow_probe_rows(
        rows,
        sample_ids=sample_ids,
        flow_steps=OBJECTIVE_AGGREGATION_HELDOUT_FLOW_STEPS,
        variant=variant,
    )


def test_phase_e5_track_checks_cover_full_cohort_and_mean(
    tmp_path,
) -> None:
    base_cfg = load_thought3_config(CONFIG)
    output = tmp_path / "thought3" / "track"
    cfg = replace(
        derive_e5_track_config(
            base_cfg,
            variant="A1",
            lr_slug="lr_1e_04",
            learning_rate=1e-4,
        ),
        experiment=replace(
            base_cfg.experiment,
            name="fake_phase_e5_a1",
            output_dir=output,
        ),
    )
    output.mkdir(parents=True)
    sample_ids = verify_frozen_phase_e4()["sample_ids"]
    zero_slot_set = set(
        OBJECTIVE_AGGREGATION_EXPECTED_ZERO_WEIGHT_SLOTS
    )
    objective_rows = []
    update_rows = []
    sample_cursor = 0
    for update in range(1, OBJECTIVE_AGGREGATION_UPDATES + 1):
        cohort = []
        for micro, base_sample_id in enumerate(sample_ids, start=1):
            sample_cursor += 1
            flow_slot = objective_aggregation_flow_slot(update, micro)
            identity = _flow_objective_identity(
                base_sample_id=base_sample_id,
                train_seed=3407,
                flow_step=flow_slot,
            )
            zero_weight = (update, micro, flow_slot) in zero_slot_set
            action_loss = 0.0 if zero_weight else 1.0
            row = {
                **identity,
                "action_hidden_norm": 10.0,
                "action_loss": action_loss,
                "action_weight": 0.0 if zero_weight else 1.0,
                "attention_residual_norm": 2.0,
                "base_sample_id": base_sample_id,
                "cohort_sample_index": micro - 1,
                "future_token_norm": 3.0,
                "gate_gradient_contribution_mean_scaled": 0.125,
                "gate_gradient_contribution_sign": 1,
                "gate_gradient_contribution_unscaled": 1.0,
                "gate_gradient_cumulative": micro * 0.125,
                "gate_raw_after_update": 0.1,
                "gate_raw_before_update": 0.0 if update == 1 else 0.1,
                "gated_delta_norm": 1.0,
                "gated_delta_to_action_hidden_ratio": 0.1,
                "gradient_reduction": "arithmetic_mean",
                "mean_scaled_backward_loss": action_loss / 8,
                "micro_index": micro,
                "nan_or_inf": False,
                "objective_index": sample_cursor,
                "optimizer_update_peak_memory_mib": 100.0,
                "optimizer_update_time_ms": 10.0,
                "optimizer_update": update,
                "sample_cursor": sample_cursor,
                "timestep": 1000.0 if zero_weight else 500.0,
                "training_flow_slot": flow_slot,
                "zero_weight_objective": zero_weight,
            }
            cohort.append(row)
            objective_rows.append(row)
        groups = {
            name: {
                "finite": True,
                "l2": (
                    1.0
                    if name == "gate"
                    or (
                        update >= 2
                        and name
                        in {
                            "attention",
                            "future_projector",
                            "non_gate",
                        }
                    )
                    else 0.0
                ),
                "nonzero_element_count": (
                    0
                    if update == 1 and name == "non_gate"
                    else 1
                ),
            }
            for name in (
                "all",
                "attention",
                "future_projector",
                "gate",
                "non_gate",
            )
        }
        raw_losses = [float(row["action_loss"]) for row in cohort]
        weights = [float(row["action_weight"]) for row in cohort]
        update_rows.append(
            {
                "action_weight_mean": statistics.fmean(weights),
                "action_weight_sum": sum(weights),
                "gate_gradient": 1.0,
                "gate_gradient_absolute_contribution_sum": 1.0,
                "gate_gradient_cancellation_ratio": 1.0,
                "gate_gradient_sign": 1,
                "gate_raw_after_update": 0.1,
                "gate_raw_before_update": 0.0 if update == 1 else 0.1,
                "gradient_groups": groups,
                "gradient_reduction": "arithmetic_mean",
                "mean_action_loss": statistics.fmean(raw_losses),
                "nan_or_inf": False,
                "objective_count": 8,
                "objective_index_end": update * 8,
                "objective_index_start": (update - 1) * 8 + 1,
                "optimizer_update": update,
                "peak_memory_mib": 100.0,
                "sample_cursor": update * 8,
                "summed_action_loss": sum(raw_losses),
                "update_time_ms": 10.0,
                "zero_weight_objective_count": sum(
                    weight == 0 for weight in weights
                ),
            }
        )

    initial = _fake_probe(
        sample_ids,
        loss=1.0,
        ratio=0.0,
        variant="A1",
    )
    final = _fake_probe(
        sample_ids,
        loss=0.8,
        ratio=0.1,
        variant="A1",
    )
    initial_row = {**initial, "global_step": 0, "learning_rate": 1e-4}
    final_row = {
        **final,
        "global_step": OBJECTIVE_AGGREGATION_UPDATES,
        "learning_rate": 1e-4,
    }
    objective_path = output / "train_objective_metrics.jsonl"
    update_path = output / "train_update_metrics.jsonl"
    probe_path = output / "heldout_multiflow_metrics.jsonl"
    objective_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in objective_rows
        )
    )
    update_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in update_rows
        )
    )
    probe_path.write_text(
        json.dumps(initial_row, sort_keys=True)
        + "\n"
        + json.dumps(final_row, sort_keys=True)
        + "\n"
    )
    (output / "training_manifest.json").write_text("{}\n")
    (output / "training_state.json").write_text("{}\n")
    result = {
        "checkpoint_roundtrip": {
            "global_step": OBJECTIVE_AGGREGATION_UPDATES,
            "state_equal": True,
        },
        "completed_objectives": 1600,
        "completed_steps": OBJECTIVE_AGGREGATION_UPDATES,
        "final_probe": final_row,
        "first_attention_nonzero_gradient_update": 2,
        "first_non_gate_nonzero_gradient_update": 2,
        "first_projector_nonzero_gradient_update": 2,
        "identity_schedule_sha256": (
            PHASE_E5_FROZEN_IDENTITY_SCHEDULE_SHA256
        ),
        "initial_probe": initial_row,
        "max_peak_memory_mib": 100.0,
        "objective_metrics": str(objective_path),
        "optimizer_parameter_scope": "adapter_only",
        "outcome": multiflow_subset_outcome(initial, final),
        "probe_metrics": str(probe_path),
        "sample_count": 8,
        "sample_ids": sample_ids,
        "status": "complete",
        "train_flow_schedule_sha256": (
            objective_aggregation_schedule_sha256(objective_rows)
        ),
        "update_metrics": str(update_path),
        "uses_development_outcomes": False,
        "uses_ground_truth_future_input": False,
        "uses_ood_or_success_outcomes": False,
        "zero_weight_objective_count": 24,
        "zero_weight_slots": [
            list(value)
            for value in OBJECTIVE_AGGREGATION_EXPECTED_ZERO_WEIGHT_SLOTS
        ],
    }
    checks, artifacts = _track_checks(cfg, result)
    assert all(checks.values()), {
        key: value for key, value in checks.items() if not value
    }
    assert (
        artifacts["identity_schedule_sha256"]
        == PHASE_E5_FROZEN_IDENTITY_SCHEDULE_SHA256
    )


def test_schedule_hash_rejects_partial_or_corrupt_cohorts() -> None:
    with pytest.raises(RuntimeError, match="full-update prefix"):
        objective_aggregation_schedule_sha256([])
    rows = []
    for micro in range(1, 9):
        identity = _flow_objective_identity(
            base_sample_id=f"sample-{micro}",
            train_seed=3407,
            flow_step=objective_aggregation_flow_slot(1, micro),
        )
        rows.append(
            {
                **identity,
                "action_weight": 1.0,
                "base_sample_id": f"sample-{micro}",
                "cohort_sample_index": micro - 1,
                "micro_index": micro,
                "objective_index": micro,
                "optimizer_update": 1,
                "sample_cursor": micro,
                "timestep": 500.0,
                "training_flow_slot": (
                    objective_aggregation_flow_slot(1, micro)
                ),
            }
        )
    assert len(objective_aggregation_schedule_sha256(rows)) == 64
    rows[4]["micro_index"] = 4
    with pytest.raises(RuntimeError, match="not contiguous"):
        objective_aggregation_schedule_sha256(rows)


def test_resume_manifest_binds_both_metric_prefixes() -> None:
    objective_rows = []
    sample_ids = [f"sample-{index}" for index in range(8)]
    for micro, base_sample_id in enumerate(sample_ids, start=1):
        flow_slot = objective_aggregation_flow_slot(1, micro)
        identity = _flow_objective_identity(
            base_sample_id=base_sample_id,
            train_seed=3407,
            flow_step=flow_slot,
        )
        objective_rows.append(
            {
                **identity,
                "action_weight": 1.0,
                "base_sample_id": base_sample_id,
                "cohort_sample_index": micro - 1,
                "micro_index": micro,
                "objective_index": micro,
                "optimizer_update": 1,
                "sample_cursor": micro,
                "timestep": 500.0,
                "training_flow_slot": flow_slot,
            }
        )
    update_rows = [{"optimizer_update": 1, "mean_action_loss": 1.0}]
    identity_sha = objective_aggregation_identity_schedule_sha256(
        sample_ids,
        train_seed=3407,
        update_count=1,
    )
    extra = {
        "gate_e5_objective_aggregation": True,
        "gradient_reduction": "arithmetic_mean",
        "heldout_flow_steps": [1, 2, 3, 4, 5],
        "identity_schedule_sha256": identity_sha,
        "objective_count": 8,
        "objective_metrics_prefix_sha256": (
            objective_aggregation_metric_rows_sha256(objective_rows)
        ),
        "objectives_per_update": 8,
        "train_flow_schedule_sha256": (
            objective_aggregation_schedule_sha256(objective_rows)
        ),
        "training_flow_slot_offset": 20_000,
        "update_metrics_prefix_sha256": (
            objective_aggregation_metric_rows_sha256(update_rows)
        ),
    }
    _validate_resume_metric_provenance(
        global_step=1,
        sample_cursor=8,
        extra=extra,
        objective_rows=objective_rows,
        update_rows=update_rows,
        identity_schedule_sha256=identity_sha,
    )
    corrupted = [dict(row) for row in update_rows]
    corrupted[0]["mean_action_loss"] = 2.0
    with pytest.raises(RuntimeError, match="metric-prefix provenance"):
        _validate_resume_metric_provenance(
            global_step=1,
            sample_cursor=8,
            extra=extra,
            objective_rows=objective_rows,
            update_rows=corrupted,
            identity_schedule_sha256=identity_sha,
        )


def test_objective_aggregation_api_has_no_outcome_input() -> None:
    parameters = set(
        inspect.signature(
            run_full_cohort_objective_aggregation
        ).parameters
    )
    assert parameters == {
        "cfg",
        "device",
        "frozen_parameter_sha256",
        "model",
        "prepared",
        "progress",
        "resume",
    }
    assert not parameters & {
        "development",
        "future_rgb",
        "ood",
        "success",
    }
