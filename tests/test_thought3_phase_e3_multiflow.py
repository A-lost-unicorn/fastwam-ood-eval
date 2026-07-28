from __future__ import annotations

import inspect

import pytest

from fastwam_ood_eval.thought3.config import load_thought3_config
from fastwam_ood_eval.thought3.phase_e2_eight_sample import (
    performance_checks,
)
from fastwam_ood_eval.thought3.phase_e3_multiflow import (
    PHASE_E2_FROZEN_ARTIFACTS,
    PHASE_E3_FLOW_STEPS,
    PHASE_E3_SCHEMA,
    _assert_phase_e3_scope,
    _run_phase_e3,
    run_phase_e3_multiflow,
)
from fastwam_ood_eval.thought3.real_training import (
    aggregate_multiflow_probe_rows,
    evaluate_multiflow_subset_probe,
    multiflow_subset_outcome,
)


SAMPLE_IDS = tuple(f"sample-{index}" for index in range(8))


def _rows(
    *,
    sample_losses: list[float],
    ratio: float,
) -> list[dict]:
    return [
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
        for flow_step in PHASE_E3_FLOW_STEPS
    ]


def _probe(sample_losses: list[float], ratio: float) -> dict:
    return aggregate_multiflow_probe_rows(
        _rows(sample_losses=sample_losses, ratio=ratio),
        sample_ids=SAMPLE_IDS,
        flow_steps=PHASE_E3_FLOW_STEPS,
        variant="A1",
    )


def test_phase_e3_protocol_is_heldout_and_scope_is_frozen() -> None:
    assert PHASE_E3_SCHEMA == "thought3.phase_e3.multiflow.v2"
    assert PHASE_E3_FLOW_STEPS == (1, 2, 3, 4, 5)
    assert 0 not in PHASE_E3_FLOW_STEPS
    assert PHASE_E2_FROZEN_ARTIFACTS == {
        "gate_e2_result.json": (
            "40f66bc50acd8e175ecb61ec150a04ef9ed5c55bf1fa9090802cc529104214bb"
        ),
        "run_status.json": (
            "570774031d338ee27754f460c46deaf2a12f77d39e1b68cd3b08cb6af1a91e58"
        ),
        "pre_validation_result.json": (
            "7aa98cfb95fbc73ab409ef47545e8a912ae221586fe57f2afa841676c6a9a7bb"
        ),
        "data_preparation.json": (
            "fb92b8c7f01129689c5a4ddd7ab96aaa184687dcec15b07b9f180d049dc01b4e"
        ),
    }
    cfg = load_thought3_config(
        "configs/thought3/phase_e3_multiflow_diagnostic_v2.yaml"
    )
    _assert_phase_e3_scope(cfg)
    legacy_cfg = load_thought3_config(
        "configs/thought3/phase_e3_multiflow_diagnostic.yaml"
    )
    with pytest.raises(RuntimeError, match="experiment name changed"):
        _assert_phase_e3_scope(legacy_cfg)


def test_phase_e3_legacy_config_is_rejected_before_output_write(
    monkeypatch,
) -> None:
    legacy_cfg = load_thought3_config(
        "configs/thought3/phase_e3_multiflow_diagnostic.yaml"
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy Gate E.3 attempted an output write")

    monkeypatch.setattr(
        "fastwam_ood_eval.thought3.phase_e3_multiflow."
        "atomic_write_json",
        forbidden,
    )
    with pytest.raises(RuntimeError, match="experiment name changed"):
        run_phase_e3_multiflow(legacy_cfg)


def test_multiflow_aggregation_and_frozen_gate() -> None:
    initial = _probe([1.0] * 8, ratio=0.0)
    final = _probe([0.8] * 6 + [1.05] * 2, ratio=0.4)
    assert initial["flow_objective_count"] == 40
    assert len(initial["per_sample"]) == 8
    assert len(initial["per_objective"]) == 40
    outcome = multiflow_subset_outcome(initial, final)
    assert outcome["loss_reduction_fraction"] == pytest.approx(0.1375)
    assert outcome["non_worsened_sample_count"] == 6
    assert outcome["catastrophic_sample_count"] == 0
    assert outcome["max_objective_loss_ratio"] == pytest.approx(1.05)
    assert all(performance_checks({"outcome": outcome}).values())

    corrupted = dict(final)
    corrupted["mean_action_loss"] = 0.5
    with pytest.raises(RuntimeError, match="differs from objective rows"):
        multiflow_subset_outcome(initial, corrupted)


def test_multiflow_allows_official_zero_weight_endpoint() -> None:
    initial_rows = _rows(sample_losses=[1.0] * 8, ratio=0.0)
    final_rows = _rows(sample_losses=[0.8] * 8, ratio=0.4)
    for rows in (initial_rows, final_rows):
        endpoint = rows[-1]
        endpoint["action_loss"] = 0.0
        endpoint["action_weight"] = 0.0
        endpoint["timestep"] = 1000.0
    initial = aggregate_multiflow_probe_rows(
        initial_rows,
        sample_ids=SAMPLE_IDS,
        flow_steps=PHASE_E3_FLOW_STEPS,
        variant="A1",
    )
    final = aggregate_multiflow_probe_rows(
        final_rows,
        sample_ids=SAMPLE_IDS,
        flow_steps=PHASE_E3_FLOW_STEPS,
        variant="A1",
    )

    outcome = multiflow_subset_outcome(initial, final)

    assert initial["zero_weight_objective_count"] == 1
    assert initial["zero_action_loss_objective_count"] == 1
    assert outcome["objective_loss_ratio_count"] == 39
    assert outcome["zero_initial_loss_objective_count"] == 1
    assert outcome["zero_weight_objective_count"] == 1
    assert (
        outcome["positive_final_from_zero_initial_loss_count"] == 0
    )
    assert outcome["max_objective_loss_ratio"] == pytest.approx(0.8)


def test_multiflow_rejects_nonzero_loss_at_zero_weight() -> None:
    rows = _rows(sample_losses=[1.0] * 8, ratio=0.0)
    rows[-1]["action_weight"] = 0.0
    with pytest.raises(RuntimeError, match="invalid"):
        aggregate_multiflow_probe_rows(
            rows,
            sample_ids=SAMPLE_IDS,
            flow_steps=PHASE_E3_FLOW_STEPS,
            variant="A1",
        )


def test_multiflow_grid_rejects_missing_or_duplicate_objectives() -> None:
    rows = _rows(sample_losses=[1.0] * 8, ratio=0.0)
    with pytest.raises(RuntimeError, match="grid mismatch"):
        aggregate_multiflow_probe_rows(
            rows[:-1],
            sample_ids=SAMPLE_IDS,
            flow_steps=PHASE_E3_FLOW_STEPS,
            variant="A0",
        )
    with pytest.raises(RuntimeError, match="duplicate"):
        aggregate_multiflow_probe_rows(
            [*rows, rows[0]],
            sample_ids=SAMPLE_IDS,
            flow_steps=PHASE_E3_FLOW_STEPS,
            variant="A0",
        )


def test_phase_e3_refuses_without_confirmation_before_model_load(
    monkeypatch,
) -> None:
    cfg = load_thought3_config(
        "configs/thought3/phase_e3_multiflow_diagnostic_v2.yaml"
    )
    monkeypatch.delenv(
        "CONFIRM_THOUGHT3_PHASE_E3_V2",
        raising=False,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Gate E.3 loaded Fast-WAM before confirmation")

    monkeypatch.setattr(
        "fastwam_ood_eval.thought3.phase_e3_multiflow."
        "_load_upstream_model",
        forbidden,
    )
    with pytest.raises(
        RuntimeError,
        match="CONFIRM_THOUGHT3_PHASE_E3_V2",
    ):
        _run_phase_e3(cfg)


def test_multiflow_probe_api_has_no_dev_or_outcome_input() -> None:
    parameters = set(
        inspect.signature(
            evaluate_multiflow_subset_probe
        ).parameters
    )
    assert parameters == {
        "adapter",
        "cfg",
        "device",
        "flow_steps",
        "injector",
        "model",
        "samples",
    }
    assert not parameters & {
        "development",
        "future_rgb",
        "ood",
        "success",
    }
