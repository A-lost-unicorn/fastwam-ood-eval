from __future__ import annotations

import inspect

import pytest

from fastwam_ood_eval.thought3.config import load_thought3_config
from fastwam_ood_eval.thought3.phase_e2_eight_sample import (
    PHASE_E2_LR_GRID,
    PHASE_E2_MAX_MEDIAN_DELTA_HIDDEN_RATIO,
    PHASE_E2_MAX_SAMPLE_DELTA_HIDDEN_RATIO,
    PHASE_E2_MIN_LOSS_REDUCTION_FRACTION,
    PHASE_E2_MIN_NON_WORSENED_SAMPLES,
    _assert_phase_e2_scope,
    _initial_probe_signature,
    _matched_recipe_payload,
    _run_phase_e2,
    derive_e2_track_config,
    performance_checks,
    select_smallest_eligible_lr,
)
from fastwam_ood_eval.thought3.real_training import (
    fixed_subset_outcome,
    run_fixed_subset_training,
)


def _probe(
    *,
    loss: float,
    ratio: float,
    residual_offset: float = 0.0,
) -> dict:
    rows = [
        {
            "action_hidden_norm": 10.0,
            "action_loss": loss,
            "attention_residual_norm": residual_offset + index,
            "base_sample_id": f"sample-{index}",
            "gated_delta_nonzero_fraction": (
                0.0 if ratio == 0 else 1.0
            ),
            "gated_delta_norm": ratio * 10.0,
            "gated_delta_to_action_hidden_ratio": ratio,
        }
        for index in range(8)
    ]
    return {
        "max_gated_delta_to_action_hidden_ratio": ratio,
        "mean_action_loss": loss,
        "median_gated_delta_to_action_hidden_ratio": ratio,
        "per_sample": rows,
        "sample_count": 8,
        "sample_ids": [f"sample-{index}" for index in range(8)],
    }


def test_phase_e2_tracks_are_matched_except_preregistered_lr() -> None:
    cfg = load_thought3_config(
        "configs/thought3/phase_e2_eight_sample_diagnostic.yaml"
    )
    _assert_phase_e2_scope(cfg)
    assert PHASE_E2_LR_GRID == (
        ("lr_1e_04", 1e-4),
        ("lr_3e_04", 3e-4),
        ("lr_1e_03", 1e-3),
    )
    for lr_slug, learning_rate in PHASE_E2_LR_GRID:
        a0 = derive_e2_track_config(
            cfg,
            variant="A0",
            lr_slug=lr_slug,
            learning_rate=learning_rate,
        )
        a1 = derive_e2_track_config(
            cfg,
            variant="A1",
            lr_slug=lr_slug,
            learning_rate=learning_rate,
        )
        assert _matched_recipe_payload(a0) == _matched_recipe_payload(a1)
        assert (a0.variant, a0.sampler.active_k) == ("A0", 0)
        assert (a1.variant, a1.sampler.active_k) == ("A1", 1)
        assert (
            a0.training.learning_rate
            == a1.training.learning_rate
            == learning_rate
        )


def test_phase_e2_thresholds_and_smallest_lr_selection_are_frozen() -> None:
    assert PHASE_E2_MIN_LOSS_REDUCTION_FRACTION == 0.10
    assert PHASE_E2_MIN_NON_WORSENED_SAMPLES == 6
    assert PHASE_E2_MAX_MEDIAN_DELTA_HIDDEN_RATIO == 0.50
    assert PHASE_E2_MAX_SAMPLE_DELTA_HIDDEN_RATIO == 1.00
    eligibility = {
        "lr_1e_04": False,
        "lr_3e_04": True,
        "lr_1e_03": True,
    }
    assert select_smallest_eligible_lr(eligibility) == "lr_3e_04"
    assert (
        select_smallest_eligible_lr(
            {slug: False for slug, _ in PHASE_E2_LR_GRID}
        )
        is None
    )


def test_fixed_subset_outcome_and_performance_gate() -> None:
    initial = _probe(loss=1.0, ratio=0.0)
    final = _probe(loss=0.8, ratio=0.4)
    outcome = fixed_subset_outcome(initial, final)
    assert outcome["loss_reduction_fraction"] == pytest.approx(0.2)
    assert outcome["non_worsened_sample_count"] == 8
    assert outcome["catastrophic_sample_count"] == 0
    assert all(performance_checks({"outcome": outcome}).values())

    excessive = _probe(loss=0.8, ratio=1.1)
    excessive_checks = performance_checks(
        {"outcome": fixed_subset_outcome(initial, excessive)}
    )
    assert excessive_checks["max_delta_hidden_at_most_1_0"] is False

    corrupted = _probe(loss=0.8, ratio=0.4)
    corrupted["mean_action_loss"] = 0.7
    with pytest.raises(RuntimeError, match="differs from per-sample"):
        fixed_subset_outcome(initial, corrupted)


def test_zero_gate_pairing_ignores_inactive_residual_but_not_action() -> None:
    a0 = {"initial_probe": _probe(loss=1.0, ratio=0.0)}
    a1 = {
        "initial_probe": _probe(
            loss=1.0,
            ratio=0.0,
            residual_offset=100.0,
        )
    }
    assert _initial_probe_signature(a0) == _initial_probe_signature(a1)
    a1["initial_probe"]["per_sample"][0]["action_loss"] = 1.01
    assert _initial_probe_signature(a0) != _initial_probe_signature(a1)


def test_phase_e2_refuses_without_confirmation_before_model_load(
    monkeypatch,
) -> None:
    cfg = load_thought3_config(
        "configs/thought3/phase_e2_eight_sample_diagnostic.yaml"
    )
    monkeypatch.delenv("CONFIRM_THOUGHT3_PHASE_E2", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Gate E.2 loaded Fast-WAM before confirmation")

    monkeypatch.setattr(
        "fastwam_ood_eval.thought3.phase_e2_eight_sample."
        "_load_upstream_model",
        forbidden,
    )
    with pytest.raises(RuntimeError, match="CONFIRM_THOUGHT3_PHASE_E2"):
        _run_phase_e2(cfg, resume=False)


def test_fixed_subset_training_api_has_no_dev_or_outcome_input() -> None:
    parameters = set(
        inspect.signature(run_fixed_subset_training).parameters
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
        "ood",
        "success",
        "future_rgb",
    }
