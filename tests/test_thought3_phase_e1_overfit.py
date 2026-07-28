from __future__ import annotations

import inspect

import pytest

from fastwam_ood_eval.thought3.config import load_thought3_config
from fastwam_ood_eval.thought3.phase_e1_overfit import (
    PHASE_E1_MIN_LOSS_REDUCTION_FRACTION,
    _assert_phase_e1_scope,
    _matched_recipe_payload,
    _run_phase_e1,
    derive_overfit_variant_config,
)
from fastwam_ood_eval.thought3.real_training import (
    run_fixed_sample_overfit,
)


def test_phase_e1_a0_a1_tracks_have_matched_recipe() -> None:
    cfg = load_thought3_config(
        "configs/thought3/phase_e1_overfit_diagnostic.yaml"
    )
    _assert_phase_e1_scope(cfg)
    a0 = derive_overfit_variant_config(cfg, variant="A0")
    a1 = derive_overfit_variant_config(cfg, variant="A1")
    assert _matched_recipe_payload(a0) == _matched_recipe_payload(a1)
    assert (a0.variant, a0.sampler.active_k) == ("A0", 0)
    assert (a1.variant, a1.sampler.active_k) == ("A1", 1)
    assert a0.training.max_steps == a1.training.max_steps == 200
    assert a0.training.learning_rate == a1.training.learning_rate == 1e-3
    assert PHASE_E1_MIN_LOSS_REDUCTION_FRACTION == 0.50


def test_phase_e1_refuses_without_confirmation_before_model_load(
    monkeypatch,
) -> None:
    cfg = load_thought3_config(
        "configs/thought3/phase_e1_overfit_diagnostic.yaml"
    )
    monkeypatch.delenv("CONFIRM_THOUGHT3_PHASE_E1", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Gate E.1 loaded Fast-WAM before confirmation")

    monkeypatch.setattr(
        "fastwam_ood_eval.thought3.phase_e1_overfit."
        "_load_upstream_model",
        forbidden,
    )
    with pytest.raises(RuntimeError, match="CONFIRM_THOUGHT3_PHASE_E1"):
        _run_phase_e1(cfg, resume=False)


def test_fixed_overfit_api_has_no_development_or_outcome_input() -> None:
    parameters = set(inspect.signature(run_fixed_sample_overfit).parameters)
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
