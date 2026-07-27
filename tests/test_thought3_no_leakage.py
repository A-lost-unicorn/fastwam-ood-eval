from __future__ import annotations

import pytest

from fastwam_ood_eval.thought3.safety import (
    Thought3SafetyError,
    ensure_standard_training_source,
    ensure_thought3_output_path,
    validate_training_batch_keys,
)


def test_output_guard_rejects_frozen_and_upstream_paths(tmp_path):
    ensure_thought3_output_path(tmp_path / "thought3" / "safe")
    for path in (
        tmp_path / "outputs" / "thought1" / "bad",
        tmp_path / "outputs" / "thought2" / "thought3",
        tmp_path / "third_party" / "thought3",
        tmp_path / "ordinary",
    ):
        with pytest.raises(Thought3SafetyError):
            ensure_thought3_output_path(path)


def test_training_source_rejects_ood_and_formal_trajectory_roots(tmp_path):
    ensure_standard_training_source(tmp_path / "standard_libero" / "lerobot")
    for path in (
        tmp_path / "LIBERO-plus" / "data",
        tmp_path / "libero_plus" / "data",
        tmp_path / "outputs" / "thought1",
        tmp_path / "outputs" / "thought2",
        tmp_path / "outputs" / "thought3",
    ):
        with pytest.raises(Thought3SafetyError):
            ensure_standard_training_source(path)


def test_real_future_or_outcome_cannot_enter_adapter_batch():
    legal = {
        "current_rgb": object(),
        "current_proprio": object(),
        "context": object(),
        "context_mask": object(),
        "target_action": object(),
        "action_is_pad": object(),
        "future_latent": object(),
        "future_mask": object(),
    }
    validate_training_batch_keys(legal)
    for key in ("actual_future", "next_observation", "success"):
        with pytest.raises(Thought3SafetyError, match="forbidden"):
            validate_training_batch_keys({**legal, key: object()})
