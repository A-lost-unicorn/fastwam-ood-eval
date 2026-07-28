from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

import fastwam_ood_eval.thought3.phase_e_training_smoke as phase_e
from fastwam_ood_eval.thought3.adapter import (
    FutureAdapterSpec,
    FutureToActionAdapter,
)
from fastwam_ood_eval.thought3.config import load_thought3_config
from fastwam_ood_eval.thought3.phase_e_training_smoke import (
    _assert_phase_e_scope,
    _matched_recipe_payload,
    _progress,
    _run_phase_e,
    _verify_phase_d_gate,
    derive_variant_config,
)
from fastwam_ood_eval.thought3.real_training import (
    CurrentActionLiberoSource,
    CurrentActionObservation,
    adapter_gradient_groups,
    preprocess_current_action_target,
)


class _IdentityTransform:
    def __call__(self, batch):
        return batch


class _Normalizer:
    def forward(self, batch):
        batch["action"]["action"] += 1
        batch["state"]["state"] += 2
        return batch


class _Merger:
    def forward(self, batch):
        batch["action"] = batch["action"]["action"]
        batch["state"] = batch["state"]["state"]
        return batch


def test_current_action_target_uses_official_processor_stages_and_padding() -> None:
    processor = SimpleNamespace(
        shape_meta={
            "action": [{"key": "action"}],
            "state": [{"key": "state"}],
        },
        delta_action_dim_mask={
            "action": torch.tensor(
                [True, False, False, False, False, False, False]
            )
        },
        action_state_transform=_IdentityTransform(),
        normalizer=_Normalizer(),
        action_state_merger=_Merger(),
    )
    action = torch.arange(32 * 7, dtype=torch.float32).reshape(32, 7)
    state = torch.arange(8, dtype=torch.float32)
    padding = torch.zeros(32, dtype=torch.bool)
    padding[-1] = True
    processed_action, processed_state, processed_padding = (
        preprocess_current_action_target(
            action,
            state,
            padding,
            processor=processor,
        )
    )
    assert processed_action.shape == (32, 7)
    assert processed_state.shape == (1, 8)
    assert torch.equal(processed_padding, padding)
    assert processed_action[-1, 0] == 1
    assert processed_action[-1, 1] == action[-1, 1] + 1
    assert torch.equal(processed_state, state.unsqueeze(0) + 2)


def test_current_action_source_api_contains_no_future_rgb_argument() -> None:
    parameters = set(
        inspect.signature(CurrentActionLiberoSource.load_training).parameters
    )
    assert parameters == {"self", "entry"}
    fields = set(CurrentActionObservation.__dataclass_fields__)
    assert fields == {
        "action_is_pad",
        "image",
        "proprio",
        "source",
        "target_action",
    }
    assert not fields & {
        "actual_future",
        "future_frames",
        "future_rgb",
        "next_observation",
        "success",
    }


def test_zero_gate_opens_non_gate_paths_on_second_step() -> None:
    torch.manual_seed(3407)
    adapter = FutureToActionAdapter(
        FutureAdapterSpec(
            input_channels=2,
            action_hidden_dim=8,
            future_dim=4,
            attention_dim=8,
            num_heads=2,
            max_projected_grid=(2, 2, 2),
        )
    )
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=0.1,
        weight_decay=0,
    )
    action = torch.randn(1, 3, 8)
    future = torch.randn(1, 2, 2, 4, 4)
    target = torch.randn_like(action)

    optimizer.zero_grad(set_to_none=True)
    first_loss = (adapter(action, future) - target).square().mean()
    first_loss.backward()
    first = adapter_gradient_groups(adapter)
    assert first["gate"]["l2"] > 0
    assert first["non_gate"]["nonzero_element_count"] == 0
    optimizer.step()
    assert float(adapter.gate.detach()) != 0

    optimizer.zero_grad(set_to_none=True)
    second_loss = (adapter(action, future) - target).square().mean()
    second_loss.backward()
    second = adapter_gradient_groups(adapter)
    assert second["future_projector"]["nonzero_element_count"] > 0
    assert second["attention"]["nonzero_element_count"] > 0
    assert second["non_gate"]["finite"] is True


def test_phase_e_a0_a1_tracks_have_matched_recipe() -> None:
    cfg = load_thought3_config(
        "configs/thought3/phase_e_training_smoke.yaml"
    )
    _assert_phase_e_scope(cfg)
    values = [
        derive_variant_config(cfg, variant=variant, track=track)
        for variant in ("A0", "A1")
        for track in ("resumed", "uninterrupted")
    ]
    payloads = [_matched_recipe_payload(value) for value in values]
    assert all(payload == payloads[0] for payload in payloads)
    assert {value.variant for value in values} == {"A0", "A1"}
    assert {
        (value.variant, value.sampler.active_k) for value in values
    } == {("A0", 0), ("A1", 1)}
    assert len(
        {value.adapter_structural_fingerprint for value in values}
    ) == 1


def test_phase_e_refuses_without_confirmation_before_model_load(
    monkeypatch,
) -> None:
    cfg = load_thought3_config(
        "configs/thought3/phase_e_training_smoke.yaml"
    )
    monkeypatch.delenv("CONFIRM_THOUGHT3_PHASE_E", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Phase E loaded Fast-WAM before confirmation")

    monkeypatch.setattr(
        "fastwam_ood_eval.thought3.phase_e_training_smoke."
        "_load_upstream_model",
        forbidden,
    )
    with pytest.raises(RuntimeError, match="CONFIRM_THOUGHT3_PHASE_E"):
        _run_phase_e(cfg, resume=False)


def test_phase_e_reads_split_fingerprint_from_frozen_gate_d_result(
    monkeypatch,
) -> None:
    cfg = load_thought3_config(
        "configs/thought3/phase_e_training_smoke.yaml"
    )
    monkeypatch.setattr(phase_e, "PHASE_D_FROZEN", {})
    monkeypatch.setattr(
        phase_e,
        "load_json",
        lambda path: (
            {"gate_d_passed": True}
            if str(path).endswith("run_status.json")
            else {
                "gate_d_passed": True,
                "plan": {
                    "split_fingerprint": (
                        phase_e.PHASE_D_SPLIT_FINGERPRINT
                    )
                },
            }
        ),
    )
    monkeypatch.setattr(
        phase_e,
        "validate_cache",
        lambda root: {
            "cache_fingerprint": phase_e.PHASE_D_CACHE_FINGERPRINT,
            "entry_count": 96,
            "shard_count": 12,
            "uses_ground_truth_future": False,
        },
    )
    report = _verify_phase_d_gate(cfg)
    assert (
        report["split_fingerprint"]
        == phase_e.PHASE_D_SPLIT_FINGERPRINT
    )


def test_phase_e_progress_accepts_training_callback_contract(
    capsys,
) -> None:
    _progress("checkpoint", {"step": 25}, variant="A1")
    output = capsys.readouterr().out
    assert '"stage": "checkpoint"' in output
    assert '"step": 25' in output
    assert '"variant": "A1"' in output
