from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from fastwam_ood_eval.thought6 import BOOTSTRAP_SEED, SIGMA_THRESHOLD
from fastwam_ood_eval.thought6.config import load_thought6_config
from fastwam_ood_eval.thought6.future_modes import FusionMode, decide_future_fusion
from fastwam_ood_eval.thought6.rollout_policy import SigmaAwareFutureInjector, mock_online_contract
from fastwam_ood_eval.thought6.schemas import Thought6Error, tensor_sha256
from fastwam_ood_eval.thought6.sigma_gate import (
    build_inference_sigma_schedule,
    offline_sigma_from_seed,
    shifted_sigma,
    sigma_gate,
    validate_runtime_schedule,
)


class CountingAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, hidden: torch.Tensor, future: torch.Tensor, _mask=None) -> torch.Tensor:
        self.calls += 1
        return hidden + future.reshape(-1)[0].to(hidden)


def test_01_sigma_threshold_is_fixed_half() -> None:
    assert SIGMA_THRESHOLD == 0.5


def test_02_config_rejects_sigma_override(tmp_path) -> None:
    source = open("configs/thought6/phase6_audit.yaml", encoding="utf-8").read()
    path = tmp_path / "bad.yaml"
    path.write_text(source + "\nsigma_0: 0.4\n", encoding="utf-8")
    with pytest.raises(Thought6Error, match="forbidden"):
        load_thought6_config(path)


def test_03_gate_boundary() -> None:
    assert sigma_gate(0.499999) == 0
    assert sigma_gate(0.5) == 1


def test_04_gate_rejects_nonfinite() -> None:
    with pytest.raises(Thought6Error):
        sigma_gate(math.nan)


def test_05_shift_formula() -> None:
    assert shifted_sigma(0.5) == pytest.approx(5 / 6)


def test_06_offline_sigma_reproducible() -> None:
    assert offline_sigma_from_seed(123) == offline_sigma_from_seed(123)


def test_07_offline_effective_sigma_uses_bf16_timestep() -> None:
    row = offline_sigma_from_seed(123)
    expected = float(
        (torch.tensor(row["sampled_timestep_bf16"], dtype=torch.bfloat16) / 1000.0)
        .float()
        .item()
    )
    assert row["effective_sigma_bf16"] == expected


def test_08_online_schedule_has_20_actual_steps() -> None:
    assert len(build_inference_sigma_schedule()) == 20


def test_09_online_schedule_has_17_high_sigma_steps() -> None:
    assert sum(row.gate for row in build_inference_sigma_schedule()) == 17


def test_10_runtime_schedule_reads_values_not_step_number() -> None:
    rows = build_inference_sigma_schedule()
    timesteps = torch.tensor([row.scheduler_timestep_bf16 for row in rows], dtype=torch.bfloat16)
    deltas = torch.tensor([row.delta_bf16 for row in rows], dtype=torch.bfloat16)
    observed = validate_runtime_schedule(timesteps, deltas)
    assert [row.gate for row in observed] == [row.gate for row in rows]


def test_11_b0_never_calls_adapter() -> None:
    encoder, adapter = nn.Identity(), CountingAdapter()
    decision = decide_future_fusion("B0", condition="camera", effective_sigma=1.0)
    with SigmaAwareFutureInjector(encoder, adapter) as injector:
        with injector.activate_step(decision, future_latent=None, future_mask=None):
            output = encoder(torch.ones(1, 2, 3))
    assert adapter.calls == 0 and torch.equal(output, torch.ones_like(output))


def test_12_b0_identity_is_bitwise() -> None:
    encoder, adapter = nn.Identity(), CountingAdapter()
    value = torch.randn(2, 3)
    before = tensor_sha256(value)
    decision = decide_future_fusion("B0", condition="clean", effective_sigma=0.8)
    with SigmaAwareFutureInjector(encoder, adapter) as injector:
        with injector.activate_step(decision, future_latent=None, future_mask=None):
            output = encoder(value)
    assert tensor_sha256(output) == before


def test_13_f0_calls_adapter_20_of_20() -> None:
    assert mock_online_contract()["counts"]["F0"] == 20


def test_14_fsigma_calls_adapter_17_of_20() -> None:
    assert mock_online_contract()["counts"]["Fsigma"] == 17


def test_15_low_sigma_contribution_is_strict_zero() -> None:
    encoder, adapter = nn.Identity(), CountingAdapter()
    decision = decide_future_fusion("Fsigma", condition="camera", effective_sigma=0.468)
    with SigmaAwareFutureInjector(encoder, adapter) as injector:
        with injector.activate_step(decision, future_latent=None, future_mask=None) as scope:
            encoder(torch.ones(2, 2))
    assert scope.diagnostic["adapter_output_rms"] == 0.0
    assert scope.diagnostic["pre_fusion_hidden_sha256"] == scope.diagnostic["post_fusion_hidden_sha256"]


def test_16_high_sigma_fsigma_matches_f0_adapter_output() -> None:
    value, future = torch.ones(1, 2), torch.tensor([[[2.0]]])
    outputs = []
    for mode in (FusionMode.F0, FusionMode.FSIGMA):
        encoder, adapter = nn.Identity(), CountingAdapter()
        decision = decide_future_fusion(mode, condition="camera", effective_sigma=0.8)
        with SigmaAwareFutureInjector(encoder, adapter) as injector:
            with injector.activate_step(decision, future_latent=future, future_mask=None):
                outputs.append(encoder(value))
    assert torch.equal(*outputs)


def test_17_label_oracle_is_clean_identity() -> None:
    assert not decide_future_fusion("Label-Oracle", condition="clean", effective_sigma=1).adapter_called


def test_18_label_oracle_is_diagnostic_camera_fusion() -> None:
    row = decide_future_fusion("Label-Oracle", condition="camera", effective_sigma=0.1)
    assert row.adapter_called and row.label_oracle_used


def test_19_shuffle_uses_same_sigma_gate() -> None:
    for sigma in (0.2, 0.5, 0.9):
        assert decide_future_fusion("Shuffle+Fsigma", condition="camera", effective_sigma=sigma).external_gate == decide_future_fusion("Fsigma", condition="camera", effective_sigma=sigma).external_gate


def test_20_bootstrap_seed_is_fixed() -> None:
    assert BOOTSTRAP_SEED == 6607


def test_20b_b0_matches_thought3_formal_null_bitwise() -> None:
    from fastwam_ood_eval.thought3.injection import ActionEncoderFutureInjector

    value = torch.randn(2, 3)
    phase6_encoder, phase6_adapter = nn.Identity(), CountingAdapter()
    with SigmaAwareFutureInjector(phase6_encoder, phase6_adapter) as injector:
        decision = decide_future_fusion("B0", condition="clean", effective_sigma=1.0)
        with injector.activate_step(decision, future_latent=None, future_mask=None):
            phase6 = phase6_encoder(value)
    formal_encoder, formal_adapter = nn.Identity(), CountingAdapter()
    with ActionEncoderFutureInjector(formal_encoder, formal_adapter) as injector:
        with injector.activate_null(expected_calls=1):
            formal = formal_encoder(value)
    assert torch.equal(phase6, formal) and phase6_adapter.calls == formal_adapter.calls == 0
