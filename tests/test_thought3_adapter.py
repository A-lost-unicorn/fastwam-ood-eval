from __future__ import annotations

import pytest
import torch
from torch import nn

from fastwam_ood_eval.thought3.adapter import (
    FutureAdapterError,
    FutureAdapterSpec,
    FutureToActionAdapter,
)


def _small_adapter() -> FutureToActionAdapter:
    return FutureToActionAdapter(
        FutureAdapterSpec(
            input_channels=4,
            action_hidden_dim=16,
            future_dim=8,
            attention_dim=16,
            num_heads=4,
            max_projected_grid=(3, 4, 5),
        )
    )


def test_default_adapter_has_audited_parameter_count():
    adapter = FutureToActionAdapter()
    assert adapter.parameter_count == 1_371_137
    assert adapter.trainable_parameter_count == 1_371_137


def test_fp32_adapter_accepts_bfloat16_fastwam_inputs():
    adapter = FutureToActionAdapter()
    action = torch.randn(1, 4, 1024, dtype=torch.bfloat16)
    future = torch.randn(1, 48, 2, 14, 28, dtype=torch.bfloat16)
    output = adapter(action, future)
    assert output.dtype == torch.bfloat16
    assert output.shape == action.shape
    assert torch.equal(output, action)


def test_adapter_supports_variable_tokens_and_latent_mask():
    adapter = _small_adapter()
    action = torch.randn(2, 5, 16)
    future = torch.randn(2, 4, 3, 8, 10)
    mask = torch.ones(2, 3, 8, 10, dtype=torch.bool)
    mask[0, :, :2, :2] = False
    output, diagnostics = adapter(
        action,
        future,
        mask,
        return_diagnostics=True,
    )
    assert output.shape == action.shape
    assert diagnostics.projected_grid == (3, 4, 5)
    assert 0 < diagnostics.valid_token_fraction < 1


@pytest.mark.parametrize(
    "future,match",
    [
        (torch.randn(2, 4, 8, 8), r"\[B,C,T,H,W\]"),
        (torch.randn(2, 5, 2, 8, 8), "channel mismatch"),
        (torch.randn(3, 4, 2, 8, 8), "batch sizes differ"),
    ],
)
def test_adapter_rejects_wrong_future_shape(future, match):
    adapter = _small_adapter()
    with pytest.raises(FutureAdapterError, match=match):
        adapter(torch.randn(2, 5, 16), future)


def test_adapter_rejects_empty_mask_without_nan():
    adapter = _small_adapter()
    action = torch.randn(2, 5, 16)
    future = torch.randn(2, 4, 2, 8, 8)
    mask = torch.ones(2, 2, 8, 8, dtype=torch.bool)
    mask[1] = False
    with pytest.raises(FutureAdapterError, match="at least one valid"):
        adapter(action, future, mask)
