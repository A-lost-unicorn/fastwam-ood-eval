from __future__ import annotations

import pytest
import torch

from fastwam_ood_eval.thought3.phase_c_smoke import (
    compute_upstream_action_loss,
)


def test_phase_c_action_loss_matches_explicit_masked_formula() -> None:
    pred = torch.tensor(
        [[[1.0, 3.0], [2.0, 8.0], [99.0, 99.0]]],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [[[0.0, 1.0], [4.0, 4.0], [0.0, 0.0]]],
        dtype=torch.float32,
    )
    pad = torch.tensor([[False, False, True]])
    weight = torch.tensor([1.5])
    loss = compute_upstream_action_loss(
        pred,
        target,
        pad,
        weight,
        loss_lambda_action=2.0,
    )
    token0 = ((1.0 - 0.0) ** 2 + (3.0 - 1.0) ** 2) / 2
    token1 = ((2.0 - 4.0) ** 2 + (8.0 - 4.0) ** 2) / 2
    expected = 2.0 * 1.5 * (token0 + token1) / 2
    assert float(loss) == pytest.approx(expected)


def test_phase_c_action_loss_without_pad_uses_all_tokens() -> None:
    pred = torch.zeros((1, 2, 3))
    target = torch.ones_like(pred)
    loss = compute_upstream_action_loss(
        pred,
        target,
        None,
        torch.tensor([0.5]),
        loss_lambda_action=1.0,
    )
    assert float(loss) == pytest.approx(0.5)
