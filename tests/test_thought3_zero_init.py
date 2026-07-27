from __future__ import annotations

import torch
from torch import nn

from fastwam_ood_eval.thought3.adapter import (
    FutureAdapterSpec,
    FutureToActionAdapter,
)
from fastwam_ood_eval.thought3.model_wrapper import AdapterConditionedModel


def test_zero_gate_is_exact_identity_and_non_mutating():
    torch.manual_seed(7)
    adapter = FutureToActionAdapter(
        FutureAdapterSpec(
            input_channels=4,
            action_hidden_dim=16,
            future_dim=8,
            attention_dim=16,
            num_heads=4,
            max_projected_grid=(2, 4, 4),
        )
    )
    action = torch.randn(2, 5, 16)
    original = action.clone()
    output = adapter(action, torch.randn(2, 4, 2, 8, 8))
    assert torch.equal(output, original)
    assert torch.equal(action, original)


def test_a0_a1_a2_a4_have_identical_structure_and_count():
    adapters = [
        FutureToActionAdapter(
            FutureAdapterSpec(
                input_channels=4,
                action_hidden_dim=16,
                future_dim=8,
                attention_dim=16,
                num_heads=4,
                max_projected_grid=(2, 4, 4),
            )
        )
        for _ in ("A0", "A1", "A2", "A4")
    ]
    counts = {adapter.parameter_count for adapter in adapters}
    fingerprints = {adapter.spec.fingerprint for adapter in adapters}
    state_keys = {tuple(adapter.state_dict()) for adapter in adapters}
    assert len(counts) == len(fingerprints) == len(state_keys) == 1


def test_a0_wrapper_matches_b0_backbone_action_at_initialization():
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_encoder = nn.Linear(3, 16)
            self.head = nn.Linear(16, 3)

        def forward(self, action):
            return self.head(self.action_encoder(action))

    torch.manual_seed(31)
    backbone = Backbone()
    action = torch.randn(2, 5, 3)
    b0_action = backbone(action).detach()
    wrapped = AdapterConditionedModel(
        backbone,
        FutureToActionAdapter(
            FutureAdapterSpec(
                input_channels=4,
                action_hidden_dim=16,
                future_dim=8,
                attention_dim=16,
                num_heads=4,
                max_projected_grid=(2, 4, 4),
            )
        ),
    )
    a0_action = wrapped(
        action,
        future_latent=torch.zeros(2, 4, 2, 8, 8),
    ).detach()
    assert torch.equal(a0_action, b0_action)
    wrapped.close()
