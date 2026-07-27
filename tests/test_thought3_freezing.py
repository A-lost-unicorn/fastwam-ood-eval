from __future__ import annotations

import pytest
import torch
from torch import nn

from fastwam_ood_eval.thought3.adapter import (
    FutureAdapterSpec,
    FutureToActionAdapter,
)
from fastwam_ood_eval.thought3.injection import FutureInjectionError
from fastwam_ood_eval.thought3.model_wrapper import AdapterConditionedModel


class TinyActionBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_encoder = nn.Linear(3, 16)
        self.action_head = nn.Linear(16, 3)

    def forward(self, action):
        return self.action_head(self.action_encoder(action))


def _wrapped() -> AdapterConditionedModel:
    return AdapterConditionedModel(
        TinyActionBackbone(),
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


def test_only_adapter_is_trainable_and_backward_is_finite():
    model = _wrapped()
    action = torch.randn(2, 5, 3)
    future = torch.randn(2, 4, 2, 8, 8)
    loss = model(action, future_latent=future).square().mean()
    loss.backward()
    assert all(name.startswith("adapter.") for name in model.trainable_parameter_names)
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    adapter_gradients = [
        parameter.grad
        for parameter in model.adapter.parameters()
        if parameter.grad is not None
    ]
    assert adapter_gradients
    assert all(torch.isfinite(gradient).all() for gradient in adapter_gradients)
    model.close()


def test_hook_context_is_cleaned_after_backbone_exception():
    model = _wrapped()

    def explode(*args, **kwargs):
        raise RuntimeError("intentional")

    model.backbone.forward = explode
    with pytest.raises(RuntimeError, match="intentional"):
        model(
            torch.randn(1, 2, 3),
            future_latent=torch.randn(1, 4, 2, 8, 8),
        )
    assert not model.injector.has_active_context
    model.close()


def test_hook_requires_exactly_one_action_encoder_call():
    model = _wrapped()

    def skip_encoder(action):
        return action

    model.backbone.forward = skip_encoder
    with pytest.raises(FutureInjectionError, match="call mismatch"):
        model(
            torch.randn(1, 2, 3),
            future_latent=torch.randn(1, 4, 2, 8, 8),
        )
    assert not model.injector.has_active_context
    model.close()
