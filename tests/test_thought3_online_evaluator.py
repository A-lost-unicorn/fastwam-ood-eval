from __future__ import annotations

import inspect

import torch

from fastwam_ood_eval.thought3.adapter import (
    FutureAdapterSpec,
    FutureToActionAdapter,
)
from fastwam_ood_eval.thought3.evaluator import OnlineFutureActionEvaluator
from fastwam_ood_eval.thought3.future_sampler import make_mock_future_sampler
from fastwam_ood_eval.thought3.model_wrapper import AdapterConditionedModel
from fastwam_ood_eval.thought3.trainer import MockActionBackbone


def test_online_evaluator_api_cannot_accept_training_cache():
    parameters = inspect.signature(OnlineFutureActionEvaluator).parameters
    assert "cache" not in parameters
    assert "cache_root" not in parameters
    assert "cache_reader" not in parameters


def test_online_future_is_generated_from_current_and_stays_latent():
    backbone = MockActionBackbone(32)
    model = AdapterConditionedModel(
        backbone,
        FutureToActionAdapter(
            FutureAdapterSpec(
                input_channels=48,
                action_hidden_dim=32,
                future_dim=8,
                attention_dim=16,
                num_heads=4,
                max_projected_grid=(2, 7, 14),
            )
        ),
    )
    evaluator = OnlineFutureActionEvaluator(
        backbone=backbone,
        conditioned_model=model,
        sampler=make_mock_future_sampler(),
    )
    result = evaluator.predict(
        torch.zeros(1, 48, 1, 14, 28),
        initial_noise_seed=11,
        action_noise_seed=22,
        k=1,
    )
    assert result.future_latent.shape == (1, 48, 2, 14, 28)
    assert result.latency.future_decoded_to_video is False
    assert result.latency.future_sampling_ms > 0
    model.close()
