from __future__ import annotations

import inspect

import pytest
import torch

from fastwam_ood_eval.thought3.future_sampler import (
    FutureSamplingError,
    VideoOnlyFutureSampler,
    make_initial_video_state,
    make_mock_future_sampler,
)
from fastwam_ood_eval.thought3.schemas import build_sampler_schedule


@pytest.mark.parametrize("k", [1, 2, 4])
def test_complete_shifted_schedule_reaches_zero(k):
    schedule = build_sampler_schedule(k)
    assert schedule.sigma_nodes[0] == 1.0
    assert schedule.sigma_nodes[-1] == 0.0
    assert len(schedule.deltas) == k
    assert sum(schedule.deltas) == pytest.approx(-1.0)


def test_initial_noise_is_independent_of_k_and_batch_order():
    current = torch.zeros(2, 4, 1, 3, 5)
    state, hashes = make_initial_video_state(current, [11, 29])
    reversed_state, reversed_hashes = make_initial_video_state(
        current.flip(0), [29, 11]
    )
    assert hashes == tuple(reversed(reversed_hashes))
    assert torch.equal(state[0, :, 1:], reversed_state[1, :, 1:])
    assert torch.equal(state[:, :, :1], current)


def test_mock_sampler_outputs_native_shape_and_distinct_k():
    sampler = make_mock_future_sampler()
    current = torch.full((1, 48, 1, 14, 28), 0.25)
    values = [
        sampler.sample(current, initial_noise_seeds=[1234], k=k)
        for k in (1, 2, 4)
    ]
    assert all(value.future_latent.shape == (1, 48, 2, 14, 28) for value in values)
    assert len({value.initial_state_sha256 for value in values}) == 1
    assert not torch.equal(values[0].future_latent, values[-1].future_latent)
    assert all(torch.equal(value.full_state[:, :, :1], current) for value in values)


def test_video_only_callback_rejects_action_parameter():
    def forbidden(state, timestep, conditions, action):
        return state

    with pytest.raises(FutureSamplingError, match="forbidden inputs"):
        VideoOnlyFutureSampler(forbidden)
    assert "action" not in inspect.signature(
        make_mock_future_sampler().sample
    ).parameters
