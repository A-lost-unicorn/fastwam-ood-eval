from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fastwam_ood_eval.thought4.feature_hooks import (
    FeatureHookError,
    HookSpec,
    ScopedFeatureCapture,
    ScopedFeatureReplacement,
    ScopedVideoKVCacheCapture,
    ScopedVideoKVCacheReplacement,
    VideoKVCacheSpec,
    action_hook_specs,
    assert_backbone_frozen,
    assert_probe_only_trainable,
    freeze_backbone,
    parameter_state_sha256,
    resolve_module,
    validate_layer_indices,
    video_hook_specs,
)
from fastwam_ood_eval.thought4.probe_models import build_probe


class ToyAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)


class ToyBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = ToyAttention(dim)


class ToyExpert(nn.Module):
    def __init__(self, layers: int, dim: int, *, action: bool = False) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([ToyBlock(dim) for _ in range(layers)])
        self.hidden_dim = dim
        if action:
            self.action_encoder = nn.Linear(dim, dim)
            self.head = nn.Linear(dim, dim)


class ToyFastWAM(nn.Module):
    def __init__(self, layers: int = 30, dim: int = 8) -> None:
        super().__init__()
        self.video_expert = ToyExpert(layers, dim)
        self.action_expert = ToyExpert(layers, dim, action=True)

    def video_forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = value
        for block in self.video_expert.blocks:
            normalized = block.norm1(hidden)
            hidden = hidden + block.self_attn.k(normalized) + block.self_attn.v(
                normalized
            )
        return hidden

    def action_forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.action_expert.action_encoder(value)
        for block in self.action_expert.blocks:
            hidden = hidden + block.norm1(hidden)
        return self.action_expert.head(hidden)


class ToyMoT(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward_action_with_video_cache(self, **kwargs: object) -> torch.Tensor:
        action = kwargs["action_tokens"]
        cache = kwargs["video_kv_cache"]
        assert isinstance(action, torch.Tensor)
        assert isinstance(cache, list)
        return action + cache[0]["v"].mean()


def test_capture_hook_is_read_only_and_removed() -> None:
    torch.manual_seed(1)
    model = ToyFastWAM(layers=3)
    value = torch.randn(2, 5, 8)
    reference = model.video_forward(value)
    specs = video_hook_specs((0, 2), include_kv=True)
    with ScopedFeatureCapture(model, specs) as capture:
        hooked = model.video_forward(value)
    after = model.video_forward(value)
    assert torch.equal(reference, hooked)
    assert torch.equal(reference, after)
    assert all(len(capture.captured[spec.name]) == 1 for spec in specs)
    assert all(not tensor.requires_grad for rows in capture.captured.values() for tensor in rows)


def test_video_and_action_real_call_boundaries_are_captured() -> None:
    model = ToyFastWAM()
    video = torch.randn(1, 5, 8)
    action = torch.randn(1, 4, 8)
    specs = (*video_hook_specs((0, 7, 15, 22, 29)), *action_hook_specs())
    with ScopedFeatureCapture(model, specs) as capture:
        model.video_forward(video)
        model.action_forward(action)
    assert set(capture.captured) == {spec.name for spec in specs}
    assert all(len(rows) == 1 for rows in capture.captured.values())


def test_identity_replacement_preserves_output_and_is_removed() -> None:
    model = ToyFastWAM(layers=2)
    value = torch.randn(1, 3, 8)
    reference = model.video_forward(value)
    spec = HookSpec(
        "video_hidden",
        "video_expert.blocks.1.norm1",
        "input",
        expected_calls=1,
    )
    with ScopedFeatureReplacement(model, spec, lambda tensor: tensor.clone()):
        replaced = model.video_forward(value)
    after = model.video_forward(value)
    assert torch.equal(reference, replaced)
    assert torch.equal(reference, after)


def test_invalid_module_or_layer_fails_closed() -> None:
    model = ToyFastWAM(layers=3)
    with pytest.raises(FeatureHookError):
        resolve_module(model, "video_expert.blocks.30.norm1")
    with pytest.raises(FeatureHookError):
        validate_layer_indices(model.video_expert, (0, 3), "video")
    with pytest.raises(FeatureHookError):
        ScopedFeatureCapture(
            model,
            (HookSpec("never", "video_expert.blocks.0.norm1", "input"),),
        ).__enter__().__exit__(None, None, None)


def test_backbone_frozen_probe_only_trainable_and_sha_stable() -> None:
    model = ToyFastWAM(layers=2)
    freeze_backbone(model)
    assert_backbone_frozen(model)
    probe = build_probe("linear", input_dim=8, output_dim=3)
    assert_probe_only_trainable(model, probe)
    before = parameter_state_sha256(model)
    value = torch.randn(2, 4, 8)
    with ScopedFeatureCapture(
        model,
        (HookSpec("hidden", "video_expert.blocks.0.norm1", "input"),),
    ):
        model.video_forward(value)
    assert parameter_state_sha256(model) == before
    model.video_expert.blocks[0].norm1.weight.requires_grad_(True)
    with pytest.raises(FeatureHookError):
        assert_backbone_frozen(model)


def test_nan_activation_fails_closed() -> None:
    model = ToyFastWAM(layers=1)
    value = torch.full((1, 2, 8), float("nan"))
    with pytest.raises(FeatureHookError, match="NaN/Inf"):
        with ScopedFeatureCapture(
            model,
            (HookSpec("hidden", "video_expert.blocks.0.norm1", "input"),),
        ):
            model.video_forward(value)


def test_actual_video_kv_cache_capture_and_replacement_are_scoped() -> None:
    mot = ToyMoT()
    action = torch.ones(1, 2, 3)
    cache = [
        {
            "k": torch.full((1, 5, 8), 2.0),
            "v": torch.full((1, 5, 8), 3.0),
        }
    ]
    reference = mot.forward_action_with_video_cache(
        action_tokens=action, video_kv_cache=cache
    )
    specs = (
        VideoKVCacheSpec(0, "k", expected_calls=2),
        VideoKVCacheSpec(0, "v", expected_calls=2),
    )
    with ScopedVideoKVCacheCapture(mot, specs) as capture:
        first = mot.forward_action_with_video_cache(
            action_tokens=action, video_kv_cache=cache
        )
        second = mot.forward_action_with_video_cache(
            action_tokens=action, video_kv_cache=cache
        )
    assert torch.equal(first, reference)
    assert torch.equal(second, reference)
    assert capture.calls == 2
    assert torch.equal(capture.captured["video_l0_cache_k"][0], cache[0]["k"])
    assert torch.equal(
        mot.forward_action_with_video_cache(
            action_tokens=action, video_kv_cache=cache
        ),
        reference,
    )

    replacement_spec = VideoKVCacheSpec(0, "v", expected_calls=1)
    with ScopedVideoKVCacheReplacement(
        mot, replacement_spec, lambda value: torch.zeros_like(value)
    ):
        replaced = mot.forward_action_with_video_cache(
            action_tokens=action, video_kv_cache=cache
        )
    assert not torch.equal(replaced, reference)
    assert torch.equal(cache[0]["v"], torch.full((1, 5, 8), 3.0))
    assert torch.equal(
        mot.forward_action_with_video_cache(
            action_tokens=action, video_kv_cache=cache
        ),
        reference,
    )


def test_actual_video_kv_cache_capture_supports_inference_tensors() -> None:
    mot = ToyMoT()
    spec = VideoKVCacheSpec(0, "k", expected_calls=2)
    with torch.inference_mode():
        action = torch.ones(1, 2, 3)
        cache = [
            {
                "k": torch.full((1, 5, 8), 2.0),
                "v": torch.full((1, 5, 8), 3.0),
            }
        ]
        with ScopedVideoKVCacheCapture(mot, (spec,)) as capture:
            first = mot.forward_action_with_video_cache(
                action_tokens=action, video_kv_cache=cache
            )
            second = mot.forward_action_with_video_cache(
                action_tokens=action, video_kv_cache=cache
            )
    assert torch.equal(first, second)
    assert capture.calls == 2
    assert torch.equal(capture.captured[spec.name][0], cache[0]["k"])


def test_inference_cache_mutation_still_fails_closed() -> None:
    class MutatingToyMoT(nn.Module):
        def forward_action_with_video_cache(self, **kwargs: object) -> torch.Tensor:
            cache = kwargs["video_kv_cache"]
            assert isinstance(cache, list)
            cache[0]["k"].add_(1)
            action = kwargs["action_tokens"]
            assert isinstance(action, torch.Tensor)
            return action

    mot = MutatingToyMoT()
    spec = VideoKVCacheSpec(0, "k", expected_calls=1)
    with torch.inference_mode():
        action = torch.ones(1, 2, 3)
        cache = [{"k": torch.zeros(1, 5, 8), "v": torch.zeros(1, 5, 8)}]
        with pytest.raises(FeatureHookError, match="content changed"):
            with ScopedVideoKVCacheCapture(mot, (spec,)):
                mot.forward_action_with_video_cache(
                    action_tokens=action, video_kv_cache=cache
                )
