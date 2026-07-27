"""Adapter-only wrapper and freezing invariants."""

from __future__ import annotations

import hashlib
from typing import Any, Iterator

import torch
from torch import Tensor, nn

from fastwam_ood_eval.thought3.adapter import FutureToActionAdapter
from fastwam_ood_eval.thought3.injection import ActionEncoderFutureInjector


class Thought3FreezingError(RuntimeError):
    """Raised when parameters outside the Adapter are trainable."""


def _tensor_bytes(tensor: Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    return value.reshape(-1).view(torch.uint8).numpy().tobytes()


def parameter_state_sha256(
    named_parameters: Iterator[tuple[str, nn.Parameter]],
) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(named_parameters, key=lambda item: item[0]):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(parameter.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(_tensor_bytes(parameter))
    return digest.hexdigest()


def freeze_backbone(backbone: nn.Module) -> None:
    backbone.requires_grad_(False)
    backbone.eval()


def assert_adapter_only_trainable(
    module: nn.Module,
    *,
    allowed_prefix: str = "adapter.",
) -> tuple[str, ...]:
    trainable = tuple(
        name for name, parameter in module.named_parameters() if parameter.requires_grad
    )
    unexpected = tuple(name for name in trainable if not name.startswith(allowed_prefix))
    if unexpected:
        raise Thought3FreezingError(
            f"parameters outside {allowed_prefix!r} are trainable: {unexpected}"
        )
    if not trainable:
        raise Thought3FreezingError("no Adapter parameters are trainable")
    return trainable


class AdapterConditionedModel(nn.Module):
    """Wrap a frozen model without modifying its source or forward signature."""

    def __init__(
        self,
        backbone: nn.Module,
        adapter: FutureToActionAdapter,
        *,
        action_encoder_path: str = "action_encoder",
    ) -> None:
        super().__init__()
        freeze_backbone(backbone)
        self.backbone = backbone
        self.adapter = adapter
        self.adapter.requires_grad_(True)
        try:
            action_encoder = backbone.get_submodule(action_encoder_path)
        except AttributeError as exc:
            raise Thought3FreezingError(
                f"backbone has no module at {action_encoder_path!r}"
            ) from exc
        self.injector = ActionEncoderFutureInjector(action_encoder, adapter)
        assert_adapter_only_trainable(self)

    def forward(
        self,
        *args: Any,
        future_latent: Tensor,
        future_mask: Tensor | None = None,
        expected_action_encoder_calls: int = 1,
        **kwargs: Any,
    ) -> Any:
        with self.injector.activate(
            future_latent,
            future_mask,
            expected_calls=expected_action_encoder_calls,
        ):
            return self.backbone(*args, **kwargs)

    @property
    def trainable_parameter_names(self) -> tuple[str, ...]:
        return assert_adapter_only_trainable(self)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @property
    def frozen_parameter_sha256(self) -> str:
        return parameter_state_sha256(iter(self.backbone.named_parameters()))

    def train(self, mode: bool = True) -> "AdapterConditionedModel":
        super().train(mode)
        # The wrapper call above would otherwise put the frozen backbone into
        # train mode and change dropout/batchnorm behavior.
        self.backbone.eval()
        self.adapter.train(mode)
        return self

    def close(self) -> None:
        self.injector.close()
