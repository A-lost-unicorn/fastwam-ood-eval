"""Fixed-capacity lightweight probes; these are the only trainable modules."""

from __future__ import annotations

from typing import Any


class ProbeModelError(ValueError):
    """Raised for an unsupported or unsafe probe architecture."""


def build_probe(
    kind: str,
    *,
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 256,
) -> Any:
    import torch
    from torch import nn

    if min(input_dim, output_dim) <= 0:
        raise ProbeModelError("probe input/output dimensions must be positive")
    if kind == "linear":
        model = nn.Linear(input_dim, output_dim)
    elif kind == "mlp":
        if hidden_dim <= 0:
            raise ProbeModelError("MLP hidden dimension must be positive")
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    else:
        raise ProbeModelError(f"unknown probe kind: {kind}")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def probe_parameter_count(model: Any) -> int:
    return sum(int(value.numel()) for value in model.parameters())


def linear_weight(model: Any) -> Any:
    """Return the raw-hidden/raw-target equivalent linear weight."""

    from torch import nn

    trained = model
    core = getattr(trained, "model", trained)
    if not isinstance(core, nn.Linear):
        raise ProbeModelError("geometry subspace requires a linear probe")
    weight = core.weight.detach().float()
    if core is trained:
        return weight
    feature_std = getattr(trained, "feature_std", None)
    target_std = getattr(trained, "target_std", None)
    if feature_std is None or target_std is None:
        raise ProbeModelError(
            "standardized linear probe lacks train-only scale metadata"
        )
    feature_scale = feature_std.detach().float().reshape(1, -1)
    target_scale = target_std.detach().float().reshape(-1, 1)
    if (
        feature_scale.shape[1] != weight.shape[1]
        or target_scale.shape[0] != weight.shape[0]
    ):
        raise ProbeModelError("linear probe standardization shape mismatch")
    return target_scale * weight / feature_scale
