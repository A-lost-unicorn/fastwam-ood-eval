"""Training-only projector from action-consumed Video K/V features to geometry."""

from __future__ import annotations

from typing import Any


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise RuntimeError("GeoProjector requires PyTorch") from exc
    return torch


class GeoProjector:  # constructed as a real nn.Module by __new__
    """Factory-compatible wrapper that avoids importing torch at package import."""

    def __new__(
        cls,
        input_dim: int = 3072,
        hidden_dim: int = 256,
        output_dim: int = 11,
    ) -> Any:
        torch = _torch()

        class _GeoProjector(torch.nn.Module):
            geometry_layout = (
                "depth:1,depth_relation:1,point_camera:3,"
                "point_world:3,eef_object_world:3"
            )
            training_only = True

            def __init__(self) -> None:
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.LayerNorm(input_dim),
                    torch.nn.Linear(input_dim, hidden_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(hidden_dim, output_dim),
                )

            def forward(self, hidden: Any) -> Any:
                if hidden.shape[-1] != input_dim:
                    raise ValueError(
                        f"GeoProjector expected hidden dim {input_dim}, "
                        f"received {hidden.shape[-1]}"
                    )
                return self.net(hidden.float())

            @staticmethod
            def unpack(packed: Any) -> dict[str, Any]:
                if packed.shape[-1] != output_dim:
                    raise ValueError("invalid packed geometry dimension")
                return {
                    "depth": packed[..., 0:1],
                    "depth_relation": packed[..., 1:2],
                    "point_camera": packed[..., 2:5],
                    "point_world": packed[..., 5:8],
                    "eef_object_world": packed[..., 8:11],
                }

        return _GeoProjector()
