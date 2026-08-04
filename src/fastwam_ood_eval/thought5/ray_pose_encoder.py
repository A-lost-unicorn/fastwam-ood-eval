"""Lightweight per-token camera-ray and relative-pose conditioning."""

from __future__ import annotations

from typing import Any


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("RayPoseEncoder requires PyTorch") from exc
    return torch


class RayPoseEncoder:
    def __new__(
        cls,
        *,
        model_dim: int = 3072,
        hidden_dim: int = 128,
        pose_dim: int = 12,
    ) -> Any:
        torch = _torch()

        class _RayPoseEncoder(torch.nn.Module):
            requires_depth = False
            inference_inputs = ("camera_rays", "camera_to_world")

            def __init__(self) -> None:
                super().__init__()
                self.pose_dim = pose_dim
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(3 + pose_dim, hidden_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(hidden_dim, model_dim),
                )
                self.pose_aux = torch.nn.Sequential(
                    torch.nn.LayerNorm(model_dim),
                    torch.nn.Linear(model_dim, pose_dim),
                )
                self.gate = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

            def forward(self, rays: Any, pose: Any) -> Any:
                if rays.ndim != 3 or rays.shape[-1] != 3:
                    raise ValueError("rays must be [batch,tokens,3]")
                if pose.ndim != 2 or pose.shape[-1] != pose_dim:
                    raise ValueError(f"pose must be [batch,{pose_dim}]")
                if rays.shape[0] != pose.shape[0]:
                    raise ValueError("ray and pose batch sizes differ")
                expanded_pose = pose[:, None, :].expand(-1, rays.shape[1], -1)
                features = torch.cat([rays.float(), expanded_pose.float()], dim=-1)
                return torch.tanh(self.gate) * self.net(features)

            def predict_pose(self, encoded: Any) -> Any:
                if encoded.ndim != 3:
                    raise ValueError("encoded ray/pose tensor must be rank 3")
                # The residual injected into Fast-WAM must match the BF16
                # backbone, while the small training-only auxiliary head is
                # deliberately kept in FP32.  Cast before pooling so both the
                # reduction and LayerNorm/Linear run at the head's precision.
                # ``Tensor.to`` preserves autograd, so gradients still reach
                # the injected encoding and RayPoseEncoder.
                reference = next(self.pose_aux.parameters())
                pooled = encoded.to(
                    device=reference.device,
                    dtype=reference.dtype,
                ).mean(dim=1)
                return self.pose_aux(pooled)

        return _RayPoseEncoder()
