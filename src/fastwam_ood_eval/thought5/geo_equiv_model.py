"""Fast-WAM-GeoEq attachment, conditioning context, and trainable whitelist."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Mapping, Sequence

from fastwam_ood_eval.thought5.geo_projector import GeoProjector
from fastwam_ood_eval.thought5.lora_targets import (
    DEFAULT_LORA_TARGETS,
    InstalledLoRA,
    install_lora_targets,
)
from fastwam_ood_eval.thought5.ray_pose_encoder import RayPoseEncoder


class GeoEquivModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class Conditioning:
    rays: Any
    camera_pose_12: Any
    enable_ray_pose: bool


class GeoEqAttachment:
    """Attach Phase 5 modules without editing ``third_party/FastWAM``.

    Ray/pose features are injected into the input of the exact layer-15 Video
    value projection whose output is exposed as ``mot.video_kv_cache.15.v``.
    The output hook captures that action-consumed tensor for Geo-REPA.
    """

    selected_feature = "mot.video_kv_cache.15.v"
    value_module_path = "video_expert.blocks.15.self_attn.v"

    def __init__(
        self,
        model: Any,
        *,
        lora_targets: Sequence[str] = DEFAULT_LORA_TARGETS,
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        projector_hidden_dim: int = 256,
        ray_pose_hidden_dim: int = 128,
    ) -> None:
        import torch

        if not isinstance(model, torch.nn.Module):
            raise GeoEquivModelError("Fast-WAM backbone must be a torch module")
        self.model = model
        self.model.requires_grad_(False)
        self.installed_lora: tuple[InstalledLoRA, ...] = install_lora_targets(
            model,
            lora_targets,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        self.geo_projector = GeoProjector(hidden_dim=projector_hidden_dim)
        self.ray_pose_encoder = RayPoseEncoder(hidden_dim=ray_pose_hidden_dim)
        try:
            model_device = next(model.parameters()).device
        except StopIteration as exc:
            raise GeoEquivModelError("Fast-WAM backbone has no parameters") from exc
        self.geo_projector.to(device=model_device, dtype=torch.float32)
        self.ray_pose_encoder.to(device=model_device, dtype=torch.float32)
        self._conditioning: Conditioning | None = None
        self._captured_value: Any | None = None
        self._injected_encoding: Any | None = None
        value_module = self._module(self.value_module_path)
        self._pre_handle = value_module.register_forward_pre_hook(self._inject)
        self._post_handle = value_module.register_forward_hook(self._capture)

    def _module(self, dotted: str) -> Any:
        value = self.model
        for part in dotted.split("."):
            value = value[int(part)] if part.isdigit() else getattr(value, part)
        return value

    def _inject(self, module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...] | None:
        if self._conditioning is None or not self._conditioning.enable_ray_pose:
            self._injected_encoding = None
            return None
        if len(inputs) != 1:
            raise GeoEquivModelError("Video value projection input contract changed")
        hidden = inputs[0]
        encoding = self.ray_pose_encoder(
            self._conditioning.rays.to(hidden.device),
            self._conditioning.camera_pose_12.to(hidden.device),
        ).to(hidden.dtype)
        if (
            encoding.shape[0] != hidden.shape[0]
            or encoding.shape[2] != hidden.shape[2]
            or hidden.shape[1] % encoding.shape[1] != 0
        ):
            raise GeoEquivModelError(
                f"ray/pose shape {encoding.shape} != Video hidden {hidden.shape}"
            )
        # The action path has one current frame (98 tokens), whereas K=1
        # video sampling has current + two latent future frames (294 tokens).
        # Video tokens are frame-major, so repeat the same camera-ray field for
        # every temporal slice.  No generated RGB/depth is read here.
        temporal_factor = hidden.shape[1] // encoding.shape[1]
        expanded = encoding.repeat(1, temporal_factor, 1)
        self._injected_encoding = expanded
        return (hidden + expanded,)

    def _capture(self, module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        self._captured_value = output

    @contextmanager
    def conditioning(
        self, *, rays: Any, camera_pose_12: Any, enable_ray_pose: bool
    ) -> Iterator[None]:
        if self._conditioning is not None:
            raise GeoEquivModelError("nested camera conditioning is forbidden")
        self._captured_value = None
        self._injected_encoding = None
        self._conditioning = Conditioning(rays, camera_pose_12, enable_ray_pose)
        try:
            yield
        finally:
            self._conditioning = None

    @property
    def captured_value(self) -> Any:
        if self._captured_value is None:
            raise GeoEquivModelError("selected Video value projection was not executed")
        return self._captured_value

    @property
    def injected_encoding(self) -> Any:
        if self._injected_encoding is None:
            raise GeoEquivModelError("ray/pose injection was not executed")
        return self._injected_encoding

    def geometry_prediction(self) -> Any:
        return self.geo_projector(self.captured_value)

    def trainable_modules(self) -> tuple[Any, ...]:
        return (self.model, self.geo_projector, self.ray_pose_encoder)

    def named_parameters(self) -> Iterator[tuple[str, Any]]:
        for name, parameter in self.model.named_parameters():
            yield f"backbone.{name}", parameter
        for name, parameter in self.geo_projector.named_parameters():
            yield f"geo_projector.{name}", parameter
        for name, parameter in self.ray_pose_encoder.named_parameters():
            yield f"ray_pose_encoder.{name}", parameter

    def parameter_manifest(self) -> dict[str, Any]:
        trainable = [
            (name, parameter)
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]
        frozen = [
            (name, parameter)
            for name, parameter in self.named_parameters()
            if not parameter.requires_grad
        ]
        invalid = [
            name
            for name, _ in trainable
            if not (
                ".lora_A" in name
                or ".lora_B" in name
                or name.startswith("geo_projector.")
                or name.startswith("ray_pose_encoder.")
            )
        ]
        if invalid:
            raise GeoEquivModelError(f"trainable whitelist violation: {invalid}")
        return {
            "schema_version": "thought5.phase5.trainable_manifest.v1",
            "selected_feature": self.selected_feature,
            "injection_module": self.value_module_path,
            "lora_targets": [asdict(item) for item in self.installed_lora],
            "trainable_parameter_names": [name for name, _ in trainable],
            "trainable_parameter_count": sum(p.numel() for _, p in trainable),
            "frozen_parameter_count": sum(p.numel() for _, p in frozen),
            "geometry_target_detached": True,
            "inference_uses_gt_depth": False,
            "action_dit_trainable": False,
        }

    def inference_modules(self) -> Mapping[str, Any]:
        """Return deployable modules; the training-only projector is absent."""

        return {"backbone": self.model, "ray_pose_encoder": self.ray_pose_encoder}

    def close(self) -> None:
        self._pre_handle.remove()
        self._post_handle.remove()
