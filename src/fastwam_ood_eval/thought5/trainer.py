"""Matched B1/G1/G2/G3/G4 training contracts for real or mock Fast-WAM."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from fastwam_ood_eval.thought5.geo_projector import GeoProjector
from fastwam_ood_eval.thought5.losses import (
    LossWeights,
    equivariance_loss,
    geo_repa_loss,
    total_loss,
)


class GeoEqTrainingError(RuntimeError):
    pass


VARIANT_FLAGS = {
    "B1": {"repa": False, "ray_pose": False, "shuffle": False},
    "G1": {"repa": True, "ray_pose": False, "shuffle": False},
    "G2": {"repa": False, "ray_pose": True, "shuffle": False},
    "G3": {"repa": True, "ray_pose": True, "shuffle": False},
    "G4": {"repa": True, "ray_pose": True, "shuffle": True},
}


@dataclass
class GeoEqForwardBatch:
    pair_ids: Sequence[str]
    clean_current_latent: Any
    camera_current_latent: Any
    context: Any
    context_mask: Any
    noisy_action: Any
    timestep_action: Any
    velocity_target: Any
    action_is_pad: Any
    action_weight: Any
    clean_rays: Any
    camera_rays: Any
    clean_pose_12: Any
    camera_pose_12: Any
    clean_camera_to_world: Any
    camera_camera_to_world: Any
    clean_geometry_target: Mapping[str, Any]
    camera_geometry_target: Mapping[str, Any]
    valid_mask: Any
    shuffled_clean_geometry_target: Mapping[str, Any] | None = None
    shuffled_camera_geometry_target: Mapping[str, Any] | None = None
    shuffled_pair_ids: Sequence[str] | None = None


def weights_for_variant(values: Mapping[str, float]) -> LossWeights:
    return LossWeights(
        lambda_repa=float(values["lambda_repa"]),
        lambda_equiv=float(values["lambda_equiv"]),
        lambda_pose_aux=float(values["lambda_pose_aux"]),
    )


def trainable_parameters(attachment: Any) -> list[Any]:
    values = [
        parameter
        for _name, parameter in attachment.named_parameters()
        if parameter.requires_grad
    ]
    if not values:
        raise GeoEqTrainingError("no trainable parameters")
    return values


def gradient_report(attachment: Any) -> dict[str, Any]:
    groups: dict[str, list[tuple[str, Any]]] = {
        "lora": [],
        "geo_projector": [],
        "ray_pose_encoder": [],
    }
    for name, parameter in attachment.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".lora_" in name:
            groups["lora"].append((name, parameter))
        elif name.startswith("geo_projector."):
            groups["geo_projector"].append((name, parameter))
        elif name.startswith("ray_pose_encoder."):
            groups["ray_pose_encoder"].append((name, parameter))
        else:
            raise GeoEqTrainingError(f"unexpected trainable parameter: {name}")
    report: dict[str, Any] = {}
    for group, values in groups.items():
        gradients = [p.grad.detach().float() for _, p in values if p.grad is not None]
        finite = all(bool(g.isfinite().all()) for g in gradients)
        l2 = math.sqrt(sum(float(g.square().sum().cpu()) for g in gradients))
        report[group] = {
            "parameter_count": sum(p.numel() for _, p in values),
            "gradient_tensor_count": len(gradients),
            "gradient_l2": l2,
            "finite": finite,
            "nonzero_elements": sum(int(g.count_nonzero()) for g in gradients),
        }
    return report


def _forward_cache_and_action(
    attachment: Any,
    *,
    current_latent: Any,
    context: Any,
    context_mask: Any,
    noisy_action: Any,
    timestep_action: Any,
    rays: Any,
    pose_12: Any,
    enable_ray_pose: bool,
) -> tuple[Any, Any, Any]:
    from fastwam_ood_eval.thought3.phase_c_smoke import (
        _action_from_video_cache,
        _prepare_video_cache,
    )

    model = attachment.model
    shape_probe = model.action_expert.pre_dit(
        action_tokens=noisy_action,
        timestep=timestep_action,
        context=context,
        context_mask=context_mask,
    )
    with attachment.conditioning(
        rays=rays,
        camera_pose_12=pose_12,
        enable_ray_pose=enable_ray_pose,
    ):
        video_cache, attention_mask, video_seq_len = _prepare_video_cache(
            model,
            current_latent,
            context,
            context_mask,
            action_seq_len=int(shape_probe["tokens"].shape[1]),
        )
        prediction = _action_from_video_cache(
            model,
            noisy_action,
            timestep_action,
            context,
            context_mask,
            video_cache,
            attention_mask,
            video_seq_len,
        )
        video_value = attachment.captured_value
    return prediction, video_value, attachment._injected_encoding


def paired_training_loss(
    attachment: Any,
    batch: GeoEqForwardBatch,
    *,
    variant: str,
    weights: LossWeights,
) -> tuple[Any, dict[str, Any]]:
    import torch
    from fastwam_ood_eval.thought3.phase_c_smoke import compute_upstream_action_loss

    if variant not in VARIANT_FLAGS:
        raise GeoEqTrainingError(f"unsupported training variant: {variant}")
    flags = VARIANT_FLAGS[variant]
    has_external_shuffle = (
        batch.shuffled_clean_geometry_target is not None
        and batch.shuffled_camera_geometry_target is not None
        and batch.shuffled_pair_ids is not None
    )
    if flags["shuffle"] and not has_external_shuffle and len(set(batch.pair_ids)) < 2:
        raise GeoEqTrainingError(
            "G4 needs two in-batch pair IDs or an explicit external derangement"
        )
    clean_action, clean_hidden, clean_encoding = _forward_cache_and_action(
        attachment,
        current_latent=batch.clean_current_latent,
        context=batch.context,
        context_mask=batch.context_mask,
        noisy_action=batch.noisy_action,
        timestep_action=batch.timestep_action,
        rays=batch.clean_rays,
        pose_12=batch.clean_pose_12,
        enable_ray_pose=bool(flags["ray_pose"]),
    )
    camera_action, camera_hidden, camera_encoding = _forward_cache_and_action(
        attachment,
        current_latent=batch.camera_current_latent,
        context=batch.context,
        context_mask=batch.context_mask,
        noisy_action=batch.noisy_action,
        timestep_action=batch.timestep_action,
        rays=batch.camera_rays,
        pose_12=batch.camera_pose_12,
        enable_ray_pose=bool(flags["ray_pose"]),
    )
    original_clean = compute_upstream_action_loss(
        clean_action,
        batch.velocity_target,
        batch.action_is_pad,
        batch.action_weight,
        loss_lambda_action=attachment.model.loss_lambda_action,
    )
    original_camera = compute_upstream_action_loss(
        camera_action,
        batch.velocity_target,
        batch.action_is_pad,
        batch.action_weight,
        loss_lambda_action=attachment.model.loss_lambda_action,
    )
    original = 0.5 * (original_clean + original_camera)
    clean_geo = attachment.geo_projector.unpack(
        attachment.geo_projector(clean_hidden)
    )
    camera_geo = attachment.geo_projector.unpack(
        attachment.geo_projector(camera_hidden)
    )
    target_clean = batch.clean_geometry_target
    target_camera = batch.camera_geometry_target
    if flags["shuffle"]:
        if has_external_shuffle:
            if len(batch.shuffled_pair_ids or ()) != len(batch.pair_ids):
                raise GeoEqTrainingError("external shuffled identity count differs")
            if any(
                source == donor
                for source, donor in zip(batch.pair_ids, batch.shuffled_pair_ids or ())
            ):
                raise GeoEqTrainingError("external shuffled geometry contains a fixed point")
            target_clean = batch.shuffled_clean_geometry_target or {}
            target_camera = batch.shuffled_camera_geometry_target or {}
            if set(target_clean) != set(batch.clean_geometry_target) or set(
                target_camera
            ) != set(batch.camera_geometry_target):
                raise GeoEqTrainingError("external shuffled target schema differs")
        else:
            permutation = torch.roll(
                torch.arange(clean_hidden.shape[0], device=clean_hidden.device), shifts=1
            )
            target_clean = {
                key: value[permutation] for key, value in target_clean.items()
            }
            target_camera = {
                key: value[permutation] for key, value in target_camera.items()
            }
    repa = None
    repa_components: dict[str, Any] = {}
    if weights.lambda_repa > 0:
        clean_repa, clean_components = geo_repa_loss(
            clean_geo, target_clean, batch.valid_mask, weights
        )
        camera_repa, camera_components = geo_repa_loss(
            camera_geo, target_camera, batch.valid_mask, weights
        )
        repa = 0.5 * (clean_repa + camera_repa)
        repa_components = {
            key: 0.5 * (clean_components[key] + camera_components[key])
            for key in clean_components
        }
    equiv = None
    if weights.lambda_equiv > 0:
        equiv = equivariance_loss(
            clean_geo,
            camera_geo,
            batch.clean_camera_to_world,
            batch.camera_camera_to_world,
            batch.valid_mask,
        )
    pose_aux = None
    if weights.lambda_pose_aux > 0:
        if clean_encoding is None or camera_encoding is None:
            raise GeoEqTrainingError("pose auxiliary loss requires ray/pose injection")
        clean_pose_prediction = attachment.ray_pose_encoder.predict_pose(clean_encoding)
        camera_pose_prediction = attachment.ray_pose_encoder.predict_pose(camera_encoding)
        pose_aux = 0.5 * (
            torch.nn.functional.smooth_l1_loss(
                clean_pose_prediction.float(), batch.clean_pose_12.detach().float()
            )
            + torch.nn.functional.smooth_l1_loss(
                camera_pose_prediction.float(), batch.camera_pose_12.detach().float()
            )
        )
    total, components = total_loss(
        original_fastwam_loss=original,
        weights=weights,
        repa=repa,
        equiv=equiv,
        pose_aux=pose_aux,
    )
    components.update({f"geo_{key}": value for key, value in repa_components.items()})
    components["original_clean"] = original_clean
    components["original_camera"] = original_camera
    return total, components


def matched_optimizer(
    attachment: Any, *, learning_rate: float, weight_decay: float
) -> Any:
    import torch

    return torch.optim.AdamW(
        trainable_parameters(attachment),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )


def assert_matched_parameter_budgets(manifests: Mapping[str, Mapping[str, Any]]) -> None:
    trainable = {
        variant: int(manifest["trainable_parameter_count"])
        for variant, manifest in manifests.items()
        if variant != "B0"
    }
    if len(set(trainable.values())) > 1:
        raise GeoEqTrainingError(f"control parameter budgets differ: {trainable}")
