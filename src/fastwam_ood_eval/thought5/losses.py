"""Original Fast-WAM plus preregistered Geo-REPA/equivariance losses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class LossWeights:
    lambda_repa: float
    lambda_equiv: float
    lambda_pose_aux: float
    depth_relation: float = 1.0
    point3d: float = 1.0
    eef_object_relation: float = 1.0

    def __post_init__(self) -> None:
        if any(value < 0 for value in asdict(self).values()):
            raise ValueError("loss weights must be non-negative")


def _masked_smooth_l1(prediction: Any, target: Any, mask: Any) -> Any:
    import torch

    if prediction.shape != target.shape:
        raise ValueError(f"loss shape mismatch: {prediction.shape} != {target.shape}")
    expanded = mask
    while expanded.ndim < prediction.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(prediction).bool()
    if not bool(expanded.any()):
        raise ValueError("geometry loss has no valid tokens")
    return torch.nn.functional.smooth_l1_loss(
        prediction[expanded].float(), target[expanded].detach().float()
    )


def geo_repa_loss(
    prediction: Mapping[str, Any],
    target: Mapping[str, Any],
    valid_mask: Any,
    weights: LossWeights,
) -> tuple[Any, dict[str, Any]]:
    depth_absolute = _masked_smooth_l1(
        prediction["depth"], target["depth"], valid_mask
    )
    depth_relative = _masked_smooth_l1(
        prediction["depth_relation"], target["depth_relation"], valid_mask
    )
    depth = 0.5 * (depth_absolute + depth_relative)
    camera = _masked_smooth_l1(
        prediction["point_camera"], target["point_camera"], valid_mask
    )
    world = _masked_smooth_l1(
        prediction["point_world"], target["point_world"], valid_mask
    )
    point3d = 0.5 * (camera + world)
    relation = _masked_smooth_l1(
        prediction["eef_object_world"], target["eef_object_world"], valid_mask
    )
    total = (
        weights.depth_relation * depth
        + weights.point3d * point3d
        + weights.eef_object_relation * relation
    )
    return total, {
        "depth_absolute": depth_absolute,
        "depth_relation": depth,
        "depth_relation_only": depth_relative,
        "point3d": point3d,
        "eef_object_relation": relation,
    }


def camera_points_to_world_torch(points: Any, camera_to_world: Any) -> Any:
    import torch

    if points.shape[-1] != 3 or camera_to_world.shape[-2:] != (4, 4):
        raise ValueError("invalid point/transform shape")
    rotation = camera_to_world[..., :3, :3].float()
    translation = camera_to_world[..., :3, 3].float()
    return torch.einsum("bij,btj->bti", rotation, points.float()) + translation[:, None]


def equivariance_loss(
    clean_prediction: Mapping[str, Any],
    camera_prediction: Mapping[str, Any],
    clean_camera_to_world: Any,
    camera_camera_to_world: Any,
    valid_mask: Any,
) -> Any:
    clean_world = camera_points_to_world_torch(
        clean_prediction["point_camera"], clean_camera_to_world
    )
    camera_world = camera_points_to_world_torch(
        camera_prediction["point_camera"], camera_camera_to_world
    )
    return 0.5 * (
        _masked_smooth_l1(clean_world, camera_world.detach(), valid_mask)
        + _masked_smooth_l1(camera_world, clean_world.detach(), valid_mask)
    )


def total_loss(
    *,
    original_fastwam_loss: Any,
    weights: LossWeights,
    repa: Any | None = None,
    equiv: Any | None = None,
    pose_aux: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Preserve the original loss and omit zero-weight graphs entirely."""

    zero = original_fastwam_loss.detach().new_zeros(())
    components: dict[str, Any] = {"original_fastwam": original_fastwam_loss}
    total = original_fastwam_loss
    for name, weight, value in (
        ("geo_repa", weights.lambda_repa, repa),
        ("equivariance", weights.lambda_equiv, equiv),
        ("pose_aux", weights.lambda_pose_aux, pose_aux),
    ):
        if weight == 0:
            components[name] = zero
            continue
        if value is None:
            raise ValueError(f"positive {name} weight requires a loss tensor")
        total = total + weight * value
        components[name] = value
    components["total"] = total
    return total, components
