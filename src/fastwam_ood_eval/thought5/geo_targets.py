"""Detached simulator geometry targets aligned with Video DiT tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fastwam_ood_eval.thought5.camera_rays import TokenRayGrid
from fastwam_ood_eval.thought5.pose_transforms import camera_to_world_points


class GeometryTargetError(ValueError):
    pass


@dataclass(frozen=True)
class GeometryTargets:
    depth: np.ndarray
    depth_relation: np.ndarray
    points_camera: np.ndarray
    points_world: np.ndarray
    eef_object_translation: np.ndarray
    valid_mask: np.ndarray
    detached: bool = True

    @property
    def packed(self) -> np.ndarray:
        relation = np.repeat(
            self.eef_object_translation.reshape(1, 3), len(self.depth), axis=0
        )
        return np.concatenate(
            [
                self.depth[:, None],
                self.depth_relation[:, None],
                self.points_camera,
                self.points_world,
                relation,
            ],
            axis=1,
        ).astype(np.float32)


def _sample_depth(depth_map: Any, pixel_centers: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_map, dtype=np.float64)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2 or not np.isfinite(depth).all():
        raise GeometryTargetError("depth must be a finite HxW metric map")
    u = np.clip(np.rint(pixel_centers[:, 0]).astype(int), 0, depth.shape[1] - 1)
    v = np.clip(np.rint(pixel_centers[:, 1]).astype(int), 0, depth.shape[0] - 1)
    return depth[v, u]


def build_geometry_targets(
    *,
    depth_map: Any,
    ray_grid: TokenRayGrid,
    camera_to_world: Any,
    eef_position_world: Any,
    object_position_world: Any,
) -> GeometryTargets:
    depth = _sample_depth(depth_map, ray_grid.pixel_centers)
    valid = ray_grid.valid_mask & np.isfinite(depth) & (depth > 0)
    points_camera = ray_grid.rays_camera.astype(np.float64) * depth[:, None]
    points_world = camera_to_world_points(camera_to_world, points_camera)
    eef = np.asarray(eef_position_world, dtype=np.float64).reshape(3)
    obj = np.asarray(object_position_world, dtype=np.float64).reshape(3)
    if not np.isfinite(eef).all() or not np.isfinite(obj).all():
        raise GeometryTargetError("EEF/object positions must be finite")
    relation = obj - eef
    grid_depth = depth.reshape(ray_grid.token_height, ray_grid.token_width)
    relation_depth = np.zeros_like(grid_depth)
    relation_depth[:, :-1] = grid_depth[:, 1:] - grid_depth[:, :-1]
    relation_depth[:, -1] = relation_depth[:, -2]
    return GeometryTargets(
        depth=depth.astype(np.float32),
        depth_relation=relation_depth.reshape(-1).astype(np.float32),
        points_camera=points_camera.astype(np.float32),
        points_world=points_world.astype(np.float32),
        eef_object_translation=relation.astype(np.float32),
        valid_mask=valid,
    )


def shuffled_target_indices(pair_ids: list[str], *, seed: int) -> list[int]:
    """Return a deterministic derangement for the G4 geometry control."""

    if len(pair_ids) < 2 or len(set(pair_ids)) != len(pair_ids):
        raise GeometryTargetError("shuffled control needs >=2 unique pair IDs")
    rng = np.random.default_rng(seed)
    candidate = np.arange(len(pair_ids))
    for _ in range(1000):
        rng.shuffle(candidate)
        if np.all(candidate != np.arange(len(pair_ids))):
            return candidate.tolist()
    raise GeometryTargetError("could not construct shuffled geometry derangement")


def assert_no_future_crossing(
    episode_index: int, current_frame: int, future_frames: list[tuple[int, int]]
) -> None:
    for future_episode, future_frame in future_frames:
        if future_episode != episode_index or future_frame <= current_frame:
            raise GeometryTargetError("future label crosses episode or time boundary")
