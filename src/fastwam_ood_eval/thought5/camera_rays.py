"""Camera-ray construction aligned to Fast-WAM's 7x14 two-camera tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class CameraRayError(ValueError):
    pass


@dataclass(frozen=True)
class TokenRayGrid:
    rays_camera: np.ndarray
    pixel_centers: np.ndarray
    valid_mask: np.ndarray
    token_height: int
    token_width: int
    layout: str = "frame_major_row_major"


def _intrinsic(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise CameraRayError("intrinsic must be a finite 3x3 matrix")
    if abs(np.linalg.det(matrix)) < 1e-12:
        raise CameraRayError("intrinsic is singular")
    return matrix


def token_camera_rays(
    intrinsic: Any,
    *,
    image_height: int,
    image_width: int,
    token_height: int,
    token_width: int,
    normalize: bool = True,
) -> TokenRayGrid:
    """Generate rays through area-centred spatial tokens.

    Tokens are flattened row-major, matching Fast-WAM's ``(f h w)`` order for
    a single current frame. Pixel coordinates follow the robosuite/OpenCV
    convention ``[u, v, 1]`` and rays are ``K^-1 [u, v, 1]``.
    """

    if min(image_height, image_width, token_height, token_width) <= 0:
        raise CameraRayError("image and token dimensions must be positive")
    matrix = _intrinsic(intrinsic)
    u = (np.arange(token_width, dtype=np.float64) + 0.5) * (
        image_width / token_width
    ) - 0.5
    v = (np.arange(token_height, dtype=np.float64) + 0.5) * (
        image_height / token_height
    ) - 0.5
    vv, uu = np.meshgrid(v, u, indexing="ij")
    pixels = np.stack([uu, vv, np.ones_like(uu)], axis=-1).reshape(-1, 3)
    rays = (np.linalg.inv(matrix) @ pixels.T).T
    if normalize:
        norms = np.linalg.norm(rays, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise CameraRayError("zero-length ray")
        rays = rays / norms
    return TokenRayGrid(
        rays_camera=rays.astype(np.float32),
        pixel_centers=pixels[:, :2].astype(np.float32),
        valid_mask=np.ones(len(rays), dtype=bool),
        token_height=token_height,
        token_width=token_width,
    )


def two_camera_token_rays(
    primary_intrinsic: Any,
    *,
    image_height: int = 224,
    camera_width: int = 224,
    token_height: int = 7,
    tokens_per_camera_width: int = 7,
    wrist_intrinsic: Any | None = None,
) -> TokenRayGrid:
    """Build the 7x14 ray grid; mask wrist tokens if metadata is unavailable."""

    primary = token_camera_rays(
        primary_intrinsic,
        image_height=image_height,
        image_width=camera_width,
        token_height=token_height,
        token_width=tokens_per_camera_width,
    )
    if wrist_intrinsic is None:
        wrist_rays = np.zeros_like(primary.rays_camera)
        wrist_pixels = np.zeros_like(primary.pixel_centers)
        wrist_valid = np.zeros_like(primary.valid_mask)
    else:
        wrist = token_camera_rays(
            wrist_intrinsic,
            image_height=image_height,
            image_width=camera_width,
            token_height=token_height,
            token_width=tokens_per_camera_width,
        )
        wrist_rays, wrist_pixels, wrist_valid = (
            wrist.rays_camera,
            wrist.pixel_centers,
            wrist.valid_mask,
        )
    primary_rays = primary.rays_camera.reshape(token_height, -1, 3)
    wrist_rays = wrist_rays.reshape(token_height, -1, 3)
    primary_pixels = primary.pixel_centers.reshape(token_height, -1, 2)
    wrist_pixels = wrist_pixels.reshape(token_height, -1, 2)
    valid = np.concatenate(
        [
            primary.valid_mask.reshape(token_height, -1),
            wrist_valid.reshape(token_height, -1),
        ],
        axis=1,
    )
    return TokenRayGrid(
        rays_camera=np.concatenate([primary_rays, wrist_rays], axis=1).reshape(-1, 3),
        pixel_centers=np.concatenate(
            [primary_pixels, wrist_pixels], axis=1
        ).reshape(-1, 2),
        valid_mask=valid.reshape(-1),
        token_height=token_height,
        token_width=2 * tokens_per_camera_width,
    )


def assert_fastwam_token_layout(grid: TokenRayGrid, token_count: int) -> None:
    if grid.layout != "frame_major_row_major":
        raise CameraRayError("unsupported token flattening layout")
    if grid.rays_camera.shape != (token_count, 3):
        raise CameraRayError(
            f"ray/token mismatch: rays={grid.rays_camera.shape}, tokens={token_count}"
        )
    if token_count != grid.token_height * grid.token_width:
        raise CameraRayError("token count is inconsistent with the spatial grid")
