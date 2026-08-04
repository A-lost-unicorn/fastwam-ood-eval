"""Explicit camera/world transform conventions used by Phase 5."""

from __future__ import annotations

from typing import Any

import numpy as np


class PoseTransformError(ValueError):
    pass


def homogeneous(value: Any, *, name: str = "transform") -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise PoseTransformError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise PoseTransformError(f"{name} has an invalid homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise PoseTransformError(f"{name} rotation is not orthonormal")
    if np.linalg.det(rotation) < 0.999 or np.linalg.det(rotation) > 1.001:
        raise PoseTransformError(f"{name} rotation determinant is not +1")
    return matrix


def invert_transform(value: Any) -> np.ndarray:
    transform = homogeneous(value)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation.T
    result[:3, 3] = -(rotation.T @ translation)
    return result


def transform_points(transform: Any, points: Any) -> np.ndarray:
    matrix = homogeneous(transform)
    values = np.asarray(points, dtype=np.float64)
    if values.shape[-1] != 3 or not np.isfinite(values).all():
        raise PoseTransformError("points must be finite with final dimension 3")
    original_shape = values.shape
    flat = values.reshape(-1, 3)
    output = (matrix[:3, :3] @ flat.T).T + matrix[:3, 3]
    return output.reshape(original_shape)


def camera_to_world_points(camera_to_world: Any, points_camera: Any) -> np.ndarray:
    return transform_points(camera_to_world, points_camera)


def world_to_camera_points(camera_to_world: Any, points_world: Any) -> np.ndarray:
    return transform_points(invert_transform(camera_to_world), points_world)


def relative_clean_to_camera(
    clean_camera_to_world: Any, condition_camera_to_world: Any
) -> np.ndarray:
    """Map coordinates in the clean camera frame into condition camera frame."""

    clean = homogeneous(clean_camera_to_world, name="clean_camera_to_world")
    condition = homogeneous(
        condition_camera_to_world, name="condition_camera_to_world"
    )
    return invert_transform(condition) @ clean


def pose_embedding_12(camera_to_world: Any) -> np.ndarray:
    """Flatten the 3x4 camera-to-world matrix for a lightweight encoder."""

    return homogeneous(camera_to_world)[:3, :4].reshape(12).astype(np.float32)
