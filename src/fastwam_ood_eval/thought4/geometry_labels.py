"""Geometry and future-trajectory labels with explicit coordinate frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


class GeometryLabelError(ValueError):
    """Raised for invalid geometry, coordinates or episode boundaries."""


def _array(value: Any, shape: tuple[int, ...], name: str) -> Any:
    import numpy as np

    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise GeometryLabelError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise GeometryLabelError(f"{name} contains NaN/Inf")
    return array


def validate_rotation_matrix(rotation: Any, *, atol: float = 1e-5) -> Any:
    import numpy as np

    matrix = _array(rotation, (3, 3), "rotation")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=atol):
        raise GeometryLabelError("rotation is not orthonormal")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=atol):
        raise GeometryLabelError("rotation determinant is not +1")
    return matrix


def rotation_to_6d(rotation: Any) -> Any:
    """Return the first two rotation columns in column-major order."""

    matrix = validate_rotation_matrix(rotation)
    return matrix[:, :2].T.reshape(6)


def rotation_6d_to_matrix(value: Any) -> Any:
    import numpy as np

    vectors = _array(value, (6,), "rotation_6d").reshape(2, 3)
    first = vectors[0]
    first_norm = np.linalg.norm(first)
    if first_norm <= 1e-12:
        raise GeometryLabelError("rotation_6d first vector is degenerate")
    first = first / first_norm
    second = vectors[1] - np.dot(first, vectors[1]) * first
    second_norm = np.linalg.norm(second)
    if second_norm <= 1e-12:
        raise GeometryLabelError("rotation_6d second vector is degenerate")
    second = second / second_norm
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=1)


def quaternion_xyzw_to_matrix(quaternion: Any) -> Any:
    import numpy as np

    q = _array(quaternion, (4,), "quaternion_xyzw")
    norm = np.linalg.norm(q)
    if norm <= 1e-12:
        raise GeometryLabelError("quaternion is degenerate")
    x, y, z, w = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def axis_angle_to_matrix(axis_angle: Any) -> Any:
    import numpy as np

    vector = _array(axis_angle, (3,), "axis_angle")
    angle = float(np.linalg.norm(vector))
    if angle <= 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = vector / angle
    skew = np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return (
        np.eye(3)
        + np.sin(angle) * skew
        + (1.0 - np.cos(angle)) * (skew @ skew)
    )


def rotation_geodesic_degrees(prediction: Any, target: Any) -> float:
    import numpy as np

    pred = validate_rotation_matrix(prediction)
    truth = validate_rotation_matrix(target)
    cosine = np.clip((np.trace(pred.T @ truth) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def invert_transform(transform: Any) -> Any:
    import numpy as np

    value = _array(transform, (4, 4), "transform")
    rotation = validate_rotation_matrix(value[:3, :3])
    if not np.allclose(value[3], [0, 0, 0, 1], atol=1e-7):
        raise GeometryLabelError("invalid homogeneous transform bottom row")
    result = np.eye(4)
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ value[:3, 3]
    return result


def transform_points(transform: Any, points: Any) -> Any:
    import numpy as np

    value = _array(transform, (4, 4), "transform")
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.shape[-1] != 3 or not np.isfinite(point_array).all():
        raise GeometryLabelError("points must be finite [...,3]")
    homogeneous = np.concatenate(
        (point_array, np.ones((*point_array.shape[:-1], 1))), axis=-1
    )
    return (homogeneous @ value.T)[..., :3]


def world_to_camera_points(camera_to_world: Any, points_world: Any) -> Any:
    return transform_points(invert_transform(camera_to_world), points_world)


def relative_camera_pose(
    current_camera_to_world: Any,
    clean_camera_to_world: Any,
) -> dict[str, Any]:
    """Pose mapping current-camera coordinates into Clean-camera coordinates."""

    current = _array(current_camera_to_world, (4, 4), "current_camera_to_world")
    clean = _array(clean_camera_to_world, (4, 4), "clean_camera_to_world")
    relative = invert_transform(clean) @ current
    return {
        "translation": relative[:3, 3].astype("float32"),
        "rotation_6d": rotation_to_6d(relative[:3, :3]).astype("float32"),
        "transform_current_to_clean": relative.astype("float32"),
    }


def eef_object_relation(
    *,
    eef_position_world: Any,
    object_position_world: Any,
    camera_to_world: Any,
    eef_rotation_world: Any | None = None,
    object_rotation_world: Any | None = None,
) -> dict[str, Any]:
    import numpy as np

    eef = _array(eef_position_world, (3,), "eef_position_world")
    obj = _array(object_position_world, (3,), "object_position_world")
    world_delta = obj - eef
    camera_rotation = invert_transform(camera_to_world)[:3, :3]
    result: dict[str, Any] = {
        "eef_to_object_world": world_delta.astype("float32"),
        "eef_to_object_camera": (camera_rotation @ world_delta).astype("float32"),
    }
    if eef_rotation_world is not None and object_rotation_world is not None:
        eef_rotation = validate_rotation_matrix(eef_rotation_world)
        object_rotation = validate_rotation_matrix(object_rotation_world)
        relative_world = eef_rotation.T @ object_rotation
        # Expressing both bodies in another common frame must not reverse the
        # relative transform: R_eef^T R_object is frame invariant.
        relative_camera = (
            camera_rotation @ eef_rotation
        ).T @ (camera_rotation @ object_rotation)
        result["relative_orientation_world_6d"] = rotation_to_6d(
            relative_world
        ).astype("float32")
        result["relative_orientation_camera_6d"] = rotation_to_6d(
            relative_camera
        ).astype("float32")
    return result


def relative_depth(depth: Any, *, epsilon: float = 1e-6) -> Any:
    import numpy as np

    value = np.asarray(depth, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all() or bool((value <= 0).any()):
        raise GeometryLabelError("depth must be finite positive [H,W]")
    scale = float(np.median(value))
    if scale <= epsilon:
        raise GeometryLabelError("depth median is degenerate")
    return (value / scale).astype("float32")


def low_resolution_depth(depth: Any, output_hw: tuple[int, int]) -> Any:
    """Area-average depth into a frozen grid; dimensions must divide exactly."""

    import numpy as np

    value = relative_depth(depth)
    out_h, out_w = output_hw
    if min(out_h, out_w) <= 0:
        raise GeometryLabelError("output depth dimensions must be positive")
    height, width = value.shape
    if height % out_h or width % out_w:
        raise GeometryLabelError(
            f"depth shape {(height, width)} not divisible by {output_hw}"
        )
    return value.reshape(
        out_h, height // out_h, out_w, width // out_w
    ).mean(axis=(1, 3))


def ordinal_depth_labels(
    depth: Any,
    pairs: Sequence[tuple[tuple[int, int], tuple[int, int]]],
    *,
    tolerance: float = 1e-4,
) -> Any:
    import numpy as np

    value = relative_depth(depth)
    labels: list[int] = []
    for first, second in pairs:
        try:
            delta = float(value[first] - value[second])
        except IndexError as exc:
            raise GeometryLabelError("ordinal depth coordinate is out of bounds") from exc
        labels.append(0 if abs(delta) <= tolerance else 1 if delta > 0 else -1)
    return np.asarray(labels, dtype=np.int8)


@dataclass(frozen=True)
class FutureTrajectoryLabel:
    input_index: int
    label_indices: tuple[int, ...]
    translation_world: Any
    translation_camera: Any
    rotation_6d: Any
    gripper: Any
    valid_mask: Any


def build_future_trajectory_label(
    *,
    input_index: int,
    episode_ids: Sequence[str],
    eef_positions_world: Any,
    eef_rotations_world: Any,
    gripper_values: Any,
    camera_to_world: Any,
    horizon: int,
) -> FutureTrajectoryLabel:
    """Build t+1...t+H labels without crossing an episode boundary."""

    import numpy as np

    positions = np.asarray(eef_positions_world, dtype=np.float64)
    rotations = np.asarray(eef_rotations_world, dtype=np.float64)
    gripper = np.asarray(gripper_values, dtype=np.float64)
    total = len(episode_ids)
    if positions.shape != (total, 3):
        raise GeometryLabelError("EEF positions must have [N,3] shape")
    if rotations.shape != (total, 3, 3):
        raise GeometryLabelError("EEF rotations must have [N,3,3] shape")
    if gripper.shape not in {(total,), (total, 1)}:
        raise GeometryLabelError("gripper values must have [N] or [N,1] shape")
    if not 0 <= input_index < total or horizon <= 0:
        raise GeometryLabelError("invalid input_index or horizon")
    if not (
        np.isfinite(positions).all()
        and np.isfinite(rotations).all()
        and np.isfinite(gripper).all()
    ):
        raise GeometryLabelError("future trajectory source contains NaN/Inf")
    current_episode = str(episode_ids[input_index])
    current_position = positions[input_index]
    world_to_camera = invert_transform(camera_to_world)[:3, :3]
    translation_world = np.zeros((horizon, 3), dtype=np.float32)
    translation_camera = np.zeros((horizon, 3), dtype=np.float32)
    rotation_labels = np.zeros((horizon, 6), dtype=np.float32)
    gripper_labels = np.zeros((horizon, 1), dtype=np.float32)
    valid = np.zeros(horizon, dtype=bool)
    indices: list[int] = []
    for offset in range(1, horizon + 1):
        index = input_index + offset
        if index >= total or str(episode_ids[index]) != current_episode:
            indices.append(-1)
            continue
        delta = positions[index] - current_position
        translation_world[offset - 1] = delta
        translation_camera[offset - 1] = world_to_camera @ delta
        rotation_labels[offset - 1] = rotation_to_6d(rotations[index])
        gripper_labels[offset - 1, 0] = float(gripper[index])
        valid[offset - 1] = True
        indices.append(index)
    return FutureTrajectoryLabel(
        input_index=input_index,
        label_indices=tuple(indices),
        translation_world=translation_world,
        translation_camera=translation_camera,
        rotation_6d=rotation_labels,
        gripper=gripper_labels,
        valid_mask=valid,
    )
