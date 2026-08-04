from __future__ import annotations

import numpy as np
import pytest

from fastwam_ood_eval.thought5.camera_rays import (
    CameraRayError,
    assert_fastwam_token_layout,
    token_camera_rays,
    two_camera_token_rays,
)
from fastwam_ood_eval.thought5.geo_targets import (
    GeometryTargetError,
    assert_no_future_crossing,
    build_geometry_targets,
    shuffled_target_indices,
)
from fastwam_ood_eval.thought5.pose_transforms import (
    camera_to_world_points,
    relative_clean_to_camera,
    world_to_camera_points,
)
from fastwam_ood_eval.thought5.schemas import EpisodeKey, PairIdentity


def intrinsic() -> np.ndarray:
    return np.asarray([[100.0, 0.0, 49.5], [0.0, 100.0, 49.5], [0, 0, 1]])


def test_camera_ray_through_principal_point_is_forward() -> None:
    grid = token_camera_rays(
        intrinsic(),
        image_height=100,
        image_width=100,
        token_height=1,
        token_width=1,
    )
    np.testing.assert_allclose(grid.rays_camera[0], [0, 0, 1], atol=1e-7)


def test_singular_intrinsic_fails_closed() -> None:
    with pytest.raises(CameraRayError, match="singular"):
        token_camera_rays(
            np.zeros((3, 3)),
            image_height=10,
            image_width=10,
            token_height=1,
            token_width=1,
        )


def test_two_camera_token_layout_matches_98_tokens() -> None:
    grid = two_camera_token_rays(intrinsic())
    assert_fastwam_token_layout(grid, 98)
    assert grid.rays_camera.shape == (98, 3)
    assert grid.valid_mask.sum() == 49


def test_camera_world_transform_direction_roundtrip() -> None:
    camera_to_world = np.eye(4)
    camera_to_world[:3, 3] = [1, 2, 3]
    point = np.asarray([[0.2, -0.1, 1.0]])
    world = camera_to_world_points(camera_to_world, point)
    np.testing.assert_allclose(world, [[1.2, 1.9, 4.0]])
    np.testing.assert_allclose(world_to_camera_points(camera_to_world, world), point)


def test_relative_clean_to_camera_maps_clean_coordinates() -> None:
    clean = np.eye(4)
    condition = np.eye(4)
    condition[0, 3] = 1.0
    relative = relative_clean_to_camera(clean, condition)
    np.testing.assert_allclose(
        camera_to_world_points(relative, [[1.0, 0.0, 0.0]]), [[0.0, 0.0, 0.0]]
    )


def test_exact_camera_pair_requires_equal_state_hash() -> None:
    pair = PairIdentity("p", EpisodeKey(0, 1, 2), "camera", "a", "a", True)
    pair.validate()
    with pytest.raises(ValueError, match="unequal"):
        PairIdentity("p", EpisodeKey(0, 1, 2), "camera", "a", "b", True).validate()


def test_lighting_pair_is_exact_state() -> None:
    PairIdentity("p", EpisodeKey(0, 1, 2), "lighting", "x", "x", True).validate()


def test_robot_init_must_not_be_exact_state() -> None:
    PairIdentity("p", EpisodeKey(0, 1, 2), "robot_init", "x", "y", False).validate()
    with pytest.raises(ValueError, match="must not"):
        PairIdentity("p", EpisodeKey(0, 1, 2), "robot_init", "x", "x", True).validate()


def test_geometry_target_aligns_with_spatial_tokens_and_is_detached() -> None:
    grid = two_camera_token_rays(intrinsic(), image_height=100, camera_width=100)
    target = build_geometry_targets(
        depth_map=np.ones((100, 100), dtype=np.float32),
        ray_grid=grid,
        camera_to_world=np.eye(4),
        eef_position_world=[0, 0, 0],
        object_position_world=[1, 2, 3],
    )
    assert target.packed.shape == (98, 11)
    assert target.detached is True
    assert target.valid_mask.sum() == 49


def test_shuffled_geometry_is_a_true_derangement() -> None:
    indices = shuffled_target_indices(["a", "b", "c", "d"], seed=1)
    assert all(index != value for index, value in enumerate(indices))


def test_future_labels_cannot_cross_episode() -> None:
    assert_no_future_crossing(1, 2, [(1, 3), (1, 4)])
    with pytest.raises(GeometryTargetError, match="crosses"):
        assert_no_future_crossing(1, 2, [(2, 3)])

