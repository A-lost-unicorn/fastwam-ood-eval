from __future__ import annotations

import numpy as np
import pytest

from fastwam_ood_eval.thought4.geometry_labels import (
    axis_angle_to_matrix,
    build_future_trajectory_label,
    eef_object_relation,
    invert_transform,
    low_resolution_depth,
    relative_camera_pose,
    rotation_6d_to_matrix,
    rotation_geodesic_degrees,
    rotation_to_6d,
    transform_points,
    world_to_camera_points,
)
from fastwam_ood_eval.thought4.paired_rendering import (
    PairedRenderingError,
    PairedStateRenderer,
    array_sha256,
    validate_exact_state_group,
)
from fastwam_ood_eval.thought4.real_runtime import (
    DemonstrationEpisode,
    Thought4RuntimeError,
    _demonstration_state_alignment,
    _regenerate_input_observations,
    _robot_state_snapshot,
    _robot_states_matching_clean,
    _simulator_action_trajectory_world,
    _trajectory_labels_for_camera,
    _validate_input_robot_states,
)
from fastwam_ood_eval.thought4.schemas import (
    CameraMetadata,
    PairedRenderRecord,
    SampleIdentity,
    Thought4SchemaError,
    deterministic_episode_split,
    validate_episode_split,
)


DIGEST = "a" * 64


def identity(split: str = "test", episode: str = "e0") -> SampleIdentity:
    return SampleIdentity("task0", episode, 3, split, 0.15, "t3-to-t35")


def camera() -> CameraMetadata:
    return CameraMetadata.from_values(
        "agentview",
        np.eye(3),
        np.eye(4),
    )


def render_record(condition: str, *, state: str = DIGEST) -> PairedRenderRecord:
    return PairedRenderRecord(
        identity=identity(),
        condition=condition,
        condition_variant=f"{condition}_v1",
        exact_state_pair=condition != "robot_init",
        clean_reference_sample_id=DIGEST,
        clean_reference_state_sha256=DIGEST,
        simulator_state_sha256=state,
        object_eef_state_sha256=DIGEST,
        rgb_sha256=DIGEST,
        depth_sha256=DIGEST,
        camera=camera(),
        lighting_config_sha256=DIGEST,
    )


def test_episode_split_has_no_leakage_and_is_deterministic() -> None:
    records = (identity("train", "e0"), identity("train", "e0"), identity("test", "e1"))
    validate_episode_split(records)
    with pytest.raises(Thought4SchemaError, match="leakage"):
        validate_episode_split((identity("train", "e0"), identity("test", "e0")))
    first = deterministic_episode_split(
        [("t", f"e{i}") for i in range(8)],
        seed=4,
        train_count=4,
        development_count=2,
        test_count=2,
    )
    second = deterministic_episode_split(
        reversed([("t", f"e{i}") for i in range(8)]),
        seed=4,
        train_count=4,
        development_count=2,
        test_count=2,
    )
    assert first == second


def test_exact_state_pair_and_robot_init_semantics() -> None:
    records = [render_record(value) for value in ("clean", "camera", "lighting")]
    validate_exact_state_group(records)
    with pytest.raises(Thought4SchemaError, match="state hash differs"):
        render_record("camera", state="b" * 64)
    with pytest.raises(Thought4SchemaError, match="robot_init"):
        PairedRenderRecord(
            identity=identity(),
            condition="robot_init",
            condition_variant="robot",
            exact_state_pair=True,
            clean_reference_sample_id=DIGEST,
            clean_reference_state_sha256=DIGEST,
            simulator_state_sha256=DIGEST,
            object_eef_state_sha256=DIGEST,
            rgb_sha256=DIGEST,
            depth_sha256=DIGEST,
            camera=camera(),
            lighting_config_sha256=DIGEST,
        )


def test_camera_identity_is_stable_and_matrix_shapes_are_checked() -> None:
    first = camera()
    second = camera()
    assert first.identity_sha256 == second.identity_sha256
    with pytest.raises(Thought4SchemaError):
        CameraMetadata.from_values("agentview", np.eye(4), np.eye(4))


def test_coordinate_transforms_pose_and_rotation_roundtrip() -> None:
    transform = np.eye(4)
    transform[:3, 3] = [1, 2, 3]
    point = np.asarray([[1.0, 1.0, 1.0]])
    assert np.allclose(transform_points(transform, point), [[2, 3, 4]])
    assert np.allclose(
        transform_points(invert_transform(transform), [[2, 3, 4]]), point
    )
    assert np.allclose(world_to_camera_points(transform, [[1, 2, 4]]), [[0, 0, 1]])
    rotation = axis_angle_to_matrix([0, 0, np.pi / 2])
    six = rotation_to_6d(rotation)
    restored = rotation_6d_to_matrix(six)
    assert rotation_geodesic_degrees(restored, rotation) < 1e-5
    pose = relative_camera_pose(transform, np.eye(4))
    assert np.allclose(pose["translation"], [1, 2, 3])


def test_eef_object_relation_and_depth_labels() -> None:
    eef_rotation = axis_angle_to_matrix([0.0, 0.0, np.pi / 2])
    object_rotation = axis_angle_to_matrix([0.0, np.pi / 2, 0.0])
    camera_to_world = np.eye(4)
    camera_to_world[:3, :3] = axis_angle_to_matrix([np.pi / 4, 0.0, 0.0])
    relation = eef_object_relation(
        eef_position_world=[1, 1, 1],
        object_position_world=[2, 3, 4],
        camera_to_world=camera_to_world,
        eef_rotation_world=eef_rotation,
        object_rotation_world=object_rotation,
    )
    assert np.allclose(relation["eef_to_object_world"], [1, 2, 3])
    expected_camera = camera_to_world[:3, :3].T @ np.asarray([1, 2, 3])
    assert np.allclose(relation["eef_to_object_camera"], expected_camera)
    assert np.allclose(
        rotation_6d_to_matrix(relation["relative_orientation_camera_6d"]),
        rotation_6d_to_matrix(relation["relative_orientation_world_6d"]),
    )
    depth = np.arange(1, 17, dtype=float).reshape(4, 4)
    low = low_resolution_depth(depth, (2, 2))
    assert low.shape == (2, 2)
    assert np.isfinite(low).all()


def test_future_trajectory_never_crosses_episode() -> None:
    episodes = ["a", "a", "a", "b", "b"]
    positions = np.arange(15, dtype=float).reshape(5, 3)
    rotations = np.repeat(np.eye(3)[None], 5, axis=0)
    label = build_future_trajectory_label(
        input_index=1,
        episode_ids=episodes,
        eef_positions_world=positions,
        eef_rotations_world=rotations,
        gripper_values=np.zeros(5),
        camera_to_world=np.eye(4),
        horizon=4,
    )
    assert label.label_indices == (2, -1, -1, -1)
    assert label.valid_mask.tolist() == [True, False, False, False]


def test_demonstration_prefix_alignment_is_checked_at_input_time() -> None:
    episode = DemonstrationEpisode(
        actions=np.zeros((2, 7), dtype=np.float32),
        eef_states=np.zeros((2, 6), dtype=np.float64),
        gripper_values=np.zeros((2, 1), dtype=np.float32),
        episode_ids=("0", "0"),
    )
    observation = {
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
    }
    result = _demonstration_state_alignment(observation, episode, 1)
    assert result["passed"] is True
    with pytest.raises(Thought4RuntimeError, match="alignment failed"):
        _demonstration_state_alignment(
            {**observation, "robot0_eef_pos": np.asarray([0.1, 0.0, 0.0])},
            episode,
            1,
        )
    disclosed = _demonstration_state_alignment(
        {**observation, "robot0_eef_pos": np.asarray([0.1, 0.0, 0.0])},
        episode,
        1,
        strict=False,
    )
    assert disclosed["passed"] is False
    assert disclosed["enforcement"] == "disclosure_only_3cm_15deg"
    assert disclosed["translation_error_m"] == pytest.approx(0.1)


class _ReplayEnv:
    def __init__(self, position: np.ndarray) -> None:
        self.position = np.asarray(position, dtype=np.float64).copy()
        self.step_count = 0

    def get_sim_state(self) -> np.ndarray:
        return self.position.copy()

    def step(self, action: np.ndarray):
        self.step_count += 1
        self.position += np.asarray(action[:3], dtype=np.float64)
        observation = {
            "robot0_eef_pos": self.position.copy(),
            "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
        }
        return observation, 0.0, False, {}


class _ReplayAdapter:
    def __init__(self, position: np.ndarray) -> None:
        self.env = _ReplayEnv(position)


def test_simulator_action_labels_share_world_replay_and_restore_input_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = np.asarray([1.0, 2.0, 3.0])
    adapter = _ReplayAdapter(initial)
    actions = np.zeros((4, 7), dtype=np.float32)
    actions[1, :3] = [0.1, 0.0, 0.0]
    actions[1, -1] = 0.2
    actions[2, :3] = [0.0, 0.2, 0.0]
    actions[2, -1] = 0.8
    episode = DemonstrationEpisode(
        actions=actions,
        eef_states=np.zeros((4, 6), dtype=np.float64),
        gripper_values=actions[:, -1:].copy(),
        episode_ids=("0", "0", "0", "0"),
    )

    def restore_state(replay_adapter, state):
        replay_adapter.env.position = np.asarray(state, dtype=np.float64).copy()
        return {
            "robot0_eef_pos": replay_adapter.env.position.copy(),
            "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
        }

    monkeypatch.setattr(
        "fastwam_ood_eval.thought4.real_runtime._observation_for_state",
        restore_state,
    )
    world = _simulator_action_trajectory_world(
        adapter,
        episode,
        frame_index=1,
        horizon=2,
        current_observation={
            "robot0_eef_pos": initial.copy(),
            "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
        },
    )
    result = _trajectory_labels_for_camera(world, np.eye(4))
    same_camera = _trajectory_labels_for_camera(world, np.eye(4))
    rotated_camera_to_world = np.eye(4)
    rotated_camera_to_world[:3, :3] = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    rotated = _trajectory_labels_for_camera(
        world, rotated_camera_to_world
    )
    assert np.allclose(
        result["translation_camera"],
        [[0.1, 0.0, 0.0], [0.1, 0.2, 0.0]],
    )
    assert np.array_equal(
        result["translation_camera"], same_camera["translation_camera"]
    )
    assert np.allclose(
        rotated["translation_camera"],
        [[0.0, -0.1, 0.0], [0.2, -0.1, 0.0]],
    )
    assert np.allclose(result["gripper"].reshape(-1), [0.2, 0.8])
    assert result["valid_mask"].tolist() == [True, True]
    assert adapter.env.step_count == 2
    assert np.allclose(adapter.env.position, initial)


def _robot_observation(joint_offset: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "robot0_joint_pos": np.asarray([joint_offset, 0.2]),
        "robot0_eef_pos": np.asarray([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.asarray([0.01, -0.01]),
    }


def test_robot_init_is_validated_at_model_input_not_reset() -> None:
    reset = {
        condition: _robot_state_snapshot(_robot_observation())
        for condition in ("clean", "camera", "lighting", "robot_init")
    }
    # Shared demonstration reset state is allowed to mask Robot-init at reset.
    assert _robot_states_matching_clean(reset)["robot_init"] is True

    input_states = dict(reset)
    input_states["robot_init"] = _robot_state_snapshot(
        _robot_observation(joint_offset=0.05)
    )
    matches = _validate_input_robot_states(input_states)
    assert matches["clean"] is True
    assert matches["camera"] is True
    assert matches["lighting"] is True
    assert matches["robot_init"] is False

    with pytest.raises(Thought4RuntimeError, match="model input time"):
        _validate_input_robot_states(reset)

    bad_camera = dict(input_states)
    bad_camera["camera"] = _robot_state_snapshot(
        _robot_observation(joint_offset=0.02)
    )
    with pytest.raises(Thought4RuntimeError, match="camera robot state"):
        _validate_input_robot_states(bad_camera)


class _FakeStateEnv:
    def __init__(self, state: np.ndarray) -> None:
        self.state = np.asarray(state, dtype=np.float64)

    def get_sim_state(self) -> np.ndarray:
        return self.state.copy()


class _FakeStateAdapter:
    def __init__(self, name: str, state: np.ndarray) -> None:
        self.name = name
        self.env = _FakeStateEnv(state)


def test_all_input_observations_use_the_same_refresh_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clean_state = np.asarray([1.0, 2.0], dtype=np.float64)
    adapters = {
        condition: _FakeStateAdapter(
            condition,
            (
                np.asarray([9.0, 8.0], dtype=np.float64)
                if condition == "robot_init"
                else np.asarray([-1.0, -1.0], dtype=np.float64)
            ),
        )
        for condition in ("clean", "camera", "lighting", "robot_init")
    }
    calls: list[tuple[str, np.ndarray]] = []

    def fake_observation_for_state(
        adapter: _FakeStateAdapter, state: np.ndarray
    ) -> dict[str, np.ndarray]:
        value = np.asarray(state).copy()
        calls.append((adapter.name, value))
        return {"state": value}

    monkeypatch.setattr(
        "fastwam_ood_eval.thought4.real_runtime._observation_for_state",
        fake_observation_for_state,
    )
    observations = _regenerate_input_observations(adapters, clean_state)
    assert set(observations) == set(adapters)
    by_condition = {condition: state for condition, state in calls}
    for condition in ("clean", "camera", "lighting"):
        assert np.array_equal(by_condition[condition], clean_state)
    assert np.array_equal(
        by_condition["robot_init"], np.asarray([9.0, 8.0])
    )


def test_robot_init_render_requires_input_simulator_state_difference() -> None:
    renderer = PairedStateRenderer(
        camera_name="agentview",
        height=2,
        width=2,
        render_function=lambda _env, _camera, _height, _width: (
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.ones((2, 2), dtype=np.float32),
            camera(),
            {"eef": [0.0, 0.0, 0.0]},
        ),
    )
    clean_state = np.asarray([0.0, 1.0], dtype=np.float64)
    kwargs = {
        "identity": identity(),
        "variant": "robot_init_v1",
        "clean_reference_sample_id": identity().sample_id,
        "clean_reference_state_sha256": array_sha256(clean_state),
    }
    with pytest.raises(PairedRenderingError, match="model input time"):
        renderer.render_robot_init(
            robot_init_env=_FakeStateEnv(clean_state),
            **kwargs,
        )
    rendered = renderer.render_robot_init(
        robot_init_env=_FakeStateEnv(np.asarray([0.1, 1.0])),
        **kwargs,
    )
    assert rendered.record.exact_state_pair is False
