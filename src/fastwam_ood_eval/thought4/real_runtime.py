"""Real LIBERO/LIBERO-Plus and frozen Fast-WAM runtime for Phase 4.

The renderer and model are intentionally used in separate stages so MuJoCo EGL
and a 24 GiB Fast-WAM load do not compete for GPU memory.  This module never
queries or records task success.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from fastwam_ood_eval.thought4.cohort import PlannedBaseState
from fastwam_ood_eval.thought4.config import Thought4Config
from fastwam_ood_eval.thought4.geometry_labels import (
    axis_angle_to_matrix,
    build_future_trajectory_label,
    eef_object_relation,
    relative_camera_pose,
    rotation_to_6d,
)
from fastwam_ood_eval.thought4.paired_rendering import (
    PairedRenderingError,
    PairedStateRenderer,
    RenderedCondition,
    array_sha256,
    camera_metadata,
    render_rgb_depth,
)
from fastwam_ood_eval.thought4.schemas import SampleIdentity


class Thought4RuntimeError(RuntimeError):
    """Raised when the real diagnostic runtime cannot preserve its contract."""


@dataclass
class DemonstrationEpisode:
    actions: Any
    eef_states: Any
    gripper_values: Any
    episode_ids: tuple[str, ...]


@dataclass
class RenderedProbeSample:
    plan: PlannedBaseState
    condition: str
    rendered: RenderedCondition
    observation: Mapping[str, Any]
    labels: Mapping[str, Any]
    masks: Mapping[str, Any]
    task_description: str
    trajectory_label_source: str
    initial_object_layout_sha256: str
    initial_object_layout_matches_clean: bool
    reset_robot_state_sha256: str
    reset_robot_state_matches_clean: bool
    input_robot_state_sha256: str
    input_robot_state_matches_clean: bool
    demonstration_state_alignment: Mapping[str, Any]


@dataclass
class FrozenFastWAMRuntime:
    model: Any
    upstream_cfg: Any
    processor: Any
    official: Any
    action_horizon: int
    input_height: int
    input_width: int
    load_latency_s: float


def _classification_rows() -> list[dict[str, Any]]:
    path = Path(
        "third_party/LIBERO-plus/libero/libero/benchmark/"
        "task_classification.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("libero_goal")
    if not isinstance(rows, list):
        raise Thought4RuntimeError("LIBERO-Plus libero_goal classification missing")
    return rows


def _condition_task_catalog(
    cfg: Thought4Config,
) -> dict[str, tuple[tuple[int, str], ...]]:
    ids = dict(cfg.cohort.condition_task_ids)
    rows = {
        int(row["id"]): (str(row["name"]), str(row["category"]))
        for row in _classification_rows()
    }
    base_names = {
        0: "open_the_middle_drawer_of_the_cabinet",
        1: "put_the_bowl_on_the_stove",
        2: "put_the_wine_bottle_on_top_of_the_cabinet",
        3: "open_the_top_drawer_and_put_the_bowl_inside",
        4: "put_the_bowl_on_top_of_the_cabinet",
        5: "push_the_plate_to_the_front_of_the_stove",
        6: "put_the_cream_cheese_in_the_bowl",
        7: "turn_on_the_stove",
        8: "put_the_bowl_on_the_plate",
        9: "put_the_wine_bottle_on_the_rack",
    }
    if len(cfg.cohort.task_ids) != 1:
        raise Thought4RuntimeError(
            "v1 real renderer supports one preregistered base task per run"
        )
    base_task = cfg.cohort.task_ids[0]
    clean_id = ids["clean"][0]
    if clean_id != base_task + 1:
        raise Thought4RuntimeError(
            "clean classification ID must equal one-based base task ID"
        )
    catalog: dict[str, tuple[tuple[int, str], ...]] = {
        "clean": ((clean_id, base_names[base_task]),)
    }
    for condition in ("camera", "lighting", "robot_init"):
        expected_category = {
            "camera": "Camera Viewpoints",
            "lighting": "Light Conditions",
            "robot_init": "Robot Initial States",
        }[condition]
        values: list[tuple[int, str]] = []
        for classification_id in ids[condition]:
            if classification_id not in rows:
                raise Thought4RuntimeError(
                    f"unknown LIBERO-Plus classification ID: {classification_id}"
                )
            name, category = rows[classification_id]
            if category != expected_category:
                raise Thought4RuntimeError(
                    f"{condition} ID {classification_id} has category {category}"
                )
            if not name.startswith(base_names[base_task] + "_"):
                raise Thought4RuntimeError(
                    f"{condition} variant does not match frozen base task"
                )
            values.append((classification_id, name))
        catalog[condition] = tuple(values)
    return catalog


def _select_condition_task(
    cfg: Thought4Config,
    plan: PlannedBaseState,
    condition: str,
    *,
    catalog: Mapping[str, Sequence[tuple[int, str]]] | None = None,
) -> tuple[int, str]:
    import hashlib

    values = (
        catalog[condition]
        if catalog is not None
        else _condition_task_catalog(cfg)[condition]
    )
    digest = hashlib.sha256(
        (
            f"thought4-condition-variant-v1\0{cfg.cohort.split_seed}\0"
            f"{condition}\0{plan.task_index}\0{plan.episode_index}\0"
            f"{plan.frame_index}"
        ).encode()
    ).digest()
    return values[int.from_bytes(digest[:8], "big") % len(values)]


def _make_job(
    cfg: Thought4Config,
    *,
    condition: str,
    plan: PlannedBaseState,
    classification_id: int,
    task_name: str,
    clean_task_name: str,
) -> Any:
    from fastwam_ood_eval.evaluation.jobs import EvaluationJob

    return EvaluationJob(
        experiment_id=cfg.experiment.name,
        job_id=f"thought4-{condition}-{plan.identity.sample_id[:16]}",
        suite=cfg.cohort.suite,
        task_id=plan.task_index,
        task_name=clean_task_name,
        upstream_task_id=classification_id - 1,
        upstream_task_name=task_name,
        episode_index=plan.task_local_episode_index,
        episode_seed=cfg.experiment.seed + plan.task_local_episode_index,
        initial_state_index=plan.task_local_episode_index,
        condition=condition,
        perturbation_category=condition if condition != "clean" else None,
        perturbation_level="frozen_v1" if condition != "clean" else None,
        perturbation_parameters={
            "classification_id": classification_id,
            "diagnostic_only": True,
        },
        policy_variant="fastwam",
        test_time_future_imagination=False,
        comparison_group="thought4_exact_state",
    )


def load_demonstration_episode(
    dataset_root: Path,
    episode_index: int,
) -> DemonstrationEpisode:
    """Load only actions/EEF labels; no future RGB is read."""

    try:
        import numpy as np
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise Thought4RuntimeError(
            "real Phase 4 requires NumPy and PyArrow in the Fast-WAM environment"
        ) from exc
    path = (
        dataset_root
        / "data"
        / f"chunk-{episode_index // 1000:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )
    if not path.is_file():
        raise Thought4RuntimeError(f"demonstration parquet missing: {path}")
    table = parquet.read_table(
        path,
        columns=[
            "action",
            "observation.states.ee_state",
            "episode_index",
        ],
    )
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    eef = np.asarray(
        table["observation.states.ee_state"].to_pylist(), dtype=np.float64
    )
    episode_column = tuple(str(value.as_py()) for value in table["episode_index"])
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise Thought4RuntimeError(f"unexpected action shape: {actions.shape}")
    if eef.ndim != 2 or eef.shape[1] != 6:
        raise Thought4RuntimeError(f"unexpected EEF-state shape: {eef.shape}")
    if len(actions) != len(eef):
        raise Thought4RuntimeError("action/EEF episode lengths differ")
    return DemonstrationEpisode(
        actions=actions,
        eef_states=eef,
        gripper_values=actions[:, -1:].copy(),
        episode_ids=episode_column,
    )


def _replay_demo_prefix(adapter: Any, episode: DemonstrationEpisode, frame: int) -> Any:
    import numpy as np

    if not 0 <= frame < len(episode.actions):
        raise Thought4RuntimeError("planned frame is outside demonstration")
    observation = None
    for action in episode.actions[:frame]:
        env_action = np.asarray(action, dtype=np.float64).copy()
        # LeRobot: gripper g in [0,1]; LIBERO OSC action: +1 close, -1 open.
        env_action[-1] = 1.0 - 2.0 * float(env_action[-1])
        observation, _reward, _done, _info = adapter.env.step(env_action)
    if observation is None:
        observation = adapter.env.regenerate_obs_from_state(
            adapter.env.get_sim_state()
        )
    return observation


def _observation_for_state(adapter: Any, state: Any) -> Mapping[str, Any]:
    # Avoid regenerate_obs_from_state(), which calls check_success internally.
    adapter.env.set_state(state)
    adapter.env.sim.forward()
    adapter.env._post_process()
    adapter.env._update_observables(force=True)
    observation = adapter.env.env._get_observations()
    if not isinstance(observation, Mapping):
        raise Thought4RuntimeError("LIBERO state regeneration returned no observation")
    return observation


def _demonstration_state_alignment(
    observation: Mapping[str, Any],
    episode: DemonstrationEpisode,
    frame_index: int,
) -> dict[str, Any]:
    """Verify that action-prefix recovery lands on demonstration time t."""

    import numpy as np

    from fastwam_ood_eval.thought4.geometry_labels import (
        quaternion_xyzw_to_matrix,
        rotation_geodesic_degrees,
    )

    expected_position = np.asarray(
        episode.eef_states[frame_index, :3], dtype=np.float64
    )
    observed_position = np.asarray(
        observation["robot0_eef_pos"], dtype=np.float64
    )
    expected_rotation = axis_angle_to_matrix(
        episode.eef_states[frame_index, 3:6]
    )
    observed_rotation = quaternion_xyzw_to_matrix(
        observation["robot0_eef_quat"]
    )
    translation_error = float(np.linalg.norm(observed_position - expected_position))
    rotation_error = rotation_geodesic_degrees(
        observed_rotation, expected_rotation
    )
    translation_limit = 0.03
    rotation_limit = 15.0
    passed = (
        translation_error <= translation_limit
        and rotation_error <= rotation_limit
    )
    if not passed:
        raise Thought4RuntimeError(
            "demonstration prefix/input-time alignment failed: "
            f"translation={translation_error:.6f}m, rotation={rotation_error:.3f}deg"
        )
    return {
        "applicable": True,
        "input_frame_index": int(frame_index),
        "translation_error_m": translation_error,
        "rotation_geodesic_error_degrees": rotation_error,
        "translation_limit_m": translation_limit,
        "rotation_limit_degrees": rotation_limit,
        "passed": True,
    }


def _inner_task_env(adapter: Any) -> Any:
    value = adapter.env
    while hasattr(value, "env") and not hasattr(value, "obj_body_id"):
        value = value.env
    if not hasattr(value, "sim"):
        raise Thought4RuntimeError("could not resolve LIBERO task environment")
    return value


def _geometry_state(
    adapter: Any,
    observation: Mapping[str, Any],
    *,
    target_object_name: str,
) -> dict[str, Any]:
    import numpy as np

    inner = _inner_task_env(adapter)
    object_ids = getattr(inner, "obj_body_id", {})
    if target_object_name not in object_ids:
        # Fixtures can use a root body name not exposed through obj_body_id.
        candidates = [
            name
            for name in getattr(inner.sim.model, "body_names", [])
            if target_object_name in str(name)
        ]
        if not candidates:
            raise Thought4RuntimeError(
                f"target object body unavailable: {target_object_name}"
            )
        body_id = inner.sim.model.body_name2id(candidates[0])
    else:
        body_id = object_ids[target_object_name]
    eef_pos = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
    eef_quat = np.asarray(observation["robot0_eef_quat"], dtype=np.float64)
    object_pos = np.asarray(inner.sim.data.body_xpos[body_id], dtype=np.float64)
    object_quat_wxyz = np.asarray(
        inner.sim.data.body_xquat[body_id], dtype=np.float64
    )
    object_quat_xyzw = object_quat_wxyz[[1, 2, 3, 0]]
    payload = {
        "eef_position_world": eef_pos.tolist(),
        "eef_quaternion_xyzw": eef_quat.tolist(),
        "target_object_name": target_object_name,
        "object_position_world": object_pos.tolist(),
        "object_quaternion_xyzw": object_quat_xyzw.tolist(),
    }
    # Reject silent non-finite conversion.
    json.dumps(payload, allow_nan=False)
    return payload


def _lighting_state(adapter: Any) -> dict[str, Any]:
    import numpy as np

    model = adapter.env.sim.model
    payload: dict[str, Any] = {}
    for name in (
        "light_pos",
        "light_dir",
        "light_diffuse",
        "light_specular",
        "light_ambient",
        "light_active",
    ):
        value = getattr(model, name, None)
        if value is not None:
            payload[name] = np.asarray(value).tolist()
    return payload


def _initial_object_layout(adapter: Any) -> tuple[tuple[str, ...], Any, str]:
    """Snapshot all LIBERO task objects/fixtures before any action replay."""

    import numpy as np

    from fastwam_ood_eval.thought4.schemas import sha256_canonical

    inner = _inner_task_env(adapter)
    object_ids = getattr(inner, "obj_body_id", None)
    if not isinstance(object_ids, Mapping) or not object_ids:
        raise Thought4RuntimeError("LIBERO object layout is unavailable")
    items = tuple(
        sorted(
            ((str(name), int(body_id)) for name, body_id in object_ids.items()),
            key=lambda item: item[0],
        )
    )
    names = tuple(name for name, _body_id in items)
    rows = []
    for _name, body_id in items:
        rows.append(
            np.concatenate(
                (
                    np.asarray(inner.sim.data.body_xpos[body_id], dtype=np.float64),
                    np.asarray(inner.sim.data.body_xquat[body_id], dtype=np.float64),
                )
            )
        )
    values = np.stack(rows)
    digest = sha256_canonical(
        {"names": names, "values_sha256": array_sha256(values)}
    )
    return names, values, digest


def _robot_state_snapshot(
    observation: Mapping[str, Any],
) -> tuple[tuple[str, ...], Any, str]:
    """Hash robot-only fields from an owned observation snapshot."""

    import numpy as np

    from fastwam_ood_eval.thought4.schemas import sha256_canonical

    preferred = (
        "robot0_joint_pos",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    )
    keys = tuple(key for key in preferred if key in observation)
    if not keys:
        raise Thought4RuntimeError("LIBERO observation has no robot state")
    arrays = [np.asarray(observation[key], dtype=np.float64).reshape(-1) for key in keys]
    if any(not np.isfinite(value).all() for value in arrays):
        raise Thought4RuntimeError("robot state contains NaN/Inf")
    values = np.concatenate(arrays)
    digest = sha256_canonical(
        {"keys": keys, "values_sha256": array_sha256(values)}
    )
    return keys, values, digest


def _robot_states_matching_clean(
    states: Mapping[str, tuple[tuple[str, ...], Any, str]],
) -> dict[str, bool]:
    """Compare robot observations without assigning perturbation semantics."""

    import numpy as np

    if "clean" not in states:
        raise Thought4RuntimeError("robot-state comparison requires Clean")
    clean_keys, clean_values, _clean_sha = states["clean"]
    matches: dict[str, bool] = {}
    for condition, (keys, values, _sha) in states.items():
        if keys != clean_keys:
            raise Thought4RuntimeError(
                f"{condition} robot-state fields differ from Clean"
            )
        matches[condition] = bool(
            np.allclose(values, clean_values, atol=1e-7, rtol=1e-7)
        )
    return matches


def _validate_input_robot_states(
    states: Mapping[str, tuple[tuple[str, ...], Any, str]],
) -> dict[str, bool]:
    """Validate perturbation semantics at the actual model input time t.

    LIBERO's shared demonstration state can overwrite the variant qpos during
    reset.  Robot-init therefore becomes observable only after replaying the
    common action prefix.  Camera and Lighting remain exact-state controls.
    """

    matches = _robot_states_matching_clean(states)
    for condition in ("clean", "camera", "lighting"):
        if condition in matches and not matches[condition]:
            raise Thought4RuntimeError(
                f"{condition} robot state differs from Clean at model input time"
            )
    if "robot_init" in matches and matches["robot_init"]:
        raise Thought4RuntimeError(
            "Robot-init variant did not change the robot state at model input time"
        )
    return matches


def _renderer_for_adapter(
    cfg: Thought4Config,
    observation_by_adapter_id: Mapping[int, Mapping[str, Any]],
) -> PairedStateRenderer:
    def render_function(
        env: Any,
        camera_name: str,
        height: int,
        width: int,
    ) -> tuple[Any, Any, Any, Mapping[str, Any]]:
        observation = observation_by_adapter_id[id(env)]
        rgb, depth = render_rgb_depth(
            env,
            camera_name=camera_name,
            height=height,
            width=width,
        )
        camera = camera_metadata(
            env,
            camera_name=camera_name,
            height=height,
            width=width,
        )
        geometry = _geometry_state(
            env,
            observation,
            target_object_name=cfg.cohort.target_object_name,
        )
        return rgb, depth, camera, geometry

    return PairedStateRenderer(
        camera_name=cfg.rendering.camera_name,
        height=cfg.rendering.image_height,
        width=cfg.rendering.image_width,
        render_function=render_function,
    )


def _labels_for_condition(
    rendered: RenderedCondition,
    episode: DemonstrationEpisode,
    *,
    frame_index: int,
    horizon: int,
    clean_camera_to_world: Any,
    action_label_override: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    from fastwam_ood_eval.thought4.geometry_labels import (
        low_resolution_depth,
        quaternion_xyzw_to_matrix,
    )

    camera_to_world = np.asarray(
        rendered.record.camera.extrinsic_camera_to_world, dtype=np.float64
    )
    geometry = rendered.geometry_state
    eef_rotation = quaternion_xyzw_to_matrix(
        geometry["eef_quaternion_xyzw"]
    )
    object_rotation = quaternion_xyzw_to_matrix(
        geometry["object_quaternion_xyzw"]
    )
    relation = eef_object_relation(
        eef_position_world=geometry["eef_position_world"],
        object_position_world=geometry["object_position_world"],
        camera_to_world=camera_to_world,
        eef_rotation_world=eef_rotation,
        object_rotation_world=object_rotation,
    )
    if action_label_override is None:
        rotations = np.stack(
            [axis_angle_to_matrix(value[3:6]) for value in episode.eef_states]
        )
        trajectory = build_future_trajectory_label(
            input_index=frame_index,
            episode_ids=episode.episode_ids,
            eef_positions_world=episode.eef_states[:, :3],
            eef_rotations_world=rotations,
            gripper_values=episode.gripper_values,
            camera_to_world=camera_to_world,
            horizon=horizon,
        )
        camera_rotation = np.linalg.inv(camera_to_world)[:3, :3]
        rotation_camera = np.stack(
            [
                rotation_to_6d(camera_rotation @ rotation)
                for rotation in rotations[
                    frame_index + 1 : frame_index + horizon + 1
                ]
            ]
        )
        if len(rotation_camera) < horizon:
            padded = np.zeros((horizon, 6), dtype=np.float32)
            padded[: len(rotation_camera)] = rotation_camera
            rotation_camera = padded
        translation_camera = trajectory.translation_camera
        gripper = trajectory.gripper
        valid_mask = trajectory.valid_mask
    else:
        required = {
            "translation_camera",
            "rotation_camera_6d",
            "gripper",
            "valid_mask",
        }
        if set(action_label_override) != required:
            raise Thought4RuntimeError(
                "robot-init action-label override has invalid fields"
            )
        translation_camera = np.asarray(
            action_label_override["translation_camera"], dtype=np.float32
        )
        rotation_camera = np.asarray(
            action_label_override["rotation_camera_6d"], dtype=np.float32
        )
        gripper = np.asarray(
            action_label_override["gripper"], dtype=np.float32
        )
        valid_mask = np.asarray(
            action_label_override["valid_mask"], dtype=bool
        )
        if (
            translation_camera.shape != (horizon, 3)
            or rotation_camera.shape != (horizon, 6)
            or gripper.shape != (horizon, 1)
            or valid_mask.shape != (horizon,)
        ):
            raise Thought4RuntimeError(
                "robot-init action-label override has invalid shapes"
            )
        for value in (translation_camera, rotation_camera, gripper):
            if not np.isfinite(value).all():
                raise Thought4RuntimeError(
                    "robot-init action-label override contains NaN/Inf"
                )
    se3 = np.concatenate(
        (
            translation_camera,
            rotation_camera.astype("float32"),
            gripper,
        ),
        axis=-1,
    )
    pose = relative_camera_pose(camera_to_world, clean_camera_to_world)
    labels = {
        "depth": low_resolution_depth(rendered.depth, (7, 7)),
        "relative_camera_translation": pose["translation"],
        "relative_camera_rotation_6d": pose["rotation_6d"],
        "eef_object_translation_camera": relation[
            "eef_to_object_camera"
        ],
        "eef_object_translation_world": relation[
            "eef_to_object_world"
        ],
        "eef_object_relative_orientation_camera_6d": relation[
            "relative_orientation_camera_6d"
        ],
        "eef_object_relative_orientation_world_6d": relation[
            "relative_orientation_world_6d"
        ],
        "action_translation_trajectory": translation_camera,
        "action_rotation_6d": rotation_camera.astype("float32"),
        "action_gripper": gripper,
        "action_se3_trajectory": se3.astype("float32"),
    }
    masks = {
        "action_translation_trajectory": valid_mask,
        "action_rotation_6d": valid_mask,
        "action_gripper": valid_mask,
        "action_se3_trajectory": valid_mask,
    }
    return labels, masks


def _robot_init_action_labels(
    adapter: Any,
    episode: DemonstrationEpisode,
    *,
    frame_index: int,
    horizon: int,
    camera_to_world: Any,
    current_observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay future demonstration actions from robot-init state, reading no RGB."""

    import numpy as np

    from fastwam_ood_eval.thought4.geometry_labels import (
        quaternion_xyzw_to_matrix,
    )

    state_at_t = np.asarray(adapter.env.get_sim_state()).copy()
    current_position = np.asarray(
        current_observation["robot0_eef_pos"], dtype=np.float64
    ).copy()
    world_to_camera = np.linalg.inv(
        np.asarray(camera_to_world, dtype=np.float64)
    )[:3, :3]
    translation_camera = np.zeros((horizon, 3), dtype=np.float32)
    rotation_camera = np.zeros((horizon, 6), dtype=np.float32)
    gripper = np.zeros((horizon, 1), dtype=np.float32)
    valid = np.zeros(horizon, dtype=bool)
    try:
        for offset, action_index in enumerate(
            range(frame_index, min(frame_index + horizon, len(episode.actions)))
        ):
            env_action = np.asarray(
                episode.actions[action_index], dtype=np.float64
            ).copy()
            env_action[-1] = 1.0 - 2.0 * float(env_action[-1])
            observation, _reward, _done, _info = adapter.env.step(env_action)
            future_position = np.asarray(
                observation["robot0_eef_pos"], dtype=np.float64
            )
            future_rotation = quaternion_xyzw_to_matrix(
                observation["robot0_eef_quat"]
            )
            translation_camera[offset] = (
                world_to_camera @ (future_position - current_position)
            )
            rotation_camera[offset] = rotation_to_6d(
                world_to_camera @ future_rotation
            )
            gripper[offset, 0] = float(
                episode.gripper_values[action_index, 0]
            )
            valid[offset] = True
    finally:
        _observation_for_state(adapter, state_at_t)
    if not valid.any():
        raise Thought4RuntimeError(
            "robot-init future action replay produced no valid labels"
        )
    return {
        "translation_camera": translation_camera,
        "rotation_camera_6d": rotation_camera,
        "gripper": gripper,
        "valid_mask": valid,
    }


def render_probe_samples(
    cfg: Thought4Config,
    plans: Sequence[PlannedBaseState],
) -> tuple[list[RenderedProbeSample], dict[str, Any]]:
    """Render diagnostic states; no policy model and no success outcome."""

    import numpy as np

    from fastwam_ood_eval.envs.libero_plus_adapter import LiberoPlusAdapter

    conditions = tuple(cfg.cohort.conditions)
    adapters = {
        condition: LiberoPlusAdapter(
            image_size=(
                cfg.rendering.image_height,
                cfg.rendering.image_width,
            ),
            root=Path("third_party/LIBERO-plus"),
            config_dir=cfg.experiment.output_dir / "runtime" / "libero_plus",
        )
        for condition in conditions
    }
    samples: list[RenderedProbeSample] = []
    states: dict[str, Any] = {}
    task_catalog = _condition_task_catalog(cfg)
    try:
        for plan in plans:
            episode = load_demonstration_episode(
                cfg.cohort.dataset_root, plan.episode_index
            )
            observations: dict[str, Mapping[str, Any]] = {}
            selected_tasks = {
                condition: _select_condition_task(
                    cfg,
                    plan,
                    condition,
                    catalog=task_catalog,
                )
                for condition in conditions
            }
            for condition, adapter in adapters.items():
                classification_id, task_name = selected_tasks[condition]
                observations[condition] = adapter.reset(
                    _make_job(
                        cfg,
                        condition=condition,
                        plan=plan,
                        classification_id=classification_id,
                        task_name=task_name,
                        clean_task_name=task_catalog["clean"][0][1],
                    )
                )
            initial_layouts = {
                condition: _initial_object_layout(adapter)
                for condition, adapter in adapters.items()
            }
            reset_robots = {
                condition: _robot_state_snapshot(observations[condition])
                for condition in adapters
            }
            clean_layout_names, clean_layout_values, _clean_layout_sha = (
                initial_layouts["clean"]
            )
            layout_matches_clean: dict[str, bool] = {}
            for condition, (names, values, _sha) in initial_layouts.items():
                matches = names == clean_layout_names and bool(
                    np.allclose(
                        values,
                        clean_layout_values,
                        atol=1e-7,
                        rtol=1e-7,
                    )
                )
                layout_matches_clean[condition] = matches
                if not matches:
                    raise Thought4RuntimeError(
                        f"{condition} initial object layout differs from Clean"
                    )
            # Disclosure only: the shared demonstration state may make the
            # reset observations identical even for a Robot-init variant.
            reset_robot_matches_clean = _robot_states_matching_clean(
                reset_robots
            )
            observations["clean"] = _replay_demo_prefix(
                adapters["clean"], episode, plan.frame_index
            )
            clean_alignment = _demonstration_state_alignment(
                observations["clean"], episode, plan.frame_index
            )
            if "robot_init" in adapters:
                observations["robot_init"] = _replay_demo_prefix(
                    adapters["robot_init"], episode, plan.frame_index
                )
            clean_state = adapters["clean"].env.get_sim_state()
            states[plan.identity.sample_id] = clean_state.copy()
            for condition in ("camera", "lighting"):
                observations[condition] = _observation_for_state(
                    adapters[condition], clean_state
                )
            observations = {
                condition: deepcopy(dict(observation))
                for condition, observation in observations.items()
            }
            input_robots = {
                condition: _robot_state_snapshot(observations[condition])
                for condition in adapters
            }
            input_robot_matches_clean = _validate_input_robot_states(
                input_robots
            )
            observation_by_id = {
                id(adapters[condition].env): observations[condition]
                for condition in adapters
            }
            renderer = _renderer_for_adapter(cfg, observation_by_id)
            identities = {
                condition: SampleIdentity(
                    task_id=plan.identity.task_id,
                    episode_id=plan.identity.episode_id,
                    frame_index=plan.identity.frame_index,
                    split=plan.identity.split,
                    timestamp=plan.identity.timestamp,
                    label_identity=plan.identity.label_identity,
                )
                for condition in ("clean", "camera", "lighting")
            }
            exact = renderer.render_exact_state_conditions(
                clean_env=adapters["clean"].env,
                camera_env=adapters["camera"].env,
                lighting_env=adapters["lighting"].env,
                identity_by_condition=identities,
                variants={
                    condition: selected_tasks[condition][1]
                    for condition in ("clean", "camera", "lighting")
                },
                lighting_configs={
                    condition: _lighting_state(adapters[condition])
                    for condition in ("clean", "camera", "lighting")
                },
            )
            rendered_by_condition = dict(exact)
            if "robot_init" in adapters:
                rendered_by_condition["robot_init"] = renderer.render_robot_init(
                    robot_init_env=adapters["robot_init"].env,
                    identity=plan.identity,
                    variant=selected_tasks["robot_init"][1],
                    clean_reference_sample_id=plan.identity.sample_id,
                    clean_reference_state_sha256=exact[
                        "clean"
                    ].record.simulator_state_sha256,
                )
            clean_camera = exact[
                "clean"
            ].record.camera.extrinsic_camera_to_world
            robot_action_override = None
            if "robot_init" in rendered_by_condition:
                robot_action_override = _robot_init_action_labels(
                    adapters["robot_init"],
                    episode,
                    frame_index=plan.frame_index,
                    horizon=cfg.probe.horizon,
                    camera_to_world=rendered_by_condition[
                        "robot_init"
                    ].record.camera.extrinsic_camera_to_world,
                    current_observation=observations["robot_init"],
                )
            for condition, rendered in rendered_by_condition.items():
                labels, masks = _labels_for_condition(
                    rendered,
                    episode,
                    frame_index=plan.frame_index,
                    horizon=cfg.probe.horizon,
                    clean_camera_to_world=clean_camera,
                    action_label_override=(
                        robot_action_override
                        if condition == "robot_init"
                        else None
                    ),
                )
                samples.append(
                    RenderedProbeSample(
                        plan=plan,
                        condition=condition,
                        rendered=rendered,
                        # Environments are closed before the 5B model loads;
                        # retain an owned time-t snapshot, never simulator views.
                        observation=deepcopy(dict(observations[condition])),
                        labels=labels,
                        masks=masks,
                        task_description=adapters[condition].task_description,
                        trajectory_label_source=(
                            "simulator_action_replay_from_robot_init_t"
                            if condition == "robot_init"
                            else "lerobot_demonstration_t_plus_1_to_t_plus_h"
                        ),
                        initial_object_layout_sha256=initial_layouts[condition][2],
                        initial_object_layout_matches_clean=(
                            layout_matches_clean[condition]
                        ),
                        reset_robot_state_sha256=reset_robots[condition][2],
                        reset_robot_state_matches_clean=(
                            reset_robot_matches_clean[condition]
                        ),
                        input_robot_state_sha256=input_robots[condition][2],
                        input_robot_state_matches_clean=(
                            input_robot_matches_clean[condition]
                        ),
                        demonstration_state_alignment=(
                            {
                                "applicable": False,
                                "reason": (
                                    "robot_init_changes_physical_state_and_uses_"
                                    "simulator_replay_labels"
                                ),
                                "input_frame_index": int(plan.frame_index),
                            }
                            if condition == "robot_init"
                            else dict(clean_alignment)
                        ),
                    )
                )
    finally:
        for adapter in adapters.values():
            adapter.close()
    return samples, states


def load_frozen_fastwam(cfg: Thought4Config) -> FrozenFastWAMRuntime:
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from fastwam.datasets.lerobot.utils.normalizer import (
        load_dataset_stats_from_json,
    )

    fastwam_root = Path("third_party/FastWAM").resolve()
    experiment_root = fastwam_root / "experiments" / "libero"
    for path in (experiment_root, fastwam_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from experiments.libero import eval_libero_single as official

    sim_config = fastwam_root / "configs" / "sim_libero.yaml"
    overrides = [
        "task=libero_uncond_2cam224_1e-4",
        f"ckpt={cfg.backbone.checkpoint_path.resolve()}",
        "mixed_precision=bf16",
        f"EVALUATION.dataset_stats_path={cfg.backbone.dataset_stats_path.resolve()}",
        f"EVALUATION.device={cfg.runtime.device}",
        "EVALUATION.replan_steps=10",
        f"EVALUATION.num_inference_steps={cfg.runtime.action_denoise_steps}",
        "EVALUATION.visualize_future_video=false",
    ]
    started = time.perf_counter()
    with initialize_config_dir(
        version_base="1.3", config_dir=str(sim_config.parent)
    ):
        upstream_cfg = compose(config_name=sim_config.stem, overrides=overrides)
    if bool(upstream_cfg.EVALUATION.get("visualize_future_video", False)):
        raise Thought4RuntimeError(
            "Thought4 forbids future-video generation during action inference"
        )
    model = instantiate(
        upstream_cfg.model,
        model_dtype=torch.bfloat16,
        device=cfg.runtime.device,
    )
    if type(model).__name__ != "FastWAM":
        raise Thought4RuntimeError(
            f"expected official FastWAM, got {type(model).__name__}"
        )
    model.load_checkpoint(str(cfg.backbone.checkpoint_path))
    model.requires_grad_(False)
    model.eval()
    if getattr(model.video_expert, "action_conditioned", None) is not False:
        raise Thought4RuntimeError("release Video DiT must be unconditional")
    if str(model.video_expert.video_attention_mask_mode) != "first_frame_causal":
        raise Thought4RuntimeError("expected first_frame_causal Video mask")
    processor = instantiate(upstream_cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(
        load_dataset_stats_from_json(str(cfg.backbone.dataset_stats_path))
    )
    video_size = upstream_cfg.data.train.get("video_size", [224, 448])
    action_horizon = int(
        upstream_cfg.EVALUATION.get("action_horizon")
        or (int(upstream_cfg.data.train.num_frames) - 1)
    )
    if action_horizon != cfg.probe.horizon:
        raise Thought4RuntimeError(
            f"official action horizon {action_horizon} differs from frozen "
            f"probe horizon {cfg.probe.horizon}"
        )
    return FrozenFastWAMRuntime(
        model=model,
        upstream_cfg=upstream_cfg,
        processor=processor,
        official=official,
        action_horizon=action_horizon,
        input_height=int(video_size[0]),
        input_width=int(video_size[1]),
        load_latency_s=time.perf_counter() - started,
    )


def extract_probe_examples(
    cfg: Thought4Config,
    runtime: FrozenFastWAMRuntime,
    samples: Sequence[RenderedProbeSample],
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Run one action inference per condition and capture sources A/B together."""

    import hashlib

    import torch

    from fastwam_ood_eval.thought4.feature_hooks import (
        ScopedFeatureCapture,
        ScopedVideoKVCacheCapture,
        action_hook_specs,
        video_kv_cache_specs,
        video_hook_specs,
    )
    from fastwam_ood_eval.thought4.pipeline import ProbeExample
    from fastwam_ood_eval.thought4.video_feature_extractor import (
        build_primary_camera_token_masks,
        pool_tokens,
        tensor_sha256,
    )

    video_specs = video_hook_specs(cfg.backbone.video_layers, include_kv=False)
    cache_specs = video_kv_cache_specs(
        cfg.backbone.video_layers,
        expected_calls=cfg.runtime.action_denoise_steps,
    )
    action_specs = action_hook_specs(cfg.backbone.action_hooks)
    specs = (*video_specs, *action_specs)
    spec_by_name = {spec.name: spec for spec in specs}
    examples: list[ProbeExample] = []
    inference_rows: list[dict[str, Any]] = []
    for sample in samples:
        geometry = sample.rendered.geometry_state
        camera = sample.rendered.record.camera
        masks = build_primary_camera_token_masks(
            depth=sample.rendered.depth,
            eef_position_world=geometry["eef_position_world"],
            object_position_world=geometry["object_position_world"],
            intrinsic=camera.intrinsic,
            camera_to_world=camera.extrinsic_camera_to_world,
            token_grid=(7, 14),
            primary_image_hw=(
                cfg.rendering.image_height,
                cfg.rendering.image_width,
            ),
        )
        action_seed = int.from_bytes(
            hashlib.sha256(
                f"thought4-action-feature-v1\0{cfg.experiment.seed}\0"
                f"{sample.plan.identity.sample_id}".encode()
            ).digest()[:4],
            "big",
        )
        # The base-state seed is identical across conditions, eliminating
        # diffusion-noise variation from paired condition comparisons.
        runtime.upstream_cfg.seed = action_seed
        started = time.perf_counter()
        with torch.inference_mode(), ScopedFeatureCapture(
            runtime.model,
            specs,
            clone=True,
            to_cpu=True,
        ) as capture, ScopedVideoKVCacheCapture(
            runtime.model.mot,
            cache_specs,
            to_cpu=True,
        ) as cache_capture:
            action, _images, future_frames = runtime.official._predict_action_chunk(
                obs=dict(sample.observation),
                task_description=sample.task_description,
                model=runtime.model,
                processor=runtime.processor,
                cfg=runtime.upstream_cfg,
                action_horizon=runtime.action_horizon,
                input_w=runtime.input_width,
                input_h=runtime.input_height,
                model_device=cfg.runtime.device,
            )
            if future_frames is not None:
                raise Thought4RuntimeError(
                    "action inference unexpectedly returned future RGB"
                )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if capture.captured.keys() != spec_by_name.keys():
            raise Thought4RuntimeError("captured hook identity mismatch")
        sample_id = sample.plan.identity.sample_id
        video_labels = {
            key: value
            for key, value in sample.labels.items()
            if key
            in {
                "depth",
                "relative_camera_translation",
                "relative_camera_rotation_6d",
                "eef_object_translation_camera",
                "eef_object_translation_world",
                "eef_object_relative_orientation_camera_6d",
                "eef_object_relative_orientation_world_6d",
            }
        }
        action_labels = {
            key: value
            for key, value in sample.labels.items()
            if key.startswith("action_")
        }
        action_geometry_labels = {
            key: value
            for key, value in video_labels.items()
            if key.startswith("eef_object_")
        }
        video_activations: list[tuple[str, int, Any]] = []
        for spec in video_specs:
            activations = capture.captured[spec.name]
            if len(activations) != 1:
                raise Thought4RuntimeError(
                    f"Video hook {spec.name} fired {len(activations)} times"
                )
            layer_index = int(spec.name.split("_l", 1)[1].split("_", 1)[0])
            video_activations.append((spec.module_path, layer_index, activations[0]))
        for cache_spec in cache_specs:
            activations = cache_capture.captured[cache_spec.name]
            if len(activations) != 1:
                raise Thought4RuntimeError(
                    f"Video cache {cache_spec.name} was not captured exactly once"
                )
            video_activations.append(
                (
                    cache_spec.module_path,
                    cache_spec.layer_index,
                    activations[0],
                )
            )
        for module_path, layer_index, tensor in video_activations:
            if (
                tensor.ndim != 3
                or tensor.shape[0] != 1
                or tensor.shape[1] != 98
                or tensor.shape[2] != 3072
            ):
                raise Thought4RuntimeError(
                    f"Video feature {module_path} expected [B,98,D], got "
                    f"{tuple(tensor.shape)}"
                )
            for pooling in cfg.probe.pooling:
                token_mask = None
                if pooling in masks:
                    token_mask = masks[pooling]
                pooled = pool_tokens(
                    tensor,
                    rule=pooling,
                    token_mask=token_mask,
                    has_cls_token=False,
                )[0].to(dtype=torch.float32)
                examples.append(
                    ProbeExample(
                        sample_id=sample_id,
                        episode_id=sample.plan.identity.episode_id,
                        split=sample.plan.identity.split,
                        condition=sample.condition,
                        source="A",
                        module_path=module_path,
                        layer_index=layer_index,
                        denoise_step_index=None,
                        pooling=pooling,
                        feature=pooled,
                        labels=video_labels,
                        masks={},
                    )
                )
        for spec in action_specs:
            activations = capture.captured[spec.name]
            if len(activations) != cfg.runtime.action_denoise_steps:
                raise Thought4RuntimeError(
                    f"Action hook {spec.name} expected "
                    f"{cfg.runtime.action_denoise_steps} calls, got "
                    f"{len(activations)}"
                )
            # Frozen v1 rule: the final denoising-step representation, then
            # temporal mean over the 32 action tokens.
            tensor = activations[-1]
            if (
                tensor.ndim != 3
                or tensor.shape[0] != 1
                or tensor.shape[1] != runtime.action_horizon
                or tensor.shape[2] != 1024
            ):
                raise Thought4RuntimeError(
                    f"Action feature {spec.name} has unexpected shape "
                    f"{tuple(tensor.shape)}"
                )
            pooled = tensor.mean(dim=1)[0].detach().to(dtype=torch.float32)
            layer_index = (
                15
                if "blocks.15" in spec.module_path
                else 29
                if "blocks.29" in spec.module_path
                else None
            )
            examples.append(
                ProbeExample(
                    sample_id=sample_id,
                    episode_id=sample.plan.identity.episode_id,
                    split=sample.plan.identity.split,
                    condition=sample.condition,
                    source="B",
                    module_path=spec.module_path,
                    layer_index=layer_index,
                    denoise_step_index=(
                        cfg.runtime.action_denoise_steps - 1
                    ),
                    pooling="action_token_mean_final_denoise_step",
                    feature=pooled,
                    labels={**action_geometry_labels, **action_labels},
                    masks={
                        key: value
                        for key, value in sample.masks.items()
                        if key.startswith("action_")
                    },
                )
            )
        action_tensor = torch.as_tensor(action)
        inference_rows.append(
            {
                "sample_id": sample_id,
                "condition": sample.condition,
                "action_seed": action_seed,
                "action_sha256": tensor_sha256(action_tensor),
                "action_shape": list(action_tensor.shape),
                "latency_ms": elapsed_ms,
                "video_hook_calls": {
                    spec.name: len(capture.captured[spec.name])
                    for spec in video_specs
                },
                "video_cache_consumer_calls": cache_capture.calls,
                "video_cache_entries": {
                    spec.name: {
                        "module_path": spec.module_path,
                        "feature_sha256": tensor_sha256(
                            cache_capture.captured[spec.name][0]
                        ),
                    }
                    for spec in cache_specs
                },
                "action_hook_calls": {
                    spec.name: len(capture.captured[spec.name])
                    for spec in action_specs
                },
                "future_rgb_read": False,
            }
        )
    return examples, inference_rows


def release_fastwam(runtime: FrozenFastWAMRuntime) -> None:
    try:
        import torch

        del runtime.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        gc.collect()
