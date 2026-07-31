"""Exact-state paired rendering contracts for LIBERO/LIBERO-Plus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from fastwam_ood_eval.thought4.schemas import (
    CameraMetadata,
    PairedRenderRecord,
    SampleIdentity,
    Thought4SchemaError,
    canonical_json,
    sha256_bytes,
    sha256_canonical,
)


class PairedRenderingError(RuntimeError):
    """Raised when a claimed pair does not preserve its physical state."""


def array_sha256(value: Any) -> str:
    import numpy as np

    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise PairedRenderingError("array contains NaN/Inf")
    header = canonical_json(
        {"dtype": str(array.dtype), "shape": [int(v) for v in array.shape]}
    ).encode("utf-8")
    return sha256_bytes(header + b"\0" + array.tobytes(order="C"))


def simulator_state_sha256(state: Any) -> str:
    """Hash a flattened simulator state with dtype/shape identity."""

    return array_sha256(state)


def get_simulator_state(env: Any) -> Any:
    import numpy as np

    if hasattr(env, "get_sim_state"):
        state = env.get_sim_state()
    elif hasattr(env, "sim") and hasattr(env.sim, "get_state"):
        state = env.sim.get_state()
        if hasattr(state, "flatten"):
            state = state.flatten()
    else:
        raise PairedRenderingError("environment exposes no simulator-state getter")
    array = np.asarray(state).copy()
    if array.ndim != 1:
        array = array.reshape(-1)
    if not np.isfinite(array).all():
        raise PairedRenderingError("simulator state contains NaN/Inf")
    return array


def set_simulator_state(env: Any, state: Any) -> None:
    import numpy as np

    value = np.asarray(state).copy()
    if hasattr(env, "set_state"):
        env.set_state(value)
    elif hasattr(env, "sim") and hasattr(env.sim, "set_state_from_flattened"):
        env.sim.set_state_from_flattened(value)
    else:
        raise PairedRenderingError("environment exposes no simulator-state setter")
    if hasattr(env, "sim") and hasattr(env.sim, "forward"):
        env.sim.forward()


def camera_metadata(
    env: Any,
    *,
    camera_name: str,
    height: int,
    width: int,
) -> CameraMetadata:
    """Read robosuite camera matrices; extrinsic is camera-to-world."""

    try:
        from robosuite.utils.camera_utils import (
            get_camera_extrinsic_matrix,
            get_camera_intrinsic_matrix,
        )
    except ImportError as exc:
        raise PairedRenderingError("robosuite camera utilities are unavailable") from exc
    sim = getattr(env, "sim", None)
    if sim is None:
        raise PairedRenderingError("environment has no sim for camera metadata")
    intrinsic = get_camera_intrinsic_matrix(
        sim=sim,
        camera_name=camera_name,
        camera_height=height,
        camera_width=width,
    )
    extrinsic = get_camera_extrinsic_matrix(
        sim=sim,
        camera_name=camera_name,
    )
    return CameraMetadata.from_values(camera_name, intrinsic, extrinsic)


def render_rgb_depth(
    env: Any,
    *,
    camera_name: str,
    height: int,
    width: int,
) -> tuple[Any, Any]:
    """Render RGB and convert MuJoCo normalized depth to metric depth."""

    import numpy as np

    if not hasattr(env, "sim"):
        raise PairedRenderingError("environment has no simulator")
    result = env.sim.render(
        camera_name=camera_name,
        height=height,
        width=width,
        depth=True,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise PairedRenderingError("sim.render(depth=True) did not return RGB/depth")
    rgb, normalized_depth = result
    try:
        from robosuite.utils.camera_utils import get_real_depth_map
    except ImportError as exc:
        raise PairedRenderingError("robosuite depth conversion is unavailable") from exc
    depth = get_real_depth_map(env.sim, normalized_depth)
    rgb_array = np.asarray(rgb)
    depth_array = np.asarray(depth)
    if rgb_array.ndim != 3 or rgb_array.shape[-1] != 3:
        raise PairedRenderingError(f"invalid RGB shape: {rgb_array.shape}")
    if depth_array.shape[:2] != rgb_array.shape[:2]:
        raise PairedRenderingError("RGB/depth spatial shapes differ")
    if not np.isfinite(depth_array).all():
        raise PairedRenderingError("metric depth contains NaN/Inf")
    return rgb_array, depth_array


def _pose_payload(
    eef_position: Any,
    eef_quaternion: Any,
    object_positions: Mapping[str, Any],
    object_quaternions: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    payload = {
        "eef_position": np.asarray(eef_position, dtype=float).tolist(),
        "eef_quaternion": np.asarray(eef_quaternion, dtype=float).tolist(),
        "object_positions": {
            str(key): np.asarray(value, dtype=float).tolist()
            for key, value in sorted(object_positions.items())
        },
        "object_quaternions": {
            str(key): np.asarray(value, dtype=float).tolist()
            for key, value in sorted(object_quaternions.items())
        },
    }
    # canonical_json also rejects non-finite numbers.
    canonical_json(payload)
    return payload


@dataclass(frozen=True)
class RenderedCondition:
    record: PairedRenderRecord
    rgb: Any
    depth: Any
    simulator_state: Any
    geometry_state: Mapping[str, Any]


RenderFunction = Callable[
    [Any, str, int, int],
    tuple[Any, Any, CameraMetadata, Mapping[str, Any]],
]


def default_render_function(
    env: Any,
    camera_name: str,
    height: int,
    width: int,
) -> tuple[Any, Any, CameraMetadata, Mapping[str, Any]]:
    """Default renderer; the caller supplies task-specific geometry extraction."""

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
    geometry = getattr(env, "_thought4_geometry_state", None)
    if not isinstance(geometry, Mapping):
        raise PairedRenderingError(
            "task-specific EEF/object geometry extractor was not installed"
        )
    return rgb, depth, camera, geometry


class PairedStateRenderer:
    """Render Clean/Camera/Lighting from one state and Robot-init separately."""

    def __init__(
        self,
        *,
        camera_name: str,
        height: int,
        width: int,
        render_function: RenderFunction = default_render_function,
    ) -> None:
        self.camera_name = camera_name
        self.height = int(height)
        self.width = int(width)
        self.render_function = render_function
        if self.height <= 0 or self.width <= 0:
            raise PairedRenderingError("render size must be positive")

    def _render(
        self,
        env: Any,
        *,
        identity: SampleIdentity,
        condition: str,
        variant: str,
        clean_sample_id: str,
        clean_state_sha: str,
        exact_state_pair: bool,
        lighting_config: Mapping[str, Any],
    ) -> RenderedCondition:
        state = get_simulator_state(env)
        state_sha = simulator_state_sha256(state)
        rgb, depth, camera, geometry = self.render_function(
            env, self.camera_name, self.height, self.width
        )
        geometry_sha = sha256_canonical(dict(geometry))
        record = PairedRenderRecord(
            identity=identity,
            condition=condition,
            condition_variant=variant,
            exact_state_pair=exact_state_pair,
            clean_reference_sample_id=clean_sample_id,
            clean_reference_state_sha256=clean_state_sha,
            simulator_state_sha256=state_sha,
            object_eef_state_sha256=geometry_sha,
            rgb_sha256=array_sha256(rgb),
            depth_sha256=array_sha256(depth),
            camera=camera,
            lighting_config_sha256=sha256_canonical(dict(lighting_config)),
        )
        return RenderedCondition(record, rgb, depth, state, geometry)

    def render_exact_state_conditions(
        self,
        *,
        clean_env: Any,
        camera_env: Any,
        lighting_env: Any,
        identity_by_condition: Mapping[str, SampleIdentity],
        variants: Mapping[str, str],
        lighting_configs: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, RenderedCondition]:
        clean_state = get_simulator_state(clean_env)
        clean_sha = simulator_state_sha256(clean_state)
        clean_identity = identity_by_condition["clean"]
        clean_sample_id = clean_identity.sample_id
        result: dict[str, RenderedCondition] = {}
        for condition, env in (
            ("clean", clean_env),
            ("camera", camera_env),
            ("lighting", lighting_env),
        ):
            set_simulator_state(env, clean_state)
            observed = get_simulator_state(env)
            if simulator_state_sha256(observed) != clean_sha:
                raise PairedRenderingError(
                    f"{condition} failed exact-state restoration"
                )
            rendered = self._render(
                env,
                identity=identity_by_condition[condition],
                condition=condition,
                variant=variants[condition],
                clean_sample_id=clean_sample_id,
                clean_state_sha=clean_sha,
                exact_state_pair=True,
                lighting_config=lighting_configs.get(condition, {}),
            )
            if rendered.record.simulator_state_sha256 != clean_sha:
                raise PairedRenderingError(
                    f"{condition} mutated simulator state while rendering"
                )
            result[condition] = rendered
        return result

    def render_robot_init(
        self,
        *,
        robot_init_env: Any,
        identity: SampleIdentity,
        variant: str,
        clean_reference_sample_id: str,
        clean_reference_state_sha256: str,
    ) -> RenderedCondition:
        return self._render(
            robot_init_env,
            identity=identity,
            condition="robot_init",
            variant=variant,
            clean_sample_id=clean_reference_sample_id,
            clean_state_sha=clean_reference_state_sha256,
            exact_state_pair=False,
            lighting_config={},
        )


def validate_exact_state_group(records: Sequence[PairedRenderRecord]) -> None:
    by_condition = {record.condition: record for record in records}
    if set(by_condition) != {"clean", "camera", "lighting"}:
        raise Thought4SchemaError(
            "exact-state group must contain clean/camera/lighting exactly once"
        )
    hashes = {record.simulator_state_sha256 for record in records}
    if len(hashes) != 1 or not all(record.exact_state_pair for record in records):
        raise Thought4SchemaError("exact-state group has mismatched state hashes")
    geometry_hashes = {record.object_eef_state_sha256 for record in records}
    if len(geometry_hashes) != 1:
        raise Thought4SchemaError(
            "exact-state group has mismatched EEF/object state hashes"
        )
