"""Outcome-blind paired noise, future, and initial-state identities."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

from fastwam_ood_eval.thought6.schemas import Thought6Error, object_sha256


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("\0".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


@dataclass(frozen=True)
class OfflineNoiseIdentity:
    row_id: str
    suite: str
    task_id: int
    episode_id: str
    flow_slot: int
    action_noise_seed: int
    action_timestep_seed: int
    future_noise_seed: int


@dataclass(frozen=True)
class RolloutNoiseIdentity:
    pair_id: str
    stage: int
    suite: str
    task_id: int
    initial_state_index: int
    environment_seed: int
    episode_seed: int
    action_seed: int
    future_noise_seed: int


def offline_noise_identity(
    *, suite: str, task_id: int, episode_id: str, flow_slot: int, seed: int
) -> OfflineNoiseIdentity:
    if flow_slot < 0:
        raise Thought6Error("flow slot must be non-negative")
    row_id = hashlib.sha256(
        f"thought6-offline-row-v1\0{suite}\0{task_id}\0{episode_id}\0{flow_slot}".encode()
    ).hexdigest()
    return OfflineNoiseIdentity(
        row_id=row_id,
        suite=suite,
        task_id=int(task_id),
        episode_id=str(episode_id),
        flow_slot=int(flow_slot),
        action_noise_seed=stable_seed(
            "thought6-action-noise-v1", seed, suite, task_id, episode_id, flow_slot
        ),
        action_timestep_seed=stable_seed(
            "thought6-action-time-v1", seed, suite, task_id, episode_id, flow_slot
        ),
        future_noise_seed=stable_seed(
            "thought6-future-noise-v1", seed, suite, task_id, episode_id
        ),
    )


def rollout_noise_identity(
    *, stage: int, suite: str, task_id: int, initial_state_index: int, seed: int
) -> RolloutNoiseIdentity:
    if stage not in {1, 2}:
        raise Thought6Error("rollout stage must be 1 or 2")
    if initial_state_index < 0:
        raise Thought6Error("initial state index must be non-negative")
    pair_id = hashlib.sha256(
        f"thought6-rollout-pair-v1\0{stage}\0{suite}\0{task_id}\0{initial_state_index}".encode()
    ).hexdigest()
    return RolloutNoiseIdentity(
        pair_id=pair_id,
        stage=stage,
        suite=suite,
        task_id=int(task_id),
        initial_state_index=int(initial_state_index),
        environment_seed=stable_seed(
            "thought6-environment-v1", seed, stage, suite, task_id, initial_state_index
        ),
        episode_seed=stable_seed(
            "thought6-episode-v1", seed, stage, suite, task_id, initial_state_index
        ),
        action_seed=stable_seed(
            "thought6-action-v1", seed, stage, suite, task_id, initial_state_index
        ),
        future_noise_seed=stable_seed(
            "thought6-future-v1", seed, stage, suite, task_id, initial_state_index
        ),
    )


def build_noise_pairing_manifest(
    offline: Iterable[OfflineNoiseIdentity],
    rollout: Iterable[RolloutNoiseIdentity],
) -> dict[str, object]:
    offline_rows = [asdict(row) for row in offline]
    rollout_rows = [asdict(row) for row in rollout]
    payload: dict[str, object] = {
        "schema_version": "thought6.noise_pairing_manifest.v1",
        "outcome_fields_read": False,
        "correct_null_shuffle_share_action_noise": True,
        "correct_shuffle_share_future_noise": True,
        "clean_camera_share_rollout_pair_identity": True,
        "offline": offline_rows,
        "rollout": rollout_rows,
    }
    payload["manifest_sha256"] = object_sha256(payload)
    return payload


def validate_arm_pairing(rows: Iterable[Mapping[str, object]]) -> None:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["row_id"]), []).append(row)
    if not grouped:
        raise Thought6Error("paired rows are empty")
    for row_id, values in grouped.items():
        for key in (
            "action_noise_seed",
            "action_timestep_seed",
            "future_noise_seed",
            "initial_state_sha256",
            "scheduler_sha256",
        ):
            observed = {str(value[key]) for value in values}
            if len(observed) != 1:
                raise Thought6Error(f"paired {key} differs for {row_id}")


def validate_camera_only_pair(
    *,
    clean_physical_state_sha256: str,
    camera_physical_state_sha256: str,
    clean_camera_sha256: str,
    camera_camera_sha256: str,
) -> None:
    """Require exact physics and a changed camera for a Clean/Camera pair."""

    if clean_physical_state_sha256 != camera_physical_state_sha256:
        raise Thought6Error("Camera condition changed the physical state")
    if clean_camera_sha256 == camera_camera_sha256:
        raise Thought6Error("Camera condition did not change camera parameters")
