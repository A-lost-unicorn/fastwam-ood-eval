"""Outcome-blind episode planning and materialized cohort manifests."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from fastwam_ood_eval.thought4 import THOUGHT4_COHORT_SCHEMA
from fastwam_ood_eval.thought4.config import CohortConfig
from fastwam_ood_eval.thought4.schemas import (
    CohortRecord,
    SampleIdentity,
    Thought4SchemaError,
    sha256_canonical,
    validate_episode_split,
)


class CohortPlanningError(RuntimeError):
    """Raised when the frozen cohort cannot be built without leakage."""


@dataclass(frozen=True)
class EpisodeInventory:
    episode_index: int
    task_index: int
    task_name: str
    length: int
    task_local_episode_index: int

    @property
    def episode_id(self) -> str:
        return f"episode_{self.episode_index:06d}"


@dataclass(frozen=True)
class PlannedBaseState:
    task_id: str
    task_index: int
    episode_id: str
    episode_index: int
    task_local_episode_index: int
    frame_index: int
    split: str
    timestamp: float
    replay_action_count: int

    @property
    def label_identity(self) -> str:
        return (
            f"lerobot:{self.task_index}:{self.episode_index}:"
            f"t{self.frame_index}:future_h32"
        )

    @property
    def identity(self) -> SampleIdentity:
        return SampleIdentity(
            task_id=self.task_id,
            episode_id=self.episode_id,
            frame_index=self.frame_index,
            split=self.split,
            timestamp=self.timestamp,
            label_identity=self.label_identity,
        )

    @property
    def replay_locator(self) -> str:
        return (
            f"lerobot://episode/{self.episode_index:06d}"
            f"?frame={self.frame_index}&init={self.task_local_episode_index}"
        )


def load_lerobot_episode_inventory(dataset_root: str | Path) -> list[EpisodeInventory]:
    root = Path(dataset_root)
    episodes_path = root / "meta" / "episodes.jsonl"
    tasks_path = root / "meta" / "tasks.jsonl"
    if not episodes_path.is_file() or not tasks_path.is_file():
        raise CohortPlanningError("LeRobot episode/task metadata is missing")
    tasks: dict[str, int] = {}
    task_names: dict[int, str] = {}
    for raw in tasks_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        task_index = int(row["task_index"])
        task_name = str(row["task"])
        tasks[task_name] = task_index
        task_names[task_index] = task_name
    counters: dict[int, int] = {}
    result: list[EpisodeInventory] = []
    for raw in episodes_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        row_tasks = row.get("tasks")
        if not isinstance(row_tasks, list) or len(row_tasks) != 1:
            raise CohortPlanningError("each episode must identify exactly one task")
        task_name = str(row_tasks[0])
        if task_name not in tasks:
            raise CohortPlanningError(f"episode references unknown task: {task_name}")
        task_index = tasks[task_name]
        local = counters.get(task_index, 0)
        counters[task_index] = local + 1
        result.append(
            EpisodeInventory(
                episode_index=int(row["episode_index"]),
                task_index=task_index,
                task_name=task_name,
                length=int(row["length"]),
                task_local_episode_index=local,
            )
        )
    if not result:
        raise CohortPlanningError("LeRobot inventory is empty")
    return result


def _stable_order(values: Iterable[Any], *, namespace: str, seed: int) -> list[Any]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"{namespace}\0{seed}\0{value}".encode("utf-8")
        ).hexdigest(),
    )


def plan_base_states(
    cohort: CohortConfig,
    *,
    fps: int = 20,
    horizon: int = 32,
) -> tuple[PlannedBaseState, ...]:
    """Plan frames using only task/episode/frame identity and lengths."""

    if fps <= 0 or horizon <= 0:
        raise CohortPlanningError("fps/horizon must be positive")
    inventory = [
        episode
        for episode in load_lerobot_episode_inventory(cohort.dataset_root)
        if episode.task_index in set(cohort.task_ids)
        and episode.length > horizon + 2
    ]
    if not inventory:
        raise CohortPlanningError("no eligible episodes for frozen task IDs")
    requested = {
        "train": cohort.train_base_states,
        "development": cohort.development_base_states,
        "test": cohort.test_base_states,
    }
    episodes_needed = {
        split: math.ceil(count / cohort.frames_per_episode)
        for split, count in requested.items()
    }
    ordered = _stable_order(
        inventory,
        namespace="thought4-episode-plan-v1",
        seed=cohort.split_seed,
    )
    total_needed = sum(episodes_needed.values())
    if len(ordered) < total_needed:
        raise CohortPlanningError(
            f"need {total_needed} distinct episodes, found {len(ordered)}"
        )
    selected = ordered[:total_needed]
    cursor = 0
    result: list[PlannedBaseState] = []
    for split in ("train", "development", "test"):
        split_episodes = selected[cursor : cursor + episodes_needed[split]]
        cursor += episodes_needed[split]
        candidates: list[tuple[EpisodeInventory, int]] = []
        for episode in split_episodes:
            maximum = episode.length - horizon - 1
            possible = range(1, maximum + 1)
            ordered_frames = _stable_order(
                possible,
                namespace=(
                    f"thought4-frame-plan-v1:{episode.task_index}:"
                    f"{episode.episode_index}"
                ),
                seed=cohort.split_seed,
            )
            for frame in ordered_frames[: cohort.frames_per_episode]:
                candidates.append((episode, int(frame)))
        candidates = _stable_order(
            candidates,
            namespace=f"thought4-split-state-v1:{split}",
            seed=cohort.split_seed,
        )[: requested[split]]
        if len(candidates) != requested[split]:
            raise CohortPlanningError(
                f"could not plan requested {split} base-state count"
            )
        for episode, frame in candidates:
            result.append(
                PlannedBaseState(
                    task_id=str(episode.task_index),
                    task_index=episode.task_index,
                    episode_id=episode.episode_id,
                    episode_index=episode.episode_index,
                    task_local_episode_index=episode.task_local_episode_index,
                    frame_index=frame,
                    split=split,
                    timestamp=frame / fps,
                    replay_action_count=frame,
                )
            )
    validate_episode_split(value.identity for value in result)
    return tuple(sorted(result, key=lambda value: value.identity.sample_id))


def planned_cohort_manifest(
    values: Sequence[PlannedBaseState],
    *,
    config_fingerprint: str,
) -> dict[str, Any]:
    rows = [
        {
            **asdict(value),
            "sample_id": value.identity.sample_id,
            "label_identity": value.label_identity,
            "replay_locator": value.replay_locator,
        }
        for value in values
    ]
    payload: dict[str, Any] = {
        "schema_version": "thought4.phase4.planned_cohort.v1",
        "config_fingerprint": config_fingerprint,
        "selection_is_outcome_blind": True,
        "split_unit": "episode",
        "rows": rows,
    }
    payload["manifest_sha256"] = sha256_canonical(payload)
    return payload


def materialized_cohort_manifest(
    planned: Sequence[PlannedBaseState],
    state_by_sample_id: Mapping[str, Any],
    *,
    config_fingerprint: str,
) -> dict[str, Any]:
    from fastwam_ood_eval.thought4.paired_rendering import (
        simulator_state_sha256,
    )

    records: list[CohortRecord] = []
    for value in planned:
        sample_id = value.identity.sample_id
        if sample_id not in state_by_sample_id:
            raise CohortPlanningError(f"missing materialized state for {sample_id}")
        records.append(
            CohortRecord(
                identity=value.identity,
                simulator_state_sha256=simulator_state_sha256(
                    state_by_sample_id[sample_id]
                ),
                simulator_state_locator=value.replay_locator,
                episode_seed_namespace=(
                    f"thought4:{value.task_index}:{value.task_local_episode_index}"
                ),
            )
        )
    validate_episode_split(record.identity for record in records)
    rows = [record.to_dict() for record in records]
    payload: dict[str, Any] = {
        "schema_version": THOUGHT4_COHORT_SCHEMA,
        "config_fingerprint": config_fingerprint,
        "split_unit": "episode",
        "input_time": "t",
        "label_time": "t+1...t+H",
        "outcome_fields_read": False,
        "rows": rows,
    }
    payload["manifest_sha256"] = sha256_canonical(payload)
    return payload

