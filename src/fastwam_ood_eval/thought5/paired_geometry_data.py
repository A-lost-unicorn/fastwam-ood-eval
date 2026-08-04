"""Outcome-blind multi-task Phase 5 cohort and perturbation catalog."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from fastwam_ood_eval.thought4.cohort import (
    EpisodeInventory,
    load_lerobot_episode_inventory,
)
from fastwam_ood_eval.thought5.config import CohortConfig
from fastwam_ood_eval.thought5.schemas import CohortRow, object_sha256


class Phase5CohortError(RuntimeError):
    pass


UPSTREAM_GOAL_ORDER = (
    "open_the_middle_drawer_of_the_cabinet",
    "put_the_bowl_on_the_stove",
    "put_the_wine_bottle_on_top_of_the_cabinet",
    "open_the_top_drawer_and_put_the_bowl_inside",
    "put_the_bowl_on_top_of_the_cabinet",
    "push_the_plate_to_the_front_of_the_stove",
    "put_the_cream_cheese_in_the_bowl",
    "turn_on_the_stove",
    "put_the_bowl_on_the_plate",
    "put_the_wine_bottle_on_the_rack",
)

CATEGORY_BY_CONDITION = {
    "camera": "Camera Viewpoints",
    "lighting": "Light Conditions",
    "robot_init": "Robot Initial States",
}


def normalized_task_name(name: str) -> str:
    return "_".join(name.strip().lower().split())


@dataclass(frozen=True)
class ConditionVariant:
    condition: str
    classification_id: int
    upstream_task_id: int
    task_name: str
    category: str


def load_condition_catalog(
    classification_path: str | Path,
    task_names: Iterable[str],
) -> dict[str, dict[str, tuple[ConditionVariant, ...]]]:
    payload = json.loads(Path(classification_path).read_text(encoding="utf-8"))
    rows = payload.get("libero_goal")
    if not isinstance(rows, list):
        raise Phase5CohortError("LIBERO-Plus libero_goal classification is missing")
    result: dict[str, dict[str, tuple[ConditionVariant, ...]]] = {}
    for human_name in task_names:
        name = normalized_task_name(human_name)
        if name not in UPSTREAM_GOAL_ORDER:
            raise Phase5CohortError(f"task absent from upstream goal order: {name}")
        clean_id = UPSTREAM_GOAL_ORDER.index(name) + 1
        by_condition: dict[str, tuple[ConditionVariant, ...]] = {
            "clean": (
                ConditionVariant(
                    condition="clean",
                    classification_id=clean_id,
                    upstream_task_id=clean_id - 1,
                    task_name=name,
                    category="Original Task",
                ),
            )
        }
        for condition, category in CATEGORY_BY_CONDITION.items():
            candidates = tuple(
                ConditionVariant(
                    condition=condition,
                    classification_id=int(row["id"]),
                    upstream_task_id=int(row["id"]) - 1,
                    task_name=str(row["name"]),
                    category=str(row["category"]),
                )
                for row in rows
                if row.get("category") == category
                and str(row.get("name", "")).startswith(name + "_")
            )
            if not candidates:
                raise Phase5CohortError(
                    f"no {condition} variants found for task {human_name}"
                )
            by_condition[condition] = candidates
        result[human_name] = by_condition
    return result


def choose_variant(
    values: Sequence[ConditionVariant],
    *,
    namespace: str,
    task_index: int,
    episode_index: int,
    seed: int,
) -> ConditionVariant:
    digest = hashlib.sha256(
        f"{namespace}\0{task_index}\0{episode_index}\0{seed}".encode("utf-8")
    ).digest()
    return values[int.from_bytes(digest[:8], "big") % len(values)]


def historical_exclusions(cfg: CohortConfig) -> dict[str, Any]:
    """Resolve frozen Thought3-dev and Thought4-test episode identities."""

    inventory = load_lerobot_episode_inventory(cfg.dataset_root)
    thought3_payload = json.loads(
        cfg.thought3_split_manifest.read_text(encoding="utf-8")
    )
    thought3_dev_hashes = set(thought3_payload["development_episode_ids"])
    thought3_episodes: list[tuple[int, int]] = []
    for episode in inventory:
        identity = object_sha256(
            {
                "suite": cfg.suite,
                "task_id": f"task_{episode.task_index}",
                "demonstration_id": f"episode_{episode.episode_index:06d}",
                "episode_index": episode.episode_index,
            }
        )
        if identity in thought3_dev_hashes:
            thought3_episodes.append((episode.task_index, episode.episode_index))
    if len(thought3_episodes) != len(thought3_dev_hashes):
        raise Phase5CohortError("could not resolve every Thought3 development episode")
    thought4_payload = json.loads(
        cfg.thought4_formal_manifest.read_text(encoding="utf-8")
    )
    thought4_test = sorted(
        {
            (
                int(row["identity"]["task_id"]),
                int(str(row["identity"]["episode_id"]).split("_")[-1]),
            )
            for row in thought4_payload["rows"]
            if row["identity"]["split"] == "test"
        }
    )
    return {
        "thought3_development": sorted(thought3_episodes),
        "thought4_formal_test": thought4_test,
        "all": sorted(set(thought3_episodes) | set(thought4_test)),
    }


def _stable_order(
    values: Iterable[EpisodeInventory], *, namespace: str, seed: int
) -> list[EpisodeInventory]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"{namespace}\0{seed}\0{value.task_index}\0{value.episode_index}".encode()
        ).hexdigest(),
    )


def _stable_frames(
    episode: EpisodeInventory, *, count: int, horizon: int, seed: int
) -> list[int]:
    candidates = range(1, episode.length - horizon)
    ordered = sorted(
        candidates,
        key=lambda frame: hashlib.sha256(
            f"thought5-frame-v1\0{seed}\0{episode.episode_index}\0{frame}".encode()
        ).hexdigest(),
    )
    if len(ordered) < count:
        raise Phase5CohortError("episode is too short for requested frame count")
    return ordered[:count]


def plan_cohorts(cfg: CohortConfig) -> tuple[CohortRow, ...]:
    inventory = load_lerobot_episode_inventory(cfg.dataset_root)
    excluded = set(map(tuple, historical_exclusions(cfg)["all"]))
    split_tasks = {
        "train": cfg.train_tasks,
        "development": cfg.development_tasks,
        "formal": cfg.formal_tasks,
    }
    requested = {
        "train": cfg.train_episodes_per_task,
        "development": cfg.development_episodes_per_task,
        "formal": cfg.formal_episodes_per_task,
    }
    split_offsets = {"train": 0, "development": 1_000_000, "formal": 2_000_000}
    result: list[CohortRow] = []
    used_episodes: set[tuple[int, int]] = set()
    used_seeds: set[int] = set()
    for split in ("train", "development", "formal"):
        for task_index in split_tasks[split]:
            eligible = [
                row
                for row in inventory
                if row.task_index == task_index
                and row.length > cfg.horizon + 2
                and (row.task_index, row.episode_index) not in excluded
                and (row.task_index, row.episode_index) not in used_episodes
            ]
            ordered = _stable_order(
                eligible,
                namespace=f"{cfg.seed_namespace}:{split}",
                seed=cfg.split_seed,
            )
            selected = ordered[: requested[split]]
            if len(selected) != requested[split]:
                raise Phase5CohortError(
                    f"not enough episodes for {split} task {task_index}"
                )
            for episode in selected:
                episode_key = (task_index, episode.episode_index)
                if episode_key in used_episodes:
                    raise Phase5CohortError("episode leakage across Phase5 splits")
                used_episodes.add(episode_key)
                seed = (
                    cfg.split_seed
                    + split_offsets[split]
                    + task_index * 10_000
                    + episode.task_local_episode_index
                )
                if seed in used_seeds:
                    raise Phase5CohortError("seed leakage across Phase5 splits")
                used_seeds.add(seed)
                for frame in _stable_frames(
                    episode,
                    count=cfg.frames_per_episode,
                    horizon=cfg.horizon,
                    seed=cfg.split_seed,
                ):
                    result.append(
                        CohortRow(
                            split=split,
                            task_index=task_index,
                            task_name=episode.task_name,
                            episode_index=episode.episode_index,
                            task_local_episode_index=episode.task_local_episode_index,
                            seed=seed,
                            frame_index=frame,
                        )
                    )
    return tuple(sorted(result, key=lambda row: row.sample_id))


def cohort_manifest(cfg: CohortConfig) -> dict[str, Any]:
    rows = plan_cohorts(cfg)
    exclusions = historical_exclusions(cfg)
    catalog = load_condition_catalog(
        cfg.classification_path, sorted({row.task_name for row in rows})
    )
    payload: dict[str, Any] = {
        "schema_version": "thought5.phase5.cohort_manifest.v1",
        "selection_is_outcome_blind": True,
        "success_fields_read": False,
        "split_unit": "task_then_episode_then_seed",
        "task_split": {
            "train": list(cfg.train_tasks),
            "development": list(cfg.development_tasks),
            "formal": list(cfg.formal_tasks),
        },
        "historical_exclusions": exclusions,
        "rows": [asdict(row) | {"sample_id": row.sample_id} for row in rows],
        "condition_catalog_counts": {
            task: {condition: len(values) for condition, values in groups.items()}
            for task, groups in catalog.items()
        },
    }
    payload["manifest_sha256"] = object_sha256(payload)
    return payload


def assert_formal_exclusion(manifest: Mapping[str, Any]) -> None:
    excluded = {
        tuple(item)
        for values in manifest["historical_exclusions"].values()
        if isinstance(values, list)
        for item in values
        if isinstance(item, list) and len(item) == 2
    }
    formal = {
        (int(row["task_index"]), int(row["episode_index"]))
        for row in manifest["rows"]
        if row["split"] == "formal"
    }
    if formal & excluded:
        raise Phase5CohortError("formal cohort overlaps frozen historical exclusions")
