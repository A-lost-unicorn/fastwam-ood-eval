"""Episode-level split logic and leak-resistant training sample validation."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Iterable, Mapping, Sequence

from fastwam_ood_eval.thought3.safety import validate_training_batch_keys
from fastwam_ood_eval.thought3.schemas import (
    EpisodeDescriptor,
    EpisodeSplitManifest,
    Thought3SchemaError,
    sha256_canonical,
)


def _stable_episode_order(
    episode: EpisodeDescriptor,
    seed: int,
) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"thought3-split-v1\0{seed}\0{episode.episode_id}".encode("utf-8")
    ).hexdigest()
    return digest, episode.episode_id


def build_episode_split(
    episodes: Sequence[EpisodeDescriptor],
    *,
    dataset_revision: str,
    seed: int = 3407,
    development_fraction: float = 0.1,
) -> EpisodeSplitManifest:
    """Split whole episodes within each suite×task stratum.

    A stratum with at least two episodes always contributes at least one
    episode to both train and development.  A one-episode stratum is rejected
    rather than silently leaking or dropping a task.
    """

    if seed < 0:
        raise Thought3SchemaError("split seed must be non-negative")
    if not 0 < development_fraction < 1:
        raise Thought3SchemaError("development_fraction must be between 0 and 1")
    if not episodes:
        raise Thought3SchemaError("cannot split an empty episode inventory")
    by_id: dict[str, EpisodeDescriptor] = {}
    strata: dict[tuple[str, str], list[EpisodeDescriptor]] = defaultdict(list)
    for episode in episodes:
        if episode.episode_id in by_id:
            raise Thought3SchemaError(f"duplicate episode identity: {episode.episode_id}")
        by_id[episode.episode_id] = episode
        strata[(episode.suite, episode.task_id)].append(episode)

    train: list[str] = []
    development: list[str] = []
    stratum_counts: dict[str, dict[str, int]] = {}
    for (suite, task_id), values in sorted(strata.items()):
        if len(values) < 2:
            raise Thought3SchemaError(
                f"suite={suite} task={task_id} has only {len(values)} episode; "
                "episode-level train/development split requires at least two"
            )
        ordered = sorted(values, key=lambda item: _stable_episode_order(item, seed))
        dev_count = max(1, int(math.ceil(len(ordered) * development_fraction)))
        dev_count = min(dev_count, len(ordered) - 1)
        dev_values = ordered[:dev_count]
        train_values = ordered[dev_count:]
        development.extend(item.episode_id for item in dev_values)
        train.extend(item.episode_id for item in train_values)
        key = f"{suite}/{task_id}"
        stratum_counts[key] = {
            "total": len(ordered),
            "train": len(train_values),
            "development": len(dev_values),
        }

    return EpisodeSplitManifest(
        dataset_revision=dataset_revision,
        seed=seed,
        development_fraction=development_fraction,
        train_episode_ids=tuple(sorted(train)),
        development_episode_ids=tuple(sorted(development)),
        strata=stratum_counts,
    )


def split_for_episode(
    episode_id: str,
    manifest: EpisodeSplitManifest,
) -> str:
    if episode_id in set(manifest.train_episode_ids):
        return "train"
    if episode_id in set(manifest.development_episode_ids):
        return "development"
    raise Thought3SchemaError(f"episode is absent from split manifest: {episode_id}")


def validate_training_example(
    example: Mapping[str, Any],
    *,
    require_future: bool = True,
) -> None:
    validate_training_batch_keys(example, require_future=require_future)
    metadata = example.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise Thought3SchemaError("training metadata must be a mapping")
        forbidden_metadata = {
            "success",
            "termination_reason",
            "actual_future",
            "future_observation",
            "next_observation",
        }
        overlap = sorted(forbidden_metadata & {str(key) for key in metadata})
        if overlap:
            raise Thought3SchemaError(
                f"training metadata contains forbidden outcome/future fields: {overlap}"
            )


def episode_inventory_fingerprint(episodes: Iterable[EpisodeDescriptor]) -> str:
    rows = sorted((asdict(episode) for episode in episodes), key=sha256_canonical)
    return sha256_canonical(rows)
