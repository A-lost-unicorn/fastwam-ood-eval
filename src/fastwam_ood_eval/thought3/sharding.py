"""Deterministic, auditable rank and shard assignment."""

from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar("T")


def validate_rank_world_size(rank: int, world_size: int) -> None:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0,{world_size}), got {rank}")


def shard_by_index(items: Sequence[T], rank: int, world_size: int) -> list[T]:
    validate_rank_world_size(rank, world_size)
    return [item for index, item in enumerate(items) if index % world_size == rank]


def shard_indices(count: int, rank: int, world_size: int) -> tuple[int, ...]:
    if count < 0:
        raise ValueError("count must be non-negative")
    validate_rank_world_size(rank, world_size)
    return tuple(index for index in range(count) if index % world_size == rank)


def validate_complete_partition(
    partitions: Sequence[Sequence[int]],
    *,
    expected_count: int,
) -> None:
    flat = [int(index) for partition in partitions for index in partition]
    if len(flat) != len(set(flat)):
        raise ValueError("rank partitions contain duplicate indices")
    if set(flat) != set(range(expected_count)):
        missing = sorted(set(range(expected_count)) - set(flat))
        extra = sorted(set(flat) - set(range(expected_count)))
        raise ValueError(f"rank partition is incomplete: missing={missing}, extra={extra}")
