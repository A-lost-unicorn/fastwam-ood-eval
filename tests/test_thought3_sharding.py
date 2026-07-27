from __future__ import annotations

from fastwam_ood_eval.thought3.sharding import (
    shard_indices,
    validate_complete_partition,
)


def test_three_rank_partition_has_no_duplicate_or_missing_sample():
    partitions = [shard_indices(101, rank, 3) for rank in range(3)]
    validate_complete_partition(partitions, expected_count=101)
    assert not set(partitions[0]) & set(partitions[1])
    assert not set(partitions[0]) & set(partitions[2])
    assert not set(partitions[1]) & set(partitions[2])
