"""Tests for the deterministic dataset split and sharding (no Blender required)."""

from __future__ import annotations

from collections import Counter

import pytest

from cube_tracker.render import dataset_plan as dp


def test_split_holds_out_the_right_fraction() -> None:
    plan = dp.plan_frames(100, 0.2)
    counts = Counter(fp.split for fp in plan)
    assert counts["val"] == 20
    assert counts["train"] == 80


def test_split_is_deterministic() -> None:
    first = dp.plan_frames(50, 0.3)
    second = dp.plan_frames(50, 0.3)
    assert first == second


def test_shards_partition_every_index_exactly_once() -> None:
    count, num_shards = 37, 4
    seen: list[int] = []
    for shard in range(num_shards):
        seen.extend(dp.shard_indices(count, shard, num_shards))
    assert sorted(seen) == list(range(count))  # complete and disjoint


def test_invalid_arguments_rejected() -> None:
    with pytest.raises(ValueError):
        dp.plan_frames(10, 1.5)
    with pytest.raises(ValueError):
        dp.shard_indices(10, 2, 2)
