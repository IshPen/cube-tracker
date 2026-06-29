"""Deterministic dataset planning: train/val split and shard assignment (M3).

Pure Python (no Blender), so the split is reproducible and unit-testable. Both the
under-Blender shard runner and the venv launcher import this so they always agree on which
frame goes where. The split uses a stable hash of the frame index (not Python's salted
``hash``) so the same ``count`` and ``val_fraction`` give the same split on every machine.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class FramePlan:
    """One frame's planned destination."""

    index: int
    split: str  # "train" or "val"


def _stable_key(index: int) -> int:
    digest = hashlib.sha256(str(index).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def plan_frames(count: int, val_fraction: float) -> list[FramePlan]:
    """Assign each of ``count`` frames to train or val, holding out ``val_fraction`` for val."""
    if count < 0:
        raise ValueError("count must be non-negative.")
    if not 0.0 <= val_fraction <= 1.0:
        raise ValueError("val_fraction must lie in [0, 1].")
    val_count = round(count * val_fraction)
    held_out = set(sorted(range(count), key=_stable_key)[:val_count])
    return [FramePlan(index, "val" if index in held_out else "train") for index in range(count)]


def shard_indices(count: int, shard_index: int, num_shards: int) -> list[int]:
    """Return the frame indices this shard is responsible for (round-robin partition)."""
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1.")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards).")
    return [index for index in range(count) if index % num_shards == shard_index]
