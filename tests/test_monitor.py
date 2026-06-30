"""Test the dataset progress monitor's frame counting (no Blender required)."""

from __future__ import annotations

from pathlib import Path

from cube_tracker.render.monitor_dataset import _count


def test_count_sums_train_and_val_pngs(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    for name in ("a.png", "b.png"):
        (tmp_path / "train" / name).write_bytes(b"x")
    (tmp_path / "val" / "c.png").write_bytes(b"x")
    (tmp_path / "train" / "notes.txt").write_bytes(b"x")  # ignored: not a png

    assert _count(tmp_path) == 3
