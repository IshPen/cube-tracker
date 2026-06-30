"""Tests for the YOLO label conversion (no torch/ultralytics required)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cube_tracker.models.detector_data import bbox_to_yolo, write_data_yaml, write_yolo_labels


def test_bbox_to_yolo_centre_and_size() -> None:
    bbox = {"x_min": 100.0, "y_min": 200.0, "x_max": 300.0, "y_max": 400.0}
    cx, cy, w, h = bbox_to_yolo(bbox, 800, 800)
    assert cx == pytest.approx(200.0 / 800)
    assert cy == pytest.approx(300.0 / 800)
    assert w == pytest.approx(200.0 / 800)
    assert h == pytest.approx(200.0 / 800)


def test_bbox_to_yolo_clamps_to_image() -> None:
    bbox = {"x_min": -50.0, "y_min": -10.0, "x_max": 900.0, "y_max": 810.0}
    cx, cy, w, h = bbox_to_yolo(bbox, 800, 800)
    assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
    assert w == pytest.approx(1.0) and h == pytest.approx(1.0)


def test_write_yolo_labels_creates_matching_txt(tmp_path: Path) -> None:
    split = tmp_path / "train"
    split.mkdir()
    labels = {
        "image": {"width": 768, "height": 768},
        "bbox": {"x_min": 192.0, "y_min": 192.0, "x_max": 576.0, "y_max": 576.0},
    }
    (split / "frame_000001.labels.json").write_text(json.dumps(labels), encoding="utf-8")

    assert write_yolo_labels(split) == 1
    line = (split / "frame_000001.txt").read_text(encoding="utf-8").strip()
    parts = line.split()
    assert parts[0] == "0"
    assert pytest.approx(float(parts[1])) == 0.5  # centred
    assert pytest.approx(float(parts[3])) == 0.5  # half width


def test_write_data_yaml(tmp_path: Path) -> None:
    out = tmp_path / "cube_detect.yaml"
    write_data_yaml(Path("/data/cubes"), out)
    text = out.read_text(encoding="utf-8")
    assert "names:" in text and "cube" in text and "nc: 1" in text
