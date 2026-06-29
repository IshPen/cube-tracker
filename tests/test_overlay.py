"""Test the debug overlay drawing (no Blender; OpenCV only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from cube_tracker.eval.reproject_overlay import draw_overlay


def _labels() -> dict[str, Any]:
    return {
        "image": {"width": 32, "height": 32},
        "bbox": {"x_min": 2, "y_min": 2, "x_max": 30, "y_max": 30},
        "landmarks": [
            {"name": "lm_00", "kind": "corner", "x": 5, "y": 5, "visible": True},
            {"name": "lm_01", "kind": "edge", "x": 10, "y": 10, "visible": False},
        ],
        "facelets": [
            {
                "face": "U",
                "row": 0,
                "col": 0,
                "color": "U",
                "coverage": 1.0,
                "center": [16, 16],
                "corners": [[12, 12], [20, 12], [20, 20], [12, 20]],
            }
        ],
    }


def test_draw_overlay_preserves_shape_and_draws() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    out = draw_overlay(image, _labels())
    assert out.shape == image.shape
    assert out.dtype == image.dtype
    assert out.any()  # something was drawn
