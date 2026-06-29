"""Draw a frame's labels back onto its render, for eyeball verification (M2).

This is the debug overlay: if the projected facelet grid sits on the cube, the landmark
dots sit on the corners/intersections (green = visible, red = occluded), and the coverage
dots shrink where the cube is covered, then the render-and-label stage is producing correct
labels. It runs in the project venv (OpenCV), reading only the PNG and its ``labels.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# OpenCV uses BGR. Rough cube-colour swatches for drawing the facelet grid.
_FACE_BGR: dict[str, tuple[int, int, int]] = {
    "U": (245, 245, 245),
    "D": (40, 220, 230),
    "F": (60, 170, 40),
    "B": (200, 70, 20),
    "L": (30, 140, 240),
    "R": (50, 50, 220),
}


def draw_overlay(image: np.ndarray, labels: dict[str, Any]) -> np.ndarray:
    """Return a copy of ``image`` annotated with the bounding box, grid, and landmarks."""
    canvas = image.copy()

    box = labels["bbox"]
    cv2.rectangle(
        canvas,
        (int(box["x_min"]), int(box["y_min"])),
        (int(box["x_max"]), int(box["y_max"])),
        (0, 255, 255),
        1,
    )

    # Only the camera-facing facelets are drawn; back faces project onto the front and would
    # just clutter the grid. A visible sticker has non-zero ray coverage.
    for facelet in labels["facelets"]:
        if facelet["coverage"] <= 0.0:
            continue
        color = _FACE_BGR[facelet["color"]]
        pts = np.array(facelet["corners"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=1)
        cx, cy = (int(facelet["center"][0]), int(facelet["center"][1]))
        # Dot radius encodes coverage: a fully visible sticker is big, a barely-seen one tiny.
        cv2.circle(canvas, (cx, cy), 2 + int(5 * facelet["coverage"]), color, -1)

    for landmark in labels["landmarks"]:
        point = (int(landmark["x"]), int(landmark["y"]))
        dot = (0, 220, 0) if landmark["visible"] else (0, 0, 255)
        cv2.circle(canvas, point, 3, dot, -1)

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay frame labels onto its render.")
    parser.add_argument("--labels", required=True, help="Path to the frame's labels.json.")
    parser.add_argument("--image", default=None, help="Render PNG; defaults to the labelled path.")
    parser.add_argument("--out", default=None, help="Output PNG; defaults to <stem>.overlay.png.")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))

    image_path = Path(args.image) if args.image else labels_path.with_name(labels["image"]["path"])
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read render image: {image_path}")

    out_path = (
        Path(args.out) if args.out else labels_path.with_suffix("").with_suffix(".overlay.png")
    )
    cv2.imwrite(str(out_path), draw_overlay(image, labels))
    print(f"[overlay] wrote {out_path}")


if __name__ == "__main__":
    main()
