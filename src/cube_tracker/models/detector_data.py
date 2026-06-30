"""Convert the cube dataset labels to YOLO detection format and write a data.yaml (M4).

Each frame's ``labels.json`` carries a pixel-space bounding box; YOLO wants one ``.txt`` per
image with a normalised ``class cx cy w h`` line. The cube is the single class (0). The label
``.txt`` is written *next to* its image -- Ultralytics looks for labels beside images when the
path has no ``images/`` segment -- so the 50k-frame dataset is never duplicated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

CUBE_CLASS = 0


def bbox_to_yolo(
    bbox: dict[str, float], width: int, height: int
) -> tuple[float, float, float, float]:
    """Convert a pixel ``{x_min,y_min,x_max,y_max}`` box to normalised ``(cx, cy, w, h)``."""
    x_min = max(0.0, bbox["x_min"])
    y_min = max(0.0, bbox["y_min"])
    x_max = min(float(width), bbox["x_max"])
    y_max = min(float(height), bbox["y_max"])
    cx = (x_min + x_max) / 2.0 / width
    cy = (y_min + y_max) / 2.0 / height
    return cx, cy, (x_max - x_min) / width, (y_max - y_min) / height


def write_yolo_labels(split_dir: Path) -> int:
    """Write a YOLO ``.txt`` next to every ``*.labels.json`` in ``split_dir``; return the count."""
    count = 0
    for labels_path in sorted(split_dir.glob("*.labels.json")):
        data: dict[str, Any] = json.loads(labels_path.read_text(encoding="utf-8"))
        cx, cy, w, h = bbox_to_yolo(data["bbox"], data["image"]["width"], data["image"]["height"])
        txt_path = labels_path.with_name(labels_path.name.replace(".labels.json", ".txt"))
        txt_path.write_text(f"{CUBE_CLASS} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8")
        count += 1
    return count


def write_data_yaml(dataset_dir: Path, out_path: Path) -> None:
    """Write an Ultralytics ``data.yaml`` pointing at the dataset's train/ and val/ image dirs."""
    config = {
        "path": dataset_dir.as_posix(),
        "train": "train",
        "val": "val",
        "nc": 1,
        "names": ["cube"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert cube labels to YOLO format.")
    parser.add_argument("--dataset-dir", required=True, help="Folder with train/ and val/.")
    parser.add_argument("--data-yaml", default="data/cube_detect.yaml")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    n_train = write_yolo_labels(dataset_dir / "train")
    n_val = write_yolo_labels(dataset_dir / "val")
    write_data_yaml(dataset_dir, Path(args.data_yaml))
    print(f"[detector_data] wrote {n_train} train + {n_val} val YOLO labels")
    print(f"[detector_data] data.yaml -> {args.data_yaml}")


if __name__ == "__main__":
    main()
