"""Fine-tune the detector on synthetic + reviewed real frames (M4 hardening).

Mixes a sample of the synthetic training set with the (oversampled) real frames into one train
list and continues training from ``best.pt`` at a low learning rate. The real frames adapt the
detector to real webcam footage -- fixing face/poster false positives and missed real cubes --
while the synthetic anchor and a low LR prevent forgetting or overfitting to the small real set.

    python -m cube_tracker.models.finetune_detector \
        --weights models_out/detector/yolo11n_cube_v2/weights/best.pt \
        --synthetic P:/Development/cube-tracker-dataset/train \
        --val P:/Development/cube-tracker-dataset/val --real-dir data/real_frames
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import yaml


def build_mix(
    synthetic_dir: Path,
    real_dir: Path,
    val_dir: Path,
    synthetic_count: int,
    real_repeat: int,
    out_yaml: Path,
) -> tuple[int, int]:
    """Write a train image list (synthetic sample + oversampled real) and a data.yaml."""
    rng = random.Random(0)
    synthetic = sorted(synthetic_dir.glob("*.png"))
    synthetic_sample = rng.sample(synthetic, min(synthetic_count, len(synthetic)))
    real = sorted(p for p in real_dir.glob("*.jpg"))  # top-level only -> skips _discarded/

    lines = [str(p.resolve()) for p in synthetic_sample]
    lines += [str(p.resolve()) for p in real for _ in range(real_repeat)]
    rng.shuffle(lines)

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    train_txt = out_yaml.parent / "finetune_train.txt"
    train_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_yaml.write_text(
        yaml.safe_dump(
            {
                "train": train_txt.resolve().as_posix(),
                "val": val_dir.resolve().as_posix(),
                "nc": 1,
                "names": ["cube"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return len(synthetic_sample), len(real)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the detector on synthetic + real.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--synthetic", required=True, help="Synthetic train image dir.")
    parser.add_argument("--val", required=True, help="Synthetic val image dir.")
    parser.add_argument("--real-dir", default="data/real_frames")
    parser.add_argument("--out-yaml", default="data/finetune.yaml")
    parser.add_argument("--synthetic-count", type=int, default=6000)
    parser.add_argument("--real-repeat", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="finetune")
    args = parser.parse_args()

    n_syn, n_real = build_mix(
        Path(args.synthetic),
        Path(args.real_dir),
        Path(args.val),
        args.synthetic_count,
        args.real_repeat,
        Path(args.out_yaml),
    )
    print(f"[finetune] mix: {n_syn} synthetic + {n_real} real x{args.real_repeat}")

    from cube_tracker.models.detector import install_webcam_augmentation

    install_webcam_augmentation()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.train(
        data=str(Path(args.out_yaml).resolve()),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        optimizer="AdamW",
        lr0=args.lr0,
        project=str(Path("models_out/detector").resolve()),
        name=args.name,
        patience=8,
        seed=0,
        plots=True,
    )
    metrics: Any = model.val()
    print(f"[finetune] val mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
