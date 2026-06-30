"""Fine-tune and evaluate the YOLO cube detector (M4).

Trains ``yolo11n`` from pretrained COCO weights on the synthetic dataset's YOLO labels, on
the GPU, with Ultralytics' built-in augmentation plus our webcam-degradation pipeline
(injected by monkeypatching Ultralytics' Albumentations transform, since it has no public
hook for custom pixel transforms). Reports val mAP/precision/recall at the end.

    python -m cube_tracker.models.detector --data-yaml data/cube_detect.yaml \
        --epochs 30 --imgsz 640 --batch 16 --device 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def install_webcam_augmentation() -> None:
    """Make Ultralytics apply our webcam-degradation pipeline as its Albumentations step.

    Ultralytics builds its ``Albumentations`` transform during dataloader setup; replacing the
    class ``__init__`` before training swaps in our image-only pipeline. ``contains_spatial`` is
    False so Ultralytics applies it to the image alone and leaves the boxes untouched.
    """
    from ultralytics.data import augment as ul_augment

    from cube_tracker.models.augment import webcam_degradation

    transform = webcam_degradation()

    def _patched_init(self: Any, p: float = 1.0, transforms: Any = None) -> None:
        # Ultralytics calls Albumentations(p=..., transforms=...); we ignore its default
        # transforms and substitute our image-only webcam-degradation pipeline.
        self.p = p
        self.contains_spatial = False
        self.transform = transform

    ul_augment.Albumentations.__init__ = _patched_init


def train(
    data_yaml: Path,
    model_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    project: Path,
    name: str,
    patience: int,
    degrade: bool,
    fraction: float,
    workers: int,
) -> Any:
    """Fine-tune the detector and run a final validation pass; return the val metrics."""
    if degrade:
        install_webcam_augmentation()

    from ultralytics import YOLO

    model = YOLO(model_name)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project),
        name=name,
        patience=patience,
        fraction=fraction,
        workers=workers,
        seed=0,
        plots=True,
    )
    metrics = model.val()
    print(
        f"[detector] val  mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}  "
        f"precision={metrics.box.mp:.4f}  recall={metrics.box.mr:.4f}"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the YOLO cube detector.")
    parser.add_argument("--data-yaml", default="data/cube_detect.yaml")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="models_out/detector")
    parser.add_argument("--name", default="yolo11n_cube")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of train set to use.")
    # 4 avoids a Windows DataLoader deadlock seen at Ultralytics' default of 8; 0 = in-process.
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--no-degrade", dest="degrade", action="store_false")
    args = parser.parse_args()

    train(
        Path(args.data_yaml),
        args.model,
        args.epochs,
        args.imgsz,
        args.batch,
        args.device,
        Path(
            args.project
        ).resolve(),  # absolute so Ultralytics writes under the project, not C:\runs
        args.name,
        args.patience,
        args.degrade,
        args.fraction,
        args.workers,
    )


if __name__ == "__main__":
    main()
