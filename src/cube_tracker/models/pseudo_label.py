"""Pseudo-label the cube in real frames with Grounding DINO, in YOLO format (M4 hardening).

Grounding DINO is a text-prompted detector; we ask it for "a rubik's cube" in each captured
real frame and write a YOLO ``.txt`` next to the image -- the highest-scoring box, or an empty
file for a hard-negative frame where no cube is found (faces, posters, clutter). Mixing these
real labels into detector training closes the sim-to-real gap. Runs in the venv on the GPU
(needs the ``label`` extra: transformers). These tools never ship in the deployed pipeline.

    python -m cube_tracker.models.pseudo_label --frames-dir data/real_frames
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

_MODEL = "IDEA-Research/grounding-dino-tiny"
_IMG_EXT = {".jpg", ".jpeg", ".png"}


def pseudo_label(
    frames_dir: Path,
    prompt: str,
    threshold: float,
    text_threshold: float,
    model_id: str,
    device: str,
) -> tuple[int, int]:
    """Write a YOLO ``.txt`` per frame (best cube box, or empty); return (labeled, negatives)."""
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()

    frames = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in _IMG_EXT)
    labeled = 0
    negatives = 0
    for path in frames:
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs["input_ids"],
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]

        txt_path = path.with_suffix(".txt")
        boxes = result["boxes"]
        if len(boxes) == 0:
            txt_path.write_text("", encoding="utf-8")
            negatives += 1
            continue
        best = int(torch.argmax(result["scores"]))
        x1, y1, x2, y2 = (float(v) for v in boxes[best])
        cx = (x1 + x2) / 2.0 / image.width
        cy = (y1 + y2) / 2.0 / image.height
        bw = (x2 - x1) / image.width
        bh = (y2 - y1) / image.height
        txt_path.write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")
        labeled += 1

    print(f"[pseudo_label] {labeled} cube-labeled, {negatives} negatives -> {frames_dir}")
    return labeled, negatives


def main() -> None:
    parser = argparse.ArgumentParser(description="Pseudo-label cubes with Grounding DINO.")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--prompt", default="a rubik's cube.")
    parser.add_argument("--threshold", type=float, default=0.35, help="Box confidence threshold.")
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--model", default=_MODEL)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pseudo_label(
        Path(args.frames_dir), args.prompt, args.threshold, args.text_threshold, args.model, device
    )


if __name__ == "__main__":
    main()
