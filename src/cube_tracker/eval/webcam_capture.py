"""Capture real webcam frames to fine-tune the detector on real footage (M4 hardening).

The synthetic-trained detector has a sim-to-real gap (e.g. faces read as cubes, since training
never contained people). Collecting a few hundred real frames -- cube in varied poses *and*
faces/hands, plus some frames with no cube at all -- lets us pseudo-label them and mix them into
training to close that gap. No preview window is shown (so it works with headless OpenCV); just
move the cube, your hands, and your face through the frame while it captures.

    python -m cube_tracker.eval.webcam_capture --out-dir data/real_frames --count 200
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


def _next_index(out_dir: Path) -> int:
    existing = sorted(out_dir.glob("real_*.jpg"))
    return int(existing[-1].stem.split("_")[1]) + 1 if existing else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture real webcam frames for fine-tuning.")
    parser.add_argument("--out-dir", required=True, help="Folder to save frames into.")
    parser.add_argument("--count", type=int, default=200, help="Number of frames to capture.")
    parser.add_argument("--interval", type=float, default=0.3, help="Seconds between saved frames.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--warmup", type=float, default=3.0, help="Seconds before capture starts.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not capture.isOpened():
        raise SystemExit(f"could not open webcam {args.camera}")

    start = _next_index(out_dir)
    print(f"warming up {args.warmup:.0f}s — get ready to move the cube, hands, and face around...")
    time.sleep(args.warmup)

    saved = 0
    last = 0.0
    try:
        while saved < args.count:
            ok, frame = capture.read()
            if not ok:
                break
            now = time.monotonic()
            if now - last >= args.interval:
                cv2.imwrite(str(out_dir / f"real_{start + saved:04d}.jpg"), frame)
                saved += 1
                last = now
                print(f"\rcaptured {saved}/{args.count}", end="", flush=True)
    finally:
        capture.release()
    print(f"\nsaved {saved} frames to {out_dir}")


if __name__ == "__main__":
    main()
