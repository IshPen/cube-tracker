"""Split video clips into frames for the real-data fine-tune set (M4 hardening).

Record short clips (phone or webcam) of the cube in varied conditions, plus no-cube footage of
your room/face, then extract frames at a fixed interval. More convenient than live capture --
you control framing and can shoot several clips. Point ``--videos`` at one file or a folder of
clips; frames are written to one folder (continuing the numbering) ready for pseudo-labelling.

    python -m cube_tracker.eval.video_to_frames \
        --videos clips --out-dir data/real_frames --every 0.4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
_IMG_GLOB = "real_*.jpg"


def _video_paths(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.suffix.lower() in _VIDEO_EXT)
    return [source]


def _next_index(out_dir: Path) -> int:
    existing = sorted(out_dir.glob(_IMG_GLOB))
    return int(existing[-1].stem.split("_")[1]) + 1 if existing else 0


def extract(videos: list[Path], out_dir: Path, every_seconds: float, max_per_video: int) -> int:
    """Write a frame every ``every_seconds`` from each video; return total frames saved."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = _next_index(out_dir)
    start = saved
    for video in videos:
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(fps * every_seconds))
        frame_index = 0
        per_video = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % step == 0 and (max_per_video <= 0 or per_video < max_per_video):
                cv2.imwrite(str(out_dir / f"real_{saved:04d}.jpg"), frame)
                saved += 1
                per_video += 1
            frame_index += 1
        capture.release()
        print(f"  {video.name}: {per_video} frames")
    total = saved - start
    print(f"[video_to_frames] saved {total} frames to {out_dir}")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Split video clips into frames.")
    parser.add_argument("--videos", required=True, help="A video file or a folder of clips.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--every", type=float, default=0.4, help="Seconds between saved frames.")
    parser.add_argument("--max-per-video", type=int, default=0, help="Cap per clip (0 = no cap).")
    args = parser.parse_args()

    videos = _video_paths(Path(args.videos))
    if not videos:
        raise SystemExit(f"no videos found at {args.videos}")
    extract(videos, Path(args.out_dir), args.every, args.max_per_video)


if __name__ == "__main__":
    main()
