"""Live progress bar for a dataset render, by polling the output directory (M3).

The launcher prints its own bar, but when a render runs in the background (or in another
session) you can watch it from any terminal by counting the frames on disk:

    python -m cube_tracker.render.monitor_dataset \
        --out-dir P:/Development/cube-tracker-dataset --count 50000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from tqdm import tqdm


def _count(out_dir: Path) -> int:
    return sum(len(list((out_dir / split).glob("*.png"))) for split in ("train", "val"))


def monitor(out_dir: Path, count: int, interval: float) -> None:
    """Show a tqdm bar that tracks the number of rendered frames until it reaches ``count``."""
    with tqdm(total=count, desc="frames", unit="frame") as bar:
        done = 0
        while done < count:
            current = _count(out_dir)
            bar.update(current - done)
            done = current
            if current >= count:
                break
            time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a dataset render's progress live.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--interval", type=float, default=3.0, help="Seconds between polls.")
    args = parser.parse_args()
    monitor(Path(args.out_dir), args.count, args.interval)


if __name__ == "__main__":
    main()
