"""Launch the dataset render across several Blender processes in parallel (M3).

Runs in the project venv (no Blender). It spawns one Blender process per shard, each running
:mod:`cube_tracker.render.generate_dataset` over a disjoint slice of frame indices, shows
overall progress, then writes ``manifest.json`` describing the split. The Blender processes
do the rendering and the venv never imports ``bpy`` -- the two environments still only talk
through files on disk.

    python -m cube_tracker.render.dataset_launcher \
        --blender "/path/to/blender" --out-dir data/dataset --count 500 \
        --val-fraction 0.15 --num-shards 4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import IO

from tqdm import tqdm

from cube_tracker.render import dataset_plan


def _generate_dataset_script() -> Path:
    import cube_tracker.render as render_pkg

    return Path(render_pkg.__file__).resolve().parent / "generate_dataset.py"


def _count_rendered(out_dir: Path) -> int:
    return sum(len(list((out_dir / split).glob("*.png"))) for split in ("train", "val"))


def _write_manifest(out_dir: Path, args: argparse.Namespace) -> dict[str, int]:
    plan = dataset_plan.plan_frames(args.count, args.val_fraction)
    frames: list[dict[str, object]] = []
    counts = {"train": 0, "val": 0}
    for frame in plan:
        stem = f"frame_{frame.index:06d}"
        if (out_dir / frame.split / f"{stem}.png").exists():
            frames.append(
                {
                    "index": frame.index,
                    "split": frame.split,
                    "image": f"{frame.split}/{stem}.png",
                    "labels": f"{frame.split}/{stem}.labels.json",
                }
            )
            counts[frame.split] += 1
    manifest = {
        "count": args.count,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "num_shards": args.num_shards,
        "render_config": args.render_config,
        "cube_config": args.cube_config,
        "asset": args.asset,
        "counts": counts,
        "frames": frames,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return counts


def launch(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    script = _generate_dataset_script()

    processes: list[tuple[int, subprocess.Popen[bytes], IO[bytes]]] = []
    for shard in range(args.num_shards):
        command = [
            args.blender, "--background", "--python", str(script), "--",
            "--render-config", args.render_config, "--cube-config", args.cube_config,
            "--asset", args.asset, "--model-points", args.model_points,
            "--out-dir", str(out_dir), "--count", str(args.count),
            "--val-fraction", str(args.val_fraction),
            "--shard-index", str(shard), "--num-shards", str(args.num_shards),
        ]  # fmt: skip
        if args.seed is not None:
            command += ["--seed", str(args.seed)]
        log = (out_dir / f"shard_{shard}.log").open("wb")
        processes.append(
            (shard, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log)
        )

    with tqdm(total=args.count, desc="frames", unit="frame") as bar:
        done = 0
        while any(proc.poll() is None for _, proc, _ in processes):
            current = _count_rendered(out_dir)
            bar.update(current - done)
            done = current
            time.sleep(1.0)
        bar.update(_count_rendered(out_dir) - done)

    failed = []
    for shard, proc, handle in processes:
        handle.close()
        if proc.wait() != 0:
            failed.append(shard)
    if failed:
        raise RuntimeError(f"shards failed: {failed} (see {out_dir}/shard_*.log)")

    counts = _write_manifest(out_dir, args)
    print(f"[dataset_launcher] done: {counts['train']} train + {counts['val']} val -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the dataset in parallel Blender shards.")
    parser.add_argument("--blender", required=True, help="Path to the Blender executable.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--render-config", default="configs/render.yaml")
    parser.add_argument("--cube-config", default="configs/cube.yaml")
    parser.add_argument("--asset", default="assets/cube_model/cube_model.blend")
    parser.add_argument("--model-points", default="assets/cube_model/model_points.json")
    launch(parser.parse_args())


if __name__ == "__main__":
    main()
