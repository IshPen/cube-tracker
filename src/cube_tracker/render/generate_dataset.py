"""Render a shard of the dataset in Blender by looping ``render_and_label`` (M3).

This is the thin orchestrator: it does no rendering of its own, it just calls the *exact*
per-frame function the single-frame path uses, so a frame that passes the single-frame test
behaves identically in bulk. It writes each frame into ``<out-dir>/train`` or
``<out-dir>/val`` per :mod:`cube_tracker.render.dataset_plan`, skips frames whose outputs
already exist (crash-resume), and handles only the indices assigned to its shard so several
Blender processes can run in parallel (see :mod:`cube_tracker.render.dataset_launcher`).

Leak-safety comes from ``render_and_label`` re-opening the asset every call, so occluders,
motion blur, and lights from one frame cannot bleed into the next even within one process.

    blender --background --python src/cube_tracker/render/generate_dataset.py -- \
        --render-config configs/render.yaml --cube-config configs/cube.yaml \
        --asset assets/cube_model/cube_model.blend \
        --model-points assets/cube_model/model_points.json \
        --out-dir data/dataset --count 500 --val-fraction 0.15
"""

from __future__ import annotations

import argparse
import json
import site
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
_USER_SITE = site.getusersitepackages()
if _USER_SITE and _USER_SITE not in sys.path:
    sys.path.append(_USER_SITE)

from cube_tracker.common.config import load_cube_config, load_render_config  # noqa: E402
from cube_tracker.render import dataset_plan  # noqa: E402
from cube_tracker.render.render_core import render_and_label  # noqa: E402


def _script_args(argv: list[str]) -> list[str]:
    return argv[argv.index("--") + 1 :] if "--" in argv else []


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one shard of the dataset.")
    parser.add_argument("--render-config", required=True)
    parser.add_argument("--cube-config", required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--model-points", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=None, help="Override the render-config seed.")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(_script_args(sys.argv))
    render_config = load_render_config(args.render_config)
    if args.seed is not None:
        render_config = render_config.model_copy(update={"seed": args.seed})
    cube_config = load_cube_config(args.cube_config)
    model_points = json.loads(Path(args.model_points).read_text(encoding="utf-8"))
    asset = Path(args.asset).resolve()
    out_dir = Path(args.out_dir).resolve()

    split_by_index = {
        fp.index: fp.split for fp in dataset_plan.plan_frames(args.count, args.val_fraction)
    }
    indices = dataset_plan.shard_indices(args.count, args.shard_index, args.num_shards)

    rendered = 0
    skipped = 0
    for position, index in enumerate(indices, start=1):
        split_dir = out_dir / split_by_index[index]
        split_dir.mkdir(parents=True, exist_ok=True)
        stem = f"frame_{index:06d}"
        if (split_dir / f"{stem}.png").exists() and (split_dir / f"{stem}.labels.json").exists():
            skipped += 1
            continue
        render_and_label(index, render_config, cube_config, asset, model_points, split_dir)
        rendered += 1
        print(
            f"[generate_dataset] shard {args.shard_index}/{args.num_shards} "
            f"{position}/{len(indices)} index {index} -> {split_by_index[index]}",
            flush=True,
        )

    print(
        f"[generate_dataset] shard {args.shard_index} done: {rendered} rendered, {skipped} skipped",
        flush=True,
    )


if __name__ == "__main__":
    main()
