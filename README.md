# Cube Tracker

Reconstruct the moves and rotations of a human Rubik's cube solve from **ordinary webcam
video** — no Bluetooth smart cube, no special hardware.

The core idea: the cube's structure is known in advance, so the learned models only ever
*locate* points or *read* colors, while all 3D reasoning and reconstruction of unobserved
moves is done by deterministic geometry and combinatorics.

> **Status:** early (M1). The project is built one milestone at a time.

## Quickstart (development)

Requires **Python 3.11+**.

```bash
git clone https://github.com/IshPen/cube-tracker.git
cd cube-tracker
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install

ruff check . && mypy && pytest
```

## Building the cube asset

The synthetic-data pipeline starts from a parametric cube built in Blender. This step runs
under **Blender's** bundled Python, not the project venv.

1. Install **Blender 4.x or newer** and note the path to `blender` (`blender.exe` on Windows).
2. Install the two config dependencies into Blender's Python (one-time):

   ```bash
   "<blender>/python/bin/python" -m pip install --user pydantic pyyaml
   ```

3. Build the asset:

   ```bash
   blender --background --python src/cube_tracker/render/build_cube.py -- \
     --config configs/cube.yaml --out-dir assets/cube_model
   ```

This writes `cube_model.blend` and `model_points.json` into `assets/cube_model/`
(gitignored). Open the `.blend` in Blender to inspect the cube and its landmark empties.

## Rendering a labelled frame

`render_core.py` renders one synthetic frame (scrambled, randomly lit and posed, optionally
occluded) and writes its labels — bounding box, per-landmark 2D position and visibility, and
per-facelet colour and coverage — by ray casting against the real geometry. It runs under
Blender; install `pycuber` into Blender's Python once (`<blender>/python/bin/python -m pip
install --user pycuber`), then:

```bash
blender --background --python src/cube_tracker/render/render_core.py -- \
  --render-config configs/render.yaml --cube-config configs/cube.yaml \
  --asset assets/cube_model/cube_model.blend \
  --model-points assets/cube_model/model_points.json --index 0 --out-dir data/frames
```

Draw the labels back onto the render (in the project venv) to check them by eye:

```bash
python -m cube_tracker.eval.reproject_overlay --labels data/frames/frame_000000.labels.json
```

Frames and labels land in `data/` (gitignored). Add `--force-occluders` / `--no-occluders`
to the render command to force an occluded or clean frame.

## Generating a dataset

Mass-render many labelled frames into a train/val split, in parallel and reproducibly. The
launcher runs in the venv and spawns one Blender process per shard (it never imports `bpy`):

```bash
python -m cube_tracker.render.dataset_launcher \
  --blender "<path-to-blender>" --out-dir data/dataset \
  --count 500 --val-fraction 0.15 --num-shards 4 --seed 0
```

This writes `data/dataset/{train,val}/frame_XXXXXX.{png,labels.json}` plus a `manifest.json`.
Re-running **skips frames that already exist** (crash-resume), and a fixed `--seed` reproduces
the same dataset. Each frame is produced by the *same* `render_and_label` used for a single
frame, so the single-frame test and the bulk run share one code path. To render one shard
directly under Blender (no launcher), run `generate_dataset.py` with `--shard-index` /
`--num-shards`.

## License

[Apache-2.0](LICENSE).
