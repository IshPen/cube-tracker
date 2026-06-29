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

## License

[Apache-2.0](LICENSE).
