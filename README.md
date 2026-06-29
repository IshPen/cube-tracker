# Cube Tracker

Reconstruct the moves and rotations of a human Rubik's cube solve from **ordinary webcam
video** — no Bluetooth smart cube, no special hardware.

The core idea: the cube's structure is known in advance, so the learned models only ever
*locate* points or *read* colors, while all 3D reasoning and reconstruction of unobserved
moves is done by deterministic geometry and combinatorics.

> **Status:** early scaffold (M0). The project is built one milestone at a time.

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

## License

[Apache-2.0](LICENSE).
