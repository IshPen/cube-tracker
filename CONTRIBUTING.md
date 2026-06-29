# Contributing to Cube Tracker

Thanks for your interest! This project reconstructs the moves of a human Rubik's cube solve
from ordinary webcam video, building the pipeline one milestone at a time.

## Development setup

```bash
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Milestone-by-milestone build

This project is built **strictly one milestone at a time**. Each stage silently corrupts the
next if it is subtly wrong, so every milestone is verified in isolation before the next begins.
Please scope pull requests to the current milestone and do not build ahead.

## Engineering standards

- **Python 3.11+**, full type hints on public functions, clean under `mypy --strict`.
- Lint and format with **ruff**; test with **pytest**. New logic ships **with tests**.
- **No magic numbers in code** — randomization ranges and hyperparameters live in `configs/*.yaml`,
  validated by `pydantic` schemas.
- **Determinism:** thread a `--seed` through anything stochastic (NumPy, PyTorch, Python `random`).
- Geometry, state, and reconstruction code must be unit-testable **without Blender and without a GPU**.
- Docstrings explain **why** a thing is done, not merely what. No dead or commented-out code.

## Commits and pull requests

- Follow **[Conventional Commits](https://www.conventionalcommits.org/)**: `feat:`, `fix:`,
  `docs:`, `test:`, `refactor:`, `chore:`, `ci:`, `build:`.
- Keep each commit **small and focused** on one logical change.
- **Never commit** datasets, rendered images, model weights, `.blend` files, downloaded assets,
  secrets, or tokens. The `.gitignore` and the `check-added-large-files` hook help enforce this.
- Work on a feature branch and open a pull request into `main`. Keep `main` green: `ruff`, `mypy`,
  and `pytest` must pass in CI.
