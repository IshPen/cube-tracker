"""Smoke tests: the package imports and exposes a version (M0 scaffold check)."""

from __future__ import annotations

import cube_tracker


def test_package_exposes_version() -> None:
    assert isinstance(cube_tracker.__version__, str)
    assert cube_tracker.__version__
