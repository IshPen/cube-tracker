"""Tests for the pure cube geometry (no Blender required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cube_tracker.common.config import load_cube_config
from cube_tracker.render import cube_geometry as cg

_SIZE = 0.057
_HALF = _SIZE / 2.0
_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "cube.yaml"


def test_landmark_counts() -> None:
    landmarks = cg.compute_landmarks(_SIZE)
    assert len(landmarks) == 56
    kinds = [lm.kind for lm in landmarks]
    assert kinds.count("corner") == 8
    assert kinds.count("edge") == 24
    assert kinds.count("face") == 24


def test_corners_are_cube_vertices() -> None:
    corners = [lm for lm in cg.compute_landmarks(_SIZE) if lm.kind == "corner"]
    for lm in corners:
        assert all(abs(abs(c) - _HALF) < 1e-9 for c in lm.position)
        assert len(lm.faces) == 3


def test_landmark_indices_and_names_are_stable() -> None:
    first = cg.compute_landmarks(_SIZE)
    second = cg.compute_landmarks(_SIZE)
    assert [(lm.index, lm.name, lm.position) for lm in first] == [
        (lm.index, lm.name, lm.position) for lm in second
    ]
    assert [lm.index for lm in first] == list(range(56))
    assert first[0].name == "lm_00"


def test_facelet_counts_and_layout() -> None:
    facelets = cg.compute_facelets(_SIZE)
    assert len(facelets) == 54
    for face in ("U", "D", "F", "B", "L", "R"):
        assert sum(1 for f in facelets if f.face == face) == 9
    materials = {f.material for f in facelets}
    assert len(materials) == 54


def test_center_facelet_sits_at_face_center() -> None:
    facelets = cg.compute_facelets(_SIZE)
    centers = {f.face: f for f in facelets if f.row == 1 and f.col == 1}
    # The U center sticker must sit at the middle of the +Z face.
    assert centers["U"].center == pytest.approx((0.0, 0.0, _HALF))
    assert centers["D"].center == pytest.approx((0.0, 0.0, -_HALF))
    assert centers["R"].center == pytest.approx((_HALF, 0.0, 0.0))


def test_facelet_corners_span_one_third_of_the_face() -> None:
    facelets = cg.compute_facelets(_SIZE)
    up_corner = next(f for f in facelets if f.face == "U" and f.row == 0 and f.col == 0)
    xs = [c[0] for c in up_corner.corners]
    ys = [c[1] for c in up_corner.corners]
    assert max(xs) - min(xs) == pytest.approx(_SIZE / 3.0)
    assert max(ys) - min(ys) == pytest.approx(_SIZE / 3.0)
    assert all(c[2] == pytest.approx(_HALF) for c in up_corner.corners)


def test_model_points_payload() -> None:
    config = load_cube_config(_CONFIG)
    payload = cg.build_model_points(config)
    assert payload["cube_size_m"] == pytest.approx(_SIZE)
    assert isinstance(payload["landmarks"], list) and len(payload["landmarks"]) == 56
    assert isinstance(payload["facelets"], list) and len(payload["facelets"]) == 54
    scheme = payload["color_scheme"]
    assert isinstance(scheme, dict) and set(scheme) == {"U", "D", "F", "B", "L", "R"}
