"""Tests for the cube configuration schema (no Blender required)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from cube_tracker.common.config import CubeConfig, load_cube_config, load_render_config

_CONFIGS = Path(__file__).resolve().parents[1] / "configs"
_CONFIG = _CONFIGS / "cube.yaml"


def _valid_dict() -> dict[str, object]:
    return {
        "size_m": 0.057,
        "geometry": {
            "gap_m": 0.0016,
            "bevel_radius_m": 0.0018,
            "bevel_segments": 4,
            "tile_margin_m": 0.0016,
            "tile_height_m": 0.0004,
        },
        "materials": {
            "body_color": [0.0, 0.0, 0.0],
            "body_roughness": 0.45,
            "tile_roughness": 0.3,
            "face_colors": {
                "U": [0.9, 0.9, 0.9],
                "D": [0.95, 0.8, 0.05],
                "F": [0.0, 0.5, 0.18],
                "B": [0.0, 0.18, 0.6],
                "L": [0.9, 0.35, 0.0],
                "R": [0.7, 0.03, 0.03],
            },
        },
    }


def test_default_config_loads() -> None:
    config = load_cube_config(_CONFIG)
    assert config.size_m == pytest.approx(0.057)
    assert config.geometry.bevel_segments == 4
    assert pytest.approx((0.92, 0.92, 0.92)) == config.materials.face_colors.U


def test_default_config_has_weighted_variants() -> None:
    config = load_cube_config(_CONFIG)
    assert len(config.variants) >= 2
    names = {v.name for v in config.variants}
    assert "stickered_thick" in names  # the thick/Rubik's look is still represented
    assert all(v.weight > 0 for v in config.variants)


def test_default_render_config_loads() -> None:
    config = load_render_config(_CONFIGS / "render.yaml")
    assert config.device in ("CPU", "GPU")
    assert 0.0 <= config.background.hdri_probability <= 1.0
    assert config.distractors.count_max >= config.distractors.count_min


def test_rejects_out_of_range_color() -> None:
    data = _valid_dict()
    data["materials"]["face_colors"]["U"] = [1.5, 0.0, 0.0]  # type: ignore[index]
    with pytest.raises(ValidationError):
        CubeConfig.model_validate(data)


def test_rejects_unknown_field() -> None:
    data = _valid_dict()
    data["wobble"] = 1
    with pytest.raises(ValidationError):
        CubeConfig.model_validate(data)


def test_rejects_gap_larger_than_cubie() -> None:
    data = _valid_dict()
    data["geometry"]["gap_m"] = 0.05  # type: ignore[index]  # exceeds size_m / 3
    with pytest.raises(ValidationError):
        CubeConfig.model_validate(data)
