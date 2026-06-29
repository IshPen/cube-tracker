"""Configuration schemas, validated with pydantic.

Centralising the schema here lets both the project virtualenv and Blender's bundled
Python validate the *same* versioned YAML configs. This module imports only pydantic,
PyYAML, and the standard library, so it loads cleanly under Blender's interpreter
(which does not have the heavier parts of the project venv such as torch or OpenCV).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


def _unit_rgb(value: tuple[float, float, float]) -> tuple[float, float, float]:
    """Reject colors whose channels fall outside the renderable [0, 1] range."""
    if any(not 0.0 <= channel <= 1.0 for channel in value):
        raise ValueError(f"RGB channels must lie in [0, 1]; got {value!r}")
    return value


Color = Annotated[tuple[float, float, float], AfterValidator(_unit_rgb)]
"""A linear RGB base color: three channels, each in [0, 1]."""


class FaceColors(BaseModel):
    """Base sticker color for each face, keyed by Singmaster face letter (U D F B L R)."""

    model_config = ConfigDict(extra="forbid")

    U: Color
    D: Color
    F: Color
    B: Color
    L: Color
    R: Color


class GeometrySettings(BaseModel):
    """Parameters that govern how convincingly the cube body and tiles are printed.

    These are the realism knobs called out in the build plan: gap width, bevel radius,
    and tile geometry. They are deliberately kept out of the source so a render looks
    different by editing config, never by editing code.
    """

    model_config = ConfigDict(extra="forbid")

    gap_m: float = Field(gt=0, description="Gap between adjacent cubies, in metres.")
    bevel_radius_m: float = Field(gt=0, description="Cubie edge bevel radius, in metres.")
    bevel_segments: int = Field(ge=1, description="Bevel resolution (number of segments).")
    tile_margin_m: float = Field(gt=0, description="Border from cubie face edge to tile edge.")
    tile_height_m: float = Field(
        gt=0, description="Sticker thickness: how far each tile sits above the cubie body surface."
    )


class MaterialSettings(BaseModel):
    """Principled-BSDF material parameters for the plastic body and the colored tiles."""

    model_config = ConfigDict(extra="forbid")

    body_color: Color
    body_roughness: float = Field(ge=0, le=1)
    tile_roughness: float = Field(ge=0, le=1)
    face_colors: FaceColors


class CubeConfig(BaseModel):
    """Top-level cube build configuration (see ``configs/cube.yaml``)."""

    model_config = ConfigDict(extra="forbid")

    size_m: float = Field(gt=0, description="Full edge length of the cube, in metres.")
    geometry: GeometrySettings
    materials: MaterialSettings

    @model_validator(mode="after")
    def _check_fits(self) -> CubeConfig:
        """Ensure the geometry knobs leave a physically buildable cubie and tile.

        A 3x3 cube divides each axis into three cubies of ``size_m / 3``. Each cubie is
        shrunk by the inter-cubie gap, and each tile is inset from the cubie face by a
        margin; if these exceed the available room the asset would self-intersect.
        """
        cubie = self.size_m / 3.0
        body = cubie - self.geometry.gap_m
        if body <= 0:
            raise ValueError("gap_m is too large: it leaves no cubie body within size_m / 3.")
        if 2.0 * self.geometry.bevel_radius_m >= body:
            raise ValueError("bevel_radius_m is too large for the cubie body size.")
        if 2.0 * self.geometry.tile_margin_m >= body:
            raise ValueError("tile_margin_m is too large for the cubie face size.")
        return self


def load_cube_config(path: str | Path) -> CubeConfig:
    """Load and validate a cube configuration from a YAML file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return CubeConfig.model_validate(raw)


# --- Render configuration (M2): the domain-randomisation ranges for a single frame ---


class Range(BaseModel):
    """An inclusive ``[min, max]`` interval that a value is uniformly sampled from."""

    model_config = ConfigDict(extra="forbid")

    min: float
    max: float

    @model_validator(mode="after")
    def _ordered(self) -> Range:
        if self.max < self.min:
            raise ValueError(f"range max ({self.max}) must be >= min ({self.min}).")
        return self


class ImageSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    samples: int = Field(ge=1, description="Cycles path-tracing samples per pixel.")


class CameraSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distance_m: Range = Field(description="Camera distance from the cube centre.")
    focal_mm: Range = Field(description="Lens focal length (varies field of view).")
    sensor_mm: float = Field(gt=0)
    elevation_deg: Range = Field(description="Angle above/below the cube's equator.")
    roll_deg: Range = Field(description="Camera roll about the view axis.")


class LightingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    world_strength: Range
    world_value: Range = Field(description="Brightness of the ambient world colour.")
    num_lights_min: int = Field(ge=0)
    num_lights_max: int = Field(ge=0)
    light_energy: Range = Field(description="Per-light power, in watts.")
    hdri_dir: str | None = Field(
        default=None, description="Optional folder of .hdr/.exr environment maps; used if present."
    )


class PaletteJitter(BaseModel):
    """Per-frame jitter applied to the six base colours and body, for cross-brand robustness."""

    model_config = ConfigDict(extra="forbid")

    hue_deg: float = Field(ge=0)
    saturation: float = Field(ge=0, description="Max fractional saturation shift.")
    value: float = Field(ge=0, description="Max fractional brightness shift.")


class FinishSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tile_roughness: Range
    body_roughness: Range


class OccluderSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    probability: float = Field(ge=0, le=1, description="Chance a frame spawns occluders at all.")
    count_min: int = Field(ge=0)
    count_max: int = Field(ge=0)
    size_m: Range


class ScrambleSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_moves: int = Field(ge=0, description="Random scramble length; 0 renders a solved cube.")


class MotionBlurSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    probability: float = Field(ge=0, le=1)
    shutter: float = Field(gt=0)
    max_rotation_deg: float = Field(ge=0, description="Cube spin across the shutter when blurred.")


class RenderConfig(BaseModel):
    """Top-level per-frame render configuration (see ``configs/render.yaml``)."""

    model_config = ConfigDict(extra="forbid")

    seed: int
    device: Literal["CPU", "GPU"] = Field(
        default="GPU", description="Cycles render device; GPU uses OptiX/CUDA, falling back to CPU."
    )
    image: ImageSettings
    camera: CameraSettings
    lighting: LightingSettings
    palette: PaletteJitter
    finish: FinishSettings
    occluders: OccluderSettings
    scramble: ScrambleSettings
    motion_blur: MotionBlurSettings
    visibility_tol_m: float = Field(
        gt=0, description="Ray hit-distance tolerance when deciding landmark visibility."
    )

    @model_validator(mode="after")
    def _counts_ordered(self) -> RenderConfig:
        if self.lighting.num_lights_max < self.lighting.num_lights_min:
            raise ValueError("lighting.num_lights_max must be >= num_lights_min.")
        if self.occluders.count_max < self.occluders.count_min:
            raise ValueError("occluders.count_max must be >= count_min.")
        return self


def load_render_config(path: str | Path) -> RenderConfig:
    """Load and validate a render configuration from a YAML file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return RenderConfig.model_validate(raw)
