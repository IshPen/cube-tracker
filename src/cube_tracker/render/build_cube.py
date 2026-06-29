"""Build the canonical 3x3 cube asset in Blender and export its exact labels.

Run under Blender's bundled Python (which provides ``bpy``), never the project venv::

    blender --background --python src/cube_tracker/render/build_cube.py -- \
        --config configs/cube.yaml --out-dir assets/cube_model

It builds beveled cubies with visible gaps, near-black Principled-BSDF plastic, and 54
individually named, recolorable tile materials, places a labelled empty at every surface
lattice point, then writes ``cube_model.blend`` and ``model_points.json``. The landmark and
facelet coordinates come from :mod:`cube_tracker.render.cube_geometry`, the same code that
positions the Blender objects, so the exported labels are exact by construction.

This script is deliberately deterministic: building the canonical asset involves no random
choices (palette and pose jitter belong to the per-frame render stage, M2), so it takes no
``--seed``.
"""

from __future__ import annotations

import argparse
import json
import site
import sys
from pathlib import Path
from typing import Any

import bmesh
import bpy

# build_cube.py lives at <repo>/src/cube_tracker/render/; add <repo>/src so the pure helper
# modules import under Blender's interpreter without installing the whole project.
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

# Blender ships its own Python without our config deps. They are installed with
# `<blender_python> -m pip install --user pydantic pyyaml`, which lands in the per-user site
# that Blender's embedded interpreter does not search by default, so add it explicitly.
_USER_SITE = site.getusersitepackages()
if _USER_SITE and _USER_SITE not in sys.path:
    sys.path.append(_USER_SITE)

from cube_tracker.common.config import CubeConfig, load_cube_config  # noqa: E402
from cube_tracker.render import cube_geometry as cg  # noqa: E402


def _script_args(argv: list[str]) -> list[str]:
    """Return the arguments after the ``--`` separator Blender uses to pass them through."""
    return argv[argv.index("--") + 1 :] if "--" in argv else []


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the cube asset and export labels.")
    parser.add_argument("--config", required=True, help="Path to the cube YAML config.")
    parser.add_argument(
        "--out-dir",
        default="assets/cube_model",
        help="Directory for cube_model.blend and model_points.json.",
    )
    parser.add_argument("--blend-name", default="cube_model.blend")
    parser.add_argument("--points-name", default="model_points.json")
    return parser.parse_args(argv)


def reset_scene() -> None:
    """Start from a guaranteed-empty file so repeated runs are reproducible."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def make_collection(name: str) -> Any:
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def make_material(name: str, color: tuple[float, float, float], roughness: float) -> Any:
    """Create a Principled-BSDF plastic material with the given base color and roughness."""
    material = bpy.data.materials.new(name)
    # Newer Blender enables the shader node tree by default; only opt in when it is absent
    # (older Blender) to avoid the deprecated ``use_nodes`` assignment.
    if material.node_tree is None:
        material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    return material


def make_cubie(
    name: str,
    body_size: float,
    bevel_radius: float,
    bevel_segments: int,
    location: cg.Vec3,
    material: Any,
    collection: Any,
) -> Any:
    """Create one beveled cubie cube, centered at ``location``, with the body material."""
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=body_size)
    bmesh.ops.bevel(
        bm,
        geom=[*bm.verts, *bm.edges],
        offset=bevel_radius,
        offset_type="OFFSET",
        segments=bevel_segments,
        profile=0.5,
        affect="EDGES",
        clamp_overlap=True,
    )
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    obj.data.materials.append(material)
    collection.objects.link(obj)
    return obj


def tile_corners(
    facelet: cg.Facelet, cell_half: float, margin: float, height: float
) -> list[cg.Vec3]:
    """Inset the facelet's nominal corners by ``margin`` and lift them ``height`` off the body."""
    inset = (cell_half - margin) / cell_half
    cx, cy, cz = facelet.center
    ox, oy, oz = (
        facelet.normal[0] * height,
        facelet.normal[1] * height,
        facelet.normal[2] * height,
    )
    corners: list[cg.Vec3] = []
    for px, py, pz in facelet.corners:
        corners.append(
            (
                cx + (px - cx) * inset + ox,
                cy + (py - cy) * inset + oy,
                cz + (pz - cz) * inset + oz,
            )
        )
    return corners


def make_tile(facelet: cg.Facelet, corners: list[cg.Vec3], material: Any, collection: Any) -> Any:
    """Create a flat quad tile from four corner points and assign its unique material."""
    mesh = bpy.data.meshes.new(facelet.material)
    mesh.from_pydata([list(c) for c in corners], [], [[0, 1, 2, 3]])
    mesh.update()
    obj = bpy.data.objects.new(facelet.material, mesh)
    obj.data.materials.append(material)
    collection.objects.link(obj)
    return obj


def make_landmark(landmark: cg.Landmark, display_size: float, collection: Any) -> Any:
    """Place a labelled plain-axes empty exactly on a surface lattice point."""
    empty = bpy.data.objects.new(landmark.name, None)
    empty.empty_display_type = "PLAIN_AXES"
    empty.empty_display_size = display_size
    empty.location = landmark.position
    collection.objects.link(empty)
    return empty


def build(config: CubeConfig) -> dict[str, object]:
    """Construct the full asset in the current Blender scene and return the model points."""
    reset_scene()

    cubie_collection = make_collection("Cubies")
    tile_collection = make_collection("Tiles")
    landmark_collection = make_collection("Landmarks")

    geom = config.geometry
    size = config.size_m
    cell = size / 3.0
    body_size = cell - geom.gap_m
    cell_half = size / 6.0

    body_material = make_material(
        "cube_body", config.materials.body_color, config.materials.body_roughness
    )

    # 26 visible cubies (the fully interior core piece is never seen, so it is skipped).
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                if i == 0 and j == 0 and k == 0:
                    continue
                make_cubie(
                    name=f"cubie_{i}_{j}_{k}",
                    body_size=body_size,
                    bevel_radius=geom.bevel_radius_m,
                    bevel_segments=geom.bevel_segments,
                    location=(i * cell, j * cell, k * cell),
                    material=body_material,
                    collection=cubie_collection,
                )

    face_color = {
        "U": config.materials.face_colors.U,
        "D": config.materials.face_colors.D,
        "F": config.materials.face_colors.F,
        "B": config.materials.face_colors.B,
        "L": config.materials.face_colors.L,
        "R": config.materials.face_colors.R,
    }
    for facelet in cg.compute_facelets(size):
        material = make_material(
            facelet.material, face_color[facelet.face], config.materials.tile_roughness
        )
        corners = tile_corners(facelet, cell_half, geom.tile_margin_m, geom.tile_height_m)
        make_tile(facelet, corners, material, tile_collection)

    landmark_display = size * 0.03
    for landmark in cg.compute_landmarks(size):
        make_landmark(landmark, landmark_display, landmark_collection)

    return cg.build_model_points(config)


def main() -> None:
    args = parse_args(_script_args(sys.argv))
    config = load_cube_config(args.config)

    model_points = build(config)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    blend_path = out_dir / args.blend_name
    points_path = out_dir / args.points_name

    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    points_path.write_text(json.dumps(model_points, indent=2) + "\n", encoding="utf-8")

    landmarks = model_points["landmarks"]
    facelets = model_points["facelets"]
    assert isinstance(landmarks, list) and isinstance(facelets, list)
    print(f"[build_cube] wrote {blend_path}")
    print(f"[build_cube] wrote {points_path}")
    print(f"[build_cube] {len(landmarks)} landmarks, {len(facelets)} facelets")


if __name__ == "__main__":
    main()
