"""Render and label a single synthetic frame in Blender (the per-frame unit, M2).

``render_and_label`` is the function that produces one training example and is the exact
code path the mass dataset run (M3) will call in a loop. For one frame it:

1. opens the cube asset fresh (so nothing leaks between frames),
2. paints a scrambled state onto the 54 tiles and jitters the palette and finish,
3. randomises camera, lighting, world/background, and optional occluders,
4. renders a PNG, and
5. exports labels by **ray casting against the real geometry**: a loose bounding box, each
   landmark's 2D location with a binary visible/occluded flag (one ray per point), and each
   facelet's colour with a coverage fraction (a 5x5 ray bundle per area).

Rays exist only here, at label time. At inference there is no Blender and no ray casting --
only the projected grid and pixel sampling.

Run one frame directly::

    blender --background --python src/cube_tracker/render/render_core.py -- \
        --render-config configs/render.yaml --cube-config configs/cube.yaml \
        --asset assets/cube_model/cube_model.blend \
        --model-points assets/cube_model/model_points.json --index 0 --out-dir data/frames
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import random
import site
import sys
from pathlib import Path
from typing import Any

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Quaternion, Vector

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
_USER_SITE = site.getusersitepackages()
if _USER_SITE and _USER_SITE not in sys.path:
    sys.path.append(_USER_SITE)

from cube_tracker.common.config import (  # noqa: E402
    CubeConfig,
    RenderConfig,
    load_cube_config,
    load_render_config,
)
from cube_tracker.common.config import Range as ConfigRange  # noqa: E402
from cube_tracker.render import cube_state  # noqa: E402

# Coverage rays land on the real tile surface, so the hit should be near-exact; this small
# tolerance only absorbs bevel/numerical slack (unlike the looser landmark tolerance, whose
# targets sit on the nominal surface a fraction of a millimetre above the tiles).
_COVERAGE_TOL_M = 0.0015
_FACE_LETTERS: tuple[str, ...] = ("U", "D", "F", "B", "L", "R")

# Outward normal axis of each face, used to decide self-occlusion (a landmark is a candidate
# for visibility only if one of its faces points toward the camera). Landmarks sit on the
# grid-line gaps, so a ray aimed exactly at one would slip through; normals handle self-
# occlusion and a ray handles external occluders.
_FACE_NORMAL: dict[str, tuple[float, float, float]] = {
    "U": (0.0, 0.0, 1.0),
    "D": (0.0, 0.0, -1.0),
    "F": (0.0, 1.0, 0.0),
    "B": (0.0, -1.0, 0.0),
    "R": (1.0, 0.0, 0.0),
    "L": (-1.0, 0.0, 0.0),
}


def _sample(rng: random.Random, span: ConfigRange) -> float:
    return rng.uniform(span.min, span.max)


def _jitter_color(
    rgb: tuple[float, float, float], palette: Any, rng: random.Random
) -> tuple[float, float, float]:
    """Jitter a base colour in HSV so the same cube-colour stays consistent across a frame."""
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    h = (h + rng.uniform(-palette.hue_deg, palette.hue_deg) / 360.0) % 1.0
    s = min(1.0, max(0.0, s * (1.0 + rng.uniform(-palette.saturation, palette.saturation))))
    v = min(1.0, max(0.0, v * (1.0 + rng.uniform(-palette.value, palette.value))))
    return colorsys.hsv_to_rgb(h, s, v)


def _set_principled(material: Any, color: tuple[float, float, float], roughness: float) -> None:
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
    bsdf.inputs["Roughness"].default_value = roughness


def paint_state(
    cube_config: CubeConfig,
    render_config: RenderConfig,
    facelet_map: dict[tuple[str, int, int], str],
    rng: random.Random,
) -> None:
    """Recolor the 54 tiles to a scrambled state, jittering palette and finish once per frame."""
    base = {
        "U": cube_config.materials.face_colors.U,
        "D": cube_config.materials.face_colors.D,
        "F": cube_config.materials.face_colors.F,
        "B": cube_config.materials.face_colors.B,
        "L": cube_config.materials.face_colors.L,
        "R": cube_config.materials.face_colors.R,
    }
    jittered = {letter: _jitter_color(base[letter], render_config.palette, rng) for letter in base}
    tile_roughness = _sample(rng, render_config.finish.tile_roughness)
    for (face, row, col), shown in facelet_map.items():
        material = bpy.data.materials[f"tile_{face}_r{row}_c{col}"]
        _set_principled(material, jittered[shown], tile_roughness)

    body_color = _jitter_color(cube_config.materials.body_color, render_config.palette, rng)
    _set_principled(
        bpy.data.materials["cube_body"],
        body_color,
        _sample(rng, render_config.finish.body_roughness),
    )


def place_camera(scene: Any, rng: random.Random, settings: Any) -> Any:
    """Create a camera at a random pose on a sphere around the cube, aimed at its centre."""
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = _sample(rng, settings.focal_mm)
    cam_data.sensor_width = settings.sensor_mm
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)

    azimuth = math.radians(rng.uniform(0.0, 360.0))
    elevation = math.radians(_sample(rng, settings.elevation_deg))
    distance = _sample(rng, settings.distance_m)
    cam.location = Vector(
        (
            distance * math.cos(elevation) * math.cos(azimuth),
            distance * math.cos(elevation) * math.sin(azimuth),
            distance * math.sin(elevation),
        )
    )
    look = (Vector((0.0, 0.0, 0.0)) - cam.location).to_track_quat("-Z", "Y")
    roll = Quaternion((0.0, 0.0, 1.0), math.radians(_sample(rng, settings.roll_deg)))
    cam.rotation_euler = (look @ roll).to_euler()
    scene.camera = cam
    return cam


def setup_world(scene: Any, rng: random.Random, lighting: Any) -> None:
    """Set a randomised grey world colour and strength as the ambient light and background."""
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    if world.node_tree is None:
        world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    value = _sample(rng, lighting.world_value)
    background.inputs["Color"].default_value = (value, value, value, 1.0)
    background.inputs["Strength"].default_value = _sample(rng, lighting.world_strength)


def add_lights(scene: Any, rng: random.Random, lighting: Any) -> None:
    """Add a few area lights at random directions, each aimed at the cube."""
    count = rng.randint(lighting.num_lights_min, lighting.num_lights_max)
    for index in range(count):
        light_data = bpy.data.lights.new(f"light_{index}", type="AREA")
        light_data.energy = _sample(rng, lighting.light_energy)
        light_data.size = 0.25
        light = bpy.data.objects.new(f"light_{index}", light_data)
        azimuth = math.radians(rng.uniform(0.0, 360.0))
        elevation = math.radians(rng.uniform(10.0, 80.0))
        radius = 0.4
        light.location = Vector(
            (
                radius * math.cos(elevation) * math.cos(azimuth),
                radius * math.cos(elevation) * math.sin(azimuth),
                radius * math.sin(elevation),
            )
        )
        light.rotation_euler = (
            (Vector((0.0, 0.0, 0.0)) - light.location).to_track_quat("-Z", "Y").to_euler()
        )
        scene.collection.objects.link(light)


def spawn_occluders(scene: Any, rng: random.Random, settings: Any, cam_location: Any) -> None:
    """Place random primitives between the camera and the cube to occlude parts of it."""
    count = rng.randint(settings.count_min, settings.count_max)
    for index in range(count):
        # Closer to the cube (higher fraction) means a smaller, more localised occlusion,
        # like a fingertip over a few stickers rather than a wall in front of the lens.
        fraction = rng.uniform(0.55, 0.8)
        center = cam_location.lerp(Vector((0.0, 0.0, 0.0)), fraction)
        jitter = Vector(
            (rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02))
        )
        size = _sample(rng, settings.size_m)
        mesh = bpy.data.meshes.new(f"occluder_{index}")
        if rng.random() < 0.5:
            _fill_cube(mesh, size)
        else:
            _fill_octahedron(mesh, size)
        obj = bpy.data.objects.new(f"occluder_{index}", mesh)
        obj.location = center + jitter
        material = bpy.data.materials.new(f"occluder_mat_{index}")
        if material.node_tree is None:
            material.use_nodes = True
        skin = (rng.uniform(0.5, 0.9), rng.uniform(0.3, 0.6), rng.uniform(0.25, 0.5))
        _set_principled(material, skin, 0.5)
        obj.data.materials.append(material)
        scene.collection.objects.link(obj)


def _fill_cube(mesh: Any, size: float) -> None:
    h = size / 2.0
    verts = [
        (-h, -h, -h),
        (h, -h, -h),
        (h, h, -h),
        (-h, h, -h),
        (-h, -h, h),
        (h, -h, h),
        (h, h, h),
        (-h, h, h),
    ]
    faces = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    mesh.from_pydata(verts, [], faces)
    mesh.update()


def _fill_octahedron(mesh: Any, size: float) -> None:
    h = size / 2.0
    verts = [(h, 0, 0), (-h, 0, 0), (0, h, 0), (0, -h, 0), (0, 0, h), (0, 0, -h)]
    faces = [
        (0, 2, 4),
        (2, 1, 4),
        (1, 3, 4),
        (3, 0, 4),
        (2, 0, 5),
        (1, 2, 5),
        (3, 1, 5),
        (0, 3, 5),
    ]
    mesh.from_pydata(verts, [], faces)
    mesh.update()


def setup_render(scene: Any, image: Any, seed: int) -> None:
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = image.width
    scene.render.resolution_y = image.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.cycles.samples = image.samples
    scene.cycles.use_denoising = True
    scene.cycles.seed = seed


def _project(
    scene: Any, cam: Any, world_co: Any, width: int, height: int
) -> tuple[float, float, float]:
    ndc = world_to_camera_view(scene, cam, Vector(world_co))
    return (ndc.x * width, (1.0 - ndc.y) * height, ndc.z)


def _ray_reaches(scene: Any, depsgraph: Any, origin: Any, target: Any, tol: float) -> bool:
    """True if the first surface hit travelling from origin to target is the target itself."""
    delta = target - origin
    distance = delta.length
    if distance == 0.0:
        return False
    hit, location, _normal, _index, _obj, _matrix = scene.ray_cast(
        depsgraph, origin, delta / distance
    )
    return bool(hit) and (location - target).length < tol


def _landmark_visible(
    scene: Any, depsgraph: Any, cam_origin: Any, position: Any, faces: list[str], margin: float
) -> bool:
    """Visible if a face points at the camera and no occluder is nearer than the landmark.

    Self-occlusion (back of the cube) is decided by face normals because the landmark lies
    in a gap a ray would pass through. External occlusion is decided by a single ray: if it
    strikes anything appreciably nearer than the landmark, something is in the way.
    """
    to_cam = cam_origin - position
    distance = to_cam.length
    if distance == 0.0:
        return False
    to_cam = to_cam / distance
    front = any(
        to_cam.x * n[0] + to_cam.y * n[1] + to_cam.z * n[2] > 0.0
        for n in (_FACE_NORMAL[f] for f in faces)
    )
    if not front:
        return False
    hit, location, _normal, _index, _obj, _matrix = scene.ray_cast(
        depsgraph, cam_origin, (position - cam_origin) / distance
    )
    return not (hit and (location - cam_origin).length < distance - margin)


def _bilinear(corners: list[Any], s: float, t: float) -> Any:
    top = corners[0].lerp(corners[1], s)
    bottom = corners[3].lerp(corners[2], s)
    return top.lerp(bottom, t)


def compute_labels(
    scene: Any,
    cam: Any,
    depsgraph: Any,
    model_points: dict[str, Any],
    facelet_map: dict[tuple[str, int, int], str],
    render_config: RenderConfig,
    scramble: str,
) -> dict[str, Any]:
    """Project landmarks/facelets and ray-cast visibility to build the frame's labels."""
    width = render_config.image.width
    height = render_config.image.height
    cam_origin = cam.matrix_world.translation

    landmarks: list[dict[str, Any]] = []
    xs: list[float] = []
    ys: list[float] = []
    for entry in model_points["landmarks"]:
        position = Vector(entry["position"])
        x, y, _z = _project(scene, cam, position, width, height)
        xs.append(x)
        ys.append(y)
        visible = _landmark_visible(
            scene, depsgraph, cam_origin, position, entry["faces"], render_config.visibility_tol_m
        )
        landmarks.append(
            {"name": entry["name"], "kind": entry["kind"], "x": x, "y": y, "visible": visible}
        )

    bbox = {
        "x_min": max(0.0, min(xs)),
        "y_min": max(0.0, min(ys)),
        "x_max": min(float(width), max(xs)),
        "y_max": min(float(height), max(ys)),
    }

    facelets: list[dict[str, Any]] = []
    for entry in model_points["facelets"]:
        face, row, col = entry["face"], entry["row"], entry["col"]
        tile = bpy.data.objects[entry["material"]]
        corners3d = [tile.matrix_world @ vertex.co for vertex in tile.data.vertices]
        hits = 0
        for i in range(5):
            for j in range(5):
                sample = _bilinear(corners3d, i / 4.0, j / 4.0)
                if _ray_reaches(scene, depsgraph, cam_origin, sample, _COVERAGE_TOL_M):
                    hits += 1
        centroid = sum(corners3d, Vector((0.0, 0.0, 0.0))) / len(corners3d)
        cx, cy, _z = _project(scene, cam, centroid, width, height)
        facelets.append(
            {
                "face": face,
                "row": row,
                "col": col,
                "color": facelet_map[(face, row, col)],
                "coverage": hits / 25.0,
                "center": [cx, cy],
                "corners": [list(_project(scene, cam, c, width, height)[:2]) for c in corners3d],
            }
        )

    return {
        "image": {"width": width, "height": height},
        "scramble": scramble,
        "bbox": bbox,
        "landmarks": landmarks,
        "facelets": facelets,
    }


def render_and_label(
    index: int,
    render_config: RenderConfig,
    cube_config: CubeConfig,
    asset_path: Path,
    model_points: dict[str, Any],
    out_dir: Path,
    force_occluders: bool | None = None,
) -> dict[str, Any]:
    """Render frame ``index`` and return its labels (also written next to the PNG)."""
    rng = random.Random(render_config.seed + index)
    bpy.ops.wm.open_mainfile(filepath=str(asset_path))
    scene = bpy.context.scene

    scramble = cube_state.random_scramble(render_config.scramble.num_moves, rng)
    facelet_map = cube_state.facelet_colors(scramble)
    paint_state(cube_config, render_config, facelet_map, rng)

    cam = place_camera(scene, rng, render_config.camera)
    setup_world(scene, rng, render_config.lighting)
    add_lights(scene, rng, render_config.lighting)

    spawn = (
        force_occluders
        if force_occluders is not None
        else rng.random() < render_config.occluders.probability
    )
    if spawn:
        spawn_occluders(scene, rng, render_config.occluders, cam.location)

    setup_render(scene, render_config.image, render_config.seed + index)

    stem = f"frame_{index:06d}"
    png_path = out_dir / f"{stem}.png"
    scene.render.filepath = str(png_path)
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    labels = compute_labels(
        scene, cam, depsgraph, model_points, facelet_map, render_config, scramble
    )
    labels["index"] = index
    labels["image"]["path"] = png_path.name

    (out_dir / f"{stem}.labels.json").write_text(
        json.dumps(labels, indent=2) + "\n", encoding="utf-8"
    )
    return labels


def _script_args(argv: list[str]) -> list[str]:
    return argv[argv.index("--") + 1 :] if "--" in argv else []


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render and label one synthetic frame.")
    parser.add_argument("--render-config", required=True)
    parser.add_argument("--cube-config", required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--model-points", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out-dir", default="data/frames")
    occ = parser.add_mutually_exclusive_group()
    occ.add_argument("--force-occluders", dest="force", action="store_true", default=None)
    occ.add_argument("--no-occluders", dest="force", action="store_false")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(_script_args(sys.argv))
    render_config = load_render_config(args.render_config)
    cube_config = load_cube_config(args.cube_config)
    model_points = json.loads(Path(args.model_points).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = render_and_label(
        args.index,
        render_config,
        cube_config,
        Path(args.asset).resolve(),
        model_points,
        out_dir,
        args.force,
    )
    visible = sum(1 for lm in labels["landmarks"] if lm["visible"])
    print(
        f"[render_core] frame {args.index}: {visible}/{len(labels['landmarks'])} landmarks visible"
    )
    print(f"[render_core] wrote {out_dir / f'frame_{args.index:06d}.png'}")


if __name__ == "__main__":
    main()
