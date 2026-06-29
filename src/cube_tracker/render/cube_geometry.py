"""Pure geometry of the canonical 3x3 cube model — no Blender, no GPU.

This module is the single source of truth for *where things are* on the cube: the
landmark points used later for pose recovery, and the 54 facelet quads. ``build_cube.py``
imports it to place Blender empties and tiles at exactly these coordinates and to write
``model_points.json`` from the same numbers, so the exported labels are exact by
construction rather than measured back off the mesh.

Coordinate system: a right-handed frame with +Z up, the cube centred at the origin, and
each axis spanning ``[-size/2, +size/2]``. Faces are named with Singmaster letters and map
to axes as U=+Z, D=-Z, F=+Y, B=-Y, R=+X, L=-X (white up, green front, red right).
"""

from __future__ import annotations

from dataclasses import dataclass

from cube_tracker.common.config import CubeConfig

Vec3 = tuple[float, float, float]
Axis = tuple[int, int, int]

# Tolerance for deciding whether a coordinate lies exactly on the outer surface (±size/2)
# when classifying and de-duplicating landmarks. The cube is metre-scale, so 1e-9 m is far
# below any meaningful geometric feature yet well above floating-point noise.
_EPS = 1e-9


@dataclass(frozen=True)
class Face:
    """A face frame: outward ``normal`` plus in-plane ``u`` (column) and ``v`` (row) axes."""

    name: str
    normal: Axis
    u: Axis
    v: Axis


# Each entry is an outward-facing unit axis. The order fixes the deterministic face order.
FACES: tuple[Face, ...] = (
    Face("U", (0, 0, 1), (1, 0, 0), (0, 1, 0)),
    Face("D", (0, 0, -1), (1, 0, 0), (0, -1, 0)),
    Face("F", (0, 1, 0), (1, 0, 0), (0, 0, 1)),
    Face("B", (0, -1, 0), (-1, 0, 0), (0, 0, 1)),
    Face("R", (1, 0, 0), (0, -1, 0), (0, 0, 1)),
    Face("L", (-1, 0, 0), (0, 1, 0), (0, 0, 1)),
)

# (axis index, sign) -> face name, used to report which faces a landmark sits on.
_AXIS_SIGN_TO_FACE: dict[tuple[int, int], str] = {
    (0, 1): "R",
    (0, -1): "L",
    (1, 1): "F",
    (1, -1): "B",
    (2, 1): "U",
    (2, -1): "D",
}


@dataclass(frozen=True)
class Landmark:
    """A surface lattice point: a cube corner or a grid-line intersection.

    ``kind`` is "corner" (3 coords on the surface), "edge" (2), or "face" (1). ``faces``
    lists the face letters the point lies on. These are the known 3D points that the 2D
    keypoint model is trained to locate and that PnP fits the camera pose against.
    """

    index: int
    name: str
    kind: str
    faces: tuple[str, ...]
    position: Vec3


@dataclass(frozen=True)
class Facelet:
    """One of the 54 stickers: its slot identity, material name, centre, and quad corners."""

    index: int
    face: str
    row: int
    col: int
    material: str
    center: Vec3
    corners: tuple[Vec3, Vec3, Vec3, Vec3]
    normal: Axis


def _scaled(axis: Axis, scalar: float) -> Vec3:
    return (axis[0] * scalar, axis[1] * scalar, axis[2] * scalar)


def _point_on_face(face: Face, half: float, u_coord: float, v_coord: float) -> Vec3:
    """Return the 3D point at in-plane offsets ``(u_coord, v_coord)`` on ``face``."""
    nx, ny, nz = _scaled(face.normal, half)
    ux, uy, uz = _scaled(face.u, u_coord)
    vx, vy, vz = _scaled(face.v, v_coord)
    return (nx + ux + vx, ny + uy + vy, nz + uz + vz)


def division_coords(size_m: float) -> tuple[float, float, float, float]:
    """The four grid-line offsets along a face axis: the tile boundaries [-h, -h/3, h/3, h]."""
    half = size_m / 2.0
    third = size_m / 3.0
    return (-half, -half + third, -half + 2.0 * third, half)


def cell_center_coords(size_m: float) -> tuple[float, float, float]:
    """The three tile-centre offsets along a face axis: [-size/3, 0, +size/3]."""
    third = size_m / 3.0
    return (-third, 0.0, third)


def compute_facelets(size_m: float) -> list[Facelet]:
    """Build all 54 facelets in deterministic (face, row, col) order."""
    div = division_coords(size_m)
    centers = cell_center_coords(size_m)
    half = size_m / 2.0
    facelets: list[Facelet] = []
    index = 0
    for face in FACES:
        for row in range(3):
            for col in range(3):
                center = _point_on_face(face, half, centers[col], centers[row])
                corners = (
                    _point_on_face(face, half, div[col], div[row]),
                    _point_on_face(face, half, div[col + 1], div[row]),
                    _point_on_face(face, half, div[col + 1], div[row + 1]),
                    _point_on_face(face, half, div[col], div[row + 1]),
                )
                facelets.append(
                    Facelet(
                        index=index,
                        face=face.name,
                        row=row,
                        col=col,
                        material=f"tile_{face.name}_r{row}_c{col}",
                        center=center,
                        corners=corners,
                        normal=face.normal,
                    )
                )
                index += 1
    return facelets


def _classify(position: Vec3, half: float) -> tuple[str, tuple[str, ...]]:
    """Return the landmark kind and the faces it lies on, from its surface coordinates."""
    faces: list[str] = []
    for axis_index, coord in enumerate(position):
        if abs(coord - half) < _EPS:
            faces.append(_AXIS_SIGN_TO_FACE[(axis_index, 1)])
        elif abs(coord + half) < _EPS:
            faces.append(_AXIS_SIGN_TO_FACE[(axis_index, -1)])
    kind = {3: "corner", 2: "edge", 1: "face"}[len(faces)]
    return kind, tuple(sorted(faces))


def compute_landmarks(size_m: float) -> list[Landmark]:
    """Build the unique surface lattice points (8 corners + 24 edge + 24 face = 56).

    Every face contributes its 4x4 grid of intersections; points shared between faces
    (corners and edge points) are de-duplicated. The result is sorted into a stable order
    so landmark indices are reproducible across runs and machines.
    """
    div = division_coords(size_m)
    half = size_m / 2.0
    seen: dict[tuple[int, int, int], Vec3] = {}
    for face in FACES:
        for u_coord in div:
            for v_coord in div:
                point = _point_on_face(face, half, u_coord, v_coord)
                key = tuple(round(c / _EPS) for c in point)
                seen.setdefault(key, point)  # type: ignore[arg-type]

    ordered = sorted(seen.values(), key=lambda p: (p[2], p[1], p[0]))
    landmarks: list[Landmark] = []
    for index, position in enumerate(ordered):
        kind, faces = _classify(position, half)
        landmarks.append(
            Landmark(index=index, name=f"lm_{index:02d}", kind=kind, faces=faces, position=position)
        )
    return landmarks


def build_model_points(config: CubeConfig) -> dict[str, object]:
    """Assemble the serialisable known-model description written to ``model_points.json``."""
    landmarks = compute_landmarks(config.size_m)
    facelets = compute_facelets(config.size_m)
    colors = config.materials.face_colors
    return {
        "schema_version": 1,
        "units": "meters",
        "coordinate_system": "right_handed_z_up_centered_at_cube_center",
        "cube_size_m": config.size_m,
        "color_scheme": {
            "U": list(colors.U),
            "D": list(colors.D),
            "F": list(colors.F),
            "B": list(colors.B),
            "L": list(colors.L),
            "R": list(colors.R),
        },
        "landmarks": [
            {
                "index": lm.index,
                "name": lm.name,
                "kind": lm.kind,
                "faces": list(lm.faces),
                "position": list(lm.position),
            }
            for lm in landmarks
        ],
        "facelets": [
            {
                "index": f.index,
                "face": f.face,
                "row": f.row,
                "col": f.col,
                "material": f.material,
                "center": list(f.center),
                "corners": [list(c) for c in f.corners],
                "normal": list(f.normal),
            }
            for f in facelets
        ],
    }
