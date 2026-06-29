"""Cube state for render-time painting: scrambles and per-facelet colours.

This wraps pycuber to turn a scramble into the colour shown by each of the 54 stickers.
The colour is reported as the Singmaster *face letter* it belongs to (the cube-colour
identity), so the render stage can both paint the tile and record the ground-truth label
from the same source. It is pure Python (no Blender), so it is unit-testable in the venv.

There is **no smart cube**, so a written scramble applied to a solved cube is the ground
truth: it makes every facelet's colour known with certainty.
"""

from __future__ import annotations

import random

import pycuber as pc

FACES: tuple[str, ...] = ("U", "D", "F", "B", "L", "R")
_SUFFIXES: tuple[str, ...] = ("", "'", "2")


def random_scramble(num_moves: int, rng: random.Random) -> str:
    """Build a random scramble of ``num_moves`` quarter/half turns, no immediate repeats."""
    moves: list[str] = []
    last: str | None = None
    for _ in range(num_moves):
        face = rng.choice(FACES)
        while face == last:
            face = rng.choice(FACES)
        last = face
        moves.append(face + rng.choice(_SUFFIXES))
    return " ".join(moves)


def _colour_to_face() -> dict[str, str]:
    """Map each pycuber colour name to the face letter it is the home colour of."""
    solved = pc.Cube()
    return {solved.get_face(face)[1][1].colour: face for face in FACES}


def facelet_colors(scramble: str) -> dict[tuple[str, int, int], str]:
    """Return, for every (face, row, col), the face-letter colour that sticker shows.

    An empty scramble yields the solved cube, where each (face, r, c) maps to its own face.
    The (row, col) indexing follows pycuber's per-face convention; render painting and the
    exported colour label read from this same map, so they always agree.
    """
    cube = pc.Cube()
    if scramble.strip():
        cube(pc.Formula(scramble))
    colour_to_face = _colour_to_face()
    result: dict[tuple[str, int, int], str] = {}
    for face in FACES:
        grid = cube.get_face(face)
        for row in range(3):
            for col in range(3):
                result[(face, row, col)] = colour_to_face[grid[row][col].colour]
    return result
