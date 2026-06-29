"""Tests for scramble generation and per-facelet colours (no Blender required)."""

from __future__ import annotations

import random
from collections import Counter
from itertools import pairwise

from cube_tracker.render import cube_state as cs


def test_solved_cube_each_facelet_shows_its_own_face() -> None:
    colors = cs.facelet_colors("")
    assert len(colors) == 54
    for (face, _row, _col), shown in colors.items():
        assert shown == face


def test_scramble_preserves_nine_of_each_color() -> None:
    scramble = cs.random_scramble(25, random.Random(123))
    counts = Counter(cs.facelet_colors(scramble).values())
    assert dict(counts) == dict.fromkeys(cs.FACES, 9)


def test_random_scramble_is_well_formed_and_deterministic() -> None:
    first = cs.random_scramble(20, random.Random(7))
    second = cs.random_scramble(20, random.Random(7))
    assert first == second  # seeded -> reproducible

    tokens = first.split()
    assert len(tokens) == 20
    faces = [t[0] for t in tokens]
    assert all(f in cs.FACES for f in faces)
    assert all(a != b for a, b in pairwise(faces))  # no immediate repeats
