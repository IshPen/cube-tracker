"""Cube Tracker: reconstruct a human Rubik's cube solve from ordinary webcam video.

The guiding principle is that the cube's structure is known in advance, so the learned
models only ever *locate* points or *read* colors, while all 3D reasoning and
reconstruction of unobserved moves is deterministic.
"""

__version__ = "0.0.0"
