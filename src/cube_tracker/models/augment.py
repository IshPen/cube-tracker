"""Webcam-capture degradation augmentation: label-safe, pixel-only (M4+).

Emulates a real webcam on top of clean renders -- sensor noise, motion/defocus blur, JPEG
compression, resolution loss, and brightness/contrast/colour shifts -- so the models are
robust to cheap, noisy, compressed footage instead of only pristine renders. Every transform
is non-spatial, so it never moves the cube and the geometry labels stay valid. Shared by the
detector (M4) and reused by the keypoint/colour models (M5/M6).

Built against albumentations 2.x signatures.
"""

from __future__ import annotations

import albumentations as A


def webcam_degradation(p: float = 0.95) -> A.Compose:
    """Return a moderate webcam-degradation pipeline (applied to the image only)."""
    return A.Compose(
        [
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.02, 0.12), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.4), p=1.0),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=(3, 9), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.Defocus(radius=(2, 5), alias_blur=(0.1, 0.3), p=1.0),
                ],
                p=0.4,
            ),
            A.ImageCompression(quality_range=(35, 85), p=0.5),
            A.Downscale(scale_range=(0.4, 0.9), p=0.25),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=12, p=0.3),
        ],
        p=p,
    )
