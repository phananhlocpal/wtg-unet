from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class AugmentConfig:
    rotate_prob: float = 0.25
    rotate_degrees: float = 15.0
    horizontal_flip_prob: float = 0.50
    vertical_flip_prob: float = 0.50
    brightness_prob: float = 0.25
    brightness_range: tuple[float, float] = (0.80, 1.20)
    noise_prob: float = 0.15
    noise_std: float = 0.01


def apply_train_augmentation(
    image: np.ndarray,
    mask: np.ndarray,
    config: AugmentConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply geometry to image/mask together and photometry to image only."""
    image = image.astype(np.float32, copy=False)
    mask = mask.astype(np.float32, copy=False)

    if rng.random() < config.rotate_prob:
        angle = float(rng.uniform(-config.rotate_degrees, config.rotate_degrees))
        image = ndimage.rotate(
            image, angle, axes=(0, 1), reshape=False, order=1, mode="reflect"
        )
        mask = ndimage.rotate(
            mask, angle, axes=(0, 1), reshape=False, order=0, mode="reflect"
        )

    if rng.random() < config.horizontal_flip_prob:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=1)
    if rng.random() < config.vertical_flip_prob:
        image = np.flip(image, axis=0)
        mask = np.flip(mask, axis=0)

    if rng.random() < config.brightness_prob:
        scale = float(rng.uniform(*config.brightness_range))
        image = image * scale
    if rng.random() < config.noise_prob:
        image = image + rng.normal(0.0, config.noise_std, size=image.shape).astype(np.float32)

    image = np.clip(image, 0.0, 1.0)
    mask = (mask > 0.5).astype(np.float32)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)

