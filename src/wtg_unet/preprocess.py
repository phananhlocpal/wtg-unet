from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask(path: str | Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return mask


def preprocess_image(
    image_rgb: np.ndarray,
    image_size: tuple[int, int] = (512, 512),
    clahe_clip_limit: float = 2.0,
    clahe_grid: tuple[int, int] = (8, 8),
    gaussian_sigma: float = 0.5,
) -> np.ndarray:
    """Return a normalized one-channel float32 image in H x W form."""
    height, width = image_size
    resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    if resized.ndim == 2:
        green = resized
    elif resized.shape[2] == 1:
        green = resized[..., 0]
    elif resized.shape[2] == 3:
        green = resized[..., 1]
    else:
        raise ValueError(f"Expected grayscale or RGB image, got shape {resized.shape}")

    green = green.astype(np.float32, copy=False)
    if float(green.max()) <= 1.5:
        green = green * 255.0
    green = np.clip(green, 0.0, 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid)
    enhanced = clahe.apply(green).astype(np.float32) / 255.0
    smoothed = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=gaussian_sigma)
    low, high = float(smoothed.min()), float(smoothed.max())
    if high > low:
        normalized = (smoothed - low) / (high - low)
    else:
        normalized = np.zeros_like(smoothed)
    mean, std = float(normalized.mean()), float(normalized.std())
    return ((normalized - mean) / (std + 1e-8)).astype(np.float32)


def preprocess_mask(
    mask: np.ndarray, image_size: tuple[int, int] = (512, 512)
) -> np.ndarray:
    height, width = image_size
    resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return (resized > 127).astype(np.float32)
