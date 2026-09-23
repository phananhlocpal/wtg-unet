from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .augment import AugmentConfig, apply_train_augmentation
from .labels import make_structural_targets
from .preprocess import preprocess_image, preprocess_mask, read_mask, read_rgb


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
_MASK_SUFFIXES = (
    "_ground_truth",
    "_groundtruth",
    "_gt",
    "_mask",
    "_segmentation",
    "_label",
)


@dataclass(frozen=True)
class ImageMaskPair:
    image: Path
    mask: Path

    @property
    def identifier(self) -> str:
        return self.image.stem


def _canonical_stem(path: Path) -> str:
    value = path.stem.lower().replace(" ", "_")
    for suffix in _MASK_SUFFIXES:
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value


def _image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def discover_pairs(image_dir: str | Path, mask_dir: str | Path) -> list[ImageMaskPair]:
    image_dir, mask_dir = Path(image_dir), Path(mask_dir)
    image_files, mask_files = _image_files(image_dir), _image_files(mask_dir)
    if not image_files:
        raise FileNotFoundError(f"No image files found under {image_dir}")
    if not mask_files:
        raise FileNotFoundError(f"No mask files found under {mask_dir}")

    masks_by_key: dict[str, list[Path]] = {}
    for path in mask_files:
        masks_by_key.setdefault(_canonical_stem(path), []).append(path)

    pairs: list[ImageMaskPair] = []
    unmatched: list[Path] = []
    for image in image_files:
        candidates = masks_by_key.get(_canonical_stem(image), [])
        if len(candidates) == 1:
            pairs.append(ImageMaskPair(image, candidates[0]))
        elif not candidates:
            unmatched.append(image)
        else:
            raise ValueError(f"Multiple masks match {image}: {candidates}")

    if unmatched:
        if len(image_files) != len(mask_files):
            names = ", ".join(path.name for path in unmatched[:3])
            raise ValueError(f"Could not match {len(unmatched)} images; examples: {names}")
        warnings.warn(
            "Falling back to sorted image/mask pairing because file stems do not match.",
            stacklevel=2,
        )
        pairs = [ImageMaskPair(image, mask) for image, mask in zip(image_files, mask_files)]

    return sorted(pairs, key=lambda pair: pair.identifier)


def _first_existing(root: Path, candidates: Iterable[str]) -> Path | None:
    for relative in candidates:
        candidate = root / relative
        if _image_files(candidate):
            return candidate
    return None


def resolve_data_dirs(
    root: str | Path,
    train_images: str | Path | None = None,
    train_masks: str | Path | None = None,
    test_images: str | Path | None = None,
    test_masks: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(root)

    def explicit(value: str | Path | None, fallback: Path | None) -> Path:
        if value is not None:
            path = Path(value)
            return path if path.is_absolute() else root / path
        if fallback is None:
            raise FileNotFoundError("Could not infer a data directory; pass it explicitly.")
        return fallback

    train_image_guess = _first_existing(
        root,
        (
            "train/images",
            "train/Images",
            "train/Original",
            "train/original",
            "training/images",
            "1_Training_Set/Original",
            "1_Training_Set/original",
        ),
    )
    train_mask_guess = _first_existing(
        root,
        (
            "train/masks",
            "train/Masks",
            "train/Ground truth",
            "train/Ground Truth",
            "train/ground_truth",
            "training/masks",
            "1_Training_Set/Ground truth",
            "1_Training_Set/Ground Truth",
        ),
    )
    test_image_guess = _first_existing(
        root,
        (
            "test/images",
            "test/Images",
            "test/Original",
            "test/original",
            "testing/images",
            "2_Testing_Set/Original",
            "2_Testing_Set/original",
        ),
    )
    test_mask_guess = _first_existing(
        root,
        (
            "test/masks",
            "test/Masks",
            "test/Ground truth",
            "test/Ground Truth",
            "test/ground_truth",
            "testing/masks",
            "2_Testing_Set/Ground truth",
            "2_Testing_Set/Ground Truth",
        ),
    )
    resolved = {
        "train_images": explicit(train_images, train_image_guess),
        "train_masks": explicit(train_masks, train_mask_guess),
        "test_images": explicit(test_images, test_image_guess),
        "test_masks": explicit(test_masks, test_mask_guess),
    }
    for name, path in resolved.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} directory does not exist: {path}")
    return resolved


class FIVESDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        pairs: list[ImageMaskPair],
        image_size: tuple[int, int] = (512, 512),
        train: bool = False,
        augmentation: dict[str, Any] | None = None,
        radius_normalization: str = "max_per_image",
        seed: int = 0,
    ) -> None:
        self.pairs = pairs
        self.image_size = image_size
        self.train = train
        self.radius_normalization = radius_normalization
        self.seed = seed
        self.epoch = 0
        augmentation = augmentation or {}
        self.augmentation = AugmentConfig(
            rotate_prob=float(augmentation.get("rotate_prob", 0.25)),
            rotate_degrees=float(augmentation.get("rotate_degrees", 15.0)),
            horizontal_flip_prob=float(augmentation.get("horizontal_flip_prob", 0.5)),
            vertical_flip_prob=float(augmentation.get("vertical_flip_prob", 0.5)),
            brightness_prob=float(augmentation.get("brightness_prob", 0.25)),
            brightness_range=tuple(augmentation.get("brightness_range", (0.8, 1.2))),
            noise_prob=float(augmentation.get("noise_prob", 0.15)),
            noise_std=float(augmentation.get("noise_std", 0.01)),
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        image = read_rgb(pair.image).astype(np.float32) / 255.0
        mask = read_mask(pair.mask).astype(np.float32)
        if float(mask.max()) > 1.5:
            mask /= 255.0
        if self.train:
            rng = np.random.default_rng(self.seed + index + self.epoch * 1_000_003)
            image, mask = apply_train_augmentation(image, mask, self.augmentation, rng)

        image = preprocess_image(image, self.image_size)
        mask = preprocess_mask(mask * 255.0, self.image_size)
        centerline, radius = make_structural_targets(mask, self.radius_normalization)
        _, radius_px = make_structural_targets(mask, "none")
        return {
            "image": torch.from_numpy(image[None, ...]),
            "mask": torch.from_numpy(mask[None, ...]),
            "centerline": torch.from_numpy(centerline[None, ...]),
            "radius": torch.from_numpy(radius[None, ...]),
            "radius_px": torch.from_numpy(radius_px[None, ...]),
            "id": pair.identifier,
            "image_path": str(pair.image),
            "mask_path": str(pair.mask),
        }


def split_pairs(
    pairs: list[ImageMaskPair],
    val_fraction: float,
    seed: int,
) -> tuple[list[ImageMaskPair], list[ImageMaskPair]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    indices = list(range(len(pairs)))
    random.Random(seed).shuffle(indices)
    n_val = int(np.floor(val_fraction * len(pairs)))
    val_indices = set(indices[:n_val])
    train = [pair for idx, pair in enumerate(pairs) if idx not in val_indices]
    val = [pair for idx, pair in enumerate(pairs) if idx in val_indices]
    return train, val


def _limit(items: list[ImageMaskPair], limit: int | None) -> list[ImageMaskPair]:
    return items if limit is None else items[: int(limit)]


def build_dataloaders(
    config: dict[str, Any],
    data_root: str | Path | None = None,
    seed: int | None = None,
    augmentation_seed: int | None = None,
    limit_train: int | None = None,
    limit_val: int | None = None,
    limit_test: int | None = None,
) -> tuple[dict[str, DataLoader], dict[str, list[ImageMaskPair]]]:
    data_cfg = config.get("data", {})
    root = data_root or data_cfg.get("root")
    if root is None:
        raise ValueError("Pass --data-root or set data.root in the YAML file.")
    directories = resolve_data_dirs(
        root,
        data_cfg.get("train_images"),
        data_cfg.get("train_masks"),
        data_cfg.get("test_images"),
        data_cfg.get("test_masks"),
    )
    development = discover_pairs(directories["train_images"], directories["train_masks"])
    test_pairs = discover_pairs(directories["test_images"], directories["test_masks"])
    split_seed = int(data_cfg.get("split_seed", 2026) if seed is None else seed)
    train_pairs, val_pairs = split_pairs(
        development, float(data_cfg.get("val_fraction", 0.2)), split_seed
    )
    train_pairs, val_pairs, test_pairs = (
        _limit(train_pairs, limit_train),
        _limit(val_pairs, limit_val),
        _limit(test_pairs, limit_test),
    )
    image_size = tuple(int(v) for v in data_cfg.get("image_size", [512, 512]))
    loss_cfg = config.get("loss", {})
    augmentation = config.get("augmentation", {})
    radius_norm = str(loss_cfg.get("radius_normalization", "max_per_image"))
    base_seed = split_seed if augmentation_seed is None else int(augmentation_seed)
    datasets = {
        "train": FIVESDataset(
            train_pairs, image_size, True, augmentation, radius_norm, base_seed + 11
        ),
        "val": FIVESDataset(
            val_pairs, image_size, False, augmentation, radius_norm, base_seed + 23
        ),
        "test": FIVESDataset(
            test_pairs, image_size, False, augmentation, radius_norm, base_seed + 37
        ),
    }
    optim_cfg = config.get("optim", {})
    batch_size = int(optim_cfg.get("batch_size", 4))
    num_workers = int(optim_cfg.get("num_workers", 0))
    generator = torch.Generator()
    generator.manual_seed(base_seed)

    def loader(dataset: Dataset[dict[str, Any]], shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            generator=generator if shuffle else None,
        )

    return (
        {
            "train": loader(datasets["train"], True),
            "val": loader(datasets["val"], False),
            "test": loader(datasets["test"], False),
        },
        {"train": train_pairs, "val": val_pairs, "test": test_pairs},
    )
