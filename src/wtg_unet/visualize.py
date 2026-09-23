from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def _display_image(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, (1, 99))
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def error_map(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = prediction.astype(bool)
    gt = target.astype(bool)
    result = np.zeros((*gt.shape, 3), dtype=np.float32)
    result[np.logical_and(pred, gt)] = (0.0, 0.8, 0.2)  # true positive
    result[np.logical_and(pred, ~gt)] = (0.9, 0.1, 0.1)  # false positive
    result[np.logical_and(~pred, gt)] = (0.1, 0.3, 0.9)  # false negative
    return result


@torch.no_grad()
def save_example_grid(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str | Path,
    threshold: float = 0.5,
    limit: int = 4,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    for batch in loader:
        images = batch["image"].to(device)
        outputs = model(images)
        probabilities = outputs["mask"].detach().cpu().numpy()[:, 0]
        for idx in range(images.shape[0]):
            if saved >= limit:
                return
            image = batch["image"][idx, 0].numpy()
            target = batch["mask"][idx, 0].numpy() >= 0.5
            prediction = probabilities[idx] >= threshold
            fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
            axes[0].imshow(_display_image(image), cmap="gray")
            axes[0].set_title("Preprocessed")
            axes[1].imshow(target, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("Ground truth")
            axes[2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
            axes[2].set_title("Prediction")
            axes[3].imshow(probabilities[idx], cmap="magma", vmin=0, vmax=1)
            axes[3].set_title("Probability")
            axes[4].imshow(error_map(prediction, target))
            axes[4].set_title("Error map")
            for axis in axes:
                axis.axis("off")
            identifier = str(batch["id"][idx])
            fig.suptitle(identifier)
            fig.tight_layout()
            fig.savefig(output_dir / f"example-{saved + 1:02d}.png", dpi=160)
            plt.close(fig)
            saved += 1


def save_confusion_plot(
    counts: dict[str, float], output_path: str | Path
) -> None:
    matrix = np.array(
        [[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]],
        dtype=np.float64,
    )
    fig, axis = plt.subplots(figsize=(4, 3.5))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks([0, 1], ["Nền", "Mạch"])
    axis.set_yticks([0, 1], ["Nền", "Mạch"])
    axis.set_xlabel("Dự đoán")
    axis.set_ylabel("Nhãn chuẩn")
    axis.set_title("Ma trận nhầm lẫn cộng gộp")
    for (row, col), value in np.ndenumerate(matrix):
        axis.text(col, row, f"{int(value):,}", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

