from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .losses import MultitaskLoss
from .metrics import finalize_metric_sums, merge_metric_sums, metrics_from_batch
from .utils import AverageMeter


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: MultitaskLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool = False,
    epoch: int = 0,
) -> dict[str, float]:
    model.train()
    if hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(epoch)
    meters = {key: AverageMeter() for key in ("total", "mask", "centerline", "radius", "topology")}
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    progress = tqdm(loader, desc=f"train {epoch + 1}", leave=False)
    for batch in progress:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            outputs = model(batch["image"])
            losses = criterion(outputs, batch)
        scaler.scale(losses["total"]).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(batch["image"].shape[0])
        for key, meter in meters.items():
            meter.update(float(losses[key].detach().cpu()), batch_size)
        progress.set_postfix(loss=f"{meters['total'].average:.4f}")
    return {key: meter.average for key, meter in meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: MultitaskLoss,
    device: torch.device,
    threshold: float = 0.5,
    skeleton_iterations: int = 10,
) -> dict[str, float]:
    model.eval()
    meters = {key: AverageMeter() for key in ("total", "mask", "centerline", "radius", "topology")}
    metric_sum: dict[str, float] = {}
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_batch(batch, device)
        outputs = model(batch["image"])
        losses = criterion(outputs, batch)
        batch_size = int(batch["image"].shape[0])
        for key, meter in meters.items():
            meter.update(float(losses[key].detach().cpu()), batch_size)
        metrics = metrics_from_batch(
            outputs, batch, threshold, skeleton_iterations=skeleton_iterations
        )
        metric_sum = merge_metric_sums(metric_sum, metrics, batch_size)
    result = {f"loss_{key}": meter.average for key, meter in meters.items()}
    result.update(finalize_metric_sums(metric_sum))
    return result


class WarmupPlateau:
    """Linear warm-up followed by validation-Dice plateau reduction."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        warmup_epochs: int,
        start_factor: float,
        patience: int,
        factor: float,
        min_lr: float,
    ) -> None:
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.warmup_epochs = int(warmup_epochs)
        self.start_factor = float(start_factor)
        self.patience = int(patience)
        self.factor = float(factor)
        self.min_lr = float(min_lr)
        self.epoch = 0
        self.best = -np.inf
        self.bad_epochs = 0
        self._set_lr(self.base_lr * self.start_factor)

    def _set_lr(self, value: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(value)

    @property
    def learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def step(self, validation_dice: float) -> None:
        self.epoch += 1
        if self.epoch <= self.warmup_epochs:
            fraction = self.epoch / max(self.warmup_epochs, 1)
            self._set_lr(
                self.base_lr
                * (self.start_factor + fraction * (1.0 - self.start_factor))
            )
            return
        if validation_dice > self.best + 1e-8:
            self.best = validation_dice
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self._set_lr(max(self.learning_rate * self.factor, self.min_lr))
            self.bad_epochs = 0
