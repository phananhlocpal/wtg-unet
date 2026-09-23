from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def weighted_dice_loss(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    positive_weight: float = 1.0,
    negative_weight: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    weights = target * positive_weight + (1.0 - target) * negative_weight
    dims = tuple(range(1, probabilities.ndim))
    intersection = (weights * probabilities * target).sum(dim=dims)
    denominator = (weights * probabilities.square()).sum(dim=dims)
    denominator = denominator + (weights * target.square()).sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return (1.0 - dice).mean()


def weighted_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    positive_weight: float = 1.0,
    negative_weight: float = 1.0,
) -> torch.Tensor:
    weights = target * positive_weight + (1.0 - target) * negative_weight
    return F.binary_cross_entropy_with_logits(logits, target, weight=weights)


def soft_erode(image: torch.Tensor) -> torch.Tensor:
    vertical = -F.max_pool2d(-image, kernel_size=(3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-image, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def soft_dilate(image: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(image, kernel_size=3, stride=1, padding=1)


def soft_open(image: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(image))


def soft_skeletonize(image: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    image = image.clamp(0.0, 1.0)
    opened = soft_open(image)
    skeleton = F.relu(image - opened)
    current = image
    for _ in range(max(int(iterations), 1)):
        current = soft_erode(current)
        delta = F.relu(current - soft_open(current))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton.clamp(0.0, 1.0)


def soft_cldice(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    iterations: int = 10,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred_skeleton = soft_skeletonize(probabilities, iterations)
    target_skeleton = soft_skeletonize(target, iterations)
    dims = tuple(range(1, probabilities.ndim))
    topology_precision = (pred_skeleton * target).sum(dim=dims) / (
        pred_skeleton.sum(dim=dims) + eps
    )
    topology_sensitivity = (target_skeleton * probabilities).sum(dim=dims) / (
        target_skeleton.sum(dim=dims) + eps
    )
    score = 2.0 * topology_precision * topology_sensitivity / (
        topology_precision + topology_sensitivity + eps
    )
    return score.mean()


class MultitaskLoss(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        loss_cfg = config.get("loss", {})
        self.dice_weight = float(loss_cfg.get("dice_weight", 0.7))
        self.bce_weight = float(loss_cfg.get("bce_weight", 0.3))
        self.centerline_weight = float(loss_cfg.get("centerline_weight", 1.0))
        self.radius_weight = float(loss_cfg.get("radius_weight", 1.0))
        self.topology_weight = float(loss_cfg.get("topology_weight", 1.0))
        self.use_centerline = bool(loss_cfg.get("use_centerline", True))
        self.use_radius = bool(loss_cfg.get("use_radius", True))
        self.use_topology = bool(loss_cfg.get("use_topology", True))
        self.positive_weight = float(loss_cfg.get("positive_weight", 1.0))
        self.negative_weight = float(loss_cfg.get("negative_weight", 1.0))
        self.skeleton_iterations = int(loss_cfg.get("skeleton_iterations", 10))

    def _region_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        dice = weighted_dice_loss(
            probabilities,
            target,
            self.positive_weight,
            self.negative_weight,
        )
        bce = weighted_bce_loss(
            logits,
            target,
            self.positive_weight,
            self.negative_weight,
        )
        return self.dice_weight * dice + self.bce_weight * bce

    def forward(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        mask_loss = self._region_loss(outputs["mask_logits"], batch["mask"])
        centerline_loss = self._region_loss(
            outputs["centerline_logits"], batch["centerline"]
        )
        radius_error = F.smooth_l1_loss(
            outputs["radius"],
            batch["radius"],
            reduction="none",
        )
        center = batch["centerline"]
        radius_loss = (radius_error * center).sum() / (center.sum() + 1e-6)
        if self.use_topology:
            topology_loss = 1.0 - soft_cldice(
                outputs["mask"],
                batch["mask"],
                iterations=self.skeleton_iterations,
            )
        else:
            topology_loss = outputs["mask_logits"].sum() * 0.0
        if not self.use_centerline:
            centerline_loss = outputs["centerline_logits"].sum() * 0.0
        if not self.use_radius:
            radius_loss = outputs["radius_logits"].sum() * 0.0
        total = (
            mask_loss
            + self.centerline_weight * centerline_loss
            + self.radius_weight * radius_loss
            + self.topology_weight * topology_loss
        )
        return {
            "total": total,
            "mask": mask_loss,
            "centerline": centerline_loss,
            "radius": radius_loss,
            "topology": topology_loss,
        }
