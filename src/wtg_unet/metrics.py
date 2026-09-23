from __future__ import annotations

from typing import Any

import numpy as np
import torch
from skimage.morphology import skeletonize

from .losses import soft_cldice


def confusion_counts(prediction: np.ndarray, target: np.ndarray) -> dict[str, int]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    return {
        "tp": int(np.logical_and(prediction, target).sum()),
        "fp": int(np.logical_and(prediction, ~target).sum()),
        "fn": int(np.logical_and(~prediction, target).sum()),
        "tn": int(np.logical_and(~prediction, ~target).sum()),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def roc_auc(scores: np.ndarray, target: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=bool).ravel()
    positives = int(target.sum())
    negatives = int((~target).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_target = target[order].astype(np.int64)
    tpr = np.cumsum(sorted_target) / positives
    fpr = np.cumsum(1 - sorted_target) / negatives
    return float(np.trapz(np.r_[0.0, tpr], np.r_[0.0, fpr]))


def branch_break_rate(prediction: np.ndarray, target: np.ndarray) -> float:
    """Approximate broken-branch rate from missing 8-neighbour skeleton edges."""
    target_skeleton = skeletonize(target.astype(bool))
    prediction_skeleton = skeletonize(prediction.astype(bool))
    total_edges = 0
    covered_edges = 0
    for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1)):
        source = target_skeleton[max(0, dy) : target_skeleton.shape[0] + min(0, dy),
                                 max(0, dx) : target_skeleton.shape[1] + min(0, dx)]
        shifted_target = target_skeleton[max(0, -dy) : target_skeleton.shape[0] - max(0, dy),
                                         max(0, -dx) : target_skeleton.shape[1] - max(0, dx)]
        source_prediction = prediction_skeleton[max(0, dy) : target_skeleton.shape[0] + min(0, dy),
                                                max(0, dx) : target_skeleton.shape[1] + min(0, dx)]
        shifted_prediction = prediction_skeleton[max(0, -dy) : target_skeleton.shape[0] - max(0, dy),
                                                 max(0, -dx) : target_skeleton.shape[1] - max(0, dx)]
        edge = source & shifted_target
        total_edges += int(edge.sum())
        covered_edges += int((edge & source_prediction & shifted_prediction).sum())
    return float("nan") if total_edges == 0 else float(1.0 - covered_edges / total_edges)


def single_sample_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    centerline: np.ndarray,
    radius: np.ndarray,
    radius_px: np.ndarray | None,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    prediction = probability >= threshold
    counts = confusion_counts(prediction, target >= 0.5)
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    dice = _safe_ratio(2 * tp, 2 * tp + fp + fn)
    iou = _safe_ratio(tp, tp + fp + fn)
    precision = _safe_ratio(tp, tp + fp)
    sensitivity = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    thin = (radius_px > 0) & (radius_px <= 2.0) if radius_px is not None else centerline.astype(bool)
    thin_recall = _safe_ratio(float((prediction & thin).sum()), float(thin.sum()))
    valid_radius = centerline > 0.5
    return {
        **counts,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "roc_auc": roc_auc(probability, target >= 0.5),
        "thin_recall": thin_recall,
        "branch_break_rate": branch_break_rate(prediction, target >= 0.5),
        "radius_mae": float("nan"),
    }


def metrics_from_batch(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    threshold: float = 0.5,
    skeleton_iterations: int = 10,
) -> dict[str, float]:
    probabilities = outputs["mask"].detach().cpu().numpy()[:, 0]
    predicted_radius = outputs["radius"].detach().cpu().numpy()[:, 0]
    targets = batch["mask"].detach().cpu().numpy()[:, 0]
    centerlines = batch["centerline"].detach().cpu().numpy()[:, 0]
    radii = batch["radius"].detach().cpu().numpy()[:, 0]
    radii_px = batch.get("radius_px")
    radii_px_array = None if radii_px is None else radii_px.detach().cpu().numpy()[:, 0]
    rows: list[dict[str, float | int]] = []
    for idx in range(probabilities.shape[0]):
        row = single_sample_metrics(
            probabilities[idx],
            targets[idx],
            centerlines[idx],
            radii[idx],
            None if radii_px_array is None else radii_px_array[idx],
            threshold,
        )
        valid = centerlines[idx] > 0.5
        row["radius_mae"] = (
            float(np.abs(predicted_radius[idx][valid] - radii[idx][valid]).mean())
            if valid.any()
            else float("nan")
        )
        rows.append(row)

    keys = ("dice", "iou", "precision", "sensitivity", "specificity",
            "roc_auc", "thin_recall", "branch_break_rate", "radius_mae")
    result = {key: float(np.nanmean([float(row[key]) for row in rows])) for key in keys}
    for key in ("tp", "fp", "fn", "tn"):
        result[key] = float(sum(int(row[key]) for row in rows))
    result["cldice"] = float(
        soft_cldice(
            outputs["mask"],
            batch["mask"].to(outputs["mask"].device),
            iterations=skeleton_iterations,
        ).detach().cpu()
    )
    return result


def merge_metric_sums(
    total: dict[str, float], current: dict[str, float], weight: int
) -> dict[str, float]:
    """Accumulate sample metrics while preserving confusion-count totals."""
    if not total:
        total.update({key: 0.0 for key in current})
        total["_n"] = 0.0
    for key, value in current.items():
        if key in {"tp", "fp", "fn", "tn"}:
            total[key] += value
        elif np.isfinite(value):
            total[key] += value * weight
    total["_n"] += weight
    return total


def finalize_metric_sums(total: dict[str, float]) -> dict[str, float]:
    count = max(total.pop("_n", 1.0), 1.0)
    result = dict(total)
    for key in result:
        if key not in {"tp", "fp", "fn", "tn"}:
            result[key] /= count
    return result
