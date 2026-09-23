from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from wtg_unet.config import load_config
from wtg_unet.data import build_dataloaders
from wtg_unet.losses import MultitaskLoss
from wtg_unet.model import build_model
from wtg_unet.training import WarmupPlateau, evaluate, train_one_epoch
from wtg_unet.utils import count_parameters, resolve_device, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WTG-U-Net on FIVES.")
    parser.add_argument("--config", default="configs/fives.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default="runs/wtg-unet")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    return parser.parse_args()


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    keys = sorted({key for row in history for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def train_seed(
    config: dict[str, Any],
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    seed_everything(seed)
    loaders, pairs = build_dataloaders(
        config,
        data_root=args.data_root,
        seed=int(config.get("data", {}).get("split_seed", 2026)),
        augmentation_seed=seed,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        limit_test=args.limit_test,
    )
    model = build_model(config).to(device)
    criterion = MultitaskLoss(config)
    optim_cfg = config.get("optim", {})
    learning_rate = float(optim_cfg.get("learning_rate", 1e-4))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
    )
    scheduler = WarmupPlateau(
        optimizer,
        base_lr=learning_rate,
        warmup_epochs=int(optim_cfg.get("warmup_epochs", 5)),
        start_factor=float(optim_cfg.get("warmup_start_factor", 0.2)),
        patience=int(optim_cfg.get("plateau_patience", 10)),
        factor=float(optim_cfg.get("plateau_factor", 0.5)),
        min_lr=float(optim_cfg.get("min_learning_rate", 1e-8)),
    )
    epochs = int(args.epochs or optim_cfg.get("epochs", 25))
    threshold = float(config.get("evaluation", {}).get("threshold", 0.5))
    skeleton_iterations = int(config.get("loss", {}).get("skeleton_iterations", 10))
    amp = bool(optim_cfg.get("amp", False))
    run_dir = Path(args.output_dir) / f"seed-{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        run_dir / "split.json",
        {
            split: [
                {"image": str(pair.image), "mask": str(pair.mask)}
                for pair in split_pairs
            ]
            for split, split_pairs in pairs.items()
        },
    )

    best_dice = float("-inf")
    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, amp=amp, epoch=epoch
        )
        val_metrics = evaluate(
            model,
            loaders["val"],
            criterion,
            device,
            threshold=threshold,
            skeleton_iterations=skeleton_iterations,
        )
        scheduler.step(float(val_metrics["dice"]))
        row: dict[str, Any] = {"epoch": epoch + 1, "lr": scheduler.learning_rate}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        print(
            f"seed={seed} epoch={epoch + 1:02d}/{epochs} "
            f"train_loss={train_metrics['total']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} lr={scheduler.learning_rate:.2e}"
        )
        if float(val_metrics["dice"]) > best_dice:
            best_dice = float(val_metrics["dice"])
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "seed": seed,
                    "epoch": epoch + 1,
                    "val_metrics": val_metrics,
                },
                run_dir / "best.pt",
            )
    write_history(run_dir / "history.csv", history)
    save_json(run_dir / "final.json", {"seed": seed, "best_val_dice": best_dice})
    return {"seed": seed, "best_val_dice": best_dice, "parameters": count_parameters(model)}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seeds = args.seeds or [int(value) for value in config.get("optim", {}).get("seeds", [1, 2, 3])]
    device = resolve_device(args.device)
    print(f"device={device}")
    summaries = [train_seed(config, seed, args, device) for seed in seeds]
    save_json(Path(args.output_dir) / "summary.json", {"device": str(device), "runs": summaries})


if __name__ == "__main__":
    main()
