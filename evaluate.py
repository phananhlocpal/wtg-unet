from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from wtg_unet.config import load_config
from wtg_unet.data import build_dataloaders
from wtg_unet.losses import MultitaskLoss
from wtg_unet.model import build_model
from wtg_unet.training import evaluate
from wtg_unet.utils import resolve_device, save_json
from wtg_unet.visualize import save_confusion_plot, save_example_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a WTG-U-Net checkpoint.")
    parser.add_argument("--config", default="configs/fives.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="runs/evaluation")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = checkpoint.get("config", config)
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    loaders, _ = build_dataloaders(config, data_root=args.data_root)
    criterion = MultitaskLoss(config)
    eval_cfg = config.get("evaluation", {})
    metrics = evaluate(
        model,
        loaders["test"],
        criterion,
        device,
        threshold=float(eval_cfg.get("threshold", 0.5)),
        skeleton_iterations=int(config.get("loss", {}).get("skeleton_iterations", 10)),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "metrics.json", metrics)
    save_confusion_plot(metrics, output_dir / "confusion-matrix.png")
    save_example_grid(
        model,
        loaders["test"],
        device,
        output_dir / "examples",
        threshold=float(eval_cfg.get("threshold", 0.5)),
        limit=int(eval_cfg.get("save_examples", 4)),
    )
    print(f"saved evaluation to {output_dir}")


if __name__ == "__main__":
    main()

