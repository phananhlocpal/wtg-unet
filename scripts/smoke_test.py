from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wtg_unet.config import load_config
from wtg_unet.losses import MultitaskLoss
from wtg_unet.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a forward/loss smoke test.")
    parser.add_argument("--config", default=str(ROOT / "configs/fives.yaml"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()
    config = load_config(args.config)
    model = build_model(config).to(args.device)
    model.eval()
    image = torch.randn(1, 1, args.size, args.size, device=args.device)
    target = torch.zeros_like(image)
    batch = {
        "image": image,
        "mask": target,
        "centerline": target,
        "radius": target,
        "radius_px": target,
    }
    with torch.no_grad():
        outputs = model(image)
    criterion = MultitaskLoss(config)
    losses = criterion(outputs, batch)
    expected = (1, 1, args.size, args.size)
    for name in ("mask", "centerline", "radius"):
        if tuple(outputs[name].shape) != expected:
            raise RuntimeError(f"{name} shape is {tuple(outputs[name].shape)}, expected {expected}")
    print(f"forward and loss smoke test passed; total={float(losses['total']):.4f}")


if __name__ == "__main__":
    main()

