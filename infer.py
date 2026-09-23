from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from wtg_unet.config import load_config
from wtg_unet.model import build_model
from wtg_unet.preprocess import preprocess_image, read_rgb
from wtg_unet.utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WTG-U-Net on one fundus image.")
    parser.add_argument("--config", default="configs/fives.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = checkpoint.get("config", config)
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    image_size = tuple(int(v) for v in config.get("data", {}).get("image_size", [512, 512]))
    image = preprocess_image(read_rgb(args.image), image_size)
    tensor = torch.from_numpy(image[None, None]).to(device)
    with torch.no_grad():
        output = model(tensor)
    probability = output["mask"][0, 0].cpu().numpy()
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(config.get("evaluation", {}).get("threshold", 0.5))
    )
    binary = (probability >= threshold).astype(np.uint8) * 255
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), binary):
        raise RuntimeError(f"Could not write prediction to {target}")
    np.save(target.with_suffix(".npy"), probability)
    print(f"saved mask to {target}")


if __name__ == "__main__":
    main()

