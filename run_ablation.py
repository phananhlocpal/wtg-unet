from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from train import train_seed
from wtg_unet.config import deep_update, load_config
from wtg_unet.utils import resolve_device, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the report's WTG ablation variants.")
    parser.add_argument("--config", default="configs/ablation.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="runs/ablation")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    args = parser.parse_args()

    ablation = load_config(args.config)
    base_path = Path(str(ablation.get("base_config", "configs/fives.yaml")))
    if not base_path.is_absolute():
        candidate = Path(args.config).parent / base_path
        base_path = candidate if candidate.exists() else ROOT / base_path
    base = load_config(base_path)
    device = resolve_device(args.device)
    summaries = []
    for variant in ablation.get("variants", []):
        name = str(variant["name"])
        config = deep_update(base, variant.get("overrides", {}))
        variant_dir = Path(args.output_dir) / name
        train_args = SimpleNamespace(
            data_root=args.data_root,
            output_dir=str(variant_dir),
            epochs=args.epochs,
            limit_train=args.limit_train,
            limit_val=args.limit_val,
            limit_test=args.limit_test,
        )
        for seed in args.seeds:
            summary = train_seed(config, int(seed), train_args, device)
            summary["variant"] = name
            summaries.append(summary)
    save_json(Path(args.output_dir) / "summary.json", {"runs": summaries})


if __name__ == "__main__":
    main()
