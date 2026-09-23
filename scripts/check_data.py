from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wtg_unet.data import discover_pairs, resolve_data_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Check FIVES image/mask pairing.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-images", default=None)
    parser.add_argument("--train-masks", default=None)
    parser.add_argument("--test-images", default=None)
    parser.add_argument("--test-masks", default=None)
    args = parser.parse_args()
    directories = resolve_data_dirs(
        args.data_root,
        args.train_images,
        args.train_masks,
        args.test_images,
        args.test_masks,
    )
    development = discover_pairs(directories["train_images"], directories["train_masks"])
    test = discover_pairs(directories["test_images"], directories["test_masks"])
    print("Resolved directories:")
    for key, value in directories.items():
        print(f"  {key}: {value}")
    print(f"Development pairs: {len(development)} (expected 600)")
    print(f"Test pairs:        {len(test)} (expected 200)")
    if len(development) != 600 or len(test) != 200:
        raise SystemExit("Pair counts differ from the report protocol.")


if __name__ == "__main__":
    main()

