# WTG-U-Net reproducibility code

This directory contains an executable implementation of the WTG-U-Net
protocol for retinal vessel segmentation on FIVES.

## Installation

From this directory:

    python -m venv .venv
    . .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .

The code requires Python 3.10 or newer and a PyTorch build appropriate for the
available CPU or CUDA device.

## Evaluate and infer

    python evaluate.py --config configs/fives.yaml \
      --data-root /data/FIVES \
      --checkpoint runs/wtg-unet/seed-1/best.pt \
      --output-dir runs/wtg-unet/seed-1/test

    python infer.py --config configs/fives.yaml \
      --checkpoint runs/wtg-unet/seed-1/best.pt \
      --image /data/FIVES/test/images/example.png \
      --output runs/wtg-unet/example_mask.png