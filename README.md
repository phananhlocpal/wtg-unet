# WTG-U-Net reproducibility code

This directory contains an executable PyTorch implementation of the WTG-U-Net
protocol described in the report for retinal vessel segmentation on FIVES.
The code is deliberately separated from the LaTeX report and expects the FIVES
images and masks to be supplied locally.

## What is implemented

- deterministic 480/120 development split from the 600 FIVES development pairs;
- independent 200-image test split;
- 512 x 512 preprocessing, green-channel extraction, CLAHE (clip limit 2.0,
  8 x 8 tiles), Gaussian smoothing (sigma 0.5), min-max and per-image
  z-score normalization;
- training augmentation with rotation, horizontal/vertical flips, brightness
  scaling, and Gaussian noise;
- four-level U-Net with 64/128/256/512 encoder channels, a 1024-channel
  bottleneck, residual blocks, four WTG gates, and mask/centerline/radius
  outputs;
- weighted Dice plus weighted BCE for the mask and centerline, SmoothL1 radius
  loss, soft clDice topology loss, Adam, five warm-up epochs, plateau
  reduction, best validation Dice checkpointing, and seed-wise runs;
- test metrics, confusion counts, error maps, prediction overlays, and a
  machine-readable metrics file.

The report does not specify the numerical class weights, the exact soft
skeleton iteration count, or the radius normalization constant used by the
original run. These are explicit configuration fields in
configs/fives.yaml; the defaults are conservative and must be replaced if
the original experiment log provides different values.

## Installation

From this directory:

    python -m venv .venv
    . .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .

The code requires Python 3.10 or newer and a PyTorch build appropriate for the
available CPU or CUDA device.

## Expected data layout

The loader accepts either explicit directories in the YAML file or common
FIVES layouts. The safest layout is:

    /data/FIVES/
      train/images/
      train/masks/
      test/images/
      test/masks/

train must contain the 600 development pairs and test the 200 independent
pairs. The image and mask file names should share a stem. The loader also
recognizes Original and Ground truth directories under train, test,
1_Training_Set, and 2_Testing_Set.

## Check the data and run training

    cd wtg-unet
    python scripts/check_data.py --data-root /data/FIVES
    python train.py --config configs/fives.yaml \
      --data-root /data/FIVES --output-dir runs/wtg-unet

The default configuration runs seeds 1, 2, and 3. For a quick protocol check,
override the seed list and epoch count:

    python train.py --config configs/fives.yaml \
      --data-root /data/FIVES --output-dir runs/smoke \
      --seeds 1 --epochs 1 --limit-train 8 --limit-val 4

This command is only a smoke run; it is not the report experiment.

The ablation table can be reproduced with the supplied variant definitions:

    python run_ablation.py --data-root /data/FIVES \
      --output-dir runs/ablation --seeds 1 2 3

## Evaluate and infer

    python evaluate.py --config configs/fives.yaml \
      --data-root /data/FIVES \
      --checkpoint runs/wtg-unet/seed-1/best.pt \
      --output-dir runs/wtg-unet/seed-1/test

    python infer.py --config configs/fives.yaml \
      --checkpoint runs/wtg-unet/seed-1/best.pt \
      --image /data/FIVES/test/images/example.png \
      --output runs/wtg-unet/example_mask.png

## Reproducibility notes

All random generators used by the loader and training loop are seeded. The
test set is never used for scheduling or checkpoint selection. Augmentation is
only applied to the training split. Structural labels are generated from the
mask with skeletonization and a distance transform at load time.

The final centerline and radius heads are attached to the full-resolution
decoder feature. Each intermediate WTG gate uses lightweight auxiliary
structure projections from its current decoder feature, which makes the
structure-conditioned skip path well-defined in a single forward pass.
