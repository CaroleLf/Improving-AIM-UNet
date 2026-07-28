# Improving-AIM-UNet

PyTorch implementation of AIM-UNet, a Mamba-based U-Net architecture for breast tumor segmentation in ultrasound images.

Reproduction of:
> Mai et al., "AIM-UNet: Adaptive Inception Vision Mamba U-Net with Selective Kernel Gating for Breast Ultrasound Tumor Segmentation", 2026.

Progression in 5 steps: U-Net baseline, then VSS/Mamba blocks, then the AIM module, then the DESL loss, then full training with the paper's hyperparameters. Followed by a proposed improvement: a boundary-aware loss term (Solution A), compared against the reproduced baseline on BUSI and Dataset B2.

Real training runs on a remote GPU machine (SSH, CUDA, RTX 3090). This repo is developed locally, then transferred manually.

## Status

All 5 reproduction steps done, plus Solution A. Best results on BUSI so far:

| Model | Loss | Test Dice | Test IoU | Precision | Recall | Boundary Dice |
|-------|------|-----------|----------|-----------|--------|----------------|
| U-Net baseline | BCE+Dice | 0.7710 | 0.6747 | 0.8406 | 0.7596 | n/a |
| AIM-UNet | DESL | 0.7712 | 0.6869 | 0.8284 | 0.7778 | 0.5747 |
| AIM-UNet | DESL + Solution A (lambda_bdry=0.2) | 0.7722 | 0.6858 | 0.8018 | 0.7832 | 0.5823 |

Full history, hyperparameter sweeps, and known open questions (unconfirmed N per stage and base channel width, a sign error found in the paper's eq. 10, a structural difference from the official VMamba SS2D) are tracked in [docs/results.md](docs/results.md).

## Dataset

BUSI (Breast Ultrasound Images), Al-Dhabyani et al. 2020:

```bash
python scripts/download_dataset.py
```

Expected structure once unzipped:

```
Dataset_BUSI_with_GT/
    benign/
        benign (1).png
        benign (1)_mask.png
        ...
    malignant/
        malignant (1).png
        malignant (1)_mask.png
        ...
    normal/        # excluded from train/val/test (no lesion)
```

## Setup

```bash
pip install -r requirements.txt
```

## Published baselines (BUSI, paper Table 1)

| Model       | Dice   | IoU    |
|-------------|--------|--------|
| U-Net       | 70.10  | 60.70  |
| ViM-UNet    | 76.92  | 66.04  |
| MSVM-UNet   | 79.33  | 70.37  |
| AIM-UNet    | 82.44  | 72.63  |

See [docs/results.md](docs/results.md) for the full table and the reproduction results log, and [docs/legacy_reference.md](docs/legacy_reference.md) for supporting notes (an earlier boundary-aware loss draft that fed into Solution A's design).
