# Improving-AIM-UNet

PyTorch implementation of AIM-UNet, a Mamba-based U-Net architecture for breast tumor segmentation in ultrasound images.

Reproduction of:
> Mai et al., "AIM-UNet: Adaptive Inception Vision Mamba U-Net with Selective Kernel Gating for Breast Ultrasound Tumor Segmentation", 2026.

Progression en 5 étapes : U-Net baseline → blocs VSS/Mamba → module AIM → loss DESL → entraînement complet (hyperparamètres papier). Puis amélioration proposée : loss boundary-aware (Solution A), comparée à la baseline sur BUSI et Dataset B2.

Entraînement réel exécuté sur machine GPU distante (SSH, CUDA, RTX 3090). Ce dépôt est développé en local puis transféré manuellement.

## Statut

Étape 1 en cours : U-Net baseline (pipeline de données + entraînement) sur BUSI.

## Dataset

BUSI (Breast Ultrasound Images), Al-Dhabyani et al. 2020 :

```bash
python scripts/download_dataset.py
```

Structure attendue une fois décompressé (à vérifier sur disque avant de faire confiance au Dataset — non encore vérifiée) :

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
    normal/        # exclu du train/val/test (pas de lésion)
```

## Setup

```bash
pip install -r requirements.txt
```

## Baselines publiées (BUSI, papier)

| Model       | Dice   | IoU    |
|-------------|--------|--------|
| U-Net       | 0.742  | 0.628  |
| ViM-UNet    | 0.783  | 0.673  |
| MSVM-UNet   | 0.801  | 0.694  |
| AIM-UNet*   | 0.847  | 0.743  |

\* Valeurs rapportées par Mai et al., 2026.

Voir [docs/legacy_reference.md](docs/legacy_reference.md) pour des notes de référence complémentaires (ébauche de boundary-aware loss pour l'étape Solution A).
