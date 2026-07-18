# Improving-AIM-UNet

PyTorch implementation of AIM-UNet, a Mamba-based U-Net architecture for breast tumor segmentation in ultrasound images.

Reproduction of:
> Mai et al., "AIM-UNet: Adaptive Inception Vision Mamba U-Net with Selective Kernel Gating for Breast Ultrasound Tumor Segmentation", 2026.

Progression en 5 étapes : U-Net baseline → blocs VSS/Mamba → module AIM → loss DESL → entraînement complet (hyperparamètres papier). Puis amélioration proposée : loss boundary-aware (Solution A), comparée à la baseline sur BUSI et Dataset B2.

Entraînement réel exécuté sur machine GPU distante (SSH, CUDA, RTX 3090). Ce dépôt est développé en local puis transféré manuellement.

## Statut

Étape 1 terminée : U-Net baseline sur BUSI — Test Dice 0.7710, IoU 0.6747 (papier, Table 1 : U-Net 0.7010 / 0.6070). Détails dans [docs/results.md](docs/results.md). Étape 2 (blocs VSS/Mamba) à venir.

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

## Baselines publiées (BUSI, papier — Table 1)

| Model       | Dice   | IoU    |
|-------------|--------|--------|
| U-Net       | 70.10  | 60.70  |
| ViM-UNet    | 76.92  | 66.04  |
| MSVM-UNet   | 79.33  | 70.37  |
| AIM-UNet    | 82.44  | 72.63  |

Voir [docs/results.md](docs/results.md) pour la table complète (Table 1) et le suivi des résultats de reproduction, et [docs/legacy_reference.md](docs/legacy_reference.md) pour des notes complémentaires (ébauche de boundary-aware loss pour l'étape Solution A).
