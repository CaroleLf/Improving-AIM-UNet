# Notes de référence — essai précédent (archivé)

Contenu extrait de l'essai antérieur (`archive/legacy_mac_attempt/` à la racine de `Vietnam/`) avant archivage, pour ne pas perdre l'info. Ce n'est PAS du code réutilisé tel quel dans l'étape 1 — juste des notes de référence pour les étapes suivantes.

## Référence papier

> Mai et al., "AIM-UNet: Adaptive Inception Vision Mamba U-Net with Selective Kernel Gating for Breast Ultrasound Tumor Segmentation", 2026.

## Baselines publiées (BUSI) — table du papier

| Model       | Dice   | IoU    |
|-------------|--------|--------|
| U-Net       | 0.742  | 0.628  |
| ViM-UNet    | 0.783  | 0.673  |
| MSVM-UNet   | 0.801  | 0.694  |
| AIM-UNet*   | 0.847  | 0.743  |

\* Valeurs rapportées par Mai et al., 2026. À utiliser comme référence dans le tableau de comparaison final baseline vs Solution A.

## Ébauche de boundary-aware loss (essai précédent, pour l'étape Solution A — PAS l'étape 1)

Extraction de la bande de contour, approche dilatation morphologique via `max_pool2d` (pas besoin de scipy, portable) :

```python
def _boundary_dice_loss(self, pred, target):
    dilated = F.max_pool2d(target, kernel_size=5, stride=1, padding=2)
    boundary_mask = (dilated - target).clamp(0.0, 1.0)
    pred_boundary = pred * boundary_mask
    target_boundary = target * boundary_mask
    if boundary_mask.sum() < 1.0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    return _soft_dice_loss(pred_boundary, target_boundary)
```

C'est une variante concrète de l'option "boundary-weighted band + Dice restreinte" discutée en détail par ailleurs (bande de tolérance autour du contour GT, plutôt qu'un contour 1px exact). Point commun avec ce qu'on a conçu : même motivation (précision de contour), même famille de technique (bande morphologique). Différence : ici Dice restreinte à la bande, alors qu'on avait aussi évoqué une variante edge-map (Sobel) + Dice/L1 entre cartes de contour continues — à trancher au moment de concevoir Solution A pour de vrai.

**Point d'attention** : l'implémentation `DESLLoss` de cet essai (`_diversity_loss` = cosinus brut sur branches aplaties, sans pooling global ni terme de variance) NE correspond PAS à la formule DESL réelle du papier (section 3.4, confirmée par l'utilisatrice : λ_cos·L_cos + λ_var·L_var sur des embeddings globalement pooled). Ne pas réutiliser cette implémentation de DESL comme référence à l'étape 3 — repartir de la formule exacte du papier.

## Script de téléchargement dataset (récupéré, adapté dans `scripts/download_dataset.py`)

Dataset : `aryashah2k/breast-ultrasound-images-dataset` sur Kaggle, via `kagglehub`.
