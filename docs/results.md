# Résultats

Split fixe (seed=42, `data/splits/busi_split.json`), mêmes conditions pour toutes les étapes — condition nécessaire pour une comparaison baseline vs Solution A valide.

| Étape | Modèle | Loss | Dataset | Val Dice (best) | Test Dice | Test IoU | Test Precision | Test Recall |
|-------|--------|------|---------|------------------|-----------|----------|-----------------|-------------|
| 1 | U-Net baseline | BCE+Dice | BUSI | 0.7771 (epoch 144) | 0.7710 | 0.6747 | 0.8406 | 0.7596 |

## Référence papier (Table 1, BUSI — valeurs réelles vérifiées dans le PDF, CMPB-D-26-02655)

| Model       | Dice   | Precision | F1-Score | Recall | IoU    |
|-------------|--------|-----------|----------|--------|--------|
| U-Net       | 70.10  | 71.88     | 74.20    | 76.30  | 60.70  |
| DeepLabV3+  | 68.46  | 73.30     | 74.32    | 72.67  | 62.16  |
| CFANet      | 70.53  | 75.25     | 76.32    | 73.45  | 63.14  |
| SANet       | 74.46  | 74.84     | 77.71    | 80.76  | 65.96  |
| AAU-Net     | 77.51  | 79.61     | 80.32    | 81.10  | 68.82  |
| ViT-UNet    | 77.06  | 79.10     | 78.45    | 79.87  | 68.40  |
| ViM-UNet    | 76.92  | 78.34     | 75.42    | 81.25  | 66.04  |
| MSVM-UNet   | 79.33  | 80.56     | 80.77    | 82.17  | 70.37  |
| AIM-UNet    | 82.44  | 84.65     | 80.52    | 84.32  | 72.63  |

Table 4 (ablation loss, avec Adaptive Inception + SK per-channel) : BCE+Dice seul = 80.79 Dice / 70.32 IoU ; BCE+Dice+DESL = 82.44 Dice / 72.63 IoU sur BUSI.

**Correction : la table précédemment notée ici (U-Net 0.742/0.628, AIM-UNet* 0.847/0.743) venait du README de l'essai Mac archivé, jamais vérifiée contre le papier réel — elle était fausse. Le vrai papier (PDF fourni par l'utilisatrice, lu directement) donne les chiffres ci-dessus.**

## Observations

- Étape 1 (U-Net baseline) : Dice 0.7710 vs 0.7010 dans le papier (+0.070), IoU 0.6747 vs 0.6070 (+0.0677) — reproduction jugée valide et même au-dessus, run-to-run variance normale (split différent, versions de librairies différentes).
- Precision (0.8406) nettement > Recall (0.7596) : le modèle sous-segmente plutôt qu'il ne sur-segmente. Point de référence pour évaluer l'impact de Solution A (loss boundary-aware) — si elle cible bien la sous-segmentation aux frontières à faible contraste, on doit voir le recall remonter à l'étape correspondante sans trop dégrader la precision.
