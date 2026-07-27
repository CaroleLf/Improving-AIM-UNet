# Résultats

Split fixe (seed=42, `data/splits/busi_split.json`), mêmes conditions pour toutes les étapes — condition nécessaire pour une comparaison baseline vs Solution A valide.

| Étape | Modèle | Loss | Dataset | Val Dice (best) | Test Dice | Test IoU | Test Precision | Test Recall |
|-------|--------|------|---------|------------------|-----------|----------|-----------------|-------------|
| 1 | U-Net baseline | BCE+Dice | BUSI | 0.7771 (epoch 144) | 0.7710 | 0.6747 | 0.8406 | 0.7596 |
| 2 | VSS-UNet (VSS seul, skip concat simple, N=2/étage, base=64 — hypothèses non confirmées) | BCE+Dice | BUSI | 0.7804 (epoch 150) | 0.7350 | 0.6457 | 0.7652 | 0.7581 |
| 3 | AIM-UNet sans DESL (VSS + AIM + SAB/CAB, mêmes hypothèses N/canal) | BCE+Dice | BUSI | 0.7726 (epoch 150) | 0.7091 | 0.6255 | 0.7216 | 0.7550 |

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
- Étape 2 (VSS seul) : Test Dice **inférieur** au baseline (0.7350 vs 0.7710, -0.036), malgré un Val Dice légèrement supérieur (0.7804 vs 0.7771) — écart val/test à noter, pas un bug de pipeline (chaque brique validée unitairement). Hypothèses : (a) la skip connection en concaténation simple n'est peut-être pas adaptée à un backbone hiérarchique à base de Patch Embedding — le SAB/CAB prévu à l'étape 3 pourrait être un composant nécessaire plutôt qu'un raffinement optionnel ; (b) N=2 blocs/étage et canal de base 64 restent des hypothèses non confirmées par le Professeur Nguyen ; (c) train_loss final plus haut que le baseline (0.15 vs 0.12) au même nombre d'epochs, possible sous-entraînement/optimisation différente pour les blocs Mamba. À rediscuter avec le Professeur Nguyen avant l'étape 3.
- Étape 3 (AIM + SAB/CAB, sans DESL) : Test Dice encore plus bas (0.7091), **contredit** l'hypothèse de l'étape 2 — SAB/CAB n'a pas compensé, la tendance continue de baisser à chaque composant ajouté. Precision continue de chuter (0.8406 → 0.7652 → 0.7216) alors que le recall reste quasi stable (0.7596 → 0.7581 → 0.7550) : le modèle sur-segmente de plus en plus sans gagner en détection, à mesure que la complexité augmente.
  - Point le plus significatif : ceci **contredit l'ablation du papier lui-même**. Table 3 du papier montre qu'ajouter l'Inception adaptative à ViM-UNet, même sans DESL (BCE+Dice seul), fait *monter* le Dice de 76.92 à 80.51 (+3.59). Chez nous, sous les mêmes conditions de loss, AIM+SAB/CAB fait baisser le Dice. Signal fort que l'architecture reconstruite diverge de l'implémentation réelle sur un point structurel (au-delà des deux hyperparamètres déjà identifiés comme non confirmés) — à clarifier avec le Professeur Nguyen avant d'aller plus loin.
  - Reste cohérent avec la logique du papier sur un point : Table 4 montre que DESL apporte +1.65 Dice à AIM (80.79 → 82.44) précisément parce que BCE+Dice seul ne force pas les branches d'AIM à être complémentaires (motivation explicite de DESL dans l'introduction du papier). Sans DESL, les branches d'AIM peuvent converger vers des features redondantes plutôt que multi-échelle — étape 4 (DESL) permettra de vérifier si ça corrige la tendance.
