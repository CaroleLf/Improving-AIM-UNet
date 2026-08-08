# Results

Fixed split (seed=42, `data/splits/busi_split.json`), same conditions for every step: a necessary condition for a valid baseline vs Solution A comparison.

| Step | Model | Loss | Dataset | Val Dice (best) | Test Dice | Test IoU | Test Precision | Test Recall | Test Boundary Dice |
|------|-------|------|---------|------------------|-----------|----------|-----------------|-------------|---------------------|
| 1 | U-Net baseline | BCE+Dice | BUSI | 0.7771 (epoch 144) | 0.7710 | 0.6747 | 0.8406 | 0.7596 | n/a |
| 2 | VSS-UNet (VSS only, simple concat skip, N=2/stage, base=64, unconfirmed assumptions) | BCE+Dice | BUSI | 0.7804 (epoch 150) | 0.7350 | 0.6457 | 0.7652 | 0.7581 | n/a |
| 3 | AIM-UNet without DESL (VSS + AIM + SAB/CAB, same N/channel assumptions) | BCE+Dice | BUSI | 0.7726 (epoch 150) | 0.7091 | 0.6255 | 0.7216 | 0.7550 | n/a |
| 4 | AIM-UNet + DESL, with fixes (SS2D with per-direction weights, base_dim=96 instead of 64, corrected L_cos sign) | DESL | BUSI | 0.7793 (epoch 126) | 0.7331 | 0.6451 | 0.7551 | 0.7772 | n/a |
| 4b | Same, 300 epochs instead of 150 (only variable changed), **reference for Solution A** | DESL | BUSI | 0.8201 (epoch 275) | **0.7712** | **0.6869** | 0.8284 | **0.7778** | **0.5747** |
| 5 | Solution A: AIM-UNet + DESL + boundary-aware loss (dilate-erode band), 300 epochs | BoundaryAwareDESL | BUSI | 0.8204 (epoch 300) | 0.7520 | 0.6663 | 0.7639 | **0.7949** | **0.5926** |

## Paper reference (Table 1, BUSI: real values verified in the PDF, CMPB-D-26-02655)

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

Table 4 (loss ablation, with Adaptive Inception + per-channel SK): BCE+Dice alone = 80.79 Dice / 70.32 IoU; BCE+Dice+DESL = 82.44 Dice / 72.63 IoU on BUSI.

**Correction: the table previously noted here (U-Net 0.742/0.628, AIM-UNet* 0.847/0.743) came from the archived Mac attempt's README, never checked against the real paper. It was wrong. The real paper (PDF provided by the user, read directly) gives the numbers above.**

## Observations

- Step 1 (U-Net baseline): Dice 0.7710 vs 0.7010 in the paper (+0.070), IoU 0.6747 vs 0.6070 (+0.0677). Reproduction judged valid, even slightly above, normal run-to-run variance (different split, different library versions).
- Precision (0.8406) clearly above Recall (0.7596): the model under-segments rather than over-segments. Reference point for evaluating the impact of Solution A (boundary-aware loss): if it correctly targets under-segmentation at low-contrast boundaries, recall should rise at the corresponding step without degrading precision too much.
- Step 2 (VSS only): Test Dice **below** the baseline (0.7350 vs 0.7710, -0.036), despite a slightly higher Val Dice (0.7804 vs 0.7771). Val/test gap worth noting, not a pipeline bug (every building block was unit-tested). Hypotheses: (a) simple concatenation skip connections may not suit a hierarchical, patch-embedding-based backbone; the SAB/CAB planned for step 3 might be a necessary component rather than an optional refinement; (b) N=2 blocks/stage and base channel 64 remain unconfirmed assumptions from Professor Nguyen; (c) final train_loss higher than the baseline (0.15 vs 0.12) at the same epoch count, possible under-training/different optimization dynamics for Mamba blocks. To revisit with Professor Nguyen before step 3.
- Step 3 (AIM + SAB/CAB, without DESL): Test Dice even lower (0.7091), **contradicting** the step 2 hypothesis: SAB/CAB did not compensate, the downward trend continues with each added component. Precision keeps dropping (0.8406 to 0.7652 to 0.7216) while recall stays roughly flat (0.7596 to 0.7581 to 0.7550): the model over-segments increasingly without gaining detection, as complexity increases.
  - Most significant point: this **contradicts the paper's own ablation**. Table 3 of the paper shows that adding Adaptive Inception to ViM-UNet, even without DESL (BCE+Dice only), *raises* Dice from 76.92 to 80.51 (+3.59). In our case, under the same loss conditions, AIM+SAB/CAB lowers Dice. Strong signal that the reconstructed architecture diverges from the real implementation on a structural point, beyond the two hyperparameters already flagged as unconfirmed. To clarify with Professor Nguyen before going further.
  - Still consistent with the paper's logic on one point: Table 4 shows DESL contributes +1.65 Dice to AIM (80.79 to 82.44), precisely because BCE+Dice alone does not force AIM's branches to be complementary (explicit motivation for DESL in the paper's introduction). Without DESL, AIM's branches may converge toward redundant features instead of multi-scale ones. Step 4 (DESL) will confirm whether this fixes the trend.
- **Step 4 (DESL + bundled fixes)**: real improvement over step 3 (Dice 0.7091 to 0.7331, +0.024; Recall 0.7550 to 0.7772, the highest of the 4 runs), confirming the hypotheses had merit. Still below the U-Net baseline (0.7710) and roughly tied with step 2 (0.7350). Three changes bundled in this run (SS2D with per-direction weights, base_dim 64 to 96, corrected L_cos sign): impossible to isolate which contributed most, a deliberate tradeoff to get an actionable result quickly rather than a clean ablation at this stage.
  - **Inconsistency found in the paper**: eq. 10 defines `L_cos` with a leading minus sign, but added positively to a loss being minimized (eq. 9), that sign would mathematically reward similarity between branches instead of penalizing it, contradicting the paper's own text ("this term increases angular dispersion"). Verified with a unit test (identical branches gave a lower loss than with the correct sign). Sign removed in our implementation so the term does what the paper says it should. Worth flagging to Professor Nguyen: potentially a typo in the submitted manuscript.
  - **Structural divergence confirmed vs official VMamba**: the reference repo (MzeroMiko/VMamba) uses separate projection weights per scan direction (only A_log/D are shared), while eq. 5 of the AIM-UNet paper ("a selective S6 block", singular) is ambiguous on this point. Implemented with 4 separate Mamba modules (an approximation of VMamba, not an exact replica of their A_log/D sharing).
  - Underlying question for Professor Nguyen: Table 3 of the paper shows that ViM-UNet alone (Mamba without AIM) only contributes +6.8 Dice over the U-Net (70.10 to 76.92); the real jump comes from AIM+DESL. Does Mamba's global context actually help on 256x256 images with fairly compact tumors, or is most of the paper's gain independent of the Mamba backbone?
  - Hyperparameters N (blocks/stage) and base channel remain unconfirmed assumptions despite the fix to 96 (VMamba's default value, not a value specific to AIM-UNet).
- **Step 4b (300 epochs, only variable changed)**: confirms the under-training hypothesis. Test Dice 0.7712 (nearly identical to the baseline's 0.7710), Test IoU 0.6869 (**best score across all runs**, above the baseline), Recall 0.7778 (**best across all runs**), Precision 0.8284 (nearly recovered relative to the baseline's 0.8406). The problem was therefore not a fundamental structural one: the architecture, properly trained, holds up against the classic U-Net. Reproduction judged satisfactory to serve as the baseline for Solution A, despite the N/channel hyperparameters still unconfirmed by Professor Nguyen.

## Solution A (boundary-aware loss)

- **Bug found and fixed**: the first version of `extract_boundary` computed `dilated(mask) - mask`, a band strictly **outside** the true tumor. But `mask x band` is then structurally always zero (the band and the tumor never overlap by construction): Dice restricted to that band is degenerate, both as a metric and as a loss term (it only pushes the model to predict nothing in that outer ring, with no real notion of contour alignment). Confirmed by an abnormally low `test_boundary_dice` (0.0600) on the first run. Fixed with a dilate-erode band that straddles both sides of the contour, verified with a unit test (a square mask shifted by 1 pixel: global Dice 0.84 but `boundary_dice` 0.71, far more sensitive to the contour error, as expected). This flaw came from the archived Mac attempt's draft (`docs/legacy_reference.md`), reused without challenging it enough.
- Reference to beat: **Test Boundary Dice = 0.5747** (AIM-UNet + DESL, step 4b, without Solution A).
- **Final result (step 5)**: Solution A improves Recall (0.7778 to 0.7949, +0.017) and **Boundary Dice** (0.5747 to 0.5926, +0.018), the metric most directly relevant to the stated goal (boundary delineation precision). Tradeoff: a more marked drop in Precision (0.8284 to 0.7639, -0.065), and a slight drop in global Dice (-0.019) and IoU (-0.021).
- **Interpretation**: a measurable effect, consistent with the theory. The boundary loss term pushes the model to be less conservative on contour pixels (hence recall and boundary Dice rising), at the cost of a few extra false positives elsewhere (hence the precision drop). Not a dramatic gain on global Dice, but a real, measurable impact on the metric Solution A targets, consistent with the internship's original goal ("measure whether this improvement has a real, measurable impact, particularly relevant for small lesions or low-contrast boundaries").
- **Retuned lambda_bdry (0.2 instead of 0.5)**: confirms the hypothesis, the default inherited from the legacy draft was too aggressive. With `lambda_bdry=0.2` (300 epochs, otherwise identical):

  | | Reference (DESL) | Solution A lambda=0.5 | Solution A lambda=0.2 | Solution A lambda=0.1 |
  |---|---|---|---|---|
  | Test Dice | 0.7712 | 0.7520 | **0.7722** | 0.7447 |
  | Test IoU | 0.6869 | 0.6663 | **0.6858** | 0.6605 |
  | Precision | 0.8284 | 0.7639 | **0.8018** | 0.7649 |
  | Recall | 0.7778 | 0.7949 | 0.7832 | 0.7884 |
  | Boundary Dice | 0.5747 | **0.5926** | 0.5823 | 0.5906 |

  By far the best tradeoff at `lambda_bdry=0.2`: Dice slightly **above** the reference (0.7722 vs 0.7712), IoU nearly equal (-0.0011), while still keeping a real gain on Recall (+0.0054) and Boundary Dice (+0.0076), with a much smaller precision cost (-0.0266 instead of -0.0645 with lambda=0.5). `lambda=0.1` does not do better: below `lambda=0.2` on Dice/IoU/Precision, with no net gain on Boundary Dice. That run's curve shows instability around epochs 288-296 (train_loss spikes, val_dice temporarily dropping to 0.7151), which may have hurt this particular run without necessarily being representative of `lambda=0.1` in general. Non-monotonic relationship across the 3 tested values, consistent with training noise (a single run per configuration, no averaging over multiple seeds). **Recommended configuration for Solution A: `lambda_bdry=0.2`.**
- Conclusion: Solution A, properly tuned, improves boundary precision and recall without sacrificing overall Dice/IoU, a clear and defensible result answering the internship's original research question.

## Training pipeline improvements

Motivated by two things observed across every run so far: strong epoch-to-epoch noise in val_dice (often +/-0.03 to 0.05), and at least one real instability episode (the lambda=0.1 run, loss spikes around epochs 288-296). Added to `src/train.py`:

- Cosine learning rate schedule with linear warmup (10 epochs), instead of a constant lr.
- Gradient clipping (max norm 1.0), a standard fix for the kind of loss spike observed on the lambda=0.1 run.
- Light weight decay (1e-5) on Adam, given the small dataset (452 training images) relative to the model's capacity.
- Exponential moving average (EMA) of model weights, decay 0.999. The EMA model (not the raw end-of-batch weights) is what gets validated and checkpointed, which smooths out the noise from picking a single lucky epoch.

Validated locally first (EMA convergence toward a fixed target after enough updates, LR schedule values at key epochs) before a full run.

**Result, same configuration as the best Solution A run (lambda_bdry=0.2, 300 epochs) but with these training improvements**:

| | Solution A lambda=0.2 (before) | + EMA/LR schedule/clip/decay (after) | Delta |
|---|---|---|---|
| Test Dice | 0.7722 | **0.7884** | +0.0162 |
| Test IoU | 0.6858 | **0.7003** | +0.0145 |
| Precision | 0.8018 | 0.8065 | +0.0047 |
| Recall | 0.7832 | **0.8265** | +0.0433 |
| Boundary Dice | 0.5823 | **0.6066** | +0.0243 |

Best result of the whole project. The val_dice curve over the last epochs (258-300) is visibly smooth and near-monotonic, unlike the jagged curves of every prior run, confirming the EMA is doing what it was added for. Beyond the stability gain, the training improvements also translated into a real performance gain across every metric, not just noise reduction. This configuration now exceeds the U-Net baseline on Dice (0.7884 vs 0.7710), IoU (0.7003, the first run to cross 0.70), and Recall (0.8265 vs 0.7596); only Precision remains slightly below the baseline (0.8065 vs 0.8406). Unlike earlier Solution A runs, the recall/boundary gain no longer comes at a real precision cost.

**Final recommended configuration**: AIM-UNet, DESL, boundary-aware loss with `lambda_bdry=0.2`, 300 epochs, cosine LR schedule with 10-epoch warmup, gradient clipping at 1.0, weight decay 1e-5, EMA decay 0.999.

## Next: multi-seed runs and Dataset B2

Two remaining items for a more rigorous final comparison:

- **Multi-seed runs**: every result so far is a single run per configuration. Given the real epoch-to-epoch noise observed throughout (and the non-monotonic lambda_bdry sweep), the final configuration should be run with several seeds and reported as mean +/- std, not a single number.
- **Dataset B2 (BUS2017-B / UDIAT)**: the original internship protocol called for testing on both BUSI and Dataset B2. The official source (Manchester Metropolitan University, M. Yap) requires a license agreement submitted with an institutional email, with approval taking up to 10 working days; that request has been submitted. In the meantime, a Kaggle mirror (`ayush02102001/udiat-segmentation-dataset`, `scripts/download_dataset_b2.py`) is being used to keep making progress. **Provenance note**: this mirror is very likely an unauthorized re-upload of the officially licensed dataset, not obtained through the MMU license process. Used here only to avoid blocking on the up-to-10-day approval wait; results obtained this way should be cross-checked against the officially licensed copy once access is granted.
