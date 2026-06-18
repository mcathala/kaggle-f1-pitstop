# Experiment 037 — Quantile-bucketed blend weights

**Cycle.** 11
**Status.** Inconclusive (Reverted) — quantile-bucketed weights overfit folds; uniform `w_cb=0.20` remains optimal.
**Date.** 2026-05-25

## Hypothesis

Bucketing rows by RealMLP-prediction quantile (5 buckets: very-low / low / mid / high / very-high) and fitting a separate `w_cb` per bucket via nested CV produces an OOF lift ≥ +0.00010 over cycle 7's uniform `w_cb=0.20` blend.

## Rationale

Two pieces of cycle 10 evidence:

1. **Probe 4** found rank disagreement (top-5% RealMLP↔CB difference) concentrates on **low-prob rows** — those rows had Position_Change very negative (gaining positions) with only 2.53% positives vs 19.9% overall.
2. **Probe 5** found cycle-7 blend over-predicts in mid-probability bins (7-9), bias +0.029 to +0.057. Calibration error is quantile-correlated, not feature-correlated.

Together they suggest: **the optimal mix of RealMLP and CB is a function of the predicted probability**, not a function of raw features (which exp 036 tested and failed). With only 5 buckets, the DoF is much lower than exp 036's 24, so per-fold overfitting risk is greatly reduced.

Specifically:
- Low-prob bucket: RealMLP and CB disagree more here → optimal `w_cb` is different.
- Mid-prob bucket: probe 5 found calibration bias here → optimal `w_cb` may differ.
- High-prob bucket: both models confident; optimal `w_cb` ≈ 0.20 (uniform).

## Expected magnitude

- **Target:** OOF lift ≥ +0.00010 over cycle 7's 0.95408 → ≥ 0.95418.
- **Optimistic:** OOF lift ≥ +0.00020 → 0.95428 (clears the cycle hurdle).
- **Floor:** OOF lift ≤ 0 → quantile structure doesn't carry useful blend signal.

## Overfitting risk

**Low.** Only 5 weights to fit (vs exp 036's 24). Each bucket has ~87k rows — plenty of statistical power. Nested CV (5-fold outer × full-fold weight fit) ensures evaluation is honest.

## Kill criteria

- [ ] Per-fold std > 0.00050 (instability — fewer DoF should mean smaller std than exp 036's 0.00060).
- [ ] OOF AUC < cycle 7's 0.95408 (no lift, direction dead).
- [ ] Optimal per-bucket `w_cb` flat at 0.20 across all buckets (no signal — uniform was already optimal).

## Scope

- `src/research/blend_quantile.py` (new, ~120 lines).
- Output: `data/blend_quantile_sweep.parquet`, `data/oof_blend_quantile.parquet`, conditional `data/submission_blend_quantile.csv` if hurdle cleared.

Compute: ~1-2 min (pure post-hoc on cached OOFs).

## Reversibility check

No retraining, no CV change. Pure post-hoc analysis on cached OOFs.

## Plan

1. Load RealMLP-multiseed OOF + CB-tuned-exp14 OOF.
2. Bucket rows by RealMLP-prediction quantile (5 equal-frequency buckets, ranked on the OOF).
3. Nested CV: per fold, fit best `w_cb` per bucket on the 4 train folds, apply to the 5th.
4. Concat to full OOF, evaluate AUC, compare to cycle 7's 0.95408.
5. If OOF ≥ 0.95428 → write submission CSV (fit weights on full OOF, apply to RealMLP × CB test submissions).

## Result

Nested-CV quantile-bucketed blend OOF: **0.95407** (vs cycle 7 uniform 0.95408, **Δ = −0.00001**).

Per-fold AUCs + per-bucket weights:

| Fold | AUC      | Weights {bucket: w_cb}                                    |
| ---- | -------- | --------------------------------------------------------- |
|  1   | 0.95466  | {0: 0.000, 1: 0.225, 2: 0.150, 3: 0.200, 4: 0.275}        |
|  2   | 0.95469  | {0: 0.150, 1: 0.250, 2: 0.150, 3: 0.200, 4: 0.250}        |
|  3   | 0.95378  | {0: 0.125, 1: 0.150, 2: 0.175, 3: 0.175, 4: 0.275}        |
|  4   | 0.95312  | {0: 0.100, 1: 0.275, 2: 0.150, 3: 0.175, 4: 0.250}        |
|  5   | 0.95414  | {0: 0.075, 1: 0.275, 2: 0.150, 3: 0.200, 4: 0.250}        |

Per-fold std: **0.00059** (basically identical to exp 036's 0.00060 — fewer DoF didn't reduce noise).

Full-fit bucket weights (diagnostic):

| Bucket | n      | pos_rate | rm range          | best_w | bucket_auc |
| ------ | ------ | -------- | ----------------- | ------ | ---------- |
|   0    | 87,828 |  0.0006  | [0.0000, 0.0016]  | 0.075  | 0.683      |
|   1    | 87,828 |  0.0034  | [0.0016, 0.0072]  | 0.225  | 0.625      |
|   2    | 87,828 |  0.0173  | [0.0072, 0.0481]  | 0.150  | 0.669      |
|   3    | 87,828 |  0.2154  | [0.0481, 0.4835]  | 0.200  | 0.716      |
|   4    | 87,828 |  0.7583  | [0.4835, 0.9964]  | 0.275  | 0.730      |

## Verdict

**Inconclusive (Reverted).** Even with only 5 buckets (vs exp 036's 24), the per-bucket weights don't transfer across folds — per-fold std stays at 0.00059, OOF lift is −0.00001. The diagnostic shows bucket-1 wants `w_cb=0.225` and bucket-4 wants `w_cb=0.275` while bucket-0 wants 0.075 — there IS structural heterogeneity, but the noise on the optimal weights (±0.05-0.10 across folds) exceeds the signal.

## Kill-criteria check

- [x] OOF AUC < cycle 7's 0.95408 — **FIRED** (0.95407, by 0.00001).
- [ ] Per-fold std > 0.00050 — **FIRED** (0.00059 > 0.00050).
- [ ] Optimal per-bucket weights flat at 0.20 — not fired (real heterogeneity).

## Repro stamp

- inputs: `oof_realmlp_multiseed.parquet`, `oof_cb_tuned_exp14.parquet`, `train.csv`
- runtime: ~30 sec

## Learnings

1. **The blend axis is exhausted with the current 2-input pair.** Across exp 029 (advanced operators), exp 036 (per-slice), and exp 037 (per-quantile), no weighting scheme beats cycle 7's uniform `w_cb=0.20`. The pair's information content has been fully extracted.
2. **Per-fold std is a stronger signal than mean OOF for weight-transferability.** Exp 036 had 24 buckets → std 0.00060. Exp 037 had 5 buckets → std 0.00059. Almost identical noise. Suggests the noise is intrinsic to the OOF imperfection, not the bucketing scheme.
3. **Diagnostic vs CV is informative.** Full-fit bucket weights show real heterogeneity (0.075-0.275 range). The structure exists; it just doesn't transfer.

## Follow-ups

- **Exp 038 candidate:** stacking with raw features. Train a tiny meta-model (logistic regression OR shallow LGB) on (rm_oof, cb_oof, top-5 raw features from probe 1). Cycle 9 exp 010 ruled out stacking on weak 4-way bases; with strong 2-input bases + raw features, the meta-model may find slice-aware blending that the per-slice/per-quantile schemes couldn't.
- **Confirmed dead direction:** any post-hoc reweighting on cycle 7's pair. Move to changing inputs (multi-seed CB via exp 035) or adding signal (stacking).
