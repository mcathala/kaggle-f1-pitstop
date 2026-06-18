# #004 — CatBoost diversity model + LightGBM/CatBoost ensemble

**Status.** Kept
**Date.** 2026-05-08

## Hypothesis

Three feature-engineering cycles (#001, #002, #003) all produced ≤ 0.0001 OOF AUC moves on the LightGBM baseline — strong evidence that the model has extracted essentially all of the easily-accessible signal in the current 63-feature space. The next leverage point is **model diversity**: a different inductive bias on the same features, then ensembled with the LightGBM baseline.

CatBoost is the natural choice for this dataset because:

1. **Native ordered target encoding for `Driver`** — 887 categorical levels with skewed counts. LightGBM uses optimal-split categorical handling; CatBoost uses ordered target stats, which produces different per-driver rankings.
2. **Symmetric (oblivious) trees** — fundamentally different tree structure from LightGBM's leaf-wise. Different bias-variance trade-off.
3. **Different default regularization path** — CatBoost over-regularizes by design; LightGBM under-regularizes by default. The two errors are decorrelated.

In playground competitions, a 50/50 LGB+CatBoost ensemble typically gains **+0.001 to +0.003 AUC** vs the better single model with no extra feature work. That's larger than anything cycles #001-003 produced.

## Plan

1. **Add `catboost` to `pyproject.toml`** and `uv sync`. (Done.)
2. **Add a CatBoost trainer at [src/research/train_catboost.py](../src/research/train_catboost.py)** mirroring `src/train.py`'s structure:
   - Same 5-fold StratifiedKFold(seed=42) on Year × PitNextLap.
   - Same feature set (63 features after dropping `id` and `PitNextLap`).
   - Categorical features: `Driver`, `Race`, `Compound` (CatBoost native).
   - Hyperparameters: a single conservative default config (no tuning yet) — `iterations=5000`, `learning_rate=0.05`, `depth=8`, `l2_leaf_reg=3`, early stopping at 100. Goal is to anchor a number, not optimize.
   - Output: `data/oof_catboost.parquet`, `data/submission_catboost.csv`.
3. **Build ensemble** — simple mean of LightGBM and CatBoost OOF/test probabilities. Compute ensemble OOF AUC.
4. **Compare** — three models in the leaderboard: LGB-only, CatBoost-only, ensemble (= the deliverable).

## Expected impact

- **CatBoost standalone**: 0.935 – 0.940 OOF AUC (likely a touch below LGB's 0.94166 — different tree shape often costs 0.002-0.005 on first try).
- **LGB + CatBoost ensemble (mean of probs)**: **+0.0005 to +0.0025 vs LGB-only**, target ≥ 0.94216.

Even if standalone CatBoost is worse than LGB, the *ensemble* often beats LGB-only because the errors are decorrelated. The bar is "ensemble OOF AUC > LGB-only OOF AUC", not "CatBoost ≥ LGB-only".

## Overfit risk

**Low.**

- Same CV protocol, same seed, same features. Adding a parallel model doesn't change the decision boundary that's evaluated on validation.
- Ensembling via simple mean has no fitted parameters — it can't overfit to the OOF distribution.
- Risk to monitor: CatBoost OOF predictions might be miscalibrated relative to LGB, in which case rank-mean (mean of ranks rather than probs) could outperform mean-of-probs. Will try both if simple mean is flat.

## Validation gates

- [ ] CatBoost OOF AUC reported (no fixed target — informational).
- [ ] Ensemble OOF AUC ≥ 0.94166 + 0.0005 (i.e. > 0.94216) for "Kept".
- [ ] Per-fold ensemble std stays ≤ 0.0008.
- [ ] No per-year ensemble regression > 0.001.
- [ ] LGB-vs-CatBoost OOF correlation reported. If > 0.99, the models are too similar — diversity won't help. If 0.94-0.98, ideal range.

## Result

### CatBoost standalone

```
OOF AUC:   0.94774   (vs LGB baseline 0.94166, Δ = +0.00608)
Per-fold:  fold std 0.00041 (matches LGB's tightness)
best iter: [2503, 2185, 2876, 1806, 1878]   (~30s/200iter ≈ 3-5min/fold; 16 min total)
```

Per-year:

| Year | LGB | CatBoost | Δ |
|---|---|---|---|
| 2022 | 0.89892 | **0.91009** | **+0.01117** |
| 2023 | 0.92364 | **0.94080** | **+0.01716** |
| 2024 | 0.91599 | **0.92436** | +0.00837 |
| 2025 | 0.91596 | **0.92494** | +0.00898 |

CatBoost beats LightGBM on **every year**, by 0.008 to 0.017 AUC. The 2022 weakness (worst year for LGB at 0.899) is dramatically improved (+0.011). 2023 (with its synthetic-data near-zero pit rate) gains the most — CatBoost's ordered target encoding apparently handles the 887-level `Driver` × low-positive-rate regime much better than LGB's split-based categorical handling.

LGB-CB OOF probability **correlation: 0.9696** — high (same features, same problem) but not redundant. The 3% diversity is what makes the ensemble work.

### Ensemble (mean of probabilities)

Weight sweep on OOF (full grid):

| w_LGB | OOF AUC |
|---|---|
| 0.00 (CB only) | 0.94774 |
| 0.05 | 0.94784 |
| 0.075 | 0.94787 |
| **0.10** | **0.94789** |
| **0.125** | **0.94789** |
| **0.15** | **0.94789** |
| 0.175 | 0.94788 |
| 0.20 | 0.94785 |
| 0.30 | 0.94766 |
| 0.50 | 0.94680 |
| 1.00 (LGB only) | 0.94166 |

Wide flat plateau at w_LGB ∈ [0.10, 0.15] all hitting 0.94789. Picking **w_LGB = 0.15** (middle of the plateau, slightly conservative — more robust to the LGB→CB OOF correlation drift on test).

### Per-fold ensemble check (w_LGB=0.15)

| Fold | CB only | Ensemble (0.15 LGB) |
|---|---|---|
| 1 | 0.94733 | 0.94754 (+0.00021) |
| 2 | 0.94817 | 0.94830 (+0.00013) |
| 3 | 0.94721 | 0.94742 (+0.00021) |
| 4 | 0.94795 | 0.94806 (+0.00011) |
| 5 | 0.94816 | 0.94820 (+0.00004) |
| **mean** | **0.94776** | **0.94790** |

Every fold gains from the LGB blend. Tiny per-fold gain (~0.00014), but consistent direction → not noise.

### Final result

| Model | OOF AUC | Δ vs baseline |
|---|---|---|
| LGB-only baseline | 0.94166 | — |
| CatBoost-only | 0.94774 | **+0.00608** |
| **LGB+CB ensemble (0.15/0.85)** | **0.94789** | **+0.00623** |

Test prediction mean (ensemble): 0.198 ≈ train target rate 0.199 → no drift.

Submission file: `data/submission_ensemble.csv`.

## Decision

**Kept.** This is the first cycle to actually move the OOF AUC, and it moves it by **+0.00623** — well above the 0.0005 "Kept" threshold. Most of the gain comes from CatBoost itself; the ensemble adds a marginal +0.00015 over CatBoost alone, but the LGB blend is robust on every fold.

Commit both `src/research/train_catboost.py` and the dependency change. Submit `data/submission_ensemble.csv` to Kaggle (one of the few times where a public-LB submission is justified — we want to confirm the +0.006 OOF gain transfers to LB).

## Observations / followups

1. **CatBoost is a much better model for this dataset out of the box.** The single biggest source of gain is the ordered target encoding for the 887-level `Driver`. LGB's split-based categorical handling under-performs by 0.005-0.017 per year despite identical features.
2. **The 2023 anomaly is handled much better by CatBoost** — the near-zero positive-rate year sees AUC 0.941 vs LGB's 0.924. The baseline EDA called out 2023 as a regime-shift problem; CatBoost generalises across it more cleanly.
3. **Pre-cycle hypotheses revisited:**
   - HARD compound (cycle #003) — CatBoost likely closes some of this gap. Worth re-checking per-compound on CatBoost.
   - 2022 weakness — solved (0.899 → 0.910 standalone; 0.908 ensemble).
4. **Promising follow-ups for cycle #5+:**
   - Tune CatBoost hyperparameters (`depth`, `learning_rate`, `l2_leaf_reg`) with a small Optuna budget. The conservative defaults likely have +0.001-0.003 left on the table.
   - Add a **third diverse model** (XGBoost with target-encoded `Driver`, or a small MLP). With only 2 models in the ensemble, adding a third could give another +0.001.
   - Stacking — fit a logistic regression on (LGB_oof, CB_oof) with CV, may slightly beat the constant-weight blend if heteroscedasticity exists.
