# Experiment 081 — differentiated (stripped) FE on CatBoost

**Cycle.** 18
**Status.** Inconclusive — stronger base (+0.00068 OOF) but low blend leverage.
**Date.** 2026-05-29

## Hypothesis

The diffFE win on XGB (exp 080, +0.00028 from stripping cross-cats/frequency/group-stats) should transfer to CatBoost — possibly more, since CatBoost's native ordered target-encoding makes the hand-built cross-categoricals especially redundant.

## Method

Forked `train_cb_tuned_exp14.py` → `train_cb_diffFE.py`. Same strip: dropped 9 cross-categoricals, union-frequency on cross/bins (kept base-cat freq), all group-stats. 132 → 49 features. Same HP (iterations 8000, depth 8, etc.), CV, external augmentation.

## Result

| | OOF AUC | per-fold mean ± std |
| --- | --- | --- |
| CB-tuned-exp14 (132 feat) | 0.95114 | — |
| **diffFE-CB (49 feat)** | **0.95182** | 0.95182 ± 0.00058 |
| **Δ** | **+0.00068** | (bigger than XGB's +0.00028) |

Per-fold: 0.95247 / 0.95249 / 0.95143 / 0.95102 / 0.95167. Note all folds hit the 8000-iter cap without early-stopping firing → mildly undertrained (could squeeze a little more with a higher cap).

### Blend

Clean rebuild including diffFE-CB: OOF 0.95448 → **0.95449** (diffFE-CB enters at w≈0.045). Negligible blend movement.

## Verdict

**Inconclusive.** diffFE makes CatBoost substantially stronger standalone (+0.00068, the biggest single-base FE gain of the cycle), confirming the lever generalizes across GBDT families — and confirming CatBoost's native cat handling made the engineered cross-cats redundant + harmful. But CatBoost's weight in the RealMLP-dominated blend is ~0.045, so the standalone gain barely moves the blend (same low-leverage story as exp 068/071 pseudo-CB).

## Repro stamp

- Trainer: [src/train_cb_diffFE.py](../src/train_cb_diffFE.py) (49 features).
- Output: `data/oof_cb_diffFE.parquet` (0.95182), `data/submission_cb_diffFE.csv`.
- 5-fold StratifiedKFold(42); CPU, ~100 min.

## Learnings

1. **diffFE generalizes across GBDTs (XGB +0.00028, CB +0.00068)** — the hand-engineered cross-cats/union-stats were net-harmful overfitting for both. CatBoost benefited more because its ordered-TE already encodes high-card cats optimally.
2. **CatBoost's blend leverage remains low** regardless of standalone strength (RealMLP dominates).
3. Mildly undertrained at the 8000-iter cap; not worth re-running given the low blend leverage.

## Follow-ups

- The diffFE strength gain on GBDTs is real but blend-diluted. Next: pseudo on diffFE-XGB (exp 083, higher blend weight than CB).
