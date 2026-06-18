# Experiment 034 — XGBoost as 3rd diverse base

**Cycle.** 11
**Status.** Inconclusive (Reverted) — diversity gain insufficient to overcome weak standalone AUC.
**Date.** 2026-05-26

## Hypothesis

XGBoost on cycle 14's exact FE produces lower rank-corr with both RealMLP-multiseed and CB-tuned-exp14 than CB14 has with RealMLP. This diversity, even at lower standalone AUC, makes XGBoost a viable 3rd blend base — the 3-way blend OOF clears the cycle hurdle of 0.95428.

## Rationale

Cycle 8 exp 023 ruled out LGB as the 3rd base (rank-corr 0.943 vs CB14 was too high for diversity). XGBoost has a completely different split mechanism (level-wise + histogram, vs LGB's leaf-wise + voting) and different regularization (L1+L2 leaf weight penalty vs LGB's min-split-gain). Possibility: XGBoost gives different rank ordering than LGB, allowing diversity to win.

## Result

5-fold StratifiedKFold (same as cycle 14), same 132-feature recipe (16 categoricals via `enable_categorical=True`).

### XGBoost standalone

| Fold | AUC      | iters | runtime |
| ---- | -------- | ----- | ------- |
|  1   | 0.94672  | 339   | 36 sec  |
|  2   | 0.94657  | 371   | 39 sec  |
|  3   | 0.94544  | 426   | 43 sec  |
|  4   | 0.94569  | 308   | 35 sec  |
|  5   | 0.94637  | 371   | 40 sec  |
| **OOF** | **0.94615** | — | — |

**Standalone AUC 0.94615 — significantly below RealMLP (0.95383) and CB14 (0.95114).** Early stopping fired at iter ~300-430 in every fold; the recipe converged fast and at a low ceiling.

### Rank-correlation diagnostics

| Pair | Spearman ρ |
| ---- | ---------- |
| **XGB vs RealMLP-multiseed** | **0.96230** ✓ |
| **XGB vs CB-tuned-exp14** | **0.96447** ✓ |
| (reference: RealMLP-multiseed vs CB14) | 0.97569 |

**XGBoost is meaningfully more diverse** from both bases than they are from each other (rank-corr ~0.0134 lower). Diversity is real.

### 3-way blend grid search

Linear blend, 2-D simplex sweep over w_cb14 × w_xgb ∈ [0.0, 0.40] step 0.025:

| Best | w_rm | w_cb14 | w_xgb | AUC |
| ---- | ---- | ------ | ----- | --- |
| Linear | 0.800 | 0.200 | **0.000** | 0.95408 |
| Rank-blend | 0.775 | 0.225 | **0.000** | 0.95408 |

**w_xgb = 0.000** for both operators. The optimizer judges XGBoost contributes zero positive marginal value.

## Verdict

**Inconclusive (Reverted).** XGBoost's diversity advantage (Δρ ~−0.013 vs CB14's reference) is real but insufficient to overcome its standalone AUC gap. At 0.94615 standalone (vs RM 0.95383, CB14 0.95114), XGBoost is **0.00800 below RealMLP** and **0.00500 below CB14**. The blend math:

- For inclusion at any weight, the marginal `∂AUC/∂w_xgb` must be positive at w_xgb=0.
- Numerically: tested weight 0.025 → AUC drop. Tested weight 0.05 → larger drop. Tested any positive w_xgb → drop.
- The standalone gap is too large for the diversity to compensate.

This mirrors cycle 8 exp 023's LGB verdict: a 3rd base needs standalone AUC close enough to the existing pair (within ~0.003-0.005) for its diversity to make the blend optimizer want any weight.

## Kill-criteria check

- [x] OOF AUC < cycle 7's 0.95408 — not applicable directly (3-way at w_xgb=0 = cycle 7 result).
- [x] Optimal w_xgb = 0 → XGBoost adds no value in the blend — **FIRED**.

## Repro stamp

- packages: xgboost 3.2.0
- runtime: ~3 min total (5 folds, ~40 sec each)

## Learnings

1. **Standalone AUC is the binding constraint, not rank-corr.** Even with meaningfully lower rank-corr (0.9623 vs CB-RM's 0.9757), XGBoost at 0.946 standalone can't earn blend weight against bases at 0.954/0.951. The blend optimizer treats AUC and rank diversity in a non-linear trade-off; a 0.005-0.008 standalone gap is too wide.
2. **XGBoost's categorical handling underperforms CatBoost's ordered TE.** With identical 132 features (16 categoricals), XGBoost at depth=8 lr=0.05 reached AUC 0.946 in ~400 iters and stopped. CB14 with the same recipe (depth=8 lr=0.018) reached 0.951 over ~5000 iters. XGBoost's split-based cat handling can't extract as much signal from high-cardinality categoricals as CB's ordered TE.
3. **Convergence speed is fast for XGBoost but at a low ceiling.** 5 folds × 40 sec = 3 minutes total — 60× faster than CB. But the ceiling on this dataset is ~0.946 with default-ish HPs; tuning is needed to push it higher.
4. **Confirmed pattern across cycles 8 + 11**: adding any non-RealMLP/non-CB14 base at its naïve HPs fails to clear the blend hurdle. Future 3rd-base attempts must establish standalone AUC ≥ ~0.949 before testing the blend.

## Follow-ups

- **Exp 041 candidate:** XGBoost HP sweep targeting standalone AUC ≥ 0.949. Variables: depth ∈ {8, 10, 12}, lr ∈ {0.01, 0.02, 0.03, 0.05}, iter cap 10000 with longer early-stop. ~30-60 min compute. If reaches 0.949, retry 3-way blend.
- **Alternative direction:** different model family with stronger standalone tabular AUC. Candidates: TabNet, SAINT, deep CatBoost variant. Compute-heavy.
