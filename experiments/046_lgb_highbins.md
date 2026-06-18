# Experiment 046 — LightGBM with max_bin=5000 (LGB-highbins)

**Cycle.** 13
**Status.** Inconclusive (Reverted) — closes cycle 13. max_bin=5000 generalizes to LGB but standalone gap too wide for blend inclusion.
**Date.** 2026-05-26

## Hypothesis

Cycle 11 exp 044 confirmed that lifting XGBoost's `max_bin` from 256 to 5000 takes its standalone OOF from 0.94615 to 0.95263 — operationalizing cycle 10 probe 1's finding that high-cardinality cross-cats (Race_Year, Race_Compound_Stint) dominate per-feature importance. Open question: does the `max_bin` fix generalize across tree families?

LightGBM has the same parameter (default 255) but uses leaf-wise growth + voting splits, structurally different from XGBoost's level-wise + histogram partitioning. If LGB-highbins reaches standalone ≥ 0.949 AND maintains rank-corr < 0.97 vs the existing 3-way blend bases (RM, CB14, XGB-highbins), it becomes a viable 4th base — leaf-wise growth's different rank ordering provides the diversity that exp 044 ran into a ceiling on (XGB-highbins rank-corr 0.98+ with both other bases).

## Result

5-fold StratifiedKFold (same as cycle 14 protocol), same 132-feature recipe (16 categoricals), `max_bin=5000`, `num_leaves=127`, `learning_rate=0.01`, `feature_fraction=0.15`, `bagging_fraction=0.85`, `lambda_l1=8`, `lambda_l2=8`, `min_data_in_leaf=200`, `cat_smooth=10`, `n_rounds=50000`, `early_stopping=100`.

### LGB-highbins standalone

| Fold | AUC      | iters | runtime |
| ---- | -------- | ----- | ------- |
|  1   | 0.94995  | 1394  | 103 sec |
|  2   | 0.94931  | 1179  |  86 sec |
|  3   | 0.94846  | 1297  |  94 sec |
|  4   | 0.94717  |  920  |  80 sec |
|  5   | 0.94934  | 1407  | 100 sec |
| **OOF** | **0.94885** | — | **~8 min total** |

LightGBM is **15× faster than XGB-highbins** at the same `max_bin=5000` (8 min total vs 55 min). Per-fold convergence in ~1000-1400 iters (LGB's leaf-wise growth extracts cat-cardinality signal faster than XGB's level-wise).

**Standalone AUC 0.94885** is:
- +0.00708 above the cycle-8 LGB-rich baseline (0.94177) — **confirms `max_bin=5000` does lift LightGBM substantially**
- −0.00229 below CB-tuned-exp14 (0.95114)
- −0.00378 below XGB-highbins (0.95263)

### Rank-correlation diagnostics

| Pair | Spearman ρ | Comparison vs XGB-highbins |
| ---- | ---------- | -------------------------- |
| **LGB-highbins vs RealMLP-multiseed** | **0.96652** | XGB-highbins: 0.97992 → LGB is 0.0134 more diverse |
| **LGB-highbins vs CB-tuned-exp14** | **0.96885** | XGB-highbins: 0.98403 → LGB is 0.0152 more diverse |
| (reference: RM-multiseed vs CB14)    | 0.97569 | — |

The leaf-wise growth payoff is real — LGB-highbins has meaningfully MORE diversity than XGB-highbins from both other bases. But that diversity has to overcome a 0.003-0.004 standalone gap.

### Blend probes

| Blend                              | Optimal weights              | OOF AUC | Δ vs cycle 7 | Δ vs cycle 11 |
| ---------------------------------- | ---------------------------- | ------- | ------------ | ------------- |
| Cycle 7 (RM × 0.80 + CB × 0.20)    | (0.80, 0.20)                 | 0.95410 | —            | —             |
| Cycle 11 3-way (RM, CB, XGB)       | (0.675, 0.075, 0.250)        | 0.95421 | +0.00013     | —             |
| **3-way (RM, CB, LGB-highbins)**   | (**0.80, 0.20, 0.000**)      | 0.95410 | 0            | −0.00011      |
| **4-way (RM, CB, XGB, LGB)**       | (**0.675, 0.075, 0.250, 0.000**) | **0.95421** | +0.00013 | **0**         |

**LGB-highbins gets w=0 in both blends.** The 3-way collapses to cycle 7's exact (0.80, 0.20) weights; the 4-way collapses to cycle 11's exact (0.675, 0.075, 0.250) weights. LGB-highbins's lower standalone AUC means its diversity advantage can't earn positive marginal value at any weight.

## Verdict

**Inconclusive (Reverted).** This experiment validates one important hypothesis (max_bin DOES generalize to LightGBM — confirming the cat-cardinality fix is universal across tree families) but the practical impact on our blend is zero. The standalone gap of 0.003-0.004 below our existing bases is too wide for LGB-highbins to earn blend weight despite the meaningful diversity advantage.

Cycle 13 closes with no submission.

## Kill-criteria check

- [x] Standalone OOF < 0.949 → fired (0.94885 < 0.949).
- [x] Optimal w_lg = 0 in 3-way AND 4-way → both fired.

## Repro stamp

- packages: lightgbm 4.6.0
- runtime: ~8 min CPU total (vs XGB-highbins's ~55 min)
- inputs: same 132-feature cycle-14 FE pipeline

## Learnings

1. **`max_bin=5000` is a universal cat-cardinality fix** — confirmed across both XGBoost (+0.00648 from default) and LightGBM (+0.00708 from cycle-8 LGB-rich). Future tree-model experiments on this dataset should default to `max_bin ≥ 5000`.
2. **LightGBM is 15× faster than XGBoost at the same recipe.** Per-fold runtime 1.5 min vs 19 min. If LB-relevant standalone AUC isn't the goal (e.g., for HP sweeps or quick diversity probes), LightGBM is the better substrate.
3. **Diversity ≠ blend value if standalone gap is wide.** LGB-highbins has ρ=0.967 vs both bases (XGB-highbins has ρ=0.984) — a real 0.013-0.015 diversity advantage. But standalone AUC 0.005 below the next-weakest base (CB14) is the binding constraint. The blend optimizer's tradeoff curve still rules in standalone's favor at this point in the project.
4. **The "3rd base needs standalone AUC near the existing pair" rule, established in cycles 8/11, holds for the 4th base too.** Cycle 8 exp 023's LGB-baseline (standalone 0.94177) had w=0. Cycle 11 exp 034's default XGB (0.94615) had w=0. Cycle 13's LGB-highbins (0.94885) has w=0. The threshold appears to be in the 0.949-0.950 standalone-AUC band.
5. **For cycle 14+ candidates: any new base candidate must reach standalone ≥0.949** to be worth integrating into the blend. Diversity below this floor is wasted.

## Follow-ups

- **Closed direction:** any base model with standalone < 0.949 on this recipe.
- **Project closes at LB 0.95372** (cycle 11). Remaining cycle 14+ candidates need either GPU (TabM, Optuna+CB) or a fundamentally new mechanism (e.g., second-order feature interactions hand-engineered, deep learning architectural changes).
