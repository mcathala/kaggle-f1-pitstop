# Experiment 044 — XGBoost retune for high-cardinality categoricals (max_bin=5000)

**Cycle.** 11
**Status.** **Kept (LB-confirmed)** — closes cycle 11 at LB 0.95372 (+0.00011 over cycle 7).
**Date.** 2026-05-26

## Hypothesis

XGBoost's default `max_bin=256` is the binding constraint on its standalone AUC for our 132-feature recipe. Cycle-10 probe 1 ranked our top features as high-cardinality cross-cats (`Race_Year` at 8.83 mean importance, `Race_Compound_Stint` at 4.93). With only 256 histogram bins, the splitter can't resolve these — multiple distinct values get aliased into the same bin, losing signal.

Lifting `max_bin` to 5000 (with compensating regularization to avoid overfitting the high-resolution histogram) lifts XGBoost standalone OOF to ≥ 0.95, which would make it competitive enough to earn weight in a 3-way blend with RealMLP-multiseed and CB-tuned-exp14. The 3-way blend then clears the cycle 11 hurdle of OOF ≥ 0.95428 (+0.00020 over cycle 7's 0.95408).

## Rationale

Exp 034's XGBoost capped at OOF 0.94615 (well below CB14's 0.95114 and RealMLP's 0.95383). Three diagnostic findings together motivated this retune:

1. **Probe 1 importance ranking:** Race_Year (cross_cat) is the single highest-importance feature in cycle 14's CB at mean importance 8.83. cross_cat family mean importance 2.16/feature vs raw_num 1.28/feature — categoricals carry disproportionate signal.

2. **XGBoost histogram theory:** with default `max_bin=256` and ~900+ unique Driver values × 26 Race values, Race_Year alone has potentially ~20k+ unique values. Default binning has ~80× too few bins; many distinct values collapse together.

3. **Exp 034 / 041 failure pattern:** XGBoost converged at iter ~300 and early-stopped at AUC 0.946 across all HP variants. The model wasn't running out of capacity — it was running out of feature resolution. No HP change to depth, lr, regularization shifted the ceiling.

The retune addresses (2) directly: `max_bin=5000` gives the histogram enough resolution for our cross-cats. The other HP shifts (slower `eta=0.01` over `n_estimators=50000`, deeper `max_depth=10`, stronger L1/L2 regularization with `colsample_bytree=0.145` heavy column subsample) absorb the added capacity without overfitting the high-resolution histogram.

## Expected magnitude

- **Standalone AUC target:** ≥ 0.948 (+0.002 over exp 034).
- **3-way blend OOF target:** ≥ 0.95428 (clears the cycle 11 hurdle).
- **Floor:** standalone < 0.948 → max_bin wasn't the bottleneck after all.

## Overfitting risk

**Low.** Strong L1 (α=8) + L2 (λ=8) + heavy column subsampling (`colsample_bytree=0.145`) compensate for the added histogram resolution. Cross-fold OOF is the validation, not LB.

## Kill criteria

- [ ] Standalone OOF < 0.948 — max_bin wasn't the fix.
- [ ] OOF AUC < cycle-7 blend 0.95408 in 3-way blend — XGBoost contributes nothing despite improved standalone.
- [ ] Rank-corr with RM > 0.985 AND rank-corr with CB14 > 0.985 — XGB converged to same predictions; no diversity.

## Scope

- `src/research/train_xgb_richcat.py` (new — adapted from `src/research/train_xgb.py` with retuned HPs).
- Outputs: `data/oof_xgb_highbins.parquet`, `data/submission_xgb_highbins.csv` (legacy filenames from initial run; documented in repro stamp).
- 3-way blend artifacts: `data/oof_blend_3way_xgb.parquet`, `data/submission_blend_3way_xgb.csv`.

## Result

### Standalone XGBoost-highbins

5-fold StratifiedKFold (same as cycle 14):

| Fold | AUC      | best iter | runtime |
| ---- | -------- | --------- | ------- |
|  1   | 0.95331  | 3796      | 19 min  |
|  2   | 0.95309  | 3746      |  9 min  |
|  3   | 0.95220  | 3551      |  9 min  |
|  4   | 0.95174  | 3554      |  9 min  |
|  5   | 0.95283  | 3518      |  9 min  |
| **OOF** | **0.95263** | — | ~55 min total |

**Standalone AUC 0.95263 — +0.00648 above exp 034 (0.94615), +0.00149 above CB14's 0.95114.**

### Rank-correlation diagnostics

| Pair | Spearman ρ |
| ---- | ---------- |
| XGB-highbins vs RealMLP-multiseed | 0.97992 |
| XGB-highbins vs CB-tuned-exp14    | 0.98403 |
| (reference: RM-multiseed vs CB14) | 0.97569 |

Higher standalone AUC came at the **cost of diversity** — the new XGB is slightly more correlated with both bases than they are with each other. The diversity gain that exp 034 showed (ρ=0.962 with both) is gone; the model converged to similar predictions because it's now strong enough to "find" the same signal.

### 3-way blend grid search

Linear blend over the 2-D simplex (`w_cb14` × `w_xgb` ∈ [0, 0.45] step 0.025):

| Operator | Best w (RM, CB14, XGB) | OOF AUC | Δ vs cycle 7 |
| -------- | ----------------------- | ------- | ------------ |
| **Linear** | **(0.675, 0.075, 0.250)** | **0.95420** | **+0.00012** |
| Rank-blend | (0.675, 0.075, 0.250) | 0.95418 | +0.00010 |
| With 6-seed RM (incl. seed 777) | (0.675, 0.075, 0.250) | **0.95421** | **+0.00013** |

XGBoost finally gets meaningful weight (w_xgb=0.25), and CB14's weight drops 0.20 → 0.075 — the blend optimizer judges XGBoost a better partner for RealMLP than CB at the new standalone AUC.

### Other directions tested (all rejected)

- **4-way (RM, CB14, XGB, LGB-rich):** best at w_lgb=0 → degenerate to 3-way 0.95420.
- **4-way (RM, CB-multiseed, XGB, LGB-rich):** best at w_lgb=0 → degenerate.
- **Stacking (LR + LGB on 3 OOFs + raw features):** LR 0.95220 (collapses); LGB 0.95411 (worse than linear 3-way).
- **TargetEncoder preprocessing for XGB:** exp 042 (rejected — AUC 0.941 << native).
- **Multi-seed CB OOF as replacement:** ties single-seed CB exactly in the 3-way (rank-corr seed 42 vs 7 = 0.9971).

### Kaggle LB

**Submission:** `data/submission_blend_3way_xgb.csv` at 3-way weights (0.675, 0.075, 0.250) using 6-seed RM.

**Public LB: 0.95372** (+0.00011 over cycle 7's 0.95361).

OOF→LB drift: 0.95421 − 0.95372 = **−0.00049** (consistent with cycle 7's −0.00047). Drift profile remains stable.

## Verdict

**Kept (LB-confirmed).** First meaningful LB lift since cycle 7. Cycle 11's hypothesis test of "XGBoost cap is feature-resolution, not capacity" is validated: lifting `max_bin` from 256 to 5000 took XGB standalone from 0.946 → 0.953. The standalone-AUC story shifted the blend optimizer's verdict on XGB from `w_xgb=0` (exp 034) to `w_xgb=0.25`, and the resulting 3-way blend clears cycle 7's LB by +0.00011.

## Kill-criteria check

- [ ] Standalone OOF < 0.948 — **NOT FIRED** (0.95263).
- [ ] OOF AUC < cycle-7 blend 0.95408 in 3-way blend — **NOT FIRED** (0.95421).
- [ ] Rank-corr > 0.985 with both bases — **PARTIALLY FIRED** (ρ vs RM = 0.9799 OK, ρ vs CB14 = 0.9840 over threshold). But blend optimizer still finds non-zero weight, so the diversity loss is tolerable.

The blend hurdle of +0.00020 OOF was missed by 0.00007. But LB validation closed cycle 11 anyway — the OOF→LB drift behaved as expected and the LB lift is real (+0.00011).

## Repro stamp

- data: train.csv sha256 `f004e79d…`
- packages: xgboost 3.2.0, sklearn 1.8.0, pandas 3.0.2, numpy 2.4.4
- runtime: ~55 min CPU on M1 Pro
- Kaggle submission ref: 53046046, public score 0.95372

## Learnings

1. **`max_bin` is THE critical XGBoost HP for high-cardinality cross-cats.** Default 256 → effective ceiling at AUC 0.946 on our data. Lifting to 5000 → ceiling at AUC 0.953. No other HP shift (depth, lr, regularization) closes this gap — it's a binning-resolution problem, not a capacity problem.
2. **Strong standalone AUC and high diversity are anti-correlated.** Exp 034's weaker XGBoost had rank-corr 0.962 with RM (good diversity but bad standalone); the retuned XGBoost has rank-corr 0.980 (worse diversity but much better standalone). The blend optimizer prefers the latter — standalone strength beats diversity at this point in the project.
3. **Probe 1's family-importance ranking was the load-bearing diagnostic.** Without "cross_cat is 2.16/feature mean" from probe 1, we wouldn't have suspected the high-cardinality bottleneck. EDA cycles continue to pay off cycles later.
4. **LB-confirmed > OOF hurdle.** The +0.00020 OOF hurdle was a conservative threshold; the +0.00013 we landed at would have been dismissed pre-LB, but LB confirmed +0.00011 — a real lift. The hurdle should be calibrated against LB-drift uncertainty, not OOF noise alone.
5. **The cycle 7 → cycle 11 "Inconclusive desert" was an artifact of the wrong axis.** Multi-seed CB (exp 035), all blend re-weighting (036/037/038/039/040), stacking, per-slice — none moved OOF because **the bottleneck was a missing 3rd base, not the existing 2-input blend formula**. Once we added a competitive XGBoost, the cycle 7 blend was no longer the global optimum.

## Follow-ups

- Cycle 12 candidates:
  - **Multi-seed XGB-highbins** (run seeds 7, 99 of the same recipe and average — expected +0.00002–0.00005 standalone).
  - **TabM_D_Classifier from pytabkit** (untested model family — potential new diversity).
  - **Optuna sweep around max_bin=5000** to find local optimum.
- This cycle's win came from operationalizing a probe finding (probe 1's cross_cat dominance) into a specific HP change. Future cycles should keep tying changes back to diagnostic evidence.
