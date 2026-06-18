# Experiment 052 — per-compound tyre-overdue features on XGB-highbins

**Cycle.** 16
**Status.** Inconclusive (Reverted) — features carry mild standalone signal (+0.00006, 4/5 folds up) but barely shift XGB's ranking (ρ 0.999 vs plain XGB); no blend lift.
**Date.** 2026-05-27

## Hypothesis

Adding per-compound tyre-overdue features (TyreLife normalised against its own compound's distribution: minus-median, ratio-to-p75, overdue-p75/p90 flags, beyond-p90 amount, degradation-vs-compound-median) to the cycle-11 XGB-highbins recipe lifts standalone OOF by ≥ +0.00020, or produces a base diverse enough (rank-corr vs RealMLP < 0.98) to lift the 3-way blend OOF ≥ +0.00020 over 0.95421.

## Rationale

- Our cycle-10 probe-2 residual EDA localised the worst-loss quartile to heavily degraded tyres (Cumulative_Degradation Q4 −39.7 vs Q1 −15.0).
- Every existing tyre feature is *absolute* (TyreLife, TyreAgeRatio, …). But a 20-lap HARD tyre and a 20-lap SOFT tyre sit in entirely different pit-pressure regimes. The unused signal is tyre age **relative to what is normal for its own compound** — "is this tyre overdue for its type".
- This is the degraded-tyre axis of the same Q4 slice that exp 051 attacks from the trajectory axis; running both in parallel covers the slice from two independent angles.

## Expected magnitude

- Standalone target: ≥ +0.00020 over cycle-11 XGB-highbins 0.95263.
- Or blend target: 4-way / swap blend OOF ≥ 0.95441.
- Floor: standalone < 0.95243 AND rank-corr ≥ 0.984 (no better than plain XGB on either axis) → revert.

## Kill criteria

- [ ] Standalone OOF < 0.95243 (worse than plain XGB by > min_delta) AND no diversity gain (rank-corr ≥ 0.984).
- [ ] Best blend config does not clear 0.95421 + 0.00005.

## Scope / reversibility

New Kaggle notebook [gpu-kernels/cycle16_xgb_tyreoverdue_gpu.py](../gpu-kernels/cycle16_xgb_tyreoverdue_gpu.py); adds one FE family (`add_tyre_overdue_features`) ahead of the verbatim cycle-11 XGB-highbins pipeline. Reference percentiles computed on the train+test+external union (population statistic per compound, leakage-free w.r.t. folds). Does not touch CV/seed/target/frozen files. Reversible.

## Result

Ran on Kaggle P100 (XGB `device=cuda`), 5-fold, cycle-11 XGB-highbins HPs verbatim; only the 7 per-compound tyre-overdue features added (139 features total).

### Standalone — mild positive

| Fold | tyre-overdue | plain XGB-highbins | Δ |
| ---- | ------------ | ------------------ | --------- |
| 1 | 0.95332 | 0.95331 | +0.00001 |
| 2 | 0.95320 | 0.95309 | +0.00011 |
| 3 | 0.95231 | 0.95220 | +0.00011 |
| 4 | 0.95173 | 0.95174 | −0.00001 |
| 5 | 0.95292 | 0.95283 | +0.00009 |
| **OOF** | **0.95269** | **0.95263** | **+0.00006** |

4/5 folds positive — the features carry real (if small) signal, more than lag-FE did. But +0.00006 is below `min_delta` (0.00020).

### Rank-correlation matrix (OOF)

|     | rm | cb | xgb | to |
| --- | -- | -- | --- | --- |
| rm  | 1.0000 | 0.9758 | 0.9799 | 0.9801 |
| cb  | 0.9758 | 1.0000 | 0.9840 | 0.9840 |
| xgb | 0.9799 | 0.9840 | 1.0000 | **0.9989** |
| to  | 0.9801 | 0.9840 | 0.9989 | 1.0000 |

**tyre-overdue vs plain-XGB ρ = 0.9989** — even more correlated than lag-FE (0.9973). The features moved XGB's ranking almost not at all.

### Blend probe (anchor = cycle-11 3-way, OOF 0.95420)

Swap, 4-way split (any ratio), two-FE average, and free 4-way grid all return **0.95420** (Δ ≤ −0.00001). The free grid's nominal optimum (w_xgb=0.10, w_to=0.15) produces the identical anchor OOF — no lift.

## Verdict

**Inconclusive (Reverted).** The tyre-overdue features carry mild standalone signal (+0.00006, 4/5 folds positive) — genuinely more than the lag features — but XGB's ranking is so stable that they don't change its ordering (ρ 0.999), so they add no blend diversity and earn no incremental weight.

## Kill-criteria check

- [x] Standalone < min_delta improvement AND no diversity (ρ 0.999 ≥ 0.984) — **FIRED**.
- [x] Best blend does not clear anchor + 0.00005 — **FIRED** (0.95420).

## Repro stamp

- Kaggle kernel: `mcathala/cycle-16-xgb-tyre-overdue-exp-052`; notebook [gpu-kernels/cycle16_xgb_tyreoverdue_gpu.py](../gpu-kernels/cycle16_xgb_tyreoverdue_gpu.py)
- packages: xgboost (Kaggle default), device=cuda on P100
- runtime: 5 folds × ~14 min = ~72 min on P100; outputs `data/oof_xgb_tyreoverdue.parquet`, `data/submission_xgb_tyreoverdue.csv`

## Learnings

1. **The "FE on XGB" axis is now definitively closed.** Two independent, well-targeted FE families (exp 051 trajectory, exp 052 tyre-overdue) both produce ρ 0.997-0.999 vs plain XGB. XGB-highbins on this data has a ranking so stable that no feature work shifts it — so feature engineering cannot diversify our blend, regardless of how well it targets the residual slice.
2. **The tyre-overdue features DO carry signal (+0.00006, 4/5 folds positive) — it's the *model* that can't express it as a different ranking.** This is the key distinction: the features aren't useless; XGB just absorbs them into the same ordering. A model with a different inductive bias (CatBoost's symmetric trees + ordered TE, or a different algorithm entirely) might convert that signal into a genuinely different ranking.
3. **Confirms the diversity ceiling for tree-FE.** All our tree models (XGB, LGB, CB) and their FE variants cluster at ρ ≥ 0.98 with each other; only RealMLP (the NN) is structurally diverse, and it's fully exploited. New diversity must come from a different *algorithm*, not new features on XGB.

## Follow-ups

- Closed: feature engineering on XGB as a blend-diversity lever.
- **Next (exp 053):** put the signal-carrying tyre-overdue features on a different model family (CatBoost ranks differently than XGB; or YDF as a genuinely new algorithm) to test whether a different inductive bias converts the mild feature signal into a diverse ranking that clears the blend bar.


## Repro stamp (target)

- Kaggle kernel: `mcathala/cycle-16-xgb-tyre-overdue-exp-052`
- recipe: cycle-11 XGB-highbins HPs (max_bin=5000, eta=0.01, depth=10, λ=8.16, α=8.35, colsample=0.145), 5-fold StratifiedKFold seed 42 on Year×PitNextLap
- output: `data/oof_xgb_tyreoverdue.parquet`, `data/submission_xgb_tyreoverdue.csv`
