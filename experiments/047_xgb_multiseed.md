# Experiment 047 — Multi-seed XGB-highbins (cycle 11 recipe with second seed)

**Cycle.** 14
**Status.** Inconclusive (Reverted) — variance reduction is real on standalone but doesn't translate to blend OOF.
**Date.** 2026-05-26

## Hypothesis

Running cycle 11's XGB-highbins recipe with a different `MODEL_SEED` (7 instead of 42-derived) and averaging the two OOFs lifts the 3-way blend by ≥ +0.00010, via the same variance-reduction mechanism that helped cycle 5's RealMLP-multiseed.

Specifically: if seed-to-seed rank-corr is < 0.99, averaging meaningfully reduces per-row prediction noise, which should translate into a higher blend optimum (the blend optimizer can place more weight on a less-noisy XGB contribution).

## Rationale

Cycle 14's recipe needed a defensible follow-up to cycle 11 exp 044 (XGB-highbins as 3rd blend base, LB +0.00011). The natural variance-reduction extension was to multi-seed XGB-highbins. We hadn't directly tested whether XGBoost is stochastic-enough-across-seeds for averaging to help — cycle 11 exp 035 found CatBoost was effectively deterministic (rank-corr 0.9971 between seeds 42 and 7), but XGBoost has different stochastic elements (subsample=0.857, colsample_bytree=0.145, random_state controls both row+column sampling) that could plausibly produce more inter-seed variation.

## Expected magnitude

- **Standalone 2-seed OOF target:** +0.00005 to +0.00010 over single-seed 0.95263.
- **3-way blend OOF target:** ≥ 0.95431 (cycle 11 + +0.00010).
- **Floor:** standalone lift < +0.00003 → seeds are too correlated for averaging to help. Confirms variance-reduction axis is exhausted.

## Kill criteria

- [ ] Seed-to-seed rank-corr > 0.998 (deterministic — averaging is wasted).
- [ ] 3-way blend OOF unchanged from cycle 11's 0.95421.

## Result

### Standalone XGB seed 7

5-fold StratifiedKFold (CV_SEED=42 fixed, MODEL_SEED=7).

| Fold | Seed 7 AUC | Seed 42 (cycle 11) | Δ |
| ---- | ---------- | ------------------- | --------- |
|  1   | 0.95329 | 0.95331 | −0.00002 |
|  2   | 0.95306 | 0.95309 | −0.00003 |
|  3   | 0.95226 | 0.95220 | +0.00006 |
|  4   | 0.95176 | 0.95174 | +0.00002 |
|  5   | 0.95277 | 0.95283 | −0.00006 |
| **OOF** | **0.95262** | **0.95263** | **−0.00001** |

Per-fold per-seed Δ averages to essentially zero. Mean wall-clock per fold: 575s (vs cycle 11's average of 665s — slightly faster this run due to system being less loaded).

### Rank-correlation diagnostics

| Pair | ρ |
| ---- | -- |
| Seed 42 vs Seed 7                  | **0.999007** |
| Seed 42 vs RealMLP-multiseed       | 0.97992 |
| Seed 42 vs CB-tuned-exp14          | 0.98403 |
| **Seed 7 vs RealMLP-multiseed**    | **0.97994** |
| **Seed 7 vs CB-tuned-exp14**       | **0.98407** |

Seeds 42 and 7 produce **near-identical** predictions (ρ=0.999). The rank-corr against other bases is identical between the two seeds to the 4th decimal. XGBoost-highbins is **essentially deterministic** for this dataset/recipe.

### 2-seed average

| OOF | AUC |
| --- | --- |
| Seed 42 alone | 0.95263 |
| Seed 7 alone | 0.95262 |
| **2-seed average** | **0.95268** |

Multi-seed lift: **+0.00005** standalone — within expected variance-reduction math (≈ √(2/1) × ρ-related factor).

### Blend probes

| Variant | Weights (RM, CB, XGB, LGB) | OOF AUC | Δ vs cycle 11 3-way |
| ------- | -------------------------- | ------- | ------------------- |
| Cycle 7 baseline (RM, CB)        | (0.80, 0.20, —, —)       | 0.95410 | −0.00011 |
| Cycle 11 3-way (RM, CB, XGB-seed42) | (0.675, 0.075, 0.250, —) | 0.95421 | 0 |
| **3-way (RM, CB, XGB-2seed)**    | **(0.675, 0.075, 0.250, —)** | **0.95421** | **0** |
| **4-way (RM, CB, XGB-2seed, LGB-highbins)** | (0.675, 0.075, 0.250, 0.0)   | 0.95421 | 0 |

**Multi-seed XGB ties cycle 11 exactly.** The standalone +0.00005 doesn't translate to blend OOF improvement. Optimal weights stay at the same (RM=0.675, CB=0.075, XGB=0.250) — the blend optimizer was already at its operating point.

## Verdict

**Inconclusive (Reverted).** The hypothesis is falsified by rank-corr 0.999 between seeds — XGB-highbins is effectively deterministic on this dataset. The standalone +0.00005 variance reduction is real but the blend optimizer's weight allocation already absorbs that level of noise. Cycle 14 closes with no submission.

This is also a useful **constraint discovery**: variance reduction via multi-seed is conclusively exhausted across our three primary tree-model bases (CatBoost cycle 11: ρ=0.997; XGBoost cycle 14: ρ=0.999; LightGBM didn't help anyway). Only RealMLP retains real seed-to-seed variation (cycle 5's 5-seed avg lifted +0.00028) — and we've already squeezed that.

## Kill-criteria check

- [x] Seed-to-seed rank-corr > 0.998 — **FIRED** (0.999007 > 0.998).
- [x] 3-way blend OOF unchanged from cycle 11 — **FIRED** (both at 0.95421).

## Repro stamp

- packages: xgboost 3.2.0
- runtime: 5 folds × ~575s = ~48 min total
- inputs: cycle 14 FE pipeline + `data/oof_xgb_richcat_seed7.parquet`

## Learnings

1. **XGB-highbins is essentially deterministic** at our HP combo (subsample=0.857 + colsample_bytree=0.145). Despite stochastic row + column sampling, seed-to-seed rank-corr is 0.999. The bagging-style randomization inside XGBoost converges to nearly the same predictions because the dataset is large (439k+ rows) and the trees are heavily regularized.
2. **Variance-reduction axis is now fully closed across our three tree-model bases.** CatBoost (cycle 11): ρ=0.997. XGBoost (cycle 14): ρ=0.999. LightGBM (cycle 13): didn't matter — its standalone was too low for blend inclusion.
3. **Standalone +0.00005 doesn't translate to blend +0.00005 at this configuration.** When the existing blend weights are already at the optimum surface, small base-model improvements get absorbed without lifting the optimum. The blend OOF is genuinely capped at 0.95421 for this 3-input pair.
4. **The 0.95421 OOF / 0.95372 LB plateau is real, not an artifact of HP tuning.** Across exp 035 (multi-seed CB), exp 046 (LGB-highbins), and exp 047 (multi-seed XGB), no variance-reduction or family-diversity move clears the +0.0001 noise floor on blend OOF.

## Follow-ups

- **Closed direction:** multi-seed averaging on any of our three tree-model bases.
- **For cycle 15+:** the only paths that could plausibly clear the plateau are (a) fundamentally new model families (TabM/SAINT on GPU — deferred), (b) Optuna HP sweeps with substantially different HP regions (e.g., XGB depth=12 or depth=6 — 3-5h each), or (c) novel FE additions that the existing models haven't been trained on (no candidates identified in cycle 10's diagnostics that would close the standalone gap of any 4th base).
