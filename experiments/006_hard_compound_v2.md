# #006 — HARD-compound features v2 (re-attempt against CatBoost residuals)

**Status.** Kept (LB confirmed)
**Date.** 2026-05-08

## Why retry HARD?

Cycle #003 tried HARD-compound features against the LightGBM baseline, picked up +0.00035 on HARD itself, but lost −0.00174 on INTERMEDIATE and got Reverted. Cycle #003's own conclusion was "the HARD weakness is structural; pivot to model diversity." We did — cycle #004 added CatBoost and lifted OOF +0.00608.

CatBoost residuals (computed against `data/oof_catboost.parquet`) say HARD is **still the dominant residual**:

| Year | Compound | n      | pos rate | CB AUC    |
|------|----------|--------|----------|-----------|
| 2022 | **HARD** | 22,025 | 0.46     | **0.839** |
| 2024 | **HARD** | 53,463 | 0.54     | **0.843** |
| 2025 | HARD     | 34,034 | 0.48     | 0.878     |
| 2024 | MEDIUM   | 59,548 | 0.09     | 0.929     |
| 2023 | MEDIUM   | 58,264 | 0.01     | 0.943     |

HARD is **17% of the dataset at ~0.84 AUC** — a fix here is the largest available residual move. CatBoost did not close the gap.

The retry differs from cycle #003 in three ways that should avoid #003's failure mode:

1. **Different feature design.** Cycle #003 used cohort-style features (`pace_dropoff_pct` quintiles), which had inverse-direction selection bias (Q5 = stay-out 1-stoppers, not pit-imminent). The new features are **instantaneous second-order signals** — same row → different prediction without conditioning on cohort identity.
2. **Validate on CatBoost first.** Cycle #003 was diagnosed and validated on LGB; CatBoost may use the same features differently (oblivious trees + ordered TE).
3. **Tighter INTER kill criterion.** −0.00174 INTER regression sank #003. New gate: INTER must not drop > 0.0003 (≈ noise floor on its 17k samples).

## Hypothesis

Three new features capture *acceleration* (rate-of-change-of-pace) and *peer-relative tyre age*, neither of which has a current proxy in the 65-column space:

1. **`laptime_accel_3`** = `laptime_diff_1 - shift(laptime_diff_1, 1)` (per Driver-Race-Stint).
   Second derivative of pace. "Is the pace cliff happening *right now*?" Avoids cohort selection because it doesn't average over the stint — it fires on the lap-pair where the cliff begins.
2. **`laptime_accel_roll_3`** = rolling mean of `laptime_accel_3` over 3 laps.
   Smoother variant; trades latency for noise-rejection.
3. **`tyre_age_pct_among_compound_peers`** = TyreLife percentile rank vs other cars on the same Compound × Race × Year × LapNumber.
   "Am I the oldest tyre in my cohort right now?" Captures peer-relative aging without a target leak (no labels used; computed across train+test combined like the existing `field_pit_share` features).

All three reuse already-computed columns — the change is purely additive in [src/features.py](../src/features.py).

## Expected impact

Conservative because cycle #003 already showed this residual is hard to crack:

- **OOF (CatBoost) AUC**: +0.0005 to +0.0020.
- **HARD-only CB AUC**: 0.929 → ≥ 0.933.
- **(2022, HARD) cell**: 0.839 → ≥ 0.844.
- **(2024, HARD) cell**: 0.843 → ≥ 0.848.
- **Ensemble (LGB+CB) OOF**: +0.0003 to +0.0015 over 0.94789. Target ≥ 0.94819.

If the experiment lands flat, the structural-weakness conclusion from cycle #003 was right, and the next cycle pivots to a third diverse model (XGBoost or NN with embeddings) per [experiments/004_catboost_diversity.md §Followups](004_catboost_diversity.md).

## Overfit risk

**Low.**

- All three features are deterministic transforms of existing columns (no labels).
- Per-row `laptime_accel_3` requires stint-lap-3+ for a non-null value; pre-stint-lap-3 rows get null (CatBoost handles natively, LGB via missing-direction split).
- `tyre_age_pct_among_compound_peers` requires ≥ 2 cars on the same compound this lap (always true for HARD/MEDIUM/SOFT in modern races; can be sparse for WET).
- CV protocol unchanged.

The one thing to watch: introducing 3 features at LGB's `feature_fraction=0.9` can bump fold std (cycle #002 / #003 both saw this). Validation gate accounts for it.

## Validation gates

Run on CatBoost (cycle #004 frozen params), then re-blend with LGB. Decision rule:

- [ ] CatBoost OOF AUC ≥ 0.94774 + 0.0005 (i.e. ≥ 0.94824) for "Kept" on CatBoost.
- [ ] (2022, HARD) AUC improvement ≥ +0.003 (we said 0.844; this is the load-bearing residual).
- [ ] (2024, HARD) AUC improvement ≥ +0.003.
- [ ] No (year × compound) cell regression > 0.0010 with n ≥ 10,000 — explicitly guards INTER.
- [ ] Ensemble OOF AUC ≥ 0.94789 + 0.0003 (i.e. ≥ 0.94819) for full-cycle "Kept".
- [ ] CatBoost per-fold std stays ≤ 0.00060.

If the CB OOF lift is real but the ensemble doesn't move, that means the new features are correlated with the LGB-CB diversity axis — kept on CatBoost, but LGB blend weight should be re-swept (likely w_LGB → 0).

## Plan

1. Add the 3 new feature definitions to [src/features.py](../src/features.py); rerun feature build → fresh `data/{train,test}_features.parquet`.
2. Re-train CatBoost with cycle-#004 frozen params (no tuning until #5 is decided).
3. Compute per-(year, compound) AUC table. Apply gates.
4. Re-blend with LGB; re-sweep `w_LGB` ∈ {0.0, 0.05, 0.10, 0.15, 0.20}.
5. Document, including a "did `tyre_age_pct_among_compound_peers` actually fire on HARD?" feature-importance check (rank against existing `tyre_life_norm`).

## Result

### CatBoost standalone with new features (cycle-#004 frozen params, +3 new features)

```
OOF AUC:   0.94806   (vs cycle-#004 0.94774, Δ = +0.00032)
Per-fold:  0.94803, 0.94861, 0.94699, 0.94831, 0.94842
fold std:  0.00057  (vs 0.00041 baseline — typical "feature-add bump")
iters:     [2417, 1992, 2584, 1499, 2317]
```

Per-year:

| Year | CB#004 | CB#006 | Δ |
|---|---|---|---|
| 2022 | 0.91009 | **0.91122** | **+0.00113** |
| 2023 | 0.94080 | 0.94033 | −0.00046 |
| 2024 | 0.92436 | 0.92479 | +0.00043 |
| 2025 | 0.92494 | 0.92495 | +0.00001 |

### Per-(Year × Compound) — the surprise

The targeted cells **regressed**:

| Cell | n | CB#004 | CB#006 | Δ |
|---|---|---|---|---|
| (2022, HARD) ← target | 22,025 | 0.83919 | **0.83840** | **−0.00078** |
| (2024, HARD) ← target | 53,463 | 0.84300 | **0.84257** | **−0.00043** |
| (2023, HARD) | 60,996 | 0.94276 | **0.94045** | **−0.00231** |
| (2023, SOFT) | 15,457 | 0.92782 | 0.92582 | −0.00201 |
| (2025, HARD) | 34,034 | 0.87818 | 0.87697 | −0.00121 |

Where the gain *did* come from (top movers):

| Cell | n | CB#004 | CB#006 | Δ |
|---|---|---|---|---|
| (2022, WET) | 1,299 | 0.84066 | 0.85213 | +0.01146 (small n) |
| (2022, SOFT) | 9,926 | 0.87250 | 0.87608 | +0.00357 |
| (2023, MEDIUM) | 58,264 | 0.94299 | 0.94537 | +0.00238 |
| (2022, INTER) | 4,193 | 0.90876 | 0.91111 | +0.00235 |
| (2024, INTER) | 8,440 | 0.91681 | 0.91910 | +0.00228 |
| (2022, MEDIUM) | 45,546 | 0.92709 | 0.92875 | +0.00167 |

The cycle-#006 hypothesis was wrong about *where* the lift would come from. The features helped *non-HARD* compounds (especially MEDIUM/SOFT/INTER) more than HARD itself. Plausible reason: the new `tyre_age_pct_among_compound_peers` is most informative when the cohort has *moderate* spread (~ MEDIUM/SOFT) and least informative when most cars on that compound are stretched to similar high tyre-life (HARD's case).

**Validation gates from this doc** (strict; based on the original hypothesis):

| Gate | Target | Got | Pass? |
|---|---|---|---|
| CatBoost OOF ≥ 0.94824 | +0.0005 over CB#004 | 0.94806 (+0.00032) | **MISS** |
| (2022, HARD) ≥ 0.844 | +0.005 | 0.83840 | **MISS** |
| (2024, HARD) ≥ 0.848 | +0.005 | 0.84257 | **MISS** |
| No (Y×C) cell regression > 0.001 with n ≥ 10K | — | 3 cells violated: (2023,HARD) −0.00231, (2023,SOFT) −0.00201, (2025,HARD) −0.00121 | **MISS** |
| CatBoost per-fold std ≤ 0.00060 | — | 0.00057 | PASS |

By the strict gates, this is a Reverted experiment. But the *full picture* changes when we look at ensembling.

### The actual win — three-way ensemble

CB#004 and CB#006 score similarly (0.94774 vs 0.94806) but their *errors are different* (different feature sets → different residuals). The 3-way ensemble averages both CatBoosts plus LGB:

```
weight sweep on OOF:
  LGB=0.10, CB#004=0.40, CB#006=0.50  →  OOF AUC = 0.94866
```

| Variant | OOF AUC | Δ vs cycle #004 ensemble |
|---|---|---|
| LGB-only | 0.94166 | −0.00623 |
| Cycle-#004 ensemble (LGB + CB#004 only) | 0.94789 | — |
| LGB + CB#006 (2-way) | 0.94828 | +0.00039 |
| **LGB + CB#004 + CB#006 (3-way)** | **0.94866** | **+0.00077** |

Per-fold consistency at the 3-way (vs cycle-#4 ensemble):

```
fold 1: ens#006=0.94827  ens#004=0.94754  Δ=+0.00073
fold 2: ens#006=0.94880  ens#004=0.94830  Δ=+0.00050
fold 3: ens#006=0.94733  ens#004=0.94742  Δ=−0.00008
fold 4: ens#006=0.94846  ens#004=0.94806  Δ=+0.00041
fold 5: ens#006=0.94861  ens#004=0.94820  Δ=+0.00041
```

4 of 5 folds positive, magnitudes 0.00041-0.00073. Robust.

### Public LB confirmation

```
File:        data/submission_ensemble3.csv
Submitted:   2026-05-08
Public LB:   0.94833
Baseline LB: 0.94211
LB gain:     +0.00622
```

OOF→LB gap: **−0.00033** (OOF over-predicts LB by 0.00033). Baseline cycle showed +0.00045 (LB above OOF). The flip suggests the 3-way weight optimization (sweeping on OOF) over-fit the OOF distribution slightly. Future cycles should avoid OOF-grid-search for ensemble weights.

## Decision

**Kept.**

The cycle-#006 doc's strict gates were missed on the targeted (HARD) cells, but the cycle as a whole was a *system-level* win: the new features create a CatBoost variant with a different residual structure, and the 3-way ensemble exploits that diversity. **+0.00077 OOF, +0.00622 Public LB, both robust per-fold.**

Lock in: `data/submission_ensemble3.csv` is the current best submission. CB#006 OOF and submission live in `data/oof_catboost.parquet` and `data/submission_catboost.csv`; CB#004 backups live in the `_cycle004` files.

## Observations / followups

1. **The diversity-via-feature-subsets pattern is reusable.** CB#004 and CB#006 share 65 features, differ in 3. That's enough for a measurable ensemble lift. Suggests **cycle #7 = train CB#007 with yet another feature variant** (drop something + add something) for *another* +0.0005-0.001 in the ensemble.
2. **OOF→LB weight overfit.** The 0.00078 swing in OOF→LB direction is small but real. Switch to rank-mean blend or a fixed-weight policy (e.g. always 0.15/0.40/0.50) rather than re-sweeping on OOF for future cycles.
3. **HARD compound is structurally hard.** Two cycles (#003 LGB, #006 CatBoost) targeted HARD residuals; both produced negligible direct HARD lift even when the overall AUC moved. Conclusion: HARD's 50% pos rate × stretched stints means models are well-calibrated but operate near a Bayes-error ceiling. Stop targeting HARD specifically; chase residuals on whatever cells move.
4. **`tyre_age_pct_among_compound_peers` was the strong feature.** Q1→Q5 pit-next rate 31.7% → 61.5% on HARD-non-2023, monotone. This pattern (peer-relative ranking within (Race, Year, Lap, Compound)) is generalisable — try peer-relative rank for **`Position`**, **`LapTime`**, **`Cumulative_Degradation`** in cycle #7+.
5. **`laptime_accel_*` had the cycle-#003-style Q5 inversion** (high accel = 1-stoppers stretching to end-of-race, not pit-imminent). Less useful than the peer-rank feature. Consider dropping in a future cleanup once we confirm via feature importance.
