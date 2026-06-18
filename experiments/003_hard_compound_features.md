# #003 — Pace-dropoff features for stint-end physics

**Status.** Reverted
**Date.** 2026-05-08

## Hypothesis

The current weakest non-2023 compound is **HARD at OOF AUC 0.839** — far below MEDIUM (0.925) and INTERMEDIATE (0.919). Pit decisions on HARD are *physics-driven* (the tyre runs long until pace drops below a threshold), but the existing features encode degradation in absolute terms (`laptime_vs_stint_start`, `Cumulative_Degradation`, `cum_deg_diff_1`) that don't scale across tracks with different lap times.

A **percentage** pace-dropoff signal — "this driver is N% slower than they were at stint start" — should be more informative than the absolute `laptime_vs_stint_start` because it's invariant to track lap-time magnitude. It should give the model a clean "pace has fallen off the cliff" signal that trips the pit decision, especially for HARD where TyreLife alone isn't enough (50+ laps is normal for HARD).

## Concrete changes — three new features in [src/features.py](../src/features.py)

1. **`pace_dropoff_pct`** = `(laptime_roll_mean_3 - stint_start_pace) / stint_start_pace`
   "% slower than stint start, smoothed over 3 laps." Scale-invariant.
2. **`pace_dropoff_pct_5`** = `(laptime_roll_mean_5 - stint_start_pace) / stint_start_pace`
   Same with 5-lap window — captures longer-term decline.
3. **`pace_dropoff_per_lap`** = `pace_dropoff_pct / max(TyreLife, 1)`
   Average % degradation per lap — distinguishes "fast-degrading" from "stable" tyres at the same TyreLife.

All three use already-computed columns (`laptime_roll_mean_3/5`, `stint_start_pace`, `TyreLife`), so the change is purely additive in the per-row derivation step.

## Expected impact

- **OOF AUC: +0.0005 to +0.0020.** The signal is most useful for HARD where existing TyreLife/Compound features under-discriminate. Wider gain bound than cycle #1 because we're *adding signal*, not just recalibrating existing features.
- HARD-compound AUC target: 0.839 → ≥ 0.842.
- Modest gain on SOFT (currently 0.850) is also possible.

## Overfit risk

**Very low.**
- Deterministic transforms of existing features; computed on train+test combined, no labels involved.
- CV protocol unchanged.
- All three features use `stint_start_pace` (mean of first 3 laps of stint). The denominator is well-defined for all rows after stint-lap-2; pre-stint-lap-3 rows get `pace_dropoff_pct` near zero (stint just started, no degradation yet).
- Tree-based models are robust to the multicollinearity introduced (pace_dropoff_pct vs pace_dropoff_pct_5).

## Validation gates

- [ ] OOF AUC ≥ 0.94166 + 0.0005 (i.e. > 0.94216) for "Kept"; flat → "Pending" / inconclusive.
- [ ] Per-fold std stays ≤ 0.0008 (current 0.00042).
- [ ] Train-val gap on fold-1 must not widen by > 0.005 vs baseline.
- [ ] Per-compound HARD AUC ≥ 0.842.
- [ ] No per-year regression > 0.001.

## Result

```
OOF AUC:   0.94158   (vs baseline 0.94166, Δ = −0.00008)
Per-fold:  fold std 0.00061 (vs 0.00042 baseline)
best iter: [454, 531, 483, 454, 596]
```

Per-year:
- 2022: 0.89874 (Δ −0.00018)
- 2023: 0.92347 (Δ −0.00017)
- 2024: 0.91602 (Δ +0.00003)
- 2025: 0.91562 (Δ −0.00034)

Per-compound (excl 2023):
- HARD: 0.83861 → **0.83896** (Δ +0.00035) — the targeted compound improved by a hair
- SOFT: 0.84962 → 0.84953 (Δ −0.00009)
- MEDIUM: 0.92518 → 0.92475 (Δ −0.00043)
- INTERMEDIATE: 0.91894 → **0.91720** (Δ −0.00174) — meaningful regression
- WET: 0.80448 → 0.80924 (Δ +0.00476, n=1,308 — noise)

Sanity check on the new features themselves — `pace_dropoff_pct` quintile vs target on HARD-only (excl 2023):

| Quintile | mean dropoff | pit_next |
|---|---|---|
| Q1 | −0.068 | **0.545** |
| Q2 | −0.010 | 0.563 |
| Q3 | −0.001 | 0.495 |
| Q4 | +0.005 | 0.475 |
| Q5 | +0.047 | **0.449** |

**Inverse relationship.** Drivers with the highest pace dropoff on HARD are *less* likely to pit next. Most likely interpretation: by the time a HARD-tyre driver shows large pace dropoff, they're committed to running to end-of-race (1-stoppers). Drivers who pit earlier do so *before* dropoff peaks. The feature is signaling "this driver has already absorbed degradation = stay-out commitment", not the assumed "pace cliff = pit imminent."

This is a real signal, just opposite of the hypothesis.

## Decision

**Reverted.** HARD-specific gain (+0.00035) is too small and is more than offset by the INTER regression. Net OOF Δ negative.

## Observations / followups

1. **Three cycles, three flat results.** Cycles #001 (calibration), #002 (drop dead features), #003 (pace-dropoff) all produced ≤ 0.0001 OOF moves, all swamped by per-fold std. The current LightGBM baseline appears to have extracted essentially all of the signal in the existing 63-feature space.
2. **Adding/removing features at `feature_fraction=0.9` reliably bumps fold std from 0.00042 to ~0.00060** even when the feature itself is genuinely null. Bagging is sensitive to candidate-pool size.
3. **Next move: model diversity, not more features.** Pivot to CatBoost (cycle #4) — different inductive bias should pull predictions in slightly different directions, and a 50/50 ensemble averaging often gives +0.001–0.003 AUC for free in playground competitions.
4. **The HARD-compound weakness is structural** (50% pos rate → less class imbalance → harder to discriminate among near-pit rows). Probably needs ensembling or a model class change to break, not more features.
