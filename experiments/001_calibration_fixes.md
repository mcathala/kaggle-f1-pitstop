# #001 — Calibration fixes for three known-miscalibrated features

**Status.** Reverted
**Date.** 2026-05-08

## Hypothesis

Three features in the current pipeline are documented as miscalibrated in [docs/feature_engineering.md §5](../docs/feature_engineering.md#5-known-calibration-issues). A single-fold feature-importance run on the baseline confirms three features have **literally zero gain** in the LightGBM model: `sc_likely`, `cant_finish_on_current_tyres`, `lap_is_anomalous`. Two of those map directly to documented calibration issues; the third (`sc_likely`) is redundant with `field_pace_ratio` (which the tree already splits on) and is a candidate for either removal or a tighter threshold.

If we tighten the calibration of these features so the binary thresholds carry information rather than firing on ~all rows or ~no rows, the model gains usable splits — particularly in regimes where the underlying physics matters (HARD compound, late-stint, end-of-race).

## Expected impact

- **Modest gain: +0.0005 to +0.0020 OOF AUC.**
- LightGBM trees can already split on the underlying continuous columns (`compound_max_life`, `pit_window_start/end`, `typical_stint_length`), so the gain is bounded by how much *binary saturation* is hurting the splits and how much `tyre_life_norm`'s scale matters as a feature in its own right.
- Greater lift expected on the **HARD compound subset** (current AUC 0.839, the worst non-2023 compound), because end-of-stint physics features like `cant_finish_on_current_tyres` matter most there.

## Overfit risk

**Very low.**

- All three changes are deterministic transforms computed on `train + test` combined. No labels involved, no leakage path introduced.
- CV protocol unchanged (5-fold StratifiedKFold on `Year × PitNextLap`, seed=42).
- Hyperparameters unchanged.
- Validation gate: train-val gap must not widen by > 0.005 without OOF improvement. Per-fold std must stay ≤ 0.0008.

## Concrete changes to [src/features.py](../src/features.py)

### Change 1 — `compound_max_life`: p95 → max of completed stints

Before (line ~322-326):

```python
compound_max = (
    stint_lengths.group_by("__compound")
    .agg(pl.col("__stint_length").quantile(0.95).alias("compound_max_life"))
    .rename({"__compound": "Compound"})
)
```

After: use the maximum of completed stints (or a high quantile like 0.99) so the cap is a true upper bound, not the 95th percentile of mostly-short synthetic stints.

```python
compound_max = (
    stint_lengths.group_by("__compound")
    .agg(pl.col("__stint_length").max().alias("compound_max_life"))
    .rename({"__compound": "Compound"})
)
```

This widens `compound_max_life` so that `cant_finish_on_current_tyres = LapsRemaining > (compound_max_life - TyreLife)` no longer saturates at 99.9% true.

### Change 2 — `in_pit_window`: p5/p95 → p15/p85

Before (line ~352-358):

```python
pit_window = (
    df.filter(pl.col("PitStop") == 1)
    .group_by("Race")
    .agg(
        pl.col("LapNumber").quantile(0.05).alias("pit_window_start"),
        pl.col("LapNumber").quantile(0.95).alias("pit_window_end"),
    )
)
```

After: tighten to p15/p85.

```python
pit_window = (
    df.filter(pl.col("PitStop") == 1)
    .group_by("Race")
    .agg(
        pl.col("LapNumber").quantile(0.15).alias("pit_window_start"),
        pl.col("LapNumber").quantile(0.85).alias("pit_window_end"),
    )
)
```

This drops `in_pit_window`'s fire rate from 83% toward something more discriminative (target ~50-60%).

### Change 3 — `typical_stint_length`: median of all completed stints → median of stints with length ≥ 5

The synthetic data has many 1-3-lap stints that pull the median to ~3, inflating `tyre_life_norm` to mean 2.83 and breaking its "fraction of typical life used" interpretation.

Before (line ~313-318):

```python
typical = (
    stint_lengths.group_by(["__compound", "Race"])
    .agg(pl.col("__stint_length").median().alias("typical_stint_length"))
    .rename({"__compound": "Compound"})
)
```

After: filter to stints of length ≥ 5 before taking the median.

```python
typical = (
    stint_lengths.filter(pl.col("__stint_length") >= 5)
    .group_by(["__compound", "Race"])
    .agg(pl.col("__stint_length").median().alias("typical_stint_length"))
    .rename({"__compound": "Compound"})
)
```

If a (Compound, Race) pair has no stints of length ≥ 5, fall back to the overall per-Compound median (so the column doesn't go null and break downstream `tyre_life_norm` for those rows).

## Validation gates

- [ ] OOF AUC ≥ 0.94166 + 0.0005 (i.e. > 0.94216) for "Kept"; flat or worse → "Reverted".
- [ ] Per-fold std ≤ 0.0008 (vs current 0.00042).
- [ ] Train-val AUC gap measured per fold; must not widen > 0.005 without OOF gain.
- [ ] Per-year AUC table — confirm 2022 doesn't regress.
- [ ] Per-compound AUC for HARD: target ≥ 0.842 (vs current 0.839).
- [ ] `cant_finish_on_current_tyres` fire rate well under 99.9%.
- [ ] `in_pit_window` fire rate < 83%.
- [ ] `tyre_life_norm` mean closer to 1 (currently 2.83).

## Result

### OOF AUC (5-fold StratifiedKFold, seed=42)

```
OOF AUC:   0.94168   (vs baseline 0.94166, Δ = +0.00002)
Per-fold:  0.94180, 0.94208, 0.94097, 0.94202, 0.94160
fold std:  0.00040
best iter: 591, 488, 755, 755, 473
```

### Per-fold delta vs baseline

| Fold | Baseline | This experiment | Δ |
|---|---|---|---|
| 1 | 0.94158 | 0.94180 | +0.00022 |
| 2 | 0.94215 | 0.94208 | −0.00007 |
| 3 | 0.94091 | 0.94097 | +0.00006 |
| 4 | 0.94193 | 0.94202 | +0.00009 |
| 5 | 0.94177 | 0.94160 | −0.00017 |

3/5 folds slightly improve, 2/5 slightly regress. Mean Δ +0.00002 is well within the per-fold std (0.00040) — i.e. statistical noise.

### Per-year OOF AUC

| Year | Baseline | This | Δ |
|---|---|---|---|
| 2022 | 0.89892 | 0.89856 | −0.00036 |
| 2023 | 0.92364 | 0.92626 | +0.00262 |
| 2024 | 0.91599 | 0.91615 | +0.00016 |
| 2025 | 0.91596 | 0.91580 | −0.00016 |

2023 gains 0.0026 (mostly from the better-calibrated `tyre_life_norm` correctly handling the sparse-pit-stop year), 2022 loses a touch. Net wash.

### Per-compound OOF AUC (excluding 2023)

| Compound | Baseline | This | Δ |
|---|---|---|---|
| HARD | 0.83861 | 0.83900 | +0.00039 |
| SOFT | 0.84962 | 0.84977 | +0.00015 |
| MEDIUM | 0.92518 | 0.92472 | −0.00046 |
| INTERMEDIATE | 0.91894 | 0.91844 | −0.00050 |
| WET | 0.80448 | 0.82584 | +0.02136 (n=1,308 — noise) |

The hypothesis predicted the biggest lift in HARD; we got +0.00039 — directionally correct, magnitude trivial.

### Feature-importance changes (single fold-1 retrain, gain)

| Feature | Baseline gain | This gain | Δ |
|---|---|---|---|
| `tyre_life_norm` | 67,872 | **84,755** | +25% (now better-calibrated, more useful) |
| `cant_finish_on_current_tyres` | **0** | 975 | now nonzero, but still a small player |
| `in_pit_window` | 65 | 793 | now nonzero |
| `pit_window_start` | n/a | 13,068 | newly informative top-25 feature |
| `compound_max_life` | 12,396 | 12,161 | flat |
| `typical_stint_length` | 8,110 | 9,861 | +22% |
| `sc_likely` | **0** | **0** | dead — still |
| `lap_is_anomalous` | **0** | **0** | dead — still |

### Train-val gap

Fold-1 train AUC 0.97886 vs val AUC 0.94180 → **gap 0.0371**. No baseline gap recorded but per-fold std (0.00040, near identical to baseline 0.00042) shows no overfit.

### Calibration verification (all three fixes effective)

| Feature | Before | After |
|---|---|---|
| `cant_finish_on_current_tyres` fire rate | 99.9% | **89.5%** |
| `in_pit_window` fire rate | 83% | **58%** |
| `tyre_life_norm` mean | 2.83 | **1.41** (median bucket) |
| `tyre_life_norm` Q1→Q5 pit-rate spread | 3.5% → 35.9% | **3.7% → 40.8%** (slightly wider) |

The features themselves are now properly calibrated and discriminative.

## Decision

**Reverted.**

Calibration fixes succeeded *as features* but not *as model improvement*. The flat OOF AUC tells us LightGBM was already extracting all the relevant signal from the underlying continuous columns (`compound_max_life`, `pit_window_start/end`, `TyreLife`, `typical_stint_length`). The binary saturation problem was real but not load-bearing.

Reverting `src/features.py` to baseline. Branch retained for reference but not merged.

## Observations / followups

1. **`sc_likely` and `lap_is_anomalous` are pure dead weight (gain = 0).** Both are redundant with the continuous `field_pace_ratio` / `driver_pace_ratio` that the tree already splits on. **Drop them** in a future cleanup cycle (zero risk, marginal training-time saving).
2. **HARD compound at AUC 0.839 is the real bottleneck** — the worst non-2023 compound by 0.07 vs MEDIUM. Calibration didn't move it (+0.00039). The signal for "when does a HARD-tyre driver pit" needs a different angle: probably stint-finishing-pace dynamics or compound-aware pit-window features rather than threshold tightening.
3. **2022 (AUC 0.899) is the worst year.** Year 2023 gained from this experiment, but 2022 lost. 2022 needs its own diagnostic pass — possibly a feature or label-distribution shift specific to 2022.
4. **The valuable diagnostic from this cycle**: feature importance is far more reliable than calibration metrics for prioritising work. Two of three "miscalibrated" features were already useful via the underlying continuous; one (`sc_likely`) is permanently dead.

## Cycle #2 candidate hypotheses (ranked)

1. **HARD-compound stint-end signals** — add features that capture late-stint pace dropoff (`laptime_diff_3` and `laptime_vs_stint_start` interacted with Compound, or compound-aware tyre-life-percentile). Highest expected impact.
2. **2022 diagnostic** — error analysis specifically on 2022 residuals, find what's missing.
3. **CatBoost or XGBoost diversity model** — add a second base learner before any further feature work. Larger investment, but ensemble averaging often gives +0.001-0.003 AUC for free.
4. **Drop the four zero-gain features** (`sc_likely`, `lap_is_anomalous`, original `cant_finish_on_current_tyres`, original `in_pit_window`) — clean no-op, but only as a cleanup, not an experiment.
