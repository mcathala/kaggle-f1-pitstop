# Feature Engineering — Kaggle Playground S6E5: F1 Pit-Stop Prediction

Reproduce: `.venv/bin/python src/features.py`. Outputs `data/train_features.parquet` (439,140 × 65 cols) and `data/test_features.parquet` (188,165 × 64 cols). The implementation lives in [src/features.py](../src/features.py).

This document explains what we added on top of the raw 16-column schema described in [dataset.md](dataset.md), how each feature is computed, and why it should help the model.

## TL;DR

- **49 new columns** added on top of the 16 raw ones, computed on `train + test` combined (no labels used).
- Five logical blocks: race-level aggregates, field-wide per-lap statistics, safety-car detection, positional pit signals, per-driver timeline lags.
- Strongest empirical signals (target rate Q5/Q1 ratio):
  - `field_pit_share` quintile: **2.2% → 38.2%** (17×)
  - `tyre_life_norm` quintile: **3.5% → 35.9%** (10×)
  - Ahead/behind both pitted recently: **7.8% → 26.3%** (3.4×)
- Three known calibration weaknesses (`cant_finish_on_current_tyres` saturates, `in_pit_window` too wide, `tyre_life_norm` denominator too small) — kept for now since they're still ordinal-monotonic and trees can use the underlying continuous values.
- The dataset's documented identities (`LapTime_Delta`, `Position_Change`, `PitNextLap ⟺ PitStop_{i+1}`) do not hold empirically — see §6. Our recomputed timeline features bypass this.

---

## 1. Why feature engineering

The raw schema gives ~5 useful predictors (`TyreLife`, `LapNumber`, `Stint`, `RaceProgress`, `Cumulative_Degradation`) and a handful of context columns. Two structural facts about the data make engineering more profitable than usual:

1. **Row-level train/test split.** The same `(Race, Year, Driver)` appears in both train and test with disjoint `LapNumber`s ([eda.md §8](eda.md)). Per-driver timelines, per-lap field statistics, and per-race aggregates can all be computed on `train + test` combined — no labels involved, no leakage — so test rows benefit from the full population's context.
2. **Pit-stop strategy is shaped by signals not present in the raw row.** Two examples: a) when several cars pit on the same lap, the rest of the field reacts; b) safety cars are a "free" pit window, but the dataset has no SC flag. Both can be reconstructed from the dataset itself.

We avoided external data (FastF1, Ergast, weather APIs) because the dataset is synthetic — the 2023 anomaly, synthetic `D###` driver codes, and "Pre-Season Testing" as a "race" mean real-world joins don't line up cleanly.

---

## 2. Constraints

- **Synthetic data quirks.** 2023 has ~30× fewer pit stops than other years; ~60% of train rows belong to synthetic `D###` driver codes; "Pre-Season Testing" appears as a 26th "race" with non-race strategy dynamics. We surface these as flags rather than try to clean them away — the test set has the same distribution.
- **Noisy labels.** `PitNextLap` does not strictly equal `PitStop_{i+1}` (~81% agreement, see §6). Treat the label as ground truth; treat `PitStop` as a feature that mostly tracks it.
- **Row-level split.** OK to compute aggregates on `train+test`; not OK to use `PitNextLap` from any row in any feature.

---

## 3. Feature catalog

Features are grouped by aggregation level. Each entry has a one-line definition, the polars groupby key it's computed at, and a brief note on what it captures.

### 3.1 Per-race aggregates

Computed by grouping on `Race` only (across years, for stability — 2023's near-zero pit rate makes per-(Race, Year) estimates noisy).

| Column | Definition | Captures |
|---|---|---|
| `pit_cost_seconds` | `median(LapTime \| PitStop=1) - median(LapTime \| PitStop=0)` per Race | Track-specific pit-lane cost proxy. São Paulo +8s top, Monaco +1.7s, Spa −1.5s. (Negative values indicate the pit-lap LapTime is faster than non-pit median, likely because `PitStop=1` rows correspond to the outlap on fresh tyres.) |
| `pit_window_start` | 5th percentile of `LapNumber` over rows with `PitStop=1` per Race | Earliest typical pit lap. |
| `pit_window_end` | 95th percentile of same | Latest typical pit lap. |
| `in_pit_window` | `LapNumber ∈ [start, end]` | Binary flag. Saturates at 83% — see §5. |

### 3.2 Per-(Race, Year) aggregates

| Column | Definition | Captures |
|---|---|---|
| `total_laps_year` | `max(LapNumber)` per (Race, Year) | Race length this year. |
| `LapsRemaining` | `total_laps_year - LapNumber` | Inverse of `RaceProgress` in absolute laps. More direct end-of-race signal than the [0,1] progress fraction. |
| `is_wet_race` | `any(Compound ∈ {INTERMEDIATE, WET})` per (Race, Year) | Binary. Currently fires on 48% of (Race, Year) pairs — too generous (see §5). |
| `expected_stops_for_race` | `median(total_PitStop_per_driver)` per (Race, Year) | "1-stopper vs 2-stopper" race typology. |

### 3.3 Per-(Compound, Race) and per-Compound

Computed from completed stints (those that ended in `PitStop=1`).

| Column | Definition | Captures |
|---|---|---|
| `typical_stint_length` | Median completed-stint length per (Compound, Race) | Compound-and-track-specific stint norm. |
| `compound_max_life` | 95th percentile of completed-stint length per Compound | Soft upper bound on viable laps per compound. |

### 3.4 Per-(Race, Year, LapNumber) — field-wide statistics

Same lap across all drivers — captures phenomena that affect the whole field (safety cars, weather transitions, pit clusters).

| Column | Definition | Captures |
|---|---|---|
| `field_median_laptime` | Median `LapTime` across all drivers in (Race, Year, LapNumber) | Pace baseline for SC detection. |
| `field_max_laptime` | Max `LapTime` in same group | Anomaly detector. |
| `wet_compound_share` | Fraction of cars on INTER/WET this lap | Captures rain transitions. Field-wide deltas are stronger than per-row Compound. |
| `field_pace_ratio` | `field_median_laptime / quantile(field_median_laptime, 0.25 \| Race, Year)` | Lap-time spike vs the race-year's clean baseline. |
| `sc_likely` | `field_pace_ratio > 1.15` | Safety car / VSC detector. Fires on 1.7% of rows; Monaco tops the per-race ranking at 13% (matches reality). |
| `sc_prev_lap`, `sc_lap_minus2` | `sc_likely` shifted by 1 / 2 within (Race, Year) | "SC was on a lap or two ago." Pit reactions cluster on the lap *after* SC deployment. |
| `laps_since_sc` | Lap counter that resets on `sc_likely=1` | "How long since the last SC" — null until the first SC of the race. |

### 3.5 Per-(Race, Year, LapNumber) — pit-cluster features

| Column | Definition | Captures |
|---|---|---|
| `field_pit_share` | Leave-one-out: `(sum_pit_in_lap - PitStop) / (n_in_lap - 1)` | "What fraction of the field pitted this lap, excluding me." Strongest single new feature — quintile target rate climbs 2.2% → 38.2%. |
| `field_pit_share_prev_lap` | Field pit share shifted by 1 lap | Last lap's pit wave, viewed from this lap. |
| `field_pit_share_lap_minus_2` | Shifted by 2 | Two laps ago. |
| `field_pit_share_window_3` | 3-lap rolling mean of field pit share | Smoothed "is the field actively pitting around now". |

### 3.6 Per-(Driver, Race, Year)

| Column | Definition | Captures |
|---|---|---|
| `pits_so_far_this_race` | Cumulative sum of `PitStop` ordered by `LapNumber` within (Driver, Race, Year) | Stops the driver has already taken; combined with `expected_stops_for_race`, tells the model where this driver is in their planned strategy. |
| `stops_remaining_proxy` | `expected_stops_for_race - pits_so_far_this_race` | "Expected stops left." |
| `prev_stint_length` | Length of the previous completed stint for this (Driver, Race, Year) | Driver/team strategy consistency within a race. |

### 3.7 Per-row derived

#### Tyre-life features (using §3.3 aggregates)

| Column | Definition | Captures |
|---|---|---|
| `tyre_life_norm` | `TyreLife / typical_stint_length` | "How worn for this compound on this track" — clean ordinal even with the calibration caveat in §5. |
| `tyre_life_remaining` | `compound_max_life - TyreLife` | Soft "laps left on these tyres." |
| `cant_finish_on_current_tyres` | `LapsRemaining > tyre_life_remaining` | Hard physical constraint: must pit. Currently saturates (see §5). |

#### Positional pit signals (the undercut/overcut feature)

The neighbour at Position±1 in the same `(Race, Year, LapNumber)`. Because the synthetic data has up to 43 "drivers" sharing the same Position in a single lap (887 distinct codes squeezed into Position 1–20), we aggregate by mean across same-Position rows before looking up the neighbouring Position. The result is a soft version of the classic undercut signal, but still strongly informative:

| Column | Definition | Captures |
|---|---|---|
| `ahead_pitted_last_3` | Mean rolling-3-lap PitStop sum at Position−1 in this lap | Pit activity of the car(s) directly ahead. |
| `ahead_pitted_last_5` | Same, 5-lap window | Wider memory. |
| `behind_pitted_last_3`, `behind_pitted_last_5` | Same at Position+1 | Pit activity of the car(s) directly behind. |

Empirically: when neither neighbour pitted recently, pit-next rate is 7.8%; when both pitted, 26.3% — 3.4× lift, captured at a positional level that `field_pit_share` cannot.

#### Per-driver pace

| Column | Definition | Captures |
|---|---|---|
| `driver_pace_ratio` | `LapTime / field_median_laptime` | "Is this driver dropping off relative to the field this lap." Single-row, no rolling required. |

### 3.8 Per-(Driver, Race, Year) timeline features

Computed by sorting on `(Race, Year, Driver, LapNumber)` and applying rolling / shift operations within (Driver, Race, Year). These bypass the suspect raw `LapTime_Delta` and `Position_Change` columns (see §6).

| Column | Definition | Captures |
|---|---|---|
| `laptime_roll_mean_3`, `laptime_roll_mean_5` | Rolling 3 / 5-lap mean of `LapTime` | Pace baseline. |
| `laptime_roll_std_3`, `laptime_roll_std_5` | Rolling 3 / 5-lap std of `LapTime` | Pace stability — high std around SC laps, weather changes. |
| `laptime_diff_1` | `LapTime - LapTime.shift(1)` | The "true" lap-to-lap delta. |
| `laptime_diff_3` | `LapTime - LapTime.shift(3)` | 3-lap pace change — degradation rate proxy. |
| `position_change_3`, `position_change_5` | `Position - Position.shift(N)` | Recent overtakes / losses. |
| `position_roll_std_5` | Rolling 5-lap std of `Position` | Position volatility. |
| `cum_deg_diff_1` | `Cumulative_Degradation - shift(1)` | Per-lap degradation rate, regardless of how the original column is defined. |
| `stint_start_pace` | Mean `LapTime` over first 3 laps of this stint per (Driver, Race, Year, Stint) | Stint-baseline pace. |
| `laptime_vs_stint_start` | `LapTime - stint_start_pace` | Direct degradation signal that doesn't rely on `Cumulative_Degradation`. |

### 3.9 Cheap flags

| Column | Definition | Captures |
|---|---|---|
| `is_pre_season` | `Race == "Pre-Season Testing"` | 5.1% of train. Different strategy dynamics. |
| `is_synthetic_driver` | `Driver` matches `D` + 3 digits (length 4 starting with D) | 60% of train. Synthetic personas in the data generator. |
| `is_2023` | `Year == 2023` | 31% of train. Anomaly year with near-zero pit rate. |
| `lap_is_anomalous` | `LapTime > field_median_laptime * 1.5` | 0.3% of train. Per-row extreme-laptime flag, complementary to `sc_likely`. |

---

## 4. Empirical signal — quintile / contingency breakdowns

Verified on the train split. Useful for sanity-checking that each block is doing real work.

### `field_pit_share` quintile vs target

```
Q1 (~0%):  2.2% pit-next
Q2 (~3%):  10.6%
Q3 (~9%):  21.6%
Q4 (~18%): 27.1%
Q5 (~38%): 38.2%
```
Cleanly monotone, 17× spread. Strongest individual signal we added.

### Ahead/behind pitted in last 3 laps

```
neither pitted recently: 7.8%
only behind pitted:     19.9%
only ahead pitted:      20.3%
both pitted:            26.3%
```
3.4× lift. Captures the positional pit-wave that `field_pit_share` (field-wide) cannot.

### `tyre_life_norm` quintile

```
Q1 (norm 0.48): 3.5%
Q2 (norm 1.29): 10.1%
Q3 (norm 2.28): 20.1%
Q4 (norm 3.70): 30.6%
Q5 (norm 6.57): 35.9%
```
10× spread. The norm values themselves are inflated (denominator too small — see §5), but the ordering is clean.

### `sc_likely` per-race fire rates (top 5)

```
Monaco GP    13.2%
São Paulo GP  3.3%
Austrian GP   3.2%
Qatar GP      3.2%
Emilia Romagna 2.5%
```
Matches real F1 SC propensity. Monaco being 4× any other circuit is consistent with its history of disproportionately many safety cars.

### Target rate when SC is detected

`sc_likely=1`: 6.4% pit-next vs `sc_likely=0`: 20.1%. Lower under SC because SC pit decisions tend to fall on the *current* lap (`PitStop=1`, `PitNextLap=0`); the strategic lift lives in `sc_prev_lap`.

---

## 5. Known calibration issues

Three features have miscalibrated thresholds. None are bugs — they're still ordinal-monotonic and the underlying continuous columns are also exposed — but they're worth tightening before final submission.

1. **`cant_finish_on_current_tyres` saturates at 99.9%.** `compound_max_life` is the 95th percentile of *completed* stints, which underestimates max useful life because drivers who run to the end of the race without pitting are excluded. Fix: use `max` of completed stints, include uncompleted stints, or hard-code per-compound max life from real F1 norms (Hard ~50, Medium ~35, Soft ~25).
2. **`in_pit_window` fires on 83% of rows.** The p5/p95 band is too wide — only excludes the very first and very last laps. Fix: tighten to p15/p85, or use the actual concentrated-density region.
3. **`tyre_life_norm` denominator is too small.** `typical_stint_length` is the median *completed* stint per (Compound, Race), but the EDA noted the median stint length in this synthetic data is ~3 laps (lots of very short stints). Most stints exceed that, so the ratio inflates to a mean of 2.83. The ordering remains correct but the "fraction of typical life used" interpretation is broken. Fix: use mean rather than median, or filter stints to length ≥ 5 before taking the median.

A fourth note: `is_wet_race` triggers on 48% of (Race, Year) pairs because a single synthetic driver doing a single INTER lap flags the whole race-year. Consider gating on `wet_compound_share > 0.2 for at least N laps`.

---

## 6. Documented identities that don't hold

The dataset's column documentation implies three relationships:

1. `LapTime_Delta_i = LapTime_i - LapTime_{i-1}` (when `PitStop_i ≠ 1`)
2. `Position_Change_i = Position_{i-1} - Position_i`
3. `PitNextLap_i = 1 ⟺ PitStop_{i+1} = 1`

Verified empirically (see [src/features.py](../src/features.py) and the verification script in our chat history):

| Identity | Rows checked | Holds |
|---|---|---|
| (1) `LapTime_Delta` ≈ adjacent-lap delta | 230,750 | **0.03% within 1 ms**, 21% within 1 s, 53% off by > 5 s |
| (2) `Position_Change` ≈ adjacent-lap position delta | 260,744 | **7.6% exact match**, median diff 5, max 32 |
| (3) `PitNextLap_i = PitStop_{i+1}` | 182,243 (train, next-lap available) | **80.95% agreement**, ~19% disagreement, symmetric mismatches |

Interpretation:

- **`LapTime_Delta` and `Position_Change` are not adjacent-lap deltas.** Most likely they're computed against a different reference (stint baseline, starting grid, rolling window). We keep them as features in case the reference they actually use carries signal, but **do not rely on their documented semantics**. Our `laptime_diff_1` and `position_change_3/5` recompute the adjacent-lap versions explicitly from raw `LapTime` and `Position`.
- **`PitNextLap` is the ground truth label; `PitStop` is a noisy observed feature, not a derivable mirror.** Our pit-derived features (`field_pit_share`, `pits_so_far_this_race`, `ahead_pitted_*`, `behind_pitted_*`) are built on `PitStop` and inherit a few percent of noise. This sets a soft AUC ceiling but doesn't change the modelling approach.
- **No leakage worry.** If `PitNextLap = PitStop_{i+1}` were exact, including next-row `PitStop` as a feature would be a perfect leak. With 81% agreement, it isn't — meaning the modelling signal we're building is doing genuine work.

---

## 7. Output

Two parquet files:

- `data/train_features.parquet` — 439,140 rows × 65 columns (16 raw + 49 engineered)
- `data/test_features.parquet` — 188,165 rows × 64 columns (15 raw + 49 engineered, no `PitNextLap`)

The original CSVs are unchanged. All engineered columns have no missing values except where structurally undefined (e.g. `sc_prev_lap` is null on the first lap of each race; `prev_stint_length` is null for `Stint=1`).

To regenerate:

```bash
.venv/bin/python src/features.py
```

The script prints a sanity summary at the end: per-feature stats, the field-pit-share / tyre-life-norm / undercut / SC quintile breakdowns, and per-Race rankings for `pit_cost_seconds` and `sc_likely`.
