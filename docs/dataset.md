# Dataset — Column Reference

Kaggle Playground S6E5: F1 pit-stop prediction. Data is **synthetically generated from real F1 telemetry** (per the competition overview). Definitions below combine standard F1 domain meaning with what was observed in [eda.md](eda.md). Where a column is engineered and not officially documented, the interpretation is noted as such.

Files: `data/train.csv` (439,140 rows), `data/test.csv` (188,165 rows), `data/sample_submission.csv`. The test schema is the train schema minus `PitNextLap`.

---

## Identifier

### `id` — Int64
Row identifier. Train ids are `0 … 439,139`; test ids are `439,140 … 627,304`. No overlap. Use only to align predictions with `sample_submission.csv` — has no predictive value.

---

## Categorical context

### `Driver` — string
Driver code. Two encoding styles coexist:
- **Real F1 driver codes** — three-letter abbreviations (`VER`, `ALO`, `HAM`, `LEC`, …).
- **Synthetic driver IDs** — `D` + three digits (`D109`, `D552`, …). These are likely synthetic personas added by the data generator to enrich the training set.

887 unique drivers in train, 801 in test. Every test driver also appears in train (test ⊆ train), so no cold-start problem. High cardinality + skewed counts → target encoding or CatBoost-style native categorical handling is sensible.

### `Compound` — string
Tyre compound on the lap. Five values, identical set in train and test:
- `SOFT` — softest dry compound, fastest but degrades quickly.
- `MEDIUM` — balanced dry compound (most common in the data, ~48% of rows).
- `HARD` — durable dry compound; long stints, latest pit windows.
- `INTERMEDIATE` — for damp / drying tracks.
- `WET` — for full wet conditions.

Empirically, base `PitNextLap` rate by compound is `HARD 33% > SOFT 19% > INTER 15% > MEDIUM 10% > WET 3%` — Hard is run longest and so produces the most "pit next lap" labels at high `TyreLife`.

### `Race` — string
Grand Prix name (e.g. `Monaco Grand Prix`, `Dutch Grand Prix`). 26 unique values, identical set in train and test. **Includes `Pre-Season Testing`** as a "race" — its strategy/pit dynamics are not race-representative, worth flagging.

### `Year` — Int64
Season year, `2022 – 2025`. Distribution differs noticeably from the on-track 2026 timeline of the competition; the data appears to span 4 modelled seasons.

**Important quirk:** 2023 has the most rows but `PitStop` rate of just 1.2% and `PitNextLap` rate of 0.96% (vs 27–30% other years). This is a synthetic-data artifact, not real-world strategy — treat 2023 as a near-zero-label regime in the training set.

---

## Lap state

### `LapNumber` — Int64
1-indexed lap counter for the driver in this race. Range `1 – 78`. Together with `Race`, `Year`, `Driver` it uniquely identifies a row (verified — no duplicate keys in either file).

### `Stint` — Int64
Stint number within the race, 1-indexed. A new stint starts after every pit stop, so `Stint = 1` is the opening stint, `Stint = 2` the next, etc. Range `1 – 8` (most rows are stints 1–3). Increment from one row to the next implies a pit stop happened in between.

### `TyreLife` — Float64
Number of laps the **current set of tyres** has done. Range `1 – 77`, median 12. Resets when a stint changes. The **single strongest predictor** in the data — `PitNextLap` rate climbs monotonically from 5% (life ≤ 5) to 72% (life > 50). Strong interaction with `Compound` (Hard tolerates much higher life than Soft).

### `PitStop` — Int64 (0/1)
Whether the driver pits **on this lap**. Distinct from the target, which is about the *next* lap. Mean ≈ 13.6% (excluding the 2023 anomaly where it's near zero). Note: `PitStop=1` only co-occurs with `PitNextLap=1` in 24.8% of cases — they describe consecutive-lap events, so the relationship is structural, not redundant.

### `Position` — Int64
On-track running position at the end of the lap. Range `1 – 20`. Has near-zero direct correlation with the target (rate ≈ 0.15–0.21 across all positions) — pit decisions are tyre/strategy-driven, not position-driven.

### `LapTime (s)` — Float64
Lap time in seconds. Median 90.5s, p99 124.9s, but **max 2507s** — the long tail captures safety-car, virtual safety-car, red-flag, and outlap/inlap pit cycles. Heavy-tailed; either clip or add an "abnormal lap" flag.

---

## Engineered / derived features

These columns are not directly observed F1 telemetry — they are computed by the data generator. Interpretation below is based on observed ranges and the column names; treat as best-guess unless the competition documentation says otherwise.

### `LapTime_Delta` — Float64
**Interpretation:** lap-time delta to a reference (likely the driver's previous lap, the stint-fastest lap, or a rolling baseline). Median −0.30s, p1/p99 ≈ ±40s, but extremes reach ±2400s. Negative values mean "faster than the reference," positive values "slower."

The huge tails again come from safety-car / red-flag laps where the lap is dramatically slower than baseline. Near-zero correlation with target (-0.005). Treat as noisy; consider clipping to ±5s for "normal" laps and flagging the rest.

### `Cumulative_Degradation` — Float64
**Interpretation:** running tally of tyre degradation expressed as a lap-time delta from some baseline (likely the start-of-stint pace or a fresh-tyre reference). Range `-274 … 2412`, median **negative** (~−21).

The negative median is the key clue: degradation here is **signed**, not unsigned. A negative value means the driver is *currently faster than the baseline* — fuel-burn-off, track evolution, or the baseline being a slow installation lap can produce this. Positive values indicate the tyre has slowed the driver down vs baseline. Correlation with target is **−0.17** (more-degraded-than-baseline → less likely to pit *next* lap, which is counter-intuitive and likely a side-effect of the negative-median convention combined with the synthetic 2023 regime). Worth careful inspection before trusting it as a physics signal.

### `RaceProgress` — Float64
Fraction of the race completed, in `[0.013, 1.0]`. **Effectively `LapNumber / total_laps_for_this_race`** — Pearson correlation with `LapNumber` is 0.965, so the two are near-redundant. Drop one for linear models; trees can use both safely.

`PitNextLap` rate vs `RaceProgress` traces the classic mid-race pit window: ~6% in lap 1, peaks at ~39% in the 50–70% window, falls to ~5% in the final 10%.

### `Position_Change` — Float64
**Interpretation:** position delta over some short window (likely change since lap-1 or change since previous lap). Range `−18 … +18`, median 0, p1/p99 ±11. Negative = gained places (lower position number is better), positive = lost places. Mild negative correlation with `Position` (−0.32) — leaders gain less.

Weak direct signal to the target (corr 0.046).

---

## Target

### `PitNextLap` — Float64 (0/1, train only)
**Target.** 1 if the driver pits on lap `LapNumber + 1`, else 0. Encoded as float but takes only `{0.0, 1.0}`. Class balance: 19.90% positives in train.

Submission format: predicted **probability** in `[0, 1]`, evaluated by **ROC-AUC**. The `sample_submission.csv` file contains `id,PitNextLap` with all-zero predictions.

**Caveat:** the target is *not* a perfect mechanical function of the next row's `Stint` increment in train. On rows where the next lap is also in train, only 22,180 of 73,802 `PitNextLap=1` rows have a next-row stint increment, and 77,019 `PitNextLap=0` rows do show a next-row stint increment. The target carries genuine label noise relative to the structural columns — model the label, don't try to reverse-engineer it from `Stint`.

---

## Quick reference — typical values

| Column | Type | Range / cardinality | Median |
|---|---|---|---|
| `id` | Int64 | 0 – 627,304 | — |
| `Driver` | str | 887 unique (train) | — |
| `Compound` | str | 5 values | — |
| `Race` | str | 26 values | — |
| `Year` | Int64 | 2022 – 2025 | 2024 |
| `PitStop` | Int64 | 0 / 1 | 0 (mean 0.136) |
| `LapNumber` | Int64 | 1 – 78 | — |
| `Stint` | Int64 | 1 – 8 | — |
| `TyreLife` | Float64 | 1 – 77 | 12 |
| `Position` | Int64 | 1 – 20 | — |
| `LapTime (s)` | Float64 | 67.7 – 2507.6 | 90.5 |
| `LapTime_Delta` | Float64 | −2403.9 – 2423.9 | −0.30 |
| `Cumulative_Degradation` | Float64 | −274.6 – 2412.0 | −21.0 |
| `RaceProgress` | Float64 | 0.013 – 1.0 | 0.27 |
| `Position_Change` | Float64 | −18 – 18 | 0 |
| `PitNextLap` | Float64 | 0 / 1 (train only) | 0 (mean 0.199) |

No missing values in any column of either file.
