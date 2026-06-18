# EDA — Kaggle Playground S6E5: F1 Pit-Stop Prediction

Reproduce: `.venv/bin/python /tmp/eda_script.py` and `/tmp/eda_script2.py` (raw output: `/tmp/eda_output.txt`, `/tmp/eda_output2.txt`).

## TL;DR

- **Task** — binary classification, predict `PitNextLap` (probability the driver pits on the *next* lap). Metric is ROC-AUC.
- **Sizes** — train 439,140 × 16, test 188,165 × 15 (≈70/30 split). No missing values anywhere.
- **Class balance** — 19.90% positives. Mild imbalance, AUC metric so no resampling needed.
- **Train/test split is row-level, not group-level.** The same `(Race, Year, Driver)` appears in both train and test with disjoint `LapNumber`s — test laps are interleaved gaps inside the same race timelines you see in train. This shapes CV strategy (see §8).
- **2023 is anomalous** — PitNextLap rate 0.96% (vs 27–30% other years) and PitStop rate 1.24% (vs 18–20%). Looks like a synthetic-data artifact, not a real-season signal. Treat year as a feature *and* be ready to model 2023 separately.
- **Strongest individual signals** — `TyreLife` (ρ=0.27), `LapNumber` (0.27), `Stint` (0.20), `RaceProgress` (0.19), `Cumulative_Degradation` (−0.17). `LapNumber` and `RaceProgress` are 0.965 correlated — nearly redundant.
- **Strong interaction** — `TyreLife × Compound`. Hard at 50+ laps → 71% pit rate; Medium at the same range → 61%; Wet barely ever pits.
- **Outliers** — `LapTime (s)` median 90.5s, max 2507s; `LapTime_Delta` and `Cumulative_Degradation` have ±2000+ spikes (p1/p99 are ±40/±200). Almost certainly safety-car / red-flag laps. Decide: clip, log-transform, or flag.

---

## 1. Files & schema

| File | Rows | Cols |
|---|---|---|
| `train.csv` | 439,140 | 16 |
| `test.csv` | 188,165 | 15 |
| `sample_submission.csv` | 188,165 | 2 (`id,PitNextLap`) |

Columns (test = train minus `PitNextLap`):

| Column | Dtype | Notes |
|---|---|---|
| `id` | Int64 | train 0–439,139, test 439,140–627,304, no overlap |
| `Driver` | str | mix of 3-letter codes (`VER`, `ALO`, …) and synthetic IDs (`D109`, `D552`, …). 887 unique in train, 801 in test, **all test drivers present in train** |
| `Compound` | str | `MEDIUM` / `HARD` / `SOFT` / `INTERMEDIATE` / `WET` (5 values, identical set in test) |
| `Race` | str | 26 races, identical set in train and test (incl. `Pre-Season Testing`) |
| `Year` | Int64 | 2022–2025 |
| `PitStop` | Int64 | 0/1, "did the driver pit on **this** lap". Feature, not target. |
| `LapNumber` | Int64 | 1–78 |
| `Stint` | Int64 | 1–8 (stint number within race) |
| `TyreLife` | Float64 | 1–77 laps, median 12 |
| `Position` | Int64 | 1–20 |
| `LapTime (s)` | Float64 | median 90.5s, but extreme tail to 2507s |
| `LapTime_Delta` | Float64 | gap to reference (median −0.30, but ±2400 outliers) |
| `Cumulative_Degradation` | Float64 | median −21.0, p1 −205, p99 122, max 2412 |
| `RaceProgress` | Float64 | 0.013–1.0, basically `LapNumber / race_total_laps` |
| `Position_Change` | Float64 | bounded −18 to +18 |
| `PitNextLap` | Float64 | **target**, 0/1 |

**No missing values in any column** of either file.

---

## 2. Target distribution

```
PitNextLap = 1 :  87,381   (19.90%)
PitNextLap = 0 : 351,759   (80.10%)
```

Mildly imbalanced — fine for AUC. Don't oversample unless using a model that needs it.

---

## 3. Year distribution & the 2023 anomaly

```
Year   train       test       PitStop_rate   PitNextLap_rate
2022   82,989      35,348     0.187          0.267
2023   136,147     58,160     0.012          0.010   ← anomaly
2024   127,110     54,532     0.192          0.295
2025   92,894      40,125     0.196          0.284
```

2023 has the most rows but ~30× fewer positives. Both the input feature `PitStop` and the target `PitNextLap` are suppressed in 2023 across **every** compound (Hard 0.8% vs 47–54% other years). This is a generator artifact, not a real strategy shift. Implication:

- Use `Year` as a model feature.
- Consider whether removing 2023 helps or hurts on the public LB. It likely contributes to test-set AUC because that distribution shift is also in test.
- Do not use 2023 to fit "physics" (compound × tyre-life curves) — its labels are largely zeros and will flatten the relationship.

---

## 4. Categorical breakdowns

### Compound

```
Compound       train    test    PitNextLap_rate(train)
MEDIUM         211,141  90,897  0.101
HARD           170,518  72,677  0.328   ← highest base rate
SOFT            38,744  16,615  0.193
INTERMEDIATE    17,382   7,408  0.152
WET              1,355     568  0.025
```

Hard has by far the highest pit-next rate — drivers run hards long and pit them when worn. Wet barely pits (likely full-distance under safety conditions).

### Race

26 unique races including `Pre-Season Testing`. **Identical race set** in train and test. Top by row count:

```
Dutch GP     24,462
Mexico City  23,672
Pre-Season   22,492
Hungarian    22,481
Monaco       21,539
…
```

`Pre-Season Testing` is included as a "race" — its strategy/pit dynamics are very different from a real race. Worth a binary `is_test_session` flag.

### Driver

- 887 unique drivers in train, 801 in test, **801 of 801 test drivers appear in train** (test ⊆ train).
- 86 train-only drivers, 0 test-only.
- Two encoding styles coexist: real F1 codes (`VER`, `ALO`, `ALB`) and synthetic IDs (`D109`, `D552`). The synthetic ones likely correspond to drivers that appear sparsely (few laps each). Worth a `is_synthetic_driver` flag (`startswith("D") and len==4`) and target encoding.

---

## 5. Numerical distributions & outliers

| Column | min | p1 | p50 | p99 | max |
|---|---|---|---|---|---|
| `LapTime (s)` | 67.7 | 70.7 | 90.5 | 124.9 | **2507.6** |
| `LapTime_Delta` | −2403.9 | −40.3 | −0.30 | 30.9 | **2423.9** |
| `Cumulative_Degradation` | −274.6 | −205.0 | −21.0 | 122.2 | **2412.0** |
| `Position_Change` | −18 | −11 | 0 | 11 | 18 |

The three "huge tail" columns (LapTime, LapTime_Delta, Cumulative_Degradation) are dominated by safety-car/red-flag laps where lap times spike. Two practical options:

1. **Clip** at p1/p99, keep raw + clipped versions.
2. **Add an "abnormal lap" indicator** — e.g. `LapTime > p99` flag. The model can use the spike pattern itself as a signal (cars often pit the lap after a safety car).

`Cumulative_Degradation` having a negative median is suspicious — likely a delta-from-baseline already, where negative means "faster than the start of the stint." Don't assume this is "absolute degradation."

---

## 6. Correlations with target

```
TyreLife                  0.274
LapNumber                 0.267
Stint                     0.198
RaceProgress              0.185
Cumulative_Degradation   -0.167
Year                      0.125
PitStop                   0.049
Position_Change           0.046
LapTime (s)              -0.034
Position                  0.021
LapTime_Delta            -0.005
```

Pairwise notes (from full corr matrix):

- `LapNumber ↔ RaceProgress` = **0.965** — nearly redundant. Drop one or use just one in linear models; trees are fine with both.
- `LapNumber ↔ Stint` = 0.724, `LapNumber ↔ TyreLife` = 0.648 — also strong.
- `Position ↔ Position_Change` = −0.316 — leaders gain less.
- `LapTime ↔ Cumulative_Degradation` = 0.206 — slow laps correlate with high degradation, sane.

---

## 7. Pit dynamics (what actually drives the target)

### TyreLife is the dominant single feature, and it interacts with Compound

```
TyreLife bucket    PitNextLap_rate    n
(0, 5]             0.051              90,638
(5, 10]            0.134              92,543
(10, 15]           0.196              87,313
(15, 20]           0.263              66,479
(20, 25]           0.311              45,310
(25, 30]           0.348              28,318
(30, 40]           0.393              21,878
(40, 50]           0.427               5,280
(50, 70]           0.721               1,281
```

Smooth monotone climb to ~43% then a step to 72% at 50+ laps. The 50+ region is where strategists almost always pit.

By compound:

| Compound | (0,5] | (5,10] | (10,15] | (15,20] | (20,30] | (30,50] | 50+ |
|---|---|---|---|---|---|---|---|
| HARD | 0.083 | 0.229 | 0.320 | 0.389 | 0.427 | 0.429 | **0.711** |
| MEDIUM | 0.030 | 0.074 | 0.108 | 0.150 | 0.192 | 0.308 | **0.610** |
| SOFT | 0.103 | 0.201 | 0.243 | 0.235 | 0.224 | 0.180 | 0.000 |
| INTER | 0.043 | 0.109 | 0.164 | 0.203 | 0.232 | 0.392 | 0.556 |
| WET | 0.010 | 0.024 | 0.025 | 0.047 | 0.070 | — | — |

SOFT plateaus around 24% (drivers either pit early or never quite stretch them long), HARD/MEDIUM rise consistently. Strong case for a `Compound × TyreLife` interaction (or just let a tree model find it).

### RaceProgress: classic pit window shape

```
0.0–0.1   0.062
0.1–0.2   0.108
0.2–0.3   0.175
0.3–0.4   0.277
0.4–0.5   0.352
0.5–0.6   0.386       ← peak pit window
0.6–0.7   0.389
0.7–0.8   0.287
0.8–0.9   0.149
0.9+      0.051       ← almost no one pits in last 10%
```

### Position has near-zero impact

Rate is roughly flat across positions 1–20 (range 0.15–0.21). Pit decisions are about tyres/strategy, not running order.

### Stint and laps-per-stint

Stint count per (driver, race, year): 1 → 2 → 3 → … → 8. 2-stoppers and 3-stoppers are common (Stint distribution: 216k / 130k / 69k / 19k / 4k / …). Laps-per-stint median 3, mean 3.9 — lots of very short stints, again partly a synthetic-data feature.

### `PitStop` (current-lap pit) ≠ `PitNextLap` (target)

`PitStop=1` rate: 13.6% of rows. Yet the target rate when `PitStop=1` is only 24.8% — pitting *now* doesn't strongly predict pitting *next*. The two columns describe consecutive-lap events; you can use `PitStop` as a feature but expect it to be a weak direct signal. The structural relationship between the two is preserved only when both rows of the (driver, race) sequence are in the same set, which often they are not (see §8).

---

## 8. Train/test split structure (important for CV)

The split is **row-level random, not group-level**:

- 40,869 unique `(Race, Year, Driver)` groups in train; 37,038 in test; **35,674 overlap**.
- For overlapping groups, train and test contain **disjoint sets of LapNumbers** from the same race timeline (verified on samples — laps 1, 5, 6, 12, … in train; laps 3, 10, 13, 24, … in test for the same race-driver).
- 5,195 groups appear only in train (likely retired drivers / very short stints) and 1,364 only in test.

Implications:

1. **Do not GroupKFold by `(Race, Year, Driver)`.** That's a stricter split than what Kaggle scores you on, so your CV will under-estimate test AUC.
2. **Random KFold on rows is the correct mirror** of the public/private split. Standard `StratifiedKFold` on `PitNextLap` is fine.
3. **Adjacent-lap leakage** — for any test lap, you usually have access to lap-1 and lap+1 of the same driver-race in **train**. This is a legitimate feature source: per-driver-race timelines combining train+test (sorted by `LapNumber`) let you compute lagged/lead features (e.g. `TyreLife - prev_TyreLife`, `LapTime - rolling_mean_3`). The competition allows test features for feature engineering since labels are not used.
4. **`PitStop` of the next-row in train**, when it's in train, is essentially a label leak for `PitNextLap` — but only for ~half the rows, so don't model on it directly. Useful for sanity checks though.

Sanity check on the target's mechanical definition — for the 398,271 train rows where the *next* lap is also in train:

```
PitNextLap=1 with next-row Stint+1   :  22,180   ← clean transitions
PitNextLap=1 with no Stint increment :  51,622
PitNextLap=0 with next-row Stint+1   :  77,019   ← target undercount
```

`PitNextLap` does **not** strictly equal "next row's stint > current stint." There's noise (or the synthetic generator decoupled them in places). Treat the label as ground truth, but don't expect clean recovery from `Stint` shifts.

---

## 9. Feature engineering shortlist

Ranked rough by expected lift:

1. **Lag/lead features per (Driver, Race, Year)** — `LapTime` rolling mean/std (3, 5 laps), `Position` change vs N laps ago, `TyreLife` slope. Build on combined train+test.
2. **`Compound × TyreLife` interaction** explicitly (or rely on tree splits).
3. **`StintCompoundLapsLeft`** — for each row, how many laps remain in this stint by looking ahead in the (driver, race) timeline (only valid because of the row-level split). Strong but borderline leak — verify on CV.
4. **Driver target-encoding** (out-of-fold) — 887 drivers, many low-volume.
5. **`is_synthetic_driver`** flag — `Driver.startswith("D")` heuristic.
6. **`is_test_session`** = `Race == "Pre-Season Testing"`.
7. **Year-2023 indicator** — keeps trees from blending its abnormal labels with other years.
8. **Outlier flags** for `LapTime > p99`, `Cumulative_Degradation` outside ±150 (safety-car/anomaly).
9. **Race-specific features** — track length proxy = `max(LapNumber)` per race; stint-strategy frequencies per race.
10. **Drop or merge `RaceProgress`** for linear models (redundant with `LapNumber`).

---

## 10. Modelling notes / what to try first

- **Baseline**: gradient boosted trees (LightGBM / XGBoost / CatBoost) on raw + a handful of lag features, 5-fold stratified CV. With these features the corr-with-target structure suggests a starting AUC in the 0.85–0.90 range is plausible; the public benchmark will tell.
- **CatBoost** is a natural fit because of `Driver` (887 levels) and `Race` (26 levels). Otherwise target-encode.
- **Calibration matters for AUC less** but if you ensemble, calibrate logits before averaging.
- **Don't mock the temporal split** — random KFold on rows mirrors the actual evaluation, see §8.
- **Watch 2023 in your CV folds** — its extreme label rate makes it a small-effective-N year. Stratify by `Year × PitNextLap` to keep folds comparable.
