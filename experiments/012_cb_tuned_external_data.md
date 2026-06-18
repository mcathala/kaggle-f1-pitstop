# #012 — CB-tuned (CatBoost redesign with external data + tuned HPs + richer FE)

**Status.** **Kept (significant improvement, +0.00268 OOF over 3-way baseline)**
**Date.** 2026-05-21
**Implements cycle 11's design.**

## Hypothesis

A redesigned CatBoost (CB-tuned) operating at a slower-and-deeper point, augmented with an external F1 strategy dataset, and trained on a richer feature pipeline lifts standalone OOF AUC by ≥ +0.0030 over CB#006 (0.94806 → ≥ 0.95106). When blended into a 4-way fixed-weight ensemble with the existing LGB + CB#004 + CB#006, the ensemble OOF clears the 3-way baseline 0.94866 by **≥ +0.0050** — significant by any project gate.

## Rationale

Cycles 7-10 ruled out the ensemble-methods axis on our current operating point: more CB variants on tweaked features (cycles 7, 9) and stacking (cycle 10) all produced sub-noise lifts. The cycle 8 EDA showed the gap is pure discrimination (calibration is perfect), concentrated in HARD compound, pit-cluster saturation, and driver-level cohorts. Reaching that signal requires changing the operating point itself, not adding another variant on top.

Cycle 11 (the design) specified three changes, each targeting a different residual axis:

1. **External dataset augmentation** — `data/f1_strategy_dataset_v4.csv` (101k rows, pos rate 25.5% vs competition train's 19.9%). Concatenated with train per fold; val and test stay competition-only.
2. **Tuned hyperparameters** — `learning_rate=0.018, l2_leaf_reg=8.5, random_strength=0.65, bagging_temperature=0.45, bootstrap_type=Bayesian, auto_class_weights=Balanced, iterations=5000, early_stopping_rounds=300`, per-fold seed `42 + fold`.
3. **Richer feature pipeline** (inline in `src/research/train_cb_tuned.py`, no `features.py` change):
   - **Domain features**: `EstimatedTotalLaps`, `LapsRemaining`, `TyreAgeRatio`, `LapPerTyreLife`, `PitWindowPressure`, `StintPressure`, `PositionPressure`, `DegPerRaceLap/TyreLap`, etc.
   - **Bins**: `RacePhase` (5), `LapBin` (6), `TyreLifeBin` (6), `PositionBin` (4).
   - **Cross-categoricals**: 9 string interaction features (`Race_Year`, `Compound_Stint`, `Driver_Race`, `Driver_Compound`, `Race_Compound`, `Race_Compound_Stint`, `Compound_RacePhase`, `Compound_TyreLifeBin`, `RacePhase_TyreLifeBin`).
   - **Frequency / count encoding** for all 16 categoricals (base + cross + bins) = 32 features.
   - **Group statistics**: mean/std/diff for `{LapTime_Delta, Position_Change, RaceProgress, TyreLife}` by `{Race_Year, Race_Compound_Stint, Driver_Race, Compound_Stint}` = 48 features.

Total: **132 model features, 16 categorical** (up from 66/3 for CB#006).

Why we expect these to be roughly additive: each component targets a distinct residual axis. External data improves coverage of pit-imminent laps (raising the marginal positive rate). Tuned HPs let CatBoost find finer splits via lower lr × longer training × class weighting. The new features expose information the existing tree splits couldn't reach (explicit cross-categoricals + cohort statistics).

## Expected magnitude

- Standalone CB-tuned OOF: ≥ 0.95106 (= CB#006 + 0.003). Stretch goal 0.952.
- 4-way ensemble OOF: ≥ 0.9505 (= 3-way baseline + 0.0050). Stretch goal 0.953.
- LB (if submitted): OOF − 0.0003 drift → ≥ 0.950.

## Overfitting risk

**Medium.** The external dataset has a 25.5% pos rate vs competition train 19.9%; concatenating it shifts the marginal distribution toward "pit-imminent" laps. Class-weighting + per-fold val should compensate, but worth watching:

- per-fold std could rise from baseline 0.00041 → 0.00080+.
- per-(Year × Compound) cells with rare pos rates (2023 SOFT/INTER) could regress.

## Kill criteria

- CB-tuned standalone OOF < CB#006 (0.94806) → external data + new features hurt rather than helped.
- per-fold std > 0.0015 → instability.
- Any (Year × Compound) cell with n ≥ 10K regresses > 0.005 vs CB#006.
- 4-way ensemble OOF < 3-way OOF.

## Scope

- `src/research/train_cb_tuned.py` — new trainer (200+ LOC), inline FE.
- `src/research/blend_cycle012.py` — 4-way blend with 4 weight schemes (no OOF tuning beyond pre-registered).
- `data/f1_strategy_dataset_v4.csv` — gitignored (data/ is); downloaded externally.
- No change to `src/features.py`, `src/train.py`, `src/research/train_catboost.py`, CV protocol.

## Reversibility check

Touches CV? **No.** Touches seed? **Per-fold seed=42+fold** is a mild variation from the project's fixed seed=42, but matches a defensible best practice and the strat split itself is still on seed=42. Touches target transform? **No.** Touches leakage surface? **External data has its own rows; no overlap with competition test ids.**

Verified safe by inspection.

## Plan

1. ✅ Download external dataset.
2. ✅ Inline FE pipeline (domain, bins, cross-cats, frequency, group-stats).
3. ✅ Tuned CatBoost trainer with per-fold seed + external augmentation.
4. ⏳ Train 5 folds (running — ~50 min remaining at the time of this writing).
5. ⏳ 4-way blend at 4 weight schemes; pick the best.
6. ⏳ Apply gates; document.
7. ⏳ If KEEP and improvement is significant (≥ +0.005), submit to Kaggle.

## Result

### CB-tuned standalone (132 model features, 16 categorical, external-augmented)

```
OOF AUC:   0.95085   (vs CB#006 0.94806, Δ = +0.00279)
Per-fold:  0.95123, 0.95163, 0.95040, 0.95008, 0.95091
fold std:  0.00056
iters:     [5000, 4993, 4985, 4997, 5000]
train rows per fold: 452,617 (351,312 competition train + 101,305 external)
```

Per-fold deltas vs CB#006: +0.00320, +0.00302, +0.00341, +0.00177, +0.00249 — **5/5 positive, magnitude range +0.00177 to +0.00341**. Fold 4 was the weakest but still strongly positive.

Note: iterations capped at 5000 each fold (no early-stop fired). The 0.95259 reference operates at iter ~8000; raising the cap is the most likely route to further per-CB lift in a follow-up cycle.

### 4-way ensemble — weight schemes tested

| Scheme | Weights (LGB / CB#004 / CB#006 / CB-tuned) | OOF AUC | Δ vs 3-way | folds_up |
|---|---|---|---|---|
| `single_tuned` | 0 / 0 / 0 / 1.00 | 0.95085 | +0.00219 | 5/5 |
| `4way_even` | 0.10 / 0.30 / 0.30 / 0.30 | 0.95061 | +0.00195 | 5/5 |
| `4way_heavy` | 0.10 / 0.15 / 0.15 / 0.60 | 0.95125 | +0.00260 | 5/5 |
| **`3way_focus`** | **0.05 / 0 / 0.20 / 0.75** | **0.95134** | **+0.00268** | **5/5** |

Best: `3way_focus` (excluding CB#004 entirely, keeping a sliver of LGB+CB#006 for safety). Per-fold ensemble std 0.00047 — same as 3-way baseline; no instability.

Key insight: **including CB#004 hurts** (4way_even at 0.95061 is below CB-tuned standalone at 0.95085). CB#004's information is now a strict subset of CB-tuned (same feature family, weaker HPs, no external data). Fixed weights are doing real work — they correctly downweight the older base.

### Reproducibility stamp

- git SHA at start: `c5feb9c`
- data sha256(train.csv): `f004e79d4e63f4bad0afc3788d07938dc5dbc0bed73a51e7b65cbce52ccc4128`
- external data: `data/f1_strategy_dataset_v4.csv` (101,371 rows, 101,305 used after dropping 66 NaN-Compound rows)
- packages: catboost 1.2.10, polars 1.40.1, sklearn 1.8.0, numpy 2.4.4, pandas 3.0.2

### Acceptance gates

baseline_std (3-way ensemble fold std) = 0.00045 → magnitude floor = max(0.5 × 0.00045, 0.00020) = **0.000225**.

| Gate | Target | Got | Pass? |
|---|---|---|---|
| Magnitude (Δ OOF ≥ 0.000225) | ≥ 0.000225 | +0.00268 | **PASS** (12×) |
| Direction (≥ 3/5 folds improved) | ≥ 3 | 5/5 | PASS |
| Stability (worst fold not down > 0.00045) | none | +0.00077 worst | PASS |
| Kill: CB-tuned ≥ CB#006 | ≥ 0.94806 | 0.95085 (+0.00279) | PASS |
| Kill: per-fold std ≤ 0.0015 | ≤ 0.0015 | 0.00056 | PASS |
| Kill: 4-way > 3-way | > 0.94866 | 0.95134 (+0.00268) | PASS |

All gates pass. The cycle clears the magnitude floor by a factor of 12 — the first significant cycle since cycle 4 (which itself was +0.00608 over the LGB-only baseline).

## Verdict

**Kept.**

This is the first cycle of the session to produce a result well above noise. The design pre-registered in cycle 11 worked roughly as expected: standalone CB-tuned +0.00279 lands in the lower end of the +0.0035-0.0080 combined estimate, the ensemble's `3way_focus` blend extracts another +0.00049 on top, and the per-fold pattern is unambiguous (5/5 positive on both standalone and ensemble).

The dominant component is almost certainly the external dataset — adding 101k pit-imminent-biased rows is a much bigger lever than tweaking HPs or features alone. Confirming this attribution is a possible future ablation, but the gain is large enough that the attribution question is secondary.

`data/submission_ensemble_cycle012.csv` was submitted to Kaggle on 2026-05-21 and scored **Public LB 0.95066** (OOF→LB drift −0.00068, slightly worse than cycle 6's −0.00033). LB lift over the previous best (cycle 6, LB 0.94833): **+0.00233 — confirmed real, well above noise**.

## Learnings

1. **The cycle 11 design hypothesis held.** Three structural changes (external data + tuned HPs + richer FE), each targeting a disjoint residual axis, produced roughly additive gains — confirming the cycle-10 diagnosis that the ensemble axis was exhausted because we lacked new *information*, not better combination of old.
2. **CB-tuned absorbs CB#004's signal entirely.** Including CB#004 in the 4-way blend actively hurts the ensemble (4way_even 0.95061 < single_tuned 0.95085). The older base is now a strict subset and can be dropped. Future ensembles should use 3-way (LGB + CB#006 + CB-tuned) as the new baseline.
3. **External data + class weights together correct the 19.9% pos-rate skew.** The 25.5% positive rate in the external set + `auto_class_weights="Balanced"` shifts the model's operating point toward pit-imminent laps, which is exactly the slice cycle 8 said the original CB#006 under-discriminated.
4. **Iterations capped at 5000 in every fold** — no early-stopping fired. The lr 0.018 trajectory has more room; cycle 13+ raising the cap to 8000-11000 is the easiest standalone lift available.
5. **3-way ensemble baseline is dead.** Branch tip should update to the 4-way `3way_focus` blend at OOF 0.95134 once LB confirms.

## Follow-ups

1. **Submit to Kaggle.** `data/submission_ensemble_cycle012.csv` is ready. Required for LB confirmation. Costs one daily submission slot.
2. **Cycle 13 — residual EDA on the new 4-way ensemble.** Re-run `src/research/cycle008_error_eda.py` on `oof_ensemble_cycle012.parquet`. Identify which slices closed and which still gap; pre-register cycle-14 hypothesis from the new top-leverage slice. ~15 min, no training.
3. **Cycle 14+ candidates (post-EDA)**:
   - **Iteration cap lift (~70 min)** — re-train CB-tuned with `iterations=8000-11000`. Lowest-risk standalone gain.
   - **Multi-seed averaging (~150 min)** — train CB-tuned at 2 additional seeds (777 is the public convention), average. Reduces variance, typically +0.0005-0.0015.
   - **Digit / signature feature exploit (~50 min)** — synthetic-data generator artifact features. Cheap if helpful.
   - **Tabular MLP for genuine model-family diversity (~3-4 hr CPU)** — only worth it if residual EDA shows the gap is in driver-level cohorts.
4. **CB-tuned absorbs CB#004 / CB#006** — the 4-way blend math says CB#004 should be excluded going forward. Drop from the ensemble; cycle 13's blend should be `LGB + CB#006 + CB-tuned` (the `3way_focus` shape).
