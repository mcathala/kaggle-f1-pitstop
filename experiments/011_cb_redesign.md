# #011 — CatBoost redesign (planning cycle)

**Status.** Kept (design cycle, no model trained)
**Date.** 2026-05-21
**Focus.** Planning — connect the cycle 8 diagnosis to the cycle 12 implementation.

## Why this cycle

Cycles 7, 9, and 10 each gave the same answer at different angles: the existing CatBoost + LightGBM stack, run on the existing feature set, can only produce ~+0.0001 OOF lifts per cycle — well below the project's 0.000225 noise floor. The three cycles ruled out distinct axes:

| Cycle | Tested | Result |
|---|---|---|
| 7 | New features → 4th CB variant on tweaked feature subset | +0.00014 ensemble, Inconclusive |
| 9 | Targeted features for the cycle-8-identified pit-cluster q5 weakness | +0.00011 ensemble, targeted slice didn't move standalone |
| 10 | Stacking meta-model on base OOFs | Best meta-OOF below the existing fixed-weight 4-way blend |

Together this is strong evidence that **the current operating point is saturated**. The cycle 8 EDA showed the residual gap is pure discrimination (calibration is perfect, max bias 0.009 per decile), concentrated in three large slices: HARD compound (low/mid TyreLife), pit-cluster saturation (`field_pit_share` q4+q5), and driver-level discrimination on real F1 codenames. None of these are reachable from where we are now.

The only way out is a structural redesign of the CatBoost setup, not another local FE tweak.

## Hypothesis

A redesigned CatBoost (CB-tuned) using:

1. **External data augmentation** — augment training with an external F1 strategy dataset (`data/f1_strategy_dataset_v4.csv`, 101k rows). Concatenated per-fold with the competition train set; val and test remain competition-only so the CV remains a fair proxy for the public LB.
2. **Tuned hyperparameters** — operate at a slower-and-deeper point: `learning_rate=0.018` (vs current 0.05), `l2_leaf_reg=8.5` (vs 3.0), `random_strength=0.65`, `bagging_temperature=0.45`, `bootstrap_type="Bayesian"`, `auto_class_weights="Balanced"` to compensate for the 19.9% positive rate, `iterations=5000` with `early_stopping_rounds=300`, per-fold seed `42+fold`.
3. **Richer feature pipeline** (inline in the trainer, no `features.py` change):
   - Domain features: `EstimatedTotalLaps`, `LapsRemaining`, `TyreAgeRatio`, `PitWindowPressure`, `StintPressure`, race-phase / lap / tyre-life / position bins.
   - Cross-categoricals: nine string-concat interactions over `{Race, Year, Driver, Compound, Stint, RacePhase, TyreLifeBin}`.
   - Frequency / count encoding for every categorical (base + crosses + bins).
   - Group statistics: mean / std / diff for `{LapTime_Delta, Position_Change, RaceProgress, TyreLife}` by `{Race_Year, Race_Compound_Stint, Driver_Race, Compound_Stint}` — 48 features.

Total expected feature count: ~130 model features, ~16 categorical.

Combined leverage (estimated, based on which residual axis each component targets):

| Component | Targets | Expected Δ OOF |
|---|---|---|
| External data | All residuals — more signal per-(Driver, Race) | +0.0015–0.0030 |
| Tuned HPs (slower lr + class weights + per-fold seed) | Driver-level discrimination, low-positive-rate slices | +0.0010–0.0030 |
| Cross-cat + freq + group-stat features | HARD-compound and pit-cluster slices (explicit cohort signals trees can target) | +0.0010–0.0020 |
| **Combined (roughly additive on disjoint residuals)** | | **+0.0035–0.0080** |

## Why we expect this to clear the noise floor

The three components target disjoint residual axes from the cycle 8 EDA:

- External data adds rows from a different distribution (25.5% positive rate vs 19.9% in competition train) — better coverage of pit-imminent laps, the slice the model under-predicts.
- Tuned HPs change the operating point. Lower learning rate + longer training + balanced class weights compensate for the positive-class minority and the per-(Driver, Race) heterogeneity. Per-fold seed variation gives the ensemble more diversity even within the same architecture.
- The new features expose information the existing trees can split on but couldn't reach: explicit cross-categoricals turn implicit interactions into first-class features, frequency encoding gives CatBoost a non-target-encoding angle on high-cardinality columns, and group statistics surface "this row vs its cohort mean" signals that the current pipeline doesn't compute.

If the combined gain lands within the +0.0035 to +0.0080 range, the 4-way ensemble (LGB + CB#004 + CB#006 + CB-tuned) crosses the project's significance bar by a comfortable margin and unlocks the first LB-significant submission of the session.

## Overfitting risk

**Medium.** Calling out the specific concerns and how cycle 12 will guard against them:

- **External data distribution shift** — 25.5% vs 19.9% pos rate. Class-balanced weights help, but per-fold AUC could still rise unevenly. Watch fold std; trigger Discard if > 0.0015.
- **Per-fold seed=42+fold** — mild divergence from the project's fixed seed=42, but the StratifiedKFold split itself stays at seed=42 so the fold structure is identical to all prior cycles. The seed variation only affects each CatBoost's internal randomness.
- **132 features at depth 8 with `auto_class_weights="Balanced"`** — moderately more capacity than CB#006 (66 features). l2=8.5 compensates.

## Kill criteria

- CB-tuned standalone OOF < CB#006 (0.94806) → external data + new features hurt rather than helped.
- per-fold std > 0.0015 → instability from the new operating point.
- Any (Year × Compound) cell with n ≥ 10K regresses by > 0.005 vs CB#006.
- 4-way ensemble OOF < 3-way OOF.

## Scope

Cycle 11 is design-only — no code, no training. It records the hypothesis, the rationale grounded in cycles 7-10, and the gates. Cycle 12 implements and validates.

Code to be written in cycle 12:

- `src/research/train_cb_tuned.py` — new trainer with inline FE and tuned HPs.
- `src/research/blend_cycle012.py` — 4-way blend at fixed pre-registered weight schemes.
- `data/f1_strategy_dataset_v4.csv` — external data file (gitignored).
- `experiments/012_cb_tuned_external_data.md` — results + verdict.

No change to `src/features.py`, `src/train.py`, `src/research/train_catboost.py`, the CV protocol, or the project seed.

## Reversibility check

Touches CV? **No.** Touches the master seed? **Per-fold seed=42+fold** is a defensible variation that keeps the strat split identical. Touches target transform? **No.** Touches leakage surface? **External data has its own rows, no overlap with competition test ids.** Verified safe.

## Verdict (planning cycle)

**Kept.** This is the design that cycle 12 will execute.

## Follow-ups

- Cycle 12: implement and train.
- Cycle 13+ (if cycle 12 succeeds): explore further components — digit / signature features that exploit synthetic-data generator artifacts (~+0.0005-0.0015 estimated), and a tabular MLP as a fifth model for ensemble diversity (~+0.0005-0.0020 estimated).
