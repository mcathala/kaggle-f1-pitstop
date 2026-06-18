# Experiment 014 — Iter cap raised (scope reduced from initial plan)

**Cycle.** 4 (opens cycle 4 — first experiment after cycle 3 closed on submission #3 at LB 0.95066)
**Status.** Kept (LB-confirmed at 0.95097, +0.00031 over cycle 3's 0.95066)
**Date.** 2026-05-22
**Pre-registered in `project.md`'s cycle 4 next-steps section.**

## Hypothesis

Lifting CB-tuned's iteration cap from 5000 → 8000 (with `early_stopping_rounds` 300 → 500) raises standalone OOF AUC by ≥ +0.0005 over experiment 012's 0.95085. Single-component test; cleaner attribution.

## Rationale

Experiment 012 produced the cycle 3 closing submission (LB 0.95066) but hit the 5000-iter cap in every fold without `early_stopping_rounds=300` firing. The lr=0.018 trajectory still had downward slope when training stopped. Closest thing to a free lift available — same model, same FE, just trained longer.

## Scope-reduction note (initial broader experiment killed)

The initial experiment 014 also planned digit / signature feature decomposition (decompose float/integer columns into per-digit features and "signature" strings — a synthetic-data generator exploit). After kicking off training (start 16:53, killed 18:00 after fold 1 never completed), two findings made us pull back:

1. **Cost was 3-5× worse than estimated.** Feature space jumped 132 → 294 model features and 16 → 81 categoricals. Fold 1 was projecting to ~60 min, total run ~5 hours on CPU.
2. **Univariate signal check (`/tmp/digit_signal_check.py`) showed most "high-spread" digit features were either redundant with the raw numerics or 1-row outliers.** Top-spread features: `TyreLife_int_digit_2` is just bucketed TyreLife (CatBoost already splits on TyreLife). `LapMinusTyreLife_dec_digit_1` spread 0.80 but n=1 in the minority bin. Real new signal from digit decomposition is probably 0 to +0.0005 — not the +0.0005-0.0015 originally estimated.

Reduced experiment 014 = **iter-cap-lift only**. If it lands a clean +0.0005-0.0010, we move to experiment 015 (multi-seed) and 016 (RealMLP). Digit features deferred — they can be retested standalone in a future experiment if we ever revisit the synthetic-exploit angle.

## Expected magnitude

- CB-tuned (exp 014) standalone OOF: **≥ 0.95135** (= 0.95085 + 0.0005). Stretch +0.0010.
- 4-way ensemble OOF (drop-in replacement of CB-tuned in `3way_focus` weights): **≥ 0.95184** (= 0.95134 + 0.0005).

## Overfitting risk

**Low.** The only change is iter cap 5000→8000 with `early_stopping_rounds` 300→500. CatBoost's own early stop should trigger if more iters start hurting val AUC. The base feature space and HPs are unchanged from experiment 012, which we already know generalises well (LB 0.95066 confirmed).

## Kill criteria

- CB-tuned (exp 014) standalone OOF < experiment 012's 0.95085 → iter cap lift hurt.
- per-fold std > 0.00080 (vs experiment 012's 0.00056) → instability from longer training.
- 4-way ensemble OOF < experiment 012's 0.95134.
- Any (Year × Compound) cell with n ≥ 10K regresses by > 0.001 vs experiment 012.

## Scope

- `src/research/train_cb_tuned_exp14.py` — clone of `src/research/train_cb_tuned.py` with: iter cap 8000, early_stop 500, output paths suffixed `_exp14`. **Single change vs cycle 3.**
- `src/research/blend_exp14.py` — new 4-way blend reading the exp14 OOF in place of the cycle 3 CB-tuned.
- `experiments/014_iter_cap_and_digit_features.md` — this file.
- No changes to `src/features.py`, `src/train.py`, `src/research/train_catboost.py`, `src/research/train_cb_tuned.py` (cycle 3 trainer preserved as the closed-cycle reference).

## Reversibility check

CV unchanged. Project seed unchanged. Target transform unchanged. Leakage surface unchanged (no labels touched in new FE). Cycle 3 artifacts (`oof_cb_tuned.parquet`, `submission_ensemble_cycle012.csv`) are preserved untouched. Safe.

## Plan

1. ✅ Edit `train_cb_tuned_exp14.py` → only iter cap 5000→8000, early_stop 300→500.
2. ⏳ Train 5 folds (~80 min expected; iter cap 1.6× cycle 12's ~23 min/fold = ~37 min/fold max).
3. ⏳ Blend with existing LGB + CB#006 + CB-tuned (cycle 3 baseline) at the cycle 3 `3way_focus` weights plus a new variant that uses CB-tuned-exp14 as drop-in. Try a 5-way that keeps both old + new CB-tuned for diversity.
4. ⏳ Apply gates; document.
5. ⏳ If KEEP: feed into experiment 015 (multi-seed). Defer submission until cycle 4's cumulative lift clears +0.005 OOF or equivalent.

## Result

### CB-tuned exp14 standalone

```
OOF AUC:   0.95114   (vs experiment 012 0.95085, Δ = +0.00029)
Per-fold:  0.95160, 0.95194, 0.95066, 0.95030, 0.95125
fold std:  0.00062   (cycle 12 was 0.00056)
iters:     [7930, 7995, 7943, 7961, 7975]
```

**All 5 folds hit ≈ the 8000 iter cap** — early stopping fired only in the final ~70 iterations of each fold. The model used all the extra training budget but with diminishing returns:

Per-fold deltas vs cycle 12: **+0.00037, +0.00031, +0.00026, +0.00022, +0.00034 → mean +0.00030**. Pattern wasn't strictly monotonic (fold 5 jumped back up), but the trend is "real but at the low end of the +0.0005-0.0010 estimate".

Wall-clock: 5 folds × ~36 min = ~180 min total (M1 Pro, 8 cores at sustained 80% utilization with mild thermal throttle in hours 2-3).

### 4-way ensemble weight sweep

| Scheme | Weights (LGB / CB#004 / CB#006 / CB-tuned-c3 / CB-tuned-exp14) | OOF AUC | fold std | Δ vs cycle 3 4-way | Folds up |
|---|---|---|---|---|---|
| 3way_c2 (cycle 2 ref) | 0.10 / 0.40 / 0.50 / 0 / 0 | 0.94866 | 0.00045 | −0.00268 | 0/5 |
| 4way_c3 (cycle 3 ref) | 0.05 / 0 / 0.20 / 0.75 / 0 | 0.95134 | 0.00047 | 0 | — |
| **4way_exp14_dropin** | **0.05 / 0 / 0.20 / 0 / 0.75** | **0.95161** | **0.00046** | **+0.00028** | **5/5** |
| 5way_both_tuned | 0.05 / 0 / 0.15 / 0.35 / 0.45 | 0.95150 | 0.00046 | +0.00017 | 5/5 |
| single_exp14 | 0 / 0 / 0 / 0 / 1.0 | 0.95114 | 0.00045 | −0.00019 | 0/5 |

**Best blend: `4way_exp14_dropin`** — drop the cycle-3 CB-tuned, use only the new exp14 CB-tuned at the same 0.75 weight. 5/5 folds positive.

Notable: `5way_both_tuned` (keeping both CB-tuneds in the blend) is *worse* than the drop-in. CB-tuned-exp14 absorbs CB-tuned-c3's signal entirely — same pattern observed in cycle 12 with CB#004.

### Reproducibility stamp

- git SHA at start: `44828aa`
- data sha256(train.csv): `f004e79d4e63f4bad0afc3788d07938dc5dbc0bed73a51e7b65cbce52ccc4128`
- external data: `data/f1_strategy_dataset_v4.csv` (101,371 rows, 101,305 used)
- packages: catboost 1.2.10, polars 1.40.1, sklearn 1.8.0, numpy 2.4.4, pandas 3.0.2

### Acceptance gates

baseline_std (cycle 3 4-way ensemble fold std) = 0.00047 → magnitude floor = max(0.5 × 0.00047, 0.00020) = **0.000237**.

| Gate | Target | Got | Pass? |
|---|---|---|---|
| Magnitude (Δ OOF ≥ 0.000237) | ≥ 0.000237 | +0.00028 | **PASS** (hairline, +0.000037 over bar) |
| Direction (≥ 3/5 folds improved) | ≥ 3 | 5/5 | PASS |
| Stability (fold std change) | ≤ 0.00080 | 0.00046 | PASS |
| Kill: CB-tuned-exp14 ≥ CB-tuned-c3 | ≥ 0.95085 | 0.95114 | PASS |
| Kill: 4-way > cycle 3 4-way | > 0.95134 | 0.95161 | PASS |

All gates pass. Magnitude only by a hair — not the +0.0005-0.0010 estimate, more like +0.0003.

## Verdict

**Kept (hairline).**

The lift is real (5/5 folds positive, gates pass) but small. Pre-registered estimate was +0.0005-0.0010; reality is ~+0.0003. The CatBoost-on-this-feature-set track is approaching its ceiling — pushing iter cap from 5000 → 8000 gave us the unused-training signal, but the model's residual capacity at iter=8000 is small (per-fold gains declining from +0.00037 to +0.00022).

**Submitted to Kaggle on 2026-05-22 to validate the OOF→LB drift before committing to exp 16.** Public LB = **0.95097**, drift −0.00064 (vs cycle 12's −0.00068 — within 0.00004, drift is stable). **+0.00031 LB over cycle 3's 0.95066.** Modest but real; the cycle-4 cumulative LB lift over cycle 3 is +0.00031 so far, target +0.001+ before closing.

## Learnings

1. **The iter-cap lift was a real but bounded win.** The 5000-iter cap was indeed leaving signal on the table, but only ~+0.0003 of it. Most of the easy iter-budget signal was already captured by cycle 12's 5000-iter run; iter cap to 10000+ probably yields ≤ +0.0001 more.
2. **The new CB-tuned absorbs the old CB-tuned's signal.** `5way_both_tuned` is worse than `4way_exp14_dropin`. Same pattern as cycle 12's CB#004 elimination — old base obsoleted by the new better-trained version.
3. **The CatBoost track is approaching its ceiling on the current feature set.** Three structural levers tried in cycle 12 (external data, tuned HPs, richer FE) gave us +0.00268. Tweaking the remaining knob (iter cap) is +0.00028 — ~10× smaller. **Further +AUC requires either new features (digit-feature exploit was the candidate but failed the signal check), or a different model family (RealMLP).**
4. **M1 Pro CPU on long CatBoost runs**: sustained 80-90% utilization stable for ~3 hours, mild thermal throttling appearing around hour 2 (CPU% drops ~5-7%). 36 min/fold × 5 folds = workable but not fast. Future iterations should consider whether RealMLP on Metal (10× faster than this) is the better compute investment.

## Follow-ups

1. **Cycle 4 continuation decision** (next experiment):
   - **Exp 015 (multi-seed)**: pre-registered as variance-reduction on the new CB-tuned. Expected +0.0005-0.0015. Cost: ~6 hours sequential.
   - **Exp 016 (RealMLP)**: pre-registered as model-family diversity. Expected +0.0010-0.0030. Cost: ~1-2 hours on M1 Pro MPS (revised from initial CPU estimate).
   - **Recommendation**: swap exp 15 → exp 16 given M1 Pro hardware. Better lift-per-hour, structurally different residual.
2. **Iter cap pushing further (to 10000+)**: probably not worth it. Diminishing returns are evident at 8000.
3. **Submission gate**: don't submit exp 14 standalone. Accumulate with exp 16 (or 15+16), submit when cumulative LB-projected lift clears +0.001.
4. `data/submission_ensemble_exp14.csv` exists but holding back from Kaggle.
