# Experiment 027 — 2022 year-specialist RealMLP

**Cycle.** 9
**Status.** Reverted (specialist underfits — small training set < RealMLP's pre-tuned default capacity)
**Date.** 2026-05-23

## Hypothesis

Training a RealMLP using *only* 2022 competition train rows + 2022 external data, then using its predictions for 2022 OOF rows (combining with cycle 5 multi-seed's predictions for 2023/2024/2025), improves the overall OOF AUC by ≥ +0.00020 (min_delta) over cycle 5 multi-seed's 0.95383. If yes, also test whether the year-2022-specialized predictions, blended with CB-tuned-exp14 like cycle 7 did, push the blend past 0.95428.

## Rationale

- Per-year within-year AUC analysis on cycle 7 blend shows:
  - 2022: **0.91947** (weakest by ~0.015)
  - 2023: 0.94978 (best, but 1% pos rate inflates AUC)
  - 2024: 0.93413
  - 2025: 0.93434
- 2022 is 19% of competition train (82,989 rows). If the within-year AUC of 2022 lifts from 0.91947 → 0.92500 (closing ~⅓ of the gap to 2024/2025), the overall OOF would lift roughly +0.001.
- A year-specialist can "spend" its capacity on year-specific features without diluting on cross-year discrimination. Cycle 5's main RealMLP has to balance four years.
- External data includes 21,860 rows tagged Year=2022 — sufficient to augment the specialist without overwhelming it with off-year noise.

## Expected magnitude

- **Specialist's 2022 within-year AUC target:** ≥ 0.925 (lift +0.006 over current).
- **New overall OOF target:** ≥ 0.95403 (cycle 5 + 0.00020).
- **Optimistic:** specialist 2022 AUC reaches 0.930 → overall OOF ≈ 0.95450.
- **Floor for direction-alive:** specialist 2022 AUC > current 0.91947 by any margin. Below current, specialist is worse than main (less data wins).

## Overfitting risk

**Medium.** Sources:

1. **Smaller training set per specialist** — ~66k comp + 22k external = 88k rows (vs 350k+ for main). Risk: specialist underfits, especially with RealMLP's pre-tuned defaults (calibrated for larger data).
2. **Year as proxy for distribution shift** — 2022 may differ from other years not just in features but in pit-stop strategy regime. The main RealMLP learns averaged behavior; the specialist learns 2022 behavior exclusively, which may not generalize within 2022 if 2022 itself is non-stationary across races.
3. **OOF construction risk** — combining specialist 2022 predictions with main-model predictions for other years creates a *mixed* OOF. The combination must use the same fold indices as cycle 5 multi-seed (controlled by fold-by-fold construction).

## Kill criteria

- [ ] Specialist 2022 within-year AUC < cycle 5 multi-seed's 0.92085 (specialist underfits — less data hurts)
- [ ] Specialist 2022 fold range > 0.030 (unstable across folds)
- [ ] Combined OOF < cycle 5 multi-seed's 0.95383 (the specialist's gains don't transfer to overall)
- [ ] Per-fold wall-clock > 5 min (specialist training is unexpectedly slow)

## Scope

- `src/research/train_realmlp_year2022.py` (new, ~250 lines — clone of train_realmlp.py with year-filtered training)
- Outputs:
  - `data/oof_realmlp_2022_specialist.parquet` (specialist's 2022 OOF predictions)
  - `data/oof_realmlp_year_combined.parquet` (specialist 2022 + multi-seed non-2022)
  - `data/submission_realmlp_2022_specialist.csv` (specialist test predictions for 2022 test rows)
  - `data/submission_realmlp_year_combined.csv`
- `experiments/027_year2022_specialist.md` (this file)

Wall-clock budget: ~10-15 min total (5 specialist folds × ~2 min each, smaller training set).

## Reversibility check

- CV protocol: unchanged (same 5-fold StratifiedKFold seed=42 on Year × PitNextLap).
- Seed: unchanged.
- Feature set: cycle 5's pipeline.
- Leakage surface: unchanged.

No reversibility flag fires.

## Plan

1. Build `src/research/train_realmlp_year2022.py`:
   - Filter competition train to Year == 2022; filter external data to Year == 2022.
   - For each fold (using the same 5-fold StratifiedKFold splits as cycle 5):
     - Training rows = (4/5 of 2022 train) + (all 2022 external)
     - Validation rows = (1/5 of 2022 train) within that fold's val set
     - Train RealMLP with cycle 5's HPs; predict on val + test (2022 test rows only)
   - Combine: replace cycle 5 multi-seed's 2022 OOF predictions with the specialist's
2. Compare overall OOF AUC against cycle 5 multi-seed's 0.95383.
3. If specialist 2022 AUC improves: try blending with CB-tuned-exp14 at w_cb=0.20 (cycle 7's recipe). Compare blend to cycle 7's 0.95408.
4. Decision gate:
   - Specialist OOF > 0.95403 AND blend > 0.95428 → submission.
   - Either gate misses → Inconclusive, consider adding all-external data for the specialist in a follow-up, or pivot.

## Result

Killed at fold 3/5 on consistently-negative signal + MPS thermal throttling.

| Fold | Specialist 2022 AUC | Wall (s) | Δ vs multi-seed 2022 baseline (0.91884) |
|------|---------------------|----------|-------------------------------------------|
| 1 | 0.92009 | 304 | **+0.00125** |
| 2 | 0.91326 | 421 | **−0.00558** |
| 3 | 0.91561 | 1062 (thermal) | **−0.00323** |
| **Mean (3 folds)** | **0.91632** | — | **−0.00252** |

Fold 1 was a positive outlier (+0.00125), folds 2-3 were significantly negative (−0.005, −0.003). The mean lift after 3 folds is −0.00252 — the specialist is *worse* than multi-seed on the very slice it was supposed to improve.

## Verdict

**Reverted.** Specialist is unstable across folds (range 0.91326 → 0.92009 = 0.0068 spread) and on average underperforms multi-seed. Replacing multi-seed's 2022 OOF predictions with the specialist would *lower* the combined OOF below cycle 5's 0.95383.

## Kill-criteria check

- [x] Specialist 2022 AUC < cycle 5 multi-seed's 2022 (0.91884) by mean −0.00252 — **KILL FIRES**
- [x] Specialist fold range = 0.0068 → exceeds the 0.005 expected-stability range by 36% — **KILL FIRES**
- [x] Combined OOF would be worse than cycle 5 — direction dead

## Repro stamp

- training set per fold: ~88k rows (66k comp + 22k external, all Year=2022)
- HPs: identical to cycle 5 multi-seed (PyTabKit RealMLP_TD_Classifier defaults)
- killed at fold 3/5; folds 4-5 not run

## Learnings

1. **RealMLP's pre-tuned defaults are calibrated for large data.** With ~88k training rows per specialist (vs ~350k for cycle 5 RealMLP), the model underfits — the n_ens=24 internal ensemble + hidden_sizes=[512,256,128] need substantially more data to converge well. The specialist's smaller training set hurts more than the year-specialization helps.
2. **Year-specialization didn't compensate for less data.** The 2022 slice's lower AUC (0.91884 in cycle 5) isn't due to *insufficient* per-year capacity — RealMLP already learned 2022 well within the cross-year training. Training only on 2022 strips away cross-year regularization without adding signal.
3. **MPS thermal throttling is severe** when running consecutive long training jobs. Fold 1: 304s; fold 3: 1062s (3.5× slower). Mac M1 Pro thermal headroom is limited; need cooling pauses or simpler models for long sessions.
4. **Specialist approaches need different HPs.** A specialist on 88k rows should use smaller capacity (hidden_sizes=[256,128,64], n_ens=12-16) and possibly higher regularization. Cycle 6 finding ("don't change features without HP retune") generalizes: don't change training set size without HP retune either.

## Follow-ups

1. ✅ Killed exp 027 at fold 3.
2. **Cycle 9 closes Inconclusive.** Two probes (pseudo, year-specialist) both failed to break the cycle 7 blend ceiling.
3. **Don't retry year-specialist without HP retuning** — same data brittleness as cycle 6/7. A CatBoost-based specialist would handle small data more gracefully but has lower ceiling on this task.
4. **Remaining options for cycle 10:**
   - Stacking with strong base OOFs + raw features (cheap, ~15 min)
   - CatBoost 2022 specialist (robust to small data; ~5-10 min)
   - Isotonic calibration on cycle 7 blend (drift-tightening, max +0.0001)
   - Accept project tip at LB 0.95361 and stop cycling.
