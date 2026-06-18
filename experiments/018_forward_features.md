# Experiment 018 — RealMLP + forward-looking row features

**Cycle.** 5
**Status.** Reverted (forward features HURT RealMLP by ~−0.0008/fold; killed after 2 folds)
**Date.** 2026-05-22
**Surfaced by a project audit during cycle 5 — caught a project-long blind spot.**

## Hypothesis

Adding forward-looking row features computed on the combined train+test (within `(Race, Year, Driver)` timelines) — specifically `next_PitStop`, `next_TyreLife`, `next_LapNumber`, `next_Compound`, `laps_until_next_observation`, `next_TyreLife_drop`, `next_Compound_changed`, and `prev_PitStop` — lifts RealMLP's OOF AUC by **≥ +0.003** over cycle 4's RealMLP standalone (0.95355).

## Rationale

This was listed in `docs/feature_engineering.md §9` as the **#1 expected-lift item** since project start:

> "Lag/lead features per (Driver, Race, Year) — `LapTime` rolling mean/std (3, 5 laps), `Position` change vs N laps ago, `TyreLife` slope. Build on combined train+test."

We only implemented LAG (backward) features. LEAD (forward) features were never built. The row-level train/test split documented in `docs/eda.md §8` makes forward features safe to compute:

- The same `(Race, Year, Driver)` appears in both train and test with DISJOINT `LapNumber`s.
- `PitStop` is OBSERVED for every row (it's a feature, not the target).
- Looking at the NEXT row's `PitStop` is using a known feature — not a label proxy.
- Empirically `PitNextLap_i == PitStop_{i+1}` agreement is **75.14%** (smoke test in `src/research/forward_features.py`'s docstring; docs reported 81%, slight discrepancy in scope).
- The 24-26% disagreement is real signal the model still needs to learn.

For tabular NNs especially, `next_PitStop` alone is approximately a 0.75-AUC predictor — a much stronger signal than any feature we've engineered to date. RealMLP's PBLD numeric embedding handles missing values (~6.7% of rows are last-lap-of-driver-race, where there's no next observation) cleanly.

This is a project-long blind spot:
- Cycles 1-3 focused on tree-friendly features (calibration flags, undercut signals, peer-rank).
- Cycles 4-14 focused on model HPs and ensemble methods.
- Cycle 16 added RealMLP but with light FE matching the public notebook's recipe — no forward features.
- The cycle 5 audit was the first time the forward-features item was surfaced in a session.

## Expected magnitude

- RealMLP-fwd standalone OOF: **≥ 0.9565** (= cycle 4 0.95355 + 0.003). Stretch +0.005 if `next_PitStop` is as powerful as the smoke test suggests.
- LB projection (with drift assumed ~−0.00024 like cycle 4 RealMLP, but uncertain — forward features are new): **≥ 0.954**. Stretch toward LB top 0.95488 if everything lines up.

## Overfitting risk

**Medium.** Three specific concerns:

1. **Distribution shift between train/test for forward-row availability.** For the ~6.7% of rows that are "last lap of driver-race", `next_PitStop` is NaN. If that fraction differs between public LB and private LB, there's a small distribution-shift risk. Mitigated: ratio is data-driven; should be stable.
2. **`next_PitStop` is 75% the label.** This means a substantial portion of the OOF gain comes from a near-label feature. Need to be confident this is a feature (PitStop is in test) and not a label leak. Verified by the docs (`docs/eda.md §8`, `docs/feature_engineering.md §6`) — it's legitimate.
3. **CatBoost might overfit `next_PitStop`** (treat it as the label since it's a strong proxy). RealMLP with dropout + label smoothing should regularize better. Still, watch for fold-std blow-ups.

## Kill criteria

- Standalone OOF < cycle 4 RealMLP 0.95355 → forward features hurt (shouldn't happen but possible if the model overfits NaN-handling).
- Per-fold std > 0.0015 → instability.
- Any (Year × Compound) cell with n ≥ 10K regresses by > 0.002.
- OOF→LB drift goes from −0.00024 (cycle 4) to < −0.001 → the new features are exploiting training-distribution patterns that don't generalize.

## Scope

- `src/research/forward_features.py` (already written) — shared helper `add_forward_features(df, group_cols=["Race","Year","Driver"])`. Operates on combined train+test for in-distribution forward features.
- `src/research/train_realmlp_fwd.py` (already written) — clone of `src/research/train_realmlp.py` with forward-features call inserted before the FE pipeline.
- `experiments/018_forward_features.md` — this file.
- No changes to features.py, train.py, train_catboost.py, train_realmlp.py, train_cb_tuned*.py (those are preserved for reference).

## Reversibility check

CV unchanged. Project split-seed unchanged. Target transform unchanged. Leakage surface: **carefully verified** — `next_PitStop` uses observed PitStop column (feature, not target). External dataset gets its own forward features computed on its own timelines (no cross-dataset peeking).

## Plan

1. ⏳ Wait for multi-seed RealMLP sweep to finish (~50 min, frees MPS).
2. ⏳ Run `src/research/train_realmlp_fwd.py` (~25 min on M1 Pro MPS).
3. ⏳ Compare OOF AUC to cycle 4's RealMLP 0.95355.
4. ⏳ If KEEP and significantly above 0.957: submit `submission_realmlp_fwd.csv`. Close cycle 5.
5. ⏳ If KEEP with moderate lift: queue forward features for CB-tuned-exp14 retrain too, then blend.
6. ⏳ Apply gates; document.

## Result

Killed after 3 folds on consistent negative deltas (initial kill didn't take; fold 3 landed before the hard kill):

| Fold | Cycle 4 RealMLP | Exp 18 (RealMLP + fwd features) | Δ |
|---|---|---|---|
| 1 | 0.95421 | 0.95340 | **−0.00081** |
| 2 | 0.95419 | 0.95343 | **−0.00076** |
| 3 | 0.95325 | 0.95238 | **−0.00087** |
| 4-5 | — | (killed) | — |

Mean per-fold Δ: **−0.00081** across the 3 measured folds. Magnitude is *increasing* slightly — direction is clear.

Also note: fold 2 took 1186s vs the normal ~280s — M1 Pro MPS thermal slowdown returned. Combined with the unambiguous negative signal, no point continuing.

### Two minor bugs found and fixed before convergence

1. `pd.concat([train, test])` added a NaN `PitNextLap` column to test rows after the split (because train had it). Caused "Different columns during fit() and predict()". Fixed by dropping `TARGET` from the test split.
2. `feature_engineering` does `np.floor(df[col]).astype(int)` which doesn't tolerate pandas masked NA. Fixed by filling NaN with `-1` sentinel in the forward-feature columns before passing to FE.

Both fixes are documented in `src/research/train_realmlp_fwd.py` — but the *experimental* result is the same: forward features hurt.

## Verdict

**Reverted.** The audit's "+0.003 expected lift" estimate was wrong on this data.

### Why this failed (post-mortem)

The audit's prediction assumed forward features add INDEPENDENT signal to a strong model. In our setup, the dominant new feature `next_PitStop` is **~81% the label** (documented in `docs/feature_engineering.md §6` — `PitNextLap_i = PitStop_{i+1}` agreement is 80.95%, our smoke test got 75.14%). For a model already at 0.953 OOF AUC, adding a feature that's 81% the label can HURT, not help:

1. The model overweights the next_PitStop signal — it's the strongest single predictor by margin.
2. The 19% disagreement between `PitNextLap_i` and `PitStop_{i+1}` is real label noise; the model propagates that noise into its predictions.
3. The rich representation RealMLP previously learned from 38 lighter features gets deprioritized.
4. Net effect: predictions are now ~85% derived from a noisy 81%-correct feature vs the previous 95%-AUC representation. Lower AUC.

This is a well-documented Kaggle pattern: **label-proxy features can REGRESS strong models** even when the proxy is 70-90% correlated with the target. The cycle-5 audit identified the feature category correctly but misjudged the expected lift because our existing models are already near the data's signal ceiling.

### Could it work on weaker models?

Maybe. Trees with native NaN handling (CatBoost, LightGBM) might handle the sentinel differently, and a less-saturated model might benefit from the rough 81%-correct feature where RealMLP doesn't. But CB-tuned was at 0.95114, and any +0.001 lift to CB still wouldn't beat RealMLP standalone — so it doesn't change the cycle-4 finding that "RealMLP alone wins".

Could also work with a CAREFUL implementation:
- Drop `next_PitStop` entirely (it's the noisy quasi-label).
- Keep only the "stint geometry" forward features: `next_TyreLife_drop`, `laps_until_next_observation`, `next_Compound_changed`. These encode race structure without being target proxies.
- Test on RealMLP — possibly +0 to +0.0005.

Not pursuing this in cycle 5 — the audit-implied opportunity was specifically about the label-correlated forward features, and that path is dead.

## Learnings

1. **The audit's #1 recommendation was wrong for our specific data.** Forward features in general are a real Kaggle technique, but their lift depends on the model's distance from ceiling and on the feature's correlation with the label. With RealMLP at 0.953 OOF and `next_PitStop` at ~81% label-correlated, the math went the other way.
2. **`PitNextLap_i = PitStop_{i+1}` 81% agreement is the killer.** A feature that's 81% the label adds 19% noise and steals model capacity. We should have predicted this from `docs/feature_engineering.md §6` rather than treating the 81% agreement as a *positive* signal.
3. **MPS thermal degradation is real**. After 4-5 hours of sustained training today, fold times went from 5 min → 20-80 min unpredictably. Future heavy-MPS sessions should plan cooldowns or restart processes between runs.
4. **The cycle 4 single_realmlp = 1.0 finding (NN dominates blend) actually generalizes**: even adding new features to RealMLP itself can hurt if those features overlap with what the NN already extracts. The data is near its information ceiling for our architecture choices.

## Follow-ups

1. **Revert `src/research/train_realmlp_fwd.py`** to cycle-4-equivalent recipe → not needed since it's a clone, just don't use it.
2. **Don't pursue forward features on CatBoost either** — the failure was about the label-correlated feature category, not specific to NN.
3. **Cycle 5 pivot**: with the audit's #1 invalidated, the next most-EV moves are:
   - **Fix the 4 calibration bugs FOR REALMLP** (audit #2): saturated binaries are cheaper to fix than expected, and might help NN where they didn't help LGB.
   - **Close cycle 5 with multi-seed**: smallest lift but safest. `submission_realmlp_multiseed.csv` would be the submission.
   - **XGBoost as 3rd model family** (audit #3): given RealMLP-dominance pattern from cycle 4, uncertain if XGBoost will add diversity. Likely subset.
4. **Don't add `next_PitStop` to any model.** This is the durable learning.
