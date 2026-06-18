# Experiment 019 — RealMLP + historical driver aggregates

**Cycle.** 6 (opens cycle 6 — first experiment after cycle 5 closed on submission #5 at LB 0.95342)
**Status.** Reverted (historical aggregates HURT RealMLP by ~−0.0014/fold; killed after 2 folds)
**Date.** 2026-05-22
**Pre-registered after exp 18 reverted; targets the driver-level residual cycle 13 EDA flagged, via a safer mechanism than the failed forward-features attempt.**

## Hypothesis

Adding 17 historical driver-level aggregates (avg LapTime, std LapTime, avg TyreLife, avg Position, abs position change, RaceProgress, compound usage frequency for 5 compounds, per-(Driver, Race) aggregates, per-(Driver, Year) aggregates) as new features to RealMLP raises OOF AUC by **≥ +0.0005** over cycle 4's single-seed RealMLP 0.95355. All aggregates computed on combined train+test, no labels involved.

## Rationale

Cycle 13 EDA showed driver-level discrimination (real F1 codenames ALO/STR/RIC/VET/MSC) is the structurally hardest residual. Cycle 4 RealMLP partially solved this via learned categorical embeddings, but:

- RealMLP's 6-dim Driver embedding gives 887 × 6 = 5322 params for driver-level info.
- These embeddings are learned from RealMLP's training objective only — they encode whatever is predictive of PitNextLap.
- They do NOT have access to driver-level *unsupervised* signal: how fast this driver typically drives, how much position variability, what compounds they prefer, their typical race progress.
- Adding these as explicit features gives the NN more per-driver context to combine with its learned embeddings.

This is the same residual target as exp 18 but a safer mechanism:
- exp 18 used `next_PitStop` (~81% the label) → label-proxy failure
- exp 19 uses driver-level aggregates of INPUT FEATURES (LapTime, Position, etc.) → genuinely new signal, correlations with target are 0.01-0.10 (verified by smoke test).

## Expected magnitude

- OOF: ≥ 0.95405 (= 0.95355 + 0.0005). Stretch ≥ 0.9550.
- LB: with drift assumed ~−0.00024 to −0.00041, ≥ 0.95360 to 0.95380.

## Overfitting risk

**Low.** Aggregates are deterministic transforms of input features; no labels touched. Smoke test confirmed max correlation with target is 0.10 (driver_avg_TyreLife) — far below the label-proxy threshold that broke exp 18.

Small concern: driver-level aggregates create a "this is driver X's typical behavior" signal that might cluster predictions per-driver. If the test set has the same per-driver distribution as train (which it does — all test drivers are in train), this generalizes; if not, it could regress.

## Kill criteria

- OOF < cycle 4 RealMLP 0.95355 → historical aggregates hurt.
- Per-fold std > 0.0010 → instability.
- Any per-driver cell with n ≥ 200 regresses by > 0.005 AUC.

## Scope

- `src/research/historical_features.py` — new shared helper computing 17 aggregates per Driver / (Driver, Race) / (Driver, Year). PitStop/PitNextLap deliberately excluded.
- `src/research/train_realmlp_hist.py` — clone of `train_realmlp.py` with historical aggregates added before the FE pipeline.
- `experiments/019_historical_driver_features.md` — this file.

## Reversibility check

CV unchanged. Split seed unchanged. Target untouched. Leakage surface verified: no PitStop or PitNextLap in aggregates.

## Result

Killed after 2 folds:

| Fold | Cycle 4 RealMLP | Exp 19 (RealMLP + 17 hist features) | Δ |
|---|---|---|---|
| 1 | 0.95421 | 0.95292 | **−0.00129** |
| 2 | 0.95419 | 0.95278 | **−0.00141** |
| 3-5 | — | (killed) | — |

Mean Δ across 2 folds: **−0.00135** — even worse than exp 18's −0.00081.

## Verdict

**Reverted.** Historical driver aggregates HURT RealMLP. Stronger evidence than exp 18 that **feature additions break RealMLP's pre-tuned defaults**.

### Why this failed — combined with exp 18

Two consecutive feature-addition experiments failed:
- exp 18 (forward features, ~−0.0008/fold): label-proxy pattern.
- exp 19 (historical aggregates, ~−0.0014/fold): non-label-proxy, but still worse.

The common root cause: **RealMLP's "Better by Default" pre-tuned HPs assume a particular feature space.** Adding features without re-tuning breaks the carefully-balanced training dynamics. Possible specific mechanisms:

1. **Input-dim increase**: from 44 features → 52 (exp 18) or 61 (exp 19). The first-layer width (512) was tuned for ~44 features; with more inputs, the same width is undertrained relative to capacity.
2. **Embedding-redundancy**: cycle 4's Driver embedding (6-dim × 887 drivers) already captures driver-specific signal. Adding explicit driver-level aggregates duplicates info and increases overfit surface.
3. **Categorical/numeric balance**: PyTabKit's tfms transformations (one_hot, embedding, median_center, robust_scale, smooth_clip, l2_normalize) are tuned for the feature mix. Adding 17 numeric aggregates shifts the balance toward numerics.

This is consistent with cycle 4's blend-monotonicity finding: RealMLP's standalone wins because the architecture+features are tightly co-tuned. Adding ANYTHING (other models in blends, new features in inputs) disrupts the tuning.

## Learnings

1. **Feature addition is unlikely to work on RealMLP without HP re-tuning.** Both attempted feature families (label-correlated AND uncorrelated) hurt.
2. **Pre-tuned-defaults architectures are brittle to feature-space changes.** "Better by Default" tabular recipes are tuned for a fixed feature representation; changing the representation requires re-tuning.
3. **Two paths remain for cycle 6**: different architectures on the SAME feature set (FT-Transformer, RealTabR, etc.), or HP-retuned wider RealMLP (Optuna sweep).

## Follow-ups

1. ✅ Killed exp 19 after 2 folds.
2. **Pivot to FT-Transformer** (PyTabKit's `FTT_D_Classifier`) — same FE pipeline, different architecture. Cycle 6 next experiment.
3. **Don't add features to RealMLP** unless we commit to a full HP re-tuning cycle.
4. **`src/research/historical_features.py` and `src/research/train_realmlp_hist.py` are kept for reference.** Could be reused if we ever do an HP-retuned RealMLP variant.
