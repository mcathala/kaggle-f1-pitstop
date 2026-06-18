# Experiment 040 — 3-way blend with cached CB variants

**Cycle.** 11
**Status.** Inconclusive (Reverted) — closes the post-hoc blend axis definitively.
**Date.** 2026-05-25

## Hypothesis

A 3-way blend `(RealMLP-multiseed, CB-tuned-exp14, X)` where X is one of the cached CB variants from earlier cycles produces an OOF lift ≥ +0.00010 over cycle 7's 2-way 0.95408. The 3rd CB variant adds diversity that cycle 14's single CB cannot.

## Result

| Variant | standalone AUC | rank-corr RM | rank-corr CB14 | best w (rm, cb14, var) | 3-way AUC | Δ vs cycle 7 |
|---|---|---|---|---|---|---|
| cb004 (cycle 2 base) | 0.94774 | 0.9636 | 0.9678 | (0.80, 0.15, 0.05) | **0.954084** | +0.000002 |
| cb006 (cycle 2 compound) | 0.94806 | 0.9654 | 0.9695 | (0.80, 0.20, 0.00) | 0.954082 | 0 |
| cb007 (cycle 3 peer-rank) | 0.94814 | 0.9655 | 0.9689 | (0.80, 0.20, 0.00) | 0.954082 | 0 |
| cb009 (cycle 3 pit-cluster) | 0.94805 | 0.9658 | 0.9693 | (0.80, 0.20, 0.00) | 0.954082 | 0 |
| cb_tuned (cycle 3 older) | 0.95085 | 0.9772 | **0.9988** | (0.80, 0.20, 0.00) | 0.954082 | 0 |

**Variants beating cycle 7: 1 (cb004, by +0.000002 — rounding noise).** Variants reaching hurdle 0.95428: 0.

## Verdict

**Inconclusive (Reverted).** Only cb004 nudges the optimization by w_var=0.05, and the gain (+0.000002) is below any measurable threshold. Every other variant settles at w_var=0.00 — the optimizer judges the variant contributes nothing positive. Cycle_tuned (cycle 3's older CB) has rank-corr 0.9988 with cb14 (near-identical), confirming it carries zero new information.

## Learnings

1. **All cached CB variants are statistically interchangeable with cb14 for blending.** Even cb004 (the most-diverse, rank-corr 0.9636 with RealMLP) doesn't shift the blend AUC. Diversity *within the CatBoost family* is insufficient to extract more from the blend.
2. **cycle_tuned (cycle 3) has rank-corr 0.9988 with cb14 (cycle 4) — they're nearly identical predictors** despite different recipes (cycle 3: iter cap 5000, cycle 4: iter cap 8000). The iter-cap raise affected per-row probability values but not rank ordering. Consistent with the project's history of cycle 14 being a hairline improvement over cycle 3.
3. **The post-hoc blend axis is conclusively exhausted.** Six experiments (cycle 7's grid, exp 029, 036, 037, 038, 039, 040) all converge on the same answer: 0.95408 is the global optimum for any combination of cached OOFs.

## Follow-ups

- **Confirmed: only path forward is new training.** Exp 035 (multi-seed CB-tuned, in progress) is the last reasonable bet at cycle 11.
- **Closed direction:** any post-hoc transformation of cached OOFs. Future cycles should focus exclusively on new model training (XGBoost, GPU CB, NN variants).
