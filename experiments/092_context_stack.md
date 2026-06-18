# Experiment 092 — context-aware GBDT stacking → worse than linear (combiner closed, again)

**Cycle.** 18 · **Date.** 2026-05-29
**Status.** **Inconclusive (worse, −0.00012).** Context-aware meta-stacking underperforms the linear blend.

## Hypothesis

exp 061 closed pred-only linear/logistic meta (overfit). Per-year weights failed (broke cross-year calibration). The untried middle: a shallow, regularized LightGBM meta-model on [8 base OOF preds + context (Year, Stint, TyreLife, Position, RaceProgress, LapNumber, Compound)]. A context-aware combiner might learn cross-year-*consistent* context weighting that linear/per-year can't. Gated on nested-CV honest OOF (a GBDT on OOF preds overfits in-sample).

## Result

| combiner | nested honest OOF |
| --- | --- |
| **linear blend** | **0.95462** |
| context-stack (preds + context) | 0.95450 (−0.00012) |
| pred-only meta (control) | 0.95448 |

Context adds only +0.00002 over pred-only; both are worse than linear.

## Verdict

**Inconclusive — linear blend remains optimal.** The bases are too correlated (ρ≥0.97) for a non-linear combiner to extract anything; even a regularized, nested GBDT overfits the OOF idiosyncrasies. Context weighting doesn't help because (per the diagnostic) the pooled metric is dominated by cross-year structure the linear blend already captures, and within-year heterogeneity is too noisy to exploit row-wise.

**This is the 7th independent confirmation of the 0.95462 own-pipeline ceiling:** linear blend beats per-year weights, bagged weights (A1), operators (074), pred-only meta (061), context meta (this), and no diverse/FE/robust base improves it (088/089/091). The combiner axis is closed.

## Repro stamp

- [src/research/meta_stack_context.py](../src/research/meta_stack_context.py).

## Learnings

1. **Offline OOF is comprehensively saturated at 0.95462.** Every combiner and base-diversity lever has been tested and none beats the linear blend. Continuing offline OOF experiments is now busywork.
2. **The work pivots fully to the LB.** The 5 staged candidates (best/nopseudoGBDT/pure/robustincl/rankmean) test pseudo-drift, diversity, and aggregation hypotheses on the real metric — the only remaining source of genuinely-new information before the deadline.
