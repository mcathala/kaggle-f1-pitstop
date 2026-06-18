# Experiment 061 — honest cross-fitted meta-blend over the full base zoo

**Cycle.** 17
**Status.** Inconclusive — no combiner and no diverse-base addition beats the linear 3-way (0.95420). Zero-GPU. Closes the blend-combiner lever with current bases.
**Date.** 2026-05-27

## Hypothesis

After the diverse-base hunt closed (058/059/060), test the cheap (zero-GPU) question: does a smarter *combiner* — NNLS / logistic stacking / rank-averaging — or adding the genuinely-diverse-but-weak bases (lap-attention ρ0.90, embMLP ρ0.908) lift the blend above the linear 3-way OOF 0.95420?

## Method

Honest **cross-fitted** evaluation: meta-weights fit on 4/5 folds of the OOF, predict the held-out fold (so we don't reward OOF-overfitting the way the OOF-tuned linear weights do). [src/research/blend_metastack_probe.py](../src/research/blend_metastack_probe.py).

## Result

| combiner | bases | cross-fit OOF |
| -------- | ----- | ------------- |
| linear (OOF-tuned) | rm+cb+xgb | 0.95420 |
| NNLS | rm+cb+xgb | 0.95417 |
| NNLS | rm+cb+xgb+lap1 | 0.95417 |
| NNLS | rm+cb+xgb+lap1+lap3+emb | 0.95417 |
| NNLS | rm+xgb+lap1+emb | 0.95417 |
| logistic | rm+cb+xgb | 0.95358 |
| logistic | rm+cb+xgb+lap1+lap3+emb | 0.95359 |
| rank-avg | rm+cb+xgb | 0.95418 |

All NNLS variants are flat at **0.95417** — the diverse bases receive ~0 weight; honest cross-fit is a hair *below* the OOF-tuned 0.95420 (the linear weights are already near-optimal). Logistic stacking overfits/miscalibrates (0.95358). Rank-averaging ≈ linear.

## Verdict

**Inconclusive.** No combiner beats the linear 3-way, and adding the diverse-weak bases adds nothing (consistent with the oracle map: a base must be ρ≤0.92 *and* AUC≥0.951 to help; ours are diverse-XOR-strong). Closes the blend-combiner lever for the current base set. The linear weights also confirmed near-optimal and not materially overfit.

## Learnings

1. **Blend-combiner is exhausted.** With these bases, nothing beats `0.675·RM + 0.075·CB + 0.250·XGB`. Any lift must come from a stronger or genuinely-independent-and-strong *base*, not a smarter combiner.
2. **The OOF→LB drift (−0.00049) is not blend-overfit** — honest cross-fit (0.95417) ≈ OOF-tuned (0.95420), so the drift is in the bases/data, not the weight tuning.

## Follow-ups

- Closed: blend-combiner and diverse-weak-base addition.
- Remaining: strengthen a strong base (RealMLP already Optuna-tuned + n_ens24 + multiseed; XGB Optuna-tuned) or introduce a new strong+independent model. Both low-EV; the ceiling (OOF 0.95420 / LB 0.95372) is well-established.
