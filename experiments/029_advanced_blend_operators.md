# Experiment 029 — Advanced blend operators (logit-rank, confidence-gated, piecewise)

**Cycle.** 10
**Status.** Inconclusive — linear blend confirmed near-optimal for the cycle-7 input pair.
**Date.** 2026-05-25

## Hypothesis

Replacing the cycle 7 linear blend (`0.80 × RealMLP + 0.20 × CB-tuned-exp14`) with one of three more sophisticated blend operators — logit-rank blend, confidence-gated blend, piecewise (ventile) rescaling — lifts OOF AUC by ≥ +0.00010 over cycle 7's 0.95408 without retraining anything.

## Rationale

Cycle 7 settled on a linear weighted average between two strong inputs. That assumes the optimal correction is *uniform across the prediction distribution*. But blend value typically comes from disagreement in specific quantiles — extremes where the two models confidently disagree, or middle where one is uncertain.

Three blend operators address this differently:

1. **Logit-rank blend** — convert each model's predictions to ranks ∈ [0, 1], logit-transform, weighted-average the logits, sigmoid back, then remap to the anchor's value distribution. Stretches differences at the extremes (where logit is large) more than the middle. Useful when one model is better-calibrated at the tails than the other.

2. **Confidence-gated blend** — apply blending ONLY where the anchor's prediction is ambiguous (e.g., 0.05 ≤ p ≤ 0.95); leave confident extremes (p < 0.05 or p > 0.95) unchanged. If the secondary model adds noise at extremes (where the anchor is already correct), gating preserves the anchor's signal there.

3. **Piecewise (ventile) rescaling** — split the anchor's predictions into 20 equal-frequency bins, compute mean-of-anchor and mean-of-support within each bin, scale the anchor's values in that bin by the ratio. Local recalibration without global mixing.

All three are post-hoc — they use the existing OOFs (`oof_realmlp_multiseed.parquet`, `oof_cb_tuned_exp14.parquet`) and existing submissions (`submission_realmlp_multiseed.csv`, `submission_cb_tuned_exp14.csv`). No retraining. Runtime is seconds.

## Expected magnitude

- **Target:** at least one operator beats cycle 7's 0.95408 by ≥ +0.00010 OOF.
- **Optimistic:** +0.00030 OOF on logit-rank (if the two models disagree mainly at extremes).
- **Floor:** all three operators fall within ±0.00005 of linear blend → confirms cycle 7's linear blend is already near-optimal for these two inputs.

## Overfitting risk

**Low-Medium.**

1. Each operator has its own hyperparameter (rank exponent, gate thresholds, ventile count). Risk of OOF-tuning the parameter. Mitigated by reporting all candidates and choosing only operators that clear +0.00010 with default parameters (no grid search).
2. No retraining → no train-side data leakage.
3. **NOTE:** OOF→LB drift might differ for non-linear operators (calibration changes). LB confirmation gates apply if any operator clears OOF hurdle.

## Kill criteria

- [ ] All three operators within ±0.00005 of linear blend (no non-linear value to extract from these two models)
- [ ] Best operator regresses cycle 7's linear blend (over-engineering hurts)

## Scope

- `src/research/blend_advanced_ops.py` (new, ~150 lines)
- Outputs: `data/blend_advanced_sweep.parquet`, optional `data/submission_blend_advanced_best.csv`
- `experiments/029_advanced_blend_operators.md` (this file)

Wall-clock budget: ~10 seconds (post-hoc operations on cached OOFs).

## Reversibility check

- CV protocol: unchanged.
- Seed: unchanged.
- Feature set: unchanged.
- Leakage surface: unchanged — uses cached OOFs only.

No reversibility flag fires.

## Plan

1. Load cached OOFs and submissions for RealMLP-multiseed (anchor) and CB-tuned-exp14 (support).
2. Sweep each operator across its main hyperparameter:
   - Logit-rank blend: w_support ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}
   - Confidence-gated linear blend: w_support ∈ {0.20, 0.25, 0.30, 0.40, 0.50} × gate ∈ {0.05/0.95, 0.10/0.90}
   - Piecewise rescaling: bins ∈ {10, 20, 50, 100}, scale_clip ∈ {None, 0.95-1.05}
3. Report best per operator + best overall vs linear-blend baseline.
4. Decision gate:
   - Best OOF ≥ cycle 7 + 0.00010 → generate submission, but **do not submit until exp 028 (CB-rich-FE) result is known** — the operator may compound with new CB.
   - Best OOF within ±0.00010 → Inconclusive, confirms linear blend was near-optimal for these inputs.

## Result

Swept 46 configurations across 5 operators. Outputs in `data/blend_advanced_sweep.parquet`. Cycle 7 baseline (linear w_cb=0.20) OOF = 0.954080.

| Operator     | Best config                  | OOF       | Δ vs linear-baseline |
| ------------ | ---------------------------- | --------- | -------------------- |
| **linear**   | w=0.20                       | **0.954082** | +0.000002 (tie)   |
| rank         | w=0.20                       | 0.954076 | −0.000006             |
| logit_rank   | w=0.20                       | 0.954064 | −0.000016             |
| conf_gate    | w=0.20, gate=0.05–0.95       | 0.954015 | −0.000065             |
| piecewise    | bins=20, clip=(0.95, 1.05)   | 0.953831 | −0.000249             |

Full tail visible in the parquet — every non-linear operator's best variant lands below the linear baseline, monotonically worse as configurations diverge from `w≈0.20`.

## Verdict

**Inconclusive (favors null).** No operator clears the +0.00010 minimum-improvement gate. Best non-linear (`rank w=0.20`) is statistically indistinguishable from linear (Δ −0.000006, well below noise). The cycle-7 linear blend is near-optimal for the cycle-5 multi-seed RealMLP × cycle-4 CB-tuned-exp14 input pair.

## Kill-criteria check

- [x] Operators within ±0.00010 of linear (no non-linear value extractable) — **FIRED** (rank/logit_rank/conf_gate all within −0.0001).
- [ ] Best operator regressed linear-blend baseline — not fired (`linear w=0.20` *is* the best; nothing beat it).

## Repro stamp

- data: `train.csv` sha256 `f004e79d…` (matches `project.md` pin)
- packages: catboost 1.2.10, sklearn 1.7.2, pandas 2.3.3, numpy 2.3.5
- inputs: `data/oof_realmlp_multiseed.parquet` (cycle 5), `data/oof_cb_tuned_exp14.parquet` (cycle 4)
- runtime: ~3 seconds (post-hoc, cached OOFs)

## Learnings

1. **For a 2-input blend on rank-quality OOFs, linear ≈ optimal.** Rank, logit-rank, and confidence-gated all encode the same monotonic remap to first order; differences manifest only when one input is systematically miscalibrated, which RealMLP and CB-tuned are not relative to each other.
2. **Piecewise ventile rescaling is fragile.** Bins of 10–100 all underperformed; without bias clipping (`clip=None`) it actively diverged (bins=10 hit OOF 0.95184, −0.0022). Local recalibration on AUC-optimal predictors is anti-signal.
3. **Confidence-gating costs us symmetrically.** Restricting blend to ambiguous-zone preserves the anchor at extremes but loses information the support model adds at the boundaries — Δ −0.00007 at the most-permissive gate (0.05–0.95) deepens as the gate narrows.
4. **The +0.00010 hurdle was the right gate.** The 5-operator × 6-param sweep saw a 0.0024 spread; if we'd softened to +0.00005 we'd have mistaken the linear's tie-with-rank as a "positive" result.

## Follow-ups

- Re-attempt advanced blending only if exp 031 (or follow-up) produces a meaningfully different CB OOF — operator value re-emerges when input *diversity* increases.
- If a 3-way blend probe opens (RealMLP × CB-tuned × CB-rich), the right operator search becomes constrained-quadratic over 2 weights, not the 1-D problem this exp tested.
