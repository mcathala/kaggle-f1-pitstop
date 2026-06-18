# Experiment 039 — RealMLP seed-subset selection

**Cycle.** 11
**Status.** Inconclusive (Reverted)
**Date.** 2026-05-25

## Hypothesis

A k-seed subset (k < 5) of the cycle-5 RealMLP multi-seed average has higher blend AUC than the full 5-seed average, because by dropping seed(s) highly correlated with CB, the remaining subset adds more effective diversity.

## Result

| Subset | k | rm_auc | rank-corr CB | best w_cb | best_blend_auc | Δ vs cycle 7 |
|---|---|---|---|---|---|---|
| **42+7+99+137+313 (full)** | **5** | **0.95382** | **0.97569** | **0.20** | **0.95408** | **0** |
| 7+99+137+313 (drop 42) | 4 | 0.95382 | 0.97569 | 0.20 | 0.95408 | 0 |
| 7+99+313 | 3 | 0.95381 | 0.97543 | 0.20 | 0.95407 | −0.00001 |
| 99+137+313 | 3 | 0.95380 | 0.97556 | 0.20 | 0.95406 | −0.00002 |
| 99+313 | 2 | 0.95377 | 0.97509 | 0.20 | 0.95404 | −0.00004 |
| 313 (best 1-seed) | 1 | 0.95362 | 0.97447 | 0.225 | 0.95395 | −0.00013 |

**Subsets beating cycle 7's 0.95408: 0.** Subsets reaching hurdle 0.95428: 0.

## Verdict

**Inconclusive (Reverted).** Seed 42's marginal contribution is exactly zero — the 4-seed average (7+99+137+313) ties the 5-seed average down to the 5th decimal. Beyond that, every k<4 subset loses AUC monotonically. Cycle 5's 5-seed average is at the variance-reduction ceiling for this input set.

## Learnings

1. **Seed 42 contributes nothing additive to the 5-seed RealMLP average.** Likely because seed 42's predictions are highly correlated with seeds 7, 99, 137, 313 (the rank-corr-with-CB is identical to the 4-seed at 0.97569). This is a real finding: future RealMLP multi-seed experiments could drop seed 42 with zero AUC cost.
2. **Rank-corr-with-CB doesn't vary meaningfully across subsets** (range 0.9744-0.9757). Subset selection can't engineer diversity; that's set by the model family.
3. **Blend ceiling at 0.95408 is real.** Across exp 029, 036, 037, 038, 039, NOTHING in post-hoc analysis beats it. The bottleneck is the inputs, not the recipe.

## Follow-ups

- Post-hoc blend axis is **conclusively exhausted**. Only path forward: change the CB-side input via exp 035 (multi-seed CB-tuned, in progress).
- **Exp 040 candidate:** 3-way blend with one of the cached CB variants (CB#006, CB#007, CB#009, CB-tuned cycle-3) as the third slot. All have different recipes than CB-tuned-exp14. Untested 3-way combinations.
