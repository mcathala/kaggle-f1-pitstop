# Experiment 036 — Per-slice blend weights (Year × Position_Change × PitStop)

**Cycle.** 11
**Status.** Inconclusive (Reverted) — per-slice weights overfit folds; uniform `w_cb=0.20` remains optimal.
**Date.** 2026-05-25

## Hypothesis

The optimal `w_cb` in cycle 7's RealMLP × CB-tuned blend differs across data slices. Specifically, cycle 10's probe 2 found Q4 worst-loss concentrates on rows with `Position_Change < 0` AND high `PitStop` cluster. A slice-aware blend (different `w_cb` per slice, fit by per-slice grid search on OOF) extracts ≥ +0.0001 OOF over cycle 7's uniform `w_cb=0.20`.

## Rationale

Cycle 7 uses a single `w_cb=0.20` across all 439k rows. But cycle 10 probe 4 showed RealMLP × CB rank-disagreement concentrates on:
- `Position_Change` very negative (driver gaining positions) — gap 266%
- High `Cumulative_Degradation` (degraded tyres) — gap 27%
- High `Driver` cardinality slices — gap 12%

If the two models disagree systematically on these slices, the OPTIMAL `w_cb` should differ there. Possible patterns:
- **Slice where RealMLP is unreliable** → higher `w_cb` helps
- **Slice where CB is unreliable** → lower `w_cb` helps
- **Slice where both agree** → `w_cb` ≈ 0.20 is fine

By grid-searching per slice and choosing the per-slice best `w_cb`, we extract a non-uniform combination that the global `w_cb=0.20` cannot represent.

**Overfitting risk control:** the slices must be chosen *a priori* (from probe 4), not by greedy search. We use Year × sign(Position_Change) × PitStop_bin → ≤16 slices × 5 weight choices = 80 cells, well below the 439k-row sample size. Per-fold OOF AUC validates the slice weights aren't memorizing.

## Expected magnitude

- **Target:** OOF lift ≥ +0.0001 over cycle 7's 0.95408 → ≥ 0.95418.
- **Optimistic:** +0.0003 → 0.95438 (would clear the +0.00020 hurdle 0.95428).
- **Floor:** OOF lift ≤ 0 → slice variance is noise; uniform `w_cb=0.20` is genuinely optimal.

## Overfitting risk

**Medium.** Per-slice weights have more degrees of freedom than a uniform weight. Mitigations:
1. Slices chosen from probe 4 evidence (not greedy search).
2. 5-fold CV on the slice-aware blend OOF; if per-slice weights overfit, OOF AUC will regress.
3. Each slice still has ≥ 5k rows (439k / 16 slices = 27k average) — sample size is fine.

## Kill criteria

- [ ] Slice-aware blend OOF < cycle 7's 0.95408 → per-slice weights are over-fitting noise.
- [ ] Per-slice best `w_cb` shows no spatial pattern (random across slices) → no slice signal exists.

## Scope

- `src/research/blend_per_slice.py` (new, ~150 lines).
- Output: `data/blend_per_slice_sweep.parquet` (per-slice best w_cb), `data/oof_blend_per_slice.parquet`, conditional `data/submission_blend_per_slice.csv` if hurdle cleared.

Compute: ~3-5 min, low CPU (post-hoc on cached OOFs).

## Reversibility check

No retraining, no CV change, no leakage surface — pure post-hoc blending on cached OOFs.

## Plan

1. Load RealMLP-multiseed OOF + CB-tuned-exp14 OOF + raw train DataFrame.
2. Define slices: Year ∈ {2022, 2023, 2024, 2025} × sign(Position_Change) ∈ {neg, zero, pos} × PitStop_bin ∈ {low, high} = up to 24 slices.
3. For each slice, grid-search `w_cb ∈ {0.0, 0.05, ..., 0.40}` on a TRAIN portion of the OOF (leave-out-fold CV).
4. Construct full slice-aware blend OOF using per-slice best weights.
5. Compare to cycle 7's uniform-weight blend OOF.

## Result

Nested-CV per-slice blend OOF: **0.95400** (vs cycle 7 uniform 0.95408, **Δ = −0.00008**).

Per-fold AUCs (5-fold nested CV, slice weights re-fit per outer fold):

| Fold | Slice-aware AUC | Slice weights fit |
| ---- | --------------- | ----------------- |
|  1   | 0.95455         | 24                |
|  2   | 0.95465         | 24                |
|  3   | 0.95372         | 24                |
|  4   | 0.95301         | 24                |
|  5   | 0.95411         | 24                |

Per-fold std: **0.00060** (vs cycle 7's typical ~0.00040 — meaningfully higher).

Slice weight distribution from a single full-data fit (for inspection only — not used in the OOF):
- Heterogeneity is real: optimal `w_cb` ranges from 0.00 (e.g., `2023_neg_lo`, n=7,359) to 0.40 (e.g., `2022_zero_hi`, n=1,165).
- Top-15 by row count: most slices want `w_cb` between 0.10 and 0.25.
- The largest slice (2023_zero_lo, n=106,437) wants `w_cb=0.20` — same as cycle 7's global default. So the dominant slice contributes nothing.

## Verdict

**Inconclusive (Reverted).** The slice signal is real in single-fit AUC but does NOT transfer across folds: per-fold std of 0.00060 is 50% higher than the project's typical 0.00040 fold variance, and the OOF AUC actually regresses by −0.00008. The 24 slice weights have too many degrees of freedom for the underlying signal.

## Kill-criteria check

- [x] Slice-aware blend OOF < cycle 7's 0.95408 — **FIRED** (0.95400 < 0.95408).
- [ ] Per-slice best `w_cb` shows no spatial pattern — not fired (clear heterogeneity).

## Repro stamp

- inputs: `oof_realmlp_multiseed.parquet`, `oof_cb_tuned_exp14.parquet`, `train.csv`
- runtime: ~30 sec

## Learnings

1. **24-slice granularity is too fine for this OOF size.** With 439k rows / 24 slices ≈ 18k rows per slice average, but the smallest slices have only 1-2k. Fold-level noise per slice dominates the signal.
2. **The dominant slice (n=106k) wants `w_cb=0.20`** — same as the global default. This means the per-slice "improvement" is entirely concentrated in small slices, which is exactly where fold-overfitting is worst.
3. **Cycle 7's `w_cb=0.20` is robustly optimal at the global level.** Per-fold std with uniform weight stays at ~0.00040; per-slice adds 50% noise without lifting the mean.
4. **Hypothesis test design lesson:** when probing a structure with many degrees of freedom, the kill criterion should be "per-fold std + OOF AUC together" — either alone is insufficient.

## Follow-ups

- **Exp 037 candidate:** quantile-aware blending. Use prediction-quantile buckets (5 buckets, low to high probability) instead of feature-space slices. Probe 4 said disagreement concentrates on low-prob rows; quantile-bucketing tests that directly with far fewer DoF (5 vs 24).
- **NOT a candidate:** per-Year-only slicing (4 slices). Probe 1's family-importance ranking didn't single out Year as a coverage gap; the gain would just be the 4-cell average of this 24-cell result.
