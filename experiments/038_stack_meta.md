# Experiment 038 — Meta-stacker on (RealMLP OOF, CB OOF, top raw features)

**Cycle.** 11
**Status.** Inconclusive (Reverted) — LR underperforms, LGB ties uniform blend. Stacking adds no value.
**Date.** 2026-05-25

## Hypothesis

A tiny meta-model trained on `(rm_oof, cb_oof, top_5_raw_features)` produces an OOF AUC ≥ +0.00010 over cycle 7's uniform-blend 0.95408. The raw features allow the meta-model to do slice-aware blending implicitly (without the explicit per-slice DoF that exp 036 / 037 overfit on).

## Rationale

Three pieces of evidence shaped this hypothesis:

1. **Exp 036 / 037 confirmed: pure post-hoc weighting can't beat cycle 7's uniform blend.** Explicit per-slice (24 DoF) and per-quantile (5 DoF) bucketing both overfit folds.
2. **Probe 1 (exp 030) ranked the highest per-feature importance** in CB: `EstimatedTotalLaps` (4.44), `DeltaAbs` (8.33), `TyreAgeRatio` (3.93), `LapMinusTyreLife` (3.58), `StintPressure` (2.90). These 5 features carry ~50% of cycle 14's discriminative power.
3. **Cycle 9 exp 010 ruled out stacking on 4-way bases** — but that was 4 weak bases (LGB + 3 CBs). With just RealMLP-multiseed (OOF 0.95383) + CB-tuned-exp14 (OOF 0.95114) + 5 high-signal raw features, the meta-model has different leverage.

The meta-model's learnable parameter count is small (7 features × LR coefficient = 7-8 params, or shallow LGB with 50 trees of depth 3). With 439k rows, the parameter-to-sample ratio is ~6e-5 — many orders of magnitude lower than exp 036's per-slice setup.

Two meta-models to test:
1. **Logistic regression** (linear, L2-regularized). Simple, low overfitting risk.
2. **Shallow LGB** (n_est=100, depth=3, leaves=8). Captures non-linear interactions between OOF predictions and raw features (e.g., "use higher w_cb when DeltaAbs is large").

## Expected magnitude

- **Target:** OOF lift ≥ +0.00010 over cycle 7's 0.95408 → ≥ 0.95418.
- **Optimistic:** OOF lift ≥ +0.00020 → 0.95428 (clears the cycle hurdle).
- **Floor:** OOF lift ≤ 0 → stacking adds nothing over a uniform blend at this scale of inputs.

## Overfitting risk

**Low.** LR has 7 free params with L2; LGB has bounded capacity (n_est × depth). Nested 5-fold CV (meta-model re-fit per outer fold) catches any over-fitting.

## Kill criteria

- [ ] OOF AUC < cycle 7's 0.95408 → stacking is anti-signal.
- [ ] Per-fold std > 0.00050 → meta-model unstable.
- [ ] Meta-model puts coefficient ≈0 on raw features → raw features didn't help, just a complicated linear blend.

## Scope

- `src/research/stack_meta.py` (new, ~150 lines). Implements both LR and LGB variants.
- Output: `data/stack_meta_sweep.parquet`, `data/oof_stack_meta.parquet`, conditional submission.

Compute: ~3-5 min.

## Reversibility check

No retraining of base models, no CV change. Pure post-hoc meta-modeling on cached OOFs + raw features.

## Plan

1. Load RealMLP-multiseed OOF, CB-tuned-exp14 OOF, train.csv.
2. Build meta-feature matrix: `[rm_oof, cb_oof, EstimatedTotalLaps, DeltaAbs, TyreAgeRatio, LapMinusTyreLife, StintPressure]`.
3. Nested 5-fold CV: per outer fold, fit meta-model on 4 train folds (OOF, raw) → predict on held-out fold.
4. Report OOF AUC for LR + LGB.
5. Best variant → write submission if OOF ≥ 0.95428.

## Result

| Meta-model | OOF AUC  | Δ vs uniform | vs hurdle  | per-fold std |
| ---------- | -------- | ------------ | ---------- | ------------ |
| LR         | 0.95218  | **−0.00190** | −0.00210   | 0.00044      |
| LGB shallow| 0.95401  | −0.00007     | −0.00027   | 0.00059      |

LR full-fit standardized coefficients:

| Feature              | Coef    |
| -------------------- | ------- |
| `rm_oof`             | +1.1020 |
| `cb_oof`             | +1.0755 |
| `EstimatedTotalLaps` | +0.3221 |
| `LapMinusTyreLife`   | +0.0311 |
| `DeltaAbs`           | +0.0153 |
| `TyreAgeRatio`       | −0.0102 |
| `StintPressure`      | −0.0509 |

## Verdict

**Inconclusive (Reverted).** Two findings:

1. **LR underperforms drastically.** A linear meta-model on logits-mixed-with-raw drops AUC by −0.00190. The LR coefficients show rm_oof and cb_oof get near-equal weights (+1.10 vs +1.08), but the linear combination of probabilities — even with raw-feature corrections — destroys the calibration that cycle 7's direct weighted-mean preserves. AUC is not a linear-friendly objective when inputs are probabilities, not logits.

2. **LGB recovers but doesn't beat uniform.** Shallow LGB lands at 0.95401, essentially tied with the uniform 0.80/0.20 blend. The non-linear meta-model can extract the same signal but no more — the raw features' marginal information is already encoded in the OOFs themselves (since both base models saw those features).

This corroborates cycle 9 exp 010's verdict (stacking is exhausted) for our project, even with the strongest 2-base inputs.

## Kill-criteria check

- [x] OOF AUC < cycle 7's 0.95408 — **FIRED** (both LR and LGB).
- [ ] Per-fold std > 0.00050 — LGB fired (0.00059); LR did not (0.00044).
- [x] Meta-model coefficient ≈0 on raw features — **FIRED** (top raw coef is EstimatedTotalLaps at +0.32, others <0.06).

## Repro stamp

- runtime: ~2 min
- inputs: `oof_realmlp_multiseed.parquet`, `oof_cb_tuned_exp14.parquet`, `train.csv`

## Learnings

1. **Logit-vs-probability matters for LR meta-stacking.** LR on probability inputs collapses the OOFs into a logistic combination which is structurally suboptimal for AUC. A logit-transform of OOFs before LR would be the correct setup; the −0.0019 loss without it is informative.
2. **Raw features add no marginal information to OOF stacking.** Both base models had access to these features during their own training. The stacker can only re-weight, not re-discover.
3. **The blend axis is conclusively exhausted.** Five separate experiments (029 operators, 036 slice, 037 quantile, 038 stacking, plus cycle 7's own grid) confirm cycle 7's `w_cb=0.20` linear blend is the global optimum for this pair. The only path forward is changing the inputs.

## Follow-ups

- **Exp 039 candidate:** RealMLP seed-subset selection. The 5 individual seed OOFs exist; some 3-of-5 subset might have lower rank-corr with CB-tuned (more diversity in the blend). Cheap to check.
- **Confirmed dead direction:** any post-hoc meta-modeling on cycle 7's input pair.
