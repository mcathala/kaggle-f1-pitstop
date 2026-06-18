# #002 — Drop three zero-gain features

**Status.** Reverted
**Date.** 2026-05-08

## Hypothesis

Single-fold feature-importance on the baseline confirmed three features have **literally zero gain** (and zero splits) in the LightGBM model:

- `sc_likely` — redundant with the continuous `field_pace_ratio` the tree splits on instead.
- `cant_finish_on_current_tyres` — saturated (99.9% true), no information.
- `lap_is_anomalous` — redundant with `field_pace_ratio` and `driver_pace_ratio`.

Dropping these from the feature set should produce an essentially identical OOF AUC (since the tree never used them), but cleans the feature space, slightly improves `feature_fraction=0.9` sampling efficiency, and reduces noise in future feature-importance analyses.

## Expected impact

- **OOF AUC: ~+0.0000 to +0.0002.** Dropping zero-gain features can only change OOF via the bagged-feature-fraction sampling. Expect noise.
- Cleaner foundation for cycles #3+ (HARD-compound features, ensembling).

## Overfit risk

**Zero.** Pure feature subset reduction. Same CV protocol, same hyperparameters.

## Implementation

[src/train.py](../src/train.py) — add a `DROP_FEATURES` list and exclude from `feature_cols`. Features remain computed in [src/features.py](../src/features.py) (so the parquets are unchanged); only the model's input set shrinks.

## Validation gates

- [ ] OOF AUC ≥ 0.94166 − 0.0002 (i.e. ≥ 0.94146). If lower, something is wrong; revert.
- [ ] Per-fold std stays ≤ 0.0008.
- [ ] No per-year regression > 0.001.

## Result

```
OOF AUC:   0.94161   (vs baseline 0.94166, Δ = −0.00005)
Per-fold:  fold std 0.00063 (vs baseline 0.00042 — increased)
best iter: [320, 311, 312, 310, 311] mean ~313 (vs baseline mean ~676 — halved)
```

Per-year:
- 2022: 0.89895 (≈ baseline 0.89892)
- 2023: 0.92437 (vs 0.92364, +0.00073)
- 2024: 0.91594 (≈ baseline 0.91599)
- 2025: 0.91573 (vs 0.91596, −0.00023)

## Decision

**Reverted.** Two surprising effects:

1. **Fold std went up by 50%** (0.00042 → 0.00063). The model is less stable across folds when zero-gain features are removed.
2. **Best iteration halved** (~676 → ~313). LightGBM is hitting early stopping much sooner.

Hypothesis: with `feature_fraction=0.9` bagging, the three "dead" features were acting as harmless filler in random subsets, helping the trees stay diverse. Dropping them shrinks the candidate pool from 63 → 60, which means each tree's 0.9-sampled subset overlaps more with neighbouring trees, reducing diversity → faster overfit detection by early stopping → fewer trees → slightly lower OOF AUC.

This is a bagging-side effect, not a feature-quality effect. The "dead" features are still genuinely zero-gain individually but have a small positive role in the ensemble through randomness. Lesson: **don't prune purely on individual gain when the model uses bagging**.

Reverting `src/train.py` to baseline.

## Observations / followups

- If we ever raise `feature_fraction` to 1.0 (no bagging), this experiment would need re-running — at that point the dead features really would be no-ops.
- Cycle #3 (HARD-compound features) is the more promising direction — we should add useful features rather than remove fillers.
