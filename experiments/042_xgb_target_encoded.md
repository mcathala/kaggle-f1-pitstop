# Experiment 042 — XGBoost with target-encoded categoricals

**Cycle.** 11
**Status.** Inconclusive (Reverted)
**Date.** 2026-05-26

## Hypothesis

Replacing XGBoost's native categorical-split with **target-encoded categoricals** (sklearn `TargetEncoder` with cross-fold) lifts XGBoost's standalone OOF from 0.94615 (exp 034) to ≥0.948. With the higher standalone, the 3-way blend `(RealMLP, CB14, XGB-TE)` could allocate non-zero weight to XGB-TE (because XGB-TE's rank-corr with RealMLP and CB14 should remain low — the TE preprocessing changes XGBoost's internal logic but should still produce different rank ordering than CB's ordered TE).

## Rationale

Exp 034 / 041 confirmed XGBoost has a 0.946 AUC ceiling with native `enable_categorical=True`. The bottleneck is XGBoost's split-based cat handling, which is structurally weaker than CB's ordered target encoding for high-cardinality categoricals like Driver (887 levels) and the cross-cats (Driver_Race, etc.).

By pre-encoding categoricals to target-frequency means (with cross-fold to avoid leakage), XGBoost sees them as ordinary numeric features with high signal-to-noise. This should:
- Lift standalone AUC by giving XGB the same signal CB extracts via ordered TE.
- Preserve some rank diversity because XGBoost's TREE-building (level-wise, histogram) differs from CB's (symmetric oblivious + ordered boosting).

## Expected magnitude

- **Standalone AUC target:** ≥ 0.948 (+0.002 over exp 034's 0.94615).
- **Optimistic:** ≥ 0.950 → comparable to CB14's 0.95114, would meaningfully change blend math.
- **Floor:** standalone AUC < 0.947 → TE didn't help, direction dead.

## Overfitting risk

**Medium.** sklearn's `TargetEncoder` uses cross-fold by default but it's not as leak-proof as CB's ordered TE. Will use 5-fold internal CV for TE to align with the project's outer 5-fold protocol. If rank-corr with RM/CB14 stays low (< 0.96), the cross-fold encoding worked correctly.

## Kill criteria

- [ ] Standalone AUC < 0.946 — TE made things worse.
- [ ] Rank-corr with CB14 > 0.98 — TE replicated CB's predictions (no diversity).
- [ ] 3-way blend w_xgb = 0 — even with better standalone, optimizer rejects it.

## Scope

- `src/research/train_xgb_te.py` (new, ~250 lines).
- Outputs: `data/oof_xgb_te.parquet`, `data/submission_xgb_te.csv`.

Compute: ~5-10 min total (similar to exp 034 since the model is the same; only the FE step is different).

## Reversibility check

CV protocol unchanged. Different FE recipe (TE-encoded instead of native categorical), but doesn't affect base models' OOFs. No leakage concerns if cross-fold TE is correctly implemented.

## Result

Cross-fold `TargetEncoder` preprocessing *hurt* XGBoost rather than helping: fold-1 AUC 0.941, folds 2–3 ~0.940 — below exp 034's native-categorical 0.94615. Killed early once the trajectory was clearly negative.

## Verdict

**Inconclusive (Reverted).** Target-encoding the categoricals strips the high-cardinality structure XGBoost's native split was actually exploiting; the TE-mean features are lower signal-to-noise than expected here. The XGBoost cat-handling bottleneck is not fixed by pre-encoding — it was later addressed by high-resolution histograms (`max_bin=5000`, exp 044) instead. Direction closed.
