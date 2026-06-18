# Experiments — living log

Track every change we make to features, model, or CV, and whether it improved the goal. **Goal:** maximize ROC-AUC on the Kaggle test set (mirrored by 5-fold OOF AUC, see [eda.md §8](eda.md#8-traintest-split-structure-important-for-cv)).

How to use this doc:
- One section per experiment, newest at the top.
- Always record OOF AUC under the **same CV setup** as the current best so numbers compare. If you change the CV, that's its own experiment.
- Record Kaggle LB score when you submit. Note any divergence between OOF and LB.
- Mark decisions explicitly: **Kept** / **Reverted** / **Pending**.

## Current best

| | OOF AUC | Public LB | Notes |
|---|---|---|---|
| **Baseline (#001)** | **0.94166** | **0.94211** | LightGBM, 49 engineered features, 5-fold StratifiedKFold on Year × PitNextLap. LB − OOF = +0.00045 → CV is trustworthy. |

## CV protocol (the bar all results are measured against)

- 5-fold `StratifiedKFold(shuffle=True, random_state=42)` stratified on `Year × PitNextLap`. Mirrors the row-level split (see [eda.md §8](eda.md#8-traintest-split-structure-important-for-cv)).
- Test predictions are the mean of the 5 fold models, each at its own `best_iteration`.
- AUC is computed on out-of-fold predictions concatenated across folds.

If you change any of the above, log it as an experiment — it changes what "AUC" means.

---

## #001 — LightGBM baseline (2026-05-08)

**Change.** First model. LightGBM on 63 columns (49 engineered + 14 raw, dropping `id` and `PitNextLap`). Native categorical handling for `Driver`, `Race`, `Compound`. Params: `num_leaves=63`, `learning_rate=0.05`, `min_data_in_leaf=100`, `feature_fraction=0.9`, `bagging_fraction=0.9`, early stopping at 100 rounds, max 5000 boosting rounds. Code: [src/train.py](../src/train.py).

**Rationale.** Anchor a number before tuning anything. LightGBM handles the 887-level `Driver` natively and is fast enough to iterate on.

**Result.**

```
OOF AUC:   0.94166
Public LB: 0.94211   (Δ vs OOF = +0.00045 — CV is trustworthy)
Per-fold AUC: 0.94158, 0.94215, 0.94091, 0.94193, 0.94177  (std 0.00042)
Best iteration per fold: 630, 630, 783, 758, 579
```

Per-year OOF AUC:

| Year | AUC | n | pos rate |
|---|---|---|---|
| 2022 | 0.89892 | 82,989 | 0.2665 |
| 2023 | 0.92364 | 136,147 | 0.0096 |
| 2024 | 0.91599 | 127,110 | 0.2953 |
| 2025 | 0.91596 | 92,894 | 0.2844 |

Test prediction mean: 0.196 (train target rate 0.199 → no drift).

**Decision.** **Kept** as baseline.

**Observations / followups.**
- Fold std is tight (0.00042) — folds are well-balanced; CV is trustworthy.
- 2022 has the lowest per-year AUC despite "normal" pit rate; 2023's higher AUC is partly the trivially-predictable near-zero label rate. Worth digging into 2022 — possible feature gap or generator quirk.
- Blended OOF AUC (0.942) > any per-year AUC because `Year` itself separates classes; do not over-read the gap.
- Three known feature-calibration issues from [feature_engineering.md §5](feature_engineering.md#5-known-calibration-issues) (`cant_finish_on_current_tyres`, `in_pit_window`, `tyre_life_norm`) still in place — candidates for future experiments.
- LB tracks OOF within 0.0005 → we can trust 5-fold OOF as a proxy and avoid burning daily submission budget on speculative changes.

---

## Template (copy when adding an experiment)

```markdown
## #NNN — Short title (YYYY-MM-DD)

**Change.** What was changed, with file pointers.

**Rationale.** Why we expect this to help.

**Result.**
- OOF AUC: 0.XXXXX (vs prev best 0.XXXXX, Δ +/-0.XXXXX)
- Public LB: 0.XXXXX  (or "not submitted")
- Per-fold AUC: ...
- Per-year AUC: ... (only if it shifted meaningfully)

**Decision.** Kept / Reverted / Pending — and why.

**Observations / followups.** Surprises, regressions, ideas this opens up.
```
