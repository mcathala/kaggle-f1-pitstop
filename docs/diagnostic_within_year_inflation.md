# Diagnostic — the pooled OOF is inflated by between-year separation

**Cycle.** 18 · **Date.** 2026-05-29

## Finding

The best-blend pooled OOF is **0.95462**, but the **within-year** AUCs are far lower:

| Year | within-year AUC | n (train) | n (test) | pit rate |
| --- | --- | --- | --- | --- |
| **2022** | **0.920** | 82,989 | 35,348 | 0.267 |
| 2023 | 0.953 | 136,147 | 58,160 | **0.010** (anomaly) |
| 2024 | 0.935 | 127,110 | 54,532 | 0.295 |
| 2025 | 0.935 | 92,894 | 40,125 | 0.284 |

A pooled AUC of 0.9546 sitting *above* every within-year AUC is the signature of **between-group separation doing the work**: 2023 pits only 1% of the time, so 2023-negatives rank trivially below other-year-positives. Those cross-year pairs dominate the pair count and are nearly free — every competitor gets them. **The real, competitive signal is within-year ranking**, where our ceiling is 0.92–0.95 and **2022 is the weakest by a wide margin** (likely the 2022 ground-effect regulation change shifted tyre/pit dynamics our features fit worse).

Per-stint: stint 2 weakest (0.923, highest pit rate 0.391); stint 1 best (0.945).

## Per-year blend weights — tested, CLOSED

If bases were unequally good per-year, per-year blend weights should help. They don't:

| weighting | pooled OOF |
| --- | --- |
| global | **0.95462** |
| per-year (in-sample) | 0.95455 |
| per-year (nested honest) | 0.95452 (−0.0001) |

Per-year weighting is *worse* — because the pooled metric needs **cross-year calibration**, and optimizing weights within each year separately breaks the between-year ranking. **The global blend is already optimal for the pooled metric.**

## Implications

1. **The plateau is a within-year skill ceiling shared by the whole field.** Everyone gets the between-year freebie; the 0.0004-wide jam at the top reflects tiny within-year skill differences. This is consistent with the ~32% instance-dependent label noise capping within-year separability.
2. **2022 is the one concrete weak slice** (0.920). Improving it only modestly moves pooled (2022 is ~19% of rows and its within-year pairs are a fraction of total), but it's the most-defensible target → exp 091 (2022-upweight GBDT).
3. **Gating must stay on pooled OOF** (it's what Kaggle scores), but be aware small pooled gains may just reshuffle the easy between-year part — prefer changes that demonstrably lift a weak within-year slice.
