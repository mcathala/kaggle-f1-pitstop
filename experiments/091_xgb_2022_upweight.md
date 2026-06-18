# Experiment 091 — 2022-upweighted diffFE-XGB → worse everywhere (2022 weakness is intrinsic)

**Cycle.** 18 · **Date.** 2026-05-29
**Status.** **Reverted (negative).** Upweighting the weak slice made it — and the pooled metric — worse.

## Hypothesis

Diagnostic found 2022 is the weakest within-year slice (AUC 0.920). If that's under-fitting (smallest year, regime change), upweighting 2022 rows (sample_weight 1.6 on train+ext) should lift 2022 → help pooled. Gate: pooled OOF ≥ 0.95291 and 2022 within-year up.

## Result

| | pooled OOF | 2022 | 2023 | 2024 | 2025 |
| --- | --- | --- | --- | --- | --- |
| diffFE-XGB (baseline) | 0.95291 | 0.920 | 0.953 | 0.935 | 0.935 |
| **2022-upweight 1.6** | **0.95272** | **0.918** | 0.951 | 0.932 | 0.932 |
| Δ | **−0.00019** | −0.002 | −0.002 | −0.003 | −0.003 |

**Worse on every slice, including 2022 itself.**

## Verdict

**Reverted.** 2022's weakness is **not under-fitting** — it's intrinsic difficulty (2022 ground-effect regulation change shifted tyre/pit dynamics + the shared ~32% label noise). Upweighting forces the splitter to fit 2022's *noise* harder → worse 2022 generalization, and the weight imbalance distorts the global fit → all years drop. Confirms the diagnostic's read: the within-year ceiling is a data-difficulty limit, not a weighting/capacity one. **Closes 2022-targeted reweighting** (with the per-year-blend-weights closure, the whole "exploit year heterogeneity" axis is now closed).

## Repro stamp

- Trainer: [src/research/train_xgb_y2022up.py](../src/research/train_xgb_y2022up.py) (WEIGHT_2022=1.6).
- Output: `data/oof_xgb_y2022up.parquet` (0.95272) — not used.

## Learnings

1. **A diagnosed weak slice ≠ a fixable weak slice.** 2022 is hard because of its data regime + noise, not because the model under-weights it. Reweighting a noisy-hard slice amplifies its noise.
2. The own-pipeline OOF ceiling (0.95462 pooled) is confirmed from yet another angle. Loop pivots fully to LB-candidate generation: the daily slots are now the primary information source.
