# Experiment 055 — rich out-of-fold target-statistic encoding on XGB-highbins

**Cycle.** 16
**Status.** Inconclusive (Reverted) — leakage-free target stats slightly *hurt* standalone (−0.00041) and don't diversify (ρ 0.998 vs XGB); XGB already extracts group structure from frequency + group-stat features.
**Date.** 2026-05-27

## Hypothesis

Exp 051/052 closed *physics* FE on XGB (the ranking is locked, ρ 0.997–0.999). Target statistics are a different animal: the mean/std/count of the **target** within a group is near-soft-label signal, far stronger than any derived physics feature, so it could raise standalone AUC rather than just shuffle the ranking. XGB-highbins currently uses NO target encoding at all (only frequency counts + feature group-stats), so this is a genuine gap. Adding out-of-fold target mean/std/count for ~10 group keys lifts standalone OOF by ≥ +0.00020, or produces a base diverse enough (ρ < 0.98 vs XGB) to lift the 3-way blend.

## Rationale

- Target encoding is the one strong signal family our XGB base never uses.
- Higher moments (std, count) of the target within a group carry calibration/uncertainty signal beyond the mean.
- Properly cross-fit (inner 5-fold for train rows; full outer-train fold for val/test), it is leakage-free — the standard target-encoding scheme extended to higher moments and many keys.

## Expected magnitude

- Standalone target: ≥ +0.00020 over cycle-11 XGB-highbins 0.95263.
- Or blend target: 4-way / swap blend OOF ≥ 0.95441.
- Floor: standalone < 0.95243 AND rank-corr ≥ 0.984 → revert.

## Kill criteria

- [ ] Standalone OOF < 0.95243 AND no diversity gain (ρ ≥ 0.984).
- [ ] Best blend config does not clear 0.95421 + 0.00005.

## Scope / reversibility

New trainer [src/research/train_xgb_targetstats.py](../src/research/train_xgb_targetstats.py) (untracked → committed as a closed dead-end). Adds OOF target mean/std/count for ~10 group keys ahead of the verbatim cycle-11 XGB-highbins pipeline; inner-CV cross-fit on train rows, outer-train stats for val/test. Does not touch CV/seed/target/frozen files. Run locally on M1 (CPU). Reversible.

## Result

Ran locally, 5-fold, cycle-11 XGB-highbins HPs verbatim, with the OOF target-stat block added.

| Fold | iters | AUC |
| ---- | ----- | --- |
| 1 | 3650 | — |
| 2 | 3320 | — |
| 3 | 3375 | 0.95182 |
| 4 | 3544 | 0.95135 |
| 5 | 3855 | 0.95250 |
| **OOF** | — | **0.95222** |

per-fold mean 0.95223, std 0.00057. **OOF 0.95222 vs cycle-11 XGB-highbins 0.95263 → Δ −0.00041** (slightly *worse* than plain XGB).

### Rank-correlation (OOF)

| vs RealMLP-ms | vs CB-tuned14 | vs XGB-highbins |
| ------------- | ------------- | --------------- |
| 0.97964 | 0.98433 | **0.99771** |

ρ 0.998 vs plain XGB — same family as the physics-FE variants (051/052). No new diversity.

## Verdict

**Inconclusive (Reverted).** Target stats both *fail to lift standalone* (−0.00041) and *fail to diversify* (ρ 0.998). The "stronger signal → higher standalone AUC" premise didn't hold: XGB-highbins already captures the group structure through its frequency counts + feature group-stats at max_bin=5000, so the OOF target-encoding adds mostly cross-fit variance, which slightly hurts. Both the standalone and the blend kill-criteria fire.

## Kill-criteria check

- [x] Standalone 0.95222 < 0.95243 AND ρ 0.998 ≥ 0.984 — **FIRED** (worse *and* non-diverse).
- [x] No blend lift possible (strictly dominated by XGB-highbins, ρ 0.998) — **FIRED**.

## Repro stamp

- trainer: [src/research/train_xgb_targetstats.py](../src/research/train_xgb_targetstats.py); recipe = cycle-11 XGB-highbins HPs (max_bin=5000, eta=0.01, depth=10, λ=8.16, α=8.35, colsample=0.145), 5-fold StratifiedKFold seed 42 on Year×PitNextLap + external
- outputs: `data/oof_xgb_targetstats.parquet`, `data/submission_xgb_targetstats.csv`
- runtime: ~70 min on M1 CPU

## Learnings

1. **Target encoding is not a free standalone lift here.** Even the strongest signal family available (OOF target mean/std/count) slightly *hurt* XGB-highbins. The high-bin frequency/group-stat features already encode the group structure; the extra cross-fit target stats add variance without new information.
2. **Confirms the FE-on-XGB close from a third angle.** 051 (trajectory) and 052 (tyre-overdue) closed *physics* FE; 055 now closes *target-statistic* FE — ρ 0.998 again. XGB's ranking is locked to feature work of any kind. New diversity must come from a different algorithm, not new features.

## Follow-ups

- Closed: target-statistic encoding on XGB as either a standalone or diversity lever.
- The live cycle-16 thread is the external-data axis (056 no-external ablation, 057 adversarial reweighting), not more FE on XGB.
