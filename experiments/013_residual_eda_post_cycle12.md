# #013 — Residual EDA on the post-cycle-12 4-way ensemble

**Status.** Kept (EDA cycle, no model change)
**Date.** 2026-05-21
**Focus.** `eda` — re-run the cycle 8 slice scan on the new current best.

## Why this cycle

Cycle 12 lifted ensemble OOF 0.94866 → 0.95134 and Public LB 0.94833 → 0.95066 (+0.00268 OOF / +0.00233 LB). The gap to the public LB top of 0.95488 went from −0.00655 to −0.00422 — ~36% closed in one cycle. To plan cycle 14 well, we need to know which slices closed and which still gap.

## Method

`src/research/cycle013_error_eda.py` is `src/research/cycle008_error_eda.py` with `ENSEMBLE_OOF` repointed at `data/oof_ensemble_cycle012.parquet`. Same scans, same `lift_$ = n × (global_AUC − slice_AUC)` metric. Direct comparison.

## Findings

### Global
```
rows:           439,140
OOF AUC:        0.95134  (was 0.94866)
log-loss:       0.24893  (was 0.22652 — WORSE)
positive rate:  0.1990
```

Log-loss **rose** despite AUC rising — first sign of a calibration regression. Investigated below.

### Top-leverage slices: NOTHING closed differentially

| Rank | Slice | Cycle 8 lift_$ | Cycle 13 lift_$ | Closed |
|---|---|---|---|---|
| 1 | non-2023 rows | 7963 | 7572 | 4.9% |
| 2 | `field_pit_share` q5 | 6759 | 6566 | 2.9% |
| 3 | (2024, HARD) | 5589 | 5380 | 3.7% |
| 4 | Stint 2 | 4637 | 4551 | 1.9% |
| 5 | `field_pit_share` q4 | 2847 | 2629 | 7.7% |
| 6 | (2022, HARD) | 2386 | 2312 | 3.1% |
| 7 | (2025, HARD) | 2375 | 2311 | 2.7% |
| 8 | LapsRem (30,50] | 2262 | 2196 | 2.9% |
| 9 | (5,10]×HARD | 2141 | 2054 | 4.1% |
| 10 | (0,5]×HARD | 2055 | 1965 | 4.4% |

All slices shrunk ~2-8%, roughly proportional to the global AUC improvement (0.94866 → 0.95134 = +0.28% relative). **No specific residual cell closed differentially.** Cycle 12's gain came from *general* model quality (external data + better HPs + more features), not from fixing any one residual axis the EDA pointed at.

This is consistent with the cycle 12 finding that lift was uniform: q3/q4/q5 of `field_pit_share` all gained ~+0.00020 in the 4-way ensemble, not concentrated at the targeted q5.

### Calibration: regressed sharply

| Decile | mean_pred | obs_rate | bias (cycle 13) | bias (cycle 8) | Δ |
|---|---|---|---|---|---|
| d1  | 0.00157 | 0.00043 | +0.00113 | +0.00029 | +0.00084 |
| d4  | 0.01513 | 0.00474 | +0.01039 | +0.00280 | +0.00759 |
| d6  | 0.06620 | 0.02637 | **+0.03983** | +0.00356 | +0.03627 |
| d7  | 0.21752 | 0.11295 | **+0.10457** | −0.00378 | +0.10835 |
| **d8** | **0.53645** | **0.33010** | **+0.20635** | −0.00918 | **+0.21553** |
| d9  | 0.79357 | 0.62978 | +0.16379 | −0.00113 | +0.16492 |
| d10 | 0.93219 | 0.87200 | +0.06019 | −0.00599 | +0.06618 |

Cycle 8's max bias was 0.00918 (essentially perfect calibration). Cycle 13's max bias is **0.20635 — a 22× regression**. The model systematically over-predicts in deciles 6-9 (the mid-to-high-probability bins).

**Cause**: `auto_class_weights="Balanced"` in CB-tuned inflates the loss gradient on positives, which pulls predicted probabilities up at inference. AUC is rank-based so it's unaffected, but the absolute probabilities are no longer interpretable as "P(pit next lap)".

**Implications**: AUC-only metric → calibration regression doesn't cost LB points directly. But if we ever ensemble with a well-calibrated model (LGB or CB#006) via simple averaging, the CB-tuned over-predictions tilt the average upward. The current 4-way `3way_focus` weights (0.05/0.20/0.75) implicitly compensate; future ensembles should keep this in mind.

### Driver-level: real codenames are still the bottom

Bottom 5 drivers by AUC (n ≥ 200):

| Driver | n | AUC (cycle 13) | AUC (cycle 8) | Δ |
|---|---|---|---|---|
| VET | 359 | 0.85039 | 0.83090 | +0.01949 |
| MSC | 355 | 0.85749 | 0.85975 | −0.00226 |
| COL | 657 | 0.89264 | 0.88368 | +0.00896 |
| BEA | 555 | 0.89426 | 0.89406 | +0.00020 |
| ALO | 1386 | 0.89790 | 0.88937 | +0.00853 |

VET, COL, ALO improved markedly (~+0.01); MSC and BEA didn't move. **Driver-level discrimination is the residual axis least helped by cycle 12** — the external data + tuned HPs improved most of the field uniformly, but individual real F1 codenames with idiosyncratic strategies remain difficult.

If we ever want a structurally different model family, this is the case for it: the GBDT family extracts what it can from target-encoded Driver, but the per-driver behavior signature appears non-tree-friendly. Neural embeddings could do better here.

### `field_pit_share` q5 — still the largest single-slice problem

```
q5 AUC: 0.87561 (cycle 13) vs 0.87069 (cycle 8) — +0.00492 closure
n = 86,698, pos_rate 0.382
```

Cycle 12 closed about 1% of q5's AUC gap. The cycle 9 attempt with dedicated pit-cluster features didn't help; cycle 12's general improvement helped a bit. Path forward likely needs either a driver-level (NN-embedding) approach or genuinely new pit-cluster signals from the external data (which we didn't engineer separately for).

## Verdict

**Kept** (EDA cycle).

The takeaway is structural, not tactical:

1. **The remaining −0.00422 LB gap is broadly distributed.** No single slice has disproportionate leverage anymore. Cycle 14+ should use *general* model improvements (iter cap, multi-seed, NN component), not slice-targeted FE.
2. **Calibration is broken** but AUC-only metric protects us. Note for ensembling and any future stacking attempts.
3. **Driver-level cohorts (real F1 codenames) are the structurally hardest residual.** This is the case for adding a tabular NN (RealMLP-style) as a model-family-diverse ensemble component.

## Follow-ups

Cycle 14 candidate priorities, based on this EDA:

1. **Lift CB-tuned iter cap (5000 → 8000-11000) (~70 min)** — cheapest. iter=5000 hit cap in every cycle 12 fold. Expected +0.0005-0.0010 OOF. Slice-agnostic but tractable.
2. **Multi-seed average on CB-tuned (~140 min)** — variance reduction at the operating point we now know works. +0.0005-0.0015. Could combine with #1.
3. **Digit/signature synthetic-exploit features (~50 min retrain)** — slice-agnostic generator-artifact features. Reference notebooks use these; cheap to add to the inline FE in `src/research/train_cb_tuned.py`. +0.0005-0.0015.
4. **Tabular NN (RealMLP-style) as 5th ensemble component (~3-4 hr CPU)** — only branch that should help driver-level discrimination specifically. Highest expected ceiling lift but expensive on CPU. +0.0010-0.0030 in ensemble.

Order of operations going forward: do #1+#3 together as a fast cycle (CB-tuned re-train with both higher iter cap and digit features; ~80 min), then evaluate. If gain is good, ship. If still gapping, then commit to #4 (NN).
