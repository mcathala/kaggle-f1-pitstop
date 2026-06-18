# #008 — Error-bucket EDA on the 3-way ensemble

**Status.** Kept (EDA — no model change)
**Date.** 2026-05-21
**Focus.** EDA

## Why this cycle

Cycle 7 was inconclusive (+0.00014 OOF, below noise floor). The next step was to burn ~110 min on a seed-robustness sweep to confirm cycle 7's small lift, but the Public LB gap to top-300 is **+0.00554** (current 0.94833 vs top ~0.95387). Drilling deeper into the peer-rank branch — even if seed-confirmed — wouldn't close that gap. We pivoted to map the residual landscape first.

## Hypothesis

Slicing the 3-way ensemble OOF along multiple feature axes will surface 1–2 large-leverage slices (low AUC × large n) that name the next cycle's modeling branch. If residuals are uniformly distributed, the conclusion is "no targeted FE will close the gap → cycle 009 = different model family".

## Setup

- Input: `data/oof_ensemble3_seed42.parquet` (current best, OOF AUC 0.94866).
- Metric per slice: AUC, log-loss, **lift_$ = n × (global_AUC − slice_AUC)** — an estimate of how much aggregate AUC-leverage a fix in this slice would deliver.
- Code: `src/research/cycle008_error_eda.py`.

## Findings

### Global stats
```
rows:           439,140
OOF AUC:        0.94866
log-loss:       0.22652
positive rate:  0.1990
```

### Calibration

Calibration by predicted-probability decile is essentially perfect:

| Decile | n | mean_pred | obs_rate | bias |
|---|---|---|---|---|
| d1 | 43,914 | 0.00076 | 0.00048 | +0.00029 |
| d5 | 43,914 | 0.01678 | 0.01084 | +0.00594 |
| d10 | 43,914 | 0.85978 | 0.86576 | −0.00599 |

Max |bias| = 0.00918 (d8). **The gap is not a calibration problem — it's a discrimination problem.**

### Top leverage opportunities (lift_$ ranked)

| Rank | Scan | Slice | n | pos_rate | AUC | lift_$ |
|---|---|---|---|---|---|---|
| 1 | is_2023 | 0 (non-2023) | 302,993 | 0.284 | 0.92237 | **+7963** |
| 2 | field_pit_share | q5 | 86,698 | 0.382 | 0.87069 | **+6759** |
| 3 | Year × Compound | (2024, HARD) | 53,463 | 0.539 | 0.84412 | **+5589** |
| 4 | Stint | (1.5, 2.5] | 129,536 | 0.391 | 0.91286 | **+4637** |
| 5 | Stint | (0.5, 1.5] | 216,288 | 0.060 | 0.93275 | +3440 |
| 6 | field_pit_share | q4 | 88,908 | 0.271 | 0.91663 | +2847 |
| 7 | Year × Compound | (2022, HARD) | 22,025 | 0.465 | 0.84034 | +2386 |
| 8 | Year × Compound | (2025, HARD) | 34,034 | 0.480 | 0.87886 | +2375 |
| 9 | LapsRemaining | (30, 50] | 131,960 | 0.256 | 0.93151 | +2262 |
| 10 | TyreLife × Compound | (5, 10] × HARD | 26,584 | 0.229 | 0.86810 | +2141 |

### HARD compound — the largest physics branch

All-HARD-TyreLife combined lift (across 2022/2024/2025):

| Slice | n | AUC | lift_$ |
|---|---|---|---|
| (5, 10] × HARD | 26,584 | 0.868 | 2141 |
| (0, 5] × HARD | 22,739 | 0.858 | 2055 |
| (10, 15] × HARD | 29,448 | 0.899 | 1454 |
| (15, 20] × HARD | 28,929 | 0.917 | 918 |
| (20, 30] × HARD | 40,837 | 0.932 | 686 |
| **HARD total** | **~148k** | **~0.89** | **~7.3k** |

Cycle #006 said HARD was "near Bayes ceiling" — but the data says 0.89, not 0.97. **That's not Bayes; that's an unsolved branch.** Two previous attempts (cycles #003, #006-target) used features designed *against the model's mistakes after-the-fact* — they were reactive. A proactive approach (e.g. domain-specific HARD physics) hasn't been tried.

### Pit-cluster saturation — a brand new branch

`field_pit_share` quintiles (computed at the (Race, Year, Lap) level):

| Quintile | mean pit_share | n | pos_rate | AUC | lift_$ |
|---|---|---|---|---|---|
| q1 | ~0% | 87,907 | 0.022 | 0.97589 | −2394 |
| q2 | ~3% | 88,109 | 0.106 | 0.97624 | −2431 |
| q3 | ~9% | 87,518 | 0.216 | 0.94621 | +214 |
| q4 | ~18% | 88,908 | 0.271 | 0.91663 | **+2847** |
| q5 | ~38% | 86,698 | 0.382 | 0.87069 | **+6759** |

When ≥18% of the field pits, model AUC drops sharply (0.92 → 0.87). q4 + q5 combined = **+9606 lift**, comparable in size to all-HARD. The current `field_pit_share`/`ahead_pitted_*`/`behind_pitted_*` features capture pit pressure, but their granularity saturates above q3. Possible cycle-009 hypothesis: finer per-event pit-cluster features (acceleration of pit rate, per-cluster identity, post-SC pit-window flags).

### Driver-level discrimination

| Subset | n | AUC | lift_$ |
|---|---|---|---|
| Real F1 codenames (e.g. ALO, STR, RIC) | 177,516 | 0.93951 | **+1623** |
| Synthetic personas (D###) | 261,624 | 0.95417 | −1442 |

Real drivers underperform synthetic ones by 0.015 AUC. Worst real drivers by AUC (n ≥ 200): VET 0.831, DEV 0.837, MSC 0.860, COL 0.884, DOO 0.884, ALO 0.889, STR 0.893. These are drivers with **idiosyncratic** pit-decision patterns that target-encoding can't fully capture. A NN with learned Driver embeddings — pre-registered as cycle-#006 followup #1 in the cycle-#004 doc — directly targets this.

### Stint-bucket and LapsRemaining

| Slice | n | AUC | lift_$ |
|---|---|---|---|
| Stint 2 | 129,536 | 0.913 | +4637 |
| Stint 1 (only 6% pos rate) | 216,288 | 0.933 | +3440 |
| Stint 3 | 69,238 | 0.926 | +1593 |
| LapsRemaining (30, 50] | 131,960 | 0.932 | +2262 |
| LapsRemaining (20, 30] | 51,262 | 0.927 | +1110 |

Stint 2 is the weakest stint (mid-race tyre transitions); pit-window LapsRemaining buckets 20–50 are weak too. Probably overlaps with HARD-compound timing, since HARD stints are typically the longer middle/final segments.

## Verdict

**Kept** (EDA cycle — no model change, but a Mapped Territory).

The +0.00554 LB gap is concentrated in **three branches that have NOT been targeted yet**:

| Branch | Estimated leverage | Status |
|---|---|---|
| Pit-cluster saturation (field_pit_share q4+q5) | ~9.6k lift | **Never explored** |
| HARD-compound discrimination (low/mid TyreLife) | ~7.3k lift | Tried (#003, #006-target), but with reactive features — proactive physics untried |
| Driver embeddings (real-codename underperformance) | ~1.6k lift (within driver scope; could be larger via interaction effects) | **Never explored** |

## Cycle 009 candidate branches (with expected ROI per CPU minute)

Ranked by best ROI-per-CPU-minute given the EDA:

1. **Pit-cluster saturation features (~30 min, ~+0.001–0.003)** — entirely new branch, biggest single untouched lift. Concrete: `field_pit_share_diff_1` (1-lap delta), `field_pit_share_lag_5` (5-lap memory), `peak_pit_window_distance` (laps to/from local max of share), `post_sc_pit_cluster` (interaction `sc_lap_minus2 × field_pit_share`).
2. **NN with categorical embeddings (~45–60 min, ~+0.001–0.005 high variance)** — direct attack on driver discrimination; also adds ensemble diversity beyond tree models. Pre-registered as cycle-#004 followup.
3. **Stacking meta-model (~10–15 min, ~+0.0005–0.0020)** — quickest win; train a lightweight learner on OOF predictions of base models + a few raw features. Risk: OOF overfit (cycle 6 already saw −0.00033 LB drift from this).
4. **Targeted HARD-compound physics features (~30 min, ~0–0.002 low confidence)** — tyre-wear curve estimation per (Driver, Race) from prior stints, etc. Higher uncertainty since cycle #003 + #006-target both failed.

## Follow-ups

- Cycle 009: pick branch (1), (2), or (3) above. (4) is last-resort.
- Independent of cycle 009: log this EDA as a permanent reference. Future cycles should re-run `src/research/cycle008_error_eda.py` after kept cycles to track which slices closed.
