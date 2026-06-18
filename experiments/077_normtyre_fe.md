# Experiment 077 — compound-stint-normalized tyre features (Normalized_TyreLife reconstruction)

**Cycle.** 18
**Status.** Inconclusive (flat/negative; FE-axis confirmed closed).
**Date.** 2026-05-29

## Hypothesis

The host deliberately removed a `Normalized_TyreLife` column (present in the external dataset). Our analysis of stint structure suggests a pit-overdue signal — `TyreLife / compound-typical-stint-length` — that crosses 1.0 when a tyre is past its compound's normal life. If our bases lack this specific normalization, adding it lifts XGB-highbins standalone by ≥ +0.0003 OOF.

## Rationale

1. Reverse-engineering the external `Normalized_TyreLife` shows it's normalized by *eventual stint length* (forward-looking → leaky, which is why it was removed). The **non-leaky** version normalizes by a per-compound constant.
2. Our stint analysis (max TyreLife per (Driver,Race,Year,Stint), median across stints) gives per-compound stint lengths: HARD 23, MEDIUM 17, INTERMEDIATE/WET 16/14, SOFT 14.
3. Our existing tyre features normalize by *race length* (`TyreAgeVsRace = TyreLife/EstimatedTotalLaps`), never by *compound stint*. The compound-stint version is the physically meaningful "overdue" signal.

## Features added (to XGB-highbins FE)

```
NormTyreLife          = TyreLife / compound_median_stint        # pit-overdue ratio
LapsOverdue           = TyreLife - compound_median_stint        # laps past typical
TyreOverdue           = (TyreLife > compound_median_stint)      # binary
NormTyre_x_RaceProgress = NormTyreLife * RaceProgress
```

## Result

| | OOF AUC | per-fold mean ± std |
| --- | --- | --- |
| canonical XGB-highbins | 0.95263 | — |
| **+ norm-tyre (this exp)** | **0.95259** | 0.95260 ± 0.00058 |
| **Δ** | **−0.00004** | flat-to-negative |

Per-fold deltas vs canonical: f1 −0.00005, f2 −0.00006, f3 +0.00001, f4 −0.00007, f5 ~flat. Consistently within noise, slightly negative.

Univariate pre-check confirmed it: `NormTyreLife` AUC 0.664 vs raw `TyreLife` 0.699 vs existing `TyreAgeVsRace` 0.660 — the reconstruction is *weaker univariately* than raw TyreLife and near-identical to a feature we already have.

## Verdict

**Inconclusive (flat); FE-axis closed.** The compound-stint normalization is redundant with our existing race-normalized tyre features + the categorical Compound × numeric TyreLife interactions that XGB/CB/RealMLP already capture. No incremental signal.

This independently confirms the external research conclusion: there is **no missing single feature** that closes our gap to the 0.9545 plateau. Our FE recipe already captures the load-bearing tyre/compound signal.

## Kill-criteria check

- [x] Standalone OOF (0.95259) < canonical (0.95263) → magnitude gate fails; no propagation to RealMLP/CB.

## Repro stamp

- Trainer: [src/research/train_xgb_normtyre.py](../src/research/train_xgb_normtyre.py) (fork of `train_xgb_richcat.py`; +4 compound-stint features; 136 features total).
- Output: `data/oof_xgb_normtyre.parquet` (OOF 0.95259, ρ 0.980 vs RM, 0.984 vs CB — same as canonical XGB).
- 5-fold StratifiedKFold(shuffle=True, random_state=42); max_bin=5000.

## Learnings

1. **The Normalized_TyreLife "load-bearing feature" is already implicit in our pipeline.** Compound-stint normalization adds nothing over race-normalization + the categorical Compound interactions GBDTs/RealMLP form automatically.
2. **Confirms the FE axis is closed** — combined with cycles 16-17's FE closures, our feature recipe is at its information ceiling for the tree/NN families.
3. The genuine gap to 0.9545 is, per the leaderboard structure, blend-diversity of many bases (the field's shared-CSV blender) — not a feature we're missing.

## Follow-ups

- Closed: compound-stint FE.
- Next: exp 078 external-reweighting (drift mechanism, not FE) + exp on genuine base-diversity.
