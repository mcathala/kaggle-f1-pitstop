# Experiment 070 — sklearn HistGradientBoostingClassifier (untried GBDT family)

**Cycle.** 17
**Status.** Inconclusive (Reverted; lever closed) — OOF 0.94486 standalone; ρ 0.946 vs RM6 (most NN-diverse GBDT in project) but strength too weak; free 8-way grid awards w_hgbc = 0.
**Date.** 2026-05-28

## Hypothesis

A 4th GBDT family (sklearn HGBC) reaches blend-relevant diversity (ρ ≤ 0.93 vs all of {RM6, CB-exp14, XGB-highbins}) while clearing the 0.949 strength floor — i.e. lands in or near the *strong+diverse* quadrant that none of our GBDTs (all ρ 0.997-0.999 mutual) currently occupies.

## Rationale

- Cycle 17 mapped: every NN frontier maxes out at 0.943, the strong+diverse quadrant is empty from the NN side. GBDT side has the opposite: strong (0.951-0.953) but mutually saturated.
- HGBC is the one major GBDT family not yet tested. Sklearn's leaf-wise histogram backbone is *not* LightGBM (despite the surface similarity) — different bin layout, different leaf-merge logic, different regularization defaults. ρ vs LGB-highbins (which lives in our zoo at ρ 0.998 vs XGB) is an unknown.
- Local CPU is the only available compute (Kaggle GPU quota effectively exhausted). HGBC is small-overhead and proven fast on M1.

## Expected magnitude

- Strength: standalone OOF AUC 0.949-0.953 (similar to other tree bases on the same FE pipeline; floor is 0.949 = current 4-th-base minimum threshold).
- Diversity: ρ 0.95-0.99 vs other GBDTs (most-likely outcome — ρ-saturation across GBDTs has been the rule). ρ 0.93 vs RM6 — possible if HGBC picks up its own slice. *Stretch* outcome: ρ 0.92 @ AUC 0.951 — would clear the oracle-boost frontier.
- Floor below which we don't escalate: OOF < 0.949 OR all ρ > 0.99 → inconclusive, close lever.

## Overfitting risk

Low. Same CV protocol, same FE pipeline as XGB; only the learner changes. HGBC has internal early-stopping on a 10% validation split (separate from our fold val) → standard regularization.

## Kill criteria

- [ ] OOF < 0.949 (below GBDT base floor) **AND** ρ > 0.95 vs every other base → discard, lever closed.
- [ ] Best blend OOF (with HGBC included) does not exceed the current 0.95433 ceiling → discard.

## Scope

- `src/research/train_hgbc.py` (+~360 lines, new file; verbatim FE pipeline from `train_xgb.py`)
- `experiments/070_hgbc.md` (new)

## Reversibility check

No CV / seed / target changes. New artifact files only (`oof_hgbc.parquet`, `submission_hgbc.csv`). Reversible.

## Plan

1. Fork `train_xgb.py` to `train_hgbc.py`; swap learner; cast cats to pandas Categorical with HGBC's native categorical-features support (`from_dtype`).
2. **HGBC constraint:** max categorical cardinality ≤ `max_bins` (255 default). Bucket any cat over the cap (Driver: 872 levels, Driver_Race/Driver_Compound/Race_Compound_Stint) into top-253 + `__OTHER__`.
3. Smoke fold 1, gate on AUC > 0.94 & ρ < 0.99 vs RM6.
4. If smoke passes, full 5-fold OOF.
5. Free 4-way blend probe (`{RM6, CB-exp14, XGB-highbins, HGBC}`); free 5-way grid including the pseudo-RM6 strong base.

## Smoke result (fold 1)

```
fold 1/5   AUC = 0.94531   iters = 816   (50s)
rank-corr vs RealMLP-6seed     : 0.94631
rank-corr vs CB-tuned-exp14    : 0.95589
rank-corr vs XGB-highbins      : 0.96200
```

ρ 0.946 vs RM6 **is more diverse than any other GBDT** in the project — i.e. HGBC sits on a strictly better frontier than LGB-highbins or XGB-highbins in the diversity dimension. But strength is weak (0.94531 < 0.949 floor on a single fold). Decision: run the full 5-fold cheaply (~5 min) to get honest OOF ρ + blend probe verdict.

## Result (5-fold OOF)

```
per-fold AUC: mean=0.94486   std=0.00051   iters=[816, 822, 903, 780, 796]
OOF AUC:      0.94486   (vs CB-tuned-exp14 0.95114, Δ −0.00628)
                       (vs XGB-highbins   0.95263, Δ −0.00777)
```

### Per-year AUC

| Year | OOF AUC | n        | pos_rate |
| ---- | ------- | -------- | -------- |
| 2022 | 0.90580 | 82,989   | 0.2665   |
| 2023 | 0.92531 | 136,147  | 0.0096   |
| 2024 | 0.92150 | 127,110  | 0.2953   |
| 2025 | 0.91970 | 92,894   | 0.2844   |

### Rank-correlation diagnostics (full OOF)

| Pair                  | ρ |
| --------------------- | --- |
| HGBC vs RealMLP-6seed | **0.94624** ← most diverse GBDT-to-NN pair in the project |
| HGBC vs CB-exp14      | 0.95590 |
| HGBC vs XGB-highbins  | 0.96238 |

### Blend probe

Adding HGBC to the anchor or pseudo 3-way at any tested weight *reduces* the OOF — HGBC's −0.006 standalone gap to the GBDT zoo is bigger than its diversity bonus.

| Blend | OOF |
| ----- | --- |
| 3way anchor (RM6/CB/XGB)                         | 0.95421 |
| 4way RM6+CB+XGB+HGBC (w_hgbc=0.05)               | 0.95414  (−0.00007) |
| 4way RM6+CB+XGB+HGBC (w_hgbc=0.10)               | 0.95402  (−0.00019) |
| 3way pseudo (psRM6/CB/psXGB)                     | 0.95432 |
| 4way psRM6+CB+psXGB+HGBC (w_hgbc=0.05)           | 0.95425  (−0.00007) |
| Free 8-base coord-descent (with HGBC available)  | 0.95436  → assigns HGBC weight 0.000 |

## Verdict

**Inconclusive (Reverted; lever closed)** — HGBC is genuinely the most NN-diverse GBDT we have access to (ρ 0.946 vs RM6 vs 0.97-0.98 for every other GBDT), but its strength (0.94486) is too weak to clear the *strong + diverse* frontier we'd need (oracle-boost analysis: ρ ≤ 0.92 @ AUC ≥ 0.951). It lives in the same weak+diverse trap as embMLP/lap-attention from the NN side — *just on the GBDT side*. Free 8-way grid awards w_hgbc=0.

Decision: leave the trainer in `src/` (useful for future cycles if HGBC's max-bins limit is lifted), do not touch the blend, do not submit.

## Kill-criteria check

- [x] OOF 0.94486 < 0.949 floor → **kill criterion 1 partially fires** (strength below floor).
- [x] Free 4-way grid with HGBC available → w_hgbc=0 in the refined optimum → **kill criterion 2 fires**.

## Repro stamp

- Trainer: [src/research/train_hgbc.py](../src/research/train_hgbc.py) (one new file; same FE pipeline as `train_xgb.py`)
- Blend probe: [src/research/blend_hgbc_probe.py](../src/research/blend_hgbc_probe.py)
- 5-fold StratifiedKFold(shuffle=True, random_state=42) on `Year × PitNextLap`; HGBC max_iter=2000, lr=0.05, max_depth=8, l2=5.0, max_bins=255, early_stopping with 30 stalls.
- Bucketed Driver/Driver_Race/Driver_Compound/Race_Compound_Stint to top-253 + `__OTHER__` (HGBC categorical-cardinality cap = max_bins = 255).
- Outputs on disk: `data/oof_hgbc.parquet`, `data/submission_hgbc.csv`, `data/blend_hgbc_sweep.parquet`.

## Learnings

1. **HGBC's binning cap is the binding constraint here, not the algorithm.** Driver alone has 887 unique values; bucketing to 254 throws away the discriminative tail. XGB-highbins works because `max_bin=5000` resolves these. With the histogram bin cap closer to GBDT-standard 255, every modern tree library collapses to the ~0.945 floor (mirrors exp 034's pre-highbins XGB at 0.94615).
2. **Diversity profile is asymmetric.** HGBC has ρ 0.946 vs RM6 (NN axis) but ρ 0.96 vs the other GBDTs. That's *NN-axis* diversity from a GBDT model — a structural anomaly not previously seen in this project. It just doesn't pay because the strength is too low.
3. **Confirms the weak+diverse trap is symmetric.** Cycle 17 already documented this for NN-side mechanisms (embMLP, lap-attention). HGBC closes the GBDT-side branch: at low standalone AUC, no amount of diversity wins.

## Follow-ups

- Closed: HGBC as a base. Not worth re-tuning — the bin cap is intrinsic to the API.
- Worth one note: HGBC's NN-axis diversity is the kind of profile we'd want from a future model. If we ever find a *strong* HGBC-like model (or somehow get HGBC's bin cap lifted), it could clear the frontier. No such option exists today.
