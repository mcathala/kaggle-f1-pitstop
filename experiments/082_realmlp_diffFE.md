# Experiment 082 — differentiated (stripped) FE on RealMLP

**Cycle.** 18
**Status.** Inconclusive (neutral; +0.00002 OOF). diffFE helps GBDTs, not RealMLP.
**Date.** 2026-05-29

## Hypothesis

The diffFE win (exp 080/081: stripping over-engineered FE made XGB +0.00028 and CB +0.00068) might also strengthen the dominant RealMLP base (w=0.57). RM's FE has an analogous heavy block — it factorizes/floors ~22 numeric columns into extra categoricals that get one-hot/embedded.

## Method

Forked `train_realmlp_pseudo_r3.py` → `train_realmlp_diffFE.py`. Dropped the categorize-numerics block (kept only `Year_cat_`/`PitStop_cat_`, required by the count block). 44 → 30 features. Single-seed (42), same recipe otherwise.

## Result

| | OOF AUC | per-fold |
| --- | --- | --- |
| RM single-seed no-strip (exp 072) | 0.95369 | 0.95433/0.95434/0.95340/0.95268/0.95380 |
| RM single-seed diffFE (this exp) | 0.95371 | 0.95452/0.95414/0.95341/0.95262/0.95388 |
| Δ | **+0.00002** | neutral (f1 +0.00019, f2 −0.00020, rest ~flat) |

## Verdict

**Inconclusive (neutral).** Stripping the categorize-numerics block is OOF-neutral for RealMLP — unlike GBDTs. RealMLP's PLR-embeddings + `smooth_clip`/`robust_scale` transforms already model the raw numerics well, so the hand-categorized numerics were benign (neither the harmful redundancy they were for GBDTs, nor additive value). Bonus: 30-feature RM trains ~2× faster (285s/fold vs ~600s) at equal accuracy — a free efficiency win, not an accuracy win.

**Interpretation of the diffFE lever's scope:** the gain came specifically from removing the *hand-engineered cross-categoricals + union-frequency + union-group-stats* that the GBDTs carried (and which were redundant with what tree splits / native ordered-TE already capture). RealMLP never had those features, so there was nothing harmful to strip.

## Kill-criteria check

- [x] OOF Δ (+0.00002) < min_delta → no propagation; keep the existing 6-seed psRM6r2 in the blend.

## Repro stamp

- Trainer: [src/train_realmlp_diffFE.py](../src/train_realmlp_diffFE.py) (30 features; categorize-numerics block dropped).
- Output: `data/oof_realmlp_diffFE_s42.parquet` (0.95371), `data/submission_realmlp_diffFE_s42.csv`.
- 5-fold StratifiedKFold(42); single seed; MPS, 1432s total.

## Learnings

1. **diffFE is a GBDT-specific lever** — the harmful over-engineering (cross-cats / union-stats) lived only in the tree pipelines. RealMLP's lean PLR-embedding FE was already near-optimal.
2. **The 6-seed psRM6r2 (0.95396) remains the dominant base**; no diffFE swap warranted.
3. The night's banked win (LB 0.95388) is entirely from diffFE-XGB. The compounding question moves to: does pseudo-labeling the cleaner diffFE-XGB give a stronger XGB that lifts the blend (exp 083)?

## Follow-ups

- exp 083: diffFE + pseudo XGB (running).
- diffFE-RM is a free speed win; could adopt for faster future RM iterations, but no accuracy reason to switch the production base.
