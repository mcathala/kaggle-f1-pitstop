# Experiment 056 — no-external ablation (XGB-highbins + embMLP-v2)

**Cycle.** 16
**Status.** Inconclusive — **pollution hypothesis falsified.** Removing the external data lowers OOF in *both* model families (XGB −0.00073, NN −0.00088). Despite being distribution-shifted (adversarial AUC 0.78), the external augmentation is net-beneficial; the cap is not the data.
**Date.** 2026-05-27

## Hypothesis

We force-feed 101k external rows (`<external-f1-strategy-dataset>`, v4) into every training fold since cycle 3. We measured that this set is **distribution-shifted** from the competition data: an adversarial classifier (competition=0 vs external=1) reaches **AUC 0.78** (vs ~0.50 for train-vs-test), the external positive rate is **0.255 vs 0.199**, and it skews toward late-race high-pit-pressure laps. Hypothesis: this shift drags our models, and the 0.95421 OOF ceiling is partly a data-pollution artefact. **Ablation:** remove the external data entirely and re-run; if OOF *rises*, external is a net drag and should be dropped/down-weighted across the stack.

## Rationale

- We are stuck at OOF 0.95421 / LB 0.95372 after closing every own-model and FE axis. The external data is the one input we never questioned.
- A clean A/B (with vs without external) is the only way to attribute the cap to the data rather than the models.

## Expected magnitude

- If pollution is real: no-external OOF > with-external 0.95263 (XGB) / 0.95102 (NN) → drop or reweight external everywhere.
- If external helps: no-external OOF < with-external → data is beneficial, cap is elsewhere; pivot to reweighting (exp 057).

## Kill criteria

- [ ] No-external OOF ≤ with-external for both families → pollution hypothesis falsified.

## Scope / reversibility

- XGB: new Kaggle notebook [gpu-kernels/cycle16_xgb_noext_gpu.py](../gpu-kernels/cycle16_xgb_noext_gpu.py) — verbatim cycle-11 XGB-highbins recipe with external **not loaded** (FE union and training both competition-only). P100, `device=cuda`.
- NN: `NO_EXT=1` env-var branch added to [src/research/train_embmlp_v2.py](../src/research/train_embmlp_v2.py) (exp-054 trainer) — same multi-seed numeric-embedding MLP, external excluded from each training fold. Local M1 MPS.
- Diagnostic only; does not touch CV/seed/target/frozen files. Reversible.

## Result

### XGB-highbins (Kaggle P100)

| Fold | no-external AUC |
| ---- | --------------- |
| 1 | 0.95247 |
| 2 | 0.95251 |
| 3 | 0.95147 |
| 4 | 0.95084 |
| 5 | 0.95223 |
| **OOF** | **0.95190** |

per-fold mean 0.95190, std 0.00065, iters [3042, 3353, 3601, 3145, 3669].
**OOF 0.95190 vs with-external 0.95263 → Δ −0.00073.** Removing external *hurts* XGB.

### embMLP-v2 (numeric-embedding NN, local M1 MPS)

| Fold | no-external AUC |
| ---- | --------------- |
| 1 | 0.95075 |
| 2 | 0.95062 |
| 3 | 0.94991 |
| 4 | 0.94913 |
| 5 | 0.95055 |
| **OOF** | **0.95014** |

**OOF 0.95014 vs with-external (exp 054) 0.95102 → Δ −0.00088.** Removing external *hurts* the NN too.

## Verdict

**Inconclusive — pollution hypothesis fully falsified.** Both model families lose AUC when the external data is removed (XGB −0.00073, NN −0.00088). Despite the genuine distribution shift (adversarial 0.78, pos rate 0.255 vs 0.199), the external set's volume/signal **outweighs** its shift drag at our current model strength. The OOF 0.95421 / LB 0.95372 ceiling is **not** a data-pollution artefact — the cap is in the models/blend, not the data.

## Kill-criteria check

- [x] No-external OOF < with-external for **both** families — **FIRED** (XGB −0.00073, NN −0.00088). Pollution hypothesis falsified.

## Repro stamp

- XGB: Kaggle kernel `mcathala/cycle-16-xgb-noext-exp-056`, notebook [gpu-kernels/cycle16_xgb_noext_gpu.py](../gpu-kernels/cycle16_xgb_noext_gpu.py); device=cuda, P100; outputs `data/oof_xgb_noext.parquet`, `data/submission_xgb_noext.csv`; runtime ~58 min (5 folds).
- NN: `NO_EXT=1 .venv/bin/python src/research/train_embmlp_v2.py`; outputs `data/oof_embmlp_v2_noext.parquet`; runtime ~28 min on M1 MPS.

## Learnings

1. **Distribution shift ≠ harmful.** A 0.78 adversarial AUC is a large covariate shift, yet the external data still helps both a GBDT and an NN. Volume + genuine pit-timing signal beats the shift drag at this model strength. Do not drop the external augmentation.
2. **The cap is the models, not the data.** This closes the "our data is polluting us" line of inquiry. Combined with the own-model close (053/054) and the FE-on-XGB close (051/052/055), the remaining levers are: (a) reweighting external to recover *more* of its signal (exp 057), or (b) a genuinely different algorithm.
3. **Sets up exp 057 with a caveat.** Since external helps at *full* weight, down-weighting its off-distribution rows (adversarial importance weighting) risks discarding useful signal in exchange for less drag — a coin-flip, not an obvious win.

## Follow-ups

- Closed: "external data is a net drag / pollution cap" hypothesis.
- **Next (exp 057):** adversarial-importance-weighted external — keep all rows but weight each external row by P(comp|x)/P(ext|x), trying to retain volume while softening the shift. The only remaining live shot in this batch.
