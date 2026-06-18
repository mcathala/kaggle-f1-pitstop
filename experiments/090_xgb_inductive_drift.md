# Experiment 090 — fully-inductive diffFE-XGB (drift test) → no-op; drift isn't in the base-cat frequency

**Cycle.** 18
**Status.** **Inconclusive (no-op, −0.00001).** Excluding test from the frequency union changes nothing → diffFE-XGB is already effectively inductive. Re-points drift suspicion at the pseudo-labeled bases.
**Date.** 2026-05-29

## Hypothesis

A1 proved blend weights are honest (overfit +0.00001), so the −0.0006 OOF→LB drift is structural transductive-FE optimism. RealMLP's FE is already train-only (exp 079's small RM drift), so the leak should live in the GBDT bases' base-cat **frequency**, which diffFE-XGB still computes on the train+**test**+ext union. Counting frequency on [train, ext] only (test excluded) should lower OOF but reduce drift → higher LB.

## Method

Forked `train_xgb_diffFE.py` → `train_xgb_inductive.py`; `add_frequency_features` gained a `count_frames` arg, called with `[train, ext]` (test excluded from the count). Everything else identical.

## Result

| | OOF AUC | ρ vs RM |
| --- | --- | --- |
| diffFE-XGB (transductive freq) | 0.95291 | 0.980 |
| **inductive-XGB (test excluded)** | **0.95290** | **0.980** |
| **Δ** | **−0.00001** | 0.000 |

**A no-op.** OOF, ρ, and per-fold profile are all identical.

## Verdict

**Inconclusive — the base-cat frequency is not the drift source.** The reason is now obvious in hindsight: diffFE already *stripped* the heavy transductive block (cross-categoricals, union-frequency on cross/bins, union group-stats) back in exp 080. The only transductive feature left is base-cat frequency on 3 **low-cardinality** cats (Driver/Race/Compound); adding ~88k test rows to a ~450k train+ext count shifts those frequencies negligibly. **diffFE-XGB is already effectively inductive.**

**This closes the "transductive base-cat frequency → drift" hypothesis** and re-points the remaining drift suspicion at the one genuinely transductive thing left in the blend: **the round-1 pseudo-labeled bases** (`diffpsxgb`, 0.224 weight; `psxgb` already de-leaked to round-2 in A1). Pseudo-labeling trains on *test-derived* labels — a real transductive channel, and exp 067 measured ~+0.0002 round-1 OOF inflation for XGB. The next free probe tests a no-pseudo honest blend variant.

## Acceptance gates

| Gate | Got | Pass? |
| --- | --- | --- |
| OOF change (drift-reduction proxy) | −0.00001 | ❌ (no-op) |

## Repro stamp

- Trainer: [src/research/train_xgb_inductive.py](../src/research/train_xgb_inductive.py).
- Output: `data/oof_xgb_inductive.parquet` (0.95290) — a near-duplicate of `oof_xgb_diffFE.parquet`.

## Learnings

1. **diffFE already removed the transductive FE that mattered.** The exp 080 strip wasn't just a strength win — it incidentally made the XGB base ~inductive. There is no further drift to claw back from base-cat frequency.
2. **The remaining transductive channel is pseudo-labeling**, not hand-FE. That's the next (free) thing to probe for drift — a no-pseudo honest blend.
3. Adds to the closure stack: diversity (088), FE-for-RM (089), weights (A1), and now inductive-FE (090) are all closed/no-op. The own-tooling OOF ceiling (0.95462) is comprehensively confirmed.

## Follow-up — no-pseudo honest blend probe (free)

Since pseudo-labeling is the last transductive channel, a free coord-descent probe over the pseudo axis (no retraining):

| Candidate | pool | OOF | Δ vs full |
| --- | --- | --- | --- |
| `submission_blend_best.csv` (C1) | full, incl. round-1 pseudo GBDTs | **0.95462** | — |
| `submission_blend_nopseudoGBDT.csv` (C2) | drops round-1 pseudo GBDTs, keeps pseudo-RM | 0.95461 | **−0.00001** |
| `submission_blend_pure_nopseudo.csv` (C3) | no pseudo at all | 0.95449 | −0.00013 |

**Dropping the round-1 pseudo GBDTs is essentially free (−0.00001)** — `diffxgb` replaces `diffpsxgb` at w≈0.25 with no OOF loss. So C2 is the same strength but removes the suspected drift channel. The pseudo-RM (psRM6r2, round-2 de-leaked) is worth keeping (C3 drops it → −0.00013). All three are built by [src/research/build_drift_candidates.py](../src/research/build_drift_candidates.py) and **staged for the next daily submission slot**: submitting C1/C2/C3 LB-tests whether round-1 pseudo was inflating OOF without transferring — if C2 ≥ C1 on LB, the honest blend wins and the drift is (partly) pseudo.
