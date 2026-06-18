# Experiment 064 — pseudo-labeled RealMLP (single-seed feasibility)

**Cycle.** 17
**Status.** Inconclusive (mildly positive) — pseudo lifts single-seed RealMLP +0.00011 (less than XGB's +0.00032; RealMLP already robust-clips). Combined pseudo-thread blend caps ~0.95429–0.95435, below the 0.95441 hurdle. Characterizes & closes the pseudo-labeling thread.
**Date.** 2026-05-27

## Hypothesis

exp 063 lifted XGB +0.00032 but only +0.00008 in the blend (XGB 0.25 weight). Apply the same self-training to the **0.675-weight RealMLP** — ~3× the blend leverage. Single-seed feasibility (vs single-seed RealMLP 0.95355) before committing to the 6-seed cost.

## Method

[gpu-kernels/cycle17_realmlp_pseudo_gpu.py](../gpu-kernels/cycle17_realmlp_pseudo_gpu.py), Kaggle P100, ~16 min. Pass-1 quick XGB (raw features) → 103,009 confident test pseudo-labels (3,621 hi / 99,388 lo). Pass-2 verbatim cycle-5/11 RealMLP recipe (PyTabKit RealMLP_TD, n_ens=24, single seed 42), 5-fold, trained on comp-fold + external + pseudo-test.

## Result

| Fold | AUC |
| ---- | --- |
| 1 | 0.95429 |
| 2 | 0.95435 |
| 3 | 0.95338 |
| 4 | 0.95253 |
| 5 | 0.95380 |
| **OOF** | **0.95366** |

**OOF 0.95366 vs single-seed RealMLP 0.95355 → +0.00011.** Pseudo-labeling lifts RealMLP, but ~3× less than it lifted XGB — consistent with RealMLP's `robust_scale + smooth_clip` already smoothing the input distribution that pseudo-labels reshape.

### Combined pseudo-thread blend probe

Best 5-way over {6-seed RM, CB, XGB, pseudo-XGB, single-pseudo-RM}: **0.95429** (+0.00010 over anchor), at w≈(RM 0.5, CB 0.05, pseudoXGB 0.30, pseudoRM 0.15). The single-seed pseudo-RM adds essentially nothing — it's weaker than the existing 6-seed RM and ρ≈0.998 with it.

Rough estimate for a full **6-seed** pseudo-RM (carrying the +0.00011 onto the 0.95383 base → ~0.95394): combined blend ≈ **0.95435** — still below the 0.95441 hurdle.

## Verdict

**Inconclusive (mildly positive); closes the pseudo-labeling thread.** Self-training lifts both bases (XGB +0.00032, RealMLP +0.00011) and the blend by ~+0.0001–0.00015, but the combined ceiling (~0.95435) stays below the hurdle and well below the LB-0.9544 goal. Did **not** escalate to 6-seed pseudo-RM (~3 h compute for an estimated +0.00007 that still doesn't clear the bar — poor budget stewardship; left as an option).

## Kill-criteria check

- [x] Combined pseudo blend (0.95429, est. ≤0.95435) < hurdle 0.95441 — **FIRED**.
- [ ] Standalone lift confirmed on RealMLP (+0.00011) — the positive note.

## Repro stamp

- Kaggle kernel `mcathala/cycle-17-realmlp-pseudo-exp-064`; notebook [gpu-kernels/cycle17_realmlp_pseudo_gpu.py](../gpu-kernels/cycle17_realmlp_pseudo_gpu.py)
- single seed 42, n_ens=24, verbatim RealMLP recipe; outputs `data/oof_realmlp_pseudo.parquet`, `data/submission_realmlp_pseudo.csv`
- compute ~16 min P100

## Learnings & follow-ups

1. **Pseudo-labeling helps proportionally less the more robust the base.** XGB (+0.00032) > RealMLP (+0.00011); RealMLP's built-in robust scaling/clipping already absorbs much of what pseudo-labels add.
2. **The pseudo-labeling thread is mildly positive but sub-hurdle.** Combined ceiling ~0.95435 OOF (≈0.9539 LB at our drift) — a real but small gain that does not reach top-10% (0.9544).
3. **Options:** (a) submit the held pseudo-blend (0.95428–0.95429) to bank a small LB gain over 0.95372; (b) run full 6-seed pseudo-RM for ~+0.00007 more (still sub-hurdle); (c) accept the mapped ceiling.
