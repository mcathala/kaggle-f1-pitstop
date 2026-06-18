# Experiment 065 — full 6-seed pseudo-labeled RealMLP (completes the pseudo thread)

**Cycle.** 17
**Status.** Inconclusive (mildly positive) — full 6-seed pseudo-RealMLP OOF 0.95393 (+0.00010 over base); best blend OOF **0.95433** (+0.00013 over anchor) but sub-hurdle; LB transfer expected marginal. Completes & closes the pseudo-labeling thread.
**Date.** 2026-05-28

## Hypothesis

exp 064 showed single-seed pseudo lifts RealMLP +0.00011. Run the full 6-seed average (seeds 42/7/99/137/313/777, CV folds fixed) to match the production base (0.95383) and get the actual best-achievable pseudo-RM for the blend.

## Result

Kaggle P100, ~1.7 h. Pseudo-labels (103,009 confident test rows, same as exp 064). Per-seed OOF 0.95366–0.95376; **6-seed avg OOF 0.95393** vs 6-seed RealMLP base 0.95383 → **+0.00010**. ρ vs base RealMLP 0.9987 (same model, just pseudo-augmented).

### Blend probe (both pseudo bases)

| config | OOF | Δ vs anchor (0.95420) |
| ------ | --- | --------------------- |
| both-pseudo @ cycle-11 weights (0.675 psRM + 0.075 CB + 0.250 psXGB) | **0.95432** | +0.00012 |
| best free grid | 0.95433 | +0.00013 |

**OOF 0.95433 is the best blend in the project** — but still 0.00008 below the 0.95441 hurdle.

## Verdict

**Inconclusive (mildly positive); closes the pseudo-labeling thread.** Full 6-seed pseudo-RealMLP + pseudo-XGB gives the best OOF blend (0.95433, +0.00013), confirming pseudo-labeling lifts both bases. But it stays sub-hurdle, and the prior pseudo-blend showed the OOF lift does **not** transfer to LB (exp 063: 0.95428 OOF → 0.95373 LB, mostly leakage). So this blend is expected to land ~0.9537–0.9538 LB — marginal over the current 0.95373. The pseudo thread is fully characterized: real but negligible on the held-out test distribution.

## Held submission candidate

`submission_blend_pseudo6.csv` (0.675 pseudoRM6 + 0.075 CB + 0.250 pseudoXGB) built and held as a candidate (not submitted). It is the best-OOF blend; would test whether the extra pseudo-RM moves LB beyond the flat 0.95373. (1 slot used today; 4 remain.)

## Repro stamp

- Kaggle kernel `mcathala/cycle-17-realmlp-pseudo6-exp-065`; notebook [gpu-kernels/cycle17_realmlp_pseudo6_gpu.py](../gpu-kernels/cycle17_realmlp_pseudo6_gpu.py)
- 6 seeds, n_ens=24, CV folds fixed at seed 42; outputs `data/oof_realmlp_pseudo6.parquet`, `data/submission_realmlp_pseudo6.csv`, `data/submission_blend_pseudo6.csv` (held)
- compute ~1.7 h P100

## Learnings & follow-ups

1. **Pseudo-labeling, fully exploited on both bases, caps the blend at OOF ~0.95433 (sub-hurdle).** Real but small; LB-marginal due to same-distribution train/test (test carries little new info) + mild leakage in the OOF.
2. **Every lever is now exhausted** (cycles 16-17): own-model, FE, external, diverse-NN, blend-combiner, AutoML, and pseudo-labeling. Best blend OOF 0.95433 / submitted LB 0.95373. The 0.9544 top-10% goal is out of reach for our approach against a dense synthetic-data LB plateau.
3. **Remaining options:** submit the held pseudo6-blend (test the marginal gain), accept 0.95373 and lock finals, or pursue a fundamentally new idea/data source.
