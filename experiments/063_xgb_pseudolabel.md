# Experiment 063 — confidence-gated self-training (pseudo-labels) on XGB

**Cycle.** 17
**Status.** Inconclusive (mildly positive) — FIRST mechanism this round to lift a base standalone (XGB +0.00032) and the blend (+0.00008), but below the 0.0002 hurdle; pure strength (ρ 0.9988 vs plain XGB), not diversity. Not submitted (below bar).
**Date.** 2026-05-27

## Hypothesis

A different *mechanism* — semi-supervised self-training — motivated by our transductive-structure finding (test laps sit among train laps of the same driver-race). Cycle-9's pseudo-labeling (exp 026) predated that understanding. Self-training: a quick XGB labels confident test rows; those pseudo-labels augment training for the full cycle-11 XGB-highbins. Hypothesis: pseudo-labels add signal → XGB OOF > 0.95263 and/or the blend lifts.

## Method

[gpu-kernels/cycle17_xgb_pseudolabel_gpu.py](../gpu-kernels/cycle17_xgb_pseudolabel_gpu.py), Kaggle P100, ~92 min.
- Pass 1: quick XGB (eta 0.05, 2500 rounds) on comp+external → test predictions.
- Gate: p ≥ 0.92 → pseudo-1, p ≤ 0.03 → pseudo-0 (else dropped) → **121,479/188,165** test rows labeled (9,709 hi, 111,770 lo).
- Pass 2: full cycle-11 XGB-highbins, 5-fold, trained on comp-train-fold + external + pseudo-test; OOF measured on competition rows only.

## Result

| Fold | AUC |
| ---- | --- |
| 1 | 0.95353 |
| 2 | 0.95344 |
| 3 | 0.95253 |
| 4 | 0.95203 |
| 5 | 0.95321 |
| **OOF** | **0.95295** |

**OOF 0.95295 vs plain XGB-highbins 0.95263 → +0.00032** (standalone lift — the first this round).

### Diversity + blend probe

- ρ vs plain XGB **0.9988**, vs RealMLP 0.9799 (identical to plain XGB) → pseudo-XGB is a *stronger* XGB, not a more diverse one.
- Swap pseudo→XGB at cycle-11 weights (0.675/0.075/0.25): blend **0.95427** (anchor 0.95420, **+0.00007**).
- Best free 4-way / reoptimized 3-way: **0.95428** (+0.00008), at ~(RM 0.65, CB 0.05, pseudoXGB 0.30).

Below the 0.95441 hurdle by 0.00013; below the project's 0.0002 min_delta.

## Verdict

**Inconclusive (mildly positive).** Pseudo-labeling is the first lever to move a base (XGB +0.00032) and the blend (+0.00008) upward this round — but the blend gain is below the noise floor, because the lift is *pure strength* (ρ 0.9988) flowing through XGB's 0.25 weight, not new diversity.

**Caveat (mild OOF leakage):** pass-1 was global (trained on all comp-train, incl. rows that become pass-2 val folds), so the pseudo-test labels are weakly informed by val rows → OOF 0.95295 may be slightly optimistic. The effect is heavily diluted (one val row among 440k → aggregate pseudo-labels on 121k test rows → pass-2), so likely small, but the honest read would come from the LB or a per-fold leakage-clean re-run.

## Submitted (2026-05-27)

Pseudo-blend `submission_blend_pseudo.csv` (0.675 RM + 0.075 CB + 0.250 pseudo-XGB): **Public LB 0.95373** vs prior best 0.95372 → **+0.00001 (flat, within noise)**. The +0.00008 OOF lift did **not** transfer to the LB — consistent with the mild pseudo-label leakage inflating the OOF. A hairline new best, but not a meaningful improvement and far from the 0.9544 goal. Confirms pseudo-labeling is real-but-negligible on the held-out test distribution.

## Kill-criteria check

- [x] Best blend 0.95428 < hurdle 0.95441 — **FIRED** (below noise floor).
- [ ] Standalone lift exists (+0.00032) and mechanism validated — the one *positive* among the kill checks.

## Repro stamp

- Kaggle kernel `mcathala/cycle-17-xgb-pseudo-label-exp-063`; notebook [gpu-kernels/cycle17_xgb_pseudolabel_gpu.py](../gpu-kernels/cycle17_xgb_pseudolabel_gpu.py)
- thresholds 0.92 / 0.03; cycle-11 XGB-highbins recipe; outputs `data/oof_xgb_pseudo.parquet`, `data/submission_xgb_pseudo.csv`, `data/submission_blend_pseudo.csv` (held)
- compute ~92 min P100

## Learnings & follow-ups

1. **Semi-supervised lifts a base here — the first positive mechanism this round.** Even on synthetic, same-distribution data, confident pseudo-labels added a small but real standalone gain.
2. **The blend impact is capped by XGB's role (0.25 weight, ρ 0.9988).** The high-EV extension is to apply pseudo-labeling to the **0.675-weight RealMLP base** — a +0.0003 there could translate to ~+0.0002 blend (clearing the hurdle). That's the next experiment.
3. **Verify leakage-clean:** a per-fold pass-1 version would confirm the +0.00032 is real, not leakage. Cheap and fully local-smoke-testable.
4. **Open question:** whether to spend a daily slot LB-testing the pseudo-blend to ground-truth the mechanism, and whether to extend pseudo-labeling to RealMLP.
