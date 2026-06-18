# Experiment 067 — round-2 pseudo-XGB (strong-blend labeler)

**Cycle.** 17
**Status.** Inconclusive (negative). Round-2 OOF 0.95276 (+0.00013), *smaller* lift than round-1's 0.95295 (+0.00032). Blend with round-2 = 0.95427, **worse** than round-1's 0.95432. Iterative self-training does not compound — closes the round-2 lever and supports the round-1 OOF-was-partly-leakage hypothesis.
**Date.** 2026-05-28

## Hypothesis

exp 063 round-1 used a quick XGB (~0.94) as labeler → +0.00032 standalone, +0.00008 blend. Hypothesis: use a *stronger* labeler — the project-best strong-blend (OOF 0.95433, exp 065) — so confident pseudo-labels are higher-quality → bigger XGB lift → bigger blend gain. Privately uploaded labeler dataset `mcathala/f1-pitstop-blend-labeler` for the kernel.

## Result

Kaggle P100, ~75 min. Strong-blend labeler is more calibrated → fewer false-confidents: **113,753** pseudo-labeled rows (hi 6,751 / lo 107,002) vs round-1's 121,479 (hi 9,709 / lo 111,770).

| Fold | AUC |
| ---- | --- |
| 1 | 0.95341 |
| 2 | 0.95325 |
| 3 | 0.95232 |
| 4 | 0.95183 |
| 5 | 0.95303 |
| **OOF** | **0.95276** |

Standalone Δ vs plain XGB-highbins (0.95263) = **+0.00013** — *less than half* round-1's +0.00032.

### Blend probe (both rounds available)

| config | OOF |
| ------ | --- |
| anchor 3-way (RM+CB+plain XGB) | 0.95420 |
| pseudo-blend round-1 (psRM6+CB+**ps1**) | **0.95432** |
| swap round-2 (psRM6+CB+**ps2**) | 0.95427 |
| free 6-way grid (best) | 0.95433 at w=(rm 0.05, psr6 0.60, cb 0.05, **ps1 0.30, ps2 0**) |

The free grid **awards round-2 zero weight** even when both are available.

ρ ps2-vs-ps1 = 0.9994; ps2-vs-plain XGB = 0.9989; ps2-vs-RM = 0.9804 — round-2 is essentially the same ranking as round-1 / plain XGB, just at a lower AUC.

## Verdict

**Inconclusive (negative).** Iterative self-training doesn't compound here. Round-2 is strictly dominated by round-1 in both standalone and blend.

**Crucial interpretive takeaway:** the stronger, more-calibrated labeler gave a *smaller* lift. That is exactly what the leakage analysis predicted — round-1's pass-1 saw val rows, so its higher pseudo-label confidence (9,709 high-conf vs round-2's 6,751) was partly OOF-leakage. Round-2's labeler (the strong blend's test predictions) is much less leakage-prone, and its honest lift is +0.00013 standalone / 0 in the blend. So **the real, honest pseudo-labeling effect on XGB ≈ +0.00013 standalone, ≈ 0 in the blend** — round-1's 0.95432 is partly OOF-overfit.

Implication for LB: round-1's pseudo-blend submission to LB read 0.95373 (vs prior 0.95372, flat). That matches a real lift near zero — consistent with the leakage hypothesis. Round-2 confirms it independently. The pseudo-labeling thread is now decisively closed.

## Kill-criteria check

- [x] Round-2 OOF (0.95276) < round-1 OOF (0.95295) — **FIRED** (no improvement from stronger labels).
- [x] Best blend with round-2 (0.95427) < round-1 (0.95432); free grid picks w_ps2=0 — **FIRED**.

## Repro stamp

- Kaggle kernel `mcathala/cycle-17-xgb-pseudo2-exp-067` (v1 ERROR due to my own `dt` undefined bug; v2 fixed)
- private labeler dataset: `mcathala/f1-pitstop-blend-labeler` (submission_blend_pseudo6.csv, OOF 0.95433)
- outputs: `data/oof_xgb_pseudo2.parquet`, `data/submission_xgb_pseudo2.csv`
- compute ~75 min P100 v2 (~16 min wasted on v1's `dt` bug — cheap learning)

## Learnings

1. **Pseudo-labeling does not iterate.** Higher-quality labels don't grow the lift — they reveal that round-1's "lift" was partly leakage-inflated OOF.
2. **The honest standalone effect of pseudo-labeling here is ≈ +0.00013 on XGB**, with ≈ 0 blend impact. That matches the LB result (flat 0.95373 vs 0.95372).
3. **Confirms the OOF→LB drift story:** when the OOF advantage is real, LB transfers; when it's partly leakage, it doesn't. We have a clean example of each.

## Follow-ups

- Closed: pseudo-labeling thread (round-1 OOF advantage = partly leakage; honest lift is too small to matter).
- exp 068 (pseudo-CatBoost local) is in flight to complete the symmetric pseudo-on-all-bases coverage; expected to show the same marginal pattern (+~0.0001 standalone, ~0 blend).
