# Experiment 057 — adversarial-importance-weighted external data on XGB-highbins

**Cycle.** 16
**Status.** Inconclusive (Reverted) — covariate-shift reweighting *hurts* standalone (−0.00046) and stays non-diverse (ρ 0.997 vs plain XGB); no blend lift. The last live shot of the external-data batch fails; closes the external-data axis.
**Date.** 2026-05-27

## Hypothesis

Exp 056 falsified the "external is pollution" hypothesis — external helps both families at full weight. But external is still distribution-shifted (adversarial AUC 0.78). Remaining idea: keep external's volume benefit while cutting its shift drag by **importance-weighting** each external row by the covariate-shift ratio P(comp|x)/P(ext|x). If the shift drag is separable from the signal, this should beat full-weight inclusion: standalone > 0.95263, or a more diverse base that lifts the 3-way blend ≥ +0.00020.

## Rationale

- Textbook covariate-shift correction: train an adversarial classifier (competition=0 vs external=1), get p = P(ext|x), weight external rows by w = clip((1−p)/p, 0.1, 10). Rows that look most like competition data get the most weight; rows that look most off-distribution get suppressed.
- This is the principled middle ground between full inclusion (exp 044/cycle-11, helps) and full exclusion (exp 056, hurts).

## Expected magnitude

- Standalone target: ≥ +0.00020 over cycle-11 XGB-highbins 0.95263.
- Or blend target: swap/4-way OOF ≥ 0.95441.
- Floor: standalone < 0.95243 AND ρ ≥ 0.984 → revert.

## Kill criteria

- [ ] Standalone OOF < 0.95243 AND no diversity gain (ρ ≥ 0.984).
- [ ] Best blend config does not clear 0.95421 + 0.00020.

## Scope / reversibility

New Kaggle notebook [gpu-kernels/cycle16_xgb_advweight_gpu.py](../gpu-kernels/cycle16_xgb_advweight_gpu.py) — verbatim cycle-11 XGB-highbins recipe; the **only** change is per-row sample weights on the external rows (competition rows keep 1.0). Adversarial classifier is a 4-fold OOF XGB (depth 6, 150 rounds). P100, `device=cuda`. Does not touch CV/seed/target/frozen files. Reversible.

## Result

Ran on Kaggle P100, 5-fold, cycle-11 XGB-highbins HPs verbatim, external rows importance-weighted.

- **Adversarial classifier OOF AUC = 0.777** (confirms the shift). External weights: mean **3.596**, median 2.743, min 0.100, max 10.000 — i.e. the importance ratio *up-weights* external on average (most external rows look only moderately off-distribution, so (1−p)/p > 1), with the most-competition-like rows hitting the cap of 10.

| Fold | adv-weighted AUC | iters |
| ---- | ---------------- | ----- |
| 1 | 0.95297 | 3766 |
| 2 | 0.95248 | 2921 |
| 3 | 0.95174 | 4002 |
| 4 | 0.95122 | 3926 |
| 5 | 0.95245 | 3799 |
| **OOF** | **0.95217** | — |

per-fold mean 0.95217, std 0.00062.
**OOF 0.95217 vs plain XGB-highbins 0.95263 → Δ −0.00046.** Reweighting is *worse* than uniform full-weight inclusion (and better than no-external 0.95190 — so external still helps, just not when reweighted).

### Rank-correlation (OOF)

|     | rm | cb | xgb | adv |
| --- | -- | -- | --- | --- |
| rm  | 1.0000 | 0.9758 | 0.9799 | 0.9762 |
| cb  | 0.9758 | 1.0000 | 0.9840 | 0.9818 |
| xgb | 0.9799 | 0.9840 | 1.0000 | **0.9966** |
| adv | 0.9762 | 0.9818 | 0.9966 | 1.0000 |

adv vs plain-XGB ρ 0.9966 — marginally less correlated than tyre-overdue (0.9989), but firmly inside the tree-family cluster.

### Blend probe (anchor = cycle-11 3-way, OOF 0.95420)

| Config | OOF | Δ |
| ------ | --- | - |
| A swap adv→xgb | 0.95418 | −0.00003 |
| B 4-way (best split) | 0.95420 | −0.00001 |
| C xgb-avg(plain,adv) | 0.95420 | −0.00001 |
| D free 4-way grid | 0.95420 | −0.00001 |

Best 0.95420, below the 0.95441 hurdle and below the 0.95421 anchor. No-lift.

## Verdict

**Inconclusive (Reverted).** Importance-weighting external rows neither beats full-weight inclusion standalone (−0.00046) nor adds blend diversity (ρ 0.997). The shift drag and the useful signal are **not separable** by covariate-shift weighting — reweighting toward the competition distribution discards external rows that, despite being off-distribution, carry genuine pit-timing signal. Both kill-criteria fire.

## Kill-criteria check

- [x] Standalone 0.95217 < 0.95243 AND ρ 0.997 ≥ 0.984 — **FIRED** (worse *and* non-diverse).
- [x] Best blend 0.95420 does not clear 0.95441 — **FIRED**.

## Repro stamp

- Kaggle kernel: `mcathala/cycle-16-xgb-advweight-exp-057`; notebook [gpu-kernels/cycle16_xgb_advweight_gpu.py](../gpu-kernels/cycle16_xgb_advweight_gpu.py)
- recipe: cycle-11 XGB-highbins HPs (max_bin=5000, eta=0.01, depth=10, λ=8.16, α=8.35, colsample=0.145), 5-fold StratifiedKFold seed 42 on Year×PitNextLap + importance-weighted external
- probe: [src/research/blend_advweight_probe.py](../src/research/blend_advweight_probe.py)
- outputs: `data/oof_xgb_advweight.parquet`, `data/submission_xgb_advweight.csv`
- runtime: ~72 min compute on P100 (5 folds; long wall-clock was Kaggle GPU queue, not compute)

## Learnings

1. **Covariate-shift reweighting does not recover the external drag.** Full-weight inclusion (0.95263) > importance-weighted (0.95217) > no external (0.95190). The ordering shows the external signal and its distribution shift are entangled — you cannot down-weight the "off-distribution" part without also throwing away signal. Uniform inclusion is the better policy.
2. **Closes the external-data axis.** 055 (target-stats), 056 (ablation), 057 (reweighting) together establish: the external data helps, can't be improved by reweighting, and isn't the cap. The bottleneck is the model/blend, full stop.
3. **The mean importance weight > 1 is the tell.** Because the adversarial separation is moderate (AUC 0.78, not 0.95+), most external rows get up-weighted, so the scheme amplified external influence rather than suppressing it — and it still hurt. Even a *gentler* down-weighting would not have helped, since the direction of effect is wrong.

## Follow-ups

- Closed: external-data treatment (inclusion / exclusion / reweighting) as a lever. Keep external at uniform full weight, as cycle 3+ already does.
- Cycle 16 is fully exhausted: own-model (053/054), FE-on-XGB (051/052/055), and external-data (056/057) axes all closed. The OOF 0.95421 / LB 0.95372 ceiling holds. Next cycle, if any, must introduce a genuinely new algorithm or information source — not a variation on the existing stack.
