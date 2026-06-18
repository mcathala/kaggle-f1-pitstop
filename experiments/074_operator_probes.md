# Experiment 074 — operator-family blend probes on the 9-base zoo

**Cycle.** 17 (post-audit Phase-1 #4)
**Status.** Inconclusive (linear is optimal; operator-family axis closed).
**Date.** 2026-05-28

## Hypothesis

Non-linear blend operators (rank-remap, logit-rank, geometric mean) extract a small additional lift (≥ +0.00010 OOF) over the linear coord-descent optimum on the now-9-base zoo, by exploiting distribution shape differences across bases.

## Rationale

Audit Phase-1 #4. Linear blending has been the workhorse since cycle 7; the audit hypothesized that with 9 bases (more distribution shapes than ever before) the operator family might pull a small lift. exp 029 closed this on 2 inputs; the question was whether more inputs change the picture.

## Result

Loaded 9 bases (8 from prior cycles + self-distill from exp 075). Coord-descent on weights per operator, starting from the canonical 3-way anchor:

| Operator | Best OOF | Δ vs anchor (0.95436) | Weights (>1e-3) |
| -------- | -------- | --------------------- | --------------- |
| **linear**   | **0.95455** | **+0.00019** | psrm6r2 0.508 / psxgb 0.265 / selfdistill 0.190 / cb 0.038 |
| rank_avg | 0.95454 | +0.00019 | psrm6r2 0.503 / psxgb 0.315 / selfdistill 0.182 |
| remap    | 0.95455 | +0.00019 | psrm6r2 0.505 / psxgb 0.315 / selfdistill 0.180 |
| logit    | 0.95453 | +0.00018 | psrm6r2 0.514 / psxgb 0.313 / selfdistill 0.174 |
| gmean    | 0.95453 | +0.00017 | psrm6r2 0.497 / psxgb 0.255 / selfdistill 0.199 / cb 0.049 |

All operators within Δ +0.00001 of linear. Linear remains optimal.

## Verdict

**Inconclusive — linear is optimal; operator-family lever closed.** Worth noting: the free-grid optimum *also* prefers self-distill at w≈0.19, essentially reproducing the exp 075 submission that LB-regressed by −0.00026. So the +0.00019 OOF lift the operator probe identifies is the same OOF mirage as exp 075. The operator family probe does **not** discover any new submittable blend — the best OOF candidate it produces was already empirically falsified at the LB.

## Kill-criteria check

- [x] No operator beats linear by ≥+0.00010 OOF → operator-family lever closed.
- [x] Best operator candidate is the same blend that LB-regressed in exp 075 → no new submission warranted.

## Implications

- Closes the audit's P1-#4 "logit-rank + power-mean grid" lever cleanly.
- The free-grid optimum's preference for self-distill at w=0.19 is itself confirmation of the exp 075 false-positive: removing or downweighting self-distill in the blend means we lose the OOF lift, but the LB pattern says that lift wasn't real anyway.
- Combined with exp 073 / 075 / 076, this is the fourth and final Phase-1 lever to close empty in this round.

## Repro stamp

- Script: [src/research/blend_operator_probes.py](../src/research/blend_operator_probes.py)
- Output: `data/blend_operator_sweep.parquet`
- Bases loaded: rm6, psrm6, psrm6r2, cb, xgb_highbins, psxgb, psxgb2, hgbc, selfdistill (9)
- Anchor starting weights: psrm6r2 0.675 / cb 0.075 / psxgb 0.250

## Learnings

1. **At our scale (9 bases, ρ ∈ [0.95, 0.99] across the zoo, AUC range 0.945–0.954), linear blending sits within +0.0001 of every alternative operator.** The operator-family hypothesis was the audit's lowest-EV item; the result is consistent with that.
2. **The coord-descent's preference for self-distill confirms the false-positive nature.** Without LB ground-truth, the optimizer happily picks up the same OOF mirage; this is structural evidence that the gap between OOF-best and LB-best is fundamental, not solver-dependent.
3. **The remaining live lever is exp 071 (pseudo-CB-exp14).** Mechanism-different: lifts the CB base, not introduces a diversity-forcing technique. If 071 lands and its blend lift transfers (unlike self-distill), we have a real Phase-1 win.

## Follow-ups

- Closed: operator-family probes (all 5 operators tested, all within +0.0001 of linear).
- Open: exp 071 verdict + comprehensive 10-base blend probe after psCB-exp14 lands.
