# Experiment 087 — alt-HP RealMLP (HP-diverse RM view) → cosmetic, not escalated

**Cycle.** 18
**Status.** **Inconclusive — passes the loose gate but cosmetic; no 6-seed.**
**Date.** 2026-05-29

## Hypothesis

exp 084 showed FE-view diversity breaks the RealMLP-saturation ceiling (a 2nd RM view splits weight, transfers ~1:1 to LB). Test the complementary axis — *hyperparameter* diversity: same diffFE lean-FE + r2-pseudo, but a deliberately different training regime (lr=0.04 cos-anneal, sq_mom=0.99, first_layer_lr_factor=0.25, hidden=[512,256,128] silu, p_drop=0.05, retuned PLR, ls_eps=0.01). A different optimizer trajectory + capacity should land in a different basin → a decorrelated RM view earning separate blend weight.

**Gate (single-seed feasibility):** OOF ≥ 0.9537 AND ρ < 0.99 vs psRM6r2 → escalate to 6-seed and add as a 3rd RM view.

## Result

| Metric | Value | Gate | Pass? |
| --- | --- | --- | --- |
| Single-seed OOF | **0.95370** (vs psRM6r2 0.95396, Δ −0.00026) | ≥ 0.9537 | ✅ (by 0.00000) |
| ρ vs psRM6r2 | **0.9892** | < 0.99 | ✅ |

Per-fold AUC: 0.95449 / 0.95414 / 0.95368 / 0.95250 / (f5) — high fold variance.

**It passes both gates — but barely, and on the wrong side of meaningful.** The free blend probe is the real test:

### Free blend-weight probe (single-seed altHP added to the best-blend pool)

| Blend | OOF | altHP weight |
| --- | --- | --- |
| best blend (8 bases) | 0.95462 | — |
| **+ altHP single-seed** | **0.95463** | **0.108** |

altHP earns a non-trivial 0.108 weight (displacing psRM6r2 0.292 and diffrm6 0.209) **but lifts OOF by only +0.00001.** It earns weight as a different *noise realization* of the same model, not as new signal — the classic near-duplicate pattern. ρ 0.9892 is nowhere near the ρ ≤ 0.96 that a genuine diversity win requires (cf. exp 084's transferable view sat at ρ 0.9907 but came with a real +0.00012; this one does not).

## Verdict

**Inconclusive — not escalated to 6-seed.** A 6-seed run (~5.5h MPS) would, best case, firm the +0.00001 into ~+0.00003 OOF — but ρ 0.989 says that gain is cosmetic, not real diversity, and historically sub-+0.00005 OOF transfers at or below noise to LB. Poor stewardship to spend 5.5h chasing it.

**This is the substantive finding:** *HP-diversity within RealMLP is cosmetic* — the diffFE/pseudo recipe dominates the basin regardless of the optimizer trajectory, so alt-HP lands at ρ 0.989, same neighborhood as every other RM view. Combined with exp 088 (robust-loss GBDT diversity failed numerically), **both diversity levers tried this cycle came up empty — direct evidence the own-tooling blend is near-saturated** (audit cycle-18 thesis). Genuine diversity now needs a different *mechanism* (e.g. a prior-fitted model), not another view or HP set of our existing bases.

## Acceptance gates

| Gate | Got | Pass? |
| --- | --- | --- |
| Loose gate (OOF≥0.9537 & ρ<0.99) | 0.95370, 0.9892 | ✅ (both, barely) |
| Real-diversity (ρ ≤ 0.96) | 0.9892 | ❌ |
| Blend marginal (≥ +0.0001 OOF) | +0.00001 | ❌ |

## Repro stamp

- Trainer: [src/research/train_realmlp_altHP.py](../src/research/train_realmlp_altHP.py) (single seed 42).
- Output: `data/oof_realmlp_altHP_s42.parquet` (0.95370).

## Learnings

1. **A base earning coord-descent weight ≠ a base adding signal.** altHP took 0.108 weight for +0.00001 OOF — coord-descent rewards independent noise realizations. The honest gauge is the OOF *delta* and ρ, not the weight.
2. **HP-diversity is cosmetic for RealMLP on this data** — the recipe/data fix the basin; the optimizer trajectory doesn't move ρ off ~0.99. Closes the HP-diversity lever, symmetric with exp 088 closing the loss-surface lever.
3. With both this-cycle diversity attempts empty, the own-tooling blend is near-saturated; the next genuine lever must change the model *mechanism*.
