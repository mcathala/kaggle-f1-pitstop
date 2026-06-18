# Experiment 071 — pseudo-CB on canonical CB-exp14 recipe (round-1 self-training)

**Cycle.** 17
**Status.** Inconclusive (negative-ish) — standalone +0.00012 OOF (below 0.00020 floor); blend lift only +0.00001 OOF; not submitted.
**Date.** 2026-05-28

## Hypothesis

Pseudo-labeling lifts the canonical CB-tuned-exp14 base (OOF 0.95114) by ≥ +0.00005 standalone, and the new blend (psRM6r2 / psCB-exp14 / psXGB) clears the prior 0.95436 ceiling — symmetric completion of round-1 pseudo-labeling on all three production bases.

## Rationale

1. exp 068 ran pseudo-CB but on the *earlier* CB-tuned recipe (5000 iters, OOF 0.95085) — not the canonical CB-exp14 (8000 iters, OOF 0.95114) that's actually in the production blend. Result: weaker base → weaker contribution → w=0 in the blend. This experiment fixes the base.
2. exp 063 (psXGB) +0.00032 standalone; exp 065 (psRM6) +0.00010 standalone; exp 069 (psRM6r2) +0.00013 standalone. Each pseudo-labeling pass produces a small positive on the right base. CB on its canonical recipe should follow the same pattern.
3. With the new 0.95436 blend as the labeler (vs the 0.95433 used for exp 069), pseudo-labels are slightly higher-quality — should propagate a small extra lift through all subsequent bases.

## Expected magnitude

- Standalone: +0.00005 to +0.00020 OOF over CB-exp14 (0.95114).
- Blend `submission_blend_pseudo_r2_psCB.csv` (psRM6r2 + psCB-exp14 + psXGB at 0.675/0.075/0.250): +0.00001 to +0.00005 over 0.95436.
- Floor: if the new psCB OOF ≤ 0.95114, the lever is genuinely dead for CB regardless of base — close.

## Overfitting risk

Low. Same CV/seed/folds as every prior CB run. Pseudo-labels come from `submission_blend_pseudo_r2.csv` (the new 0.95436 LB-0.95375 submitted blend), so labels are fully held out from train.

## Kill criteria

- [ ] Standalone OOF < 0.95114 (no improvement over base CB-exp14) → kill the lever for CB.
- [ ] Best blend OOF (4-way grid with psCB-exp14 available) ≤ 0.95436 → CB doesn't get weight from pseudo either.

## Scope

- `src/train_cb_pseudo_exp14.py` (new — fork of `train_cb_pseudo.py` with iterations 5000→8000, early_stop 300→500, labeler swap to `submission_blend_pseudo_r2.csv`, output renames).
- `experiments/071_cb_pseudo_exp14.md` (new).
- Outputs: `data/oof_cb_pseudo_exp14.parquet`, `data/submission_cb_pseudo_exp14.csv`.

## Reversibility check

No CV / seed / target changes. Reversible.

## Result

5-fold M1 CPU, 5h 39min wall (one fold throttled by sleep before caffeinate kicked in). All folds ran to ~iter 7800-7995 within the 8000 cap.

| Fold | AUC      | best iter | runtime |
| ---- | -------- | --------- | ------- |
|  1   | 0.95171  | 7775      |  54 min |
|  2   | 0.95201  | 7995      | 129 min (lid-closed throttle) |
|  3   | 0.95083  | 7951      |  68 min |
|  4   | 0.95043  | 7814      |  50 min |
|  5   | 0.95135  | 7975      |  50 min |
| **OOF** | **0.95126** | per-fold mean ± std = 0.95127 ± 0.00057 | — |

Standalone Δ vs CB-tuned-exp14 (0.95114) = **+0.00012**. Below the +0.00020 magnitude gate; above the +0.00005 floor.

### Per-year AUC

| Year | OOF AUC | n        | pos_rate |
| ---- | ------- | -------- | -------- |
| 2022 | 0.91573 | 82,989   | 0.2665   |
| 2023 | 0.94597 | 136,147  | 0.0096   |
| 2024 | 0.92993 | 127,110  | 0.2953   |
| 2025 | 0.92938 | 92,894   | 0.2844   |

### Comprehensive 10-base blend probe (with psCB-exp14)

| Blend (with self-distill) | OOF | Δ vs anchor (0.95436) |
| ------------------------- | --- | --------------------- |
| Linear free grid | **0.95456** | +0.00020 |
|   weights: psrm6r2 0.500 / psxgb 0.256 / **selfdistill 0.194** / pscb_exp14 0.050 / (CB 0) | | |
| **Blend WITHOUT self-distill** (cleaner isolation of psCB contribution) |  |  |
| Linear free grid (psrm6/r2/r2 + psxgb + pscb_exp14) | **0.95437** | **+0.00001** |
|   weights: psrm6r2 0.584 / psxgb 0.289 / psrm6 0.078 / pscb_exp14 0.049 / (CB 0) |  |  |

The +0.00020 figure in the 10-base optimum is **dominated by self-distill** (the same OOF-mirage base that LB-regressed in exp 075). With self-distill excluded, the genuine incremental contribution of psCB-exp14 over the production blend is **+0.00001 OOF** — below the noise floor.

Notably, both grids drop the original CB in favor of psCB-exp14 (CB at w=0 in both refined optima). psCB-exp14 is strictly better than CB for the blend, but the gain is tiny because CB's weight in the anchor was already only 0.075.

## Verdict

**Inconclusive (negative-ish for blend; mildly positive for base).** psCB-exp14's standalone +0.00012 OOF over CB-exp14 confirms the pseudo-labeling lift exists on the canonical CB recipe (vs exp 068's accidental fork with no lift). But blend leverage is +0.00001 — below the +0.00005 gate and far below the +0.00020 noise floor. Not submitted (slot conservation).

## Held submission candidates

None constructed (the +0.00001 blend gain doesn't justify a slot; the +0.00020 grid optimum is the same self-distill mirage from exp 075).

## Kill-criteria check

- [x] Standalone OOF (0.95126) > CB-exp14 baseline (0.95114) by +0.00012 → kill criterion 1 does not fire (real positive).
- [x] Best blend OOF without self-distill (0.95437) ≤ anchor (0.95436) within noise → effective blend lift = 0 → **practical lever closure**.

## Repro stamp

- Trainer: [src/train_cb_pseudo_exp14.py](../src/train_cb_pseudo_exp14.py) — forked from `train_cb_pseudo.py` (exp 068) with iterations 5000→8000, early_stop 300→500, labeler swap to `submission_blend_pseudo_r2.csv`.
- Outputs: `data/oof_cb_pseudo_exp14.parquet`, `data/submission_cb_pseudo_exp14.csv`.
- Blend probe: [src/research/blend_operator_probes.py](../src/research/blend_operator_probes.py).
- 5-fold StratifiedKFold(shuffle=True, random_state=42) on `Year × PitNextLap`.

## Learnings

1. **The lever was real but small.** +0.00012 standalone is a clean positive — pseudo-labeling lifts CB on its canonical recipe (unlike exp 068's wrong-base fork). But CB at w=0.075 doesn't have enough blend leverage for this lift to matter.
2. **Pseudo-labeling has now been characterized on all three production bases on their canonical recipes**: psRM6 (+0.00010 round-1, +0.00013 round-2), psXGB (+0.00032 round-1 leakage-tainted / +0.00013 round-2 honest), psCB-exp14 (+0.00012). Standalone lifts are all in the +0.00010 - +0.00015 range. Blend leverage is gated by each base's anchor weight.
3. **The free-grid swap CB → psCB-exp14 at the SAME weight (0.075) is the right way to think about this lift.** It's a marginal-component upgrade, not a new base. Worth keeping in the pipeline registry.

## Follow-ups

- Closed: pseudo-CB-exp14 as a "new base" candidate.
- Implicit upgrade: future blends should default to psCB-exp14 in place of CB-exp14 (same weight, slightly better).
- Pseudo-labeling thread now fully exhausted across all three production bases.
