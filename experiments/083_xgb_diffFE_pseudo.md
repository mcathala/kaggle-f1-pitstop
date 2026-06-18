# Experiment 083 — diffFE + pseudo-label XGB (compounding the night's win)

**Cycle.** 18
**Status.** Kept (strongest XGB base ever; blend marginal). New best **LB 0.95389**.
**Date.** 2026-05-29

## Hypothesis

The diffFE-XGB (exp 080, stripped 49-feat, OOF 0.95291) is nearly as strong as the rich pseudo-XGB (0.95295) but cleaner. Applying our pseudo-labeling self-training to the *cleaner* base should give our strongest XGB and lift the saturated blend.

## Result

| XGB variant | OOF |
| --- | --- |
| rich (132 feat) | 0.95263 |
| diffFE (49 feat, exp 080) | 0.95291 |
| rich + pseudo (psXGB) | 0.95295 |
| **diffFE + pseudo (this exp)** | **0.95299** ← strongest XGB in project |

Pseudo added +0.00008 on the diffFE base (vs +0.0003 on rich — smaller, consistent with the leakage-explained pseudo story: a cleaner base has less leakage headroom). ρ(diffFE-psXGB, psXGB) = 0.989.

### Blend (v3, free coord-descent over strongest pool)

`psRM6r2 0.576 / diffFE-psXGB 0.192 / psXGB 0.131 / diffFE-XGB 0.02 / cbdiff 0.04 / pscb14 0.04` → **OOF 0.95449**.

Submitted `submission_blend_diffFE_v3.csv` → **LB 0.95389** (new best, +0.00001 over the diffFE-XGB blend's 0.95388).

## Verdict

**Kept** (strongest XGB base; new best LB 0.95389 by a hair). But the blend is **saturated**: OOF stuck at 0.95449 regardless of which strong XGB view we use, because all XGB views are ρ≈0.988 to each other and the blend is dominated by RealMLP (w=0.576). The diffFE lever's blend value was banked entirely in the first step (exp 080, +0.00013 LB); subsequent stronger XGB views add ~+0.00001.

## The blend ceiling (cycle-18 conclusion)

| | OOF | LB |
| --- | --- | --- |
| pre-cycle-18 best | 0.95436 | 0.95375 |
| **cycle-18 best (diffFE)** | **0.95449** | **0.95389** |

The blend is **RealMLP-limited**: psRM6r2 (0.95396, w=0.576) dominates; the GBDT views (ρ 0.98 to RM, ρ 0.988 among themselves) are mutually saturated. Breaking OOF 0.95449 requires a base that is BOTH strong (≥0.953) AND ρ≤0.95 to RealMLP — the empty quadrant confirmed across cycles 16-18 (architecture diversity, FE diversity, rank-blend of weak-diverse bases all closed).

## Repro stamp

- Trainer: [src/train_xgb_diffFE_pseudo.py](../src/train_xgb_diffFE_pseudo.py) (49-feat diffFE + strong-blend pseudo-labels, 114,551 rows).
- Outputs: `data/oof_xgb_diffFE_pseudo.parquet` (0.95299), `data/submission_xgb_diffFE_pseudo.csv`, `data/submission_blend_diffFE_v3.csv` (LB 0.95389).
- Submission 53137489. 4/5 daily slots used.

## Learnings

1. **diffFE + pseudo stack additively on the base** (diffFE +0.00028, pseudo +0.00008 → strongest XGB 0.95299), but the blend is saturated so the marginal blend/LB gain is ~+0.00001.
2. **Our pipeline blend ceiling is OOF 0.95449 / LB 0.95389** — RealMLP-limited and confirmed robust. Net cycle-18 gain +0.00014 LB from the diffFE insight.
3. **Top-7% (LB 0.95453, +0.00064 away) is the shared-public-submission plateau** (research-confirmed: ~190 teams blend the same published CSVs). Our independent pipeline does not reach it; closing the gap requires either a RealMLP breakthrough (recipe is canonical; diffFE was neutral on it) or ingesting external submission files (held off-limits under the project's integrity rules).

## Follow-ups

- The diffFE lever is fully exploited (XGB win banked; CB/RM/pseudo-stack all blend-neutral due to RM saturation).
- Open options: (a) accept LB 0.95389 as the honest independent-pipeline ceiling; (b) the shared-CSV blend for top-7% (ruled out on integrity grounds); (c) a fundamentally new strong+decorrelated base.
