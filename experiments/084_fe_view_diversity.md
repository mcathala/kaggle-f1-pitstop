# Experiment 084 — FE-view diversity (two RealMLP FE-views) → new best LB 0.95401

**Cycle.** 18
**Status.** **KEPT — new project-best LB 0.95401** (+0.00012 over 0.95389).
**Date.** 2026-05-29

## Hypothesis

diffFE-RM (exp 082) was OOF-*neutral* standalone (0.95371 vs 0.95369), so I'd shelved it. But the diffFE-XGB blend win came from FE-*view* diversity (rich + stripped XGB, ρ 0.988), not standalone strength. So the diffFE-RM — a genuinely different FE-view of RealMLP — might add blend value despite neutral standalone, breaking the RealMLP saturation that caps the blend.

## Result

ρ(diffFE-RM, psRM6r2) = **0.9907** (the two RM FE-views are mildly decorrelated). Adding diffFE-RM to the v3 blend:

| Blend | OOF | LB |
| --- | --- | --- |
| v3 (single RM view) | 0.95449 | 0.95389 |
| **v4 (TWO RM views: psRM6r2 0.356 + diffFE-RM 0.28)** | **0.95460** | **0.95401** |

The RealMLP weight *splits* across the two FE-views (0.356 + 0.28 = 0.636 total RM), and the blend gains +0.00011 OOF → **+0.00012 LB (~1:1 transfer)**.

Full v4 weights: psRM6r2 0.356 / diffFE-RM 0.28 / diffFE-psXGB 0.216 / psXGB 0.086 / cbdiff 0.039 / pscb14 0.023.

## Verdict

**KEPT.** This is the generalization of the diffFE insight and the lever that **breaks the RealMLP saturation**: a strong model trained on a *different FE view* adds blend diversity that transfers ~1:1 to LB, even when the new view is OOF-neutral or slightly weaker standalone. The RM-dominated ceiling is not fixed — it lifts when we give RealMLP a second FE-view to split weight with.

New best **LB 0.95401** (~top 15%, from top 17% at session start).

## Repro stamp

- diffFE-RM from exp 082 (`oof_realmlp_diffFE_s42.parquet`, single-seed, 30 feat).
- Blend: `submission_blend_diffFE_v4.csv` → LB 0.95401 (submission 53137600). 5/5 daily slots used.
- Weights via free coord-descent over {psRM6r2, diffFE-RM, diffFE-psXGB, psXGB, cbdiff, pscb14}.

## Learnings

1. **FE-view diversity is the live lever.** Each independent FE recipe of a strong model family is a decorrelated-enough base to add blend value that transfers to LB. This sidesteps the "strong+diverse quadrant is empty" closure — the diversity comes from FE, not architecture, and doesn't require the new view to be *stronger*.
2. **The RealMLP saturation lifts with a 2nd RM view** — the single biggest structural ceiling of the project (RM w=0.576 dominating) is now addressable.
3. **Even a single-seed, OOF-neutral diffFE-RM added +0.00012 LB.** A 6-seed diffFE-RM (stronger view-2) should add more (exp 084b, running).

## Follow-ups

- **exp 084b: 6-seed diffFE-RM** (running, ~2.4h) — upgrades view-2 from single-seed (0.95371) to ~0.95390, should push the blend higher.
- Then: a THIRD FE-view per family (medium-FE XGB; a different RM FE recipe). Each decorrelated view compounds.
- Path to top-7% (0.95453, +0.00052): stack 3-4 FE-views × {RM, XGB} via the diffFE lever.

## Update — exp 084b: 6-seed diffFE-RM (stronger view-2)

Ran the full 6-seed diffFE-RM (seeds 42/7/99/137/313/777, per-seed 0.95370-0.95374): **6-seed OOF 0.95390** (vs single-seed 0.95371; vs psRM6r2 0.95396, −0.00006 standalone but a distinct FE-view at ρ~0.99).

Rebuilt best blend with the stronger view-2:

| Blend | OOF |
| --- | --- |
| v4 (single-seed diffFE-RM view-2) | 0.95460 → LB 0.95401 |
| **best (6-seed diffFE-RM view-2)** | **0.95462** |

Weights: psRM6r2 0.307 / diffFE-RM-6seed 0.308 / diffFE-psXGB 0.224 / psXGB 0.061 / cbdiff 0.032 / pscb14 0.023 / diffFE-XGB 0.011. The two RM FE-views now split weight ~evenly (0.307/0.308). `submission_blend_best.csv` ready (OOF 0.95462) — to submit when daily slots reset.

Diminishing within-lever: the 6-seed view-2 added only +0.00002 OOF over single-seed. The +0.00011 came from *having* a 2nd view; strengthening it is secondary. Next lever step: a THIRD distinct RM FE-view (exp 085, rich-RM-FE + extra high-card cross-cats) for fresh diversity.
