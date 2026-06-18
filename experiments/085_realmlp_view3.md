# Experiment 085 — RealMLP view-3 (rich-RM-FE + extra high-card cross-cats)

**Cycle.** 18
**Status.** Reverted (killed at seed 2/6) — cross-cats weaken RealMLP; non-additive.
**Date.** 2026-05-29

## Hypothesis

A THIRD RealMLP FE-view (rich-RM-FE + 3 extra high-card cross-cat combos: Driver_Race, Compound_Stint, Driver_Compound) would add more blend diversity, building on the exp-084 FE-view-diversity win.

## Result

Single-seed OOF (seed 42) **0.95288**; seed 7 **0.95282** — both ~−0.0008 weaker than diffFE-RM (0.95371) and psRM6r2 (0.95396). Adding factorized high-cardinality cross-cats **hurts** RealMLP (large embeddings → overfitting), consistent with the diffFE finding that hand-built cross-cats are net-harmful for models that already encode base cats well.

Killed after seed 2 (2 consistent weak seeds = sufficient signal; saved ~1.6h MPS). A weak view (0.9528) at ρ~0.99 would not earn blend weight.

## Verdict

**Reverted.** Cross-cats are the wrong direction for a RealMLP FE-view. The productive RM-view axis is *stripping* FE (diffFE-RM, exp 082/084), not *adding* cross-cats. Confirms diffFE's mechanism: NNs want lean numeric inputs (PLR-embeddings), not hand-built high-card categoricals.

## Repro stamp

- Trainer: [src/research/train_realmlp_view3.py](../src/research/train_realmlp_view3.py) (rich-RM-FE + 3 extra combos, 47 feat). Killed mid-run; no OOF persisted.

## Follow-ups

- Closed: cross-cat RM view. Next RM view via different *optimization* (exp 087 altHP-RM: lr/schedule/dropout change) rather than more features.
