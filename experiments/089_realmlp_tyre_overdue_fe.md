# Experiment 089 — tyre-overdue FE on RealMLP → neutral (FE-for-RM closed)

**Cycle.** 18
**Status.** **Inconclusive (neutral, +0.00003).** Tyre-overdue features are rank-invariant for RealMLP; closes FE-for-RM.
**Date.** 2026-05-29

## Hypothesis

exp 088 closed the diverse-base lever (even ρ 0.959 @ 0.952 adds 0 to the saturated, RealMLP-dominated blend — the hurdle is ρ≤0.92, unreachable at strength). The only remaining path to lift a saturated blend is raising the **absolute** strength of the dominant base. Pit decisions are tyre-cliff driven, and the one mechanism-aligned real feature never tried on RealMLP is **compound-relative tyre-overdue** (exp 052 tried it on XGB → rank-invariant; never on RM). Gate: single-seed OOF ≥ 0.9542 (real lift over the 0.95370 altHP baseline → port to canonical 6-seed).

## Method

Clean FE **ablation** at fixed altHP hyperparameters (`train_realmlp_tyrefe.py`, forked from `train_realmlp_altHP.py`): identical recipe, ONLY 4 tyre-overdue features added —
`TyreLife − compound_p75`, `TyreLife − compound_p90`, `TyreLife / compound_p50`, `TyreLife > compound_p90` flag. Compound TyreLife quantiles fit on **train only** (inductive → drift-safe), global-quantile fallback for unseen compounds. Single seed. Real features only — no prediction-stacking (avoids the exp 075 OOF-mirage).

## Result

| | single-seed OOF |
| --- | --- |
| altHP baseline (exp 087) | 0.95370 |
| **+ tyre-overdue FE** | **0.95373** |
| **Δ** | **+0.00003** (noise; gate ≥0.9542 not cleared) |

## Verdict

**Inconclusive — neutral.** Tyre-overdue FE is rank-invariant for RealMLP, exactly as it was for XGB (exp 052). RealMLP's PLR-embeddings + existing tyre/lap/degradation features already capture the tyre-cliff signal; expressing it relative to the compound's pit window adds nothing to the ranking. **Closes FE-for-RealMLP** (third confirmation with diffFE-RM exp 082 neutral and the rich-FE baseline): RealMLP's ranking is FE-saturated, just as XGB's is.

## Acceptance gates

| Gate | Got | Pass? |
| --- | --- | --- |
| Absolute lift (OOF ≥ 0.9542) | 0.95373 | ❌ |

## Repro stamp

- Trainer: [src/research/train_realmlp_tyrefe.py](../src/research/train_realmlp_tyrefe.py) (single seed 42).
- Output: `data/oof_realmlp_tyrefe_s42.parquet` (0.95373).

## Learnings

1. **FE is saturated for both model families.** GBDT (051/052/055/077/080-diffFE-was-a-strip-not-an-add) and now RealMLP (082/089) both ignore new hand-features in their ranking. The only FE that ever helped was *removing* transductive over-engineering (diffFE), not adding signal.
2. Combined with exp 088 (diversity closed) and A1 (weights honest), the absolute-strength and diversity levers within our own tooling are both exhausted. The remaining direction with LB upside is the structural OOF→LB drift, not OOF gains.
