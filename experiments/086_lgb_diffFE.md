# Experiment 086 — diffFE LightGBM (new GBDT family, lean FE)

**Cycle.** 18
**Status.** Inconclusive — stronger-than-rich LGB base but w=0 in blend (ρ-saturated).
**Date.** 2026-05-29

## Hypothesis

LightGBM is leaf-wise (vs XGB level-wise) — a structurally different GBDT. With the lean diffFE recipe (49 feat, which strengthened XGB/CB), a diffFE-LGB might be both stronger than the rich LGB-highbins AND a distinct enough view to add blend diversity.

## Result

| | OOF | ρ vs RM | ρ vs CB |
| --- | --- | --- | --- |
| diffFE-LGB | **0.95208** | 0.9775 | 0.9736 |

ρ 0.9775 vs RealMLP is *slightly* lower (more diverse) than diffFE-XGB's 0.980 — leaf-wise growth does decorrelate a touch. But standalone 0.95208 is weaker than the XGB views (0.9529-0.9530) already in the blend.

**Blend test** (`build_best_blend.py` free coord-descent, 11 bases): diffFE-LGB gets **w=0**; blend OOF stays 0.95462. The marginal extra diversity doesn't compensate for being −0.001 weaker than the XGB views.

## Verdict

**Inconclusive — w=0.** diffFE generalizes to LightGBM too (stronger than the rich LGB-highbins, which had been w=0), but in the RealMLP-dominated blend it's redundant with the stronger XGB views. Same low-leverage outcome as diffFE-CB (exp 081).

## Learnings

1. diffFE is now confirmed across ALL four GBDT-ish families (XGB +0.00028, CB +0.00068, LGB stronger-but-weak, all standalone gains). The lever is real but only the XGB views earn blend weight (they're the strongest GBDT family here).
2. The blend's GBDT slot is saturated by the two XGB FE-views; adding more GBDT families/views at ρ≈0.98 and lower strength is closed.
3. The remaining lever is RealMLP FE-views (different story — RM dominates the blend weight).

## Repro stamp

- Trainer: [src/research/train_lgb_diffFE.py](../src/research/train_lgb_diffFE.py) (49 feat, max_bin=5000, num_leaves=127).
- Output: `data/oof_lgb_diffFE.parquet` (0.95208), `data/submission_lgb_diffFE.csv`.

## Follow-ups

- Closed: LGB as a blend base. Continue RealMLP FE-view diversity (exp 087 altHP-RM).
