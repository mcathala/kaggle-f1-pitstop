# #010 — Stacking meta-model on base OOFs

**Status.** Inconclusive
**Date.** 2026-05-21
**Pre-registered as cycle #009 followup #2.**

## Hypothesis

A stacking meta-model trained on logit-transformed base OOFs (LGB, CB#004, CB#006, CB#007) extracts diversity beyond fixed-weight blending and beats the 3-way baseline OOF 0.94866 by ≥ +0.00020.

## Rationale

After two consecutive Inconclusive CB-variant cycles (7 and 9, both +0.0001 ensemble lift, both 5/5 folds positive but sub-noise), the CB-variant branch is exhausted. Cycle 10 pivots to ensemble-methods exploration before committing the larger NN/XGBoost cost. Stacking is a textbook "extract more from existing bases" move — if it works, that's information; if it doesn't (the OOFs are *already* additively combinable), that's *also* information and rules out a whole branch cheaply.

## Setup

- Meta-features: `logit(oof_lgb)`, `logit(oof_cb004)`, `logit(oof_cb006)`, `logit(oof_cb007)` — clipped probabilities transformed to log-odds, where additive blending is the natural operation.
- 5-fold StratifiedKFold on Year × PitNextLap, same seed=42 and same split as base models.
- Three meta-models tested in parallel:
  1. Logistic regression on 3 base OOFs (no CB#007).
  2. Logistic regression on 4 base OOFs (incl CB#007).
  3. Shallow LightGBM on 4 base OOFs (depth 3, num_leaves 7, min_data_in_leaf 200, lr 0.05). Slightly more capacity than logreg; can learn slice-specific weights.
- No raw features added — pure stacking, to isolate the meta-learning effect from feature engineering.

## Result

```
Base OOF AUCs (sanity):
  lgb   = 0.94166
  cb004 = 0.94774
  cb006 = 0.94806
  cb007 = 0.94814

Fixed-weight baselines:
  3-way (LGB+CB#004+CB#006 at 0.10/0.40/0.50): OOF = 0.94866  fold std 0.00045
  4-way cycle 7  (+CB#007 at 0.30 each CB):    OOF = 0.94880  fold std 0.00048
```

| Meta-model | Inputs | OOF AUC | fold std | Δ vs 3-way | Δ vs 4-way |
|---|---|---|---|---|---|
| Logreg-3 | LGB+CB#004+CB#006 logits | 0.94859 | 0.00048 | **−0.00006** | — |
| Logreg-4 | + CB#007 logit | 0.94875 | 0.00051 | +0.00009 | −0.00005 |
| LGB-meta-4 | (same 4 logits, shallow LGB) | 0.94871 | 0.00050 | +0.00006 | −0.00008 |

Per-fold (best meta-model, Logreg-4): 0.94864, 0.94929, 0.94785, 0.94894, 0.94911 (4/5 improved vs 3-way; tied/below 4-way fixed-weight).

### Reproducibility stamp

- git SHA at start: `690c4b9`
- data sha256(train.csv): `f004e79d4e63f4bad0afc3788d07938dc5dbc0bed73a51e7b65cbce52ccc4128`
- packages: lightgbm 4.6.0, scikit-learn 1.8.0, scipy 1.x (logit)

### Acceptance gates

| Gate | Target | Got (best meta) | Pass? |
|---|---|---|---|
| Magnitude (Δ OOF ≥ 0.000226) | ≥ 0.000226 | +0.00009 | **FAIL** |
| Direction (≥ 3/5 folds improved vs 3-way) | ≥ 3 | 4/5 | PASS |
| Beats existing fixed-weight 4-way (0.94880) | ≥ 0.94880 | 0.94875 | **FAIL** |
| Stability (fold std ≤ 1.5 × baseline) | ≤ 0.000675 | 0.00051 | PASS |

## Verdict

**Inconclusive — but with strong diagnostic value.**

The best meta-model (Logreg-4) lifts +0.00009 over 3-way but is **−0.00005 *below* the fixed-weight 4-way ensemble we already have**. Stacking added zero useful capacity over fixed weights. Three pieces of evidence:

1. **Logreg-3 (no CB#007) ran −0.00006 below the 3-way fixed-weight baseline.** The learned linear weights were *worse* than the hand-picked 0.10/0.40/0.50. Means the base OOFs are already in their best linear combination — the meta-LR can only find a slightly worse one.
2. **Logreg-4's improvement over Logreg-3 (+0.00016) came entirely from including CB#007's information**, not from any meta-learning advantage — the fixed-weight 4-way captures the same with no fitting needed.
3. **Shallow LightGBM didn't outperform logistic regression.** The base-OOF-to-target relationship is essentially linear in logit space; non-linear meta-learning has nothing to add.

## Learnings — the meaningful finding

**The ensemble-methods axis is exhausted.** Across cycles 4, 6, 7, 9, 10 we've established:

- Fixed-weight blending of LGB + multiple CB variants: works (cycle 6 = +0.00622 LB)
- Adding another CB variant on tweaked features: ~+0.0001 each, below noise floor
- Stacking on those same OOFs: same +0.0001 ballpark, doesn't beat fixed weights
- Non-linear meta-model on OOFs: no advantage over linear

**The +0.00554 remaining LB gap cannot be closed by ensemble methods.** It requires *new information* — a model family that sees the data differently (NN with embeddings), or features that move the existing models more (and the cycle-7/9 evidence says incremental peer-rank / pit-cluster features don't).

This is a definitive negative result. Cycle 11 must pivot to model family.

## Follow-ups

1. **Cycle 11 = NN with categorical embeddings.** Pre-registered since cycle 4. Highest expected variance, only untried branch.
2. **No code change to revert** — stacking didn't touch features.py or any model trainer; only added `src/research/stack_cycle010.py`.
3. **Side benefit kept**: the 4-way fixed-weight ensemble OOF 0.94880 from cycle 7 was just re-confirmed by cycle 10's analysis. If we never beat it, that's the eventual LB submission.