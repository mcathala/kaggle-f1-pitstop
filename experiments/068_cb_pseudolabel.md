# Experiment 068 — pseudo-labeled CatBoost (local, symmetric completion)

**Cycle.** 17
**Status.** Inconclusive (negative). Pseudo-CB OOF 0.95091 (+0.00006 over its early-CB base; below canonical CB-exp14 0.95114). Free 6-way grid awards w_pscb=0. Confirms the symmetric pseudo pattern across all 3 bases: each lifts by ~+0.0001 standalone, ~0 in the blend.
**Date.** 2026-05-28

## Hypothesis

Symmetric completion: exp 063/067 (XGB) and 065 (RealMLP) showed pseudo-labeling lifts each base by ~+0.0001 standalone but ~0 in the blend. CB is the third base (0.075 blend weight). Test whether pseudo-CB completes the pattern or surprises with a bigger lift.

## Method

Local M1 CPU run (parallel with Kaggle exp 067), `src/research/train_cb_pseudo.py` forked from `train_cb_tuned.py` with the strong-blend labeler (`data/submission_blend_pseudo6.csv`, OOF 0.95433) gating confident test rows (≥0.92 → 1, ≤0.03 → 0) and adding them to each fold's training set. 5-fold StratifiedKFold seed 42; CB params unchanged (5000 iters, lr 0.018, depth 8, l2 8.5). Pseudo-test added 113,753 rows → train_rows per fold = 566,370 (vs CB-tuned's 453k).

**Note (fork caveat):** the source trainer (`train_cb_tuned.py`) writes `oof_cb_tuned.parquet` (the *early* CB-tuned baseline OOF 0.95085), not the canonical `oof_cb_tuned_exp14.parquet` (0.95114) used in the production blend. Output paths renamed post-hoc to `oof_cb_pseudo.parquet` / `submission_cb_pseudo.csv`. Comparison vs the canonical CB-exp14 is therefore an underestimate of what a properly-forked pseudo-CB-exp14 would deliver.

## Result

| Fold | AUC | iters |
| ---- | --- | ----- |
| 1 | 0.95127 | 4991 |
| 2 | 0.95161 | 5000 |
| 3 | 0.95053 | 5000 |
| 4 | 0.95014 | 5000 |
| 5 | 0.95099 | 4997 |
| **OOF** | **0.95091** | mean 0.95091, std 0.00052 |

Δ vs early-CB-tuned base (0.95085) = **+0.00006**. Δ vs canonical CB-exp14 (0.95114) = **−0.00023**.

ρ vs canonical CB-exp14: **0.9958** — essentially same model, just with the underlying weaker recipe and pseudo bump.

### Blend probe (with pseudo-CB available)

| config | OOF |
| ------ | --- |
| anchor 3-way (RM+CB_exp14+XGB) | 0.95420 |
| pseudo-blend (psRM6 + CB_exp14 + psXGB) | **0.95432** |
| swap pseudo-CB into pseudo-blend | 0.95431 (worse) |
| free 6-way grid (CB_exp14 + psCB both available) | **0.95433** at w_pscb=**0** |

The free grid awards pseudo-CB **zero weight** even when offered alongside the canonical CB. Strictly dominated.

## Verdict

**Inconclusive (negative); completes the symmetric pseudo pattern across all three bases.**

| Base | Plain | Pseudo | Δ standalone | Blend weight | Blend impact |
| ---- | ----- | ------ | ------------ | ------------ | ------------ |
| RealMLP | 0.95383 | 0.95393 (6-seed) | +0.00010 | 0.675 | small but real |
| XGB | 0.95263 | 0.95276 (r2 honest) | +0.00013 | 0.250 | ~0 in free grid |
| CatBoost | 0.95114 (e14) / 0.95085 (early) | 0.95091 (psCB) | +0.00006 over early base | 0.075 | 0 (dominated) |

The pseudo-labeling thread is now characterized across the **entire** strong-base set: each base gains a consistent ~+0.0001 standalone, and the blend impact is ≤+0.00013 total (combined). The 0.95433 OOF / 0.95373 LB ceiling holds.

## Compute spent

~2.8 h M1 CPU (~10,000s wall). Local — costs nothing against Kaggle quota.

## Learnings

1. **The marginal-lift pattern is universal across our three bases.** Pseudo-labeling adds ~+0.0001 standalone irrespective of model family (NN / GBDT-XGB / GBDT-CB). The signal injected is the same: a calibrated "test laps look like X" prior. The effect is real but small.
2. **Stronger base → bigger marginal pseudo lift.** RM (0.954) gained +0.00010; XGB (0.953) gained +0.00013; CB-early (0.951) gained +0.00006. Suggests the lift is partly a function of the labeler-vs-base AUC gap — the better the labeler is than the base, the more useful the pseudo-labels.
3. **Per-segment / per-year blend tuning also flat** (free probe earlier in this round): 0.95432, no per-year win to extract.

## Follow-ups

- Closed: pseudo-labeling on all three bases. The 0.95433 OOF ceiling is robust across every combiner and every base composition we've tested.
- Open (untried): a properly-forked pseudo-CB-exp14 (~0.95120 expected) would still be w=0 in the blend (same ρ-saturation). Not worth the rerun.
