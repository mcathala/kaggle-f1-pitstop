# Experiment 062 — AutoGluon WITH full stacking (the untried variant)

**Cycle.** 17
**Status.** Inconclusive — AG full-stack honest comp-OOF 0.95085, below every base; its 0.9598 "val" is inflated by external in AG's internal validation. Closes the AutoML / different-combiner lever.
**Date.** 2026-05-27

## Hypothesis

exp 024 (cycle 8) ran AutoGluon but **disabled its core mechanism** (`num_bag_folds=0, num_stack_levels=0`) to drive an external CV — so AG's multi-layer stacking was never tested here. This run enables it (high_quality, bagging + 1 stack level over LGB/CAT/XGB/NN, comp + external) and uses AG's bagged OOF. Hypothesis: AG's stacker beats our hand-tuned 3-way (0.95420) standalone, or is independent enough to lift the blend.

## Result

Kaggle P100, ~99 min, fit cap 5400s. AG trained the full bagged/stacked zoo.

- **AG bagged OOF AUC on competition rows: 0.95085** — below RealMLP (0.95383), XGB (0.95263), CB (0.95114), and the 3-way blend (0.95420).
- AG's reported leaderboard (its own internal validation):

  | model | AG score_val |
  | ----- | ------------ |
  | WeightedEnsemble_L3 | 0.95975 |
  | CatBoost_BAG_L2 | 0.95966 |
  | LightGBMXT_BAG_L2 | 0.95860 |
  | LightGBMXT_BAG_L1 | 0.95498 |
  | CatBoost_BAG_L1 | 0.95435 |

**The 0.9598 vs 0.95085 gap is the headline finding.** AG's score_val is computed on internal holdouts that **include the distribution-shifted external rows** (pos rate 0.255 vs comp 0.199, adversarial train-vs-ext AUC 0.78), which are easier to discriminate and inflate AUC by ~+0.009. The honest number — bagged OOF restricted to the actual competition distribution — is 0.95085.

(The kernel ended in ERROR on a trivial post-result bug: `predictor.predict_proba(Xte[feat])` where `feat` still contained the target column → `KeyError: PitNextLap`. This fired *after* the OOF AUC + leaderboard were printed, so the result stands; output parquet/submission were not written. Not re-run — at 0.95085 the base is strictly dominated, so it cannot earn blend weight regardless.)

## Verdict

**Inconclusive.** AutoGluon's full multi-layer stacking over a broad zoo, evaluated honestly on competition rows (0.95085), does not beat our existing bases or blend. Even L1 CatBoost (0.95435 val) collapses to comp-distribution reality. The different-combiner / AutoML lever is closed; our hand-tuned RealMLP + tree blend remains superior on the true objective.

## Kill-criteria check

- [x] AG comp-OOF (0.95085) < blend anchor (0.95420), same model families → w=0 — **FIRED**.

## Learnings

1. **External-in-validation inflates AUC ~+0.009 here.** This is the single most useful takeaway: any CV that scores on holdouts containing the external dataset reads ~0.959–0.960, but the competition distribution is ~0.951. Our project's protocol (OOF on competition rows only) is correct; this likely explains gaps between public "CV ~0.96" claims and LB ~0.954.
2. **AG stacking ≠ free lift.** A broad bagged/stacked zoo did not beat the hand-tuned 3-way on the honest objective. The ceiling is not a combiner problem (consistent with exp 061).

## Repro stamp

- Kaggle kernel `mcathala/cycle-17-autogluon-stack-exp-062`; notebook [gpu-kernels/cycle17_autogluon_stack_gpu.py](../gpu-kernels/cycle17_autogluon_stack_gpu.py)
- autogluon high_quality, num_bag_folds=5, num_stack_levels=1, excluded KNN/RF/XT, time_limit 5400s; comp + external
- compute: ~99 min P100

## Follow-ups

- Closed: AutoGluon / AutoML stacking as a lever.
- Continue overnight queue: RealMLP strengthening (063), TabR (064).
