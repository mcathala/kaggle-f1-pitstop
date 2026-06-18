# Experiment 015 — Multi-seed CB-tuned

**Cycle.** 4
**Status.** Deferred — superseded by exp 035 (multi-seed CB-tuned, run later)
**Date.** TBD
**Pre-registered in `project.md`'s cycle 4 next-steps section.**

## Hypothesis

Averaging CB-tuned's OOF across 3 random seeds {42, 777, 99} reduces per-prediction variance enough to lift OOF AUC by ≥ +0.0005 over experiment 014's single-seed result. Seed variation is the cleanest way to reduce model variance without changing the underlying recipe — same data, same FE, same HPs, just different `random_seed` in CatBoost.

## Rationale

Experiment 014 will give us the strongest single-seed (=42) CB-tuned. The remaining variance from random tree-building / Bayesian-bagging-temperature sampling is real but uncorrelated across seeds. Two more seeds at 777 and 99 (the cycle-#006 public-convention seeds and a third unrelated one) give a 3-seed average that should:

- preserve standalone OOF (averaging doesn't degrade rank if seeds are equally good)
- reduce per-fold std → tighter generalization
- improve LB if part of the OOF→LB drift is noise rather than systematic bias

Public 0.95259 single-CatBoost notebook uses a 2-seed (42, 777) average for the same reason. We add seed 99 for more independence.

## Expected magnitude

- Multi-seed CB-tuned standalone OOF: **≥ 0.95135** (= exp14 OOF + 0.0005). Stretch +0.0010.
- 4-way ensemble (LGB=0.05, CB#006=0.20, CB-tuned-multi=0.75): expected to lift over experiment 014's ensemble by similar magnitude.
- LB target after submission: ≥ 0.9515 (~+0.0008 over cycle 3's 0.95066).

## Overfitting risk

**Very low.** Pure variance reduction. The seeds only vary CatBoost's internal randomness (Bayesian bagging temperature draws, ordered TE permutations); they do not change CV splits, data, FE, or HPs. Per-fold std should DECREASE with 3-seed averaging, not increase.

## Kill criteria

- 3-seed-averaged CB-tuned OOF < experiment 014's single-seed CB-tuned → averaging hurt (suggests one of seeds 777/99 is worse than 42, dragging the average down).
- 4-way ensemble OOF ≤ experiment 014's ensemble → no diversity gain.

## Scope

- `src/research/train_cb_tuned_exp15.py` (ready) — clone of `train_cb_tuned_exp14.py` parameterized by `SEED` env var. Outputs `data/oof_cb_tuned_exp15_seed{N}.parquet`.
- `src/research/blend_exp15.py` (ready) — averages exp14's seed-42 OOF + exp15's two new OOFs, then blends with LGB + CB#006.
- `experiments/015_multiseed_cb_tuned.md` — this file.
- No changes to other source files.

## Cost

- 2 extra CB-tuned trainings at iter cap 8000. Each ≈ same time as experiment 014's single fold-5 run.
- Sequential: ~5-6 hours total if exp 14 took ~3 hours.
- Can chain after experiment 014 lands with `SEED=777 ... && SEED=99 ...`. Fire-and-forget overnight.

## Plan

1. ⏳ Wait for experiment 014 to finish.
2. ⏳ `SEED=777 .venv/bin/python -u src/research/train_cb_tuned_exp15.py` (~2-3h).
3. ⏳ `SEED=99 .venv/bin/python -u src/research/train_cb_tuned_exp15.py` (~2-3h).
4. ⏳ `.venv/bin/python src/research/blend_exp15.py` — averages 3 seeds, builds 4-way ensemble.
5. ⏳ Apply gates; document.
6. ⏳ If cumulative cycle-4 lift (exp 014 + 015) clears +0.001 LB-equivalent: submit. Otherwise queue experiment 016 (RealMLP) for more diversity.

## Outcome

Deferred at the time: after exp 014, the multi-seed budget (~6 h) was redirected to the higher-ROI RealMLP build (exp 016), which became the model-family pivot. The multi-seed-CB idea itself was picked up later as **exp 035** (seeds 42 + 7), where it landed Inconclusive — CB-tuned is near-deterministic across seeds (rank-corr 0.997), so multi-seed adds essentially nothing. This file records the original pre-registration.
