# Experiment 035 — Multi-seed CB-tuned (deferred exp 015, finally run)

**Cycle.** 11
**Status.** Inconclusive — 2-seed avg blend lifted only +0.00002 over cycle 7 (well below +0.00020 hurdle). Pipeline correctly skipped seed 99.
**Date.** 2026-05-25

## Hypothesis

Averaging 3 seeds of cycle-14's `train_cb_tuned_exp14.py` (seeds 42, 7, 99) lifts the standalone CB OOF by ≥ +0.00015 over cycle 14's single-seed 0.95114 → target OOF ≥ 0.95129. When this multi-seed CB replaces CB-tuned-exp14 in the cycle 7 blend, the blend OOF lifts by ≥ +0.00020 over 0.95408 → target blend OOF ≥ 0.95428.

## Rationale

Exp 015 was pre-registered for cycle 4 but never ran (RealMLP took precedence). Cycle 5's parallel experiment — averaging 5 seeds of RealMLP — netted +0.00028 OOF over the cycle-4 single-seed. Multi-seed CB is the same mechanism applied to CB-tuned-exp14.

Three reasons the gain might transfer to the blend:
1. **Direct OOF lift:** 3-seed averaging reduces per-fold noise; expected +0.00010 to +0.00020 OOF for CB (less than RealMLP's gain because CB's fold noise is already lower).
2. **Optimal w_cb shifts:** a less-noisy CB input means the blend optimizer can place more weight on it without amplifying noise; w_cb_opt may shift from 0.20 → 0.25.
3. **Rank-correlation unchanged:** averaging preserves rank ordering; rank-corr with RealMLP stays ~0.976 (no diversity loss).

CV protocol bit-identical to cycle 14: `StratifiedKFold(5, shuffle=True, random_state=42)` on `Year × PitNextLap`. Only the CatBoost `random_seed` differs per seed.

## Expected magnitude

- **Standalone 3-seed OOF target:** ≥ 0.95129 (+0.00015 over cycle 14).
- **Optimistic:** ≥ 0.95150 (+0.00036).
- **Blend OOF target (vs RealMLP-multiseed at best w_cb):** ≥ 0.95428.
- **Floor:** standalone OOF lift < +0.00005 → averaging is noise; close cycle 11 Inconclusive.

## Overfitting risk

**Very low.** No HP search, no FE change, no CV change. Three independent training runs at fixed HPs, averaged. Mathematically equivalent to ensemble variance reduction with no leakage surface introduced.

## Kill criteria

- [ ] Seed 7 standalone OOF < 0.95080 (no AUC at all — recipe is broken at non-42 seeds).
- [ ] Per-fold std (across 5 folds) > 0.00080 (instability).
- [ ] 2-seed blend OOF < 0.95420 (seeds 42+7 combined adds nothing → seed 99 not worth running).
- [ ] 3-seed blend OOF < 0.95428 → standalone hurdle missed → no submission.

## Scope

- `src/research/train_cb_multiseed.py` (new — CLI `--seed N`, otherwise bit-identical to `src/research/train_cb_tuned_exp14.py`).
- `src/research/combine_cb_seeds.py` (new — averages OOFs and submissions across seeds into multi-seed outputs).
- `src/research/blend_realmlp_cb_multiseed.py` (new — linear + rank blend sweep, conditional submission write).
- Outputs:
  - `data/oof_cb_tuned_seed{7,99}.parquet` and matching submission CSVs.
  - `data/oof_cb_multiseed.parquet` and `data/submission_cb_multiseed.csv` (combined).
  - `data/blend_cb_multiseed_sweep.parquet` and conditionally `data/submission_blend_multiseed_best.csv`.

Wall-clock budget:
- Seed 7: ~3.25 h (started 22:16 CEST, ETA ~01:30 CEST). nohup → survives session end.
- Combine + blend probe: ~5 min (post-hoc on cached OOFs).
- Seed 99 (conditional on 2-seed blend clearing 0.95420): another ~3.25 h.

## Reversibility check

CV / seed (for folds) / target / leakage surface: **all unchanged**. The only thing that varies is CatBoost's `random_seed` per training run.

## Plan

1. Run seed 7 (5 folds, ~3.25h). [In progress]
2. Combine seed 42 (existing cycle 14 OOF) + seed 7 → 2-seed OOF + submission.
3. Run blend probe (RealMLP × 2-seed CB at w_cb ∈ [0.05, 0.30] linear+rank).
4. Decision gate:
   - If 2-seed blend OOF ≥ 0.95428 → submit, close cycle 11.
   - If 2-seed blend OOF ∈ [0.95420, 0.95428) → run seed 99, retry blend probe with 3-seed CB.
   - If 2-seed blend OOF < 0.95420 → cycle 11 Inconclusive, accept cycle 7 as final.

## Result

### Seed 7 standalone

5-fold per-fold AUCs:

| Fold | AUC      | iters | runtime |
| ---- | -------- | ----- | ------- |
|  1   | 0.95164  | 7997  | 39 min  |
|  2   | 0.95199  | 7996  | 37 min  |
|  3   | 0.95077  | 7991  | 35 min  |
|  4   | 0.95036  | 7621  | 36 min  |
|  5   | 0.95119  | 7962  | 36 min  |

**Seed 7 full OOF AUC = 0.95118** (vs cycle 14 seed-42's 0.95114, Δ +0.00004).
Per-fold std: 0.00060 (vs cycle 14's typical 0.00042 — slightly noisier seed).

### 2-seed combine (42 + 7)

| OOF | AUC |
| --- | --- |
| seed 42 alone (cycle 14) | 0.95114 |
| seed 7 alone | 0.95118 |
| **2-seed average** | **~0.95116** |

The 2-seed average is **essentially tied** with each individual seed — no meaningful variance reduction visible at this level.

### 2-seed blend probe vs RealMLP-multiseed

Sweep `w_cb ∈ [0.00, 0.40]` linear + rank:

| Operator | Best w_cb | Blend AUC |
| -------- | --------- | --------- |
| Linear   | 0.200     | **0.95410** |
| Rank     | 0.225     | 0.95409     |

- **vs cycle 7's 0.95408: Δ +0.00002** — at the noise floor, indistinguishable.
- **vs hurdle 0.95428: −0.00018** — well below.

### Diagnostic: rank-correlation between seeds

| Pair | Spearman ρ |
| ---- | ---------- |
| RealMLP-multiseed vs seed 7 | 0.9759 |
| CB-tuned-exp14 (seed 42) vs seed 7 | **0.9971** |

Seed 7's predictions are **nearly identical** to seed 42's at the rank level (ρ=0.9971). This is why variance reduction failed: the two seeds learn essentially the same model. The cycle-14 recipe is highly deterministic with respect to the `random_seed` HP for this dataset.

## Verdict

**Inconclusive.** Multi-seed CB doesn't add meaningful diversity at this recipe. The CV protocol (fixed CV_SEED) plus the same data + HPs produces near-identical models across random seeds. The variance reduction mechanism that worked for cycle 5's RealMLP (Δ +0.00028 from 1→5 seeds) does not transfer to CatBoost because CB's ordered-TE + Bayesian bootstrap appears to dominate the seed effect.

The overnight pipeline correctly determined the 2-seed result was clear (0.95410, below the [0.95420, 0.95430) marginal zone) and skipped seed 99 — saving ~3.25 hours of pointless compute.

## Kill-criteria check

- [ ] Seed 7 standalone OOF < 0.95080 — not fired (0.95118).
- [x] Per-fold std (across 5 folds) > 0.00080 — not fired (0.00060).
- [x] 2-seed blend OOF < 0.95420 — **FIRED** (0.95410 < 0.95420).
- [ ] 3-seed blend OOF < 0.95428 — not evaluated (seed 99 not run).

## Repro stamp

- data: train.csv sha256 `f004e79d…`
- packages: catboost 1.2.10, sklearn 1.8.0, pandas 3.0.2, numpy 2.4.4
- seed 7 runtime: 5 folds × ~36 min ≈ 180 min total CPU on M1 Pro (run overnight, combined with seed 42 the next morning)

## Learnings

1. **CatBoost's `random_seed` has minimal effect on rank ordering for this recipe.** ρ=0.9971 between seed 42 and seed 7 means the random-seed-driven variance is ~1% of the rank scale. Most of the inter-seed variation is in absolute probability values, not ordering. AUC depends only on ordering, so it's near-invariant.
2. **Cycle 5's RealMLP-multiseed result doesn't generalize to other model families.** RealMLP (a neural network) has stochasticity in init + minibatch + dropout that produces rank-different predictions across seeds. CatBoost (tree boosting with ordered TE) does not.
3. **The overnight auto-pipeline worked exactly as designed.** Detected seed 7 outputs within 30s of write, ran combine + blend probe in <30s, applied the decision-tree thresholds correctly (avoiding wasted seed-99 compute), and produced a clean verdict report. The adaptive design (only run seed 99 if 2-seed is marginal) saved ~3.25h.
4. **The +0.00020 hurdle is rigorous.** A naive Δ +0.00002 would have looked like "real improvement" without the pre-registered min_delta. Pre-registration of the kill criteria saved a marginal-but-noise submission.

## Follow-ups

- **Confirmed dead axis:** multi-seed averaging on CatBoost recipes is unproductive. Future cycles should test multi-seed on RealMLP (cycle 5 path) or on stochastic boosters like XGBoost.
- **Exp 034 (XGBoost) running now** as the next attempt — completely different model family, real diversity potential.
