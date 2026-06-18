# Experiment 031 — CB-rich-FE slim (retry of exp 028 with tractable HPs)

**Cycle.** 10
**Status.** Inconclusive (dry-run only — fold-1 wall-clock kill fired; scale-up cost too high to justify the marginal expected blend gain).
**Date.** 2026-05-25

## Hypothesis

Exp 028's hypothesis — that a categorical-dense CatBoost recipe (4 raw cats + ~55 bigrams + light num-as-cat) outperforms cycle 14's CB-tuned-exp14 (mostly-numeric, OOF 0.95114) — is testable within compute budget once `depth`, `lr`, and feature count are sized to M1 Pro CPU. Specifically: a slim variant achieves **standalone OOF ≥ 0.95150** with per-fold wall-clock ≤ 20 min. When blended at `w_cb ∈ [0.10, 0.30]` against the cycle-5 RealMLP multi-seed, the result lifts ≥ +0.00020 OOF over cycle 7's 0.95408.

## Rationale

Exp 028 went infra-fail per its own kill criterion #3 (fold-1 wall-clock > 30 min; CatBoost projected 5.5 h/fold). The recipe was sound *but unsized* — `depth=8`, `iter=10000`, 113 categoricals together triggered O(2^d × n_cat) ordered-TE work per node split that compounded into ~55 h projected total.

Phase-1 diagnostics (exp 030) gave us evidence to make smart cuts without losing the hypothesis:

1. **Drop the digit-position features.** Probe 1's family-importance ranking gave per-feature mean: `cross_cat` 2.16 / `bin_cat` 0.62 / `freq_encoding` 0.16. Digit features fall structurally in the freq/encoding tier — high cardinality, low joint signal. They cost feature-matrix budget that bigrams use better.
2. **Reduce `depth` 8 → 7.** Halves leaves per tree (256 → 128). CatBoost's symmetric-oblivious-tree time scales as `O(2^d × n_features × n_cat)`; this is the single biggest knob.
3. **Raise `lr` 0.03 → 0.06.** With `iter=2000` cap and `early_stopping=150`, expected convergence at iter ~1000–1500 (vs 5000+ at lr=0.03). Equivalent regularization given the iter cap.
4. **Slim num-as-cat from 23 cols → 6 cols.** Keep exact + 1-decimal cats for the 3 high-importance numerics (LapTime(s), LapTime_Delta, Cumulative_Degradation); drop the step-rounded variants. Bigrams already capture the multi-resolution effect.
5. **Keep all 55 bigrams.** Probe 1 showed cross_cat dominance on a 9-bigram slice; expanding to the full C(11,2) grid is the philosophical pivot we're testing. CatBoost's ordered TE handles 55 cats without overfitting if `depth ≤ 7`.
6. **Drop `auto_class_weights="Balanced"`.** Probe 5 attributes the cycle-7 bin-8 over-prediction (+0.057) to this; AUC is rank-only so removing it doesn't cost ranking, and may slightly improve calibration for downstream blends.

Net feature count: **79** (vs 124 in 028, 132 in cycle 14). 14 raw nums + 4 raw cats + 6 num-as-cat + 55 bigrams = 79; 65 of those are CB categoricals.

Estimated runtime: per-iter cost scales roughly as `depth × n_cat` per fold; 28% of exp 028's per-iter cost (`7/8 × 65/113 ≈ 0.50`). With `iter=2000` cap and ~1500 actual convergence at lr=0.06, **expected fold-1 single-seed wall-clock ~12–20 min**.

## Expected magnitude

- **Standalone CB OOF target:** ≥ 0.95150 (+0.00036 over CB-tuned-exp14).
- **Stretch:** ≥ 0.95250 (matches exp 028's stated target).
- **Blend (RealMLP-multiseed × new CB at best w_cb):** ≥ 0.95428 (cycle 7 + 0.00020).
- **Floor:** standalone CB OOF < 0.95080 → categorical-dense direction confirmed dead for this dataset; close cycle 10 Inconclusive.

## Overfitting risk

**Low.**

1. CatBoost's ordered TE is cross-fold-safe by construction; 55 high-cardinality bigrams don't leak.
2. Reduced `depth=7` is more regularizing than 028's 8.
3. 2-seed averaging per fold (in the scale-up phase) reduces fold variance.
4. CV unchanged from cycle 14 — same `StratifiedKFold(5, shuffle=True, random_state=42)` on `Year × PitNextLap`.
5. No HP search, no FE search — direct application of a diagnostics-informed recipe.

## Kill criteria

- [ ] **Fold-1 dry-run wall-clock > 30 min** — if it fires, abandon the categorical-dense direction; pivot to exp 032 (class-weighted CB-tuned-exp14 minus Balanced, targeting probe-2's Q4 positives).
- [ ] **Fold-1 dry-run AUC < 0.95080** — standalone signal is below cycle-14's CB-tuned-exp14; FE pivot adds nothing meaningful. Pivot to exp 032.
- [ ] **Fold-1 rank-corr with RealMLP-multiseed (on fold-1 val rows) > 0.97** — new CB is too similar to RealMLP; blend has no diversity to extract. Pivot.
- [ ] **Full-OOF standalone AUC < 0.95150 (after scale-up)** — FE direction adds nothing meaningful at full scale; mark cycle 10 Inconclusive.
- [ ] Per-fold std > 0.00080 — instability; reconsider.

## Scope

- `src/research/train_cb_rich_fe_slim.py` (new, ~250 lines — adapted from `src/research/train_cb_rich_fe.py` with `--fold`, `--seeds`, `--iters`, `--depth`, `--lr` CLI flags).
- Outputs: `data/oof_cb_rich_slim.parquet`, `data/submission_cb_rich_slim.csv`.
- `experiments/031_cb_rich_slim.md` (this file).

Wall-clock budget:
- **Dry-run** (fold 1, seed 42): ~12–20 min.
- **Scale-up** (5-fold × 2-seed): ~120–180 min if dry-run was on the low end; ~150–250 if on the high end.

## Reversibility check

- CV protocol: **unchanged** — `StratifiedKFold(5, shuffle=True, random_state=42)` on `Year × PitNextLap`.
- Seed: 42 (model seeds 42, 43 within each fold during scale-up).
- Feature set: **NEW** (slim variant of 028's recipe; same philosophical direction).
- Leakage surface: unchanged — same external data as cycle 12+.

No reversibility flag fires.

## Plan

### Phase A — dry-run (fold 1, single seed, ~12–20 min)

1. `.venv/bin/python -u src/research/train_cb_rich_fe_slim.py --fold 1 --seeds 42 --iters 2000 --depth 7 --lr 0.06`.
2. Capture: per-iter trajectory, fold-1 AUC at best_iter, wall-clock, rank-corr-with-RealMLP-multiseed on the fold-1 val subset.
3. Check all four dry-run kill criteria.

### Phase B — decision gate

- If dry-run passes all four → **scale up** (Phase C).
- If any kill fires → **pivot to exp 032** (class-weighted CB on cycle-14 recipe, targeting probe-2's Q4 worst-loss slice).

### Phase C — scale-up (5-fold × 2-seed, ~120–180 min)

1. `.venv/bin/python -u src/research/train_cb_rich_fe_slim.py --all-folds --seeds 42,43 --iters 2000 --depth 7 --lr 0.06`.
2. Report standalone OOF, per-fold AUCs, rank-corr vs RealMLP-multiseed and CB-tuned-exp14.
3. If standalone OOF ≥ 0.95150 → blend probe (Phase D).

### Phase D — blend probe (post-OOF, ~2 min)

1. Sweep `w_cb ∈ {0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.30}` linearly against RealMLP-multiseed.
2. If best blend OOF ≥ 0.95428 → submit `data/submission_blend_realmlp_cb_rich_slim_best.csv`.

## Result

### Phase A — fold-1 dry-run (single seed)

`.venv/bin/python -u src/research/train_cb_rich_fe_slim.py --fold 1 --seeds 42 --iters 2000 --depth 7 --lr 0.06 --early-stop 150`

Per-iter trajectory (val AUC on fold-1, 87,828 rows):

| iter | val AUC | Δ vs prev | elapsed |
| ---- | ------- | --------- | ------- |
|    0 | 0.92435 |    —      |   1.3 s |
|  250 | 0.94801 | +0.02366  |   6m 28s |
|  500 | 0.94950 | +0.00149  |  11m 32s |
|  750 | 0.95012 | +0.00062  |  17m 18s |
| 1000 | 0.95054 | +0.00042  |  22m 28s |
| 1250 | 0.95081 | +0.00027  |  27m 49s |
| 1500 | 0.95098 | +0.00017  |  33m 13s |
| 1750 | 0.95110 | +0.00012  |  38m 52s |
| 1999 | 0.95124 | +0.00014  |  44m 24s |

- **fold-1 ensemble AUC = 0.95124**, best_iter=1999 (iter cap reached; early-stop=150 never fired).
- **Wall-clock = 2669 s ≈ 44 min** (over the 30-min spec budget).
- **Rank-corr with RealMLP-multiseed on fold-1 val = 0.96555** (vs cycle-7 blend's 0.9758 for CB-tuned-exp14).

### Phase B — decision gate

| Kill criterion                  | Threshold     | Observed   | Status   |
| ------------------------------- | ------------- | ---------- | -------- |
| Fold-1 wall-clock               | ≤ 30 min       | 44 min     | **FAIL** |
| Fold-1 AUC                      | ≥ 0.95080      | 0.95124    | Pass     |
| Rank-corr with RealMLP          | < 0.97         | 0.96555    | Pass     |

Phase C (scale-up) and Phase D (blend probe) **NOT EXECUTED.**

### Scale-up cost-benefit (what we didn't run)

- **single-seed × 5-fold:** 5 × 44 min ≈ 3.7 h (long; session-kill risk is real after the exp-028 dead-process incident on the same branch).
- **2-seed × 5-fold:** 10 × 44 min ≈ 7.3 h (infeasible for one session).
- **Expected blend OOF (rough projection):** rank-corr 0.965 (vs cycle-7's 0.976) suggests ~1.5× more variance reduction at the blend; cycle-7's blend gain over RealMLP-standalone was +0.00025 → exp-031's projected blend gain over RealMLP-standalone ≈ +0.00037 → projected blend OOF ≈ **0.95420**, which is **below** the +0.00020-over-cycle-7 hurdle of 0.95428.

Given (a) the wall-clock kill, (b) the projected blend OOF straddles-but-likely-misses the hurdle, and (c) the standalone OOF is only marginally above the 0.95080 kill-floor and well below the 0.95150 pass-gate target, scale-up is **not the cost-justified next step.** Pivot to exp 032.

## Verdict

**Inconclusive (dry-run only).** The slim recipe is *runnable* on M1 Pro CPU (vs exp 028's infeasibility) but the expected return at full scale does not clear the cycle's +0.00020 hurdle. The categorical-dense FE direction is not adding value beyond cycle 14's mostly-numeric recipe at depth-7 / iter-2000 / lr-0.06 on this dataset. Direction not killed (single-fold AUC is *almost* matching cycle 14's 0.95114), but the expected blend gain is too thin to justify 3.7+ hours of compute.

## Kill-criteria check

- [x] Fold-1 dry-run wall-clock > 30 min — **FIRED** (44 min; 47% over budget).
- [ ] Fold-1 dry-run AUC < 0.95080 — not fired (0.95124).
- [ ] Fold-1 rank-corr with RealMLP > 0.97 — not fired (0.96555).
- [ ] Full-OOF standalone AUC < 0.95150 (post-scale-up) — not evaluated (scale-up skipped).
- [ ] Per-fold std > 0.00080 — not evaluated.

One of three pre-Phase-C criteria fired; combined with the marginal expected blend gain, this is sufficient to short-circuit the scale-up.

## Repro stamp

- data: `train.csv` sha256 `f004e79d…`
- packages: catboost 1.2.10, sklearn 1.7.2, pandas 2.3.3, numpy 2.3.5
- runtime: 44 min CPU on M1 Pro (single fold, single seed)
- output log: `(local background task log)`

## Learnings

1. **Slim recipe is computationally feasible but not philosophically transformative.** Cutting digits + step-rounded num-as-cat + dropping depth 8→7 + lr 0.03→0.06 reduces per-fold runtime from ~5.5 h to ~44 min — enough to run, not enough to win. The cat-dense direction's predictive gain at this dataset size is bounded by the *information overlap* with cycle 14's 132-numeric recipe, which is high.
2. **Early stopping at 150 never fired** because the val AUC was still rising at iter 2000. The recipe wants ~3000–4000 iterations to fully converge, which compounds the wall-clock problem — extending the iter cap moves us from 44 → ~88 min per fold.
3. **Rank-corr 0.96555 is a real but small diversity signal.** Cycle 7's 2-input blend (corr 0.976) gained +0.00025 over RealMLP-standalone. Exp 031's blend (corr ~0.965) would gain maybe +0.00035–0.00045 — below the +0.00045 needed to clear the cycle's hurdle. The structural ceiling on a 2-input linear blend is set by the *minimum-rank-corr you can achieve while keeping standalone AUC near the baseline*, and we're hitting it.
4. **The infra cost of dry-run-before-scale-up is its own value.** A 44-min fold-1 single-seed run informed a confident no-scale-up decision; the alternative (4-7 hour blind scale-up) would have produced the same conclusion at 4-7× the cost.

## Follow-ups

- **exp 032 (next):** drop `auto_class_weights="Balanced"` AND apply mild `class_weights=[1.0, 1.5]` to a CB on cycle-14's existing 132-feature recipe. Targets probe-2's Q4 worst-loss positive concentration (46.5% positives vs 19.9% overall) via a CB mechanism (loss-weighted boosting) rather than an FE mechanism. Expected runtime ~25-35 min on the existing trainer.
- If exp 032 also fails to clear the hurdle: close cycle 10 Inconclusive with three null results (028 infra-fail, 029 inconclusive, 031 inconclusive) + one diagnostic Kept (030). Ship the diagnostic findings; defer the cat-dense direction until a faster CB substrate (GPU, or LightGBM/XGBoost cat handling) is available.
