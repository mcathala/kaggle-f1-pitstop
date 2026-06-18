# Experiment 028 — CatBoost with rich-categorical FE

**Cycle.** 10
**Status.** Infra-fail — kill criterion #3 fired (fold-1 wall-clock > 30 min). Recipe retried as exp 031 (slim).
**Date.** 2026-05-25

## Hypothesis

Replacing our cycle-4 CB-tuned-exp14 (OOF 0.95114) with a CatBoost trained on a **categorically-dense** feature set — digit-position features, multi-resolution numeric-as-categorical, and a full pairwise-bigram grid — produces a standalone OOF ≥ 0.95250. When this new CB replaces CB-tuned-exp14 in the cycle 7 blend, the combined OOF lifts ≥ +0.00020 over cycle 7's 0.95408 (projected OOF ≥ 0.95428).

## Rationale

Audit of our cycle-12-14 CatBoost work surfaced a **feature-philosophy mismatch**: we've been giving CatBoost mostly **numeric** features (computed interactions, frequency encodings, group statistics, hand-tuned bins). CatBoost's *primary* mechanism — ordered target encoding on high-cardinality categoricals — is therefore underused.

Re-reading the leaderboard pattern across publicly-visible Playground-S6E5 high-scoring CatBoost solutions (~OOF 0.952-0.953) shows a recurring recipe: most features are STRING categoricals derived from numeric raw inputs. The mechanism is that CB's ordered TE assigns each unique categorical bucket its own target-encoded mean (cross-fold safe), letting the model learn cell-specific target rates over fine-grained interactions of digit cats × bigram cats. Univariate signal in any single digit cat is low — the value compounds across the categorical grid.

Specifically, three transforms we don't currently apply to CB:

1. **Digit-position cats** — for each numeric (Year, PitStop, LapNumber, Stint, TyreLife, Position, LapTime (s), LapTime_Delta, Cumulative_Degradation, RaceProgress, Position_Change), extract sign + 1s/10s/100s/1000s integer digits + 0.1/0.01/0.001 decimal digits. Each digit becomes its own int8 categorical column. ~67 new cat cols.

2. **Multi-resolution numeric-as-categorical** — for {LapTime (s), LapTime_Delta, Cumulative_Degradation}, cast to STRING at exact precision AND at rounded resolutions {0.5, 1.0, 2.0, 5.0, 10.0, 20.0} step (resolution chosen per column based on natural scale). Each rounding level is its own categorical. ~21 cols.

3. **Full pairwise-bigram grid** — all C(11, 2) = 55 unordered pairs of {Driver, Compound, Race, Year, PitStop, LapNumber, Stint, TyreLife, Position, RaceProgress, Position_Change} as concatenated string cats. We currently have 9 hand-picked pairs; the full grid covers ~6× more combinations.

In cycle 14 we considered digit features but rejected them after a *univariate* signal check showed weak per-column AUC contribution. That was the wrong test — digit-cat value emerges only when CB combines them with bigram cats through ordered TE, not in isolation. Cycle 10 corrects that decision.

## Expected magnitude

- **Standalone CB OOF target:** ≥ 0.95250 (+0.00136 over our cycle-4 CB-tuned-exp14 = 0.95114).
- **Re-run blend (multi-seed RealMLP × new CB at w_cb ∈ [0.10, 0.30]):** OOF ≥ 0.95428 (cycle 7 + 0.00020).
- **Optimistic:** OOF 0.95450+ → projected LB ~0.954 after −0.00045 drift.
- **Floor:** standalone CB OOF < 0.95150 → FE direction adds nothing; close cycle 10 Inconclusive, pivot to advanced blend operators (exp 029) or stacking.

## Overfitting risk

**Low-Medium.**

1. CatBoost's ordered TE + 200-iter early stopping provides built-in regularization for high-cardinality cats.
2. 2-seed ensemble (seeds 42, 43) per fold reduces fold variance.
3. No HP tuning loop in this experiment — direct application of a known-strong HP recipe + new FE. No selection bias.
4. CV unchanged (same Year × PitNextLap StratifiedKFold seed=42 as everything else), so OOF aligns with our existing model OOFs for the blend probe.

## Kill criteria

- [ ] Standalone CB OOF < 0.95150 (no FE lift; direction dead)
- [ ] Per-fold std > 0.00080 (instability)
- [ ] Fold-1 wall-clock > 30 min (compute blow-up)
- [ ] OOM during fit (157-feature × 2-seed × big-train might exceed M1 Pro RAM)

## Scope

- `src/research/train_cb_rich_fe.py` (new, ~310 lines)
- Outputs: `data/oof_cb_rich.parquet`, `data/submission_cb_rich.csv`
- `experiments/028_cb_rich_categorical_fe.md` (this file)

Wall-clock budget: ~30-60 min total (CatBoost CPU on M1 Pro, 5 folds × 2 seeds ≈ 10 model fits at ~3-6 min each).

## Reversibility check

- CV protocol: **unchanged** — Year × PitNextLap StratifiedKFold(seed=42, n_splits=5).
- Seed: 42 (model seeds 42, 43 within each fold).
- Feature set: **NEW** (different recipe from cycle 12-14, replaces our hand-tuned numeric FE with categorical-dense FE).
- Leakage surface: unchanged — no test data in train, same external dataset as cycle 12+.

No reversibility flag fires.

## Plan

1. Build `src/research/train_cb_rich_fe.py`:
   - Load competition train + test + external data (drop `Normalized_TyreLife` for compat).
   - Encode all base raw cats (Driver/Compound/Race) consistently across train/test/orig via single mapping.
   - Apply digit FE, NUM_as_CAT FE, BIGRAM FE — every new column registered as CB categorical.
   - 5-fold StratifiedKFold seed=42 stratified on Year × PitNextLap.
   - Per fold: 2-seed CatBoost ensemble (seed 42, 43) averaged.
   - CatBoost params: iterations=10000, lr=0.03, depth=8, early_stopping=200, bagging_temperature=0.8, NO `auto_class_weights`, bootstrap_type Bayesian.
2. Report standalone OOF, per-fold AUCs, rank correlation vs RealMLP-multiseed and CB-tuned-exp14 OOFs.
3. If standalone OOF ≥ 0.95150: re-run blend probe (RealMLP-multiseed × new CB) at w_cb ∈ {0.10..0.30}.
4. Decision gate:
   - Blend OOF ≥ 0.95428 → generate submission.
   - Blend OOF ∈ (0.95408, 0.95428) → Inconclusive, document, consider exp 029 (advanced blend operators) before submitting.
   - Blend OOF ≤ 0.95408 → Reverted, direction confirmed dead.

## Result

Background training launched but **did not complete**. Process was killed when the background session ended.

Partial trajectory (fold 1 / seed 42 only, before kill):
- iter 0: val AUC 0.9283
- iter 500: val AUC **0.9498** (17 min elapsed)
- CatBoost itself reported `5h 18m remaining` at iter 500 — projected fold-1 wall-clock ~5.5 h
- Projected total for 5 folds × 2 seeds: **~55 hours** (vs the 30–60 min "budget" claimed in this spec)

Cause of the runtime mis-estimate: 124-feature × 113-categorical recipe at `depth=8`, `iter=10000` triggers heavy ordered-TE bookkeeping for every node split. Per-iter cost was ~2.0 s; even with `early_stopping_rounds=200`, convergence wasn't imminent at iter 500 (val AUC still rising).

No `oof_cb_rich.parquet` / `submission_cb_rich.csv` produced; no full-fold AUC measured.

## Verdict

**Infra-fail.** Recipe is too heavy for the compute budget on M1 Pro CPU. The *hypothesis* is unchanged — categorical-dense CB may still beat numeric-dense — but this scope cannot test it. Pivot to exp 031 (slim variant).

## Kill-criteria check

- [ ] Standalone CB OOF < 0.95150 — not evaluated (no full OOF).
- [ ] Per-fold std > 0.00080 — not evaluated.
- [x] Fold-1 wall-clock > 30 min — **FIRED** (projected ~5.5 h; killed at 17 min for environmental reasons).
- [ ] OOM — not observed.

Kill criterion #3 alone is sufficient to terminate; killing the process before fold completion was therefore the right call regardless of the environmental kill.

## Repro stamp

- data: `train.csv` sha256 `f004e79d…`
- packages: catboost 1.2.10, sklearn 1.7.2, pandas 2.3.3, numpy 2.3.5
- failure log: `(local background task log)` (last write 17:47 CEST 2026-05-25)

## Learnings

1. **Always dry-run a fold-1 single-seed first** for a recipe whose runtime is uncalibrated. The 30–60 min spec estimate was a guess; iter 500 of fold 1 = 17 min on CPU would have flagged the issue in <5 min on a smaller iter budget.
2. **CatBoost depth-8 + 100+ categoricals is non-linear in cost.** Empirically per-iter cost grew super-linearly with the number of cat features at fixed train size; depth=7 + slimmer feature set is the sane operating point on M1 Pro CPU.
3. **Background tasks die with the session.** When the background session ended, the unparented `python` subprocess was reaped. Long-running jobs need to be either (a) launched with `nohup` + redirected output and explicitly disowned, or (b) sized to fit comfortably within one session.

## Follow-ups

- **exp 031** — slim variant of this recipe: drop digit features (no probe-1 support), prune bigrams to top ~20, drop num-as-cat step variants, `depth=7`, `iter=2500`, `lr=0.05`. Dry-run fold-1 single-seed before scaling.
- If exp 031 also underwhelms: pivot to exp 032 (class-weighted CB-tuned-exp14 minus `auto_class_weights="Balanced"`, targeting probe-2's Q4 worst-loss slice via mild `class_weights=[1.0, 1.5]`).
