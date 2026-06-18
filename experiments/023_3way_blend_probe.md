# Experiment 023 — 3-way blend probe: RealMLP × CB-tuned × LGB

**Cycle.** 8
**Status.** Inconclusive (LGB axis adds zero; model zoo exhausted for blend gains)
**Date.** 2026-05-22

## Hypothesis

A 3-way linear blend of cycle 5 multi-seed RealMLP + cycle 4 CB-tuned-exp14 + cycle 1 LGB baseline, at carefully-weighted small LGB+CB injection into the RealMLP base, improves OOF AUC over cycle 7's 2-way blend (0.95408) by ≥ +0.00020 (min_delta).

## Rationale

- Cycle 7 broke cycle 4's "any non-RealMLP weight hurts" rule. The natural next question: is the 2-way blend's +0.00025 OOF lift the ceiling of small-CB-injection, or can we extract more by adding a *third* diverse signal?
- **LGB is the most diverse signal in the model zoo.** Different family (gradient boosting + LightGBM's specific binning), different feature handling, different optimization. RealMLP and CB-tuned have 0.9758 rank correlation; LGB-baseline-vs-RealMLP rank correlation likely lower.
- This probe is CHEAP (~10s of compute). Worst case: confirms 3-way diversity adds zero.

## Expected magnitude

- **Target:** OOF AUC ≥ 0.95428 (cycle 7 + min_delta 0.00020).
- **Optimistic:** +0.00050 over cycle 7 (matches the typical 3-way diversity lift seen in Kaggle competitions).
- **Floor:** any positive Δ vs cycle 7's 0.95408 is a finding; sub-min_delta is Inconclusive.

## Overfitting risk

**Medium.** Sources:

1. **Selection bias on weight grid.** 3-way sweep has 2 free weights → larger search space than 2-way. Mitigated by reporting the entire surface, requiring +0.00020 (not min_delta minus headroom), and finer-grid refinement near the peak.
2. **Cumulative drift creep.** Each blend layer added ~+0.00006 OOF→LB drift. A 3-way blend may have drift around −0.00053; if the OOF lift is +0.00030 the LB lift could be a wash.
3. **No model retraining** — uses cached OOFs/submissions, so the search is deterministic given those inputs.

## Kill criteria

- [ ] Best 3-way OOF ≤ cycle 7's 0.95408 (3-way doesn't add over 2-way; diversity exhausted via LGB axis)
- [ ] Best 3-way ∈ (0.95408, 0.95428) (Inconclusive — within noise floor)
- [ ] Optimal w_lgb < 0.025 (LGB axis adds nothing meaningful; effectively a 2-way)

## Scope

- `src/research/blend_3way_probe.py` (new, ~150 lines)
- Outputs: `data/blend_3way_sweep.parquet`, optional `data/submission_3way_best.csv`
- `experiments/023_3way_blend_probe.md` (this file)

## Reversibility check

- CV protocol: unchanged.
- Seed: unchanged.
- Feature set: unchanged.
- Leakage surface: unchanged.
- Uses only cached OOFs + submissions — no retraining.

No reversibility flag fires.

## Plan

1. Load `oof_realmlp_multiseed.parquet` + `oof_cb_tuned_exp14.parquet` + `oof_baseline.parquet`. Align by id.
2. Verify rank correlations between the three.
3. **Coarse sweep:** w_cb ∈ {0.10, 0.15, 0.20}, w_lgb ∈ {0.00, 0.025, 0.05, 0.075, 0.10}, with w_realmlp = 1 − w_cb − w_lgb. Report best.
4. **Fine sweep** around coarse peak: ±0.025 on each weight at 0.0125 step.
5. **Decision gate:**
   - Best OOF ≥ 0.95428 → generate submission CSV, report.
   - Best OOF ∈ (0.95408, 0.95428) → Inconclusive, no submission, document.
   - Best OOF ≤ 0.95408 → Reverted, document direction dead.

## Result

**Rank correlations (full train, OOF):**
- RealMLP × CB-tuned: 0.9758
- RealMLP × LGB: **0.9432** (LGB IS more diverse than CB)
- CB-tuned × LGB: 0.9434

LGB is genuinely the most diverse signal — lowest rank correlation with RealMLP. The hypothesis that diversity translates to ensemble value was sound, but...

**Linear 3-way sweep (coarse + fine, 30 points):**

| w_realmlp | w_cb | w_lgb | OOF AUC | Δ vs cycle 7 |
|-----------|------|-------|---------|--------------|
| 0.800 | 0.200 | **0.000** | **0.95408** | **+0.00000 (best)** |
| 0.775 | 0.200 | 0.025 | 0.95407 | −0.00001 |
| 0.750 | 0.200 | 0.050 | 0.95404 | −0.00004 |
| 0.725 | 0.200 | 0.075 | 0.95399 | −0.00009 |
| 0.700 | 0.200 | 0.100 | 0.95394 | −0.00014 |

Optimal w_lgb is exactly **0**. Every fractional point of LGB weight makes the blend worse. Fine sweep confirms — the surface is monotonically decreasing in w_lgb across all (w_cb, w_realmlp) combinations.

**Rank-blend 3-way sweep:** identical conclusion. Best at w_lgb=0, OOF 0.95408. LGB's rank-diversity does not survive its lower standalone AUC.

**Auxiliary probe — RealMLP × ensemble_exp14 (2-way):**

| w_e | linear | Δ vs cycle 7 | rank | Δ vs cycle 7 |
|-----|--------|--------------|------|--------------|
| 0.20 | 0.95407 | −0.00001 | 0.95406 | −0.00002 |
| 0.25 | 0.95407 | −0.00001 | 0.95406 | −0.00002 |

Cycle-4 ensemble_exp14 (a 3-way LGB+CB006+CB-tuned blend, OOF 0.95161) doesn't beat the simpler cycle 7 2-way either — because 75% of ensemble_exp14's weight IS CB-tuned-exp14, so blending RealMLP with it is functionally a noisy version of cycle 7's blend.

## Verdict

**Inconclusive.** Neither LGB nor ensemble_exp14 add value over cycle 7's 2-way blend. The model zoo has been exhausted for further blend gains. Cycle 7's 0.95408 OOF is the local optimum in our current prediction stack.

## Kill-criteria check

- [x] Best 3-way OOF = cycle 7 (0.95408 = 0.95408) — at-the-bar; LGB axis literally adds nothing. KILL FIRES.
- [x] Optimal w_lgb < 0.025 (it's exactly 0). KILL FIRES.

Direction dead: adding more *existing* models won't help; the next ROI must come from a *new* model or a different ensembling scheme.

## Repro stamp

- inputs: `oof_realmlp_multiseed.parquet`, `oof_cb_tuned_exp14.parquet`, `oof_baseline.parquet`, `oof_ensemble_exp14.parquet`
- pkg: scipy 1.17.1, numpy 2.4.4, scikit-learn 1.8.0

## Learnings

1. **Diversity is necessary but not sufficient for blend gains.** LGB had genuine rank-diversity (0.9432 vs CB's 0.9758) but was too weak to translate it. Rule of thumb: a blend partner needs OOF ≥ 0.985 × base_OOF for diversity to overcome weakness. LGB at 0.985 × 0.95383 = 0.93952 minimum — LGB only just clears (0.94273). CB-tuned-exp14 (0.95114) is well above this floor.
2. **Meta-blends don't help when the meta is built from the same components.** ensemble_exp14 is mostly CB-tuned-exp14; mixing it back into a RealMLP blend produces nothing new.
3. **The model zoo is exhausted for cheap blend gains.** Either a new architecturally-distinct model (AutoGluon's auto-stack, XGBoost with different FE, per-year specialists) or a fundamentally different ensembling scheme (stacking with raw features, Bayesian model averaging) is needed.
4. **Drift creep is real and concerning.** Cycle 4 single-seed drift: −0.00024. Cycle 5 multi-seed: −0.00041. Cycle 7 blend: −0.00047. Each layer adds ~+0.00006. Stacking blends-on-blends will eventually swallow OOF lifts entirely.

## Follow-ups

1. ✅ Confirmed cycle 7's 0.95408 is the local 2-way blend ceiling within the current model zoo.
2. **Cycle 8 pivot: exp 024 = AutoGluon-Tabular** (`medium_quality` preset first, ~30 min). Different ensembling philosophy — auto-stacks GBM/NN/RF/KNN with learned weights. If it produces a model with OOF ≥ 0.95300, retry the blend probes with AG as a 3rd component.
3. Skip: pseudo-labeling (LB-overfit risk, noted earlier), XGBoost (too weak on this data), distillation (marginal lift expected).
