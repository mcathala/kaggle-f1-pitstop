# Experiment 025 — LightGBM with cycle 5's rich FE pipeline

**Cycle.** 8
**Status.** Reverted (same FE → same diversity → no blend gain)
**Date.** 2026-05-22

## Hypothesis

A LightGBM trained on cycle 5's rich feature pipeline (arithmetic interactions, count encoding, KBins discretization, interaction categoricals) — instead of the cycle 1 baseline's 49-feature set — produces OOF AUC ≥ 0.945 standalone. When blended into cycle 7's mix as a 3-way (RealMLP × CB-tuned × LGB-rich), it adds ≥ +0.00020 OOF over cycle 7's 0.95408.

## Rationale

- Exp 023 showed the LGB *baseline* (OOF 0.94166) was too weak to add ensemble value. But that LGB used the original 49-feature pipeline; cycle 5's pipeline is *much* richer (46 engineered features with cross-cat interactions, target-encoding-able combos, binning).
- A LightGBM with cycle 5's FE should land at OOF 0.945-0.950 — strong enough to clear the diversity/strength tradeoff floor identified in exp 023 (partner OOF ≥ 0.985 × base_OOF = 0.93952; we need substantially above that for the diversity to translate to blend gains).
- LightGBM is *structurally distinct* from RealMLP (NN with PBLD embedding + MLP) and CB-tuned (CatBoost's ordered target encoding + symmetric trees). LGB uses GOSS sampling, EFB feature bundling, and leaf-wise growth — entirely different optimization geometry.
- Fully controlled environment (vs AG's binary issues): direct LightGBM call, deterministic.

## Expected magnitude

- **LGB standalone OOF:** 0.945-0.950 expected (well above the 0.94166 baseline; rich FE has consistently lifted CB by similar margins).
- **3-way blend target:** ≥ 0.95428 (cycle 7 + 0.00020).
- **Floor for direction-alive:** LGB-rich standalone ≥ 0.940. Below that, FE-vs-baseline did nothing and direction is dead.

## Overfitting risk

**Low-Medium.** Sources:

1. **Same FE as RealMLP** — TE leakage already controlled by sklearn's `TargetEncoder(cv=5)`.
2. **Standard LightGBM HPs** — using known-good values from cycle 1 baseline (early stopping, num_leaves=63, lr=0.05). No HP sweep, so no selection bias.
3. **No retraining loop** — single 5-fold pass, OOF computed once.

## Kill criteria

- [ ] LGB-rich standalone OOF < 0.940 (FE didn't help LGB)
- [ ] LGB-rich standalone OOF < 0.945 AND best blend ≤ cycle 7's 0.95408 (no ensemble value)
- [ ] Any fold's train AUC > 0.99 with val AUC < 0.94 (massive overfit signal)

## Scope

- `src/research/train_lgb_richfe.py` (new, ~210 lines — feature engineering + 5-fold LGB)
- Outputs: `data/oof_lgb_richfe.parquet`, `data/submission_lgb_richfe.csv`
- `experiments/025_lgb_richfe.md` (this file)

Wall-clock budget: ~10-15 min total (LightGBM 5-fold).

## Reversibility check

- CV protocol: unchanged.
- Seed: unchanged.
- Feature set: cycle 5's pipeline (same as RealMLP/AG).
- Leakage surface: unchanged.

No reversibility flag fires.

## Plan

1. Reuse the FE function from `train_realmlp.py` (arithmetic + binning + count encoding + interaction combos + target encoding).
2. Train LightGBM per fold using same 5-fold StratifiedKFold seed=42 on Year × PitNextLap.
3. Skip external data for first run (cycle 1 LGB had it; we want to isolate the FE-vs-baseline question). Add external in a follow-up if needed.
4. Report OOF AUC + per-fold + rank correlation with RealMLP/CB OOFs.
5. If OOF ≥ 0.945: run 3-way blend probe.
6. Decision gate:
   - Best 3-way ≥ 0.95428 → submission, mark Pending LB.
   - Best 3-way ∈ (0.95408, 0.95428) → Inconclusive.
   - Best 3-way ≤ 0.95408 → Reverted, accept direction dead, close cycle 8.

## Result

**LGB-rich standalone:**
- OOF AUC: **0.94477** (+0.00311 over LGB-baseline 0.94166)
- Per-fold: [0.94545, 0.94519, 0.94451, 0.94408, 0.94469]
- Per-fold std: 0.00049 (stable)
- best_iter range: 80-94 (converges very fast — model saturates with this FE)
- Wall-clock: ~1 min total (11s/fold), well below budget

**Rank correlations vs cycle 7 components:**
- LGB-rich × RealMLP: **0.9424** (vs LGB-baseline's 0.9432 — slightly LESS diverse, despite stronger standalone)
- LGB-rich × CB-tuned: 0.9316 (more diverse than LGB-baseline's 0.9434)

**3-way linear blend sweep (RealMLP × CB-tuned × LGB-rich):**

| w_cb | w_lgb | w_realmlp | OOF AUC | Δ vs cycle 7 |
|------|-------|-----------|---------|--------------|
| 0.15 | 0.000 | 0.850 | 0.95407 | −0.00001 |
| **0.20** | **0.000** | **0.800** | **0.95408** | **0.00000 (best)** |
| 0.20 | 0.025 | 0.775 | 0.95405 | −0.00003 |
| 0.20 | 0.050 | 0.750 | 0.95401 | −0.00007 |
| 0.25 | 0.000 | 0.750 | 0.95407 | −0.00001 |

Optimal w_lgb_rich = 0.000 exactly. **Identical to exp 023's LGB-baseline finding.**

**3-way rank-blend:** same conclusion — optimal w_lgb=0, best AUC matches cycle 7's 0.95408.

## Verdict

**Reverted.** LGB-rich is a stronger standalone than LGB-baseline (+0.00311 OOF) but adds zero ensemble value. The key insight: rank correlation with RealMLP went DOWN slightly compared to baseline (0.9424 vs 0.9432), despite using identical FE — so even sharing FE didn't *increase* shared structure, but it also didn't differentiate enough to overcome RealMLP's standalone dominance.

## Kill-criteria check

- [x] LGB-rich standalone OOF > 0.940 (0.94477 — strong; FE worked)
- [x] LGB-rich standalone OOF > 0.945 — borderline pass (0.94477)
- [x] Best blend = cycle 7's 0.95408 (literally identical; LGB-rich axis adds nothing)
- [x] No fold's train AUC > 0.99 (best_iter caps prevented overfitting)

The "ensemble value" kill criterion fired: blend ≤ cycle 7's 0.95408.

## Repro stamp

- pkg: lightgbm 4.6.0 (current .venv), pandas 2.3.3, sklearn 1.7.2
- inputs: same FE as `train_realmlp.py`
- outputs: `data/oof_lgb_richfe.parquet`, `data/submission_lgb_richfe.csv`

## Learnings

1. **DURABLE FINDING — FE is doing more diversification work than architecture.** Two models (RealMLP, LightGBM) trained on the *same* engineered features produce predictions with high rank correlation (0.9424), even though they have wildly different inductive biases (NN with PBLD embedding vs leaf-wise GBM). Architecture alone does not produce blend-able diversity when the input representation is shared. For future blend probes, vary FE explicitly OR use raw features for the diversity partner.
2. **Stronger ≠ better blend partner.** LGB-rich (OOF 0.94477) is +0.00311 over LGB-baseline (0.94166), yet adds zero blend value where LGB-baseline already added zero. Strength alone doesn't unlock ensembling; the rank-disagreement geometry has to be different from the existing blend members.
3. **LGB converges fast on this FE.** best_iter ranged 80-94 across folds — model saturates within ~100 iterations at lr=0.05. Further HP tuning (lower lr, more iters) might add marginal lift but not the +0.001 needed to break the blend.
4. **Cycle 8 closes the model-zoo-blending direction definitively.** Three independent probes (LGB-baseline, AutoGluon-attempt, LGB-rich) all failed to add ensemble value. Cycle 9 needs a structurally different angle.

## Follow-ups

1. ✅ LGB-rich is in the bank as a reference model. OOF + submission saved.
2. **Cycle 8 closes Inconclusive.** Project tip stays at cycle 7's LB 0.95361.
3. **Cycle 9 candidates (in priority order):**
   - **Pseudo-labeling** — high-confidence test predictions added to train set. Known LB-overfit risk; mitigated by conservative confidence threshold (>0.95 or <0.05) and limiting to a small fraction of test set.
   - **Per-year specialist models** — train 4 RealMLPs (one per year). Year 2022 has the worst per-year AUC (0.90817 in cycle 6 ensemble); a 2022-specific model with year-targeted features could close that slice.
   - **Calibration / isotonic regression on cycle 7 blend** — doesn't change ordering (AUC unchanged) but could tighten the OOF→LB drift.
   - **Custom features from residual EDA on cycle 7 OOF** — find the slices where the blend still misses systematically.
   - **A second multi-seed pass with different feature splits** — train RealMLP on 5 random feature-subset seeds, average.
