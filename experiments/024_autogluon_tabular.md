# Experiment 024 — AutoGluon-Tabular

**Cycle.** 8
**Status.** Infra-fail (SIGSEGV during LightGBMXT fit, abandoned)
**Date.** 2026-05-22

## Hypothesis

AutoGluon-Tabular trained on cycle 5's feature pipeline, run as 5 standalone folds matching our fixed CV protocol, produces a model whose OOF AUC is ≥ 0.95300 standalone AND whose OOF residuals are diverse enough from RealMLP that a 3-way blend (RealMLP × CB-tuned × AG) clears cycle 7's 0.95408 OOF by ≥ +0.00020.

## Rationale

- Exp 023 confirmed the model zoo is exhausted for cheap blend gains — LGB (most diverse non-blend partner) added zero.
- The remaining ROI requires a *new* model with two properties:
  - **Standalone strong enough** (OOF ≥ 0.953) to overcome diversity-discount in blends
  - **Architecturally distinct** from RealMLP and CB-tuned
- AutoGluon-Tabular fits both: it auto-stacks GBM (CatBoost, LightGBM, XGBoost), Neural Nets (its own MLP), Random Forests, ExtraTrees, KNN — and learns weights between them. The stack is structurally different from a hand-tuned single model.
- On past Kaggle Playground competitions, `medium_quality` AutoGluon has hit OOF AUC within −0.005 of best hand-tuned solutions.

## Expected magnitude

- **AG standalone OOF target:** ≥ 0.95300 (within −0.0008 of RealMLP, would qualify for blend probe).
- **Best 3-way blend target:** ≥ 0.95428 (cycle 7 + min_delta 0.00020).
- **Optimistic:** AG standalone reaches 0.95400+, 3-way blend reaches 0.95450+.
- **Floor:** if AG standalone < 0.95200, blend won't help (too weak — cycle 7's CB-tuned was already at 0.95114 and could only add +0.00025).

## Overfitting risk

**Medium.** Sources:

1. **AG includes its own NN/GBM stacks** — risk of overlap with our RealMLP / CB-tuned signal. Mitigated by inspecting AG's component leaderboard.
2. **`medium_quality` preset has internal bagging** — could overfit to fold-1's holdout if its internal CV doesn't align with ours. Mitigated by manual 5-fold loop using OUR splits.
3. **Time budget pressure** — 10 min/fold caps quality. May not reach the standalone target.

## Kill criteria

- [ ] AG standalone OOF < 0.95200 (too weak to add ensemble value, by the cycle 4 blend-monotonicity adjusted rule)
- [ ] Any single fold takes > 25 min (cumulative wall-clock > 2 hr → unsustainable)
- [ ] Best 3-way blend < cycle 7's 0.95408 (AG signal correlated to existing stack, adds nothing)

## Scope

- `src/research/train_autogluon.py` (new, ~180 lines)
- Outputs: `data/oof_autogluon.parquet`, `data/submission_autogluon.csv`, AG model artifacts in `ag_models/` (gitignored)
- `experiments/024_autogluon_tabular.md` (this file)

Wall-clock budget: 5 folds × ~10 min = ~50 min. If first fold > 15 min, scale down.

## Reversibility check

- CV protocol: unchanged (manual 5-fold StratifiedKFold seed=42 on Year × PitNextLap).
- Seed: unchanged.
- Feature set: cycle 5's pipeline (same as RealMLP).
- Leakage surface: unchanged.

No reversibility flag fires.

## Plan

1. Build `src/research/train_autogluon.py` using `autogluon.tabular.TabularPredictor` with `time_limit=600` per fold, `presets="medium_quality"`, `eval_metric="roc_auc"`.
2. Same 5-fold StratifiedKFold seed=42 on Year × PitNextLap as everything else.
3. Train one AG predictor per fold; predict on val + test; accumulate OOF + averaged test predictions.
4. Report:
   - AG standalone OOF AUC + per-fold AUCs
   - AG component leaderboard (which sub-models AG trained, their internal scores)
   - Rank correlation AG-OOF vs RealMLP-OOF vs CB-tuned-OOF
5. **If AG standalone clears 0.95200**: 3-way blend probe RealMLP × CB-tuned × AG via the existing `blend_3way_probe.py` pattern.
6. **Decision gates:**
   - Best 3-way ≥ 0.95428 → generate submission, mark Pending LB.
   - Best 3-way ∈ (0.95408, 0.95428) → Inconclusive, document.
   - Best 3-way ≤ 0.95408 → Reverted, document direction dead, consider escalating AG to `high_quality` (2 hr) before fully closing.

## Result

**Attempt 1** (medium_quality preset, 11 L1 models including NN_TORCH/FASTAI/RF/XT, time_limit=600s, train rows incl. external = ~432k):
- AG started, did feature inference, began fitting LightGBMXT
- Process died silently mid-fit. Bash exited 0 (tee'd pipe closed cleanly) but Python process was gone. Likely OOM-killed by macOS — 16.6% memory pre-fit warning + parallel model fits on M1 Pro 16 GB.

**Attempt 2** (constrained zoo: GBM+CAT+XGB only, skip external data → ~351k rows, time_limit=480s):
- AG started, did feature inference
- Process **crashed with SIGSEGV (exit code 139)** during LightGBMXT fit, no Python-level exception
- Root cause likely a binary incompatibility in AG's bundled LightGBM vs our pandas 2.3.3 / sklearn 1.7.2 (these were *downgraded* by the AG install from pandas 3.0.2 / sklearn 1.8.0). M1 Pro ARM64 native libraries are notoriously sensitive to ABI changes across pandas major versions.

## Verdict

**Infra-fail.** Cannot get AutoGluon to complete a single fold without crashing on this machine. Not a model-quality finding — purely an environment problem. Could be resolved by a fresh isolated venv just for AG, but the time investment isn't worth it when a simpler pivot (LightGBM with cycle 5's FE pipeline) provides the same value (a structurally distinct model with strong OOF) at 1/10th the wall-clock and zero env risk.

## Kill-criteria check

Not applicable — never reached a result. The 25-min/fold ceiling was crossed in *negative* sense (process died before any fold completed).

## Repro stamp

- pkg attempted: autogluon-tabular 1.4.1 (current PyPI), pandas 2.3.3 (downgraded from 3.0.2 by AG install)
- failure: exit code 139 (SIGSEGV) during LightGBMXT model fit, no traceback emitted

## Learnings

1. **AutoGluon-Tabular's dep tree is hostile to mature environments.** The install forced downgrades on numpy/pandas/sklearn/scipy/pyarrow, and the downgraded pandas (2.x) combined with M1 Pro ARM64 LightGBM binaries produced SIGSEGV. Fresh isolated venv would be safer — but the indirect cost of switching venvs mid-cycle is high.
2. **Memory headroom on M1 Pro is tight for AG.** Even the constrained zoo (GBM+CAT+XGB, no external data) triggered memory warnings. AG's parallel-model-fit strategy isn't well-suited to a 16 GB laptop without significant tuning.
3. **The hypothesis is still interesting**, just not via AG. The pivot to a hand-trained LightGBM with cycle 5's rich FE pipeline (exp 025) tests the same underlying question — *does a structurally distinct, strong tabular model add ensemble value to the cycle 7 mix?* — with full environment control.

## Follow-ups

1. ✅ Abandoned exp 024 as Infra-fail.
2. **Cycle 8 pivot continuation: exp 025 = LightGBM with cycle 5 FE pipeline.** Same data, same features as RealMLP — but a tree model, fully controlled environment, ~10 min to train.
3. If exp 025 also fails to break the cycle 7 blend ceiling, accept that the model zoo is exhausted and close cycle 8 Inconclusive. Cycle 9 would then need a fundamentally different angle: pseudo-labeling, custom features driven by per-year residual EDA, or a hand-trained XGBoost.
