# Experiment 033 — CB-tuned-exp14 with `class_weights=[1.0, 1.5]` on GPU

**Cycle.** 11
**Status.** Infra-fail
**Date.** 2026-05-25

## Hypothesis

Replacing cycle 14's `auto_class_weights="Balanced"` with a **mild explicit `class_weights=[1.0, 1.5]`** (everything else identical, on GPU) lifts standalone CB OOF AUC by ≥ +0.00036 (cycle 14's 0.95114 → ≥ 0.95150) AND reduces probe 5's bin-8 over-prediction from +0.057 → ≤ +0.030. When blended with RealMLP-multiseed, lifts ≥ +0.00020 over cycle 7's 0.95408.

## Rationale

Cycle 10 produced two findings that together motivate this experiment:

1. **Probe 5 (exp 030):** cycle 7's blend over-predicts positives in bins 7–9 (max bias +0.057 at bin 8). Structural cause: `auto_class_weights="Balanced"` on a 21%-positive dataset applies an implicit ~3.8× upweight to positives, pulling mid-confidence predictions toward positive class.

2. **Exp 032:** completely removing Balanced (`class_weights=None`, equivalent to `[1, 1]`) hurts standalone AUC by ~0.001 (killed at iter 1750, projected asymptote 0.9499). Balanced was load-bearing for cycle 14's HP combo — the model needs *some* positive-class emphasis to converge to its 0.95114 AUC point.

The natural test is the **middle of these two extremes**: explicit `class_weights=[1.0, 1.5]` retains 39% of Balanced's implicit upweight (≈1.5 / 3.8). Hypothesis: this preserves the AUC-relevant positive emphasis while halving the bin-8 over-prediction.

Two co-motivating considerations from cycle 10:
- **Exp 032's secondary signal:** at iter 1500→1750 the decay ratio flattened to 0.95× (vs prior 0.62×), hinting at a delayed convergence pattern when positive emphasis is removed. A mild 1.5× weight may recover the lost gradient amplitude without the over-prediction side-effect.
- **GPU eliminates the compute barrier.** Exp 032 was killed for trajectory reasons; on GPU, full convergence at iter 8000 is ~10 min/fold vs 60+ min on CPU. We can let it run.

## Expected magnitude

- **Standalone CB OOF target:** ≥ 0.95150 (+0.00036 over cycle 14).
- **Stretch:** ≥ 0.95180 (+0.00066).
- **Bin-8 bias target:** ≤ +0.030 (vs cycle 7's +0.057; halved).
- **Floor:** OOF < 0.95080 → mild upweighting was the wrong middle-ground; mark cycle 11 Inconclusive and try `class_weights=[1.0, 2.0]` or a 4-point sweep in cycle 12.
- **Blend (RealMLP × new CB at best `w_cb`):** ≥ 0.95428.

## Overfitting risk

**Very low.**

1. Single explicit HP swap (Balanced → [1.0, 1.5]). No FE change, no CV change.
2. CV unchanged from cycle 14 (`StratifiedKFold(5, shuffle=True, random_state=42)` on `Year × PitNextLap`).
3. No HP search: a single point, no selection bias.
4. GPU determinism is slightly weaker than CPU's, but `random_seed=SEED+fold` is set per fold so reproducibility holds.

## Kill criteria

- [ ] Full-OOF standalone AUC < 0.95080 (no lift over cycle 14; mild upweight is the wrong middle ground).
- [ ] Bin-8 bias > +0.045 (the upweight didn't move calibration meaningfully; probe-5 mechanism wasn't the bottleneck).
- [ ] Per-fold std > 0.00080 (instability).
- [ ] OOM on Kaggle T4 GPU (~16 GB VRAM should be ample for this recipe; if it fires, drop `iterations` to 6000 and retry).
- [ ] Total runtime > 90 min (would suggest GPU not actually being used; check `task_type` and `devices` flags).

## Scope

- `gpu-kernels/cycle11_kaggle_gpu.py` (new — single-file script, paste into Kaggle Notebook).
- `experiments/033_cb_class_weights_gpu.md` (this file).
- Outputs (produced ON Kaggle, downloaded locally afterwards):
  - `data/oof_cb_classweighted_gpu.parquet`
  - `data/submission_cb_classweighted_gpu.csv`

Wall-clock budget (estimated):
- Kaggle T4 GPU, 5-fold × 1 seed × cycle 14 recipe: ~60 min total.
- If a per-fold timing report comes back > 20 min, GPU isn't engaged — kill and check params.

## Reversibility check

- CV protocol: **unchanged** — same seed, same fold construction.
- Seed: 42, per-fold offset preserved from cycle 14.
- Feature set: **unchanged** — identical 132-feature recipe (cycle 12 base FE + cross-cats + freq + group-stats).
- Target transform: unchanged.
- Leakage surface: unchanged.

No reversibility flag fires.

## Plan

1. Build `gpu-kernels/cycle11_kaggle_gpu.py` — single-file paste-into-Kaggle script.
2. Create Kaggle Notebook: enable GPU, add inputs (competition + external dataset), paste script, run.
3. Collect the final OOF AUC + per-fold AUCs + bin-8 bias report.
4. Decision gate:
   - If standalone OOF ≥ 0.95150 AND bin-8 bias ≤ 0.045: blend probe (locally) → if blend ≥ 0.95428, generate submission.
   - If standalone OOF < 0.95080: kill, plan cycle 12 with `class_weights=[1, 2]` or 4-point sweep.
   - Marginal (0.95080–0.95150): write up Inconclusive, but check if the blend probe still clears even at flat standalone AUC (the new rank-corr-with-RealMLP might compensate).

## Result

Infra-fail. The Kaggle kernel silently allocated CPU instead of the requested GPU (an account-level allocation issue), compounded by nested input-mount paths and a CatBoost CUDA driver mismatch. The class-weight comparison never produced a clean OOF on GPU.

## Verdict

**Infra-fail.** The `class_weights=[1.0, 1.5]` hypothesis was untestable on the available Kaggle GPU allocation. The notebook (`gpu-kernels/cycle11_kaggle_gpu.py`) is retained for a future GPU revisit; the calibration question it targets was not blocking (AUC is rank-based, unaffected by the bin-8 bias). The same silent-CPU-downgrade failure mode recurred later and was handled by failing fast on `torch.cuda.is_available()` (see exp 049).

## Repro stamp

- notebook: `gpu-kernels/cycle11_kaggle_gpu.py` (paste-into-Kaggle, GPU)
- inputs: competition data + external strategy dataset
- outcome: no usable OOF (GPU not allocated)
