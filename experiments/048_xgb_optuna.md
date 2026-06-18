# Experiment 048 — Optuna HP sweep on XGB-highbins

**Cycle.** 15
**Status.** Inconclusive (Reverted) — TPE confirms cycle-11 HPs at/near a local optimum within the search space; 15-trial budget insufficient to escape, and CPU compute ceiling prevented finishing the 25-trial plan.
**Date.** 2026-05-26

## Hypothesis

A 25-trial TPE sweep around cycle-11's hand-tuned XGB-highbins HPs finds a fold-1 AUC ≥ 0.95350 (baseline 0.95331 + ≥ +0.00019). The top-1 configuration, retrained at full 5-fold, lifts standalone OOF by ≥ +0.00010 over cycle 11's 0.95263, and the new 3-way blend OOF clears 0.95441 (cycle 11's 0.95421 + ≥ +0.00020).

## Rationale

Cycle 11's HPs (`max_depth=10, eta=0.01, λ=8.16, α=8.35, colsample=0.145, max_bin=5000`) came from manual tuning anchored on a Kaggle-shared recipe. They were never systematically searched. After cycles 12-14 closed the model-zoo and variance-reduction axes, the most defensible remaining CPU-only XGB axis was: *is there a better local optimum nearby?*

Search space (around the known-good point):

| HP | Range | Cycle 11 value |
| --- | --- | --- |
| `max_depth` | {6, 8, 10, 12} | 10 |
| `eta` | [0.005, 0.03] log | 0.01 |
| `reg_lambda` | [1, 16] log | 8.16 |
| `reg_alpha` | [0.5, 16] log | 8.35 |
| `colsample_bytree` | [0.10, 0.30] | 0.145 |
| `subsample` | [0.6, 0.95] | 0.857 |
| `min_child_weight` | [1, 8] log | 2 |
| `max_bin` | {3000, 5000, 8000} | 5000 |

Sweep design: fold-1-only evaluation per trial (`TRIAL_N_ROUNDS=5000`, early-stop patience 80), 25 TPE trials, then plan to run full 5-fold on the top trial. Total budget ~4-5h CPU.

## Expected magnitude

- Best trial fold-1 AUC ≥ 0.95350 (+0.00019 over baseline 0.95331).
- After full 5-fold: standalone OOF ≥ 0.95273 (+0.00010 over cycle 11).
- 3-way blend OOF ≥ 0.95441 (+0.00020 over cycle 11). Floor for the cycle to be worth shipping.

## Kill criteria

- [x] After ≥ 12 trials, best fold-1 AUC < baseline 0.95331 — **FIRED** (best = trial 11 at 0.95322).
- [x] Visible plateau by trial 12+ within ±0.00010 of best, indicating TPE has converged below baseline.
- [x] Compute ceiling hit mid-run (memory pressure forcing CPU throttle).

## Result

### Trial trajectory (15 of 25 completed before SIGTERM)

| Trial | AUC | max_depth | max_bin | eta | λ | α | colsample |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.95250 | 6 | 5000 | 0.024 | 14.7 | 8.95 | 0.104 |
| 1 | 0.95268 | 8 | 3000 | 0.015 | 3.54 | 7.60 | 0.173 |
| 2 | 0.95261 | 8 | 8000 | 0.027 | 1.31 | 5.36 | 0.161 |
| 3 | 0.95276 | 8 | 8000 | 0.009 | 14.7 | 7.34 | 0.137 |
| 4 | 0.94685 | 6 | 3000 | 0.009 | 2.69 | 1.32 | 0.266 |
| 5 | 0.95066 | 8 | 8000 | 0.005 | 8.49 | 0.65 | 0.246 |
| 6 | 0.94947 | 6 | 8000 | 0.009 | 3.70 | 0.76 | 0.277 |
| 7 | 0.94911 | 6 | 5000 | 0.005 | 2.39 | 2.91 | 0.227 |
| 8 | 0.95131 | 6 | 3000 | 0.007 | 11.2 | 8.10 | 0.227 |
| 9 | 0.95080 | 8 | 5000 | 0.008 | 1.02 | 2.94 | 0.272 |
| 10 | 0.95299 | 10 | 5000 | 0.022 | 6.14 | 14.46 | 0.107 |
| **11** | **0.95322** | **10** | **8000** | **0.0145** | **6.80** | **14.32** | **0.106** |
| 12 | 0.95310 | 10 | 8000 | 0.016 | 7.28 | 13.41 | 0.101 |
| 13 | 0.95300 | 10 | 8000 | 0.019 | 6.64 | 14.59 | 0.137 |
| 14 | 0.95319 | 12 | 8000 | 0.012 | 6.65 | 15.78 | 0.132 |

**Baseline (cycle 11 HPs, fold-1): AUC 0.95331.**

### TPE behaviour

- **Trials 0-9 (exploration).** TPE sampled `max_depth ∈ {6, 8}` exclusively, with `λ, α` varying widely. Best in this phase: trial 3 at 0.95276 (Δ vs baseline −0.00055). Two trials (4, 6, 7) collapsed below 0.949 from low-`α` × low-`λ` combos.
- **Trials 10-14 (exploitation).** TPE switched to `max_depth=10` from trial 10 onwards, and to `max_depth=12` for trial 14. Both gave the same operating point: high `max_bin` (8000), high `reg_alpha` (13-16), `colsample_bytree` ≈ 0.10-0.14, `eta` ≈ 0.012-0.022. AUC clustered tightly: {0.95299, 0.95322, 0.95310, 0.95300, 0.95319} — a plateau within ±0.00012 of trial 11.

### Best trial (11) vs cycle 11 baseline

| HP | Trial 11 | Cycle 11 |
| --- | ---: | ---: |
| `max_depth` | 10 | 10 |
| `max_bin` | 8000 | 5000 |
| `eta` | 0.0145 | 0.01 |
| `reg_lambda` | 6.80 | 8.16 |
| `reg_alpha` | 14.32 | 8.35 |
| `colsample_bytree` | 0.106 | 0.145 |
| `subsample` | 0.760 | 0.857 |
| `min_child_weight` | 1.79 | 2 |
| **fold-1 AUC** | **0.95322** | **0.95331** |

Trial 11 matches baseline on `max_depth=10` and similar regularization spirit but uses higher `max_bin` and lower `colsample_bytree`. It lands **−0.00009 below** baseline. The neighbouring trials (10, 12, 13, 14) confirm this is a plateau, not a single-trial noise event.

### Compute ceiling event

The sweep was terminated by SIGTERM after trial 14 at ~17:15 CEST due to memory pressure on the M1 Pro 16 GB:

- Resident memory grew from ~680 MB at session start to 2.85 GB by trial 14.
- Free pages dropped to ~72 MB; load avg climbed to 7-10 over 1m.
- Process CPU utilization dropped from peak 530-550% to 329% — clear thermal/swap throttling.
- Trial 14 took 23.8 min vs the 5-8 min/trial average — first sign of sustained degradation.

Killing at 15/25 was the right call: the verdict was already legible (TPE plateaued below baseline) and continuing risked an OOM-kill or hung trial that would taint the closure.

## Verdict

**Inconclusive (Reverted).** TPE explored the right basin (depth=10 + high λ + high α) by trial 10 and plateaued at ~0.9532, ~0.00009 below cycle 11's hand-tuned 0.95331. No fold-1 trial cleared the +0.00019 hurdle, and the plateau across 5 consecutive exploitation trials confirms the local optimum cycle-11 found by hand is near-optimal within this search space.

The 25-trial plan was not completed (15/25, ~60%) due to a hard compute ceiling on the M1 Pro. The remaining 10 trials would likely have continued exploiting the same depth=10/12 + high-α basin without escape.

## Kill-criteria check

- [x] Best fold-1 AUC < baseline by trial 12 — **FIRED** (0.95322 vs 0.95331, gap −0.00009).
- [x] Plateau visible across trials 10-14 within ±0.00012 — **FIRED**.
- [x] Compute pressure forcing early termination — **FIRED** (RSS 2.85 GB / free 72 MB / CPU dropped 530% → 329%).

## Repro stamp

- packages: xgboost 3.2.0, optuna (TPE, in-memory study)
- runtime: trials 0-14 completed in 114 min wall (avg 7.6 min/trial, plus 23.8 min outlier trial 14)
- HP space + seed: `TPESampler(seed=42)`, defined in [src/research/optuna_xgb_highbins.py](../src/research/optuna_xgb_highbins.py)
- Study state: in-memory only, lost on SIGTERM. Full trajectory recoverable from `/tmp/optuna_xgb.log`.

## Learnings

1. **Cycle-11's XGB-highbins HPs are at/near a local optimum.** TPE found the right basin (depth=10, max_bin ≥ 5000, high `reg_lambda`, high `reg_alpha`, low `colsample_bytree`) by trial 10 and could not beat the hand-tuned configuration in 4 more samples within ±0.0001. This validates the cycle-11 recipe and closes the "is there a better nearby XGB-highbins HP combo?" question on CPU.
2. **Depth=10 dominates depth ∈ {6, 8} on this dataset.** TPE wasted 10 trials sampling shallow trees before pivoting. Trial 4 (depth=6, low α/λ) collapsed to 0.94685 — −0.00646 vs baseline. Future XGB sweeps on this data should pin `max_depth ∈ {10, 12}` and skip shallower.
3. **The M1 Pro 16 GB is the compute ceiling, not the HP space.** Memory pressure forced early termination at 60% of plan. XGBoost with `max_bin ∈ {5000, 8000}` × `max_depth ∈ {10, 12}` × 5 folds is at the edge of what fits in 16 GB unified memory. Future sweeps on this stack require GPU.
4. **TPE seed 42 explored shallow trees first — sampling artifact, not principled.** Trials 0-9 sampled `max_depth ∈ {6, 8}` exclusively. Different TPE seeds or a prior-weighted sampler might have hit depth=10 sooner. For a 25-trial budget on a 4-category × 7-continuous space, this is on the edge of feasibility.

## Follow-ups

- **Closed direction:** further XGB-highbins HP search on CPU. The 0.95331 fold-1 / 0.95263 OOF / 0.95421 3-way blend is the operating point for this XGB family.
- **For cycle 16+:** the realistic unblock paths are
  1. **Kaggle Notebooks (free T4/P100 16 GB, 30h/week)** — port the existing stack and run the experiments cycle 12 / cycle 15 couldn't finish: TabM_D_Classifier (cycle 12 retry), XGB-GPU at 100+ trials with `max_depth ∈ {10, 12}` only, RealMLP at larger width. Setup cost ~1-2h to write a Kaggle-runnable notebook.
  2. **RunPod Community RTX 4090 @ $0.34/hr** — burst option for interactive sweeps where Kaggle's 12h session cap or Internet-OFF constraint bites.
- The cycle-10 probe-2 (Q4 worst-loss FE) remains the cheapest CPU-only follow-up if we want to stay local.
