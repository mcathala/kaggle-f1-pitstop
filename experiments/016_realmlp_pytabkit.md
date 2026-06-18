# Experiment 016 — RealMLP via PyTabKit (tabular NN)

**Cycle.** 4 (closes cycle 4 — LB confirmed at 0.95331, +0.00265 over cycle 3)
**Status.** Kept (LB-confirmed) — closes cycle 4 with the largest single-experiment LB lift of the project
**Date.** 2026-05-22
**Pre-registered in `project.md`'s cycle 4 next-steps section; promoted over exp 015 after M1 Pro hardware context.**

## Hypothesis

A pre-tuned tabular MLP (RealMLP via PyTabKit) trained on a light FE pipeline (arithmetic interactions + count encoding + KBins discretization + interaction-cat target encoding) + external dataset augmentation produces a 5th ensemble component whose residuals are structurally different from CatBoost. When blended into the cycle 4 ensemble (LGB + CB#006 + CB-tuned-exp14 + RealMLP), the ensemble OOF AUC lifts by ≥ +0.0010 over experiment 014's 0.95161.

## Rationale

Cycle 13's residual EDA showed the remaining LB gap is concentrated in driver-level cohorts where CatBoost's ordered target encoding has hit its ceiling. The 5 worst-AUC drivers (real F1 codenames VET/MSC/COL/BEA/ALO) are exactly the cohorts NN embeddings target — a learned low-dimensional embedding can capture per-driver style signatures that target encoding flattens.

Two reasons to prefer RealMLP over a hand-rolled NN:
1. **Pre-tuned defaults from the Holzmüller "Better by Default" paper** — saves us from hyperparameter tuning a tabular NN from scratch.
2. **Public evidence**: the most-voted single-model notebook in this competition uses PyTabKit's RealMLP_TD_Classifier with the same defaults we're using. We're not pioneering — we're replicating a proven recipe in our own pipeline.

M1 Pro Metal (MPS) makes this affordable: an experiment that would have been 3-4h on x86 CPU is ~1-2h on Apple Silicon. The decision to swap experiment 015 (multi-seed, ~6 hr) for this experiment was driven by ROI per wall-clock hour.

## Expected magnitude

- RealMLP standalone OOF: expected 0.94 – 0.95 (lower than CatBoost; that's the public pattern — RealMLP standalone underperforms CatBoost by a hair, but the residual diversity adds ensemble value).
- 5-way ensemble OOF (`5way_with_realmlp` = LGB=0.05, CB#006=0.20, CB-tuned-exp14=0.55, RealMLP=0.20, or similar): **≥ 0.95261** (= 0.95161 + 0.0010). Stretch +0.0025.
- LB projection (with cycle 12's −0.00068 OOF→LB drift): ≥ 0.9519.

## Overfitting risk

**Medium-high.** Three concerns:

- **Tabular NN on small features is touchy.** The pre-tuned defaults reduce this but don't eliminate it. n_ens=24 internal ensemble helps; SiLU + p_drop=0.05 + label smoothing eps=0.01 also help.
- **MPS backend on PyTorch is less battle-tested than CUDA.** Possible non-determinism / numerical surprises. Mitigated by `torch.manual_seed(42)` and `n_ens=24` averaging.
- **External dataset has 25.5% pos rate vs train's 19.9%.** Same shift as in CatBoost cycles; class-balanced loss / label smoothing helps.

## Kill criteria

- RealMLP standalone OOF < 0.93 → architecture / FE not converging (NN can be 1 fold-AUC away from collapse on tabular).
- 5-way ensemble OOF ≤ experiment 014's 0.95161 → diversity didn't add anything.
- Per-fold std > 0.0020 on RealMLP → instability.

## Scope

- Add `torch` + `pytabkit` to `pyproject.toml`.
- `src/research/train_realmlp.py` — new trainer, mirrors public PyTabKit notebook recipe with M1 Pro Metal device override.
- `src/research/blend_exp16.py` — 5-way ensemble blend (LGB + CB#006 + CB-tuned-exp14 + RealMLP).
- `experiments/016_realmlp_pytabkit.md` — this file.

## Reversibility check

CV unchanged. Project seed unchanged. Target transform unchanged. Leakage surface unchanged. Adding torch + pytabkit is a structural dep change but reversible. Safe.

## Plan

1. ✅ Add deps to pyproject.toml.
2. ⏳ Install via pip (`uv sync` hit private-mirror 401; falling back to PyPI).
3. ⏳ Train RealMLP on 5 folds with external augmentation; expect ~10-20 min/fold on M1 MPS.
4. ⏳ Blend 5-way; pick best weight scheme from a small pre-registered set.
5. ⏳ Apply gates; document.
6. ⏳ If KEEP and cumulative cycle 4 lift over cycle 3 ≥ +0.001 LB-equivalent: submit and close cycle 4.

## Result

### RealMLP standalone

```
OOF AUC:   0.95355   (vs CB-tuned-exp14 0.95114, Δ = +0.00241)
                    (vs exp 14 ensemble 0.95161, Δ = +0.00194)
Per-fold:  0.95421, 0.95419, 0.95325, 0.95252, 0.95364
fold std:  0.00064
wall:      5 folds × ~5 min = ~25 min on M1 Pro MPS (vs ~3-4 hr estimate on CPU)
```

All 5 folds consistently beat CB-tuned-exp14 by ~+0.0022 to +0.0026. Per-fold deltas vs CB-tuned-exp14: +0.00261, +0.00225, +0.00259, +0.00222, +0.00239. **Remarkably consistent.**

M1 Pro MPS was the right call — 5-10× faster than CPU as expected. Training cost dropped from "overnight commitment" to "lunch break".

### 5-way ensemble weight sweep — surprising winner

| Scheme | Weights | OOF AUC | Δ vs exp 14 | Folds up |
|---|---|---|---|---|
| 4way_exp14_ref (reference) | LGB=0.05, CB#006=0.20, CB-tuned-exp14=0.75 | 0.95161 | 0 | — |
| 5way_realmlp_15 | + RealMLP=0.15 | 0.95251 | +0.00090 | 5/5 |
| 5way_realmlp_20 | + RealMLP=0.20 | 0.95275 | +0.00113 | 5/5 |
| 5way_realmlp_30 | + RealMLP=0.30 | 0.95314 | +0.00152 | 5/5 |
| 3way_cb_realmlp | LGB=0.05, CB-tuned-exp14=0.65, RealMLP=0.30 | 0.95304 | +0.00143 | 5/5 |
| **single_realmlp** | **RealMLP=1.0** | **0.95355** | **+0.00194** | **5/5** |

**Surprise**: `single_realmlp = 1.0` is the best. Adding any LGB / CB#006 / CB-tuned-exp14 weight HURTS the blend — the lift is monotonic in RealMLP's weight.

Interpretation: RealMLP's residuals on this data are not just diverse from the CatBoost models — they're STRICTLY BETTER on the slices that matter. The CB stack's information is contained within RealMLP's information set (or close enough that mixing dilutes). This is the opposite of cycles 4 / 6 where CB+LGB blends beat solo. Possible reason: RealMLP's learned categorical embeddings for Driver / Race capture the per-driver style signatures that CatBoost target encoding flattens (exactly the residual cycle 13's EDA flagged).

### Reproducibility stamp

- git SHA at start: `9363cf1`
- data sha256(train.csv): `f004e79d4e63f4bad0afc3788d07938dc5dbc0bed73a51e7b65cbce52ccc4128`
- external data: `data/f1_strategy_dataset_v4.csv` (101,371 rows)
- packages: torch 2.12.0, pytabkit 1.7.3, MPS available + built
- device used: `mps`

### Acceptance gates

baseline_std (exp 14 4-way ensemble fold std) = 0.00046 → magnitude floor = max(0.5 × 0.00046, 0.00020) = **0.00023**.

| Gate | Target | Got | Pass? |
|---|---|---|---|
| Magnitude (Δ OOF ≥ 0.00023) | ≥ 0.00023 | +0.00194 | **PASS** (8× the floor) |
| Direction (≥ 3/5 folds improved) | ≥ 3 | 5/5 | PASS |
| Stability (fold std ≤ 0.0020) | ≤ 0.0020 | 0.00064 | PASS |
| Kill: RealMLP OOF ≥ 0.93 | ≥ 0.93 | 0.95355 | PASS |
| Kill: 5-way > exp 14 | > 0.95161 | 0.95355 | PASS |

All gates pass with large margin. This is the largest single-experiment lift since exp 012 (CB-tuned redesign at +0.00268).

## Verdict

**Kept (significant improvement).**

This is the cycle-4 closing submission candidate. LB projection (using cycle 12's −0.00064 drift, stable across cycle 14): **~0.9529**, a **+0.00224 lift over cycle 3's 0.95066** and **+0.00193 over exp 14's 0.95097**. Well above the cycle-close threshold.

`data/submission_ensemble_exp16.csv` (= `submission_realmlp.csv` since the winning weight scheme is single_realmlp at 1.0) submitted to Kaggle on 2026-05-22. **Public LB = 0.95331.** OOF→LB drift **−0.00024** (vs cycle 12's −0.00068 and exp 14's −0.00064 — drift sharply IMPROVED, RealMLP transfers cleaner than CatBoost). **LB lift +0.00234 over exp 14 (0.95097)** and **+0.00265 over cycle 3 (0.95066)**. Cycle 4 closes here.

## Learnings

1. **RealMLP+PyTabKit's pre-tuned defaults are a single-model breakthrough on this data.** The 111-vote public notebook was right — this architecture (PBLD numeric embedding + n_ens=24 internal ensemble + SiLU + the right normalization) extracts signal CatBoost can't reach on tabular F1 telemetry. We got +0.00241 over our best CatBoost from a single, untuned recipe.
2. **The CatBoost residual subset is contained within RealMLP's.** Every blend that adds CB to RealMLP HURTS. This means: future ensembles in this competition should treat RealMLP as the anchor, not CatBoost. Reverses the typical Kaggle pattern.
3. **M1 Pro MPS is dramatically faster than CPU for tabular NN.** 25 min for 5-fold RealMLP vs ~3-4 hr estimated on CPU. Should have asked about hardware sooner; would have prioritised this branch earlier.
4. **Driver-level discrimination was indeed the residual axis.** Cycle 13's EDA flagged real F1 codenames (ALO, STR, RIC, VET, MSC) as the structurally hardest cohorts. RealMLP's categorical embedding for Driver is exactly the right tool — learned 6-dim embeddings capture per-driver style much more flexibly than CatBoost's ordered target encoding.

## Follow-ups

1. **Submit `data/submission_ensemble_exp16.csv` to Kaggle** to confirm the OOF→LB transfer. If LB lands ≥ 0.9525, close cycle 4 with this as the kept submission.
2. **Cycle 5 candidates** (post-LB):
   - **Multi-seed RealMLP** (~125 min, 5 extra MPS runs): variance reduction on the new champion. Expected +0.0005-0.0015.
   - **RealMLP HP tuning** (cycle 5): n_ens scaling, hidden size, dropout. The defaults are pre-tuned but per-task tuning could find another +0.001.
   - **Multi-input blender** (rank-remap, H-blend from the public notebook): now that we have BOTH a strong CatBoost (LB 0.95097) AND a strong NN (~LB 0.9529), the public-blender pattern becomes viable.
   - **Stacking on top of RealMLP+CatBoost OOFs** (with raw features): meta-model could find slice-specific weights that fixed-weight blends miss. The cycle-10 stacking failure was on weak inputs; with strong+diverse inputs it might work.
3. **Don't burn budget on multi-seed CatBoost** (exp 15) anymore — the CB track is dominated by RealMLP.
