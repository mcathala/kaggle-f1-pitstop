# `src/` — code index

This directory is split in two:

- **`src/` (root)** — the **final pipeline**: the ~12 scripts that produce the two submitted models. Start here.
- **[`src/research/`](research/)** — the **research log**: the ~95 exploratory scripts behind the [88 experiment write-ups](../experiments/). Kept for completeness; not needed to reproduce the result.

> Convention: `train_*` builds a base model and writes an out-of-fold (OOF) parquet + a test submission; `blend_*` / `build_*` combine bases.

## The final pipeline (this directory)

Run order and the full lineage are in **[`../REPRODUCE.md`](../REPRODUCE.md)** (one command: `bash ../reproduce.sh`).

| Script | Role |
|---|---|
| `features.py` | the shared engineered-feature pipeline |
| `train.py` | LightGBM baseline — the reference (**public LB 0.94211**) |
| `train_xgb_diffFE.py` | diffFE XGBoost base |
| `train_xgb_diffFE_pseudo.py` | diffFE XGBoost + pseudo-labels |
| `train_cb_diffFE.py` | diffFE CatBoost base |
| `train_cb_pseudo_exp14.py` | pseudo-labeled CatBoost (exp-14 recipe) |
| `train_xgb_robust.py` | noise-robust (GCE) XGBoost base |
| `train_realmlp_diffFE.py` | diffFE RealMLP (seed 42) |
| `train_realmlp_diffFE_6seed.py` | diffFE RealMLP, 6-seed average |
| `train_realmlp_seed.py` | RealMLP single-seed (`SEED` env) — base of the hedge |
| `blend_realmlp_multiseed.py` | averages the seeds → **hedge** (`submission_realmlp_multiseed.csv`, 0.95371) |
| `build_contenders.py` | the transfer-robust blend → **primary** (`submission_blend_best.csv`, 0.95427) |

One base of the primary blend (the pseudo-RealMLP) was trained on a Kaggle GPU — see [`../gpu-kernels/README.md`](../gpu-kernels/README.md).

## The research log (`src/research/`)

The ~95 scripts there trace the full journey; each maps to a write-up in [`../experiments/`](../experiments/). The arc, by phase:

1. **Baseline & first diversity** — `train_catboost.py`, error-bucket EDA.
2. **Tuned-CatBoost redesign** (→ 0.95066) — `train_cb_tuned*.py`, stacking probes (ensemble axis saturated).
3. **Model-family pivot** (largest jump, → 0.95331) — `train_realmlp.py`; other NN families (`train_ftt.py`, `train_embmlp*.py`, `train_tabm.py`) too weak/correlated to add.
4. **XGBoost high-bin diagnosis** (→ 0.95372) — `train_xgb_richcat.py` (`max_bin=5000`), `optuna_xgb_highbins.py`.
5. **Blend-space mapping** — `blend_*` weight/rank/quantile/operator probes, `build_greedy_blend.py`, `meta_stack_context.py`.
6. **Feature-engineering probes** — `train_lgb_diffFE.py`, targeted FE (mostly reverted).
7. **Pseudo-labeling & robustness** — self-training rounds, noise-aware / self-distill probes.
8. **Plateau analysis & selection** — `decorr_analysis.py`, `build_drift_candidates.py`; written up in [`../docs/transfer_analysis.md`](../docs/transfer_analysis.md) and [`../docs/plateau_analysis.md`](../docs/plateau_analysis.md).
