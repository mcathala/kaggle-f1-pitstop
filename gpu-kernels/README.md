# GPU kernels — Kaggle CUDA notebooks

Single-file, paste-into-Kaggle **GPU kernel scripts** (`cycle*_gpu.py`). Each is the runnable Kaggle-notebook
version of one experiment that needed a CUDA GPU.

## Why only some cycles appear here

Most of the modelling ran locally on an M1 Pro (CatBoost, LightGBM, XGBoost, and RealMLP via MPS), so those
cycles have **no** kernel here — their code is in [`../src/`](../src/) and the write-ups are in
[`../experiments/`](../experiments/). A Kaggle kernel exists only when an experiment needed a CUDA GPU: the
neural-net / foundation-model attempts and a handful of GPU-XGB runs. That's why the files cluster around
cycles 16–17 and skip the rest.

## Kernel → experiment map

| Kernel | Experiment |
|---|---|
| `cycle11_kaggle_gpu.py` | [033 — CatBoost class-weights on GPU](../experiments/033_cb_class_weights_gpu.md) (infra-fail: GPU not allocated) |
| `cycle16_kaggle_gpu.py` | [049 — TabM on Kaggle P100](../experiments/049_tabm_kaggle_gpu.md) |
| `cycle16_xgb_noext_gpu.py` | [056 — no-external ablation](../experiments/056_noext_ablation.md) |
| `cycle16_xgb_tyreoverdue_gpu.py` | [052 — tyre-overdue FE on XGB](../experiments/052_xgb_tyre_overdue.md) |
| `cycle16_xgb_advweight_gpu.py` | [057 — adversarial-weighted external XGB](../experiments/057_xgb_advweight.md) |
| `cycle17_lap_attention_gpu.py` | [058 — lap-attention (feasibility)](../experiments/058_lap_attention.md) |
| `cycle17_lap_attention_v2_gpu.py` | [059 — lap-attention v2](../experiments/059_lap_attention_v2.md) |
| `cycle17_lap_attention_v3_gpu.py` | [060 — lap-attention v3](../experiments/060_lap_attention_v3.md) |
| `cycle17_autogluon_stack_gpu.py` | [062 — AutoGluon with stacking](../experiments/062_autogluon_stack.md) |
| `cycle17_xgb_pseudolabel_gpu.py` | [063 — XGB pseudo-labeling](../experiments/063_xgb_pseudolabel.md) |
| `cycle17_realmlp_pseudo_gpu.py` | [064 — RealMLP pseudo-labeling](../experiments/064_realmlp_pseudolabel.md) |
| `cycle17_realmlp_pseudo6_gpu.py` | [065 — RealMLP 6-seed pseudo](../experiments/065_realmlp_pseudo6.md) |
| `cycle17_xgb_pseudo2_gpu.py` | [067 — round-2 XGB pseudo](../experiments/067_xgb_pseudo_round2.md) |
| `cycle17_realmlp_pseudo62_gpu.py` | [069 — round-2 RealMLP pseudo](../experiments/069_realmlp_pseudo_r2.md) · [072 — round-3](../experiments/072_realmlp_pseudo_r3.md) |

Each kernel is self-contained: it reads the competition data (plus the external strategy dataset) from the
Kaggle input mounts, runs the experiment's CV protocol, and writes OOF + test predictions back out for
blending locally.
