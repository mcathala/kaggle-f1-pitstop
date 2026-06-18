# Reproducing the best model

This walks the whole path from raw data to the **best submitted model** — the transfer-robust
blend that scored **private LB 0.95427**. A second, simpler model (the RealMLP hedge) is
documented at the end but kept out of the main path to keep it readable.

```bash
uv sync            # deps (Python ≥ 3.12); or: pip install -e .
bash reproduce.sh  # raw data → 7 bases → blend → data/submission_blend_best.csv
```

That's it. Everything the blend needs is committed under `data/` (see §1), so there's no
download step for the main path. One base was trained on a free Kaggle GPU — its output
is committed too, so the blend reproduces locally; §4 explains how to rebuild it from scratch.

| Model | File | OOF | Public | **Private** |
|---|---|---:|---:|---:|
| **Best** — transfer-robust blend | `data/submission_blend_best.csv` | 0.95462 | 0.95402 | **0.95427** |
| Hedge — RealMLP multi-seed | `data/submission_realmlp_multiseed.csv` | 0.95383 | 0.95342 | **0.95371** |

---

## 1. The data

Two raw inputs live in `data/` (committed):

| Source | Files | What it is |
|---|---|---|
| Competition — [playground-series-s6e5](https://www.kaggle.com/competitions/playground-series-s6e5/data) | `train.csv`, `test.csv`, `sample_submission.csv` | the labelled task |
| External F1-strategy augmentation | `f1_strategy_dataset.csv` | **raw augmentation rows only** — +101k laps, same 16-column host schema |

> Both files are committed, so reproduction is bit-exact and needs no downloads. You can re-fetch the
> competition CSVs with `bash scripts/get_data.sh` if you ever need to.

**The external set contributes rows, not features.** Every engineered column is built by our
pipeline, not taken from the file — and it joins **training folds only**: validation and test stay
competition-only, so cross-validation is an honest proxy for the LB (leak-checked,
[`docs/dataset_review_no_leak.md`](docs/dataset_review_no_leak.md)).

## 2. Raw → features: `src/data.py`

All feature engineering lives in one readable module, [`src/data.py`](src/data.py) — the single
answer to *"what data trained this and how was it built?"* It exposes the two recipes the blend uses:

- `build_gbdt_diffFE()` — the XGBoost / CatBoost view: domain features + base-category frequency
  (49 features). The over-engineered cross-cats and group-stats are deliberately **stripped**
  ("diffFE"), which made each tree model stronger and the blend more transfer-robust.
- `realmlp_features()` — the RealMLP view: ratio features, count encodings, KBins quantile bins,
  and target-encoded Race×Compound / Race×Year combos.

Every trainer imports from here, so the data story is defined once.

## 3. The blend: 7 bases, two-thirds RealMLP

`reproduce.sh` trains each base (writing an out-of-fold parquet + a test submission), then
`src/build_contenders.py` combines them with fixed, OOF-tuned weights → `submission_blend_best.csv`.

| Weight | Base | Built by |
|---:|---|---|
| 0.315 | pseudo-RealMLP, 6-seed (round 2) | [`gpu-kernels/cycle17_realmlp_pseudo62_gpu.py`](gpu-kernels/cycle17_realmlp_pseudo62_gpu.py) · **GPU** |
| 0.293 | diffFE-RealMLP, 6-seed | `src/train_realmlp_diffFE_6seed.py` |
| 0.242 | diffFE-XGBoost + pseudo | `src/train_xgb_diffFE_pseudo.py` · pseudo |
| 0.058 | diffFE-RealMLP (seed 42) | `src/train_realmlp_diffFE.py` |
| 0.036 | pseudo-CatBoost (exp-14 recipe) | `src/train_cb_pseudo_exp14.py` · pseudo |
| 0.035 | diffFE-CatBoost | `src/train_cb_diffFE.py` |
| 0.020 | diffFE-XGBoost | `src/train_xgb_diffFE.py` |

RealMLP (a tuned tabular neural net) is strictly strongest and anchors the blend at two-thirds; the
trees fill the residual. Weights are found by constrained coordinate descent on the OOF matrix, then
frozen — OOF is the only honest surface while test labels stay hidden.

## 4. The two honest caveats

Reproduction is one command for the local part, but two pieces are inherently manual:

1. **One base needs a Kaggle GPU.** The pseudo-RealMLP (largest weight, 0.315) was trained on a
   Kaggle P100/T4. **Its output is committed**, so `reproduce.sh` builds the blend without a GPU.
   To rebuild it from scratch, run
   [`gpu-kernels/cycle17_realmlp_pseudo62_gpu.py`](gpu-kernels/cycle17_realmlp_pseudo62_gpu.py) on
   Kaggle and drop its output into `data/` (see [`gpu-kernels/README.md`](gpu-kernels/README.md)).
2. **The pseudo-labeled bases are semi-supervised.** They train on competition + external +
   high-confidence test pseudo-labels taken from an earlier-round blend
   (`data/submission_blend_pseudo_r2.csv`, committed). The mechanism and its leak-clean round-2 fix
   are in [experiments 063–069](experiments/063_xgb_pseudolabel.md).

## 5. Why this model was selected

The two locked submissions were chosen for **transfer-robustness, not raw OOF** — a deliberate guard
against the OOF-overfitting this project diagnosed. On the private split the reasoning held: the
selected blend was the single best-transferring submission produced, and the highest-OOF greedy blend
(which we did *not* pick) scored worse. Full reasoning in
[`docs/plateau_analysis.md`](docs/plateau_analysis.md) and [`docs/transfer_analysis.md`](docs/transfer_analysis.md).

## The hedge (0.95371)

A decorrelated, single-model insurance pick. `src/train_realmlp_seed.py` (5 seeds: 42, 7, 99, 137,
313) averaged by `src/blend_realmlp_multiseed.py` → `data/submission_realmlp_multiseed.csv`.
`reproduce.sh` builds it too.
