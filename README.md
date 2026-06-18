# F1 Pit-Stop Prediction — Kaggle Playground S6E5

Kaggle **Playground Series S6E5**: one row is one lap of one Formula-1 driver, and the task is to predict
whether that driver pits on the **next** lap — a yes/no over a synthetic, noise-heavy season, scored on
**ROC-AUC** across 3,023 teams.
Competition: <https://www.kaggle.com/competitions/playground-series-s6e5>

**What we did, in one line:** a 7-model blend anchored by a tuned tabular **neural network** (RealMLP), with
**gradient-boosted trees** (XGBoost, CatBoost) filling the residual — the whole pipeline trained on a laptop
(M1 Pro · 16 GB), no paid cloud.

> **📊 Full visual case study → [`case-study.html`](https://mcathala.github.io/kaggle-f1-pitstop/case-study.html)** (click to open) · plain-text
> companion: [`case-study.md`](case-study.md).

## Result

**Top 10% of 3,023 teams** — private-LB AUC **0.95427**, **+0.0122** over a 0.94211 baseline, across 88 logged
experiments on free compute.

| | AUC | Notes |
|---|---|---|
| Baseline (LightGBM, 49 feats) | 0.94211 | reference |
| **Final ensemble** | **0.95427** | private LB · **top 10%** of 3,023 teams |
| Lift over baseline | **+0.0122** | across 88 logged experiments |

> Effective top-10% after collapsing a 260-team bloc that all submitted one identical copied-notebook score
> (0.95454) — independent models never tie to five decimals. The full leaderboard breakdown is in the
> [case study](case-study.html), and is regenerable with [`scripts/plot_leaderboard.py`](scripts/plot_leaderboard.py).

The two submissions locked for scoring were chosen for **transfer-robustness, not raw out-of-fold (OOF)
score** — a deliberate guard against the OOF-overfitting the project diagnosed. The selected blend was the
single best-transferring submission produced; the highest-OOF blend would have scored worse.

## Documentation

Start with the case study, then go as deep as you like:

| Doc | What's in it |
|---|---|
| [`case-study.html`](case-study.html) · [`case-study.md`](case-study.md) | The full story, baseline → final, with the reasoning behind every step |
| [`REPRODUCE.md`](REPRODUCE.md) | Exact lineage + commands to rebuild the two final submissions |
| [`experiments/README.md`](experiments/README.md) | The 88-experiment log — one write-up per hypothesis — plus the leaderboard |
| [`docs/eda.md`](docs/eda.md) | Exploratory analysis: target structure, the synthetic-noise finding, the split |
| [`docs/feature_engineering.md`](docs/feature_engineering.md) | The 49 engineered features and what each one buys |
| [`docs/plateau_analysis.md`](docs/plateau_analysis.md) | Why the score plateaus — the ~32% label-noise ceiling |
| [`docs/transfer_analysis.md`](docs/transfer_analysis.md) | The OOF→LB transfer function and how the final submissions were picked |
| [`src/README.md`](src/README.md) | The ~12-script final pipeline |

## How it got there (one-paragraph version)

A LightGBM baseline was diversified with CatBoost, then redesigned with external data + tuned
hyperparameters. The largest single gain came from switching model family to a tuned tabular neural net
(**RealMLP / PyTabKit**), which became the ensemble anchor. XGBoost (with high-resolution histograms),
feature-stripping ("diffFE"), pseudo-labeling, and feature-view diversity added the rest. Past ~0.954 the
work shifted to *understanding the ceiling*: an OOF→LB transfer function (`LB ≈ 0.927·OOF + 0.069`) and an
exhaustive finding that no model can be both strong **and** decorrelated on this data — ~32% synthetic label
noise forces every good model onto the same ordering.

## Repo map

```
case-study.html         Visual case study (single page, no build step)
case-study.md           Plain-text companion to the case study
REPRODUCE.md            Exact lineage + commands for the two final submissions
reproduce.sh            Rebuild both submissions end-to-end
src/                    The final pipeline (~12 scripts) — see src/README.md
  research/             The ~95 exploratory scripts behind the experiment log
experiments/            88 experiment write-ups (one per hypothesis) + leaderboard in experiments/README.md
docs/                   EDA, feature notes, plateau & OOF→LB transfer analysis
notebooks/              EDA notebook (eda.ipynb)
gpu-kernels/            Kaggle GPU kernels (1 feeds the final blend)
scripts/                Data fetch + leaderboard plot
data/                   Committed reproduction inputs (raw sets + unregenerable GPU / pseudo-label prereqs)
```

## Reproduce

```bash
uv sync                          # deps (Python ≥ 3.12); or: pip install -e .
bash reproduce.sh                # data is committed → rebuilds BOTH submissions (lineage in REPRODUCE.md)
```

The reproduction inputs are **committed under `data/`** — the raw competition + external sets, plus
the Kaggle-GPU base output that can't be regenerated locally; all generated artifacts stay
gitignored. The whole feature pipeline lives in one module, [`src/data.py`](src/data.py); everything
under `src/research/` is the research log, not the pipeline. One base of the primary blend was
trained on a free Kaggle GPU (its output is committed — see [`REPRODUCE.md`](REPRODUCE.md));
`bash scripts/get_data.sh` refetches the competition CSVs if you ever need to.

A public external F1 strategy dataset augments training in the CatBoost/RealMLP phases — **raw rows
only, training folds only**; validation and test stay competition-only and leak-checked (see
[`docs/dataset_review_no_leak.md`](docs/dataset_review_no_leak.md)).

## Method notes

- **CV:** 5-fold `StratifiedKFold` on `Year × PitNextLap`, mirroring the row-level public/private split.
  AUC is computed on concatenated OOF predictions, not a mean of per-fold scores.
- **No leakage / no magic feature:** the target is synthetic and noised — a naive "did they really pit next lap"
  predictor scores only **AUC 0.547**, and only **~30%** of pit labels even coincide with a real tyre change. The
  work is honest extraction near a noise ceiling.
- **Every input is self-trained** — no external submission files were blended in.
