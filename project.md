# project.md — methodology, CV protocol & cycle log

Terminology, the CV protocol, the acceptance gates every result is measured against, and the full cycle
history. Per-experiment detail lives in `experiments/NNN_*.md`; the narrative is in
[`case-study.md`](case-study.md).

## Terminology

- **Experiment** — a single hypothesis test. One file `experiments/<NNN>_<slug>.md`. May or may not change the model. Has Kept / Reverted / Inconclusive / Infra-fail verdict. Numbered sequentially (001, 002, …).
- **Cycle** — a phase of work that groups multiple experiments and is closed by a successful Kaggle submission. The submission file becomes the new baseline for the next cycle.
- **Session** — one focused working block. A session may span multiple experiments within a cycle, or span cycles.

The `experiments/` directory is the experiment log. Cycles are tracked in the **Cycle history** section below.

## Cycle history

| Cycle | Experiments | Closing submission | Public LB |
|---|---|---|---|
| **Cycle 1** | (baseline build) | `submission_baseline.csv` (LGB 63-feat) | **0.94211** |
| **Cycle 2** | exp 001–006 | 3-way ensemble (LGB + CB#004 + CB#006) | **0.94833** |
| **Cycle 3** | exp 007–013 | 4-way `3way_focus` (LGB + CB#006 + CB-tuned + external data) | **0.95066** |
| **Cycle 4** | exp 014, 016 | RealMLP via PyTabKit, standalone (the structural pivot) | **0.95331** |
| **Cycle 5** | exp 017, 018 | `submission_realmlp_multiseed.csv` (RealMLP avg of 5 seeds) | **0.95342** |
| Cycle 6 (Inconclusive — ceiling probe) | exp 019, 020 | (no submission — 3 paths all failed) | unchanged |
| **Cycle 7** | exp 021, 022 | RealMLP-multiseed × 0.80 + CB-tuned-exp14 × 0.20 | **0.95361** |
| Cycle 8 (Inconclusive — model zoo exhausted) | exp 023, 024, 025 | (no submission) | unchanged |
| Cycle 9 (Inconclusive — pseudo + year-specialist both dead) | exp 026, 027 | (no submission) | unchanged |
| Cycle 10 (Inconclusive — CB axis exhausted) | exp 028–032 | (no submission) | unchanged |
| **Cycle 11** | exp 033–044 (**044: XGBoost retune w/ max_bin=5000 — Kept**) | RM-6seed × 0.675 + CB-tuned-exp14 × 0.075 + XGB-highbins × 0.250 | **0.95372** |
| Cycle 12 (Infra-fail — TabM untestable on M1 Pro/MPS; later run on GPU as exp 049) | TabM_D_Classifier | (no submission) | unchanged |
| Cycle 13 (Inconclusive — LGB-highbins w=0 in blend) | exp 046 | (no submission) | unchanged |
| Cycle 14 (Inconclusive — XGB multi-seed near-deterministic ρ=0.999) | exp 047 | (no submission) | unchanged |
| Cycle 15 (Inconclusive — Optuna XGB plateau; cycle-11 HPs at local optimum) | exp 048 | (no submission) | unchanged |
| Cycle 16 (Inconclusive — own-model, FE-on-XGB, and external-data axes all closed) | exp 049–057 | (no submission) | unchanged |
| Cycle 17 (Inconclusive — diverse-NN, blend-combiner, AutoML, pseudo-labeling closed) | exp 058–064 | `submission_blend_pseudo.csv` | **0.95373** |
| **Final push** | exp 065–092 | `submission_blend_best.csv` — transfer-robust 7-model blend (RealMLP × XGBoost × CatBoost on a lean **diffFE** feature set) | **0.95402** (public) |

Final shipped submission: public LB **0.95402**, **private 0.95427** — the transfer-robust 7-model blend. The
closing push (exp 065–092) added pseudo-labeling round 2, **diffFE** feature-stripping (XGBoost 0.95263 →
0.95291), and feature-view diversity, lifting the public LB from 0.95373 to 0.95401–0.95402. Cumulative lift
across the project: **0.94211 → 0.95427 = +0.01216**. A higher-OOF greedy blend (OOF 0.95479) was deliberately
**not** selected — it scored lower on both boards (public 0.95380, private 0.95408), confirming the
transfer-robust pick.

## Project
- Name: kaggle-f1-pitstop
- Goal: binary classification — predict `PitNextLap` (probability the driver pits on the next lap)
- Metric: ROC-AUC (lower-better: **false**)
- Competition: https://www.kaggle.com/competitions/playground-series-s6e5 (deadline 2026-05-31)

## Data
- Train (raw): `data/train.csv` (439,140 × 16)
- Test (raw): `data/test.csv` (188,165 × 15)
- External (cycle 3+): `data/f1_strategy_dataset.csv` (101,371 rows, 16 cols) — raw rows only, added to training folds; validation and test stay competition-only and leak-checked.
- Target column: `PitNextLap`
- Group/time column for CV: **none** — the split is row-level random, not group-level (see `docs/eda.md §8`). CV is stratified on `Year × PitNextLap` to mirror the public/private split.
- Data version / hash:
  - `data/train.csv` sha256 = `f004e79d4e63f4bad0afc3788d07938dc5dbc0bed73a51e7b65cbce52ccc4128` (pinned 2026-05-21)
  - `data/test.csv`  sha256 = `95b449a8af322ae3c88793f5aa1f17fcadfc6b492a9fc029772f02e0140c2ea7`
  - `data/` is committed (raw inputs + the unregenerable GPU / pseudo-label prerequisites). Refetch the competition CSVs with `bash scripts/get_data.sh` if needed. If the train.csv hash changes, treat it as a new data version and re-run the baseline.

## CV
- Strategy: `StratifiedKFold(shuffle=True, random_state=42)` on the composite key `f"{Year}_{PitNextLap}"`. Mirrors the row-level split; do **not** GroupKFold by `(Race, Year, Driver)`.
- n_splits: 5
- Random seed: 42
- Robustness seeds (for periodic re-validation): [7, 99]
- AUC is computed on concatenated OOF predictions, not the mean of per-fold AUCs.
- Test predictions = mean of fold models, each at its own `best_iteration`.

## Thresholds (acceptance gates)
- min_delta: **0.00020** (≈ 0.5 × current baseline fold std 0.00042; matches the LB↔OOF noise envelope of ±0.0005).
- max_gap_increase_relative: 0.20
- max_gap_increase_absolute: 0.0005 (LB tracks OOF within 0.0005; any larger gap is a generalization red flag).
- robustness_check_every: 10 cycles
- Per-cell guard (year × compound, n ≥ 10K): no cell may regress by more than 0.0010. This caught experiment 3's INTER blow-up retroactively.

## Conventions
- Source dir: `src/` (final pipeline at the root; `src/research/` is the research log, not the pipeline).
- Outputs dir: `data/`.
- Models dir: none — fold models are not persisted; OOF and submission parquets/CSVs are the artifacts.
- Frozen files (encode the original CV protocol; changing them invalidates comparison to prior numbers): `src/train.py`, `src/research/train_catboost.py`. `src/features.py` is the per-experiment change surface for tree features; `src/research/train_cb_tuned.py` is the cycle-3+ trainer with inline FE.
- Evaluation entry points:
  - `python src/features.py` — (re)build feature parquets.
  - `python src/train.py` — LGB baseline.
  - `python src/research/train_catboost.py` — CatBoost (cycle-2 frozen params).
  - `python src/research/train_cb_tuned.py` — CB-tuned with external data + tuned HPs (cycle 3+).
