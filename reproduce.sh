#!/usr/bin/env bash
# Rebuild the two final submissions. Full lineage + the two manual prerequisites
# (Kaggle-GPU bases and pseudo-labels) are documented in REPRODUCE.md.
#
#   data/submission_blend_best.csv         private 0.95427  (primary)
#   data/submission_realmlp_multiseed.csv  private 0.95371  (hedge)
#
# Everything here lives in src/ root; src/research/ is the research log, not the pipeline.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python}"   # set PY=.venv/bin/python to use the project venv

# Raw inputs must be present first. Fetch them with scripts/get_data.sh (see REPRODUCE.md §0).
# The external file is raw rows only; all feature engineering happens below in our pipeline.
for f in data/train.csv data/test.csv data/f1_strategy_dataset.csv; do
  [ -f "$f" ] || { echo "Missing $f — run 'bash scripts/get_data.sh' first (see REPRODUCE.md §0)."; exit 1; }
done

echo "==> 0. Build the shared feature parquets"
$PY src/features.py

echo "==> 1. Hedge: RealMLP multi-seed (5 seeds) -> averaged submission"
for s in 42 7 99 137 313; do SEED=$s $PY src/train_realmlp_seed.py; done
$PY src/blend_realmlp_multiseed.py     # -> data/submission_realmlp_multiseed.csv

echo "==> 2. Primary-blend bases (local, M1/MPS)"
$PY src/train_xgb_diffFE.py            # diffxgb
$PY src/train_cb_diffFE.py             # cbdiff
$PY src/train_realmlp_diffFE.py        # diffrm1 (seed 42)
$PY src/train_realmlp_diffFE_6seed.py  # diffrm6
$PY src/train_xgb_robust.py            # robust (contender pool)

# --- pseudo-labeled bases: need pseudo-labels from an earlier-round blend (REPRODUCE.md, prereq #2) ---
$PY src/train_xgb_diffFE_pseudo.py     # diffpsxgb
$PY src/train_cb_pseudo_exp14.py       # pscb14

# --- GPU bases (REPRODUCE.md, prereq #1): run on Kaggle, drop the outputs into data/ ---
#   gpu-kernels/cycle17_realmlp_pseudo62_gpu.py  -> oof_realmlp_pseudo62.parquet  (psrm6r2, largest weight)
#   gpu-kernels/cycle17_xgb_pseudo2_gpu.py       -> oof_xgb_pseudo2.parquet       (psxgb)

echo "==> 3. Primary: transfer-robust blend -> submission_blend_best.csv"
$PY src/build_contenders.py            # writes data/submission_blend_best.csv (the 'best' contender)

echo
echo "Done. Final submissions:"
echo "  data/submission_blend_best.csv          private 0.95427  (primary)"
echo "  data/submission_realmlp_multiseed.csv   private 0.95371  (hedge)"
