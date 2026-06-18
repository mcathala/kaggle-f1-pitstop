"""Experiment 049 (cycle 16) — TabM_D_Classifier on Kaggle GPU.

Paste this single file into a Kaggle Notebook with the following inputs added:
  1. Competition data:  Playground Series — Season 6, Episode 5
                        path: /kaggle/input/playground-series-s6e5/
  2. External dataset:  <external-f1-strategy-dataset>
                        path: /kaggle/input/<external-f1-strategy-dataset>/

GPU: enable T4 x1 (or P100) in the notebook settings (Settings → Accelerator).
This script fails fast if CUDA is not actually allocated — cycle 11 exp 033
hit Kaggle's silent CPU-downgrade bug; we refuse to proceed without GPU.

Reproduces cycle 12 (exp 045) verbatim: same FE pipeline, same 5-fold CV,
same TabM library-default HPs. The only changes from `src/train_tabm.py`:
  - device forced to "cuda"
  - batch_size left at PyTabKit default (None) — T4/P100 has ample VRAM
  - paths use Kaggle's /kaggle/input/ structure

Expected runtime: ~30-60 min total on T4 (vs cycle 12's M1 Pro infra-fail).

Outputs (in /kaggle/working/):
  oof_tabm_kaggle.parquet           — OOF predictions to download
  submission_tabm_kaggle.csv        — test predictions to download
  Per-fold AUC, OOF AUC, per-year AUC all printed to the log.

After running: download the two output files, place in `data/` locally, then
compute rank-corr vs RealMLP-multiseed + CB-tuned-exp14 + XGB-highbins, and
run the 4-way blend probe (cycle-11 weights as starting point).
"""

# Print GPU driver info FIRST so we know what we're dealing with.
# Fail-fast if Kaggle silently allocated CPU (the exp 033 failure mode).
import subprocess, sys
print("=== nvidia-smi ===")
try:
    print(subprocess.check_output(["nvidia-smi"], text=True))
except Exception as e:
    print(f"nvidia-smi failed: {e}")
print("==================")

# Pin pytabkit to match local repro (1.7.3). With internet enabled, this falls
# through to PyPI if Kaggle's cache doesn't have it.
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pytabkit==1.7.3"])

# Kaggle's default PyTorch (2.10.0+cu128) dropped sm_60 kernels — on a Tesla P100
# (Pascal, sm_60) basic CUDA ops raise "no kernel image available for execution".
# Pin torch 2.5.1 (latest on the cu121 index; still ships sm_60, works on T4 sm_75).
# The cu121 wheel bundles its own CUDA runtime and is forward-compatible with the
# P100's CUDA 13 driver. Installed LAST so it wins over pytabkit's torch dep.
# torchvision must be the ABI-matched pair (0.20.1 ↔ torch 2.5.1) — pytabkit's
# pytorch_lightning → torchmetrics chain eagerly imports torchvision, which
# crashes if its compiled extension was built against a different torch ABI.
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "torch==2.5.1", "torchvision==0.20.1",
                       "--index-url", "https://download.pytorch.org/whl/cu121"])

import os
import random
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytabkit import TabM_D_Classifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

warnings.filterwarnings("ignore")

print(f"torch    version: {torch.__version__}")
print(f"pytabkit version: {version('pytabkit')}")
print(f"cuda available  : {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA not available — Kaggle silently allocated CPU. "
        "Verify Settings → Accelerator is set to T4 or P100 and re-run."
    )
print(f"cuda device     : {torch.cuda.get_device_name(0)}")
device = "cuda"

# ============================================================
# Kaggle input paths
# ============================================================

KAGGLE_INPUT = Path("/kaggle/input")


def find_one(filename: str) -> Path:
    """Locate a file anywhere under /kaggle/input/, handling both flat
    (`/kaggle/input/<slug>/`) and nested (`/kaggle/input/competitions/<slug>/`)
    mount structures."""
    hits = list(KAGGLE_INPUT.rglob(filename))
    if not hits:
        print(f"=== /kaggle/input tree (looking for {filename}) ===")
        for p in sorted(KAGGLE_INPUT.rglob("*")):
            print(f"  {p}")
        print("====================================================")
        raise FileNotFoundError(f"{filename} not found under {KAGGLE_INPUT}")
    if len(hits) > 1:
        print(f"WARN: multiple {filename} found, using first: {hits}")
    return hits[0]


TRAIN_CSV = find_one("train.csv")
TEST_CSV = find_one("test.csv")
EXTERNAL_CSV = find_one("f1_strategy_dataset.csv")
print(f"resolved TRAIN_CSV    = {TRAIN_CSV}")
print(f"resolved TEST_CSV     = {TEST_CSV}")
print(f"resolved EXTERNAL_CSV = {EXTERNAL_CSV}")

WORKING = Path("/kaggle/working")
OOF_OUT = WORKING / "oof_tabm_kaggle.parquet"
SUB_OUT = WORKING / "submission_tabm_kaggle.csv"

# ============================================================
# Config — frozen CV protocol, library-default TabM HPs
# ============================================================

TARGET = "PitNextLap"
ID_COL = "id"
N_SPLITS = 5
SPLIT_SEED = 42  # frozen across all cycles
SEED = int(os.environ.get("SEED", "42"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(SEED)

# v7 (tabm_k=8, batch=1024, lr=2e-3 default) ran fast (~2.9 min/epoch) but val AUC
# PEAKED AT EPOCH 0 (0.9436) then degraded to 0.928 — classic lr-too-high collapse
# for the lean big-batch setup. v8 fixes the training dynamics for a fair test:
#   - lr 2e-3 → 5e-4        (4x lower; let the model improve past epoch 0)
#   - batch_size 1024 → 512 (smaller batch → more, smaller steps)
#   - tabm_k 8 → 16         (restore half the ensemble heads for capacity)
#   - patience 8 → 12       (give the corrected curve room to converge)
TABM_PARAMS = {
    "random_state": SEED,
    "verbosity": 2,
    "tabm_k": 16,
    "batch_size": 512,
    "lr": 5e-4,
    "n_epochs": 100,
    "patience": 12,
    "compile_model": False,        # Triton needs sm_70+; P100 is sm_60
    "allow_amp": True,             # FP16 autocast — P100 has full-rate FP16
    "gradient_clipping_norm": 1.0,
    "val_metric_name": "1-auc_ovr",
    "n_repeats": 1,
    "device": device,
}

# Fast probe: run only fold 1 to get a representative AUC + per-fold timing,
# then decide whether the full 5-fold is worth committing GPU time to.
PROBE_FOLD1 = os.environ.get("PROBE_FOLD1", "1") == "1"


# ============================================================
# Feature engineering — verbatim from src/train_tabm.py (cycle 12)
# ============================================================

def feature_engineering(df: pd.DataFrame, fit: bool, state: dict):
    """FE pipeline shared between RealMLP-family and TabM trainers."""
    df["_LapNumber_/_RaceProgress"] = (df["LapNumber"] / (df["RaceProgress"] + 1e-6)).astype("float32")
    df["_TyreLife_/_LapNumber"] = (df["TyreLife"] / df["LapNumber"].clip(lower=1)).astype("float32")
    df["_LapTime (s)_*_Cumulative_Degradation"] = (df["LapTime (s)"] * df["Cumulative_Degradation"]).astype("float32")
    df["_LapTime (s)_*_Cumulative_Degradation_abs"] = (df["LapTime (s)"] * df["Cumulative_Degradation"].abs()).astype("float32")
    df["_LapTime (s)_/_Cumulative_Degradation_abs"] = (df["LapTime (s)"] / (df["Cumulative_Degradation"].abs() + 1e-6)).astype("float32")

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in df.select_dtypes(exclude=["object"]).columns.tolist() if c not in (ID_COL, TARGET)]

    for col in num_cols + ["_LapNumber_/_RaceProgress", "_TyreLife_/_LapNumber"]:
        cat_name = f"{col}_cat_" if col in num_cols else f"{col[1:]}_cat_"
        if fit:
            codes, uniques = np.floor(df[col]).astype(int).factorize()
            state[col] = uniques
        else:
            uniques = state[col]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = np.floor(df[col]).astype(int).map(code_map).fillna(-1).astype("int32")
        df[cat_name] = codes.astype(str)

    for col in cat_cols + ["Year_cat_", "PitStop_cat_"]:
        count_name = f"_{col}_count" if col in cat_cols else f"_{col[:-1]}_count"
        if fit:
            count_map = df[col].astype(object).value_counts()
            state[count_name] = count_map
        else:
            count_map = state[count_name]
        df[count_name] = df[col].astype(object).map(count_map).fillna(0).astype("int32")

    bin_config = {"RaceProgress": [200], "LapTime (s)": [7]}
    for col, bins_list in bin_config.items():
        for n_bins in bins_list:
            bin_name = f"{col}_{n_bins}_quantile_bin_"
            if fit:
                kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
                binned = kb.fit_transform(df[[col]]).ravel().astype("int32")
                state[bin_name] = kb
            else:
                kb = state[bin_name]
                binned = kb.transform(df[[col]]).ravel().astype("int32")
            df[bin_name] = binned.astype(str)

    combo_names: list[str] = []
    for cols in [("Race", "Compound"), ("Race", "Year")]:
        combo_name = "_".join(cols) + "_"
        combo_names.append(combo_name)
        combo_series = df[cols[0]].astype(str)
        for col in cols[1:]:
            combo_series = combo_series + "_" + df[col].astype(str)
        if fit:
            codes, uniques = pd.factorize(combo_series, sort=False)
            state[combo_name] = uniques
        else:
            uniques = state[combo_name]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = combo_series.map(code_map).fillna(-1).astype("int32")
        df[combo_name] = codes.astype(str)

    new_cat_cols = [c for c in df.columns if c.endswith("_")]
    new_num_cols = [c for c in df.columns if c.startswith("_") and not c.endswith("_")]
    return df, new_cat_cols, new_num_cols, combo_names


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("Loading data...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    orig = pd.read_csv(EXTERNAL_CSV).drop(columns=["Normalized_TyreLife"], errors="ignore")
    print(f"  train {train.shape}  test {test.shape}  orig {orig.shape}")

    y = train[TARGET].astype(int)
    y_orig = orig[TARGET].astype(int)
    train_id = train[ID_COL]
    test_id = test[ID_COL]

    X = train.drop([ID_COL, TARGET], axis=1)
    X_test = test.drop([ID_COL], axis=1)
    X_orig = orig.drop([TARGET], axis=1)

    print("Applying FE...")
    state: dict = {}
    X, new_cat_cols, new_num_cols, combo_names = feature_engineering(X, fit=True, state=state)
    X_test, _, _, _ = feature_engineering(X_test, fit=False, state=state)
    X_orig, _, _, _ = feature_engineering(X_orig, fit=False, state=state)
    print(f"  X      shape: {X.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  X_orig shape: {X_orig.shape}")

    strat_key = train["Year"].astype(str) + "_" + train[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_SEED)
    oof = np.zeros(len(train), dtype=np.float64)
    test_preds = np.zeros(len(test), dtype=np.float64)
    fold_aucs: list[float] = []

    kf_orig = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_SEED)
    orig_splits = list(kf_orig.split(X_orig, y_orig))

    print(f"\nTabM params: {TABM_PARAMS}\n")

    for fold, ((tr_idx, va_idx), (or_tr_idx, or_va_idx)) in enumerate(
        zip(kf.split(X, strat_key), orig_splits), start=1
    ):
        t0 = time.time()
        X_tr = pd.concat([X.iloc[tr_idx], X_orig.iloc[or_tr_idx]], axis=0).reset_index(drop=True)
        y_tr = pd.concat([y.iloc[tr_idx], y_orig.iloc[or_tr_idx]], axis=0).reset_index(drop=True)
        X_va = X.iloc[va_idx].copy()
        y_va = y.iloc[va_idx]

        te_cols = combo_names
        te = TargetEncoder(cv=N_SPLITS, smooth="auto", shuffle=True, random_state=SPLIT_SEED)
        tr_enc = te.fit_transform(X_tr[te_cols], y_tr)
        va_enc = te.transform(X_va[te_cols])
        tst_enc = te.transform(X_test[te_cols])
        te_names = [f"_{c}TE" for c in te_cols]
        X_tr[te_names] = tr_enc
        X_va[te_names] = va_enc
        X_tst = X_test.copy()
        X_tst[te_names] = tst_enc

        if fold == 1:
            print(f"  fold 1: train rows = {len(X_tr):,}  features = {X_tr.shape[1]}  (cat + num + TE)")

        model = TabM_D_Classifier(**TABM_PARAMS)
        model.fit(X_tr, y_tr, X_va, y_va)
        va_pred = model.predict_proba(X_va)[:, 1]
        tst_pred = model.predict_proba(X_tst)[:, 1]

        oof[va_idx] = va_pred
        test_preds += tst_pred / N_SPLITS
        fold_auc = roc_auc_score(y_va, va_pred)
        fold_aucs.append(fold_auc)
        fold_secs = time.time() - t0
        print(
            f"fold {fold}/{N_SPLITS}  AUC={fold_auc:.5f}  train_rows={len(X_tr):,}  ({fold_secs:.1f}s)",
            flush=True,
        )
        torch.cuda.empty_cache()

        if PROBE_FOLD1:
            print(f"\n=== FOLD-1 PROBE ===")
            print(f"fold-1 AUC = {fold_auc:.5f}  wall = {fold_secs/60:.1f} min")
            print(f"  (vs RealMLP fold-1 ~0.95421, XGB-highbins fold-1 ~0.95331)")
            print(f"  (vs project blend-floor 0.949, Δ = {fold_auc - 0.949:+.5f})")
            print(f"extrapolated full 5-fold wall ≈ {5*fold_secs/60:.0f} min")
            print("If fold-1 AUC is competitive (≳0.952), re-run with PROBE_FOLD1=0 for full OOF.")
            return

    oof_auc = roc_auc_score(y, oof)
    print()
    print(f"per-fold AUC: mean={np.mean(fold_aucs):.5f}  std={np.std(fold_aucs):.5f}  "
          f"min={np.min(fold_aucs):.5f}  max={np.max(fold_aucs):.5f}")
    print(f"OOF AUC:      {oof_auc:.5f}")
    print(f"  (vs cycle-11 XGB-highbins 0.95263, Δ = {oof_auc - 0.95263:+.5f})")
    print(f"  (vs cycle-13 LGB-highbins 0.94885, Δ = {oof_auc - 0.94885:+.5f})")
    print(f"  (vs project blend-floor 0.949, Δ = {oof_auc - 0.949:+.5f})")

    # ============================================================
    # Per-year AUC (qualification diagnostic — see project.md per-cell guard)
    # ============================================================
    print()
    print("OOF AUC by Year:")
    oof_df = pd.DataFrame(
        {"id": train_id, "Year": train["Year"], "target": y.to_numpy(), "oof": oof}
    )
    for year, g in oof_df.groupby("Year"):
        if g["target"].nunique() > 1:
            auc = roc_auc_score(g["target"], g["oof"])
            print(f"  {year}: AUC={auc:.5f}  n={len(g):,}  pos_rate={g['target'].mean():.4f}")

    # ============================================================
    # Save outputs
    # ============================================================
    oof_df.to_parquet(OOF_OUT, index=False)
    sub = (
        pd.DataFrame({"id": test_id, TARGET: test_preds})
        .sort_values("id")
        .reset_index(drop=True)
    )
    sub.to_csv(SUB_OUT, index=False)
    print(f"\nwrote {OOF_OUT.name}  ({len(oof_df):,} rows)")
    print(f"wrote {SUB_OUT.name}  ({len(sub):,} rows)")
    print("\nDownload both files from the Kaggle Notebook sidebar (Output tab) and "
          "place in `data/` locally for the 4-way blend probe.")


if __name__ == "__main__":
    main()
