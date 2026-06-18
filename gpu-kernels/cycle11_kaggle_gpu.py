"""Experiment 033 (cycle 11) — Kaggle Notebook GPU script.

Paste this single file into a Kaggle Notebook with the following inputs added:
  1. Competition data:  Playground Series — Season 6, Episode 5
                        path: /kaggle/input/playground-series-s6e5/
  2. External dataset:  <external-f1-strategy-dataset>
                        path: /kaggle/input/<external-f1-strategy-dataset>/

GPU: enable T4 x1 in the notebook settings (Settings → Accelerator).

The script reproduces cycle 14's CB-tuned-exp14 FE recipe verbatim and trains
CatBoost on GPU with the ONE HP change being tested in exp 033:

    auto_class_weights="Balanced"   →   class_weights=[1.0, 1.5]

Expected runtime: ~60 min total on T4 (vs ~3 h on M1 Pro CPU).

Outputs (in /kaggle/working/):
  oof_cb_classweighted_gpu.parquet  — OOF predictions to download
  submission_cb_classweighted_gpu.csv  — test predictions to download
  Per-fold AUC, OOF AUC, bin-8 calibration bias all printed to the log.

After running: download the two output files, place in `data/` locally, then
report back here with the printed OOF AUC + per-fold + bin-8 bias.
"""

# Print GPU driver info FIRST so we know what we're dealing with.
import subprocess, sys
print("=== nvidia-smi ===")
try:
    print(subprocess.check_output(["nvidia-smi"], text=True))
except Exception as e:
    print(f"nvidia-smi failed: {e}")
print("==================")

# Pin CatBoost to match local repro (1.2.10). With internet enabled, this falls
# through to PyPI if Kaggle's cache doesn't have it.
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost==1.2.10"])

from pathlib import Path
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

print(f"catboost version: {__import__('catboost').__version__}")

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
OOF_OUT = WORKING / "oof_cb_classweighted_gpu.parquet"
SUB_OUT = WORKING / "submission_cb_classweighted_gpu.csv"

TARGET = "PitNextLap"
ID_COL = "id"
BASE_CATS = ["Driver", "Race", "Compound"]
N_SPLITS = 5
SEED = 42

# ============================================================
# Hyperparameters — cycle 14 recipe with ONE change
# ============================================================

CB_PARAMS = {
    "iterations": 8000,
    "learning_rate": 0.018,
    "depth": 8,
    "l2_leaf_reg": 8.5,
    "random_strength": 0.65,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 0.45,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    # === THE EXPERIMENTAL CHANGE ===
    # Cycle 14 used auto_class_weights="Balanced" (~3.8x positive upweight on a 21% pos dataset).
    # Probe 5 (exp 030) showed this caused +0.057 bin-8 over-prediction.
    # Exp 032 showed full removal hurts AUC by ~0.001 (Balanced was load-bearing).
    # Exp 033 tests the middle: mild explicit [1.0, 1.5] upweight.
    "class_weights": [1.0, 1.5],
    # ================================
    "early_stopping_rounds": 500,
    "task_type": "GPU",
    "devices": "0",
    "allow_writing_files": False,
    "verbose": 250,
}


# ============================================================
# Feature engineering — verbatim from src/train_cb_tuned_exp14.py
# ============================================================

def safe_div(a, b, eps=1e-6):
    return a / (b + eps)


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6
    out = df.copy()

    race_progress = out["RaceProgress"].clip(lower=eps)
    out["EstimatedTotalLaps"] = (out["LapNumber"] / race_progress).clip(1, 120).astype("float32")
    out["LapsRemaining"] = (out["EstimatedTotalLaps"] - out["LapNumber"]).clip(lower=0).astype("float32")
    out["RemainingRaceProgress"] = (1.0 - out["RaceProgress"]).astype("float32")
    out["LapProgress_x_LapNumber"] = (out["LapNumber"] * out["RaceProgress"]).astype("float32")

    out["RacePhase"] = pd.cut(
        out["RaceProgress"], bins=[-np.inf, 0.20, 0.40, 0.60, 0.80, np.inf],
        labels=["P1", "P2", "P3", "P4", "P5"],
    ).astype(str)
    out["LapBin"] = pd.cut(
        out["LapNumber"], bins=[-np.inf, 5, 10, 20, 35, 50, np.inf],
        labels=["L005", "L010", "L020", "L035", "L050", "Lplus"],
    ).astype(str)

    out["TyreAgeRatio"] = safe_div(out["TyreLife"], out["LapNumber"].clip(lower=1), eps).astype("float32")
    out["LapPerTyreLife"] = safe_div(out["LapNumber"], out["TyreLife"] + 1, eps).astype("float32")
    out["TyreLife_x_RaceProgress"] = (out["TyreLife"] * out["RaceProgress"]).astype("float32")
    out["PitWindowPressure"] = (out["TyreLife"] * out["RaceProgress"]).astype("float32")
    out["TyreAgeVsRace"] = safe_div(out["TyreLife"], out["EstimatedTotalLaps"].clip(lower=1), eps).astype("float32")
    out["TyreLife_to_LapsRemaining"] = safe_div(out["TyreLife"], out["LapsRemaining"] + 1, eps).astype("float32")
    out["LapMinusTyreLife"] = (out["LapNumber"] - out["TyreLife"]).astype("float32")

    out["TyreLifeBin"] = pd.cut(
        out["TyreLife"], bins=[-np.inf, 3, 7, 12, 20, 30, np.inf],
        labels=["T003", "T007", "T012", "T020", "T030", "Tplus"],
    ).astype(str)

    out["StintPressure"] = (out["Stint"] * out["TyreLife"]).astype("float32")
    out["Is_First_Stint"] = (out["Stint"] == 1).astype(np.int8)
    out["Is_Late_Stint"] = (out["Stint"] >= 3).astype(np.int8)

    out["PositionBin"] = pd.cut(
        out["Position"], bins=[-np.inf, 3, 8, 14, np.inf],
        labels=["front", "upper_mid", "lower_mid", "back"],
    ).astype(str)
    out["PositionPressure"] = (out["Position"] * out["RaceProgress"]).astype("float32")

    out["DegPerRaceLap"] = safe_div(out["Cumulative_Degradation"], out["LapNumber"].clip(lower=1), eps).astype("float32")
    out["DegPerTyreLap"] = safe_div(out["Cumulative_Degradation"], out["TyreLife"].clip(lower=1), eps).astype("float32")
    out["Abs_Cumulative_Degradation"] = out["Cumulative_Degradation"].abs().astype("float32")
    out["Positive_Degradation"] = (out["Cumulative_Degradation"] > 0).astype(np.int8)

    out["DeltaAbs"] = out["LapTime_Delta"].abs().astype("float32")
    out["LapTimeDeltaPositive"] = (out["LapTime_Delta"] > 0).astype(np.int8)
    out["DeltaPerTyreLap"] = safe_div(out["LapTime_Delta"], out["TyreLife"].clip(lower=1), eps).astype("float32")

    out["Abs_Position_Change"] = out["Position_Change"].abs().astype("float32")
    out["Gained_Position"] = (out["Position_Change"] > 0).astype(np.int8)
    out["Lost_Position"] = (out["Position_Change"] < 0).astype(np.int8)

    return out


def add_cross_categoricals(out: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("Race_Year", ["Race", "Year"]),
        ("Compound_Stint", ["Compound", "Stint"]),
        ("Driver_Race", ["Driver", "Race"]),
        ("Driver_Compound", ["Driver", "Compound"]),
        ("Race_Compound", ["Race", "Compound"]),
        ("Race_Compound_Stint", ["Race", "Compound", "Stint"]),
        ("Compound_RacePhase", ["Compound", "RacePhase"]),
        ("Compound_TyreLifeBin", ["Compound", "TyreLifeBin"]),
        ("RacePhase_TyreLifeBin", ["RacePhase", "TyreLifeBin"]),
    ]
    for name, cols in pairs:
        if all(c in out.columns for c in cols):
            value = out[cols[0]].astype(str)
            for col in cols[1:]:
                value = value + "_" + out[col].astype(str)
            out[name] = value
    return out


def add_frequency_features(frames: list, cat_cols: list) -> None:
    if not cat_cols:
        return
    total = sum(len(f) for f in frames)
    for col in cat_cols:
        if not all(col in f.columns for f in frames):
            continue
        union = pd.concat([f[col].astype("string").fillna("__NA__") for f in frames], axis=0)
        counts = union.value_counts(dropna=False)
        for f in frames:
            keys = f[col].astype("string").fillna("__NA__")
            f[f"{col}_count"] = keys.map(counts).fillna(0).astype(np.int32)
            f[f"{col}_freq"] = (f[f"{col}_count"] / total).astype("float32")


def add_group_stats(frames: list) -> list:
    group_cols = ["Race_Year", "Race_Compound_Stint", "Driver_Race", "Compound_Stint"]
    value_cols = ["LapTime_Delta", "Position_Change", "RaceProgress", "TyreLife"]
    added: list = []

    combined_pieces = []
    keep_cols = list(set(group_cols + value_cols))
    for f in frames:
        cols_here = [c for c in keep_cols if c in f.columns]
        combined_pieces.append(f[cols_here].copy())
    combined = pd.concat(combined_pieces, axis=0, ignore_index=True)

    for g in group_cols:
        if g not in combined.columns:
            continue
        for v in value_cols:
            if v not in combined.columns:
                continue
            stats = combined.groupby(g, dropna=False)[v].agg(["mean", "std"])
            mean_col = f"{v}_mean_by_{g}"
            std_col = f"{v}_std_by_{g}"
            diff_col = f"{v}_diff_mean_by_{g}"
            for f in frames:
                if g not in f.columns or v not in f.columns:
                    continue
                key = f[g]
                f[mean_col] = key.map(stats["mean"]).astype("float32")
                f[std_col] = key.map(stats["std"]).fillna(0).astype("float32")
                f[diff_col] = (f[v] - f[mean_col]).astype("float32")
            added.extend([mean_col, std_col, diff_col])
    return added


def normalize_cats(out: pd.DataFrame, cat_cols: list) -> None:
    for col in cat_cols:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("__NA__").astype(str)


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("Loading data...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    ext = pd.read_csv(EXTERNAL_CSV).dropna(subset=["Compound"]).drop(
        columns=["Normalized_TyreLife"], errors="ignore"
    )
    ext[ID_COL] = -1
    print(f"  train {train.shape}  test {test.shape}  external {ext.shape}")
    print(f"  external pos rate: {ext[TARGET].mean():.4f}  "
          f"competition train pos rate: {train[TARGET].mean():.4f}")

    print("Applying domain FE...")
    train = add_domain_features(train)
    test = add_domain_features(test)
    ext = add_domain_features(ext)

    print("Adding cross-categoricals...")
    train = add_cross_categoricals(train)
    test = add_cross_categoricals(test)
    ext = add_cross_categoricals(ext)

    cross_cats = [
        "Race_Year", "Compound_Stint", "Driver_Race", "Driver_Compound",
        "Race_Compound", "Race_Compound_Stint", "Compound_RacePhase",
        "Compound_TyreLifeBin", "RacePhase_TyreLifeBin",
    ]
    bins = ["RacePhase", "LapBin", "TyreLifeBin", "PositionBin"]
    all_cats = BASE_CATS + cross_cats + bins

    print(f"Adding frequency features for {len(all_cats)} categorical fields...")
    add_frequency_features([train, test, ext], all_cats)

    print("Adding group-statistic features...")
    group_stat_cols = add_group_stats([train, test, ext])
    print(f"  added {len(group_stat_cols)} group-stat columns")

    normalize_cats(train, all_cats)
    normalize_cats(test, all_cats)
    normalize_cats(ext, all_cats)

    feature_cols = [c for c in train.columns if c not in (TARGET, ID_COL)]
    feature_cols = [c for c in feature_cols if c in test.columns and c in ext.columns]
    cat_indices = [feature_cols.index(c) for c in all_cats if c in feature_cols]
    print(f"using {len(feature_cols)} features, {len(cat_indices)} categorical")

    X = train[feature_cols]
    y = train[TARGET].astype(int).to_numpy()
    X_test = test[feature_cols]
    X_ext = ext[feature_cols]
    y_ext = ext[TARGET].astype(int).to_numpy()

    strat_key = train["Year"].astype(str) + "_" + train[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train), dtype=np.float64)
    test_preds = np.zeros(len(test), dtype=np.float64)
    fold_aucs: list = []
    fold_iters: list = []

    print(f"\nCB params: {CB_PARAMS}\n")

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, strat_key), start=1):
        t0 = time.time()
        X_tr = pd.concat([X.iloc[tr_idx], X_ext], axis=0, ignore_index=True)
        y_tr = np.concatenate([y[tr_idx], y_ext])
        X_va = X.iloc[va_idx]
        y_va = y[va_idx]

        params = dict(CB_PARAMS)
        params["random_seed"] = SEED + fold
        train_pool = Pool(X_tr, label=y_tr, cat_features=cat_indices)
        valid_pool = Pool(X_va, label=y_va, cat_features=cat_indices)
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

        va_pred = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = va_pred
        test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
        fold_auc = roc_auc_score(y_va, va_pred)
        fold_aucs.append(fold_auc)
        fold_iters.append(int(model.tree_count_))
        print(
            f"fold {fold}/{N_SPLITS}  AUC={fold_auc:.5f}  iters={model.tree_count_}  "
            f"train_rows={len(X_tr):,}  ({time.time()-t0:.1f}s)",
            flush=True,
        )

    oof_auc = roc_auc_score(y, oof)
    print()
    print(f"per-fold AUC: mean={np.mean(fold_aucs):.5f}  std={np.std(fold_aucs):.5f}  "
          f"iters={fold_iters}")
    print(f"OOF AUC:      {oof_auc:.5f}")
    print(f"  (vs CB-tuned-exp14 0.95114, Δ = {oof_auc - 0.95114:+.5f})")
    print(f"  (vs cycle-11 pass gate 0.95150, Δ = {oof_auc - 0.95150:+.5f})")

    # ============================================================
    # Calibration: bin-8 bias check (probe-5 baseline +0.0573)
    # ============================================================
    print()
    print("Calibration (10 equal-frequency bins, 1-indexed):")
    order = np.argsort(oof)
    n = len(oof)
    bin_size = n // 10
    for b in range(10):
        start = b * bin_size
        end = (b + 1) * bin_size if b < 9 else n
        bin_idx = order[start:end]
        pred_mean = oof[bin_idx].mean()
        actual_rate = y[bin_idx].mean()
        bias = pred_mean - actual_rate
        marker = "  *" if b == 7 else ""
        print(f"  bin {b+1:2d}  n={len(bin_idx):>6}  pred={pred_mean:.4f}  "
              f"actual={actual_rate:.4f}  bias={bias:+.4f}{marker}")
    print("  (bin 8 = probe-5 baseline +0.0573 from cycle-7 blend)")

    # ============================================================
    # Per-year AUC
    # ============================================================
    print()
    print("OOF AUC by Year:")
    oof_df = pd.DataFrame(
        {"id": train[ID_COL], "Year": train["Year"], "target": y, "oof": oof}
    )
    for year, g in oof_df.groupby("Year"):
        if g["target"].nunique() > 1:
            auc = roc_auc_score(g["target"], g["oof"])
            print(f"  {year}: AUC={auc:.5f}  n={len(g):,}  pos_rate={g['target'].mean():.4f}")

    # ============================================================
    # Save outputs to /kaggle/working/ for download
    # ============================================================
    oof_df.to_parquet(OOF_OUT, index=False)
    sub = (
        pd.DataFrame({"id": test[ID_COL], TARGET: test_preds})
        .sort_values("id")
        .reset_index(drop=True)
    )
    sub.to_csv(SUB_OUT, index=False)
    print(f"\nwrote {OOF_OUT.name}  ({len(oof_df):,} rows)")
    print(f"wrote {SUB_OUT.name}  ({len(sub):,} rows)")
    print("\nDownload both files from the Kaggle Notebook sidebar (Output tab) and "
          "place in `data/` locally for the blend probe.")


if __name__ == "__main__":
    main()
