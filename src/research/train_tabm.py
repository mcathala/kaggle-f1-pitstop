"""Experiment 045 (cycle 12) — TabM_D_Classifier as 4th blend base.

TabM is a transformer-style tabular foundation model from pytabkit. Cycle 11's
exp 044 confirmed XGBoost can earn meaningful blend weight when standalone AUC
is competitive — the question for cycle 12 is whether TabM's architecture
produces rank ordering different enough from RealMLP / CB / XGB to break into
the 4-way blend optimum.

Recipe: same FE as RealMLP (seed_realmlp), same 5-fold CV. TabM HPs use library
defaults (all None → pre-tuned); only device, random_state, n_repeats (for
in-seed ensembling, matching RealMLP's n_ens=24) and val_metric set explicitly.

Outputs:
  data/oof_tabm.parquet
  data/submission_tabm.csv
"""

import os
from importlib.metadata import version
from pathlib import Path
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
from pytabkit import TabM_D_Classifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parent.parent.parent / "data"
TRAIN_CSV = DATA / "train.csv"
TEST_CSV = DATA / "test.csv"
EXTERNAL_CSV = DATA / "f1_strategy_dataset.csv"

TARGET = "PitNextLap"
ID_COL = "id"
N_SPLITS = 5
# Split seed stays 42 (CV protocol frozen). Only the model random_state
# varies per invocation.
SPLIT_SEED = 42
SEED = int(os.environ.get("SEED", "42"))
OOF_OUT = DATA / "oof_tabm.parquet"
SUB_OUT = DATA / "submission_tabm.csv"

print(f"torch    version: {torch.__version__}")
print(f"pytabkit version: {version('pytabkit')}")
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"device:           {device}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


seed_everything(SEED)

TABM_PARAMS = {
    "random_state": SEED,
    "verbosity": 2,
    # TabM minimal HPs — most None → library uses pre-tuned defaults.
    # NOTE: n_repeats > 1 conflicts with explicit (X_va, y_va) validation in fit().
    # For our outer 5-fold CV we need explicit val rows, so n_repeats=1.
    "val_metric_name": "1-auc_ovr",
    "n_repeats": 1,
    # M1 Pro has 16 GB unified memory shared CPU/GPU; TabM's default ~256
    # batch_size + transformer attention OOMs. 64 is the safe ceiling.
    "batch_size": 64,
    "device": device,
}


def feature_engineering(df: pd.DataFrame, fit: bool, state: dict) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """FE pipeline shared between RealMLP-family and TabM trainers in this project.

    Adds: arithmetic interactions, floor-binned versions of numerics, count
    encoding for cats + Year/PitStop bins, KBinsDiscretizer for RaceProgress
    (200 bins) and LapTime (7 bins), interaction categories Race_Compound and
    Race_Year. Returns the processed df + lists of new cat/num/combo names.
    """
    # Arithmetic interactions
    df["_LapNumber_/_RaceProgress"] = (df["LapNumber"] / (df["RaceProgress"] + 1e-6)).astype("float32")
    df["_TyreLife_/_LapNumber"] = (df["TyreLife"] / df["LapNumber"].clip(lower=1)).astype("float32")
    df["_LapTime (s)_*_Cumulative_Degradation"] = (df["LapTime (s)"] * df["Cumulative_Degradation"]).astype("float32")
    df["_LapTime (s)_*_Cumulative_Degradation_abs"] = (df["LapTime (s)"] * df["Cumulative_Degradation"].abs()).astype("float32")
    df["_LapTime (s)_/_Cumulative_Degradation_abs"] = (df["LapTime (s)"] / (df["Cumulative_Degradation"].abs() + 1e-6)).astype("float32")

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in df.select_dtypes(exclude=["object"]).columns.tolist() if c not in (ID_COL, TARGET)]

    # Floor numeric into categorical-like (one bucket per integer value)
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

    # Count encoding for categoricals + Year/PitStop bins
    for col in cat_cols + ["Year_cat_", "PitStop_cat_"]:
        count_name = f"_{col}_count" if col in cat_cols else f"_{col[:-1]}_count"
        if fit:
            count_map = df[col].astype(object).value_counts()
            state[count_name] = count_map
        else:
            count_map = state[count_name]
        df[count_name] = df[col].astype(object).map(count_map).fillna(0).astype("int32")

    # KBins discretize key numerics
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

    # Interaction categories
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

    # Orig split mirrors the public notebook (same n_splits stratified on y_orig).
    kf_orig = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_SEED)
    orig_splits = list(kf_orig.split(X_orig, y_orig))

    for fold, ((tr_idx, va_idx), (or_tr_idx, or_va_idx)) in enumerate(
        zip(kf.split(X, strat_key), orig_splits), start=1
    ):
        t0 = time.time()
        X_tr = pd.concat([X.iloc[tr_idx], X_orig.iloc[or_tr_idx]], axis=0).reset_index(drop=True)
        y_tr = pd.concat([y.iloc[tr_idx], y_orig.iloc[or_tr_idx]], axis=0).reset_index(drop=True)
        X_va = X.iloc[va_idx].copy()
        y_va = y.iloc[va_idx]

        # Target encoding on the interaction combos — cross-fold-safe via the
        # sklearn TargetEncoder.cv parameter.
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
        print(
            f"fold {fold}/{N_SPLITS}  AUC={fold_auc:.5f}  ({time.time()-t0:.1f}s)",
            flush=True,
        )
        if device == "mps":
            torch.mps.empty_cache()

    oof_auc = roc_auc_score(y, oof)
    print(f"\nOOF AUC = {oof_auc:.5f}  per-fold mean {np.mean(fold_aucs):.5f}  std {np.std(fold_aucs):.5f}")

    pd.DataFrame(
        {"id": train_id, "Year": train["Year"], "target": y.to_numpy(), "oof": oof}
    ).to_parquet(OOF_OUT, index=False)
    pd.DataFrame({"id": test_id, TARGET: test_preds}).sort_values("id").reset_index(drop=True).to_csv(
        SUB_OUT, index=False
    )
    print(f"wrote {OOF_OUT.name} and {SUB_OUT.name}")


if __name__ == "__main__":
    main()
