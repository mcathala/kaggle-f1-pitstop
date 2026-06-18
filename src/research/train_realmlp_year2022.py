"""Experiment 027 (cycle 9) — 2022-year specialist RealMLP.

For each of the 5 folds (same StratifiedKFold seed=42 as cycle 5):
  - Filter the fold's training rows to Year == 2022
  - Add all Year == 2022 rows from external data
  - Train RealMLP (cycle 5 HPs) on this subset
  - Predict on the fold's val rows where Year == 2022 (these get the specialist's prediction)
  - Predict on all test rows where Year == 2022 (used for test submission)

After running, combine the specialist's 2022 OOF predictions with cycle 5
multi-seed's predictions for 2023/2024/2025 to form a year-combined OOF.

Outputs:
  data/oof_realmlp_2022_specialist.parquet     — specialist's 2022 OOF predictions
  data/oof_realmlp_year_combined.parquet       — specialist 2022 + multi-seed non-2022
  data/submission_realmlp_year_combined.csv    — test submission (specialist for 2022, multi-seed for others)
"""

from importlib.metadata import version
from pathlib import Path
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parent.parent.parent / "data"
TRAIN_CSV = DATA / "train.csv"
TEST_CSV = DATA / "test.csv"
EXTERNAL_CSV = DATA / "f1_strategy_dataset.csv"
MULTISEED_OOF = DATA / "oof_realmlp_multiseed.parquet"
MULTISEED_SUB = DATA / "submission_realmlp_multiseed.csv"

OOF_2022_OUT = DATA / "oof_realmlp_2022_specialist.parquet"
OOF_COMBINED_OUT = DATA / "oof_realmlp_year_combined.parquet"
SUB_COMBINED_OUT = DATA / "submission_realmlp_year_combined.csv"

TARGET = "PitNextLap"
ID_COL = "id"
N_SPLITS = 5
SEED = 42
TARGET_YEAR = 2022

print(f"torch    version: {torch.__version__}")
print(f"pytabkit version: {version('pytabkit')}")
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"device:           {device}")
print(f"specialist year:  {TARGET_YEAR}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


seed_everything(SEED)

REALMLP_PARAMS = {
    "random_state": SEED,
    "verbosity": 1,
    "val_metric_name": "1-auc_ovr",
    "n_ens": 24,
    "n_epochs": 6,
    "batch_size": 256,
    "use_early_stopping": False,
    "lr": 0.01,
    "wd": 0.016,
    "sq_mom": 0.99,
    "lr_sched": "lin_cos_log_15",
    "first_layer_lr_factor": 0.25,
    "embedding_size": 6,
    "max_one_hot_cat_size": 18,
    "hidden_sizes": [512, 256, 128],
    "act": "silu",
    "p_drop": 0.05,
    "p_drop_sched": "invsqrtp1e-3",
    "plr_hidden_1": 16,
    "plr_hidden_2": 8,
    "plr_act_name": "gelu",
    "plr_lr_factor": 0.1151,
    "plr_sigma": 2.33,
    "ls_eps": 0.01,
    "ls_eps_sched": "sqrt_cos",
    "add_front_scale": False,
    "bias_init_mode": "neg-uniform-dynamic-2",
    "tfms": ["one_hot", "median_center", "robust_scale",
             "smooth_clip", "embedding", "l2_normalize"],
    "device": device,
}


def feature_engineering(df: pd.DataFrame, fit: bool, state: dict) -> tuple[pd.DataFrame, list[str]]:
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

    return df, combo_names


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
    X, combo_names = feature_engineering(X, fit=True, state=state)
    X_test, _ = feature_engineering(X_test, fit=False, state=state)
    X_orig, _ = feature_engineering(X_orig, fit=False, state=state)
    print(f"  X      shape: {X.shape}")
    print(f"  X_test shape: {X_test.shape}")

    # Build year masks (using the raw Year column from the source DataFrames)
    train_year = train["Year"].to_numpy()
    test_year = test["Year"].to_numpy()
    orig_year = orig["Year"].to_numpy()
    train_is_target = (train_year == TARGET_YEAR)
    test_is_target = (test_year == TARGET_YEAR)
    orig_is_target = (orig_year == TARGET_YEAR)

    print(f"\n  train Year=={TARGET_YEAR}:  {train_is_target.sum():,} rows")
    print(f"  test  Year=={TARGET_YEAR}:  {test_is_target.sum():,} rows")
    print(f"  orig  Year=={TARGET_YEAR}:  {orig_is_target.sum():,} rows")

    # CV — same 5-fold StratifiedKFold seed=42 as everything else
    strat_key = train["Year"].astype(str) + "_" + train[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    # Specialist OOF only on 2022 rows; -1 sentinel for other-year rows (filled later)
    oof_2022 = np.full(len(train), -1.0, dtype=np.float64)
    test_preds_2022 = np.zeros(len(test), dtype=np.float64)
    fold_aucs_2022: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, strat_key), start=1):
        t0 = time.time()
        # Restrict to TARGET_YEAR within both train and val splits
        tr_target = tr_idx[train_is_target[tr_idx]]
        va_target = va_idx[train_is_target[va_idx]]
        if len(tr_target) == 0 or len(va_target) == 0:
            print(f"fold {fold}: no rows of Year=={TARGET_YEAR}, skipping")
            continue

        X_tr = pd.concat(
            [X.iloc[tr_target], X_orig.loc[orig_is_target]],
            axis=0,
        ).reset_index(drop=True)
        y_tr = pd.concat(
            [y.iloc[tr_target], y_orig.loc[orig_is_target]],
            axis=0,
        ).reset_index(drop=True)
        X_va = X.iloc[va_target].copy()
        y_va = y.iloc[va_target]

        # TE on combos (fit on this fold's train, transform val + 2022-test)
        te = TargetEncoder(cv=N_SPLITS, smooth="auto", shuffle=True, random_state=SEED)
        tr_enc = te.fit_transform(X_tr[combo_names], y_tr)
        va_enc = te.transform(X_va[combo_names])
        te_names = [f"_{c}TE" for c in combo_names]
        X_tr[te_names] = tr_enc
        X_va[te_names] = va_enc

        # Also transform 2022 test rows for submission
        X_test_2022 = X_test.loc[test_is_target].copy()
        X_test_2022[te_names] = te.transform(X_test_2022[combo_names])

        if fold == 1:
            print(f"  fold 1: train rows = {len(X_tr):,}  (comp={len(tr_target):,}, ext={orig_is_target.sum():,})  val rows = {len(X_va):,}  features = {X_tr.shape[1]}")

        model = RealMLP_TD_Classifier(**REALMLP_PARAMS)
        model.fit(X_tr, y_tr, X_va, y_va)
        va_pred = model.predict_proba(X_va)[:, 1]
        tst_pred = model.predict_proba(X_test_2022)[:, 1]

        oof_2022[va_target] = va_pred
        # Accumulate test predictions only for 2022 test rows
        idx_2022 = np.where(test_is_target)[0]
        test_preds_2022[idx_2022] += tst_pred / N_SPLITS

        fold_auc = roc_auc_score(y_va, va_pred)
        fold_aucs_2022.append(fold_auc)
        print(f"fold {fold}/{N_SPLITS}  Year{TARGET_YEAR}_AUC={fold_auc:.5f}  ({time.time()-t0:.0f}s)", flush=True)
        if device == "mps":
            torch.mps.empty_cache()

    # Specialist 2022 within-year OOF AUC
    mask_2022 = train_is_target
    oof_2022_clean = oof_2022[mask_2022]
    y_2022 = y.to_numpy()[mask_2022]
    auc_2022_specialist = roc_auc_score(y_2022, oof_2022_clean)
    print(f"\n=== 2022 specialist ===")
    print(f"  Within-year AUC = {auc_2022_specialist:.5f}")
    print(f"  Per-fold: {[f'{a:.5f}' for a in fold_aucs_2022]}  std={np.std(fold_aucs_2022):.5f}")

    # Persist specialist OOF
    pd.DataFrame({
        "id": train_id.to_numpy()[mask_2022],
        "Year": train_year[mask_2022],
        "target": y_2022,
        "oof": oof_2022_clean,
    }).to_parquet(OOF_2022_OUT, index=False)
    print(f"wrote {OOF_2022_OUT.name}")

    # Build combined OOF: specialist for 2022, multi-seed for others
    multi = pd.read_parquet(MULTISEED_OOF).set_index("id").sort_index()
    train_sorted = train.set_index(ID_COL).sort_index()
    y_all = train_sorted[TARGET].astype(int).to_numpy()
    year_all = train_sorted["Year"].to_numpy()

    # Build the combined OOF using multi-seed as base then overwrite 2022 with specialist
    oof_combined = multi["oof"].to_numpy().copy()
    # Need to map specialist OOF (in original train order) into the sorted-by-id ordering of multi
    spec_by_id = pd.Series(oof_2022_clean, index=train_id.to_numpy()[mask_2022])
    overwrite_mask = train_sorted.index.isin(spec_by_id.index)
    oof_combined[overwrite_mask] = spec_by_id.reindex(train_sorted.index[overwrite_mask]).to_numpy()

    auc_combined = roc_auc_score(y_all, oof_combined)
    auc_multi = roc_auc_score(y_all, multi["oof"].to_numpy())
    auc_multi_2022 = roc_auc_score(y_all[year_all == TARGET_YEAR], multi["oof"].to_numpy()[year_all == TARGET_YEAR])
    print(f"\n=== Combined OOF (specialist 2022 + multi-seed non-2022) ===")
    print(f"  Overall AUC: {auc_combined:.5f}  (multi-seed: {auc_multi:.5f}, Δ={auc_combined-auc_multi:+.5f})")
    print(f"  2022 multi-seed within-year AUC: {auc_multi_2022:.5f}  (specialist: {auc_2022_specialist:.5f}, Δ={auc_2022_specialist-auc_multi_2022:+.5f})")

    pd.DataFrame({
        "id": train_sorted.index.to_numpy(),
        "Year": year_all,
        "target": y_all,
        "oof": oof_combined,
    }).to_parquet(OOF_COMBINED_OUT, index=False)
    print(f"wrote {OOF_COMBINED_OUT.name}")

    # Build combined test submission: specialist for 2022, multi-seed for others
    multi_sub = pd.read_csv(MULTISEED_SUB).sort_values(ID_COL).reset_index(drop=True)
    combined_sub = multi_sub.copy()
    test_2022_ids = test_id.to_numpy()[test_is_target]
    spec_test_by_id = pd.Series(test_preds_2022[test_is_target], index=test_2022_ids)
    overwrite_mask_test = combined_sub[ID_COL].isin(spec_test_by_id.index)
    combined_sub.loc[overwrite_mask_test, TARGET] = combined_sub.loc[overwrite_mask_test, ID_COL].map(spec_test_by_id).to_numpy()
    combined_sub.to_csv(SUB_COMBINED_OUT, index=False)
    print(f"wrote {SUB_COMBINED_OUT.name}")


if __name__ == "__main__":
    main()
