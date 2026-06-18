"""Experiment 024 (cycle 8) — AutoGluon-Tabular.

Same 5-fold StratifiedKFold seed=42 on Year × PitNextLap as everything
else in this project. Uses the cycle 5 RealMLP feature pipeline (minus
the explicit TargetEncoder — AG handles cat features natively).

Per fold:
  - Train AG on (4/5 competition train + all-but-1/5 external data)
  - Predict on the held-out 1/5 of competition train + on the test set
  - Time-cap each fold at 600s (medium_quality preset)

Outputs:
  data/oof_autogluon.parquet
  data/submission_autogluon.csv

After running this, the 3-way blend probe (realmlp x cb-tuned x autogluon)
can use the OOF + submission directly.
"""

import shutil
import time
from importlib.metadata import version
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parent.parent.parent / "data"
TRAIN_CSV = DATA / "train.csv"
TEST_CSV = DATA / "test.csv"
EXTERNAL_CSV = DATA / "f1_strategy_dataset.csv"
OOF_OUT = DATA / "oof_autogluon.parquet"
SUB_OUT = DATA / "submission_autogluon.csv"
AG_ROOT = Path(__file__).resolve().parent.parent.parent / "ag_models"

TARGET = "PitNextLap"
ID_COL = "id"
N_SPLITS = 5
SEED = 42
TIME_LIMIT_PER_FOLD = 480  # seconds
PRESET = "medium_quality"

# Restrict the model zoo: skip the memory-heavy NN_TORCH / FASTAI / RF / XT
# (we already have RealMLP separately, and RF/XT on 432k rows OOMs on 16GB).
# Keep GBM/CAT/XGB — these are AutoGluon's strongest gradient-boosting trio
# and overlap least with our existing model zoo.
AG_HYPERPARAMS = {
    "GBM": [
        {"extra_trees": True, "ag_args": {"name_suffix": "XT"}},
        {},
        {"learning_rate": 0.03, "num_leaves": 128, "feature_fraction": 0.9,
         "min_data_in_leaf": 3, "ag_args": {"name_suffix": "Large", "priority": 0,
                                            "hyperparameter_tune_kwargs": None}},
    ],
    "CAT": [{}],
    "XGB": [{}],
}

print(f"autogluon-tabular: {version('autogluon.tabular')}")
print(f"pandas:            {version('pandas')}")
print(f"sklearn:           {version('scikit-learn')}")
print(f"preset:            {PRESET}")
print(f"time limit/fold:   {TIME_LIMIT_PER_FOLD}s")


def feature_engineering(df: pd.DataFrame, fit: bool, state: dict) -> pd.DataFrame:
    """Same arithmetic interactions / binning / count encoding as train_realmlp.py.
    Skips the TargetEncoder step — AG does its own cat handling.
    """
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

    for cols in [("Race", "Compound"), ("Race", "Year")]:
        combo_name = "_".join(cols) + "_"
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

    return df


def main() -> None:
    print("\nLoading data...")
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

    print("Applying FE (RealMLP-style, no TE)...")
    state: dict = {}
    X = feature_engineering(X, fit=True, state=state)
    X_test = feature_engineering(X_test, fit=False, state=state)
    X_orig = feature_engineering(X_orig, fit=False, state=state)
    print(f"  X      shape: {X.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  X_orig shape: {X_orig.shape}")

    strat_key = train["Year"].astype(str) + "_" + train[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    kf_orig = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(kf.split(X, strat_key))
    orig_splits = list(kf_orig.split(X_orig, y_orig))

    oof = np.zeros(len(train), dtype=np.float64)
    test_preds = np.zeros(len(test), dtype=np.float64)
    fold_aucs: list[float] = []

    # Clean any prior AG model dir
    if AG_ROOT.exists():
        shutil.rmtree(AG_ROOT)

    # Skip external data to keep memory footprint manageable on M1 Pro 16GB.
    # AG fits 5 GBM/CAT/XGB models per fold concurrently; orig data adds ~80k rows that
    # blew past the OOM threshold in the first run.
    for fold, ((tr_idx, va_idx), _) in enumerate(zip(splits, orig_splits), start=1):
        t0 = time.time()

        X_tr = X.iloc[tr_idx].reset_index(drop=True)
        y_tr = y.iloc[tr_idx].reset_index(drop=True)
        X_va = X.iloc[va_idx].copy()
        y_va = y.iloc[va_idx]

        train_df = X_tr.copy()
        train_df[TARGET] = y_tr.to_numpy()

        if fold == 1:
            print(f"  fold 1: train rows = {len(train_df):,}  features = {X_tr.shape[1]}")

        fold_path = AG_ROOT / f"fold{fold}"
        predictor = TabularPredictor(
            label=TARGET,
            eval_metric="roc_auc",
            path=str(fold_path),
            verbosity=2,
        )
        predictor.fit(
            train_data=train_df,
            time_limit=TIME_LIMIT_PER_FOLD,
            hyperparameters=AG_HYPERPARAMS,
            num_bag_folds=0,  # disable internal bagging — we drive the 5-fold CV
            num_stack_levels=0,
            verbosity=2,
        )

        va_proba = predictor.predict_proba(X_va)
        # class-1 column (target=1)
        pos_col = 1 if 1 in va_proba.columns else va_proba.columns[-1]
        va_pred = va_proba[pos_col].to_numpy()
        tst_pred = predictor.predict_proba(X_test)[pos_col].to_numpy()

        oof[va_idx] = va_pred
        test_preds += tst_pred / N_SPLITS
        fold_auc = roc_auc_score(y_va, va_pred)
        fold_aucs.append(fold_auc)
        elapsed = time.time() - t0
        print(f"fold {fold}/{N_SPLITS}  AUC={fold_auc:.5f}  ({elapsed:.0f}s)", flush=True)

        # AG leaderboard snapshot for fold 1 only (avoid spam)
        if fold == 1:
            try:
                print("\nFold 1 AG component leaderboard:")
                print(predictor.leaderboard(silent=True).head(10).to_string(index=False))
                print()
            except Exception as e:
                print(f"(could not fetch leaderboard: {e})")

    oof_auc = roc_auc_score(y, oof)
    print(f"\nOOF AUC = {oof_auc:.5f}  per-fold mean {np.mean(fold_aucs):.5f}  std {np.std(fold_aucs):.5f}")
    print(f"Per-fold: {[f'{a:.5f}' for a in fold_aucs]}")

    pd.DataFrame(
        {"id": train_id, "Year": train["Year"], "target": y.to_numpy(), "oof": oof}
    ).to_parquet(OOF_OUT, index=False)
    pd.DataFrame({"id": test_id, TARGET: test_preds}).sort_values("id").reset_index(drop=True).to_csv(
        SUB_OUT, index=False
    )
    print(f"wrote {OOF_OUT.name} and {SUB_OUT.name}")


if __name__ == "__main__":
    main()
