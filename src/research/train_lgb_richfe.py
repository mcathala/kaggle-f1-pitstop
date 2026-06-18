"""Experiment 025 (cycle 8) — LightGBM with cycle 5's rich FE pipeline.

Same FE as train_realmlp.py (arithmetic interactions, count encoding,
KBins discretization, interaction categoricals + target encoding).
Goal: produce a tree model strong enough (OOF ≥ 0.945) AND structurally
distinct from RealMLP/CB-tuned to add ensemble value to cycle 7's mix.

Outputs:
  data/oof_lgb_richfe.parquet
  data/submission_lgb_richfe.csv
"""

from importlib.metadata import version
from pathlib import Path
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parent.parent.parent / "data"
TRAIN_CSV = DATA / "train.csv"
TEST_CSV = DATA / "test.csv"
OOF_OUT = DATA / "oof_lgb_richfe.parquet"
SUB_OUT = DATA / "submission_lgb_richfe.csv"

TARGET = "PitNextLap"
ID_COL = "id"
N_SPLITS = 5
SEED = 42

print(f"lightgbm: {version('lightgbm')}")
print(f"pandas:   {version('pandas')}")
print(f"sklearn:  {version('scikit-learn')}")


LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "seed": SEED,
    "n_jobs": -1,
}


def feature_engineering(df: pd.DataFrame, fit: bool, state: dict) -> tuple[pd.DataFrame, list[str]]:
    """Same as train_realmlp.py's FE — returns df + the list of TE combo column names."""
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
    print(f"  train {train.shape}  test {test.shape}")

    y = train[TARGET].astype(int)
    train_id = train[ID_COL]
    test_id = test[ID_COL]

    X = train.drop([ID_COL, TARGET], axis=1)
    X_test = test.drop([ID_COL], axis=1)

    print("Applying FE...")
    state: dict = {}
    X, combo_names = feature_engineering(X, fit=True, state=state)
    X_test, _ = feature_engineering(X_test, fit=False, state=state)
    print(f"  X      shape: {X.shape}")
    print(f"  X_test shape: {X_test.shape}")

    # Cast cat-string columns to pandas category dtype for LightGBM
    cat_features = [c for c in X.columns if X[c].dtype == "object"]
    for c in cat_features:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")
    # Align category dtypes between train and test (LGB needs identical categories)
    for c in cat_features:
        union = pd.api.types.union_categoricals([X[c], X_test[c]])
        X[c] = pd.Categorical(X[c], categories=union.categories)
        X_test[c] = pd.Categorical(X_test[c], categories=union.categories)
    print(f"  categorical features: {len(cat_features)}")

    strat_key = train["Year"].astype(str) + "_" + train[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof = np.zeros(len(train), dtype=np.float64)
    test_preds = np.zeros(len(test), dtype=np.float64)
    fold_aucs: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, strat_key), start=1):
        t0 = time.time()
        X_tr_raw = X.iloc[tr_idx].copy()
        y_tr = y.iloc[tr_idx]
        X_va_raw = X.iloc[va_idx].copy()
        y_va = y.iloc[va_idx]
        X_tst_raw = X_test.copy()

        # TE on the interaction combos with the cv-safe TargetEncoder
        te = TargetEncoder(cv=N_SPLITS, smooth="auto", shuffle=True, random_state=SEED)
        tr_enc = te.fit_transform(X_tr_raw[combo_names], y_tr)
        va_enc = te.transform(X_va_raw[combo_names])
        tst_enc = te.transform(X_tst_raw[combo_names])
        te_names = [f"_{c}TE" for c in combo_names]
        X_tr_raw[te_names] = tr_enc
        X_va_raw[te_names] = va_enc
        X_tst_raw[te_names] = tst_enc

        if fold == 1:
            print(f"  fold 1: train rows = {len(X_tr_raw):,}  features = {X_tr_raw.shape[1]} ({len(cat_features)} cat + {len(te_names)} TE + {X_tr_raw.shape[1]-len(cat_features)-len(te_names)} num)")

        dtrain = lgb.Dataset(X_tr_raw, y_tr, categorical_feature=cat_features)
        dvalid = lgb.Dataset(X_va_raw, y_va, categorical_feature=cat_features, reference=dtrain)

        model = lgb.train(
            LGB_PARAMS,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False),
                       lgb.log_evaluation(period=200)],
        )

        va_pred = model.predict(X_va_raw, num_iteration=model.best_iteration)
        tst_pred = model.predict(X_tst_raw, num_iteration=model.best_iteration)

        oof[va_idx] = va_pred
        test_preds += tst_pred / N_SPLITS
        fold_auc = roc_auc_score(y_va, va_pred)
        fold_aucs.append(fold_auc)
        print(f"fold {fold}/{N_SPLITS}  AUC={fold_auc:.5f}  best_iter={model.best_iteration}  ({time.time()-t0:.0f}s)", flush=True)

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
