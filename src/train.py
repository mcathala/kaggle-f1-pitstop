"""Baseline LightGBM trainer for the F1 pit-stop competition.

5-fold StratifiedKFold on Year x PitNextLap to keep the 2023 anomaly
distributed evenly across folds. Native LightGBM categorical handling for
Driver, Race, Compound. Logs per-fold AUC, OOF AUC, and writes:

  data/oof_baseline.parquet      OOF predictions (id, fold, oof_pred, target)
  data/submission_baseline.csv   id, PitNextLap (mean of fold test predictions)
"""

from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA = Path(__file__).resolve().parent.parent / "data"
TRAIN_PARQUET = DATA / "train_features.parquet"
TEST_PARQUET = DATA / "test_features.parquet"
OOF_OUT = DATA / "oof_baseline.parquet"
SUB_OUT = DATA / "submission_baseline.csv"

TARGET = "PitNextLap"
ID_COL = "id"
CATEGORICAL = ["Driver", "Race", "Compound"]
N_SPLITS = 5
SEED = 42

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": SEED,
}
NUM_BOOST_ROUND = 5000
EARLY_STOPPING = 100


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pl.read_parquet(TRAIN_PARQUET).to_pandas()
    test = pl.read_parquet(TEST_PARQUET).to_pandas()
    for c in CATEGORICAL:
        train[c] = train[c].astype("category")
        # Align test categories to train so unseen levels become NaN (none expected).
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)
    return train, test


def main() -> None:
    train, test = load()
    print(f"train {train.shape}  test {test.shape}")

    feature_cols = [c for c in train.columns if c not in (TARGET, ID_COL)]
    print(f"using {len(feature_cols)} features, {len(CATEGORICAL)} categorical")

    X = train[feature_cols]
    y = train[TARGET].astype(int).to_numpy()
    X_test = test[feature_cols]

    # Stratify by Year x PitNextLap: keeps 2023's near-zero positive rate even.
    strat_key = train["Year"].astype(str) + "_" + train[TARGET].astype(int).astype(str)

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train), dtype=np.float64)
    test_preds = np.zeros(len(test), dtype=np.float64)
    fold_aucs: list[float] = []
    fold_iters: list[int] = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, strat_key), start=1):
        t0 = time.time()
        dtr = lgb.Dataset(
            X.iloc[tr_idx], label=y[tr_idx], categorical_feature=CATEGORICAL
        )
        dva = lgb.Dataset(
            X.iloc[va_idx],
            label=y[va_idx],
            categorical_feature=CATEGORICAL,
            reference=dtr,
        )
        model = lgb.train(
            LGB_PARAMS,
            dtr,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dva],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(period=200),
            ],
        )
        va_pred = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
        oof[va_idx] = va_pred
        test_preds += (
            model.predict(X_test, num_iteration=model.best_iteration) / N_SPLITS
        )
        fold_auc = roc_auc_score(y[va_idx], va_pred)
        fold_aucs.append(fold_auc)
        fold_iters.append(model.best_iteration or 0)
        print(
            f"fold {fold}/{N_SPLITS}  AUC={fold_auc:.5f}  "
            f"best_iter={model.best_iteration}  ({time.time() - t0:.1f}s)"
        )

    oof_auc = roc_auc_score(y, oof)
    print()
    print(
        f"per-fold AUC: mean={np.mean(fold_aucs):.5f} "
        f"std={np.std(fold_aucs):.5f}  iters={fold_iters}"
    )
    print(f"OOF AUC:      {oof_auc:.5f}")

    # Per-year breakdown — 2023 is the regime to watch.
    oof_df = pd.DataFrame(
        {"id": train[ID_COL], "Year": train["Year"], "target": y, "oof": oof}
    )
    print("\nOOF AUC by Year:")
    for year, g in oof_df.groupby("Year"):
        if g["target"].nunique() > 1:
            auc = roc_auc_score(g["target"], g["oof"])
            print(f"  {year}: AUC={auc:.5f}  n={len(g):,}  pos_rate={g['target'].mean():.4f}")
        else:
            print(f"  {year}: skipped (single class)  n={len(g):,}")

    # Persist OOF + submission.
    oof_df.assign(fold=-1).to_parquet(OOF_OUT, index=False)
    sub = (
        pd.DataFrame({"id": test[ID_COL], TARGET: test_preds})
        .sort_values("id")
        .reset_index(drop=True)
    )
    sub.to_csv(SUB_OUT, index=False)
    print(f"\nwrote {OOF_OUT.name}  ({len(oof_df):,} rows)")
    print(f"wrote {SUB_OUT.name}  ({len(sub):,} rows)")


if __name__ == "__main__":
    main()
