"""CatBoost trainer for the F1 pit-stop competition.

Mirrors src/train.py's CV protocol exactly so the OOF predictions are directly
comparable: same 5-fold StratifiedKFold on Year x PitNextLap, same seed, same
feature set. Native CatBoost categorical handling for Driver, Race, Compound
(ordered target encoding for high-cardinality Driver).

Outputs:
  data/oof_catboost.parquet      OOF predictions (id, Year, target, oof)
  data/submission_catboost.csv   id, PitNextLap (mean of fold test predictions)
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA = Path(__file__).resolve().parent.parent.parent / "data"
TRAIN_PARQUET = DATA / "train_features.parquet"
TEST_PARQUET = DATA / "test_features.parquet"
OOF_OUT = DATA / "oof_catboost.parquet"
SUB_OUT = DATA / "submission_catboost.csv"

TARGET = "PitNextLap"
ID_COL = "id"
CATEGORICAL = ["Driver", "Race", "Compound"]
N_SPLITS = 5
SEED = 42

CB_PARAMS = {
    "iterations": 5000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": SEED,
    "od_type": "Iter",
    "od_wait": 100,
    "verbose": 200,
    "task_type": "CPU",
    "thread_count": -1,
}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pl.read_parquet(TRAIN_PARQUET).to_pandas()
    test = pl.read_parquet(TEST_PARQUET).to_pandas()
    # CatBoost wants categorical features as strings; nulls in numeric columns
    # are fine but cat-feature nulls must become a sentinel string.
    for c in CATEGORICAL:
        train[c] = train[c].astype(str).fillna("__NA__")
        test[c] = test[c].astype(str).fillna("__NA__")
    return train, test


def main() -> None:
    train, test = load()
    print(f"train {train.shape}  test {test.shape}")

    feature_cols = [c for c in train.columns if c not in (TARGET, ID_COL)]
    cat_indices = [feature_cols.index(c) for c in CATEGORICAL]
    print(f"using {len(feature_cols)} features, {len(CATEGORICAL)} categorical")

    X = train[feature_cols]
    y = train[TARGET].astype(int).to_numpy()
    X_test = test[feature_cols]

    # Same stratification as LGB trainer.
    strat_key = train["Year"].astype(str) + "_" + train[TARGET].astype(int).astype(str)

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train), dtype=np.float64)
    test_preds = np.zeros(len(test), dtype=np.float64)
    fold_aucs: list[float] = []
    fold_iters: list[int] = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, strat_key), start=1):
        t0 = time.time()
        train_pool = Pool(X.iloc[tr_idx], label=y[tr_idx], cat_features=cat_indices)
        valid_pool = Pool(X.iloc[va_idx], label=y[va_idx], cat_features=cat_indices)
        model = CatBoostClassifier(**CB_PARAMS)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        va_pred = model.predict_proba(X.iloc[va_idx])[:, 1]
        oof[va_idx] = va_pred
        test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
        fold_auc = roc_auc_score(y[va_idx], va_pred)
        fold_aucs.append(fold_auc)
        fold_iters.append(int(model.tree_count_))
        print(
            f"fold {fold}/{N_SPLITS}  AUC={fold_auc:.5f}  "
            f"iters={model.tree_count_}  ({time.time() - t0:.1f}s)"
        )

    oof_auc = roc_auc_score(y, oof)
    print()
    print(
        f"per-fold AUC: mean={np.mean(fold_aucs):.5f} "
        f"std={np.std(fold_aucs):.5f}  iters={fold_iters}"
    )
    print(f"OOF AUC:      {oof_auc:.5f}")

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
