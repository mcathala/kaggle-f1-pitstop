"""Reproduce the cycle-#006 3-way ensemble baseline on the current data hash.

Why this exists: src/features.py now outputs 68 columns (the cycle-#006 additions
laptime_accel_3, laptime_accel_roll_3, tyre_age_pct_among_compound_peers are
included). src/train.py and src/train_catboost.py auto-select all non-id/non-target
columns, so running them today gives a different LGB/CB#004 than what's documented
in experiments/006_hard_compound_v2.md (which used a 65-column features.py).

This script trains the three baselines that the cycle-#006 ensemble used:
  - LGB on the 63-feature set  (drop the 3 cycle-#006 features + id + target)
  - CB#004 on the 65-feature set (drop the 3 cycle-#006 features + id + target;
    cycle-#004 frozen CatBoost params)
  - CB#006 on the 68-feature set (full features.py; same cycle-#004 frozen params)

If the parquet also contains the cycle-#007 peer-rank features (71-col parquet),
this script additionally trains CB#007 on all 69 model features and computes the
4-way ensemble. This makes it the single per-seed driver used by cycle 008's
seed-robustness sweep.

Then it blends:
  - 3-way: LGB=0.10, CB#004=0.40, CB#006=0.50 (cycle #006 weights). Target seed=42 OOF: 0.94866.
  - 4-way (only if CB#007 trained): LGB=0.10, each CB=0.30 (cycle #007 fixed weights). Target seed=42 OOF: 0.94880.

Seed is taken from $SEED (default 42). All output filenames are suffixed with the seed.

Outputs (gitignored), where {seed} ∈ {42, 7, 99, ...}:
  data/oof_lgb_seed{seed}.parquet
  data/oof_cb004_seed{seed}.parquet
  data/oof_cb006_seed{seed}.parquet
  data/oof_cb007_seed{seed}.parquet (only if 71-col parquet)
  data/oof_ensemble3_seed{seed}.parquet
  data/oof_ensemble4_seed{seed}.parquet (only if CB#007 trained)
  data/submission_*_seed{seed}.csv

CV protocol is copied verbatim from src/train.py + src/train_catboost.py so
results are directly comparable. The two trainers remain frozen.
"""

import os
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA = Path(__file__).resolve().parent.parent.parent / "data"
TRAIN_PARQUET = DATA / "train_features.parquet"
TEST_PARQUET = DATA / "test_features.parquet"

TARGET = "PitNextLap"
ID_COL = "id"
CATEGORICAL = ["Driver", "Race", "Compound"]
N_SPLITS = 5
SEED = int(os.environ.get("SEED", "42"))
print(f"[seed sweep] SEED={SEED}")

# Cycle #006 added these 3 features to features.py; the pre-cycle-#006 LGB
# and CB#004 baselines did NOT see them.
CYCLE006_FEATURES = [
    "laptime_accel_3",
    "laptime_accel_roll_3",
    "tyre_age_pct_among_compound_peers",
]

# Cycle #007 peer-rank additions. If present in the parquet, this script will
# also train CB#007 and the 4-way ensemble.
CYCLE007_FEATURES = [
    "position_pct_among_compound_peers",
    "laptime_pct_among_compound_peers",
    "cum_deg_pct_among_compound_peers",
]

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
LGB_NUM_BOOST_ROUND = 5000
LGB_EARLY_STOPPING = 100

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


def load_for_lgb() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Matches src/train.py: convert categoricals to pandas category dtype."""
    train = pl.read_parquet(TRAIN_PARQUET).to_pandas()
    test = pl.read_parquet(TEST_PARQUET).to_pandas()
    for c in CATEGORICAL:
        train[c] = train[c].astype("category")
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)
    return train, test


def load_for_cb() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Matches src/train_catboost.py: categoricals as str with NA sentinel."""
    train = pl.read_parquet(TRAIN_PARQUET).to_pandas()
    test = pl.read_parquet(TEST_PARQUET).to_pandas()
    for c in CATEGORICAL:
        train[c] = train[c].astype(str).fillna("__NA__")
        test[c] = test[c].astype(str).fillna("__NA__")
    return train, test


def split_indices(train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Identical strat key to src/train.py / src/train_catboost.py."""
    strat_key = train["Year"].astype(str) + "_" + train[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(kf.split(train.index.to_numpy(), strat_key))


def run_lgb(
    feature_cols: list[str], folds, tag: str
) -> tuple[np.ndarray, np.ndarray, list[float], list[int]]:
    train, test = load_for_lgb()
    cat_cols = [c for c in CATEGORICAL if c in feature_cols]
    X = train[feature_cols]
    y = train[TARGET].astype(int).to_numpy()
    X_test = test[feature_cols]
    oof = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    fold_aucs: list[float] = []
    fold_iters: list[int] = []
    print(f"\n=== LGB ({tag}) — {len(feature_cols)} features ===")
    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        t0 = time.time()
        dtr = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx], categorical_feature=cat_cols)
        dva = lgb.Dataset(
            X.iloc[va_idx], label=y[va_idx], categorical_feature=cat_cols, reference=dtr
        )
        model = lgb.train(
            LGB_PARAMS,
            dtr,
            num_boost_round=LGB_NUM_BOOST_ROUND,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(LGB_EARLY_STOPPING, verbose=False)],
        )
        va_pred = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
        oof[va_idx] = va_pred
        test_preds += model.predict(X_test, num_iteration=model.best_iteration) / N_SPLITS
        auc = roc_auc_score(y[va_idx], va_pred)
        fold_aucs.append(auc)
        fold_iters.append(model.best_iteration)
        print(
            f"  fold {fold}/{N_SPLITS}  AUC={auc:.5f}  best_iter={model.best_iteration}  "
            f"({time.time()-t0:.1f}s)"
        )
    return oof, test_preds, fold_aucs, fold_iters


def run_cb(
    feature_cols: list[str], folds, tag: str
) -> tuple[np.ndarray, np.ndarray, list[float], list[int]]:
    train, test = load_for_cb()
    cat_indices = [feature_cols.index(c) for c in CATEGORICAL if c in feature_cols]
    X = train[feature_cols]
    y = train[TARGET].astype(int).to_numpy()
    X_test = test[feature_cols]
    oof = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    fold_aucs: list[float] = []
    fold_iters: list[int] = []
    print(f"\n=== CatBoost ({tag}) — {len(feature_cols)} features ===")
    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        t0 = time.time()
        train_pool = Pool(X.iloc[tr_idx], label=y[tr_idx], cat_features=cat_indices)
        valid_pool = Pool(X.iloc[va_idx], label=y[va_idx], cat_features=cat_indices)
        model = CatBoostClassifier(**CB_PARAMS)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        va_pred = model.predict_proba(X.iloc[va_idx])[:, 1]
        oof[va_idx] = va_pred
        test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
        auc = roc_auc_score(y[va_idx], va_pred)
        fold_aucs.append(auc)
        fold_iters.append(int(model.tree_count_))
        print(
            f"  fold {fold}/{N_SPLITS}  AUC={auc:.5f}  iters={model.tree_count_}  "
            f"({time.time()-t0:.1f}s)"
        )
    return oof, test_preds, fold_aucs, fold_iters


def report(name: str, y: np.ndarray, oof: np.ndarray, folds, fold_aucs, fold_iters):
    auc = roc_auc_score(y, oof)
    print(f"\n{name} OOF AUC = {auc:.5f}")
    print(f"  per-fold mean={np.mean(fold_aucs):.5f} std={np.std(fold_aucs):.5f}")
    print(f"  per-fold AUC: {[f'{a:.5f}' for a in fold_aucs]}")
    print(f"  iters: {fold_iters}")
    return auc


def _save_oof(name: str, raw_train: pd.DataFrame, raw_test: pd.DataFrame, y, oof, test_preds) -> None:
    pd.DataFrame({"id": raw_train[ID_COL], "Year": raw_train["Year"], "target": y, "oof": oof}).to_parquet(
        DATA / f"oof_{name}_seed{SEED}.parquet", index=False
    )
    pd.DataFrame({"id": raw_test[ID_COL], TARGET: test_preds}).sort_values("id").reset_index(drop=True).to_csv(
        DATA / f"submission_{name}_seed{SEED}.csv", index=False
    )


def _per_fold_ens(y, ens_oof, folds, label: str) -> tuple[float, list[float]]:
    auc = roc_auc_score(y, ens_oof)
    per_fold = []
    print(f"\n=== {label} (seed={SEED}) ===")
    print(f"OOF AUC = {auc:.5f}")
    for fold, (_, va_idx) in enumerate(folds, start=1):
        a = roc_auc_score(y[va_idx], ens_oof[va_idx])
        per_fold.append(a)
        print(f"  fold {fold}/{N_SPLITS}  AUC={a:.5f}")
    print(f"  per-fold mean={np.mean(per_fold):.5f} std={np.std(per_fold):.5f}")
    return auc, per_fold


def main() -> None:
    raw_train = pl.read_parquet(TRAIN_PARQUET).to_pandas()
    raw_test = pl.read_parquet(TEST_PARQUET).to_pandas()
    print(f"train {raw_train.shape}  test {raw_test.shape}")
    y = raw_train[TARGET].astype(int).to_numpy()
    folds = split_indices(raw_train)

    has_cycle007 = all(c in raw_train.columns for c in CYCLE007_FEATURES)
    print(f"cycle-#007 features present: {has_cycle007}")

    # Pre-cycle-#006 base set: drop id, target, and the 3 cycle-#006 features (and
    # cycle-#007 features too, since they didn't exist for the pre-#006 baseline).
    base_cols = [
        c for c in raw_train.columns
        if c not in (TARGET, ID_COL, *CYCLE006_FEATURES, *CYCLE007_FEATURES)
    ]
    # Cycle-#006 set: drop id, target, and the 3 cycle-#007 features.
    full_cols = [
        c for c in raw_train.columns
        if c not in (TARGET, ID_COL, *CYCLE007_FEATURES)
    ]
    # Cycle-#007 set: drop only id, target.
    all_cols = [c for c in raw_train.columns if c not in (TARGET, ID_COL)]
    print(f"pre-#006 feature set: {len(base_cols)} features")
    print(f"#006 feature set:     {len(full_cols)} features")
    if has_cycle007:
        print(f"#007 feature set:     {len(all_cols)} features")

    # --- LGB on pre-#006 set ---
    lgb_oof, lgb_test, lgb_fold_aucs, lgb_fold_iters = run_lgb(
        base_cols, folds, "pre-#006"
    )
    lgb_auc = report("LGB (pre-#006)", y, lgb_oof, folds, lgb_fold_aucs, lgb_fold_iters)
    _save_oof("lgb", raw_train, raw_test, y, lgb_oof, lgb_test)

    # --- CB#004 on pre-#006 set ---
    cb004_oof, cb004_test, cb004_fold_aucs, cb004_fold_iters = run_cb(
        base_cols, folds, "CB#004 / pre-#006"
    )
    cb004_auc = report("CB#004", y, cb004_oof, folds, cb004_fold_aucs, cb004_fold_iters)
    _save_oof("cb004", raw_train, raw_test, y, cb004_oof, cb004_test)

    # --- CB#006 on #006 set ---
    cb006_oof, cb006_test, cb006_fold_aucs, cb006_fold_iters = run_cb(
        full_cols, folds, "CB#006 / #006 features"
    )
    cb006_auc = report("CB#006", y, cb006_oof, folds, cb006_fold_aucs, cb006_fold_iters)
    _save_oof("cb006", raw_train, raw_test, y, cb006_oof, cb006_test)

    # --- CB#007 on full set (only if cycle-#007 features present) ---
    cb007_auc = None
    cb007_oof = None
    cb007_test = None
    if has_cycle007:
        cb007_oof, cb007_test, cb007_fold_aucs, cb007_fold_iters = run_cb(
            all_cols, folds, "CB#007 / #007 features"
        )
        cb007_auc = report("CB#007", y, cb007_oof, folds, cb007_fold_aucs, cb007_fold_iters)
        _save_oof("cb007", raw_train, raw_test, y, cb007_oof, cb007_test)

    # --- 3-way ensemble (cycle-#006 weights) ---
    ens3_oof = 0.10 * lgb_oof + 0.40 * cb004_oof + 0.50 * cb006_oof
    ens3_test = 0.10 * lgb_test + 0.40 * cb004_test + 0.50 * cb006_test
    ens3_auc, ens3_fold = _per_fold_ens(
        y, ens3_oof, folds, "3-way ensemble (LGB=0.10, CB#004=0.40, CB#006=0.50)"
    )
    _save_oof("ensemble3", raw_train, raw_test, y, ens3_oof, ens3_test)

    # --- 4-way ensemble (cycle-#007 fixed weights), if CB#007 trained ---
    ens4_auc = None
    if has_cycle007:
        ens4_oof = 0.10 * lgb_oof + 0.30 * cb004_oof + 0.30 * cb006_oof + 0.30 * cb007_oof
        ens4_test = 0.10 * lgb_test + 0.30 * cb004_test + 0.30 * cb006_test + 0.30 * cb007_test
        ens4_auc, ens4_fold = _per_fold_ens(
            y, ens4_oof, folds, "4-way ensemble (LGB=0.10, each CB=0.30)"
        )
        _save_oof("ensemble4", raw_train, raw_test, y, ens4_oof, ens4_test)

    # --- Summary table ---
    print("\n=== Per-component summary (seed={}) ===".format(SEED))
    print(f"  LGB        OOF AUC = {lgb_auc:.5f}")
    print(f"  CB#004     OOF AUC = {cb004_auc:.5f}")
    print(f"  CB#006     OOF AUC = {cb006_auc:.5f}")
    if cb007_auc is not None:
        print(f"  CB#007     OOF AUC = {cb007_auc:.5f}")
    print(f"  3-way ens  OOF AUC = {ens3_auc:.5f}")
    if ens4_auc is not None:
        print(f"  4-way ens  OOF AUC = {ens4_auc:.5f}")


if __name__ == "__main__":
    main()
