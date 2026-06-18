"""Experiment 028 (cycle 10) — CatBoost with rich-categorical FE.

Pivots from cycle 12-14's mostly-numeric CatBoost recipe (custom interactions,
group statistics, frequency encoding) to a categorical-dense recipe where
the bulk of features are STRING categoricals: digit-position cats,
multi-resolution numeric-as-categorical, and a full pairwise-bigram grid.

The mechanism is CatBoost's ordered target encoding: each unique categorical
bucket gets its own cross-fold-safe target-encoded mean. The value emerges
not from any single digit/bigram cat (univariate signal is low — cycle 14
verified this) but from CB combining them through ordered TE.

Outputs:
  data/oof_cb_rich.parquet
  data/submission_cb_rich.csv
"""

from itertools import combinations
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parent.parent.parent / "data"
TRAIN_CSV = DATA / "train.csv"
TEST_CSV = DATA / "test.csv"
EXTERNAL_CSV = DATA / "f1_strategy_dataset.csv"
OOF_OUT = DATA / "oof_cb_rich.parquet"
SUB_OUT = DATA / "submission_cb_rich.csv"

TARGET = "PitNextLap"
ID_COL = "id"
BASE_CATS = ["Driver", "Compound", "Race"]
N_SPLITS = 5
SEED = 42
MODEL_SEEDS = [42, 43]

CB_PARAMS = {
    "iterations": 10000,
    "learning_rate": 0.03,
    "depth": 8,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "early_stopping_rounds": 200,
    "bagging_temperature": 0.8,
    "bootstrap_type": "Bayesian",
    "task_type": "CPU",
    "thread_count": -1,
    "allow_writing_files": False,
    "verbose": 500,
}


# ============================================================
# Feature engineering — categorical-dense
# ============================================================

# Numerics from which we extract digit-position cats
DIGIT_BASE = [
    # Phase-1 probe 1 found these 4 continuous numerics carry top per-feature importance.
    # Restricting digits to these 4 keeps the FE matrix lean (24 digit cols vs 67 if all 11
    # raw numerics were included) and avoids spurious digits from discrete cols like PitStop.
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
]
DECIMAL_BASE = ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress"]

# Numerics treated as multi-resolution string categoricals
NUM_AS_CAT_CONFIG = {
    "LapTime (s)":             {"round_digits": [1, 0], "round_steps": [0.5, 1.0, 2.0, 5.0]},
    "LapTime_Delta":           {"round_digits": [1, 0], "round_steps": [0.5, 1.0, 2.0, 5.0, 10.0]},
    "Cumulative_Degradation":  {"round_digits": [1, 0], "round_steps": [1.0, 2.0, 5.0, 10.0, 20.0]},
}

# Source columns for the full pairwise BIGRAM grid (C(11, 2) = 55 pairs)
BIGRAM_BASE = [
    "Driver", "Compound", "Race", "Year", "PitStop", "LapNumber", "Stint",
    "TyreLife", "Position", "RaceProgress", "Position_Change",
]

INT_POSITIONS = [1, 10, 100, 1000]
DECIMAL_POSITIONS = [1, 2, 3]


def safe_colname(c: str) -> str:
    return (c.replace(" ", "_").replace("(", "").replace(")", "")
              .replace("/", "_").replace("-", "_"))


def add_digit_features(df: pd.DataFrame) -> list[str]:
    """For each base numeric, add sign + integer-position-digit + decimal-position-digit
    columns as int8 categoricals. Mutates df in place; returns the list of added column names.
    """
    added = []
    for c in DIGIT_BASE:
        if c not in df.columns:
            continue
        sc = safe_colname(c)
        x = pd.to_numeric(df[c], errors="coerce").astype(float).to_numpy()
        x = np.round(x, 6)
        x_filled = np.nan_to_num(x, nan=0.0)

        sign_col = f"{sc}_sign"
        df[sign_col] = np.sign(x_filled).astype(np.int8)
        added.append(sign_col)

        x_abs = np.abs(x_filled)
        int_part = np.floor(x_abs).astype(np.int64)
        for p in INT_POSITIONS:
            nc = f"{sc}_digit_{p}s"
            df[nc] = ((int_part // p) % 10).astype(np.int8)
            added.append(nc)

    for c in DECIMAL_BASE:
        if c not in df.columns:
            continue
        sc = safe_colname(c)
        x = pd.to_numeric(df[c], errors="coerce").astype(float).to_numpy()
        x = np.round(x, 6)
        x_abs = np.abs(np.nan_to_num(x, nan=0.0))
        for d in DECIMAL_POSITIONS:
            nc = f"{sc}_decimal_digit_{d}"
            df[nc] = (np.floor(x_abs * (10 ** d)).astype(np.int64) % 10).astype(np.int8)
            added.append(nc)
    return added


def add_num_as_cat_features(df: pd.DataFrame) -> list[str]:
    """For each configured numeric column, add an exact-precision string + several rounded
    versions, each as a string categorical. Mutates df in place; returns added columns.
    """
    added = []
    for c, cfg in NUM_AS_CAT_CONFIG.items():
        if c not in df.columns:
            continue
        # Exact precision string
        exact_col = f"{c}_cat"
        df[exact_col] = df[c].astype(str)
        added.append(exact_col)
        # Decimal-rounded variants
        for d in cfg["round_digits"]:
            nc = f"{c}_round{d}_cat"
            df[nc] = df[c].round(d).astype(str)
            added.append(nc)
        # Step-rounded variants
        for step in cfg["round_steps"]:
            step_name = str(step).replace(".", "p")
            nc = f"{c}_round_step_{step_name}_cat"
            df[nc] = (np.round(df[c] / step) * step).astype(str)
            added.append(nc)
    return added


def add_bigram_features(df: pd.DataFrame) -> list[str]:
    """Add all C(BIGRAM_BASE, 2) pairwise bigram string-concat features."""
    added = []
    pairs = list(combinations(BIGRAM_BASE, 2))
    for c1, c2 in pairs:
        if c1 not in df.columns or c2 not in df.columns:
            continue
        nc = f"bigram__{c1}__{c2}"
        df[nc] = (df[c1].astype(str).fillna("__MISSING__") + "__"
                  + df[c2].astype(str).fillna("__MISSING__"))
        added.append(nc)
    return added


def normalize_cat_columns(frames: list[pd.DataFrame], cat_cols: list[str]) -> None:
    """Cast every cat column to string with consistent missing token. CatBoost expects strings."""
    for f in frames:
        for c in cat_cols:
            if c in f.columns:
                f[c] = f[c].where(f[c].notna(), "__MISSING__").astype(str)


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("Loading data...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    orig = pd.read_csv(EXTERNAL_CSV).drop(columns=["Normalized_TyreLife"], errors="ignore")
    print(f"  train {train.shape}  test {test.shape}  orig {orig.shape}")

    # Build the digit features BEFORE we touch the cat columns, since digits use the numeric values
    for frame in (train, test, orig):
        add_digit_features(frame)
        add_num_as_cat_features(frame)
        add_bigram_features(frame)

    digit_cols = [c for c in train.columns if "_digit_" in c or c.endswith("_sign")]
    num_as_cat_cols = [c for c in train.columns if c.endswith("_cat")]
    bigram_cols = [c for c in train.columns if c.startswith("bigram__")]
    base_features = [c for c in train.columns if c not in (ID_COL, TARGET)]

    print(f"  digit features:       {len(digit_cols)}")
    print(f"  num-as-cat features:  {len(num_as_cat_cols)}")
    print(f"  bigram features:      {len(bigram_cols)}")
    print(f"  total features:       {len(base_features)}")

    # CatBoost categorical column list = BASE_CATS + all new cat-type cols.
    # digit_cols are integer-typed but treated as categoricals by CB.
    cat_features = sorted(set(BASE_CATS + digit_cols + num_as_cat_cols + bigram_cols))
    cat_features = [c for c in cat_features if c in train.columns]
    print(f"  CatBoost cat_features: {len(cat_features)}")

    # Normalize ALL cat columns to string (CatBoost prefers explicit strings)
    normalize_cat_columns([train, test, orig], cat_features)

    # Build feature matrices
    X = train[base_features].copy()
    X_test = test[base_features].copy()
    X_orig = orig[base_features].copy()

    y = train[TARGET].astype(int)
    y_orig = orig[TARGET].astype(int).reset_index(drop=True)
    train_id = train[ID_COL]
    test_id = test[ID_COL]

    # CV stratified on Year × PitNextLap (matches everything else in the project)
    strat_key = train["Year"].astype(str) + "_" + train[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof = np.zeros(len(train), dtype=np.float64)
    test_preds = np.zeros(len(test), dtype=np.float64)
    fold_aucs = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, strat_key), start=1):
        t0 = time.time()
        X_tr = pd.concat([X.iloc[tr_idx].reset_index(drop=True),
                          X_orig.reset_index(drop=True)], axis=0, ignore_index=True)
        y_tr = pd.concat([y.iloc[tr_idx].reset_index(drop=True), y_orig], axis=0, ignore_index=True)
        X_va = X.iloc[va_idx].copy()
        y_va = y.iloc[va_idx]

        if fold == 1:
            print(f"  fold 1: train rows = {len(X_tr):,}  val rows = {len(X_va):,}  features = {X_tr.shape[1]}")

        train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
        val_pool = Pool(X_va, y_va, cat_features=cat_features)
        test_pool = Pool(X_test, cat_features=cat_features)

        # 2-seed ensemble per fold for variance reduction
        fold_va_pred = np.zeros(len(X_va), dtype=np.float64)
        fold_test_pred = np.zeros(len(X_test), dtype=np.float64)

        for s_idx, seed in enumerate(MODEL_SEEDS):
            t_seed = time.time()
            model_seed = seed + fold * 100
            params = dict(CB_PARAMS)
            params["random_seed"] = model_seed

            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=val_pool, use_best_model=True)
            va_pred = model.predict_proba(val_pool)[:, 1]
            tst_pred = model.predict_proba(test_pool)[:, 1]

            fold_va_pred += va_pred / len(MODEL_SEEDS)
            fold_test_pred += tst_pred / len(MODEL_SEEDS)

            seed_auc = roc_auc_score(y_va, va_pred)
            print(f"    fold {fold} seed {seed} (model_seed={model_seed})  AUC={seed_auc:.5f}  best_iter={model.tree_count_}  ({time.time()-t_seed:.0f}s)", flush=True)

        oof[va_idx] = fold_va_pred
        test_preds += fold_test_pred / N_SPLITS
        fold_auc = roc_auc_score(y_va, fold_va_pred)
        fold_aucs.append(fold_auc)
        print(f"fold {fold}/{N_SPLITS}  ensemble AUC={fold_auc:.5f}  ({time.time()-t0:.0f}s)", flush=True)

    oof_auc = roc_auc_score(y, oof)
    print(f"\nOOF AUC = {oof_auc:.5f}  per-fold mean {np.mean(fold_aucs):.5f}  std {np.std(fold_aucs):.5f}")
    print(f"Per-fold: {[f'{a:.5f}' for a in fold_aucs]}")
    print(f"Baselines: CB-tuned-exp14 = 0.95114, target for cycle 10 = >=0.95150")

    pd.DataFrame(
        {"id": train_id, "Year": train["Year"], "target": y.to_numpy(), "oof": oof}
    ).to_parquet(OOF_OUT, index=False)
    pd.DataFrame({"id": test_id, TARGET: test_preds}).sort_values("id").reset_index(drop=True).to_csv(
        SUB_OUT, index=False
    )
    print(f"wrote {OOF_OUT.name} and {SUB_OUT.name}")


if __name__ == "__main__":
    main()
