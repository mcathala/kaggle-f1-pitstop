"""Experiment 031 (cycle 10) — slim CB-rich-FE retry of exp 028.

Differences vs `train_cb_rich_fe.py`:
  - Digit-position features dropped entirely (probe 1: freq/encoding family is bottom-tier).
  - Num-as-cat slimmed: exact + 1-decimal only for {LapTime(s), LapTime_Delta,
    Cumulative_Degradation} = 6 cats (was 23).
  - Bigrams kept at full C(11, 2) = 55.
  - HPs sized for tractable CPU runtime: depth=7, lr=0.06, iter=2000, early_stop=150.
  - CLI: --fold {1..5|-1}, --seeds 42[,43], --iters, --depth, --lr, --early-stop.

Dry-run usage:
  .venv/bin/python -u src/train_cb_rich_fe_slim.py --fold 1 --seeds 42

Full-CV usage:
  .venv/bin/python -u src/train_cb_rich_fe_slim.py --fold -1 --seeds 42,43

Outputs (full-CV only):
  data/oof_cb_rich_slim.parquet
  data/submission_cb_rich_slim.csv

Dry-run prints AUC + wall-clock + rank-corr-with-RealMLP-on-fold-1-val.
"""

import argparse
from itertools import combinations
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parent.parent.parent / "data"
TRAIN_CSV = DATA / "train.csv"
TEST_CSV = DATA / "test.csv"
EXTERNAL_CSV = DATA / "f1_strategy_dataset.csv"
OOF_OUT = DATA / "oof_cb_rich_slim.parquet"
SUB_OUT = DATA / "submission_cb_rich_slim.csv"
REALMLP_OOF = DATA / "oof_realmlp_multiseed.parquet"

TARGET = "PitNextLap"
ID_COL = "id"
BASE_CATS = ["Driver", "Compound", "Race"]
N_SPLITS = 5
CV_SEED = 42

# Numerics with rich tail signal — exact + 1-decimal cat encodings.
NUM_AS_CAT_BASE = ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation"]

# All 11 base columns for the pairwise BIGRAM grid (C(11, 2) = 55 pairs).
BIGRAM_BASE = [
    "Driver", "Compound", "Race", "Year", "PitStop", "LapNumber", "Stint",
    "TyreLife", "Position", "RaceProgress", "Position_Change",
]


# ============================================================
# Feature engineering — categorical-dense (slim)
# ============================================================

def add_num_as_cat_features(df: pd.DataFrame) -> list[str]:
    """For each NUM_AS_CAT_BASE col, add exact-precision string + 1-decimal-rounded string."""
    added = []
    for c in NUM_AS_CAT_BASE:
        if c not in df.columns:
            continue
        exact_col = f"{c}_cat"
        df[exact_col] = df[c].astype(str)
        added.append(exact_col)
        round1_col = f"{c}_round1_cat"
        df[round1_col] = df[c].round(1).astype(str)
        added.append(round1_col)
    return added


def add_bigram_features(df: pd.DataFrame) -> list[str]:
    """Full C(BIGRAM_BASE, 2) = 55 pairwise bigram string-concat features."""
    added = []
    for c1, c2 in combinations(BIGRAM_BASE, 2):
        if c1 not in df.columns or c2 not in df.columns:
            continue
        nc = f"bigram__{c1}__{c2}"
        df[nc] = (df[c1].astype(str).fillna("__MISSING__") + "__"
                  + df[c2].astype(str).fillna("__MISSING__"))
        added.append(nc)
    return added


def normalize_cat_columns(frames: list[pd.DataFrame], cat_cols: list[str]) -> None:
    """Cast every cat col to string with consistent missing token."""
    for f in frames:
        for c in cat_cols:
            if c in f.columns:
                f[c] = f[c].where(f[c].notna(), "__MISSING__").astype(str)


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=-1, help="1..N_SPLITS for one fold; -1 for all")
    p.add_argument("--seeds", type=str, default="42", help="comma-sep model seeds")
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--depth", type=int, default=7)
    p.add_argument("--lr", type=float, default=0.06)
    p.add_argument("--early-stop", type=int, default=150)
    p.add_argument("--bagging-temp", type=float, default=0.8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    dry_run = args.fold != -1

    print(f"Args: fold={args.fold}  seeds={seeds}  iters={args.iters}  depth={args.depth}  "
          f"lr={args.lr}  early_stop={args.early_stop}")

    cb_params = {
        "iterations": args.iters,
        "learning_rate": args.lr,
        "depth": args.depth,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "early_stopping_rounds": args.early_stop,
        "bagging_temperature": args.bagging_temp,
        "bootstrap_type": "Bayesian",
        "task_type": "CPU",
        "thread_count": -1,
        "allow_writing_files": False,
        "verbose": 250,
    }

    print("Loading data...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    orig = pd.read_csv(EXTERNAL_CSV).drop(columns=["Normalized_TyreLife"], errors="ignore")
    print(f"  train {train.shape}  test {test.shape}  orig {orig.shape}")

    for frame in (train, test, orig):
        add_num_as_cat_features(frame)
        add_bigram_features(frame)

    num_as_cat_cols = [c for c in train.columns if c.endswith("_cat")]
    bigram_cols = [c for c in train.columns if c.startswith("bigram__")]
    base_features = [c for c in train.columns if c not in (ID_COL, TARGET)]

    print(f"  num-as-cat features:  {len(num_as_cat_cols)}")
    print(f"  bigram features:      {len(bigram_cols)}")
    print(f"  total features:       {len(base_features)}")

    cat_features = sorted(set(BASE_CATS + num_as_cat_cols + bigram_cols))
    cat_features = [c for c in cat_features if c in train.columns]
    print(f"  CatBoost cat_features: {len(cat_features)}")

    normalize_cat_columns([train, test, orig], cat_features)

    X = train[base_features].copy()
    X_test = test[base_features].copy()
    X_orig = orig[base_features].copy()
    y = train[TARGET].astype(int)
    y_orig = orig[TARGET].astype(int).reset_index(drop=True)
    train_id = train[ID_COL]
    test_id = test[ID_COL]

    strat_key = train["Year"].astype(str) + "_" + train[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)

    oof = np.zeros(len(train), dtype=np.float64)
    test_preds = np.zeros(len(test), dtype=np.float64)
    fold_aucs: list[float] = []

    folds_to_run = [args.fold] if dry_run else list(range(1, N_SPLITS + 1))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, strat_key), start=1):
        if fold not in folds_to_run:
            continue
        t0 = time.time()
        X_tr = pd.concat(
            [X.iloc[tr_idx].reset_index(drop=True), X_orig.reset_index(drop=True)],
            axis=0, ignore_index=True,
        )
        y_tr = pd.concat([y.iloc[tr_idx].reset_index(drop=True), y_orig], axis=0, ignore_index=True)
        X_va = X.iloc[va_idx].copy()
        y_va = y.iloc[va_idx]

        print(f"  fold {fold}: train rows = {len(X_tr):,}  val rows = {len(X_va):,}  "
              f"features = {X_tr.shape[1]}")

        train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
        val_pool = Pool(X_va, y_va, cat_features=cat_features)
        test_pool = Pool(X_test, cat_features=cat_features)

        fold_va_pred = np.zeros(len(X_va), dtype=np.float64)
        fold_test_pred = np.zeros(len(X_test), dtype=np.float64)

        for seed in seeds:
            t_seed = time.time()
            model_seed = seed + fold * 100
            params = dict(cb_params)
            params["random_seed"] = model_seed

            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=val_pool, use_best_model=True)
            va_pred = model.predict_proba(val_pool)[:, 1]
            tst_pred = model.predict_proba(test_pool)[:, 1]

            fold_va_pred += va_pred / len(seeds)
            fold_test_pred += tst_pred / len(seeds)

            seed_auc = roc_auc_score(y_va, va_pred)
            print(f"    fold {fold} seed {seed} (model_seed={model_seed})  AUC={seed_auc:.5f}  "
                  f"best_iter={model.tree_count_}  ({time.time()-t_seed:.0f}s)", flush=True)

        oof[va_idx] = fold_va_pred
        test_preds += fold_test_pred / N_SPLITS
        fold_auc = roc_auc_score(y_va, fold_va_pred)
        fold_aucs.append(fold_auc)
        print(f"fold {fold}/{N_SPLITS}  ensemble AUC={fold_auc:.5f}  "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)

        # Dry-run extra diagnostics: rank-corr with RealMLP-multiseed on this fold's val rows.
        if dry_run and REALMLP_OOF.exists():
            try:
                realmlp_oof = pd.read_parquet(REALMLP_OOF)
                rm_map = dict(zip(realmlp_oof["id"], realmlp_oof["oof"]))
                rm_va = train.iloc[va_idx]["id"].map(rm_map).to_numpy()
                if not np.isnan(rm_va).any():
                    rho, _ = spearmanr(fold_va_pred, rm_va)
                    print(f"  fold {fold} rank-corr-with-RealMLP-multiseed = {rho:.5f}", flush=True)
                else:
                    print(f"  fold {fold} rank-corr skipped: RealMLP OOF missing ids", flush=True)
            except Exception as e:
                print(f"  fold {fold} rank-corr skipped: {e}", flush=True)

    if dry_run:
        print(f"\nDRY-RUN fold-{args.fold} AUC = {fold_aucs[0]:.5f}")
        return

    oof_auc = roc_auc_score(y, oof)
    print(f"\nOOF AUC = {oof_auc:.5f}  per-fold mean {np.mean(fold_aucs):.5f}  "
          f"std {np.std(fold_aucs):.5f}")
    print(f"Per-fold: {[f'{a:.5f}' for a in fold_aucs]}")
    print(f"Baselines: CB-tuned-exp14 = 0.95114, target for cycle 10 = >= 0.95150")

    pd.DataFrame(
        {"id": train_id, "Year": train["Year"], "target": y.to_numpy(), "oof": oof}
    ).to_parquet(OOF_OUT, index=False)
    pd.DataFrame({"id": test_id, TARGET: test_preds}).sort_values("id").reset_index(drop=True).to_csv(
        SUB_OUT, index=False
    )
    print(f"wrote {OOF_OUT.name} and {SUB_OUT.name}")


if __name__ == "__main__":
    main()
