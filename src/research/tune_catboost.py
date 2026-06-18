"""Optuna search over CatBoost hyperparameters.

Two stages:
  1) Search — 30 trials, 3-fold CV (fast proxy), 2000 max iters, early stop 50.
  2) Validate winner — full 5-fold, 5000 iters, early stop 100. Writes
     data/oof_catboost.parquet and data/submission_catboost.csv if the winner
     beats the cycle-#004 standalone (0.94774).

CV protocol: same seed=42, StratifiedKFold on Year × PitNextLap.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import polars as pl
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA = Path(__file__).resolve().parent.parent.parent / "data"
TRAIN_PARQUET = DATA / "train_features.parquet"
TEST_PARQUET = DATA / "test_features.parquet"
OOF_OUT = DATA / "oof_catboost.parquet"
SUB_OUT = DATA / "submission_catboost.csv"
STUDY_LOG = DATA / "optuna_catboost_study.json"

TARGET = "PitNextLap"
ID_COL = "id"
CATEGORICAL = ["Driver", "Race", "Compound"]
SEED = 42

# Search stage: 3-fold CV, lower iter cap.
SEARCH_FOLDS = 3
SEARCH_ITERS = 1500
SEARCH_EARLY_STOP = 50
N_TRIALS = 20

# Validation stage: full 5-fold, full iter budget.
VAL_FOLDS = 5
VAL_ITERS = 5000
VAL_EARLY_STOP = 100

CB_BASELINE_AUC = 0.94774  # cycle #004 standalone OOF


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pl.read_parquet(TRAIN_PARQUET).to_pandas()
    test = pl.read_parquet(TEST_PARQUET).to_pandas()
    for c in CATEGORICAL:
        train[c] = train[c].astype(str).fillna("__NA__")
        test[c] = test[c].astype(str).fillna("__NA__")
    return train, test


def cv_auc(params: dict, train: pd.DataFrame, n_folds: int, iters: int, early_stop: int) -> tuple[float, list[float], list[int]]:
    """Run n_folds CV with the given CatBoost params. Returns (mean AUC, per-fold AUC, per-fold iters)."""
    feature_cols = [c for c in train.columns if c not in (TARGET, ID_COL)]
    cat_indices = [feature_cols.index(c) for c in CATEGORICAL]
    X = train[feature_cols]
    y = train[TARGET].astype(int).to_numpy()
    strat = train["Year"].astype(str) + "_" + train[TARGET].astype(int).astype(str)

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    aucs: list[float] = []
    iters_list: list[int] = []
    full_params = {
        "iterations": iters,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": SEED,
        "od_type": "Iter",
        "od_wait": early_stop,
        "verbose": 0,
        "task_type": "CPU",
        "thread_count": -1,
        **params,
    }
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, strat), start=1):
        train_pool = Pool(X.iloc[tr_idx], label=y[tr_idx], cat_features=cat_indices)
        valid_pool = Pool(X.iloc[va_idx], label=y[va_idx], cat_features=cat_indices)
        model = CatBoostClassifier(**full_params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        va_pred = model.predict_proba(X.iloc[va_idx])[:, 1]
        aucs.append(roc_auc_score(y[va_idx], va_pred))
        iters_list.append(int(model.tree_count_))
    return float(np.mean(aucs)), aucs, iters_list


def search_objective(trial: optuna.Trial, train: pd.DataFrame) -> float:
    params = {
        "depth": trial.suggest_int("depth", 6, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
        "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
        "min_data_in_leaf": trial.suggest_categorical("min_data_in_leaf", [1, 5, 20, 50]),
    }
    t0 = time.time()
    auc, fold_aucs, fold_iters = cv_auc(
        params, train, n_folds=SEARCH_FOLDS, iters=SEARCH_ITERS, early_stop=SEARCH_EARLY_STOP
    )
    elapsed = time.time() - t0
    print(
        f"trial {trial.number:>3}: AUC={auc:.5f} "
        f"folds={[f'{a:.5f}' for a in fold_aucs]} iters={fold_iters} "
        f"({elapsed:.0f}s) | {params}"
    )
    return auc


def fit_full_5fold(params: dict, train: pd.DataFrame, test: pd.DataFrame) -> tuple[float, list[float], np.ndarray, np.ndarray]:
    """Full 5-fold validation. Returns (OOF AUC, fold AUCs, OOF preds, test preds)."""
    feature_cols = [c for c in train.columns if c not in (TARGET, ID_COL)]
    cat_indices = [feature_cols.index(c) for c in CATEGORICAL]
    X = train[feature_cols]
    y = train[TARGET].astype(int).to_numpy()
    X_test = test[feature_cols]
    strat = train["Year"].astype(str) + "_" + train[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=VAL_FOLDS, shuffle=True, random_state=SEED)

    oof = np.zeros(len(train), dtype=np.float64)
    test_preds = np.zeros(len(test), dtype=np.float64)
    fold_aucs: list[float] = []

    full_params = {
        "iterations": VAL_ITERS,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": SEED,
        "od_type": "Iter",
        "od_wait": VAL_EARLY_STOP,
        "verbose": 0,
        "task_type": "CPU",
        "thread_count": -1,
        **params,
    }
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, strat), start=1):
        t0 = time.time()
        train_pool = Pool(X.iloc[tr_idx], label=y[tr_idx], cat_features=cat_indices)
        valid_pool = Pool(X.iloc[va_idx], label=y[va_idx], cat_features=cat_indices)
        model = CatBoostClassifier(**full_params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        va_pred = model.predict_proba(X.iloc[va_idx])[:, 1]
        oof[va_idx] = va_pred
        test_preds += model.predict_proba(X_test)[:, 1] / VAL_FOLDS
        fold_auc = roc_auc_score(y[va_idx], va_pred)
        fold_aucs.append(fold_auc)
        print(
            f"validate fold {fold}/{VAL_FOLDS}  AUC={fold_auc:.5f}  "
            f"iters={model.tree_count_}  ({time.time() - t0:.1f}s)"
        )
    oof_auc = roc_auc_score(y, oof)
    return oof_auc, fold_aucs, oof, test_preds


def main() -> None:
    train, test = load()
    print(f"train {train.shape}  test {test.shape}")

    print(f"\n=== Stage 1: Optuna search, {N_TRIALS} trials, {SEARCH_FOLDS}-fold CV ===\n")
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda t: search_objective(t, train), n_trials=N_TRIALS, show_progress_bar=False)

    print("\n=== Top 5 trials ===")
    sorted_trials = sorted(study.trials, key=lambda t: -t.value)
    for t in sorted_trials[:5]:
        print(f"  trial {t.number}: AUC={t.value:.5f} {t.params}")

    best = study.best_trial
    print(f"\nBest: trial {best.number} AUC={best.value:.5f}")
    print(f"Params: {best.params}")

    print("\n=== Stage 2: Full 5-fold validation of best params ===\n")
    val_oof_auc, val_fold_aucs, oof, test_preds = fit_full_5fold(best.params, train, test)
    print(f"\n>>> Full 5-fold OOF AUC: {val_oof_auc:.5f}  (vs cycle-#004 baseline {CB_BASELINE_AUC:.5f}, "
          f"Δ {val_oof_auc - CB_BASELINE_AUC:+.5f})")
    print(f"Per-fold: {val_fold_aucs}, std={np.std(val_fold_aucs):.5f}")

    # Persist study + winner regardless of outcome.
    out = {
        "best_trial": {"number": best.number, "value": best.value, "params": best.params},
        "search_n_trials": N_TRIALS,
        "search_n_folds": SEARCH_FOLDS,
        "validation_5fold_oof_auc": val_oof_auc,
        "validation_5fold_per_fold": val_fold_aucs,
        "validation_5fold_std": float(np.std(val_fold_aucs)),
        "all_trials": [{"number": t.number, "value": t.value, "params": t.params} for t in sorted_trials[:10]],
    }
    STUDY_LOG.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {STUDY_LOG.name}")

    # Persist OOF + submission only if the tuned model beats the cycle-#004 baseline.
    if val_oof_auc > CB_BASELINE_AUC:
        oof_df = pd.DataFrame({
            "id": train[ID_COL],
            "Year": train["Year"],
            "target": train[TARGET].astype(int).to_numpy(),
            "oof": oof,
            "fold": -1,
        })
        oof_df.to_parquet(OOF_OUT, index=False)
        sub = (
            pd.DataFrame({"id": test[ID_COL], TARGET: test_preds})
            .sort_values("id")
            .reset_index(drop=True)
        )
        sub.to_csv(SUB_OUT, index=False)
        print(f"wrote {OOF_OUT.name}  ({len(oof_df):,} rows)")
        print(f"wrote {SUB_OUT.name}   ({len(sub):,} rows)")
    else:
        print(f"\nTuned model did NOT beat cycle-#004 baseline ({val_oof_auc:.5f} ≤ {CB_BASELINE_AUC:.5f}); "
              f"skipping OOF/submission overwrite. Cycle-#004 outputs preserved.")


if __name__ == "__main__":
    main()
