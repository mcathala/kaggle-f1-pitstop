"""Cycle #010 — stacking meta-model on base-model OOFs.

Tests whether a meta-learner on the base models' OOF predictions extracts
diversity beyond fixed-weight blending. After two consecutive Inconclusive
CB-variant cycles (7 and 9) each producing ~+0.0001 ensemble lift, this
cycle pivots to ensemble-method exploration before committing to model-family
changes (NN, etc.).

Base inputs (all at seed=42, identical CV split):
  data/oof_lgb_seed42.parquet
  data/oof_cb004_seed42.parquet
  data/oof_cb006_seed42.parquet
  data/oof_cb007_seed42.parquet   (cycle 7 — free to include)

Meta-models trained in 5-fold StratifiedKFold on Year × PitNextLap (same
random_state=42 as base):
  (1) Logistic regression on logit(base_oofs) — closed-form, near-zero overfit risk.
  (2) Shallow LightGBM on logit(base_oofs) — depth 3, num_leaves 7, lr 0.05.
      Slightly more flexible: can learn slice-specific weights.

Why logit-transform: probabilities are bounded [0,1] and additive blending
implicitly assumes log-odds are additive. Stacking in logit space lets a linear
meta-model recover an additive log-odds weighting that's calibrated by sigmoid.

Reports meta-OOF AUC and compares to the fixed-weight 3-way baseline (0.94866).
No test-set submission written this cycle — that's a follow-up only if the
verdict is Keep.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA = Path(__file__).resolve().parent.parent.parent / "data"
TRAIN_PARQUET = DATA / "train_features.parquet"

OOF_PATHS = {
    "lgb": DATA / "oof_lgb_seed42.parquet",
    "cb004": DATA / "oof_cb004_seed42.parquet",
    "cb006": DATA / "oof_cb006_seed42.parquet",
    "cb007": DATA / "oof_cb007_seed42.parquet",
}

TARGET = "PitNextLap"
ID_COL = "id"
N_SPLITS = 5
SEED = 42

# Fixed-weight baselines for comparison
FIXED_3WAY = {"lgb": 0.10, "cb004": 0.40, "cb006": 0.50}
FIXED_4WAY = {"lgb": 0.10, "cb004": 0.30, "cb006": 0.30, "cb007": 0.30}

LGB_META_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 7,
    "max_depth": 3,
    "min_data_in_leaf": 200,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "verbose": -1,
    "seed": SEED,
}


def safe_logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    return logit(np.clip(p, eps, 1 - eps))


def load_meta_features() -> tuple[pd.DataFrame, np.ndarray]:
    feats = (
        pl.read_parquet(TRAIN_PARQUET)
        .select([ID_COL, "Year", TARGET])
        .to_pandas()
        .set_index(ID_COL)
    )
    for k, p in OOF_PATHS.items():
        df = pd.read_parquet(p).set_index("id")
        feats[f"oof_{k}"] = df["oof"]
    feats = feats.reset_index()
    y = feats[TARGET].astype(int).to_numpy()
    return feats, y


def fixed_blend(feats: pd.DataFrame, weights: dict) -> np.ndarray:
    return sum(w * feats[f"oof_{k}"].to_numpy() for k, w in weights.items())


def stack_logreg(X: np.ndarray, y: np.ndarray, folds) -> np.ndarray:
    oof = np.zeros(len(y))
    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        clf = LogisticRegression(max_iter=500, C=1.0, solver="liblinear")
        clf.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = clf.predict_proba(X[va_idx])[:, 1]
    return oof


def stack_lgb(X: np.ndarray, y: np.ndarray, folds, feature_names: list[str]) -> np.ndarray:
    oof = np.zeros(len(y))
    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        dtr = lgb.Dataset(X[tr_idx], label=y[tr_idx], feature_name=feature_names)
        dva = lgb.Dataset(X[va_idx], label=y[va_idx], feature_name=feature_names, reference=dtr)
        model = lgb.train(
            LGB_META_PARAMS,
            dtr,
            num_boost_round=1000,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        oof[va_idx] = model.predict(X[va_idx], num_iteration=model.best_iteration)
    return oof


def per_fold(y, oof, folds) -> tuple[list[float], float, float]:
    p = [roc_auc_score(y[va_idx], oof[va_idx]) for _, va_idx in folds]
    return p, float(np.mean(p)), float(np.std(p))


def main() -> None:
    feats, y = load_meta_features()
    print(f"loaded {len(feats):,} rows, {y.sum():,} positives ({y.mean():.4f} rate)")

    base_keys = list(OOF_PATHS.keys())
    print("\n=== Base OOF AUCs (sanity) ===")
    for k in base_keys:
        a = roc_auc_score(y, feats[f"oof_{k}"])
        print(f"  {k:5s} = {a:.5f}")

    strat_key = feats["Year"].astype(str) + "_" + feats[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(kf.split(feats, strat_key))

    # Fixed blends as reference
    print("\n=== Fixed-weight baselines ===")
    b3 = fixed_blend(feats, FIXED_3WAY)
    b3_auc = roc_auc_score(y, b3)
    b3_folds, _, _ = per_fold(y, b3, folds)
    print(f"3-way (LGB+CB#004+CB#006 at 0.10/0.40/0.50): OOF = {b3_auc:.5f}  per-fold std = {np.std(b3_folds):.5f}")
    b4 = fixed_blend(feats, FIXED_4WAY)
    b4_auc = roc_auc_score(y, b4)
    b4_folds, _, _ = per_fold(y, b4, folds)
    print(f"4-way cycle 7 (+CB#007 at 0.30 each CB):     OOF = {b4_auc:.5f}  per-fold std = {np.std(b4_folds):.5f}")

    # Stack 1: logistic regression on 3-base logits
    print("\n=== Meta-model 1: Logistic regression on logit(3 base OOFs) ===")
    X3 = np.column_stack([safe_logit(feats[f"oof_{k}"].to_numpy()) for k in ("lgb", "cb004", "cb006")])
    oof_lr3 = stack_logreg(X3, y, folds)
    lr3_auc = roc_auc_score(y, oof_lr3)
    lr3_folds, _, _ = per_fold(y, oof_lr3, folds)
    print(f"meta-OOF AUC = {lr3_auc:.5f}  per-fold std = {np.std(lr3_folds):.5f}  Δ vs 3-way = {lr3_auc - b3_auc:+.5f}")
    print(f"per-fold: {[f'{a:.5f}' for a in lr3_folds]}")

    # Stack 2: logistic regression on 4-base logits
    print("\n=== Meta-model 2: Logistic regression on logit(4 base OOFs incl CB#007) ===")
    X4 = np.column_stack([safe_logit(feats[f"oof_{k}"].to_numpy()) for k in base_keys])
    oof_lr4 = stack_logreg(X4, y, folds)
    lr4_auc = roc_auc_score(y, oof_lr4)
    lr4_folds, _, _ = per_fold(y, oof_lr4, folds)
    print(f"meta-OOF AUC = {lr4_auc:.5f}  per-fold std = {np.std(lr4_folds):.5f}  Δ vs 3-way = {lr4_auc - b3_auc:+.5f}  Δ vs 4-way = {lr4_auc - b4_auc:+.5f}")
    print(f"per-fold: {[f'{a:.5f}' for a in lr4_folds]}")

    # Stack 3: shallow LightGBM on 4-base logits
    print("\n=== Meta-model 3: Shallow LightGBM on logit(4 base OOFs) ===")
    oof_lgb4 = stack_lgb(X4, y, folds, [f"logit_{k}" for k in base_keys])
    lgb4_auc = roc_auc_score(y, oof_lgb4)
    lgb4_folds, _, _ = per_fold(y, oof_lgb4, folds)
    print(f"meta-OOF AUC = {lgb4_auc:.5f}  per-fold std = {np.std(lgb4_folds):.5f}  Δ vs 3-way = {lgb4_auc - b3_auc:+.5f}  Δ vs 4-way = {lgb4_auc - b4_auc:+.5f}")
    print(f"per-fold: {[f'{a:.5f}' for a in lgb4_folds]}")

    # Summary + verdict
    pass_bar = b3_auc + max(0.5 * np.std(b3_folds), 0.00020)
    print(f"\n=== Verdict gate ===")
    print(f"baseline (3-way) OOF      = {b3_auc:.5f}  fold std = {np.std(b3_folds):.5f}")
    print(f"magnitude floor (0.5*std or min_delta=0.00020): {pass_bar - b3_auc:.6f}")
    print(f"pass bar (3-way + floor)  = {pass_bar:.5f}")
    print(f"\nbest meta-model:")
    candidates = [
        ("logreg-3", lr3_auc, lr3_folds),
        ("logreg-4", lr4_auc, lr4_folds),
        ("lgb-4", lgb4_auc, lgb4_folds),
    ]
    best = max(candidates, key=lambda x: x[1])
    name, auc, fold_list = best
    print(f"  {name}: OOF AUC {auc:.5f}  fold std {np.std(fold_list):.5f}")
    delta = auc - b3_auc
    n_up = sum(b > a for a, b in zip(b3_folds, fold_list))
    print(f"  Δ vs 3-way = {delta:+.5f}  ({n_up}/{N_SPLITS} folds improved)")
    if auc >= pass_bar and n_up >= 3:
        print("  >>> KEEP — clears magnitude and direction gates.")
    elif auc <= b3_auc:
        print("  >>> DISCARD — at or below the 3-way baseline.")
    else:
        print("  >>> INCONCLUSIVE — direction positive but magnitude below floor.")


if __name__ == "__main__":
    main()
