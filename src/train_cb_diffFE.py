"""Experiment 014 (cycle 4) — CB-tuned with raised iter cap only.

Single change vs experiment 012's CB-tuned recipe:
  iter cap raised: iterations 5000 → 8000, early_stopping_rounds 300 → 500.
  Experiment 012 hit the 5000 cap in every fold without early-stop firing,
  so the model wasn't done training.

Digit/signature features were initially planned but dropped after a univariate
signal check showed most "high spread" digit features were either redundant
with the raw numerics (e.g. TyreLife_int_digit_2 ≈ bucketed TyreLife) or
spurious 1-row outliers. Expected marginal lift from digit features was 0 to
+0.0005, not worth the +160-column feature explosion and 4-5h CPU cost.
Defer digit features to a future experiment if needed.

All other settings (data, CV, HPs, base FE) identical to experiment 012.

Outputs:
  data/oof_cb_tuned_exp14.parquet      OOF predictions
  data/submission_cb_tuned_exp14.csv   test predictions
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from data import DATA, ID_COL, TARGET, build_gbdt_diffFE
OOF_OUT = DATA / "oof_cb_diffFE.parquet"
SUB_OUT = DATA / "submission_cb_diffFE.csv"

N_SPLITS = 5
SEED = 42

# Experiment 014: iter cap raised 5000 → 8000, early_stopping 300 → 500. All
# other HPs identical to experiment 012.
CB_PARAMS = {
    "iterations": 8000,
    "learning_rate": 0.018,
    "depth": 8,
    "l2_leaf_reg": 8.5,
    "random_strength": 0.65,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 0.45,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
    "early_stopping_rounds": 500,
    "task_type": "CPU",
    "thread_count": -1,
    "allow_writing_files": False,
    "verbose": 200,
}

def main() -> None:
    print("Loading data + building diffFE features (src/data.py)...")
    train, test, ext, feature_cols, cat_cols = build_gbdt_diffFE(cat_dtype=False)
    cat_indices = [feature_cols.index(c) for c in cat_cols if c in feature_cols]
    print(f"  train {train.shape}  test {test.shape}  external {ext.shape}")
    print(f"  external pos rate: {ext[TARGET].mean():.4f}  competition train pos rate: {train[TARGET].mean():.4f}")
    print(f"using {len(feature_cols)} features, {len(cat_indices)} categorical")

    X = train[feature_cols]
    y = train[TARGET].astype(int).to_numpy()
    X_test = test[feature_cols]
    X_ext = ext[feature_cols]
    y_ext = ext[TARGET].astype(int).to_numpy()

    strat_key = train["Year"].astype(str) + "_" + train[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train), dtype=np.float64)
    test_preds = np.zeros(len(test), dtype=np.float64)
    fold_aucs: list[float] = []
    fold_iters: list[int] = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, strat_key), start=1):
        t0 = time.time()
        # Augment training with external data (val stays competition-only)
        X_tr = pd.concat([X.iloc[tr_idx], X_ext], axis=0, ignore_index=True)
        y_tr = np.concatenate([y[tr_idx], y_ext])
        X_va = X.iloc[va_idx]
        y_va = y[va_idx]

        params = dict(CB_PARAMS)
        params["random_seed"] = SEED + fold  # per-fold seed variation, mirroring public notebook
        train_pool = Pool(X_tr, label=y_tr, cat_features=cat_indices)
        valid_pool = Pool(X_va, label=y_va, cat_features=cat_indices)
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

        va_pred = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = va_pred
        test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
        fold_auc = roc_auc_score(y_va, va_pred)
        fold_aucs.append(fold_auc)
        fold_iters.append(int(model.tree_count_))
        print(
            f"fold {fold}/{N_SPLITS}  AUC={fold_auc:.5f}  iters={model.tree_count_}  "
            f"train_rows={len(X_tr):,}  ({time.time()-t0:.1f}s)",
            flush=True,
        )

    oof_auc = roc_auc_score(y, oof)
    print()
    print(f"per-fold AUC: mean={np.mean(fold_aucs):.5f}  std={np.std(fold_aucs):.5f}  iters={fold_iters}")
    print(f"OOF AUC:      {oof_auc:.5f}")
    print(f"  (vs CB#006 0.94806, Δ = {oof_auc - 0.94806:+.5f})")

    oof_df = pd.DataFrame(
        {"id": train[ID_COL], "Year": train["Year"], "target": y, "oof": oof}
    )
    print("\nOOF AUC by Year:")
    for year, g in oof_df.groupby("Year"):
        if g["target"].nunique() > 1:
            auc = roc_auc_score(g["target"], g["oof"])
            print(f"  {year}: AUC={auc:.5f}  n={len(g):,}  pos_rate={g['target'].mean():.4f}")

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
