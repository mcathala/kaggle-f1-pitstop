"""Experiment 071 (cycle 17) — pseudo-CB on canonical CB-exp14 recipe (round-1 self-training).

Cycle 11 mapped the +0.00655 LB gap to 5 named components. This cycle attacks
3 of them in one pass:
  1. External dataset augmentation (data/f1_strategy_dataset.csv)
  2. Tuned CatBoost HPs (lr 0.018, l2 8.5, balanced class weights, bayesian
     bootstrap with bagging temp 0.45, random_strength 0.65, per-fold seed)
  3. Cross-cat + frequency + group-statistic features (inline FE here so we
     don't disturb the existing features.py / training scripts)

Deferred to a future cycle: digit/signature exploit features (component 4),
RealMLP NN (component 5).

CV protocol: 5-fold StratifiedKFold on Year × PitNextLap, seed 42 — identical
to all prior cycles so results are directly comparable. Each fold's training
set = (4/5 competition train) + (all external data). The validation set is
competition-only.

GPU is not available locally, so we cap iterations at 5000 with
early_stopping_rounds=300 instead of the public notebook's 11000/500. The
0.95259 notebook reports best_iter around 3000-5000 at lr 0.018, so we should
not be undertraining.

Outputs:
  data/oof_cb_tuned.parquet      OOF predictions (id, Year, target, oof)
  data/submission_cb_tuned.csv   id, PitNextLap (mean of fold test predictions)
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from data import DATA, ID_COL, TARGET, build_gbdt_diffFE
OOF_OUT = DATA / "oof_cb_pseudo_exp14.parquet"
SUB_OUT = DATA / "submission_cb_pseudo_exp14.csv"

N_SPLITS = 5
SEED = 42

# Public-notebook HPs, GPU swapped to CPU + tighter early-stopping for wall-clock
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

    # ---- Strong-blend pseudo-labels (round-2 self-training on CB) ----
    print("Reading strong-blend labeler (data/submission_blend_pseudo_r2.csv)...")
    lab = pd.read_csv(DATA / "submission_blend_pseudo_r2.csv").sort_values(ID_COL).reset_index(drop=True)
    test_s = test.sort_values(ID_COL).reset_index(drop=True)
    assert (lab[ID_COL].to_numpy() == test_s[ID_COL].to_numpy()).all(), "labeler id mismatch"
    id_to_p = dict(zip(lab[ID_COL].to_numpy(), lab[TARGET].to_numpy()))
    tp = np.array([id_to_p[i] for i in test[ID_COL].to_numpy()])
    PSEUDO_HI, PSEUDO_LO = 0.92, 0.03
    hi = tp >= PSEUDO_HI; lo = tp <= PSEUDO_LO
    pl = np.where(hi, 1, np.where(lo, 0, -1))
    keep = pl >= 0
    X_pseudo = X_test[keep].reset_index(drop=True)
    y_pseudo = pl[keep].astype(int)
    print(f"  pseudo-labeled {keep.sum():,}/{len(tp):,} test rows (hi={hi.sum():,} lo={lo.sum():,})")

    strat_key = train["Year"].astype(str) + "_" + train[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train), dtype=np.float64)
    test_preds = np.zeros(len(test), dtype=np.float64)
    fold_aucs: list[float] = []
    fold_iters: list[int] = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, strat_key), start=1):
        t0 = time.time()
        # Augment training with external data (val stays competition-only)
        X_tr = pd.concat([X.iloc[tr_idx], X_ext, X_pseudo], axis=0, ignore_index=True)
        y_tr = np.concatenate([y[tr_idx], y_ext, y_pseudo])
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
