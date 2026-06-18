"""Experiment 044 (cycle 11) — XGBoost retune for high-cardinality cats.

Exp 034's XGBoost capped at OOF 0.94615 (vs CB14 0.95114) with default HPs.
Cycle-10 probe 1 (cb_feature_importance) ranked Race_Year and Race_Compound_Stint
as top features by mean importance (8.83 and 4.93). These are high-cardinality
cross-cats with ~hundreds of unique values. XGBoost's default `max_bin=256`
histogram can't resolve them — the splitter sees aliased buckets, losing signal.

This retune addresses that directly: `max_bin=5000` gives the histogram enough
resolution for our cross-cats, slower `eta=0.01` + tighter regularization
(lambda=8, alpha=8, colsample=0.15) absorb the added capacity without overfitting.

Outputs:
  data/oof_xgb_highbins.parquet      OOF predictions
  data/submission_xgb_highbins.csv   test predictions
"""

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

from data import DATA, ID_COL, TARGET, build_gbdt_diffFE
OOF_OUT = DATA / "oof_xgb_diffFE_pseudo.parquet"
SUB_OUT = DATA / "submission_xgb_diffFE_pseudo.csv"
REALMLP_OOF = DATA / "oof_realmlp_multiseed.parquet"
CB_OOF = DATA / "oof_cb_tuned_exp14.parquet"

N_SPLITS = 5
SEED = 42

# XGBoost HPs retuned to fit our high-cardinality cross-cats (probe-1 finding).
# Key differences vs exp 034's default-ish recipe:
#   max_bin: 256 → 5000           (histogram resolution for Race_Year etc.)
#   lr: 0.05 → 0.01               (slower learning matches the deeper depth)
#   n_estimators: 3000 → 50000    (capacity for slow lr; early-stop=100 will fire)
#   max_depth: 8 → 10             (deeper trees for cat-interaction discovery)
#   lambda: 5.0 → 8                (stronger L2 to absorb added capacity)
#   alpha: 0.5 → 8                 (sparse-feature-friendly L1)
#   colsample_bytree: 0.7 → 0.145 (heavy column subsample regularizes high-bin trees)
#   subsample: 0.8 → 0.86
#   min_child_weight: 5.0 → 2     (allow finer splits at high max_bin)
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "enable_categorical": True,
    "max_bin": 5000,             # ← THE KEY FIX
    "max_depth": 10,
    "eta": 0.01,
    "min_child_weight": 2,
    "subsample": 0.8570122278990485,
    "colsample_bytree": 0.1450999139156032,
    "reg_lambda": 8.162374349037115,
    "reg_alpha": 8.354463958574286,
    "nthread": -1,
    "verbosity": 1,
}
N_ROUNDS = 50000
EARLY_STOP = 100

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=-1, help="1..N_SPLITS for one fold; -1 for all")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dry_run = args.fold != -1
    print(f"Args: fold={args.fold}  dry_run={dry_run}")

    print("Loading data + building diffFE features (src/data.py)...")
    train, test, ext, feature_cols, _ = build_gbdt_diffFE()
    print(f"  train {train.shape}  test {test.shape}  external {ext.shape}")
    print(f"  external pos rate: {ext[TARGET].mean():.4f}  competition train pos rate: {train[TARGET].mean():.4f}")
    print(f"using {len(feature_cols)} features")

    X = train[feature_cols]
    y = train[TARGET].astype(int).to_numpy()
    X_test = test[feature_cols]
    X_ext = ext[feature_cols]
    y_ext = ext[TARGET].astype(int).to_numpy()

    # ---- Strong-blend pseudo-labels (self-training on the cleaner diffFE base) ----
    print("Reading strong-blend labeler (data/submission_blend_pseudo_r2.csv)...")
    _lab = pd.read_csv(DATA / "submission_blend_pseudo_r2.csv").sort_values(ID_COL).reset_index(drop=True)
    _id_to_p = dict(zip(_lab[ID_COL].to_numpy(), _lab[TARGET].to_numpy()))
    _tp = np.array([_id_to_p[i] for i in test[ID_COL].to_numpy()])
    _hi, _lo = 0.92, 0.03
    _pl = np.where(_tp >= _hi, 1, np.where(_tp <= _lo, 0, -1))
    _keep = _pl >= 0
    X_pseudo = X_test[_keep].reset_index(drop=True)
    y_pseudo = _pl[_keep].astype(int)
    print(f"  pseudo-labeled {int(_keep.sum()):,}/{len(_tp):,} test rows (hi={int((_pl==1).sum()):,} lo={int((_pl==0).sum()):,})")

    strat_key = train["Year"].astype(str) + "_" + train[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train), dtype=np.float64)
    test_preds = np.zeros(len(test), dtype=np.float64)
    fold_aucs: list[float] = []
    fold_iters: list[int] = []

    folds_to_run = [args.fold] if dry_run else list(range(1, N_SPLITS + 1))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, strat_key), start=1):
        if fold not in folds_to_run:
            continue
        t0 = time.time()
        X_tr = pd.concat([X.iloc[tr_idx], X_ext, X_pseudo], axis=0, ignore_index=True)
        # After concat, each cat column's dtype already has the unified category set.
        y_tr = np.concatenate([y[tr_idx], y_ext, y_pseudo])
        X_va = X.iloc[va_idx]
        y_va = y[va_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
        dval = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, enable_categorical=True)

        params = dict(XGB_PARAMS)
        params["seed"] = SEED + fold

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=N_ROUNDS,
            evals=[(dval, "val")],
            early_stopping_rounds=EARLY_STOP,
            verbose_eval=200,
        )

        best_iter = booster.best_iteration
        va_pred = booster.predict(dval, iteration_range=(0, best_iter + 1))
        oof[va_idx] = va_pred
        test_preds += booster.predict(dtest, iteration_range=(0, best_iter + 1)) / N_SPLITS
        fold_auc = roc_auc_score(y_va, va_pred)
        fold_aucs.append(fold_auc)
        fold_iters.append(best_iter + 1)
        print(
            f"fold {fold}/{N_SPLITS}  AUC={fold_auc:.5f}  iters={best_iter + 1}  "
            f"train_rows={len(X_tr):,}  ({time.time()-t0:.1f}s)",
            flush=True,
        )

    if dry_run:
        print(f"\nDRY-RUN fold-{args.fold} AUC = {fold_aucs[0]:.5f}")
        if REALMLP_OOF.exists():
            try:
                rm = pd.read_parquet(REALMLP_OOF)
                rm_map = dict(zip(rm["id"], rm["oof"]))
                va_idx_actual = list(kf.split(X, strat_key))[args.fold - 1][1]
                rm_va = train.iloc[va_idx_actual][ID_COL].map(rm_map).to_numpy()
                if not np.isnan(rm_va).any():
                    rho, _ = spearmanr(oof[va_idx_actual], rm_va)
                    print(f"rank-corr-with-RealMLP-multiseed (fold {args.fold}): {rho:.5f}")
            except Exception as e:
                print(f"rank-corr-with-RealMLP skipped: {e}")
        return

    oof_auc = roc_auc_score(y, oof)
    print()
    print(f"per-fold AUC: mean={np.mean(fold_aucs):.5f}  std={np.std(fold_aucs):.5f}  iters={fold_iters}")
    print(f"OOF AUC:      {oof_auc:.5f}")
    print(f"  (vs CB-tuned-exp14 0.95114, Δ = {oof_auc - 0.95114:+.5f})")

    oof_df = pd.DataFrame(
        {"id": train[ID_COL], "Year": train["Year"], "target": y, "oof": oof}
    )
    print("\nOOF AUC by Year:")
    for year, g in oof_df.groupby("Year"):
        if g["target"].nunique() > 1:
            auc = roc_auc_score(g["target"], g["oof"])
            print(f"  {year}: AUC={auc:.5f}  n={len(g):,}  pos_rate={g['target'].mean():.4f}")

    # Rank-correlation with existing base OOFs — the diversity-for-blend signal.
    print("\nRank-correlation diagnostics:")
    for name, path in [("RealMLP-multiseed", REALMLP_OOF), ("CB-tuned-exp14", CB_OOF)]:
        try:
            other = pd.read_parquet(path)
            m = pd.DataFrame({"id": train[ID_COL], "oof": oof}).merge(
                other[["id", "oof"]].rename(columns={"oof": "other"}), on="id", how="inner"
            )
            rho, _ = spearmanr(m["oof"], m["other"])
            print(f"  rank-corr vs {name:20s}: {rho:.5f}  (n={len(m):,})")
        except Exception as e:
            print(f"  rank-corr vs {name}: skipped ({e})")

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
