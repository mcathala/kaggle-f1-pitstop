"""LightGBM DART base — decorrelation via tree-dropout boosting.

Our greedy ensemble + monotone experiments (2026-05-29) showed every GBDT variant
on the shared diffFE pipeline correlates ~0.997 with each other — same data, same
FE, same gbdt dynamics. DART (Dropout meets Multiple Additive Regression Trees)
randomly drops trees during boosting, so later trees correct a *different* residual
each round → a genuinely different fit and (hypothesis) a lower rank-correlation
with the standard-gbdt bases. If it lands AUC ≳ 0.950 with rank-corr < 0.97 vs the
gbdt pool, it earns blend weight; if it correlates ~0.99 like the rest, that closes
the "different boosting dynamics decorrelate" question. Either way it's a learning.

Reuses train_lgb_diffFE's validated FE (import, no duplication). Only the booster
config changes. DART early-stopping is unreliable (dropped trees perturb the metric),
so we train a fixed number of rounds and predict with all of them.

Outputs:
  data/oof_lgb_dart.parquet
  data/submission_lgb_dart.csv
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

import train_lgb_diffFE as base  # reuse validated FE pipeline

DATA = base.DATA
TARGET, ID_COL, BASE_CATS = base.TARGET, base.ID_COL, base.BASE_CATS
N_SPLITS, SEED = base.N_SPLITS, base.SEED
OOF_OUT = DATA / "oof_lgb_dart.parquet"
SUB_OUT = DATA / "submission_lgb_dart.csv"

DART_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "dart",
    "drop_rate": 0.10,
    "skip_drop": 0.50,
    "max_drop": 50,
    "uniform_drop": False,
    "xgboost_dart_mode": False,
    "max_bin": 2048,          # moderate (DART is slow; 2048 still resolves cross-cats)
    "num_leaves": 127,
    "learning_rate": 0.03,
    "feature_fraction": 0.15,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "lambda_l1": 8.0,
    "lambda_l2": 8.0,
    "min_data_in_leaf": 200,
    "cat_smooth": 10,
    "verbosity": -1,
    "n_jobs": 4,              # capped for memory
    "random_state": SEED,
}
N_ROUNDS = 1500               # fixed: DART predicts with all trees


def build_frames():
    train = pd.read_csv(base.TRAIN_CSV)
    test = pd.read_csv(base.TEST_CSV)
    ext = pd.read_csv(base.EXTERNAL_CSV).dropna(subset=["Compound"]).drop(columns=["Normalized_TyreLife"], errors="ignore")
    ext[ID_COL] = -1
    train = base.add_domain_features(train)
    test = base.add_domain_features(test)
    ext = base.add_domain_features(ext)
    bins = ["RacePhase", "LapBin", "TyreLifeBin", "PositionBin"]
    all_cats = BASE_CATS + bins
    base.add_frequency_features([train, test, ext], BASE_CATS)
    base.normalize_cats(train, all_cats)
    base.normalize_cats(test, all_cats)
    base.normalize_cats(ext, all_cats)
    for c in all_cats:
        if c not in train.columns:
            continue
        union_vals = (pd.concat([train[c], test[c], ext[c]], axis=0)
                      .astype("string").fillna("__NA__").unique().tolist())
        cat_dtype = pd.CategoricalDtype(categories=sorted(union_vals))
        for f in (train, test, ext):
            if c in f.columns:
                f[c] = f[c].astype(cat_dtype)
    feature_cols = [c for c in train.columns if c not in (TARGET, ID_COL)]
    feature_cols = [c for c in feature_cols if c in test.columns and c in ext.columns]
    return train, test, ext, feature_cols, all_cats


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=-1)
    args = ap.parse_args()
    dry = args.fold != -1
    print(f"DART  fold={args.fold}  dry_run={dry}  rounds={N_ROUNDS}")

    train, test, ext, feature_cols, all_cats = build_frames()
    n_cat = sum(1 for c in feature_cols if c in all_cats)
    print(f"using {len(feature_cols)} features, {n_cat} categorical")

    X = train[feature_cols]; y = train[TARGET].astype(int).to_numpy()
    X_test = test[feature_cols]
    X_ext = ext[feature_cols]; y_ext = ext[TARGET].astype(int).to_numpy()
    cat_idx = [i for i, c in enumerate(feature_cols) if c in all_cats]

    strat = train["Year"].astype(str) + "_" + train[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train)); test_preds = np.zeros(len(test)); aucs = []
    folds = [args.fold] if dry else list(range(1, N_SPLITS + 1))

    for fold, (tr, va) in enumerate(kf.split(X, strat), 1):
        if fold not in folds:
            continue
        t0 = time.time()
        X_tr = pd.concat([X.iloc[tr], X_ext], ignore_index=True)
        y_tr = np.concatenate([y[tr], y_ext])
        dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_idx, free_raw_data=False)
        params = dict(DART_PARAMS); params["random_state"] = SEED + fold
        booster = lgb.train(params, dtrain, num_boost_round=N_ROUNDS,
                            callbacks=[lgb.log_evaluation(period=300)])
        va_pred = booster.predict(X.iloc[va])          # all trees
        oof[va] = va_pred
        test_preds += booster.predict(X_test) / N_SPLITS
        a = roc_auc_score(y[va], va_pred); aucs.append(a)
        print(f"fold {fold}/{N_SPLITS}  AUC={a:.5f}  ({time.time()-t0:.1f}s)", flush=True)

    if dry:
        print(f"\nDRY-RUN fold-{args.fold} AUC = {aucs[0]:.5f}")
        rm = pd.read_parquet(DATA / "oof_xgb_diffFE.parquet")
        rm_map = dict(zip(rm["id"], rm["oof"]))
        va_idx = list(kf.split(X, strat))[args.fold - 1][1]
        rv = train.iloc[va_idx][ID_COL].map(rm_map).to_numpy()
        if not np.isnan(rv).any():
            print(f"rank-corr vs diffFE-XGB (fold {args.fold}): {spearmanr(oof[va_idx], rv).statistic:.5f}")
        return

    oof_auc = roc_auc_score(y, oof)
    print(f"\nper-fold AUC mean={np.mean(aucs):.5f} std={np.std(aucs):.5f}")
    print(f"OOF AUC: {oof_auc:.5f}")
    oof_df = pd.DataFrame({"id": train[ID_COL], "Year": train["Year"], "target": y, "oof": oof})
    print("\nRank-correlation diagnostics:")
    for name, p in [("RealMLP-multiseed", DATA / "oof_realmlp_multiseed.parquet"),
                    ("diffFE-XGB", DATA / "oof_xgb_diffFE.parquet"),
                    ("lgb_diffFE", DATA / "oof_lgb_diffFE.parquet")]:
        try:
            o = pd.read_parquet(p)
            m = pd.DataFrame({"id": train[ID_COL], "oof": oof}).merge(
                o[["id", "oof"]].rename(columns={"oof": "o"}), on="id")
            print(f"  vs {name:20s}: {spearmanr(m['oof'], m['o']).statistic:.5f}")
        except Exception as e:
            print(f"  vs {name}: skip ({e})")
    oof_df.assign(fold=-1).to_parquet(OOF_OUT, index=False)
    pd.DataFrame({"id": test[ID_COL], TARGET: test_preds}).sort_values("id").reset_index(drop=True).to_csv(SUB_OUT, index=False)
    print(f"\nwrote {OOF_OUT.name}, {SUB_OUT.name}")


if __name__ == "__main__":
    main()
