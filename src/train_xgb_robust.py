"""Experiment 088 — noise-robust GCE objective on the diffFE-XGB base.

Our own EDA found 32% of train rows are label-noise (bidirectional stint-mismatch,
audit §2.1). exp 076 attacked this by *sample-downweighting* the noisy rows and lost
−0.00091 fold-1 — discarding partially-noisy rows throws away too much signal. This
experiment attacks the same noise mechanism via a *different loss surface* instead
(audit §2.4 lists "GBDT with a robust loss" as untested): Generalized Cross-Entropy
(GCE, L_q = (1 - p_t^q)/q). All rows are kept, but the per-row gradient is bounded —
for a y=1 row where the model predicts p≈0 (a confidently-"mislabeled" row), standard
logloss applies a near-maximal gradient (overfitting to the noise) whereas GCE's
gradient g = -p^q·(1-p) → 0. q interpolates CE (q→0) ↔ MAE (q=1); we use q=0.5.

Goal: a base that is (a) strong enough to earn blend weight (standalone OOF ≥ 0.951)
and (b) diverse — its error structure differs from logloss-trained diffFE-XGB because
it has learned a noise-robust decision surface.

Forked from train_xgb_diffFE.py (identical 49-feature diffFE recipe & HPs); only the
objective changes from binary:logistic to the custom GCE grad/hess below.

Outputs:
  data/oof_xgb_robust.parquet      OOF predictions (sigmoid of margin)
  data/submission_xgb_robust.csv   test predictions
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
OOF_OUT = DATA / "oof_xgb_robust.parquet"
SUB_OUT = DATA / "submission_xgb_robust.csv"
GCE_Q = 0.5  # robustness knob: q→0 ≈ logloss, q=1 ≈ MAE (max robustness)
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
    # objective set via custom `obj` (GCE) passed to xgb.train; do not set here.
    "eval_metric": "auc",
    "base_score": 0.0,           # custom obj works in margin space; 0.0 = neutral
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
EARLY_STOP = 400  # GCE ramps very slowly; 100 fired prematurely on folds 4-5 (exp 088 diag)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def gce_obj(q: float):
    """Generalized Cross-Entropy custom objective for xgb.train.

    L_q = (1 - p_t^q)/q,  p_t = p if y=1 else 1-p,  p = sigmoid(margin).
    Gradient (w.r.t. margin):
      y=1:  g = -p^q (1-p)
      y=0:  g = +(1-p)^q p
    A confidently-mislabeled row (p→0 when y=1) gets g→0 instead of logloss's
    g→-1, so the splitter stops chasing the 32% noisy labels. Hessian uses the
    standard logloss form p(1-p) (a stable positive approximation; the robust
    *gradient* is what reshapes the decision surface).
    """
    def _obj(preds: np.ndarray, dtrain: "xgb.DMatrix"):
        y = dtrain.get_label()
        p = _sigmoid(preds)
        pc = np.clip(p, 1e-7, 1 - 1e-7)
        grad = np.where(y == 1, -(pc ** q) * (1.0 - pc), ((1.0 - pc) ** q) * pc)
        hess = np.maximum(pc * (1.0 - pc), 1e-6)
        return grad, hess
    return _obj

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
        X_tr = pd.concat([X.iloc[tr_idx], X_ext], axis=0, ignore_index=True)
        # After concat, each cat column's dtype already has the unified category set
        # (because we set it on the source frames before this loop). No re-cast needed.
        y_tr = np.concatenate([y[tr_idx], y_ext])
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
            verbose_eval=500,
            obj=gce_obj(GCE_Q),
        )

        best_iter = booster.best_iteration
        # custom obj → predict() returns raw margins; map to probability via sigmoid
        va_pred = _sigmoid(booster.predict(dval, iteration_range=(0, best_iter + 1)))
        oof[va_idx] = va_pred
        test_preds += _sigmoid(booster.predict(dtest, iteration_range=(0, best_iter + 1))) / N_SPLITS
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
