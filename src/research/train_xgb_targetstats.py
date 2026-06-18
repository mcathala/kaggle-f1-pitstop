"""Experiment 055 (cycle 16) — rich out-of-fold target-statistic encoding on XGB-highbins.

Cycle 16 established that *physics* feature work (lag, tyre-overdue) can't move
XGB — its ranking is locked (ρ 0.997-0.999) because those features carry weak
signal the model already extracts. Target statistics are a different animal: the
mean/std/count of the TARGET within a group is near-soft-label signal, much
stronger than any derived physics feature, so it can raise standalone AUC rather
than just shuffle the ranking. And our XGB-highbins currently uses NO target
encoding at all (only frequency counts + feature group-stats) — so this is a
genuine gap, not a tweak.

We add, for ~10 group keys (categoricals + binned numerics), the out-of-fold
target mean (count-smoothed), std, and count. Leakage control:
  - val / test rows: stats from the full outer-train fold (comp_train + external).
  - train rows: inner 5-fold cross-fit (each inner-val encoded from inner-train
    only), so no row sees its own target. This is the standard leakage-free
    target-encoding scheme, extended to higher moments and many keys.

Base recipe otherwise identical to cycle-11 XGB-highbins (max_bin=5000, eta=0.01,
depth=10, heavy L1/L2, colsample=0.145).

Outputs:
  data/oof_xgb_targetstats.parquet
  data/submission_xgb_targetstats.csv
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

# reuse the cycle-11 XGB-highbins FE verbatim (DRY; same engineered base)
from train_xgb_richcat import (
    add_domain_features, add_cross_categoricals, add_frequency_features,
    add_group_stats, normalize_cats, XGB_PARAMS, N_ROUNDS, EARLY_STOP,
    BASE_CATS, TARGET, ID_COL, N_SPLITS, SEED,
)

DATA = Path(__file__).resolve().parent.parent.parent / "data"
TRAIN_CSV = DATA / "train.csv"
TEST_CSV = DATA / "test.csv"
EXTERNAL_CSV = DATA / "f1_strategy_dataset.csv"
OOF_OUT = DATA / "oof_xgb_targetstats.parquet"
SUB_OUT = DATA / "submission_xgb_targetstats.csv"
REALMLP_OOF = DATA / "oof_realmlp_multiseed.parquet"
CB_OOF = DATA / "oof_cb_tuned_exp14.parquet"
XGB_OOF = DATA / "oof_xgb_highbins.parquet"

# group keys for target-statistic encoding (all already exist as columns post-FE)
TARGET_STAT_KEYS = [
    "Driver", "Race_Year", "Driver_Race", "Driver_Compound", "Race_Compound",
    "Race_Compound_Stint", "Compound_TyreLifeBin", "RacePhase_TyreLifeBin",
    "Compound_Stint", "TyreLifeBin", "PositionBin",
]
SMOOTHING = 20.0
INNER_SPLITS = 5


def _group_stats(keys_series: pd.Series, y: np.ndarray, global_mean: float):
    """Return (smoothed_mean, std, count) Series indexed by group value."""
    g = pd.DataFrame({"k": keys_series.to_numpy(), "y": y})
    agg = g.groupby("k")["y"].agg(["mean", "std", "count"])
    sm = (agg["mean"] * agg["count"] + global_mean * SMOOTHING) / (agg["count"] + SMOOTHING)
    return sm, agg["std"].fillna(0.0), agg["count"]


def add_target_stats(X_tr, y_tr, X_va, X_te, keys):
    """Cross-fitted target mean/std/count. Adds columns in-place to copies; returns
    (X_tr2, X_va2, X_te2) with the new columns. No row sees its own target."""
    X_tr, X_va, X_te = X_tr.copy(), X_va.copy(), X_te.copy()
    gm = float(y_tr.mean())

    # val / test: full-train stats
    for key in keys:
        sm, sd, cnt = _group_stats(X_tr[key].astype(str), y_tr, gm)
        for frame in (X_va, X_te):
            ks = frame[key].astype(str)
            frame[f"_ts_{key}_mean"] = ks.map(sm).fillna(gm).astype("float32")
            frame[f"_ts_{key}_std"] = ks.map(sd).fillna(0.0).astype("float32")
            frame[f"_ts_{key}_cnt"] = ks.map(cnt).fillna(0).astype("float32")

    # train: inner cross-fit
    inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED)
    tr_cols = {f"_ts_{key}_{stat}": np.full(len(X_tr), np.nan, dtype="float32")
               for key in keys for stat in ("mean", "std", "cnt")}
    strat_inner = (np.asarray(y_tr)).astype(int)
    keys_str = {key: X_tr[key].astype(str).reset_index(drop=True) for key in keys}
    y_arr = np.asarray(y_tr)
    for in_tr, in_va in inner.split(np.zeros(len(X_tr)), strat_inner):
        gm_i = float(y_arr[in_tr].mean())
        for key in keys:
            sm, sd, cnt = _group_stats(keys_str[key].iloc[in_tr], y_arr[in_tr], gm_i)
            kv = keys_str[key].iloc[in_va]
            tr_cols[f"_ts_{key}_mean"][in_va] = kv.map(sm).fillna(gm_i).to_numpy(dtype="float32")
            tr_cols[f"_ts_{key}_std"][in_va] = kv.map(sd).fillna(0.0).to_numpy(dtype="float32")
            tr_cols[f"_ts_{key}_cnt"][in_va] = kv.map(cnt).fillna(0).to_numpy(dtype="float32")
    for c, v in tr_cols.items():
        X_tr[c] = np.nan_to_num(v, nan=gm if c.endswith("_mean") else 0.0)
    return X_tr, X_va, X_te


def main():
    print("Loading data...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    ext = pd.read_csv(EXTERNAL_CSV).dropna(subset=["Compound"]).drop(columns=["Normalized_TyreLife"], errors="ignore")
    ext[ID_COL] = -1
    print(f"  train {train.shape}  test {test.shape}  ext {ext.shape}")

    train = add_domain_features(train); test = add_domain_features(test); ext = add_domain_features(ext)
    train = add_cross_categoricals(train); test = add_cross_categoricals(test); ext = add_cross_categoricals(ext)
    cross_cats = ["Race_Year", "Compound_Stint", "Driver_Race", "Driver_Compound", "Race_Compound",
                  "Race_Compound_Stint", "Compound_RacePhase", "Compound_TyreLifeBin", "RacePhase_TyreLifeBin"]
    bins = ["RacePhase", "LapBin", "TyreLifeBin", "PositionBin"]
    all_cats = BASE_CATS + cross_cats + bins
    add_frequency_features([train, test, ext], all_cats)
    gs = add_group_stats([train, test, ext]); print(f"  added {len(gs)} group-stat cols")
    normalize_cats(train, all_cats); normalize_cats(test, all_cats); normalize_cats(ext, all_cats)

    # unified categorical dtype (XGB enable_categorical needs identical category sets)
    for c in all_cats:
        if c not in train.columns:
            continue
        uv = pd.concat([train[c], test[c], ext[c]], axis=0).astype("string").fillna("__NA__").unique().tolist()
        dt = pd.CategoricalDtype(categories=sorted(uv))
        for f in (train, test, ext):
            if c in f.columns:
                f[c] = f[c].astype(dt)

    ts_cols = [f"_ts_{k}_{s}" for k in TARGET_STAT_KEYS for s in ("mean", "std", "cnt")]
    base_feature_cols = [c for c in train.columns if c not in (TARGET, ID_COL)]
    base_feature_cols = [c for c in base_feature_cols if c in test.columns and c in ext.columns]
    feature_cols = base_feature_cols + ts_cols
    n_cat = sum(1 for c in feature_cols if c in all_cats)
    print(f"using {len(feature_cols)} features ({len(ts_cols)} target-stat, {n_cat} categorical)")

    y = train[TARGET].astype(int).to_numpy()
    y_ext = ext[TARGET].astype(int).to_numpy()
    X = train[base_feature_cols]; X_test = test[base_feature_cols]; X_ext = ext[base_feature_cols]

    strat = train["Year"].astype(str) + "_" + train[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train)); test_preds = np.zeros(len(test)); fold_aucs = []; fold_iters = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, strat), start=1):
        t0 = time.time()
        X_tr = pd.concat([X.iloc[tr_idx], X_ext], axis=0, ignore_index=True)
        y_tr = np.concatenate([y[tr_idx], y_ext])
        X_va = X.iloc[va_idx].reset_index(drop=True); y_va = y[va_idx]

        # cross-fitted target-statistic features (leakage-free)
        X_tr, X_va, X_te = add_target_stats(X_tr, y_tr, X_va, X_test.reset_index(drop=True), TARGET_STAT_KEYS)

        dtr = xgb.DMatrix(X_tr[feature_cols], label=y_tr, enable_categorical=True)
        dva = xgb.DMatrix(X_va[feature_cols], label=y_va, enable_categorical=True)
        dte = xgb.DMatrix(X_te[feature_cols], enable_categorical=True)
        params = dict(XGB_PARAMS); params["seed"] = SEED + fold
        bst = xgb.train(params, dtr, num_boost_round=N_ROUNDS, evals=[(dva, "val")],
                        early_stopping_rounds=EARLY_STOP, verbose_eval=500)
        bi = bst.best_iteration
        va_pred = bst.predict(dva, iteration_range=(0, bi + 1))
        oof[va_idx] = va_pred
        test_preds += bst.predict(dte, iteration_range=(0, bi + 1)) / N_SPLITS
        a = roc_auc_score(y_va, va_pred); fold_aucs.append(a); fold_iters.append(bi + 1)
        print(f"fold {fold}/{N_SPLITS}  AUC={a:.5f}  iters={bi+1}  ({time.time()-t0:.1f}s)", flush=True)

    oof_auc = roc_auc_score(y, oof)
    print(f"\nper-fold mean={np.mean(fold_aucs):.5f} std={np.std(fold_aucs):.5f} iters={fold_iters}")
    print(f"OOF AUC: {oof_auc:.5f}  (vs cycle-11 XGB-highbins 0.95263, Δ={oof_auc-0.95263:+.5f}; vs floor 0.949, Δ={oof_auc-0.949:+.5f})")

    print("\nRank-correlation vs existing bases:")
    for name, path in [("RealMLP-ms", REALMLP_OOF), ("CB-tuned14", CB_OOF), ("XGB-highbins", XGB_OOF)]:
        try:
            other = pd.read_parquet(path)
            m = pd.DataFrame({"id": train[ID_COL], "oof": oof}).merge(
                other[["id", "oof"]].rename(columns={"oof": "o"}), on="id")
            rho, _ = spearmanr(m["oof"], m["o"]); print(f"  vs {name:14s}: {rho:.5f}")
        except Exception as e:
            print(f"  vs {name}: skipped ({e})")

    pd.DataFrame({"id": train[ID_COL], "Year": train["Year"], "target": y, "oof": oof, "fold": -1}).to_parquet(OOF_OUT, index=False)
    pd.DataFrame({"id": test[ID_COL], TARGET: test_preds}).sort_values("id").reset_index(drop=True).to_csv(SUB_OUT, index=False)
    print(f"\nwrote {OOF_OUT.name} and {SUB_OUT.name}")


if __name__ == "__main__":
    main()
