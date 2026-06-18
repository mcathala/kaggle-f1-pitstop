"""Monotone-constrained XGB — a transfer-robust base.

Motivation (our own analysis, 2026-05-29): our submission history shows the
OOF->LB transfer is a shrink with slope ~0.927 and the worst-transferring model
was the self-distilled RealMLP (most overfit). Less-overfit models transfer
relatively better. Domain-monotone constraints (older tyre / more degradation =>
higher pit probability) are a principled regulariser that (a) cannot fit noise in
the physically-wrong direction and (b) make the error structure different from the
unconstrained XGB bases, a decorrelation source for the blend.

Forked from train_xgb_diffFE.py (same validated FE pipeline) — the ONLY change is
adding monotone_constraints on a conservative set of strongly-justified numeric
features. Everything else (folds, external data, HPs) is identical so the
comparison is controlled.

Outputs:
  data/oof_xgb_monotone.parquet
  data/submission_xgb_monotone.csv
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

DATA = Path(__file__).resolve().parent.parent.parent / "data"
TRAIN_CSV = DATA / "train.csv"
TEST_CSV = DATA / "test.csv"
EXTERNAL_CSV = DATA / "f1_strategy_dataset.csv"
OOF_OUT = DATA / "oof_xgb_monotone.parquet"
SUB_OUT = DATA / "submission_xgb_monotone.csv"
REALMLP_OOF = DATA / "oof_realmlp_multiseed.parquet"
CB_OOF = DATA / "oof_cb_tuned_exp14.parquet"

TARGET = "PitNextLap"
ID_COL = "id"
BASE_CATS = ["Driver", "Race", "Compound"]
N_SPLITS = 5
SEED = 42

# Domain-monotone numeric features: relationship with pit probability has a
# physically-determined sign. Conservative set (only strongly-justified ones) so
# the constraint doesn't crush OOF.  +1 = pit prob non-decreasing in the feature.
MONOTONE = {
    "TyreLife": 1,                    # older tyre -> more likely to pit
    "TyreAgeRatio": 1,                # tyre age relative to lap
    "TyreAgeVsRace": 1,
    "TyreLife_to_LapsRemaining": 1,
    "PitWindowPressure": 1,           # TyreLife * RaceProgress
    "StintPressure": 1,               # Stint * TyreLife
    "Cumulative_Degradation": 1,      # more degradation -> pit
    "DegPerTyreLap": 1,
    "Abs_Cumulative_Degradation": 1,
}

# Same HPs as train_xgb_diffFE.py (controlled comparison).
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "enable_categorical": True,
    "max_bin": 5000,
    "max_depth": 10,
    "eta": 0.01,
    "min_child_weight": 2,
    "subsample": 0.8570122278990485,
    "colsample_bytree": 0.1450999139156032,
    "reg_lambda": 8.162374349037115,
    "reg_alpha": 8.354463958574286,
    "nthread": 4,                     # capped: M1 memory is tight, avoid OOM
    "verbosity": 1,
}
N_ROUNDS = 50000
EARLY_STOP = 100


def safe_div(a, b, eps=1e-6):
    return a / (b + eps)


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6
    out = df.copy()
    race_progress = out["RaceProgress"].clip(lower=eps)
    out["EstimatedTotalLaps"] = (out["LapNumber"] / race_progress).clip(1, 120).astype("float32")
    out["LapsRemaining"] = (out["EstimatedTotalLaps"] - out["LapNumber"]).clip(lower=0).astype("float32")
    out["RemainingRaceProgress"] = (1.0 - out["RaceProgress"]).astype("float32")
    out["LapProgress_x_LapNumber"] = (out["LapNumber"] * out["RaceProgress"]).astype("float32")
    out["RacePhase"] = pd.cut(out["RaceProgress"], bins=[-np.inf, 0.20, 0.40, 0.60, 0.80, np.inf],
                              labels=["P1", "P2", "P3", "P4", "P5"]).astype(str)
    out["LapBin"] = pd.cut(out["LapNumber"], bins=[-np.inf, 5, 10, 20, 35, 50, np.inf],
                           labels=["L005", "L010", "L020", "L035", "L050", "Lplus"]).astype(str)
    out["TyreAgeRatio"] = safe_div(out["TyreLife"], out["LapNumber"].clip(lower=1), eps).astype("float32")
    out["LapPerTyreLife"] = safe_div(out["LapNumber"], out["TyreLife"] + 1, eps).astype("float32")
    out["TyreLife_x_RaceProgress"] = (out["TyreLife"] * out["RaceProgress"]).astype("float32")
    out["PitWindowPressure"] = (out["TyreLife"] * out["RaceProgress"]).astype("float32")
    out["TyreAgeVsRace"] = safe_div(out["TyreLife"], out["EstimatedTotalLaps"].clip(lower=1), eps).astype("float32")
    out["TyreLife_to_LapsRemaining"] = safe_div(out["TyreLife"], out["LapsRemaining"] + 1, eps).astype("float32")
    out["LapMinusTyreLife"] = (out["LapNumber"] - out["TyreLife"]).astype("float32")
    out["TyreLifeBin"] = pd.cut(out["TyreLife"], bins=[-np.inf, 3, 7, 12, 20, 30, np.inf],
                                labels=["T003", "T007", "T012", "T020", "T030", "Tplus"]).astype(str)
    out["StintPressure"] = (out["Stint"] * out["TyreLife"]).astype("float32")
    out["Is_First_Stint"] = (out["Stint"] == 1).astype(np.int8)
    out["Is_Late_Stint"] = (out["Stint"] >= 3).astype(np.int8)
    out["PositionBin"] = pd.cut(out["Position"], bins=[-np.inf, 3, 8, 14, np.inf],
                                labels=["front", "upper_mid", "lower_mid", "back"]).astype(str)
    out["PositionPressure"] = (out["Position"] * out["RaceProgress"]).astype("float32")
    out["DegPerRaceLap"] = safe_div(out["Cumulative_Degradation"], out["LapNumber"].clip(lower=1), eps).astype("float32")
    out["DegPerTyreLap"] = safe_div(out["Cumulative_Degradation"], out["TyreLife"].clip(lower=1), eps).astype("float32")
    out["Abs_Cumulative_Degradation"] = out["Cumulative_Degradation"].abs().astype("float32")
    out["Positive_Degradation"] = (out["Cumulative_Degradation"] > 0).astype(np.int8)
    out["DeltaAbs"] = out["LapTime_Delta"].abs().astype("float32")
    out["LapTimeDeltaPositive"] = (out["LapTime_Delta"] > 0).astype(np.int8)
    out["DeltaPerTyreLap"] = safe_div(out["LapTime_Delta"], out["TyreLife"].clip(lower=1), eps).astype("float32")
    out["Abs_Position_Change"] = out["Position_Change"].abs().astype("float32")
    out["Gained_Position"] = (out["Position_Change"] > 0).astype(np.int8)
    out["Lost_Position"] = (out["Position_Change"] < 0).astype(np.int8)
    return out


def add_frequency_features(frames, cat_cols):
    if not cat_cols:
        return
    total = sum(len(f) for f in frames)
    for col in cat_cols:
        if not all(col in f.columns for f in frames):
            continue
        union = pd.concat([f[col].astype("string").fillna("__NA__") for f in frames], axis=0)
        counts = union.value_counts(dropna=False)
        for f in frames:
            keys = f[col].astype("string").fillna("__NA__")
            f[f"{col}_count"] = keys.map(counts).fillna(0).astype(np.int32)
            f[f"{col}_freq"] = (f[f"{col}_count"] / total).astype("float32")


def normalize_cats(out, cat_cols):
    for col in cat_cols:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("__NA__").astype(str)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=-1, help="1..N_SPLITS for one fold; -1 for all")
    return p.parse_args()


def main():
    args = parse_args()
    dry_run = args.fold != -1
    print(f"Args: fold={args.fold}  dry_run={dry_run}")

    print("Loading data...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    ext = pd.read_csv(EXTERNAL_CSV).dropna(subset=["Compound"]).drop(columns=["Normalized_TyreLife"], errors="ignore")
    ext[ID_COL] = -1
    print(f"  train {train.shape}  test {test.shape}  external {ext.shape}")

    print("Applying domain FE...")
    train = add_domain_features(train)
    test = add_domain_features(test)
    ext = add_domain_features(ext)

    # diffFE recipe: base cats only, no cross-cats / group-stats (controlled vs diffFE).
    bins = ["RacePhase", "LapBin", "TyreLifeBin", "PositionBin"]
    all_cats = BASE_CATS + bins
    add_frequency_features([train, test, ext], BASE_CATS)

    normalize_cats(train, all_cats)
    normalize_cats(test, all_cats)
    normalize_cats(ext, all_cats)
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
    n_cat = sum(1 for c in feature_cols if c in all_cats)

    # Build monotone-constraint tuple aligned to feature_cols (0 for unconstrained
    # & all categoricals). Verify the constrained features are numeric.
    mono = []
    applied = []
    for c in feature_cols:
        if c in MONOTONE and c not in all_cats:
            mono.append(MONOTONE[c]); applied.append(c)
        else:
            mono.append(0)
    print(f"using {len(feature_cols)} features, {n_cat} categorical")
    print(f"monotone constraints applied to {len(applied)}: {applied}")

    X = train[feature_cols]
    y = train[TARGET].astype(int).to_numpy()
    X_test = test[feature_cols]
    X_ext = ext[feature_cols]
    y_ext = ext[TARGET].astype(int).to_numpy()

    strat_key = train["Year"].astype(str) + "_" + train[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train), dtype=np.float64)
    test_preds = np.zeros(len(test), dtype=np.float64)
    fold_aucs, fold_iters = [], []
    folds_to_run = [args.fold] if dry_run else list(range(1, N_SPLITS + 1))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, strat_key), start=1):
        if fold not in folds_to_run:
            continue
        t0 = time.time()
        X_tr = pd.concat([X.iloc[tr_idx], X_ext], axis=0, ignore_index=True)
        y_tr = np.concatenate([y[tr_idx], y_ext])
        X_va = X.iloc[va_idx]
        y_va = y[va_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True, feature_names=feature_cols)
        dval = xgb.DMatrix(X_va, label=y_va, enable_categorical=True, feature_names=feature_cols)
        dtest = xgb.DMatrix(X_test, enable_categorical=True, feature_names=feature_cols)

        params = dict(XGB_PARAMS)
        params["seed"] = SEED + fold
        params["monotone_constraints"] = "(" + ",".join(str(m) for m in mono) + ")"

        booster = xgb.train(params, dtrain, num_boost_round=N_ROUNDS,
                            evals=[(dval, "val")], early_stopping_rounds=EARLY_STOP,
                            verbose_eval=200)
        best_iter = booster.best_iteration
        va_pred = booster.predict(dval, iteration_range=(0, best_iter + 1))
        oof[va_idx] = va_pred
        test_preds += booster.predict(dtest, iteration_range=(0, best_iter + 1)) / N_SPLITS
        fold_auc = roc_auc_score(y_va, va_pred)
        fold_aucs.append(fold_auc); fold_iters.append(best_iter + 1)
        print(f"fold {fold}/{N_SPLITS}  AUC={fold_auc:.5f}  iters={best_iter + 1}  "
              f"train_rows={len(X_tr):,}  ({time.time()-t0:.1f}s)", flush=True)

    if dry_run:
        print(f"\nDRY-RUN fold-{args.fold} AUC = {fold_aucs[0]:.5f}")
        return

    oof_auc = roc_auc_score(y, oof)
    print(f"\nper-fold AUC: mean={np.mean(fold_aucs):.5f}  std={np.std(fold_aucs):.5f}  iters={fold_iters}")
    print(f"OOF AUC:      {oof_auc:.5f}  (vs unconstrained diffFE-XGB 0.95299)")

    oof_df = pd.DataFrame({"id": train[ID_COL], "Year": train["Year"], "target": y, "oof": oof})
    print("\nOOF AUC by Year:")
    for year, g in oof_df.groupby("Year"):
        if g["target"].nunique() > 1:
            print(f"  {year}: AUC={roc_auc_score(g['target'], g['oof']):.5f}  n={len(g):,}")

    print("\nRank-correlation diagnostics:")
    for name, path in [("RealMLP-multiseed", REALMLP_OOF), ("CB-tuned-exp14", CB_OOF),
                       ("diffFE-XGB", DATA / "oof_xgb_diffFE.parquet")]:
        try:
            other = pd.read_parquet(path)
            m = pd.DataFrame({"id": train[ID_COL], "oof": oof}).merge(
                other[["id", "oof"]].rename(columns={"oof": "other"}), on="id", how="inner")
            rho, _ = spearmanr(m["oof"], m["other"])
            print(f"  rank-corr vs {name:20s}: {rho:.5f}  (n={len(m):,})")
        except Exception as e:
            print(f"  rank-corr vs {name}: skipped ({e})")

    oof_df.assign(fold=-1).to_parquet(OOF_OUT, index=False)
    sub = pd.DataFrame({"id": test[ID_COL], TARGET: test_preds}).sort_values("id").reset_index(drop=True)
    sub.to_csv(SUB_OUT, index=False)
    print(f"\nwrote {OOF_OUT.name}  ({len(oof_df):,} rows)")
    print(f"wrote {SUB_OUT.name}  ({len(sub):,} rows)")


if __name__ == "__main__":
    main()
