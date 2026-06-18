"""Experiment 042 (cycle 11) — XGBoost with target-encoded categoricals.

Replace XGBoost's native categorical-split (the 0.946 bottleneck) with sklearn's
`TargetEncoder` (cross-fold, low-leakage). Same 132-feature recipe but the 16
categoricals are now floats (target-frequency means) rather than category dtype.

Outputs:
  data/oof_xgb_te.parquet
  data/submission_xgb_te.csv
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
import xgboost as xgb

DATA = Path(__file__).resolve().parent.parent.parent / "data"
TRAIN_CSV = DATA / "train.csv"
TEST_CSV = DATA / "test.csv"
EXTERNAL_CSV = DATA / "f1_strategy_dataset.csv"
OOF_OUT = DATA / "oof_xgb_te.parquet"
SUB_OUT = DATA / "submission_xgb_te.csv"
REALMLP_OOF = DATA / "oof_realmlp_multiseed.parquet"
CB_OOF = DATA / "oof_cb_tuned_exp14.parquet"

TARGET = "PitNextLap"
ID_COL = "id"
BASE_CATS = ["Driver", "Race", "Compound"]
N_SPLITS = 5
SEED = 42

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "max_depth": 8,
    "eta": 0.05,
    "min_child_weight": 5.0,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_lambda": 5.0,
    "reg_alpha": 0.5,
    "nthread": -1,
    "verbosity": 1,
}
N_ROUNDS = 5000
EARLY_STOP = 300


# ============================================================
# Feature engineering — copy of cycle 14's pipeline
# ============================================================

def safe_div(a, b, eps=1e-6):
    return a / (b + eps)


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6
    out = df.copy()
    rp = out["RaceProgress"].clip(lower=eps)
    out["EstimatedTotalLaps"] = (out["LapNumber"] / rp).clip(1, 120).astype("float32")
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


def add_cross_categoricals(out: pd.DataFrame) -> pd.DataFrame:
    pairs = [("Race_Year", ["Race", "Year"]), ("Compound_Stint", ["Compound", "Stint"]),
             ("Driver_Race", ["Driver", "Race"]), ("Driver_Compound", ["Driver", "Compound"]),
             ("Race_Compound", ["Race", "Compound"]), ("Race_Compound_Stint", ["Race", "Compound", "Stint"]),
             ("Compound_RacePhase", ["Compound", "RacePhase"]), ("Compound_TyreLifeBin", ["Compound", "TyreLifeBin"]),
             ("RacePhase_TyreLifeBin", ["RacePhase", "TyreLifeBin"])]
    for name, cols in pairs:
        if all(c in out.columns for c in cols):
            value = out[cols[0]].astype(str)
            for col in cols[1:]:
                value = value + "_" + out[col].astype(str)
            out[name] = value
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


def add_group_stats(frames):
    group_cols = ["Race_Year", "Race_Compound_Stint", "Driver_Race", "Compound_Stint"]
    value_cols = ["LapTime_Delta", "Position_Change", "RaceProgress", "TyreLife"]
    added = []
    combined = pd.concat([f[[c for c in (group_cols + value_cols) if c in f.columns]].copy() for f in frames],
                         axis=0, ignore_index=True)
    for g in group_cols:
        if g not in combined.columns:
            continue
        for v in value_cols:
            if v not in combined.columns:
                continue
            stats = combined.groupby(g, dropna=False)[v].agg(["mean", "std"])
            for f in frames:
                if g not in f.columns or v not in f.columns:
                    continue
                key = f[g]
                f[f"{v}_mean_by_{g}"] = key.map(stats["mean"]).astype("float32")
                f[f"{v}_std_by_{g}"] = key.map(stats["std"]).fillna(0).astype("float32")
                f[f"{v}_diff_mean_by_{g}"] = (f[v] - f[f"{v}_mean_by_{g}"]).astype("float32")
            added.extend([f"{v}_mean_by_{g}", f"{v}_std_by_{g}", f"{v}_diff_mean_by_{g}"])
    return added


def normalize_cats(out, cat_cols):
    for col in cat_cols:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("__NA__").astype(str)


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("Loading data...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    ext = pd.read_csv(EXTERNAL_CSV).dropna(subset=["Compound"]).drop(columns=["Normalized_TyreLife"], errors="ignore")
    ext[ID_COL] = -1
    print(f"  train {train.shape}  test {test.shape}  external {ext.shape}")

    print("FE...")
    train = add_domain_features(train); test = add_domain_features(test); ext = add_domain_features(ext)
    train = add_cross_categoricals(train); test = add_cross_categoricals(test); ext = add_cross_categoricals(ext)

    cross_cats = ["Race_Year", "Compound_Stint", "Driver_Race", "Driver_Compound",
                  "Race_Compound", "Race_Compound_Stint", "Compound_RacePhase",
                  "Compound_TyreLifeBin", "RacePhase_TyreLifeBin"]
    bins = ["RacePhase", "LapBin", "TyreLifeBin", "PositionBin"]
    all_cats = BASE_CATS + cross_cats + bins

    add_frequency_features([train, test, ext], all_cats)
    add_group_stats([train, test, ext])
    normalize_cats(train, all_cats); normalize_cats(test, all_cats); normalize_cats(ext, all_cats)

    feature_cols = [c for c in train.columns if c not in (TARGET, ID_COL)]
    feature_cols = [c for c in feature_cols if c in test.columns and c in ext.columns]
    print(f"using {len(feature_cols)} features, {len(all_cats)} categorical (will be target-encoded)")

    y = train[TARGET].astype(int).to_numpy()
    y_ext = ext[TARGET].astype(int).to_numpy()

    strat_key = train["Year"].astype(str) + "_" + train[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train), dtype=np.float64)
    test_preds = np.zeros(len(test), dtype=np.float64)
    fold_aucs, fold_iters = [], []

    cat_cols_in_features = [c for c in all_cats if c in feature_cols]
    print(f"  will target-encode {len(cat_cols_in_features)} categorical columns")

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train[feature_cols], strat_key), start=1):
        t0 = time.time()

        # Build train+ext for this fold
        X_tr_raw = pd.concat([train[feature_cols].iloc[tr_idx], ext[feature_cols]], axis=0, ignore_index=True)
        y_tr = np.concatenate([y[tr_idx], y_ext])
        X_va_raw = train[feature_cols].iloc[va_idx].copy()
        X_test_raw = test[feature_cols].copy()

        # Target-encode the categorical columns
        te = TargetEncoder(target_type="binary", smooth="auto", cv=5, random_state=SEED + fold)
        te.fit(X_tr_raw[cat_cols_in_features], y_tr)
        X_tr_te = te.transform(X_tr_raw[cat_cols_in_features])
        X_va_te = te.transform(X_va_raw[cat_cols_in_features])
        X_test_te = te.transform(X_test_raw[cat_cols_in_features])

        # Substitute encoded values back in
        X_tr = X_tr_raw.copy()
        X_tr[cat_cols_in_features] = X_tr_te
        X_va = X_va_raw.copy()
        X_va[cat_cols_in_features] = X_va_te
        X_test_fold = X_test_raw.copy()
        X_test_fold[cat_cols_in_features] = X_test_te

        # Ensure all features are numeric
        X_tr = X_tr.astype("float32")
        X_va = X_va.astype("float32")
        X_test_fold = X_test_fold.astype("float32")

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_va, label=y[va_idx])
        dtest = xgb.DMatrix(X_test_fold)

        params = dict(XGB_PARAMS)
        params["seed"] = SEED + fold

        booster = xgb.train(params, dtrain, num_boost_round=N_ROUNDS,
                            evals=[(dval, "val")], early_stopping_rounds=EARLY_STOP, verbose_eval=200)

        best_iter = booster.best_iteration
        va_pred = booster.predict(dval, iteration_range=(0, best_iter + 1))
        oof[va_idx] = va_pred
        test_preds += booster.predict(dtest, iteration_range=(0, best_iter + 1)) / N_SPLITS
        fold_auc = roc_auc_score(y[va_idx], va_pred)
        fold_aucs.append(fold_auc)
        fold_iters.append(best_iter + 1)
        print(f"fold {fold}/{N_SPLITS}  AUC={fold_auc:.5f}  iters={best_iter + 1}  ({time.time()-t0:.0f}s)", flush=True)

    oof_auc = roc_auc_score(y, oof)
    print()
    print(f"per-fold AUC: mean={np.mean(fold_aucs):.5f}  std={np.std(fold_aucs):.5f}  iters={fold_iters}")
    print(f"OOF AUC:      {oof_auc:.5f}")
    print(f"  (vs xgb-native 0.94615, Δ = {oof_auc - 0.94615:+.5f})")
    print(f"  (vs CB-tuned-exp14 0.95114, Δ = {oof_auc - 0.95114:+.5f})")

    print("\nRank-correlation diagnostics:")
    for name, path in [("RealMLP-multiseed", REALMLP_OOF), ("CB-tuned-exp14", CB_OOF)]:
        try:
            other = pd.read_parquet(path)
            m = pd.DataFrame({"id": train[ID_COL], "oof": oof}).merge(
                other[[ID_COL, "oof"]].rename(columns={"oof": "other"}), on=ID_COL, how="inner")
            rho, _ = spearmanr(m["oof"], m["other"])
            print(f"  rank-corr vs {name:20s}: {rho:.5f}")
        except Exception as e:
            print(f"  rank-corr vs {name}: skipped ({e})")

    pd.DataFrame({ID_COL: train[ID_COL], "Year": train["Year"], "target": y, "oof": oof}).to_parquet(OOF_OUT, index=False)
    pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_preds}).sort_values(ID_COL).reset_index(drop=True).to_csv(SUB_OUT, index=False)
    print(f"\nwrote {OOF_OUT.name}, {SUB_OUT.name}")


if __name__ == "__main__":
    main()
