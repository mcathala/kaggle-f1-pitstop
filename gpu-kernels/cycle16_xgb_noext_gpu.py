"""Experiment 056 (cycle 16) — NO-EXTERNAL ablation of XGB-highbins (Kaggle GPU).

Diagnostic motivated by the hypothesis that our external augmentation dataset may
be capping us. We measured it: the external set is distribution-shifted from the
competition data — adversarial train-vs-external AUC 0.78 (vs ~0.5 train-vs-test),
pos rate 0.255 vs 0.199, skewed toward late-race high-pit-pressure laps. We have
force-fed 101k of those rows into training since cycle 3.

This ablation removes the external data ENTIRELY (not loaded; FE union and training
both competition-only) and re-runs the verbatim cycle-11 XGB-highbins recipe. If
the OOF rises above the with-external 0.95263, the external augmentation is a net
drag at current model strength and should be dropped (or down-weighted) across the
stack. If OOF falls, external still helps and the cap is elsewhere.

Runs on the P100 via device=cuda (XGBoost supports sm_60).

Inputs (add in Kaggle): competition playground-series-s6e5 (external NOT needed).
Outputs (/kaggle/working/): oof_xgb_noext.parquet, submission_xgb_noext.csv
"""

import subprocess
print("=== nvidia-smi ===")
try:
    print(subprocess.check_output(["nvidia-smi"], text=True))
except Exception as e:
    print(f"nvidia-smi failed: {e}")
print("==================")

from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

print(f"xgboost version: {xgb.__version__}")

# ---- Kaggle input/output paths ----
KAGGLE_INPUT = Path("/kaggle/input")


def find_one(filename: str) -> Path:
    hits = list(KAGGLE_INPUT.rglob(filename))
    if not hits:
        for p in sorted(KAGGLE_INPUT.rglob("*")):
            print(f"  {p}")
        raise FileNotFoundError(f"{filename} not found under {KAGGLE_INPUT}")
    if len(hits) > 1:
        print(f"WARN: multiple {filename}: {hits}")
    return hits[0]


TRAIN_CSV = find_one("train.csv")
TEST_CSV = find_one("test.csv")
WORKING = Path("/kaggle/working")
OOF_OUT = WORKING / "oof_xgb_noext.parquet"
SUB_OUT = WORKING / "submission_xgb_noext.csv"

TARGET = "PitNextLap"
ID_COL = "id"
BASE_CATS = ["Driver", "Race", "Compound"]
N_SPLITS = 5
SEED = 42

# cycle-11 XGB-highbins HPs, device=cuda for the P100.
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "device": "cuda",
    "enable_categorical": True,
    "max_bin": 5000,
    "max_depth": 10,
    "eta": 0.01,
    "min_child_weight": 2,
    "subsample": 0.8570122278990485,
    "colsample_bytree": 0.1450999139156032,
    "reg_lambda": 8.162374349037115,
    "reg_alpha": 8.354463958574286,
    "verbosity": 1,
}
N_ROUNDS = 50000
EARLY_STOP = 100


def safe_div(a, b, eps=1e-6):
    return a / (b + eps)


# ---------- THE EXPERIMENTAL FEATURE FAMILY ----------
def add_tyre_overdue_features(frames: list[pd.DataFrame]) -> list[str]:
    """Per-compound tyre-age normalisation, computed on the union of all frames.

    Reference percentiles are taken over train+test+external combined, so they
    don't leak fold-specific information (they're a fixed property of each
    compound, like a population statistic). Returns the list of added columns.
    """
    combined = pd.concat([f[["Compound", "TyreLife", "Cumulative_Degradation"]] for f in frames],
                         axis=0, ignore_index=True)
    grp = combined.groupby("Compound")["TyreLife"]
    ref = pd.DataFrame({
        "median": grp.median(),
        "p75": grp.quantile(0.75),
        "p90": grp.quantile(0.90),
    })
    deg_ref = combined.groupby("Compound")["Cumulative_Degradation"].median()

    added: list[str] = []
    for f in frames:
        c = f["Compound"]
        med = c.map(ref["median"]).astype("float32")
        p75 = c.map(ref["p75"]).astype("float32")
        p90 = c.map(ref["p90"]).astype("float32")
        f["_tyre_minus_compound_median"] = (f["TyreLife"] - med).astype("float32")
        f["_tyre_over_compound_p75"] = safe_div(f["TyreLife"], p75).astype("float32")
        f["_tyre_overdue_p75"] = (f["TyreLife"] > p75).astype(np.int8)
        f["_tyre_overdue_p90"] = (f["TyreLife"] > p90).astype(np.int8)
        f["_tyre_beyond_p90_amt"] = (f["TyreLife"] - p90).clip(lower=0).astype("float32")
        f["_deg_vs_compound_median"] = (
            f["Cumulative_Degradation"] - c.map(deg_ref).astype("float32")
        ).astype("float32")
    added = ["_tyre_minus_compound_median", "_tyre_over_compound_p75", "_tyre_overdue_p75",
             "_tyre_overdue_p90", "_tyre_beyond_p90_amt", "_deg_vs_compound_median"]

    # per-compound percentile rank of TyreLife (within each frame, ref is global ordering)
    order = combined["TyreLife"].rank(pct=True)  # global pct as a stable proxy
    # cheap per-frame: rank TyreLife within compound using the combined ranking map is
    # overkill; use the global tyre-life pct mapped by value bins instead.
    val_to_pct = (pd.Series(order.values, index=combined["TyreLife"].values)
                  .groupby(level=0).mean())
    for f in frames:
        f["_tyre_global_pct"] = f["TyreLife"].map(val_to_pct).astype("float32")
    added.append("_tyre_global_pct")
    return added


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


def add_cross_categoricals(out: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("Race_Year", ["Race", "Year"]), ("Compound_Stint", ["Compound", "Stint"]),
        ("Driver_Race", ["Driver", "Race"]), ("Driver_Compound", ["Driver", "Compound"]),
        ("Race_Compound", ["Race", "Compound"]), ("Race_Compound_Stint", ["Race", "Compound", "Stint"]),
        ("Compound_RacePhase", ["Compound", "RacePhase"]), ("Compound_TyreLifeBin", ["Compound", "TyreLifeBin"]),
        ("RacePhase_TyreLifeBin", ["RacePhase", "TyreLifeBin"]),
    ]
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
    keep_cols = list(set(group_cols + value_cols))
    combined = pd.concat([f[[c for c in keep_cols if c in f.columns]].copy() for f in frames],
                         axis=0, ignore_index=True)
    for g in group_cols:
        if g not in combined.columns:
            continue
        for v in value_cols:
            if v not in combined.columns:
                continue
            stats = combined.groupby(g, dropna=False)[v].agg(["mean", "std"])
            mean_col, std_col, diff_col = f"{v}_mean_by_{g}", f"{v}_std_by_{g}", f"{v}_diff_mean_by_{g}"
            for f in frames:
                if g not in f.columns or v not in f.columns:
                    continue
                key = f[g]
                f[mean_col] = key.map(stats["mean"]).astype("float32")
                f[std_col] = key.map(stats["std"]).fillna(0).astype("float32")
                f[diff_col] = (f[v] - f[mean_col]).astype("float32")
            added.extend([mean_col, std_col, diff_col])
    return added


def normalize_cats(out, cat_cols):
    for col in cat_cols:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("__NA__").astype(str)


def main():
    print("Loading data (NO EXTERNAL — ablation)...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    print(f"  train {train.shape}  test {test.shape}  (external deliberately excluded)")

    print("Applying domain FE...")
    train, test = add_domain_features(train), add_domain_features(test)
    train, test = add_cross_categoricals(train), add_cross_categoricals(test)

    cross_cats = ["Race_Year", "Compound_Stint", "Driver_Race", "Driver_Compound", "Race_Compound",
                  "Race_Compound_Stint", "Compound_RacePhase", "Compound_TyreLifeBin", "RacePhase_TyreLifeBin"]
    bins = ["RacePhase", "LapBin", "TyreLifeBin", "PositionBin"]
    all_cats = BASE_CATS + cross_cats + bins

    add_frequency_features([train, test], all_cats)
    gs = add_group_stats([train, test])
    print(f"  added {len(gs)} group-stat columns")

    normalize_cats(train, all_cats); normalize_cats(test, all_cats)
    for c in all_cats:
        if c not in train.columns:
            continue
        union_vals = (pd.concat([train[c], test[c]], axis=0).astype("string").fillna("__NA__").unique().tolist())
        cat_dtype = pd.CategoricalDtype(categories=sorted(union_vals))
        for f in (train, test):
            if c in f.columns:
                f[c] = f[c].astype(cat_dtype)

    feature_cols = [c for c in train.columns if c not in (TARGET, ID_COL)]
    feature_cols = [c for c in feature_cols if c in test.columns]
    n_cat = sum(1 for c in feature_cols if c in all_cats)
    print(f"using {len(feature_cols)} features, {n_cat} categorical")

    X = train[feature_cols]; y = train[TARGET].astype(int).to_numpy()
    X_test = test[feature_cols]

    strat_key = train["Year"].astype(str) + "_" + train[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train), dtype=np.float64)
    test_preds = np.zeros(len(test), dtype=np.float64)
    fold_aucs, fold_iters = [], []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, strat_key), start=1):
        t0 = time.time()
        X_tr = X.iloc[tr_idx]               # competition-only training (no external)
        y_tr = y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
        dval = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, enable_categorical=True)
        params = dict(XGB_PARAMS); params["seed"] = SEED + fold
        booster = xgb.train(params, dtrain, num_boost_round=N_ROUNDS, evals=[(dval, "val")],
                            early_stopping_rounds=EARLY_STOP, verbose_eval=500)
        bi = booster.best_iteration
        va_pred = booster.predict(dval, iteration_range=(0, bi + 1))
        oof[va_idx] = va_pred
        test_preds += booster.predict(dtest, iteration_range=(0, bi + 1)) / N_SPLITS
        a = roc_auc_score(y_va, va_pred); fold_aucs.append(a); fold_iters.append(bi + 1)
        print(f"fold {fold}/{N_SPLITS}  AUC={a:.5f}  iters={bi+1}  ({time.time()-t0:.1f}s)", flush=True)

    oof_auc = roc_auc_score(y, oof)
    print(f"\nper-fold mean={np.mean(fold_aucs):.5f} std={np.std(fold_aucs):.5f} iters={fold_iters}")
    print(f"OOF AUC: {oof_auc:.5f}  (vs cycle-11 XGB-highbins 0.95263, Δ={oof_auc-0.95263:+.5f})")

    oof_df = pd.DataFrame({"id": train[ID_COL], "Year": train["Year"], "target": y, "oof": oof, "fold": -1})
    oof_df.to_parquet(OOF_OUT, index=False)
    sub = pd.DataFrame({"id": test[ID_COL], TARGET: test_preds}).sort_values("id").reset_index(drop=True)
    sub.to_csv(SUB_OUT, index=False)
    print(f"wrote {OOF_OUT.name} ({len(oof_df):,}) and {SUB_OUT.name} ({len(sub):,})")
    print("Download both from the Output tab; place in data/ for the blend probe.")


if __name__ == "__main__":
    main()
