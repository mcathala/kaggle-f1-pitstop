"""Experiment 048 (cycle 15) — Optuna HP sweep on XGB-highbins.

Cycle 11 exp 044 established XGB-highbins at OOF 0.95263 standalone using
Mikhail-style HPs. This sweep searches around that point for a better local
optimum that could close more of the gap to the top-10% LB tier.

Search budget: fold-1-only evaluation per trial (~5 min each), 25 TPE trials,
then full 5-fold on the top 3 trials. Total ~4-5h CPU.

Search space (around cycle 11's known-good point):
  max_depth        ∈ {6, 8, 10, 12}                (cycle 11: 10)
  eta              ∈ [0.005, 0.03]                  (cycle 11: 0.01)
  reg_lambda       ∈ [1, 16]                        (cycle 11: 8.16)
  reg_alpha        ∈ [0.5, 16]                      (cycle 11: 8.35)
  colsample_bytree ∈ [0.10, 0.30]                   (cycle 11: 0.145)
  subsample        ∈ [0.6, 0.95]                    (cycle 11: 0.857)
  min_child_weight ∈ [1, 8]                         (cycle 11: 2)
  max_bin          ∈ {3000, 5000, 8000}             (cycle 11: 5000)

Outputs:
  data/optuna_xgb_highbins_trials.parquet   trial history
  data/optuna_xgb_highbins_top3.parquet     top-3 trials by fold-1 AUC
"""

from pathlib import Path
import time
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import optuna

DATA = Path(__file__).resolve().parent.parent.parent / "data"
TRAIN_CSV = DATA / "train.csv"
EXTERNAL_CSV = DATA / "f1_strategy_dataset.csv"
TRIALS_OUT = DATA / "optuna_xgb_highbins_trials.parquet"
TOP3_OUT = DATA / "optuna_xgb_highbins_top3.parquet"

TARGET = "PitNextLap"
ID_COL = "id"
BASE_CATS = ["Driver", "Race", "Compound"]
N_SPLITS = 5
CV_SEED = 42
N_TRIALS = 25

# Capped iters per trial for HP search (full eval uses 50000 + 100 patience).
TRIAL_N_ROUNDS = 5000
TRIAL_EARLY_STOP = 50

# Cycle 14 FE pipeline (verbatim from train_xgb_richcat.py)

def safe_div(a, b, eps=1e-6):
    return a / (b + eps)

def add_domain_features(df):
    eps = 1e-6
    out = df.copy()
    rp = out["RaceProgress"].clip(lower=eps)
    out["EstimatedTotalLaps"] = (out["LapNumber"] / rp).clip(1, 120).astype("float32")
    out["LapsRemaining"] = (out["EstimatedTotalLaps"] - out["LapNumber"]).clip(lower=0).astype("float32")
    out["RemainingRaceProgress"] = (1.0 - out["RaceProgress"]).astype("float32")
    out["LapProgress_x_LapNumber"] = (out["LapNumber"] * out["RaceProgress"]).astype("float32")
    out["RacePhase"] = pd.cut(out["RaceProgress"], bins=[-np.inf, 0.20, 0.40, 0.60, 0.80, np.inf],
                              labels=["P1","P2","P3","P4","P5"]).astype(str)
    out["LapBin"] = pd.cut(out["LapNumber"], bins=[-np.inf, 5, 10, 20, 35, 50, np.inf],
                            labels=["L005","L010","L020","L035","L050","Lplus"]).astype(str)
    out["TyreAgeRatio"] = safe_div(out["TyreLife"], out["LapNumber"].clip(lower=1), eps).astype("float32")
    out["LapPerTyreLife"] = safe_div(out["LapNumber"], out["TyreLife"]+1, eps).astype("float32")
    out["TyreLife_x_RaceProgress"] = (out["TyreLife"]*out["RaceProgress"]).astype("float32")
    out["PitWindowPressure"] = (out["TyreLife"]*out["RaceProgress"]).astype("float32")
    out["TyreAgeVsRace"] = safe_div(out["TyreLife"], out["EstimatedTotalLaps"].clip(lower=1), eps).astype("float32")
    out["TyreLife_to_LapsRemaining"] = safe_div(out["TyreLife"], out["LapsRemaining"]+1, eps).astype("float32")
    out["LapMinusTyreLife"] = (out["LapNumber"]-out["TyreLife"]).astype("float32")
    out["TyreLifeBin"] = pd.cut(out["TyreLife"], bins=[-np.inf,3,7,12,20,30,np.inf],
                                 labels=["T003","T007","T012","T020","T030","Tplus"]).astype(str)
    out["StintPressure"] = (out["Stint"]*out["TyreLife"]).astype("float32")
    out["Is_First_Stint"] = (out["Stint"]==1).astype(np.int8)
    out["Is_Late_Stint"] = (out["Stint"]>=3).astype(np.int8)
    out["PositionBin"] = pd.cut(out["Position"], bins=[-np.inf,3,8,14,np.inf],
                                 labels=["front","upper_mid","lower_mid","back"]).astype(str)
    out["PositionPressure"] = (out["Position"]*out["RaceProgress"]).astype("float32")
    out["DegPerRaceLap"] = safe_div(out["Cumulative_Degradation"], out["LapNumber"].clip(lower=1), eps).astype("float32")
    out["DegPerTyreLap"] = safe_div(out["Cumulative_Degradation"], out["TyreLife"].clip(lower=1), eps).astype("float32")
    out["Abs_Cumulative_Degradation"] = out["Cumulative_Degradation"].abs().astype("float32")
    out["Positive_Degradation"] = (out["Cumulative_Degradation"]>0).astype(np.int8)
    out["DeltaAbs"] = out["LapTime_Delta"].abs().astype("float32")
    out["LapTimeDeltaPositive"] = (out["LapTime_Delta"]>0).astype(np.int8)
    out["DeltaPerTyreLap"] = safe_div(out["LapTime_Delta"], out["TyreLife"].clip(lower=1), eps).astype("float32")
    out["Abs_Position_Change"] = out["Position_Change"].abs().astype("float32")
    out["Gained_Position"] = (out["Position_Change"]>0).astype(np.int8)
    out["Lost_Position"] = (out["Position_Change"]<0).astype(np.int8)
    return out

def add_cross_categoricals(out):
    pairs = [("Race_Year",["Race","Year"]),("Compound_Stint",["Compound","Stint"]),
             ("Driver_Race",["Driver","Race"]),("Driver_Compound",["Driver","Compound"]),
             ("Race_Compound",["Race","Compound"]),("Race_Compound_Stint",["Race","Compound","Stint"]),
             ("Compound_RacePhase",["Compound","RacePhase"]),("Compound_TyreLifeBin",["Compound","TyreLifeBin"]),
             ("RacePhase_TyreLifeBin",["RacePhase","TyreLifeBin"])]
    for name, cols in pairs:
        if all(c in out.columns for c in cols):
            v = out[cols[0]].astype(str)
            for col in cols[1:]:
                v = v + "_" + out[col].astype(str)
            out[name] = v
    return out

def add_frequency_features(frames, cat_cols):
    if not cat_cols: return
    total = sum(len(f) for f in frames)
    for col in cat_cols:
        if not all(col in f.columns for f in frames): continue
        union = pd.concat([f[col].astype("string").fillna("__NA__") for f in frames], axis=0)
        counts = union.value_counts(dropna=False)
        for f in frames:
            keys = f[col].astype("string").fillna("__NA__")
            f[f"{col}_count"] = keys.map(counts).fillna(0).astype(np.int32)
            f[f"{col}_freq"] = (f[f"{col}_count"]/total).astype("float32")

def add_group_stats(frames):
    group_cols = ["Race_Year","Race_Compound_Stint","Driver_Race","Compound_Stint"]
    value_cols = ["LapTime_Delta","Position_Change","RaceProgress","TyreLife"]
    combined = pd.concat([f[[c for c in (group_cols+value_cols) if c in f.columns]].copy() for f in frames],
                         axis=0, ignore_index=True)
    for g in group_cols:
        if g not in combined.columns: continue
        for v in value_cols:
            if v not in combined.columns: continue
            stats = combined.groupby(g, dropna=False)[v].agg(["mean","std"])
            for f in frames:
                if g not in f.columns or v not in f.columns: continue
                key = f[g]
                f[f"{v}_mean_by_{g}"] = key.map(stats["mean"]).astype("float32")
                f[f"{v}_std_by_{g}"] = key.map(stats["std"]).fillna(0).astype("float32")
                f[f"{v}_diff_mean_by_{g}"] = (f[v]-f[f"{v}_mean_by_{g}"]).astype("float32")

def normalize_cats(out, cat_cols):
    for col in cat_cols:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("__NA__").astype(str)


def prepare_data():
    print("Loading data + applying FE...")
    train = pd.read_csv(TRAIN_CSV)
    ext = pd.read_csv(EXTERNAL_CSV).dropna(subset=["Compound"]).drop(columns=["Normalized_TyreLife"], errors="ignore")
    ext[ID_COL] = -1
    train = add_domain_features(train); ext = add_domain_features(ext)
    train = add_cross_categoricals(train); ext = add_cross_categoricals(ext)
    cross_cats = ["Race_Year","Compound_Stint","Driver_Race","Driver_Compound","Race_Compound",
                  "Race_Compound_Stint","Compound_RacePhase","Compound_TyreLifeBin","RacePhase_TyreLifeBin"]
    bins = ["RacePhase","LapBin","TyreLifeBin","PositionBin"]
    all_cats = BASE_CATS + cross_cats + bins
    # need both for freq feats — use just train+ext (test not needed for HP search)
    add_frequency_features([train, ext], all_cats)
    add_group_stats([train, ext])
    normalize_cats(train, all_cats); normalize_cats(ext, all_cats)

    # Unified categorical dtype across frames
    for c in all_cats:
        if c not in train.columns: continue
        vals = pd.concat([train[c], ext[c]], axis=0).astype("string").fillna("__NA__").unique().tolist()
        ctype = pd.CategoricalDtype(categories=sorted(vals))
        for f in (train, ext):
            if c in f.columns:
                f[c] = f[c].astype(ctype)

    feature_cols = [c for c in train.columns if c not in (TARGET, ID_COL) and c in ext.columns]
    X = train[feature_cols]
    y = train[TARGET].astype(int).to_numpy()
    X_ext = ext[feature_cols]
    y_ext = ext[TARGET].astype(int).to_numpy()
    strat_key = train["Year"].astype(str) + "_" + train[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
    # Get fold 1 indices only
    splits = list(kf.split(X, strat_key))
    tr_idx, va_idx = splits[0]
    print(f"FE done. {len(feature_cols)} features. fold-1 train rows {len(tr_idx):,} val rows {len(va_idx):,}")
    return X, y, X_ext, y_ext, tr_idx, va_idx


# Globals — initialized in main(), used by objective()
_X = None
_y = None
_X_ext = None
_y_ext = None
_tr_idx = None
_va_idx = None
_dtrain = None
_dval = None


def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "enable_categorical": True,
        "max_bin": trial.suggest_categorical("max_bin", [3000, 5000, 8000]),
        "max_depth": trial.suggest_categorical("max_depth", [6, 8, 10, 12]),
        "eta": trial.suggest_float("eta", 0.005, 0.03, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 8, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.10, 0.30),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 16.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.5, 16.0, log=True),
        "nthread": -1,
        "verbosity": 0,
        "seed": CV_SEED + 1,
    }

    booster = xgb.train(
        params, _dtrain, num_boost_round=TRIAL_N_ROUNDS,
        evals=[(_dval, "val")],
        early_stopping_rounds=TRIAL_EARLY_STOP,
        verbose_eval=False,
    )
    best_iter = booster.best_iteration
    va_pred = booster.predict(_dval, iteration_range=(0, best_iter + 1))
    return roc_auc_score(_y[_va_idx], va_pred)


def main():
    global _X, _y, _X_ext, _y_ext, _tr_idx, _va_idx, _dtrain, _dval
    _X, _y, _X_ext, _y_ext, _tr_idx, _va_idx = prepare_data()

    # Build fold-1 train/val DMatrix once (re-used by all trials)
    X_tr_full = pd.concat([_X.iloc[_tr_idx], _X_ext], axis=0, ignore_index=True)
    y_tr_full = np.concatenate([_y[_tr_idx], _y_ext])
    _dtrain = xgb.DMatrix(X_tr_full, label=y_tr_full, enable_categorical=True)
    _dval = xgb.DMatrix(_X.iloc[_va_idx], label=_y[_va_idx], enable_categorical=True)

    print(f"\nStarting Optuna sweep — {N_TRIALS} trials × ~5 min each.")
    print(f"Baseline (cycle 11 HPs): fold-1 AUC 0.95331")
    print()

    sampler = optuna.samplers.TPESampler(seed=CV_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    t_start = time.time()
    def callback(study, trial):
        elapsed = (time.time() - t_start) / 60
        print(f"  trial {trial.number:2d} done  AUC={trial.value:.5f}  best so far={study.best_value:.5f}  ({elapsed:.1f} min total)", flush=True)
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[callback])

    print(f"\nFinished. Best AUC: {study.best_value:.5f}")
    print(f"Best params: {study.best_params}")

    # Persist trial history
    rows = [
        {"trial": t.number, "auc": t.value, **t.params}
        for t in study.trials
    ]
    df = pd.DataFrame(rows).sort_values("auc", ascending=False)
    df.to_parquet(TRIALS_OUT, index=False)
    df.head(3).to_parquet(TOP3_OUT, index=False)
    print(f"\nwrote {TRIALS_OUT.name} and {TOP3_OUT.name}")
    print("\nTop 5 trials:")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
