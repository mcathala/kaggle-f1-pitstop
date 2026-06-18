"""Experiment 065 (cycle 17) — pseudo-labeled RealMLP, FULL 6-seed (completes the thread).

exp 063 showed self-training lifts XGB standalone +0.00032 but only +0.00008 in the
blend (XGB is 0.25 weight, rho 0.9988). Apply the SAME idea to the 0.675-weight
RealMLP base: a standalone lift there has ~3x the blend leverage.

exp 064 showed single-seed pseudo lifts RealMLP +0.00011 (0.95355->0.95366).
This runs the FULL 6-seed average (seeds 42,7,99,137,313,777; CV folds fixed) to
match the production base (0.95383), giving the actual best-achievable pseudo-RM
for the blend. Est. ~0.95394 standalone -> blend ~0.95435 (still sub-hurdle but the
real best candidate over LB 0.95372/0.95373).

  Pass 1 — quick XGB on raw numerics + ordinal-encoded cats -> test predictions.
  Gate   — p>=0.92 -> pseudo-1, p<=0.03 -> pseudo-0 (else dropped).
  Pass 2 — verbatim cycle-5/11 RealMLP recipe (PyTabKit RealMLP_TD), 5-fold,
           trained on comp-train-fold + external + pseudo-test; OOF on comp rows.

Inputs: competition playground-series-s6e5 + external dataset.
Outputs: oof_realmlp_pseudo6.parquet, submission_realmlp_pseudo6.csv
"""

import subprocess, sys
print("=== nvidia-smi ===")
try:
    print(subprocess.check_output(["nvidia-smi"], text=True))
except Exception as e:
    print(f"nvidia-smi failed: {e}")
print("==================")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pytabkit==1.7.3"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "torch==2.5.1", "torchvision==0.20.1",
                       "--index-url", "https://download.pytorch.org/whl/cu121"])

import time
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import torch
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder, OrdinalEncoder
import xgboost as xgb
warnings.filterwarnings("ignore")

print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available — Kaggle allocated CPU. Set Accelerator P100/T4.")
device = "cuda"

KAGGLE_INPUT = Path("/kaggle/input")
def find_one(fn):
    hits = list(KAGGLE_INPUT.rglob(fn))
    if not hits:
        for p in sorted(KAGGLE_INPUT.rglob("*")): print("  ", p)
        raise FileNotFoundError(fn)
    return hits[0]
TRAIN_CSV, TEST_CSV = find_one("train.csv"), find_one("test.csv")
EXTERNAL_CSV = find_one("f1_strategy_dataset.csv")
WORKING = Path("/kaggle/working")
OOF_OUT = WORKING / "oof_realmlp_pseudo6.parquet"
SUB_OUT = WORKING / "submission_realmlp_pseudo6.csv"
TARGET, ID_COL = "PitNextLap", "id"
N_SPLITS, SEED = 5, 42
SEEDS = [42, 7, 99, 137, 313, 777]   # full multiseed avg; CV folds fixed at SEED
PSEUDO_HI, PSEUDO_LO = 0.92, 0.03
RAW_NUM = ["LapNumber","Stint","TyreLife","Position","LapTime (s)","LapTime_Delta","Cumulative_Degradation","RaceProgress","Position_Change","PitStop","Year"]
RAW_CAT = ["Driver","Race","Compound"]

REALMLP_PARAMS = {
    "random_state": SEED, "verbosity": 1, "val_metric_name": "1-auc_ovr",
    "n_ens": 24, "n_epochs": 6, "batch_size": 256, "use_early_stopping": False,
    "lr": 0.01, "wd": 0.016, "sq_mom": 0.99, "lr_sched": "lin_cos_log_15",
    "first_layer_lr_factor": 0.25, "embedding_size": 6, "max_one_hot_cat_size": 18,
    "hidden_sizes": [512, 256, 128], "act": "silu", "p_drop": 0.05,
    "p_drop_sched": "invsqrtp1e-3", "plr_hidden_1": 16, "plr_hidden_2": 8,
    "plr_act_name": "gelu", "plr_lr_factor": 0.1151, "plr_sigma": 2.33,
    "ls_eps": 0.01, "ls_eps_sched": "sqrt_cos", "add_front_scale": False,
    "bias_init_mode": "neg-uniform-dynamic-2",
    "tfms": ["one_hot","median_center","robust_scale","smooth_clip","embedding","l2_normalize"],
    "device": device,
}


def feature_engineering(df, fit, state):
    df["_LapNumber_/_RaceProgress"] = (df["LapNumber"] / (df["RaceProgress"] + 1e-6)).astype("float32")
    df["_TyreLife_/_LapNumber"] = (df["TyreLife"] / df["LapNumber"].clip(lower=1)).astype("float32")
    df["_LapTime (s)_*_Cumulative_Degradation"] = (df["LapTime (s)"] * df["Cumulative_Degradation"]).astype("float32")
    df["_LapTime (s)_*_Cumulative_Degradation_abs"] = (df["LapTime (s)"] * df["Cumulative_Degradation"].abs()).astype("float32")
    df["_LapTime (s)_/_Cumulative_Degradation_abs"] = (df["LapTime (s)"] / (df["Cumulative_Degradation"].abs() + 1e-6)).astype("float32")
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in df.select_dtypes(exclude=["object"]).columns.tolist() if c not in (ID_COL, TARGET)]
    for col in num_cols + ["_LapNumber_/_RaceProgress", "_TyreLife_/_LapNumber"]:
        cat_name = f"{col}_cat_" if col in num_cols else f"{col[1:]}_cat_"
        if fit:
            codes, uniques = np.floor(df[col]).astype(int).factorize(); state[col] = uniques
        else:
            code_map = {c: i for i, c in enumerate(state[col])}
            codes = np.floor(df[col]).astype(int).map(code_map).fillna(-1).astype("int32")
        df[cat_name] = codes.astype(str)
    for col in cat_cols + ["Year_cat_", "PitStop_cat_"]:
        count_name = f"_{col}_count" if col in cat_cols else f"_{col[:-1]}_count"
        if fit:
            cm = df[col].astype(object).value_counts(); state[count_name] = cm
        else:
            cm = state[count_name]
        df[count_name] = df[col].astype(object).map(cm).fillna(0).astype("int32")
    for col, bins_list in {"RaceProgress": [200], "LapTime (s)": [7]}.items():
        for n_bins in bins_list:
            bn = f"{col}_{n_bins}_quantile_bin_"
            if fit:
                kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
                df[bn] = kb.fit_transform(df[[col]]).ravel().astype("int32").astype(str); state[bn] = kb
            else:
                df[bn] = state[bn].transform(df[[col]]).ravel().astype("int32").astype(str)
    combo_names = []
    for cols in [("Race", "Compound"), ("Race", "Year")]:
        cn = "_".join(cols) + "_"; combo_names.append(cn)
        cs = df[cols[0]].astype(str)
        for c in cols[1:]: cs = cs + "_" + df[c].astype(str)
        if fit:
            codes, uniques = pd.factorize(cs, sort=False); state[cn] = uniques
        else:
            cmap = {c: i for i, c in enumerate(state[cn])}
            codes = cs.map(cmap).fillna(-1).astype("int32")
        df[cn] = codes.astype(str)
    return df, combo_names


def gen_pseudo(train, test, ext):
    """Quick XGB on raw numerics + ordinal cats -> confident test pseudo-labels."""
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    tr_cat = oe.fit_transform(pd.concat([train, ext], ignore_index=True)[RAW_CAT].astype(str))
    te_cat = oe.transform(test[RAW_CAT].astype(str))
    Xtr = np.hstack([pd.concat([train, ext], ignore_index=True)[RAW_NUM].fillna(-999).to_numpy(), tr_cat])
    ytr = pd.concat([train[TARGET], ext[TARGET]], ignore_index=True).astype(int).to_numpy()
    Xte = np.hstack([test[RAW_NUM].fillna(-999).to_numpy(), te_cat])
    dm = xgb.DMatrix(Xtr, label=ytr)
    b = xgb.train({"objective":"binary:logistic","eval_metric":"auc","tree_method":"hist","device":"cuda","max_depth":8,"eta":0.05,"subsample":0.8,"colsample_bytree":0.6,"verbosity":0}, dm, num_boost_round=400)
    p = b.predict(xgb.DMatrix(Xte))
    pl = np.where(p >= PSEUDO_HI, 1, np.where(p <= PSEUDO_LO, 0, -1))
    keep = pl >= 0
    print(f"  pseudo-labeled {keep.sum():,}/{len(p):,} test rows (hi={int((pl==1).sum()):,} lo={int((pl==0).sum()):,})")
    return keep, pl


def main():
    t0 = time.time()
    train = pd.read_csv(TRAIN_CSV); test = pd.read_csv(TEST_CSV)
    ext = pd.read_csv(EXTERNAL_CSV).drop(columns=["Normalized_TyreLife"], errors="ignore")
    print(f"train {train.shape} test {test.shape} ext {ext.shape}")

    # ---- pseudo-labels from a quick XGB (raw features) ----
    keep, pl = gen_pseudo(train, test, ext)

    y = train[TARGET].astype(int); y_ext = ext[TARGET].astype(int)
    train_id, test_id = train[ID_COL], test[ID_COL]
    X = train.drop([ID_COL, TARGET], axis=1)
    Xt = test.drop([ID_COL], axis=1)
    Xe = ext.drop([TARGET], axis=1)

    state = {}
    X, combo = feature_engineering(X, True, state)
    Xt, _ = feature_engineering(Xt, False, state)
    Xe, _ = feature_engineering(Xe, False, state)

    # confident pseudo-test rows (post-FE) + their labels
    Xp = Xt[keep].reset_index(drop=True); yp = pd.Series(pl[keep].astype(int))
    print(f"FE done. X {X.shape} Xt {Xt.shape} Xe {Xe.shape} pseudo {Xp.shape}")

    strat = train["Year"].astype(str) + "_" + y.astype(str)
    kf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    kfe = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(Xe, y_ext))
    splits = list(kf.split(X, strat))
    # precompute per-fold TE (seed-independent) once
    fold_data = []
    for (tr, va), (etr, _) in zip(splits, kfe):
        X_tr = pd.concat([X.iloc[tr], Xe.iloc[etr], Xp], axis=0).reset_index(drop=True)
        y_tr = pd.concat([y.iloc[tr], y_ext.iloc[etr], yp], axis=0).reset_index(drop=True)
        X_va = X.iloc[va].copy(); y_va = y.iloc[va]
        te = TargetEncoder(cv=N_SPLITS, smooth="auto", shuffle=True, random_state=SEED)
        names = [f"_{c}TE" for c in combo]
        X_tr[names] = te.fit_transform(X_tr[combo], y_tr)
        X_va[names] = te.transform(X_va[combo])
        X_ts = Xt.copy(); X_ts[names] = te.transform(Xt[combo])
        fold_data.append((tr, va, X_tr, y_tr, X_va, y_va, X_ts))

    oof = np.zeros(len(train)); test_preds = np.zeros(len(test))
    for seed in SEEDS:
        params = dict(REALMLP_PARAMS); params["random_state"] = seed
        oof_s = np.zeros(len(train)); test_s = np.zeros(len(test))
        for fi, (tr, va, X_tr, y_tr, X_va, y_va, X_ts) in enumerate(fold_data, 1):
            t1 = time.time()
            m = RealMLP_TD_Classifier(**params); m.fit(X_tr, y_tr, X_va, y_va)
            oof_s[va] = m.predict_proba(X_va)[:, 1]
            test_s += m.predict_proba(X_ts)[:, 1] / N_SPLITS
            print(f"seed {seed} fold {fi}/{N_SPLITS} AUC={roc_auc_score(y_va, oof_s[va]):.5f} ({time.time()-t1:.0f}s)", flush=True)
        oof += oof_s / len(SEEDS); test_preds += test_s / len(SEEDS)
        print(f"  seed {seed} OOF AUC = {roc_auc_score(y, oof_s):.5f}", flush=True)

    oa = roc_auc_score(y, oof)
    print(f"\n{len(SEEDS)}-seed OOF AUC: {oa:.5f}  (vs 6-seed RealMLP base 0.95383)")
    pd.DataFrame({"id": train_id, "Year": train["Year"], "target": y.to_numpy(), "oof": oof}).to_parquet(OOF_OUT, index=False)
    pd.DataFrame({"id": test_id, TARGET: test_preds}).sort_values("id").reset_index(drop=True).to_csv(SUB_OUT, index=False)
    print(f"wrote outputs ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
