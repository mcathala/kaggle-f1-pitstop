"""Experiment 063 (cycle 17) — confidence-gated self-training (pseudo-labels) on XGB.

Different MECHANISM from everything tried so far: semi-supervised learning.
Motivated by our transductive-structure finding (test laps sit among train laps of
the same driver-race) — re-attempt of cycle-9's pseudo-labeling now that we
understand the split. Self-training, self-contained:

  Pass 1 — quick XGB (eta 0.05) on comp+external -> test predictions.
  Gate   — confident test rows become pseudo-labels: p>0.92 -> 1, p<0.03 -> 0
           (others dropped).
  Pass 2 — full cycle-11 XGB-highbins recipe on comp + external + pseudo-test,
           5-fold; OOF measured on COMPETITION rows only (pseudo-test never in val,
           so OOF is leakage-free), test predicted.

Hypothesis: pseudo-labels add signal -> XGB OOF rises above 0.95263 and/or the
blend lifts. (Prior: low — train/test are same-distribution, adversarial ~0.5, so
test carries little new info; but it's a genuinely different mechanism, cheap, and
the transductive framing is new.)

Inputs: competition playground-series-s6e5 + external dataset.
Outputs: oof_xgb_pseudo.parquet, submission_xgb_pseudo.csv
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

print(f"xgboost {xgb.__version__}")
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
OOF_OUT = WORKING / "oof_xgb_pseudo.parquet"
SUB_OUT = WORKING / "submission_xgb_pseudo.csv"
TARGET, ID_COL = "PitNextLap", "id"
BASE_CATS = ["Driver", "Race", "Compound"]
N_SPLITS, SEED = 5, 42
PSEUDO_HI, PSEUDO_LO = 0.92, 0.03

XGB_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist",
    "device": "cuda", "enable_categorical": True, "max_bin": 5000, "max_depth": 10,
    "eta": 0.01, "min_child_weight": 2, "subsample": 0.8570122278990485,
    "colsample_bytree": 0.1450999139156032, "reg_lambda": 8.162374349037115,
    "reg_alpha": 8.354463958574286, "verbosity": 0,
}
N_ROUNDS, EARLY_STOP = 50000, 100


def safe_div(a, b, eps=1e-6):
    return a / (b + eps)


def add_domain_features(df):
    eps = 1e-6; out = df.copy()
    rp = out["RaceProgress"].clip(lower=eps)
    out["EstimatedTotalLaps"] = (out["LapNumber"] / rp).clip(1, 120).astype("float32")
    out["LapsRemaining"] = (out["EstimatedTotalLaps"] - out["LapNumber"]).clip(lower=0).astype("float32")
    out["RemainingRaceProgress"] = (1.0 - out["RaceProgress"]).astype("float32")
    out["LapProgress_x_LapNumber"] = (out["LapNumber"] * out["RaceProgress"]).astype("float32")
    out["RacePhase"] = pd.cut(out["RaceProgress"], bins=[-np.inf, .2, .4, .6, .8, np.inf], labels=["P1","P2","P3","P4","P5"]).astype(str)
    out["LapBin"] = pd.cut(out["LapNumber"], bins=[-np.inf,5,10,20,35,50,np.inf], labels=["L005","L010","L020","L035","L050","Lplus"]).astype(str)
    out["TyreAgeRatio"] = safe_div(out["TyreLife"], out["LapNumber"].clip(lower=1), eps).astype("float32")
    out["LapPerTyreLife"] = safe_div(out["LapNumber"], out["TyreLife"]+1, eps).astype("float32")
    out["TyreLife_x_RaceProgress"] = (out["TyreLife"]*out["RaceProgress"]).astype("float32")
    out["PitWindowPressure"] = (out["TyreLife"]*out["RaceProgress"]).astype("float32")
    out["TyreAgeVsRace"] = safe_div(out["TyreLife"], out["EstimatedTotalLaps"].clip(lower=1), eps).astype("float32")
    out["TyreLife_to_LapsRemaining"] = safe_div(out["TyreLife"], out["LapsRemaining"]+1, eps).astype("float32")
    out["LapMinusTyreLife"] = (out["LapNumber"]-out["TyreLife"]).astype("float32")
    out["TyreLifeBin"] = pd.cut(out["TyreLife"], bins=[-np.inf,3,7,12,20,30,np.inf], labels=["T003","T007","T012","T020","T030","Tplus"]).astype(str)
    out["StintPressure"] = (out["Stint"]*out["TyreLife"]).astype("float32")
    out["Is_First_Stint"] = (out["Stint"]==1).astype(np.int8)
    out["Is_Late_Stint"] = (out["Stint"]>=3).astype(np.int8)
    out["PositionBin"] = pd.cut(out["Position"], bins=[-np.inf,3,8,14,np.inf], labels=["front","upper_mid","lower_mid","back"]).astype(str)
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
    pairs = [("Race_Year",["Race","Year"]),("Compound_Stint",["Compound","Stint"]),("Driver_Race",["Driver","Race"]),
             ("Driver_Compound",["Driver","Compound"]),("Race_Compound",["Race","Compound"]),
             ("Race_Compound_Stint",["Race","Compound","Stint"]),("Compound_RacePhase",["Compound","RacePhase"]),
             ("Compound_TyreLifeBin",["Compound","TyreLifeBin"]),("RacePhase_TyreLifeBin",["RacePhase","TyreLifeBin"])]
    for name, cols in pairs:
        if all(c in out.columns for c in cols):
            v = out[cols[0]].astype(str)
            for c in cols[1:]: v = v + "_" + out[c].astype(str)
            out[name] = v
    return out


def add_frequency_features(frames, cat_cols):
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
    group_cols=["Race_Year","Race_Compound_Stint","Driver_Race","Compound_Stint"]
    value_cols=["LapTime_Delta","Position_Change","RaceProgress","TyreLife"]
    keep=list(set(group_cols+value_cols))
    combined=pd.concat([f[[c for c in keep if c in f.columns]].copy() for f in frames], axis=0, ignore_index=True)
    added=[]
    for g in group_cols:
        if g not in combined.columns: continue
        for v in value_cols:
            if v not in combined.columns: continue
            stats=combined.groupby(g, dropna=False)[v].agg(["mean","std"])
            mc,sc,dc=f"{v}_mean_by_{g}",f"{v}_std_by_{g}",f"{v}_diff_mean_by_{g}"
            for f in frames:
                if g not in f.columns or v not in f.columns: continue
                key=f[g]; f[mc]=key.map(stats["mean"]).astype("float32")
                f[sc]=key.map(stats["std"]).fillna(0).astype("float32"); f[dc]=(f[v]-f[mc]).astype("float32")
            added+=[mc,sc,dc]
    return added


def normalize_cats(out, cat_cols):
    for col in cat_cols:
        if col in out.columns: out[col]=out[col].astype("string").fillna("__NA__").astype(str)


def build_features(train, test, ext):
    train, test, ext = add_domain_features(train), add_domain_features(test), add_domain_features(ext)
    train, test, ext = add_cross_categoricals(train), add_cross_categoricals(test), add_cross_categoricals(ext)
    cross=["Race_Year","Compound_Stint","Driver_Race","Driver_Compound","Race_Compound","Race_Compound_Stint","Compound_RacePhase","Compound_TyreLifeBin","RacePhase_TyreLifeBin"]
    bins=["RacePhase","LapBin","TyreLifeBin","PositionBin"]
    all_cats=BASE_CATS+cross+bins
    add_frequency_features([train,test,ext], all_cats)
    add_group_stats([train,test,ext])
    for df in (train,test,ext): normalize_cats(df, all_cats)
    for c in all_cats:
        if c not in train.columns: continue
        uv=pd.concat([train[c],test[c],ext[c]],axis=0).astype("string").fillna("__NA__").unique().tolist()
        cd=pd.CategoricalDtype(categories=sorted(uv))
        for f in (train,test,ext):
            if c in f.columns: f[c]=f[c].astype(cd)
    feats=[c for c in train.columns if c not in (TARGET,ID_COL) and c in test.columns and c in ext.columns]
    return train, test, ext, feats


def main():
    t0=time.time()
    train=pd.read_csv(TRAIN_CSV); test=pd.read_csv(TEST_CSV)
    ext=pd.read_csv(EXTERNAL_CSV).dropna(subset=["Compound"]).drop(columns=["Normalized_TyreLife"],errors="ignore"); ext[ID_COL]=-1
    print(f"train {train.shape} test {test.shape} ext {ext.shape}")
    train, test, ext, feats = build_features(train, test, ext)
    print(f"{len(feats)} features")
    X=train[feats]; y=train[TARGET].astype(int).to_numpy(); Xt=test[feats]; Xe=ext[feats]; ye=ext[TARGET].astype(int).to_numpy()

    # ---- Pass 1: quick XGB for pseudo-labels ----
    print("Pass 1: quick XGB for test pseudo-labels...")
    q=dict(XGB_PARAMS); q["eta"]=0.05; q["seed"]=SEED
    dtr=xgb.DMatrix(pd.concat([X,Xe],ignore_index=True), label=np.concatenate([y,ye]), enable_categorical=True)
    dt=xgb.DMatrix(Xt, enable_categorical=True)
    b1=xgb.train(q, dtr, num_boost_round=2500)
    tp1=b1.predict(dt)
    hi=tp1>=PSEUDO_HI; lo=tp1<=PSEUDO_LO
    pl=np.where(hi,1,np.where(lo,0,-1))
    keep=pl>=0
    print(f"  pseudo-labeled {keep.sum():,}/{len(tp1):,} test rows (hi={hi.sum():,} lo={lo.sum():,})")

    # ---- Pass 2: full XGB-highbins on comp + external + pseudo-test ----
    Xp=Xt[keep].copy(); yp=pl[keep]
    strat=train["Year"].astype(str)+"_"+train[TARGET].astype(int).astype(str)
    kf=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof=np.zeros(len(train)); test_preds=np.zeros(len(test)); fa=[]
    for fold,(tr,va) in enumerate(kf.split(X,strat),1):
        t1=time.time()
        Xtr=pd.concat([X.iloc[tr],Xe,Xp],ignore_index=True)
        ytr=np.concatenate([y[tr],ye,yp])
        dtrain=xgb.DMatrix(Xtr,label=ytr,enable_categorical=True)
        dval=xgb.DMatrix(X.iloc[va],label=y[va],enable_categorical=True)
        p=dict(XGB_PARAMS); p["seed"]=SEED+fold
        bst=xgb.train(p,dtrain,num_boost_round=N_ROUNDS,evals=[(dval,"val")],early_stopping_rounds=EARLY_STOP,verbose_eval=1000)
        bi=bst.best_iteration
        oof[va]=bst.predict(dval,iteration_range=(0,bi+1))
        test_preds+=bst.predict(dt,iteration_range=(0,bi+1))/N_SPLITS
        a=roc_auc_score(y[va],oof[va]); fa.append(a)
        print(f"fold {fold}/{N_SPLITS} AUC={a:.5f} iters={bi+1} ({time.time()-t1:.0f}s)",flush=True)
    oa=roc_auc_score(y,oof)
    print(f"\nOOF AUC: {oa:.5f}  (vs plain XGB-highbins 0.95263, d={oa-0.95263:+.5f})")
    pd.DataFrame({"id":train[ID_COL],"Year":train["Year"],"target":y,"oof":oof}).to_parquet(OOF_OUT,index=False)
    pd.DataFrame({"id":test[ID_COL],TARGET:test_preds}).sort_values("id").reset_index(drop=True).to_csv(SUB_OUT,index=False)
    print(f"wrote outputs ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
