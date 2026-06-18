"""Experiment 087 (cycle 18) — alt-HP RealMLP as an HP-diverse 3rd RM view.

exp 084 showed FE-view diversity breaks the RealMLP-saturation ceiling: a second
RM view (diffFE) splits RM blend weight and transfers ~1:1 to LB. This tests a
complementary diversity axis — *hyperparameter* diversity within RealMLP. Same
diffFE lean-FE recipe + r2-pseudo labels, but a deliberately different training
regime: lr=0.04 with cos-anneal (vs the canonical schedule), sq_mom=0.99,
first_layer_lr_factor=0.25, hidden=[512,256,128] silu, p_drop=0.05, retuned PLR
embeddings, ls_eps=0.01. A different optimizer trajectory + capacity should land
in a different basin → a decorrelated RM view that earns separate blend weight.

Single seed first as a feasibility probe (~1h on M1 MPS). Gate: single-seed
OOF ≥ 0.9537 AND ρ < 0.99 vs psRM6r2 → escalate to 6-seed and add as a 3rd RM
view. If OOF caps low or ρ ≥ 0.99, the HP-diversity lever is dead for RealMLP.

  - device='mps' (M1)
  - labeler = data/submission_blend_pseudo_r2.csv
  - SEEDS = [42] (single-seed feasibility)
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

warnings.filterwarnings("ignore")

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch {torch.__version__}  device={device}")

DATA = Path(__file__).resolve().parent.parent.parent / "data"
TRAIN_CSV = DATA / "train.csv"
TEST_CSV = DATA / "test.csv"
EXTERNAL_CSV = DATA / "f1_strategy_dataset.csv"
LABELER_CSV = DATA / "submission_blend_pseudo_r2.csv"
OOF_OUT = DATA / "oof_realmlp_altHP_s42.parquet"
SUB_OUT = DATA / "submission_realmlp_altHP_s42.csv"

TARGET, ID_COL = "PitNextLap", "id"
N_SPLITS, SEED = 5, 42
SEEDS = [42]  # single-seed feasibility probe
PSEUDO_HI, PSEUDO_LO = 0.92, 0.03

REALMLP_PARAMS = {
    "random_state": SEED, "verbosity": 1, "val_metric_name": "1-auc_ovr",
    "n_ens": 24, "n_epochs": 10, "batch_size": 256, "use_early_stopping": False,
    "lr": 0.04, "wd": 0.016, "sq_mom": 0.99, "lr_sched": "cos_anneal",
    "first_layer_lr_factor": 0.25, "embedding_size": 6, "max_one_hot_cat_size": 18,
    "hidden_sizes": [512, 256, 128], "act": "silu", "p_drop": 0.05,
    "p_drop_sched": "expm4t", "plr_hidden_1": 16, "plr_hidden_2": 8,
    "plr_act_name": "gelu", "plr_lr_factor": 0.1151, "plr_sigma": 2.33,
    "ls_eps": 0.01, "ls_eps_sched": "sqrt_cos", "add_front_scale": False,
    "bias_init_mode": "neg-uniform-dynamic-2",
    "tfms": ["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding", "l2_normalize"],
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
    # diffFE (exp 082): drop categorize-numerics over-engineering; keep only
    # Year_cat_ / PitStop_cat_ (needed by the count block below). Mirrors the
    # exp 080/081 finding that stripping heavy engineered cats strengthens GBDTs.
    for col in ["Year", "PitStop"]:
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
        for c in cols[1:]:
            cs = cs + "_" + df[c].astype(str)
        if fit:
            codes, uniques = pd.factorize(cs, sort=False); state[cn] = uniques
        else:
            cmap = {c: i for i, c in enumerate(state[cn])}
            codes = cs.map(cmap).fillna(-1).astype("int32")
        df[cn] = codes.astype(str)
    return df, combo_names


def main():
    t0 = time.time()
    print(f"\nLoading data from {DATA}...")
    train = pd.read_csv(TRAIN_CSV); test = pd.read_csv(TEST_CSV)
    ext = pd.read_csv(EXTERNAL_CSV).drop(columns=["Normalized_TyreLife"], errors="ignore")
    print(f"train {train.shape}  test {test.shape}  ext {ext.shape}")

    print(f"Reading strong-blend labeler ({LABELER_CSV.name})...")
    lab = pd.read_csv(LABELER_CSV).sort_values(ID_COL).reset_index(drop=True)
    test_s = test.sort_values(ID_COL).reset_index(drop=True)
    assert (lab[ID_COL].to_numpy() == test_s[ID_COL].to_numpy()).all(), "labeler id mismatch"
    id_to_p = dict(zip(lab[ID_COL].to_numpy(), lab[TARGET].to_numpy()))
    tp = np.array([id_to_p[i] for i in test[ID_COL].to_numpy()])
    hi = tp >= PSEUDO_HI; lo = tp <= PSEUDO_LO
    pl = np.where(hi, 1, np.where(lo, 0, -1))
    keep = pl >= 0
    print(f"  pseudo-labeled {keep.sum():,}/{len(tp):,} test rows (hi={int((pl==1).sum()):,} lo={int((pl==0).sum()):,})")

    y = train[TARGET].astype(int); y_ext = ext[TARGET].astype(int)
    train_id, test_id = train[ID_COL], test[ID_COL]
    X = train.drop([ID_COL, TARGET], axis=1)
    Xt = test.drop([ID_COL], axis=1)
    Xe = ext.drop([TARGET], axis=1)

    state = {}
    X, combo = feature_engineering(X, True, state)
    Xt, _ = feature_engineering(Xt, False, state)
    Xe, _ = feature_engineering(Xe, False, state)

    Xp = Xt[keep].reset_index(drop=True); yp = pd.Series(pl[keep].astype(int))
    print(f"FE done. X {X.shape} Xt {Xt.shape} Xe {Xe.shape} pseudo {Xp.shape}")

    strat = train["Year"].astype(str) + "_" + y.astype(str)
    kf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    kfe = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(Xe, y_ext))
    splits = list(kf.split(X, strat))

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
        oof_s = np.zeros(len(train)); test_s_pred = np.zeros(len(test))
        for fi, (tr, va, X_tr, y_tr, X_va, y_va, X_ts) in enumerate(fold_data, 1):
            t1 = time.time()
            m = RealMLP_TD_Classifier(**params); m.fit(X_tr, y_tr, X_va, y_va)
            oof_s[va] = m.predict_proba(X_va)[:, 1]
            test_s_pred += m.predict_proba(X_ts)[:, 1] / N_SPLITS
            print(f"seed {seed} fold {fi}/{N_SPLITS} AUC={roc_auc_score(y_va, oof_s[va]):.5f} ({time.time()-t1:.0f}s)", flush=True)
        oof += oof_s / len(SEEDS); test_preds += test_s_pred / len(SEEDS)
        print(f"  seed {seed} OOF AUC = {roc_auc_score(y, oof_s):.5f}", flush=True)

    oa = roc_auc_score(y, oof)
    print(f"\n{len(SEEDS)}-seed OOF AUC: {oa:.5f}  (vs round-2 0.95396, Δ = {oa - 0.95396:+.5f})")
    pd.DataFrame({"id": train_id, "Year": train["Year"], "target": y.to_numpy(), "oof": oof}).to_parquet(OOF_OUT, index=False)
    pd.DataFrame({"id": test_id, TARGET: test_preds}).sort_values("id").reset_index(drop=True).to_csv(SUB_OUT, index=False)
    print(f"wrote {OOF_OUT.name} and {SUB_OUT.name}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
