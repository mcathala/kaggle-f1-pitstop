"""Experiment 072 (cycle 17) — round-3 pseudo-RealMLP-6seed (local M1 MPS).

Builds on exp 069 (round-2 won for RealMLP +0.00003). Tests whether iterative
self-training continues to compound: use the *new* round-2 blend (OOF 0.95436,
LB 0.95375) as the labeler for a round-3 RealMLP run.

Two hypotheses being tested simultaneously:
  - **H1 (compound):** round-3 OOF > round-2 OOF — labeler quality keeps lifting
    the base. If so, we're climbing a slow ramp.
  - **H2 (plateau):** round-3 OOF ≈ round-2 OOF — labeler quality has saturated
    relative to the test-distribution signal. This closes the iteration loop.

Single seed first as a *feasibility probe* (~1h on M1 MPS). If single-seed AUC
clears 0.9536, escalate to 6-seed. If not, the round-3 lever is dead.

Recipe verbatim from cycle17_realmlp_pseudo62_gpu.py except:
  - device='mps' (M1) instead of 'cuda'
  - labeler = data/submission_blend_pseudo_r2.csv (new 0.95436 blend)
  - SEEDS = [42, 7, 99, 137, 313, 777] (single-seed feasibility)
  - reads data from local data/ dir
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

from data import DATA, ID_COL, TARGET, load_competition, load_external, realmlp_features
LABELER_CSV = DATA / "submission_blend_pseudo_r2.csv"
OOF_OUT = DATA / "oof_realmlp_diffFE_6seed.parquet"
SUB_OUT = DATA / "submission_realmlp_diffFE_6seed.csv"

N_SPLITS, SEED = 5, 42
SEEDS = [42, 7, 99, 137, 313, 777]  # single-seed feasibility probe
PSEUDO_HI, PSEUDO_LO = 0.92, 0.03

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
    "tfms": ["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding", "l2_normalize"],
    "device": device,
}

def main():
    t0 = time.time()
    print(f"\nLoading data from {DATA}...")
    train, test = load_competition()
    ext = load_external(dropna_compound=False)
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
    X, combo = realmlp_features(X, True, state)
    Xt, _ = realmlp_features(Xt, False, state)
    Xe, _ = realmlp_features(Xe, False, state)

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
