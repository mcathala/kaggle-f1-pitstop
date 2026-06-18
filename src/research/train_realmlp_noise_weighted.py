"""Experiment 076 (cycle 17) — noise-weighted RealMLP (bidirectional, ablate-first).

Audit Phase-1 #3 (corrected to bidirectional). The label-noise mechanism is
bidirectional:
  - PitNextLap=1 AND no stint increment on the next sampled row: ~65k rows
    (74.6% of all PitNextLap=1 labels — likely noisy positives).
  - PitNextLap=0 AND stint increment on the next sampled row: ~77k rows
    (21.9% of all PitNextLap=0 — likely noisy negatives).
  - Total bidirectional noise: ~142k / 439k = 32% of train (upper bound; data
    is row-sampled so "no increment on next sampled row" may understate noise).

Since pytabkit's `fit()` doesn't accept sample_weight, we implement weighting
via row-subsampling: keep "agree" rows at 1×, subsample "disagree" rows to 50%
of their count per fold. Effective weight ratio w_agree : w_disagree = 1 : 0.5
under stochastic gradient training.

This is an *ablate-first* run: single seed, build on the round-2 pseudo-RM
recipe (exp 069, which gave OOF 0.95396 currently the strongest RM base) by
swapping the train-set construction. If single-fold AUC ≥ 0.9540 (above the
exp 069 fold-1 baseline), continue all 5 folds. Otherwise kill.

Outputs:
  data/oof_realmlp_noise_weighted_s42.parquet
  data/submission_realmlp_noise_weighted_s42.csv
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
OOF_OUT = DATA / "oof_realmlp_noise_weighted_s42.parquet"
SUB_OUT = DATA / "submission_realmlp_noise_weighted_s42.csv"

TARGET, ID_COL = "PitNextLap", "id"
N_SPLITS, SEED = 5, 42
SEEDS = [42]
PSEUDO_HI, PSEUDO_LO = 0.92, 0.03
NOISY_SUBSAMPLE_RATE = 0.5  # keep this fraction of noisy rows
ABLATE_FIRST_GATE = 0.9540  # single-fold AUC; below this → kill

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
        for c in cols[1:]:
            cs = cs + "_" + df[c].astype(str)
        if fit:
            codes, uniques = pd.factorize(cs, sort=False); state[cn] = uniques
        else:
            cmap = {c: i for i, c in enumerate(state[cn])}
            codes = cs.map(cmap).fillna(-1).astype("int32")
        df[cn] = codes.astype(str)
    return df, combo_names


def compute_noise_mask(train: pd.DataFrame) -> np.ndarray:
    """Return boolean mask: True = 'noisy' row, False = 'agree' row.

    A row is considered 'noisy' if its PitNextLap label disagrees with the
    stint-transition signal on the next sampled row within (Driver, Race, Year):
      - PitNextLap=1 but no stint increment → noisy positive
      - PitNextLap=0 but stint increment ≥1 → noisy negative
    """
    df = train.sort_values(["Year", "Race", "Driver", "LapNumber"]).reset_index(drop=False)
    df["next_stint"] = df.groupby(["Year", "Race", "Driver"])["Stint"].shift(-1)
    df["stint_inc"] = (df["next_stint"] - df["Stint"]).fillna(0).astype(int)
    noisy_pos = (df[TARGET] == 1) & (df["stint_inc"] == 0)
    noisy_neg = (df[TARGET] == 0) & (df["stint_inc"] >= 1)
    noisy = (noisy_pos | noisy_neg).to_numpy()
    # Restore original order using the 'index' column captured by reset_index(drop=False)
    df = df.assign(_noisy=noisy).sort_values("index").reset_index(drop=True)
    return df["_noisy"].to_numpy()


def main():
    t0 = time.time()
    print(f"\nLoading data from {DATA}...")
    train = pd.read_csv(TRAIN_CSV); test = pd.read_csv(TEST_CSV)
    ext = pd.read_csv(EXTERNAL_CSV).drop(columns=["Normalized_TyreLife"], errors="ignore")
    print(f"train {train.shape}  test {test.shape}  ext {ext.shape}")

    # Sort train by id so the noise mask aligns with X
    train = train.sort_values(ID_COL).reset_index(drop=True)
    test = test.sort_values(ID_COL).reset_index(drop=True)

    print("Building noise mask (bidirectional)...")
    noise_mask = compute_noise_mask(train.copy())
    n_noisy = int(noise_mask.sum())
    n_agree = len(train) - n_noisy
    print(f"  agree: {n_agree:,} ({n_agree/len(train)*100:.1f}%)   noisy: {n_noisy:,} ({n_noisy/len(train)*100:.1f}%)")

    print(f"Reading strong-blend labeler ({LABELER_CSV.name})...")
    lab = pd.read_csv(LABELER_CSV).sort_values(ID_COL).reset_index(drop=True)
    assert (lab[ID_COL].to_numpy() == test[ID_COL].to_numpy()).all(), "labeler id mismatch"
    tp = lab[TARGET].to_numpy()
    hi = tp >= PSEUDO_HI; lo = tp <= PSEUDO_LO
    pl = np.where(hi, 1, np.where(lo, 0, -1))
    keep = pl >= 0
    print(f"  pseudo-labeled {keep.sum():,}/{len(tp):,} test rows (hi={int((pl==1).sum()):,} lo={int((pl==0).sum()):,})")

    y = train[TARGET].astype(int); y_ext = ext[TARGET].astype(int)
    train_id, test_id = train[ID_COL], test[ID_COL]
    year_train = train["Year"].to_numpy()
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

    # Pre-build per-fold training data with noisy-row subsampling
    rng = np.random.default_rng(SEED)
    fold_data = []
    for (tr, va), (etr, _) in zip(splits, kfe):
        noise_tr = noise_mask[tr]
        agree_idx = tr[~noise_tr]
        noisy_idx = tr[noise_tr]
        n_keep = int(NOISY_SUBSAMPLE_RATE * len(noisy_idx))
        kept_noisy = rng.choice(noisy_idx, size=n_keep, replace=False)
        tr_idx = np.concatenate([agree_idx, kept_noisy])
        rng.shuffle(tr_idx)

        X_tr = pd.concat([X.iloc[tr_idx], Xe.iloc[etr], Xp], axis=0).reset_index(drop=True)
        y_tr = pd.concat([y.iloc[tr_idx], y_ext.iloc[etr], yp], axis=0).reset_index(drop=True)
        X_va = X.iloc[va].copy(); y_va = y.iloc[va]
        te = TargetEncoder(cv=N_SPLITS, smooth="auto", shuffle=True, random_state=SEED)
        names = [f"_{c}TE" for c in combo]
        X_tr[names] = te.fit_transform(X_tr[combo], y_tr)
        X_va[names] = te.transform(X_va[combo])
        X_ts = Xt.copy(); X_ts[names] = te.transform(Xt[combo])
        print(f"  fold prep: agree={len(agree_idx):,}  noisy_kept={n_keep:,}/{len(noisy_idx):,}  total_tr={len(X_tr):,}")
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
            auc = roc_auc_score(y_va.to_numpy(), oof_s[va])
            print(f"seed {seed} fold {fi}/{N_SPLITS} AUC={auc:.5f} ({time.time()-t1:.0f}s)", flush=True)
            if fi == 1 and auc < ABLATE_FIRST_GATE:
                print(f"\n*** ABLATE-FIRST GATE FAILED *** fold-1 AUC {auc:.5f} < {ABLATE_FIRST_GATE:.4f}; killing run.")
                return
        oof += oof_s / len(SEEDS); test_preds += test_s_pred / len(SEEDS)
        print(f"  seed {seed} OOF AUC = {roc_auc_score(y, oof_s):.5f}", flush=True)

    oa = roc_auc_score(y, oof)
    print(f"\n{len(SEEDS)}-seed OOF AUC: {oa:.5f}  (vs round-2 0.95396, Δ {oa - 0.95396:+.5f})")

    print("\nrank-correlation diagnostics:")
    from scipy.stats import spearmanr
    for name, path in [("RM6", "oof_realmlp_6seed.parquet"),
                       ("psRM6r2", "oof_realmlp_pseudo62.parquet"),
                       ("CB-exp14", "oof_cb_tuned_exp14.parquet")]:
        p = DATA / path
        if not p.exists():
            continue
        other = pd.read_parquet(p)
        m = pd.DataFrame({"id": train_id, "oof": oof}).merge(
            other[["id", "oof"]].rename(columns={"oof": "other"}), on="id", how="inner"
        )
        rho, _ = spearmanr(m["oof"], m["other"])
        print(f"  ρ vs {name:14s}: {rho:.5f}  (n={len(m):,})")

    pd.DataFrame({"id": train_id, "Year": year_train, "target": y.to_numpy(), "oof": oof}).to_parquet(OOF_OUT, index=False)
    pd.DataFrame({"id": test_id, TARGET: test_preds}).sort_values("id").reset_index(drop=True).to_csv(SUB_OUT, index=False)
    print(f"\nwrote {OOF_OUT.name} and {SUB_OUT.name}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
