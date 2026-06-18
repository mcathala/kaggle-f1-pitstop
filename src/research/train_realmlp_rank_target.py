"""Experiment 073 (cycle 17) — rank-target RealMLP on per-race rank_pct of laps_to_next_PitStop.

Audit Phase-1 #2 (post-rerank). Training-objective diversity probe: instead of
predicting the binary (noisy) PitNextLap, predict the per-(Year, Race) rank
percentile of `laps_to_next_PitStop` derived from the PitStop timeline. Goal:
generate a strong base whose ρ vs RealMLP-6seed is in the strong+diverse
quadrant we never reached via architecture changes.

Recipe:
  - Target: `rank_pct` from `data/rank_target_train.parquet` (per-(Year,Race),
    censored rows at `race_max_lap − LapNumber + 1`, indicator in `censored`).
  - Model: RealMLP_TD_Regressor with MSE on rank_pct.
  - FE pipeline: identical to `train_realmlp_pseudo_r3.py` (FE + Race_Compound /
    Race_Year TE in-fold).
  - No external (different per-group density would distort rank_pct).
  - No pseudo (this run is a clean novel-target probe; can be combined later).
  - Single-seed feasibility (seed 42). Escalate to 3-seed (42/7/99) if Gate A.

Gate A (5-fold OOF AUC vs PitNextLap):
  - ≥ 0.949 AND ρ vs RM6 ≤ 0.95 → strong+diverse → escalate to 3-seed.
  - 0.94 ≤ AUC < 0.949 → stays "gated on self-distill" for DAE descope.
  - < 0.94 → close lever; partially falsifies "non-PitNextLap objective" axis.

Outputs:
  data/oof_realmlp_rank_target_s42.parquet
  data/submission_realmlp_rank_target_s42.csv
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytabkit import RealMLP_TD_Regressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

warnings.filterwarnings("ignore")

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch {torch.__version__}  device={device}")

DATA = Path(__file__).resolve().parent.parent.parent / "data"
TRAIN_CSV = DATA / "train.csv"
TEST_CSV = DATA / "test.csv"
RANK_TRAIN = DATA / "rank_target_train.parquet"
RANK_TEST = DATA / "rank_target_test.parquet"
OOF_OUT = DATA / "oof_realmlp_rank_target_s42.parquet"
SUB_OUT = DATA / "submission_realmlp_rank_target_s42.csv"

TARGET = "PitNextLap"  # evaluation target (binary)
RANK_COL = "rank_pct"  # training target (continuous in [0, 1])
ID_COL = "id"
N_SPLITS, SEED = 5, 42
SEEDS = [42]

REALMLP_PARAMS = {
    "random_state": SEED, "verbosity": 1, "val_metric_name": "rmse",
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
    num_cols = [c for c in df.select_dtypes(exclude=["object"]).columns.tolist() if c not in (ID_COL, TARGET, RANK_COL, "censored", "laps_to_next_pit")]
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


def maybe_flip_pred(pred: np.ndarray, y_binary: np.ndarray) -> tuple[np.ndarray, bool]:
    """If AUC of pred < 0.5, return 1-pred (signal is reversed). Returns (pred, flipped)."""
    auc = roc_auc_score(y_binary, pred)
    if auc < 0.5:
        return 1.0 - pred, True
    return pred, False


def main():
    t0 = time.time()
    print(f"\nLoading data from {DATA}...")
    train = pd.read_csv(TRAIN_CSV); test = pd.read_csv(TEST_CSV)
    print(f"  train {train.shape}  test {test.shape}")

    print("Loading rank-target labels...")
    rt_train = pd.read_parquet(RANK_TRAIN).sort_values(ID_COL).reset_index(drop=True)
    rt_test = pd.read_parquet(RANK_TEST).sort_values(ID_COL).reset_index(drop=True)
    train_s = train.sort_values(ID_COL).reset_index(drop=True)
    test_s = test.sort_values(ID_COL).reset_index(drop=True)
    assert (rt_train[ID_COL].to_numpy() == train_s[ID_COL].to_numpy()).all()
    assert (rt_test[ID_COL].to_numpy() == test_s[ID_COL].to_numpy()).all()
    train = train_s.merge(rt_train[[ID_COL, RANK_COL, "censored"]], on=ID_COL, how="left")
    test = test_s.merge(rt_test[[ID_COL, RANK_COL, "censored"]], on=ID_COL, how="left")
    print(f"  rank_pct(train): mean={train[RANK_COL].mean():.4f}, std={train[RANK_COL].std():.4f}")
    print(f"  censored(train): {train['censored'].mean()*100:.2f}%")

    y_binary = train[TARGET].astype(int).to_numpy()
    y_rank = train[RANK_COL].astype("float32").to_numpy()
    train_id, test_id = train[ID_COL], test[ID_COL]
    year_train = train["Year"].to_numpy()

    X = train.drop([ID_COL, TARGET, RANK_COL, "censored", "laps_to_next_pit"], axis=1, errors="ignore")
    Xt = test.drop([ID_COL, RANK_COL, "censored", "laps_to_next_pit"], axis=1, errors="ignore")

    state = {}
    X, combo = feature_engineering(X, True, state)
    Xt, _ = feature_engineering(Xt, False, state)
    print(f"FE done. X {X.shape}  Xt {Xt.shape}")

    strat = train["Year"].astype(str) + "_" + pd.Series(y_binary).astype(str)
    kf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(kf.split(X, strat))

    fold_data = []
    for tr, va in splits:
        X_tr = X.iloc[tr].copy(); y_tr = y_rank[tr]
        X_va = X.iloc[va].copy(); y_va = y_rank[va]
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
            m = RealMLP_TD_Regressor(**params); m.fit(X_tr, y_tr, X_va, y_va)
            oof_s[va] = m.predict(X_va).ravel()
            test_s_pred += m.predict(X_ts).ravel() / N_SPLITS
            auc_fold, flipped = maybe_flip_pred(oof_s[va], y_binary[va])
            # Use raw direction (no flip) for the per-fold print; AUC is recomputed on raw
            auc_raw = roc_auc_score(y_binary[va], oof_s[va])
            print(f"seed {seed} fold {fi}/{N_SPLITS} rmse={np.sqrt(np.mean((oof_s[va]-y_va)**2)):.4f} "
                  f"AUC(PitNextLap, pred)={auc_raw:.5f} flip={'1-pred' if flipped else 'raw'} "
                  f"({time.time()-t1:.0f}s)", flush=True)
        oof += oof_s / len(SEEDS); test_preds += test_s_pred / len(SEEDS)

    # Pick sign once on full OOF
    oof_final, flipped = maybe_flip_pred(oof, y_binary)
    test_preds_final = 1.0 - test_preds if flipped else test_preds
    oa = roc_auc_score(y_binary, oof_final)
    print(f"\n{len(SEEDS)}-seed OOF AUC vs PitNextLap: {oa:.5f}  (flip={'1-pred' if flipped else 'raw'})")
    print(f"  (vs RM6 0.95383, Δ {oa - 0.95383:+.5f})")

    # ρ vs existing bases
    print("\nrank-correlation diagnostics (full OOF, post-flip):")
    from scipy.stats import spearmanr
    for name, path in [("RM6", "oof_realmlp_6seed.parquet"),
                       ("psRM6r2", "oof_realmlp_pseudo62.parquet"),
                       ("CB-exp14", "oof_cb_tuned_exp14.parquet"),
                       ("XGB-highbins", "oof_xgb_highbins.parquet")]:
        p = DATA / path
        if not p.exists():
            continue
        other = pd.read_parquet(p)
        m = pd.DataFrame({"id": train_id, "oof": oof_final}).merge(
            other[["id", "oof"]].rename(columns={"oof": "other"}), on="id", how="inner"
        )
        rho, _ = spearmanr(m["oof"], m["other"])
        print(f"  ρ vs {name:14s}: {rho:.5f}  (n={len(m):,})")

    pd.DataFrame({"id": train_id, "Year": year_train, "target": y_binary, "oof": oof_final}).to_parquet(OOF_OUT, index=False)
    pd.DataFrame({"id": test_id, TARGET: test_preds_final}).sort_values("id").reset_index(drop=True).to_csv(SUB_OUT, index=False)
    print(f"\nwrote {OOF_OUT.name} and {SUB_OUT.name}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
