"""Experiment 075 (cycle 17) — self-distillation MLP on our own leakage-clean blend.

Audit Phase-1 #5. Student RealMLP trained to predict the *teacher's soft OOF*
(continuous in [0,1]), where the teacher is our own leakage-clean blend
(`oof_blend_pseudo_r2_xgb2.parquet`, OOF 0.95430).

Hypothesis: by replacing the noisy hard binary target with the teacher's
smoothed probability, the student inherits the strength but converges to a
slightly different decision boundary — generating ρ-diverse output usable in
a higher-order blend. Audit-locked constraint: teacher is the leakage-clean
blend, not the leaky one.

To force structural diversity beyond just label-smoothing, this run uses a
*subset* of features (excluding the categoricals that dominate the original
RealMLP's encoder) — forcing the student to find a different representation.

Outputs:
  data/oof_realmlp_selfdistill_s42.parquet
  data/submission_realmlp_selfdistill_s42.csv
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
from sklearn.preprocessing import KBinsDiscretizer

warnings.filterwarnings("ignore")

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch {torch.__version__}  device={device}")

DATA = Path(__file__).resolve().parent.parent.parent / "data"
TRAIN_CSV = DATA / "train.csv"
TEST_CSV = DATA / "test.csv"
TEACHER_OOF = DATA / "oof_blend_pseudo_r2_xgb2.parquet"
TEACHER_SUB = DATA / "submission_blend_pseudo_r2_xgb2.csv"
OOF_OUT = DATA / "oof_realmlp_selfdistill_s42.parquet"
SUB_OUT = DATA / "submission_realmlp_selfdistill_s42.csv"

TARGET, ID_COL = "PitNextLap", "id"
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


def feature_engineering(df, fit, state, exclude_cats: bool):
    """Same FE as the pseudo trainers, but optionally skip the high-cardinality
    categoricals to force the student to a different representation."""
    df["_LapNumber_/_RaceProgress"] = (df["LapNumber"] / (df["RaceProgress"] + 1e-6)).astype("float32")
    df["_TyreLife_/_LapNumber"] = (df["TyreLife"] / df["LapNumber"].clip(lower=1)).astype("float32")
    df["_LapTime (s)_*_Cumulative_Degradation"] = (df["LapTime (s)"] * df["Cumulative_Degradation"]).astype("float32")
    df["_LapTime (s)_*_Cumulative_Degradation_abs"] = (df["LapTime (s)"] * df["Cumulative_Degradation"].abs()).astype("float32")
    df["_LapTime (s)_/_Cumulative_Degradation_abs"] = (df["LapTime (s)"] / (df["Cumulative_Degradation"].abs() + 1e-6)).astype("float32")
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in df.select_dtypes(exclude=["object"]).columns.tolist() if c not in (ID_COL, TARGET, "_teacher_soft")]

    if exclude_cats:
        # Drop the high-cardinality categoricals to force the student to find
        # a different representation. Keep low-cardinality ordinals.
        keep_obj = ["Compound"]
        for c in cat_cols:
            if c not in keep_obj:
                df = df.drop(columns=[c])
        cat_cols = keep_obj

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
    return df


def maybe_flip_pred(pred: np.ndarray, y_binary: np.ndarray) -> tuple[np.ndarray, bool]:
    auc = roc_auc_score(y_binary, pred)
    return (1.0 - pred, True) if auc < 0.5 else (pred, False)


def main():
    t0 = time.time()
    print(f"\nLoading data + teacher from {DATA}...")
    train = pd.read_csv(TRAIN_CSV).sort_values(ID_COL).reset_index(drop=True)
    test = pd.read_csv(TEST_CSV).sort_values(ID_COL).reset_index(drop=True)
    teacher_oof = pd.read_parquet(TEACHER_OOF).sort_values(ID_COL).reset_index(drop=True)
    teacher_sub = pd.read_csv(TEACHER_SUB).sort_values(ID_COL).reset_index(drop=True)
    assert (teacher_oof[ID_COL].to_numpy() == train[ID_COL].to_numpy()).all()
    assert (teacher_sub[ID_COL].to_numpy() == test[ID_COL].to_numpy()).all()
    train["_teacher_soft"] = teacher_oof["oof"].astype("float32").to_numpy()
    print(f"  train {train.shape}  test {test.shape}")
    print(f"  teacher OOF AUC = {roc_auc_score(train[TARGET].astype(int), train['_teacher_soft']):.5f}")

    y_binary = train[TARGET].astype(int).to_numpy()
    y_soft = train["_teacher_soft"].astype("float32").to_numpy()
    train_id, test_id = train[ID_COL], test[ID_COL]
    year_train = train["Year"].to_numpy()

    X = train.drop([ID_COL, TARGET, "_teacher_soft"], axis=1, errors="ignore")
    Xt = test.drop([ID_COL], axis=1, errors="ignore")

    state = {}
    EXCLUDE_CATS = True   # diversity-forcing: skip high-cardinality cats (Driver, Race)
    X = feature_engineering(X, True, state, EXCLUDE_CATS)
    Xt = feature_engineering(Xt, False, state, EXCLUDE_CATS)
    print(f"FE done (exclude_cats={EXCLUDE_CATS}). X {X.shape}  Xt {Xt.shape}")

    strat = train["Year"].astype(str) + "_" + pd.Series(y_binary).astype(str)
    kf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(kf.split(X, strat))

    oof = np.zeros(len(train)); test_preds = np.zeros(len(test))
    for seed in SEEDS:
        params = dict(REALMLP_PARAMS); params["random_state"] = seed
        oof_s = np.zeros(len(train)); test_s_pred = np.zeros(len(test))
        for fi, (tr, va) in enumerate(splits, 1):
            t1 = time.time()
            X_tr = X.iloc[tr]; y_tr = y_soft[tr]
            X_va = X.iloc[va]; y_va = y_soft[va]
            m = RealMLP_TD_Regressor(**params); m.fit(X_tr, y_tr, X_va, y_va)
            oof_s[va] = m.predict(X_va).ravel()
            test_s_pred += m.predict(Xt).ravel() / N_SPLITS
            auc_raw = roc_auc_score(y_binary[va], oof_s[va])
            print(f"seed {seed} fold {fi}/{N_SPLITS} "
                  f"rmse={np.sqrt(np.mean((oof_s[va]-y_va)**2)):.4f} "
                  f"AUC(PitNextLap, pred)={auc_raw:.5f}  "
                  f"({time.time()-t1:.0f}s)", flush=True)
        oof += oof_s / len(SEEDS); test_preds += test_s_pred / len(SEEDS)

    oof_final, flipped = maybe_flip_pred(oof, y_binary)
    test_preds_final = 1.0 - test_preds if flipped else test_preds
    oa = roc_auc_score(y_binary, oof_final)
    print(f"\n{len(SEEDS)}-seed OOF AUC vs PitNextLap: {oa:.5f}  (flip={'1-pred' if flipped else 'raw'})")
    print(f"  teacher  OOF 0.95430  (Δ student − teacher = {oa - 0.95430:+.5f})")
    print(f"  vs RM6   0.95386  (Δ = {oa - 0.95386:+.5f})")

    print("\nrank-correlation diagnostics (full OOF, post-flip):")
    from scipy.stats import spearmanr
    for name, path in [("RM6", "oof_realmlp_6seed.parquet"),
                       ("psRM6r2", "oof_realmlp_pseudo62.parquet"),
                       ("CB-exp14", "oof_cb_tuned_exp14.parquet"),
                       ("XGB-highbins", "oof_xgb_highbins.parquet"),
                       ("teacher_blend", "oof_blend_pseudo_r2_xgb2.parquet")]:
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
