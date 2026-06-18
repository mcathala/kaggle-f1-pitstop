"""Experiment 021 (cycle 7) — RealMLP HP search via single-fold Optuna proxy.

Sweeps RealMLP's capacity / regularization HPs on fold 1 only:
  - n_ens          {12, 24, 36, 48}
  - hidden_sizes   one of 4 width-depth recipes
  - embedding_size {4, 6, 8, 12}
  - p_drop         {0.0, 0.05, 0.1, 0.15, 0.2}
  - lr             log-uniform [3e-3, 3e-2]

Feature set is FROZEN at cycle 5's pipeline — cycle 6 closed the
"add features" direction (exp 18/19) and the "swap architecture"
direction (exp 20), so the only remaining ROI is HP perturbation
around the existing pre-tuned defaults.

Outputs:
  data/realmlp_optuna_trials.parquet  — full trial-by-trial log
  data/realmlp_optuna_top3.json        — top-3 configs to validate in exp 022
"""

import json
import random
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.INFO)

DATA = Path(__file__).resolve().parent.parent.parent / "data"
TRAIN_CSV = DATA / "train.csv"
TEST_CSV = DATA / "test.csv"
EXTERNAL_CSV = DATA / "f1_strategy_dataset.csv"
TRIALS_OUT = DATA / "realmlp_optuna_trials.parquet"
TOP3_OUT = DATA / "realmlp_optuna_top3.json"

TARGET = "PitNextLap"
ID_COL = "id"
N_SPLITS = 5
SEED = 42
N_TRIALS = 20
FOLD_PROXY = 1  # 1-indexed; this is the proxy fold used for ranking

ARCH_CHOICES = {
    0: [256, 128, 64],
    1: [512, 256, 128],   # default
    2: [768, 384, 192],
    3: [1024, 512, 256],
}

print(f"torch     version: {torch.__version__}")
print(f"pytabkit  version: {version('pytabkit')}")
print(f"optuna    version: {version('optuna')}")
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"device:            {device}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def feature_engineering(df: pd.DataFrame, fit: bool, state: dict):
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
            codes, uniques = np.floor(df[col]).astype(int).factorize()
            state[col] = uniques
        else:
            uniques = state[col]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = np.floor(df[col]).astype(int).map(code_map).fillna(-1).astype("int32")
        df[cat_name] = codes.astype(str)

    for col in cat_cols + ["Year_cat_", "PitStop_cat_"]:
        count_name = f"_{col}_count" if col in cat_cols else f"_{col[:-1]}_count"
        if fit:
            count_map = df[col].astype(object).value_counts()
            state[count_name] = count_map
        else:
            count_map = state[count_name]
        df[count_name] = df[col].astype(object).map(count_map).fillna(0).astype("int32")

    bin_config = {"RaceProgress": [200], "LapTime (s)": [7]}
    for col, bins_list in bin_config.items():
        for n_bins in bins_list:
            bin_name = f"{col}_{n_bins}_quantile_bin_"
            if fit:
                kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
                binned = kb.fit_transform(df[[col]]).ravel().astype("int32")
                state[bin_name] = kb
            else:
                kb = state[bin_name]
                binned = kb.transform(df[[col]]).ravel().astype("int32")
            df[bin_name] = binned.astype(str)

    combo_names: list[str] = []
    for cols in [("Race", "Compound"), ("Race", "Year")]:
        combo_name = "_".join(cols) + "_"
        combo_names.append(combo_name)
        combo_series = df[cols[0]].astype(str)
        for col in cols[1:]:
            combo_series = combo_series + "_" + df[col].astype(str)
        if fit:
            codes, uniques = pd.factorize(combo_series, sort=False)
            state[combo_name] = uniques
        else:
            uniques = state[combo_name]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = combo_series.map(code_map).fillna(-1).astype("int32")
        df[combo_name] = codes.astype(str)

    return df, combo_names


def prepare_fold(fold_idx: int):
    """One-shot data prep: read CSVs, FE, build the proxy fold's train/val splits."""
    print(f"Loading data + FE for proxy fold {fold_idx}...")
    train = pd.read_csv(TRAIN_CSV)
    orig = pd.read_csv(EXTERNAL_CSV).drop(columns=["Normalized_TyreLife"], errors="ignore")
    print(f"  train {train.shape}  orig {orig.shape}")

    y = train[TARGET].astype(int)
    y_orig = orig[TARGET].astype(int)
    X = train.drop([ID_COL, TARGET], axis=1)
    X_orig = orig.drop([TARGET], axis=1)

    state: dict = {}
    X, combo_names = feature_engineering(X, fit=True, state=state)
    X_orig, _ = feature_engineering(X_orig, fit=False, state=state)
    print(f"  X      shape: {X.shape}")

    strat_key = train["Year"].astype(str) + "_" + train[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    kf_orig = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(kf.split(X, strat_key))
    orig_splits = list(kf_orig.split(X_orig, y_orig))

    tr_idx, va_idx = splits[fold_idx - 1]
    or_tr_idx, _ = orig_splits[fold_idx - 1]

    X_tr = pd.concat([X.iloc[tr_idx], X_orig.iloc[or_tr_idx]], axis=0).reset_index(drop=True)
    y_tr = pd.concat([y.iloc[tr_idx], y_orig.iloc[or_tr_idx]], axis=0).reset_index(drop=True)
    X_va = X.iloc[va_idx].copy()
    y_va = y.iloc[va_idx]

    te = TargetEncoder(cv=N_SPLITS, smooth="auto", shuffle=True, random_state=SEED)
    tr_enc = te.fit_transform(X_tr[combo_names], y_tr)
    va_enc = te.transform(X_va[combo_names])
    te_names = [f"_{c}TE" for c in combo_names]
    X_tr[te_names] = tr_enc
    X_va[te_names] = va_enc

    print(f"  fold {fold_idx}: train rows = {len(X_tr):,}  val rows = {len(X_va):,}  features = {X_tr.shape[1]}")
    return X_tr, y_tr, X_va, y_va


# Pre-tuned default HPs (everything we hold fixed; the search space overrides the 5 swept knobs).
BASE_PARAMS = {
    "random_state": SEED,
    "verbosity": 0,
    "val_metric_name": "1-auc_ovr",
    "n_epochs": 6,
    "batch_size": 256,
    "use_early_stopping": False,
    "wd": 0.016,
    "sq_mom": 0.99,
    "lr_sched": "lin_cos_log_15",
    "first_layer_lr_factor": 0.25,
    "max_one_hot_cat_size": 18,
    "act": "silu",
    "p_drop_sched": "invsqrtp1e-3",
    "plr_hidden_1": 16,
    "plr_hidden_2": 8,
    "plr_act_name": "gelu",
    "plr_lr_factor": 0.1151,
    "plr_sigma": 2.33,
    "ls_eps": 0.01,
    "ls_eps_sched": "sqrt_cos",
    "add_front_scale": False,
    "bias_init_mode": "neg-uniform-dynamic-2",
    "tfms": ["one_hot", "median_center", "robust_scale",
             "smooth_clip", "embedding", "l2_normalize"],
    "device": device,
}


def build_objective(X_tr, y_tr, X_va, y_va):
    def objective(trial: optuna.Trial) -> float:
        t0 = time.time()
        n_ens = trial.suggest_categorical("n_ens", [12, 24, 36, 48])
        arch_idx = trial.suggest_categorical("arch_idx", [0, 1, 2, 3])
        embedding_size = trial.suggest_categorical("embedding_size", [4, 6, 8, 12])
        p_drop = trial.suggest_categorical("p_drop", [0.0, 0.05, 0.1, 0.15, 0.2])
        lr = trial.suggest_float("lr", 3e-3, 3e-2, log=True)

        hidden_sizes = ARCH_CHOICES[arch_idx]

        params = dict(BASE_PARAMS)
        params.update({
            "n_ens": n_ens,
            "hidden_sizes": hidden_sizes,
            "embedding_size": embedding_size,
            "p_drop": p_drop,
            "lr": lr,
        })

        seed_everything(SEED)
        model = RealMLP_TD_Classifier(**params)
        model.fit(X_tr, y_tr, X_va, y_va)
        va_pred = model.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, va_pred)

        elapsed = time.time() - t0
        print(
            f"trial {trial.number:02d}  n_ens={n_ens:>2}  arch={arch_idx}  emb={embedding_size:>2}  "
            f"p_drop={p_drop:.2f}  lr={lr:.4f}  AUC={auc:.5f}  ({elapsed:.0f}s)",
            flush=True,
        )

        if device == "mps":
            torch.mps.empty_cache()

        trial.set_user_attr("elapsed_s", elapsed)
        trial.set_user_attr("hidden_sizes", str(hidden_sizes))
        return auc

    return objective


def main() -> None:
    seed_everything(SEED)
    X_tr, y_tr, X_va, y_va = prepare_fold(FOLD_PROXY)

    print(f"\nStarting Optuna study — {N_TRIALS} trials, fold {FOLD_PROXY} proxy.")
    print(f"Default-config single-fold reference (cycle 4 RealMLP fold 1) = 0.95421")
    print(f"Hurdle: any trial > 0.95421 + 0.0005 = 0.95471 worth full-5-fold validation.\n")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=6),
        study_name="realmlp_hp_cycle7",
    )
    study.optimize(build_objective(X_tr, y_tr, X_va, y_va), n_trials=N_TRIALS)

    # Persist all trials
    rows = []
    for t in study.trials:
        rows.append({
            "trial": t.number,
            "auc": t.value if t.value is not None else float("nan"),
            "n_ens": t.params.get("n_ens"),
            "arch_idx": t.params.get("arch_idx"),
            "hidden_sizes": t.user_attrs.get("hidden_sizes"),
            "embedding_size": t.params.get("embedding_size"),
            "p_drop": t.params.get("p_drop"),
            "lr": t.params.get("lr"),
            "elapsed_s": t.user_attrs.get("elapsed_s"),
            "state": t.state.name,
        })
    trials_df = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    trials_df.to_parquet(TRIALS_OUT, index=False)
    print(f"\nWrote {TRIALS_OUT.name}  ({len(trials_df)} trials)")

    # Top-3 export
    top3 = trials_df.head(3).to_dict(orient="records")
    with open(TOP3_OUT, "w") as f:
        json.dump({
            "study_name": "realmlp_hp_cycle7",
            "n_trials": N_TRIALS,
            "fold_proxy": FOLD_PROXY,
            "default_fold1_baseline": 0.95421,
            "top3": top3,
        }, f, indent=2, default=str)
    print(f"Wrote {TOP3_OUT.name}")

    print("\n=== Top 5 ===")
    print(trials_df.head(5).to_string(index=False))

    best = trials_df.iloc[0]
    delta = best["auc"] - 0.95421
    print(f"\nBest trial #{int(best['trial'])}: AUC = {best['auc']:.5f}  (Δ vs default = {delta:+.5f})")
    if delta >= 0.0005:
        print(f"PASS — proceed to exp 022 (full-5-fold validation of top-3 configs).")
    elif delta >= 0.0002:
        print(f"WEAK PASS — Inconclusive at the single-fold level; consider exp 022 with caveats.")
    else:
        print(f"FAIL — HP space is effectively flat. Direction dead; consider AutoGluon or pseudo-labeling for cycle 8.")


if __name__ == "__main__":
    main()
