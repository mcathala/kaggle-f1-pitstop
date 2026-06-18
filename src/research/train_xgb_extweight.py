"""Experiment 044 (cycle 11) — XGBoost retune for high-cardinality cats.

Exp 034's XGBoost capped at OOF 0.94615 (vs CB14 0.95114) with default HPs.
Cycle-10 probe 1 (cb_feature_importance) ranked Race_Year and Race_Compound_Stint
as top features by mean importance (8.83 and 4.93). These are high-cardinality
cross-cats with ~hundreds of unique values. XGBoost's default `max_bin=256`
histogram can't resolve them — the splitter sees aliased buckets, losing signal.

This retune addresses that directly: `max_bin=5000` gives the histogram enough
resolution for our cross-cats, slower `eta=0.01` + tighter regularization
(lambda=8, alpha=8, colsample=0.15) absorb the added capacity without overfitting.

Outputs:
  data/oof_xgb_highbins.parquet      OOF predictions
  data/submission_xgb_highbins.csv   test predictions
"""

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

DATA = Path(__file__).resolve().parent.parent.parent / "data"
TRAIN_CSV = DATA / "train.csv"
TEST_CSV = DATA / "test.csv"
EXTERNAL_CSV = DATA / "f1_strategy_dataset.csv"
OOF_OUT = DATA / "oof_xgb_extweight.parquet"
SUB_OUT = DATA / "submission_xgb_extweight.csv"
REALMLP_OOF = DATA / "oof_realmlp_multiseed.parquet"
CB_OOF = DATA / "oof_cb_tuned_exp14.parquet"

TARGET = "PitNextLap"
ID_COL = "id"
BASE_CATS = ["Driver", "Race", "Compound"]
N_SPLITS = 5
SEED = 42
EXT_WEIGHT = 0.7   # down-weight distribution-shifted external rows (adv-AUC 0.78; pit-rate 25.5% vs synthetic 19.9%)

# XGBoost HPs retuned to fit our high-cardinality cross-cats (probe-1 finding).
# Key differences vs exp 034's default-ish recipe:
#   max_bin: 256 → 5000           (histogram resolution for Race_Year etc.)
#   lr: 0.05 → 0.01               (slower learning matches the deeper depth)
#   n_estimators: 3000 → 50000    (capacity for slow lr; early-stop=100 will fire)
#   max_depth: 8 → 10             (deeper trees for cat-interaction discovery)
#   lambda: 5.0 → 8                (stronger L2 to absorb added capacity)
#   alpha: 0.5 → 8                 (sparse-feature-friendly L1)
#   colsample_bytree: 0.7 → 0.145 (heavy column subsample regularizes high-bin trees)
#   subsample: 0.8 → 0.86
#   min_child_weight: 5.0 → 2     (allow finer splits at high max_bin)
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "enable_categorical": True,
    "max_bin": 5000,             # ← THE KEY FIX
    "max_depth": 10,
    "eta": 0.01,
    "min_child_weight": 2,
    "subsample": 0.8570122278990485,
    "colsample_bytree": 0.1450999139156032,
    "reg_lambda": 8.162374349037115,
    "reg_alpha": 8.354463958574286,
    "nthread": -1,
    "verbosity": 1,
}
N_ROUNDS = 50000
EARLY_STOP = 100

# Digit / signature feature decomposition was planned but dropped after a
# univariate signal check (see git history). Kept the iter-cap change only.


def safe_div(a, b, eps=1e-6):
    return a / (b + eps)


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cycle-12 inline FE. Same recipes as the 0.95259 public notebook."""
    eps = 1e-6
    out = df.copy()

    # Race progress / lap geometry
    race_progress = out["RaceProgress"].clip(lower=eps)
    out["EstimatedTotalLaps"] = (out["LapNumber"] / race_progress).clip(1, 120).astype("float32")
    out["LapsRemaining"] = (out["EstimatedTotalLaps"] - out["LapNumber"]).clip(lower=0).astype("float32")
    out["RemainingRaceProgress"] = (1.0 - out["RaceProgress"]).astype("float32")
    out["LapProgress_x_LapNumber"] = (out["LapNumber"] * out["RaceProgress"]).astype("float32")

    out["RacePhase"] = pd.cut(
        out["RaceProgress"], bins=[-np.inf, 0.20, 0.40, 0.60, 0.80, np.inf],
        labels=["P1", "P2", "P3", "P4", "P5"],
    ).astype(str)
    out["LapBin"] = pd.cut(
        out["LapNumber"], bins=[-np.inf, 5, 10, 20, 35, 50, np.inf],
        labels=["L005", "L010", "L020", "L035", "L050", "Lplus"],
    ).astype(str)

    # Tyre features
    out["TyreAgeRatio"] = safe_div(out["TyreLife"], out["LapNumber"].clip(lower=1), eps).astype("float32")
    out["LapPerTyreLife"] = safe_div(out["LapNumber"], out["TyreLife"] + 1, eps).astype("float32")
    out["TyreLife_x_RaceProgress"] = (out["TyreLife"] * out["RaceProgress"]).astype("float32")
    out["PitWindowPressure"] = (out["TyreLife"] * out["RaceProgress"]).astype("float32")
    out["TyreAgeVsRace"] = safe_div(out["TyreLife"], out["EstimatedTotalLaps"].clip(lower=1), eps).astype("float32")
    out["TyreLife_to_LapsRemaining"] = safe_div(out["TyreLife"], out["LapsRemaining"] + 1, eps).astype("float32")
    out["LapMinusTyreLife"] = (out["LapNumber"] - out["TyreLife"]).astype("float32")

    out["TyreLifeBin"] = pd.cut(
        out["TyreLife"], bins=[-np.inf, 3, 7, 12, 20, 30, np.inf],
        labels=["T003", "T007", "T012", "T020", "T030", "Tplus"],
    ).astype(str)

    # Stint
    out["StintPressure"] = (out["Stint"] * out["TyreLife"]).astype("float32")
    out["Is_First_Stint"] = (out["Stint"] == 1).astype(np.int8)
    out["Is_Late_Stint"] = (out["Stint"] >= 3).astype(np.int8)

    # Position
    out["PositionBin"] = pd.cut(
        out["Position"], bins=[-np.inf, 3, 8, 14, np.inf],
        labels=["front", "upper_mid", "lower_mid", "back"],
    ).astype(str)
    out["PositionPressure"] = (out["Position"] * out["RaceProgress"]).astype("float32")

    # Degradation
    out["DegPerRaceLap"] = safe_div(out["Cumulative_Degradation"], out["LapNumber"].clip(lower=1), eps).astype("float32")
    out["DegPerTyreLap"] = safe_div(out["Cumulative_Degradation"], out["TyreLife"].clip(lower=1), eps).astype("float32")
    out["Abs_Cumulative_Degradation"] = out["Cumulative_Degradation"].abs().astype("float32")
    out["Positive_Degradation"] = (out["Cumulative_Degradation"] > 0).astype(np.int8)

    # Lap-time delta
    out["DeltaAbs"] = out["LapTime_Delta"].abs().astype("float32")
    out["LapTimeDeltaPositive"] = (out["LapTime_Delta"] > 0).astype(np.int8)
    out["DeltaPerTyreLap"] = safe_div(out["LapTime_Delta"], out["TyreLife"].clip(lower=1), eps).astype("float32")

    # Position change
    out["Abs_Position_Change"] = out["Position_Change"].abs().astype("float32")
    out["Gained_Position"] = (out["Position_Change"] > 0).astype(np.int8)
    out["Lost_Position"] = (out["Position_Change"] < 0).astype(np.int8)

    return out


def add_cross_categoricals(out: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("Race_Year", ["Race", "Year"]),
        ("Compound_Stint", ["Compound", "Stint"]),
        ("Driver_Race", ["Driver", "Race"]),
        ("Driver_Compound", ["Driver", "Compound"]),
        ("Race_Compound", ["Race", "Compound"]),
        ("Race_Compound_Stint", ["Race", "Compound", "Stint"]),
        ("Compound_RacePhase", ["Compound", "RacePhase"]),
        ("Compound_TyreLifeBin", ["Compound", "TyreLifeBin"]),
        ("RacePhase_TyreLifeBin", ["RacePhase", "TyreLifeBin"]),
    ]
    for name, cols in pairs:
        if all(c in out.columns for c in cols):
            value = out[cols[0]].astype(str)
            for col in cols[1:]:
                value = value + "_" + out[col].astype(str)
            out[name] = value
    return out


def add_frequency_features(frames: list[pd.DataFrame], cat_cols: list[str]) -> None:
    """In-place: add `<col>_count` and `<col>_freq` to every frame, computed on the union."""
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


def add_group_stats(frames: list[pd.DataFrame]) -> list[str]:
    """In-place: add mean/std/diff for value_cols by group_cols, computed on union. Returns added columns."""
    group_cols = ["Race_Year", "Race_Compound_Stint", "Driver_Race", "Compound_Stint"]
    value_cols = ["LapTime_Delta", "Position_Change", "RaceProgress", "TyreLife"]
    added: list[str] = []

    combined_pieces = []
    keep_cols = list(set(group_cols + value_cols))
    for f in frames:
        cols_here = [c for c in keep_cols if c in f.columns]
        combined_pieces.append(f[cols_here].copy())
    combined = pd.concat(combined_pieces, axis=0, ignore_index=True)

    for g in group_cols:
        if g not in combined.columns:
            continue
        for v in value_cols:
            if v not in combined.columns:
                continue
            stats = combined.groupby(g, dropna=False)[v].agg(["mean", "std"])
            mean_col = f"{v}_mean_by_{g}"
            std_col = f"{v}_std_by_{g}"
            diff_col = f"{v}_diff_mean_by_{g}"
            for f in frames:
                if g not in f.columns or v not in f.columns:
                    continue
                key = f[g]
                f[mean_col] = key.map(stats["mean"]).astype("float32")
                f[std_col] = key.map(stats["std"]).fillna(0).astype("float32")
                f[diff_col] = (f[v] - f[mean_col]).astype("float32")
            added.extend([mean_col, std_col, diff_col])
    return added


def normalize_cats(out: pd.DataFrame, cat_cols: list[str]) -> None:
    for col in cat_cols:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("__NA__").astype(str)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=-1, help="1..N_SPLITS for one fold; -1 for all")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dry_run = args.fold != -1
    print(f"Args: fold={args.fold}  dry_run={dry_run}")

    print("Loading data...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    ext = pd.read_csv(EXTERNAL_CSV).dropna(subset=["Compound"]).drop(columns=["Normalized_TyreLife"], errors="ignore")
    # External data has no `id`; we'll insert a synthetic id far above the test range.
    ext[ID_COL] = -1  # marker; never used in OOF/sub
    print(f"  train {train.shape}  test {test.shape}  external {ext.shape}")
    print(f"  external pos rate: {ext[TARGET].mean():.4f}  competition train pos rate: {train[TARGET].mean():.4f}")

    print("Applying domain FE...")
    train = add_domain_features(train)
    test = add_domain_features(test)
    ext = add_domain_features(ext)

    print("Adding cross-categoricals...")
    train = add_cross_categoricals(train)
    test = add_cross_categoricals(test)
    ext = add_cross_categoricals(ext)

    cross_cats = [
        "Race_Year", "Compound_Stint", "Driver_Race", "Driver_Compound",
        "Race_Compound", "Race_Compound_Stint", "Compound_RacePhase",
        "Compound_TyreLifeBin", "RacePhase_TyreLifeBin",
    ]
    bins = ["RacePhase", "LapBin", "TyreLifeBin", "PositionBin"]
    all_cats = BASE_CATS + cross_cats + bins

    print(f"Adding frequency features for {len(all_cats)} categorical fields...")
    add_frequency_features([train, test, ext], all_cats)

    print("Adding group-statistic features...")
    group_stat_cols = add_group_stats([train, test, ext])
    print(f"  added {len(group_stat_cols)} group-stat columns")

    # Final cat list (everything string-like, then cast to pandas categorical for XGBoost)
    normalize_cats(train, all_cats)
    normalize_cats(test, all_cats)
    normalize_cats(ext, all_cats)
    # XGBoost with enable_categorical=True requires identical category sets across
    # train/val/test. Build a unified categorical dtype per column from the UNION,
    # then cast all frames to that dtype. Otherwise val/test rows with categories
    # absent from train trigger "Found a category not in the training set" errors.
    for c in all_cats:
        if c not in train.columns:
            continue
        union_vals = (
            pd.concat([train[c], test[c], ext[c]], axis=0)
            .astype("string")
            .fillna("__NA__")
            .unique()
            .tolist()
        )
        cat_dtype = pd.CategoricalDtype(categories=sorted(union_vals))
        for f in (train, test, ext):
            if c in f.columns:
                f[c] = f[c].astype(cat_dtype)

    train["is_original"] = np.int8(0)
    test["is_original"] = np.int8(0)
    ext["is_original"] = np.int8(1)
    feature_cols = [c for c in train.columns if c not in (TARGET, ID_COL)]
    feature_cols = [c for c in feature_cols if c in test.columns and c in ext.columns]
    n_cat = sum(1 for c in feature_cols if c in all_cats)
    print(f"using {len(feature_cols)} features, {n_cat} categorical")

    X = train[feature_cols]
    y = train[TARGET].astype(int).to_numpy()
    X_test = test[feature_cols]
    X_ext = ext[feature_cols]
    y_ext = ext[TARGET].astype(int).to_numpy()

    strat_key = train["Year"].astype(str) + "_" + train[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train), dtype=np.float64)
    test_preds = np.zeros(len(test), dtype=np.float64)
    fold_aucs: list[float] = []
    fold_iters: list[int] = []

    folds_to_run = [args.fold] if dry_run else list(range(1, N_SPLITS + 1))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, strat_key), start=1):
        if fold not in folds_to_run:
            continue
        t0 = time.time()
        X_tr = pd.concat([X.iloc[tr_idx], X_ext], axis=0, ignore_index=True)
        # After concat, each cat column's dtype already has the unified category set
        # (because we set it on the source frames before this loop). No re-cast needed.
        y_tr = np.concatenate([y[tr_idx], y_ext])
        # Down-weight external rows: competition rows weight 1.0, external EXT_WEIGHT.
        w_tr = np.concatenate([np.ones(len(tr_idx), dtype=np.float64),
                               np.full(len(X_ext), EXT_WEIGHT, dtype=np.float64)])
        X_va = X.iloc[va_idx]
        y_va = y[va_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, enable_categorical=True)
        dval = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, enable_categorical=True)

        params = dict(XGB_PARAMS)
        params["seed"] = SEED + fold

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=N_ROUNDS,
            evals=[(dval, "val")],
            early_stopping_rounds=EARLY_STOP,
            verbose_eval=200,
        )

        best_iter = booster.best_iteration
        va_pred = booster.predict(dval, iteration_range=(0, best_iter + 1))
        oof[va_idx] = va_pred
        test_preds += booster.predict(dtest, iteration_range=(0, best_iter + 1)) / N_SPLITS
        fold_auc = roc_auc_score(y_va, va_pred)
        fold_aucs.append(fold_auc)
        fold_iters.append(best_iter + 1)
        print(
            f"fold {fold}/{N_SPLITS}  AUC={fold_auc:.5f}  iters={best_iter + 1}  "
            f"train_rows={len(X_tr):,}  ({time.time()-t0:.1f}s)",
            flush=True,
        )

    if dry_run:
        print(f"\nDRY-RUN fold-{args.fold} AUC = {fold_aucs[0]:.5f}")
        if REALMLP_OOF.exists():
            try:
                rm = pd.read_parquet(REALMLP_OOF)
                rm_map = dict(zip(rm["id"], rm["oof"]))
                va_idx_actual = list(kf.split(X, strat_key))[args.fold - 1][1]
                rm_va = train.iloc[va_idx_actual][ID_COL].map(rm_map).to_numpy()
                if not np.isnan(rm_va).any():
                    rho, _ = spearmanr(oof[va_idx_actual], rm_va)
                    print(f"rank-corr-with-RealMLP-multiseed (fold {args.fold}): {rho:.5f}")
            except Exception as e:
                print(f"rank-corr-with-RealMLP skipped: {e}")
        return

    oof_auc = roc_auc_score(y, oof)
    print()
    print(f"per-fold AUC: mean={np.mean(fold_aucs):.5f}  std={np.std(fold_aucs):.5f}  iters={fold_iters}")
    print(f"OOF AUC:      {oof_auc:.5f}")
    print(f"  (vs CB-tuned-exp14 0.95114, Δ = {oof_auc - 0.95114:+.5f})")

    oof_df = pd.DataFrame(
        {"id": train[ID_COL], "Year": train["Year"], "target": y, "oof": oof}
    )
    print("\nOOF AUC by Year:")
    for year, g in oof_df.groupby("Year"):
        if g["target"].nunique() > 1:
            auc = roc_auc_score(g["target"], g["oof"])
            print(f"  {year}: AUC={auc:.5f}  n={len(g):,}  pos_rate={g['target'].mean():.4f}")

    # Rank-correlation with existing base OOFs — the diversity-for-blend signal.
    print("\nRank-correlation diagnostics:")
    for name, path in [("RealMLP-multiseed", REALMLP_OOF), ("CB-tuned-exp14", CB_OOF)]:
        try:
            other = pd.read_parquet(path)
            m = pd.DataFrame({"id": train[ID_COL], "oof": oof}).merge(
                other[["id", "oof"]].rename(columns={"oof": "other"}), on="id", how="inner"
            )
            rho, _ = spearmanr(m["oof"], m["other"])
            print(f"  rank-corr vs {name:20s}: {rho:.5f}  (n={len(m):,})")
        except Exception as e:
            print(f"  rank-corr vs {name}: skipped ({e})")

    oof_df.assign(fold=-1).to_parquet(OOF_OUT, index=False)
    sub = (
        pd.DataFrame({"id": test[ID_COL], TARGET: test_preds})
        .sort_values("id")
        .reset_index(drop=True)
    )
    sub.to_csv(SUB_OUT, index=False)
    print(f"\nwrote {OOF_OUT.name}  ({len(oof_df):,} rows)")
    print(f"wrote {SUB_OUT.name}  ({len(sub):,} rows)")


if __name__ == "__main__":
    main()
