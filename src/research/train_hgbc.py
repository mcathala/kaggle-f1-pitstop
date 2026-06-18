"""Experiment 070 (cycle 17) — sklearn HistGradientBoostingClassifier as a 4th GBDT base.

Untried GBDT family in this project. LightGBM/XGBoost/CatBoost are all in the
zoo; sklearn HGBC has the same histogram-binning backbone as LightGBM but with
sklearn's own implementation (leaf-wise depth-first, OpenMP-parallel). Even if
correlated, it's a cheap local-CPU probe and could provide rank-diversity.

Recipe: identical FE pipeline to `train_xgb.py` (132 features, ~16+ categoricals).
HGBC receives categoricals as pandas `category` dtype via `categorical_features='from_dtype'`.

Outputs:
  data/oof_hgbc.parquet      OOF predictions
  data/submission_hgbc.csv   test predictions
"""

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA = Path(__file__).resolve().parent.parent.parent / "data"
TRAIN_CSV = DATA / "train.csv"
TEST_CSV = DATA / "test.csv"
EXTERNAL_CSV = DATA / "f1_strategy_dataset.csv"
OOF_OUT = DATA / "oof_hgbc.parquet"
SUB_OUT = DATA / "submission_hgbc.csv"
REALMLP_OOF = DATA / "oof_realmlp_6seed.parquet"
CB_OOF = DATA / "oof_cb_tuned_exp14.parquet"
XGB_OOF = DATA / "oof_xgb.parquet"

TARGET = "PitNextLap"
ID_COL = "id"
BASE_CATS = ["Driver", "Race", "Compound"]
N_SPLITS = 5
SEED = 42

# HGBC HPs — match XGB-highbins depth/lr; max_bins capped at 255 (sklearn limit).
HGBC_PARAMS = dict(
    loss="log_loss",
    learning_rate=0.05,
    max_iter=2000,
    max_depth=8,
    l2_regularization=5.0,
    min_samples_leaf=20,
    max_bins=255,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=30,
    tol=1e-5,
    categorical_features="from_dtype",
    random_state=SEED,
    verbose=0,
)


def safe_div(a, b, eps=1e-6):
    return a / (b + eps)


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6
    out = df.copy()

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

    out["StintPressure"] = (out["Stint"] * out["TyreLife"]).astype("float32")
    out["Is_First_Stint"] = (out["Stint"] == 1).astype(np.int8)
    out["Is_Late_Stint"] = (out["Stint"] >= 3).astype(np.int8)

    out["PositionBin"] = pd.cut(
        out["Position"], bins=[-np.inf, 3, 8, 14, np.inf],
        labels=["front", "upper_mid", "lower_mid", "back"],
    ).astype(str)
    out["PositionPressure"] = (out["Position"] * out["RaceProgress"]).astype("float32")

    out["DegPerRaceLap"] = safe_div(out["Cumulative_Degradation"], out["LapNumber"].clip(lower=1), eps).astype("float32")
    out["DegPerTyreLap"] = safe_div(out["Cumulative_Degradation"], out["TyreLife"].clip(lower=1), eps).astype("float32")
    out["Abs_Cumulative_Degradation"] = out["Cumulative_Degradation"].abs().astype("float32")
    out["Positive_Degradation"] = (out["Cumulative_Degradation"] > 0).astype(np.int8)

    out["DeltaAbs"] = out["LapTime_Delta"].abs().astype("float32")
    out["LapTimeDeltaPositive"] = (out["LapTime_Delta"] > 0).astype(np.int8)
    out["DeltaPerTyreLap"] = safe_div(out["LapTime_Delta"], out["TyreLife"].clip(lower=1), eps).astype("float32")

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


def to_categories(frames: list[pd.DataFrame], cat_cols: list[str], max_cardinality: int = 254) -> dict[str, int]:
    """Cast each cat column to a unified pd.CategoricalDtype across frames.

    HGBC requires cardinality <= max_bins (default 255). For columns over the cap,
    keep top-(max_cardinality-1) by union count and bucket the rest into __OTHER__.
    Returns {col: cardinality} for diagnostics.
    """
    cards: dict[str, int] = {}
    for c in cat_cols:
        if not all(c in f.columns for f in frames):
            continue
        union = pd.concat([f[c].astype("string").fillna("__NA__") for f in frames], axis=0)
        counts = union.value_counts(dropna=False)
        if len(counts) > max_cardinality:
            keep = set(counts.head(max_cardinality - 1).index)
            for f in frames:
                vals = f[c].astype("string").fillna("__NA__")
                f[c] = vals.where(vals.isin(keep), "__OTHER__")
            union_vals = sorted(keep) + ["__OTHER__"]
        else:
            union_vals = sorted(counts.index.tolist())
        cat_dtype = pd.CategoricalDtype(categories=union_vals)
        for f in frames:
            f[c] = f[c].astype("string").fillna("__NA__").astype(cat_dtype)
        cards[c] = len(union_vals)
    return cards


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
    ext[ID_COL] = -1
    print(f"  train {train.shape}  test {test.shape}  external {ext.shape}")

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

    print(f"Adding frequency features for {len(all_cats)} cats...")
    add_frequency_features([train, test, ext], all_cats)

    print("Adding group stats...")
    group_stat_cols = add_group_stats([train, test, ext])
    print(f"  added {len(group_stat_cols)} group-stat columns")

    print("Casting cats to pd.CategoricalDtype (bucketing >254 into __OTHER__)...")
    cards = to_categories([train, test, ext], all_cats, max_cardinality=254)
    over = [(c, cards[c]) for c in all_cats if c in cards and cards[c] >= 254]
    print(f"  cardinalities: {[(c, cards[c]) for c in all_cats if c in cards]}")
    if over:
        print(f"  bucketed (>=254): {over}")

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
        y_tr = np.concatenate([y[tr_idx], y_ext])
        X_va = X.iloc[va_idx]
        y_va = y[va_idx]

        params = dict(HGBC_PARAMS)
        params["random_state"] = SEED + fold

        model = HistGradientBoostingClassifier(**params)
        model.fit(X_tr, y_tr)

        va_pred = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = va_pred
        test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
        fold_auc = roc_auc_score(y_va, va_pred)
        fold_aucs.append(fold_auc)
        fold_iters.append(int(model.n_iter_))
        print(
            f"fold {fold}/{N_SPLITS}  AUC={fold_auc:.5f}  iters={model.n_iter_}  "
            f"train_rows={len(X_tr):,}  ({time.time()-t0:.1f}s)",
            flush=True,
        )

    if dry_run:
        print(f"\nDRY-RUN fold-{args.fold} AUC = {fold_aucs[0]:.5f}")
        for name, path in [("RealMLP-6seed", REALMLP_OOF), ("CB-tuned-exp14", CB_OOF), ("XGB-highbins", XGB_OOF)]:
            if not path.exists():
                continue
            try:
                other = pd.read_parquet(path)
                other_map = dict(zip(other["id"], other["oof"]))
                va_idx_actual = list(kf.split(X, strat_key))[args.fold - 1][1]
                other_va = train.iloc[va_idx_actual][ID_COL].map(other_map).to_numpy()
                if not np.isnan(other_va).any():
                    rho, _ = spearmanr(oof[va_idx_actual], other_va)
                    print(f"  rank-corr vs {name:18s}: {rho:.5f}")
            except Exception as e:
                print(f"  rank-corr vs {name}: skipped ({e})")
        return

    oof_auc = roc_auc_score(y, oof)
    print()
    print(f"per-fold AUC: mean={np.mean(fold_aucs):.5f}  std={np.std(fold_aucs):.5f}  iters={fold_iters}")
    print(f"OOF AUC:      {oof_auc:.5f}")
    print(f"  (vs XGB-highbins 0.95254, Δ = {oof_auc - 0.95254:+.5f})")
    print(f"  (vs CB-tuned-exp14 0.95114, Δ = {oof_auc - 0.95114:+.5f})")

    oof_df = pd.DataFrame(
        {"id": train[ID_COL], "Year": train["Year"], "target": y, "oof": oof}
    )
    print("\nOOF AUC by Year:")
    for year, g in oof_df.groupby("Year"):
        if g["target"].nunique() > 1:
            auc = roc_auc_score(g["target"], g["oof"])
            print(f"  {year}: AUC={auc:.5f}  n={len(g):,}  pos_rate={g['target'].mean():.4f}")

    print("\nRank-correlation diagnostics:")
    for name, path in [("RealMLP-6seed", REALMLP_OOF), ("CB-tuned-exp14", CB_OOF), ("XGB-highbins", XGB_OOF)]:
        try:
            other = pd.read_parquet(path)
            m = pd.DataFrame({"id": train[ID_COL], "oof": oof}).merge(
                other[["id", "oof"]].rename(columns={"oof": "other"}), on="id", how="inner"
            )
            rho, _ = spearmanr(m["oof"], m["other"])
            print(f"  rank-corr vs {name:18s}: {rho:.5f}  (n={len(m):,})")
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
