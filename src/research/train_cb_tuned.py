"""Cycle #012 — CatBoost with public-notebook techniques.

Cycle 11 mapped the +0.00655 LB gap to 5 named components. This cycle attacks
3 of them in one pass:
  1. External dataset augmentation (data/f1_strategy_dataset.csv)
  2. Tuned CatBoost HPs (lr 0.018, l2 8.5, balanced class weights, bayesian
     bootstrap with bagging temp 0.45, random_strength 0.65, per-fold seed)
  3. Cross-cat + frequency + group-statistic features (inline FE here so we
     don't disturb the existing features.py / training scripts)

Deferred to a future cycle: digit/signature exploit features (component 4),
RealMLP NN (component 5).

CV protocol: 5-fold StratifiedKFold on Year × PitNextLap, seed 42 — identical
to all prior cycles so results are directly comparable. Each fold's training
set = (4/5 competition train) + (all external data). The validation set is
competition-only.

GPU is not available locally, so we cap iterations at 5000 with
early_stopping_rounds=300 instead of the public notebook's 11000/500. The
0.95259 notebook reports best_iter around 3000-5000 at lr 0.018, so we should
not be undertraining.

Outputs:
  data/oof_cb_tuned.parquet      OOF predictions (id, Year, target, oof)
  data/submission_cb_tuned.csv   id, PitNextLap (mean of fold test predictions)
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA = Path(__file__).resolve().parent.parent.parent / "data"
TRAIN_CSV = DATA / "train.csv"
TEST_CSV = DATA / "test.csv"
EXTERNAL_CSV = DATA / "f1_strategy_dataset.csv"
OOF_OUT = DATA / "oof_cb_tuned.parquet"
SUB_OUT = DATA / "submission_cb_tuned.csv"

TARGET = "PitNextLap"
ID_COL = "id"
BASE_CATS = ["Driver", "Race", "Compound"]
N_SPLITS = 5
SEED = 42

# Public-notebook HPs, GPU swapped to CPU + tighter early-stopping for wall-clock
CB_PARAMS = {
    "iterations": 5000,
    "learning_rate": 0.018,
    "depth": 8,
    "l2_leaf_reg": 8.5,
    "random_strength": 0.65,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 0.45,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
    "early_stopping_rounds": 300,
    "task_type": "CPU",
    "thread_count": -1,
    "allow_writing_files": False,
    "verbose": 200,
}


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


def main() -> None:
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

    # Final cat list (everything string-like)
    normalize_cats(train, all_cats)
    normalize_cats(test, all_cats)
    normalize_cats(ext, all_cats)

    # Feature columns = everything except id, target
    feature_cols = [c for c in train.columns if c not in (TARGET, ID_COL)]
    # Ensure test has same columns; align
    feature_cols = [c for c in feature_cols if c in test.columns and c in ext.columns]
    cat_indices = [feature_cols.index(c) for c in all_cats if c in feature_cols]
    print(f"using {len(feature_cols)} features, {len(cat_indices)} categorical")

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

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, strat_key), start=1):
        t0 = time.time()
        # Augment training with external data (val stays competition-only)
        X_tr = pd.concat([X.iloc[tr_idx], X_ext], axis=0, ignore_index=True)
        y_tr = np.concatenate([y[tr_idx], y_ext])
        X_va = X.iloc[va_idx]
        y_va = y[va_idx]

        params = dict(CB_PARAMS)
        params["random_seed"] = SEED + fold  # per-fold seed variation, mirroring public notebook
        train_pool = Pool(X_tr, label=y_tr, cat_features=cat_indices)
        valid_pool = Pool(X_va, label=y_va, cat_features=cat_indices)
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

        va_pred = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = va_pred
        test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
        fold_auc = roc_auc_score(y_va, va_pred)
        fold_aucs.append(fold_auc)
        fold_iters.append(int(model.tree_count_))
        print(
            f"fold {fold}/{N_SPLITS}  AUC={fold_auc:.5f}  iters={model.tree_count_}  "
            f"train_rows={len(X_tr):,}  ({time.time()-t0:.1f}s)",
            flush=True,
        )

    oof_auc = roc_auc_score(y, oof)
    print()
    print(f"per-fold AUC: mean={np.mean(fold_aucs):.5f}  std={np.std(fold_aucs):.5f}  iters={fold_iters}")
    print(f"OOF AUC:      {oof_auc:.5f}")
    print(f"  (vs CB#006 0.94806, Δ = {oof_auc - 0.94806:+.5f})")

    oof_df = pd.DataFrame(
        {"id": train[ID_COL], "Year": train["Year"], "target": y, "oof": oof}
    )
    print("\nOOF AUC by Year:")
    for year, g in oof_df.groupby("Year"):
        if g["target"].nunique() > 1:
            auc = roc_auc_score(g["target"], g["oof"])
            print(f"  {year}: AUC={auc:.5f}  n={len(g):,}  pos_rate={g['target'].mean():.4f}")

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
