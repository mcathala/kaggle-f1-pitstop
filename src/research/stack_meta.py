"""Experiment 038 (cycle 11) — Meta-stacker on (OOFs + top raw features).

Builds a meta-feature matrix from cached OOFs + top-5 raw features (per probe 1
importance), fits two meta-models (LR and shallow LGB) under nested 5-fold CV,
and writes a submission iff OOF clears the 0.95428 hurdle.

Outputs:
  data/stack_meta_sweep.parquet      per-model OOF AUC + per-fold AUCs
  data/oof_stack_meta_lr.parquet
  data/oof_stack_meta_lgb.parquet
  data/submission_stack_meta_best.csv (if OOF ≥ 0.95428)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

DATA = Path(__file__).resolve().parent.parent.parent / "data"
TRAIN_CSV = DATA / "train.csv"
TEST_CSV = DATA / "test.csv"
REALMLP_OOF = DATA / "oof_realmlp_multiseed.parquet"
CB_OOF = DATA / "oof_cb_tuned_exp14.parquet"
REALMLP_SUB = DATA / "submission_realmlp_multiseed.csv"
CB_SUB = DATA / "submission_cb_tuned_exp14.csv"

SWEEP_OUT = DATA / "stack_meta_sweep.parquet"
OOF_LR_OUT = DATA / "oof_stack_meta_lr.parquet"
OOF_LGB_OUT = DATA / "oof_stack_meta_lgb.parquet"
SUB_OUT = DATA / "submission_stack_meta_best.csv"

TARGET = "PitNextLap"
ID_COL = "id"
HURDLE = 0.95428

RAW_FEATURES = ["EstimatedTotalLaps", "TyreLife", "Cumulative_Degradation", "LapTime_Delta", "Stint"]


def derive_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the raw-derived features used as meta inputs (matches cycle 14 FE)."""
    eps = 1e-6
    rp = df["RaceProgress"].clip(lower=eps)
    out = df.copy()
    out["EstimatedTotalLaps"] = (out["LapNumber"] / rp).clip(1, 120).astype("float32")
    out["DeltaAbs"] = out["LapTime_Delta"].abs().astype("float32")
    out["TyreAgeRatio"] = (out["TyreLife"] / out["LapNumber"].clip(lower=1).clip(lower=eps)).astype("float32")
    out["LapMinusTyreLife"] = (out["LapNumber"] - out["TyreLife"]).astype("float32")
    out["StintPressure"] = (out["Stint"] * out["TyreLife"]).astype("float32")
    return out


META_RAW = ["EstimatedTotalLaps", "DeltaAbs", "TyreAgeRatio", "LapMinusTyreLife", "StintPressure"]


def main() -> None:
    print("Loading...")
    train = pd.read_csv(TRAIN_CSV)
    rm = pd.read_parquet(REALMLP_OOF).rename(columns={"oof": "rm_oof"})
    cb = pd.read_parquet(CB_OOF).rename(columns={"oof": "cb_oof"})
    df = train.merge(rm[[ID_COL, "rm_oof"]], on=ID_COL).merge(cb[[ID_COL, "cb_oof"]], on=ID_COL)
    df = derive_raw(df)
    print(f"  {df.shape}")
    assert df["rm_oof"].notna().all() and df["cb_oof"].notna().all()

    y = df[TARGET].astype(int).to_numpy()
    feature_cols = ["rm_oof", "cb_oof"] + META_RAW
    X = df[feature_cols].to_numpy().astype("float32")
    print(f"meta features: {feature_cols}")
    print(f"shape: {X.shape}")

    # Cycle 7 baseline
    auc_uniform = roc_auc_score(y, 0.80 * df["rm_oof"].to_numpy() + 0.20 * df["cb_oof"].to_numpy())
    print(f"\nBaseline (cycle 7, uniform w_cb=0.20): OOF AUC = {auc_uniform:.5f}")

    # Nested 5-fold CV — same folds as the rest of the project
    strat_key = df["Year"].astype(str) + "_" + df[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_lr = np.zeros(len(df))
    oof_lgb = np.zeros(len(df))
    fold_lr = []
    fold_lgb = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(df, strat_key), start=1):
        # === Logistic regression meta ===
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X[tr_idx])
        X_va_s = scaler.transform(X[va_idx])
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=42)
        lr.fit(X_tr_s, y[tr_idx])
        oof_lr[va_idx] = lr.predict_proba(X_va_s)[:, 1]
        fold_lr_auc = roc_auc_score(y[va_idx], oof_lr[va_idx])
        fold_lr.append(fold_lr_auc)

        # === Shallow LGB meta ===
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=3, num_leaves=7, learning_rate=0.05,
            min_child_samples=200, reg_lambda=5.0, random_state=42, n_jobs=-1, verbose=-1,
        )
        lgb_model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_lgb[va_idx] = lgb_model.predict_proba(X[va_idx])[:, 1]
        fold_lgb_auc = roc_auc_score(y[va_idx], oof_lgb[va_idx])
        fold_lgb.append(fold_lgb_auc)

        print(f"  fold {fold}  LR AUC={fold_lr_auc:.5f}  LGB AUC={fold_lgb_auc:.5f}")

    auc_lr = roc_auc_score(y, oof_lr)
    auc_lgb = roc_auc_score(y, oof_lgb)
    print(f"\n--- meta-stacker OOF results ---")
    print(f"LR meta:        OOF AUC = {auc_lr:.5f}  (Δ vs uniform: {auc_lr - auc_uniform:+.5f}, vs hurdle: {auc_lr - HURDLE:+.5f})")
    print(f"  per-fold std:  {np.std(fold_lr):.5f}")
    print(f"LGB meta:       OOF AUC = {auc_lgb:.5f}  (Δ vs uniform: {auc_lgb - auc_uniform:+.5f}, vs hurdle: {auc_lgb - HURDLE:+.5f})")
    print(f"  per-fold std:  {np.std(fold_lgb):.5f}")

    # LR coefficients (full-fit diagnostic)
    scaler_full = StandardScaler()
    X_full = scaler_full.fit_transform(X)
    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=42)
    lr_full.fit(X_full, y)
    print("\nLR full-fit standardized coefficients:")
    for name, coef in zip(feature_cols, lr_full.coef_[0]):
        print(f"  {name:25s} {coef:+.4f}")

    # Save outputs
    pd.DataFrame({
        "model": ["lr", "lgb"],
        "oof_auc": [auc_lr, auc_lgb],
        "delta_uniform": [auc_lr - auc_uniform, auc_lgb - auc_uniform],
        "fold_std": [np.std(fold_lr), np.std(fold_lgb)],
    }).to_parquet(SWEEP_OUT, index=False)
    pd.DataFrame({ID_COL: df[ID_COL].values, "Year": df["Year"].values, "target": y, "oof": oof_lr}).to_parquet(OOF_LR_OUT, index=False)
    pd.DataFrame({ID_COL: df[ID_COL].values, "Year": df["Year"].values, "target": y, "oof": oof_lgb}).to_parquet(OOF_LGB_OUT, index=False)
    print(f"\nwrote {SWEEP_OUT.name}, {OOF_LR_OUT.name}, {OOF_LGB_OUT.name}")

    best_auc = max(auc_lr, auc_lgb)
    best_name = "lr" if auc_lr >= auc_lgb else "lgb"
    print(f"\nBest meta-model: {best_name} (OOF AUC {best_auc:.5f})")

    if best_auc >= HURDLE:
        print(f"\n✓ CLEARS HURDLE — generating test submission")
        # Build TEST meta-features
        test = pd.read_csv(TEST_CSV)
        test = derive_raw(test)
        rm_sub = pd.read_csv(REALMLP_SUB).sort_values(ID_COL).reset_index(drop=True)
        cb_sub = pd.read_csv(CB_SUB).sort_values(ID_COL).reset_index(drop=True)
        test = test.merge(rm_sub.rename(columns={TARGET: "rm_oof"})[[ID_COL, "rm_oof"]], on=ID_COL)
        test = test.merge(cb_sub.rename(columns={TARGET: "cb_oof"})[[ID_COL, "cb_oof"]], on=ID_COL)
        X_test = test[feature_cols].to_numpy().astype("float32")

        if best_name == "lr":
            X_test_s = scaler_full.transform(X_test)
            sub_pred = lr_full.predict_proba(X_test_s)[:, 1]
        else:
            lgb_full = lgb.LGBMClassifier(
                n_estimators=200, max_depth=3, num_leaves=7, learning_rate=0.05,
                min_child_samples=200, reg_lambda=5.0, random_state=42, n_jobs=-1, verbose=-1,
            )
            lgb_full.fit(X, y)
            sub_pred = lgb_full.predict_proba(X_test)[:, 1]

        pd.DataFrame({ID_COL: test[ID_COL], TARGET: sub_pred}).to_csv(SUB_OUT, index=False)
        print(f"wrote {SUB_OUT.name}")
    else:
        print(f"\n✗ Below hurdle ({best_auc:.5f} < {HURDLE}). No submission written.")


if __name__ == "__main__":
    main()
