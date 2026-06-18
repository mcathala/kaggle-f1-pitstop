"""4-way ensemble blend for cycle #009 (pit-cluster saturation features).

Reads:
  data/oof_lgb_seed42.parquet   LGB on 63 features (cycle-#006 baseline)
  data/oof_cb004_seed42.parquet CB#004 on 63 features
  data/oof_cb006_seed42.parquet CB#006 on 66 features (cycle-#006 set)
  data/oof_cb009.parquet        CB#009 on 70 features (cycle-#009 set = #006 + 4)

Blends at fixed weights (no OOF tuning):
  LGB=0.10, CB#004=0.30, CB#006=0.30, CB#009=0.30

Reports OOF AUC for each component, the 3-way (baseline) and 4-way (cycle 9),
per-fold breakdown, per-year, per-(Year, Compound), and per-field_pit_share
quintile (the targeted slice from cycle 8's EDA). Writes the test submission.

Outputs:
  data/oof_ensemble4_cycle009.parquet
  data/submission_ensemble4_cycle009.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA = Path(__file__).resolve().parent.parent.parent / "data"
TRAIN_PARQUET = DATA / "train_features.parquet"

TARGET = "PitNextLap"
ID_COL = "id"
N_SPLITS = 5
SEED = 42

WEIGHTS = {"lgb": 0.10, "cb004": 0.30, "cb006": 0.30, "cb009": 0.30}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

OOF_PATHS = {
    "lgb": DATA / "oof_lgb_seed42.parquet",
    "cb004": DATA / "oof_cb004_seed42.parquet",
    "cb006": DATA / "oof_cb006_seed42.parquet",
    "cb009": DATA / "oof_cb009.parquet",
}


def load_aligned() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    feats = (
        pl.read_parquet(TRAIN_PARQUET)
        .select([ID_COL, "Year", "Compound", "field_pit_share", TARGET])
        .to_pandas()
        .set_index(ID_COL)
    )
    preds: dict[str, np.ndarray] = {}
    for k, p in OOF_PATHS.items():
        df = pd.read_parquet(p).set_index("id")
        feats[f"oof_{k}"] = df["oof"]
        preds[k] = feats[f"oof_{k}"].to_numpy()
    return feats.reset_index(), preds


def main() -> None:
    meta, preds = load_aligned()
    y = meta[TARGET].astype(int).to_numpy()
    print(f"aligned OOFs on {len(meta):,} train rows")

    for k in preds:
        auc = roc_auc_score(y, preds[k])
        print(f"  {k:5s} OOF AUC = {auc:.5f}")

    # 3-way baseline (cycle-6 weights)
    ens3 = 0.10 * preds["lgb"] + 0.40 * preds["cb004"] + 0.50 * preds["cb006"]
    ens3_auc = roc_auc_score(y, ens3)
    print(f"\n3-way baseline (LGB=0.10, CB#004=0.40, CB#006=0.50):  OOF AUC = {ens3_auc:.5f}")

    # 4-way cycle 9 (fixed weights, no OOF tuning)
    ens4 = sum(WEIGHTS[k] * preds[k] for k in WEIGHTS)
    ens4_auc = roc_auc_score(y, ens4)
    print(
        f"4-way cycle 9 (LGB={WEIGHTS['lgb']}, CB#004={WEIGHTS['cb004']}, "
        f"CB#006={WEIGHTS['cb006']}, CB#009={WEIGHTS['cb009']}): OOF AUC = {ens4_auc:.5f}"
    )
    print(f"\nΔ(4-way − 3-way) = {ens4_auc - ens3_auc:+.5f}")

    # Per-fold
    strat_key = meta["Year"].astype(str) + "_" + meta[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    print("\nper-fold AUC (3-way → 4-way):")
    p3_fold = []
    p4_fold = []
    for fold, (_, va_idx) in enumerate(kf.split(meta, strat_key), start=1):
        a3 = roc_auc_score(y[va_idx], ens3[va_idx])
        a4 = roc_auc_score(y[va_idx], ens4[va_idx])
        p3_fold.append(a3)
        p4_fold.append(a4)
        print(f"  fold {fold}/{N_SPLITS}  3way={a3:.5f}  4way={a4:.5f}  Δ={a4-a3:+.5f}")
    print(f"per-fold mean 3way={np.mean(p3_fold):.5f} std={np.std(p3_fold):.5f}")
    print(f"per-fold mean 4way={np.mean(p4_fold):.5f} std={np.std(p4_fold):.5f}")
    n_folds_up = sum(b > a for a, b in zip(p3_fold, p4_fold))
    print(f"folds where 4way > 3way: {n_folds_up}/{N_SPLITS}")

    # Per-year
    print("\nper-year ensemble OOF AUC (4-way):")
    df2 = meta.assign(ens=ens4)
    for year, g in df2.groupby("Year"):
        if g[TARGET].nunique() > 1:
            a = roc_auc_score(g[TARGET].astype(int), g["ens"])
            print(f"  {year}: AUC={a:.5f}  n={len(g):,}")

    # Per-(Year, Compound)
    print("\nper-(Year, Compound) ensemble OOF AUC (4-way), n≥1000:")
    for (year, comp), g in df2.groupby(["Year", "Compound"]):
        if g[TARGET].nunique() > 1 and len(g) >= 1000:
            a = roc_auc_score(g[TARGET].astype(int), g["ens"])
            print(f"  ({year}, {comp}): AUC={a:.5f}  n={len(g):,}")

    # field_pit_share quintile — the targeted slice from cycle 8 EDA
    print("\nensemble OOF AUC by field_pit_share quintile (3-way → 4-way):")
    df2["fps_q"] = pd.qcut(
        df2["field_pit_share"], q=5, labels=["q1", "q2", "q3", "q4", "q5"], duplicates="drop"
    )
    df2["ens3"] = ens3
    for q, g in df2.groupby("fps_q", observed=True):
        if g[TARGET].nunique() > 1:
            a3 = roc_auc_score(g[TARGET].astype(int), g["ens3"])
            a4 = roc_auc_score(g[TARGET].astype(int), g["ens"])
            print(
                f"  {q}: 3way={a3:.5f}  4way={a4:.5f}  Δ={a4-a3:+.5f}  "
                f"n={len(g):,}  pos_rate={g[TARGET].astype(int).mean():.4f}"
            )

    # Persist OOF + submission
    pd.DataFrame(
        {"id": meta[ID_COL], "Year": meta["Year"], "target": y, "oof": ens4}
    ).to_parquet(DATA / "oof_ensemble4_cycle009.parquet", index=False)

    sub_parts = []
    for k in WEIGHTS:
        sub_path = DATA / f"submission_{k}_seed42.csv" if k != "cb009" else DATA / "submission_cb009.csv"
        # LGB / CB#004 / CB#006 share the cycle-#006 baseline suffix
        if k in {"lgb", "cb004", "cb006"}:
            sub_path = DATA / f"submission_{k}_seed42.csv"
        else:
            sub_path = DATA / "submission_cb009.csv"
        s = pd.read_csv(sub_path).set_index("id").rename(columns={TARGET: f"pred_{k}"})
        sub_parts.append(s)
    subs = pd.concat(sub_parts, axis=1)
    subs["ens"] = sum(WEIGHTS[k] * subs[f"pred_{k}"] for k in WEIGHTS)
    sub_out = subs.reset_index()[["id", "ens"]].rename(columns={"ens": TARGET})
    sub_out = sub_out.sort_values("id").reset_index(drop=True)
    sub_out.to_csv(DATA / "submission_ensemble4_cycle009.csv", index=False)
    print(
        f"\nwrote oof_ensemble4_cycle009.parquet ({len(meta):,} rows) "
        f"and submission_ensemble4_cycle009.csv ({len(sub_out):,} rows)"
    )


if __name__ == "__main__":
    main()
