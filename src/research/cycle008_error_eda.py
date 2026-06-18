"""Cycle #008 — error-bucket EDA on the current best ensemble OOF.

Goal: locate where the +0.00554 Public LB gap (current 0.94833 vs top-300 ~0.95387)
hides, so cycle 009+ can target the right modeling branch instead of nibbling on
already-explored ones (peer-rank features, etc.).

We score the 3-way ensemble (LGB+CB#004+CB#006, OOF 0.94866) on multiple slices:
  - Per-(Year × Compound)
  - Per-Race (top by row count)
  - Per-Driver (top by row count and bottom by AUC)
  - Per-TyreLife bucket × Compound
  - Per-Stint bucket
  - Per-Position bucket
  - Per-field_pit_share quintile
  - Per-LapsRemaining bucket
  - is_synthetic_driver, is_2023, sc_likely, is_pre_season, is_wet_race

For each slice we report:
  - n          row count
  - pos_rate   target rate within slice
  - auc        OOF AUC inside the slice
  - logloss    log loss inside the slice
  - lift_$     "expected aggregate error reduction if this slice's AUC matched the
                global AUC" — n × (global_auc − slice_auc). Negative if the slice is
                actually OUT-performing global.

The output: a ranked list of slices by `lift_$`, naming the largest-leverage
weaknesses for cycle 009's hypothesis selection.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score

DATA = Path(__file__).resolve().parent.parent.parent / "data"
TRAIN_PARQUET = DATA / "train_features.parquet"

ENSEMBLE_OOF = DATA / "oof_ensemble3_seed42.parquet"  # current best (3-way)


def load() -> pd.DataFrame:
    feats = (
        pl.read_parquet(TRAIN_PARQUET)
        .select(
            [
                "id",
                "Year",
                "Compound",
                "Race",
                "Driver",
                "TyreLife",
                "Stint",
                "Position",
                "LapNumber",
                "LapsRemaining",
                "field_pit_share",
                "is_synthetic_driver",
                "is_2023",
                "is_pre_season",
                "is_wet_race",
                "sc_likely",
                "PitNextLap",
            ]
        )
        .to_pandas()
    )
    oof = pd.read_parquet(ENSEMBLE_OOF)
    df = feats.merge(oof[["id", "oof"]], on="id", how="inner")
    df["target"] = df["PitNextLap"].astype(int)
    return df


def slice_score(g: pd.DataFrame, global_auc: float) -> dict:
    """Return per-slice diagnostics. Skip slices with single-class target."""
    if g["target"].nunique() < 2 or len(g) < 50:
        return {}
    auc = roc_auc_score(g["target"], g["oof"])
    ll = log_loss(g["target"], np.clip(g["oof"], 1e-7, 1 - 1e-7))
    return {
        "n": len(g),
        "pos_rate": g["target"].mean(),
        "auc": auc,
        "logloss": ll,
        "lift_n_x_aucgap": len(g) * (global_auc - auc),
    }


def slice_table(df: pd.DataFrame, by, global_auc: float, top_n: int = 20) -> pd.DataFrame:
    if isinstance(by, str):
        by = [by]
    rows = []
    for key, g in df.groupby(by):
        s = slice_score(g, global_auc)
        if not s:
            continue
        s["slice"] = key if len(by) > 1 else (key,)
        rows.append(s)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("lift_n_x_aucgap", ascending=False)
    return out.head(top_n).reset_index(drop=True)


def bucket(s: pd.Series, edges: list[float]) -> pd.Categorical:
    return pd.cut(s, bins=edges, include_lowest=True)


def main() -> None:
    df = load()
    print(f"loaded {len(df):,} rows; ensemble3 OOF AUC = {roc_auc_score(df['target'], df['oof']):.5f}")
    global_auc = roc_auc_score(df["target"], df["oof"])
    global_ll = log_loss(df["target"], np.clip(df["oof"], 1e-7, 1 - 1e-7))
    print(f"  global log-loss = {global_ll:.5f}")
    print(f"  pos rate        = {df['target'].mean():.4f}")

    # Add bucketed versions of continuous features.
    df["TyreLife_bucket"] = bucket(df["TyreLife"], [-0.1, 5, 10, 15, 20, 30, 50, 100])
    df["Position_bucket"] = bucket(df["Position"], [0, 3, 6, 10, 15, 20])
    df["LapsRemaining_bucket"] = bucket(df["LapsRemaining"], [-0.1, 5, 10, 20, 30, 50, 100])
    df["Stint_bucket"] = bucket(df["Stint"], [0.5, 1.5, 2.5, 3.5, 8.5])
    df["field_pit_share_q"] = pd.qcut(
        df["field_pit_share"], q=5, labels=["q1", "q2", "q3", "q4", "q5"], duplicates="drop"
    )
    df["TyreLife_x_Compound"] = (
        df["TyreLife_bucket"].astype(str) + " × " + df["Compound"].astype(str)
    )

    # === Per-slice scans ===
    scans = [
        ("Year × Compound", ["Year", "Compound"]),
        ("Race", ["Race"]),
        ("TyreLife × Compound", ["TyreLife_x_Compound"]),
        ("Stint bucket", ["Stint_bucket"]),
        ("Position bucket", ["Position_bucket"]),
        ("LapsRemaining bucket", ["LapsRemaining_bucket"]),
        ("field_pit_share quintile", ["field_pit_share_q"]),
        ("is_synthetic_driver", ["is_synthetic_driver"]),
        ("is_2023", ["is_2023"]),
        ("is_pre_season", ["is_pre_season"]),
        ("is_wet_race", ["is_wet_race"]),
        ("sc_likely", ["sc_likely"]),
    ]
    for name, by in scans:
        t = slice_table(df, by, global_auc, top_n=15)
        if t.empty:
            continue
        print(f"\n=== {name} (top by lift_n_x_aucgap) ===")
        with pd.option_context(
            "display.float_format", lambda x: f"{x:.5f}", "display.max_colwidth", 60
        ):
            print(t.to_string(index=False))

    # === Driver scan: large drivers AND worst-AUC drivers (controlling for n>=200) ===
    drv = []
    for d, g in df.groupby("Driver"):
        s = slice_score(g, global_auc)
        if s and len(g) >= 200:
            s["Driver"] = d
            drv.append(s)
    drv_df = pd.DataFrame(drv).sort_values("lift_n_x_aucgap", ascending=False)
    print("\n=== Drivers (n≥200): top by lift_n_x_aucgap ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.5f}"):
        print(drv_df.head(20).to_string(index=False))
    print("\n=== Drivers (n≥200): worst by AUC (bottom 15) ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.5f}"):
        print(drv_df.sort_values("auc", ascending=True).head(15).to_string(index=False))

    # === Calibration: residual distribution by predicted-probability decile ===
    df["pred_decile"] = pd.qcut(
        df["oof"], q=10, labels=[f"d{i}" for i in range(1, 11)], duplicates="drop"
    )
    cal = []
    for d, g in df.groupby("pred_decile", observed=True):
        cal.append(
            {
                "decile": d,
                "n": len(g),
                "mean_pred": g["oof"].mean(),
                "obs_rate": g["target"].mean(),
                "logloss": log_loss(g["target"], np.clip(g["oof"], 1e-7, 1 - 1e-7))
                if g["target"].nunique() > 1
                else np.nan,
            }
        )
    cal_df = pd.DataFrame(cal)
    cal_df["bias"] = cal_df["mean_pred"] - cal_df["obs_rate"]
    print("\n=== Calibration by prediction decile ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.5f}"):
        print(cal_df.to_string(index=False))

    # === Aggregate summary: which scan has the largest single-slice leverage? ===
    print("\n=== Top 10 single-slice leverage opportunities (all scans combined) ===")
    all_rows = []
    for name, by in scans:
        for key, g in df.groupby(by):
            s = slice_score(g, global_auc)
            if s:
                s["scan"] = name
                s["slice"] = str(key)
                all_rows.append(s)
    big = (
        pd.DataFrame(all_rows)
        .sort_values("lift_n_x_aucgap", ascending=False)
        .head(15)
    )
    with pd.option_context(
        "display.float_format", lambda x: f"{x:.5f}", "display.max_colwidth", 60
    ):
        print(big[["scan", "slice", "n", "pos_rate", "auc", "lift_n_x_aucgap"]].to_string(index=False))


if __name__ == "__main__":
    main()
