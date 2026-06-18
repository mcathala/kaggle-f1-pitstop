"""Combine multiple CB-tuned seed OOFs and submissions into a multi-seed ensemble.

Usage:
  .venv/bin/python src/combine_cb_seeds.py --seeds 42,7        # 2-seed avg
  .venv/bin/python src/combine_cb_seeds.py --seeds 42,7,99     # 3-seed avg

Reads:
  data/oof_cb_tuned_exp14.parquet       (seed 42, from cycle 14 — special case)
  data/oof_cb_tuned_seed{S}.parquet     (seeds 7, 99 from exp 035)
  data/submission_cb_tuned_exp14.csv    (seed 42)
  data/submission_cb_tuned_seed{S}.csv  (seeds 7, 99)

Writes:
  data/oof_cb_multiseed.parquet
  data/submission_cb_multiseed.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DATA = Path(__file__).resolve().parent.parent.parent / "data"
TARGET = "PitNextLap"
ID_COL = "id"

OOF_OUT = DATA / "oof_cb_multiseed.parquet"
SUB_OUT = DATA / "submission_cb_multiseed.csv"


def oof_path(seed: int) -> Path:
    # Cycle 14's seed-42 OOF lives at the legacy filename.
    return DATA / "oof_cb_tuned_exp14.parquet" if seed == 42 else DATA / f"oof_cb_tuned_seed{seed}.parquet"


def sub_path(seed: int) -> Path:
    return DATA / "submission_cb_tuned_exp14.csv" if seed == 42 else DATA / f"submission_cb_tuned_seed{seed}.csv"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", required=True, help="comma-separated seeds e.g. '42,7'")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"Combining seeds: {seeds}")

    # === OOFs ===
    oof_frames = []
    for s in seeds:
        path = oof_path(s)
        if not path.exists():
            raise FileNotFoundError(f"OOF for seed {s} missing: {path}")
        df = pd.read_parquet(path)[[ID_COL, "Year", "target", "oof"]].rename(
            columns={"oof": f"oof_seed{s}"}
        )
        print(f"  seed {s:3d}: {len(df):,} rows, AUC={roc_auc_score(df['target'], df[f'oof_seed{s}']):.5f}")
        oof_frames.append(df)

    base = oof_frames[0]
    for f in oof_frames[1:]:
        base = base.merge(f[[ID_COL, f"oof_seed{int(f.columns[-1].split('seed')[-1])}"]], on=ID_COL, how="inner")

    seed_cols = [f"oof_seed{s}" for s in seeds]
    base["oof"] = base[seed_cols].mean(axis=1)
    combined_auc = roc_auc_score(base["target"], base["oof"])
    print(f"\nMulti-seed OOF AUC (N={len(seeds)}): {combined_auc:.5f}")
    for s in seeds:
        s_auc = roc_auc_score(base["target"], base[f"oof_seed{s}"])
        print(f"  vs seed {s:3d} alone: Δ = {combined_auc - s_auc:+.5f}")

    out = base[[ID_COL, "Year", "target", "oof"]]
    out.to_parquet(OOF_OUT, index=False)
    print(f"wrote {OOF_OUT.name}  ({len(out):,} rows)")

    # === Submissions ===
    sub_frames = []
    for s in seeds:
        path = sub_path(s)
        if not path.exists():
            raise FileNotFoundError(f"Submission for seed {s} missing: {path}")
        df = pd.read_csv(path).sort_values(ID_COL).reset_index(drop=True)
        df = df.rename(columns={TARGET: f"pred_seed{s}"})
        sub_frames.append(df)

    base_sub = sub_frames[0]
    for f in sub_frames[1:]:
        merge_col = f.columns[-1]
        base_sub = base_sub.merge(f[[ID_COL, merge_col]], on=ID_COL, how="inner")

    pred_cols = [f"pred_seed{s}" for s in seeds]
    base_sub[TARGET] = base_sub[pred_cols].mean(axis=1)
    sub = base_sub[[ID_COL, TARGET]]
    sub.to_csv(SUB_OUT, index=False)
    print(f"wrote {SUB_OUT.name}  ({len(sub):,} rows)")


if __name__ == "__main__":
    main()
