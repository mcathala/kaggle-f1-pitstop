"""Experiment 017 (cycle 5) — multi-seed RealMLP averaging.

Reads all available RealMLP OOFs across seeds (cycle 4's seed=42 + the cycle 5
seeds {7, 99, 137, 313, 777}) and averages them into a single multi-seed
ensemble. Pure variance reduction — equal weights, no variant selection, no
public-LB-overfit risk.

Inputs (whichever exist):
  data/oof_realmlp.parquet               cycle 4 seed=42
  data/oof_realmlp_seed7.parquet
  data/oof_realmlp_seed99.parquet
  data/oof_realmlp_seed137.parquet
  data/oof_realmlp_seed313.parquet
  data/oof_realmlp_seed777.parquet

Outputs:
  data/oof_realmlp_multiseed.parquet
  data/submission_realmlp_multiseed.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA = Path(__file__).resolve().parent.parent / "data"
TRAIN_PARQUET = DATA / "train_features.parquet"
TARGET = "PitNextLap"
ID_COL = "id"
N_SPLITS = 5
SPLIT_SEED = 42

SEED_OOF_PATHS = {
    42: DATA / "oof_realmlp.parquet",
    7: DATA / "oof_realmlp_seed7.parquet",
    99: DATA / "oof_realmlp_seed99.parquet",
    137: DATA / "oof_realmlp_seed137.parquet",
    313: DATA / "oof_realmlp_seed313.parquet",
    777: DATA / "oof_realmlp_seed777.parquet",
}

SEED_SUB_PATHS = {
    42: DATA / "submission_realmlp.csv",
    7: DATA / "submission_realmlp_seed7.csv",
    99: DATA / "submission_realmlp_seed99.csv",
    137: DATA / "submission_realmlp_seed137.csv",
    313: DATA / "submission_realmlp_seed313.csv",
    777: DATA / "submission_realmlp_seed777.csv",
}


def main() -> None:
    feats = (
        pl.read_parquet(TRAIN_PARQUET)
        .select([ID_COL, "Year", TARGET])
        .to_pandas()
        .set_index(ID_COL)
    )
    y = feats[TARGET].astype(int).to_numpy()

    seed_oofs = {}
    for seed, p in SEED_OOF_PATHS.items():
        if not p.exists():
            print(f"  missing seed {seed}: {p.name}")
            continue
        df = pd.read_parquet(p).set_index("id")
        seed_oofs[seed] = df["oof"].reindex(feats.index).to_numpy()
        auc = roc_auc_score(y, seed_oofs[seed])
        print(f"  seed {seed:>4} OOF AUC = {auc:.5f}")

    if len(seed_oofs) < 2:
        print(f"\nNot enough seeds available ({len(seed_oofs)}). Need at least 2 to average.")
        return

    avg_oof = np.mean(list(seed_oofs.values()), axis=0)
    avg_auc = roc_auc_score(y, avg_oof)
    print(f"\n=== Multi-seed average ({len(seed_oofs)} seeds) ===")
    print(f"OOF AUC = {avg_auc:.5f}")

    # Per-fold for stats
    strat_key = feats.reset_index()["Year"].astype(str) + "_" + feats.reset_index()[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_SEED)
    folds = list(kf.split(feats.reset_index(), strat_key))
    fold_aucs = [roc_auc_score(y[va], avg_oof[va]) for _, va in folds]
    print(f"per-fold mean = {np.mean(fold_aucs):.5f}  std = {np.std(fold_aucs):.5f}")
    print(f"per-fold:  {[f'{a:.5f}' for a in fold_aucs]}")

    # Comparison: best single seed vs average
    best_seed = max(seed_oofs, key=lambda s: roc_auc_score(y, seed_oofs[s]))
    best_auc = roc_auc_score(y, seed_oofs[best_seed])
    print(f"\nbest single seed ({best_seed}): {best_auc:.5f}")
    print(f"multi-seed average:     {avg_auc:.5f}")
    print(f"Δ (avg − best single):  {avg_auc - best_auc:+.5f}")

    # Save OOF + submission
    pd.DataFrame(
        {"id": feats.reset_index()[ID_COL], "Year": feats.reset_index()["Year"], "target": y, "oof": avg_oof}
    ).to_parquet(DATA / "oof_realmlp_multiseed.parquet", index=False)

    # Test predictions: average submissions across the same seeds
    sub_parts = []
    for seed in seed_oofs:
        p = SEED_SUB_PATHS[seed]
        if p.exists():
            sub_parts.append(pd.read_csv(p).set_index("id")[TARGET])
    if len(sub_parts) < 2:
        print(f"Not enough seed submissions to average ({len(sub_parts)}).")
        return
    avg_sub = pd.concat(sub_parts, axis=1).mean(axis=1)
    sub_out = avg_sub.reset_index().rename(columns={0: TARGET, "PitNextLap": TARGET})
    sub_out.columns = [ID_COL, TARGET]
    sub_out = sub_out.sort_values(ID_COL).reset_index(drop=True)
    sub_out.to_csv(DATA / "submission_realmlp_multiseed.csv", index=False)
    print(f"\nwrote oof_realmlp_multiseed.parquet ({len(avg_oof):,} rows) and submission_realmlp_multiseed.csv ({len(sub_out):,} rows)")

    # Verdict gate
    ref_auc = 0.95355  # cycle 4 RealMLP single-seed OOF
    pass_bar = ref_auc + 0.00020  # variance reduction is small but should clear noise floor
    print(f"\n=== Verdict gate ===")
    print(f"reference (cycle 4 RealMLP single-seed) OOF = {ref_auc:.5f}")
    print(f"pass bar (cycle 4 + 0.00020 floor)          = {pass_bar:.5f}")
    print(f"multi-seed OOF                              = {avg_auc:.5f}  ({'PASS' if avg_auc >= pass_bar else 'FAIL'})")
    if avg_auc >= pass_bar:
        print(">>> KEEP — clears magnitude gate")


if __name__ == "__main__":
    main()
