"""Cycle #008 aggregator — seed-robustness sweep for cycle 7's 4-way ensemble.

Reads per-seed component OOFs (LGB, CB#004, CB#006, CB#007) for each seed in
{7, 42, 99}, computes the 4-way ensemble OOF at fixed weights (LGB=0.10, each
CB=0.30, matching cycle #007), and reports per-seed + median + verdict.

Pass bar (Inconclusive → Keep on cycle 7): median 4-way OOF AUC across the 3
seeds must clear the same magnitude gate that cycle 7 missed:

  median_oof_auc >= baseline_3way_seed42 + max(0.5 * baseline_std_seed42, project min_delta)
                 = 0.94866              + max(0.5 * 0.00045,             0.00020)
                 = 0.94866              + 0.000225
                 = 0.94888

If the median clears 0.94888 → flip cycle 7 verdict to Keep.
If not → cycle 7 stays Inconclusive (or downgrade to Discard if median ≤ 0.94866).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score

DATA = Path(__file__).resolve().parent.parent.parent / "data"
TRAIN_PARQUET = DATA / "train_features.parquet"

SEEDS = [7, 42, 99]
WEIGHTS = {"lgb": 0.10, "cb004": 0.30, "cb006": 0.30, "cb007": 0.30}

BASELINE_3WAY_SEED42 = 0.94866
BASELINE_3WAY_STD_SEED42 = 0.00045
MIN_DELTA = 0.00020
MAGNITUDE_FLOOR = max(0.5 * BASELINE_3WAY_STD_SEED42, MIN_DELTA)
PASS_BAR = BASELINE_3WAY_SEED42 + MAGNITUDE_FLOOR


def load_y() -> np.ndarray:
    train = pl.read_parquet(TRAIN_PARQUET).select(["id", "PitNextLap"]).to_pandas()
    return train.sort_values("id")["PitNextLap"].astype(int).to_numpy()


def load_seed(seed: int) -> dict[str, np.ndarray] | None:
    """Returns {model: oof_array} aligned on sorted id, or None if any file missing."""
    preds = {}
    for model in WEIGHTS:
        p = DATA / f"oof_{model}_seed{seed}.parquet"
        if not p.exists():
            print(f"  missing: {p.name}")
            return None
        df = pd.read_parquet(p).sort_values("id")
        preds[model] = df["oof"].to_numpy()
    return preds


def main() -> None:
    y = load_y()
    print(f"loaded target ({len(y):,} rows, pos rate {y.mean():.4f})")
    print(f"\npass bar (median 4-way OOF) = {PASS_BAR:.5f}")
    print(
        f"  = baseline_3way(seed=42) + max(0.5 * baseline_std, min_delta)"
        f" = {BASELINE_3WAY_SEED42} + {MAGNITUDE_FLOOR:.6f}"
    )

    per_seed = {}
    for seed in SEEDS:
        print(f"\n--- seed={seed} ---")
        preds = load_seed(seed)
        if preds is None:
            print(f"  seed {seed} incomplete, skipping")
            continue
        # Component AUCs
        comp = {}
        for k, v in preds.items():
            comp[k] = roc_auc_score(y, v)
        # 3-way (LGB=0.10, CB#004=0.40, CB#006=0.50)
        ens3 = 0.10 * preds["lgb"] + 0.40 * preds["cb004"] + 0.50 * preds["cb006"]
        ens3_auc = roc_auc_score(y, ens3)
        # 4-way (cycle #007 weights)
        ens4 = sum(WEIGHTS[k] * preds[k] for k in WEIGHTS)
        ens4_auc = roc_auc_score(y, ens4)
        per_seed[seed] = {
            **{f"oof_{k}": comp[k] for k in comp},
            "3way": ens3_auc,
            "4way": ens4_auc,
            "delta_4_vs_3": ens4_auc - ens3_auc,
        }
        print(f"  LGB     {comp['lgb']:.5f}")
        print(f"  CB#004  {comp['cb004']:.5f}")
        print(f"  CB#006  {comp['cb006']:.5f}")
        print(f"  CB#007  {comp['cb007']:.5f}")
        print(f"  3-way   {ens3_auc:.5f}")
        print(f"  4-way   {ens4_auc:.5f}   Δ(4-3) = {ens4_auc - ens3_auc:+.5f}")

    if len(per_seed) < 2:
        print(f"\nNot enough seeds finished ({len(per_seed)}). Run more sweeps.")
        return

    # Aggregate
    print(f"\n=== Summary across seeds {sorted(per_seed)} ===")
    df = pd.DataFrame(per_seed).T
    print(df.to_string(float_format=lambda x: f"{x:.5f}"))

    if "4way" in df.columns:
        med_4way = df["4way"].median()
        med_delta = df["delta_4_vs_3"].median()
        n_above = int((df["4way"] >= PASS_BAR).sum())
        n_4way_better_than_3way = int((df["delta_4_vs_3"] > 0).sum())

        print(f"\nMedian 4-way OOF AUC = {med_4way:.5f}")
        print(f"Median Δ(4-3)        = {med_delta:+.5f}")
        print(f"Seeds with 4-way >= pass bar ({PASS_BAR:.5f}): {n_above}/{len(df)}")
        print(f"Seeds with 4-way > 3-way:                       {n_4way_better_than_3way}/{len(df)}")

        if med_4way >= PASS_BAR:
            verdict = "KEEP"
            note = (
                f"Median 4-way ({med_4way:.5f}) clears pass bar ({PASS_BAR:.5f}). "
                "Cycle 7 hypothesis confirmed: peer-rank features carry real signal."
            )
        elif med_4way <= BASELINE_3WAY_SEED42:
            verdict = "DISCARD"
            note = (
                f"Median 4-way ({med_4way:.5f}) is at or below the 3-way baseline "
                f"({BASELINE_3WAY_SEED42:.5f}). Cycle 7 was seed-lucky."
            )
        else:
            verdict = "INCONCLUSIVE (still)"
            note = (
                f"Median 4-way ({med_4way:.5f}) is above the 3-way baseline "
                f"({BASELINE_3WAY_SEED42:.5f}) but below the pass bar "
                f"({PASS_BAR:.5f}). Real but underpowered signal."
            )
        print(f"\n>>> CYCLE 8 VERDICT: {verdict}")
        print(f"    {note}")


if __name__ == "__main__":
    main()
