"""Experiment 015 (cycle 4) — blend multi-seed CB-tuned + cycle-3 components.

Averages CB-tuned across 3 seeds:
  - data/oof_cb_tuned_exp14.parquet              (seed 42, from exp 14)
  - data/oof_cb_tuned_exp15_seed777.parquet      (seed 777)
  - data/oof_cb_tuned_exp15_seed99.parquet       (seed 99)

Then blends the seed-averaged CB-tuned with the cycle-3 base components:
  - data/oof_lgb_seed42.parquet
  - data/oof_cb006_seed42.parquet

Two weight schemes (fixed, no OOF tuning):
  A. drop-in 3way_focus: LGB=0.05, CB#006=0.20, CB-tuned-3seed-avg=0.75
  B. heavier-on-multi:  LGB=0.05, CB#006=0.10, CB-tuned-3seed-avg=0.85

Outputs:
  data/oof_ensemble_exp15.parquet
  data/submission_ensemble_exp15.csv
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

MULTISEED_PATHS = {
    42: DATA / "oof_cb_tuned_exp14.parquet",
    777: DATA / "oof_cb_tuned_exp15_seed777.parquet",
    99: DATA / "oof_cb_tuned_exp15_seed99.parquet",
}

BASE_OOF_PATHS = {
    "lgb": DATA / "oof_lgb_seed42.parquet",
    "cb006": DATA / "oof_cb006_seed42.parquet",
}

BASE_SUB_PATHS = {
    "lgb": DATA / "submission_lgb_seed42.csv",
    "cb006": DATA / "submission_cb006_seed42.csv",
}

MULTISEED_SUB_PATHS = {
    42: DATA / "submission_cb_tuned_exp14.csv",
    777: DATA / "submission_cb_tuned_exp15_seed777.csv",
    99: DATA / "submission_cb_tuned_exp15_seed99.csv",
}

WEIGHT_SCHEMES = {
    "3way_focus":   {"lgb": 0.05, "cb006": 0.20, "cb_multi": 0.75},
    "heavier_multi": {"lgb": 0.05, "cb006": 0.10, "cb_multi": 0.85},
    "single_multi": {"cb_multi": 1.0},
}


def main() -> None:
    feats = (
        pl.read_parquet(TRAIN_PARQUET)
        .select([ID_COL, "Year", TARGET])
        .to_pandas()
        .set_index(ID_COL)
    )
    y = feats[TARGET].astype(int).to_numpy()

    # Load multi-seed OOFs
    seed_preds = {}
    for seed, p in MULTISEED_PATHS.items():
        if not p.exists():
            print(f"  WARN: missing {p.name}")
            continue
        df = pd.read_parquet(p).set_index("id")
        feats[f"oof_seed{seed}"] = df["oof"]
        seed_preds[seed] = feats[f"oof_seed{seed}"].to_numpy()
        auc = roc_auc_score(y, seed_preds[seed])
        print(f"  seed {seed} OOF AUC = {auc:.5f}")

    if len(seed_preds) < 2:
        print(f"\nNot enough seeds available ({len(seed_preds)}). Run more multi-seed training.")
        return

    # Average all available seeds
    cb_multi = np.mean(list(seed_preds.values()), axis=0)
    cb_multi_auc = roc_auc_score(y, cb_multi)
    print(f"\nseed-averaged CB-tuned OOF AUC = {cb_multi_auc:.5f}  (over {len(seed_preds)} seeds)")

    # Load base OOFs
    base_preds = {}
    for k, p in BASE_OOF_PATHS.items():
        df = pd.read_parquet(p).set_index("id")
        feats[f"oof_{k}"] = df["oof"]
        base_preds[k] = feats[f"oof_{k}"].to_numpy()

    # Reference: cycle 3 4-way ensemble at 0.95134
    ref_4way = 0.05 * base_preds["lgb"] + 0.20 * base_preds["cb006"] + 0.75 * seed_preds[42]
    ref_auc = roc_auc_score(y, ref_4way)
    print(f"\ncycle 3 4-way reference OOF AUC = {ref_auc:.5f}")

    # Per-fold split for stats
    strat_key = feats.reset_index()["Year"].astype(str) + "_" + feats.reset_index()[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(kf.split(feats.reset_index(), strat_key))
    ref_folds = [roc_auc_score(y[va], ref_4way[va]) for _, va in folds]
    print(f"  per-fold std = {np.std(ref_folds):.5f}")

    print("\n=== Weight schemes ===")
    best = ("", -np.inf, None, None)
    for name, w in WEIGHT_SCHEMES.items():
        oof = w.get("lgb", 0) * base_preds["lgb"] + w.get("cb006", 0) * base_preds["cb006"] + w["cb_multi"] * cb_multi
        auc = roc_auc_score(y, oof)
        per_fold = [roc_auc_score(y[va], oof[va]) for _, va in folds]
        delta = auc - ref_auc
        n_up = sum(b > a for a, b in zip(ref_folds, per_fold))
        print(f"\n{name}:  {w}")
        print(f"  OOF AUC = {auc:.5f}  per-fold std = {np.std(per_fold):.5f}  Δ vs cycle 3 = {delta:+.5f}  folds_up = {n_up}/{N_SPLITS}")
        if auc > best[1]:
            best = (name, auc, w, oof)

    name, auc, w, oof = best
    print(f"\n>>> Best scheme: {name} at OOF {auc:.5f} (Δ vs cycle 3 {auc-ref_auc:+.5f})")

    # Test predictions: average submissions across seeds, then blend with base subs
    cb_multi_sub_parts = []
    for seed, p in MULTISEED_SUB_PATHS.items():
        if seed in seed_preds and p.exists():
            cb_multi_sub_parts.append(pd.read_csv(p).set_index("id")[TARGET])
    cb_multi_sub = pd.concat(cb_multi_sub_parts, axis=1).mean(axis=1)

    base_sub_lgb = pd.read_csv(BASE_SUB_PATHS["lgb"]).set_index("id")[TARGET]
    base_sub_cb006 = pd.read_csv(BASE_SUB_PATHS["cb006"]).set_index("id")[TARGET]

    sub_blend = (
        w.get("lgb", 0) * base_sub_lgb + w.get("cb006", 0) * base_sub_cb006 + w["cb_multi"] * cb_multi_sub
    )
    sub_out = sub_blend.reset_index().rename(columns={0: TARGET, "PitNextLap": TARGET})
    sub_out.columns = [ID_COL, TARGET]
    sub_out = sub_out.sort_values(ID_COL).reset_index(drop=True)
    sub_out.to_csv(DATA / "submission_ensemble_exp15.csv", index=False)

    pd.DataFrame(
        {"id": feats.reset_index()[ID_COL], "Year": feats.reset_index()["Year"], "target": y, "oof": oof}
    ).to_parquet(DATA / "oof_ensemble_exp15.parquet", index=False)

    pass_bar = ref_auc + max(0.5 * np.std(ref_folds), 0.00020)
    print(f"\n=== Verdict gate ===")
    print(f"reference (cycle 3 4-way) OOF      = {ref_auc:.5f}")
    print(f"pass bar (cycle 3 + floor)         = {pass_bar:.5f}")
    print(f"best ensemble OOF                  = {auc:.5f}  ({'PASS' if auc >= pass_bar else 'FAIL'})")


if __name__ == "__main__":
    main()
