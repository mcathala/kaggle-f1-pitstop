"""Experiment 016 (cycle 4) — RealMLP + CB ensemble blend.

5-way (or fewer) fixed-weight blend testing whether RealMLP adds ensemble
diversity over the cycle-4 best (exp 14's 4-way `4way_exp14_dropin`).

Reads:
  data/oof_lgb_seed42.parquet           LGB on 63 features (cycle 2 baseline)
  data/oof_cb006_seed42.parquet         CB#006 on 66 features (cycle 2)
  data/oof_cb_tuned_exp14.parquet       CB-tuned-exp14 (cycle 4, iter 8000)
  data/oof_realmlp.parquet              RealMLP-PyTabKit (cycle 4, exp 16)

Reference: exp 14's 4-way ensemble at LGB=0.05, CB#006=0.20, CB-tuned-exp14=0.75
(OOF 0.95161, LB 0.95097).

Fixed weight schemes (no OOF tuning beyond this pre-registered set):
  - 4way_exp14_ref                LGB=0.05, CB#006=0.20, CB-tuned-exp14=0.75    (reference)
  - 5way_realmlp_15               LGB=0.05, CB#006=0.15, CB-tuned-exp14=0.65, RealMLP=0.15
  - 5way_realmlp_20               LGB=0.05, CB#006=0.15, CB-tuned-exp14=0.60, RealMLP=0.20
  - 5way_realmlp_30               LGB=0.05, CB#006=0.10, CB-tuned-exp14=0.55, RealMLP=0.30
  - 3way_cb_realmlp               LGB=0.05, CB-tuned-exp14=0.65, RealMLP=0.30   (drop CB#006)
  - single_realmlp                RealMLP=1.0                                    (sanity)

Outputs:
  data/oof_ensemble_exp16.parquet
  data/submission_ensemble_exp16.csv
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

OOF_PATHS = {
    "lgb": DATA / "oof_lgb_seed42.parquet",
    "cb006": DATA / "oof_cb006_seed42.parquet",
    "cb_tuned_exp14": DATA / "oof_cb_tuned_exp14.parquet",
    "realmlp": DATA / "oof_realmlp.parquet",
}

SUB_PATHS = {
    "lgb": DATA / "submission_lgb_seed42.csv",
    "cb006": DATA / "submission_cb006_seed42.csv",
    "cb_tuned_exp14": DATA / "submission_cb_tuned_exp14.csv",
    "realmlp": DATA / "submission_realmlp.csv",
}

WEIGHT_SCHEMES = {
    "4way_exp14_ref":  {"lgb": 0.05, "cb006": 0.20, "cb_tuned_exp14": 0.75},
    "5way_realmlp_15": {"lgb": 0.05, "cb006": 0.15, "cb_tuned_exp14": 0.65, "realmlp": 0.15},
    "5way_realmlp_20": {"lgb": 0.05, "cb006": 0.15, "cb_tuned_exp14": 0.60, "realmlp": 0.20},
    "5way_realmlp_30": {"lgb": 0.05, "cb006": 0.10, "cb_tuned_exp14": 0.55, "realmlp": 0.30},
    "3way_cb_realmlp": {"lgb": 0.05, "cb_tuned_exp14": 0.65, "realmlp": 0.30},
    "single_realmlp":  {"realmlp": 1.0},
}


def load_aligned() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    feats = (
        pl.read_parquet(TRAIN_PARQUET)
        .select([ID_COL, "Year", TARGET])
        .to_pandas()
        .set_index(ID_COL)
    )
    preds: dict[str, np.ndarray] = {}
    for k, p in OOF_PATHS.items():
        if not p.exists():
            print(f"  WARN: missing {p.name}, skipping")
            continue
        df = pd.read_parquet(p).set_index("id")
        feats[f"oof_{k}"] = df["oof"]
        preds[k] = feats[f"oof_{k}"].to_numpy()
    return feats.reset_index(), preds


def main() -> None:
    meta, preds = load_aligned()
    y = meta[TARGET].astype(int).to_numpy()
    print(f"aligned OOFs on {len(meta):,} train rows")

    print("\n=== Per-component OOF AUC ===")
    for k, p in preds.items():
        a = roc_auc_score(y, p)
        print(f"  {k:18s} = {a:.5f}")

    # Reference: exp 14's best blend
    ref = 0.05 * preds["lgb"] + 0.20 * preds["cb006"] + 0.75 * preds["cb_tuned_exp14"]
    ref_auc = roc_auc_score(y, ref)
    print(f"\nReference (exp 14 4-way) OOF AUC = {ref_auc:.5f}")

    strat_key = meta["Year"].astype(str) + "_" + meta[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(kf.split(meta, strat_key))
    ref_folds = [roc_auc_score(y[va], ref[va]) for _, va in folds]
    print(f"  per-fold std = {np.std(ref_folds):.5f}")

    print("\n=== Weight schemes ===")
    best = ("", -np.inf, None, None)
    for name, w in WEIGHT_SCHEMES.items():
        missing = [k for k in w if k not in preds]
        if missing:
            print(f"\n{name}: SKIP (missing: {missing})")
            continue
        assert abs(sum(w.values()) - 1.0) < 1e-9, f"{name} weights don't sum to 1: {w}"
        oof = sum(weight * preds[k] for k, weight in w.items())
        auc = roc_auc_score(y, oof)
        per_fold = [roc_auc_score(y[va], oof[va]) for _, va in folds]
        delta = auc - ref_auc
        n_up = sum(b > a for a, b in zip(ref_folds, per_fold))
        print(f"\n{name}:  {w}")
        print(f"  OOF AUC = {auc:.5f}  per-fold std = {np.std(per_fold):.5f}  Δ vs exp 14 = {delta:+.5f}  folds_up = {n_up}/{N_SPLITS}")
        if auc > best[1] and name != "4way_exp14_ref":
            best = (name, auc, w, oof)

    if best[0] == "":
        print("\nNo improvement over exp 14's ensemble.")
        return

    name, auc, w, oof = best
    print(f"\n>>> Best scheme: {name} at OOF {auc:.5f} (Δ vs exp 14 {auc-ref_auc:+.5f})")

    # Build test predictions
    sub_parts = {}
    for k in w:
        sub_parts[k] = pd.read_csv(SUB_PATHS[k]).set_index("id")[TARGET]
    subs = pd.DataFrame(sub_parts)
    subs["ens"] = sum(w[k] * subs[k] for k in w)
    sub_out = subs.reset_index()[["id", "ens"]].rename(columns={"ens": TARGET})
    sub_out = sub_out.sort_values("id").reset_index(drop=True)
    sub_out.to_csv(DATA / "submission_ensemble_exp16.csv", index=False)

    pd.DataFrame(
        {"id": meta[ID_COL], "Year": meta["Year"], "target": y, "oof": oof}
    ).to_parquet(DATA / "oof_ensemble_exp16.parquet", index=False)
    print(f"\nwrote oof_ensemble_exp16.parquet and submission_ensemble_exp16.csv (scheme={name})")

    pass_bar = ref_auc + max(0.5 * np.std(ref_folds), 0.00020)
    print(f"\n=== Verdict gate ===")
    print(f"reference (exp 14 4-way) OOF       = {ref_auc:.5f}")
    print(f"pass bar (exp 14 + floor)          = {pass_bar:.5f}")
    print(f"best ensemble OOF                  = {auc:.5f}  ({'PASS' if auc >= pass_bar else 'FAIL'})")
    if auc >= pass_bar:
        if auc >= ref_auc + 0.0010:
            print(">>> KEEP — significant improvement (>= +0.0010 over exp 14)")
        else:
            print(">>> KEEP — clears magnitude gate")


if __name__ == "__main__":
    main()
