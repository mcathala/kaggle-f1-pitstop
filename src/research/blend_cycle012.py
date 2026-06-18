"""5-way ensemble blend for cycle #012.

Combines:
  data/oof_lgb_seed42.parquet           LGB on 63 features
  data/oof_cb004_seed42.parquet         CB#004 on 63 features (cycle 6 baseline component)
  data/oof_cb006_seed42.parquet         CB#006 on 66 features (cycle 6 baseline component)
  data/oof_cb_tuned.parquet             CB-tuned on 132 features + external data (cycle 12)

Optionally adds CB#007 (cycle 7, Inconclusive) as a 5-way ablation.

Weight strategies tested:
  (1) 4-way fixed:  LGB=0.10, CB#004=CB#006=0.30, CB-tuned=0.30
  (2) 4-way heavy:  LGB=0.10, CB#004=CB#006=0.15, CB-tuned=0.60  (heavier on the strong model)
  (3) Single:       CB-tuned alone (sanity check)

No OOF grid search — fixed weights only, per cycle #006 followup #2.

Outputs:
  data/oof_ensemble_cycle012.parquet
  data/submission_ensemble_cycle012.csv
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
    "cb004": DATA / "oof_cb004_seed42.parquet",
    "cb006": DATA / "oof_cb006_seed42.parquet",
    "cb_tuned": DATA / "oof_cb_tuned.parquet",
}

SUB_PATHS = {
    "lgb": DATA / "submission_lgb_seed42.csv",
    "cb004": DATA / "submission_cb004_seed42.csv",
    "cb006": DATA / "submission_cb006_seed42.csv",
    "cb_tuned": DATA / "submission_cb_tuned.csv",
}

WEIGHT_SCHEMES = {
    "4way_even":    {"lgb": 0.10, "cb004": 0.30, "cb006": 0.30, "cb_tuned": 0.30},
    "4way_heavy":   {"lgb": 0.10, "cb004": 0.15, "cb006": 0.15, "cb_tuned": 0.60},
    "3way_focus":   {"lgb": 0.05, "cb006": 0.20, "cb_tuned": 0.75},
    "single_tuned": {"cb_tuned": 1.0},
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
        df = pd.read_parquet(p).set_index("id")
        feats[f"oof_{k}"] = df["oof"]
        preds[k] = feats[f"oof_{k}"].to_numpy()
    return feats.reset_index(), preds


def main() -> None:
    meta, preds = load_aligned()
    y = meta[TARGET].astype(int).to_numpy()
    print(f"aligned OOFs on {len(meta):,} train rows")

    print("\n=== Per-component OOF AUC ===")
    for k in preds:
        a = roc_auc_score(y, preds[k])
        print(f"  {k:10s} = {a:.5f}")
    cb006_oof = roc_auc_score(y, preds["cb006"])
    cb_tuned_oof = roc_auc_score(y, preds["cb_tuned"])
    print(f"\n  Δ(CB-tuned − CB#006) = {cb_tuned_oof - cb006_oof:+.5f}")

    # 3-way baseline (existing)
    ens3 = 0.10 * preds["lgb"] + 0.40 * preds["cb004"] + 0.50 * preds["cb006"]
    ens3_auc = roc_auc_score(y, ens3)
    print(f"\n3-way baseline (cycle 6) OOF AUC = {ens3_auc:.5f}")

    # Same StratifiedKFold for per-fold tables
    strat_key = meta["Year"].astype(str) + "_" + meta[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_splits = list(kf.split(meta, strat_key))

    p3_folds = [roc_auc_score(y[va], ens3[va]) for _, va in fold_splits]
    print(f"  per-fold std = {np.std(p3_folds):.5f}")

    print("\n=== Weight schemes ===")
    best = ("", -np.inf, None)
    results = []
    for name, w in WEIGHT_SCHEMES.items():
        assert abs(sum(w.values()) - 1.0) < 1e-9, f"{name} weights don't sum to 1: {w}"
        oof = sum(weight * preds[k] for k, weight in w.items())
        auc = roc_auc_score(y, oof)
        per_fold = [roc_auc_score(y[va], oof[va]) for _, va in fold_splits]
        delta = auc - ens3_auc
        n_up = sum(b > a for a, b in zip(p3_folds, per_fold))
        print(f"\n{name}:  {w}")
        print(f"  OOF AUC = {auc:.5f}  per-fold std = {np.std(per_fold):.5f}  Δ vs 3-way = {delta:+.5f}  folds_up = {n_up}/{N_SPLITS}")
        results.append((name, auc, per_fold, oof))
        if auc > best[1]:
            best = (name, auc, oof)

    # Best variant: write OOF + submission
    name, auc, oof = best
    print(f"\n>>> Best scheme: {name} at OOF {auc:.5f} (Δ vs 3-way {auc-ens3_auc:+.5f})")

    # Build test predictions for the best scheme
    w_best = WEIGHT_SCHEMES[name]
    sub_parts = {}
    for k in w_best:
        s = pd.read_csv(SUB_PATHS[k]).set_index("id")
        sub_parts[k] = s[TARGET]
    sub_df = pd.DataFrame(sub_parts)
    sub_df["ens"] = sum(w_best[k] * sub_df[k] for k in w_best)
    sub_out = sub_df.reset_index()[["id", "ens"]].rename(columns={"ens": TARGET})
    sub_out = sub_out.sort_values("id").reset_index(drop=True)
    sub_out.to_csv(DATA / "submission_ensemble_cycle012.csv", index=False)

    pd.DataFrame(
        {"id": meta[ID_COL], "Year": meta["Year"], "target": y, "oof": oof}
    ).to_parquet(DATA / "oof_ensemble_cycle012.parquet", index=False)
    print(
        f"\nwrote oof_ensemble_cycle012.parquet ({len(meta):,} rows) "
        f"and submission_ensemble_cycle012.csv ({len(sub_out):,} rows, scheme={name})"
    )

    # Final gates summary
    pass_bar = ens3_auc + max(0.5 * np.std(p3_folds), 0.00020)
    print(f"\n=== Verdict gate ===")
    print(f"3-way baseline OOF       = {ens3_auc:.5f}")
    print(f"magnitude floor          = {pass_bar - ens3_auc:.6f}")
    print(f"pass bar (3-way + floor) = {pass_bar:.5f}")
    print(f"best ensemble OOF        = {auc:.5f}  ({'PASS' if auc >= pass_bar else 'FAIL'})")
    if auc >= pass_bar:
        if auc >= ens3_auc + 0.005:
            print(">>> KEEP — significant improvement (>= +0.005)")
        else:
            print(">>> KEEP — clears magnitude gate")


if __name__ == "__main__":
    main()
