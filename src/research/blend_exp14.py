"""Experiment 014 (cycle 4) — 4-way / 5-way ensemble blends.

Reads:
  data/oof_lgb_seed42.parquet           LGB on 63 features (cycle-2 baseline)
  data/oof_cb006_seed42.parquet         CB#006 on 66 features (cycle 2)
  data/oof_cb_tuned.parquet             CB-tuned cycle 3 (132 features + external)
  data/oof_cb_tuned_exp14.parquet       CB-tuned exp14 (cycle 3 features + digit/signature + iter 8000)

Compares several fixed-weight blends:
  - 3-way (cycle 2 baseline):        LGB=0.10, CB#004=0.40, CB#006=0.50  (reference)
  - 4-way (cycle 3 3way_focus):      LGB=0.05, CB#006=0.20, CB-tuned-c3=0.75  (reference)
  - 4-way exp14 (drop-in replace):   LGB=0.05, CB#006=0.20, CB-tuned-exp14=0.75
  - 5-way (keep both CB-tuneds):     LGB=0.05, CB#006=0.15, CB-tuned-c3=0.35, CB-tuned-exp14=0.45
  - single CB-tuned-exp14:           1.0  (sanity)

No OOF grid search — fixed weights only. Pre-registered before training.

Outputs:
  data/oof_ensemble_exp14.parquet
  data/submission_ensemble_exp14.csv
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
    "cb_tuned_c3": DATA / "oof_cb_tuned.parquet",
    "cb_tuned_exp14": DATA / "oof_cb_tuned_exp14.parquet",
}

SUB_PATHS = {
    "lgb": DATA / "submission_lgb_seed42.csv",
    "cb004": DATA / "submission_cb004_seed42.csv",
    "cb006": DATA / "submission_cb006_seed42.csv",
    "cb_tuned_c3": DATA / "submission_cb_tuned.csv",
    "cb_tuned_exp14": DATA / "submission_cb_tuned_exp14.csv",
}

WEIGHT_SCHEMES = {
    "3way_c2":           {"lgb": 0.10, "cb004": 0.40, "cb006": 0.50},
    "4way_c3":           {"lgb": 0.05, "cb006": 0.20, "cb_tuned_c3": 0.75},
    "4way_exp14_dropin": {"lgb": 0.05, "cb006": 0.20, "cb_tuned_exp14": 0.75},
    "5way_both_tuned":   {"lgb": 0.05, "cb006": 0.15, "cb_tuned_c3": 0.35, "cb_tuned_exp14": 0.45},
    "single_exp14":      {"cb_tuned_exp14": 1.0},
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

    if "cb_tuned_exp14" in preds and "cb_tuned_c3" in preds:
        a14 = roc_auc_score(y, preds["cb_tuned_exp14"])
        ac3 = roc_auc_score(y, preds["cb_tuned_c3"])
        print(f"\n  Δ(CB-tuned-exp14 − CB-tuned-c3) = {a14 - ac3:+.5f}")

    strat_key = meta["Year"].astype(str) + "_" + meta[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(kf.split(meta, strat_key))

    # Reference cycle-3 ensemble for the verdict gate
    ref_4way = 0.05 * preds["lgb"] + 0.20 * preds["cb006"] + 0.75 * preds["cb_tuned_c3"]
    ref_auc = roc_auc_score(y, ref_4way)
    ref_folds = [roc_auc_score(y[va], ref_4way[va]) for _, va in folds]
    print(f"\n=== Reference: cycle-3 4-way ensemble OOF AUC = {ref_auc:.5f}  fold std {np.std(ref_folds):.5f} ===")

    print("\n=== Weight schemes ===")
    best = ("", -np.inf, None, None)
    results = []
    for name, w in WEIGHT_SCHEMES.items():
        missing = [k for k in w if k not in preds]
        if missing:
            print(f"\n{name}: SKIP (missing components: {missing})")
            continue
        assert abs(sum(w.values()) - 1.0) < 1e-9, f"{name} weights don't sum to 1: {w}"
        oof = sum(weight * preds[k] for k, weight in w.items())
        auc = roc_auc_score(y, oof)
        per_fold = [roc_auc_score(y[va], oof[va]) for _, va in folds]
        delta = auc - ref_auc
        n_up = sum(b > a for a, b in zip(ref_folds, per_fold))
        print(f"\n{name}:  {w}")
        print(f"  OOF AUC = {auc:.5f}  per-fold std = {np.std(per_fold):.5f}  Δ vs cycle-3 = {delta:+.5f}  folds_up = {n_up}/{N_SPLITS}")
        results.append((name, auc, w))
        if auc > best[1] and name != "3way_c2":  # don't count the cycle-2 baseline as winner
            best = (name, auc, w, oof)

    name, auc, w, oof = best
    print(f"\n>>> Best scheme: {name} at OOF {auc:.5f} (Δ vs cycle-3 4-way {auc-ref_auc:+.5f})")

    # Build test predictions for the best scheme
    sub_parts = {}
    for k in w:
        if not SUB_PATHS[k].exists():
            print(f"  WARN: missing submission file {SUB_PATHS[k].name} — skipping submission build.")
            sub_parts = None
            break
        sub_parts[k] = pd.read_csv(SUB_PATHS[k]).set_index("id")[TARGET]

    pd.DataFrame(
        {"id": meta[ID_COL], "Year": meta["Year"], "target": y, "oof": oof}
    ).to_parquet(DATA / "oof_ensemble_exp14.parquet", index=False)

    if sub_parts:
        subs = pd.DataFrame(sub_parts)
        subs["ens"] = sum(w[k] * subs[k] for k in w)
        sub_out = subs.reset_index()[["id", "ens"]].rename(columns={"ens": TARGET})
        sub_out = sub_out.sort_values("id").reset_index(drop=True)
        sub_out.to_csv(DATA / "submission_ensemble_exp14.csv", index=False)
        print(
            f"\nwrote oof_ensemble_exp14.parquet ({len(meta):,} rows) "
            f"and submission_ensemble_exp14.csv ({len(sub_out):,} rows, scheme={name})"
        )

    pass_bar = ref_auc + max(0.5 * np.std(ref_folds), 0.00020)
    print(f"\n=== Verdict gate ===")
    print(f"reference (cycle-3 4-way) OOF      = {ref_auc:.5f}  fold std {np.std(ref_folds):.5f}")
    print(f"magnitude floor                    = {pass_bar - ref_auc:.6f}")
    print(f"pass bar (cycle-3 + floor)         = {pass_bar:.5f}")
    print(f"best ensemble OOF                  = {auc:.5f}  ({'PASS' if auc >= pass_bar else 'FAIL'})")
    if auc >= pass_bar:
        if auc >= ref_auc + 0.0010:
            print(">>> KEEP — significant improvement (>= +0.0010 OOF)")
        else:
            print(">>> KEEP — clears magnitude gate")


if __name__ == "__main__":
    main()
