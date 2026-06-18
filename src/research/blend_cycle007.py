"""4-way ensemble blend for cycle #007.

Reads the four OOF parquets (LGB on 63-col, CB#004 on 65-col, CB#006 on 68-col,
CB#007 on 71-col) and blends them at fixed weights:

  LGB=0.10, CB#004=0.30, CB#006=0.30, CB#007=0.30

Fixed weights (no OOF grid-search) per cycle-#006 followup #2: the −0.00033
OOF→LB drift in cycle #006 came from sweeping ensemble weights on OOF. We hold
weights constant; only the underlying models move.

Reports OOF AUC for each component and the 4-way ensemble, per-fold breakdown,
per-year and per-(Year, Compound) AUC, and writes the test-set submission.

Inputs (must exist before running):
  data/oof_lgb_cycle006_repro.parquet   from src/repro_cycle006.py
  data/oof_cb004_repro.parquet          from src/repro_cycle006.py
  data/oof_cb006_repro.parquet          from src/repro_cycle006.py
  data/oof_cb007.parquet                from src/train_cb007.py
  data/submission_lgb_cycle006_repro.csv
  data/submission_cb004_repro.csv
  data/submission_cb006_repro.csv
  data/submission_cb007.csv

Outputs:
  data/oof_ensemble4.parquet
  data/submission_ensemble4.csv
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

WEIGHTS = {"lgb": 0.10, "cb004": 0.30, "cb006": 0.30, "cb007": 0.30}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

OOF_PATHS = {
    "lgb": DATA / "oof_lgb_cycle006_repro.parquet",
    "cb004": DATA / "oof_cb004_repro.parquet",
    "cb006": DATA / "oof_cb006_repro.parquet",
    "cb007": DATA / "oof_cb007.parquet",
}
SUB_PATHS = {
    "lgb": DATA / "submission_lgb_cycle006_repro.csv",
    "cb004": DATA / "submission_cb004_repro.csv",
    "cb006": DATA / "submission_cb006_repro.csv",
    "cb007": DATA / "submission_cb007.csv",
}


def load_oof_aligned() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Align all OOF parquets on `id`, return (meta_df, dict of preds).

    Also pull Compound from the train features parquet for per-cell breakdowns.
    """
    feats = pl.read_parquet(TRAIN_PARQUET).select([ID_COL, "Year", "Compound", TARGET]).to_pandas()
    feats = feats.set_index(ID_COL)
    preds: dict[str, np.ndarray] = {}
    for k, p in OOF_PATHS.items():
        df = pd.read_parquet(p).set_index("id")
        feats[f"oof_{k}"] = df["oof"]
        preds[k] = feats[f"oof_{k}"].to_numpy()
    return feats.reset_index(), preds


def main() -> None:
    meta, preds = load_oof_aligned()
    y = meta[TARGET].astype(int).to_numpy()
    print(f"aligned OOFs on {len(meta):,} train rows")

    # Per-component sanity
    for k in preds:
        auc = roc_auc_score(y, preds[k])
        print(f"  {k:5s} OOF AUC = {auc:.5f}")

    ens = sum(WEIGHTS[k] * preds[k] for k in WEIGHTS)
    ens_auc = roc_auc_score(y, ens)
    print(
        f"\n4-way ensemble (LGB={WEIGHTS['lgb']}, "
        f"CB#004={WEIGHTS['cb004']}, CB#006={WEIGHTS['cb006']}, "
        f"CB#007={WEIGHTS['cb007']}): OOF AUC = {ens_auc:.5f}"
    )

    # Per-fold (use the same StratifiedKFold split as training).
    strat_key = meta["Year"].astype(str) + "_" + meta[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    per_fold = []
    print("\nper-fold ensemble AUC:")
    for fold, (_, va_idx) in enumerate(kf.split(meta, strat_key), start=1):
        a = roc_auc_score(y[va_idx], ens[va_idx])
        per_fold.append(a)
        print(f"  fold {fold}/{N_SPLITS}  AUC={a:.5f}")
    print(f"per-fold mean={np.mean(per_fold):.5f} std={np.std(per_fold):.5f}")

    # Per-year
    print("\nensemble OOF AUC by Year:")
    for year, g in meta.assign(ens=ens).groupby("Year"):
        if g[TARGET].nunique() > 1:
            a = roc_auc_score(g[TARGET].astype(int), g["ens"])
            print(f"  {year}: AUC={a:.5f}  n={len(g):,}")

    # Per-(Year, Compound)
    print("\nensemble OOF AUC by (Year, Compound):")
    df2 = meta.assign(ens=ens)
    for (year, comp), g in df2.groupby(["Year", "Compound"]):
        if g[TARGET].nunique() > 1 and len(g) >= 1000:
            a = roc_auc_score(g[TARGET].astype(int), g["ens"])
            print(f"  ({year}, {comp}): AUC={a:.5f}  n={len(g):,}")

    # OOF parquet
    pd.DataFrame(
        {"id": meta[ID_COL], "Year": meta["Year"], "target": y, "oof": ens}
    ).to_parquet(DATA / "oof_ensemble4.parquet", index=False)

    # Test submission — blend the same way.
    sub_parts = []
    for k, p in SUB_PATHS.items():
        s = pd.read_csv(p).set_index("id").rename(columns={TARGET: f"pred_{k}"})
        sub_parts.append(s)
    subs = pd.concat(sub_parts, axis=1)
    subs["ens"] = sum(WEIGHTS[k] * subs[f"pred_{k}"] for k in WEIGHTS)
    sub_out = subs.reset_index()[["id", "ens"]].rename(columns={"ens": TARGET})
    sub_out = sub_out.sort_values("id").reset_index(drop=True)
    sub_out.to_csv(DATA / "submission_ensemble4.csv", index=False)
    print(
        f"\nwrote oof_ensemble4.parquet ({len(meta):,} rows) "
        f"and submission_ensemble4.csv ({len(sub_out):,} rows)"
    )


if __name__ == "__main__":
    main()
