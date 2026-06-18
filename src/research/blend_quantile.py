"""Experiment 037 (cycle 11) — quantile-bucketed blend weights.

Different w_cb per RealMLP-prediction quantile bucket. Directly responds to
probe 4 (rank disagreement on low-prob rows) and probe 5 (calibration bias in
mid-prob bins).

Method:
  1. Bucket rows by RealMLP OOF quantile (5 equal-frequency buckets).
  2. Nested 5-fold CV: per outer fold, fit best w_cb per bucket on the 4 train
     folds, apply to the held-out fold.
  3. Compare to cycle 7's uniform w_cb=0.20 blend.

Outputs:
  data/blend_quantile_sweep.parquet     per-bucket best w_cb table
  data/oof_blend_quantile.parquet       full OOF
  data/submission_blend_quantile.csv    if OOF ≥ 0.95428
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA = Path(__file__).resolve().parent.parent.parent / "data"
TRAIN_CSV = DATA / "train.csv"
TEST_CSV = DATA / "test.csv"
REALMLP_OOF = DATA / "oof_realmlp_multiseed.parquet"
CB_OOF = DATA / "oof_cb_tuned_exp14.parquet"
REALMLP_SUB = DATA / "submission_realmlp_multiseed.csv"
CB_SUB = DATA / "submission_cb_tuned_exp14.csv"

SWEEP_OUT = DATA / "blend_quantile_sweep.parquet"
OOF_OUT = DATA / "oof_blend_quantile.parquet"
SUB_OUT = DATA / "submission_blend_quantile.csv"

TARGET = "PitNextLap"
ID_COL = "id"
HURDLE = 0.95428
WEIGHTS = np.round(np.arange(0.0, 0.41, 0.025), 3).tolist()
N_BUCKETS = 5


def assign_buckets(scores: np.ndarray, n_buckets: int = N_BUCKETS) -> np.ndarray:
    """Equal-frequency bucket assignment based on rank of scores."""
    ranks = pd.Series(scores).rank(method="first").to_numpy()
    return ((ranks - 1) // (len(scores) / n_buckets)).astype(int).clip(0, n_buckets - 1)


def best_w_for_bucket(y: np.ndarray, rm: np.ndarray, cb: np.ndarray) -> tuple[float, float]:
    if len(y) < 100 or y.sum() < 5 or y.sum() > len(y) - 5:
        return 0.20, 0.0
    best_w, best_auc = 0.20, 0.0
    for w in WEIGHTS:
        pred = (1 - w) * rm + w * cb
        try:
            auc = roc_auc_score(y, pred)
            if auc > best_auc:
                best_auc, best_w = auc, w
        except ValueError:
            continue
    return best_w, best_auc


def main() -> None:
    print("Loading...")
    train = pd.read_csv(TRAIN_CSV).set_index(ID_COL)
    rm = pd.read_parquet(REALMLP_OOF).set_index(ID_COL)
    cb = pd.read_parquet(CB_OOF).set_index(ID_COL)
    df = train.join(rm["oof"].rename("rm_oof")).join(cb["oof"].rename("cb_oof"))
    print(f"  {df.shape}")
    assert df["rm_oof"].notna().all() and df["cb_oof"].notna().all()

    y = df[TARGET].astype(int).to_numpy()
    rm_oof = df["rm_oof"].to_numpy()
    cb_oof = df["cb_oof"].to_numpy()

    auc_uniform = roc_auc_score(y, 0.80 * rm_oof + 0.20 * cb_oof)
    print(f"\nBaseline (cycle 7, uniform w_cb=0.20): OOF AUC = {auc_uniform:.5f}")

    # Buckets are based on RealMLP OOF rank (deterministic, no leakage)
    buckets = assign_buckets(rm_oof, N_BUCKETS)
    print(f"\nBucket counts:")
    for b in range(N_BUCKETS):
        mask = buckets == b
        print(f"  bucket {b}  n={mask.sum():>6,}  pos_rate={y[mask].mean():.4f}  "
              f"rm_range=[{rm_oof[mask].min():.4f}, {rm_oof[mask].max():.4f}]")

    # Nested 5-fold CV
    strat_key = df["Year"].astype(str) + "_" + df[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    per_quantile_oof = np.zeros(len(df))
    fold_aucs = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(df, strat_key), start=1):
        bucket_w = {}
        for b in range(N_BUCKETS):
            mask_tr = (buckets[tr_idx] == b)
            if mask_tr.sum() == 0:
                bucket_w[b] = 0.20
                continue
            w, _ = best_w_for_bucket(y[tr_idx][mask_tr], rm_oof[tr_idx][mask_tr], cb_oof[tr_idx][mask_tr])
            bucket_w[b] = w
        # Apply
        for b in range(N_BUCKETS):
            sel = np.zeros(len(df), dtype=bool)
            sel[va_idx] = buckets[va_idx] == b
            w = bucket_w[b]
            per_quantile_oof[sel] = (1 - w) * rm_oof[sel] + w * cb_oof[sel]
        fold_auc = roc_auc_score(y[va_idx], per_quantile_oof[va_idx])
        fold_aucs.append(fold_auc)
        print(f"  fold {fold}  AUC={fold_auc:.5f}  weights={ {b: round(bucket_w[b], 3) for b in range(N_BUCKETS)} }")

    auc_q = roc_auc_score(y, per_quantile_oof)
    print(f"\nQuantile-bucketed blend OOF AUC: {auc_q:.5f}")
    print(f"  vs uniform-w blend             : {auc_uniform:.5f}")
    print(f"  Δ                              : {auc_q - auc_uniform:+.5f}")
    print(f"  per-fold std                   : {np.std(fold_aucs):.5f}")
    print(f"  hurdle (0.95428)               : Δ {auc_q - HURDLE:+.5f}")

    # Inspect full-fit bucket weights for diagnostic
    print("\n--- full-fit bucket weights (diagnostic, not used in OOF) ---")
    rows = []
    for b in range(N_BUCKETS):
        mask = buckets == b
        w, b_auc = best_w_for_bucket(y[mask], rm_oof[mask], cb_oof[mask])
        rows.append({"bucket": b, "n": int(mask.sum()), "pos_rate": float(y[mask].mean()),
                     "rm_min": float(rm_oof[mask].min()), "rm_max": float(rm_oof[mask].max()),
                     "best_w": w, "bucket_auc": b_auc})
    sweep = pd.DataFrame(rows)
    print(sweep.to_string(index=False))
    sweep.to_parquet(SWEEP_OUT, index=False)
    print(f"\nwrote {SWEEP_OUT.name}")

    oof_df = pd.DataFrame({ID_COL: df.index, "Year": df["Year"].values, "target": y, "oof": per_quantile_oof})
    oof_df.to_parquet(OOF_OUT, index=False)
    print(f"wrote {OOF_OUT.name}")

    if auc_q >= HURDLE:
        print(f"\n✓ CLEARS HURDLE — generating test submission")
        # Re-fit bucket weights on full OOF; bucket TEST by RealMLP test submission's quantile rank
        full_w = {b: sweep.loc[sweep["bucket"] == b, "best_w"].iloc[0] for b in range(N_BUCKETS)}

        test = pd.read_csv(TEST_CSV)
        rm_sub = pd.read_csv(REALMLP_SUB).sort_values(ID_COL).reset_index(drop=True)
        cb_sub = pd.read_csv(CB_SUB).sort_values(ID_COL).reset_index(drop=True)
        assert (rm_sub[ID_COL] == cb_sub[ID_COL]).all() and (rm_sub[ID_COL] == test[ID_COL]).all()

        rm_arr = rm_sub[TARGET].to_numpy()
        cb_arr = cb_sub[TARGET].to_numpy()
        test_buckets = assign_buckets(rm_arr, N_BUCKETS)
        blend = np.zeros(len(test))
        for b in range(N_BUCKETS):
            mask = test_buckets == b
            w = full_w[b]
            blend[mask] = (1 - w) * rm_arr[mask] + w * cb_arr[mask]

        pd.DataFrame({ID_COL: test[ID_COL], TARGET: blend}).to_csv(SUB_OUT, index=False)
        print(f"wrote {SUB_OUT.name}")
    else:
        print(f"\n✗ Below hurdle ({auc_q:.5f} < {HURDLE}). No submission written.")


if __name__ == "__main__":
    main()
