"""Experiment 036 (cycle 11) — per-slice blend weights.

Tests whether the optimal w_cb in cycle 7's RealMLP × CB-tuned-exp14 blend differs
across data slices defined by probe 2/4 findings (Year × Position_Change sign ×
PitStop bucket).

Method:
  1. Load RealMLP-multiseed OOF + CB-tuned-exp14 OOF + raw train.csv (for slice keys).
  2. Define slices and bucket the rows.
  3. For each slice, grid-search the optimal w_cb using nested CV (leave-out-fold:
     fit weights on 4 of 5 folds, evaluate on the 5th).
  4. Construct full slice-aware OOF and compare to uniform-weight baseline.

Outputs:
  data/blend_per_slice_sweep.parquet     per-slice best w_cb table
  data/oof_blend_per_slice.parquet       full OOF (id, target, oof)
  data/submission_blend_per_slice.csv    if OOF ≥ 0.95428 hurdle
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

SWEEP_OUT = DATA / "blend_per_slice_sweep.parquet"
OOF_OUT = DATA / "oof_blend_per_slice.parquet"
SUB_OUT = DATA / "submission_blend_per_slice.csv"

TARGET = "PitNextLap"
ID_COL = "id"
HURDLE = 0.95428
WEIGHTS = np.round(np.arange(0.0, 0.41, 0.05), 3).tolist()


def make_slices(df: pd.DataFrame) -> pd.Series:
    """Build slice key: Year × sign(Position_Change) × PitStop_bin."""
    year = df["Year"].astype(int).astype(str)
    pc_sign = np.sign(df["Position_Change"]).astype(int).map({-1: "neg", 0: "zero", 1: "pos"})
    pit_bin = pd.cut(df["PitStop"], bins=[-np.inf, 0.5, np.inf], labels=["lo", "hi"]).astype(str)
    return year + "_" + pc_sign + "_" + pit_bin


def best_w_for_slice(y_slice: np.ndarray, rm_slice: np.ndarray, cb_slice: np.ndarray) -> tuple[float, float]:
    """Grid-search best w_cb for a slice on its own data. Returns (best_w, best_auc)."""
    if len(y_slice) < 100 or y_slice.sum() < 5 or y_slice.sum() > len(y_slice) - 5:
        return 0.20, 0.0  # degenerate slice — fallback to global default
    best_w, best_auc = 0.20, 0.0
    for w in WEIGHTS:
        pred = (1 - w) * rm_slice + w * cb_slice
        try:
            auc = roc_auc_score(y_slice, pred)
            if auc > best_auc:
                best_auc, best_w = auc, w
        except ValueError:
            continue
    return best_w, best_auc


def main() -> None:
    print("Loading...")
    train = pd.read_csv(TRAIN_CSV)
    rm = pd.read_parquet(REALMLP_OOF).set_index(ID_COL)
    cb = pd.read_parquet(CB_OOF).set_index(ID_COL)
    train = train.set_index(ID_COL)
    df = train.join(rm["oof"].rename("rm_oof")).join(cb["oof"].rename("cb_oof"))

    print(f"  train {df.shape}  RealMLP OOF {rm.shape}  CB OOF {cb.shape}")
    assert df["rm_oof"].notna().all() and df["cb_oof"].notna().all(), "OOF alignment broken"

    df["slice"] = make_slices(df)
    print(f"\nslice counts (top 20 of {df['slice'].nunique()}):")
    print(df["slice"].value_counts().head(20))

    y = df[TARGET].astype(int).to_numpy()
    rm_oof = df["rm_oof"].to_numpy()
    cb_oof = df["cb_oof"].to_numpy()

    # Baseline: uniform w_cb=0.20 (cycle 7)
    uniform_pred = 0.80 * rm_oof + 0.20 * cb_oof
    auc_uniform = roc_auc_score(y, uniform_pred)
    print(f"\nBaseline (cycle 7, uniform w_cb=0.20): OOF AUC = {auc_uniform:.5f}")

    # === Nested CV for per-slice weights ===
    # Use the same StratifiedKFold the project uses; per outer fold, fit slice weights
    # on the other 4 folds and apply to the held-out fold's slices.
    strat_key = df["Year"].astype(str) + "_" + df[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    slice_keys = df["slice"].to_numpy()
    per_slice_oof = np.full(len(df), np.nan)

    fold_aucs = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(df, strat_key), start=1):
        y_tr = y[tr_idx]
        rm_tr = rm_oof[tr_idx]
        cb_tr = cb_oof[tr_idx]
        s_tr = slice_keys[tr_idx]

        # Fit per-slice weights using train portion of THIS fold
        slice_w: dict[str, float] = {}
        for sk in np.unique(slice_keys):
            mask = s_tr == sk
            if mask.sum() == 0:
                slice_w[sk] = 0.20
                continue
            w, _ = best_w_for_slice(y_tr[mask], rm_tr[mask], cb_tr[mask])
            slice_w[sk] = w

        # Apply to held-out fold
        for sk in np.unique(slice_keys[va_idx]):
            sel = (slice_keys == sk) & np.isin(np.arange(len(df)), va_idx)
            w = slice_w[sk]
            per_slice_oof[sel] = (1 - w) * rm_oof[sel] + w * cb_oof[sel]

        fold_auc = roc_auc_score(y[va_idx], per_slice_oof[va_idx])
        fold_aucs.append(fold_auc)
        print(f"  fold {fold}  AUC={fold_auc:.5f}  ({len(slice_w)} slice weights fit)")

    auc_per_slice = roc_auc_score(y, per_slice_oof)
    print(f"\nPer-slice blend OOF AUC: {auc_per_slice:.5f}")
    print(f"  vs uniform-w blend     : {auc_uniform:.5f}")
    print(f"  Δ                      : {auc_per_slice - auc_uniform:+.5f}")
    print(f"  per-fold std           : {np.std(fold_aucs):.5f}")
    print(f"  hurdle (0.95428)       : Δ {auc_per_slice - HURDLE:+.5f}")

    # Inspect a single full-data fit (not the CV — just to see slice-weight distribution)
    print("\n--- slice weight distribution (single full fit, for inspection) ---")
    rows = []
    for sk in sorted(np.unique(slice_keys)):
        mask = slice_keys == sk
        w, slice_auc = best_w_for_slice(y[mask], rm_oof[mask], cb_oof[mask])
        rows.append({"slice": sk, "n": int(mask.sum()), "pos_rate": float(y[mask].mean()),
                     "best_w": w, "slice_auc": slice_auc})
    sweep = pd.DataFrame(rows).sort_values("n", ascending=False)
    print(sweep.head(20).to_string(index=False))
    sweep.to_parquet(SWEEP_OUT, index=False)
    print(f"\nwrote {SWEEP_OUT.name}")

    # Save OOF
    oof_df = pd.DataFrame({
        ID_COL: df.index, "Year": df["Year"].values, "target": y, "oof": per_slice_oof,
    })
    oof_df.to_parquet(OOF_OUT, index=False)
    print(f"wrote {OOF_OUT.name}")

    # Submission — only if hurdle is cleared
    if auc_per_slice >= HURDLE:
        print(f"\n✓ CLEARS HURDLE — generating test submission")
        # Re-fit slice weights on FULL train OOF, apply to test
        full_slice_w: dict[str, float] = {}
        for sk in np.unique(slice_keys):
            mask = slice_keys == sk
            w, _ = best_w_for_slice(y[mask], rm_oof[mask], cb_oof[mask])
            full_slice_w[sk] = w

        test = pd.read_csv(TEST_CSV)
        test["slice"] = make_slices(test)
        rm_sub = pd.read_csv(REALMLP_SUB).sort_values(ID_COL).reset_index(drop=True)
        cb_sub = pd.read_csv(CB_SUB).sort_values(ID_COL).reset_index(drop=True)
        assert (rm_sub[ID_COL] == cb_sub[ID_COL]).all() and (rm_sub[ID_COL] == test[ID_COL]).all()

        rm_arr = rm_sub[TARGET].to_numpy()
        cb_arr = cb_sub[TARGET].to_numpy()
        blend = np.zeros(len(test))
        for sk in test["slice"].unique():
            w = full_slice_w.get(sk, 0.20)
            mask = test["slice"].to_numpy() == sk
            blend[mask] = (1 - w) * rm_arr[mask] + w * cb_arr[mask]

        pd.DataFrame({ID_COL: test[ID_COL], TARGET: blend}).to_csv(SUB_OUT, index=False)
        print(f"wrote {SUB_OUT.name}")
    else:
        print(f"\n✗ Below hurdle ({auc_per_slice:.5f} < {HURDLE}). No submission written.")


if __name__ == "__main__":
    main()
