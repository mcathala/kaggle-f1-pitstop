"""Experiment 022 (cycle 7) — rank-blend probe.

Cheap probe pivoting from exp 021 (HP tuning dead). Cycle 4 found that
LINEAR blends with any CB weight hurt the RealMLP-standalone OOF
monotonically. Rank-blend has different geometry — it ignores the
probability scale and only uses the within-prediction ordering — so it
can in principle extract diversity that linear-mean cannot.

Inputs (OOF + test submissions, aligned by id):
  - cycle 5 multi-seed RealMLP: OOF 0.95383, LB 0.95342
  - cycle 4 CB-tuned-exp14:     OOF 0.95114, LB 0.95097

Approach:
  1. Sweep w_cb ∈ {0.00, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25}
  2. For each w_cb, compute BOTH linear-blend and rank-blend OOF AUC
  3. Report the best of each
  4. If best rank-blend > 0.95383 + 0.00020 = 0.95403, write submission CSV
     (held as a candidate; submit deliberately)

Outputs:
  data/blend_rankblend_sweep.parquet — sweep results
  data/oof_rankblend_best.parquet    — best rank-blend OOF
  data/submission_rankblend_best.csv — best rank-blend test submission (if hurdle cleared)
"""

from pathlib import Path
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

DATA = Path(__file__).resolve().parent.parent.parent / "data"

# Inputs
REALMLP_OOF = DATA / "oof_realmlp_multiseed.parquet"
CB_OOF = DATA / "oof_cb_tuned_exp14.parquet"
REALMLP_SUB = DATA / "submission_realmlp_multiseed.csv"
CB_SUB = DATA / "submission_cb_tuned_exp14.csv"

# Outputs
SWEEP_OUT = DATA / "blend_rankblend_sweep.parquet"
OOF_OUT = DATA / "oof_rankblend_best.parquet"
SUB_OUT = DATA / "submission_rankblend_best.csv"

TARGET = "PitNextLap"
ID_COL = "id"
WEIGHTS = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25]
HURDLE = 0.95403  # cycle 5 OOF (0.95383) + min_delta (0.00020)


def to_rank(x: np.ndarray) -> np.ndarray:
    """Convert to dense ranks in [0, 1]. Ties get the average rank."""
    r = rankdata(x, method="average")
    return r / (len(r) - 1)


def main() -> None:
    print(f"scipy: {version('scipy')}")
    print(f"numpy: {version('numpy')}")

    print("\nLoading OOFs...")
    m = pd.read_parquet(REALMLP_OOF).set_index(ID_COL).sort_index()
    c = pd.read_parquet(CB_OOF).set_index(ID_COL).sort_index()
    assert (m["target"] == c["target"]).all(), "target mismatch — alignment broken"

    y = m["target"].to_numpy()
    p_realmlp = m["oof"].to_numpy()
    p_cb = c["oof"].to_numpy()

    auc_realmlp = roc_auc_score(y, p_realmlp)
    auc_cb = roc_auc_score(y, p_cb)
    print(f"  RealMLP OOF AUC: {auc_realmlp:.5f}")
    print(f"  CB-tuned OOF AUC: {auc_cb:.5f}")
    print(f"  rank corr: {pd.Series(p_realmlp).rank().corr(pd.Series(p_cb).rank()):.4f}")

    # Pre-compute rank-transforms (one-time cost)
    r_realmlp = to_rank(p_realmlp)
    r_cb = to_rank(p_cb)

    print(f"\nSweeping w_cb ∈ {WEIGHTS} (cycle 5 baseline = {auc_realmlp:.5f}, hurdle = {HURDLE:.5f})\n")
    print(f"  {'w_cb':>6}  {'linear':>8}  {'Δ_lin':>8}  {'rank':>8}  {'Δ_rank':>8}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    rows = []
    for w in WEIGHTS:
        p_lin = (1 - w) * p_realmlp + w * p_cb
        auc_lin = roc_auc_score(y, p_lin)
        p_rank = (1 - w) * r_realmlp + w * r_cb
        auc_rank = roc_auc_score(y, p_rank)
        rows.append({
            "w_cb": w,
            "auc_linear": auc_lin,
            "delta_linear": auc_lin - auc_realmlp,
            "auc_rank": auc_rank,
            "delta_rank": auc_rank - auc_realmlp,
        })
        print(f"  {w:6.3f}  {auc_lin:8.5f}  {auc_lin-auc_realmlp:+8.5f}  "
              f"{auc_rank:8.5f}  {auc_rank-auc_realmlp:+8.5f}")

    df = pd.DataFrame(rows)
    df.to_parquet(SWEEP_OUT, index=False)
    print(f"\nWrote {SWEEP_OUT.name}")

    best_rank = df.loc[df["auc_rank"].idxmax()]
    best_lin = df.loc[df["auc_linear"].idxmax()]

    print(f"\nBest linear: w_cb={best_lin['w_cb']:.3f}  AUC={best_lin['auc_linear']:.5f}  Δ={best_lin['delta_linear']:+.5f}")
    print(f"Best rank:   w_cb={best_rank['w_cb']:.3f}  AUC={best_rank['auc_rank']:.5f}  Δ={best_rank['delta_rank']:+.5f}")

    if best_rank["auc_rank"] >= HURDLE:
        print(f"\n✓ Rank-blend CLEARED hurdle ({best_rank['auc_rank']:.5f} ≥ {HURDLE:.5f}). Generating submission.")

        # Persist best rank-blend OOF
        pd.DataFrame({
            "id": m.index,
            "Year": m["Year"].values,
            "target": y,
            "oof": (1 - best_rank["w_cb"]) * r_realmlp + best_rank["w_cb"] * r_cb,
        }).to_parquet(OOF_OUT, index=False)
        print(f"Wrote {OOF_OUT.name}")

        # Generate test submission via rank-blend
        print("\nLoading test submissions...")
        sub_m = pd.read_csv(REALMLP_SUB).sort_values(ID_COL).reset_index(drop=True)
        sub_c = pd.read_csv(CB_SUB).sort_values(ID_COL).reset_index(drop=True)
        assert (sub_m[ID_COL] == sub_c[ID_COL]).all(), "test id mismatch"

        rt_m = to_rank(sub_m[TARGET].to_numpy())
        rt_c = to_rank(sub_c[TARGET].to_numpy())
        blended = (1 - best_rank["w_cb"]) * rt_m + best_rank["w_cb"] * rt_c

        pd.DataFrame({
            "id": sub_m[ID_COL],
            TARGET: blended,
        }).to_csv(SUB_OUT, index=False)
        print(f"Wrote {SUB_OUT.name}")
    elif best_rank["auc_rank"] > auc_realmlp:
        print(f"\n~ Rank-blend improved over RealMLP by {best_rank['delta_rank']:+.5f} but below hurdle ({HURDLE:.5f}). No submission generated.")
    else:
        print(f"\n✗ Rank-blend did NOT beat RealMLP standalone. Direction dead.")


if __name__ == "__main__":
    main()
