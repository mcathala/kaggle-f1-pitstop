"""Experiment 023 (cycle 8) — 3-way blend probe.

Sweeps a 3-way linear blend of:
  - cycle 5 multi-seed RealMLP (OOF 0.95383, LB 0.95342)
  - cycle 4 CB-tuned-exp14    (OOF 0.95114, LB 0.95097)
  - cycle 1 LGB baseline       (OOF 0.94166, LB 0.94211)

LGB is the most structurally diverse signal in the model zoo (different
family entirely from NN + gradient boosting). Cycle 7 showed CB at small
weight helps multi-seed RealMLP; this probe asks whether a third diverse
signal pushes the blend further.

Two-stage sweep:
  1. Coarse: w_cb x w_lgb grid (15 points)
  2. Fine: ±0.025 around coarse peak at 0.0125 step (25 points)

Outputs:
  data/blend_3way_sweep.parquet           — full sweep results
  data/oof_3way_best.parquet              — best 3-way OOF
  data/submission_3way_best.csv           — best 3-way submission (if hurdle cleared)
"""

from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DATA = Path(__file__).resolve().parent.parent.parent / "data"

REALMLP_OOF = DATA / "oof_realmlp_multiseed.parquet"
CB_OOF = DATA / "oof_cb_tuned_exp14.parquet"
LGB_OOF = DATA / "oof_baseline.parquet"

REALMLP_SUB = DATA / "submission_realmlp_multiseed.csv"
CB_SUB = DATA / "submission_cb_tuned_exp14.csv"
LGB_SUB = DATA / "submission_baseline.csv"

SWEEP_OUT = DATA / "blend_3way_sweep.parquet"
OOF_OUT = DATA / "oof_3way_best.parquet"
SUB_OUT = DATA / "submission_3way_best.csv"

TARGET = "PitNextLap"
ID_COL = "id"

CYCLE7_OOF = 0.95408
HURDLE = CYCLE7_OOF + 0.00020  # 0.95428


def load_oof(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "id" not in df.columns:
        raise ValueError(f"{path}: missing 'id' column")
    df = df.set_index("id").sort_index()
    return df


def main() -> None:
    print(f"numpy {version('numpy')}  pandas {version('pandas')}  sklearn {version('scikit-learn')}")

    print("\nLoading OOFs (aligned by id)...")
    m = load_oof(REALMLP_OOF, "realmlp")
    c = load_oof(CB_OOF, "cb_tuned")
    l = load_oof(LGB_OOF, "lgb")

    # Targets must match across all 3
    assert (m["target"] == c["target"]).all(), "target mismatch realmlp vs cb"
    assert (m["target"] == l["target"]).all(), "target mismatch realmlp vs lgb"

    y = m["target"].to_numpy()
    p_realmlp = m["oof"].to_numpy()
    p_cb = c["oof"].to_numpy()
    p_lgb = l["oof"].to_numpy()

    auc_realmlp = roc_auc_score(y, p_realmlp)
    auc_cb = roc_auc_score(y, p_cb)
    auc_lgb = roc_auc_score(y, p_lgb)
    print(f"  RealMLP:  OOF AUC = {auc_realmlp:.5f}")
    print(f"  CB-tuned: OOF AUC = {auc_cb:.5f}")
    print(f"  LGB:      OOF AUC = {auc_lgb:.5f}")

    # Rank correlations
    rs = pd.DataFrame({"realmlp": p_realmlp, "cb": p_cb, "lgb": p_lgb}).rank()
    print(f"  rank correlations:\n{rs.corr().round(4).to_string()}")

    # ===== Coarse sweep =====
    print(f"\nCoarse sweep (cycle 7 baseline = {CYCLE7_OOF:.5f}, hurdle = {HURDLE:.5f})")
    print(f"  {'w_cb':>6}  {'w_lgb':>6}  {'w_realmlp':>10}  {'AUC':>8}  {'Δ vs cycle 7':>13}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*13}")

    rows = []
    for w_cb in [0.10, 0.15, 0.20]:
        for w_lgb in [0.00, 0.025, 0.05, 0.075, 0.10]:
            w_realmlp = 1.0 - w_cb - w_lgb
            if w_realmlp <= 0:
                continue
            p = w_realmlp * p_realmlp + w_cb * p_cb + w_lgb * p_lgb
            auc = roc_auc_score(y, p)
            delta = auc - CYCLE7_OOF
            rows.append({
                "w_realmlp": w_realmlp, "w_cb": w_cb, "w_lgb": w_lgb,
                "auc": auc, "delta_vs_cycle7": delta, "stage": "coarse",
            })
            print(f"  {w_cb:6.3f}  {w_lgb:6.3f}  {w_realmlp:10.3f}  {auc:8.5f}  {delta:+13.5f}")

    df = pd.DataFrame(rows)
    best_coarse = df.loc[df["auc"].idxmax()]
    print(f"\nCoarse best: w_cb={best_coarse['w_cb']:.3f}  w_lgb={best_coarse['w_lgb']:.3f}  "
          f"AUC={best_coarse['auc']:.5f}  Δ={best_coarse['delta_vs_cycle7']:+.5f}")

    # ===== Fine sweep around coarse peak =====
    cb0 = float(best_coarse["w_cb"])
    lgb0 = float(best_coarse["w_lgb"])
    print(f"\nFine sweep around (w_cb={cb0:.3f}, w_lgb={lgb0:.3f}) at 0.0125 step")
    print(f"  {'w_cb':>6}  {'w_lgb':>6}  {'w_realmlp':>10}  {'AUC':>8}  {'Δ vs cycle 7':>13}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*13}")
    fine_rows = []
    for w_cb in np.arange(max(0.0, cb0 - 0.025), cb0 + 0.0251, 0.0125):
        for w_lgb in np.arange(max(0.0, lgb0 - 0.025), lgb0 + 0.0251, 0.0125):
            w_realmlp = 1.0 - w_cb - w_lgb
            if w_realmlp <= 0:
                continue
            p = w_realmlp * p_realmlp + w_cb * p_cb + w_lgb * p_lgb
            auc = roc_auc_score(y, p)
            delta = auc - CYCLE7_OOF
            fine_rows.append({
                "w_realmlp": w_realmlp, "w_cb": float(w_cb), "w_lgb": float(w_lgb),
                "auc": auc, "delta_vs_cycle7": delta, "stage": "fine",
            })
            print(f"  {w_cb:6.4f}  {w_lgb:6.4f}  {w_realmlp:10.4f}  {auc:8.5f}  {delta:+13.5f}")

    df = pd.concat([df, pd.DataFrame(fine_rows)], ignore_index=True)
    df.to_parquet(SWEEP_OUT, index=False)
    print(f"\nWrote {SWEEP_OUT.name}  ({len(df)} sweep rows total)")

    best = df.loc[df["auc"].idxmax()]
    print(f"\nBest overall: w_realmlp={best['w_realmlp']:.4f}  w_cb={best['w_cb']:.4f}  w_lgb={best['w_lgb']:.4f}")
    print(f"            OOF AUC = {best['auc']:.5f}   Δ vs cycle 7 = {best['delta_vs_cycle7']:+.5f}")

    if best["auc"] >= HURDLE:
        print(f"\n✓ Cleared hurdle ({best['auc']:.5f} ≥ {HURDLE:.5f}). Generating submission.")

        # Persist best OOF
        wm, wc, wl = best["w_realmlp"], best["w_cb"], best["w_lgb"]
        pd.DataFrame({
            "id": m.index,
            "Year": m["Year"].values,
            "target": y,
            "oof": wm * p_realmlp + wc * p_cb + wl * p_lgb,
        }).to_parquet(OOF_OUT, index=False)
        print(f"Wrote {OOF_OUT.name}")

        # Test submission
        sub_m = pd.read_csv(REALMLP_SUB).sort_values(ID_COL).reset_index(drop=True)
        sub_c = pd.read_csv(CB_SUB).sort_values(ID_COL).reset_index(drop=True)
        sub_l = pd.read_csv(LGB_SUB).sort_values(ID_COL).reset_index(drop=True)
        assert (sub_m[ID_COL] == sub_c[ID_COL]).all(), "test id mismatch realmlp vs cb"
        assert (sub_m[ID_COL] == sub_l[ID_COL]).all(), "test id mismatch realmlp vs lgb"

        blended = wm * sub_m[TARGET].to_numpy() + wc * sub_c[TARGET].to_numpy() + wl * sub_l[TARGET].to_numpy()
        pd.DataFrame({"id": sub_m[ID_COL], TARGET: blended}).to_csv(SUB_OUT, index=False)
        print(f"Wrote {SUB_OUT.name}")
    elif best["auc"] > CYCLE7_OOF:
        print(f"\n~ 3-way improved over cycle 7 by {best['delta_vs_cycle7']:+.5f} but below hurdle ({HURDLE:.5f}). Inconclusive.")
    else:
        print(f"\n✗ 3-way did NOT beat cycle 7's 2-way. LGB axis adds nothing here. Direction dead.")


if __name__ == "__main__":
    main()
