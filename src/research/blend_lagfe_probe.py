"""Experiment 051 blend probe — does the lag-FE XGB lift the blend?

Lag-XGB (exp 051) is marginally weaker standalone than plain XGB-highbins
(fold-1 0.95311 vs 0.95331) but more rank-diverse from RealMLP (rho 0.980 vs
0.984). The standalone gate is the wrong gate; the question is whether the added
diversity lifts the 3-way blend OOF (0.95421) by >= +0.00020.

Tests four configs against the cycle-11 anchor:
  A  swap:    RM + CB + lag-XGB           at cycle-11 ratios (0.675/0.075/0.250)
  B  4-way:   RM + CB + XGB + lag-XGB     (split the 0.250 XGB slot, sweep the split)
  C  xgb-avg: RM + CB + 0.5*(XGB+lagXGB)  (two-FE XGB average in the 0.250 slot)
  D  free 4-way grid:  coarse (w_cb, w_xgb, w_lag) with w_rm = 1 - rest

Anchor recorded OOF = 0.95421; hurdle = +0.00020 -> 0.95441.
"""

from pathlib import Path
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DATA = Path(__file__).resolve().parent.parent.parent / "data"
RM = "oof_realmlp_multiseed.parquet"
CB = "oof_cb_tuned_exp14.parquet"
XGB = "oof_xgb_highbins.parquet"
LAG = "oof_xgb_lagfe.parquet"

ANCHOR_OOF = 0.95421
HURDLE = ANCHOR_OOF + 0.00020
W11 = {"rm": 0.675, "cb": 0.075, "xgb": 0.250}


def load(p):
    return pd.read_parquet(DATA / p).set_index("id").sort_index()


def main():
    rm, cb, xg, lag = load(RM), load(CB), load(XGB), load(LAG)
    y = rm["target"].to_numpy()
    for nm, d in [("CB", cb), ("XGB", xg), ("LAG", lag)]:
        assert (rm["target"] == d["target"]).all(), f"target mismatch RM vs {nm}"
    p_rm, p_cb, p_xg, p_lag = (d["oof"].to_numpy() for d in (rm, cb, xg, lag))

    auc = lambda p: roc_auc_score(y, p)
    print(f"solo:  RM={auc(p_rm):.5f}  CB={auc(p_cb):.5f}  XGB={auc(p_xg):.5f}  LAG={auc(p_lag):.5f}")

    # rank-corr matrix
    rk = pd.DataFrame({"rm": p_rm, "cb": p_cb, "xgb": p_xg, "lag": p_lag}).rank()
    print("\nrank-corr:\n" + rk.corr().round(4).to_string())

    anchor = W11["rm"] * p_rm + W11["cb"] * p_cb + W11["xgb"] * p_xg
    print(f"\nanchor (cycle-11 3-way) OOF = {auc(anchor):.5f}  (recorded {ANCHOR_OOF}, hurdle {HURDLE:.5f})")

    rows = []

    # A — straight swap (lag-XGB replaces plain XGB at cycle-11 ratios)
    pA = W11["rm"] * p_rm + W11["cb"] * p_cb + W11["xgb"] * p_lag
    rows.append(("A swap lag->xgb", auc(pA)))

    # B — 4-way, split the 0.250 XGB slot between plain and lag
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        w_xgb = W11["xgb"] * (1 - frac)
        w_lag = W11["xgb"] * frac
        p = W11["rm"] * p_rm + W11["cb"] * p_cb + w_xgb * p_xg + w_lag * p_lag
        rows.append((f"B 4way xgb={w_xgb:.3f} lag={w_lag:.3f}", auc(p)))

    # C — two-FE XGB average in the slot
    pC = W11["rm"] * p_rm + W11["cb"] * p_cb + W11["xgb"] * (0.5 * p_xg + 0.5 * p_lag)
    rows.append(("C xgb-avg(plain,lag)", auc(pC)))

    # D — coarse free 4-way grid (w_rm = 1 - rest)
    best_d = (0.0, None)
    grid = [round(x, 3) for x in np.arange(0.0, 0.41, 0.05)]
    for w_cb, w_xgb, w_lag in itertools.product([0.0, 0.05, 0.075, 0.10], grid, grid):
        w_rm = 1.0 - w_cb - w_xgb - w_lag
        if w_rm < 0.4:
            continue
        p = w_rm * p_rm + w_cb * p_cb + w_xgb * p_xg + w_lag * p_lag
        a = auc(p)
        if a > best_d[0]:
            best_d = (a, (round(w_rm, 3), w_cb, w_xgb, w_lag))

    print("\nconfig results:")
    for name, a in rows:
        print(f"  {name:32s} OOF={a:.5f}  Δ={a-ANCHOR_OOF:+.5f}")
    print(f"  {'D best free 4-way':32s} OOF={best_d[0]:.5f}  Δ={best_d[0]-ANCHOR_OOF:+.5f}  w(rm,cb,xgb,lag)={best_d[1]}")

    best_overall = max([a for _, a in rows] + [best_d[0]])
    verdict = "KEEP" if best_overall >= HURDLE else ("MARGINAL" if best_overall > ANCHOR_OOF + 0.00005 else "no-lift")
    print(f"\nbest OOF = {best_overall:.5f}  (anchor {ANCHOR_OOF}, hurdle {HURDLE:.5f}) -> {verdict}")


if __name__ == "__main__":
    main()
