"""Audit P1-#4 — operator-family probes on the existing 8-base zoo.

Tests whether non-linear blend operators (rank-remap, logit-rank, geometric mean)
extract a small lift over the linear Nelder-Mead optimum we've been using.

Bases (all in `data/`):
  rm6, psrm6, psrm6r2, cb, xgb_highbins, psxgb, psxgb2, hgbc, selfdistill

For each operator, we coord-descent from the current best linear weights and
record the per-operator optimum + the OOF lift over the anchor (0.95436).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

DATA = Path(__file__).resolve().parent.parent.parent / "data"

BASES = [
    ("rm6",         "oof_realmlp_6seed.parquet"),
    ("psrm6",       "oof_realmlp_pseudo6.parquet"),
    ("psrm6r2",     "oof_realmlp_pseudo62.parquet"),
    ("cb",          "oof_cb_tuned_exp14.parquet"),
    ("xgb",         "oof_xgb_highbins.parquet"),
    ("psxgb",       "oof_xgb_pseudo.parquet"),
    ("psxgb2",      "oof_xgb_pseudo2.parquet"),
    ("hgbc",        "oof_hgbc.parquet"),
    ("selfdistill", "oof_realmlp_selfdistill_s42.parquet"),
    ("pscb_exp14",  "oof_cb_pseudo_exp14.parquet"),
]

ANCHOR_WEIGHTS = {"psrm6r2": 0.675, "cb": 0.075, "psxgb": 0.250}  # OOF 0.95436 starting point


def to_rank(x: np.ndarray) -> np.ndarray:
    return rankdata(x, method="average") / (len(x) - 1)


def to_logit_rank(x: np.ndarray) -> np.ndarray:
    r = np.clip(to_rank(x), 1e-6, 1 - 1e-6)
    return np.log(r / (1 - r))


def remap_to_anchor(x: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    rx = rankdata(x, method="average").astype(np.int64) - 1
    sorted_anchor = np.sort(anchor)
    return sorted_anchor[rx]


def auc_blend(weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    if weights.sum() <= 0:
        return float("nan")
    w = weights / weights.sum()
    return float(roc_auc_score(y, X @ w))


def coord_descent(w0: np.ndarray, X: np.ndarray, y: np.ndarray, step: float = 0.01, n_passes: int = 50) -> tuple[np.ndarray, float]:
    w = w0.copy().astype(np.float64)
    s = w.sum()
    if s > 0:
        w = w / s
    else:
        w = np.ones_like(w) / len(w)
    best = auc_blend(w, X, y)
    for _ in range(n_passes):
        improved = False
        for i in range(len(w)):
            for delta in (step, -step, step * 3, -step * 3):
                w_try = w.copy()
                w_try[i] = max(0.0, w_try[i] + delta)
                s = w_try.sum()
                if s <= 0:
                    continue
                w_try = w_try / s
                a = auc_blend(w_try, X, y)
                if a > best + 1e-7:
                    best = a; w = w_try; improved = True
        if not improved:
            break
    return w, best


def main():
    print("Loading 9-base zoo...")
    cols = []
    target = None; ids = None
    for name, path in BASES:
        p = DATA / path
        if not p.exists():
            print(f"  {name:14s}  (missing — skipped)")
            continue
        df = pd.read_parquet(p).sort_values("id").reset_index(drop=True)
        if ids is None:
            ids = df["id"].to_numpy()
            target = df["target"].astype(int).to_numpy()
        else:
            assert (df["id"].to_numpy() == ids).all()
        cols.append((name, df["oof"].to_numpy()))
        print(f"  {name:14s}  AUC={roc_auc_score(target, df['oof']):.5f}")
    y = target
    names = [n for n, _ in cols]
    X_lin = np.stack([v for _, v in cols], axis=1)
    X_rank = np.stack([to_rank(v) for _, v in cols], axis=1)
    X_logit = np.stack([to_logit_rank(v) for _, v in cols], axis=1)
    # Power-mean (geometric) — use clipped log-space then exp at end. Coord-desc
    # on weights as if linear on log-clipped values is equivalent to geometric
    # weighted mean.
    X_log = np.log(np.clip(X_lin, 1e-6, 1 - 1e-6))
    # For anchor-remap: pick the current best fixed-weight blend as the
    # "anchor distribution" target.
    anchor_oof = sum(ANCHOR_WEIGHTS.get(n, 0.0) * v for n, v in cols)
    X_remap = np.stack([remap_to_anchor(v, anchor_oof) for _, v in cols], axis=1)

    # Starting weights — current best anchor in our 9-base space, padded with zeros
    w0 = np.array([ANCHOR_WEIGHTS.get(n, 0.0) for n in names], dtype=np.float64)
    w0 = w0 / w0.sum()

    operators = {
        "linear":   X_lin,
        "rank_avg": X_rank,
        "logit":    X_logit,
        "remap":    X_remap,
        "gmean":    X_log,  # special handling below
    }
    anchor_auc = roc_auc_score(y, X_lin @ w0)
    print(f"\nAnchor (linear, w={dict(zip(names, w0.round(3)))}): OOF AUC = {anchor_auc:.5f}")

    rows = []
    for op_name, Xop in operators.items():
        if op_name == "gmean":
            # geometric mean: predictions = exp(Xop @ w); AUC invariant to exp
            def auc_g(w):
                if w.sum() <= 0: return float("nan")
                return roc_auc_score(y, Xop @ (w / w.sum()))
            # Use coord-desc using the auc_g closure via a thin wrapper
            X = Xop  # AUC of weighted sum of logs equals AUC of geometric mean
            w_opt, auc_opt = coord_descent(w0, X, y, step=0.01, n_passes=50)
        else:
            w_opt, auc_opt = coord_descent(w0, Xop, y, step=0.01, n_passes=50)
        delta = auc_opt - anchor_auc
        active = {n: float(w_opt[i]) for i, n in enumerate(names) if w_opt[i] > 1e-3}
        rows.append({"operator": op_name, "auc": auc_opt, "delta": delta, "weights": active})
        print(f"\n  {op_name:8s}: AUC={auc_opt:.5f}  Δ {delta:+.5f}")
        print(f"            weights = {active}")

    out = pd.DataFrame(rows)
    out_path = DATA / "blend_operator_sweep.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\nwrote {out_path.name}")

    # If any operator beats linear by ≥+0.0001 OOF, flag it
    best = max(rows, key=lambda r: r["auc"])
    if best["auc"] - anchor_auc >= 0.0001:
        print(f"\n*** Best operator '{best['operator']}' clears anchor by Δ {best['delta']:+.5f} OOF.")
        print(f"    (Reminder: OOF lifts < +0.0005 transfer poorly per exp 075 — do not auto-submit.)")
    else:
        print(f"\n  All operators within Δ +0.0001 of linear anchor. Linear remains optimal — closes the operator-family lever.")


if __name__ == "__main__":
    main()
