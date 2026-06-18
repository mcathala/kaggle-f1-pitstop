"""Experiment 070 — blend probe with HGBC as 4th GBDT.

Reads up to 7 OOFs (RM6, psRM6, CB-exp14, XGB-highbins, psXGB, HGBC, psRM6_r2 if exists)
and evaluates:
  - 3-way anchor and 3-way pseudo baselines
  - 4-way: anchor + HGBC and pseudo + HGBC
  - Free non-negative grid over all available bases (random search + local refine)

Outputs:
  data/blend_hgbc_sweep.parquet
"""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

DATA = Path(__file__).resolve().parent.parent.parent / "data"

# (name, path, optional) — optional bases skipped if missing
BASES = [
    ("rm6",     DATA / "oof_realmlp_6seed.parquet",       False),
    ("psrm6",   DATA / "oof_realmlp_pseudo6.parquet",     False),
    ("psrm6r2", DATA / "oof_realmlp_pseudo62.parquet",    True),   # exp 069 (round-2)
    ("cb",      DATA / "oof_cb_tuned_exp14.parquet",      False),
    ("xgb",     DATA / "oof_xgb_highbins.parquet",         False),  # canonical XGB-highbins
    ("psxgb",   DATA / "oof_xgb_pseudo.parquet",          False),
    ("psxgb2",  DATA / "oof_xgb_pseudo2.parquet",         True),   # exp 067 (round-2)
    ("hgbc",    DATA / "oof_hgbc.parquet",                False),
]


def load_oofs() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    print("Loading OOFs...")
    target = None
    ids = None
    oofs: dict[str, np.ndarray] = {}
    for name, path, optional in BASES:
        if not path.exists():
            if optional:
                print(f"  {name:8s} (optional, missing — skipped)")
                continue
            raise FileNotFoundError(path)
        df = pd.read_parquet(path).sort_values("id").reset_index(drop=True)
        if ids is None:
            ids = df["id"].to_numpy()
            target = df["target"].to_numpy().astype(int)
        else:
            assert (df["id"].to_numpy() == ids).all(), f"id mismatch in {name}"
            assert (df["target"].to_numpy().astype(int) == target).all(), f"target mismatch in {name}"
        oofs[name] = df["oof"].to_numpy()
        print(f"  {name:8s}  AUC={roc_auc_score(target, oofs[name]):.5f}  n={len(df):,}")
    return target, oofs


def rho_table(oofs: dict[str, np.ndarray]) -> None:
    names = list(oofs.keys())
    print("\nRank-correlation matrix:")
    print("        " + "  ".join(f"{n:>8s}" for n in names))
    for i, a in enumerate(names):
        row = []
        for j, b in enumerate(names):
            if j < i:
                row.append("       .")
            else:
                rho, _ = spearmanr(oofs[a], oofs[b])
                row.append(f"{rho:8.4f}")
        print(f"{a:>8s}  " + "  ".join(row))


def blend_auc(weights: dict[str, float], y: np.ndarray, oofs: dict[str, np.ndarray]) -> float:
    w_arr = np.array([weights.get(n, 0.0) for n in oofs.keys()], dtype=np.float64)
    if w_arr.sum() <= 0:
        return float("nan")
    w_arr = w_arr / w_arr.sum()
    pred = np.zeros_like(next(iter(oofs.values())))
    for w, name in zip(w_arr, oofs.keys()):
        if w > 0:
            pred = pred + w * oofs[name]
    return float(roc_auc_score(y, pred))


def coord_descent(w0: dict[str, float], y: np.ndarray, oofs: dict[str, np.ndarray], step: float = 0.01, n_passes: int = 60) -> tuple[dict[str, float], float]:
    """Cheap coordinate-descent refinement around an initial weight dict (in-place +/- step)."""
    names = list(oofs.keys())
    X = np.stack([oofs[n] for n in names], axis=1)
    w = np.array([w0.get(n, 0.0) for n in names], dtype=np.float64)
    s = w.sum()
    if s <= 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / s
    best_auc = roc_auc_score(y, X @ w)
    for _ in range(n_passes):
        improved = False
        for i in range(len(names)):
            for delta in (step, -step, step * 3, -step * 3):
                w_try = w.copy()
                w_try[i] = max(0.0, w_try[i] + delta)
                s = w_try.sum()
                if s <= 0:
                    continue
                w_try = w_try / s
                auc_try = roc_auc_score(y, X @ w_try)
                if auc_try > best_auc + 1e-7:
                    best_auc = auc_try
                    w = w_try
                    improved = True
        if not improved:
            break
    return dict(zip(names, w.tolist())), best_auc


def main() -> None:
    y, oofs = load_oofs()
    rho_table(oofs)

    print("\n--- Fixed-weight reference blends ---")
    refs = []

    # 3-way anchor
    if all(n in oofs for n in ["rm6", "cb", "xgb"]):
        w = {"rm6": 0.675, "cb": 0.075, "xgb": 0.250}
        auc = blend_auc(w, y, oofs)
        refs.append(("3way_anchor (RM6/CB/XGB 0.675/0.075/0.250)", auc, w))

    # 3-way pseudo
    if all(n in oofs for n in ["psrm6", "cb", "psxgb"]):
        w = {"psrm6": 0.675, "cb": 0.075, "psxgb": 0.250}
        auc = blend_auc(w, y, oofs)
        refs.append(("3way_pseudo (psRM6/CB/psXGB 0.675/0.075/0.250)", auc, w))

    # 4-way HGBC-added
    if all(n in oofs for n in ["rm6", "cb", "xgb", "hgbc"]):
        for w_hgbc in [0.05, 0.10, 0.15, 0.20]:
            rest = 1.0 - w_hgbc
            w = {"rm6": 0.675 * rest, "cb": 0.075 * rest, "xgb": 0.250 * rest, "hgbc": w_hgbc}
            auc = blend_auc(w, y, oofs)
            refs.append((f"4way RM6+CB+XGB+HGBC (w_hgbc={w_hgbc})", auc, w))

    if all(n in oofs for n in ["psrm6", "cb", "psxgb", "hgbc"]):
        for w_hgbc in [0.05, 0.10, 0.15, 0.20]:
            rest = 1.0 - w_hgbc
            w = {"psrm6": 0.675 * rest, "cb": 0.075 * rest, "psxgb": 0.250 * rest, "hgbc": w_hgbc}
            auc = blend_auc(w, y, oofs)
            refs.append((f"4way psRM6+CB+psXGB+HGBC (w_hgbc={w_hgbc})", auc, w))

    # Round-2 pseudo-RM (exp 069): 3-way swap and 4-way merge
    if "psrm6r2" in oofs:
        if all(n in oofs for n in ["psrm6r2", "cb", "psxgb"]):
            w = {"psrm6r2": 0.675, "cb": 0.075, "psxgb": 0.250}
            refs.append(("3way_pseudo_r2 (psRM6r2/CB/psXGB)", blend_auc(w, y, oofs), w))
        if all(n in oofs for n in ["psrm6", "psrm6r2", "cb", "psxgb"]):
            for split in [0.5, 0.7, 0.3]:
                w = {"psrm6": 0.675 * split, "psrm6r2": 0.675 * (1 - split), "cb": 0.075, "psxgb": 0.250}
                refs.append((f"4way psRM6+psRM6r2 (split={split:.1f})+CB+psXGB", blend_auc(w, y, oofs), w))

    # Round-2 pseudo-XGB (exp 067): symmetric swap and merge
    if "psxgb2" in oofs:
        if all(n in oofs for n in ["psrm6", "cb", "psxgb2"]):
            w = {"psrm6": 0.675, "cb": 0.075, "psxgb2": 0.250}
            refs.append(("3way swap psXGB→psXGB2", blend_auc(w, y, oofs), w))

    for name, auc, w in refs:
        print(f"  {name:60s}  OOF={auc:.5f}")

    print("\n--- Coordinate-descent refinement starting from best fixed weights ---")
    best_ref = max(refs, key=lambda x: x[1])
    print(f"  start: {best_ref[0]}  OOF={best_ref[1]:.5f}")
    w_opt, auc_opt = coord_descent(best_ref[2], y, oofs, step=0.01, n_passes=60)
    print(f"  refined OOF: {auc_opt:.5f}")
    print(f"  weights : " + "  ".join(f"{k}={v:.3f}" for k, v in sorted(w_opt.items(), key=lambda x: -x[1]) if v >= 1e-3))

    rows = []
    for name, auc, w in refs:
        rows.append({"config": name, "auc": auc, "weights": w})
    rows.append({"config": "coord_descent_refined", "auc": auc_opt, "weights": w_opt})
    out = pd.DataFrame(rows)
    out_path = DATA / "blend_hgbc_sweep.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\nwrote {out_path.name}")


if __name__ == "__main__":
    main()
