"""A1 — honest, overfit-aware blend-weight selection.

Three questions this answers:
  1. How inflated is our in-sample coord-descent blend OOF vs a nested-CV honest
     estimate? (the in-sample bias we've been reporting)
  2. What does the clean psXGB2 de-leak swap (oof_xgb_pseudo -> oof_xgb_pseudo2)
     do to the honest OOF?
  3. Robust (bagged) weights for the actual submission, lower-variance than a
     single full-data coord-descent fit.

CV here is over the 439k OOF *rows* (the bases are already fold-honest); it
estimates blend-weight overfit, which the single full-data fit cannot see.

Usage: .venv/bin/python src/blend_nested_weights.py
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA = Path(__file__).resolve().parent.parent.parent / "data"

# (name, oof, submission). psxgb is the LEAKY round-1; psxgb2 is the clean swap.
POOL_LEAKY = [
    ("psrm6r2",   "oof_realmlp_pseudo62.parquet",     "submission_realmlp_pseudo62.csv"),
    ("diffrm6",   "oof_realmlp_diffFE_6seed.parquet", "submission_realmlp_diffFE_6seed.csv"),
    ("diffpsxgb", "oof_xgb_diffFE_pseudo.parquet",    "submission_xgb_diffFE_pseudo.csv"),
    ("psxgb",     "oof_xgb_pseudo.parquet",           "submission_xgb_pseudo.csv"),
    ("diffxgb",   "oof_xgb_diffFE.parquet",           "submission_xgb_diffFE.csv"),
    ("cbdiff",    "oof_cb_diffFE.parquet",            "submission_cb_diffFE.csv"),
    ("pscb14",    "oof_cb_pseudo_exp14.parquet",      "submission_cb_pseudo_exp14.csv"),
]
# clean variant: swap the leaky psxgb round-1 for the de-leaked round-2.
SWAP = {"psxgb": ("psxgb2", "oof_xgb_pseudo2.parquet", "submission_xgb_pseudo2.csv")}

STEPS = (0.01, -0.01, 0.03, -0.03, 0.005, -0.005)


def load(pool):
    oofs, subs, names = {}, {}, []
    y = ids = sids = None
    for n, of, sf in pool:
        if not (DATA / of).exists() or not (DATA / sf).exists():
            print(f"  skip {n}: missing"); continue
        d = pd.read_parquet(DATA / of).sort_values("id").reset_index(drop=True)
        if y is None:
            y = d["target"].astype(int).to_numpy(); ids = d["id"].to_numpy()
        elif not (d["id"].to_numpy() == ids).all():
            print(f"  skip {n}: id mismatch"); continue
        s = pd.read_csv(DATA / sf).sort_values("id").reset_index(drop=True)
        if sids is None: sids = s["id"].to_numpy()
        oofs[n] = d["oof"].to_numpy(); subs[n] = s["PitNextLap"].to_numpy(); names.append(n)
    X = np.stack([oofs[n] for n in names], 1)
    S = np.stack([subs[n] for n in names], 1)
    return names, X, S, y, sids


def coord_descent(X, y, w0, n_passes=600):
    w = w0 / w0.sum() if w0.sum() > 0 else np.ones(X.shape[1]) / X.shape[1]
    best = roc_auc_score(y, X @ w)
    for _ in range(n_passes):
        imp = False
        for i in range(len(w)):
            for dl in STEPS:
                t = w.copy(); t[i] = max(0.0, t[i] + dl); ss = t.sum()
                if ss <= 0: continue
                t = t / ss; a = roc_auc_score(y, X @ t)
                if a > best + 1e-7: best = a; w = t; imp = True
        if not imp: break
    return w, best


def nested_oof(X, y, w0, n_splits=5, seed=42):
    """Fit weights on each train fold, score on held-out fold. Honest blend OOF."""
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    held = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        w, _ = coord_descent(X[tr], y[tr], w0.copy())
        held[va] = X[va] @ w
    return roc_auc_score(y, held)


def bagged_weights(X, y, w0, n_boot=20, frac=0.5, seed=42):
    rng = np.random.default_rng(seed)
    acc = np.zeros(X.shape[1])
    for b in range(n_boot):
        idx = rng.choice(len(y), int(frac * len(y)), replace=True)
        w, _ = coord_descent(X[idx], y[idx], w0.copy())
        acc += w
    return acc / acc.sum()


def run(tag, pool):
    print(f"\n===== {tag} =====")
    names, X, S, y, sids = load(pool)
    for i, n in enumerate(names):
        print(f"  {n:10s} OOF {roc_auc_score(y, X[:, i]):.5f}")
    # init spread across RM views + strong XGB (same spirit as build_best_blend)
    base_init = {"psrm6r2": 0.31, "diffrm6": 0.31, "diffpsxgb": 0.22,
                 "psxgb": 0.06, "psxgb2": 0.06, "cbdiff": 0.03, "pscb14": 0.02}
    w0 = np.array([base_init.get(n, 0.0) for n in names])
    if w0.sum() == 0: w0 = np.ones(len(names))
    w_full, auc_insample = coord_descent(X, y, w0.copy())
    auc_honest = nested_oof(X, y, w0.copy())
    w_bag = bagged_weights(X, y, w0.copy())
    auc_bag_full = roc_auc_score(y, X @ w_bag)
    wd_full = {n: round(float(w_full[i]), 3) for i, n in enumerate(names) if w_full[i] > 1e-3}
    wd_bag = {n: round(float(w_bag[i]), 3) for i, n in enumerate(names) if w_bag[i] > 1e-3}
    print(f"  in-sample coord-descent OOF : {auc_insample:.5f}   (what we've been reporting)")
    print(f"  nested-CV honest OOF        : {auc_honest:.5f}   (overfit = {auc_insample-auc_honest:+.5f})")
    print(f"  bagged-weights full OOF     : {auc_bag_full:.5f}")
    print(f"  full-fit weights : {wd_full}")
    print(f"  bagged   weights : {wd_bag}")
    return names, S, sids, w_bag


def main():
    run("LEAKY pool (current best lineage)", POOL_LEAKY)
    clean_pool = [SWAP[n] if n in SWAP else (n, of, sf) for (n, of, sf) in POOL_LEAKY]
    names, S, sids, w_bag = run("CLEAN pool (psXGB2 de-leak swap)", clean_pool)
    sub = S @ w_bag
    out = DATA / "submission_blend_nested_clean.csv"
    pd.DataFrame({"id": sids, "PitNextLap": sub}).sort_values("id").reset_index(drop=True).to_csv(out, index=False)
    print(f"\nwrote {out.name}  (clean pool, bagged-robust weights)")


if __name__ == "__main__":
    main()
