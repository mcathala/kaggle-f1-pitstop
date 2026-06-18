"""Caruana greedy ensemble selection over the full base library.

Our hand-tuned 7-member coord-descent blend plateaued at OOF 0.95462 and we'd
concluded the blend was saturated. Re-auditing with greedy ensemble selection
(Caruana 2004) over ALL alignable base OOFs found OOF 0.95479 — and a nested
5-fold check (weights chosen on 4 folds, scored on the held fold) confirmed the
gain generalizes (held-out 0.95474 vs in-sample 0.95479, ~no overfit). The gain
comes almost entirely from including the self-distilled RealMLP at a modest weight:
it transfers poorly *standalone* (worst LB residual in our history) but adds OOF
value as a *component*. Whether that survives LB transfer is the open question the
greedy_full vs greedy_nosd submissions test.

Builds:
  data/submission_blend_greedy_full.csv   greedy over all bases (incl self-distill)
  data/submission_blend_greedy_nosd.csv   greedy with self-distill excluded (transfer hedge)

Derived blends (blend_/stack_/rankblend_) are excluded from the candidate pool to
avoid circular selection. A base needs BOTH an oof_*.parquet and a submission_*.csv.

Usage: .venv/bin/python src/build_greedy_blend.py
"""
import glob
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA = Path(__file__).resolve().parent.parent.parent / "data"
REF = DATA / "oof_realmlp_pseudo62.parquet"
N_ITER = 150
INIT_K = 3


def _pct(a):
    """Rank-percentile to [0,1] — makes a non-probability base (e.g. rank:pairwise
    scores) blendable on the same scale as probabilities without changing its AUC."""
    from scipy.stats import rankdata
    return rankdata(a) / len(a)


def load_pool():
    ref = pd.read_parquet(REF).sort_values("id").reset_index(drop=True)
    ids = ref["id"].to_numpy()
    y = ref["target"].astype(int).to_numpy()
    cands = {}
    norm = set()  # bases rank-normalized because they're outside [0,1]
    for p in sorted(glob.glob(str(DATA / "oof_*.parquet"))):
        name = os.path.basename(p)[4:-8]
        if name.startswith(("blend", "stack")) or "rankblend" in name:
            continue
        if not (DATA / f"submission_{name}.csv").exists():
            continue
        d = pd.read_parquet(p).sort_values("id").reset_index(drop=True)
        if len(d) != len(ref) or not (d["id"].to_numpy() == ids).all():
            continue
        o = d["oof"].to_numpy()
        if np.isnan(o).any():
            continue
        if o.min() < -1e-6 or o.max() > 1 + 1e-6:   # not a probability (e.g. rank scores)
            o = _pct(o)
            norm.add(name)
        cands[name] = o
    names = list(cands)
    P = np.column_stack([cands[n] for n in names])
    return names, P, y, ids, norm


def greedy(P, y, cols, n_iter=N_ITER, init_k=INIT_K):
    """Greedy selection-with-replacement over column indices `cols`. Returns full-width weights."""
    M = P[:, cols]
    a = [roc_auc_score(y, M[:, j]) for j in range(M.shape[1])]
    order = np.argsort(a)[::-1]
    ens = list(order[:init_k])
    cur = M[:, ens].mean(1)
    best = roc_auc_score(y, cur)
    for _ in range(n_iter):
        bn, ba = None, best
        for j in range(M.shape[1]):
            c = (cur * len(ens) + M[:, j]) / (len(ens) + 1)
            s = roc_auc_score(y, c)
            if s > ba:
                ba, bn = s, j
        if bn is None:
            break
        ens.append(bn)
        cur = (cur * (len(ens) - 1) + M[:, bn]) / len(ens)
        best = ba
    w = np.zeros(P.shape[1])
    for j, c in Counter(ens).items():
        w[cols[j]] = c
    return w / w.sum(), best


def nested_check(P, y, cols):
    kf = StratifiedKFold(5, shuffle=True, random_state=0)
    held = np.zeros(len(y))
    for tr, va in kf.split(P, y):
        a = [roc_auc_score(y[tr], P[tr, j]) for j in cols]
        order = [cols[i] for i in np.argsort(a)[::-1]]
        ens = order[:INIT_K]
        cur = P[tr][:, ens].mean(1)
        best = roc_auc_score(y[tr], cur)
        for _ in range(N_ITER):
            bn, ba = None, best
            for j in cols:
                c = (cur * len(ens) + P[tr, j]) / (len(ens) + 1)
                s = roc_auc_score(y[tr], c)
                if s > ba:
                    ba, bn = s, j
            if bn is None:
                break
            ens.append(bn)
            cur = (cur * (len(ens) - 1) + P[tr, bn]) / len(ens)
            best = ba
        w = np.zeros(P.shape[1])
        for j, c in Counter(ens).items():
            w[j] = c
        held[va] = P[va] @ (w / w.sum())
    return roc_auc_score(y, held)


def bagged_greedy(P, y, cols, n_bags=20, frac=0.6, seed=0):
    """Average greedy weights over random candidate subsets — Caruana's overfit guard.
    Lower OOF than point-greedy but less OOF-overfit, so a better TRANSFER bet
    (our self-distill finding: OOF-overfit components transfer worse)."""
    rng = np.random.default_rng(seed)
    cols = np.asarray(cols)
    W = np.zeros(P.shape[1])
    for _ in range(n_bags):
        sub = rng.choice(cols, size=max(3, int(frac * len(cols))), replace=False)
        w, _ = greedy(P, y, list(sub))
        W += w
    W /= W.sum()
    return W, roc_auc_score(y, P @ W)


def build_submission(names, w, ids_sub, out, norm=frozenset()):
    sb = None
    for j, wt in enumerate(w):
        if wt <= 0:
            continue
        s = pd.read_csv(DATA / f"submission_{names[j]}.csv").sort_values("id").reset_index(drop=True)
        col = "PitNextLap" if "PitNextLap" in s.columns else s.columns[-1]
        v = s[col].to_numpy()
        if names[j] in norm:        # same rank-normalization applied to its OOF
            v = _pct(v)
        sb = wt * v if sb is None else sb + wt * v
    pd.DataFrame({"id": ids_sub, "PitNextLap": sb}).to_csv(DATA / out, index=False)


def main():
    names, P, y, ids, norm = load_pool()
    if norm:
        print(f"rank-normalized (non-probability) bases: {sorted(norm)}")
    print(f"{len(names)} bases with both OOF and submission")
    s0 = pd.read_csv(DATA / f"submission_{names[0]}.csv").sort_values("id").reset_index(drop=True)
    ids_sub = s0["id"].to_numpy()

    all_cols = list(range(len(names)))
    w_full, auc_full = greedy(P, y, all_cols)
    print(f"\n[greedy_full] OOF={auc_full:.5f}  nested-held={nested_check(P, y, all_cols):.5f}")
    for j in np.argsort(w_full)[::-1]:
        if w_full[j] > 0:
            print(f"    {names[j]:26s}{w_full[j]:7.4f}")
    build_submission(names, w_full, ids_sub, "submission_blend_greedy_full.csv", norm)

    nosd_cols = [j for j, n in enumerate(names) if "selfdistill" not in n]
    w_nosd, auc_nosd = greedy(P, y, nosd_cols)
    print(f"\n[greedy_nosd] OOF={auc_nosd:.5f}  (self-distill excluded)")
    for j in np.argsort(w_nosd)[::-1]:
        if w_nosd[j] > 0:
            print(f"    {names[j]:26s}{w_nosd[j]:7.4f}")
    build_submission(names, w_nosd, ids_sub, "submission_blend_greedy_nosd.csv", norm)

    w_bag, auc_bag = bagged_greedy(P, y, all_cols)
    print(f"\n[bagged_greedy] OOF={auc_bag:.5f}  (20 bags x 60% subsets — transfer-robust)")
    for j in np.argsort(w_bag)[::-1][:12]:
        if w_bag[j] > 0.01:
            print(f"    {names[j]:26s}{w_bag[j]:7.4f}")
    build_submission(names, w_bag, ids_sub, "submission_blend_bagged_greedy.csv", norm)
    print("\nwrote greedy_full, greedy_nosd, bagged_greedy")


if __name__ == "__main__":
    main()
