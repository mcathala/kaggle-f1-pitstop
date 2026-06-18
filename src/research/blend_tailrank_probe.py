"""Experiment 050 (cycle 16 pivot) — tail-gated rank-blend probe.

Motivation (from our own analysis): ROC-AUC depends only on the ordering of
predictions, and is most sensitive to the ranking of the high-confidence
positive tail. Our cycle-10 probe-2 residual EDA localised the worst-loss
quartile to a coherent slice (degraded tyre x losing position x pit-cluster),
which sits in that tail. This probe tests whether re-ranking ONLY the anchor's
tail using a rank-diverse base — blended in percentile space — lifts OOF by
>= +0.00020 over our cycle-11 3-way anchor (0.95420).

Rank space is AUC-invariant to monotonic transforms, so we operate on percentile
ranks directly and read OOF AUC. No model training — pure post-processing.

Sweep: source x tail-quantile x weight. Baseline = anchor AUC (no blend).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

DATA = Path(__file__).resolve().parent.parent.parent / "data"
ANCHOR = "oof_blend_3way_xgb.parquet"
SOURCES = {
    "lgb_highbins": "oof_lgb_highbins.parquet",   # most rank-diverse (rho~0.967 vs RM)
    "xgb_highbins": "oof_xgb_highbins.parquet",
    "cb_tuned14": "oof_cb_tuned_exp14.parquet",
}
TAIL_Q = [0.80, 0.88, 0.92, 0.95, 0.98]   # gate: re-rank rows with anchor-percentile >= q
WEIGHTS = [0.02, 0.05, 0.10, 0.15, 0.20]


def load(p):
    return pd.read_parquet(DATA / p).set_index("id").sort_index()


def pct(x):
    return rankdata(x) / len(x)


def main():
    anchor = load(ANCHOR)
    y = anchor["target"].to_numpy()
    a = anchor["oof"].to_numpy()
    ra = pct(a)
    base_auc = roc_auc_score(y, a)
    print(f"anchor (cycle-11 3-way) OOF AUC = {base_auc:.5f}  (n={len(y):,})\n")

    # rank-corr of each source vs anchor (diversity check)
    print("source rank-corr vs anchor:")
    for name, f in SOURCES.items():
        s = load(f)["oof"].to_numpy()
        rho = np.corrcoef(pct(a), pct(s))[0, 1]
        print(f"  {name:14s} rho={rho:.4f}  solo_auc={roc_auc_score(y, s):.5f}")
    print()

    rows = []
    best = (base_auc, "none", None, None)
    for name, f in SOURCES.items():
        rs = pct(load(f)["oof"].to_numpy())
        for q in TAIL_Q:
            thr = np.quantile(ra, q)
            tail = ra >= thr
            for w in WEIGHTS:
                rb = ra.copy()
                rb[tail] = (1 - w) * ra[tail] + w * rs[tail]
                auc = roc_auc_score(y, rb)
                d = auc - base_auc
                rows.append({"source": name, "tail_q": q, "weight": w,
                             "n_tail": int(tail.sum()), "auc": auc, "delta": d})
                if auc > best[0]:
                    best = (auc, name, q, w)

    res = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    pd.set_option("display.width", 120)
    print("Top 15 tail-gated rank-blend configs:")
    print(res.head(15).to_string(index=False,
          formatters={"auc": "{:.5f}".format, "delta": "{:+.5f}".format}))
    print()
    auc, name, q, w = best
    print(f"BEST: source={name} tail_q={q} weight={w} -> OOF {auc:.5f} "
          f"(delta {auc-base_auc:+.5f} vs anchor; hurdle +0.00020)")
    res.to_parquet(DATA / "blend_tailrank_sweep.parquet", index=False)
    print(f"wrote blend_tailrank_sweep.parquet ({len(res)} configs)")


if __name__ == "__main__":
    main()
