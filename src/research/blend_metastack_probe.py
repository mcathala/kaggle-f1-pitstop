"""Experiment 061 — honest cross-fitted meta-blend over the full base zoo.

After the diverse-base hunt closed (lap-attention 058/059/060), this asks the
zero-GPU question: does any smarter combiner — or adding the genuinely-diverse
(but weak) lap/embMLP bases — beat the linear 3-way (OOF 0.95420)?

Honest evaluation: weights are CROSS-FITTED (fit on 4/5 folds of the OOF, predict
the held-out fold), so we don't reward OOF-overfitting the way the OOF-tuned
linear weights do. Tries NNLS, logistic stacking, and rank-averaging over several
base sets.

Result (2026-05-27): all NNLS combos = 0.95417 (diverse bases get ~0 weight);
logistic = 0.95358 (overfits); rank-avg = 0.95418. None beats the linear 3-way.
The blend-combiner lever is closed with the current bases.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA = Path(__file__).resolve().parent.parent.parent / "data"
BASES = {
    "rm": "oof_realmlp_multiseed.parquet", "xgb": "oof_xgb_highbins.parquet", "cb": "oof_cb_tuned_exp14.parquet",
    "lap1": "oof_lap_attention.parquet", "lap3": "oof_lap_attention_v3.parquet", "emb": "oof_embmlp.parquet",
}


def main():
    ref = pd.read_parquet(DATA / BASES["rm"]).set_index("id").sort_index()
    y, yr = ref["target"].to_numpy(), ref["Year"].to_numpy()
    P = {k: np.nan_to_num(pd.read_parquet(DATA / v).set_index("id").sort_index()["oof"].to_numpy())
         for k, v in BASES.items()}
    auc = lambda p: roc_auc_score(y, p)
    print(f"linear 3-way anchor: {auc(0.675*P['rm']+0.075*P['cb']+0.250*P['xgb']):.5f}")

    strat = np.char.add(yr.astype(str), np.char.add("_", y.astype(str)))
    skf = StratifiedKFold(5, shuffle=True, random_state=42)

    def crossfit(keys, method):
        M = np.column_stack([P[k] for k in keys]); oof = np.zeros(len(y))
        for tr, va in skf.split(M, strat):
            if method == "nnls":
                w, _ = nnls(M[tr], y[tr]); oof[va] = M[va] @ w
            else:
                lr = LogisticRegression(C=1.0, max_iter=2000).fit(M[tr], y[tr]); oof[va] = lr.predict_proba(M[va])[:, 1]
        return auc(oof)

    print("\ncross-fitted NNLS:")
    for keys in [("rm", "cb", "xgb"), ("rm", "cb", "xgb", "lap1"),
                 ("rm", "cb", "xgb", "lap1", "lap3", "emb"), ("rm", "xgb", "lap1", "emb")]:
        print(f"  {'+'.join(keys):28s} {crossfit(keys, 'nnls'):.5f}")
    print("\ncross-fitted logistic:")
    for keys in [("rm", "cb", "xgb"), ("rm", "cb", "xgb", "lap1", "lap3", "emb")]:
        print(f"  {'+'.join(keys):28s} {crossfit(keys, 'logit'):.5f}")
    rk = 0.675*rankdata(P["rm"]) + 0.075*rankdata(P["cb"]) + 0.250*rankdata(P["xgb"])
    print(f"\nrank-avg 3-way: {auc(rk):.5f}")


if __name__ == "__main__":
    main()
