"""Rebuild the best CLEAN blend from all available strong bases (free coord-descent).

Picks up whichever bases exist on disk (diffFE 6-seed RM, view-3 RM, etc. as they
land). Reports OOF + weights and writes submission_blend_best.csv.

Usage: .venv/bin/python src/build_best_blend.py
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

DATA = Path(__file__).resolve().parent.parent.parent / "data"
# (name, oof, submission) — skipped if missing
POOL = [
    ("psrm6r2",    "oof_realmlp_pseudo62.parquet",      "submission_realmlp_pseudo62.csv"),
    ("diffrm1",    "oof_realmlp_diffFE_s42.parquet",    "submission_realmlp_diffFE_s42.csv"),
    ("diffrm6",    "oof_realmlp_diffFE_6seed.parquet",  "submission_realmlp_diffFE_6seed.csv"),
    ("rmview3",    "oof_realmlp_view3_6seed.parquet",   "submission_realmlp_view3_6seed.csv"),
    ("rm6",        "oof_realmlp_6seed.parquet",         "submission_realmlp_multiseed.csv"),
    ("diffpsxgb",  "oof_xgb_diffFE_pseudo.parquet",     "submission_xgb_diffFE_pseudo.csv"),
    # A1 honesty fix: round-2 pseudo-XGB (leakage-clean) replaces round-1 (audit §2.2).
    # Standalone leak +0.00019 → only +0.00001 in-blend at this weight, but round-2 is honest.
    ("psxgb",      "oof_xgb_pseudo2.parquet",           "submission_xgb_pseudo2.csv"),
    ("diffxgb",    "oof_xgb_diffFE.parquet",            "submission_xgb_diffFE.csv"),
    ("difflgb",    "oof_lgb_diffFE.parquet",            "submission_lgb_diffFE.csv"),
    ("cbdiff",     "oof_cb_diffFE.parquet",             "submission_cb_diffFE.csv"),
    ("pscb14",     "oof_cb_pseudo_exp14.parquet",       "submission_cb_pseudo_exp14.csv"),
]


def main():
    oofs, subs, names = {}, {}, []
    y = ids = sids = None
    for n, of, sf in POOL:
        if not (DATA/of).exists() or not (DATA/sf).exists():
            continue
        d = pd.read_parquet(DATA/of).sort_values("id").reset_index(drop=True)
        if y is None:
            y = d["target"].astype(int).to_numpy(); ids = d["id"].to_numpy()
        elif not (d["id"].to_numpy()==ids).all():
            print(f"  skip {n}: id mismatch"); continue
        s = pd.read_csv(DATA/sf).sort_values("id").reset_index(drop=True)
        if sids is None: sids = s["id"].to_numpy()
        oofs[n]=d["oof"].to_numpy(); subs[n]=s["PitNextLap"].to_numpy(); names.append(n)
        print(f"  {n:10s} OOF {roc_auc_score(y, oofs[n]):.5f}")

    X = np.stack([oofs[n] for n in names], 1)
    # init: spread across RM views + strong XGB
    init = {"psrm6r2":0.35,"diffrm6":0.25,"diffrm1":0.0,"rmview3":0.0,"diffpsxgb":0.22,"psxgb":0.09,"cbdiff":0.05,"pscb14":0.04}
    w = np.array([init.get(n,0.0) for n in names]);
    w = w/w.sum() if w.sum()>0 else np.ones(len(names))/len(names)
    best = roc_auc_score(y, X@w)
    for _ in range(600):
        imp=False
        for i in range(len(names)):
            for dl in (0.01,-0.01,0.03,-0.03,0.005,-0.005):
                t=w.copy(); t[i]=max(0,t[i]+dl); ss=t.sum()
                if ss<=0: continue
                t=t/ss; a=roc_auc_score(y, X@t)
                if a>best+1e-7: best=a; w=t; imp=True
        if not imp: break
    wd={n:round(float(w[i]),3) for i,n in enumerate(names) if w[i]>1e-3}
    print(f"\nBEST clean blend OOF: {best:.5f}")
    print(f"weights: {wd}")
    sub = sum(w[i]*subs[n] for i,n in enumerate(names))
    out = DATA/"submission_blend_best.csv"
    pd.DataFrame({"id":sids,"PitNextLap":sub}).sort_values("id").reset_index(drop=True).to_csv(out,index=False)
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
