"""Rebuild the best CLEAN blend including the diffFE bases, then build the submission.

Clean = no self-distill (exp 075 was an OOF mirage). Free coord-descent over the
strong base pool. Reports OOF and writes the submission for the best blend.

Pool (skips any missing):
  psrm6r2  (RealMLP round-2 pseudo, 6-seed)   — anchor NN, 0.95396
  rm6      (RealMLP 6-seed base)               — 0.95386
  cb       (CB-exp14 rich FE)                  — 0.95114
  pscb14   (pseudo-CB-exp14)                   — 0.95126
  cbdiff   (diffFE CatBoost, exp 081)          — NEW
  psxgb    (pseudo-XGB rich FE)                — 0.95295
  xgbdiff  (diffFE XGB, exp 080)               — 0.95291
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

DATA = Path(__file__).resolve().parent.parent.parent / "data"
POOL = [
    ("psrm6r2", "oof_realmlp_pseudo62.parquet",  "submission_realmlp_pseudo62.csv"),
    ("rm6",     "oof_realmlp_6seed.parquet",     "submission_realmlp_multiseed.csv"),
    ("cb",      "oof_cb_tuned_exp14.parquet",    "submission_cb_tuned_exp14.csv"),
    ("pscb14",  "oof_cb_pseudo_exp14.parquet",   "submission_cb_pseudo_exp14.csv"),
    ("cbdiff",  "oof_cb_diffFE.parquet",         "submission_cb_diffFE.csv"),
    ("psxgb",   "oof_xgb_pseudo.parquet",        "submission_xgb_pseudo.csv"),
    ("xgbdiff", "oof_xgb_diffFE.parquet",        "submission_xgb_diffFE.csv"),
]


def main():
    oofs, subs, names = {}, {}, []
    y = ids = sids = None
    for name, of, sf in POOL:
        if not (DATA/of).exists():
            print(f"  {name:9s} MISSING oof — skip"); continue
        d = pd.read_parquet(DATA/of).sort_values("id").reset_index(drop=True)
        if y is None:
            y = d["target"].astype(int).to_numpy(); ids = d["id"].to_numpy()
        else:
            assert (d["id"].to_numpy()==ids).all(), f"id mismatch {of}"
        s = pd.read_csv(DATA/sf).sort_values("id").reset_index(drop=True)
        if sids is None: sids = s["id"].to_numpy()
        else: assert (s["id"].to_numpy()==sids).all(), f"id mismatch {sf}"
        oofs[name] = d["oof"].to_numpy(); subs[name] = s["PitNextLap"].to_numpy(); names.append(name)
        print(f"  {name:9s} OOF {roc_auc_score(y, oofs[name]):.5f}")

    X = np.stack([oofs[n] for n in names], 1)
    # init from the known-good clean blend
    init = {"psrm6r2":0.578,"psxgb":0.151,"xgbdiff":0.221,"pscb14":0.05}
    w = np.array([init.get(n,0.0) for n in names]); w = w/w.sum() if w.sum()>0 else np.ones(len(names))/len(names)
    best = roc_auc_score(y, X@w)
    for _ in range(400):
        imp=False
        for i in range(len(names)):
            for dlt in (0.01,-0.01,0.03,-0.03,0.005,-0.005):
                t=w.copy(); t[i]=max(0,t[i]+dlt); s=t.sum()
                if s<=0: continue
                t=t/s; a=roc_auc_score(y,X@t)
                if a>best+1e-7: best=a; w=t; imp=True
        if not imp: break
    wd = {n:round(float(w[i]),3) for i,n in enumerate(names) if w[i]>1e-3}
    print(f"\nbest clean blend OOF: {best:.5f}")
    print(f"weights: {wd}")
    print(f"(anchor diffFE blend was OOF 0.95448 -> LB 0.95388)")

    # build submission
    sub = sum(w[i]*subs[n] for i,n in enumerate(names))
    out = DATA/"submission_blend_diffFE_v2.csv"
    pd.DataFrame({"id":sids,"PitNextLap":sub}).sort_values("id").reset_index(drop=True).to_csv(out,index=False)
    print(f"wrote {out.name}")
    # save weights for the doc
    pd.DataFrame([{"name":n,"w":float(w[i])} for i,n in enumerate(names)]).to_parquet(DATA/"diffFE_blend_v2_weights.parquet",index=False)


if __name__ == "__main__":
    main()
