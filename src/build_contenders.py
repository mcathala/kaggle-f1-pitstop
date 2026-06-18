"""Build the full contender pool of submission candidates for the daily LB reset.

Offline OOF is saturated at 0.95462 (linear blend optimal, confirmed 7 ways), so
the candidates here are NOT about chasing OOF — each is a distinct LB-TRANSFER
hypothesis to test on the real metric across the daily 5-slot budget:

  best          full coord-descent blend (incl. round-1 pseudo)      [anchor]
  nopseudoGBDT  drops round-1 pseudo GBDTs (same OOF, drift-cleaner)  [pseudo-drift]
  pure_nopseudo no pseudo at all (lower OOF, maximally clean)         [pseudo-drift extreme]
  robustincl    + the ρ0.959 robust GCE base (diversity-on-LB)       [diversity]
  rankmean      weighted mean of rank-percentiles vs probabilities    [aggregation]
  geomean       geometric mean                                        [aggregation]
  rmonly        RealMLP views only (do GBDTs even help LB?)           [composition]
  gbdtonly      GBDT views only (floor)                               [composition]

New bases (e.g. the rank-objective XGB) are folded in by adding their oof/sub to
ALL and extending CANDS. Run after any new base lands.

Usage: .venv/bin/python src/build_contenders.py
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

DATA = Path(__file__).resolve().parent.parent / "data"
ALL = {
    "psrm6r2":  ("oof_realmlp_pseudo62.parquet",     "submission_realmlp_pseudo62.csv"),
    "diffrm6":  ("oof_realmlp_diffFE_6seed.parquet", "submission_realmlp_diffFE_6seed.csv"),
    "diffrm1":  ("oof_realmlp_diffFE_s42.parquet",   "submission_realmlp_diffFE_s42.csv"),
    "diffpsxgb":("oof_xgb_diffFE_pseudo.parquet",    "submission_xgb_diffFE_pseudo.csv"),
    "psxgb":    ("oof_xgb_pseudo2.parquet",          "submission_xgb_pseudo2.csv"),
    "diffxgb":  ("oof_xgb_diffFE.parquet",           "submission_xgb_diffFE.csv"),
    "cbdiff":   ("oof_cb_diffFE.parquet",            "submission_cb_diffFE.csv"),
    "pscb14":   ("oof_cb_pseudo_exp14.parquet",      "submission_cb_pseudo_exp14.csv"),
    "robust":   ("oof_xgb_robust.parquet",           "submission_xgb_robust.csv"),
}
# best-blend coord-descent weights (full pool)
WBEST = {"psrm6r2":0.315,"diffrm6":0.293,"diffrm1":0.058,"diffpsxgb":0.242,"diffxgb":0.02,"cbdiff":0.035,"pscb14":0.036}
# (name, weights, mode)  mode in {lin, geo, rank}
CANDS = {
    "best":          (WBEST, "lin"),
    "nopseudoGBDT":  ({"psrm6r2":0.323,"diffrm6":0.312,"diffrm1":0.057,"diffxgb":0.252,"cbdiff":0.056}, "lin"),
    "pure_nopseudo": ({"diffrm6":0.531,"diffrm1":0.08,"diffxgb":0.337,"cbdiff":0.052}, "lin"),
    "rankmean":      (WBEST, "rank"),
    "geomean":       (WBEST, "geo"),
    "rmonly":        ({"psrm6r2":0.5,"diffrm6":0.45,"diffrm1":0.05}, "lin"),
    "gbdtonly":      ({"diffpsxgb":0.5,"diffxgb":0.2,"cbdiff":0.2,"pscb14":0.1}, "lin"),
}


def main():
    base = pd.read_parquet(DATA/ALL["psrm6r2"][0]).sort_values("id").reset_index(drop=True)
    y = base["target"].astype(int).to_numpy()
    O = {n: pd.read_parquet(DATA/f[0]).sort_values("id").reset_index(drop=True)["oof"].to_numpy()
         for n, f in ALL.items() if (DATA/f[0]).exists()}
    sids = None
    for name, (w, mode) in CANDS.items():
        if not all(n in O for n in w):
            print(f"  skip {name}: missing base"); continue
        ob = sb = None
        for n, wt in w.items():
            o = O[n]
            s = pd.read_csv(DATA/ALL[n][1]).sort_values("id").reset_index(drop=True)
            if sids is None: sids = s["id"].to_numpy()
            c = "PitNextLap" if "PitNextLap" in s.columns else s.columns[-1]
            sv = s[c].to_numpy()
            if mode == "geo":
                o2, s2 = np.log(np.clip(o,1e-6,1-1e-6)), np.log(np.clip(sv,1e-6,1-1e-6))
            elif mode == "rank":
                o2, s2 = rankdata(o)/len(o), rankdata(sv)/len(sv)
            else:
                o2, s2 = o, sv
            ob = wt*o2 if ob is None else ob+wt*o2
            sb = wt*s2 if sb is None else sb+wt*s2
        tot = sum(w.values())
        out = f"submission_blend_{name}.csv"
        pd.DataFrame({"id": sids, "PitNextLap": sb/tot}).sort_values("id").reset_index(drop=True).to_csv(DATA/out, index=False)
        print(f"  {name:14s} OOF {roc_auc_score(y, ob):.5f}  -> {out}")


if __name__ == "__main__":
    main()
