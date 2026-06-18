"""Build the 3 reset-submission candidates that LB-test the pseudo-drift hypothesis.

A1 showed blend weights are honest and exp 090 showed diffFE-XGB is already
~inductive — so the only transductive channel left in the blend is round-1
pseudo-labeling (trains on test-derived labels). We can't LB-validate offline
(slots spent), so we stage three candidates spanning the pseudo axis and submit
them at the 00:00 UTC reset to see which transfers best to LB:

  C1 best            (OOF 0.95462) — full pool, incl. round-1 pseudo GBDTs.
  C2 nopseudoGBDT    (OOF 0.95461) — drops round-1 pseudo GBDTs, keeps pseudo-RM
                                     (round-2, de-leaked). Same OOF, drift-cleaner.
  C3 pure_nopseudo   (OOF 0.95449) — no pseudo at all. Lowest OOF, maximally clean.

If C2/C3 match or beat C1 on LB despite lower OOF, round-1 pseudo was the drift
source and the honest blend is the better submission.

Usage: .venv/bin/python src/build_drift_candidates.py
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

DATA = Path(__file__).resolve().parent.parent.parent / "data"

# (name, oof_file, submission_file)
ALL = {
    "psrm6r2":  ("oof_realmlp_pseudo62.parquet",     "submission_realmlp_pseudo62.csv"),
    "diffrm6":  ("oof_realmlp_diffFE_6seed.parquet", "submission_realmlp_diffFE_6seed.csv"),
    "diffrm1":  ("oof_realmlp_diffFE_s42.parquet",   "submission_realmlp_diffFE_s42.csv"),
    "diffpsxgb":("oof_xgb_diffFE_pseudo.parquet",    "submission_xgb_diffFE_pseudo.csv"),
    "psxgb":    ("oof_xgb_pseudo2.parquet",          "submission_xgb_pseudo2.csv"),
    "diffxgb":  ("oof_xgb_diffFE.parquet",           "submission_xgb_diffFE.csv"),
    "cbdiff":   ("oof_cb_diffFE.parquet",            "submission_cb_diffFE.csv"),
    "pscb14":   ("oof_cb_pseudo_exp14.parquet",      "submission_cb_pseudo_exp14.csv"),
}
CANDIDATES = {
    "submission_blend_best.csv":          ["psrm6r2","diffrm6","diffrm1","diffpsxgb","psxgb","diffxgb","cbdiff","pscb14"],
    "submission_blend_nopseudoGBDT.csv":  ["psrm6r2","diffrm6","diffrm1","diffxgb","cbdiff"],
    "submission_blend_pure_nopseudo.csv": ["diffrm6","diffrm1","diffxgb","cbdiff"],
}


def main():
    base = pd.read_parquet(DATA/ALL["psrm6r2"][0]).sort_values("id").reset_index(drop=True)
    y = base["target"].astype(int).to_numpy(); ids = base["id"].to_numpy()
    oof = {n: pd.read_parquet(DATA/f[0]).sort_values("id").reset_index(drop=True)["oof"].to_numpy() for n, f in ALL.items()}

    def coord_descent(names):
        M = np.stack([oof[n] for n in names], 1); w = np.ones(len(names))/len(names)
        best = roc_auc_score(y, M@w)
        for _ in range(800):
            imp = False
            for i in range(len(names)):
                for dl in (0.01,-0.01,0.03,-0.03,0.005,-0.005):
                    t = w.copy(); t[i] = max(0, t[i]+dl); s = t.sum()
                    if s <= 0: continue
                    t = t/s; a = roc_auc_score(y, M@t)
                    if a > best+1e-7: best = a; w = t; imp = True
            if not imp: break
        return w, best

    for out, names in CANDIDATES.items():
        w, best = coord_descent(names)
        sids = None; acc = None
        for i, n in enumerate(names):
            s = pd.read_csv(DATA/ALL[n][2]).sort_values("id").reset_index(drop=True)
            if sids is None: sids = s["id"].to_numpy()
            c = "PitNextLap" if "PitNextLap" in s.columns else s.columns[-1]
            acc = w[i]*s[c].to_numpy() if acc is None else acc + w[i]*s[c].to_numpy()
        pd.DataFrame({"id": sids, "PitNextLap": acc}).sort_values("id").reset_index(drop=True).to_csv(DATA/out, index=False)
        wd = {n: round(float(w[i]),3) for i, n in enumerate(names) if w[i] > 1e-3}
        print(f"{out:38s} OOF {best:.5f}  {wd}")


if __name__ == "__main__":
    main()
