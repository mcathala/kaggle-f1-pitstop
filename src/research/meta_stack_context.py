"""Context-aware stacking — regularized GBDT meta-model on [base preds + context].

exp 061 closed pred-only linear/logistic meta (overfit, flat). Per-year blend
weights failed (broke cross-year calibration). This tries the untried middle:
a SHALLOW, heavily-regularized LightGBM meta-model on the 8 base OOF predictions
PLUS a few context features (Year, Stint, TyreLife, Position, RaceProgress,
Compound). The hypothesis: a context-aware combiner can learn cross-year-CONSISTENT
context weighting (trust RM more here, XGB more there) that a global linear blend
and per-year weights both miss.

Gated HONESTLY: nested 5-fold (meta trained on fold-train base-OOF, scored on
held-out fold) — the only fair test against the 0.95462 linear blend, since a
GBDT on OOF preds overfits in-sample. Build a submission only if honest ≥ +0.0001.

Usage: .venv/bin/python src/meta_stack_context.py
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

DATA = Path(__file__).resolve().parent.parent.parent / "data"
BASES = {
    "psrm6r2": "oof_realmlp_pseudo62.parquet", "diffrm6": "oof_realmlp_diffFE_6seed.parquet",
    "diffrm1": "oof_realmlp_diffFE_s42.parquet", "diffpsxgb": "oof_xgb_diffFE_pseudo.parquet",
    "psxgb": "oof_xgb_pseudo2.parquet", "diffxgb": "oof_xgb_diffFE.parquet",
    "cbdiff": "oof_cb_diffFE.parquet", "pscb14": "oof_cb_pseudo_exp14.parquet",
}
CTX = ["Year", "Stint", "TyreLife", "Position", "RaceProgress", "LapNumber"]
PARAMS = dict(objective="binary", n_estimators=300, num_leaves=15, learning_rate=0.02,
              min_child_samples=200, subsample=0.8, colsample_bytree=0.7,
              reg_lambda=5.0, reg_alpha=2.0, max_depth=4, verbose=-1, n_jobs=-1)


def main():
    base = pd.read_parquet(DATA/BASES["psrm6r2"]).sort_values("id").reset_index(drop=True)
    y = base["target"].astype(int).to_numpy(); ids = base["id"].to_numpy()
    tr = pd.read_csv(DATA/"train.csv").sort_values("id").reset_index(drop=True)
    assert (tr["id"].to_numpy() == ids).all()
    P = {n: pd.read_parquet(DATA/f).sort_values("id").reset_index(drop=True)["oof"].to_numpy() for n, f in BASES.items()}
    cmp_codes = tr["Compound"].astype("category").cat.codes.to_numpy()

    Xmeta = np.column_stack([P[n] for n in BASES] + [tr[c].to_numpy() for c in CTX] + [cmp_codes])
    cols = list(BASES) + CTX + ["Compound_le"]

    # linear-blend reference (current best weights)
    W = {"psrm6r2":0.315,"diffrm6":0.293,"diffrm1":0.058,"diffpsxgb":0.242,"diffxgb":0.02,"cbdiff":0.035,"pscb14":0.036}
    lin = sum(w*P[n] for n, w in W.items())/sum(W.values())
    print(f"linear blend OOF: {roc_auc_score(y, lin):.5f}")

    strat = tr["Year"].astype(str) + "_" + pd.Series(y).astype(str)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    held = np.zeros(len(y))
    for tk, vk in skf.split(Xmeta, strat):
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(Xmeta[tk], y[tk])
        held[vk] = m.predict_proba(Xmeta[vk])[:, 1]
    honest = roc_auc_score(y, held)
    print(f"context-stack NESTED honest OOF: {honest:.5f}  (Δ vs linear {honest - roc_auc_score(y, lin):+.5f})")
    # also pred-only meta (no context) as a control
    Xp = np.column_stack([P[n] for n in BASES]); held2 = np.zeros(len(y))
    for tk, vk in skf.split(Xp, strat):
        m = lgb.LGBMClassifier(**PARAMS); m.fit(Xp[tk], y[tk]); held2[vk] = m.predict_proba(Xp[vk])[:, 1]
    print(f"pred-only meta NESTED honest OOF: {roc_auc_score(y, held2):.5f}  (control)")


if __name__ == "__main__":
    main()
