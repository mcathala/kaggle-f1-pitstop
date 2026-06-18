"""Phase-1 #1 — leakage-clean swap: round-1 psXGB → round-2 psXGB.

The blend we submitted (`submission_blend_pseudo_r2.csv`, OOF 0.95436, LB 0.95375)
uses round-2 psRM6 (clean: exp 069 showed round-2 wins for RM by +0.00003) but
round-1 psXGB (leaky: exp 067 showed round-2 LOSES for XGB by −0.00019). The
leakage component in our submitted OOF is therefore concentrated in psXGB.

This script swaps psxgb (round-1) → psxgb2 (round-2) in our 3-way pseudo blend
at the same fixed weights (0.675 / 0.075 / 0.250), evaluates the cleaner OOF,
and writes `submission_blend_pseudo_r2_xgb2.csv` for optional submission.

Expected: OOF drops by ≈ 0.00005 (cleaner labels, slightly weaker signal). If
LB improves anyway, the cleaner blend is the better submission going forward.
"""

from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score

DATA = Path(__file__).resolve().parent.parent.parent / "data"

SUB_PARTS = [
    ("psrm6r2", DATA / "submission_realmlp_pseudo62.csv", 0.675),
    ("cb",      DATA / "submission_cb_tuned_exp14.csv",   0.075),
    ("psxgb2",  DATA / "submission_xgb_pseudo2.csv",      0.250),
]
OOF_PARTS = [
    ("psrm6r2", DATA / "oof_realmlp_pseudo62.parquet",   0.675),
    ("cb",      DATA / "oof_cb_tuned_exp14.parquet",     0.075),
    ("psxgb2",  DATA / "oof_xgb_pseudo2.parquet",        0.250),
]

OUT_OOF = DATA / "oof_blend_pseudo_r2_xgb2.parquet"
OUT_SUB = DATA / "submission_blend_pseudo_r2_xgb2.csv"


def main():
    print("Building OOF blend (psRM6r2 / CB / psXGB-round2)...")
    blend = None; target = None; ids = None
    for name, path, w in OOF_PARTS:
        df = pd.read_parquet(path).sort_values("id").reset_index(drop=True)
        if blend is None:
            blend = w * df["oof"].to_numpy()
            target = df["target"].astype(int).to_numpy()
            ids = df["id"].to_numpy()
        else:
            assert (df["id"].to_numpy() == ids).all()
            blend = blend + w * df["oof"].to_numpy()
        print(f"  + {w:.3f} × {name}  AUC={roc_auc_score(target, df['oof']):.5f}")
    auc = roc_auc_score(target, blend)
    print(f"\n  leakage-clean blend OOF AUC: {auc:.5f}  (vs submitted 0.95436, Δ {auc - 0.95436:+.5f})")
    pd.DataFrame({"id": ids, "target": target, "oof": blend}).to_parquet(OUT_OOF, index=False)
    print(f"  wrote {OUT_OOF.name}")

    print("\nBuilding submission blend (test predictions)...")
    sub = None; ids_sub = None
    for name, path, w in SUB_PARTS:
        df = pd.read_csv(path).sort_values("id").reset_index(drop=True)
        if sub is None:
            sub = w * df["PitNextLap"].to_numpy()
            ids_sub = df["id"].to_numpy()
        else:
            assert (df["id"].to_numpy() == ids_sub).all()
            sub = sub + w * df["PitNextLap"].to_numpy()
        print(f"  + {w:.3f} × {name}")
    out = pd.DataFrame({"id": ids_sub, "PitNextLap": sub})
    out.to_csv(OUT_SUB, index=False)
    print(f"  wrote {OUT_SUB.name}  ({len(out):,} rows)  pred mean={sub.mean():.4f}")


if __name__ == "__main__":
    main()
