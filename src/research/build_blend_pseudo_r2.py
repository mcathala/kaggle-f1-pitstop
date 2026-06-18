"""Build 3-way blend submission using exp 069's round-2 pseudo-RealMLP-6seed.

Weights are the canonical cycle-11 3-way (0.675/0.075/0.250), with psRM6r2 swapped
in for psRM6:
  - psRM6r2 (exp 069 round-2 pseudo-RealMLP-6seed) × 0.675
  - CB-tuned-exp14                                 × 0.075
  - psXGB    (exp 063 round-1 pseudo-XGB-highbins) × 0.250

Per `blend_hgbc_probe`, this blend hits OOF 0.95436 — current project-best.
"""

from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score

DATA = Path(__file__).resolve().parent.parent.parent / "data"

PARTS = [
    ("psrm6r2", DATA / "submission_realmlp_pseudo62.csv", 0.675),
    ("cb",      DATA / "submission_cb_tuned_exp14.csv",   0.075),
    ("psxgb",   DATA / "submission_xgb_pseudo.csv",       0.250),
]

OOF_PARTS = [
    ("psrm6r2", DATA / "oof_realmlp_pseudo62.parquet",   0.675),
    ("cb",      DATA / "oof_cb_tuned_exp14.parquet",     0.075),
    ("psxgb",   DATA / "oof_xgb_pseudo.parquet",         0.250),
]

OUT_SUB = DATA / "submission_blend_pseudo_r2.csv"
OUT_OOF = DATA / "oof_blend_pseudo_r2.parquet"


def main():
    print("Building OOF blend...")
    oof_blend = None
    target = None
    ids_oof = None
    for name, path, w in OOF_PARTS:
        df = pd.read_parquet(path).sort_values("id").reset_index(drop=True)
        if oof_blend is None:
            oof_blend = w * df["oof"].to_numpy()
            target = df["target"].astype(int).to_numpy()
            ids_oof = df["id"].to_numpy()
        else:
            assert (df["id"].to_numpy() == ids_oof).all()
            oof_blend = oof_blend + w * df["oof"].to_numpy()
        print(f"  + {w:.3f} × {name}  (single-base AUC={roc_auc_score(target, df['oof']):.5f})")
    auc = roc_auc_score(target, oof_blend)
    print(f"  blend OOF AUC: {auc:.5f}")
    pd.DataFrame({"id": ids_oof, "target": target, "oof": oof_blend}).to_parquet(OUT_OOF, index=False)
    print(f"  wrote {OUT_OOF.name}")

    print("\nBuilding submission blend...")
    sub_blend = None
    ids_sub = None
    for name, path, w in PARTS:
        df = pd.read_csv(path).sort_values("id").reset_index(drop=True)
        col = "PitNextLap"
        if sub_blend is None:
            sub_blend = w * df[col].to_numpy()
            ids_sub = df["id"].to_numpy()
        else:
            assert (df["id"].to_numpy() == ids_sub).all()
            sub_blend = sub_blend + w * df[col].to_numpy()
        print(f"  + {w:.3f} × {name}")
    out = pd.DataFrame({"id": ids_sub, "PitNextLap": sub_blend})
    out.to_csv(OUT_SUB, index=False)
    print(f"  wrote {OUT_SUB.name}  ({len(out):,} rows)  pred mean={sub_blend.mean():.4f}")


if __name__ == "__main__":
    main()
