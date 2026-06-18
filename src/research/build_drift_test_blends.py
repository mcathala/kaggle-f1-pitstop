"""Build two MATCHED single-seed blends to isolate the external-treatment (drift) effect on LB.

Both use single-seed RealMLP (matched, so the single-seed penalty cancels). They
differ ONLY in the external-distribution treatment:

  blend_baseline_ss = 0.675*RM_r3_ss(no flag) + 0.075*CB + 0.250*psXGB(full-ext)
  blend_drift_ss    = 0.675*RM_isorig_ss(flag) + 0.075*CB + 0.250*XGB_extweight(0.7-ext)

If blend_drift_ss LBs above blend_baseline_ss, the external-downweight + is_original
treatment improves LB transfer (cuts the -0.0006 drift) → justifies a full 6-seed
RM-isorig rebuild. If flat/lower, the drift is not from external over-weighting.

Writes:
  data/submission_blend_baseline_ss.csv
  data/submission_blend_drift_ss.csv
and prints OOF for both (so we know the OOF starting point of each).
"""
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score

DATA = Path(__file__).resolve().parent.parent.parent / "data"

BLENDS = {
    "baseline_ss": [
        ("oof_realmlp_pseudo_r3_s42.parquet", "submission_realmlp_pseudo_r3_s42.csv", 0.675),
        ("oof_cb_tuned_exp14.parquet",        "submission_cb_tuned_exp14.csv",        0.075),
        ("oof_xgb_pseudo.parquet",            "submission_xgb_pseudo.csv",            0.250),
    ],
    "drift_ss": [
        ("oof_realmlp_isorig_s42.parquet",    "submission_realmlp_isorig_s42.csv",    0.675),
        ("oof_cb_tuned_exp14.parquet",        "submission_cb_tuned_exp14.csv",        0.075),
        ("oof_xgb_extweight.parquet",         "submission_xgb_extweight.csv",         0.250),
    ],
}


def build(name, parts):
    oof_blend = None; target = None; ids = None
    for oof_f, _, w in parts:
        df = pd.read_parquet(DATA / oof_f).sort_values("id").reset_index(drop=True)
        if oof_blend is None:
            oof_blend = w * df["oof"].to_numpy(); target = df["target"].astype(int).to_numpy(); ids = df["id"].to_numpy()
        else:
            assert (df["id"].to_numpy() == ids).all(), f"id mismatch {oof_f}"
            oof_blend = oof_blend + w * df["oof"].to_numpy()
    auc = roc_auc_score(target, oof_blend)
    # submission
    sub_blend = None; sids = None
    for _, sub_f, w in parts:
        df = pd.read_csv(DATA / sub_f).sort_values("id").reset_index(drop=True)
        if sub_blend is None:
            sub_blend = w * df["PitNextLap"].to_numpy(); sids = df["id"].to_numpy()
        else:
            assert (df["id"].to_numpy() == sids).all(), f"id mismatch sub {sub_f}"
            sub_blend = sub_blend + w * df["PitNextLap"].to_numpy()
    out = DATA / f"submission_blend_{name}.csv"
    pd.DataFrame({"id": sids, "PitNextLap": sub_blend}).sort_values("id").reset_index(drop=True).to_csv(out, index=False)
    print(f"  {name:12s}: OOF {auc:.5f}  -> wrote {out.name}")
    return auc


def main():
    print("Matched single-seed drift-test blends (RM single-seed; differ only in external treatment):")
    for name, parts in BLENDS.items():
        build(name, parts)
    print("\nSubmit both; the LB GAP between them isolates the external-downweight + is_original effect.")


if __name__ == "__main__":
    main()
