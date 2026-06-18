"""Experiment 040 (cycle 11) — 3-way blend with cached CB variants.

Final post-hoc check: does any 3-way blend `(RealMLP-multiseed, CB-tuned-exp14, X)`
beat cycle 7's 2-way 0.95408? X iterates over all cached CB variants from earlier
cycles. Cycle 8 exp 023 already ruled out LightGBM as the 3rd base; this checks
the parallel question for CB variants.

3-way constraint: w_rm + w_cb14 + w_var = 1, all ≥ 0. Grid-search over 2-D simplex.

Outputs:
  data/blend_3way_cb_variants.parquet
"""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

DATA = Path(__file__).resolve().parent.parent.parent / "data"
REALMLP_OOF = DATA / "oof_realmlp_multiseed.parquet"
CB14_OOF = DATA / "oof_cb_tuned_exp14.parquet"

VARIANTS = {
    "cb004":   DATA / "oof_cb004_seed42.parquet",
    "cb006":   DATA / "oof_cb006_seed42.parquet",
    "cb007":   DATA / "oof_cb007_seed42.parquet",
    "cb009":   DATA / "oof_cb009.parquet",
    "cb_tuned": DATA / "oof_cb_tuned.parquet",
}

ID_COL = "id"
HURDLE = 0.95428
# 2-D simplex grid: w_cb14, w_var (both ≥0, sum ≤1; w_rm = 1 − w_cb14 − w_var)
GRID = np.round(np.arange(0.0, 0.41, 0.05), 3).tolist()


def main() -> None:
    rm = pd.read_parquet(REALMLP_OOF).set_index(ID_COL).sort_index()
    cb14 = pd.read_parquet(CB14_OOF).set_index(ID_COL).sort_index()
    y = rm["target"].to_numpy()
    rm_oof = rm["oof"].to_numpy()
    cb14_oof = cb14["oof"].to_numpy()
    print(f"RealMLP-multiseed OOF AUC: {roc_auc_score(y, rm_oof):.5f}")
    print(f"CB-tuned-exp14 OOF AUC:    {roc_auc_score(y, cb14_oof):.5f}")
    auc_cycle7 = roc_auc_score(y, 0.80 * rm_oof + 0.20 * cb14_oof)
    print(f"Cycle 7 2-way blend OOF:   {auc_cycle7:.5f}  (target: > 0.95408 to be worth submitting)\n")

    rows = []
    for name, path in VARIANTS.items():
        if not path.exists():
            print(f"SKIP {name}: {path} missing")
            continue
        var = pd.read_parquet(path).set_index(ID_COL).sort_index()
        # Align by id (some variants might have different row count); inner join
        common = rm.index.intersection(var.index)
        if len(common) != len(rm):
            print(f"WARN {name}: only {len(common)} of {len(rm)} ids match — alignment issue, skipping")
            continue
        var_oof = var.loc[rm.index]["oof"].to_numpy()
        auc_var = roc_auc_score(y, var_oof)
        rho_rm, _ = spearmanr(var_oof, rm_oof)
        rho_cb14, _ = spearmanr(var_oof, cb14_oof)
        print(f"{name:10s}  standalone AUC={auc_var:.5f}  rank-corr-RM={rho_rm:.4f}  rank-corr-CB14={rho_cb14:.4f}")

        # Grid search the 2-D simplex
        best_a, best_w = auc_cycle7, (0.80, 0.20, 0.0)
        for w_cb14, w_var in product(GRID, GRID):
            w_rm = 1.0 - w_cb14 - w_var
            if w_rm < 0:
                continue
            pred = w_rm * rm_oof + w_cb14 * cb14_oof + w_var * var_oof
            a = roc_auc_score(y, pred)
            if a > best_a:
                best_a, best_w = a, (w_rm, w_cb14, w_var)
        delta = best_a - auc_cycle7
        rows.append({
            "variant": name,
            "var_auc": auc_var,
            "rank_corr_rm": rho_rm,
            "rank_corr_cb14": rho_cb14,
            "best_w_rm": best_w[0],
            "best_w_cb14": best_w[1],
            "best_w_var": best_w[2],
            "best_blend_auc": best_a,
            "delta_vs_cycle7": delta,
        })
        print(f"  best 3-way: w_rm={best_w[0]:.2f} w_cb14={best_w[1]:.2f} w_var={best_w[2]:.2f}"
              f" → AUC={best_a:.5f}  Δ={delta:+.5f}\n")

    df = pd.DataFrame(rows).sort_values("best_blend_auc", ascending=False)
    print("=== Summary (sorted by best 3-way AUC) ===")
    print(df.to_string(index=False))
    print(f"\nVariants beating cycle 7 (>0.95408): {(df['best_blend_auc'] > auc_cycle7).sum()}")
    print(f"Variants reaching hurdle (≥0.95428):  {(df['best_blend_auc'] >= HURDLE).sum()}")
    df.to_parquet(DATA / "blend_3way_cb_variants.parquet", index=False)


if __name__ == "__main__":
    main()
