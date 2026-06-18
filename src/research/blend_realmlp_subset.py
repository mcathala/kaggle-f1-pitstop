"""Experiment 039 (cycle 11) — RealMLP seed-subset selection.

The 5 individual RealMLP seed OOFs exist (seeds 42, 7, 99, 137, 313). The 5-seed
average (cycle 5's submission) blends with CB-tuned-exp14 at w_cb=0.20 for an
OOF of 0.95408. Question: is there a k-seed subset (k < 5) whose blend with CB
beats 0.95408?

Hypothesis: by removing seed(s) that are highly correlated with CB, the
remaining subset's blend has more effective diversity → higher AUC.

Outputs:
  data/blend_realmlp_subset_sweep.parquet  — per-subset metrics
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

DATA = Path(__file__).resolve().parent.parent.parent / "data"
CB_OOF = DATA / "oof_cb_tuned_exp14.parquet"
SEEDS = [42, 7, 99, 137, 313]
SEED_OOFS = {s: DATA / f"oof_realmlp_seed{s}.parquet" for s in SEEDS}

ID_COL = "id"
TARGET = "target"
HURDLE = 0.95428
WEIGHTS = np.round(np.arange(0.05, 0.41, 0.025), 3).tolist()


def main() -> None:
    print("Loading...")
    cb = pd.read_parquet(CB_OOF).set_index(ID_COL).sort_index()
    rm_seeds = {}
    for s, path in SEED_OOFS.items():
        if not path.exists():
            print(f"WARN: missing {path}")
            continue
        df = pd.read_parquet(path).set_index(ID_COL).sort_index()
        rm_seeds[s] = df["oof"].to_numpy()
        print(f"  seed {s}: AUC={roc_auc_score(df['target'], df['oof']):.5f}")

    y = cb["target"].to_numpy()
    cb_oof = cb["oof"].to_numpy()
    auc_cb = roc_auc_score(y, cb_oof)
    print(f"\nCB-tuned-exp14 OOF AUC: {auc_cb:.5f}")

    # Baseline: 5-seed avg
    full_avg = np.mean([rm_seeds[s] for s in SEEDS if s in rm_seeds], axis=0)
    auc_full = roc_auc_score(y, full_avg)
    rho_full, _ = spearmanr(full_avg, cb_oof)
    print(f"5-seed RealMLP avg: AUC={auc_full:.5f}  rank-corr-with-CB={rho_full:.5f}")

    best_blend_full = 0.0
    best_w_full = 0.20
    for w in WEIGHTS:
        a = roc_auc_score(y, (1 - w) * full_avg + w * cb_oof)
        if a > best_blend_full:
            best_blend_full, best_w_full = a, w
    print(f"Best blend (5-seed): w_cb={best_w_full}  AUC={best_blend_full:.5f}\n")

    # Enumerate all subsets of size 1..5
    available = [s for s in SEEDS if s in rm_seeds]
    rows = []
    for k in range(1, len(available) + 1):
        for subset in combinations(available, k):
            avg = np.mean([rm_seeds[s] for s in subset], axis=0)
            auc_avg = roc_auc_score(y, avg)
            rho, _ = spearmanr(avg, cb_oof)
            best_a, best_w = 0.0, 0.20
            for w in WEIGHTS:
                a = roc_auc_score(y, (1 - w) * avg + w * cb_oof)
                if a > best_a:
                    best_a, best_w = a, w
            rows.append({
                "subset": "+".join(str(s) for s in subset),
                "k": k,
                "rm_auc": auc_avg,
                "rank_corr_cb": rho,
                "best_w_cb": best_w,
                "best_blend_auc": best_a,
                "delta_vs_cycle7": best_a - best_blend_full,
            })

    df = pd.DataFrame(rows).sort_values("best_blend_auc", ascending=False)
    print("Top 15 subsets by best_blend_auc:")
    print(df.head(15).to_string(index=False))
    print(f"\nRows where best_blend_auc > 0.95408 (cycle 7): {(df['best_blend_auc'] > 0.95408).sum()}")
    print(f"Rows where best_blend_auc ≥ 0.95428 (hurdle):  {(df['best_blend_auc'] >= HURDLE).sum()}")

    df.to_parquet(DATA / "blend_realmlp_subset_sweep.parquet", index=False)
    print(f"\nwrote blend_realmlp_subset_sweep.parquet")


if __name__ == "__main__":
    main()
