"""Generalized rank-remap blend probe — pair the anchor with any candidate base.

Per the audit-locked design: when a candidate base outputs something other than
a calibrated probability (rank percentiles, soft-distillation targets, etc.),
it needs rank-remap to the anchor's empirical CDF before linear-combining.

Anchor: `oof_blend_pseudo_r2.parquet` (OOF 0.95436, current project-best blend).
Candidate: passed via --cand path (defaults to rank-target OOF if no arg).

Sweeps w_cand ∈ {0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30}
across {linear, remap, rank_avg, logit_avg, gmean} operators.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

DATA = Path(__file__).resolve().parent.parent.parent / "data"

ANCHOR_OOF = DATA / "oof_blend_pseudo_r2.parquet"
ANCHOR_SUB = DATA / "submission_blend_pseudo_r2.csv"

WEIGHTS = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30]


def to_rank(x: np.ndarray) -> np.ndarray:
    return rankdata(x, method="average") / (len(x) - 1)


def remap_to_anchor_cdf(x: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Map x's rank order onto the anchor's empirical distribution."""
    rx = rankdata(x, method="average").astype(np.int64) - 1
    sorted_anchor = np.sort(anchor)
    return sorted_anchor[rx]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cand", type=Path, default=DATA / "oof_realmlp_rank_target_s42.parquet",
                   help="path to candidate OOF parquet")
    p.add_argument("--cand-sub", type=Path, default=None,
                   help="path to candidate submission CSV (defaults to cand stem .csv)")
    p.add_argument("--tag", type=str, default=None, help="label for the candidate in output filenames")
    return p.parse_args()


def main():
    args = parse_args()
    cand_oof = args.cand
    if args.cand_sub is None:
        # Convention: oof_<name>.parquet → submission_<name>.csv
        stem = cand_oof.stem.replace("oof_", "submission_", 1)
        cand_sub = cand_oof.parent / f"{stem}.csv"
    else:
        cand_sub = args.cand_sub
    tag = args.tag or cand_oof.stem.replace("oof_", "")
    print(f"Candidate OOF : {cand_oof.name}")
    print(f"Candidate SUB : {cand_sub.name}")
    print(f"Tag           : {tag}")
    if not cand_oof.exists():
        print(f"  candidate OOF missing — abort.", file=sys.stderr)
        sys.exit(2)

    print("\nLoading bases...")
    anchor = pd.read_parquet(ANCHOR_OOF).sort_values("id").reset_index(drop=True)
    rt = pd.read_parquet(cand_oof).sort_values("id").reset_index(drop=True)
    assert (anchor["id"].to_numpy() == rt["id"].to_numpy()).all(), "id mismatch"

    y = anchor["target"].astype(int).to_numpy()
    p_anchor = anchor["oof"].to_numpy()
    p_rt = rt["oof"].to_numpy()
    auc_anchor = roc_auc_score(y, p_anchor)
    auc_rt = roc_auc_score(y, p_rt)
    rho, _ = spearmanr(p_anchor, p_rt)
    print(f"  anchor (psRM6r2/CB/psXGB) OOF AUC: {auc_anchor:.5f}")
    print(f"  {tag} OOF AUC:                      {auc_rt:.5f}")
    print(f"  ρ(anchor, {tag}):                   {rho:.5f}")

    p_rt_remap = remap_to_anchor_cdf(p_rt, p_anchor)
    r_anchor = to_rank(p_anchor)
    r_rt = to_rank(p_rt)
    p_rt_logit = np.log(np.clip(r_rt, 1e-6, 1 - 1e-6) / np.clip(1 - r_rt, 1e-6, 1 - 1e-6))
    p_anchor_logit = np.log(np.clip(r_anchor, 1e-6, 1 - 1e-6) / np.clip(1 - r_anchor, 1e-6, 1 - 1e-6))

    print(f"\nSweeping w_{tag} over operators:\n")
    header = f"  {'w':>5}  {'linear':>8}  {'remap':>8}  {'rank_avg':>9}  {'logit_avg':>10}  {'gmean':>8}"
    print(header)
    print("  " + "-" * len(header))

    rows = []
    for w in WEIGHTS:
        # Linear blend of raw probabilities (likely worse — distribution mismatch)
        p_lin = (1 - w) * p_anchor + w * p_rt
        # Remap rank-target to anchor's CDF, then linear (the audit-locked correct op)
        p_remap = (1 - w) * p_anchor + w * p_rt_remap
        # Rank-space linear blend
        p_rank = (1 - w) * r_anchor + w * r_rt
        # Logit-rank linear (more aggressive on tails)
        p_logit = (1 - w) * p_anchor_logit + w * p_rt_logit
        # Geometric mean (with eps clip)
        p_g = (np.clip(p_anchor, 1e-6, 1 - 1e-6) ** (1 - w)) * (np.clip(remap_to_anchor_cdf(p_rt, p_anchor), 1e-6, 1 - 1e-6) ** w)

        aucs = [roc_auc_score(y, x) for x in [p_lin, p_remap, p_rank, p_logit, p_g]]
        rows.append({"w_cand": w, **dict(zip(["linear", "remap", "rank_avg", "logit_avg", "gmean"], aucs))})
        print(f"  {w:5.3f}  {aucs[0]:8.5f}  {aucs[1]:8.5f}  {aucs[2]:9.5f}  {aucs[3]:10.5f}  {aucs[4]:8.5f}")

    df = pd.DataFrame(rows)

    # Identify best across operators
    best_per_op = {op: df.loc[df[op].idxmax()] for op in ["linear", "remap", "rank_avg", "logit_avg", "gmean"]}
    print("\nBest per operator:")
    for op, row in best_per_op.items():
        delta = row[op] - auc_anchor
        print(f"  {op:10s}: w={row['w_cand']:.3f}  AUC={row[op]:.5f}  Δ vs anchor (0.95436) {delta:+.5f}")

    out_path = DATA / f"blend_rt_sweep_{tag}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nwrote {out_path.name}")

    # If any operator beats anchor by ≥0.00005, build that submission
    best_overall_op = max(best_per_op, key=lambda op: best_per_op[op][op])
    best_overall_auc = best_per_op[best_overall_op][best_overall_op]
    if best_overall_auc - auc_anchor >= 0.00005:
        print(f"\n*** Best blend ({best_overall_op}) clears +0.00005 vs anchor — building submission.")
        if not cand_sub.exists():
            print(f"    candidate submission CSV {cand_sub.name} missing; cannot build LB submission.", file=sys.stderr)
            return
        a_sub = pd.read_csv(ANCHOR_SUB).sort_values("id").reset_index(drop=True)
        r_sub = pd.read_csv(cand_sub).sort_values("id").reset_index(drop=True)
        assert (a_sub["id"].to_numpy() == r_sub["id"].to_numpy()).all()
        pa = a_sub["PitNextLap"].to_numpy()
        pr = r_sub["PitNextLap"].to_numpy()
        w = float(best_per_op[best_overall_op]["w_cand"])
        if best_overall_op == "linear":
            psub = (1 - w) * pa + w * pr
        elif best_overall_op == "remap":
            psub = (1 - w) * pa + w * remap_to_anchor_cdf(pr, pa)
        elif best_overall_op == "rank_avg":
            psub = (1 - w) * to_rank(pa) + w * to_rank(pr)
        elif best_overall_op == "logit_avg":
            ra = to_rank(pa); rr = to_rank(pr)
            la = np.log(np.clip(ra, 1e-6, 1 - 1e-6) / np.clip(1 - ra, 1e-6, 1 - 1e-6))
            lr = np.log(np.clip(rr, 1e-6, 1 - 1e-6) / np.clip(1 - rr, 1e-6, 1 - 1e-6))
            psub = (1 - w) * la + w * lr
        elif best_overall_op == "gmean":
            psub = (np.clip(pa, 1e-6, 1 - 1e-6) ** (1 - w)) * (np.clip(remap_to_anchor_cdf(pr, pa), 1e-6, 1 - 1e-6) ** w)
        sub_out = DATA / f"submission_blend_{tag}_{best_overall_op}_w{int(w*1000):03d}.csv"
        pd.DataFrame({"id": a_sub["id"], "PitNextLap": psub}).sort_values("id").reset_index(drop=True).to_csv(sub_out, index=False)
        print(f"    wrote {sub_out.name}")


if __name__ == "__main__":
    main()
