"""Experiment 029 (cycle 10) — advanced blend operators on cached OOFs.

Tests three non-linear blend operators against cycle 7's linear blend baseline:
  - Logit-rank blend: blend logits-of-ranks, sigmoid back, remap to anchor's value scale.
  - Confidence-gated blend: linear blend only on the ambiguous middle of the anchor.
  - Piecewise (ventile) rescaling: per-bin scaling of anchor by anchor/support mean ratio.

Inputs (anchor + support):
  - cycle 5 multi-seed RealMLP OOF + test submission
  - cycle 4 CB-tuned-exp14 OOF + test submission

Outputs:
  data/blend_advanced_sweep.parquet  — full sweep results table
  data/oof_blend_advanced_best.parquet — best OOF (if cycle 7 hurdle cleared)
  data/submission_blend_advanced_best.csv — best test submission (if hurdle cleared)
"""

from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

DATA = Path(__file__).resolve().parent.parent.parent / "data"

REALMLP_OOF = DATA / "oof_realmlp_multiseed.parquet"
CB_OOF = DATA / "oof_cb_tuned_exp14.parquet"
REALMLP_SUB = DATA / "submission_realmlp_multiseed.csv"
CB_SUB = DATA / "submission_cb_tuned_exp14.csv"

SWEEP_OUT = DATA / "blend_advanced_sweep.parquet"
OOF_OUT = DATA / "oof_blend_advanced_best.parquet"
SUB_OUT = DATA / "submission_blend_advanced_best.csv"

TARGET = "PitNextLap"
ID_COL = "id"

CYCLE7_OOF = 0.95408
HURDLE = CYCLE7_OOF + 0.00010

EPS = 1e-6
CLIP_LOW = 1e-7
CLIP_HIGH = 1 - 1e-7


# ============================================================
# Operators
# ============================================================

def normalized_rank(values: np.ndarray) -> np.ndarray:
    """Uniformly-spaced ranks in [0, 1] (ties broken by mergesort)."""
    ranks = rankdata(values, method="average")
    return (ranks - 1) / max(1, (len(values) - 1))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, CLIP_LOW, CLIP_HIGH)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def linear_blend(anchor: np.ndarray, support: np.ndarray, w_support: float) -> np.ndarray:
    return np.clip((1 - w_support) * anchor + w_support * support, CLIP_LOW, CLIP_HIGH)


def rank_blend_to_anchor(anchor: np.ndarray, support: np.ndarray, w_support: float) -> np.ndarray:
    """Linear-blend the ranks then map the resulting ordering back to the anchor's value distribution.
    This is rank-only — AUC depends on order, so this is equivalent to using the blended ranks directly.
    Provided here for parity with the literature.
    """
    a_rank = normalized_rank(anchor)
    s_rank = normalized_rank(support)
    blended_rank = (1 - w_support) * a_rank + w_support * s_rank
    order = np.argsort(blended_rank, kind="mergesort")
    sorted_anchor = np.sort(anchor)
    out = np.empty_like(anchor)
    out[order] = sorted_anchor
    return np.clip(out, CLIP_LOW, CLIP_HIGH)


def logit_rank_blend(anchor: np.ndarray, support: np.ndarray, w_support: float) -> np.ndarray:
    """Convert ranks -> logits, weighted-average logits, sigmoid back, then remap to anchor's distribution.
    Stretches differences at extremes (large |logit|) more than the middle.
    """
    a_rank = normalized_rank(anchor)
    s_rank = normalized_rank(support)
    a_logit = logit(a_rank)
    s_logit = logit(s_rank)
    blended_logit = (1 - w_support) * a_logit + w_support * s_logit
    blended_rank = sigmoid(blended_logit)
    order = np.argsort(blended_rank, kind="mergesort")
    sorted_anchor = np.sort(anchor)
    out = np.empty_like(anchor)
    out[order] = sorted_anchor
    return np.clip(out, CLIP_LOW, CLIP_HIGH)


def confidence_gated_blend(
    anchor: np.ndarray, support: np.ndarray,
    w_support: float, low: float = 0.05, high: float = 0.95,
) -> np.ndarray:
    """Linear-blend only where anchor prob is in (low, high); leave confident extremes untouched."""
    full = linear_blend(anchor, support, w_support)
    out = anchor.copy()
    ambiguous = (anchor >= low) & (anchor <= high)
    out[ambiguous] = full[ambiguous]
    return np.clip(out, CLIP_LOW, CLIP_HIGH)


def piecewise_rescale(
    anchor: np.ndarray, support: np.ndarray,
    bins: int = 20, scale_clip: tuple | None = None,
) -> np.ndarray:
    """Split anchor by quantile bins, scale each bin by ratio (support_mean / anchor_mean)."""
    order = np.argsort(anchor, kind="mergesort")
    n = len(anchor)
    bin_size = max(1, n // bins)
    out = anchor.copy()
    for i in range(bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < bins - 1 else n
        idx = order[start:end]
        a_mean = float(np.mean(anchor[idx]))
        s_mean = float(np.mean(support[idx]))
        scalar = (s_mean + EPS) / (a_mean + EPS)
        if scale_clip is not None:
            lo, hi = scale_clip
            scalar = float(np.clip(scalar, lo, hi))
        out[idx] = np.clip(out[idx] * scalar, CLIP_LOW, CLIP_HIGH)
    return out


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(f"numpy {version('numpy')}  scipy {version('scipy')}  sklearn {version('scikit-learn')}")

    print("\nLoading OOFs + submissions (aligned by id)...")
    m = pd.read_parquet(REALMLP_OOF).set_index(ID_COL).sort_index()
    c = pd.read_parquet(CB_OOF).set_index(ID_COL).sort_index()
    assert (m["target"] == c["target"]).all(), "target mismatch realmlp vs cb"

    y = m["target"].to_numpy()
    anchor = m["oof"].to_numpy().astype(float)
    support = c["oof"].to_numpy().astype(float)

    auc_anchor = roc_auc_score(y, anchor)
    auc_support = roc_auc_score(y, support)
    print(f"  RealMLP-multiseed (anchor): {auc_anchor:.5f}")
    print(f"  CB-tuned-exp14 (support):   {auc_support:.5f}")
    print(f"  cycle 7 linear baseline:    {CYCLE7_OOF:.5f}")
    print(f"  hurdle:                     {HURDLE:.5f}")

    rows = []

    # --- Reference: re-run linear blend for parity
    for w in [0.10, 0.15, 0.20, 0.25, 0.30]:
        auc = roc_auc_score(y, linear_blend(anchor, support, w))
        rows.append({"op": "linear", "w_support": w, "extra": "", "auc": auc, "delta": auc - CYCLE7_OOF})

    # --- Rank-blend (uses blended ranks, no logit)
    for w in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        auc = roc_auc_score(y, rank_blend_to_anchor(anchor, support, w))
        rows.append({"op": "rank", "w_support": w, "extra": "", "auc": auc, "delta": auc - CYCLE7_OOF})

    # --- Logit-rank blend (stretches extremes)
    for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        auc = roc_auc_score(y, logit_rank_blend(anchor, support, w))
        rows.append({"op": "logit_rank", "w_support": w, "extra": "", "auc": auc, "delta": auc - CYCLE7_OOF})

    # --- Confidence-gated blend
    for low, high in [(0.05, 0.95), (0.10, 0.90), (0.15, 0.85)]:
        for w in [0.20, 0.30, 0.40, 0.50, 0.60]:
            auc = roc_auc_score(y, confidence_gated_blend(anchor, support, w, low, high))
            extra = f"gate={low}-{high}"
            rows.append({"op": "conf_gate", "w_support": w, "extra": extra, "auc": auc, "delta": auc - CYCLE7_OOF})

    # --- Piecewise rescaling
    for bins in [10, 20, 50, 100]:
        for clip in [None, (0.95, 1.05), (0.9, 1.1)]:
            auc = roc_auc_score(y, piecewise_rescale(anchor, support, bins, clip))
            extra = f"bins={bins} clip={clip}"
            rows.append({"op": "piecewise", "w_support": np.nan, "extra": extra, "auc": auc, "delta": auc - CYCLE7_OOF})

    df = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    df.to_parquet(SWEEP_OUT, index=False)
    print(f"\nWrote {SWEEP_OUT.name}  ({len(df)} rows)")

    # Per-operator best
    print("\n=== Best per operator ===")
    for op in ["linear", "rank", "logit_rank", "conf_gate", "piecewise"]:
        sub = df[df["op"] == op]
        if len(sub) == 0:
            continue
        b = sub.iloc[0]
        print(f"  {op:>10}  best AUC={b['auc']:.5f}  Δ vs cycle 7={b['delta']:+.5f}  w_support={b['w_support']} extra={b['extra']}")

    best = df.iloc[0]
    print(f"\n=== Top overall ===")
    print(df.head(10).to_string(index=False))
    print(f"\nBest: op={best['op']} AUC={best['auc']:.5f} Δ={best['delta']:+.5f}")

    if best["auc"] >= HURDLE:
        print(f"\n✓ Cleared hurdle ({best['auc']:.5f} ≥ {HURDLE:.5f}). Generating submission.")
        op, w = best["op"], best["w_support"]

        # Re-build the best operator's OOF + test predictions
        if op == "linear":
            oof = linear_blend(anchor, support, w)
        elif op == "rank":
            oof = rank_blend_to_anchor(anchor, support, w)
        elif op == "logit_rank":
            oof = logit_rank_blend(anchor, support, w)
        elif op == "conf_gate":
            # parse extra to recover gate
            parts = best["extra"].replace("gate=", "").split("-")
            low, high = float(parts[0]), float(parts[1])
            oof = confidence_gated_blend(anchor, support, w, low, high)
        elif op == "piecewise":
            # parse extra
            parts = best["extra"].split(" ")
            bins = int(parts[0].split("=")[1])
            clip_str = parts[1].split("=")[1]
            clip = None if clip_str == "None" else tuple(eval(clip_str))
            oof = piecewise_rescale(anchor, support, bins, clip)
        else:
            raise RuntimeError(f"Unknown op: {op}")

        pd.DataFrame({
            "id": m.index,
            "Year": m["Year"].values,
            "target": y,
            "oof": oof,
        }).to_parquet(OOF_OUT, index=False)
        print(f"Wrote {OOF_OUT.name}")

        # Build test submission with the same operator
        sub_m = pd.read_csv(REALMLP_SUB).sort_values(ID_COL).reset_index(drop=True)
        sub_c = pd.read_csv(CB_SUB).sort_values(ID_COL).reset_index(drop=True)
        assert (sub_m[ID_COL] == sub_c[ID_COL]).all(), "test id mismatch"

        a_test = sub_m[TARGET].to_numpy().astype(float)
        s_test = sub_c[TARGET].to_numpy().astype(float)
        if op == "linear":
            blended = linear_blend(a_test, s_test, w)
        elif op == "rank":
            blended = rank_blend_to_anchor(a_test, s_test, w)
        elif op == "logit_rank":
            blended = logit_rank_blend(a_test, s_test, w)
        elif op == "conf_gate":
            parts = best["extra"].replace("gate=", "").split("-")
            low, high = float(parts[0]), float(parts[1])
            blended = confidence_gated_blend(a_test, s_test, w, low, high)
        elif op == "piecewise":
            parts = best["extra"].split(" ")
            bins = int(parts[0].split("=")[1])
            clip_str = parts[1].split("=")[1]
            clip = None if clip_str == "None" else tuple(eval(clip_str))
            blended = piecewise_rescale(a_test, s_test, bins, clip)

        pd.DataFrame({ID_COL: sub_m[ID_COL], TARGET: blended}).to_csv(SUB_OUT, index=False)
        print(f"Wrote {SUB_OUT.name}")
    elif best["auc"] > CYCLE7_OOF:
        print(f"\n~ Improved over cycle 7 by {best['delta']:+.5f} but below hurdle. Inconclusive.")
    else:
        print(f"\n✗ No operator beat cycle 7's linear blend. Linear was near-optimal for these inputs.")


if __name__ == "__main__":
    main()
