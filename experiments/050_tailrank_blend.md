# Experiment 050 — tail-gated rank-blend probe

**Cycle.** 16
**Status.** Inconclusive (Reverted) — re-ranking the anchor's tail with our own bases gives zero OOF lift; our internal model zoo is blend-saturated.
**Date.** 2026-05-27

## Hypothesis

ROC-AUC depends only on the ordering of predictions and is most sensitive to the ranking of the high-confidence positive tail. Our cycle-10 probe-2 residual EDA localised the worst-loss quartile to a coherent slice (degraded tyre × losing position × pit-cluster), which sits in that tail. If we re-rank ONLY the tail of our cycle-11 3-way anchor (OOF 0.95421) using a rank-diverse base — blended in percentile space — OOF AUC lifts by ≥ +0.00020.

## Rationale

- The cycle-11 3-way linear blend (RealMLP-ms 0.675 + CB14 0.075 + XGB-highbins 0.250) is our operating point at OOF 0.95421 / LB 0.95372.
- Plain global linear blending treats every row identically. But AUC is won/lost on the ordering of the high-confidence positives; a targeted re-rank of just the tail could extract ordering gains a global blend misses.
- Rank space is AUC-invariant to monotonic transforms, so the probe is pure post-processing on existing OOFs — no model training, runs in seconds.

## Expected magnitude

- Target: ≥ +0.00020 OOF over the 0.95421 anchor (project `min_delta`).
- Floor: < +0.00005 → the tail-rank lever does not work with our current bases.

## Kill criteria

- [x] No (source × tail-quantile × weight) configuration lifts OOF by ≥ +0.00005 — **FIRED**.

## Result

Probe ([src/research/blend_tailrank_probe.py](../src/research/blend_tailrank_probe.py)) swept 3 sources × 5 tail-quantiles × 5 weights = 75 configs. Re-rank rule: for rows above the anchor's `q` percentile, blend percentiles `(1-w)·rank(anchor) + w·rank(source)`; rows below keep the anchor rank.

Source diversity vs anchor (Pearson on percentile ranks):

| source | ρ vs anchor | solo OOF |
| --- | ---: | ---: |
| lgb_highbins | 0.9763 | 0.94885 |
| xgb_highbins | 0.9910 | 0.95263 |
| cb_tuned14 | 0.9871 | 0.95114 |

**Best of all 75 configs:** `lgb_highbins, tail_q=0.98, w=0.15` → OOF **0.95421** (Δ **+0.00000** vs anchor). Every configuration landed at +0.00000 (to 5 dp).

## Verdict

**Inconclusive (Reverted).** Tail-gated rank-blending with our own bases produces zero lift. Diagnosis: the anchor's tail is already dominated by RealMLP (our strongest model), and the candidate re-rankers — LGB/XGB/CB — are all *weaker solo* and *already inside* the blend, so they carry no new ordering information for the tail. This is the same wall hit by exps 023/029/040: our internal model zoo is blend-saturated.

The technique itself is sound (it is rank-space and AUC-targeted); it simply requires a source that is **both strong and genuinely diverse** from RealMLP in the tail — which none of our existing bases are. That points the next experiment back at *creating* such a source: either a genuinely new model family, or giving an existing base new signal it currently lacks (→ exp 051, within-stint trajectory features).

## Kill-criteria check

- [x] No config lifts OOF ≥ +0.00005 — **FIRED** (best +0.00000).

## Repro stamp

- inputs: `data/oof_blend_3way_xgb.parquet` (anchor) + `data/oof_{lgb_highbins,xgb_highbins,cb_tuned_exp14}.parquet`
- packages: numpy/pandas/scipy/scikit-learn (repo `.venv`)
- runtime: seconds (post-processing only); sweep saved to `data/blend_tailrank_sweep.parquet`

## Learnings

1. **Our internal zoo is confirmed blend-saturated — now from the rank/tail angle too.** Re-confirms across four independent probes (023 linear, 029 advanced ops, 040 CB variants, 050 tail-rank) that no recombination of our existing bases beats 0.95421.
2. **A blend-useful source must be strong AND diverse in the tail, not merely diverse.** LGB-highbins is our most rank-diverse base (ρ=0.976) but too weak solo (0.94885) to reorder the tail correctly; XGB/CB are strong but too correlated (ρ≥0.987).
3. **The lever is upstream of the blend.** To move OOF we must produce a new base with genuinely independent tail ordering — hence exp 051 (within-stint trajectory features) and the parallel TabM GPU track (exp 049).

## Follow-ups

- **exp 051** — within-stint trajectory (lag) features on XGB-highbins, to create a base with new tail signal.
- If a future base clears the 0.949 floor with ρ < 0.97 vs RealMLP, re-run this tail-rank probe with it as the source.
