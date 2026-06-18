# Experiment 060 — lap-attention v3 (strengthen on the diverse axis)

**Cycle.** 17
**Status.** Inconclusive (Reverted) — v3 lands ON the same strength↔diversity frontier as v1/v2 (0.941 @ ρ 0.93), not above it. The cross-lap mechanism's frontier tops out ~0.943 @ ρ 0.94, well below the oracle's required 0.951 @ ρ≤0.92. Conclusively closes lap-attention and the NN-diverse-base line.
**Date.** 2026-05-27

## Hypothesis (from our oracle-boost analysis)

Our oracle-boost analysis (exp 059 follow-up) mapped the prize precisely: a base at lap-attention's diversity (ρ ~0.90) needs only **AUC ≈ 0.951** (below RealMLP's 0.954) to push the 3-way blend to ~0.9555 — past the 0.9544 top-10% line. lap-attention reached 0.943 (v2), ~0.008 short.

Our v1→v2 ablation pinned v2's diversity loss (ρ 0.90→0.94) on the **RealMLP-style domain-FE numerics**, not the attention mechanism. So v3 strengthens *only* along the diverse axis — no RealMLP-style FE:
- minimal cross-lap features (10 raw numerics + 4 base cats, as v1);
- **winsorized numerics** (our EDA: ~20–138 rows with |x|>500, LapTime to 2507s, wreck StandardScaler) — clip to p0.5/p99.5;
- external full-race sequences (cross-lap signal, aligned with the mechanism);
- more capacity (d_model 192, 4 layers, 6 heads, 24-dim embeddings, 50 epochs);
- 3-seed ensemble.

Target: AUC ≥ 0.949 while ρ vs RealMLP ≤ 0.92.

## Result

Kaggle P100, group-out 5-fold, 3 seeds, ~58 min compute. 44,081 groups, max_len 78.

| Seed | OOF AUC |
| ---- | ------- |
| 42 | 0.93468 |
| 7  | 0.93498 |
| 99 | 0.93484 |
| **3-seed avg** | **0.94112** |

**OOF 0.94112 — *below* v2's 0.94304.** Removing the domain FE to protect diversity cost AUC; winsorize + external + extra capacity did not make it up.

### Strength–diversity frontier (all three lap-attention variants)

| variant | features | OOF AUC | ρ vs RealMLP |
| ------- | -------- | ------- | ------------ |
| v1 (058) | minimal, raw | 0.93646 | 0.9005 |
| v3 (060) | minimal + winsorize + ext + capacity | 0.94112 | 0.9313 |
| v2 (059) | + RealMLP-style domain FE | 0.94304 | 0.9409 |

The three points trace one tight, monotone frontier: AUC and ρ rise together. v3 did **not** break above it — it's just an interior point. The oracle's usable region (AUC ≥ 0.951 at ρ ≤ 0.92) sits entirely above this frontier.

### Blend probe

4-way sweep with v3: best = anchor 0.95420 at w_lap=0. No lift, no submission.

## Verdict

**Inconclusive (Reverted).** v3 tested whether the cross-lap mechanism could gain AUC *along the diverse axis* (without RealMLP-style features). It could not: stripping the domain FE dropped AUC to 0.941, and ρ still tracked AUC (0.93). The mechanism's entire reachable frontier (0.936 @ ρ0.90 → 0.943 @ ρ0.94) lies below the 0.951 @ ρ≤0.92 needed to help the blend. The strength↔diversity coupling is a property of the (synthetic) data, not of the features or capacity.

## Kill-criteria check

- [x] AUC did not reach 0.949 (0.941) AND did not gain on the diverse axis — **FIRED**.
- [x] Best blend = anchor, w_lap=0 — **FIRED**.

## Repro stamp

- Kaggle kernel `mcathala/cycle-17-lap-attention-v3-exp-060`; notebook [gpu-kernels/cycle17_lap_attention_v3_gpu.py](../gpu-kernels/cycle17_lap_attention_v3_gpu.py)
- outputs: `data/oof_lap_attention_v3.parquet`, `data/submission_lap_attention_v3.csv`
- compute: ~58 min P100 (counts against weekly GPU quota)

## Learnings

1. **The lap-attention strength↔diversity frontier is now mapped with three points and is conclusive.** No combination of features, winsorizing, external data, or capacity moves it above 0.943 @ ρ0.94 / 0.936 @ ρ0.90. Combined with embMLP (exp 053/054), TabM (049), FTT — every NN on this data lives on a frontier that never enters the strong+diverse quadrant.
2. **Our oracle analysis remains the precise statement of the gap:** we need ρ≤0.92 @ AUC≥0.951; our diverse bases cap at 0.943 and our strong bases (RM 0.954, XGB 0.953, CB 0.951) are ρ≥0.976. The strong+diverse base does not exist in our reachable model space.
3. **Winsorizing is validated as harmless-to-mildly-helpful** but not transformative for the NN (v3 0.941 vs v1 0.936, confounded with external+capacity).

## Follow-ups

- **Closed:** lap-attention, and with it the NN-as-new-diverse-base line that cycles 16–17 pursued. The diverse-base lever is exhausted.
- Remaining levers are lower-EV: (a) strengthen the *existing* bases (RealMLP HP/architecture — the 0.675-weight base, highest leverage), (b) more independent strong bases via genuinely different pipelines (mimics the public-LB blend path), or (c) accept the ~0.9537 ceiling. Decision point — see cycle-17 close.
