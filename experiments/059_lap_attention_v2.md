# Experiment 059 — strengthened lap-attention (external + domain FE + 3-seed)

**Cycle.** 17
**Status.** Inconclusive (Reverted) — strengthening lifted lap-attention +0.0066 (0.936→0.943) but coupled it toward RealMLP (ρ 0.90→0.94); still below the 0.949 floor, w_lap=0 in blend. Closes the lap-attention thread.
**Date.** 2026-05-27

## Hypothesis

exp 058 showed the lap-attention architecture is the joint-most-diverse base (ρ 0.90) but too weak (0.936). Its diversity comes from cross-lap attention, *not* numeric embeddings — so, unlike embMLP-v2 (exp 054, whose diversity collapsed ρ 0.908→0.978 when strengthened), strengthening lap-attention might raise AUC while keeping ρ low. Add the three levers that took embMLP 0.941→0.951, without touching the architecture: external full-race sequences (training-only groups), the domain-FE numerics the trees use (+8 categorical embeddings), and a 3-seed ensemble. Target: OOF ≥ 0.949 AND ρ vs RealMLP < 0.97 → first base in the empty strong+diverse quadrant.

## Result

Ran on Kaggle P100, group-out 5-fold, 3 seeds {42,7,99}, ~25 min compute. 44,081 groups (40,869 competition + 3,212 external-only full-race sequences), max_len 78, 34 numerics + 8 categorical embeddings.

| Seed | OOF AUC |
| ---- | ------- |
| 42 | 0.93893 |
| 7  | 0.93936 |
| 99 | 0.93863 |
| **3-seed avg** | **0.94304** |

Per-fold val AUCs clustered tightly at 0.938–0.940 across all seeds. Strengthening added **+0.0066** over exp 058 (0.93646) — most of it from the 3-seed average (single seed ~0.939 → avg 0.943) — but plateaued well below the 0.949 floor.

### Rank-correlation (OOF)

|        | rm | cb | xgb | lap_v1 | lap_v2 |
| ------ | -- | -- | --- | ------ | ------ |
| rm     | 1.0000 | 0.9758 | 0.9799 | 0.9005 | **0.9409** |
| cb     | 0.9758 | 1.0000 | 0.9840 | 0.9007 | 0.9376 |
| xgb    | 0.9799 | 0.9840 | 1.0000 | 0.9015 | 0.9440 |
| lap_v1 | 0.9005 | 0.9007 | 0.9015 | 1.0000 | 0.9204 |
| lap_v2 | 0.9409 | 0.9376 | 0.9440 | 0.9204 | 1.0000 |

Strengthening raised ρ vs RealMLP from 0.9005 (v1) to **0.9409** (v2) — the same coupling embMLP showed, though milder.

### Blend probe (4th base, w_lap sweep on lap_v2)

w_lap=0 optimal; every positive weight hurts (0.05→−0.00003, 0.15→−0.00024). Best 0.95420, below the 0.95441 hurdle. No lift, no submission.

## Verdict

**Inconclusive (Reverted).** Strengthened lap-attention reaches 0.943 (below floor) and its ranking moves toward RealMLP (ρ 0.94) as it gets stronger — so it can be diverse+weak or stronger+more-correlated, never strong+diverse. w_lap=0.

**Key nuance:** at equal strength lap-attention is *more diverse* than embMLP — ρ 0.94 @ OOF 0.943 vs embMLP-v2's ρ 0.978 @ 0.951. It sits on a strictly better strength–diversity frontier. But the frontier still does not cross into the usable quadrant (strong enough to clear the floor while ρ < 0.97).

## Kill-criteria check

- [x] Standalone 0.943 < 0.949 — **FIRED** (below floor).
- [x] Strengthening coupled ρ upward (0.90→0.94) and best blend < hurdle — **FIRED** (w_lap=0).

## Repro stamp

- Kaggle kernel: `mcathala/cycle-17-lap-attention-v2-exp-059`; notebook [gpu-kernels/cycle17_lap_attention_v2_gpu.py](../gpu-kernels/cycle17_lap_attention_v2_gpu.py)
- external grouped separately (full-race dense sequences as training-only groups); torch 2.5.1 cu121, P100
- outputs: `data/oof_lap_attention_v2.parquet`, `data/submission_lap_attention_v2.csv`; probe [src/research/blend_lapattn_probe.py](../src/research/blend_lapattn_probe.py)

## Learnings

1. **The strength↔diversity tradeoff is now confirmed across TWO independent NN mechanisms.** embMLP (numeric embeddings, exp 053/054) and lap-attention (cross-lap attention, exp 058/059) both gain AUC only by becoming more RealMLP-like. The strong+diverse quadrant is robustly empty for NNs on this (synthetic) data — it is a property of the data/target, not of any one architecture.
2. **Cross-lap attention is the best diversity-retaining mechanism found** (better frontier than embeddings), but not enough. Further NN architecture search on this data is now very low-EV.
3. **Implication for the +0.0008 gap:** new diversity from our own models is exhausted. The remaining levers are (a) strengthening the *existing* strong bases (winsorize NN inputs — our EDA found extreme values distorting StandardScaler — applied to RealMLP, the 0.675-weight base, is the highest-leverage shot), or (b) adding genuinely independent bases by other means.

## Follow-ups

- Closed: lap-attention as a base (both diverse-weak and stronger-correlated forms fail). Closes the NN-as-new-diverse-base line that cycle 16 opened.
- **exp 060 — winsorized RealMLP:** our EDA found ~20–138 rows with |numeric| > 500 (LapTime up to 2507s) that distort the StandardScaler RealMLP depends on. Clip/winsorize the numerics before scaling and re-run RealMLP-multiseed. If the 0.675-weight base lifts, the blend lifts directly. Highest-leverage remaining experiment.
