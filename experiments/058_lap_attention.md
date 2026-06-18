# Experiment 058 — transductive lap-attention model (driver-race self-attention)

**Cycle.** 17
**Status.** Inconclusive (feasibility established) — architecture is genuinely diverse (ρ 0.90 vs all bases, tied most-diverse in project) but too weak standalone (OOF 0.936 < 0.949 floor) in minimal form. Diversity source is cross-lap attention, NOT numeric embeddings → strengthening may preserve diversity where embMLP's did not. → exp 059.
**Date.** 2026-05-27

## Data-structure findings that motivate this (EDA, 2026-05-27)

Cycle 16 closed every *row-independent* lever. Before cycle 17, characterized the train/test split:

1. **Sparse, non-contiguous lap samples.** Each (Year, Race, Driver) group holds a *sampled* subset of the race's laps, not all of them — only 4,078 / 42,233 groups have a contiguous lap range. Train mean 10.7 laps/group (max 38); test mean 5.1 (max 20). So this is irregular-time-series, not a dense lap-by-lap sequence.
2. **Transductive split.** Train and test laps are *interleaved within the same driver-race* — 35,674 of 37,038 test groups also appear in train. 93% of test laps have a neighboring sampled lap (same group) whose features we observe; 41.7% have an adjacent (gap=1) one.
3. **Synthetic target — no deterministic leak.** `PitNextLap[t] ≠ PitStop[t+1]` (only 18% concordance even at gap=1); a stint increment at t+1 predicts PitNextLap[t]=1 just 17% of the time. The physics relationships are deliberately noised (Playground-style synthetic generation), which is why trees cap at ~0.95, not 1.0. There is no free-lunch neighbor lookup.

**Implication.** The transductive neighbor structure carries *signal* but no *certainty*, and no model so far uses it — every base (trees, RealMLP, embMLP) predicts each lap from its own row. Lag-FE on XGB (exp 051, causal/past-only) hit the ρ≈0.997 wall. The one untried inductive bias: let each lap **attend across all sampled laps of its driver-race** (bidirectional), so the model can read the surrounding pit/stint/tyre trajectory directly rather than from per-row features.

## Hypothesis

A Transformer that self-attends over the sampled laps within each driver-race produces a base that is (a) ≥ the 0.949 standalone floor and (b) **rank-diverse** from RealMLP (ρ < 0.97) — landing in the empty "strong + diverse" quadrant cycle 16 mapped — and lifts the 3-way blend OOF (0.95421) by ≥ +0.00020.

## Why it could be diverse where embMLP/TabM/FTT were not

Those were all *row-independent* NNs whose only strength source was numeric embeddings (RealMLP's own mechanism → correlated). Lap-attention's strength source is **cross-lap context**, which neither RealMLP nor the trees can express. If it works at all, its ranking should differ structurally.

## Leakage controls

- **Inputs are observed features only** (TyreLife, Stint, PitStop, LapTime_Delta, Cumulative_Degradation, Position, RaceProgress, Position_Change, LapNumber, Compound/Driver/Race embeddings). The target `PitNextLap` is **never** fed as input. Bidirectional attention over neighbor *features* is leakage-free (those features exist for test rows too; and PitNextLap ≠ next_pit, so neighbors don't encode the answer).
- **Group-aware CV.** 5-fold split on whole driver-race groups (a group is entirely in train or entirely in val per fold), so a lap is never in the same batch as a val-fold lap during training. Produces row-level OOF for every train lap. Stratify group assignment by Year and group-positive-rate to match the existing protocol's spirit.

## Expected magnitude

- Standalone target: ≥ 0.949 (floor) and ideally ≥ 0.952.
- Diversity target: ρ vs RealMLP < 0.97 (vs embMLP-v2's 0.978, entity-embMLP's 0.908).
- Blend target: 4-way OOF ≥ 0.95441.

## Kill criteria

- [ ] Standalone OOF < 0.949 (below floor; like every other non-RealMLP NN) → architecture caps, revert.
- [ ] ρ vs RealMLP ≥ 0.978 (no more diverse than embMLP-v2) → not a new ranking, revert.
- [ ] Best blend config < 0.95421 + 0.00020 → no lift.

## Scope / reversibility

New Kaggle notebook `gpu-kernels/cycle17_lap_attention_gpu.py` (PyTorch, P100 GPU). Group sequences by (Year, Race, Driver); pad/mask to max-len; per-lap binary head. Competition-only for the feasibility probe (external integration deferred — exp 056 showed external helps ~+0.0009, a follow-up if the architecture clears the floor). Does not touch CV/seed/target/frozen files. Reversible.

## Plan

1. 1-fold feasibility (small model, ~10 epochs) → check floor + ρ vs RealMLP before investing in 5-fold.
2. If it clears floor AND is diverse → full 5-fold, pull OOF, blend-probe.
3. If promising but external-gap matters → add importance-free external sequences.

## Result

Ran on Kaggle P100, 5-fold group-out, 3-layer Transformer encoder (d_model=128, 4 heads), 10 raw numerics + 4 categorical embeddings + sinusoidal lap positional encoding, ~30 epochs/fold with early stop. **Competition-only, raw features, single seed** (deliberately minimal — testing the architecture, not maximizing strength). Total ~7 min compute (42,233 groups, max_len 51).

| Fold | best val AUC |
| ---- | ------------ |
| 1 | 0.93747 |
| 2 | 0.93778 |
| 3 | 0.93706 |
| 4 | 0.93719 |
| 5 | 0.93402 |
| **OOF** | **0.93646** (covered 1.000) |

### Rank-correlation (OOF)

|         | realmlp | cb | xgb | lap |
| ------- | ------- | -- | --- | --- |
| realmlp | 1.0000 | 0.9758 | 0.9799 | **0.9005** |
| cb      | 0.9758 | 1.0000 | 0.9840 | 0.9007 |
| xgb     | 0.9799 | 0.9840 | 1.0000 | 0.9015 |
| lap     | 0.9005 | 0.9007 | 0.9015 | 1.0000 |

ρ ≈ 0.90 against every existing base — tied with the entity-embedding MLP (exp 053) as the **most rank-diverse base in the project**. The cross-lap attention genuinely ranks differently.

### Blend probe (4th base, w_lap sweep)

w_lap=0 is optimal; every positive weight hurts (0.05→−0.00008, 0.25→−0.00099). Too weak to earn blend weight despite the diversity. No lift, no submission.

## Verdict

**Inconclusive (feasibility established).** The lap-attention architecture (a) trains cleanly and (b) produces the joint-most-diverse base in the project (ρ 0.90), but (c) at 0.936 standalone in minimal form it is below the 0.949 floor, so w=0 in the blend. This is the embMLP arc (exp 053: diverse+weak) — **but with a structurally different diversity source.** embMLP's diversity came from numeric embeddings; when strengthened (exp 054) that coupled it to RealMLP (ρ→0.978). lap-attention's diversity comes from *cross-lap context*, a mechanism neither RealMLP nor the trees have — so strengthening it might keep ρ low. That is the experiment worth running.

## Kill-criteria check

- [x] Standalone OOF 0.936 < 0.949 — **FIRED** (below floor; minimal form).
- [ ] ρ vs RealMLP ≥ 0.978 — **NOT fired** (ρ 0.90, very diverse).
- [x] Best blend < hurdle — **FIRED** (w_lap=0).

## Repro stamp

- Kaggle kernel: `mcathala/cycle-17-lap-attention-exp-058`; notebook [gpu-kernels/cycle17_lap_attention_gpu.py](../gpu-kernels/cycle17_lap_attention_gpu.py)
- probe: [src/research/blend_lapattn_probe.py](../src/research/blend_lapattn_probe.py)
- torch 2.5.1 (cu121, sm_60), P100; outputs `data/oof_lap_attention.parquet`, `data/submission_lap_attention.csv`
- local smoke test: `/tmp/smoke_lapattn.py` (CPU, 1500-group subset) validated the pipeline before the GPU push

## Learnings

1. **A non-embedding diversity source exists.** Every prior diverse-but-weak NN (embMLP) and every strong NN (RealMLP) draws strength from numeric embeddings. lap-attention is the first base whose ranking differs structurally (ρ 0.90) for a *different* reason — cross-lap context. This is the project's first evidence that the strong+diverse quadrant might be reachable by a mechanism RealMLP can't replicate.
2. **Minimal form is weak, as expected.** Raw 10 features, no external, single seed → 0.936. The trees reach 0.953 only with rich FE + external; embMLP-v2 reached 0.951 only with numeric embeddings + external + 3 seeds. lap-attention has had none of those levers yet.

## Follow-ups

- **exp 059 — strengthened lap-attention:** add the external sequences (exp 056: +~0.0009 for NNs), the domain FE numerics the trees use, and a 3-seed ensemble. Test whether structural (attention) diversity survives strengthening (ρ stays < 0.97) where embMLP's embedding diversity did not (exp 054). If it clears 0.949 while staying diverse, it is the first candidate for the empty quadrant.
