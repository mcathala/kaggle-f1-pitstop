# Experiment 054 — strengthened entity-embedding MLP (numeric embeddings + multi-seed)

**Cycle.** 16
**Status.** Inconclusive (Reverted) — strengthening worked (OOF 0.941 → 0.951, clears floor) but coupled strength to RealMLP-correlation (ρ 0.908 → 0.978); no blend lift. Mechanistically closes the strong+diverse quadrant.
**Date.** 2026-05-27

## Hypothesis

Adding quantile-binned numeric embeddings (the cheap version of RealMLP's PBLD numeric embeddings) + a 3-seed ensemble + heavier regularisation to exp-053's diverse-but-weak embMLP lifts its OOF from 0.941 toward ~0.948-0.950 **while preserving its ρ < ~0.95 diversity** — landing it in the strong+diverse quadrant where it finally lifts the 3-way blend ≥ +0.00020 over 0.95421.

## Rationale

- Exp 053 produced the most rank-diverse base in the project (ρ 0.91) but it was too weak (0.941) to earn blend weight.
- Its gap to RealMLP was largely the numeric-embedding premium; replicating that cheaply should close most of it.
- The bet: strength and diversity are independent knobs, so we can buy strength without losing diversity.

## Expected magnitude

- Target standalone ~0.948-0.950; ρ stays < ~0.95; blend ≥ 0.95441.
- Floor: standalone gain that comes *with* ρ → ~0.98 (diversity lost) → revert.

## Kill criteria

- [x] ρ vs RealMLP rises into the tree cluster (≥ 0.97), erasing the diversity that was the whole point — **FIRED** (ρ 0.978).
- [x] Best blend config does not clear anchor + 0.00005 — **FIRED** (0.95420).

## Result

Same architecture as exp 053 plus: per-numeric quantile-bin embeddings (32 bins → dim-8 embedding, concatenated with raw standardised numerics), hidden (384,192), dropout 0.20, weight-decay 5e-5, 3-seed ensemble {42,7,99}. M1 MPS, ~190s/fold.

### Standalone — strengthening succeeded

| Fold | embMLP-v2 | embMLP-v1 (053) |
| ---- | --------- | --------------- |
| 1 | 0.95189 | 0.94124 |
| 2 | 0.95154 | 0.94163 |
| 3 | 0.95084 | 0.94126 |
| 4 | 0.94990 | 0.93981 |
| 5 | 0.95126 | 0.94086 |
| **OOF** | **0.95102** | **0.94070** |

**+0.01032 over v1**, clears the 0.949 floor by +0.00202. The numeric embeddings + ensemble closed most of the gap to RealMLP (0.95383). Per-seed ~0.949; the 3-seed ensemble lifted the fold means ~+0.002.

### Rank-correlation — but diversity collapsed

| vs | v2 ρ | v1 ρ |
| -- | ---- | ---- |
| RealMLP-multiseed | **0.97769** | 0.90804 |
| CB-tuned-exp14 | 0.96800 | 0.91041 |
| XGB-highbins | 0.97687 | 0.90568 |

Adding numeric embeddings pulled the model from the diversity outlier (ρ 0.91) straight into RealMLP's cluster (ρ 0.978) — because numeric embeddings are *RealMLP's own mechanism*, so the model now ranks like RealMLP.

### Blend probe (anchor 0.95420)

| config | OOF | Δ |
| ------ | --- | --- |
| LINEAR free 4-way grid | 0.95420 | −0.00001 (w_em = 0.025) |
| RANK free 4-way grid | 0.95418 | −0.00003 (w_em = 0.025) |
| add em @ 0.10 | 0.95418 | — |
| em-for-rm @ 0.275 | 0.95395 | — |

No lift; the free grid awards embMLP-v2 a negligible 0.025 weight and the OOF is unchanged.

## Verdict

**Inconclusive (Reverted).** The strengthening worked — embMLP-v2 is a genuinely strong NN (0.951, above floor) — but it bought that strength by becoming RealMLP-correlated (ρ 0.978), so it is redundant in a blend that already contains RealMLP. Strength and diversity are *not* independent knobs for NNs on this data.

## Kill-criteria check

- [x] ρ vs RealMLP ≥ 0.97 (diversity lost) — **FIRED** (0.978).
- [x] No blend lift — **FIRED** (0.95420).

## Repro stamp

- trainer: [src/research/train_embmlp_v2.py](../src/research/train_embmlp_v2.py)
- packages: torch 2.12.0 (MPS); 5 folds × 3 seeds × ~190s = ~16 min
- inputs: `data/{train,test}.csv` + external; outputs `data/oof_embmlp_v2.parquet`, `data/submission_embmlp_v2.csv`

## Learnings

1. **The strong+diverse quadrant is empty by mechanism, not by accident.** On this data, NN strength comes from numeric embeddings — and numeric embeddings produce RealMLP-like rankings (because that is RealMLP's defining mechanism). So a NN is either *diverse + weak* (raw numerics, exp 053: 0.941 / ρ 0.91) or *strong + RealMLP-correlated* (numeric embeddings, this exp: 0.951 / ρ 0.978). There is no NN configuration that is both strong and diverse from RealMLP, because the source of strength *is* the source of correlation.
2. **This conclusively closes own-model improvement.** Combined with the tree closures (all cluster ρ ≥ 0.98) and the FE closures (XGB ranking-stable), every own-model lever is exhausted with a mechanistic explanation. **LB 0.95372 / OOF 0.95421 is the ceiling under the "own models only" constraint.**
3. **Multi-seed + numeric embeddings are a strong recipe for a *standalone* NN** (0.94070 → 0.95102, +0.0103) — useful knowledge if a future need arises for a second strong NN, even though it doesn't diversify *this* blend.

## Follow-ups

- Closed: own-model blend diversity (trees and NNs both exhausted, with mechanism understood).
- The only remaining path to the +0.0008 top-10% gap is external diversity (a different team's-style independent strong submission), which is outside the "own models only" constraint. That is a strategic/values call, not a modeling experiment. Cycle 16 closes here at the conclusively-established ceiling.
