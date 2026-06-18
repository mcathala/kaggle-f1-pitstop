# Experiment 053 — entity-embedding MLP (second NN, structurally diverse)

**Cycle.** 16
**Status.** Inconclusive (Reverted) — the most rank-diverse base we have ever built (ρ 0.91 vs everything) but too weak (OOF 0.941) to earn blend weight; adding it at any weight hurts. Maps the strength–diversity frontier and closes the own-model question.
**Date.** 2026-05-27

## Hypothesis

A plain entity-embedding MLP (explicit learned embeddings for Driver/Race/Compound/Year/Stint + two cross-cats, concatenated with standardised numerics into a 2-hidden-layer MLP) is structurally different enough from RealMLP's PyTabKit recipe to produce a low-rank-corr (< 0.97 vs RealMLP) base that lifts the 3-way blend OOF ≥ +0.00020 over 0.95421 — even if its standalone is modest.

## Rationale

- Cycle 16 closed every other lever: trees cluster at ρ ≥ 0.98 (exps 050-052), feature work doesn't shift XGB (ρ 0.997-0.999), attention NNs are dead (TabM 049, FTT 020).
- The only untried structurally-new model within "our own work" is a *second* NN that is not RealMLP's exact architecture. Explicit large categorical embeddings (Driver has 887 levels) is the obvious differentiator.
- RealMLP proves NNs train well here, so this is buildable locally on M1 MPS.

## Expected magnitude

- Standalone uncertain (other non-RealMLP NNs landed 0.941-0.945).
- Blend target: ≥ 0.95441 if diversity (ρ < 0.97) compensates for modest standalone.
- Floor: standalone < 0.945 with no blend lift → revert.

## Kill criteria

- [x] Standalone < 0.945 — **FIRED** (OOF 0.94070).
- [x] Best blend config does not clear anchor + 0.00005 — **FIRED** (adding it only hurts).

## Result

Custom PyTorch entity-embedding MLP, M1 MPS, 5-fold (same CV), external data concatenated per fold. Embeddings: Driver(888→50), Race(29→11), Compound(6→4), Year(5→4), Stint(9→5), Race_Year(107→22), Driver_Compound(3024→50). 26 standardised numerics. Adam lr 1e-3, BCE, early-stop on val AUC. ~25s/fold on MPS.

### Standalone — structurally weak

| Fold | embMLP AUC | epochs |
| ---- | ---------- | ------ |
| 1 | 0.94124 | 11 |
| 2 | 0.94163 | 13 |
| 3 | 0.94126 | 13 |
| 4 | 0.93981 | 13 |
| 5 | 0.94086 | 15 |
| **OOF** | **0.94070** | — |

Validation AUC peaked at epoch 5-7 then *declined* (overfitting), so it is at its effective capacity — not under-trained. −0.0083 below the 0.949 floor, −0.0131 below RealMLP.

### Rank-correlation — the most diverse base ever produced

| vs | ρ |
| -- | -- |
| RealMLP-multiseed | **0.90804** |
| CB-tuned-exp14 | 0.91041 |
| XGB-highbins | 0.90568 |

ρ ≈ 0.91 against everything — dramatically more independent than any prior base (the previous diversity champion, LGB-highbins, was 0.967).

### Blend probe — diversity cannot rescue weakness

| w_embmlp | linear OOF | rank OOF |
| -------- | ---------- | -------- |
| 0.00 | 0.95420 | 0.95420 |
| 0.02 | 0.95418 | 0.95417 |
| 0.05 | 0.95412 | 0.95412 |
| 0.10 | 0.95398 | 0.95397 |
| 0.20 | 0.95353 | 0.95347 |

Monotonically worse with any weight. Free 4-way grid (linear and rank space) both set w_embmlp = 0.

## Verdict

**Inconclusive (Reverted).** The entity-embedding MLP is the most rank-diverse base in the project (ρ 0.91) but its 0.94070 standalone is too weak — at that AUC, its disagreements with the strong models are wrong more often than right, so blending it adds noise. Diversity cannot compensate for an 0.008-below-floor standalone.

## Kill-criteria check

- [x] Standalone < 0.945 — **FIRED**.
- [x] No blend lift at any weight — **FIRED**.

## Repro stamp

- trainer: [src/research/train_embmlp.py](../src/research/train_embmlp.py)
- packages: torch 2.12.0 (MPS)
- runtime: 5 folds × ~25s = ~2 min on M1 MPS; outputs `data/oof_embmlp.parquet`, `data/submission_embmlp.csv`

## Learnings

1. **The strength–diversity frontier is now mapped, and the useful quadrant is empty.** Our bases fall into two groups: *strong + correlated* (trees: 0.951-0.953, ρ ≥ 0.98) and *diverse + weak* (this embMLP: 0.941, ρ 0.91). A base only helps the blend if it is *strong AND diverse* (≥ 0.951 with ρ < 0.97) — and across the whole cycle nothing lands there. RealMLP (0.954, ρ ~0.98) is the closest and is fully exploited.
2. **Every non-RealMLP NN caps at ~0.94 on this data.** FTT (cycle 6, 0.945), TabM (cycle 16, 0.941), and now a vanilla entity-embedding MLP (0.941) all land in the same band. RealMLP's PyTabKit recipe (PBLD numeric embeddings, tuned init/schedule, 24-net internal ensemble) extracts ~0.013 more AUC than a standard NN — that gap is the product, not an accident, and it does not transfer to a hand-built net.
3. **Own-model improvement is exhausted.** Combined with the tree closures, this is the conclusive answer to "why aren't we improving with the researched techniques": there is no untried own-model that is both strong and diverse. The 0.95372 LB / 0.95421 OOF is our ceiling under the "own models only" constraint.

## Follow-ups

- Closed: building new own-models (trees and NNs both exhausted).
- The only remaining levers are outside "own models built from scratch": relax the external-diversity constraint, or accept and protect the 0.95372 result. Cycle 16 closes here.
