# Experiment 072 — round-3 pseudo-RealMLP (single-seed feasibility, local M1 MPS)

**Cycle.** 17
**Status.** Inconclusive (plateau confirmed; round-3 RM lever closed).
**Date.** 2026-05-28

## Hypothesis

Iterative self-training on RealMLP continues to compound: round-3 OOF beats round-2 (0.95396) by ≥ +0.00005. Specifically, the round-2 blend `submission_blend_pseudo_r2.csv` (OOF 0.95436, LB 0.95375) used as round-3 labeler should produce pseudo-labels of marginally higher quality than the round-1 strong-blend labeler used for round-2, and RealMLP's input-clipping should continue to absorb the improvement.

## Rationale

- exp 069 showed round-2 pseudo-RM > round-1 (+0.00003 OOF). Unique among our pseudo experiments: every other base either stayed flat or regressed at round-2 (XGB regressed −0.00019). This makes RealMLP the *only* base where iterative compounding is empirically supported on this data.
- The round-2 *labeler* in exp 069 was the OOF-0.95433 round-1 blend. The round-3 labeler is the OOF-0.95436 round-2 blend — slightly cleaner. If the +0.00003 gain came from labeler cleanliness, round-3 should give another small step. If it came from a one-shot iteration boost, round-3 plateaus.
- Cheap to run: a single-seed M1 MPS pass takes ~1h. If +ve, escalate to 6-seed; if flat/negative, lever closed.

## Expected magnitude

- Single seed: 0.95370-0.95385 (vs round-2 single-seed range 0.95368-0.95377). If single-seed lands in this range, round-3 is plausibly +. If <0.95365, round-3 is dead.
- 6-seed (if we escalate): 0.95398-0.95405 standalone; blend with same 0.675/0.075/0.250 weights: 0.95437-0.95444. Crosses the 0.95441 hurdle if we hit the upper end.

## Overfitting risk

Low. Same labeler-from-test-only pseudo recipe; same CV protocol; same 5-fold seed 42. No new train-side information.

## Kill criteria

- [ ] Single-seed OOF < 0.95370 (below round-2's per-seed floor) → close round-3 lever.
- [ ] Single-seed OOF ∈ [0.95370, 0.95385] but blend with new psRM3 ≤ 0.95436 → no leverage at all, close.

## Scope

- `src/research/train_realmlp_pseudo_r3.py` (new — fork of `gpu-kernels/cycle17_realmlp_pseudo62_gpu.py`; device=mps, labeler=`submission_blend_pseudo_r2.csv`, single seed).
- `experiments/072_realmlp_pseudo_r3.md` (new).
- Outputs: `data/oof_realmlp_pseudo_r3_s42.parquet`, `data/submission_realmlp_pseudo_r3_s42.csv`.

## Reversibility check

No CV/seed/target/frozen-file changes. Reversible.

## Plan

1. Single-seed run (seed 42) on M1 MPS, ~1h. Gate on OOF >= 0.95370.
2. If gate clears, document this exp as Inconclusive→Running (5 more seeds), then launch the remaining seeds {7, 99, 137, 313, 777} sequentially on MPS (~5h). Otherwise close as Inconclusive (negative).
3. Blend probe: 4-way `{psRM6r2 (round-2), psRM3 (round-3), CB, psXGB}` plus refined free grid.

## Result

Single-seed (42), 5-fold, M1 MPS, 9400s wall (~2h37 — fold 4 took 5065s due to a lid-closed throttle; other folds 1028–1185s ≈ 17–20 min).

| Fold | Round-3 AUC (seed 42, new labeler 0.95436) | Round-2 AUC (seed 42, prior labeler 0.95433) | Δ (r3 - r2) |
| ---- | ------------------------------------------ | -------------------------------------------- | ----------- |
|  1   | 0.95433                                    | 0.95435                                       | −0.00002 |
|  2   | 0.95434                                    | 0.95436                                       | −0.00002 |
|  3   | 0.95340                                    | 0.95347                                       | −0.00007 |
|  4   | 0.95268                                    | 0.95267                                       | +0.00001 |
|  5   | 0.95374                                    | 0.95380                                       | −0.00006 |
| **Concatenated OOF** | **0.95369** | **0.95372** | **−0.00003** |

Round-3 OOF is essentially identical to round-2 (within fold-noise σ ≈ 0.00050). Iterative self-training on RealMLP has plateaued at round 2 — round-3's marginally-cleaner labeler (OOF 0.95436 vs round-2's labeler at OOF 0.95433) does not propagate to a measurable lift.

## Verdict

**Inconclusive — plateau confirmed; round-3 RM iteration lever closed.** The compound-iteration hypothesis (round-N+1 > round-N because the labeler keeps improving) is empirically false on this data for RealMLP after one iteration. The +0.00003 round-1→round-2 lift (exp 069) is a *one-shot* effect of cleaner-than-quick-XGB labels, not a recurrent process. 6-seed escalation is unjustified (single-seed Δ vs round-2 = −0.00003; 6-seed Δ would be in the same envelope).

## Kill-criteria check

- [x] Single-seed OOF (0.95369) < round-2 single-seed (0.95372) — kill criterion 1 fires (no improvement).
- [x] No blend probe needed — psRM6r2 already at the same OOF as psRM3 would be; blend swap would be flat-to-negative.

## Repro stamp

- Trainer: [src/research/train_realmlp_pseudo_r3.py](../src/research/train_realmlp_pseudo_r3.py) (fork of `gpu-kernels/cycle17_realmlp_pseudo62_gpu.py`, device='mps', labeler=`submission_blend_pseudo_r2.csv` (OOF 0.95436), single seed 42)
- Outputs on disk: `data/oof_realmlp_pseudo_r3_s42.parquet`, `data/submission_realmlp_pseudo_r3_s42.csv`
- 5-fold StratifiedKFold(shuffle=True, random_state=42) on `Year × PitNextLap`; n_ens=24

## Learnings

1. **Pseudo-iteration saturates at round 2 for RealMLP.** The round-1→round-2 gain (+0.00003 OOF) is a one-shot calibration effect, not a process — round-2→round-3 is flat. This is informative because the alternative ("each round gives us a little more") would have justified ~5 more rounds of compute.
2. **The plateau OOF for the pseudo-RM thread is 0.9537 single-seed / 0.9540 6-seed.** Any further OOF lift on the RealMLP base must come from a different mechanism (training-objective change, sample weighting, architecture diversity), not from cleaner pseudo-labels.
3. **Closes the iterative-pseudo lever for the entire project**, since exp 067 already closed it for XGB (round-2 < round-1 for XGB) and exp 068 closed it for CB (negative on either CB base). All three production bases have exhausted the iterative-self-training axis.

## Follow-ups

- Closed: pseudo-iteration thread.
- Pivot to training-objective diversity (exp 073 rank-target, P1-#2 of audit) — running next.
