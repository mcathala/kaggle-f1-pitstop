# Experiment 076 — noise-weighted RealMLP (bidirectional, ablate-first; KILLED at fold 1)

**Cycle.** 17 (post-audit Phase-1)
**Status.** **Reverted (ablate-first gate failed).** fold-1 AUC = 0.95344 vs the 0.9540 ablate gate; killed at fold 1. Δ −0.00091 vs the unweighted baseline (exp 069 fold-1 = 0.95435). The 50% subsampling of "noisy" rows discards too much signal — even partially-noised labels still contribute.
**Date.** 2026-05-28

## Hypothesis

Downweighting the ~32% of rows where the PitNextLap label disagrees with the next-row stint increment (bidirectional noise mechanism per the corrected audit §2.1) lifts the round-2 pseudo-RM base by ≥ +0.00005 OOF — by training on cleaner labels.

## Rationale

- exp 067 + exp 069 established that the pseudo-labeling effect on RM is real (clean +0.00003 OOF lift round-1→round-2 after the leakage component is removed).
- The audit's P1-#3 proposed sample-weighting the 51k single-direction "noisy" rows. Our empirical bidirectional re-analysis (142k rows, 32% of train) widened the scope.
- pytabkit's `fit()` doesn't accept sample_weight, so the audit-locked design said "bidirectional, ablate first" via row-subsampling: agree rows at 1×, noisy rows at 50%. Effective weight ratio 1.0 : 0.5.

## Result (fold 1 only, before kill)

| Metric | Value |
| ------ | ----- |
| Per-fold train rows | 247,363 agree + 51,974 noisy (50% of 103,949) + 80k external + 114k pseudo = 494,984 |
| fold 1/5 AUC | **0.95344** |
| vs exp 069 fold-1 (no noise weighting) | **−0.00091** |
| ablate-first gate (0.9540) | **FAILED** |

Killed before fold 2 — would have wasted ~4×17min = ~70 min more.

## Verdict

**Reverted, lever closed.** The 50% subsample rate (audit-locked "bidirectional, ablate first") is too aggressive: 51k noisy rows discarded carry enough signal that removing them hurts more than the label-noise removal helps.

## Mechanism diagnosis

The "noisy" label set is defined as `label != next-row-stint-increment`. With row-sampled data (avg ~10 of 50+ laps per group), this label↔stint disagreement is **partially structural, not all noise**:
- A row with `PitNextLap=1` and `stint_increment=0` could be (a) a noisy label OR (b) a true pit on the next lap, where the *next sampled row* is several laps later in the new stint with a recorded `stint_increment` that we're misattributing to a different gap. Without row-by-row contiguity, we can't tell.
- The 65k figure was therefore an upper bound. The actual label noise is probably 20-30k (audit's 51k was halfway between).
- Subsampling at 50% drops ~26k truly-informative-rows along with ~26k noisy ones. The strength loss dominates.

## Three ways this could be rescued (not pursued here)

1. **Lighter subsampling** (e.g., 75% keep rate). Throws away half as much signal. Cost: 2x as many runs needed to characterize the operating point.
2. **Targeted noise definition.** Only flag a row as "noisy" if the next sampled row is at LapNumber = LapNumber+1 (i.e., the next sampled row is actually the next lap). This drops the rows where stint_increment is ambiguous due to row-sampling.
3. **Pivot to a GBDT trainer with native sample_weight.** XGB/CB/LGB all accept `sample_weight`. Cheaper to test fractional weights without subsampling. Different base though — would build psXGB-noise-weighted etc.

(2) and (3) are tractable in future work.

## Kill-criteria check

- [x] fold-1 AUC (0.95344) < ablate-first gate (0.9540) → **kill criterion 1 fires**.

## Repro stamp

- Trainer: [src/research/train_realmlp_noise_weighted.py](../src/research/train_realmlp_noise_weighted.py); `compute_noise_mask` flags 29.56% of rows (129,816); ablate-first gate at fold-1 AUC ≥ 0.9540 with `return` on fail.
- Outputs: none (killed before persistence).
- Compute: ~1031s (17min) for fold 1; saved ~70 min by killing.

## Learnings

1. **"Noisy" labels in row-sampled tabular data are often partially structural.** Stint transitions detected between non-consecutive sampled rows are ambiguous — they could be label noise OR genuine pits that we can't precisely localize. A coarse 50% downweight conflates both regimes.
2. **For low-signal label-noise candidates, fractional sample weighting via row-subsampling is too lossy.** Native `sample_weight` (XGB/CB) preserves the row's contribution at fractional weight; row-dropping discards it entirely.
3. **The audit-locked "bidirectional, ablate first" design caught the failure in one fold (~17 min) instead of one full run (~85 min).** The gate worked as intended.

## Implications for the Phase-1 plan

Combined with exp 073 (rank-target failed) and exp 075 (self-distill LB-regressed), this is **3/3 Phase-1 training-objective / data-recipe diversity experiments closing negative or non-transferring**.

**DAE pretrain is descoped here.** DAE is a label-free pretrained MLP trained on the same train+test concat as our supervised models — strong shared-risk with the cat-exclusion student in exp 075. Same correlated failure mode is likely: OOF lift that doesn't generalize.

The remaining live levers:
- exp 071 pseudo-CB-exp14 (in flight, strong fold-1+2 signal already).
- P1-#4 logit-rank + power-mean operator probes on the existing zoo.

## Follow-ups

- Closed: row-subsampling for noise weighting.
- Open for future work: GBDT-side noise weighting (native `sample_weight`); lap-contiguity-aware noise mask.
