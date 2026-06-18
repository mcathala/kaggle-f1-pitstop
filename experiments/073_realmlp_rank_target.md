# Experiment 073 — rank-target RealMLP (recipe c, per-race rank_pct of laps_to_next_PitStop)

**Cycle.** 17
**Status.** **Reverted (target not learnable from row-features).** Killed after fold 1: single-fold AUC vs PitNextLap = 0.54439 / RMSE 0.2303, well below the 0.94 Gate-A floor.
**Date.** 2026-05-28

## Hypothesis

Predicting the per-(Year, Race) rank percentile of `laps_to_next_PitStop` derived from the PitStop timeline produces a strong+diverse base (AUC ≥ 0.949 vs PitNextLap, ρ ≤ 0.95 vs RealMLP-6seed) by changing the *training objective* rather than the architecture.

## Rationale

Audit Phase-1 #2. Cycle 16 mapped that architecture novelty plateaus at 0.94; the wins came from training-recipe changes. We've never trained on anything but supervised PitNextLap. Per-race rank_pct of an underlying timeline-derived target (laps_to_next_PitStop) is:
- A *structurally different target* (not a transform of PitNextLap).
- Derived from the deterministic `PitStop` column, not the noised `PitNextLap` label → cleaner signal.
- Uniform-distributed → needs rank-remap to blend (audit-locked operator).

## Expected magnitude

- Single seed: 0.95370-0.95385 (gate at ≥ 0.94, escalate at ≥ 0.949).
- 5-seed if escalated: 0.95398-0.95405.

## Result

Single-seed M1 MPS, fold 1 in flight when killed at ~12 min:

```
seed 42 fold 1/5 rmse=0.2303  AUC(PitNextLap, pred)=0.54439  flip=raw  (723s)
```

RMSE 0.2303 is barely better than predicting the per-fold mean (target std = 0.2886). AUC 0.544 means the model's rank order on `rank_pct` is essentially noise relative to `PitNextLap`. Killed before fold 2 — 4 more folds at the same expected level would have burned ~50 min for no information.

## Verdict

**Reverted, lever closed.** The hypothesis "model can predict per-race rank_pct from row features" is empirically falsified at fold 1 by a wide margin (0.544 vs the 0.949 we'd need).

## Mechanism diagnosis

Three candidates, ordered by likelihood:

1. **Per-race rank_pct is a transductive property of the timeline, not a row-level property.** Knowing where row X is in the within-race ordering requires comparison with the *other rows in the same race*, which a row-feature-only model can't do. The model's best guess is the per-race conditional mean (~0.5), which gives RMSE ≈ std and AUC ≈ 0.5.
2. **Censoring contaminates the target.** 53.55% of rows are censored at `race_max_lap − LapNumber + 1`, which is approximately monotone with LapNumber. The model would have to learn "is this row censored AND late in the race?" before predicting rank_pct. The censoring indicator was not provided as input — only the derived target. (Could potentially be rescued by including `censored` as an input feature, but the transductive issue above is more fundamental.)
3. **Loss-function mismatch.** RealMLP_TD_Regressor was used with MSE on `rank_pct`. Pairwise / listwise losses would be a better fit for a rank target, but require a different model class.

(1) is the showstopper — even fixing (2) and (3), a row-level model fundamentally can't predict a transductive target without batch-context.

## Implication for the audit's training-objective-diversity hypothesis

This is **partial falsification**, not full. The narrow claim "predict a row-rank from row features" failed. But the broader claim "train on a non-PitNextLap supervised target" remains testable in two forms still in the queue:
- **exp 075 self-distill MLP** — student trained on the teacher's continuous soft-OOF as target (which *is* learnable from row features by construction, since the teacher itself was learned that way).
- **A future RankNet-pairwise variant** — pairs the row-features-only constraint with a within-batch comparison loss; explicitly handles the transductive aspect. (Not in the current queue.)

Decision rule: `rank-target ≤ 0.94 AND self-distill < +0.00005 blend OOF ⇒ DAE descoped here`. Rank-target is well below 0.94 → gating on self-distill. Self-distill launched immediately on MPS (exp 075).

## Kill-criteria check

- [x] Single-fold AUC vs PitNextLap (0.544) < 0.94 floor → **kill criterion 1 fires within 1 fold**.
- [n/a] Blend probe — not run (no useful OOF generated).

## Repro stamp

- Trainer: [src/research/train_realmlp_rank_target.py](../src/research/train_realmlp_rank_target.py)
- Target source: [src/research/build_rank_target.py](../src/research/build_rank_target.py); outputs `data/rank_target_train.parquet`, `data/rank_target_test.parquet`.
- Killed at fold 1 / 5; PID 25195; partial outputs not persisted.

## Learnings

1. **Targets derived from a within-group rank are transductive and not directly learnable from row-features.** The audit's "rank-shaped output" inspiration referred to *output-distribution shape* (e.g., pairwise ordering), not *predicting a percentile rank as a regression target*. The two are different problems.
2. **Recipe (b) — train on rank_pct(teacher_OOF) — would have hit the same wall**: the *target* would still be a per-population rank, with the same transductive problem. Glad we picked (c) over (b) anyway; (c) at least tested an externally-defined target rather than re-discovering self-distillation noise.
3. **RankNet-pairwise (recipe a)** sidesteps the transductive issue because the loss compares pairs within a batch. Worth recording as a future option if we want to revisit the training-objective-diversity axis with a proper rank-style loss.
4. **Smoke-gate at fold 1 saved ~50 min compute.** The single-fold AUC was 5x worse than the floor — no value in running folds 2-5 just to lower the variance of a clearly-failed run.

## Follow-ups

- Closed (this round): per-race rank-percentile regression as a base.
- Running: exp 075 self-distill MLP (different mechanism: soft-target self-distill on the leakage-clean teacher).
- If self-distill also fails to clear +0.00005 blend OOF, the "training-objective diversity" axis closes for this round, and DAE pretrain is descoped pending future work.
