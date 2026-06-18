# Experiment 075 — self-distillation RealMLP on leakage-clean teacher (recipe: high-card cats excluded)

**Cycle.** 17 (post-audit Phase-1)
**Status.** **Inconclusive (OOF false positive).** Blend OOF cleared +0.00018 over anchor → submitted → LB regressed −0.00026. OOF→LB drift jumped from −0.00061 (anchor) to −0.00105 (this blend). The OOF win does NOT transfer.
**Date.** 2026-05-28

## Hypothesis

Training a student RealMLP on the teacher's soft OOF (continuous teacher_blend probabilities, not the noisy binary PitNextLap) AND excluding the high-cardinality cats from the input — forces the student to find a structurally different representation, producing a ρ-diverse base that lifts the linear blend by ≥ +0.00005 OOF.

## Result

| Metric | Value |
| ------ | ----- |
| Student OOF AUC (vs PitNextLap, single-seed) | 0.95068 |
| Teacher OOF AUC | 0.95430 |
| Δ student − teacher | −0.00362 |
| ρ vs RM6 | 0.97108 |
| ρ vs psRM6r2 | 0.97005 |
| ρ vs CB-exp14 | 0.95655 |
| ρ vs XGB-highbins | 0.96744 |
| ρ vs teacher_blend (leakage-clean) | 0.97251 |

### Blend sweep (rank-remap probe, anchor = `oof_blend_pseudo_r2.parquet` OOF 0.95436)

| w_student | linear | remap | rank_avg | logit_avg | gmean |
| --------- | ------ | ----- | -------- | --------- | ----- |
| 0.000 | 0.95436 | 0.95436 | 0.95436 | 0.95436 | 0.95436 |
| 0.100 | 0.95450 | 0.95450 | 0.95450 | 0.95450 | 0.95450 |
| 0.150 | 0.95453 | 0.95453 | 0.95453 | 0.95453 | 0.95453 |
| **0.200** | **0.95454** | **0.95454** | 0.95453 | 0.95452 | 0.95453 |
| 0.250 | 0.95452 | 0.95451 | 0.95451 | 0.95449 | 0.95450 |
| 0.300 | 0.95447 | 0.95446 | 0.95445 | 0.95443 | 0.95445 |

All five operators agreed: best blend at w=0.20 linear, OOF **0.95454** (Δ +0.00018 over anchor 0.95436).

### Kaggle submission (3rd of the day)

`submission_blend_selfdistill_linear_w200.csv` → **LB 0.95349** (vs anchor LB 0.95375). **Δ −0.00026 LB**. OOF→LB drift was −0.00105 vs anchor's −0.00061.

## Verdict

**Inconclusive (OOF false positive).** The student's restricted feature set (high-cardinality cats excluded) appears to overfit the OOF distribution in a way that the test set doesn't reward. Two non-exclusive interpretations:

1. **Cat-exclusion overfitting**: removing Driver/Race forces the student to over-rely on its remaining inputs in the training distribution; test predictions become more brittle.
2. **Mild distribution shift**: small train↔test feature shifts hit narrower feature sets harder. The audit's `exp 056` ablation already showed external data carries a shift signal; this might be a second-order version of the same effect.

Either way, +0.00018 OOF → −0.00026 LB is a **clean experimental rejection** of using OOF as the sole submission gate. The audit's framing — "OOF→LB drift uncertainty is wider than likely improvements" — was prescient and the submitted result is the empirical confirmation.

## Kill-criteria check

- [x] Blend OOF cleared +0.00005 (audit gate fired positive).
- [x] Submission LB regressed −0.00026 (regression triggers the practical kill criterion).

## Implications for the audit's Phase-1 plan

- **Rank-target (exp 073)**: full failure at fold-1.
- **Self-distill (this exp)**: OOF false-positive.
- Both training-objective diversity experiments closed with no transferable lift.
- Per the Slot-D rule (`rank-target ≤ 0.94 AND self-distill < +0.00005 blend OOF`), the OOF gate technically passed (+0.00018 ≥ +0.00005), so DAE remains nominally in play. **But the empirical LB regression should lower the DAE prior substantially.** DAE pretrain (6-10h MPS) on the same data manifold has correlated risk: a label-free pretrained MLP trained on the same train+test concat may yield the same kind of OOF mirage.
- **Decision**: continue with P1-#3 noise-weighted RM (different mechanism — data weighting, not target change). If P1-#3 also OOF-positive but LB-flat, **descope DAE here** even though the OOF-only rule allows it; the empirical signal is consistent with "no Phase-1 lever transfers."

## Repro stamp

- Trainer: [src/research/train_realmlp_selfdistill.py](../src/research/train_realmlp_selfdistill.py); device='mps'; teacher = `oof_blend_pseudo_r2_xgb2.parquet`; `EXCLUDE_CATS=True` (drops Driver, Race; keeps Compound).
- Outputs: `data/oof_realmlp_selfdistill_s42.parquet`, `data/submission_realmlp_selfdistill_s42.csv`, `data/submission_blend_selfdistill_linear_w200.csv`, `data/blend_rt_sweep_selfdistill.parquet`.
- Compute: 1637s (~27 min) total, M1 MPS, ~5 min/fold.
- Submission: 3/5 today (53125740) — LB 0.95349.

## Learnings

1. **OOF-only submission gates are unreliable when the candidate alters the input distribution** (here: cat-exclusion). Future Phase-1 candidates with structural feature-set changes should be gated jointly on OOF lift AND OOF→LB drift consistency (e.g., the candidate's per-base drift relative to the anchor).
2. **Cat-exclusion as a diversity mechanism over-corrects.** ρ vs anchor 0.972 — barely more diverse than the existing GBDT zoo (ρ 0.97-0.98). The structural change didn't buy meaningful new information.
3. **Drift estimation needs base-level monitoring.** The anchor's OOF→LB drift of −0.00061 is averaged across ~four well-distributed pseudo-RM seeds plus CB-exp14 plus pseudo-XGB. A narrower-trained component widens the drift envelope.

## Follow-ups

- Closed: self-distill on the leakage-clean teacher with cat-exclusion.
- Still open: self-distill *without* cat-exclusion (would be near-zero diversity though — student would just track teacher).
- Next: noise-weighted RM (P1-#3), mechanism-different.
- DAE pretrain: gated on noise-weighted outcome AND a hard LB-transfer check before commit.
