# Experiment 079 — RealMLP is_original flag + matched drift-test (external/drift hypothesis)

**Cycle.** 18
**Status.** **Reverted — drift hypothesis FALSIFIED.** Downweighting/flagging external worsened both OOF and LB; the −0.0006 drift is not caused by external over-weighting.
**Date.** 2026-05-29

## Hypothesis

Our OOF→LB drift (−0.0006) is caused by training every base on full-weight, distribution-shifted external data. Adding an `is_original` flag to RealMLP (the 0.675-weight base) + downweighting external on XGB should let the models calibrate to the competition/test distribution → smaller drift → higher LB.

## Result

### Standalone (RealMLP single-seed, is_original flag)

| | OOF AUC |
| --- | --- |
| RM single-seed (no flag, exp 072) | 0.95369 |
| RM single-seed (is_original flag, this exp) | 0.95357 |
| Δ | **−0.00012** (flag costs OOF) |

### Matched drift-test blends (single-seed RM; differ only in external treatment)

| Blend | OOF | **LB** | drift |
| ----- | --- | --- | --- |
| baseline_ss = RM(no-flag) + CB + psXGB(full-ext) | 0.95423 | **0.95367** | −0.00056 |
| drift_ss = RM(is_original) + CB + XGB-extweight(0.7) | 0.95403 | **0.95338** | **−0.00065** |
| Δ (drift − baseline) | −0.00020 | **−0.00029** | drift WIDENED |

## Verdict

**Reverted; hypothesis falsified.** The external-downweight + is_original treatment made the LB *worse* by −0.00029 (more than the −0.00020 OOF gap) and the OOF→LB drift *widened* (−0.00065 vs −0.00056). If external over-weighting caused the drift, downweighting would have *narrowed* it. It did the opposite.

**Conclusion: the −0.0006 drift is structural, not from external distribution shift.** It comes from (a) CV optimism inherent to the rich transductive FE + (b) blend-weight fitting on OOF, neither of which the external treatment touches. External data, even at full weight, is correctly contributing (consistent with exp 056, where full removal cost −0.00073 OOF).

## Kill-criteria check

- [x] drift_ss LB (0.95338) < baseline_ss LB (0.95367) → treatment hurts LB → hypothesis falsified, lever closed.

## Repro stamp

- Trainers: [src/research/train_realmlp_isorig.py](../src/research/train_realmlp_isorig.py), [src/research/train_xgb_extweight.py](../src/research/train_xgb_extweight.py)
- Blend builder: [src/research/build_drift_test_blends.py](../src/research/build_drift_test_blends.py)
- Outputs: `data/oof_realmlp_isorig_s42.parquet`, `data/submission_blend_{baseline,drift}_ss.csv`
- Submissions: 53133003 (baseline_ss → 0.95367), 53133012 (drift_ss → 0.95338). 4/5 daily slots used.

## Learnings

1. **The drift is not fixable by external reweighting.** This was the single biggest candidate LB-lever (a drift fix could have been +0.0003-0.0005); it's now cleanly ruled out.
2. **baseline_ss (single-seed) LB 0.95367 confirms the single-seed→6-seed penalty is ~−0.00008 LB** (6-seed = 0.95375). Our blend recipe is stable and well-characterized.
3. **The −0.0006 drift is the price of (rich transductive FE optimism + OOF-weight fitting).** Reducing it would require a fundamentally less-transductive FE or weight-free blending — both of which would lower OOF more than they'd help LB (tested indirectly across cycles).

## Follow-ups

- Closed: external/drift treatment.
- Remaining honest lever per research: genuine base *diversity* (our bases are ρ 0.96-0.98, saturated by shared rich FE). The 0.9545 field plateau is a blend of *decorrelated* bases. Next: build a deliberately decorrelated base (native-FE CatBoost) and test rank-blend transfer.
- The shared-public-submission blender (the field's actual 0.9545 method) remains off-limits under the project's integrity rules — not pursued.
