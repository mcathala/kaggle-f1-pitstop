# Experiment 078 — XGB external-downweight + is_original flag (drift mechanism)

**Cycle.** 18
**Status.** Inconclusive on OOF (neutral, +0.00003); LB-transfer test pending (matched-blend submission).
**Date.** 2026-05-29

## Hypothesis

Our OOF→LB drift worsens from −0.00024 (RealMLP standalone) to −0.00061 (full blend). One candidate cause: we train every base on full-weight external data that is distribution-shifted from the competition test set (adv-AUC ~0.78; external pit-rate 25.5% vs synthetic 19.9%). Down-weighting external rows (sample_weight 0.7) and adding an `is_original` flag should let the model calibrate to the competition/test distribution → smaller drift → higher LB at equal OOF.

## Rationale

1. Drift analysis (this round's audit): drift grows with reliance on the external-augmented bases.
2. Research confirmed external is the standard augmentation but is distribution-shifted; the recommended mitigation is exactly `is_original` flag + `sample_weight≈0.7`.
3. The flag lets XGB learn a distinct response to shifted rows; at inference every test row is `is_original=0`, so the model uses the competition-distribution branch.

## Result (OOF)

| | OOF AUC | per-fold mean ± std |
| --- | --- | --- |
| canonical XGB-highbins | 0.95263 | — |
| **+ extweight(0.7) + is_original (this exp)** | **0.95266** | 0.95266 ± 0.00057 |
| **Δ** | **+0.00003** | OOF-neutral |

ρ vs RM 0.980, vs CB 0.983 — identical to canonical XGB (the treatment doesn't change rank structure, only the external-row influence).

**OOF is neutral — exactly the case where OOF cannot measure the hypothesis.** The treatment is "free" on OOF; the entire question is LB transfer, which requires a submission.

## LB-transfer test (designed, pending submission)

Two matched single-seed blends (RM single-seed cancels the multi-seed penalty), differing ONLY in external treatment:

| Blend | RM (0.675) | CB (0.075) | XGB (0.250) |
| ----- | ---------- | ---------- | ----------- |
| baseline_ss | RM-r3 single (no flag) | CB-exp14 | psXGB (full-ext) |
| drift_ss    | RM-isorig single (flag) | CB-exp14 | XGB-extweight (0.7-ext) |

If `drift_ss` LBs above `baseline_ss`, the external-downweight + is_original treatment cuts the drift → justifies a full 6-seed RM-isorig rebuild. (`src/research/build_drift_test_blends.py`)

## Repro stamp

- Trainer: [src/research/train_xgb_extweight.py](../src/research/train_xgb_extweight.py) (fork of `train_xgb_richcat.py`; `EXT_WEIGHT=0.7`, `is_original` feature, DMatrix `weight=`).
- Output: `data/oof_xgb_extweight.parquet`, `data/submission_xgb_extweight.csv`.
- 5-fold StratifiedKFold(shuffle=True, random_state=42); max_bin=5000.

## Learnings (interim)

1. **External-downweight to 0.7 + is_original is OOF-neutral** — confirms external still contributes its value at reduced weight (vs exp 056's full-removal which cost −0.00073). So the treatment doesn't sacrifice OOF, making it a pure LB-transfer bet.
2. The OOF-neutral result is the strongest possible setup for an LB-only hypothesis test: any LB movement is attributable to the drift treatment, not to an OOF strength change.

## Follow-ups

- Pending: submit the matched drift_ss vs baseline_ss blends; compare LB.
- Gated: full 6-seed RM-isorig rebuild only if drift_ss > baseline_ss on LB.
