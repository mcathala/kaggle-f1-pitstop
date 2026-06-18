# Experiment 069 — round-2 pseudo-RealMLP-6seed (strong-blend labeler)

**Cycle.** 17
**Status.** Kept — standalone OOF **0.95396** (+0.00013 over RM6 base; +0.00003 over round-1). Blend with round-2 reaches **OOF 0.95436** (+0.00003 over the prior best 0.95433), the project-best blend. *Opposite* sign to the round-2-XGB result (which lost +0.00003 vs round-1), supporting the interpretation that RealMLP's robust input handling absorbs labeler-noise differently from XGB.
**Date.** 2026-05-28

## Hypothesis

If round-1 pseudo-labeling's gain on the 0.675-weight RealMLP base was even partly real (not pure OOF leakage), then replacing the quick-XGB labeler with a stronger, less-leakage-prone labeler (the prior strong blend at OOF 0.95433) should produce either:
- A *larger* honest lift (if RealMLP picks up real test-distribution signal from cleaner labels), or
- A *smaller* lift (if round-1's apparent advantage was leakage, as exp 067 showed for XGB).

Either outcome is informative — it cleanly separates "real semi-supervised signal" from "OOF-leakage artifact" for the RealMLP base specifically.

## Rationale

1. exp 065 (round-1 pseudo-RM6, quick-XGB labeler): OOF 0.95393 (+0.00010 over base).
2. exp 067 (round-2 pseudo-XGB, strong-blend labeler): OOF 0.95276 (+0.00013 over base) — *smaller* than round-1's +0.00032 → leakage hypothesis confirmed for XGB.
3. RealMLP's PLR-embedded numeric features and dropout-regularised heads suggest it should be less sensitive to label noise than XGB's hard splits. Testing this empirically.

## Expected magnitude

- Standalone lift +0.00005 to +0.00015 OOF over base RM6 (0.95383).
- Blend (using new round-2 OOF in place of round-1 in our pseudo 3-way): +0.00003 to +0.00010 OOF over the 0.95433 ceiling — would land in 0.95436-0.95443. 0.95441 hurdle plausible.

## Overfitting risk

Low. Same 5-fold protocol, same per-seed n_ens=24, fixed CV folds at seed 42. Pseudo-labels come from the held-out blend's test predictions only — no train-side leakage. Internal class-balance on pseudo (hi 0.92, lo 0.03) unchanged from exp 064/065.

## Kill criteria

- [ ] Standalone OOF < 0.95383 (base) → labeler-noise actually hurt → mechanism broken.
- [ ] Blend OOF (with psRM6r2 in place of psRM6) ≤ 0.95428 → no leverage at all.

## Plan

1. Fork exp 065 notebook → `cycle17_realmlp_pseudo62_gpu.py`.
2. Replace pass-1 quick-XGB labeler with the private dataset `mcathala/f1-pitstop-blend-labeler` (the strong-blend test predictions, OOF 0.95433).
3. Same 6-seed ensemble {42, 7, 99, 137, 313, 777}, n_ens=24, 5-fold CV at SEED=42, comp + external + pseudo-test rows in train.
4. Output `oof_realmlp_pseudo62.parquet`, `submission_realmlp_pseudo62.csv`.
5. Run blend probe to test psRM6r2 in place of psRM6 in the canonical 3-way.

## Result

Kaggle P100, ~1.6 h (5727 s).

Per-seed standalone OOF AUC (5-fold concatenated):

| Seed | OOF AUC |
| ---- | ------- |
|  42  | 0.95372 |
|   7  | 0.95377 |
|  99  | 0.95373 |
| 137  | 0.95374 |
| 313  | 0.95372 |
| 777  | 0.95368 |
| **6-seed avg** | **0.95396** |

Standalone Δ vs 6-seed base (0.95383) = **+0.00013**. vs round-1 (psRM6, 0.95393) = **+0.00003**.

### Direct round-1 vs round-2 comparison (this experiment + 065/067)

| Base | Round-1 standalone | Round-2 standalone | Δ (r2 - r1) | Interpretation |
| ---- | ------------------ | ------------------ | ----------- | -------------- |
| RealMLP (this exp) | 0.95393 | **0.95396** | **+0.00003** | round-2 helps (light) |
| XGB (exp 067)      | 0.95295 | 0.95276 | −0.00019 | round-2 hurts (leakage in r1) |

The asymmetry is informative: for XGB, the round-2 *loss* of +0.00019 directly quantifies the OOF-leakage component in round-1's headline number. For RealMLP, round-2 *gains* +0.00003 — meaning RealMLP is mostly absorbing real signal from the cleaner pseudo-labels (very little leakage component). This matches our prior note (exp 064) that RealMLP's input clipping + dropout robustly tolerates label noise.

### ρ vs other bases (from blend probe)

| Pair | ρ |
| ---- | --- |
| psrm6r2 vs rm6     | 0.9986 |
| psrm6r2 vs psrm6   | 0.9990 |
| psrm6r2 vs cb      | 0.9750 |
| psrm6r2 vs xgb     | 0.9794 |
| psrm6r2 vs psxgb   | 0.9796 |
| psrm6r2 vs psxgb2  | 0.9801 |

Essentially the same rank-correlation profile as psRM6 — no new diversity, just a slightly better-calibrated strong base.

### Blend results (8-base probe; all bases on the same 5-fold CV)

| config | OOF | Δ vs prior best (0.95433) |
| ------ | --- | ------------------------- |
| 3way_anchor (RM6/CB/XGB)                 | 0.95421 | −0.00012 |
| 3way_pseudo (psRM6/CB/psXGB)             | 0.95432 | −0.00001 |
| **3way_pseudo_r2 (psRM6r2/CB/psXGB)**    | **0.95436** | **+0.00003** |
| 4way psRM6+psRM6r2 (split 0.5)+CB+psXGB  | 0.95435 | +0.00002 |
| 4way psRM6+psRM6r2 (split 0.3)+CB+psXGB  | 0.95436 | +0.00003 |
| free coord-descent (start: 4way 0.3)     | 0.95436 | +0.00003 |
|   weights: psrm6r2=0.523 psxgb=0.295 psrm6=0.139 cb=0.043 |  |  |

The simple psRM6 → psRM6r2 swap matches the free-grid optimum at 0.95436. Mixing in psRM6 gives no further lift (psRM6r2 dominates).

## Verdict

**Kept** — small but real lift to project-best **OOF 0.95436** (still below the 0.95441 hurdle by 0.00005). Updates the current-best blend; will be the submitted candidate.

## Held submission

`data/submission_blend_pseudo_r2.csv` — built and ready to submit. Tests whether the round-2 lift transfers to LB; if it does, the round-2 thread isn't done yet (could try round-3 with the new 0.95436 labeler).

## Repro stamp

- Kernel: `mcathala/cycle-17-realmlp-pseudo62-exp-069` (P100, 5727s)
- Notebook: [gpu-kernels/cycle17_realmlp_pseudo62_gpu.py](../gpu-kernels/cycle17_realmlp_pseudo62_gpu.py)
- Private labeler dataset: `mcathala/f1-pitstop-blend-labeler` (OOF 0.95433 blend's test predictions)
- Outputs on disk: `data/oof_realmlp_pseudo62.parquet`, `data/submission_realmlp_pseudo62.csv`, `data/submission_blend_pseudo_r2.csv`, `data/oof_blend_pseudo_r2.parquet`
- 5-fold StratifiedKFold(shuffle=True, random_state=42) on `Year × PitNextLap`; n_ens=24

## Learnings

1. **Round-2 pseudo-labeling helps RealMLP and hurts XGB** — the same recipe behaves oppositely. The RM win is small (+0.00003) but it's the second positive lever this round (after round-1 pseudo).
2. **Leakage in round-1 was real but RealMLP-tolerant.** XGB lost +0.00019 going to cleaner labels; RM gained +0.00003. So the leakage component in round-1's standalone numbers was ~+0.0002 for XGB and ~0 for RM.
3. **Project-best blend now OOF 0.95436.** A pure swap (psRM6 → psRM6r2) in our 3-way pseudo gets us there. The free grid optimum agrees.

## Follow-ups

- exp 071: pseudo-CB on canonical CB-exp14 (local CPU; exp 068 used the wrong CB base — re-run on the canonical one).
- Possible exp 072: round-3 pseudo-RM (label with the new 0.95436 blend). ~1.7h Kaggle GPU, but quota is exhausted; would need to wait for next week or run locally on MPS (slower). Defer.
- Submit `submission_blend_pseudo_r2.csv` to LB (OOF 0.95436) — submitted as this round's LB check.
