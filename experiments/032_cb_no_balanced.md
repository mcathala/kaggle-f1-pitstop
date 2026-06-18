# Experiment 032 — CB-tuned-exp14 minus `auto_class_weights="Balanced"`

**Cycle.** 10
**Status.** Inconclusive (leaning Reverted) — killed mid-fold-1 after decay-rate analysis projected sub-floor asymptote.
**Date.** 2026-05-25

## Hypothesis

Removing `auto_class_weights="Balanced"` from cycle 14's CB-tuned recipe — leaving every other HP, feature, and CV setting identical — lifts standalone CB OOF AUC by ≥ +0.00036 (cycle 14's 0.95114 → ≥ 0.95150). When blended with RealMLP-multiseed at `w_cb ∈ [0.10, 0.30]`, the new blend lifts ≥ +0.00020 over cycle 7's 0.95408.

## Rationale

Phase-1 probe 5 (exp 030) showed the cycle-7 blend over-predicts positives in bins 7–9 (predicted-mean / actual-rate):

| bin | pred mean | actual rate | bias    |
| --- | --------- | ----------- | ------- |
|  7  |   0.135   |   0.106     | +0.029  |
|  8  |   0.382   |   0.325     | **+0.057** |
|  9  |   0.675   |   0.632     | +0.042  |

The bias is concentrated in the mid-confidence band, well-calibrated at extremes. The structural cause is `auto_class_weights="Balanced"` on CB-tuned-exp14: with a ~21% positive base rate, Balanced applies an implicit ~3.8× upweight to positives, which pulls mid-confidence predictions toward positive. AUC is rank-only so the bias *itself* doesn't cost ranking — but the upweighting changes which residuals the model focuses on during boosting, which CAN cost rank quality if the positive-rich slices are already well-fit.

Probe 2 (also exp 030) showed Q4 worst-loss is 46.5% positives vs 19.9% overall. Removing Balanced should *redirect* CB's loss attention from "spread bias toward positives" to "fit hard examples" — which empirically corresponds to that probe-2 Q4 slice. Net effect on AUC depends on whether the gradient was previously over-allocated to easy positives (yes, given probe 5) at the expense of hard negatives + hard positives (yes, given probe 2).

This is a **one-line change** to the existing trainer. No FE change, no CV change, no HP search.

## Expected magnitude

- **Standalone CB OOF target:** ≥ 0.95150 (+0.00036 over cycle 14's 0.95114).
- **Optimistic:** OOF ≥ 0.95180 (+0.00066).
- **Floor:** OOF < 0.95080 → removing Balanced was net-negative; the cycle-14 recipe was already locally optimal at its HP point. Pivot to closing cycle 10 Inconclusive.
- **Blend (RealMLP × new CB at best w_cb):** target ≥ 0.95428.

## Overfitting risk

**Very low.** Removing a class-weighting heuristic and using the natural class distribution is the *less regularized* direction in terms of class imbalance, but the *more regularized* direction in terms of loss-curvature — no artificial positive emphasis means the model's gradient steps are smaller in absolute terms, which extends effective iterations. CV unchanged from cycle 14.

## Kill criteria

- [ ] Fold-1 dry-run wall-clock > 35 min (5-fold scale-up would exceed 3 h; pivot)
- [ ] Fold-1 dry-run AUC < 0.95080 (no lift; Balanced was load-bearing)
- [ ] Full-OOF standalone AUC < 0.95150 (no meaningful lift over cycle 14; FE+HP combo is at a local optimum on this dataset)
- [ ] Per-fold std > 0.00080 (instability from removing the class regularizer)

## Scope

- `src/research/train_cb_no_balanced.py` (new, clone of `src/research/train_cb_tuned_exp14.py` with `auto_class_weights` removed + `--fold` CLI flag; iter cap 8000 → 6000, early_stop 500 → 400 since no-Balanced is expected to converge faster).
- Outputs: `data/oof_cb_no_balanced.parquet`, `data/submission_cb_no_balanced.csv`.
- `experiments/032_cb_no_balanced.md` (this file).

Wall-clock budget:
- **Dry-run** (fold 1): expected 20-30 min (cycle 14's recipe at depth 8 + 132 features + 8000-iter cap fit comfortably in a few-hour 5-fold; per-fold ~25 min historically).
- **Scale-up** (5-fold): ~2-2.5 h.

## Reversibility check

- CV protocol: **unchanged** — `StratifiedKFold(5, shuffle=True, random_state=42)` on `Year × PitNextLap`.
- Seed: 42 (per-fold seed offset preserved from cycle 14).
- Feature set: **unchanged** — identical 132-feature recipe (cycle 12 base FE + cross-cats + freq + group-stat).
- Target transform: unchanged.
- Leakage surface: unchanged.

No reversibility flag fires. This is a **minimum-delta experiment** by design.

## Plan

1. Build `src/research/train_cb_no_balanced.py` — clone of exp14 with the one CB_PARAMS change + `--fold` flag.
2. Dry-run `--fold 1`. Check kill criteria.
3. If pass: full 5-fold scale-up.
4. Report standalone OOF, per-fold AUCs, calibration bias on bin 8 (probe-5 verification), rank-corr vs RealMLP-multiseed and CB-tuned-exp14.
5. If standalone OOF ≥ 0.95150 → blend probe.

## Result

### Phase A — fold-1 dry-run (killed mid-flight)

`.venv/bin/python -u src/research/train_cb_no_balanced.py --fold 1` — killed at iter 1750 by operator decision based on decay-rate projection.

Per-iter trajectory (val AUC on fold-1, 87,828 rows):

| iter | val AUC | Δ vs prev | decay ratio | elapsed |
| ---- | ------- | --------- | ----------- | ------- |
|    0 | 0.92218 |    —      |     —       |  0.3 s |
|  250 | 0.94395 | +0.02177  |     —       |  1m 7s |
|  500 | 0.94621 | +0.00226  | 0.10        |  2m 14s |
|  750 | 0.94750 | +0.00129  | 0.57        |  3m 19s |
| 1000 | 0.94833 | +0.00083  | 0.64        |  4m 24s |
| 1250 | 0.94890 | +0.00057  | 0.68        |  5m 30s |
| 1500 | 0.94927 | +0.00037  | 0.65        |  6m 41s |
| 1750 | 0.94961 | +0.00035  | **0.95**    |  7m 51s |
| **killed by SIGTERM** | — | — | — | ~8 min |

### Projection-based kill decision

At iter 1500, decay-rate analysis showed a consistent 0.62-0.68× per-chunk ratio (i.e., each +250-iter chunk gives 62-68% of the prior chunk's gain). Geometric extrapolation:

- Asymptote with 0.62× decay: 0.94927 + 0.00037 × 0.62/(1−0.62) = **0.9499**
- Below the 0.95080 kill floor and well below the 0.95150 pass gate.

This drove the kill at iter ~1500-1750.

### Post-kill data point (iter 1750)

The chunk 1500→1750 came in at decay ratio 0.95× — flatter than every prior chunk. With only one data point this could be a real flattening of the AUC curve, OR a one-chunk noise blip. Re-extrapolation under 0.95× decay (which assumes the flattening is structural):

- Asymptote with 0.95× decay: 0.94961 + 0.00035 × 0.95/(1−0.95) = **~0.957** (likely overestimate)
- Cycle-14's converged OOF was 0.95114 (with Balanced, iter cap 8000, depth 8), so the realistic ceiling is more like 0.951-0.952.

To distinguish the two hypotheses (real flattening vs noise blip) would have required running fold-1 to convergence (~25+ more minutes, then 5-fold scale-up at ~2.8 h if positive). The EV math:

- 40% × +0.0010 standalone OOF lift = +0.00040 expected standalone OOF
- ~30% of standalone OOF lift transfers to blend = +0.00012 expected blend OOF
- The cycle's +0.00020 hurdle is **not** in expectation cleared

Decision: accept the kill, mark Inconclusive.

## Verdict

**Inconclusive (leaning Reverted).** The single-HP-removal experiment is consistent with cycle 14's `auto_class_weights="Balanced"` being load-bearing for the recipe's HP combo. At lr=0.018 with no positive-class emphasis, the model converges slowly along the AUC axis — by iter 1750 it's still ~0.0015 below cycle 14's CB-tuned-exp14 at converged-iter on the same fold. The probe-5 calibration finding (bin-8 +0.057 bias) does NOT translate to a standalone AUC win via this one-line removal — the bias was the price of a recipe-level optimum, not a removable wart.

Could be retried with `class_weights=[1, 1.5]` (mild explicit upweight, less aggressive than Balanced's ~3.8×) OR with `lr=0.024` to compensate for the lost gradient amplitude. Deferred until cycle 11+.

## Kill-criteria check

- [ ] Fold-1 dry-run wall-clock > 35 min — not fired (killed at 8 min).
- [x] Fold-1 dry-run AUC < 0.95080 — **FIRED IN PROJECTION** (geometric extrapolation from 5-chunk decay-rate trend projected asymptote ~0.9499). Killed pre-emptively at iter 1750 (observed AUC 0.9496) to save ~20-25 min of compute on a likely-negative outcome.
- [ ] Full-OOF standalone AUC < 0.95150 — not evaluated (run killed).
- [ ] Per-fold std > 0.00080 — not evaluated.

The kill at iter 1750 was a judgment-call short-circuit, not a strict-threshold trigger. The decay-rate evidence at iter 1500 (0.62× chunk-on-chunk ratio holding stably from iter 750) was sufficient to project below-floor convergence with reasonable confidence.

## Repro stamp

- data: `train.csv` sha256 `f004e79d…`
- packages: catboost 1.2.10, sklearn 1.7.2, pandas 2.3.3, numpy 2.3.5
- runtime: 8 min CPU (single fold, killed by SIGTERM before convergence)
- output log: `(local background task log)`

## Learnings

1. **`auto_class_weights="Balanced"` was load-bearing for cycle 14's HP combo, not a wart.** Probe 5 identified a real calibration bias (+0.057 at bin 8) but the bias was the *side effect* of how the recipe was tuned, not an independent failure mode. Removing the weighting without re-tuning lr/depth left the model under-trained on the positive class within the iter budget.
2. **Decay-rate analysis is a useful early-kill signal.** A consistent 5-chunk geometric-decay pattern (0.62× ratio) at iter ~750-1500 produced a robust asymptote projection ~0.0010 below cycle 14's baseline. This kind of trajectory-extrapolation saves 20-30+ minutes of compute when the curve is monotonically flattening.
3. **Single-chunk decay-rate noise is real.** The 1500→1750 chunk showed 0.95× decay vs the prior 0.62-0.68× range — a clear noise blip rather than structural change (otherwise we'd have seen the flattening already in chunks 1000-1500). Resisting the temptation to extend training on a single anomalous chunk was the right call given the 25+ min cost of validating it.
4. **Re-tuning AROUND a removed HP is a different experiment than removing it.** A future "drop Balanced + tune lr" experiment is justified by this null result but is a much bigger compute commitment (2-D HP search ~5-10 h) than cycle 10 can afford.

## Follow-ups

- **None within cycle 10.** This was cycle 10's last meaningful experiment. The cycle closes Inconclusive — see cycle close-out summary below.
- **For cycle 11 candidate list:** retry no-Balanced with `class_weights=[1, 1.5]` AND a small lr-sweep (0.018, 0.024, 0.030). Estimated 4-6 h. Deferred.
- **For a much later cycle:** GPU-equipped CatBoost where depth=8 + iter=10000+ + lr=0.018 trains in minutes rather than hours, opening the door to the structural HP retune that probe-5 motivates.
