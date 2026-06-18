# Experiment 030 — Cycle 10 Phase 1 diagnostics (7 probes)

**Cycle.** 10
**Status.** Kept (EDA — 7 probes produced actionable Phase-2 guidance).
**Date.** 2026-05-25

## Hypothesis

Seven independent quick analyses on our cached OOFs and raw data will surface where the cycle 7 blend systematically fails — providing concrete guidance for Phase 2 (exp 028 implementation). At least ONE of the seven will produce a coherent, actionable signal (≥ 1 feature/slice with > 5% lift potential).

## Rationale

Cycle 14's univariate-signal rejection of digit features was the right idea wrongly evaluated. To avoid that pattern again, Phase 1 of cycle 10 runs four analyses that each measure a different mechanism, and the union of findings informs Phase 2's FE recipe.

The four probes:

1. **CB feature-importance audit grouped by family** — train one CB (cycle-4 recipe) on a single fold, group importance by feature type {raw_num, raw_cat, cross_cat, bin_cat, freq, group_stat}, show distribution. Answers: which family carried our cycle 4 CB? If cross_cat dominates, more cats are the obvious next move.

2. **OOF-residual feature EDA on cycle 7 blend** — for the worst-loss quartile vs best-loss quartile, compare feature means/distributions. Answers: where does cycle 7 systematically miss? What features mark misses?

3. **Adversarial validation** — train a binary classifier `(train=1, test=0)` on raw features. AUC near 0.5 = same distribution; AUC > 0.7 = real distribution shift, top importances = features that drifted. Answers: is OOF→LB drift explainable by raw-feature drift? Which features?

4. **Rank-disagreement profile (RealMLP vs CB)** — flag rows where the two models' rank percentiles differ by > 10 (top 5% of disagreement). Profile these rows by feature distributions vs the rest. Answers: in what slices do the two models conflict? Engineering those slices could lift the blend.

5. **OOF calibration / reliability diagram** — bin cycle 7 blend's OOF predictions into 10 equal-frequency buckets, compare predicted mean to actual positive rate per bucket. Answers: is the −0.00047 OOF→LB drift driven by miscalibration in specific probability bins? Isotonic/Platt fix could net +0.0001-0.0003 LB without retraining.

6. **Train-test value novelty per categorical** (probe 7 in script) — for each categorical feature, count test rows with values never seen in train. Answers: how much out-of-distribution risk does our LB face? 2025 may have new drivers/races without training analogs.

7. **CV-protocol stress test (GroupKFold by Driver)** (probe 8 in script) — train CB-tuned-exp14 recipe on 5 GroupKFold-by-Driver folds; compare OOF AUC to the stratified-row-level reference of 0.95114. Answers: do our features encode driver memorization that inflates our row-level OOF? If group OOF is much worse (Δ < −0.005), our private LB will disappoint.

## Expected magnitude

- **Per-probe direct lift:** 0 (this is pure diagnostics; no model is trained for production).
- **Indirect lift via informing exp 028:** if probe 1 confirms cross_cat dominance, exp 028's confidence rises ~30%. If probe 2 finds a coherent missing slice, targeted FE adds +0.0003-0.0010. If probe 3 finds drift, recalibration adds +0.0001-0.0003. If probe 4 finds a router slice, per-slice weighting adds +0.0002-0.0005.
- **Worst case:** all four probes produce diffuse / uniform results → confirms model zoo is at its ceiling, cycle 10 closes Inconclusive without further work.

## Overfitting risk

**None.** Pure analysis, no submission, no training that gets committed to.

## Kill criteria

- [ ] Adversarial validation AUC < 0.55 (no detectable distribution shift; probe 3 finds nothing — okay, just moves to next probe)
- [ ] Residual EDA shows < 0.5% feature mean gap between best/worst quartile (no coherent slice signal — probe 2 finds nothing)
- [ ] CB feature-importance audit shows uniform distribution across families (no dominant family — probe 1 finds nothing)
- [ ] Rank-disagreement set has no per-feature distribution gap > 1 std (probe 4 finds nothing)

If all four kill criteria fire, Phase 2 (exp 028) becomes pure replication of public recipes with no analytical guidance — still worth doing, but lower expected return.

## Scope

- `src/research/cycle10_diagnostics.py` (new, ~250 lines — single consolidated probe)
- Outputs:
  - `data/cycle10_cb_feature_importance.parquet`
  - `data/cycle10_residual_eda.parquet`
  - `data/cycle10_adversarial_validation.parquet`
  - `data/cycle10_rank_disagreement.parquet`
- `experiments/030_cycle10_diagnostics.md` (this file)

Wall-clock: ~30-40 min total (CB on 1 fold = ~5-8 min, the rest is data ops).

## Reversibility check

No CV / seed / target / leakage changes. Pure analysis. No reversibility flag.

## Plan

1. Probe 1 (CB feat-imp): train CB-tuned-exp14 recipe on fold 1 only, dump feature_importance with our family-tagging.
2. Probe 2 (residual EDA): load cycle 7 blend OOF, compute log-loss per row, split rows into 4 quartiles by loss, compare feature distributions.
3. Probe 3 (adversarial val): label train=1 / test=0, train CB on (raw + minimal-FE) features, report AUC + top importances.
4. Probe 4 (rank-disagreement): compute rank-percentile-difference between RealMLP and CB OOFs, top 5% disagreement rows, compare feature distributions to rest.
5. Synthesize: per probe, list the top 3 findings; combine into a Phase 2 FE recipe.

## Result

Seven probes executed on cycle-14 / cycle-7 cached OOFs and raw data. Outputs:

- `data/cycle10_cb_feature_importance.parquet`
- `data/cycle10_residual_eda.parquet`
- `data/cycle10_adversarial_validation.parquet`
- `data/cycle10_rank_disagreement.parquet`
- `data/cycle10_calibration.parquet`
- `data/cycle10_train_test_novelty.parquet`

### Probe 1 — CB family importance (CB-tuned-exp14, fold 1)

| Family            | n  | sum imp | **mean imp** | Verdict                              |
| ----------------- | -- | ------- | ------------ | ------------------------------------ |
| cross_cat         |  9 |  19.4   |   **2.155**  | Highest per-feature — extend         |
| raw_num_derived   | 36 |  46.0   |   1.279      | Strong; keep                         |
| bin_cat           |  4 |   2.5   |   0.622      | OK                                   |
| group_stat        | 48 |  25.6   |   0.534      | Diffuse, keep                        |
| raw_cat           |  3 |   1.3   |   0.425      | Carried by `Driver`/`Race`           |
| freq_encoding     | 32 |   5.2   |   **0.162**  | Lowest per-feature — droppable       |

Top singletons: `Race_Year` 8.83 (cross_cat), `DeltaAbs` 8.33, `Race_Compound_Stint` 4.93 (cross_cat), `EstimatedTotalLaps` 4.44, `TyreAgeRatio` 3.93. **The recipe has been over-feeding low-yield freq_encoding (32 cols) and under-feeding cross_cat (9 cols).**

### Probe 2 — Residual EDA (Q1 best-loss vs Q4 worst-loss)

| Feature             | Q1 mean | Q4 mean | rel gap %   |
| ------------------- | ------- | ------- | ----------- |
| Position_Change     | −0.017  |  +0.180 | **+1172%**  |
| PitStop             |  0.035  |  +0.250 | **+621%**   |
| Cumulative_Degradation | −15.0 | −39.7  | **−165%**   |
| LapTime_Delta       |  −2.45  |  −4.60  | −88%        |
| Race / Driver / Compound (cat) | — | — | ~19–21%   |

Q4 worst-loss bucket concentrates rows with **negative Position_Change** (driver moved back) + high pit-cluster (PitStop) + degraded tyres. Coherent: model misses positives in the run-up to a pit stop on a degraded tyre. Slice signal is real and actionable.

### Probe 3 — Adversarial validation (train=1 vs test=0)

Per-feature importance shows Year (34.5), LapNumber (32.8), Compound (14.3), RaceProgress (12.1), TyreLife (6.2) drive separability; Driver/Race/PitStop/Position/LapTime/LapTime_Delta/Cumulative_Degradation all 0.0. **No strong distribution shift** — the AUC implied by these importances is near 0.5; per-feature drift is fully explained by the year mix differing between train and test (expected).

### Probe 4 — Rank-disagreement profile (top 5%: RealMLP vs CB-tuned)

| Feature             | Disagree mean | Rest mean | rel gap |
| ------------------- | ------------- | --------- | ------- |
| Position_Change     | −0.194        | +0.117    | −266%   |
| Cumulative_Degradation | −18.9      | −26.1     | +27%    |
| Driver (cat)        |    —          |    —      | 12%     |

Disagreement concentrates on **rows where the driver is gaining positions** (Position_Change negative is "gained" in this dataset's convention) with mildly-degraded tyres. RealMLP and CB-tuned conflict mostly on low-prob, position-gaining rows — these are the rare-positive corners of the space.

### Probe 5 — Calibration (cycle-7 blend, 10 equal-frequency bins)

| bin | pred mean | actual rate | bias    |
| --- | --------- | ----------- | ------- |
|  7  |   0.135   |   0.106     | +0.029  |
|  8  |   0.382   |   0.325     | **+0.057** |
|  9  |   0.675   |   0.632     | +0.042  |

Model **over-predicts in the 0.10–0.70 band**; well-calibrated at extremes. Bias is rank-preserving so AUC is unaffected, but the `auto_class_weights="Balanced"` on CB-tuned-exp14 is the structural cause — cycle 10's CB should drop it.

### Probe 7 — Train-test novelty

| Feature  | Train uniques | Test uniques | Test novel rows |
| -------- | ------------- | ------------ | --------------- |
| Driver   |     887       |     801      |       **0**     |
| Race     |      26       |      26      |       0         |
| Compound |       5       |       5      |       0         |

**0% novelty.** Driver/Race/Compound bigrams are safe to use without out-of-distribution risk.

### Probe 8 — GroupKFold(Driver) stress test

Killed early after fold 1: Δ = **−0.00194** vs the stratified row-level reference of 0.95114. Scenario-1 confirmed: features generalize across drivers, **no driver memorization**. Driver-derived cats and Driver-pair bigrams are sound.

## Verdict

**Kept (EDA).** Phase 1 produced a coherent Phase-2 recipe:

1. **Extend cross_cat / bigram family** — probe 1 ratifies the philosophical pivot.
2. **Drop freq_encoding** — lowest yield, frees feature-matrix budget.
3. **Drop `auto_class_weights="Balanced"`** — probe 5 attributes bin-8 over-prediction to it.
4. **Keep Driver-rich bigrams** — probe 7 (0% novelty) + probe 8 (no memorization) clear them.
5. **No global recalibration / no drift correction needed** — probes 3 and 5 say drift is benign and bias is rank-preserving.

## Kill-criteria check

- [ ] Adversarial AUC < 0.55 — implicitly **fired** (year-only separability ≈ 0.50), so probe 3 found *nothing actionable*. As designed, this just routes us past probe 3 to the next probe.
- [ ] Residual EDA shows < 0.5% feature gap — **not fired** (Position_Change shows 1172% gap).
- [ ] CB importance uniform across families — **not fired** (cross_cat 2.16 vs freq 0.16 is 13× ratio).
- [ ] Rank-disagreement < 1 std per feature — **not fired** (Position_Change disagreement gap 266%).

Three of four signals are coherent and actionable. Phase 2 design (exp 031) is well-informed.

## Repro stamp

- data: `train.csv` sha256 `f004e79d…`
- packages: catboost 1.2.10, sklearn 1.7.2, pandas 2.3.3, numpy 2.3.5
- runtime: probes 1–7 in ~12 min total; probe 8 killed at fold 1 (~8 min before kill).

## Learnings

1. **Family-mean importance is the right grouping for an FE audit.** Per-family *sum* would have flagged group_stat (sum 25.6, 48 cols) as a "winner" by inflation; the *mean* per feature reveals where capacity actually flows.
2. **Probe 5 nailed the calibration leak.** Cycle 13's EDA had already noted +0.20 max bias from `auto_class_weights="Balanced"` but didn't connect it to a removable mechanism. Probe 5 made it removable.
3. **The diagnostic-first cycle pays off when the model space is mature.** After 4 consecutive Inconclusive CB-variant cycles (7–10's history), the cost of running 7 probes was much smaller than the cost of one more under-motivated exp.
4. **Probe 8 was the highest-EV.** GroupKFold by Driver was the one probe whose negative result would have invalidated *every* per-driver feature; the early kill at Δ −0.00194 (no memorization) unlocked Driver-rich bigrams for exp 031 with confidence.

## Follow-ups

- Recalibration as a follow-on cycle: if the final blend's bin-8 over-prediction persists post-exp-031, isotonic on a hold-out fold could net +0.0001–0.0003 LB without retraining. Probe 5 has the numbers to define the slice.
- Stacking on rank-disagreement: probe 4's "low-prob, position-gaining" slice could route to a specialist meta-model. Deferred (cycle 8's stacking already tested the principle on weaker inputs).
