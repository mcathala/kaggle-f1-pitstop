# Experiment 051 — within-stint trajectory (lag) features on XGB-highbins

**Cycle.** 16
**Status.** Inconclusive (Reverted) — lag features barely move XGB's ranking (ρ 0.997 vs plain XGB) and cost −0.0002 standalone; no blend lift.
**Date.** 2026-05-27

## Hypothesis

Within-stint backward-lag features (lag-1/2 + lap-over-lap deltas of LapTime, LapTime_Delta, Cumulative_Degradation, Position, grouped by Year×Race×Driver×Stint) give XGB-highbins the recent-trajectory signal it lacks — targeting our cycle-10 probe-2 worst-loss slice (degraded tyre × losing position × pit-cluster) — lifting standalone OOF by ≥ +0.00020 or producing enough diversity (rank-corr vs RealMLP < 0.98) to lift the 3-way blend OOF ≥ +0.00020 over 0.95421.

## Rationale

- Every base sees only the *current* lap. The within-stint trajectory (pace dropping, positions slipping over recent laps) is the dynamic our probe-2 EDA flagged as the worst-loss slice, and no base captures it.
- Tree models tolerate added features (unlike RealMLP, which broke in exps 018/019), so XGB-highbins is the right host.
- Lagging only neutral inputs — never PitStop/target — avoids the label-proxy regression seen with forward features.

## Expected magnitude

- Standalone +0.0002 to +0.0010, or blend +0.0001 to +0.0003 via diversity.
- Floor: standalone < 0.95243 AND rank-corr vs RealMLP ≥ 0.984 → revert.

## Kill criteria

- [x] Standalone OOF < plain XGB-highbins by > min_delta with no diversity gain — **effectively FIRED** (−0.00021 standalone, lag-vs-plain-XGB ρ=0.997).
- [x] Best blend config does not clear anchor + 0.00005 — **FIRED** (best 0.95420).

## Result

5-fold, same CV (StratifiedKFold seed 42 on Year×PitNextLap), cycle-11 XGB-highbins HPs verbatim; only the 13 lag/trajectory features added (145 features total).

### Standalone — uniform small penalty

| Fold | lag-FE | plain XGB-highbins | Δ |
| ---- | ------ | ------------------ | --------- |
| 1 | 0.95311 | 0.95331 | −0.00020 |
| 2 | 0.95289 | 0.95309 | −0.00020 |
| 3 | 0.95206 | 0.95220 | −0.00014 |
| 4 | 0.95150 | 0.95174 | −0.00024 |
| 5 | (in mean) | 0.95283 | — |
| **OOF** | **0.95242** | **0.95263** | **−0.00021** |

per-fold std 0.00058. The lag features cost a uniform ~−0.0002 of standalone strength on every fold.

### Rank-correlation matrix (OOF)

|     | rm | cb | xgb | lag |
| --- | -- | -- | --- | --- |
| rm  | 1.0000 | 0.9758 | 0.9799 | 0.9787 |
| cb  | 0.9758 | 1.0000 | 0.9840 | 0.9819 |
| xgb | 0.9799 | 0.9840 | 1.0000 | **0.9973** |
| lag | 0.9787 | 0.9819 | **0.9973** | 1.0000 |

**lag-XGB vs plain-XGB ρ = 0.9973** — the lag features barely changed XGB's ranking. lag-XGB is *not* meaningfully more diverse from RealMLP (0.9787) than plain XGB is (0.9799).

### Blend probe (anchor = cycle-11 3-way, OOF 0.95420)

| Config | OOF | Δ |
| ------ | --- | --- |
| A swap lag→xgb | 0.95416 | −0.00005 |
| B 4-way (any split of the 0.250 slot) | 0.95416–0.95420 | ≤ −0.00001 |
| C two-FE XGB average | 0.95419 | −0.00002 |
| D free 4-way grid | 0.95420 | −0.00001 → **w_lag = 0** |

No configuration clears the anchor; the free grid awards the lag base zero weight.

## Verdict

**Inconclusive (Reverted).** Within-stint lag features do not help. They slightly weaken XGB standalone (−0.00021) without adding diversity — lag-XGB and plain-XGB agree at ρ=0.997, so the new features didn't shift the model's ordering. The free 4-way blend grid sets w_lag=0.

## Kill-criteria check

- [x] Standalone below plain XGB with no diversity — **FIRED**.
- [x] Blend does not clear anchor + 0.00005 — **FIRED** (best 0.95420).

## Repro stamp

- trainer: [src/research/train_xgb_lagfe.py](../src/research/train_xgb_lagfe.py); blend probe [src/research/blend_lagfe_probe.py](../src/research/blend_lagfe_probe.py)
- packages: xgboost 3.2.0
- runtime: 5 folds × ~13 min = ~63 min CPU (M1 Pro)
- inputs: `data/{train,test}.csv` + `data/f1_strategy_dataset_v4.csv`; outputs `data/oof_xgb_lagfe.parquet`

## Learnings

1. **XGB-highbins has a stable ranking that small input perturbations don't shift.** Adding 13 trajectory features changed its OOF ranking by only ρ=0.997 vs plain XGB — the same near-determinism signature as multi-seed XGB (exp 047, ρ=0.999). On this dataset/recipe, neither new seeds nor new features move XGB's ordering, so neither can diversify the blend.
2. **The within-stint trajectory signal is already absorbed.** The current-lap features plus the group-statistic features (mean/std/diff by Race_Year, Compound_Stint, etc.) apparently already encode what the lag features would add. Median stint length is 3 laps, so lag-2/3 are mostly NaN — the temporal window is too short to carry independent signal.
3. **Diversity must come from a different model family or a fundamentally different representation, not from feature tweaks to XGB.** Confirms the exp-050 conclusion from the upstream side.

## Follow-ups

- Closed: lag/trajectory FE on XGB.
- The parallel tyre-overdue FE (exp 052) attacks the same Q4 slice from the degradation axis; if it shows the same ρ≈0.997 stability, the "tweak XGB's features" axis is fully closed and the next lever must be a different representation (e.g., a CatBoost variant on the new features, which may rank differently than XGB).
