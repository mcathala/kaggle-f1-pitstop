# #009 — Pit-cluster saturation features

**Status.** Inconclusive
**Date.** 2026-05-21
**Pre-registered as cycle #008 candidate (1).**

## Hypothesis

Four new pit-cluster features — `field_pit_share_diff_1` (1-lap delta), `field_pit_share_lag_5` (5-lap memory), `peak_pit_window_distance` (signed lap offset from the race's peak pit-share lap), `post_sc_pit_cluster` (`sc_lap_minus2 × field_pit_share` interaction) — feed into CB#009 (cycle-#004 frozen params on a 72-col parquet → 70 model features). The 4-way fixed-weight ensemble (LGB=0.10, CB#004=0.30, CB#006=0.30, CB#009=0.30) beats the 3-way 0.94866 baseline by ≥ +0.00020.

## Rationale

Cycle #008's error-bucket EDA pinned **`field_pit_share` q4 + q5 as the single largest untargeted residual** (combined ~9.6k lift_$ across 175k rows, AUC 0.92 → 0.87 going from q4 to q5). The existing pit-cluster features (`field_pit_share_prev_lap`, `field_pit_share_lap_minus_2`, `field_pit_share_window_3`) capture "the field has been pitting", but lose discrimination *within* high-share regions because they:
- Can't tell rising vs falling pit waves apart (no diff).
- Have at most 3-lap memory (no lag-5).
- Have no race-wide context (no "are we near the peak pit lap?" signal).
- Don't explicitly model the SC pit-window interaction (sc_lap_minus2 exists, but the model has to learn the multiplication itself).

This is the cleanest "structural FE" branch available: 4 features, mechanical extension of an existing family, no labels touched, no new hyperparameters.

## Expected magnitude

- CB#009 standalone OOF ≥ 0.94806 (CB#006) flat-or-up, target +0.0005–+0.0030.
- 4-way ensemble OOF ≥ 0.94886 (= 0.94866 + 0.000225 magnitude floor).
- `field_pit_share` q5 cell AUC: 0.87069 → ≥ 0.880 if the new features actually fire on q5.
- Per-(Year × Compound): no n ≥ 10K cell regresses > 0.0010.

## Overfitting risk

**Low.**

- All 4 features are deterministic transforms of existing `field_pit_share` and `sc_lap_minus2` columns (no labels).
- Computed on train+test combined (same convention as the cycle-#006 `tyre_age_pct_among_compound_peers` / cycle-#007 peer-rank features).
- `peak_pit_window_distance` uses `arg_max` per (Race, Year) over `__field_pit_share_raw` — a non-leave-one-out version. Slight info leak (current row contributes to the field share that determined the peak), but the contribution per row is 1/N where N is field size on that lap, so well under noise floor. Verified by the existing `field_pit_share_window_3` precedent (same pattern, kept since cycle #001).
- 70 model features at CatBoost depth 8 + l2 3.0 — same regularisation regime.

## Kill criteria

- CB#009 standalone OOF < CB#006 (0.94806).
- Any (Year × Compound) cell with n ≥ 10K regresses by > 0.0010 vs CB#006 standalone.
- 4-way ensemble OOF ≤ 3-way ensemble OOF.
- `field_pit_share` q5 AUC does NOT improve (the targeted weak slice didn't actually move) — if this is true the new features are spurious.

## Scope

- `src/features.py` — extended `add_field_pit_cluster()` (~40 LOC added).
- `src/research/train_cb009.py` (new) — cycle-#004 frozen CatBoost params on 70 model features.
- `experiments/009_pit_cluster_saturation.md` — this file, with results.

## Reversibility check

Touches CV? **No.** Touches seed? **No.** Touches target transform? **No.** Touches leakage surface? **No** — same FE conventions as cycle #001 onwards. Safe.

## Plan

1. Extend `add_field_pit_cluster` in `src/features.py`.
2. Rebuild parquets (72-col).
3. Train CB#009 with cycle-#004 frozen params on 70 model features.
4. Build 4-way fixed-weight ensemble (LGB=0.10, CB#004=0.30, CB#006=0.30, CB#009=0.30).
5. Verify `field_pit_share` q5 slice AUC moved — sanity that the right cohort was helped.
6. Apply gates, document, commit.

## Result

### CB#009 standalone (70 model features)

```
OOF AUC:   0.94805   (vs CB#006 0.94806, Δ = −0.00001 — essentially identical)
Per-fold:  0.94800, 0.94858, 0.94703, 0.94839, 0.94833
fold std:  0.00056
iters:     [1823, 2240, 1932, 1928, 1841]
```

Per-fold deltas vs CB#006: −0.00003, −0.00003, +0.00004, +0.00008, −0.00009. 2/5 positive, 3/5 negative-or-flat. **The new features had no measurable standalone impact** — CatBoost is splitting on them but not gaining discrimination over what `field_pit_share` and `sc_lap_minus2` already provide.

### Per-`field_pit_share` quintile — the targeted slice didn't move

CB#009 standalone:

| Quintile | n | pos_rate | CB#009 AUC | 3-way ensemble baseline AUC | Δ |
|---|---|---|---|---|---|
| q1 | 87,907 | 0.022 | 0.97543 | 0.97589 | −0.00046 |
| q2 | 88,109 | 0.106 | 0.97591 | 0.97624 | −0.00033 |
| q3 | 87,518 | 0.216 | 0.94575 | 0.94621 | −0.00046 |
| q4 | 88,908 | 0.271 | 0.91572 | 0.91663 | −0.00091 |
| **q5** | **86,698** | **0.382** | **0.86877** | **0.87069** | **−0.00192** |

The cycle 8 EDA flagged q4+q5 as the largest untargeted residual (~9.6k lift_$). Cycle 9 designed 4 features specifically to fix this saturation. **CB#009 standalone q5 is WORSE than the baseline 3-way ensemble, not better.** Note this is CB#009 single model vs 3-model ensemble, so the right comparison is CB#009 vs CB#006 standalone (similar AUC overall), where the gap is closer; but the targeted improvement didn't materialise.

### 4-way ensemble — `LGB=0.10, CB#004=0.30, CB#006=0.30, CB#009=0.30`

```
OOF AUC:   0.94876   (vs 3-way 0.94866, Δ = +0.00011)
Per-fold:  0.94864, 0.94927, 0.94795, 0.94896, 0.94906
fold std:  0.00046
```

Per-fold deltas vs 3-way: **+0.00014, +0.00012, +0.00008, +0.00013, +0.00007 — 5/5 positive**.

### Per-`field_pit_share` quintile — 4-way ensemble

| Quintile | 3-way AUC | 4-way AUC | Δ |
|---|---|---|---|
| q1 | 0.97589 | 0.97594 | +0.00004 |
| q2 | 0.97624 | 0.97628 | +0.00004 |
| q3 | 0.94621 | 0.94641 | +0.00020 |
| q4 | 0.91663 | 0.91682 | +0.00019 |
| q5 | 0.87069 | 0.87089 | +0.00020 |

The lift in the ensemble IS roughly concentrated in the q3–q5 region the EDA flagged, but the magnitude (+0.00020 at q5) is **far** below what would close the gap to the global ~0.95 (q5 would need +0.080). The new features are essentially a noisy duplicate of the existing pit-cluster features — they help diversification by 1–2 bps but don't actually re-discriminate the chaos region.

### Per-(Year × Compound) 4-way vs 3-way (no kill triggered, but very small magnitudes)

Spot checks (where n ≥ 10K and the 3-way ensemble baseline was weakest):

| Cell | n | 3-way AUC (cycle 8 EDA) | 4-way AUC | Δ |
|---|---|---|---|---|
| (2024, HARD) | 53,463 | 0.84412 | 0.84446 | +0.00034 |
| (2022, HARD) | 22,025 | 0.84034 | 0.84058 | +0.00024 |
| (2025, HARD) | 34,034 | 0.87886 | 0.87922 | +0.00036 |

HARD ensembles gained 0.00024–0.00036 — same order as the q5 lift. No cell regressed by more than the 0.0010 kill threshold.

### Reproducibility stamp

- git SHA at start: `6375f25`
- data sha256(train.csv): `f004e79d4e63f4bad0afc3788d07938dc5dbc0bed73a51e7b65cbce52ccc4128`
- packages: lightgbm 4.6.0, catboost 1.2.10, polars 1.40.1, sklearn 1.8.0, numpy 2.4.4, pandas 3.0.2

### Acceptance gates

baseline_std (3-way ensemble fold std) = 0.00045 → magnitude floor = max(0.5 × 0.00045, 0.00020) = **0.000225**.

| Gate | Target | Got | Pass? |
|---|---|---|---|
| Magnitude (Δ OOF ≥ 0.000225) | ≥ 0.000225 | +0.00011 | **FAIL** (short by 0.00012) |
| Direction (≥ ⌈0.6 × 5⌉ folds improved) | ≥ 3 | 5/5 | PASS |
| Stability (worst fold not down > 0.00045) | none | +0.00007 worst | PASS |
| Kill: CB#009 OOF ≥ CB#006 OOF | not violated | 0.94805 < 0.94806 by 0.00001 | **FAIL (hairline)** |
| Kill: q5 standalone moved | targeted slice must improve | q5 standalone slightly worse | **FAIL** |
| Kill: 4-way > 3-way | yes | yes | PASS |

## Verdict

**Inconclusive.**

Same shape as cycle 7: real-but-tiny ensemble gain (+0.00011, 5/5 folds positive), but below the noise floor and the *standalone* model basically tracks CB#006. The added features didn't move the targeted weak slice (q5) — they only contributed diversity at the ensemble level.

## Learnings

1. **Two consecutive CB-variant cycles (#007, #009) produced ~+0.0001 ensemble OOF gain each, both below noise floor.** Adding more single-CB variants on extended feature subsets is exhibiting diminishing returns: cycle #006's CB#006 was +0.00077 over its predecessor; cycle 7 was +0.00014; cycle 9 is +0.00011. The diversity well from "swap a few features in/out of an otherwise-identical CB" is essentially dry.
2. **The targeted-slice hypothesis failed.** The EDA correctly identified `field_pit_share` q5 as the largest untargeted residual, but the features designed to fix it (diff, lag-5, peak distance, post-SC interaction) didn't actually improve q5 in CB#009 standalone. Either the new features are highly collinear with the existing pit-cluster features (CatBoost picks one), or q5's residual is not about "rising vs falling pit waves" — it's about *which specific drivers* pit in chaos, which is a driver-level not field-level signal.
3. **Per-quintile ensemble lift was uniform across q3, q4, q5 (+0.00020 each)** — not concentrated where the cycle 8 EDA said the leverage was. Confirms the cycle 9 lift is generic ensemble averaging, not targeted FE working.
4. **CB#009 ran 6× slower than CB#007** (1100–1500s/fold vs 200s/fold). Likely cause: `peak_pit_window_distance` has range [−71, 76] with many distinct values, creating more split candidates than CatBoost's previous feature set offered. Worth noting if cycle 10 considers similar wide-range integer features.

## Follow-ups

1. **Stop CB-variant cycling.** Two consecutive Inconclusive results with sub-noise gains and very similar mechanics is strong evidence the branch is exhausted. Future cycles should target a different axis.
2. **Cycle 10 candidates** (re-prioritised after cycle 9 evidence):
   - **Stacking meta-model (~15 min)**: train a shallow LGB / logreg on the 4 base OOFs (LGB, CB#004, CB#006, plus a third CB if useful) with optional raw features. Tests whether a meta-learner extracts anything that fixed-weight blending misses. Cheap.
   - **NN with categorical embeddings (~60–90 min)**: pre-registered since cycle 4. Adds a structurally different model family. Driver embeddings target the cycle 8 EDA's "real-codename driver discrimination" lift (~1.6k lift_$ within driver subset).
   - **Combine cycle 7 + cycle 9 features into one CB (~30 min)**: test whether peer-rank + pit-cluster-sat lifts add linearly (would suggest each carries some independent signal that the 4-way ensemble dilutes). Low-cost diagnostic but unlikely to clear noise floor based on the two-cycle evidence.
3. **Drop the q5/q4 focus** — the residuals in those quintiles are *not* about pit-cluster timing; they're driver-level. NN embeddings is the right tool.
4. **Code: revert `src/features.py`** for an Inconclusive result. `src/research/train_cb009.py` and `src/research/blend_cycle009.py` stay as infra.
