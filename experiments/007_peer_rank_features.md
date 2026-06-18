# #007 — Peer-rank features for Position / LapTime / Cumulative_Degradation

**Status.** Inconclusive
**Date.** 2026-05-21
**Pre-registered as cycle #006 followup #1.**

## Hypothesis

Three new peer-rank features — `position_pct_among_compound_peers`, `laptime_pct_among_compound_peers`, `cum_deg_pct_among_compound_peers`, each computed as percentile rank within `(Race, Year, LapNumber, Compound)` and mirroring the cycle-#006 `tyre_age_pct_among_compound_peers` recipe — produce a CB#007 model whose 4-way fixed-weight ensemble (LGB=0.10, each CB=0.30) beats the cycle-#006 3-way ensemble OOF AUC by ≥ +0.00020 (the project min_delta).

## Rationale

1. **Cycle #006 pre-registered this exact direction** (006_hard_compound_v2.md §Observations/followups #1 and #4). The peer-rank pattern was the strongest single new feature in cycle #006: `tyre_age_pct_among_compound_peers` Q1→Q5 pit-next rate climbed 31.7% → 61.5% on HARD-non-2023, monotone. Generalising the recipe to other continuous cohort signals is the cleanest extension.
2. **The diversity-via-feature-subset pattern has worked twice** (CB#004 vs CB#006: ~0.94774 → 0.94806 standalone, but ensemble +0.00077). CB#007 adds a third feature variant that should produce different residuals from CB#004 (no peer-rank) and CB#006 (only TyreLife peer-rank), enabling further ensemble lift.
3. **Three signals chosen with non-overlapping intent**:
   - `position_pct_among_compound_peers` → "Am I leading my compound-cohort or am I a backmarker for this compound?" Captures pit pressure from strategy mates.
   - `laptime_pct_among_compound_peers` → "Am I pacing with my cohort or dropping off?" Cleaner than `driver_pace_ratio` because it ranks vs same-compound peers, not the whole field.
   - `cum_deg_pct_among_compound_peers` → "Are my tyres in better or worse shape than peers'?" The existing `Cumulative_Degradation` is signed and noisy; ranking normalises both.

## Expected magnitude

- **CB#007 standalone OOF AUC**: ≥ 0.94806 (CB#006) — flat or up.
- **4-way ensemble OOF AUC**: 0.94866 + 0.00020 to +0.00100 → target ≥ 0.94886.
- **Per-cohort sanity**: at least one of the 3 features should show a monotone Q1→Q5 lift on a target slice ≥ 3×.

## Overfitting risk

**Low.**

- All 3 features are deterministic transforms of existing columns (no labels touched).
- Computed on train+test combined, same as cycle #006's peer-rank (no temporal/label leakage).
- WET compound can have ≤1 driver in a (Race, Year, LapNumber, WET) group — those rows get rank 1.0 (deterministic). 99%+ of rows have ≥2 cohort peers.
- 71 features at CatBoost depth 8 + l2 3.0 — same regularisation regime as cycle #004/#006.
- No change to CV protocol, seed, or train.py / train_catboost.py.

## Kill criteria

If **any** of these fire, this direction is dead, not just this cycle:

- CB#007 standalone OOF AUC < CB#006 (0.94806) — the new features hurt the model that should be best positioned to use them.
- Any `(Year × Compound)` cell with n ≥ 10K regresses by > 0.0010 vs CB#006 standalone — peer-rank broke a slice we previously handled (cycle #003's failure mode).
- 4-way ensemble OOF AUC ≤ 3-way ensemble OOF AUC — no diversity gain over the existing 3-way; peer-rank features don't add information orthogonal to what CB#006 already extracts.
- Per-fold std > 0.00090 — feature additions destabilised the CV.

## Scope

- `src/features.py` — add 3 features in `add_compound_peer_features()` (~25 LOC). Strict addition; no drops, no signature change.
- `src/research/train_cb007.py` (new) — small wrapper around the cycle-#004 frozen CatBoost params, reading the 71-col feature parquet. Writes `data/oof_cb007.parquet` + `data/submission_cb007.csv`.
- `experiments/007_peer_rank_features.md` — this file, with results filled in.
- No changes to `src/train.py`, `src/research/train_catboost.py`, CV protocol, or seeds.

## Reversibility check

Touches CV? **No.** Touches seed? **No.** Touches target transform? **No.** Touches leakage surface? **No** — peer-rank within `(Race, Year, LapNumber, Compound)` uses no labels, same recipe as the already-shipped `tyre_age_pct_among_compound_peers`.

Safe to proceed.

## Plan

1. **Reproduce cycle-#006 baseline** via `src/research/repro_cycle006.py` — in progress.
2. Add 3 peer-rank feature definitions to `src/features.py`. Rebuild parquets to 71 cols.
3. Train CB#007 with cycle-#004 frozen params on the 71-col feature set. Save OOF + test predictions.
4. Compute per-(Year × Compound) AUC table for CB#007 vs CB#006.
5. Build 4-way ensemble at fixed weights (LGB=0.10, each CB=0.30). Compute OOF AUC. Compare to 3-way 0.94866 baseline.
6. Acceptance gates. Document. Commit.

## Result

### Baseline reproduction (sanity check that data + code line up with cycle #006)

`src/research/repro_cycle006.py` retrained LGB on 63 features, CB#004 on 63 features, CB#006 on 66 features. All three reproduced **bit-exact** to docs (5-decimal match on OOF AUC, per-fold AUC, and fold std).

| Component | Docs OOF | Repro OOF | Match? |
|---|---|---|---|
| LGB (63 features) | 0.94166 | 0.94166 | ✅ |
| CB#004 (63 features) | 0.94774 | 0.94774 | ✅ |
| CB#006 (66 features) | 0.94806 | 0.94806 | ✅ |
| 3-way ensemble | 0.94866 | 0.94866 | ✅ |

Per-fold ensemble: 0.94850, 0.94915, 0.94787, 0.94882, 0.94900 (std 0.00045). Identical to docs.

### CB#007 standalone (69 model features = 66 + 3 peer-rank)

```
OOF AUC:   0.94814   (vs CB#006 0.94806, Δ = +0.00008)
Per-fold:  0.94810, 0.94872, 0.94698, 0.94843, 0.94850
fold std:  0.00062   (vs CB#006 0.00057)
iters:     [2292, 2336, 1839, 1731, 1986]
```

Per-fold deltas vs CB#006: +0.00007, +0.00011, −0.00001, +0.00012, +0.00008. **4/5 positive**, worst fold essentially flat.

#### Per-year (CB#007 vs CB#006)

| Year | CB#006 | CB#007 | Δ |
|---|---|---|---|
| 2022 | 0.91122 | 0.91129 | +0.00007 |
| 2023 | 0.94033 | 0.94142 | +0.00109 |
| 2024 | 0.92479 | 0.92496 | +0.00017 |
| 2025 | 0.92495 | 0.92497 | +0.00002 |

2023 is where most of the standalone gain came from.

#### Per-(Year × Compound) — no cell with n ≥ 10K regressed > 0.0010

Spot-checks vs CB#006 (cycle-#006 doc's per-cell table):

| Cell | n | CB#006 | CB#007 | Δ |
|---|---|---|---|---|
| (2022, HARD) | 22,025 | 0.83840 | 0.83847 | +0.00007 |
| (2022, MEDIUM) | 45,546 | 0.92875 | 0.92910 | +0.00035 |
| (2023, HARD) | 60,996 | 0.94045 | 0.94253 | **+0.00208** |
| (2023, MEDIUM) | 58,264 | 0.94537 | 0.94524 | −0.00013 |
| (2023, SOFT) | 15,457 | 0.92582 | 0.92616 | +0.00034 |
| (2024, HARD) | 53,463 | 0.84257 | 0.84260 | +0.00003 |
| (2025, HARD) | 34,034 | 0.87697 | 0.87746 | +0.00049 |

(2023, HARD) recovered most of cycle #006's HARD regression (cycle #006 lost −0.00231 there vs CB#004; CB#007 gives +0.00208 back). No large-n cell regressed by more than 0.0013 in either direction.

### 4-way ensemble (LGB=0.10, CB#004=0.30, CB#006=0.30, CB#007=0.30 — fixed weights, no OOF tuning)

```
OOF AUC:   0.94880   (vs 3-way 0.94866, Δ = +0.00014)
Per-fold:  0.94869, 0.94932, 0.94795, 0.94896, 0.94913
fold std:  0.00048
```

Per-fold deltas vs 3-way ensemble: **+0.00019, +0.00017, +0.00008, +0.00014, +0.00013 — 5/5 positive, no regression**.

### Reproducibility stamp

- git SHA at start: `faf1914`
- data sha256(train.csv): `f004e79d4e63f4bad0afc3788d07938dc5dbc0bed73a51e7b65cbce52ccc4128`
- packages: lightgbm 4.6.0, catboost 1.2.10, polars 1.40.1, sklearn 1.8.0, numpy 2.4.4, pandas 3.0.2

### Acceptance gates

baseline_std = 3-way ensemble fold std = 0.00045 → magnitude floor = max(0.5 × 0.00045, 0.00020) = **0.000225**.

| Gate | Target | Got | Pass? |
|---|---|---|---|
| Magnitude (Δ OOF ≥ 0.000225) | ≥ 0.000225 | +0.00014 | **FAIL** (short by 0.000085) |
| Direction (≥ ⌈0.6 × 5⌉ = 3 folds improved) | ≥ 3 | 5/5 | PASS |
| Stability (worst fold not down > 0.00045) | none | +0.00008 worst | PASS |
| Generalization (train-val gap) | not measured for this cycle | n/a | n/a |
| Kill criteria (none fired) | none | none fired | PASS |

## Verdict

**Inconclusive.**

Direction is unambiguous: 5/5 ensemble folds positive, 4/5 CB#007 folds positive, 2023 HARD partially recovered. But ensemble OOF magnitude (+0.00014) is below the 0.000225 noise floor by ~38%. Could be a real but small effect, could be seed-specific noise — single-seed data can't distinguish.

The hypothesis is *not* refuted. It's underpowered at one seed.

## Learnings

1. **Peer-rank generalisation works as a feature mechanism but has diminishing returns.** Cycle #006 got +0.00077 ensemble from the first peer-rank feature (`tyre_age_pct`) + two pace-acceleration features. Cycle 7 added 3 more peer-rank features (Position / LapTime / Cumulative_Degradation) and got +0.00014 ensemble. The "easy" residual the peer-rank pattern targets — within-cohort rank within `(Race, Year, LapNumber, Compound)` — was substantially absorbed by the TyreLife version in cycle #006.
2. **(2023, HARD) was where CB#007's standalone gain landed (+0.00208).** This is the cell where cycle #006 had its largest regression vs CB#004 (−0.00231). The new peer-rank features partially undo that — `cum_deg_pct_among_compound_peers` is the most plausible driver since HARD-2023's pit decisions are degradation-driven in a near-zero-label regime.
3. **`tyre_age_pct_among_compound_peers` carries most of the peer-rank signal**; adding three more variants buys little marginal lift. Feature-importance check is in the follow-ups.
4. **Fixed-weight 4-way blend (LGB 0.10, each CB 0.30) is a safe default.** No OOF tuning, comparable to cycle #006's hand-tuned 0.10/0.40/0.50 (which had −0.00033 OOF→LB drift). Cycle 7's 5-of-5 fold-positive deltas suggest this is robust enough to ship even without further tuning.

## Follow-ups

1. **Inconclusive follow-up — seed robustness sweep.** Re-train LGB + CB#004 + CB#006 + CB#007 at seeds 7 and 99. Compute 4-way ensemble OOF at each. Verdict: **keep iff median 4-way ensemble OOF across seeds {7, 42, 99} clears 0.94866 + 0.000225 = 0.94888**. ~150 min CPU. Recommended before any LB submission attempt with these features.
2. **Feature-importance check for redundancy.** CatBoost feature importance on CB#007 — if `position_pct_among_compound_peers`, `laptime_pct_among_compound_peers`, `cum_deg_pct_among_compound_peers` rank below the existing `tyre_age_pct_among_compound_peers`, drop the weakest 1-2 and retest. May tighten the standalone CB#007 (less noise).
3. **Alternative cycle 8 candidates (independent of #1)**:
   - Peer-rank with a *different* grouping (`Race, Year, LapNumber` without `Compound`) — captures cross-compound positional dynamics.
   - Train a 5th model variant (XGBoost or LightGBM with different hyperparameters) for a 5-way ensemble — model-family diversity rather than feature diversity.
   - 2022 still the weakest year — targeted EDA cycle on 2022 errors.
4. **Code stays in tree.** For an Inconclusive result, revert `src/features.py` and rebuild parquets to 68-col. The 3 new feature definitions are recoverable from git history; `src/research/train_cb007.py` and `src/research/blend_cycle007.py` stay as reusable infra for any future 4-way blend cycle.
