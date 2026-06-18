# #005 — CatBoost hyperparameter tuning

**Status.** Killed early (inconclusive — see Result)
**Date.** 2026-05-08

## Hypothesis

Cycle #004's CatBoost used a single conservative-default config (`iterations=5000, learning_rate=0.05, depth=8, l2_leaf_reg=3`) and scored 0.94774 standalone. Tuning typically lifts a fresh CatBoost +0.001 to +0.003 — and that lift propagates *fully* into the ensemble (since CB is 85% of the blend). Target: push the ensemble from 0.94789 toward 0.95.

## Plan

### Stage 1 — Optuna search (fast, single-fold)

Use 1-fold validation as a fast proxy for full 5-fold. Search 30 trials with TPE sampler. Pruner kills trials underperforming at 500 iters.

Search space:
- `depth`: {6, 7, 8, 9, 10}
- `learning_rate`: log-uniform [0.02, 0.10]
- `l2_leaf_reg`: log-uniform [1, 30]
- `bagging_temperature`: [0, 2]
- `random_strength`: [0, 5]
- `border_count`: {64, 128, 254}
- `min_data_in_leaf`: {1, 5, 20, 50}

Per-trial: 3-fold CV with 2000 max iters, early stop 50. ~3 min per trial × 30 = ~90 min.

### Stage 2 — Full 5-fold validation of winner

Take the best Optuna config, re-run full 5-fold (5000 iters, early stop 100) for a comparable OOF AUC vs cycle #004.

### Stage 3 — Re-blend with LightGBM

Re-sweep ensemble weight w_LGB on the new tuned-CB OOF, find new optimum.

## Expected impact

- **Tuned CatBoost standalone**: +0.001 to +0.003 over 0.94774. Target ≥ 0.948.
- **Ensemble**: +0.001 to +0.003 over 0.94789. Target ≥ 0.949.

If it lands above 0.949 OOF, that's a clearly significant move (+0.0073 over baseline 0.94166).

## Overfit risk

**Moderate.** 30 Optuna trials on OOF is a multi-comparison effect — best-of-30 is biased high. Mitigation: validate winner on full 5-fold, not the 3-fold proxy used in search. If the 5-fold OOF underperforms the 3-fold by > 0.001, the win is overfit to the proxy.

## Validation gates

- [ ] Full 5-fold OOF (Stage 2) ≥ 0.94774 + 0.001 (i.e. ≥ 0.94874) for a clear "Kept" on tuning alone.
- [ ] Ensemble OOF ≥ 0.94789 + 0.001 (i.e. ≥ 0.94889) for a clear cycle-#005 "Kept".
- [ ] Per-fold std ≤ 0.0008.
- [ ] Stage 1 (proxy 3-fold) vs Stage 2 (5-fold) AUC gap ≤ 0.001 — if it widens, the tuning overfit the search proxy.

## Result

Killed after **8 of 20 trials** (~50 min wall) because no clear winner was emerging.

```
trial 0: AUC=0.94668  depth=7,  lr=0.092, l2=12.1   (converged at ~1380 iters)
trial 1: AUC=0.94663  depth=10, lr=0.028, l2=1.86  (capped at 1500 iters)
trial 2: AUC=0.94670  depth=8,  lr=0.071, l2=1.97  (best so far) ← cycle-#004-equivalent
trial 3: AUC=0.94436  depth=7,  lr=0.023, l2=10.2  (capped, lr too low)
trial 4: AUC=0.94508  depth=8,  lr=0.027, l2=27.0  (capped, lr too low)
trial 5: AUC=0.94516  depth=7,  lr=0.031, l2=16.8  (capped, lr too low)
trial 6: AUC=0.94624  depth=6,  lr=0.074, l2=11.1  (capped)
trial 7: AUC=0.94520  depth=7,  lr=0.034, l2=12.0  (capped, lr too low)
```

3-fold proxy AUC range: 0.944-0.947. Cycle-#004's `(depth=8, lr=0.05, l2=3, iters=5000)` would land at ~0.946-0.947 on 3-fold, so **trial 2 (the best so far) is essentially baseline territory** — not above it.

### Why the search stalled

1. **`SEARCH_ITERS=1500` cap was too low.** Five of the eight trials (1, 3, 4, 5, 7) hit the cap without converging — anything with `learning_rate < 0.05` simply ran out of trees. The proxy AUC for those trials is biased low. TPE will steer away from low-lr configs as a result, but it's also throwing away the entire "more trees, lower lr" region of the search space, which is *the* region where modest CatBoost gains usually live.
2. **3-fold proxy is ~0.001 lower than 5-fold validation.** With trial 2 at 0.94670 (3-fold proxy) and cycle-#004 at 0.94774 (5-fold), there's no signal yet that any tuned config genuinely beats baseline.
3. **No trial above the noise floor.** The trial spread (0.944-0.947) is wider than the gain we're hunting for (~+0.002), so we'd need many more trials before TPE could distinguish "real winner" from "lucky split."

### Why we killed early instead of finishing

Continuing the remaining 12 trials would burn ~60 min for a search that, by trial 8, hasn't shown evidence it'll find a real winner. Cycle #6 has a higher expected payoff (load-bearing residual at AUC 0.84), and the parallel-run option costs roughly twice as much wall time as serial.

## Decision

**Killed early — inconclusive.** Re-run only after raising `SEARCH_ITERS` to 3000+ AND reducing `N_TRIALS` to 10-15 to keep the same wall-time budget. Or better: hand-pick 3-4 promising configs (e.g. lr=0.03 with iters=8000, lr=0.10 with iters=2500, depth=10 with full iters) and full 5-fold validate each; this avoids TPE's bias toward fast-converging configs entirely.

For the immediate cycle, we move on. Cycle-#004 frozen params remain the CatBoost reference.

## Observations / followups

1. **Defer HP tuning until features are exhausted.** Tuning gives modest gains; feature engineering on this CatBoost can still find +0.001-0.005 (cycle #6 in flight). Tuning should be the *last* +0.001-0.002 squeeze, not the first.
2. **The iter cap is the trap.** Future Optuna sweeps for CatBoost should use `iters >= 3000, early_stop=100` even if it means fewer trials.
3. **Trial 2's params (depth=8, lr=0.071, l2=1.97) are worth a one-shot full-5-fold validate** as a low-cost confirmation that nothing was hiding there. Defer.
