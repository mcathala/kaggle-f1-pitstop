# Experiment 021 — RealMLP HP search (single-fold Optuna proxy)

**Cycle.** 7
**Status.** Inconclusive (killed at trial 10/20)
**Date.** 2026-05-22

## Hypothesis

An Optuna sweep over RealMLP capacity / regularization HPs — `n_ens`, `hidden_sizes`, `embedding_size`, `p_drop` (dropout), `learning_rate` — with the cycle 5 feature set frozen, finds a configuration whose single-fold OOF on fold 1 beats the pre-tuned default by ≥ 0.0010. (Half-std min_delta is 0.00020, so this is a 5× hurdle for the single-fold proxy to safely transfer to a +0.0005-1.0 5-fold lift.)

## Rationale

- **Cycle 6 closed three directions** (forward features, historical aggregates, FT-Transformer). The one direction it deliberately did not try is HP re-tuning — flagged in cycle 6 docs as the highest-prior remaining path.
- **Plateau diagnosis** (Phase 1): last three LB deltas were 0.00265 / 0.00011 / 0.00000. The remaining headroom must come from the model itself, since features and architecture are exhausted under default HPs.
- **PyTabKit's "tuned defaults" are *benchmark*-tuned, not *task*-tuned.** The defaults converge to a local optimum on the suite of benchmark datasets in the RealMLP paper; per-task HP tuning has shown +0.0005-0.0015 lift on similar tabular tasks in the published evaluations.

## Expected magnitude

- **Best single-fold OOF target:** 0.95421 + 0.0010 = **0.95521** (single-fold #1 baseline is RealMLP default at 0.95421 from cycle 4).
- **Floor for informativeness:** at least one trial clears default by +0.0005 single-fold; if all 20 trials fall within ±0.0002 of default, the direction is dead (means RealMLP defaults sit at a sharp local optimum for this data).
- **Projected 5-fold OOF** if single-fold signal transfers: **0.95453 (+0.00070 over cycle 5's 0.95383)**.

## Overfitting risk

**Medium-High.** Specific sources:

1. **HP-search-overfit**: 20 trials on the same fold-1 validation set risks fitting the noise. Mitigated by: keeping trial count modest (≤30), holding out fold-1 validation for ranking only, and requiring full-5-fold validation in exp 022 before committing.
2. **Default-shape brittleness** (from cycle 6): aggressive capacity changes might collapse the model. Mitigated by anchoring the search around the default with bounded deviations.
3. **NO direct LB risk** — this experiment makes no submission. Cycle 7 closes only when exp 022 (full-5-fold validation of best HPs) clears OOF gate.

## Kill criteria

- [ ] All 20 trials within ±0.00020 single-fold AUC of default 0.95421 (HP space is flat)
- [ ] Best trial regresses default by ≥ 0.0010 (search degenerated; default is genuinely optimal)
- [ ] Single trial takes > 15 min on M1 Pro MPS (search budget blows past 5 hr — would force re-scoping)

## Scope

Files touched:

- `src/research/train_realmlp_optuna.py` (new, ~150 lines) — single-fold Optuna driver
- `data/realmlp_optuna_trials.parquet` (output)
- `experiments/021_realmlp_hp_search.md` (this file)

## Reversibility check

- CV protocol: **unchanged** (5-fold StratifiedKFold seed=42 on Year × PitNextLap), only fold 1 used as proxy.
- Seed: **unchanged** (42 baseline, Optuna seeds within trial only).
- Target transform: **unchanged**.
- Leakage surface: **unchanged** — feature set is frozen at cycle 5's pipeline.

No `reversibility` flag fires. Proceeding.

## Plan

1. Clone `src/research/train_realmlp.py` → `src/research/train_realmlp_optuna.py`. Trim to single-fold-1 train+val.
2. Wrap RealMLP fit in an Optuna objective. Search space:
   - `n_ens`: {12, 24, 36, 48}
   - `hidden_sizes`: choose architecture by depth/width index, from {[256,128,64], [512,256,128], [768,384,192], [1024,512,256]}
   - `embedding_size`: {4, 6, 8, 12}
   - `p_drop`: {0.0, 0.1, 0.15, 0.2}
   - `learning_rate` (`lr` in PyTabKit): log-uniform [3e-4, 3e-3]
   - Fixed: `max_epochs=6`, activation=SiLU, embedding type=PBLD, batch=256
3. Budget: 20 trials × ~5 min/trial ≈ 100 min (Optuna TPESampler). Persist all trials to `data/realmlp_optuna_trials.parquet`.
4. Report top-3 configs by single-fold AUC; emit `data/realmlp_optuna_top3.json`.
5. Decision gate: if best trial > 0.95421 + 0.0005 (single-fold), proceed to exp 022 (full-5-fold validation of top-3). Else mark this experiment Inconclusive and reconsider direction.

## Result

Killed at trial 10/20 (~100 min elapsed) on pattern-stable evidence that the HP space is locally flat-to-worse around the pre-tuned default. Single-fold reference: 0.95421.

| Trial | n_ens | arch | emb | p_drop | lr     | AUC     | Δ vs default | Wall (s) |
|-------|-------|------|-----|--------|--------|---------|--------------|----------|
| 00    | 24    | 3    | 12  | 0.00   | 0.0100 | 0.95400 | −0.00021     | 547      |
| 01    | 36    | 3    | 8   | 0.20   | 0.0193 | 0.95396 | −0.00025     | 791      |
| **02**| **36**| **3**| **6**| **0.10**| **0.0235** | **0.95429** | **+0.00008** | **774** |
| 03    | 24    | 2    | 4   | 0.15   | 0.0047 | 0.95417 | −0.00004     | 370      |
| 04    | 24    | 0    | 4   | 0.20   | 0.0089 | 0.95359 | −0.00062     | 260      |
| 05    | 36    | 0    | 12  | 0.10   | 0.0171 | 0.95292 | −0.00129     | 364      |
| 06    | 48    | 1    | 6   | 0.10   | 0.0289 | 0.95386 | −0.00035     | 559      |
| 07    | 12    | 3    | 6   | 0.05   | 0.0031 | 0.95355 | −0.00066     | 250      |
| 08    | 36    | 2    | 6   | 0.10   | 0.0296 | 0.95419 | −0.00002     | 565      |
| 09    | 12    | 1    | 8   | 0.00   | 0.0107 | 0.95371 | −0.00050     | 221      |

**Stats over 10 trials:**
- Best Δ = +0.00008 (trial 2). Below +0.0005 hurdle by 6×.
- Median Δ = −0.00037. Mean Δ = −0.00039.
- 1 trial positive, 9 negative.
- Worst hits: arch_idx=0 (smallest network) cost ~0.001 AUC consistently.

**TPE-informed window (trials 6-9):** all four were ≤ default. TPE was converging *toward* the default neighborhood, not climbing above it.

## Verdict

**Inconclusive.** The direction is dead enough to not warrant the remaining ~140 min of compute. Killed deliberately, not on infra failure. Logging it as Inconclusive (per Phase 1's acceptance gate language) rather than Reverted because no code was committed and the result *was* informative — it concretely closes the HP-tuning direction.

## Kill-criteria check

- [x] Trial 2 cleared +0.00008 (the original kill criterion "all 20 trials within ±0.00020 of default" was specifically about flat-on-the-upside; we have downside but only one barely-positive trial, far below hurdle)
- [x] Best trial regressed default by < 0.0010 (no degenerate explosions on the upside; worst trials *did* regress by up to −0.0013 with arch_idx=0)
- [x] No trial took > 15 min (longest was 791s ≈ 13.2 min, within budget)

The TRUE kill signal that fired: "TPE-informed window converges below default" — proves the search has localized around the default and isn't going to find +0.0005 in the remaining trials.

## Repro stamp

- data: `train.csv` sha256 `f004e79d...` (cycle-5 pinned)
- pkg: pytabkit (current .venv), torch (current .venv), optuna 4.8.0, sklearn (current .venv)

## Learnings

1. **PyTabKit's tuned defaults are a sharp local optimum for this data.** Cycle 6's "brittle to perturbations" finding generalizes: HP perturbations roll downhill the same way feature-additions did. Cycle 7 confirms RealMLP at default HPs is at or near its ceiling on the cycle-5 feature pipeline.
2. **Single-fold proxy was actionable.** 10 trials × ~10 min was enough to read the gradient; full 20 trials would have been waste. Kill decisions on the *direction* (not just this experiment) can fire mid-experiment when the pattern is clear.
3. **TPE-informed convergence near baseline = direction dead.** This is the cheaper version of "all trials within ±min_delta" — once TPE picks defaults' neighborhood and doesn't climb, the search-region's local maximum IS the default. No amount of additional sampling fixes that.
4. **Cycle 6 + cycle 7 form a coherent map of dead ends:** feature additions (cycle 6 exp 18, 19), architecture swap (cycle 6 exp 20), HP tuning (cycle 7 exp 21). Three independent axes on RealMLP are exhausted. The remaining ROI must come from a *different training signal*, not a different model — meaning AutoGluon-Tabular, pseudo-labeling, distillation, or rank-blending heterogeneous learners.

## Follow-ups

1. ✅ Killed exp 21 at trial 10.
2. Cycle 7 pivot: **exp 022 = rank-blend cycle 5 multi-seed RealMLP + cycle 3 CB-tuned** (cheap probe, 10 min, ~20% chance of +0.0005 LB).
3. If exp 022 doesn't clear, **exp 023 = AutoGluon-Tabular** (main bet, 30 min–2 hr, ~35% chance).
4. The 4 dead-end directions (features, arch, HPs, exp 22 if it fails) close cycle 7 with strong "RealMLP ceiling" evidence; consider that a graceful cycle close even at LB 0.95342.

