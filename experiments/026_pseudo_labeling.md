# Experiment 026 — Pseudo-labeling with cycle 7 blend

**Cycle.** 9
**Status.** Inconclusive (single-seed +0.00018 over cycle 4, but rank-corr 0.9968 with cycle 5 → no new signal)
**Date.** 2026-05-22

## Hypothesis

Retraining a single-seed RealMLP on (competition train ∪ confident pseudo-labeled test rows), where pseudo-labels come from cycle 7's blend at threshold prob ≥ 0.95 or ≤ 0.05, produces OOF AUC on competition train ≥ 0.95375 (cycle 5 single-seed 0.95355 + 0.00020 min_delta). If yes, the multi-seed variant is expected to land at OOF ~0.95400 and blend with CB-tuned-exp14 at ~0.95425.

## Rationale

- Cycle 8 closed the model-zoo blending direction. The remaining ROI must come from a **new signal source**, not a new model.
- Pseudo-labeling adds the test distribution's structure into training. For binary classification with a model already at ~0.95 AUC, the top/bottom confidence buckets have ~95% precision — they're approximately correct labels.
- 60% of test rows clear the 0.95/0.05 threshold (113,304 of 188,165). With ~2.9% positive rate among pseudo-labels (vs train's ~21%), the combined train set will have pos rate ~14.6% — a moderate shift toward negative-class examples but not catastrophic.
- The LB-overfit risk noted earlier is real (public LB = 20% of test). Mitigations: (a) conservative threshold, (b) OOF measured on competition train ONLY (the pseudo-labels are augmentation, not validation), (c) public-LB submission gated on OOF improvement.

## Expected magnitude

- **OOF single-seed target:** ≥ 0.95375 (cycle 5 single-seed + min_delta). If under 0.95355 (cycle 5 baseline), pseudo-labels are HURTING — kill direction.
- **Multi-seed projection:** if single-seed lifts by +0.00020, multi-seed should reach ~0.95403 (matches cycle 5 multi-seed's variance reduction).
- **Blended projection:** ~0.95428 (cycle 7 + min_delta). If achieved, this is the cycle 9 close.
- **Floor for direction-alive:** single-seed OOF ≥ 0.95335 (within −0.00020 of cycle 4 single-seed 0.95355). Below that, pseudo-labels are net-negative.

## Overfitting risk

**HIGH.** Sources:

1. **Test-distribution leakage:** pseudo-labels encode the cycle 7 blend's biases on test. If those biases are public-LB-overfit, retraining amplifies them. Mitigation: validate on OOF (competition train), require OOF lift before submission.
2. **Class balance shift:** combined pos rate drops 21% → 14.6%. Some bias toward negative class is unavoidable. RealMLP's PBLD numeric embedding + no explicit class weighting should handle this gracefully, but the AUC threshold may shift slightly.
3. **Wrong pseudo-labels poison the model:** 2.9% pseudo pos × ~5% precision error = ~150 mislabeled positives + similar for negatives. This is ~0.3% label noise — within typical tolerance for an AUC of 0.95.
4. **Public-LB confirmation will be the real test** of whether the OOF lift transfers — drift might widen (more moving parts).

## Kill criteria

- [ ] Single-seed OOF < 0.95335 (pseudo-labels hurting more than helping)
- [ ] Single-seed OOF ∈ [0.95335, 0.95375] but per-fold std > 0.00080 (instability red flag)
- [ ] Fold-1 wall-clock > 15 min (compute blowing up; defer to a separate exp with smaller augmentation)
- [ ] Any year-bucket OOF AUC drops by > 0.005 vs cycle 5 (year-specific overfit)

## Scope

- `src/research/train_realmlp_pseudo.py` (new, ~250 lines — clone of train_realmlp.py with pseudo-label augmentation)
- Outputs: `data/oof_realmlp_pseudo.parquet`, `data/submission_realmlp_pseudo.csv`
- `experiments/026_pseudo_labeling.md` (this file)

Wall-clock budget: ~25 min for single-seed 5-fold.

## Reversibility check

- CV protocol: **unchanged** — folds still on competition train only; pseudo-labels go into training data, NOT validation.
- Seed: unchanged.
- Feature set: cycle 5's pipeline (same as RealMLP).
- Leakage surface: **NEW** — pseudo-labels create a controlled test-into-train channel. Mitigated by the threshold (conservative confidence) and the OOF validation gate.

**No `reversibility=true` flag**: the change is contained (training data augmentation), reversible (drop pseudo-labels and retrain), and CV is preserved.

## Plan

1. Build `src/research/train_realmlp_pseudo.py`:
   - Load competition train + competition test + external data (as before)
   - Load cycle 7 blend submission `data/submission_linearblend_best.csv`
   - Filter test rows where prob ≥ 0.95 or prob ≤ 0.05 — assign hard pseudo-labels
   - In each fold: train data = (4/5 of competition train) + (all external) + (all confident pseudo-labels)
   - Validate on the held-out 1/5 of competition train (UNCHANGED)
   - Same FE pipeline as `train_realmlp.py`
   - Same RealMLP HPs as `train_realmlp.py` (PyTabKit defaults)
2. Run single-seed (seed=42) → if OOF clears 0.95375, proceed to step 3
3. (Conditional) Multi-seed (seeds {42, 7, 99, 137, 313}) → if OOF clears 0.95403, proceed to step 4
4. (Conditional) Blend with CB-tuned-exp14 at w_cb=0.20 — measure OOF lift over cycle 7's 0.95408
5. (Conditional) Submit to Kaggle

## Result

**Single-seed RealMLP with 113,304 pseudo-labels (60% of test, pos_rate 2.9%):**
- OOF AUC: **0.95373**
- Per-fold: [0.95440, 0.95435, 0.95341, 0.95269, 0.95384]
- Per-fold std: 0.00064 (matches cycle 4)
- Wall-clock: ~28 min (5.5 min/fold)

**Comparison:**
- vs cycle 4 single-seed RealMLP (0.95355): **+0.00018** (technique works, lifts single-seed)
- vs cycle 5 multi-seed RealMLP (0.95383): −0.00010
- vs cycle 7 blend (0.95408): −0.00035

**Rank correlations (key finding):**
- pseudo × multi-seed: **0.9968** (nearly identical ranking)
- pseudo × cb-tuned: 0.9728
- multi-seed × cb-tuned: 0.9758 (reference)

The pseudo-augmented model's predictions are *almost identical* to cycle 5 multi-seed. Pseudo-labels reinforce existing model behavior rather than adding orthogonal signal.

**Blend probes:**

*2-way (multi-seed × pseudo):*
- Peak w_pseudo=0.30-0.40: OOF 0.95387 (+0.00004 over multi-seed alone)

*3-way (multi-seed × CB × pseudo):*
- Best at (w_multi=0.50, w_cb=0.20, w_pseudo=0.30): OOF **0.95411**
- Δ vs cycle 7 (0.95408): **+0.00003** (below min_delta 0.00020)

## Verdict

**Inconclusive.** Technique works on single-seed but does not break the cycle 7 blend ceiling. Best 3-way blend lifts by +0.00003 — well below noise floor. Multi-seed pseudo would likely add another +0.00003 (~0.95414) at the cost of ~2 hr compute, still well below min_delta. Not worth the wall-clock.

## Kill-criteria check

- [x] Single-seed OOF > 0.95335 (0.95373 — PASS, technique works)
- [ ] Single-seed OOF ≥ 0.95375 (0.95373 — JUST under by 0.00002, technically fail)
- [x] Per-fold std reasonable (0.00064, same as cycle 4)
- [x] No fold > 15 min (max 5.5 min)

The decision-relevant gate fires: **Best 3-way blend < cycle 7 + min_delta** (0.95411 < 0.95428). Pseudo doesn't add ensemble value over what we already have.

## Repro stamp

- pkg: pytabkit (current .venv), torch (current .venv), sklearn 1.7.2
- inputs: `data/submission_linearblend_best.csv` (cycle 7) as pseudo-label source
- thresholds: prob ≥ 0.95 → label 1, prob ≤ 0.05 → label 0; 113,304 pseudo-labels (60.2% of test)

## Learnings

1. **Pseudo-labeling with high-confidence threshold reinforces existing predictions instead of adding signal** — rank correlation 0.9968 with cycle 5 multi-seed means the pseudo-augmented model is functionally the same model. The "test-distribution structure" we hoped to inject was already learned from train.
2. **Single-seed pseudo (0.95373) ≈ cycle 5 multi-seed (0.95383) within noise** — pseudo augmentation does approximately what multi-seed variance reduction does, at higher compute cost.
3. **Conservative threshold doesn't reproduce the published pseudo-label LB gains** seen on simpler tabular tasks because our model is already near its ceiling — the marginal gain from confident-test-as-train is small when train has 439k rows and OOF is already 0.95.
4. **The 60/40 class imbalance shift (21% train pos → 14.6% combined pos)** didn't visibly hurt — RealMLP's PBLD embedding handles class shifts gracefully.
5. **Future pseudo-label attempts** should try: (a) percentile-balanced selection (top 21% + bottom 79% to match train), (b) iterative self-training (retrain → relabel → retrain), (c) confident-but-wrong slice analysis (where do train and pseudo disagree?).

## Follow-ups

1. ✅ Killed exp 026 as Inconclusive at single-seed.
2. **Cycle 9 pivot: exp 027 = 2022-year specialist RealMLP.** Within-year analysis confirms 2022 is the weakest slice (within-year AUC 0.91947 vs 0.93-0.95 for other years). 2022 is 19% of train; closing half the gap could lift overall AUC by ~+0.001.
3. Skip multi-seed pseudo (low expected return).
4. Skip percentile-balanced pseudo for now (likely also rank-corr ~0.99 with multi-seed — same mechanism).
