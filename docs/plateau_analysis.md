# Plateau analysis — the integrity ceiling (cycle 19)

**Date.** 2026-05-29 (deadline 2026-05-31)
**Scope.** Fresh-eyes review at cycle 19: where the score plateaued, why, and how much clean headroom remained.
**Current best LB.** **0.95401** (100% own-pipeline, ~top 15%)
**Best offline blend.** greedy ensemble **OOF 0.95479** (nested-CV held-out 0.95474) — *not yet LB-confirmed*
**Compute.** M1 Pro 16GB default; limited Kaggle GPU quota.
**Ground rules.** No TabPFN. No external submission CSVs as inputs. No distillation from external models. No author/notebook attribution.

---

## 0. One-paragraph verdict

We are not stuck because we are doing something wrong. We are stuck because **the offline metric is genuinely saturated (~0.9548 OOF) and the public-LB cluster above us is mostly a public-CSV-blend artifact we have deliberately ruled out.** Two things in the record are *stale and worth acting on immediately*: (1) the "0.95462 OOF ceiling" was **already broken** by greedy ensemble selection (0.95479) and never LB-tested; (2) the TabPFN "the one clean gap" line is now closed by a project decision, so the plan must stop treating it as latent upside. After accounting for both, the honest reachable target before the deadline is **LB ≈ 0.9541 (top ~15%) for free**, with **top-10% (0.95449) requiring +0.0004 OOF that no surviving clean lever can plausibly produce.** The right move is to *bank the free gain and pick final submissions well*, not to keep manufacturing closed-lever runs.

---

## 1. Submitted vs offline best

Best *submitted* LB is **0.95401**; the best *offline* blend is a greedy ensemble at **OOF 0.95479** (nested-CV held-out 0.95474), built as `submission_blend_greedy_{full,nosd}.csv` but not yet sent to Kaggle. Closing that submitted-vs-offline gap is the single most actionable item below.

---

## 2. The transfer function — the math that decides what is reachable

From our own 16 OOF↔LB pairs (`docs/transfer_analysis.md`):

> **LB ≈ 0.927 · OOF + 0.069**, residual std **0.00018**

This is tight enough to plan against. Inverting it for each LB target:

| LB target | Percentile | Required OOF | Gap to greedy (0.95479) |
|---|---|---|---|
| 0.95401 (current submitted) | ~top 15% | 0.95460 | — (already passed offline) |
| **0.95409** | top-15% line | **0.95479** | **0.0000 — greedy lands here** |
| 0.95449 | top-10% | 0.95523 | **+0.00044 OOF** |
| 0.95453 | top-7% (the target) | 0.95527 | **+0.00048 OOF** |
| 0.95489 | top-1% / leaders | 0.95566 | +0.00087 OOF |

**Read this carefully.** The greedy blend we already built maps to **LB ≈ 0.9541** — a free +0.0001 over current best, landing right at the top-15% line. To reach **top-10% we need +0.00044 OOF beyond a blend that seven independent analyses (074, 061, 010, 088, 089, 091, 092 + Nelder–Mead) say is saturated.** That is the core of the plateau: not a weight-tuning gap, an *information* gap.

Slope 0.927 < 1 also means **gains shrink in transfer** (every +0.001 OOF → +0.00093 LB), and the gap *widens* at the high end. So OOF chasing has declining real-world payoff exactly where we are.

One non-obvious lever hides in the residuals: **balanced/mixed-family blends transfer with a +0.00004 residual; GBDT-heavy ones transfer at −0.00021.** That's a ~0.0002 LB swing *at equal OOF*, purely from composition. It is free and we are not currently exploiting it in final-submission selection.

---

## 3. Layer-by-layer audit (questioning every layer)

### 3.1 Data — clean, fully understood, no shortcut left
- **~32% instance-dependent label noise** is real and generator-intrinsic (only ~30% of positives align with a stint boundary). This is *why the whole leaderboard is jammed into a 0.0004-wide band* — it caps everyone. We cannot out-feature a noise ceiling.
- **No train/test shift** (adversarial AUC ≈ 0.50). There is no distribution edge to exploit, and no transductive leak (confirmed twice: `dataset_review_no_leak.md`, exp 090).
- **The real residual is driver-level discrimination** (weak drivers 0.85–0.90 AUC vs 0.95 global; `field_pit_share` q5 = 0.876, barely moved in 19 cycles). Our own EDA calls it "non-tree-friendly — the per-driver behaviour signature." This axis wants a *different inductive bias*. The two model classes that supply it — an in-context foundation model (TabPFN) and competitor-CSV blends — are both off the table. **So this residual is, for us, structurally unreachable.** This is the honest crux.

### 3.2 CV / label honesty — sound, one residual risk
- Fold assignment is consistent across all 75 bases (seed 42, Year×PitNextLap, concatenated OOF). Blending is structurally valid.
- **Pseudo-label leakage** (round-1 XGB pseudo reused across folds, ~+0.0002 OOF inflation) — already de-leaked via the `oof_xgb_pseudo→pseudo2` swap. The transfer doc shows pseudo-drift is **within the 0.00018 noise floor** after controlling for OOF, so this is no longer a live concern; the staged best/nopseudoGBDT/pure A/B is now confirmatory, not corrective.
- **Blend weights are fit in-sample** on the same 439k OOF they report. Nested CV says the greedy gain generalizes (0.95479 in-sample vs 0.95474 held-out — +0.00005 optimism, trustworthy). But every new base adds a knob; *this is the one bias that compounds as the zoo grows to 75.* Discipline: only add a base if it survives the nested check.

### 3.3 Models — the zoo is exhausted, with two caveats
- The "from-scratch NN caps at ~0.94" closure is rigorous and confirmed across lap-attention (058–060), embMLP (053/054), FTT (020): genuinely-diverse architectures top out at 0.936–0.943 — **too weak to earn blend weight.**
- The "strong+diverse quadrant is empty" closure is also rigorous: the most decorrelated *strong* base ever built (GCE-XGB, ρ0.959 @ 0.952, exp 088) earned w=0.015 / +0.00000. The hurdle is **ρ≤0.92 at AUC≥0.95, and nothing own-tooled reaches it.**
- **Caveat 1 (ruled out):** TabPFN was the one untried class that is prior-fitted rather than from-scratch, so the 0.94 cap need not apply. The local package is installed and runnable but gated behind a one-time license click (it failed the smoke test on exactly that, non-interactively). It was not pursued — out of scope under the project's ground rules.
- **Caveat 2 (low-EV):** a `rank:pairwise` XGB — a genuinely different *loss surface* (optimizes ordering directly, not logloss) — was the only remaining shot at a decorrelated strong base. Expected value was low: exp 088 showed even ρ0.959@0.952 adds ~0, so unless rank landed ρ<0.95 it would change nothing.

### 3.4 Feature engineering — closed, correctly
- diffFE (stripping over-engineered cross-cats) was the last real win (+0.00013 LB). It is now saturated: tyre-FE neutral (089), diffFE-RM neutral (082), view-3 reverted (cosmetic). A residual-correlation scan (commit `f82ee23`) found **errors uncorrelated with every feature family** — the definitive FE closure. There is no FE pointed at the driver-level residual that we haven't tried *and* that helps; the ones that target it (forward features 018, history aggregates 019) regress because RealMLP's tuned recipe is brittle to feature-set changes.

### 3.5 Blend / combiner — saturated, but selection still has free value
- Global operators (rank/logit/gmean/remap) within +0.00001 of linear (074). Per-slice/quantile/per-year (036/037, fc2a84f), pred-only & context meta-stacking (061/092) all *worse* than linear. The linear blend is optimal **— except greedy selection beats hand-tuned coord-descent by +0.00017** because it explores the weight simplex more freely (it found value in self-distill as a *component* even though self-distill is the worst standalone transferrer). That's the one combiner result still paying out, and it's banked but untested on LB.

### 3.6 Drift — structural, not fixable, but selectable-around
- −0.0006 OOF→LB, shown to be the regression shrinkage (slope 0.927), **not** a pseudo penalty and **not** external-data reweighting (exp 079). Strong, honest bases transfer ~1:1; only OOF-overfit bases (self-distill, −0.00042 residual) pay extra. **Actionable consequence:** prefer balanced-family compositions for final picks (+0.0002 transfer edge), and never submit the self-distill-heavy greedy variant as the primary.

---

## 4. Clean external recon — what the field is converging on (abstract)

Reconfirmed from public material (no names, no notebooks, no CSVs ingested):

1. **The model stack the field uses is the stack we built.** Gradient-boosted trees (XGB/LGB/CB/HGBC) + a tuned tabular-NN (the RealMLP/PyTabKit recipe) + the same public external strategy dataset + cross-categorical n-grams + KBins bucketing. Stratified (not grouped) CV despite the transductive-looking split. **We independently arrived at the field's consensus recipe — there is no architecture or feature the consensus has that we lack.**
2. **The visible top cluster (≈0.9545–0.95493) is overwhelmingly post-hoc public-submission blending**, not better models: load a curated pool of other people's submission CSVs (named by their LB scores), rank-remap onto the best probability anchor, gate to the ambiguous 0.05–0.60 band. This is exactly the method our ground rules forbid. **The cluster's edge is breadth of independent inputs, not modeling skill** — and they get that breadth by ingesting each other's outputs.
3. **The one clean modeling axis the field has and we don't is the in-context foundation model (TabPFN).** That is now closed by a project decision, not by capability.
4. **Minor unused-by-us hygiene:** extreme-outlier winsorization (LapTime/Delta/Degradation > 500 → median; ~0.03% of rows). Matters for NN StandardScaler, negligible for trees. Low EV; never tested cleanly. Cheapest "did we miss anything mechanical" check.

**Comparison to us:** we match the field on models, FE, CV, and external data. We are *behind* only on the two things we've chosen not to do (TabPFN, CSV-blending). **Our 0.95401 is competitive among honest, self-trained solutions** — the gap to the headline numbers is a methodology boundary, not a skill deficit.

---

## 5. The reachable target

The standing target is **top-7% = LB 0.95453**. Per §2 that needs **OOF ≈ 0.9553** — **+0.0005 beyond a repeatedly-confirmed saturated offline ceiling**, reachable only via a strong+diverse base that the entire cycle 16–18 program showed could not be built with own tooling, *or* via the two off-limits methods. Top-7% was therefore not reachable by clean means within the remaining time; the honest position was to say so rather than manufacture motion.

What *is* reachable, cleanly, with near-certainty:
- **LB ≈ 0.9541 (top ~15%) for free** by submitting the already-built greedy blend.
- A *chance* at a few more ten-thousandths if the rank-objective base lands unusually decorrelated, or if a more careful selection over the 75-base zoo + a balanced-transfer-optimized composition squeezes another +0.0001–0.0002.

That is the honest envelope: **0.95401 → ~0.9541 banked, with a long tail toward ~0.9542.**

---

## 6. Where the remaining clean headroom was

Ranked by expected value, the clean moves left were modest:

1. **Bank the already-built greedy blend.** Submitting `submission_blend_greedy_{full,nosd}.csv` maps to LB ≈ 0.9541 (+0.0001) — a free gain sitting unbanked. `nosd` is the transfer hedge (drops the worst transferrer); `full` tests whether self-distill's OOF value survives.
2. **Optimize final-submission selection for transfer**, not raw OOF — a balanced-family composition (down-weighting the RealMLP/GBDT dominance toward an even family mix at matched OOF) harvests the ~+0.0002 transfer residual from §2, built offline from existing OOFs with no training.
3. **Two cheap genuinely-new probes**, nested-CV gated before touching the blend: an **ExtraTrees** base (~20 min CPU, never tried — a different variance profile from boosted trees) and a **richer greedy/bagged ensemble selection** over the full base zoo (the hand-tuned → greedy jump from 0.95462 → 0.95479 showed the selection layer wasn't fully mined). A one-line winsorization hygiene check on the NN inputs closes the last "did we miss anything mechanical" question.

Everything else — new from-scratch NNs, GBDT-library swaps, HP-diversity views, robust-loss bases, FE-views, per-slice/quantile/per-year weighting, global blend operators, context/pred meta-stacking — was independently closed across cycles 16–18; re-running it would be motion, not progress. The Kaggle GPU hour was held: with TabPFN ruled out, no GPU-only experiment had expected value above the local CPU/MPS shots.

---

## 7. Final-submission selection (as important as any experiment)

Two submissions are locked at the deadline. Per the transfer analysis, the right basis is *transfer-robustness*, not raw OOF:
- **Primary:** highest-OOF blend with a **balanced family mix** and **no self-distill** (best expected transfer residual) — the `nosd` greedy or its balanced variant.
- **Secondary:** the single most robust standalone (RealMLP-multiseed family) as an uncorrelated insurance pick against a blend that over-fit the OOF.
- Spend the remaining daily slots measuring these on LB before locking, rather than on closed-lever curiosities.

---

## 8. Bottom line

- **Bank the greedy blend** — a free, real +0.0001 LB that had been sitting unbanked. The "0.95462 ceiling" framing was already superseded by the greedy 0.95479.
- **Top-7% is not honestly reachable by the deadline — *by score*.** Top-15% is banked; the top-10% *score line* (0.95449) would need an information source the project chose not to use. (This is a score-percentile statement. Separately, on *rank*: ~260 teams above us share one copied-notebook score, so collapsing that bloc puts us at effective top ~10% — see the rank note in the README. The two are different axes and do not contradict.)
- **The modeling is essentially complete** — the project independently reproduced the field's consensus stack and exhausted every clean lever with unusual rigor. The plateau is a noise ceiling plus a methodology boundary, not a mistake.
- **Last steps:** bank the greedy gain, optimize final-submission *selection* for transfer, run two cheap genuinely-new probes (ExtraTrees, richer greedy) — then stop and document the integrity ceiling.
