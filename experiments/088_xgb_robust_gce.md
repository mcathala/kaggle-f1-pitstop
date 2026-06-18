# Experiment 088 — noise-robust GCE-XGB → most-diverse strong base ever, but still blend-dead (ρ 0.96 < the ρ≤0.92 hurdle)

**Cycle.** 18
**Status.** **Inconclusive (definitive closure).** Builds a genuine strong+diverse base (OOF 0.95163, ρ 0.959 vs RM — best diversity we've ever achieved at strength) but it earns w=0.015 / +0.00000 in the blend. Closes the diverse-base lever conclusively.
**Date.** 2026-05-29

## Hypothesis

Our own EDA found 32% of train rows are instance-dependent label-noise (audit §2.1). exp 076 attacked it by *sample-downweighting* and lost −0.00091 (discards signal). Attack the same noise via a *different loss surface* instead (audit §2.4 lists robust-loss GBDT as untested): Generalized Cross-Entropy, `L_q=(1−p_t^q)/q`, which bounds the per-row gradient so confidently-"mislabeled" rows (p→0 when y=1) get g→0 instead of logloss's g→−1 — the splitter stops chasing the noise. Goal: a base strong enough to earn weight (OOF ≥ 0.951) AND diverse (ρ < 0.96 vs RM).

## Method

Forked `train_xgb_diffFE.py` → `train_xgb_robust.py`. Identical 49-feature diffFE recipe & HPs; only the objective changes to a custom GCE grad/hess (`q=0.5`), `base_score=0.0` (margin space), predictions sigmoid-mapped back to [0,1].

## The bug, the diagnosis, the fix

**First full run collapsed: OOF 0.89899.** Folds 1–3 were healthy (~0.952, ~13k iters) but folds 4–5 died at 15/28 iters (AUC 0.928/0.930). Diagnosis (fold-4 recheck, verbose curve): GCE's gradient is deliberately shrunk → the val-AUC ramp is *ultra-slow* (0.888→0.952 over ~13k iters) with an early **flat patch at iters 25–125**. The tight `early_stopping_rounds=100` fired during that plateau and killed the run before it could climb. **Fix: raise patience to 400.** Fold-4 then climbed cleanly to 0.95059. Not a real failure — a stopping-criterion artifact hiding a real result.

## Result (fixed run)

| Fold | AUC |
| --- | --- |
| 1–5 | 0.95210 / 0.95227 / 0.95153 / 0.95059 / 0.95163 |
| **OOF** | **0.95163** (mean 0.95162 ± 0.00059) |

**ρ vs RealMLP-multiseed: 0.959. ρ vs CB: 0.951.** This is the **most decorrelated strong base in the entire project** — every other base sits at ρ ≥ 0.98 vs RM. The GCE objective genuinely reshaped the decision surface (the robust loss learned a different boundary by ignoring confident-noise).

### Decisive blend test

Free coord-descent over the 8-base best-blend pool + robust:

| Blend | OOF | robust weight |
| --- | --- | --- |
| best blend (8 bases) | 0.95462 | — |
| **+ robust GCE-XGB** | **0.95462** | **0.015** |

**+0.00000.** Despite being the most diverse strong base we've ever built, it moves the blend by nothing.

## Verdict

**Inconclusive — but it definitively closes the diverse-base lever.** The audit's oracle-boost analysis estimated a base needs **ρ ≤ 0.92 @ AUC ≥ 0.951** to lift the RealMLP-dominated blend. exp 088 lands at **ρ 0.959 @ 0.952** — a real diversity gain over ρ 0.98, but still on the dead side of the hurdle. Combined with:
- exp 087 (HP-diversity, ρ 0.989 — cosmetic),
- exp 053/054 (embMLP: ρ 0.91 but only 0.94 strong → too weak),
- exp 058–060 (lap-attention: ρ 0.90–0.94 but 0.936–0.943 → too weak),

…we have now mapped the strong↔diverse frontier at fine resolution from **both** ends: the diverse side caps at ρ~0.96 once a base is strong (≥0.95), and the genuinely-diverse models (ρ≤0.92) cap at ~0.94 strength. **The strong+diverse quadrant is empty, robustly, and no own-tooling trick (FE-view, HP, robust loss) reaches it.** The blend is saturated not for lack of trying diversity, but because the achievable diversity is insufficient.

**Implication for the plan:** A2 (residual-targeted RM FE-view, gated ρ<0.96) is now pointless — even a base *clearing* ρ<0.96 adds 0 (this experiment is the proof). The only remaining path to a higher blend is **raising the absolute strength of the dominant bases (RealMLP, 0.667 weight)**, not adding diversity. Next experiments pivot accordingly (driver-residual FE for RealMLP).

## Acceptance gates

| Gate | Got | Pass? |
| --- | --- | --- |
| Standalone strong (OOF ≥ 0.951) | 0.95163 | ✅ |
| Diverse (ρ < 0.96 vs RM) | 0.959 | ✅ (barely) |
| **Blend marginal (≥ +0.0001 OOF)** | **+0.00000** | ❌ |

## Repro stamp

- Trainer: [src/train_xgb_robust.py](../src/train_xgb_robust.py) (GCE q=0.5, early_stop=400).
- Output: `data/oof_xgb_robust.parquet` (OOF 0.95163, ρ 0.959 vs RM).

## Learnings

1. **The robust-loss lever works mechanically** — GCE produced the most decorrelated strong base in the project (ρ 0.959 vs the usual 0.98). Robust losses *do* reshape the GBDT decision surface meaningfully. Worth knowing for future noisy-label problems.
2. **But ρ 0.96 is still blend-dead here.** This is the sharpest confirmation yet that the RealMLP-dominated blend needs ρ ≤ 0.92, which is unreachable at strength in our model space. **Diverse-base lever closed for good.**
3. **Always sanity-check early-stopping against the loss's convergence speed.** A robust/shrunk-gradient objective ramps slowly; a patience tuned for logloss kills it mid-ramp and masquerades as divergence. The fold-4 collapse was 100% a stopping artifact.
4. **Pivot:** stop hunting diversity; raise absolute base strength (driver-residual FE on RealMLP is the next shot — the one named, unexploited residual).
