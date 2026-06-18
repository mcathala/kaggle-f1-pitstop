# OOF→LB transfer analysis & ceiling re-audit (cycle 19, 2026-05-29)

Triggered by the standing directive to keep climbing rather than accept the
plateau. Two independent offline investigations, both reproducible from
`data/oof_*.parquet` + the Kaggle submission history.

## 1. Transfer function (own submission history, n=16 OOF↔LB pairs)

Fit: **LB ≈ 0.927·OOF + 0.069**, residual std **0.00018**.

- Slope < 1 → genuine shrinkage; the OOF→LB gap *grows* at high OOF.
- **Pseudo-drift hypothesis NOT supported.** Controlling for OOF via the fit,
  pseudo vs non-pseudo differ by only +0.00005 in transfer residual — inside the
  0.00018 noise floor. The larger raw LB−OOF gap for pseudo subs is an artifact of
  them sitting at higher OOF (more shrinkage), not a pseudo penalty. The 05-30
  controlled A/B (best vs nopseudoGBDT vs pure_nopseudo) is still worth running as
  a direct measurement, but expectations are tempered: likely within-noise.
- **Worst transferrer: self-distilled RealMLP** (residual −0.00042). Self-distill
  overfits OOF. **Permanently dropped from all contenders.**
- **"mix" (balanced) blends transfer best** (+0.00004 mean residual); gbdt-heavy
  worst (−0.00021); single/diverse slightly positive.
- **Implication for the goal:** at this transfer rate, LB 0.9544 needs OOF ≈
  **0.9551** — i.e. +0.0005 beyond the 0.95462 blend ceiling. Re-blending current
  bases cannot reach it. The climb must come from (a) a genuinely decorrelated NEW
  base that raises OOF, or (b) a composition with a positive transfer residual.

## 2. OOF ceiling re-audit (confirmed closed, 3rd independent confirmation)

- Adding any single underused base (lap_attention, embmlp, hgbc, lgb_diffFE,
  xgb_robust, …) at w=0.10 *lowers* OOF below 0.95462. Decorrelated ones
  (lap_attention, corr 0.90) are too weak (AUC 0.936); strong ones too correlated
  (0.98–0.99). Diversity-strength bind, confirmed.
- Full-pool Nelder–Mead optimization (7 best-blend members + 5 diverse bases, free
  nonneg weights) → **OOF 0.95462, identical** to the 7-member blend; diverse bases
  earn ~0 weight. The offline ceiling is real and robust.

## 3. Action taken

- **`src/research/train_xgb_monotone.py`** — new transfer-robust base. Domain-monotone
  constraints (older tyre / more degradation ⇒ higher pit prob) on a conservative
  numeric set; identical FE/HP/folds to `train_xgb_diffFE` for a controlled
  comparison. Rationale: monotone constraints can't fit noise in the physically-
  wrong direction (less overfit ⇒ better transfer per §1) and produce a decorrelated
  error structure. Queued to run when M1 memory frees (parallel agent's job is mid-
  run; memory was at 72M free — refused to risk OOM-killing their work).
- Reset safety-net: all 5 contenders auto-submit at 00:00 UTC via guarded script.

## 4. Decorrelation search — exhaustively closed (cycle 19)

To raise OOF past the greedy 0.95479 we need a base that is BOTH strong (AUC ≳0.953)
AND decorrelated (rank-corr <~0.97) so greedy can give it weight. Tested every
lever we could build on this FE/data:

| base                | AUC    | rho vs gbdt | earns greedy weight? |
|---------------------|--------|-------------|----------------------|
| monotone-XGB        | 0.9525 | 0.997       | no                   |
| LightGBM DART       | 0.9483 | 0.974       | no (AUC too low)     |
| rank:pairwise XGB   | ~0.949 | 0.979       | no (objective doesn't decorrelate) |
| lag-FE XGB          | 0.9524 | 0.987       | no                   |
| inductive XGB       | 0.9529 | 0.999       | no                   |
| attention / embMLP  | 0.936  | 0.90        | no (too weak)        |

**Conclusion:** on this target every model strong enough to matter collapses to
rho ≥0.97 with the others — the ~32% synthetic label noise forces all good models
onto the same ordering. Offline OOF is genuinely capped at **0.95479** (greedy). The
only remaining offline lever is a fundamentally different DATA VIEW (a true sequence
model over lap-order trajectory, which uses information no i.i.d. tabular model has);
everything else is LB transfer. Built `submission_blend_bagged_greedy.csv` (bagged
greedy, transfer-robust) as a 05-31 candidate alongside greedy_full/greedy_nosd.
