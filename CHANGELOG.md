# Changelog

The project's progress, newest first — one line per cycle. Per-experiment detail lives in
`experiments/NNN_*.md`; the full cycle table is in [`project.md`](project.md) §Cycle history, and the
narrative in [`case-study.md`](case-study.md).

## Final — private LB 0.95427 (top 10%)

The shipped submission is a transfer-robust **7-model blend** (RealMLP × XGBoost × CatBoost on a lean
"diffFE" feature set): public LB **0.95402**, private **0.95427** — **+0.01216** over the baseline. A
higher-OOF greedy blend (OOF 0.95479) was deliberately **not** picked and scored lower on both boards
(public 0.95380, private 0.95408), confirming the transfer-robust selection.

## Cycle history (newest first)

- **Final push (exp 065–092).** diffFE feature-stripping (XGBoost 0.95263 → 0.95291), pseudo-labeling
  round 2, and feature-view diversity lifted the public LB from 0.95373 to **0.95401–0.95402**; the final
  transfer-robust blend scored private **0.95427**.
- **Cycle 17 (exp 058–064) — Inconclusive.** Cross-lap context, blend combiners, AutoML, and pseudo-labeling
  all closed; pseudo was the one positive mechanism but sub-hurdle. Submitted LB **0.95373**.
- **Cycle 16 (exp 049–057) — Inconclusive.** Own-model, FE-on-XGB, and external-data axes all closed — a NN
  is either diverse+weak or strong+RealMLP-correlated, never both. The ceiling holds.
- **Cycle 15 (exp 048) — Inconclusive.** Optuna confirmed the cycle-11 XGB hyperparameters sit at a local optimum.
- **Cycle 14 (exp 047) — Inconclusive.** Multi-seed XGB is near-deterministic (ρ 0.999); the variance-reduction axis is closed across all tree bases.
- **Cycle 13 (exp 046) — Inconclusive.** LightGBM high-bins generalizes the `max_bin` fix but is too weak (0.949) to earn blend weight.
- **Cycle 12 — Infra-fail.** TabM untestable on the laptop (MPS); later run on GPU as exp 049.
- **Cycle 11 (exp 033–044) — Kept, LB 0.95372.** XGBoost retuned with high-resolution histograms (`max_bin=5000`) — first LB lift since cycle 7.
- **Cycle 10 (exp 028–032) — Inconclusive.** CatBoost axis exhausted; fixed-weight blends already optimal.
- **Cycle 9 (exp 026–027) — Inconclusive.** Pseudo-labeling and a year-specialist both dead.
- **Cycle 8 (exp 023–025) — Inconclusive.** Model zoo exhausted; the FE pipeline does more diversification work than the architecture.
- **Cycle 7 (exp 021–022) — Kept, LB 0.95361.** RealMLP-multiseed × CB-tuned linear blend.
- **Cycle 6 (exp 019–020) — Inconclusive.** Ceiling probe: label-correlated features, label-uncorrelated features, and an FT-Transformer all failed.
- **Cycle 5 (exp 017–018) — Kept, LB 0.95342.** RealMLP multi-seed average (pure variance reduction).
- **Cycle 4 (exp 014, 016) — Kept, LB 0.95331.** The structural pivot — a tuned tabular neural net (RealMLP / PyTabKit) becomes the ensemble anchor and beats every tree blend. The largest single-experiment LB lift of the project.
- **Cycle 3 (exp 007–013) — Kept, LB 0.95066.** CatBoost redesigned with external data + tuned hyperparameters + a richer feature pipeline.
- **Cycle 2 (exp 001–006) — Kept, LB 0.94833.** CatBoost added as a diverse base — the first cycle to move the metric.
- **Cycle 1 — baseline, LB 0.94211.** LightGBM on 49 engineered features, 5-fold StratifiedKFold on `Year × PitNextLap`.
