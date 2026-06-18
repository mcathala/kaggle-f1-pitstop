# Experiment 020 — FT-Transformer via PyTabKit

**Cycle.** 6
**Status.** Reverted (standalone OOF much weaker than RealMLP; killed after fold 1)
**Date.** 2026-05-22

## Hypothesis

PyTabKit's `FTT_D_Classifier` (FT-Transformer with tuned defaults) trained on the same FE pipeline as cycle 4 RealMLP produces a model whose OOF residuals are structurally diverse from RealMLP's. When ensembled with cycle 5 multi-seed RealMLP at fixed weights, the blend lifts OOF AUC by ≥ +0.0005 over cycle 5's 0.95383.

## Rationale

Exp 18 (forward features) and exp 19 (historical features) both failed — adding ANY new features to RealMLP regresses it. The remaining cycle-6 path is **architecture diversity**: same data, same FE, different model. FT-Transformer is the natural first probe because:

- Transformer-based attention is structurally different from RealMLP's MLP + PBLD numeric embedding.
- Same PyTabKit API → trivial swap, all our infrastructure compatible.
- Published as competitive on tabular benchmarks (Gorishniy et al, NeurIPS 2021).
- Pre-tuned defaults in PyTabKit (`_D_` suffix).

## Result

Killed after fold 1 on poor standalone performance.

```
Fold 1/5  AUC = 0.94535  (1365.5s ~= 23 min)
Cycle 4 RealMLP fold 1: 0.95421
Δ vs RealMLP: -0.00886
```

FT-Transformer fold 1 is **substantially weaker** than RealMLP standalone — about the level of CB-tuned-exp14 (0.95114) but slightly worse. Wall-clock per fold is ~4.5× RealMLP's (23 min vs 5 min).

### Why this kills the experiment

Cycle 4 established the blend monotonicity finding: adding ANY weaker model to the RealMLP blend HURTS, even CatBoost-tuned-exp14 at OOF 0.95114. FTT at OOF ~0.945 is below that bar. Continuing 4 more folds (~90 more min) almost certainly produces an OOF in the same range, which then can't add ensemble value.

The pre-tuned FT-Transformer doesn't extract enough signal from this specific feature space to compete with RealMLP. Possible reasons:

1. **FT-Transformer is more sensitive to feature scale/encoding** than RealMLP. Our FE pipeline was tuned with RealMLP's preprocessing assumptions; FTT may not get the same benefit.
2. **Attention over our 46 features doesn't beat MLP+PBLD** for this specific data shape. The "feature tokens" formulation may not match how F1 telemetry features interact.
3. **PyTabKit's `_D_` (tuned defaults) for FTT** were tuned on different benchmark datasets; not as well-calibrated to our distribution as RealMLP's defaults are.

## Verdict

**Reverted.**

Standalone OOF is too far below RealMLP to participate in the ensemble. Three independent cycle-6 paths have now failed:

| Exp | Approach | Result |
|---|---|---|
| 018 | Add label-correlated features | Reverted (label-proxy) |
| 019 | Add label-uncorrelated features | Reverted (broke pre-tuned defaults) |
| 020 | Different model family (FTT) | Reverted (standalone too weak) |

## Learnings

1. **The "architecture-loop" framing was overoptimistic** on this data. RealMLP's pre-tuned defaults converge to a higher OOF than other PyTabKit architectures, suggesting our data shape happens to be well-matched to RealMLP specifically.
2. **One-fold evidence is sufficient for the kill decision** when the gap to baseline is > 0.005 standalone — the blend monotonicity finding from cycle 4 gives us strong priors.
3. **Untried cycle-6 paths that might still work** (but we're not pursuing in this round):
   - HP-retuned wider RealMLP (Optuna sweep with feature additions)
   - RealTabR or TabM (other PyTabKit variants, but FTT result suggests low probability)
   - AutoGluon-Tabular (automated multi-architecture ensemble; could surface something)
   - Pseudo-labeling (high public-LB-overfit risk)

## Follow-ups

1. ✅ Killed exp 20 after fold 1.
2. **Close cycle 6** as Inconclusive — three independent paths all failed. Cycle 5 multi-seed (LB 0.95342) remains project tip.
3. **`src/research/train_ftt.py` kept for reference** — uses `FTT_D_Classifier` with minimal params; would need HP tuning to be competitive on this data.
4. The cycle-6 negative results are durable evidence: don't pursue "feature addition" or "different PyTabKit architecture" without HP retuning. Saved future cycles from repeating these.
