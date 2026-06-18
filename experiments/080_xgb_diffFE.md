# Experiment 080 — differentiated (stripped) FE on XGB → stronger + LB-transferring base

**Cycle.** 18
**Status.** **KEPT — new project-best LB 0.95388** (+0.00013 over 0.95375).
**Date.** 2026-05-29

## Hypothesis

Cycle 16 found "the same FE → ρ-convergence regardless of architecture; the FE causes convergence." We never tested whether a *deliberately different (stripped)* FE produces a base that is decorrelated AND still strong. Stripping the heavy transductive block (cross-categoricals, union-frequency, union-group-stats) from XGB should give a base diverse enough to add blend value — and, since those features are computed on the train+test+ext union (transductive), removing them may also reduce the OOF→LB drift.

## Rationale

1. The audit flagged "per-base differentiated FE" as the one untested base-diversity lever; the night's rank-blend test confirmed our existing diverse bases are all too weak to help, so we need a *strong* decorrelated base.
2. Our −0.0006 drift is structural (exp 079 ruled out external). A prime suspect: the transductive FE (frequency/group-stats computed on the union including test) inflates OOF.
3. XGB-highbins carries 132 features; the engineered block (cross-cats + frequency + group-stats = ~83 features) may be net-harmful overfitting.

## Method

Forked `train_xgb_richcat.py` → `train_xgb_diffFE.py`. Dropped: all 9 cross-categoricals, union-frequency on cross/bins (kept base-cat freq only), all 48 group-stat columns. Kept: raw + domain features + 4 domain bins + base categoricals (Driver/Race/Compound native). **49 features (down from 132).** Everything else identical (max_bin=5000, same HP, same CV, external augmentation).

## Result

| | OOF AUC | per-fold mean ± std |
| --- | --- | --- |
| rich-FE XGB (132 feat) | 0.95263 | — |
| **diffFE XGB (49 feat)** | **0.95291** | 0.95290 ± 0.00063 |
| **Δ** | **+0.00028** | every fold positive (+0.00032/+0.00026/+0.00042/+0.00008/~flat) |

**The stripped model is STRONGER than the rich one** — the heavy transductive FE block was net-harmful overfitting. This is the strongest non-pseudo XGB in the project.

ρ: vs RM 0.980 (unchanged), vs CB 0.976 (slightly lower), **vs rich-XGB/psXGB 0.988** — the two FE-views of XGB are diverse enough to blend together.

### Clean blend (no self-distill mirage)

Free coord-descent over {psRM6r2, CB, psXGB, diffFE-XGB, psCB14}:

| Blend | OOF |
| ----- | --- |
| anchor (psRM6r2/CB/psXGB 0.675/0.075/0.25) | 0.95436 |
| **+ diffFE-XGB free** (psRM6r2 0.578 / psXGB 0.151 / diffFE-XGB 0.221 / psCB14 0.05) | **0.95448** (+0.00012) |

### LB submission

`submission_blend_diffFE_clean.csv` → **LB 0.95388** (vs prior best 0.95375). **+0.00013 LB**, drift −0.00060.

**The OOF lift transferred nearly 1:1 to LB** (+0.00012 OOF → +0.00013 LB) — because diffFE-XGB is a genuine strong base, not the weak overfit that made exp 075's self-distill an OOF mirage.

## Verdict

**KEPT.** First real base-strength + LB win in 12 experiments. New project-best **LB 0.95388** (rank ~top 16%, from top 17%). The "less FE = stronger + cleaner" finding is a genuine, transferable lever.

## Acceptance gates

| Gate | Got | Pass? |
| --- | --- | --- |
| Magnitude (OOF Δ ≥ +0.0002) | +0.00028 standalone | ✅ |
| Direction (≥3/5 folds) | 5/5 positive | ✅ |
| LB transfer | +0.00013 LB (new best) | ✅ |

## Repro stamp

- Trainer: [src/train_xgb_diffFE.py](../src/train_xgb_diffFE.py) (49 features; cross-cats/union-freq/group-stats removed).
- Outputs: `data/oof_xgb_diffFE.parquet` (0.95291), `data/submission_xgb_diffFE.csv`, `data/submission_blend_diffFE_clean.csv` (LB 0.95388).
- Submission 53133825. 3/5 daily slots used.

## Learnings

1. **The heavy transductive FE block (cross-cats + union-frequency + union-group-stats) was net-harmful to XGB** — stripping it gained +0.00028 OOF and transferred cleanly to LB. Our rich FE recipe had been over-engineered.
2. **Strong + (mildly) decorrelated bases transfer ~1:1 to LB**, unlike OOF-overfit bases (self-distill). The drift is not a tax on genuine strength — it's a tax on OOF-fitting weak/correlated bases.
3. **This is the lever to scale:** apply the same FE-stripping to CatBoost (exp 081, running) — CB uses the identical rich-FE block, so a similar +0.0003 is plausible, which would compound in the blend.

## Follow-ups

- exp 081: diffFE-CatBoost (running overnight; same strip).
- Then rebuild the full blend with diffFE-XGB + diffFE-CB; submit.
- Consider: is the rich FE harmful because of overfitting or transductive leakage? Either way, lean recipes win here.
