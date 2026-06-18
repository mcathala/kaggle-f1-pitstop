# Experiment 049 — TabM_D_Classifier on Kaggle GPU (cycle 12 retry)

**Cycle.** 16
**Status.** Inconclusive (Reverted) — TabM caps at fold-1 0.941-0.944 across two learning rates, well below the 0.949 blend-inclusion floor; standalone validation peaks at epoch 0 and degrades, indicating a structural architecture/data mismatch rather than a tuning problem. Closes the TabM axis (answers cycle 12's deferred question).
**Date.** 2026-05-26 (design) / 2026-05-27 (result)

## Hypothesis

TabM_D_Classifier (PyTabKit transformer-style tabular model) trained on Kaggle Notebooks T4/P100 GPU produces standalone OOF AUC ≥ 0.949 (the project's blend-inclusion floor established in cycle 13) with rank-correlation vs RealMLP-multiseed < 0.98, qualifying it as a 4th blend base. The resulting 4-way blend (RealMLP-multiseed + CB-tuned-exp14 + XGB-highbins + TabM) lifts OOF AUC by ≥ +0.00020 over cycle-11's 0.95421.

## Rationale

Cycle 12 closed Infra-fail on this exact experiment: TabM_D_Classifier with library defaults was untestable on M1 Pro / MPS (Epoch 0 took 22 min, Epoch 1 ETA 120-180 min as unified-memory pressure cascaded). The model itself was never evaluated — we have no AUC, no rank-corr, no diversity diagnostic. That's the largest unknown still on the table for this competition.

Cycle 15 (exp 048 Optuna XGB) confirmed the M1 Pro compute ceiling is structural — TPE found cycle-11's HPs at a local optimum on CPU. The path forward requires GPU. Kaggle Notebooks offer 30h/week of free T4 or P100 16 GB GPU, which is more than enough for a single 5-fold TabM run (~30-60 min estimated based on RealMLP's 25-min wall on M1 MPS at smaller batch).

Three knock-on questions cycle 12 left open:
1. **Does TabM produce a competitive standalone AUC on this data?** Cycle 12's "≤0.94 projection" was a single Epoch-0 extrapolation from a non-converged run — unreliable. The honest answer is unknown.
2. **If yes, is its rank ordering different enough from RealMLP to add blend value?** Both are tabular NNs but with different architectures (TabM = attention-based, RealMLP = MLP + PBLD). Plausibly rank-corr < 0.98 (vs XGB-highbins at 0.984).
3. **If standalone < 0.949 but rank-corr < 0.97, does the diversity compensate?** Cycle 13 (LGB-highbins) showed standalone 0.949 is the floor below which no blend weight is awarded, *even with* high diversity (rank-corr 0.967 vs XGB-highbins). TabM would need to clear that floor.

## Expected magnitude

- **Floor (kill if below):** standalone OOF < 0.945 → TabM materially worse than every other base; the +0.949 floor cannot be cleared by diversity alone.
- **Inconclusive band:** standalone OOF ∈ [0.945, 0.949] OR rank-corr ≥ 0.99 → TabM viable solo but doesn't earn blend weight. Documents a third trained-NN-family ceiling.
- **Target (Kept):** standalone OOF ≥ 0.949 AND rank-corr vs RealMLP < 0.98 AND 4-way blend OOF ≥ 0.95441 (+0.00020 over cycle-11). Submission earns LB validation.
- **Best case:** standalone OOF ≥ 0.953 (RealMLP territory) AND rank-corr < 0.97 → 4-way blend ≥ 0.95460, LB lift > 0.0003. This is unlikely but the upside motivates the experiment.

## Kill criteria

- [ ] Kaggle Notebook GPU not actually allocated (silent CPU downgrade) — same failure mode as exp 033 (cycle 11). Mitigated by printing `nvidia-smi` + `torch.cuda.is_available()` first thing.
- [ ] Fold-1 wall time > 2h on T4/P100 — infra problem, not science.
- [ ] Standalone OOF < 0.945.
- [ ] Rank-corr vs RealMLP-multiseed > 0.99 (TabM produces near-identical ranking — no diversity).
- [ ] 4-way blend OOF unchanged from cycle-11's 0.95421 despite standalone clearing the 0.949 floor — closes the "more model families help" axis.

## Plan

1. **Implement [gpu-kernels/cycle16_kaggle_gpu.py](../gpu-kernels/cycle16_kaggle_gpu.py)** — adaptation of [src/research/train_tabm.py](../src/research/train_tabm.py) (cycle 12) to Kaggle paths + CUDA device + GPU-appropriate `batch_size`. Verbatim FE from cycle 12; same `StratifiedKFold(shuffle=True, random_state=42)` on `Year × PitNextLap` (CV protocol frozen).
2. **Configure the Kaggle kernel** — kernel ID for cycle 16, inputs: competition `playground-series-s6e5` + external `<external-f1-strategy-dataset>`. GPU on. Internet on (for pip install pytabkit).
3. **Push notebook via `.venv/bin/kaggle kernels push gpu-kernels/`** and run remotely (~30-60 min wall on T4).
4. **Download `oof_tabm_kaggle.parquet` + `submission_tabm_kaggle.csv`** from Kaggle Notebook Output tab; place in `data/`.
5. **Compute rank-corr vs RealMLP-multiseed, CB-tuned-exp14, XGB-highbins** — local script.
6. **Run blend probe locally** — 3-way baseline cycle-11 weights + 4-way grid sweep with TabM as new base.
7. **If 4-way OOF ≥ 0.95441 with ≥ 5/5 folds positive:** submit `submission_blend_4way_tabm.csv` via `.venv/bin/kaggle competitions submit`. Wait for LB.
8. **Document `experiments/049_tabm_kaggle_gpu.md` Result + Verdict sections.** Update `project.md` cycle history + `experiments/README.md` index.

## Result

Ran as a single-fold probe on Kaggle Notebooks (Tesla P100 16 GB). Getting TabM to run at all required climbing a dependency/hardware ladder (documented for future GPU work): Kaggle's default PyTorch 2.10 dropped sm_60 kernels → pin `torch==2.5.1`+`torchvision==0.20.1` (cu121, ABI-matched); `compile_model` must be False (Triton needs CUDA capability ≥ 7.0, P100 is 6.0); AMP on (P100 has full-rate FP16).

Two fold-1 configs, both with library-default `tabm_k` internal ensembling:

| Config | tabm_k | batch | lr | patience | fold-1 AUC | wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v7 | 8 | 1024 | 2e-3 | 8 | **0.94356** | 28.8 min |
| v8 | 16 | 512 | 5e-4 | 12 | **0.94142** | 79.4 min |

In **both** runs the validation metric peaked at **epoch 0** and degraded monotonically thereafter (best model = epoch-0 weights restored). Lowering the learning rate 4× (v7→v8) did not fix the collapse — it made fold-1 slightly worse. A model that cannot improve past its first epoch across two very different learning rates is exhibiting a structural architecture/data mismatch, not a tuning artifact.

Reference fold-1 AUCs: RealMLP ~0.95421, XGB-highbins ~0.95331, project blend-inclusion floor 0.949. TabM lands −0.0054 to −0.0076 **below the floor** — it cannot earn blend weight, so no full 5-fold or blend probe was run (v8 extrapolated to ~6.6 h for 5 folds — pure waste at this AUC).

## Verdict

**Inconclusive (Reverted).** TabM_D_Classifier is not competitive on this data. Standalone fold-1 caps at 0.941-0.944 — well below every existing base and below the 0.949 floor established in cycle 13. The epoch-0 validation peak across two learning rates indicates TabM's inductive bias extracts less signal from our feature set than RealMLP's MLP+PBLD does (echoing cycle 6's FT-Transformer result, exp 020, where another attention-style tabular model also underperformed RealMLP by ~0.009). Closes the TabM axis and answers cycle 12's deferred question: the cycle-12 infra-fail was not hiding a strong model.

## Kill-criteria check

- [x] Standalone OOF < 0.945 — **FIRED** (fold-1 0.94142-0.94356, both below 0.945).
- [x] GPU dependency ladder resolved (so this is a real model result, not infra-fail): torch 2.5.1 + torchvision 0.20.1, compile off, AMP on — confirmed running on P100.

## Repro stamp (target)

- Kaggle kernel: `mcathala/cycle-16-tabm-gpu-exp-049`
- packages on Kaggle: torch (Kaggle default ≥ 2.5), pytabkit (latest from PyPI), pandas, scikit-learn (Kaggle defaults)
- CV protocol: identical to cycles 4-15 (StratifiedKFold, 5 splits, shuffle=True, random_state=42 on `Year × PitNextLap`)

## Notes

- Cycle-11 exp 033 (`gpu-kernels/cycle11_kaggle_gpu.py`) is the working precedent for this pattern. It hit Infra-fail there too — Kaggle silently allocated CPU instead of GPU. Mitigated here by failing fast on `torch.cuda.is_available() == False`.
- The competition deadline is 2026-05-31 (5 days). Kaggle Notebook session cap is 12h; this experiment fits comfortably.
- Per `project.md` thresholds: `min_delta=0.00020` for OOF lift to qualify as Kept, and `max_gap_increase_absolute=0.0005` for LB drift. Both apply.
