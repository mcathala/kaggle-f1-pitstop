# Predicting F1 Pit Stops — Top 10%\* of 3,023 teams

> Written companion to the visual case study in [`case-study.html`](case-study.html) (open that for the live charts).
> Technical and terse — the same climb the page walks, in plain text, structured Part → sub-part.

---

## Part I · Overview

Kaggle **Playground Series S6E5**: one row is one lap of one driver in one race, across four seasons (2022–2025).
Predict a single yes/no per row — **does this driver pit on the *next* lap?** Binary classification, **3,023 teams**,
scored on **ROC-AUC** — not accuracy, but how well the predicted probabilities *rank* pit-laps above no-pit-laps
(0.5 = coin flip, 1.0 = perfect). AUC is the honest measure here: only ~20% of laps are pits, so accuracy just
rewards always guessing "no".

What follows is the whole climb in three parts: **the data**, and the strong signal I found in it; **the models** —
**neural-network and machine-learning models** working together — from baseline to the blend I shipped; and **the
levers every top solution shared** — all of which I'd built on.

| | Result |
|---|---|
| Gain over baseline | **+0.0122** AUC |
| Logged experiments | **88** over ~19 cycles |
| Private-LB AUC | **0.95427** (baseline 0.94211) |
| Compute | **M1 Pro · 16 GB** — only 1 of 7 bases on a free Kaggle GPU, no paid cloud |

Two things are the real story. The whole pipeline trains on a **laptop**. And I shipped the model I could
**defend**, not the highest-scoring one: a higher-offline blend existed, the evidence said it would transfer worse,
so I left it off my picks — and it did ([IV.2](#iv2-choosing-the-final-submission)). The `*` on *top 10%* is
explained in [How "top 10%" is measured](#how-top-10-is-measured).

---

## Part II · The Data

> Four seasons, 26 races, 439k laps. Beyond a noisy synthetic season, one thing really drives a pit — **strategy** —
> and the engineered features that turn it into usable signal.

### II.1 The brief

One row is one lap of one driver; the target `PitNextLap` is yes/no — does this driver pit on the **next** lap?
I was handed **439,140** training laps and **188,165** test laps across **887** drivers, **26** races and four
seasons, in **16** columns (driver, year, race, lap, position, stint, TyreLife, compound, lap-time and degradation
fields, race progress). About **20%** are pit-laps (a 80.1 / 19.9 split).

The data is **synthetic**, built from real F1 telemetry, so it behaves like an actual season — with the odd
generator glitch, like an almost pit-less **2023** (~1% pit rate against ~27–30% every other year; kept because
removing data hurts, but flagged `is_2023` so models treat it differently). Some driver IDs are synthetic too
(`D109` sitting beside `VER`/`HAM`), flagged the same way.

The label carries some scrambling: about **a third** of it is essentially random, the rest real. That sets a hard
ceiling on any model — my leftover errors track no feature I could build (ρ ≤ **0.012**), so they're irreducible,
not signal left on the table. You can see that ceiling on the leaderboard, where the whole top clusters within
~**0.0005** AUC of the leader. So the job is to extract every bit of the *real* signal — which is what the rest of
this is about.

### II.2 The signal that carries

In a target that's part noise, the signal is **strategy** — and it lives in tyre age, lap, stint, and what the rest
of the field is doing. No single raw column captures it: **TyreLife** correlates best at ρ = 0.27 (LapNumber 0.27,
Stint 0.20, RaceProgress 0.19, Position ≈ 0) — faint, but the best the raw inputs have.

So I **engineered 49 columns** from the raw 16 to surface it. The strongest pull the pit rate apart far harder than
any raw column does:

- **`field_pit_share`** — how much of the field is pitting now: **2.2% → 38.2%** across buckets (a 17× spread, the single strongest feature).
- **`tyre_life_norm`** — tyre age against its compound's limit: **3.5% → 35.9%**.
- **"did the car just ahead / behind pit?"** (undercut / overcut): **7.8% → 26.3%**.

The physics shows up cleanly when you measure it: pit rate climbs with tyre age from **5%** to **43%**, then steps to
**72%** past 50 laps — and at that age it's the durable compounds still running (HARD 0.71, MEDIUM 0.61, INTER 0.56;
SOFT never stretches that far). The signal lives in the features I built, not the raw inputs.

---

## Part III · The Models

> What I shipped, and why I took this path — how the blend is built, then the climb from baseline to final with the
> reasoning behind every step.

### III.1 What I shipped

The final submission is a **7-model blend, two-thirds RealMLP**. It combines both worlds — one tuned tabular
**neural network** anchors it and **machine-learning models** (gradient-boosted trees) fill the residual. Blending pays only when models *fail differently* — their uncorrelated errors cancel, so the
average lands above any single member — and here that diversity is scarce, so the blend stays small and deliberate.

| Weight | Component | Trained on |
|---:|---|---|
| 0.315 | RealMLP — 6 seeds, pseudo-labels | **free Kaggle GPU** |
| 0.293 | RealMLP — 6 seeds, lean features | M1 Pro |
| 0.058 | RealMLP — single seed | M1 Pro |
| 0.242 | XGBoost — pseudo-labels | M1 Pro |
| 0.020 | XGBoost | M1 Pro |
| 0.036 | CatBoost | M1 Pro |
| 0.035 | CatBoost | M1 Pro |

**Three families, because they fail differently.** **RealMLP** — a tuned multi-layer perceptron — learns a
continuous **embedding per driver**, capturing driver *style*, the residual the trees could only crudely
target-encode. It was strictly strongest: every tree-led blend scored *below* RealMLP alone, so it earns two-thirds
of the weight. **XGBoost** adds rule-based splits; its one fix here was raising `max_bin` (to 5000) so it stopped
blurring the high-resolution category features. **CatBoost** brings native categorical handling — well-suited to the
887 drivers without manual encoding, and decorrelated enough from XGBoost to earn its own slot.

The three RealMLP slots — a 6-seed average (cancels run-to-run wobble), a GPU pseudo-label variant, and a single
seed — sum to **0.666, exactly two-thirds**; the four trees fill the rest. All seven train on the lean **diffFE**
~**49-feature** set and on the competition data plus a **leak-checked** public F1-strategy set (**+101k rows** that
earned their place — removing them lowered the score).

**How the blend is built.** Each base emits 5-fold **out-of-fold (OOF)** predictions on the 439k train rows and
predictions on the 188k test rows. I find the weights by **constrained coordinate descent on the OOF matrix** —
maximising held-out OOF AUC, the only honest surface while the test labels stay hidden — then **freeze** them and
apply them to the test predictions as a normalised weighted average. The folds are stratified by Year × PitNextLap
and audited **leak-free**; because they're stratified rather than *grouped* by driver, OOF runs a touch optimistic,
which is why I re-checked every gain on the real board. OOF **saturates at 0.95462** (confirmed seven ways) — the cue
that the *ceiling*, not the combiner, is the limit. The two pseudo-label variants gave no score lift but were kept
for **diversity**, not a bump.

```
7 trained bases ──► each: 5-fold OOF (train) + preds (test)
        │
        ├─ constrained coord-descent on OOF ──► weights (max OOF AUC, frozen)
        │
        └─ weights × test preds ──► normalised average ──► submission.csv
```

### III.2 Baseline → final

Eleven milestones from the **0.94211** baseline to **0.95402** public — the visible spine of 88 logged experiments
over ~19 cycles. Each step was a diagnosis, kept only if it beat the noise floor on held-out data. Diagnoses, not
luck. The flat stretches taught as much as the wins: stalled stacking and feature tweaks are how I learned the
ensemble axis was tapped out and only a new model *family* could move the number.

| # | Step | Why | Public |
|---|---|---|---:|
| 1 | LightGBM baseline | reference | 0.94211 |
| 2 | Feature tweaks | hoped features alone would move it — reverted | — |
| 3 | Add CatBoost | a different tree for diversity | 0.94833 |
| 4 | Stacking | fancier combiner — stalled, lever tapped out | — |
| 5 | Redesign CatBoost + external data | more coverage + tuning | 0.95066 |
| 6 | **Pivot to RealMLP** | driver style was my worst residual; a net learns embeddings trees can't — **biggest jump (+0.00241)** | 0.95331 |
| 7 | Multi-seed averaging | cancel run-to-run wobble | 0.95342 |
| 8 | XGBoost high-resolution | raise `max_bin` to stop blurring category features | 0.95372 |
| 9 | **diffFE** | strip over-engineered features → every model stronger | 0.95388 |
| 10 | Pseudo-labeling | no lift; kept 2 variants for diversity | — |
| 11 | Feature-view diversity | a second feature "view" for the net | 0.95402 |

The turning point was step 6. Tabular Kaggle is usually won by trees, but here a tuned neural net broke through where
more tree tuning couldn't — and earned most of the final weight.

---

## Part IV · The Right Path

> When the competition closed, I read the strongest published solutions. No secret feature, no exotic model — three
> levers, and I'd built on all three. The only axis I didn't match was **scale**: seven models against 186–218.

### IV.1 What the top of the board shared

Three patterns recur in every strong solution, and I'd already built on each.

**1 · Diversity — stack models that fail differently.** The strongest entries stacked **186–218 base models** across
many families (trees, MLPs, transformer-style tabular nets — one even folded in a GNN for diversity). The logic is mine too — averaging
uncorrelated errors cancels noise — but they bought diversity with sheer count. Under this much label noise diversity
is scarce: past a handful, extra models mostly add correlated copies, so I kept my blend a deliberate **7 models**.

**2 · The finisher — a linear meta-learner on logits.** The near-universal finish was to turn each prediction into a
**logit** and learn the best combination with a small **logistic regression** (or greedily, by hill-climbing). My
**constrained, regularised** weighted blend is the same family of finisher. Left unconstrained it chases the offline
score into noise — exactly why my greedy blend topped offline but came back lower on the board
([IV.2](#iv2-choosing-the-final-submission)).

**3 · The real edge — lean on the original data and absorb its shift.** The top ran heavy feature engineering, but
the *decisive* edge was **how the original F1 data was used** — folded in as extra training rows *and* target-encoded
columns, with the train↔competition distribution shift absorbed rather than chased. That's precisely my external-data
augmentation plus **diffFE**: subtract, don't pile on. Stripping the heavy cross-category block lifted every model
(XGBoost 0.95263 → 0.95291) — for my lean blend, the fancy cross-category features were fitting the data's noise, not
its pattern.

**The only gap was scale.** Same three levers, but **27–31× more models** (186–218 vs 7) for ≈**+0.0008** AUC on a
band this tight. That's a compute story, not an insight one. I found the path; the rest was hardware.

### IV.2 Choosing the final submission

My highest offline score wasn't a single model — it was a **greedy** blend at OOF **0.95479**, and shipping it looked
like the obvious move. Greedy selection exists to squeeze exactly that number: start empty, keep adding whichever
model most improves OOF, then average the picks. That's also why I didn't trust it — near the noise ceiling the
offline score oversells, and a bump is as likely to be the blend fitting noise as real signal.

The public board, a visible slice of the test set, agreed: greedy came back **lower**. So I shipped the
**constrained** blend instead. At the close, the hidden private board confirmed the call.

| Blend | Offline (OOF) | Public | Private |
|---|---:|---:|---:|
| Greedy (highest offline, **not shipped**) | **0.95479** | 0.95380 | 0.95408 |
| **Constrained (shipped)** | 0.95462 | **0.95402** | **0.95427** |

Both came from the same weight optimiser over the same models — one **constrained / regularised**, the other not.
Folding greedy back *in* wouldn't help: same bases, so it only re-loosens the weights toward the version that scored
lower. That one constraint is the whole difference between the offline leader and the model that held up.

---

## Notes

### What I'd try next

The honest next move is the one lever I never pulled — **scale**.

- **Grow the model zoo.** The top ran 186–218 base models; I ran 7. More families (TabM, FT-Transformer,
  factorization machines) would add the uncorrelated errors that were missing.
- **Finish with a logit-stack, not fixed weights.** A logistic-regression meta-learner over model logits is what
  every strong solution used; my constrained blend is the same family, one step short.
- **Mine the original data harder** — more shift-aware, target-encoded views of the real F1 dataset.

None of these breaks the noise ceiling — but on a band this tight, that's where the remaining ≈**+0.0008** lives.

### How "top 10%" is measured

The `*` next to *top 10%*: official rank is **521 / 3,023 — top 17%** on the private leaderboard. But the score just
above me is shared by **260 teams** at the *identical* value **0.95454** — and independent models never tie to five
decimals. They didn't: that bloc is **one public notebook, run and submitted unchanged**. It trains no model — it
loads a pool of other people's public submission CSVs and rank-blends them into a fixed output, so anyone who runs it
submits the same 0.95454. Collapse that one copied bloc to a single entry and I sit **~270th of ~2,760 — top ~10%**.

This is a mechanical group-by-identical-score — **not** an accusation and **not** a score claim; see the
[full leaderboard breakdown](data/leaderboard.html). Reproducible code and the 88-experiment log live in this
repository: start with the [`README`](README.md) and the [experiment log](experiments/README.md).
