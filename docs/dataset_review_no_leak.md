# Dataset review — no exploitable structure (target is synthetic, no transductive leak)

**Cycle.** 18 · **Date.** 2026-05-29

Hunt for any exploitable data structure (the data is transductive — train/test laps interleaved within driver-race — so a leak was plausible). **All angles negative:**

| check | result | verdict |
| --- | --- | --- |
| **next-lap stint increment** → PitNextLap (gap==1, n=182k) | 78% "acc" but **AUC 0.547** | base-rate illusion; no ranking signal |
| **PitStop[next lap]** → PitNextLap (the direct next-lap pit indicator) | **AUC 0.536** (gap==1), 0.545 (any-next) | PitNextLap ≈ uncorrelated with the real next-lap pit event |
| our blend AUC on the same gap==1 rows | **0.960** | models already vastly out-predict the raw observation |
| id ↔ target correlation | −0.0001; train ids 0–439139, test 439140–627304 (clean sequential split, no overlap) | id uninformative |
| exact train/test feature-duplicate rows | **0** | no label-transfer |
| train/test distribution shift (prior, adv-AUC) | ~0.5 (no shift) | — |

## Conclusion

**PitNextLap is a heavily synthetic, feature-driven target** — it is *not* recoverable from the observed race sequence (next-lap stint/PitStop carry AUC ~0.54, near-random). There is **no transductive leak, no oracle, no duplicate label-transfer, no id signal**. Our models (AUC 0.954) already extract the synthetic generator's feature-pattern far better than any raw observed signal — confirming why the whole leaderboard jams at ~0.9545 (a shared feature-pattern ceiling under ~32% instance noise).

This closes the "exploit the data structure" lever definitively. Combined with the modeling/combiner closures, the own-pipeline ceiling (OOF 0.95462) is confirmed from the data side too. Remaining levers: LB-transfer experiments (pseudo-drift candidates) and the rank-objective base — both about *transfer*, not new OOF signal.

## FE / encoding — residual-correlation scan (the definitive FE test)

To answer "can any feature/encoding help?" rigorously: correlate each candidate feature with the **blend's residual** (its errors). A feature can only add signal if it correlates with what the model gets wrong. Scanned ~18 features across every family — peer-relative (same Year/Race/LapNumber: tyre/pos/deg percentile-rank, peer-mean-delta), temporal/sequence (pits-so-far, lap-in-stint), interactions (tyre×deg, laptime×tyre, pos×raceprog), polynomials (tyre², raceprog²), and encodings:

| family | max \|resid-corr\| |
| --- | --- |
| peer-relative (same race-lap) | 0.012 |
| temporal / sequence | 0.009 |
| interactions / polynomials | 0.009 |
| **leaky in-sample driver target-encoding** | **0.006** |

**Every family is ~0.** The decisive one: a *leaky* driver-pitrate target-encoding (computed in-sample, i.e. cheating) has resid-corr just −0.006 — if the model were missing driver-level discrimination (the audit's hypothesized residual), this would light up. It doesn't. **The blend's errors are uncorrelated with every constructible feature → they are irreducible synthetic label noise, not missing FE.** FE and encoding are conclusively saturated; no transform/interaction/encoding can move the ranking. The one historical FE win (diffFE) was *removing* over-engineering, never adding signal — consistent with this.

