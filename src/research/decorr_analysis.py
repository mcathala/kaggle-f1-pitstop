import itertools
import os

import pandas as pd
from scipy.stats import spearmanr

DATA = "/Users/maximecathala/Documents/kaggle-f1-pitstop/data"
files = [
    "submission_blend_best.csv",
    "submission_blend_diffFE_v4.csv",
    "submission_blend_greedy_nosd.csv",
    "submission_blend_greedy_full.csv",
    "submission_blend_bagged_greedy.csv",
    "submission_blend_pure_nopseudo.csv",
]

preds = {}
for f in files:
    path = os.path.join(DATA, f)
    if not os.path.exists(path):
        print(f"MISSING: {f}")
        continue
    df = pd.read_csv(path)
    # pick non-id numeric column
    cols = [c for c in df.columns if c.lower() != "id"]
    # ensure sorted by id for alignment
    idcol = [c for c in df.columns if c.lower() == "id"][0]
    df = df.sort_values(idcol).reset_index(drop=True)
    valcol = cols[0]
    preds[f] = df[valcol].to_numpy()
    print(f"{f}: col={valcol} n={len(df)}")

# verify equal lengths
lengths = {f: len(v) for f, v in preds.items()}
print("lengths:", set(lengths.values()))

results = []
for a, b in itertools.combinations(preds.keys(), 2):
    rho, _ = spearmanr(preds[a], preds[b])
    results.append((a, b, rho))

results.sort(key=lambda x: x[2])
print("\nPairwise Spearman (ascending):")
for a, b, rho in results:
    print(f"{rho:.6f}  {a} | {b}")

lowest = results[0]
print(f"\nMost decorrelated: {lowest[0]} | {lowest[1]} = {lowest[2]:.6f}")
