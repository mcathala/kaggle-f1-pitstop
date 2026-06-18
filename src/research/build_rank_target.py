"""Build rank-target dataset for exp 073.

Recipe (c): for each row, derive `laps_to_next_PitStop` from the (Driver, Race, Year)
timeline, then convert to `rank_pct` within each (Year, Race) group.

Censoring: rows whose stint never ends in a PitStop within their timeline get
laps_to_next_PitStop = max_possible_laps + 1 (puts them at the top of the
rank, meaning "farthest from pitting").

Output: data/rank_target.parquet with columns:
  id, laps_to_next_pit, rank_pct, censored
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent.parent / "data"


def build_rank_target(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.sort_values(["Year", "Race", "Driver", "LapNumber"]).reset_index(drop=True)

    # Within each (Year, Race, Driver), find for each row the LapNumber of the
    # next PitStop=1 in the same group. Censor if none.
    df["_grp"] = df["Year"].astype(str) + "|" + df["Race"].astype(str) + "|" + df["Driver"].astype(str)
    df["_lap"] = df["LapNumber"].astype(np.int32)

    pit_rows = df.loc[df["PitStop"] == 1, ["_grp", "_lap"]].rename(columns={"_lap": "pit_lap"})
    # Pre-build {group: sorted np.array of pit laps} to avoid O(n_groups × n_total) filtering
    grp_to_pits: dict[str, np.ndarray] = {
        g: sub["pit_lap"].to_numpy() for g, sub in pit_rows.sort_values(["_grp", "pit_lap"]).groupby("_grp", sort=False)
    }

    out_next_pit_lap = np.full(len(df), -1, dtype=np.int32)
    for grp, sub in df.groupby("_grp", sort=False):
        grp_pits = grp_to_pits.get(grp)
        if grp_pits is None or len(grp_pits) == 0:
            continue
        rows_idx = sub.index.to_numpy()
        rows_lap = sub["_lap"].to_numpy()
        positions = np.searchsorted(grp_pits, rows_lap, side="right")
        valid = positions < len(grp_pits)
        out_next_pit_lap[rows_idx[valid]] = grp_pits[positions[valid]]

    df["next_pit_lap"] = out_next_pit_lap
    df["censored"] = (df["next_pit_lap"] == -1).astype(np.int8)
    raw_laps_to_pit = df["next_pit_lap"] - df["_lap"]

    # Censored rows = max-laps-in-race + 1 (per-race max, so per-(Year,Race) rank is stable)
    df["_race_key"] = df["Year"].astype(str) + "|" + df["Race"].astype(str)
    race_max = df.groupby("_race_key")["_lap"].max().rename("race_max_lap")
    df = df.merge(race_max.reset_index(), on="_race_key", how="left")
    censor_value = (df["race_max_lap"] - df["_lap"] + 1).clip(lower=1)
    df["laps_to_next_pit"] = np.where(df["censored"] == 1, censor_value, raw_laps_to_pit).astype(np.int32)

    # Rank-pct per (Year, Race): 0 = soonest to pit, 1 = farthest from pit
    df["rank_pct"] = df.groupby("_race_key")["laps_to_next_pit"].rank(method="average", pct=True).astype(np.float32)

    # Sanity check: for rows where PitNextLap=1, laps_to_next_pit should be 1 (next lap is pit)
    if label == "train":
        assert (df["PitNextLap"] == 1).sum() > 0
        chk = df.loc[df["PitNextLap"] == 1, "laps_to_next_pit"]
        soon_to_pit = (chk == 1).sum()
        print(f"  {label}: of {len(chk):,} PitNextLap=1 rows, {soon_to_pit:,} have laps_to_next_pit=1 "
              f"({soon_to_pit/len(chk)*100:.1f}%; rest = noise-only positives)")

    return df.sort_values("id").reset_index(drop=True)[["id", "laps_to_next_pit", "rank_pct", "censored"]]


def main():
    print("Reading train + test...")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    test["PitNextLap"] = -1  # placeholder; not used
    print(f"  train {train.shape}  test {test.shape}")

    print("\nBuilding train rank-target...")
    train_rt = build_rank_target(train, "train")
    out_train = DATA / "rank_target_train.parquet"
    train_rt.to_parquet(out_train, index=False)
    print(f"  wrote {out_train.name}")
    print(f"  rank_pct distribution: mean={train_rt['rank_pct'].mean():.4f}  std={train_rt['rank_pct'].std():.4f}")
    print(f"  censored rate: {train_rt['censored'].mean()*100:.2f}%")

    print("\nBuilding test rank-target...")
    test_rt = build_rank_target(test, "test")
    out_test = DATA / "rank_target_test.parquet"
    test_rt.to_parquet(out_test, index=False)
    print(f"  wrote {out_test.name}")
    print(f"  rank_pct distribution: mean={test_rt['rank_pct'].mean():.4f}  std={test_rt['rank_pct'].std():.4f}")
    print(f"  censored rate: {test_rt['censored'].mean()*100:.2f}%")


if __name__ == "__main__":
    main()
