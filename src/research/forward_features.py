"""Forward-looking row features within (Race, Year, Driver) timelines.

The Kaggle Playground S6E5 train/test split is row-level random — for any test
row, the lap before and the lap after of the same (Race, Year, Driver) are
usually in train (see docs/eda.md §8 and docs/feature_engineering.md §6 and §9).
This means we can compute forward-looking features safely WITHOUT a label leak:
the `PitStop` column is observed for both train and test (it's a feature, not
the target), and ditto for `TyreLife`, `Compound`, `LapNumber`. The TARGET
column `PitNextLap` is only in train, so we never use it in forward features.

The agreement between `PitNextLap_i` and `PitStop_{i+1}` is 81% empirically
(docs/feature_engineering.md §6) — not strict, so `next_PitStop` is a strongly
correlated feature, NOT a label proxy. The 19% disagreement carries real signal
for the model to learn over.

For external data (`f1_strategy_dataset.csv`), forward features are computed
within its own timelines — no cross-dataset peeking.

Adds these columns to the dataframe:
  next_PitStop                    PitStop of next observation in same group
  next_TyreLife                   TyreLife of next observation
  next_LapNumber                  LapNumber of next observation
  laps_until_next_observation     next_LapNumber − LapNumber (≥ 1)
  next_Compound                   Compound of next observation (string)
  next_Compound_changed           1 if next_Compound != current Compound
  next_TyreLife_drop              1 if next_TyreLife < TyreLife (tyre reset)
  prev_PitStop                    PitStop of previous observation (backward — included for symmetry)

Where no next observation exists, fields are NaN or 0 (numeric) / "__NA__" (str).

Usage in a trainer:
  df = pd.concat([train, test], ignore_index=True)
  df = add_forward_features(df, group_cols=["Race", "Year", "Driver"])
  # then split back via the original id column
"""

import numpy as np
import pandas as pd


def add_forward_features(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
    lap_col: str = "LapNumber",
) -> pd.DataFrame:
    """Compute next-row and previous-row features within (group_cols) timelines.

    Operates IN PLACE on a sorted copy and returns the result re-indexed back
    to the original row order. Group identity is (Race, Year, Driver) by
    default — that's the timeline boundary in this dataset.

    Each numeric/string column is filled with a safe sentinel if no next row
    exists for the group (last lap of each driver-race):
      - numeric -> NaN
      - bool/int flags -> 0
      - string -> "__NA__"
    """
    if group_cols is None:
        group_cols = ["Race", "Year", "Driver"]

    out = df.copy()
    out["__orig_order"] = np.arange(len(out), dtype=np.int64)
    # Sort by group + lap so shift(-1) within group lookups land on the next-lap row.
    out = out.sort_values(group_cols + [lap_col], kind="mergesort").reset_index(drop=True)

    grp = out.groupby(group_cols, sort=False)

    # Forward-looking features (next observation in the same driver-race)
    out["next_PitStop"] = grp["PitStop"].shift(-1).astype("Float32")
    out["next_TyreLife"] = grp["TyreLife"].shift(-1).astype("Float32")
    out["next_LapNumber"] = grp[lap_col].shift(-1).astype("Float32")
    out["next_Compound"] = grp["Compound"].shift(-1)
    # Tyre-reset proxy: next observation's tyre life is lower → stint changed
    out["next_TyreLife_drop"] = (
        (out["next_TyreLife"].fillna(out["TyreLife"]) < out["TyreLife"]).astype(np.int8)
    )
    # Compound change between current and next row
    out["next_Compound_changed"] = (
        (out["next_Compound"].fillna(out["Compound"]) != out["Compound"]).astype(np.int8)
    )
    # Spacing — how many laps to the next observation in train+test combined.
    out["laps_until_next_observation"] = (
        (out["next_LapNumber"] - out[lap_col]).astype("Float32")
    )

    # Backward symmetry (cheap, often useful — most lag features in this project
    # already exist for LapTime, but PitStop_{i-1} is new and could matter).
    out["prev_PitStop"] = grp["PitStop"].shift(1).astype("Float32")

    # Fill missing categorical with sentinel
    out["next_Compound"] = out["next_Compound"].fillna("__NA__")

    # Restore original order
    out = out.sort_values("__orig_order", kind="mergesort").drop(columns="__orig_order").reset_index(drop=True)
    return out


__all__ = ["add_forward_features"]
