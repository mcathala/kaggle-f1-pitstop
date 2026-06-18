"""Historical driver-level + (driver, race) aggregates computed on train+test.

For each row, looks up that driver's historical behavior on input features
(LapTime, Position, TyreLife, Compound usage, RaceProgress) and adds them as
new feature columns. Targets the driver-level discrimination residual that
cycle 13's EDA flagged.

Crucially EXCLUDES PitStop and PitNextLap from aggregations to avoid the
label-proxy pattern that killed experiment 018. Driver-level avg PitStop rate
would be ~equivalent to driver-level avg PitNextLap rate (they differ by
one-lap shift), which would re-create the exp-18 failure mode.

Aggregates are computed on train+test combined (no labels involved):
  - Per Driver:
      driver_avg_LapTime, driver_std_LapTime  — pace and variability
      driver_avg_TyreLife                      — typical tyre age handled
      driver_avg_Position                      — typical running position
      driver_avg_AbsPositionChange             — typical movement
      driver_pct_<COMPOUND>                    — usage rate of each compound (5)
      driver_avg_RaceProgress                  — typical race progress reached
      driver_n_rows                            — total observations
  - Per (Driver, Race):
      driver_race_avg_LapTime
      driver_race_avg_Position
      driver_race_n_rows                       — laps observed at this track
  - Per (Driver, Year):
      driver_year_avg_Position
      driver_year_n_rows

Total ~16 new features. Each is a simple groupby-mean / count.

Usage:
  combined = pd.concat([train, test], ignore_index=True)
  combined = add_historical_features(combined)
  # then split back into train/test
"""

import numpy as np
import pandas as pd


def add_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ~16 historical aggregate features computed on the input frame.

    The frame should be train+test combined for in-distribution aggregates.
    Operates on a copy and returns the result. All new columns are
    float32/int32 — no missing values (every driver appears in train+test).
    """
    out = df.copy()

    # Per-Driver aggregates (across all years/races)
    out["driver_avg_LapTime"] = out.groupby("Driver")["LapTime (s)"].transform("mean").astype("float32")
    out["driver_std_LapTime"] = out.groupby("Driver")["LapTime (s)"].transform("std").fillna(0).astype("float32")
    out["driver_avg_TyreLife"] = out.groupby("Driver")["TyreLife"].transform("mean").astype("float32")
    out["driver_avg_Position"] = out.groupby("Driver")["Position"].transform("mean").astype("float32")
    out["__abs_pos_change"] = out["Position_Change"].abs()
    out["driver_avg_AbsPositionChange"] = (
        out.groupby("Driver")["__abs_pos_change"].transform("mean").astype("float32")
    )
    out["driver_avg_RaceProgress"] = out.groupby("Driver")["RaceProgress"].transform("mean").astype("float32")
    # transform("size") returns a Series of same length as input, broadcast.
    out["driver_n_rows"] = out.groupby("Driver")["Driver"].transform("size").astype("int32")

    # Per-Driver compound usage frequency (5 features) — encode compound as one-hot
    # rows, then groupby-mean gives the fraction of that driver's rows on that compound.
    for compound in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]:
        col = f"driver_pct_{compound}"
        marker = (out["Compound"] == compound).astype("float32")
        out["__marker"] = marker
        out[col] = out.groupby("Driver")["__marker"].transform("mean").astype("float32")

    # Per-(Driver, Race) aggregates
    out["driver_race_avg_LapTime"] = out.groupby(["Driver", "Race"])["LapTime (s)"].transform("mean").astype("float32")
    out["driver_race_avg_Position"] = out.groupby(["Driver", "Race"])["Position"].transform("mean").astype("float32")
    out["driver_race_n_rows"] = out.groupby(["Driver", "Race"])["Driver"].transform("size").astype("int32")

    # Per-(Driver, Year) aggregates
    out["driver_year_avg_Position"] = out.groupby(["Driver", "Year"])["Position"].transform("mean").astype("float32")
    out["driver_year_n_rows"] = out.groupby(["Driver", "Year"])["Driver"].transform("size").astype("int32")

    out = out.drop(columns=["__abs_pos_change", "__marker"])

    return out


__all__ = ["add_historical_features"]
