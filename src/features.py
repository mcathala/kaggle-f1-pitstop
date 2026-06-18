"""Build internal-aggregation features for the F1 pit-stop dataset.

Adds:
  - Per-Race:           pit_cost_seconds, pit_window_start, pit_window_end
  - Per-(Race, Year):   total_laps_year, LapsRemaining, is_wet_race,
                        expected_stops_for_race
  - Per-(Race, Year, LapNumber):
                        field_median_laptime, field_max_laptime,
                        wet_compound_share, field_pace_ratio,
                        sc_likely, sc_prev_lap, sc_lap_minus2, laps_since_sc,
                        field_pit_share, field_pit_share_prev_lap,
                        field_pit_share_lap_minus_2, field_pit_share_window_3
  - Per-Compound:       compound_max_life
  - Per-(Compound, Race):
                        typical_stint_length
  - Per-(Driver, Race, Year):
                        pits_so_far_this_race, stops_remaining_proxy,
                        prev_stint_length
  - Per-row derived:    tyre_life_norm, tyre_life_remaining,
                        cant_finish_on_current_tyres, in_pit_window,
                        driver_pace_ratio,
                        ahead_pitted_last_3, ahead_pitted_last_5,
                        behind_pitted_last_3, behind_pitted_last_5,
                        laptime_roll_mean_3, laptime_roll_mean_5,
                        laptime_roll_std_3, laptime_roll_std_5,
                        laptime_diff_1, laptime_diff_3,
                        position_change_3, position_change_5,
                        position_roll_std_5, cum_deg_diff_1,
                        stint_start_pace, laptime_vs_stint_start,
                        laptime_accel_3, laptime_accel_roll_3
  - Per-(Race, Year, Lap, Compound):
                        tyre_age_pct_among_compound_peers
  - Per-row flags:      is_pre_season, is_synthetic_driver, is_2023,
                        lap_is_anomalous

All features are computed on train+test combined (no labels used) and joined
back per row. Outputs:
  data/train_features.parquet
  data/test_features.parquet

Note: the original LapTime_Delta and Position_Change columns have been flagged
as inconsistent with their documented definitions for some rows; the timeline
features below recompute cleaner versions from raw LapTime / Position.
"""

from pathlib import Path

import polars as pl

DATA = Path(__file__).resolve().parent.parent / "data"
TRAIN_CSV = DATA / "train.csv"
TEST_CSV = DATA / "test.csv"
OUT_TRAIN = DATA / "train_features.parquet"
OUT_TEST = DATA / "test_features.parquet"

WET_COMPOUNDS = ["INTERMEDIATE", "WET"]
# field_median / clean_baseline above this ratio => safety car / VSC suspected
SC_THRESHOLD = 1.15


def load_combined() -> pl.DataFrame:
    train = pl.read_csv(TRAIN_CSV).with_columns(pl.lit("train").alias("__split"))
    test = pl.read_csv(TEST_CSV).with_columns(
        pl.lit("test").alias("__split"),
        pl.lit(None).cast(pl.Float64).alias("PitNextLap"),
    )
    return pl.concat([train, test], how="diagonal_relaxed")


def add_race_level(df: pl.DataFrame) -> pl.DataFrame:
    # Pit-lane cost proxy: median(pit-lap time) - median(non-pit-lap time) per Race.
    # Computed across years to keep n high (2023 has very few PitStop=1 rows).
    pit_lap_med = (
        df.filter(pl.col("PitStop") == 1)
        .group_by("Race")
        .agg(pl.col("LapTime (s)").median().alias("__pit_lap_med"))
    )
    norm_lap_med = (
        df.filter(pl.col("PitStop") == 0)
        .group_by("Race")
        .agg(pl.col("LapTime (s)").median().alias("__norm_lap_med"))
    )
    pit_cost = (
        pit_lap_med.join(norm_lap_med, on="Race", how="left")
        .with_columns(
            (pl.col("__pit_lap_med") - pl.col("__norm_lap_med")).alias("pit_cost_seconds")
        )
        .select("Race", "pit_cost_seconds")
    )
    df = df.join(pit_cost, on="Race", how="left")

    # Race length (this year) and laps remaining.
    total_laps = df.group_by(["Race", "Year"]).agg(
        pl.col("LapNumber").max().alias("total_laps_year")
    )
    df = df.join(total_laps, on=["Race", "Year"], how="left").with_columns(
        (pl.col("total_laps_year") - pl.col("LapNumber")).alias("LapsRemaining")
    )

    # Wet race indicator: any INTER/WET compound seen in this (Race, Year).
    wet_race = (
        df.with_columns(pl.col("Compound").is_in(WET_COMPOUNDS).alias("__wet"))
        .group_by(["Race", "Year"])
        .agg(pl.col("__wet").any().cast(pl.Int8).alias("is_wet_race"))
    )
    df = df.join(wet_race, on=["Race", "Year"], how="left")

    return df


def add_field_lap_features(df: pl.DataFrame) -> pl.DataFrame:
    # Per-lap field-wide statistics: SC affects every car on the same lap, so the
    # median lap time across the field is the right level to detect it.
    field = df.group_by(["Race", "Year", "LapNumber"]).agg(
        pl.col("LapTime (s)").median().alias("field_median_laptime"),
        pl.col("LapTime (s)").max().alias("field_max_laptime"),
        pl.col("Compound")
        .is_in(WET_COMPOUNDS)
        .cast(pl.Float64)
        .mean()
        .alias("wet_compound_share"),
    )
    return df.join(field, on=["Race", "Year", "LapNumber"], how="left")


def add_sc_features(df: pl.DataFrame) -> pl.DataFrame:
    # "Clean" baseline pace per (Race, Year): 25th percentile of the per-lap field
    # median. SC laps push the field median up, so the lower-quartile lap is a
    # robust estimate of the green-flag pace for that race-year.
    baseline = df.group_by(["Race", "Year"]).agg(
        pl.col("field_median_laptime").quantile(0.25).alias("__baseline_pace")
    )
    df = df.join(baseline, on=["Race", "Year"], how="left").with_columns(
        (pl.col("field_median_laptime") / pl.col("__baseline_pace")).alias("field_pace_ratio"),
        (
            pl.col("field_median_laptime") > pl.col("__baseline_pace") * SC_THRESHOLD
        )
        .cast(pl.Int8)
        .alias("sc_likely"),
    )

    # Lag SC indicator at the (Race, Year, LapNumber) level — one row per lap,
    # then joined back so every driver on that lap gets the same lag values.
    sc_per_lap = (
        df.select(["Race", "Year", "LapNumber", "sc_likely"])
        .unique(["Race", "Year", "LapNumber"])
        .sort(["Race", "Year", "LapNumber"])
    )
    sc_per_lap = sc_per_lap.with_columns(
        pl.col("sc_likely").shift(1).over(["Race", "Year"]).alias("sc_prev_lap"),
        pl.col("sc_likely").shift(2).over(["Race", "Year"]).alias("sc_lap_minus2"),
        # laps_since_sc: cumulative-sum of sc_likely defines an SC "epoch"; row
        # index within an epoch is the number of laps elapsed since the last SC.
        pl.col("sc_likely").cum_sum().over(["Race", "Year"]).alias("__sc_epoch"),
    )
    sc_per_lap = sc_per_lap.with_columns(
        pl.int_range(pl.len()).over(["Race", "Year", "__sc_epoch"]).alias("laps_since_sc")
    ).with_columns(
        # Before any SC has occurred in the race, laps_since_sc is undefined.
        pl.when(pl.col("__sc_epoch") == 0)
        .then(None)
        .otherwise(pl.col("laps_since_sc"))
        .alias("laps_since_sc")
    ).drop("__sc_epoch", "sc_likely")

    return df.join(sc_per_lap, on=["Race", "Year", "LapNumber"], how="left")


def add_field_pit_cluster(df: pl.DataFrame) -> pl.DataFrame:
    # Pit stops cluster: when several cars pit on the same lap, the rest of the
    # field reacts on the next lap. Compute a leave-one-out share so the row's
    # own PitStop doesn't tautologically inflate its feature, plus lagged and
    # 3-lap windowed versions to capture "the field has been pitting for a few
    # laps now" patterns.
    field_pit = df.group_by(["Race", "Year", "LapNumber"]).agg(
        pl.col("PitStop").sum().alias("__pit_sum_lap"),
        pl.len().alias("__n_lap"),
    )
    df = df.join(field_pit, on=["Race", "Year", "LapNumber"], how="left").with_columns(
        pl.when(pl.col("__n_lap") > 1)
        .then(
            (pl.col("__pit_sum_lap") - pl.col("PitStop"))
            / (pl.col("__n_lap") - 1)
        )
        .otherwise(0.0)
        .alias("field_pit_share")
    )

    # Lag/window versions: dedupe to per-lap, shift, rolling-mean, then join back.
    pit_per_lap = (
        df.select(["Race", "Year", "LapNumber", "__pit_sum_lap", "__n_lap"])
        .unique(["Race", "Year", "LapNumber"])
        .with_columns(
            (pl.col("__pit_sum_lap") / pl.col("__n_lap")).alias("__field_pit_share_raw")
        )
        .sort(["Race", "Year", "LapNumber"])
    )
    pit_per_lap = pit_per_lap.with_columns(
        pl.col("__field_pit_share_raw")
        .shift(1)
        .over(["Race", "Year"])
        .alias("field_pit_share_prev_lap"),
        pl.col("__field_pit_share_raw")
        .shift(2)
        .over(["Race", "Year"])
        .alias("field_pit_share_lap_minus_2"),
        pl.col("__field_pit_share_raw")
        .rolling_mean(window_size=3, min_samples=1)
        .over(["Race", "Year"])
        .alias("field_pit_share_window_3"),
    ).select(
        "Race",
        "Year",
        "LapNumber",
        "field_pit_share_prev_lap",
        "field_pit_share_lap_minus_2",
        "field_pit_share_window_3",
    )

    return df.join(pit_per_lap, on=["Race", "Year", "LapNumber"], how="left").drop(
        "__pit_sum_lap", "__n_lap"
    )


def add_pit_progress(df: pl.DataFrame) -> pl.DataFrame:
    # pits_so_far_this_race: cumulative count of PitStop within (Driver, Race, Year),
    # ordered by lap. Includes the current lap (PitStop is observed for this row).
    df = df.sort(["Race", "Year", "Driver", "LapNumber"]).with_columns(
        pl.col("PitStop")
        .cum_sum()
        .over(["Race", "Year", "Driver"])
        .alias("pits_so_far_this_race")
    )

    # expected_stops_for_race: median total stops per driver in this (Race, Year).
    # Captures "this is usually a 1-stopper" vs "this is a 2-stopper", so the
    # model can read pits_so_far against the typical strategy. 2023's near-zero
    # stop rate naturally collapses this to ~0 there, which is the right signal.
    driver_totals = df.group_by(["Race", "Year", "Driver"]).agg(
        pl.col("PitStop").sum().alias("__driver_total_stops")
    )
    expected = driver_totals.group_by(["Race", "Year"]).agg(
        pl.col("__driver_total_stops").median().alias("expected_stops_for_race")
    )

    return df.join(expected, on=["Race", "Year"], how="left").with_columns(
        (pl.col("expected_stops_for_race") - pl.col("pits_so_far_this_race"))
        .alias("stops_remaining_proxy")
    )


def add_undercut_signals(df: pl.DataFrame) -> pl.DataFrame:
    # Positional pit signal: pit-stop activity at the Position immediately
    # ahead/behind in the same lap. In the synthetic dataset many "drivers"
    # share the same Position within a lap (Position is bounded 1-20 but ~70
    # drivers populate each lap), so we aggregate by mean across same-Position
    # rows before looking up the neighbouring Position. The result is a soft
    # version of the classic undercut/overcut trigger.
    df = df.sort(["Race", "Year", "Driver", "LapNumber"]).with_columns(
        pl.col("PitStop")
        .rolling_sum(window_size=3, min_samples=1)
        .over(["Race", "Year", "Driver"])
        .alias("__pitstop_last_3"),
        pl.col("PitStop")
        .rolling_sum(window_size=5, min_samples=1)
        .over(["Race", "Year", "Driver"])
        .alias("__pitstop_last_5"),
    )

    # One row per (Race, Year, LapNumber, Position) — kills the row-multiplying
    # join that occurs when Position is non-unique within a lap.
    pos_lookup = df.group_by(["Race", "Year", "LapNumber", "Position"]).agg(
        pl.col("__pitstop_last_3").mean().alias("__last_3_pos"),
        pl.col("__pitstop_last_5").mean().alias("__last_5_pos"),
    )

    # AHEAD: a row at Position=P wants the value at Position=P-1. Shift the
    # lookup's Position UP by 1 so the row at original Position=P becomes
    # joinable as Position=P+1, which a driver at P+1 then picks up as ahead.
    ahead = pos_lookup.select(
        pl.col("Race"),
        pl.col("Year"),
        pl.col("LapNumber"),
        (pl.col("Position") + 1).alias("Position"),
        pl.col("__last_3_pos").alias("ahead_pitted_last_3"),
        pl.col("__last_5_pos").alias("ahead_pitted_last_5"),
    )
    # BEHIND: mirror — shift Position DOWN by 1.
    behind = pos_lookup.select(
        pl.col("Race"),
        pl.col("Year"),
        pl.col("LapNumber"),
        (pl.col("Position") - 1).alias("Position"),
        pl.col("__last_3_pos").alias("behind_pitted_last_3"),
        pl.col("__last_5_pos").alias("behind_pitted_last_5"),
    )
    df = df.join(ahead, on=["Race", "Year", "LapNumber", "Position"], how="left")
    df = df.join(behind, on=["Race", "Year", "LapNumber", "Position"], how="left")
    return df.drop("__pitstop_last_3", "__pitstop_last_5")


def add_stint_length_features(df: pl.DataFrame) -> pl.DataFrame:
    # Compute completed-stint lengths once, then derive typical stint per
    # (Compound, Race), max viable life per Compound, and previous stint length
    # per driver-race.
    stint_lengths = (
        df.group_by(["Driver", "Race", "Year", "Stint"])
        .agg(
            pl.col("LapNumber").count().alias("__stint_length"),
            pl.col("PitStop").sum().alias("__stint_ended_in_pit"),
            pl.col("Compound").first().alias("__compound"),
        )
        .filter(pl.col("__stint_ended_in_pit") >= 1)
    )

    # Typical (median) completed stint length per (Compound, Race).
    typical = (
        stint_lengths.group_by(["__compound", "Race"])
        .agg(pl.col("__stint_length").median().alias("typical_stint_length"))
        .rename({"__compound": "Compound"})
    )
    df = df.join(typical, on=["Compound", "Race"], how="left")

    # Max viable life per Compound = p95 of completed-stint lengths.
    compound_max = (
        stint_lengths.group_by("__compound")
        .agg(pl.col("__stint_length").quantile(0.95).alias("compound_max_life"))
        .rename({"__compound": "Compound"})
    )
    df = df.join(compound_max, on="Compound", how="left").with_columns(
        (pl.col("TyreLife") / pl.col("typical_stint_length")).alias("tyre_life_norm"),
        (pl.col("compound_max_life") - pl.col("TyreLife")).alias("tyre_life_remaining"),
        (pl.col("LapsRemaining") > (pl.col("compound_max_life") - pl.col("TyreLife")))
        .cast(pl.Int8)
        .alias("cant_finish_on_current_tyres"),
    )

    # Previous stint length: for current Stint S, length of S-1 in the same
    # (Driver, Race, Year). Captures driver/team strategy consistency.
    prev_stint = stint_lengths.select(
        pl.col("Driver"),
        pl.col("Race"),
        pl.col("Year"),
        (pl.col("Stint") + 1).alias("Stint"),
        pl.col("__stint_length").alias("prev_stint_length"),
    )
    df = df.join(prev_stint, on=["Driver", "Race", "Year", "Stint"], how="left")
    return df


def add_pit_window(df: pl.DataFrame) -> pl.DataFrame:
    # Empirical pit-window per Race: 5th–95th percentile of laps where PitStop=1.
    # Most pits cluster inside this band; lap 1 and the final laps are rare.
    pit_window = (
        df.filter(pl.col("PitStop") == 1)
        .group_by("Race")
        .agg(
            pl.col("LapNumber").quantile(0.05).alias("pit_window_start"),
            pl.col("LapNumber").quantile(0.95).alias("pit_window_end"),
        )
    )
    return df.join(pit_window, on="Race", how="left").with_columns(
        (
            (pl.col("LapNumber") >= pl.col("pit_window_start"))
            & (pl.col("LapNumber") <= pl.col("pit_window_end"))
        )
        .cast(pl.Int8)
        .alias("in_pit_window")
    )


def add_per_driver_pace(df: pl.DataFrame) -> pl.DataFrame:
    # How this driver's lap compares to the field median this lap. Captures
    # "this car is dropping off" without needing the timeline to be sorted.
    return df.with_columns(
        (pl.col("LapTime (s)") / pl.col("field_median_laptime")).alias("driver_pace_ratio")
    )


def add_timeline_features(df: pl.DataFrame) -> pl.DataFrame:
    # Per (Driver, Race, Year) timeline: rolling mean/std, lapped diffs,
    # position deltas, degradation rate. Also a per-stint "stint_start_pace"
    # (mean LapTime over the first 3 laps of the stint) and the row's deviation
    # from it — direct degradation signal that doesn't depend on the
    # potentially-buggy Cumulative_Degradation column.
    df = df.sort(["Race", "Year", "Driver", "LapNumber"]).with_columns(
        pl.col("LapTime (s)")
        .rolling_mean(window_size=3, min_samples=1)
        .over(["Race", "Year", "Driver"])
        .alias("laptime_roll_mean_3"),
        pl.col("LapTime (s)")
        .rolling_mean(window_size=5, min_samples=1)
        .over(["Race", "Year", "Driver"])
        .alias("laptime_roll_mean_5"),
        pl.col("LapTime (s)")
        .rolling_std(window_size=3, min_samples=2)
        .over(["Race", "Year", "Driver"])
        .alias("laptime_roll_std_3"),
        pl.col("LapTime (s)")
        .rolling_std(window_size=5, min_samples=2)
        .over(["Race", "Year", "Driver"])
        .alias("laptime_roll_std_5"),
        (
            pl.col("LapTime (s)")
            - pl.col("LapTime (s)").shift(1).over(["Race", "Year", "Driver"])
        ).alias("laptime_diff_1"),
        (
            pl.col("LapTime (s)")
            - pl.col("LapTime (s)").shift(3).over(["Race", "Year", "Driver"])
        ).alias("laptime_diff_3"),
        (
            pl.col("Position")
            - pl.col("Position").shift(3).over(["Race", "Year", "Driver"])
        ).alias("position_change_3"),
        (
            pl.col("Position")
            - pl.col("Position").shift(5).over(["Race", "Year", "Driver"])
        ).alias("position_change_5"),
        pl.col("Position")
        .cast(pl.Float64)
        .rolling_std(window_size=5, min_samples=2)
        .over(["Race", "Year", "Driver"])
        .alias("position_roll_std_5"),
        (
            pl.col("Cumulative_Degradation")
            - pl.col("Cumulative_Degradation")
            .shift(1)
            .over(["Race", "Year", "Driver"])
        ).alias("cum_deg_diff_1"),
    )

    stint_baseline = (
        df.sort(["Race", "Year", "Driver", "Stint", "LapNumber"])
        .with_columns(
            pl.int_range(pl.len())
            .over(["Race", "Year", "Driver", "Stint"])
            .alias("__stint_lap_idx")
        )
        .filter(pl.col("__stint_lap_idx") < 3)
        .group_by(["Race", "Year", "Driver", "Stint"])
        .agg(pl.col("LapTime (s)").mean().alias("stint_start_pace"))
    )
    df = df.join(
        stint_baseline, on=["Race", "Year", "Driver", "Stint"], how="left"
    ).with_columns(
        (pl.col("LapTime (s)") - pl.col("stint_start_pace")).alias(
            "laptime_vs_stint_start"
        )
    )

    # Pace-acceleration features (cycle #006). Second derivative of pace.
    # `laptime_diff_1` is velocity (lap-to-lap pace change). The derivative of
    # that — change in degradation rate — fires the lap a pace-cliff begins,
    # not 3 laps later when the cohort-average has already moved. Avoids the
    # selection-bias trap that sank cycle #003's quintile-style features.
    return df.with_columns(
        (
            pl.col("laptime_diff_1")
            - pl.col("laptime_diff_1").shift(1).over(["Race", "Year", "Driver"])
        ).alias("laptime_accel_3"),
    ).with_columns(
        pl.col("laptime_accel_3")
        .rolling_mean(window_size=3, min_samples=2)
        .over(["Race", "Year", "Driver"])
        .alias("laptime_accel_roll_3"),
    )


def add_compound_peer_features(df: pl.DataFrame) -> pl.DataFrame:
    # Cycle #006: percentile rank of TyreLife within (Race, Year, LapNumber,
    # Compound). "Of all drivers on this compound this lap, where is my tyre
    # age?" — captures relative wear in the peer cohort, which `tyre_life_norm`
    # (normed against typical_stint_length) doesn't. Peers near p95 are the
    # next-in-line to pit regardless of absolute TyreLife. Computed across
    # train + test combined, no labels involved.
    return df.with_columns(
        (
            pl.col("TyreLife").rank(method="average")
            / pl.len()
        )
        .over(["Race", "Year", "LapNumber", "Compound"])
        .alias("tyre_age_pct_among_compound_peers")
    )


def add_cheap_flags(df: pl.DataFrame) -> pl.DataFrame:
    # Synthetic-driver pattern: codes like D109/D552 (letter D + 3 digits).
    return df.with_columns(
        (pl.col("Race") == "Pre-Season Testing").cast(pl.Int8).alias("is_pre_season"),
        (
            pl.col("Driver").str.starts_with("D")
            & (pl.col("Driver").str.len_chars() == 4)
        )
        .cast(pl.Int8)
        .alias("is_synthetic_driver"),
        (pl.col("Year") == 2023).cast(pl.Int8).alias("is_2023"),
        (pl.col("LapTime (s)") > pl.col("field_median_laptime") * 1.5)
        .cast(pl.Int8)
        .alias("lap_is_anomalous"),
    )


def main() -> None:
    df = load_combined()
    n_train = df.filter(pl.col("__split") == "train").height
    n_test = df.filter(pl.col("__split") == "test").height
    print(f"loaded {df.height:,} rows ({n_train:,} train, {n_test:,} test)")

    df = add_race_level(df)
    df = add_field_lap_features(df)
    df = add_sc_features(df)
    df = add_field_pit_cluster(df)
    df = add_pit_progress(df)
    df = add_undercut_signals(df)
    df = add_stint_length_features(df)
    df = add_pit_window(df)
    df = add_per_driver_pace(df)
    df = add_timeline_features(df)
    df = add_compound_peer_features(df)
    df = add_cheap_flags(df)

    # Drop scratch columns and split back.
    df = df.drop("__baseline_pace")
    train = df.filter(pl.col("__split") == "train").drop("__split")
    test = df.filter(pl.col("__split") == "test").drop("__split", "PitNextLap")

    train.write_parquet(OUT_TRAIN)
    test.write_parquet(OUT_TEST)
    print(f"wrote {OUT_TRAIN.name}  ({train.height:,} rows, {train.width} cols)")
    print(f"wrote {OUT_TEST.name}   ({test.height:,} rows, {test.width} cols)")

    # Sanity report.
    new_cols = [
        "pit_cost_seconds",
        "total_laps_year",
        "LapsRemaining",
        "is_wet_race",
        "field_median_laptime",
        "field_max_laptime",
        "wet_compound_share",
        "field_pace_ratio",
        "sc_likely",
        "sc_prev_lap",
        "sc_lap_minus2",
        "laps_since_sc",
        "field_pit_share",
        "field_pit_share_prev_lap",
        "field_pit_share_lap_minus_2",
        "field_pit_share_window_3",
        "pits_so_far_this_race",
        "expected_stops_for_race",
        "stops_remaining_proxy",
        "ahead_pitted_last_3",
        "ahead_pitted_last_5",
        "behind_pitted_last_3",
        "behind_pitted_last_5",
        "typical_stint_length",
        "compound_max_life",
        "tyre_life_norm",
        "tyre_life_remaining",
        "cant_finish_on_current_tyres",
        "prev_stint_length",
        "pit_window_start",
        "pit_window_end",
        "in_pit_window",
        "driver_pace_ratio",
        "laptime_roll_mean_3",
        "laptime_roll_mean_5",
        "laptime_roll_std_3",
        "laptime_roll_std_5",
        "laptime_diff_1",
        "laptime_diff_3",
        "position_change_3",
        "position_change_5",
        "position_roll_std_5",
        "cum_deg_diff_1",
        "stint_start_pace",
        "laptime_vs_stint_start",
        "is_pre_season",
        "is_synthetic_driver",
        "is_2023",
        "lap_is_anomalous",
    ]
    print("\n--- new feature stats (train) ---")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(train.select(new_cols).describe())

    print("\n--- pit_cost_seconds by Race (top + bottom 5) ---")
    pit_cost_ranking = (
        train.select("Race", "pit_cost_seconds").unique().sort("pit_cost_seconds")
    )
    print(pit_cost_ranking.head(5))
    print(pit_cost_ranking.tail(5))

    print("\n--- sc_likely fire rate by Race (top 5) ---")
    sc_rate = (
        train.group_by("Race")
        .agg(pl.col("sc_likely").mean().alias("sc_share"))
        .sort("sc_share", descending=True)
        .head(5)
    )
    print(sc_rate)

    print("\n--- target rate when sc_likely=1 vs 0 (train) ---")
    print(
        train.group_by("sc_likely").agg(
            pl.col("PitNextLap").mean().alias("pit_next_rate"),
            pl.len().alias("n"),
        ).sort("sc_likely")
    )

    print("\n--- target rate by field_pit_share quintile (train) ---")
    print(
        train.with_columns(
            pl.col("field_pit_share")
            .qcut(5, labels=["q1", "q2", "q3", "q4", "q5"])
            .alias("__bucket")
        )
        .group_by("__bucket")
        .agg(
            pl.col("PitNextLap").mean().alias("pit_next_rate"),
            pl.col("field_pit_share").mean().alias("share_mean"),
            pl.len().alias("n"),
        )
        .sort("share_mean")
    )

    print("\n--- target rate vs stops_remaining_proxy (train, excl. 2023) ---")
    print(
        train.filter(pl.col("Year") != 2023)
        .group_by("stops_remaining_proxy")
        .agg(
            pl.col("PitNextLap").mean().alias("pit_next_rate"),
            pl.len().alias("n"),
        )
        .sort("stops_remaining_proxy")
    )

    print("\n--- cheap-flag fire rates (train) ---")
    print(
        train.select(
            pl.col("is_pre_season").mean().alias("is_pre_season"),
            pl.col("is_synthetic_driver").mean().alias("is_synthetic_driver"),
            pl.col("is_2023").mean().alias("is_2023"),
            pl.col("lap_is_anomalous").mean().alias("lap_is_anomalous"),
        )
    )

    print("\n--- target rate when ahead/behind pitted in last 3 laps (train) ---")
    print(
        train.group_by(
            (pl.col("ahead_pitted_last_3").fill_null(0) > 0)
            .cast(pl.Int8)
            .alias("ahead_pitted_recent"),
            (pl.col("behind_pitted_last_3").fill_null(0) > 0)
            .cast(pl.Int8)
            .alias("behind_pitted_recent"),
        )
        .agg(
            pl.col("PitNextLap").mean().alias("pit_next_rate"),
            pl.len().alias("n"),
        )
        .sort("ahead_pitted_recent", "behind_pitted_recent")
    )

    print("\n--- target rate vs cant_finish_on_current_tyres (train) ---")
    print(
        train.group_by("cant_finish_on_current_tyres").agg(
            pl.col("PitNextLap").mean().alias("pit_next_rate"),
            pl.len().alias("n"),
        ).sort("cant_finish_on_current_tyres")
    )

    print("\n--- target rate vs in_pit_window (train) ---")
    print(
        train.group_by("in_pit_window").agg(
            pl.col("PitNextLap").mean().alias("pit_next_rate"),
            pl.len().alias("n"),
        ).sort("in_pit_window")
    )

    print("\n--- target rate by tyre_life_norm bucket (train) ---")
    print(
        train.with_columns(
            pl.col("tyre_life_norm")
            .qcut(5, labels=["q1", "q2", "q3", "q4", "q5"], allow_duplicates=True)
            .alias("__bucket")
        )
        .group_by("__bucket")
        .agg(
            pl.col("PitNextLap").mean().alias("pit_next_rate"),
            pl.col("tyre_life_norm").mean().alias("norm_mean"),
            pl.len().alias("n"),
        )
        .sort("norm_mean")
    )


if __name__ == "__main__":
    main()
