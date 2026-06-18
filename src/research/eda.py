"""Exploratory data analysis for the Kaggle Playground S6E5 F1 pit-stop dataset.

Reproduces the analytical findings documented in docs/eda.md. Produces tables
and summary statistics to stdout; the visual companion lives in
notebooks/eda.ipynb.

Sections:
    1. Files & schema (rows, dtypes, missingness, value cardinalities)
    2. Target distribution
    3. Year distribution & the 2023 anomaly
    4. Compound / race / driver breakdowns
    5. Numerical distributions & outliers
    6. Correlations with the target
    7. Pit dynamics: TyreLife (× Compound), RaceProgress, Position
    8. Train/test split structure (row-level vs group-level)
    9. Documented-identity checks (LapTime_Delta, Position_Change,
       PitNextLap vs next-row PitStop) — see docs/feature_engineering.md §6

Run: .venv/bin/python src/eda.py
"""

from pathlib import Path

import polars as pl

DATA = Path(__file__).resolve().parent.parent.parent / "data"

COMPOUNDS = ["HARD", "MEDIUM", "SOFT", "INTERMEDIATE", "WET"]
TYRE_LIFE_BREAKS = [0, 5, 10, 15, 20, 25, 30, 40, 50, 70]
TYRE_LIFE_BREAKS_COARSE = [0, 5, 10, 15, 20, 30, 50]
RACE_PROGRESS_BREAKS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    train = pl.read_csv(DATA / "train.csv")
    test = pl.read_csv(DATA / "test.csv")
    sub = pl.read_csv(DATA / "sample_submission.csv")
    return train, test, sub


def section_1_schema(train: pl.DataFrame, test: pl.DataFrame, sub: pl.DataFrame) -> None:
    banner("1. Files & schema")
    print(f"train: {train.shape}")
    print(f"test : {test.shape}")
    print(f"sub  : {sub.shape}")
    print(f"\ncolumns only in train: {set(train.columns) - set(test.columns)}")
    print(f"columns only in test : {set(test.columns) - set(train.columns)}")

    schema = pl.DataFrame(
        {
            "column": train.columns,
            "dtype": [str(d) for d in train.dtypes],
            "in_test": [c in test.columns for c in train.columns],
            "n_unique_train": [train[c].n_unique() for c in train.columns],
            "nulls_train": [train[c].null_count() for c in train.columns],
            "nulls_test": [
                test[c].null_count() if c in test.columns else None
                for c in train.columns
            ],
        }
    )
    print("\nschema:")
    with pl.Config(tbl_rows=20, tbl_width_chars=120):
        print(schema)


def section_2_target(train: pl.DataFrame) -> None:
    banner("2. Target distribution")
    pos = int(train["PitNextLap"].sum())
    neg = train.height - pos
    print(f"PitNextLap=1: {pos:,} ({pos / train.height:.4%})")
    print(f"PitNextLap=0: {neg:,} ({neg / train.height:.4%})")


def section_3_year(train: pl.DataFrame, test: pl.DataFrame) -> None:
    banner("3. Year distribution & the 2023 anomaly")
    year_stats = (
        train.group_by("Year")
        .agg(
            pl.len().alias("n_train"),
            pl.col("PitStop").mean().alias("PitStop_rate"),
            pl.col("PitNextLap").mean().alias("PitNextLap_rate"),
        )
        .sort("Year")
    )
    test_year = test.group_by("Year").agg(pl.len().alias("n_test")).sort("Year")
    year_stats = year_stats.join(test_year, on="Year", how="left")
    print(year_stats)

    # Confirm 2023 collapses across every compound, not just one.
    print("\nPitNextLap rate by Year × Compound (2023 should be uniformly tiny):")
    yc = (
        train.group_by(["Year", "Compound"])
        .agg(pl.col("PitNextLap").mean().alias("rate"), pl.len().alias("n"))
        .sort(["Year", "Compound"])
    )
    print(yc)


def section_4_categoricals(train: pl.DataFrame, test: pl.DataFrame) -> None:
    banner("4. Categorical breakdowns")

    print("--- Compound ---")
    print(
        train.group_by("Compound")
        .agg(
            pl.len().alias("n_rows"),
            pl.col("PitNextLap").mean().alias("PitNextLap_rate"),
        )
        .sort("PitNextLap_rate", descending=True)
    )

    print("\n--- Race set comparison ---")
    tr_races = set(train["Race"].unique().to_list())
    te_races = set(test["Race"].unique().to_list())
    print(f"races in train: {len(tr_races)} | races in test: {len(te_races)}")
    print(f"train-only races: {tr_races - te_races}")
    print(f"test-only races : {te_races - tr_races}")
    print("\nrace row counts (train):")
    print(
        train.group_by("Race")
        .agg(pl.len().alias("n_rows"), pl.col("PitNextLap").mean().alias("rate"))
        .sort("n_rows", descending=True)
    )

    print("\n--- Driver overlap ---")
    tr_drivers = set(train["Driver"].unique().to_list())
    te_drivers = set(test["Driver"].unique().to_list())
    print(f"drivers in train: {len(tr_drivers)}")
    print(f"drivers in test : {len(te_drivers)}")
    print(f"test drivers absent from train: {len(te_drivers - tr_drivers)}")
    print(f"train-only drivers: {len(tr_drivers - te_drivers)}")

    # Synthetic driver heuristic: 'D' + 3 digits.
    synthetic = train.with_columns(
        (
            pl.col("Driver").str.starts_with("D")
            & (pl.col("Driver").str.len_chars() == 4)
        ).alias("is_synthetic_id")
    )
    print("\nsynthetic vs real driver IDs:")
    print(
        synthetic.group_by("is_synthetic_id").agg(
            pl.len().alias("rows"),
            pl.col("Driver").n_unique().alias("unique_drivers"),
            pl.col("PitNextLap").mean().alias("rate"),
        )
    )


def section_5_numerical(train: pl.DataFrame, test: pl.DataFrame) -> None:
    banner("5. Numerical distributions & outliers")
    num_cols = [
        c
        for c, d in zip(train.columns, train.dtypes)
        if d.is_numeric() and c != "id"
    ]
    print("describe (train):")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(train.select(num_cols).describe())

    print("\nextreme-tail percentiles for the heavy-tailed columns:")
    for col in ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation"]:
        q = train.select(
            [
                pl.col(col).min().alias("min"),
                pl.col(col).quantile(0.001).alias("p0.1"),
                pl.col(col).quantile(0.01).alias("p1"),
                pl.col(col).quantile(0.5).alias("p50"),
                pl.col(col).quantile(0.99).alias("p99"),
                pl.col(col).quantile(0.999).alias("p99.9"),
                pl.col(col).max().alias("max"),
            ]
        )
        print(f"\n{col}\n{q}")

    # Train vs test distribution sanity (means & p50 for the structural columns).
    print("\ntrain vs test means (structural columns):")
    cmp_cols = ["LapNumber", "Stint", "TyreLife", "Position", "LapTime (s)", "RaceProgress"]
    rows = []
    for col in cmp_cols:
        rows.append(
            {
                "column": col,
                "train_mean": float(train[col].mean()),
                "test_mean": float(test[col].mean()),
                "train_p50": float(train[col].median()),
                "test_p50": float(test[col].median()),
            }
        )
    print(pl.DataFrame(rows))


def section_6_correlation(train: pl.DataFrame) -> None:
    banner("6. Correlations with the target")
    num_cols = [
        c
        for c, d in zip(train.columns, train.dtypes)
        if d.is_numeric() and c not in ("id", "PitNextLap")
    ]
    rows = []
    for col in num_cols:
        rows.append(
            {
                "column": col,
                "corr_with_target": float(
                    train.select(pl.corr(col, "PitNextLap")).item()
                ),
            }
        )
    corr = pl.DataFrame(rows).with_columns(pl.col("corr_with_target").abs().alias("abs_corr"))
    print(corr.sort("abs_corr", descending=True).drop("abs_corr"))

    # Notable pairwise correlations highlighted in eda.md.
    print("\nnotable pairwise correlations:")
    for a, b in [
        ("LapNumber", "RaceProgress"),
        ("LapNumber", "Stint"),
        ("LapNumber", "TyreLife"),
        ("Position", "Position_Change"),
        ("LapTime (s)", "Cumulative_Degradation"),
    ]:
        v = float(train.select(pl.corr(a, b)).item())
        print(f"  {a:<25s} ↔ {b:<25s}  {v:+.4f}")


def section_7_pit_dynamics(train: pl.DataFrame) -> None:
    banner("7. Pit dynamics — TyreLife, Compound, RaceProgress, Position")

    print("--- PitNextLap rate vs TyreLife bucket ---")
    print(
        train.with_columns(
            pl.col("TyreLife").cut(TYRE_LIFE_BREAKS).alias("life_bucket")
        )
        .group_by("life_bucket")
        .agg(pl.col("PitNextLap").mean().alias("rate"), pl.len().alias("n"))
        .sort("life_bucket")
    )

    print("\n--- PitNextLap rate by TyreLife × Compound ---")
    tc = (
        train.with_columns(
            pl.col("TyreLife").cut(TYRE_LIFE_BREAKS_COARSE).alias("life_bucket")
        )
        .group_by(["Compound", "life_bucket"])
        .agg(pl.col("PitNextLap").mean().alias("rate"), pl.len().alias("n"))
        .sort(["Compound", "life_bucket"])
    )
    with pl.Config(tbl_rows=60):
        print(tc)

    print("\n--- PitNextLap rate vs RaceProgress (classic pit window) ---")
    print(
        train.with_columns(
            pl.col("RaceProgress").cut(RACE_PROGRESS_BREAKS).alias("rp_bucket")
        )
        .group_by("rp_bucket")
        .agg(pl.col("PitNextLap").mean().alias("rate"), pl.len().alias("n"))
        .sort("rp_bucket")
    )

    print("\n--- PitNextLap rate by Position (should be ~flat) ---")
    print(
        train.group_by("Position")
        .agg(pl.col("PitNextLap").mean().alias("rate"), pl.len().alias("n"))
        .sort("Position")
    )


def section_8_split_structure(train: pl.DataFrame, test: pl.DataFrame) -> None:
    banner("8. Train/test split structure")
    key_cols = ["Race", "Year", "Driver"]
    tr_keys = set(train.select(key_cols).unique().iter_rows())
    te_keys = set(test.select(key_cols).unique().iter_rows())
    overlap = tr_keys & te_keys
    print(f"train (Race, Year, Driver) groups: {len(tr_keys):,}")
    print(f"test  (Race, Year, Driver) groups: {len(te_keys):,}")
    print(f"overlap                          : {len(overlap):,}")
    print(f"train-only                       : {len(tr_keys - te_keys):,}")
    print(f"test-only                        : {len(te_keys - tr_keys):,}")

    # Sample 5 overlapping groups; expect disjoint LapNumber sets in train vs test.
    print("\nsample of overlapping groups (laps disjoint, no shared LapNumber):")
    for r, y, d in list(overlap)[:5]:
        tr_laps = (
            train.filter(
                (pl.col("Race") == r)
                & (pl.col("Year") == y)
                & (pl.col("Driver") == d)
            )["LapNumber"]
            .sort()
            .to_list()
        )
        te_laps = (
            test.filter(
                (pl.col("Race") == r)
                & (pl.col("Year") == y)
                & (pl.col("Driver") == d)
            )["LapNumber"]
            .sort()
            .to_list()
        )
        shared = sorted(set(tr_laps) & set(te_laps))
        print(
            f"  ({r}, {y}, {d}) train_n={len(tr_laps)} test_n={len(te_laps)} shared_laps={shared}"
        )


def section_9_identity_checks(train: pl.DataFrame, test: pl.DataFrame) -> None:
    banner("9. Documented-identity checks")
    # Combine train+test to maximise the rows where adjacent laps exist.
    combined = pl.concat(
        [
            train.with_columns(pl.lit("train").alias("__split")),
            test.with_columns(
                pl.lit("test").alias("__split"),
                pl.lit(None).cast(pl.Float64).alias("PitNextLap"),
            ),
        ],
        how="diagonal_relaxed",
    ).sort(["Race", "Year", "Driver", "LapNumber"])

    df = combined.with_columns(
        pl.col("LapNumber").shift(1).over(["Race", "Year", "Driver"]).alias("__prev_lap"),
        pl.col("LapTime (s)").shift(1).over(["Race", "Year", "Driver"]).alias("__prev_LapTime"),
        pl.col("Position").shift(1).over(["Race", "Year", "Driver"]).alias("__prev_Position"),
        pl.col("LapNumber").shift(-1).over(["Race", "Year", "Driver"]).alias("__next_lap"),
        pl.col("PitStop").shift(-1).over(["Race", "Year", "Driver"]).alias("__next_PitStop"),
    ).with_columns(
        (pl.col("LapNumber") - pl.col("__prev_lap") == 1).alias("__has_prev"),
        (pl.col("__next_lap") - pl.col("LapNumber") == 1).alias("__has_next"),
    )

    # Claim 1: LapTime_Delta == LapTime - LapTime[i-1] when PitStop != 1
    c1 = df.filter(pl.col("__has_prev") & (pl.col("PitStop") != 1)).with_columns(
        (pl.col("LapTime_Delta") - (pl.col("LapTime (s)") - pl.col("__prev_LapTime")))
        .abs()
        .alias("__diff")
    )
    eligible = c1.height
    within_1ms = c1.filter(pl.col("__diff") < 1e-3).height
    within_1s = c1.filter(pl.col("__diff") < 1.0).height
    big = c1.filter(pl.col("__diff") > 5.0).height
    print(
        "Claim 1: LapTime_Delta = LapTime - LapTime_prev (PitStop != 1)\n"
        f"  eligible rows: {eligible:,}\n"
        f"  match within 1 ms: {within_1ms:,} ({100 * within_1ms / eligible:.2f}%)\n"
        f"  match within 1 s : {within_1s:,} ({100 * within_1s / eligible:.2f}%)\n"
        f"  diff > 5 s        : {big:,} ({100 * big / eligible:.2f}%)"
    )

    # Claim 2: Position_Change == Position[i-1] - Position
    c2 = df.filter(pl.col("__has_prev")).with_columns(
        (pl.col("Position_Change") - (pl.col("__prev_Position") - pl.col("Position")))
        .abs()
        .alias("__diff")
    )
    eligible = c2.height
    exact = c2.filter(pl.col("__diff") == 0).height
    print(
        "\nClaim 2: Position_Change = Position_prev - Position\n"
        f"  eligible rows: {eligible:,}\n"
        f"  exact match  : {exact:,} ({100 * exact / eligible:.2f}%)"
    )

    # Claim 3: PitNextLap[i] = 1 iff PitStop[i+1] = 1 (train rows with next lap)
    c3 = df.filter((pl.col("__split") == "train") & pl.col("__has_next"))
    eligible = c3.height
    agree = c3.filter(pl.col("PitNextLap") == pl.col("__next_PitStop")).height
    xtab = c3.group_by(["PitNextLap", "__next_PitStop"]).len().sort(
        ["PitNextLap", "__next_PitStop"]
    )
    print(
        "\nClaim 3: PitNextLap[i] = 1  iff  PitStop[i+1] = 1 (train, next-lap available)\n"
        f"  eligible rows: {eligible:,}\n"
        f"  agreement    : {agree:,} ({100 * agree / eligible:.2f}%)"
    )
    print(xtab)


def main() -> None:
    train, test, sub = load()
    section_1_schema(train, test, sub)
    section_2_target(train)
    section_3_year(train, test)
    section_4_categoricals(train, test)
    section_5_numerical(train, test)
    section_6_correlation(train)
    section_7_pit_dynamics(train)
    section_8_split_structure(train, test)
    section_9_identity_checks(train, test)


if __name__ == "__main__":
    main()
