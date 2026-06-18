"""The data pipeline for the best submitted model (primary blend, private LB 0.95427).

This is the single, readable answer to *"what data trained the model, and how was it
built?"* — the loading + feature engineering that used to be copy-pasted across every
trainer now lives here once.

Raw inputs (fetch with `scripts/get_data.sh`; see REPRODUCE.md §0):

    data/train.csv, data/test.csv      competition data — the labelled task
    data/f1_strategy_dataset.csv     external augmentation — RAW ROWS ONLY (+101k laps)

The external set contributes **rows, not features**: every engineered column below is
built here, not taken from the file. It joins **training folds only** — validation and
test stay competition-only, so cross-validation remains an honest proxy for the LB
(leak-checked, see docs/dataset_review_no_leak.md).

The blend mixes two model families, and they use two faithful-to-submission recipes:

    build_gbdt_diffFE()   XGBoost / CatBoost bases — domain features + base-cat frequency,
                          unified categorical dtypes. Cross-cats and group-stats are
                          deliberately dropped ("diffFE" = differentiated/stripped FE,
                          which made each GBDT stronger and the blend more transfer-robust).
    realmlp_features()    RealMLP bases — ratio features, count encodings, KBins quantile
                          bins, and target-encoded Race×Compound / Race×Year combos
                          (target encoding is applied per-fold by the trainer, not here).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

DATA = Path(__file__).resolve().parent.parent / "data"
TRAIN_CSV = DATA / "train.csv"
TEST_CSV = DATA / "test.csv"
EXTERNAL_CSV = DATA / "f1_strategy_dataset.csv"

TARGET = "PitNextLap"
ID_COL = "id"
BASE_CATS = ["Driver", "Race", "Compound"]


# ---------------------------------------------------------------------------
# Raw loading
# ---------------------------------------------------------------------------
def load_competition() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The labelled task: (train, test) straight from the competition CSVs."""
    return pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV)


def load_external(*, dropna_compound: bool) -> pd.DataFrame:
    """External augmentation rows. `Normalized_TyreLife` is always dropped (it is the
    host's leaky forward-looking column, absent from the competition data). The GBDT
    recipe additionally drops the handful of NaN-Compound rows; RealMLP keeps them —
    both behaviours are preserved exactly as the submitted models used them."""
    ext = pd.read_csv(EXTERNAL_CSV)
    if dropna_compound:
        ext = ext.dropna(subset=["Compound"])
    return ext.drop(columns=["Normalized_TyreLife"], errors="ignore")


# ---------------------------------------------------------------------------
# GBDT (XGBoost / CatBoost) feature recipe — verbatim from the diffFE trainers
# ---------------------------------------------------------------------------
def safe_div(a, b, eps=1e-6):
    return a / (b + eps)


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cycle-12 inline FE. Same recipes as the 0.95259 public notebook."""
    eps = 1e-6
    out = df.copy()

    # Race progress / lap geometry
    race_progress = out["RaceProgress"].clip(lower=eps)
    out["EstimatedTotalLaps"] = (out["LapNumber"] / race_progress).clip(1, 120).astype("float32")
    out["LapsRemaining"] = (out["EstimatedTotalLaps"] - out["LapNumber"]).clip(lower=0).astype("float32")
    out["RemainingRaceProgress"] = (1.0 - out["RaceProgress"]).astype("float32")
    out["LapProgress_x_LapNumber"] = (out["LapNumber"] * out["RaceProgress"]).astype("float32")

    out["RacePhase"] = pd.cut(
        out["RaceProgress"], bins=[-np.inf, 0.20, 0.40, 0.60, 0.80, np.inf],
        labels=["P1", "P2", "P3", "P4", "P5"],
    ).astype(str)
    out["LapBin"] = pd.cut(
        out["LapNumber"], bins=[-np.inf, 5, 10, 20, 35, 50, np.inf],
        labels=["L005", "L010", "L020", "L035", "L050", "Lplus"],
    ).astype(str)

    # Tyre features
    out["TyreAgeRatio"] = safe_div(out["TyreLife"], out["LapNumber"].clip(lower=1), eps).astype("float32")
    out["LapPerTyreLife"] = safe_div(out["LapNumber"], out["TyreLife"] + 1, eps).astype("float32")
    out["TyreLife_x_RaceProgress"] = (out["TyreLife"] * out["RaceProgress"]).astype("float32")
    out["PitWindowPressure"] = (out["TyreLife"] * out["RaceProgress"]).astype("float32")
    out["TyreAgeVsRace"] = safe_div(out["TyreLife"], out["EstimatedTotalLaps"].clip(lower=1), eps).astype("float32")
    out["TyreLife_to_LapsRemaining"] = safe_div(out["TyreLife"], out["LapsRemaining"] + 1, eps).astype("float32")
    out["LapMinusTyreLife"] = (out["LapNumber"] - out["TyreLife"]).astype("float32")

    out["TyreLifeBin"] = pd.cut(
        out["TyreLife"], bins=[-np.inf, 3, 7, 12, 20, 30, np.inf],
        labels=["T003", "T007", "T012", "T020", "T030", "Tplus"],
    ).astype(str)

    # Stint
    out["StintPressure"] = (out["Stint"] * out["TyreLife"]).astype("float32")
    out["Is_First_Stint"] = (out["Stint"] == 1).astype(np.int8)
    out["Is_Late_Stint"] = (out["Stint"] >= 3).astype(np.int8)

    # Position
    out["PositionBin"] = pd.cut(
        out["Position"], bins=[-np.inf, 3, 8, 14, np.inf],
        labels=["front", "upper_mid", "lower_mid", "back"],
    ).astype(str)
    out["PositionPressure"] = (out["Position"] * out["RaceProgress"]).astype("float32")

    # Degradation
    out["DegPerRaceLap"] = safe_div(out["Cumulative_Degradation"], out["LapNumber"].clip(lower=1), eps).astype("float32")
    out["DegPerTyreLap"] = safe_div(out["Cumulative_Degradation"], out["TyreLife"].clip(lower=1), eps).astype("float32")
    out["Abs_Cumulative_Degradation"] = out["Cumulative_Degradation"].abs().astype("float32")
    out["Positive_Degradation"] = (out["Cumulative_Degradation"] > 0).astype(np.int8)

    # Lap-time delta
    out["DeltaAbs"] = out["LapTime_Delta"].abs().astype("float32")
    out["LapTimeDeltaPositive"] = (out["LapTime_Delta"] > 0).astype(np.int8)
    out["DeltaPerTyreLap"] = safe_div(out["LapTime_Delta"], out["TyreLife"].clip(lower=1), eps).astype("float32")

    # Position change
    out["Abs_Position_Change"] = out["Position_Change"].abs().astype("float32")
    out["Gained_Position"] = (out["Position_Change"] > 0).astype(np.int8)
    out["Lost_Position"] = (out["Position_Change"] < 0).astype(np.int8)

    return out


def add_frequency_features(frames: list[pd.DataFrame], cat_cols: list[str]) -> None:
    """In-place: add `<col>_count` and `<col>_freq` to every frame, computed on the union."""
    if not cat_cols:
        return
    total = sum(len(f) for f in frames)
    for col in cat_cols:
        if not all(col in f.columns for f in frames):
            continue
        union = pd.concat([f[col].astype("string").fillna("__NA__") for f in frames], axis=0)
        counts = union.value_counts(dropna=False)
        for f in frames:
            keys = f[col].astype("string").fillna("__NA__")
            f[f"{col}_count"] = keys.map(counts).fillna(0).astype(np.int32)
            f[f"{col}_freq"] = (f[f"{col}_count"] / total).astype("float32")


def normalize_cats(out: pd.DataFrame, cat_cols: list[str]) -> None:
    for col in cat_cols:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("__NA__").astype(str)


def build_gbdt_diffFE(*, cat_dtype: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Assemble the GBDT diffFE training data used by every XGBoost/CatBoost base.

    Returns ``(train, test, ext, feature_cols, cat_cols)``. The external frame carries a
    sentinel ``id = -1``; callers concatenate it onto training folds only.

    ``cat_dtype`` selects how categorical columns are encoded — the two families differ:
      - ``True``  (XGBoost): cast each categorical to a unified ``CategoricalDtype`` built
        from the union of train/test/ext, which ``enable_categorical`` requires (else
        unseen val/test categories raise).
      - ``False`` (CatBoost): leave categoricals as plain strings; the caller derives
        ``cat_features`` indices from the returned ``cat_cols`` and passes raw strings to
        ``Pool``.
    """
    train, test = load_competition()
    ext = load_external(dropna_compound=True)
    ext[ID_COL] = -1  # marker; never used in OOF/sub

    train = add_domain_features(train)
    test = add_domain_features(test)
    ext = add_domain_features(ext)

    # diffFE (exp 080-083): cross-cats and group-stats are deliberately dropped, and
    # frequency features are built on the base categoricals only — stripping the
    # over-engineered transductive block made each GBDT stronger and the blend more
    # transfer-robust. The dropped helpers are kept in git history (exp write-ups).
    bins = ["RacePhase", "LapBin", "TyreLifeBin", "PositionBin"]  # keep domain bins (cast safety)
    cat_cols = BASE_CATS + bins

    add_frequency_features([train, test, ext], BASE_CATS)  # diffFE: only base-cat freq

    normalize_cats(train, cat_cols)
    normalize_cats(test, cat_cols)
    normalize_cats(ext, cat_cols)
    if cat_dtype:
        for c in cat_cols:
            if c not in train.columns:
                continue
            union_vals = (
                pd.concat([train[c], test[c], ext[c]], axis=0)
                .astype("string")
                .fillna("__NA__")
                .unique()
                .tolist()
            )
            dtype = pd.CategoricalDtype(categories=sorted(union_vals))
            for f in (train, test, ext):
                if c in f.columns:
                    f[c] = f[c].astype(dtype)

    feature_cols = [c for c in train.columns if c not in (TARGET, ID_COL)]
    feature_cols = [c for c in feature_cols if c in test.columns and c in ext.columns]
    return train, test, ext, feature_cols, cat_cols


# ---------------------------------------------------------------------------
# RealMLP feature recipe — verbatim from the diffFE RealMLP trainers
# ---------------------------------------------------------------------------
def realmlp_features(df, fit, state):
    """RealMLP diffFE. Stateful: call with ``fit=True`` on train (populates ``state``),
    then ``fit=False`` on test/external/pseudo to reuse the fitted encoders. Returns
    ``(df, combo_names)``; the trainer target-encodes ``combo_names`` per fold."""
    df["_LapNumber_/_RaceProgress"] = (df["LapNumber"] / (df["RaceProgress"] + 1e-6)).astype("float32")
    df["_TyreLife_/_LapNumber"] = (df["TyreLife"] / df["LapNumber"].clip(lower=1)).astype("float32")
    df["_LapTime (s)_*_Cumulative_Degradation"] = (df["LapTime (s)"] * df["Cumulative_Degradation"]).astype("float32")
    df["_LapTime (s)_*_Cumulative_Degradation_abs"] = (df["LapTime (s)"] * df["Cumulative_Degradation"].abs()).astype("float32")
    df["_LapTime (s)_/_Cumulative_Degradation_abs"] = (df["LapTime (s)"] / (df["Cumulative_Degradation"].abs() + 1e-6)).astype("float32")
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in df.select_dtypes(exclude=["object"]).columns.tolist() if c not in (ID_COL, TARGET)]
    # diffFE (exp 082): drop categorize-numerics over-engineering; keep only
    # Year_cat_ / PitStop_cat_ (needed by the count block below). Mirrors the
    # exp 080/081 finding that stripping heavy engineered cats strengthens GBDTs.
    for col in ["Year", "PitStop"]:
        cat_name = f"{col}_cat_" if col in num_cols else f"{col[1:]}_cat_"
        if fit:
            codes, uniques = np.floor(df[col]).astype(int).factorize(); state[col] = uniques
        else:
            code_map = {c: i for i, c in enumerate(state[col])}
            codes = np.floor(df[col]).astype(int).map(code_map).fillna(-1).astype("int32")
        df[cat_name] = codes.astype(str)
    for col in cat_cols + ["Year_cat_", "PitStop_cat_"]:
        count_name = f"_{col}_count" if col in cat_cols else f"_{col[:-1]}_count"
        if fit:
            cm = df[col].astype(object).value_counts(); state[count_name] = cm
        else:
            cm = state[count_name]
        df[count_name] = df[col].astype(object).map(cm).fillna(0).astype("int32")
    for col, bins_list in {"RaceProgress": [200], "LapTime (s)": [7]}.items():
        for n_bins in bins_list:
            bn = f"{col}_{n_bins}_quantile_bin_"
            if fit:
                kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
                df[bn] = kb.fit_transform(df[[col]]).ravel().astype("int32").astype(str); state[bn] = kb
            else:
                df[bn] = state[bn].transform(df[[col]]).ravel().astype("int32").astype(str)
    combo_names = []
    for cols in [("Race", "Compound"), ("Race", "Year")]:
        cn = "_".join(cols) + "_"; combo_names.append(cn)
        cs = df[cols[0]].astype(str)
        for c in cols[1:]:
            cs = cs + "_" + df[c].astype(str)
        if fit:
            codes, uniques = pd.factorize(cs, sort=False); state[cn] = uniques
        else:
            cmap = {c: i for i, c in enumerate(state[cn])}
            codes = cs.map(cmap).fillna(-1).astype("int32")
        df[cn] = codes.astype(str)
    return df, combo_names
