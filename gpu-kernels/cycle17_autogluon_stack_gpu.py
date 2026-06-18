"""Experiment 062 (cycle 17) — AutoGluon WITH full stacking (the untried variant).

exp 024 (cycle 8) ran AutoGluon but DISABLED its core mechanism (num_bag_folds=0,
num_stack_levels=0) to drive an external 5-fold CV — so AG's multi-layer stacking
was never actually tested. This run enables it: bagging + 1 stack level over a
broad zoo (LGB / CAT / XGB / NN), trained on competition + external, with AG's
bagged out-of-fold predictions used as our OOF.

Hypothesis: AG's stacked ensemble exceeds our hand-tuned 3-way linear blend
(OOF 0.95420) standalone, OR is independent enough that blending it lifts past the
0.95441 hurdle. Our blend-combiner is saturated for the current bases (exp 061), so
this only helps if AG's stacker is genuinely stronger/different.

CV note: AG's internal bagged OOF uses AG's own folds (not our Year×target
StratifiedKFold). Acceptable for a blend probe — each base's OOF is unbiased.

Inputs (add in Kaggle): competition playground-series-s6e5 + external
  <external-f1-strategy-dataset>.
Outputs (/kaggle/working/): oof_autogluon_stack.parquet, submission_autogluon_stack.csv
"""

import subprocess, sys, time
print("=== install autogluon.tabular ===")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "autogluon.tabular[all]"])

from pathlib import Path
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer

KAGGLE_INPUT = Path("/kaggle/input")
def find_one(fn):
    hits = list(KAGGLE_INPUT.rglob(fn))
    if not hits:
        for p in sorted(KAGGLE_INPUT.rglob("*")): print("  ", p)
        raise FileNotFoundError(fn)
    return hits[0]

TRAIN_CSV, TEST_CSV = find_one("train.csv"), find_one("test.csv")
EXTERNAL_CSV = find_one("f1_strategy_dataset.csv")
WORKING = Path("/kaggle/working")
OOF_OUT = WORKING / "oof_autogluon_stack.parquet"
SUB_OUT = WORKING / "submission_autogluon_stack.csv"
TARGET, ID_COL = "PitNextLap", "id"
SEED = 42
TIME_LIMIT = 5400  # seconds fit cap (~1.5h compute; budget-conscious)


def feature_engineering(df, fit, state):
    df["_LapNumber_/_RaceProgress"] = (df["LapNumber"] / (df["RaceProgress"] + 1e-6)).astype("float32")
    df["_TyreLife_/_LapNumber"] = (df["TyreLife"] / df["LapNumber"].clip(lower=1)).astype("float32")
    df["_LapTime (s)_*_Cumulative_Degradation"] = (df["LapTime (s)"] * df["Cumulative_Degradation"]).astype("float32")
    df["_LapTime (s)_*_Cumulative_Degradation_abs"] = (df["LapTime (s)"] * df["Cumulative_Degradation"].abs()).astype("float32")
    df["_LapTime (s)_/_Cumulative_Degradation_abs"] = (df["LapTime (s)"] / (df["Cumulative_Degradation"].abs() + 1e-6)).astype("float32")
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in df.select_dtypes(exclude=["object"]).columns.tolist() if c not in (ID_COL, TARGET)]
    for col in num_cols + ["_LapNumber_/_RaceProgress", "_TyreLife_/_LapNumber"]:
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
    for cols in [("Race", "Compound"), ("Race", "Year")]:
        combo = "_".join(cols) + "_"
        s = df[cols[0]].astype(str)
        for c in cols[1:]:
            s = s + "_" + df[c].astype(str)
        if fit:
            codes, uniques = pd.factorize(s, sort=False); state[combo] = uniques
        else:
            cmap = {c: i for i, c in enumerate(state[combo])}
            codes = s.map(cmap).fillna(-1).astype("int32")
        df[combo] = codes.astype(str)
    return df


def main():
    t0 = time.time()
    print(f"autogluon ready; loading data...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    ext = pd.read_csv(EXTERNAL_CSV).dropna(subset=["Compound"]).drop(columns=["Normalized_TyreLife"], errors="ignore")
    print(f"train {train.shape} test {test.shape} ext {ext.shape}")

    state = {}
    Xtr = feature_engineering(train.drop(columns=[]).copy(), True, state)
    Xte = feature_engineering(test.copy(), False, state)
    Xex = feature_engineering(ext.copy(), False, state)

    feat = [c for c in Xtr.columns if c not in (ID_COL,)]
    comp = Xtr[feat].copy(); comp["_src"] = "comp"
    exa = Xex[[c for c in feat if c in Xex.columns]].copy()
    for c in feat:
        if c not in exa.columns:
            exa[c] = np.nan
    exa = exa[feat]; exa["_src"] = "ext"
    train_data = pd.concat([comp, exa], ignore_index=True)
    n_comp = len(comp)
    print(f"AG train_data {train_data.shape} (comp {n_comp:,} + ext {len(exa):,})")

    predictor = TabularPredictor(label=TARGET, eval_metric="roc_auc",
                                 path=str(WORKING / "ag")).fit(
        train_data.drop(columns=["_src"]),
        presets="high_quality",
        num_bag_folds=5, num_stack_levels=1, num_bag_sets=1,
        excluded_model_types=["KNN", "RF", "XT"],
        time_limit=TIME_LIMIT, verbosity=2,
    )

    oof_all = predictor.predict_proba_oof()  # aligned to train_data index
    pos = predictor.class_labels[-1]
    oof_comp = oof_all[pos].to_numpy()[:n_comp]
    y = train[TARGET].to_numpy(int)
    print(f"\nAG bagged OOF AUC (comp rows): {roc_auc_score(y, oof_comp):.5f}  (vs RealMLP 0.95383, blend 0.95420)")
    print("\nleaderboard (top):")
    print(predictor.leaderboard(silent=True).head(12).to_string())

    test_pred = predictor.predict_proba(Xte[feat])[pos].to_numpy()
    pd.DataFrame({"id": train[ID_COL], "Year": train["Year"], "target": y, "oof": oof_comp}).to_parquet(OOF_OUT, index=False)
    pd.DataFrame({"id": test[ID_COL], TARGET: test_pred}).sort_values("id").reset_index(drop=True).to_csv(SUB_OUT, index=False)
    print(f"\nwrote {OOF_OUT.name} and {SUB_OUT.name}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
