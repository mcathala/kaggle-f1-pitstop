"""Experiment 030 (cycle 10 Phase 1) — seven diagnostic probes.

  Probe 1: CB feature-importance audit grouped by family.
  Probe 2: OOF-residual feature EDA on cycle 7 blend (loss-quartile gap).
  Probe 3: Adversarial validation (train vs test).
  Probe 4: Rank-disagreement profile (RealMLP-multiseed vs CB-tuned-exp14).
  Probe 5: OOF calibration / reliability diagram (cycle 7 blend).
  Probe 7: Train-test value novelty per categorical (OOD fraction).
  Probe 8: CV-protocol stress test (GroupKFold by Driver vs current).

All probes write a parquet to data/ and print a short summary block.
Used to inform exp 028's FE design.
"""

import sys
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

# Force line-buffered stdout so progress streams to disk
sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parent.parent.parent / "data"
TRAIN_CSV = DATA / "train.csv"
TEST_CSV = DATA / "test.csv"

REALMLP_OOF = DATA / "oof_realmlp_multiseed.parquet"
CB_OOF = DATA / "oof_cb_tuned_exp14.parquet"

OUT_FEATIMP = DATA / "cycle10_cb_feature_importance.parquet"
OUT_RESIDUAL = DATA / "cycle10_residual_eda.parquet"
OUT_ADVERSARIAL = DATA / "cycle10_adversarial_validation.parquet"
OUT_DISAGREEMENT = DATA / "cycle10_rank_disagreement.parquet"
OUT_CALIBRATION = DATA / "cycle10_calibration.parquet"
OUT_NOVELTY = DATA / "cycle10_train_test_novelty.parquet"
OUT_GROUPCV = DATA / "cycle10_groupkfold_stress.parquet"

TARGET = "PitNextLap"
ID_COL = "id"
SEED = 42


# ============================================================
# Probe 1: CB feature-importance audit grouped by family
# ============================================================

def classify_feature_family(col: str, base_cats: set, derived_cats: set, bin_cats: set, freq_cols: set, group_stat_cols: set) -> str:
    if col in base_cats:
        return "raw_cat"
    if col in derived_cats:
        return "cross_cat"
    if col in bin_cats:
        return "bin_cat"
    if col in freq_cols:
        return "freq_encoding"
    if col in group_stat_cols:
        return "group_stat"
    if col.endswith("_cat_") or "_cat_" in col:
        return "bin_cat"
    return "raw_num_or_derived_num"


def probe_1_cb_importance():
    print("\n" + "="*60)
    print("PROBE 1: CB feature-importance audit (cycle-4 recipe, fold 1)")
    print("="*60)

    # Use train_cb_tuned_exp14's recipe directly
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train_cb_tuned_exp14 import (
        add_domain_features, add_cross_categoricals,
        add_frequency_features, add_group_stats, normalize_cats, BASE_CATS,
    )

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    train = add_domain_features(train)
    test = add_domain_features(test)
    train = add_cross_categoricals(train)
    test = add_cross_categoricals(test)

    cross_cats = ["Race_Year", "Compound_Stint", "Driver_Race", "Driver_Compound",
                  "Race_Compound", "Race_Compound_Stint", "Compound_RacePhase",
                  "Compound_TyreLifeBin", "RacePhase_TyreLifeBin"]
    bins = ["RacePhase", "LapBin", "TyreLifeBin", "PositionBin"]
    all_cats_for_freq = BASE_CATS + cross_cats + bins

    add_frequency_features([train, test], all_cats_for_freq)
    group_stat_cols = add_group_stats([train, test])

    base_cats_set = set(BASE_CATS)
    cross_cats_set = set(cross_cats)
    bin_cats_set = set(bins)
    freq_cols_set = set()
    for col in all_cats_for_freq:
        freq_cols_set.add(f"{col}_count")
        freq_cols_set.add(f"{col}_freq")
    group_stat_set = set(group_stat_cols)

    feature_cols = [c for c in train.columns if c not in (ID_COL, TARGET) and c != "id"]
    cat_cols = [c for c in feature_cols if c in base_cats_set | cross_cats_set | bin_cats_set]
    normalize_cats(train, cat_cols)
    normalize_cats(test, cat_cols)

    y = train[TARGET].astype(int)
    strat_key = train["Year"].astype(str) + "_" + train[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr_idx, va_idx = next(kf.split(train[feature_cols], strat_key))

    X_tr = train[feature_cols].iloc[tr_idx]
    y_tr = y.iloc[tr_idx]
    X_va = train[feature_cols].iloc[va_idx]
    y_va = y.iloc[va_idx]

    t0 = time.time()
    model = CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=8,
        loss_function="Logloss", eval_metric="AUC",
        early_stopping_rounds=100, random_seed=SEED,
        verbose=0, allow_writing_files=False, task_type="CPU", thread_count=-1,
    )
    model.fit(Pool(X_tr, y_tr, cat_features=cat_cols), eval_set=Pool(X_va, y_va, cat_features=cat_cols))
    va_auc = roc_auc_score(y_va, model.predict_proba(X_va)[:, 1])
    print(f"  fold 1 CB AUC: {va_auc:.5f}  best_iter={model.tree_count_}  ({time.time()-t0:.0f}s)")

    importances = model.get_feature_importance()
    df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    df["family"] = df["feature"].apply(
        lambda c: classify_feature_family(c, base_cats_set, cross_cats_set, bin_cats_set,
                                          freq_cols_set, group_stat_set)
    )
    by_family = df.groupby("family")["importance"].agg(["sum", "mean", "count"]).sort_values("sum", ascending=False)
    by_family["sum_pct"] = (by_family["sum"] / by_family["sum"].sum() * 100).round(2)

    print("\n  Importance by family (fold 1):")
    print(by_family.to_string())
    print("\n  Top 20 features overall:")
    print(df.sort_values("importance", ascending=False).head(20).to_string(index=False))

    df.to_parquet(OUT_FEATIMP, index=False)
    print(f"  -> wrote {OUT_FEATIMP.name}")
    return by_family


# ============================================================
# Probe 2: OOF-residual feature EDA on cycle 7 blend
# ============================================================

def probe_2_residual_eda():
    print("\n" + "="*60)
    print("PROBE 2: OOF-residual feature EDA on cycle 7 blend")
    print("="*60)

    m = pd.read_parquet(REALMLP_OOF).set_index(ID_COL).sort_index()
    c = pd.read_parquet(CB_OOF).set_index(ID_COL).sort_index()
    assert (m["target"] == c["target"]).all()

    blend = 0.80 * m["oof"].to_numpy() + 0.20 * c["oof"].to_numpy()
    y = m["target"].to_numpy()
    eps = 1e-7
    p = np.clip(blend, eps, 1 - eps)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))

    train = pd.read_csv(TRAIN_CSV).set_index(ID_COL).sort_index()
    assert train.index.equals(m.index), "id misalignment"

    quartiles = pd.qcut(loss, q=4, labels=["Q1_best", "Q2", "Q3", "Q4_worst"])
    print(f"\n  Loss quartiles (n per Q): {pd.Series(quartiles).value_counts().sort_index().to_dict()}")
    print(f"  Q1 mean loss: {loss[quartiles == 'Q1_best'].mean():.5f}")
    print(f"  Q4 mean loss: {loss[quartiles == 'Q4_worst'].mean():.5f}")
    print(f"  pos rate per quartile: Q1={y[quartiles=='Q1_best'].mean():.3f}  Q4={y[quartiles=='Q4_worst'].mean():.3f}")

    rows = []
    # Numeric features: mean per quartile
    num_cols = [c for c in train.columns if c != TARGET and pd.api.types.is_numeric_dtype(train[c])]
    for col in num_cols:
        q1_mean = train[quartiles == "Q1_best"][col].mean()
        q4_mean = train[quartiles == "Q4_worst"][col].mean()
        rel_gap = (q4_mean - q1_mean) / (abs(q1_mean) + 1e-6)
        rows.append({
            "feature": col, "kind": "numeric",
            "q1_best_mean": q1_mean, "q4_worst_mean": q4_mean,
            "abs_gap": q4_mean - q1_mean, "rel_gap_pct": rel_gap * 100,
        })

    # Categorical features: pos-rate of mode per quartile is less informative;
    # use entropy / value share difference
    cat_cols = [c for c in train.columns if c != TARGET and not pd.api.types.is_numeric_dtype(train[c])]
    for col in cat_cols:
        q1 = train[quartiles == "Q1_best"][col].value_counts(normalize=True)
        q4 = train[quartiles == "Q4_worst"][col].value_counts(normalize=True)
        common = q1.index.union(q4.index)
        q1r = q1.reindex(common, fill_value=0.0)
        q4r = q4.reindex(common, fill_value=0.0)
        tvd = 0.5 * (q1r - q4r).abs().sum()
        rows.append({
            "feature": col, "kind": "categorical",
            "q1_best_mean": np.nan, "q4_worst_mean": np.nan,
            "abs_gap": np.nan, "rel_gap_pct": tvd * 100,  # repurpose as TVD %
        })

    eda = pd.DataFrame(rows).sort_values("rel_gap_pct", key=lambda s: s.abs(), ascending=False)
    eda.to_parquet(OUT_RESIDUAL, index=False)
    print(f"\n  Top 15 features by |Q4 vs Q1| gap (% relative for numerics, TVD % for cats):")
    print(eda.head(15).to_string(index=False))
    print(f"  -> wrote {OUT_RESIDUAL.name}")
    return eda


# ============================================================
# Probe 3: Adversarial validation (train vs test)
# ============================================================

def probe_3_adversarial():
    print("\n" + "="*60)
    print("PROBE 3: Adversarial validation (train=1, test=0)")
    print("="*60)

    train = pd.read_csv(TRAIN_CSV).drop(columns=[TARGET])
    test = pd.read_csv(TEST_CSV)
    train["__is_train"] = 1
    test["__is_train"] = 0

    combined = pd.concat([train, test], axis=0, ignore_index=True)
    feature_cols = [c for c in combined.columns if c not in (ID_COL, "__is_train")]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(combined[c])]
    for c in cat_cols:
        combined[c] = combined[c].astype(str).fillna("__MISSING__")

    y_adv = combined["__is_train"].to_numpy()
    X_adv = combined[feature_cols]

    # Stratified train/val 80/20
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(combined))
    rng.shuffle(idx)
    cut = int(0.8 * len(idx))
    tr_idx, va_idx = idx[:cut], idx[cut:]

    t0 = time.time()
    model = CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6,
        loss_function="Logloss", eval_metric="AUC",
        early_stopping_rounds=50, random_seed=SEED,
        verbose=0, allow_writing_files=False, task_type="CPU", thread_count=-1,
    )
    model.fit(
        Pool(X_adv.iloc[tr_idx], y_adv[tr_idx], cat_features=cat_cols),
        eval_set=Pool(X_adv.iloc[va_idx], y_adv[va_idx], cat_features=cat_cols),
    )
    va_pred = model.predict_proba(X_adv.iloc[va_idx])[:, 1]
    adv_auc = roc_auc_score(y_adv[va_idx], va_pred)
    print(f"  Adversarial AUC: {adv_auc:.5f}  (0.5 = same dist, 1.0 = perfectly separable)")
    print(f"  Wall: {time.time()-t0:.0f}s, best_iter={model.tree_count_}")

    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.get_feature_importance(),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    print("\n  Top 15 features by adversarial importance:")
    print(imp.head(15).to_string(index=False))
    imp.to_parquet(OUT_ADVERSARIAL, index=False)
    print(f"  -> wrote {OUT_ADVERSARIAL.name}")
    print(f"\n  Interpretation:")
    print(f"  AUC < 0.55: train and test look the same.")
    print(f"  AUC 0.55-0.70: minor drift; recalibration could help.")
    print(f"  AUC > 0.70: real distribution shift; investigate top features.")
    return adv_auc, imp


# ============================================================
# Probe 4: Rank-disagreement profile (RealMLP vs CB)
# ============================================================

def probe_4_disagreement():
    print("\n" + "="*60)
    print("PROBE 4: Rank-disagreement profile (RealMLP vs CB-tuned-exp14)")
    print("="*60)

    from scipy.stats import rankdata

    m = pd.read_parquet(REALMLP_OOF).set_index(ID_COL).sort_index()
    c = pd.read_parquet(CB_OOF).set_index(ID_COL).sort_index()
    assert (m["target"] == c["target"]).all()

    rm = rankdata(m["oof"], method="average") / len(m)
    rc = rankdata(c["oof"], method="average") / len(c)
    diff = np.abs(rm - rc)
    pct95 = np.percentile(diff, 95)

    train = pd.read_csv(TRAIN_CSV).set_index(ID_COL).sort_index()
    assert train.index.equals(m.index)

    is_disagree = diff >= pct95
    print(f"\n  rank-diff threshold (top 5%): {pct95:.4f}")
    print(f"  rows flagged as high-disagreement: {is_disagree.sum():,} / {len(diff):,} ({100*is_disagree.mean():.2f}%)")
    y = m["target"].to_numpy()
    print(f"  pos rate in disagreement set: {y[is_disagree].mean():.4f}  vs overall: {y.mean():.4f}")

    rows = []
    num_cols = [c for c in train.columns if c != TARGET and pd.api.types.is_numeric_dtype(train[c])]
    for col in num_cols:
        disagree_mean = train.loc[is_disagree, col].mean()
        rest_mean = train.loc[~is_disagree, col].mean()
        rel_gap = (disagree_mean - rest_mean) / (abs(rest_mean) + 1e-6)
        rows.append({
            "feature": col, "kind": "numeric",
            "disagree_mean": disagree_mean, "rest_mean": rest_mean,
            "abs_gap": disagree_mean - rest_mean, "rel_gap_pct": rel_gap * 100,
        })

    cat_cols = [c for c in train.columns if c != TARGET and not pd.api.types.is_numeric_dtype(train[c])]
    for col in cat_cols:
        d = train.loc[is_disagree, col].value_counts(normalize=True)
        r = train.loc[~is_disagree, col].value_counts(normalize=True)
        common = d.index.union(r.index)
        dr = d.reindex(common, fill_value=0.0)
        rr = r.reindex(common, fill_value=0.0)
        tvd = 0.5 * (dr - rr).abs().sum()
        rows.append({
            "feature": col, "kind": "categorical",
            "disagree_mean": np.nan, "rest_mean": np.nan,
            "abs_gap": np.nan, "rel_gap_pct": tvd * 100,
        })

    prof = pd.DataFrame(rows).sort_values("rel_gap_pct", key=lambda s: s.abs(), ascending=False)
    prof.to_parquet(OUT_DISAGREEMENT, index=False)
    print(f"\n  Top 15 features marking disagreement (% relative gap for numerics, TVD % for cats):")
    print(prof.head(15).to_string(index=False))
    print(f"  -> wrote {OUT_DISAGREEMENT.name}")
    return prof


# ============================================================
# Probe 5: OOF calibration / reliability diagram
# ============================================================

def probe_5_calibration():
    print("\n" + "="*60)
    print("PROBE 5: OOF calibration on cycle 7 blend (10 equal-frequency bins)")
    print("="*60)

    m = pd.read_parquet(REALMLP_OOF).set_index(ID_COL).sort_index()
    c = pd.read_parquet(CB_OOF).set_index(ID_COL).sort_index()
    y = m["target"].to_numpy().astype(int)
    blend = 0.80 * m["oof"].to_numpy() + 0.20 * c["oof"].to_numpy()

    # Equal-frequency bins
    bins = 10
    order = np.argsort(blend, kind="mergesort")
    n = len(blend)
    bin_size = n // bins
    rows = []
    for i in range(bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < bins - 1 else n
        idx = order[start:end]
        pred_mean = float(np.mean(blend[idx]))
        actual_rate = float(np.mean(y[idx]))
        rows.append({
            "bin": i + 1,
            "n": len(idx),
            "pred_mean": pred_mean,
            "actual_rate": actual_rate,
            "bias": pred_mean - actual_rate,
            "bias_abs": abs(pred_mean - actual_rate),
        })
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_CALIBRATION, index=False)
    print(df.to_string(index=False))
    max_bias_bin = df.loc[df["bias_abs"].idxmax()]
    print(f"\n  Largest bias bin: {int(max_bias_bin['bin'])}  pred={max_bias_bin['pred_mean']:.4f}  actual={max_bias_bin['actual_rate']:.4f}  bias={max_bias_bin['bias']:+.4f}")
    overall_bias = float(blend.mean() - y.mean())
    print(f"  Overall pred-vs-actual bias: {overall_bias:+.5f}")
    print(f"  -> wrote {OUT_CALIBRATION.name}")
    print(f"\n  Interpretation:")
    print(f"  max |bias| < 0.005: well-calibrated, isotonic recalibration won't help much.")
    print(f"  max |bias| 0.005-0.020: some miscalibration; isotonic could net +0.0001-0.0003 LB.")
    print(f"  max |bias| > 0.020: heavy miscalibration in some bin; recalibration likely material.")
    return df


# ============================================================
# Probe 7: Train-test value novelty per categorical
# ============================================================

def probe_7_novelty():
    print("\n" + "="*60)
    print("PROBE 7: Train-test value novelty (out-of-distribution rows per categorical)")
    print("="*60)

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    cat_cols = [c for c in test.columns if c != ID_COL and not pd.api.types.is_numeric_dtype(test[c])]
    rows = []
    for col in cat_cols:
        train_vals = set(train[col].astype(str).fillna("__NA__").unique())
        test_vals = test[col].astype(str).fillna("__NA__")
        is_novel = ~test_vals.isin(train_vals)
        n_novel = int(is_novel.sum())
        n_test = len(test_vals)
        rows.append({
            "feature": col,
            "n_train_unique": len(train_vals),
            "n_test_unique": int(test_vals.nunique()),
            "n_test_novel_rows": n_novel,
            "pct_test_novel": 100 * n_novel / n_test,
        })
    df = pd.DataFrame(rows).sort_values("pct_test_novel", ascending=False)
    df.to_parquet(OUT_NOVELTY, index=False)
    print(df.to_string(index=False))
    max_novel = df.iloc[0]
    print(f"\n  Most novel-heavy categorical: '{max_novel['feature']}' ({max_novel['pct_test_novel']:.2f}% of test rows have unseen value)")
    print(f"  -> wrote {OUT_NOVELTY.name}")
    print(f"\n  Interpretation:")
    print(f"  < 0.5%: negligible OOD risk; no action.")
    print(f"  0.5-5%: moderate; CB handles via __MISSING__-style; RealMLP via embedding lookup of unknowns.")
    print(f"  > 5%: real OOD risk; consider fallback rules for OOD rows or strip the cat from blend.")
    return df


# ============================================================
# Probe 8: CV-protocol stress test (GroupKFold by Driver)
# ============================================================

def probe_8_groupkfold_stress():
    print("\n" + "="*60)
    print("PROBE 8: CV-protocol stress test — GroupKFold by Driver vs current stratified")
    print("="*60)

    # Use train_cb_tuned_exp14's recipe directly (same as probe 1)
    sys.path.insert(0, str(Path(__file__).parent))
    from train_cb_tuned_exp14 import (
        add_domain_features, add_cross_categoricals,
        add_frequency_features, add_group_stats, normalize_cats, BASE_CATS,
    )

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    train = add_domain_features(train)
    test = add_domain_features(test)
    train = add_cross_categoricals(train)
    test = add_cross_categoricals(test)

    cross_cats = ["Race_Year", "Compound_Stint", "Driver_Race", "Driver_Compound",
                  "Race_Compound", "Race_Compound_Stint", "Compound_RacePhase",
                  "Compound_TyreLifeBin", "RacePhase_TyreLifeBin"]
    bins = ["RacePhase", "LapBin", "TyreLifeBin", "PositionBin"]
    all_cats_for_freq = BASE_CATS + cross_cats + bins
    add_frequency_features([train, test], all_cats_for_freq)
    add_group_stats([train, test])

    feature_cols = [c for c in train.columns if c not in (ID_COL, TARGET) and c != "id"]
    cat_cols = [c for c in feature_cols if c in set(BASE_CATS) | set(cross_cats) | set(bins)]
    normalize_cats(train, cat_cols)

    y = train[TARGET].astype(int)
    driver = train["Driver"].to_numpy()

    print(f"  unique drivers: {len(set(driver)):,}")
    print(f"  fold = GroupKFold by Driver, 5 splits")

    gkf = GroupKFold(n_splits=5)
    rows = []
    fold_aucs = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train[feature_cols], y, groups=driver), start=1):
        t0 = time.time()
        X_tr = train[feature_cols].iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        X_va = train[feature_cols].iloc[va_idx]
        y_va = y.iloc[va_idx]

        model = CatBoostClassifier(
            iterations=2000, learning_rate=0.05, depth=8,
            loss_function="Logloss", eval_metric="AUC",
            early_stopping_rounds=100, random_seed=SEED,
            verbose=0, allow_writing_files=False, task_type="CPU", thread_count=-1,
        )
        model.fit(Pool(X_tr, y_tr, cat_features=cat_cols),
                  eval_set=Pool(X_va, y_va, cat_features=cat_cols))
        va_pred = model.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, va_pred)
        fold_aucs.append(auc)
        rows.append({
            "fold": fold, "n_train_drivers": len(set(driver[tr_idx])),
            "n_val_drivers": len(set(driver[va_idx])),
            "n_train_rows": len(tr_idx), "n_val_rows": len(va_idx),
            "auc": auc, "best_iter": model.tree_count_, "wall_s": time.time() - t0,
        })
        print(f"  fold {fold}: drivers tr={rows[-1]['n_train_drivers']} va={rows[-1]['n_val_drivers']}  AUC={auc:.5f}  best_iter={model.tree_count_}  ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_GROUPCV, index=False)
    group_oof = float(np.mean(fold_aucs))
    # Reference: cycle 4 CB-tuned-exp14 OOF on stratified = 0.95114
    print(f"\n  GroupKFold-by-Driver mean AUC: {group_oof:.5f}  (per-fold std: {np.std(fold_aucs):.5f})")
    print(f"  Reference (stratified Year×target, exp 14 OOF): 0.95114")
    print(f"  Δ (Group − Stratified): {group_oof - 0.95114:+.5f}")
    print(f"  -> wrote {OUT_GROUPCV.name}")
    print(f"\n  Interpretation:")
    print(f"  Δ within ±0.002: features generalize cleanly across drivers; no memorization.")
    print(f"  Δ −0.002 to −0.005: minor driver-specific signal; mostly safe.")
    print(f"  Δ < −0.005: features ENCODE driver memorization; private LB risk; consider dropping Driver-correlated features.")
    return group_oof


# ============================================================
# Main
# ============================================================

def main():
    print(f"Cycle 10 Phase 1 diagnostics — 7 probes")
    print(f"output dir: {DATA}")

    by_family = probe_1_cb_importance()
    eda = probe_2_residual_eda()
    adv_auc, adv_imp = probe_3_adversarial()
    disagree_prof = probe_4_disagreement()
    calib = probe_5_calibration()
    novelty = probe_7_novelty()
    group_oof = probe_8_groupkfold_stress()

    print("\n" + "="*60)
    print("SYNTHESIS")
    print("="*60)
    print(f"  Probe 1: top family by total importance = '{by_family.index[0]}' ({by_family.iloc[0]['sum_pct']:.1f}% of importance)")
    print(f"  Probe 2: top residual feature = '{eda.iloc[0]['feature']}' (gap {eda.iloc[0]['rel_gap_pct']:.1f}%)")
    print(f"  Probe 3: adversarial AUC = {adv_auc:.4f}, top drift feature = '{adv_imp.iloc[0]['feature']}'")
    print(f"  Probe 4: top disagreement feature = '{disagree_prof.iloc[0]['feature']}' (gap {disagree_prof.iloc[0]['rel_gap_pct']:.1f}%)")
    print(f"  Probe 5: max calibration bias = {calib['bias_abs'].max():.4f} in bin {int(calib.loc[calib['bias_abs'].idxmax(), 'bin'])}")
    print(f"  Probe 7: most novel cat = '{novelty.iloc[0]['feature']}' ({novelty.iloc[0]['pct_test_novel']:.2f}% test rows OOD)")
    print(f"  Probe 8: GroupKFold-Driver OOF = {group_oof:.5f}  (stratified ref 0.95114, Δ {group_oof-0.95114:+.5f})")


if __name__ == "__main__":
    main()
