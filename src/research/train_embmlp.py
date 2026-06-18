"""Experiment 053 (cycle 16) — entity-embedding MLP (structurally-diverse NN base).

Cycle 16 established that our blend is at the own-model ceiling: every tree model
(XGB/LGB/CB) clusters at rank-corr >= 0.98, feature work doesn't shift XGB's
ranking (exps 051/052, rho 0.997-0.999), and attention NNs are dead (TabM exp 049,
FTT exp 020). The only structurally-new lever within "our own work" is a second NN
that is NOT RealMLP's recipe.

This builds a plain entity-embedding MLP: each high-cardinality categorical
(Driver 887 levels, Race, Compound, Year, Stint, + two cross-cats) gets its own
learned embedding, concatenated with standardised continuous features into a
2-hidden-layer MLP. The explicit large embeddings + vanilla architecture make it
structurally different from RealMLP's PyTabKit recipe (numeric PBLD embeddings,
SiLU, 24-net internal ensemble). The hope: a different NN ranking (rho < ~0.97 vs
RealMLP) that finally adds blend diversity.

Same CV as every prior experiment: 5-fold StratifiedKFold(seed=42) on
Year x PitNextLap; train fold = 4/5 competition + all external; val = competition only.

Outputs:
  data/oof_embmlp.parquet
  data/submission_embmlp.csv
"""

import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parent.parent.parent / "data"
TRAIN_CSV = DATA / "train.csv"
TEST_CSV = DATA / "test.csv"
EXTERNAL_CSV = DATA / "f1_strategy_dataset.csv"
OOF_OUT = DATA / "oof_embmlp.parquet"
SUB_OUT = DATA / "submission_embmlp.csv"
REALMLP_OOF = DATA / "oof_realmlp_multiseed.parquet"
CB_OOF = DATA / "oof_cb_tuned_exp14.parquet"
XGB_OOF = DATA / "oof_xgb_highbins.parquet"

TARGET = "PitNextLap"
ID_COL = "id"
N_SPLITS = 5
SEED = 42

EMB_CATS = ["Driver", "Race", "Compound", "Year", "Stint", "Race_Year", "Driver_Compound"]

MAX_EPOCHS = 30
PATIENCE = 5
BATCH = 1024
LR = 1e-3
WD = 1e-5

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch {torch.__version__}  device {device}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def safe_div(a, b, eps=1e-6):
    return a / (b + eps)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Continuous domain features + two cross-categoricals. Mirrors the tree
    trainers' numeric recipe so the NN sees the same engineered signal."""
    eps = 1e-6
    out = df.copy()
    rp = out["RaceProgress"].clip(lower=eps)
    out["EstimatedTotalLaps"] = (out["LapNumber"] / rp).clip(1, 120).astype("float32")
    out["LapsRemaining"] = (out["EstimatedTotalLaps"] - out["LapNumber"]).clip(lower=0).astype("float32")
    out["RemainingRaceProgress"] = (1.0 - out["RaceProgress"]).astype("float32")
    out["TyreAgeRatio"] = safe_div(out["TyreLife"], out["LapNumber"].clip(lower=1), eps).astype("float32")
    out["TyreLife_x_RaceProgress"] = (out["TyreLife"] * out["RaceProgress"]).astype("float32")
    out["TyreAgeVsRace"] = safe_div(out["TyreLife"], out["EstimatedTotalLaps"].clip(lower=1), eps).astype("float32")
    out["TyreLife_to_LapsRemaining"] = safe_div(out["TyreLife"], out["LapsRemaining"] + 1, eps).astype("float32")
    out["LapMinusTyreLife"] = (out["LapNumber"] - out["TyreLife"]).astype("float32")
    out["StintPressure"] = (out["Stint"] * out["TyreLife"]).astype("float32")
    out["PositionPressure"] = (out["Position"] * out["RaceProgress"]).astype("float32")
    out["DegPerRaceLap"] = safe_div(out["Cumulative_Degradation"], out["LapNumber"].clip(lower=1), eps).astype("float32")
    out["DegPerTyreLap"] = safe_div(out["Cumulative_Degradation"], out["TyreLife"].clip(lower=1), eps).astype("float32")
    out["Abs_Cumulative_Degradation"] = out["Cumulative_Degradation"].abs().astype("float32")
    out["DeltaAbs"] = out["LapTime_Delta"].abs().astype("float32")
    out["DeltaPerTyreLap"] = safe_div(out["LapTime_Delta"], out["TyreLife"].clip(lower=1), eps).astype("float32")
    out["Abs_Position_Change"] = out["Position_Change"].abs().astype("float32")
    # cross-cats for embedding
    out["Race_Year"] = out["Race"].astype(str) + "_" + out["Year"].astype(str)
    out["Driver_Compound"] = out["Driver"].astype(str) + "_" + out["Compound"].astype(str)
    return out


NUM_COLS = [
    "LapNumber", "Stint", "TyreLife", "Position", "LapTime (s)", "LapTime_Delta",
    "Cumulative_Degradation", "RaceProgress", "Position_Change", "PitStop",
    "EstimatedTotalLaps", "LapsRemaining", "RemainingRaceProgress", "TyreAgeRatio",
    "TyreLife_x_RaceProgress", "TyreAgeVsRace", "TyreLife_to_LapsRemaining",
    "LapMinusTyreLife", "StintPressure", "PositionPressure", "DegPerRaceLap",
    "DegPerTyreLap", "Abs_Cumulative_Degradation", "DeltaAbs", "DeltaPerTyreLap",
    "Abs_Position_Change",
]


class EmbMLP(nn.Module):
    def __init__(self, cat_cards, emb_dims, n_num, hidden=(256, 128), p=0.10):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cat_cards, emb_dims)])
        self.emb_drop = nn.Dropout(p)
        self.bn_num = nn.BatchNorm1d(n_num)
        in_dim = sum(emb_dims) + n_num
        layers = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        e = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        x = torch.cat(e + [self.bn_num(x_num)], dim=1)
        x = self.emb_drop(x)
        return self.mlp(x).squeeze(1)


def emb_dim(card: int) -> int:
    return int(min(50, round(1.6 * card ** 0.56)))


def build_cat_codes(frames, col):
    """Global code map over the union; 0 reserved for unknown/NaN. Returns
    (per-frame coded arrays, cardinality incl. unknown slot)."""
    union = pd.concat([f[col].astype("string").fillna("__NA__") for f in frames], axis=0)
    cats = pd.Index(union.unique())
    code = {c: i + 1 for i, c in enumerate(cats)}  # 0 = unknown
    coded = [f[col].astype("string").fillna("__NA__").map(code).fillna(0).astype("int64").to_numpy()
             for f in frames]
    return coded, len(cats) + 1


def train_one_fold(Xc_tr, Xn_tr, y_tr, Xc_va, Xn_va, y_va, Xc_te, Xn_te, cards, dims):
    seed_everything(SEED)
    model = EmbMLP(cards, dims, Xn_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.BCEWithLogitsLoss()

    tc = torch.tensor(Xc_tr, dtype=torch.long)
    tn = torch.tensor(Xn_tr, dtype=torch.float32)
    ty = torch.tensor(y_tr, dtype=torch.float32)
    vc = torch.tensor(Xc_va, dtype=torch.long, device=device)
    vn = torch.tensor(Xn_va, dtype=torch.float32, device=device)
    ec = torch.tensor(Xc_te, dtype=torch.long, device=device)
    en = torch.tensor(Xn_te, dtype=torch.float32, device=device)

    n = len(ty)
    best_auc, best_state, bad = -1.0, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb_c = tc[idx].to(device); xb_n = tn[idx].to(device); yb = ty[idx].to(device)
            opt.zero_grad()
            out = model(xb_c, xb_n)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va_logit = model(vc, vn).float().cpu().numpy()
        auc = roc_auc_score(y_va, va_logit)
        if os.environ.get("EMBMLP_VERBOSE"):
            print(f"    epoch {epoch:2d}  val_auc={auc:.5f}", flush=True)
        if auc > best_auc + 1e-6:
            best_auc, bad = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        va_pred = torch.sigmoid(model(vc, vn)).float().cpu().numpy()
        te_pred = torch.sigmoid(model(ec, en)).float().cpu().numpy()
    return va_pred, te_pred, best_auc, epoch + 1


def main():
    print("Loading data...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    ext = pd.read_csv(EXTERNAL_CSV).dropna(subset=["Compound"]).drop(columns=["Normalized_TyreLife"], errors="ignore")
    ext[ID_COL] = -1
    print(f"  train {train.shape}  test {test.shape}  ext {ext.shape}")

    train = add_features(train); test = add_features(test); ext = add_features(ext)

    # categorical codes (global over union)
    cat_arrays, cards, dims = {}, [], []
    coded_sets = {"train": {}, "test": {}, "ext": {}}
    for col in EMB_CATS:
        (ctr, cte, cex), card = build_cat_codes([train, test, ext], col)
        coded_sets["train"][col] = ctr
        coded_sets["test"][col] = cte
        coded_sets["ext"][col] = cex
        cards.append(card); dims.append(emb_dim(card))
    print("embeddings: " + ", ".join(f"{c}({cd}->{d})" for c, cd, d in zip(EMB_CATS, cards, dims)))

    Xc_train = np.stack([coded_sets["train"][c] for c in EMB_CATS], axis=1)
    Xc_test = np.stack([coded_sets["test"][c] for c in EMB_CATS], axis=1)
    Xc_ext = np.stack([coded_sets["ext"][c] for c in EMB_CATS], axis=1)

    for f in (train, test, ext):
        for c in NUM_COLS:
            f[c] = pd.to_numeric(f[c], errors="coerce").astype("float32")
    Xn_train = train[NUM_COLS].to_numpy(dtype=np.float32)
    Xn_test = test[NUM_COLS].to_numpy(dtype=np.float32)
    Xn_ext = ext[NUM_COLS].to_numpy(dtype=np.float32)

    y = train[TARGET].astype(int).to_numpy()
    y_ext = ext[TARGET].astype(int).to_numpy()

    strat = train["Year"].astype(str) + "_" + train[TARGET].astype(int).astype(str)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train)); test_preds = np.zeros(len(test)); fold_aucs = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(Xn_train, strat), start=1):
        t0 = time.time()
        # numeric scaler fit on this fold's training rows (competition + ext)
        scaler = StandardScaler()
        Xn_tr_raw = np.vstack([Xn_train[tr_idx], Xn_ext])
        scaler.fit(np.nan_to_num(Xn_tr_raw, nan=0.0))
        def scale(a):
            return np.nan_to_num(scaler.transform(np.nan_to_num(a, nan=0.0)), nan=0.0).astype("float32")
        Xn_tr = scale(Xn_tr_raw)
        Xc_tr = np.vstack([Xc_train[tr_idx], Xc_ext])
        y_tr = np.concatenate([y[tr_idx], y_ext]).astype("float32")
        Xn_va = scale(Xn_train[va_idx]); Xc_va = Xc_train[va_idx]
        Xn_te = scale(Xn_test); Xc_te = Xc_test

        va_pred, te_pred, best_auc, ep = train_one_fold(
            Xc_tr, Xn_tr, y_tr, Xc_va, Xn_va, y[va_idx], Xc_te, Xn_te, cards, dims)
        oof[va_idx] = va_pred
        test_preds += te_pred / N_SPLITS
        a = roc_auc_score(y[va_idx], va_pred); fold_aucs.append(a)
        print(f"fold {fold}/{N_SPLITS}  AUC={a:.5f}  best_val={best_auc:.5f}  epochs={ep}  "
              f"train_rows={len(y_tr):,}  ({time.time()-t0:.1f}s)", flush=True)

    oof_auc = roc_auc_score(y, oof)
    print(f"\nper-fold mean={np.mean(fold_aucs):.5f} std={np.std(fold_aucs):.5f}")
    print(f"OOF AUC: {oof_auc:.5f}")
    print(f"  (vs RealMLP-ms 0.95383, Δ={oof_auc-0.95383:+.5f}; vs floor 0.949, Δ={oof_auc-0.949:+.5f})")

    print("\nRank-correlation vs existing bases:")
    for name, path in [("RealMLP-ms", REALMLP_OOF), ("CB-tuned14", CB_OOF), ("XGB-highbins", XGB_OOF)]:
        try:
            other = pd.read_parquet(path)
            m = pd.DataFrame({"id": train[ID_COL], "oof": oof}).merge(
                other[["id", "oof"]].rename(columns={"oof": "o"}), on="id")
            rho, _ = spearmanr(m["oof"], m["o"])
            print(f"  vs {name:14s}: {rho:.5f}")
        except Exception as e:
            print(f"  vs {name}: skipped ({e})")

    pd.DataFrame({"id": train[ID_COL], "Year": train["Year"], "target": y, "oof": oof}).to_parquet(OOF_OUT, index=False)
    pd.DataFrame({"id": test[ID_COL], TARGET: test_preds}).sort_values("id").reset_index(drop=True).to_csv(SUB_OUT, index=False)
    print(f"\nwrote {OOF_OUT.name} and {SUB_OUT.name}")


if __name__ == "__main__":
    main()
