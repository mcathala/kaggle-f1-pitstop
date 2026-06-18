"""Experiment 060 (cycle 17) — lap-attention v3: strengthen on the DIVERSE axis.

Our oracle-boost analysis maps the prize precisely: a base at lap-attention's
diversity (rho ~0.90) needs only AUC ~0.951 (BELOW RealMLP's 0.954) to push the
blend to ~0.9555 — past the 0.9544 top-10% line. lap-attention reached 0.943; it
is ~0.008 short on the diverse axis.

exp 059 (v2) gained AUC (0.936->0.943) but rho rose 0.90->0.94. Our ablation
pinpoints the cause: v1 (minimal cross-lap features) stayed rho 0.90; the coupling
appeared only after adding RealMLP-style DOMAIN-FE numerics. So the coupling is in
the FEATURES, not the attention mechanism.

v3 strengthens ONLY along the diverse axis — no RealMLP-style FE:
  1. MINIMAL features    — the 10 raw numerics + 4 base categoricals (as v1).
  2. WINSORIZE numerics  — our EDA found ~20-138 rows with |x|>500 (LapTime to
                           2507s) that wreck StandardScaler; clip to robust pctiles.
  3. EXTERNAL sequences  — full-race cross-lap signal (aligned with the mechanism).
  4. MORE CAPACITY       — d_model 192, 4 layers, wider FFN, 24-dim embeddings, 50 ep.
  5. 3-SEED ensemble.

Hypothesis: AUC rises toward >=0.949 while rho vs RealMLP stays <= 0.92 -> clears
the hurdle per the oracle map. If AUC rises but rho rises with it again, the
strength<->diversity coupling is intrinsic to the data and we close the NN path.

CV: group-out 5-fold over COMPETITION groups; external + test laps never in any
loss/OOF. Leakage-safe (inputs are observed features; target never fed).

Inputs (add in Kaggle): competition playground-series-s6e5 + external dataset
  <external-f1-strategy-dataset>.
Outputs (/kaggle/working/): oof_lap_attention_v3.parquet, submission_lap_attention_v3.csv
"""

import subprocess, sys
print("=== nvidia-smi ===")
try:
    print(subprocess.check_output(["nvidia-smi"], text=True))
except Exception as e:
    print(f"nvidia-smi failed: {e}")
print("==================")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cu121"])

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

print(f"torch version: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available — Kaggle silently allocated CPU. Set Accelerator to P100/T4.")
print(f"cuda device  : {torch.cuda.get_device_name(0)}")
device = "cuda"

KAGGLE_INPUT = Path("/kaggle/input")
def find_one(filename):
    hits = list(KAGGLE_INPUT.rglob(filename))
    if not hits:
        for p in sorted(KAGGLE_INPUT.rglob("*")):
            print("  ", p)
        raise FileNotFoundError(filename)
    return hits[0]

TRAIN_CSV = find_one("train.csv")
TEST_CSV = find_one("test.csv")
EXTERNAL_CSV = find_one("f1_strategy_dataset.csv")
WORKING = Path("/kaggle/working")
OOF_OUT = WORKING / "oof_lap_attention_v3.parquet"
SUB_OUT = WORKING / "submission_lap_attention_v3.csv"

TARGET = "PitNextLap"
ID_COL = "id"
KEY = ["Year", "Race", "Driver"]
BASE_NUM = ["LapNumber", "Stint", "TyreLife", "Position", "LapTime (s)",
            "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
            "Position_Change", "PitStop"]
BASE_CAT = ["Compound", "Driver", "Race", "Year"]
BIN_CAT = ["RacePhase", "LapBin", "TyreLifeBin", "PositionBin"]
N_SPLITS = 5
SEEDS = [42, 7, 99]
D_MODEL = 192
BS = 256
EPOCHS = 50
PATIENCE = 7
MAX_LAP = 80


def safe_div(a, b, eps=1e-6):
    return a / (b + eps)


def add_domain_features(df):
    eps = 1e-6
    out = df.copy()
    rp = out["RaceProgress"].clip(lower=eps)
    out["EstimatedTotalLaps"] = (out["LapNumber"] / rp).clip(1, 120).astype("float32")
    out["LapsRemaining"] = (out["EstimatedTotalLaps"] - out["LapNumber"]).clip(lower=0).astype("float32")
    out["RemainingRaceProgress"] = (1.0 - out["RaceProgress"]).astype("float32")
    out["LapProgress_x_LapNumber"] = (out["LapNumber"] * out["RaceProgress"]).astype("float32")
    out["RacePhase"] = pd.cut(out["RaceProgress"], bins=[-np.inf, .2, .4, .6, .8, np.inf],
                              labels=["P1", "P2", "P3", "P4", "P5"]).astype(str)
    out["LapBin"] = pd.cut(out["LapNumber"], bins=[-np.inf, 5, 10, 20, 35, 50, np.inf],
                           labels=["L005", "L010", "L020", "L035", "L050", "Lplus"]).astype(str)
    out["TyreAgeRatio"] = safe_div(out["TyreLife"], out["LapNumber"].clip(lower=1), eps).astype("float32")
    out["LapPerTyreLife"] = safe_div(out["LapNumber"], out["TyreLife"] + 1, eps).astype("float32")
    out["TyreLife_x_RaceProgress"] = (out["TyreLife"] * out["RaceProgress"]).astype("float32")
    out["TyreAgeVsRace"] = safe_div(out["TyreLife"], out["EstimatedTotalLaps"].clip(lower=1), eps).astype("float32")
    out["TyreLife_to_LapsRemaining"] = safe_div(out["TyreLife"], out["LapsRemaining"] + 1, eps).astype("float32")
    out["LapMinusTyreLife"] = (out["LapNumber"] - out["TyreLife"]).astype("float32")
    out["TyreLifeBin"] = pd.cut(out["TyreLife"], bins=[-np.inf, 3, 7, 12, 20, 30, np.inf],
                                labels=["T003", "T007", "T012", "T020", "T030", "Tplus"]).astype(str)
    out["StintPressure"] = (out["Stint"] * out["TyreLife"]).astype("float32")
    out["Is_First_Stint"] = (out["Stint"] == 1).astype("float32")
    out["Is_Late_Stint"] = (out["Stint"] >= 3).astype("float32")
    out["PositionBin"] = pd.cut(out["Position"], bins=[-np.inf, 3, 8, 14, np.inf],
                                labels=["front", "upper_mid", "lower_mid", "back"]).astype(str)
    out["PositionPressure"] = (out["Position"] * out["RaceProgress"]).astype("float32")
    out["DegPerRaceLap"] = safe_div(out["Cumulative_Degradation"], out["LapNumber"].clip(lower=1), eps).astype("float32")
    out["DegPerTyreLap"] = safe_div(out["Cumulative_Degradation"], out["TyreLife"].clip(lower=1), eps).astype("float32")
    out["Abs_Cumulative_Degradation"] = out["Cumulative_Degradation"].abs().astype("float32")
    out["Positive_Degradation"] = (out["Cumulative_Degradation"] > 0).astype("float32")
    out["DeltaAbs"] = out["LapTime_Delta"].abs().astype("float32")
    out["LapTimeDeltaPositive"] = (out["LapTime_Delta"] > 0).astype("float32")
    out["DeltaPerTyreLap"] = safe_div(out["LapTime_Delta"], out["TyreLife"].clip(lower=1), eps).astype("float32")
    out["Abs_Position_Change"] = out["Position_Change"].abs().astype("float32")
    out["Gained_Position"] = (out["Position_Change"] > 0).astype("float32")
    out["Lost_Position"] = (out["Position_Change"] < 0).astype("float32")
    return out


DOMAIN_NUM = ["EstimatedTotalLaps", "LapsRemaining", "RemainingRaceProgress", "LapProgress_x_LapNumber",
              "TyreAgeRatio", "LapPerTyreLife", "TyreLife_x_RaceProgress", "TyreAgeVsRace",
              "TyreLife_to_LapsRemaining", "LapMinusTyreLife", "StintPressure", "Is_First_Stint",
              "Is_Late_Stint", "PositionPressure", "DegPerRaceLap", "DegPerTyreLap",
              "Abs_Cumulative_Degradation", "Positive_Degradation", "DeltaAbs", "LapTimeDeltaPositive",
              "DeltaPerTyreLap", "Abs_Position_Change", "Gained_Position", "Lost_Position"]
NUM_COLS = BASE_NUM        # v3: minimal, NO domain FE
CAT_COLS = BASE_CAT        # v3: base cats only


def main():
    t_start = time.time()
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    ext = pd.read_csv(EXTERNAL_CSV).dropna(subset=["Compound"]).drop(columns=["Normalized_TyreLife"], errors="ignore")
    ext[ID_COL] = -1
    print(f"train {train.shape}  test {test.shape}  ext {ext.shape}")

    train["_role"] = 1   # competition train (has target, in CV)
    test["_role"] = 0    # test (predict only)
    ext["_role"] = 2     # external (has target, always-train, never val/test)
    test[TARGET] = -1

    both = pd.concat([train, test, ext], ignore_index=True)
    # WINSORIZE numerics to robust percentiles (our EDA: extreme outliers wreck StandardScaler)
    fit_rows = both["_role"].isin([1, 2])
    for c in BASE_NUM:
        lo, hi = both.loc[fit_rows, c].quantile([0.005, 0.995])
        both[c] = both[c].clip(lo, hi)

    # categorical encoding over the full union (0 = pad)
    cat_sizes = {}
    for c in CAT_COLS:
        codes, uniq = pd.factorize(both[c].astype("string").fillna("__NA__"))
        both[c + "_idx"] = codes.astype(np.int64) + 1
        cat_sizes[c] = len(uniq) + 1
    print("cat vocab sizes:", cat_sizes)

    both["_rawlap"] = both["LapNumber"].astype(int).clip(0, MAX_LAP - 1)
    # standardize numerics — fit on training rows (competition train + external)
    scaler = StandardScaler()
    fit_mask = both["_role"].isin([1, 2]).to_numpy()
    scaler.fit(both.loc[fit_mask, NUM_COLS].fillna(0.0).to_numpy())
    both[NUM_COLS] = np.nan_to_num(scaler.transform(both[NUM_COLS].fillna(0.0).to_numpy()), nan=0.0)

    # group external SEPARATELY from comp/test (it is dense full-race data that
    # overlaps competition keys; merging would create duplicate-lap mega-sequences
    # and break the competition transductive structure). Embedding vocab is shared.
    both["_gsrc"] = (both["_role"] == 2).astype(int)
    both = both.sort_values(KEY + ["_gsrc", "_role", "LapNumber"]).reset_index(drop=True)
    grp = both.groupby(KEY + ["_gsrc"], sort=False)
    glist = [idx.to_numpy() for _, idx in grp.groups.items()]
    n_groups = len(glist)
    max_len = max(len(r) for r in glist)
    print(f"groups={n_groups:,}  max_len={max_len}")

    n_num, n_cat = len(NUM_COLS), len(CAT_COLS)
    Xn = np.zeros((n_groups, max_len, n_num), np.float32)
    Xc = np.zeros((n_groups, max_len, n_cat), np.int64)
    Xlap = np.zeros((n_groups, max_len), np.int64)
    Mask = np.zeros((n_groups, max_len), np.float32)
    Yt = np.zeros((n_groups, max_len), np.float32)
    Wloss = np.zeros((n_groups, max_len), np.float32)   # 1 where row has target (comp-train OR ext)
    Wcv = np.zeros((n_groups, max_len), np.float32)      # 1 where comp-train (eligible for OOF/val)
    Rid = np.full((n_groups, max_len), -1, np.int64)
    Rtest = np.zeros((n_groups, max_len), np.float32)    # 1 where test row
    g_year = np.zeros(n_groups, np.int64)
    g_haspos = np.zeros(n_groups, np.int64)
    g_hascomp = np.zeros(n_groups, np.int64)             # group has any comp-train lap

    num_arr = both[NUM_COLS].to_numpy(np.float32)
    cat_arr = both[[c + "_idx" for c in CAT_COLS]].to_numpy(np.int64)
    lap_arr = both["_rawlap"].to_numpy(np.int64)
    role = both["_role"].to_numpy()
    tgt = both[TARGET].to_numpy(np.float32)
    ids = both[ID_COL].to_numpy(np.int64)
    yr = both["Year"].to_numpy(np.int64)

    for gi, rows in enumerate(glist):
        L = len(rows)
        Xn[gi, :L] = num_arr[rows]; Xc[gi, :L] = cat_arr[rows]; Xlap[gi, :L] = lap_arr[rows]
        Mask[gi, :L] = 1.0
        r = role[rows]; tt = tgt[rows]
        has_t = (r == 1) | (r == 2)
        Wloss[gi, :L] = has_t.astype(np.float32)
        Wcv[gi, :L] = (r == 1).astype(np.float32)
        Rtest[gi, :L] = (r == 0).astype(np.float32)
        Yt[gi, :L] = np.where(has_t, np.clip(tt, 0, 1), 0.0)
        Rid[gi, :L] = ids[rows]
        g_year[gi] = yr[rows][0]
        comp = (r == 1)
        g_hascomp[gi] = int(comp.any())
        g_haspos[gi] = int((tt[comp] > 0.5).any()) if comp.any() else 0

    Xn_t, Xc_t, Xlap_t, Mask_t = map(torch.from_numpy, (Xn, Xc, Xlap, Mask))
    Yt_t, Wloss_t = torch.from_numpy(Yt), torch.from_numpy(Wloss)

    pe = torch.zeros(MAX_LAP, D_MODEL)
    posv = torch.arange(0, MAX_LAP).unsqueeze(1).float()
    divv = torch.exp(torch.arange(0, D_MODEL, 2).float() * (-np.log(10000.0) / D_MODEL))
    pe[:, 0::2] = torch.sin(posv * divv); pe[:, 1::2] = torch.cos(posv * divv)
    pe = pe.to(device)

    class LapAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.embs = nn.ModuleList([nn.Embedding(cat_sizes[c], 24, padding_idx=0) for c in CAT_COLS])
            in_dim = n_num + 24 * n_cat
            self.proj = nn.Sequential(nn.Linear(in_dim, D_MODEL), nn.GELU(), nn.LayerNorm(D_MODEL))
            enc = nn.TransformerEncoderLayer(D_MODEL, nhead=6, dim_feedforward=384,
                                             dropout=0.1, batch_first=True, activation="gelu")
            self.tr = nn.TransformerEncoder(enc, num_layers=4)
            self.head = nn.Sequential(nn.Linear(D_MODEL, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))

        def forward(self, xn, xc, xlap, mask):
            e = [emb(xc[:, :, i]) for i, emb in enumerate(self.embs)]
            h = self.proj(torch.cat([xn] + e, dim=-1)) + pe[xlap]
            h = self.tr(h, src_key_padding_mask=(mask < 0.5))
            return self.head(h).squeeze(-1)

    # comp groups (eligible for CV) vs ext-only groups (always train)
    comp_groups = np.where(g_hascomp == 1)[0]
    ext_only_groups = np.where(g_hascomp == 0)[0]
    print(f"comp groups (CV): {len(comp_groups):,}   ext-only groups (always-train): {len(ext_only_groups):,}")
    strat = np.char.add(g_year[comp_groups].astype(str),
                        np.char.add("_", g_haspos[comp_groups].astype(str)))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward_groups(model, gidx):
        model.eval()
        out = np.zeros((len(gidx), max_len), np.float32)
        with torch.no_grad():
            for s in range(0, len(gidx), BS):
                b = gidx[s:s + BS]
                with torch.cuda.amp.autocast():
                    logit = model(Xn_t[b].to(device), Xc_t[b].to(device),
                                  Xlap_t[b].to(device), Mask_t[b].to(device))
                out[s:s + len(b)] = logit.float().cpu().numpy()
        return out

    def collect(gidx, logits, which):
        prob = 1.0 / (1.0 + np.exp(-logits))
        rid = Rid[gidx]; ytr = Yt[gidx]
        if which == "val":
            sel = (Wcv[gidx] > 0.5)
        else:  # test
            sel = (Rtest[gidx] > 0.5) & (rid >= 0)
        return rid[sel], prob[sel], ytr[sel]

    test_ids = test[ID_COL].to_numpy(np.int64)
    pos = {int(i): k for k, i in enumerate(test_ids)}
    train_ids = train[ID_COL].to_numpy(np.int64)

    # accumulators across seeds
    oof_seed_sum = {}
    test_seed_sum = np.zeros(len(test_ids)); test_seed_cnt = 0
    fold_auc_log = []

    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        oof_by_id = {}
        for fold, (tr_i, va_i) in enumerate(skf.split(comp_groups, strat), start=1):
            t0 = time.time()
            tr_g = np.concatenate([comp_groups[tr_i], ext_only_groups])  # external always trains
            va_g = comp_groups[va_i]
            model = LapAttn().to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            gscaler = torch.cuda.amp.GradScaler()
            best_auc = -1.0; best_state = None; bad = 0
            for ep in range(EPOCHS):
                model.train()
                order = np.random.permutation(tr_g)
                for s in range(0, len(order), BS):
                    b = order[s:s + BS]
                    w = Wloss_t[b].to(device)
                    if float(w.sum()) == 0:
                        continue
                    with torch.cuda.amp.autocast():
                        logit = model(Xn_t[b].to(device), Xc_t[b].to(device),
                                      Xlap_t[b].to(device), Mask_t[b].to(device))
                        loss = (bce(logit.float(), Yt_t[b].to(device)) * w).sum() / w.sum().clamp(min=1)
                    opt.zero_grad(); gscaler.scale(loss).backward(); gscaler.step(opt); gscaler.update()
                _, vp, vy = collect(va_g, forward_groups(model, va_g), "val")
                va = roc_auc_score(vy, vp)
                if va > best_auc + 1e-5:
                    best_auc = va; bad = 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                if bad >= PATIENCE:
                    break
            model.load_state_dict(best_state)
            vid, vp, _ = collect(va_g, forward_groups(model, va_g), "val")
            for i_, p_ in zip(vid, vp):
                oof_by_id[int(i_)] = float(p_)
            # test
            tid, tp, _ = collect(np.arange(n_groups), forward_groups(model, np.arange(n_groups)), "test")
            tarr = np.full(len(test_ids), np.nan)
            for i_, p_ in zip(tid, tp):
                j = pos.get(int(i_))
                if j is not None:
                    tarr[j] = p_
            test_seed_sum += np.nan_to_num(tarr); test_seed_cnt += 1
            fold_auc_log.append((seed, fold, best_auc))
            print(f"seed {seed} fold {fold}/{N_SPLITS}  val_auc={best_auc:.5f}  ({time.time()-t0:.1f}s)", flush=True)
        oof_arr = np.array([oof_by_id.get(int(i), np.nan) for i in train_ids])
        oof_seed_sum[seed] = oof_arr
        y = train[TARGET].to_numpy(int); v = ~np.isnan(oof_arr)
        print(f"  seed {seed} OOF AUC = {roc_auc_score(y[v], oof_arr[v]):.5f}", flush=True)

    # average across seeds
    oof = np.nanmean(np.vstack([oof_seed_sum[s] for s in SEEDS]), axis=0)
    y = train[TARGET].to_numpy(int); v = ~np.isnan(oof)
    oof_auc = roc_auc_score(y[v], oof[v])
    print(f"\n3-seed OOF AUC: {oof_auc:.5f}  (floor 0.949; vs RealMLP-ms 0.95383)  covered={v.mean():.3f}")

    pd.DataFrame({"id": train_ids, "Year": train["Year"], "target": y,
                  "oof": np.nan_to_num(oof)}).to_parquet(OOF_OUT, index=False)
    sub = pd.DataFrame({"id": test_ids, TARGET: test_seed_sum / max(test_seed_cnt, 1)}).sort_values("id").reset_index(drop=True)
    sub.to_csv(SUB_OUT, index=False)
    print(f"wrote {OOF_OUT.name} ({v.sum():,} covered) and {SUB_OUT.name} ({len(sub):,})")
    print(f"total runtime {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
