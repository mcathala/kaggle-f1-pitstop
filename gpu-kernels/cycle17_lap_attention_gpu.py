"""Experiment 058 (cycle 17) — transductive lap-attention model on Kaggle GPU.

Every base so far is ROW-INDEPENDENT. EDA (2026-05-27) showed the data is a
sparse, transductive sample: each (Year, Race, Driver) group holds a non-
contiguous subset of a race's laps, and train/test laps are interleaved within
the same group (93% of test laps have a neighbouring sampled lap we observe).
The target is synthetic-noised (PitNextLap != PitStop[t+1]; 18% concordance) so
there is no deterministic leak — but the surrounding pit/stint/tyre trajectory
carries signal no row-independent model can read.

This model lets each lap SELF-ATTEND over all sampled laps of its driver-race
(bidirectional; leakage-free because inputs are observed features, never the
target). Hypothesis: cross-lap context is a strength source distinct from
RealMLP's numeric embeddings, so this base could be both >= the 0.949 floor AND
rank-diverse from RealMLP (rho < 0.97) — the empty "strong+diverse" quadrant.

CV: group-out 5-fold (a driver-race is entirely train or entirely val). Test
laps are always included in their group's sequence as context but never in any
loss. Produces row-level OOF for every train lap.

Competition-only for this feasibility probe (external integration deferred).

Inputs (add in Kaggle): competition playground-series-s6e5.
Outputs (/kaggle/working/): oof_lap_attention.parquet, submission_lap_attention.csv
"""

import subprocess, sys
print("=== nvidia-smi ===")
try:
    print(subprocess.check_output(["nvidia-smi"], text=True))
except Exception as e:
    print(f"nvidia-smi failed: {e}")
print("==================")

# Kaggle's default torch dropped sm_60 (Tesla P100). Pin torch 2.5.1 (cu121,
# still ships sm_60; also fine on T4 sm_75). Matches the cycle-16 GPU kernels.
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
WORKING = Path("/kaggle/working")
OOF_OUT = WORKING / "oof_lap_attention.parquet"
SUB_OUT = WORKING / "submission_lap_attention.csv"

TARGET = "PitNextLap"
ID_COL = "id"
KEY = ["Year", "Race", "Driver"]
NUM_COLS = ["LapNumber", "Stint", "TyreLife", "Position", "LapTime (s)",
            "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
            "Position_Change", "PitStop"]
CAT_COLS = ["Compound", "Driver", "Race", "Year"]
N_SPLITS = 5
SEED = 42
D_MODEL = 128
BS = 256
EPOCHS = 30
PATIENCE = 5
MAX_LAP = 80  # positional-encoding table size; LapNumber clamped to [0, MAX_LAP-1]

torch.manual_seed(SEED); np.random.seed(SEED)


def main():
    t_start = time.time()
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    print(f"train {train.shape}  test {test.shape}")

    train["_is_train"] = 1
    test["_is_train"] = 0
    test[TARGET] = -1  # placeholder, never enters loss
    both = pd.concat([train, test], ignore_index=True)
    both["_rawlap"] = both["LapNumber"].astype(int).clip(0, MAX_LAP - 1)  # before scaling

    # encode categoricals over the union (0 = pad)
    cat_sizes = {}
    for c in CAT_COLS:
        codes, uniques = pd.factorize(both[c].astype("string").fillna("__NA__"))
        both[c + "_idx"] = codes.astype(np.int64) + 1
        cat_sizes[c] = len(uniques) + 1
    print("cat vocab sizes:", cat_sizes)

    # standardize numerics (fit on train laps only)
    scaler = StandardScaler()
    scaler.fit(both.loc[both["_is_train"] == 1, NUM_COLS].fillna(0.0).to_numpy())
    both[NUM_COLS] = np.nan_to_num(scaler.transform(both[NUM_COLS].fillna(0.0).to_numpy()), nan=0.0)

    # build padded per-group tensors
    both = both.sort_values(KEY + ["LapNumber"]).reset_index(drop=True)
    grp = both.groupby(KEY, sort=False)
    glist = [idx.to_numpy() for _, idx in grp.groups.items()]
    n_groups = len(glist)
    max_len = max(len(r) for r in glist)
    print(f"groups={n_groups:,}  max_len={max_len}")

    n_num = len(NUM_COLS); n_cat = len(CAT_COLS)
    Xn = np.zeros((n_groups, max_len, n_num), np.float32)
    Xc = np.zeros((n_groups, max_len, n_cat), np.int64)
    Xlap = np.zeros((n_groups, max_len), np.int64)
    Mask = np.zeros((n_groups, max_len), np.float32)   # 1 = valid lap
    Yt = np.zeros((n_groups, max_len), np.float32)     # target (train laps)
    Wt = np.zeros((n_groups, max_len), np.float32)     # 1 = train lap (loss / OOF mask)
    Rid = np.full((n_groups, max_len), -1, np.int64)
    g_year = np.zeros(n_groups, np.int64)
    g_haspos = np.zeros(n_groups, np.int64)

    num_arr = both[NUM_COLS].to_numpy(np.float32)
    cat_arr = both[[c + "_idx" for c in CAT_COLS]].to_numpy(np.int64)
    lap_arr = both["_rawlap"].to_numpy(np.int64)
    istr = both["_is_train"].to_numpy()
    tgt = both[TARGET].to_numpy(np.float32)
    ids = both[ID_COL].to_numpy(np.int64)
    yr = both["Year"].to_numpy(np.int64)

    for gi, rows in enumerate(glist):
        L = len(rows)
        Xn[gi, :L] = num_arr[rows]
        Xc[gi, :L] = cat_arr[rows]
        Xlap[gi, :L] = lap_arr[rows]
        Mask[gi, :L] = 1.0
        isr = istr[rows]; tt = tgt[rows]
        Wt[gi, :L] = (isr == 1).astype(np.float32)
        Yt[gi, :L] = np.where(isr == 1, np.clip(tt, 0, 1), 0.0)
        Rid[gi, :L] = ids[rows]
        g_year[gi] = yr[rows][0]
        g_haspos[gi] = int((tt[(isr == 1)] > 0.5).any()) if (isr == 1).any() else 0

    Xn_t, Xc_t, Xlap_t, Mask_t = map(torch.from_numpy, (Xn, Xc, Xlap, Mask))
    Yt_t, Wt_t = torch.from_numpy(Yt), torch.from_numpy(Wt)

    # sinusoidal positional encoding indexed by integer LapNumber
    pe = torch.zeros(MAX_LAP, D_MODEL)
    posv = torch.arange(0, MAX_LAP).unsqueeze(1).float()
    divv = torch.exp(torch.arange(0, D_MODEL, 2).float() * (-np.log(10000.0) / D_MODEL))
    pe[:, 0::2] = torch.sin(posv * divv); pe[:, 1::2] = torch.cos(posv * divv)
    pe = pe.to(device)

    class LapAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.embs = nn.ModuleList([nn.Embedding(cat_sizes[c], 16, padding_idx=0) for c in CAT_COLS])
            in_dim = n_num + 16 * n_cat
            self.proj = nn.Sequential(nn.Linear(in_dim, D_MODEL), nn.GELU(), nn.LayerNorm(D_MODEL))
            enc = nn.TransformerEncoderLayer(D_MODEL, nhead=4, dim_feedforward=256,
                                             dropout=0.1, batch_first=True, activation="gelu")
            self.tr = nn.TransformerEncoder(enc, num_layers=3)
            self.head = nn.Sequential(nn.Linear(D_MODEL, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))

        def forward(self, xn, xc, xlap, mask):
            e = [emb(xc[:, :, i]) for i, emb in enumerate(self.embs)]
            h = self.proj(torch.cat([xn] + e, dim=-1)) + pe[xlap]
            h = self.tr(h, src_key_padding_mask=(mask < 0.5))
            return self.head(h).squeeze(-1)

    # group-out folds, stratified by Year x has-positive
    strat = np.char.add(g_year.astype(str), np.char.add("_", g_haspos.astype(str)))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward_groups(model, gidx):
        """Return logits [len(gidx), max_len] for the given group indices (no grad)."""
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

    def collect(gidx, logits, want_train):
        """Vectorized: pull (id, prob, target) for train laps (want_train) or test laps."""
        prob = 1.0 / (1.0 + np.exp(-logits))
        w = Wt[gidx]; rid = Rid[gidx]; ytr = Yt[gidx]
        sel = (w > 0.5) if want_train else ((w < 0.5) & (rid >= 0))
        return rid[sel], prob[sel], ytr[sel]

    test_ids = test[ID_COL].to_numpy(np.int64)
    oof_by_id = {}
    test_sum = np.zeros(len(test)); test_cnt = 0
    fold_aucs = []

    for fold, (tr_g, va_g) in enumerate(skf.split(np.arange(n_groups), strat), start=1):
        t0 = time.time()
        model = LapAttn().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        gscaler = torch.cuda.amp.GradScaler()

        best_auc = -1.0; best_state = None; bad = 0
        for ep in range(EPOCHS):
            model.train()
            order = np.random.permutation(tr_g)
            tot = 0.0; nb = 0
            for s in range(0, len(order), BS):
                b = order[s:s + BS]
                w = Wt_t[b].to(device)
                if float(w.sum()) == 0:
                    continue
                xn, xc = Xn_t[b].to(device), Xc_t[b].to(device)
                xlap, msk = Xlap_t[b].to(device), Mask_t[b].to(device)
                y = Yt_t[b].to(device)
                with torch.cuda.amp.autocast():
                    logit = model(xn, xc, xlap, msk)
                    loss = (bce(logit.float(), y) * w).sum() / w.sum().clamp(min=1)
                opt.zero_grad(); gscaler.scale(loss).backward()
                gscaler.step(opt); gscaler.update()
                tot += loss.item(); nb += 1

            vlog = forward_groups(model, va_g)
            _, vp, vy = collect(va_g, vlog, want_train=True)
            va = roc_auc_score(vy, vp)
            if va > best_auc + 1e-5:
                best_auc = va; bad = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            print(f"  fold {fold} ep {ep:02d}  loss={tot/max(nb,1):.4f}  val_auc={va:.5f}  best={best_auc:.5f}", flush=True)
            if bad >= PATIENCE:
                break

        model.load_state_dict(best_state)
        vlog = forward_groups(model, va_g)
        vid, vp, _ = collect(va_g, vlog, want_train=True)
        for i_, p_ in zip(vid, vp):
            oof_by_id[int(i_)] = float(p_)
        fold_aucs.append(best_auc)

        # test laps over all groups (full context), averaged across folds
        all_g = np.arange(n_groups)
        tlog = forward_groups(model, all_g)
        tid, tp, _ = collect(all_g, tlog, want_train=False)
        pos = {int(i): k for k, i in enumerate(test_ids)}
        tarr = np.full(len(test_ids), np.nan)
        for i_, p_ in zip(tid, tp):
            j = pos.get(int(i_))
            if j is not None:
                tarr[j] = p_
        test_sum += np.nan_to_num(tarr); test_cnt += 1
        print(f"fold {fold}/{N_SPLITS}  best_val_auc={best_auc:.5f}  ({time.time()-t0:.1f}s)", flush=True)

    oof = np.array([oof_by_id.get(int(i), np.nan) for i in train[ID_COL].to_numpy(np.int64)])
    y = train[TARGET].to_numpy(int)
    valid = ~np.isnan(oof)
    oof_auc = roc_auc_score(y[valid], oof[valid])
    print(f"\nper-fold best val AUC mean={np.mean(fold_aucs):.5f} std={np.std(fold_aucs):.5f}")
    print(f"OOF AUC: {oof_auc:.5f}  (floor 0.949; vs RealMLP-ms 0.95383)  covered={valid.mean():.3f}")

    pd.DataFrame({"id": train[ID_COL], "Year": train["Year"], "target": y,
                  "oof": np.nan_to_num(oof)}).to_parquet(OOF_OUT, index=False)
    sub = pd.DataFrame({"id": test_ids, TARGET: test_sum / max(test_cnt, 1)}).sort_values("id").reset_index(drop=True)
    sub.to_csv(SUB_OUT, index=False)
    print(f"wrote {OOF_OUT.name} ({valid.sum():,} covered) and {SUB_OUT.name} ({len(sub):,})")
    print(f"total runtime {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
