#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_unified_10datasets_bgcond_residual__onehot_plus_esm.py

Fixes (relative to the version you pasted):
1) Training data stays legacy-wide with one site per row: X/Y/mask are all (N,10); no more wide_to_rows.
2) Dataset returns seq (one-hot index) + esm embedding (optional), and x/prot/global_stats/y/y_mask are all 10-dimensional.
3) Model output is changed to 10-dimensional: main + gate*delta are both (B,10).
4) main() is concise: no piling of temporary classes/functions inside main; core logic lives in global definitions.
5) Fill in the missing import (random), and provide encode_seq_to_idx.

Input (legacy-wide) must contain:
  accession, prot_seq_101aa_rep,
  PROT_WT1..5, PROT_KO1..5,
  PALM_WT1..5, PALM_KO1..5

Optional:
  --esm-npz  /path/to/esm2_3b_seq101.npz  (seqs, emb_center, emb_mean)
  --bg-npz   /path/to/background.npz (samples, combined_emb/combined) used as global_stats

Run example:
python train_unified_10datasets_bgcond_residual__onehot_plus_esm.py \
  --data  data/site_sample_wide_legacy.all.csv \
  --outdir results/run_onehot_esm \
  --esm-npz data/esm2_3b_seq101.npz \
  --esm-which center \
  --bg-npz  /path/to/bg_embeddings.npz \
  --prot-threshold 0.0 \
  --epochs 10 --batch-size 128 --lr 1e-3
"""

import os
import json
import math
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------- Config -------------------------
AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")  # 20 AA
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AA_ALPHABET)}  # 0: pad/UNK
MAX_LEN = 101
VOCAB_SIZE = 21  # 0 + 20 AAs

NUMERIC_X_COLS = [f"PROT_WT{i}" for i in range(1, 6)] + [f"PROT_KO{i}" for i in range(1, 6)]
TARGET_Y_COLS = [f"PALM_WT{i}" for i in range(1, 6)] + [f"PALM_KO{i}" for i in range(1, 6)]
SEQ_COL = "prot_seq_101aa_rep"


# ------------------------- Helpers -------------------------
def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def encode_seq_to_idx(s: str) -> np.ndarray:
    """101aa -> int64 (101,), 0 for UNK/pad."""
    s = (s or "").strip().upper()
    if len(s) < MAX_LEN:
        s = s + ("X" * (MAX_LEN - len(s)))
    elif len(s) > MAX_LEN:
        s = s[:MAX_LEN]
    arr = np.zeros((MAX_LEN,), dtype=np.int64)
    for i, aa in enumerate(s):
        arr[i] = AA_TO_IDX.get(aa, 0)
    return arr


def split_by_accession(df: pd.DataFrame, seed: int, test_ratio: float = 0.2, val_ratio_in_train: float = 0.2):
    rng = np.random.default_rng(seed)
    accs = df["accession"].dropna().astype(str).unique()
    rng.shuffle(accs)
    n = len(accs)
    n_test = max(1, int(round(n * test_ratio)))
    test_acc = set(accs[:n_test])
    train_acc = set(accs[n_test:])

    df_test = df[df["accession"].astype(str).isin(test_acc)].reset_index(drop=True)
    df_train_all = df[df["accession"].astype(str).isin(train_acc)].reset_index(drop=True)

    acc_train = df_train_all["accession"].astype(str).unique()
    rng.shuffle(acc_train)
    n_val = max(1, int(round(len(acc_train) * val_ratio_in_train)))
    val_acc = set(acc_train[:n_val])

    df_val = df_train_all[df_train_all["accession"].astype(str).isin(val_acc)].reset_index(drop=True)
    df_train = df_train_all[~df_train_all["accession"].astype(str).isin(val_acc)].reset_index(drop=True)
    return df_train, df_val, df_test


def build_global_prot_stats_vector(df: pd.DataFrame, rep_cols: List[str]) -> np.ndarray:
    vecs = []
    for c in rep_cols:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        x = x[~np.isnan(x)]
        if x.size == 0:
            vecs.append(np.zeros(7, dtype=np.float32))
        else:
            q10, q50, q90 = np.percentile(x, [10, 50, 90])
            vecs.append(np.array(
                [np.mean(x), np.std(x), q50, q10, q90, np.min(x), np.max(x)],
                dtype=np.float32
            ))
    g = np.concatenate(vecs, axis=0).astype(np.float32)
    g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    return g


def load_bg_embedding_npz(bg_npz_path: str,
                          sample_cols: List[str],
                          key_name: str = "combined_emb") -> np.ndarray:
    """
    Read each sample's embedding from the background embedding npz, and concatenate them in sample_cols order into a (len(sample_cols)*d,) vector.
    The npz must contain:
      - samples: (S,)
      - combined_emb or combined: (S, d)
    """
    bg_npz = np.load(bg_npz_path, allow_pickle=True)
    samples = bg_npz["samples"].astype(str)

    if key_name in bg_npz:
        emb = bg_npz[key_name]
    elif "combined" in bg_npz:
        emb = bg_npz["combined"]
    else:
        raise ValueError(f"bg npz missing '{key_name}' or 'combined': {bg_npz_path}")

    d = emb.shape[1]
    cond_list = []
    missing = []
    for c in sample_cols:
        idx = np.where(samples == c)[0]
        if idx.size == 0:
            missing.append(c)
        else:
            cond_list.append(emb[idx[0]])
    if missing:
        raise ValueError(f"[BG] samples missing in bg npz: {missing}")

    cond_mat = np.stack(cond_list, axis=0)  # (10, d)
    return cond_mat.reshape(-1).astype(np.float32)  # (10*d,)


def load_esm_npz_as_dict(npz_path: str, which: str = "center"):
    """
    Load esm2_3b_seq101.npz:
      seqs: (N,) object
      emb_center: (N, d)
      emb_mean:   (N, d)
    Return:
      dict[seq101] -> (d,) float32
      dim
    """
    z = np.load(npz_path, allow_pickle=True)
    seqs = z["seqs"].astype(object)

    if which == "center":
        emb = z["emb_center"]
    elif which == "mean":
        emb = z["emb_mean"]
    elif which == "center+mean":
        emb = np.concatenate([z["emb_center"], z["emb_mean"]], axis=1)
    else:
        raise ValueError("esm-which must be center / mean / center+mean")

    emb = emb.astype(np.float32, copy=False)

    d = {}
    for i, s in enumerate(seqs):
        ss = str(s).strip().upper()
        if ss:
            d[ss] = emb[i]
    return d, emb.shape[1]


def batch_metrics_with_mask(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray,
                            names: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    r2s, rmses, pears, spear = [], [], [], []
    for j, name in enumerate(names):
        m = mask[:, j] > 0.5
        if m.sum() < 3:
            out[name] = {"RMSE": np.nan, "R2": np.nan, "Pearson": np.nan, "Spearman": np.nan}
            continue
        yt = y_true[m, j]
        yp = y_pred[m, j]
        rmse = math.sqrt(mean_squared_error(yt, yp))
        try:
            pr = pearsonr(yt, yp)[0]
        except Exception:
            pr = np.nan
        try:
            sr = spearmanr(yt, yp).correlation
        except Exception:
            sr = np.nan
        try:
            r2 = r2_score(yt, yp)
        except Exception:
            r2 = np.nan
        out[name] = {"RMSE": float(rmse), "R2": float(r2) if not np.isnan(r2) else np.nan,
                     "Pearson": float(pr), "Spearman": float(sr)}
        rmses.append(rmse)
        r2s.append(r2)
        pears.append(pr)
        spear.append(sr)

    out["__macro__"] = {
        "RMSE": float(np.nanmean(rmses)),
        "R2": float(np.nanmean(r2s)),
        "Pearson": float(np.nanmean(pears)),
        "Spearman": float(np.nanmean(spear))
    }
    return out


# ------------------------- Dataset -------------------------
class ProteinWideDataset(Dataset):
    """
    One row = one site.
    Returns:
      seq: (101,) long
      esm: (D,) float (optional)
      x: (10,) float
      prot_this: (10,) float
      global_stats: (G,) float
      y: (10,) float
      y_mask: (10,) float
      accession: str
    """
    def __init__(self,
                 df: pd.DataFrame,
                 seq_col: str,
                 x_np: np.ndarray,
                 prot_raw_np: np.ndarray,
                 global_stats: np.ndarray,
                 y_np: np.ndarray,
                 y_mask_np: np.ndarray,
                 esm_dict: Optional[Dict[str, np.ndarray]] = None,
                 esm_dim: int = 0):
        self.df = df.reset_index(drop=True)
        self.acc = self.df["accession"].astype(str).tolist()
        self.seqs = self.df[seq_col].astype(str).str.upper().tolist()

        self.x = x_np.astype(np.float32)
        self.prot_raw = prot_raw_np.astype(np.float32)
        self.global_stats = global_stats.astype(np.float32)

        self.y = y_np.astype(np.float32)
        self.y_mask = y_mask_np.astype(np.float32)

        self.esm_dict = esm_dict
        self.esm_dim = int(esm_dim)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        s = self.seqs[i]
        seq_idx = encode_seq_to_idx(s)

        if self.esm_dict is not None:
            v = self.esm_dict.get(s, None)
            if v is None:
                esm = np.zeros((self.esm_dim,), dtype=np.float32)
                esm_ok = 0
            else:
                esm = v.astype(np.float32, copy=False)
                esm_ok = 1
        else:
            esm = np.zeros((0,), dtype=np.float32)
            esm_ok = 0

        return {
            "accession": self.acc[i],
            "seq": torch.from_numpy(seq_idx),                      # (101,)
            "esm": torch.from_numpy(esm),                          # (D,) or (0,)
            "esm_ok": torch.tensor(esm_ok, dtype=torch.long),
            "x": torch.from_numpy(self.x[i]),                      # (10,)
            "prot_this": torch.from_numpy(self.prot_raw[i]),       # (10,)
            "global_stats": torch.from_numpy(self.global_stats),   # (G,)
            "y": torch.from_numpy(self.y[i]),                      # (10,)
            "y_mask": torch.from_numpy(self.y_mask[i]),            # (10,)
        }


# ------------------------- Model -------------------------
class SeqCNN_Num_KwiseGate_10(nn.Module):
    """
    Output 10-dimensional: yhat = prot_main(x) + sigmoid(gate(x,cond))*delta(seq,x,esm,cond)
    """
    def __init__(self,
                 vocab_size: int,
                 seq_emb_dim: int,
                 cnn_channels: int,
                 kernel_sizes: List[int],
                 num_input_dim: int,
                 num_hidden_dim: int,
                 fusion_dim: int,
                 *,
                 cond_dim: int = 0,          # NEW: global_stats dim
                 cond_proj_dim: int = 128,   # NEW: projected condition dim
                 esm_dim: int = 0,
                 esm_proj_dim: int = 256,
                 dropout: float = 0.2):
        super().__init__()

        # seq branch
        self.seq_emb = nn.Embedding(vocab_size, seq_emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(seq_emb_dim, cnn_channels, k) for k in kernel_sizes])
        self.seq_out_dim = cnn_channels * len(kernel_sizes)

        # numeric branch
        self.num = nn.Sequential(
            nn.Linear(num_input_dim, num_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # condition (global_stats) branch  -------- NEW
        self.cond_dim = int(cond_dim)
        if self.cond_dim > 0:
            self.cond_proj = nn.Sequential(
                nn.Linear(self.cond_dim, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, cond_proj_dim),
                nn.GELU(),
            )
            cond_out = cond_proj_dim
        else:
            self.cond_proj = None
            cond_out = 0

        # ESM branch
        self.esm_dim = int(esm_dim)
        if self.esm_dim > 0:
            self.esm_proj = nn.Sequential(
                nn.Linear(self.esm_dim, 1024),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(1024, esm_proj_dim),
                nn.GELU(),
            )
            esm_out = esm_proj_dim
        else:
            self.esm_proj = None
            esm_out = 0

        # fusion
        fuse_in = self.seq_out_dim + num_hidden_dim + esm_out + cond_out
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # heads -> 10 dims
        self.delta_head = nn.Linear(fusion_dim, num_input_dim)   # (B,10)

        # let the gate see cond as well (optional but recommended) -------- NEW
        gate_in = num_input_dim + cond_out
        self.res_gate = nn.Linear(gate_in, num_input_dim)        # (B,10)

        self.prot_main = nn.Linear(num_input_dim, num_input_dim) # (B,10)

    def _seq_branch(self, seq_idx: torch.Tensor) -> torch.Tensor:
        x = self.seq_emb(seq_idx).transpose(1, 2)  # (B, C, L)
        outs = []
        for conv in self.convs:
            y = torch.relu(conv(x))
            y = torch.max(y, dim=2)[0]
            outs.append(y)
        return torch.cat(outs, dim=1)

    def forward_aux(self,
                    seq: torch.Tensor,
                    x_num: torch.Tensor,
                    prot_this: torch.Tensor,
                    global_stats: torch.Tensor,
                    esm: Optional[torch.Tensor] = None):
        # seq / num
        seq_feat = self._seq_branch(seq)
        num_feat = self.num(x_num)

        feats = [seq_feat, num_feat]

        # condition (global_stats) -------- NEW
        cond_feat = None
        if self.cond_proj is not None:
            # global_stats: both (G,) and (B,G) are supported
            if global_stats.dim() == 1:
                gs = global_stats.unsqueeze(0).expand(x_num.size(0), -1)  # (B,G)
            else:
                gs = global_stats
            cond_feat = self.cond_proj(gs)  # (B,cond_out)
            feats.append(cond_feat)

        # esm
        if self.esm_proj is not None:
            assert esm is not None, "esm_dim>0 but esm is None"
            feats.append(self.esm_proj(esm))

        z = self.fuse(torch.cat(feats, dim=1))  # (B, fusion_dim)

        delta = self.delta_head(z)              # (B,10)

        # gate uses x + cond (if available) -------- NEW
        if cond_feat is not None:
            gate_in = torch.cat([x_num, cond_feat], dim=1)
        else:
            gate_in = x_num
        gate = torch.sigmoid(self.res_gate(gate_in))  # (B,10)

        main = self.prot_main(x_num)            # (B,10)

        yhat = main + gate * delta
        return yhat, delta, gate, main

    def forward(self, seq, x_num, prot_this, global_stats, esm=None):
        yhat, _, _, _ = self.forward_aux(seq, x_num, prot_this, global_stats, esm=esm)
        return yhat


# ------------------------- Loss -------------------------
def masked_mse(yhat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # yhat,y,mask: (B,10)
    denom = torch.clamp(mask.sum(), min=1.0)
    return (((yhat - y) ** 2) * mask).sum() / denom


def train_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer,
                device: torch.device, lambda_delta: float = 0.0) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        seq = batch["seq"].to(device, non_blocking=True)
        x = batch["x"].to(device, non_blocking=True).float()
        prot_this = batch["prot_this"].to(device, non_blocking=True).float()
        gstats = batch["global_stats"].to(device, non_blocking=True).float()
        y = batch["y"].to(device, non_blocking=True).float()
        mask = batch["y_mask"].to(device, non_blocking=True).float()
        esm = batch["esm"].to(device, non_blocking=True) if batch["esm"].numel() > 0 else None

        opt.zero_grad(set_to_none=True)
        yhat, delta, gate, _ = model.forward_aux(seq, x, prot_this, gstats, esm=esm)
        loss = masked_mse(yhat, y, mask)
        if lambda_delta > 0:
            loss = loss + lambda_delta * torch.mean((gate * delta) ** 2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        bs = y.size(0)
        total += float(loss.item()) * bs
        n += bs
    return total / max(1, n)


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device, lambda_delta: float = 0.0):
    model.eval()
    total, n = 0.0, 0
    preds, trues, masks = [], [], []
    for batch in loader:
        seq = batch["seq"].to(device, non_blocking=True)
        x = batch["x"].to(device, non_blocking=True).float()
        prot_this = batch["prot_this"].to(device, non_blocking=True).float()
        gstats = batch["global_stats"].to(device, non_blocking=True).float()
        y = batch["y"].to(device, non_blocking=True).float()
        mask = batch["y_mask"].to(device, non_blocking=True).float()
        esm = batch["esm"].to(device, non_blocking=True) if batch["esm"].numel() > 0 else None

        yhat, delta, gate, _ = model.forward_aux(seq, x, prot_this, gstats, esm=esm)
        loss = masked_mse(yhat, y, mask)
        if lambda_delta > 0:
            loss = loss + lambda_delta * torch.mean((gate * delta) ** 2)

        bs = y.size(0)
        total += float(loss.item()) * bs
        n += bs

        preds.append(yhat.detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())
        masks.append(mask.detach().cpu().numpy())

    loss = total / max(1, n)
    y_pred = np.concatenate(preds, axis=0) if preds else np.zeros((0, len(TARGET_Y_COLS)), dtype=np.float32)
    y_true = np.concatenate(trues, axis=0) if trues else np.zeros((0, len(TARGET_Y_COLS)), dtype=np.float32)
    y_mask = np.concatenate(masks, axis=0) if masks else np.zeros_like(y_true)
    return loss, y_true, y_pred, y_mask


# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="legacy-wide csv/parquet")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num-workers", type=int, default=4)

    ap.add_argument("--prot-threshold", type=float, default=0.0)
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--val-ratio-in-train", type=float, default=0.2)

    # ESM
    ap.add_argument("--esm-npz", type=str, default=None)
    ap.add_argument("--esm-which", type=str, default="center",
                    choices=["center", "mean", "center+mean"])
    ap.add_argument("--esm-proj-dim", type=int, default=256)

    # BG (optional)
    ap.add_argument("--bg-npz", type=str, default=None)
    ap.add_argument("--bg-key", type=str, default="combined_emb")

    # model hparams
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seq-emb-dim", type=int, default=32)
    ap.add_argument("--cnn-ch", type=int, default=128)
    ap.add_argument("--kernels", type=str, default="5,7,9")
    ap.add_argument("--num-hid", type=int, default=256)
    ap.add_argument("--fusion-dim", type=int, default=512)

    ap.add_argument("--lambda-delta", type=float, default=0.0)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)

    device = args.device
    if device.startswith("cuda") and (not torch.cuda.is_available()):
        device = "cpu"
    device_t = torch.device(device)
    print("[device]", device_t)

    # read
    if str(args.data).endswith(".parquet"):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)

    need = ["accession", SEQ_COL] + NUMERIC_X_COLS + TARGET_Y_COLS
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"[data] missing columns: {miss}")

    # numeric clean
    for c in NUMERIC_X_COLS + TARGET_Y_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)

    # split
    df_train, df_val, df_test = split_by_accession(
        df, seed=args.seed, test_ratio=args.test_ratio, val_ratio_in_train=args.val_ratio_in_train
    )
    print(f"[split] train={len(df_train)} val={len(df_val)} test={len(df_test)}")

    # global stats / bg embedding
    if args.bg_npz:
        global_stats = load_bg_embedding_npz(args.bg_npz, sample_cols=NUMERIC_X_COLS, key_name=args.bg_key)
        print(f"[BG] global_stats dim={global_stats.shape[0]}")
    else:
        global_stats = build_global_prot_stats_vector(df_train, rep_cols=NUMERIC_X_COLS)
        print(f"[BG] (fallback stats) global_stats dim={global_stats.shape[0]}")

    # esm
    esm_dict, esm_dim = None, 0
    if args.esm_npz:
        esm_dict, esm_dim = load_esm_npz_as_dict(args.esm_npz, which=args.esm_which)
        print(f"[ESM] loaded {len(esm_dict)} seqs, dim={esm_dim}, which={args.esm_which}")
    else:
        print("[ESM] disabled")

    # build arrays (N,10)
    def build_xy_mask(df_w: pd.DataFrame):
        X = df_w[NUMERIC_X_COLS].to_numpy(dtype=np.float32)  # (N,10)
        Y = df_w[TARGET_Y_COLS].to_numpy(dtype=np.float32)   # (N,10)
        M = ((Y > 0.0) & (X > float(args.prot_threshold))).astype(np.float32)  # (N,10)
        return X, Y, M

    X_tr, Y_tr, M_tr = build_xy_mask(df_train)
    X_va, Y_va, M_va = build_xy_mask(df_val)
    X_te, Y_te, M_te = build_xy_mask(df_test)

    prot_tr = X_tr.copy()
    prot_va = X_va.copy()
    prot_te = X_te.copy()

    # dataset/loader
    pin = device_t.type == "cuda"
    train_ds = ProteinWideDataset(df_train, SEQ_COL, X_tr, prot_tr, global_stats, Y_tr, M_tr, esm_dict, esm_dim)
    val_ds = ProteinWideDataset(df_val, SEQ_COL, X_va, prot_va, global_stats, Y_va, M_va, esm_dict, esm_dim)
    test_ds = ProteinWideDataset(df_test, SEQ_COL, X_te, prot_te, global_stats, Y_te, M_te, esm_dict, esm_dim)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=pin, drop_last=False)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=pin, drop_last=False)
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=pin, drop_last=False)

    # model
    kernels = [int(x) for x in args.kernels.split(",") if x.strip()]
    model = SeqCNN_Num_KwiseGate_10(
        vocab_size=VOCAB_SIZE,
        seq_emb_dim=args.seq_emb_dim,
        cnn_channels=args.cnn_ch,
        kernel_sizes=kernels,
        num_input_dim=len(NUMERIC_X_COLS),  # 10
        num_hidden_dim=args.num_hid,
        fusion_dim=args.fusion_dim,
        cond_dim=int(global_stats.shape[0]),  # NEW
        cond_proj_dim=128,  # NEW (could be exposed as an arg)
        esm_dim=esm_dim,
        esm_proj_dim=args.esm_proj_dim,
        dropout=args.dropout,
    ).to(device_t)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # train
    best_val = None
    best_path = outdir / "best.pt"

    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_ld, opt, device_t, lambda_delta=args.lambda_delta)
        va_loss, _, _, _ = eval_epoch(model, val_ld, device_t, lambda_delta=args.lambda_delta)
        print(f"[Epoch {ep}] train_loss={tr_loss:.6f} val_loss={va_loss:.6f}")

        if (best_val is None) or (va_loss < best_val):
            best_val = va_loss
            torch.save({"model": model.state_dict(), "args": vars(args)}, str(best_path))
            print(f"[SAVE] best -> {best_path}")

    # test
    te_loss, y_true, y_pred, y_mask = eval_epoch(model, test_ld, device_t, lambda_delta=args.lambda_delta)
    print(f"[TEST] loss={te_loss:.6f}")

    metrics = batch_metrics_with_mask(y_true, y_pred, y_mask, names=TARGET_Y_COLS)
    (outdir / "metrics_test.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("[DONE] metrics_test.json")

    last_path = outdir / "last.pt"
    torch.save({"model": model.state_dict(), "args": vars(args)}, str(last_path))
    print(f"[DONE] last -> {last_path}")
    print(f"[DONE] best -> {best_path}  best_val={best_val:.6f}")


if __name__ == "__main__":
    main()
