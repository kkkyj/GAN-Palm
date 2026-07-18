#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
conditional_seq_gan_noesm.py  (site_sample_long.* version, NO ESM)

Goals:
- Generator: takes bg/cell embedding as input (here bg_vec represents the condition vector), outputs a 101aa sequence (center position forced to C)
- Discriminator: uses only the token path (Transformer + FiLM(bg)), no longer introduces ESM embedding
- Data: supports your current parquet format (seq101 + label_bin), also supports explicit --seq-col / --label-col
"""

import argparse
import json
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
# ===================== Constants =====================
SEQ_LEN = 101
CENTER_POS = 50  # 0-based in 101aa window
VOCAB = list("ACDEFGHIKLMNPQRSTVWYBX")  # 20AA + B + X(unknown)
AA2IDX = {a: i for i, a in enumerate(VOCAB)}
IDX2AA = {i: a for a, i in AA2IDX.items()}
PAD_IDX = AA2IDX.get("X", len(VOCAB) - 1)
VOCAB_SIZE = len(VOCAB)

NUMERIC_X_COLS = [
    "PALM_WT1", "PALM_WT2", "PALM_WT3", "PALM_WT4", "PALM_WT5",
    "PALM_KO1", "PALM_KO2", "PALM_KO3", "PALM_KO4", "PALM_KO5",
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_seq101(s: str) -> str:
    s = (s or "").upper().strip()
    if len(s) >= SEQ_LEN:
        s = s[:SEQ_LEN]
    else:
        s = s + "X" * (SEQ_LEN - len(s))
    s = list(s)
    s[CENTER_POS] = "C"
    return "".join(s)


def encode_seq(seq101: str) -> np.ndarray:
    s = normalize_seq101(seq101)
    out = np.full((SEQ_LEN,), PAD_IDX, dtype=np.int64)
    for i, ch in enumerate(s):
        out[i] = AA2IDX.get(ch, PAD_IDX)
    out[CENTER_POS] = AA2IDX["C"]
    return out


def tokens_to_seq101(tokens: np.ndarray) -> str:
    chars = []
    for t in tokens[:SEQ_LEN]:
        t = int(t)
        chars.append(IDX2AA.get(t, "X") if t != PAD_IDX else "X")
    if len(chars) < SEQ_LEN:
        chars += ["X"] * (SEQ_LEN - len(chars))
    chars[CENTER_POS] = "C"
    return "".join(chars)


def build_background_vector(df_protein: pd.DataFrame, bins: int = 24) -> np.ndarray:
    x = df_protein[NUMERIC_X_COLS].to_numpy(dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, None)

    feats = []
    for j in range(x.shape[1]):
        v = np.log1p(x[:, j])
        if np.all(v == 0):
            feats.append(np.zeros((bins,), dtype=np.float32))
            continue
        lo, hi = float(v.min()), float(v.max())
        if hi <= lo + 1e-8:
            h = np.zeros((bins,), dtype=np.float32)
            h[0] = 1.0
            feats.append(h)
            continue
        hist, _ = np.histogram(v, bins=bins, range=(lo, hi), density=True)
        feats.append(hist.astype(np.float32))
    bg = np.concatenate(feats, axis=0).astype(np.float32)
    bg = bg / (np.linalg.norm(bg) + 1e-8)
    return bg


def load_bg_vec(bg_npy: str, bg_npz: str, protein_table: str, bins: int) -> np.ndarray:
    if bg_npy and Path(bg_npy).exists():
        bg = np.load(bg_npy).astype(np.float32)
        if bg.ndim != 1:
            raise ValueError(f"--bg-npy expects 1D, got {bg.shape}")
        return bg

    if bg_npz and Path(bg_npz).exists():
        z = np.load(bg_npz, allow_pickle=True)

        for k in ["bg_vec", "bg", "background", "mean_bg", "emb", "embedding"]:
            if k in z.files:
                arr = z[k]
                if hasattr(arr, "ndim") and arr.ndim == 1 and arr.size > 0:
                    return arr.astype(np.float32)

        if "combined_emb" in z.files:
            arr = z["combined_emb"]
            if hasattr(arr, "ndim") and arr.ndim == 2 and arr.size > 0:
                return arr.mean(axis=0).astype(np.float32)

        # fallback: only one 1D
        cand = []
        for k in z.files:
            arr = z[k]
            if hasattr(arr, "ndim") and arr.ndim == 1 and arr.size > 0:
                cand.append(arr)
        if len(cand) == 1:
            return cand[0].astype(np.float32)

        raise ValueError(f"Cannot parse bg vector from {bg_npz}. keys={z.files}")

    if protein_table and Path(protein_table).exists():
        suf = Path(protein_table).suffix.lower()
        dfp = pd.read_parquet(protein_table) if suf == ".parquet" else pd.read_csv(protein_table)
        for c in NUMERIC_X_COLS:
            if c not in dfp.columns:
                raise ValueError(f"protein-table missing {c}")
        dfp[NUMERIC_X_COLS] = dfp[NUMERIC_X_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        dfp[NUMERIC_X_COLS] = dfp[NUMERIC_X_COLS].clip(lower=0.0)
        return build_background_vector(dfp, bins=bins)

    raise FileNotFoundError("Provide --bg-npy or --bg-npz (recommended), or a valid --protein-table.")


class SiteSeqDataset(Dataset):
    def __init__(self, site_table: str, seq_col: str = "seq101", label_col: str = "label_bin"):
        suf = Path(site_table).suffix.lower()
        df = pd.read_parquet(site_table) if suf == ".parquet" else pd.read_csv(site_table)

        if seq_col not in df.columns:
            raise ValueError(f"--seq-col '{seq_col}' not found. Available: {list(df.columns)}")
        if label_col not in df.columns:
            # fallback to common names
            for c in ["label_bin", "label", "y"]:
                if c in df.columns:
                    label_col = c
                    break
            else:
                raise ValueError(f"--label-col '{label_col}' not found and no fallback. Available: {list(df.columns)}")

        seqs = [normalize_seq101(s) for s in df[seq_col].astype(str).tolist()]
        tokens = np.stack([encode_seq(s) for s in seqs], axis=0).astype(np.int64)
        labels = pd.to_numeric(df[label_col], errors="coerce").fillna(0.0).astype(np.float32).to_numpy()

        self.seqs = seqs
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return int(self.tokens.shape[0])

    def __getitem__(self, idx: int):
        return {
            "tokens": torch.from_numpy(self.tokens[idx]),
            "label": torch.tensor(self.labels[idx]),
            "seq101": self.seqs[idx],
        }


def collate(batch: List[dict]):
    tokens = torch.stack([b["tokens"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    seqs = [b["seq101"] for b in batch]
    return {"tokens": tokens, "label": labels, "seq101": seqs}


class Generator(nn.Module):
    def __init__(self, bg_dim: int, d_model=256, nhead=8, nlayers=4, dff=512, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_IDX)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, d_model) * 0.02)

        self.film = nn.Sequential(
            nn.Linear(bg_dim, d_model * 2), nn.ReLU(),
            nn.Linear(d_model * 2, d_model * 2),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dff,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.out = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, tokens: torch.Tensor, bg_vec: torch.Tensor):
        x = self.emb(tokens) + self.pos[:tokens.size(1)].unsqueeze(0)
        gamma, beta = self.film(bg_vec).chunk(2, dim=-1)
        x = x * (1 + torch.tanh(gamma.unsqueeze(1))) + beta.unsqueeze(1)
        h = self.enc(x)
        return self.out(h)

    def sample_gumbel_soft(self, bg_vec: torch.Tensor, tau: float = 1.0):
        B = bg_vec.size(0)
        device = bg_vec.device
        start = torch.full((B, SEQ_LEN), AA2IDX.get("M", PAD_IDX), dtype=torch.long, device=device)
        logits = self.forward(start, bg_vec)
        y = nn.functional.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)

        # center clamp
        V = y.size(-1)
        center_onehot = torch.zeros((B, V), device=device, dtype=y.dtype)
        center_onehot[:, AA2IDX["C"]] = 1.0
        center_onehot = center_onehot.unsqueeze(1)

        mask = torch.ones((1, SEQ_LEN, 1), device=device, dtype=y.dtype)
        mask[:, CENTER_POS:CENTER_POS + 1, :] = 0.0
        return y * mask + center_onehot * (1.0 - mask)

    @torch.no_grad()
    def sample_discrete(self, bg_vec: torch.Tensor, start_token: int, temperature: float = 1.0, top_k: int = 0):
        B = bg_vec.size(0)
        device = bg_vec.device
        tokens = torch.full((B, SEQ_LEN), start_token, dtype=torch.long, device=device)
        logits = self.forward(tokens, bg_vec) / max(1e-6, float(temperature))

        if top_k and top_k > 0:
            v, ix = torch.topk(logits, k=int(top_k), dim=-1)
            m = torch.full_like(logits, float("-inf"))
            m.scatter_(-1, ix, v)
            logits = m

        probs = torch.softmax(logits, dim=-1)
        samp = torch.multinomial(probs.view(-1, VOCAB_SIZE), num_samples=1).view(B, SEQ_LEN)
        samp[:, CENTER_POS] = AA2IDX["C"]
        return samp



class ConditionalSequenceDisc(nn.Module):
    """
    CNN-based conditional discriminator (more stable than Transformer).
    Interface kept compatible:
        forward(tokens_or_soft, bg_vec, is_soft=False) -> logits (B,)
    If is_soft=True, tokens_or_soft is expected to be (B,L,V) soft one-hot.
    """
    def __init__(
        self,
        bg_dim: int,
        emb_dim: int = 64,
        channels: int = 128,
        kernels=(3, 5, 7),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bg_dim = int(bg_dim)

        self.emb = nn.Embedding(VOCAB_SIZE, emb_dim, padding_idx=PAD_IDX)

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(emb_dim, channels, kernel_size=int(k), padding=int(k) // 2),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for k in kernels
        ])

        # normalize bg scale (cheap + helps stability)
        self.bg_ln = nn.LayerNorm(self.bg_dim, eps=1e-5) if self.bg_dim > 0 else None

        feat_dim = channels * len(kernels) + (self.bg_dim if self.bg_dim > 0 else 0)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        nn.init.zeros_(self.mlp[-1].bias)

    def _embed(self, tokens_or_soft: torch.Tensor, is_soft: bool) -> torch.Tensor:
        # return x: (B,L,E)
        if is_soft:
            # (B,L,V) @ (V,E) -> (B,L,E)
            return torch.matmul(tokens_or_soft, self.emb.weight)
        return self.emb(tokens_or_soft)

    def forward(self, tokens_or_soft: torch.Tensor, bg_vec: torch.Tensor, is_soft: bool = False):
        x = self._embed(tokens_or_soft, is_soft=is_soft)  # (B,L,E)
        x = x.transpose(1, 2)  # (B,E,L)

        feats = []
        for block in self.convs:
            h = block(x)               # (B,C,L)
            h = torch.amax(h, dim=-1)  # global max pool -> (B,C)
            feats.append(h)
        h = torch.cat(feats, dim=1)    # (B, C * nk)

        if self.bg_dim > 0:
            bg2 = self.bg_ln(bg_vec)
            h = torch.cat([h, bg2], dim=1)

        logit = self.mlp(h).squeeze(-1)  # (B,)
        return logit


def bce_loss(logits, targets):
    return nn.functional.binary_cross_entropy_with_logits(logits, targets)


def center_penalty(tokens_soft: torch.Tensor, weight: float = 1.0):
    p_c = tokens_soft[:, CENTER_POS, AA2IDX["C"]]
    return weight * (1.0 - p_c).mean()


def seq_mlm_loss(G: Generator, batch_tokens: torch.Tensor, bg: torch.Tensor,
                 mask_ratio: float = 0.15):
    """
    Masked LM warmup for encoder-based generator.
    Predict original token at masked positions only.
    """
    x = batch_tokens.clone()
    B, L = x.shape
    device = x.device

    # do not mask center C
    mask = (torch.rand((B, L), device=device) < mask_ratio)
    mask[:, CENTER_POS] = False

    # replace masked tokens with PAD_IDX (or X) as sentinel
    x[mask] = PAD_IDX

    logits = G.forward(x, bg)  # (B,L,V)
    # compute CE only on masked positions
    tgt = batch_tokens[mask]
    pred = logits[mask]  # (Nmask,V)

    if tgt.numel() == 0:
        # rare corner: no masked token, fall back
        return nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            batch_tokens.reshape(-1),
            ignore_index=PAD_IDX
        )

    return nn.functional.cross_entropy(pred, tgt, ignore_index=PAD_IDX)



def tokens_to_fasta(tokens: np.ndarray, path: Path, header_prefix: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, row in enumerate(tokens):
            f.write(f">{header_prefix}_{i}\n{tokens_to_seq101(row)}\n")


# ===================== Plotting helpers =====================
def save_loss_plot(
    logs: list,
    outdir: Path,
    fname: str = "loss_curve.png",
    smooth_window: int = 1,
):
    """
    Save loss curves from `logs` (list of dicts) into outdir/fname.

    Expected log keys (some may be missing):
      - step, epoch
      - loss_D, loss_G, adv, lm, center
    """
    if not logs:
        print("[Warn] logs is empty; skip plotting.")
        return

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Extract series
    steps = np.array([d.get("step", i) for i, d in enumerate(logs)], dtype=np.int64)

    def get_series(key: str):
        vals = [d.get(key, np.nan) for d in logs]
        return np.array(vals, dtype=np.float64)

    series = {
        "loss_D": get_series("loss_D"),
        "loss_G": get_series("loss_G"),
        "adv": get_series("adv"),
        "lm": get_series("lm"),
        "center": get_series("center"),
    }

    def moving_avg(x: np.ndarray, w: int):
        if w is None or int(w) <= 1:
            return x
        w = int(w)
        # ignore nan by simple nanmean convolution
        # fallback: replace nan with nearest valid
        y = x.copy()
        if np.isnan(y).all():
            return y
        # fill nan with last valid
        isn = np.isnan(y)
        if isn.any():
            idx = np.where(~isn)[0]
            first = idx[0]
            y[:first] = y[first]
            for i in range(first + 1, len(y)):
                if np.isnan(y[i]):
                    y[i] = y[i - 1]
        kernel = np.ones(w, dtype=np.float64) / w
        return np.convolve(y, kernel, mode="same")

    # Plot each metric in its own figure (clean, no subplots per requirement)
    for key, vals in series.items():
        if np.isnan(vals).all():
            continue
        plt.figure()
        y = moving_avg(vals, smooth_window)
        plt.plot(steps, y)
        plt.xlabel("step")
        plt.ylabel(key)
        plt.title(key)
        plt.tight_layout()
        outpath = outdir / f"{Path(fname).stem}_{key}{Path(fname).suffix}"
        plt.savefig(outpath, dpi=150)
        plt.close()

    print(f"[OK] saved plots to: {outdir} (files: {Path(fname).stem}_*.{Path(fname).suffix.lstrip('.')})")


def dump_logs_jsonl(logs: list, outdir: Path, fname: str = "train_loss.jsonl"):
    """
    Save logs as JSONL for easy downstream plotting.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / fname
    with open(path, "w", encoding="utf-8") as f:
        for d in logs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"[OK] wrote: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein-table", default="", help="optional protein-level table for bg_vec")
    ap.add_argument("--site-table", required=True, help="site_sample_long.*.parquet or .csv")
    ap.add_argument("--seq-col", default="seq101")
    ap.add_argument("--label-col", default="label_bin")

    ap.add_argument("--bg-npy", default="")
    ap.add_argument("--bg-npz", default="")
    ap.add_argument("--bins", type=int, default=24)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr-g", type=float, default=2e-4)
    ap.add_argument("--lr-d", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--lambda-lm", type=float, default=0.2)
    ap.add_argument("--lambda-center", type=float, default=0.5)
    ap.add_argument("--d-grad-clip", type=float, default=1.0)
    ap.add_argument("--d-warmup-epochs", type=int, default=2,
                    help="pretrain discriminator on real pos/neg only")

    args = ap.parse_args()
    set_seed(args.seed)

    outdir = Path(args.outdir)
    (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (outdir / "samples").mkdir(parents=True, exist_ok=True)

    bg_vec = load_bg_vec(args.bg_npy, args.bg_npz, args.protein_table, bins=args.bins)
    bg_dim = int(bg_vec.shape[0])
    bg_vec = np.asarray(bg_vec, dtype=np.float32)
    bg_vec = np.nan_to_num(bg_vec, nan=0.0, posinf=0.0, neginf=0.0)

    # key: compress the scale (both steps recommended)
    bg_vec = bg_vec / (np.linalg.norm(bg_vec) + 1e-8)  # L2 normalize
    bg_vec = np.clip(bg_vec, -5.0, 5.0)  # optional: an additional hard-clip safeguard



    ds = SiteSeqDataset(args.site_table, seq_col=args.seq_col, label_col=args.label_col)

    def _collate(b):
        base = collate(b)
        base["bg"] = torch.tensor(bg_vec).float().unsqueeze(0).repeat(base["tokens"].size(0), 1)
        return base

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        collate_fn=_collate, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(bg_dim=bg_dim).to(device)
    D = ConditionalSequenceDisc(bg_dim=bg_dim).to(device)

    optG = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    logs = []
    step = 0

    for epoch in range(1, args.epochs + 1):
        for batch in loader:
            step += 1
            real_tok = batch["tokens"].to(device)
            labels = batch["label"].to(device)
            bg = batch["bg"].to(device)
            # -------------------------
            # Phase 0: G warmup (MLM)
            # -------------------------
            if epoch <= args.warmup_epochs:
                G.train()
                optG.zero_grad(set_to_none=True)
                lm = seq_mlm_loss(G, real_tok, bg, mask_ratio=0.15)
                lm.backward()
                optG.step()
                if step % 200 == 0:
                    print(f"[G-warmup step {step}] lm={lm.item():.4f}")
                continue

            # -------------------------
            # Phase 1: D warmup (real pos/neg only) AFTER G warmup
            # epochs: (warmup_epochs, warmup_epochs + d_warmup_epochs]
            # -------------------------
            if args.d_warmup_epochs > 0 and epoch <= args.warmup_epochs + args.d_warmup_epochs:
                D.train()
                optD.zero_grad(set_to_none=True)

                y = (labels > 0.5).float().view(-1)   # use labels, not undefined y
                logit = D(real_tok, bg, is_soft=False).view(-1)
                lossD = bce_loss(logit, y)

                lossD.backward()
                if args.d_grad_clip and args.d_grad_clip > 0:
                    nn.utils.clip_grad_norm_(D.parameters(), float(args.d_grad_clip))
                optD.step()

                if step % 200 == 0:
                    with torch.no_grad():
                        prob = torch.sigmoid(logit)
                        acc = ((prob >= 0.5) == (y >= 0.5)).float().mean().item()
                    print(f"[D-warmup step {step}] loss={lossD.item():.4f} acc={acc:.4f}")
                continue
            # ---- D step ----
            D.train()
            G.eval()
            optD.zero_grad(set_to_none=True)

            mask_pos = (labels > 0.5)
            mask_neg = ~mask_pos

            pos = float(mask_pos.sum().item())
            neg = float(mask_neg.sum().item())
            ratio = min(neg / max(1.0, pos), 20.0)
            crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ratio, device=device))

            loss_D = 0.0
            if mask_pos.any():
                loss_D = loss_D + crit(D(real_tok[mask_pos], bg[mask_pos], is_soft=False),
                                       torch.ones((int(mask_pos.sum()),), device=device))
            if mask_neg.any():
                loss_D = loss_D + crit(D(real_tok[mask_neg], bg[mask_neg], is_soft=False),
                                       torch.zeros((int(mask_neg.sum()),), device=device))

            with torch.no_grad():
                y_soft_det = G.sample_gumbel_soft(bg, tau=args.tau)
            loss_D = loss_D + crit(D(y_soft_det, bg, is_soft=True), torch.zeros((y_soft_det.size(0),), device=device))

            loss_D.backward()
            if args.d_grad_clip and args.d_grad_clip > 0:
                nn.utils.clip_grad_norm_(D.parameters(), float(args.d_grad_clip))
            optD.step()

            # ---- G step ----
            D.eval()
            G.train()
            optG.zero_grad(set_to_none=True)

            y_soft = G.sample_gumbel_soft(bg, tau=args.tau)
            logit_fake = D(y_soft, bg, is_soft=True)
            g_adv = bce_loss(logit_fake, torch.ones_like(logit_fake))

            lm = seq_mlm_loss(G, real_tok, bg, mask_ratio=0.15)
            cpen = center_penalty(y_soft, weight=1.0)
            loss_G = g_adv + args.lambda_lm * lm + args.lambda_center * cpen

            loss_G.backward()
            optG.step()

            if step % 200 == 0:
                print(f"[step {step}] D={loss_D.item():.4f} G={loss_G.item():.4f} adv={g_adv.item():.4f} lm={lm.item():.4f}")
                logs.append({"step": step, "epoch": epoch, "loss_D": float(loss_D.item()), "loss_G": float(loss_G.item()),
                             "adv": float(g_adv.item()), "lm": float(lm.item()), "center": float(cpen.item())})

            if step % 2000 == 0:
                with torch.no_grad():
                    fake_tok = G.sample_discrete(bg[:16], start_token=AA2IDX.get("M", PAD_IDX), temperature=0.9, top_k=5)
                tokens_to_fasta(fake_tok.cpu().numpy(), outdir / "samples" / f"sample_step{step}.fasta",
                                header_prefix=f"ep{epoch}_st{step}")
                torch.save({"state_dict": G.state_dict()}, outdir / "checkpoints" / "G.pt")
                torch.save({"state_dict": D.state_dict()}, outdir / "checkpoints" / "D.pt")

        print(f"[Epoch {epoch}] done.")

    torch.save({"state_dict": G.state_dict()}, outdir / "checkpoints" / "G_final.pt")
    torch.save({"state_dict": D.state_dict()}, outdir / "checkpoints" / "D_final.pt")
    with open(outdir / "train_loss.json", "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    print(f"[OK] finished. outdir={outdir}")

    # save JSON (the original one)
    with open(outdir / "train_loss.json", "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    # additionally save JSONL (more convenient for post-processing)
    dump_logs_jsonl(logs, outdir, fname="train_loss.jsonl")

    # plot curves (one figure per metric)
    save_loss_plot(
        logs=logs,
        outdir=outdir / "plots",
        fname="loss_curve.png",
        smooth_window=5,  # you can start with 5 or 11; set to 1 to see the raw fluctuations
    )


if __name__ == "__main__":
    main()
