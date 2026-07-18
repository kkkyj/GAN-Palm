#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
conditional_seq_gan.py  (site_level_merged.csv version + ESM2-3B for discriminator)

Core changes:
- The discriminator D additionally receives the ESM2-3B sequence embedding (mean/center pooling) as concatenated features.
- The ESM embedding of fake sequences uses an "online incremental cache" strategy: by default the queue is batch-refreshed every K steps;
  for fakes that are cache misses in the current batch, a small number can optionally be computed inline (capped by --esm-inline-max).

Note:
- The gradient of the ESM branch with respect to G is constant (the embedding comes from hard argmax sampling), so G's gradient still comes mainly from the token path;
  ESM mainly enhances the discriminative power and stability of D.
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import esm  # facebookresearch/esm
    HAS_ESM = True
except Exception:
    HAS_ESM = False


# ===================== Constants =====================
SEQ_LEN = 101
CENTER_POS = 50  # 0-based in 101aa window
VOCAB = list("ACDEFGHIKLMNPQRSTVWYBX")  # add X, B as unknown-ish; you can customize
AA2IDX = {a: i for i, a in enumerate(VOCAB)}
IDX2AA = {i: a for a, i in AA2IDX.items()}
PAD_IDX = AA2IDX.get("X", len(VOCAB) - 1)  # treat X as padding/unknown
VOCAB_SIZE = len(VOCAB)

# protein table numeric columns used to build bg_vec (kept from your original)
NUMERIC_X_COLS = [
    "PALM_WT1", "PALM_WT2", "PALM_WT3", "PALM_WT4", "PALM_WT5",
    "PALM_KO1", "PALM_KO2", "PALM_KO3", "PALM_KO4", "PALM_KO5",
]

# ESM constants for fixed 101 window tokens: 0=BOS, 1..101=AA, 102=EOS
ESM_CENTER_TOKEN_IDX_101 = 51


# ===================== Utilities =====================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_seq101(s: str) -> str:
    """Normalize to length=101, uppercase, pad with X, enforce center=C."""
    s = (s or "").upper().strip()
    if len(s) >= SEQ_LEN:
        s = s[:SEQ_LEN]
    else:
        s = s + "X" * (SEQ_LEN - len(s))
    s = list(s)
    s[CENTER_POS] = "C"
    return "".join(s)


def encode_seq(seq101: str) -> np.ndarray:
    """Encode seq101 -> tokens (101,), with unknown mapped to X."""
    s = normalize_seq101(seq101)
    out = np.full((SEQ_LEN,), PAD_IDX, dtype=np.int64)
    for i, ch in enumerate(s):
        out[i] = AA2IDX.get(ch, PAD_IDX)
    out[CENTER_POS] = AA2IDX["C"]
    return out


def tokens_to_seq101(tokens: np.ndarray) -> str:
    """tokens (L,) -> seq101 string; PAD/unknown -> X; enforce center=C."""
    chars = []
    for t in tokens[:SEQ_LEN]:
        if int(t) == PAD_IDX:
            chars.append("X")
        else:
            chars.append(IDX2AA.get(int(t), "X"))
    if len(chars) < SEQ_LEN:
        chars += ["X"] * (SEQ_LEN - len(chars))
    chars[CENTER_POS] = "C"
    return "".join(chars)


def soft_to_hard_tokens(y_soft: torch.Tensor) -> torch.Tensor:
    """y_soft: (B,L,V) -> hard tokens (B,L) by argmax."""
    return torch.argmax(y_soft, dim=-1)


# ===================== Background vector =====================
def build_background_vector(df_protein: pd.DataFrame, bins: int = 24) -> np.ndarray:
    """
    Simple histogram-like background vector from protein-level numeric columns.
    (kept close to your original; adjust as needed)
    """
    x = df_protein[NUMERIC_X_COLS].to_numpy(dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, None)

    # log1p then bin into histogram per column
    feats = []
    for j in range(x.shape[1]):
        v = np.log1p(x[:, j])
        if np.all(v == 0):
            hist = np.zeros((bins,), dtype=np.float32)
            feats.append(hist)
            continue
        lo, hi = float(v.min()), float(v.max())
        if hi <= lo + 1e-8:
            hist = np.zeros((bins,), dtype=np.float32)
            hist[0] = 1.0
            feats.append(hist)
            continue
        hist, _ = np.histogram(v, bins=bins, range=(lo, hi), density=True)
        feats.append(hist.astype(np.float32))
    bg = np.concatenate(feats, axis=0).astype(np.float32)
    # normalize
    bg = bg / (np.linalg.norm(bg) + 1e-8)
    return bg


# ===================== Dataset =====================
class SiteSeqDataset(Dataset):
    """
    From site_level_merged.csv:
      - Auto infer label if not provided:
          label = 1 if (any PALM_* > 0) AND (Percolator q-Value <= threshold)
          else 0
      - Auto detect sequence column: sequence_101aa / Sequence Window / Annotated Sequence
    """
    def __init__(self, site_table: str, q_threshold: float = 0.05, seq_col: str = "", label_col: str = ""):
        self.site_table = site_table
        self.q_threshold = q_threshold

        suf = Path(site_table).suffix.lower()
        df = pd.read_parquet(site_table) if suf == ".parquet" else pd.read_csv(site_table)

        # sequence column
        user_seq_col = (seq_col or "").strip()
        if user_seq_col:
            if user_seq_col not in df.columns:
                raise ValueError(f"--seq-col '{user_seq_col}' not found. Available columns: {list(df.columns)[:80]}")
            seq_col_used = user_seq_col
        else:
            # common candidates across different export scripts
            cand = [
                "sequence_101aa", "seq101", "seq_101", "seq_101aa", "seq101aa",
                "Sequence Window", "sequence_window", "SequenceWindow",
                "Annotated Sequence", "annotated_sequence",
                "sequence", "seq", "window_seq", "seq_window", "seq_window_101aa",
                "SEQ101", "SEQ", "SEQ_101AA",
            ]
            seq_col_used = None
            for c in cand:
                if c in df.columns:
                    seq_col_used = c
                    break
            # heuristic fallback: pick the first object/string column whose median length is close to 101
            if seq_col_used is None:
                obj_cols = [c for c in df.columns if df[c].dtype == object]
                best = None
                for c in obj_cols[:200]:
                    s = df[c].astype(str).head(200)
                    lens = s.map(len)
                    med = float(lens.median()) if len(lens) else 0.0
                    if 80 <= med <= 140:  # allow some extra tokens/spaces
                        best = c
                        break
                seq_col_used = best
            if seq_col_used is None:
                raise ValueError(
                    "Cannot auto-detect sequence column. "
                    "Please pass --seq-col. Available columns: "
                    f"{list(df.columns)[:120]}"
                )

        df[seq_col_used] = df[seq_col_used].astype(str)

        # label
        # -------------------------
        # Priority order:
        # 1) explicit --label-col
        # 2) common ready-made label columns: label_bin / label / y
        # 3) otherwise fall back to inference from q-value + PALM_* (this parquet does not need this branch)
        label_col = (label_col or "").strip()
        if label_col:
            if label_col not in df.columns:
                raise ValueError(f"--label-col '{label_col}' not found. Available columns: {list(df.columns)[:80]}")
            labels = pd.to_numeric(df[label_col], errors="coerce").fillna(0.0).astype(np.float32).to_numpy()

        elif "label_bin" in df.columns:
            labels = pd.to_numeric(df["label_bin"], errors="coerce").fillna(0.0).astype(np.float32).to_numpy()

        elif "label" in df.columns:
            labels = pd.to_numeric(df["label"], errors="coerce").fillna(0.0).astype(np.float32).to_numpy()

        elif "y" in df.columns:
            labels = pd.to_numeric(df["y"], errors="coerce").fillna(0.0).astype(np.float32).to_numpy()

        else:
            # q-value (optional fallback)
            qcol = None
            for c in ["Percolator q-Value", "q_value", "q-value", "qValue"]:
                if c in df.columns:
                    qcol = c
                    break
            if qcol is None:
                raise ValueError(
                    "Cannot find label column (label_bin/label/y) or Percolator q-value column to infer label. "
                    "Please pass --label-col."
                )

            qv = pd.to_numeric(df[qcol], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
            palm_cols = [c for c in df.columns if c.startswith("PALM_")]
            if not palm_cols:
                raise ValueError("Cannot find PALM_* columns to infer label from q-value.")
            palm = df[palm_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            has_palm = (palm > 0).any(axis=1)
            labels = ((has_palm) & (qv <= float(q_threshold))).astype(np.float32)

        # normalize seq101
        seqs = [normalize_seq101(s) for s in df[seq_col_used].tolist()]
        tokens = np.stack([encode_seq(s) for s in seqs], axis=0)

        self.seqs = seqs
        self.tokens = tokens.astype(np.int64)
        self.labels = labels.astype(np.float32)
        self.label_col = label_col
    def __len__(self):
        return int(self.tokens.shape[0])

    def __getitem__(self, idx: int):
        return {
            "tokens": torch.from_numpy(self.tokens[idx]),
            "label": torch.tensor(self.labels[idx]),
            "seq101": self.seqs[idx],
        }


def collate(batch: List[dict]):
    tokens = torch.stack([b["tokens"] for b in batch], dim=0)  # (B,L)
    labels = torch.stack([b["label"] for b in batch], dim=0)   # (B,)
    seqs = [b["seq101"] for b in batch]
    return {"tokens": tokens, "label": labels, "seq101": seqs}


# ===================== Models =====================
class Generator(nn.Module):
    def __init__(self, bg_dim: int, d_model=256, nhead=8, nlayers=4, dff=512, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_IDX)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, d_model) * 0.02)

        self.film = nn.Sequential(
            nn.Linear(bg_dim, d_model * 2), nn.ReLU(),
            nn.Linear(d_model * 2, d_model * 2)
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dff,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.out = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, tokens: torch.Tensor, bg_vec: torch.Tensor):
        # tokens: (B,L)
        x = self.emb(tokens) + self.pos[:tokens.size(1)].unsqueeze(0)
        cond = self.film(bg_vec)  # (B,2D)
        gamma, beta = cond.chunk(2, dim=-1)
        x = x * (1 + torch.tanh(gamma.unsqueeze(1))) + beta.unsqueeze(1)

        h = self.enc(x)
        logits = self.out(h)  # (B,L,V)
        return logits

    def sample_gumbel_soft(self, bg_vec: torch.Tensor, tau: float = 1.0):
        B = bg_vec.size(0)
        device = bg_vec.device

        start = torch.full((B, SEQ_LEN), AA2IDX.get("M", PAD_IDX), dtype=torch.long, device=device)
        logits = self.forward(start, bg_vec)  # (B,L,V)

        y = nn.functional.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)  # (B,L,V)

        # ---- non-inplace center clamp ----
        V = y.size(-1)
        center_onehot = torch.zeros((B, V), device=device, dtype=y.dtype)
        center_onehot[:, AA2IDX["C"]] = 1.0
        center_onehot = center_onehot.unsqueeze(1)  # (B,1,V)

        mask = torch.ones((1, SEQ_LEN, 1), device=device, dtype=y.dtype)
        mask[:, CENTER_POS:CENTER_POS + 1, :] = 0.0

        y = y * mask + center_onehot * (1.0 - mask)
        return y

    @torch.no_grad()
    def sample_discrete(self, bg_vec: torch.Tensor, start_token: int, temperature: float = 1.0, top_k: int = 0):
        B = bg_vec.size(0)
        device = bg_vec.device
        tokens = torch.full((B, SEQ_LEN), start_token, dtype=torch.long, device=device)
        logits = self.forward(tokens, bg_vec) / max(1e-6, float(temperature))
        if top_k and top_k > 0:
            v, ix = torch.topk(logits, k=int(top_k), dim=-1)
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(-1, ix, v)
            logits = mask
        probs = torch.softmax(logits, dim=-1)
        samp = torch.multinomial(probs.view(-1, VOCAB_SIZE), num_samples=1).view(B, SEQ_LEN)
        samp[:, CENTER_POS] = AA2IDX["C"]
        return samp


class ConditionalSequenceDisc(nn.Module):
    def __init__(
        self,
        bg_dim: int,
        d_model=256,
        nhead=8,
        nlayers=4,
        dff=512,
        dropout=0.1,
        esm_dim: int = 2560,
        d_esm: int = 256,
        esm_dropout: float = 0.0,
    ):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_IDX)
        self.film = nn.Sequential(
            nn.Linear(bg_dim, d_model * 2), nn.ReLU(),
            nn.Linear(d_model * 2, d_model * 2)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dff, dropout=dropout,
            activation="gelu", batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)

        self.esm_proj = nn.Linear(esm_dim, d_esm)
        self.esm_dropout = nn.Dropout(p=float(esm_dropout)) if esm_dropout and esm_dropout > 0 else nn.Identity()

        self.cls = nn.Sequential(
            nn.Linear(d_model + d_esm, d_model), nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def forward(
        self,
        tokens_or_soft: torch.Tensor,
        bg_vec: torch.Tensor,
        esm_vec: Optional[torch.Tensor] = None,
        is_soft: bool = False,
    ):
        """
        tokens_or_soft:
          - is_soft=False: (B,L) long
          - is_soft=True : (B,L,V) float
        esm_vec:
          - (B, 2560) float32/float16 (will be projected)
          - if None: use zeros (not recommended except warmup/debug)
        """
        if is_soft:
            W = self.emb.weight  # (V,D)
            x = torch.matmul(tokens_or_soft, W)  # (B,L,D)
        else:
            x = self.emb(tokens_or_soft)  # (B,L,D)

        cond = self.film(bg_vec)  # (B,2D)
        gamma, beta = cond.chunk(2, dim=-1)
        x = x * (1 + torch.tanh(gamma.unsqueeze(1))) + beta.unsqueeze(1)

        h = self.enc(x)  # (B,L,D)
        h_mean = h.mean(dim=1)

        if esm_vec is None:
            esm_feat = torch.zeros((h_mean.size(0), self.esm_proj.out_features), device=h_mean.device, dtype=h_mean.dtype)
        else:
            esm_feat = self.esm_proj(esm_vec.to(h_mean.dtype))
            esm_feat = self.esm_dropout(esm_feat)

        feat = torch.cat([h_mean, esm_feat], dim=-1)
        logit = self.cls(feat).squeeze(-1)
        return logit


# ===================== ESM embedding (online cache) =====================
class ESMEmbedder:
    """
    Light wrapper around esm2_t36_3B_UR50D to produce emb_mean / emb_center for seq101.
    """
    def __init__(
        self,
        layer: int = 36,
        device: str = "",
        dtype: str = "fp16",
        no_amp: bool = False,
    ):
        if not HAS_ESM:
            raise RuntimeError("esm package not available. Please install facebookresearch/esm.")
        self.layer = int(layer)
        self.no_amp = bool(no_amp)

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if dtype == "fp16":
            self.amp_dtype = torch.float16
        elif dtype == "bf16":
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float32

        self.model, self.alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.pad = self.alphabet.padding_idx

    @torch.no_grad()
    def embed(
        self,
        seqs: List[str],
        pool: str = "mean",
        batch_size: int = 8,
    ) -> np.ndarray:
        """
        Return (N,2560) float16 numpy.
        """
        pool = pool.lower().strip()
        assert pool in ("mean", "center")

        outs = []
        bs = max(1, int(batch_size))
        for i in range(0, len(seqs), bs):
            chunk = [normalize_seq101(s) for s in seqs[i:i + bs]]
            data = [(f"s{i+j}", s) for j, s in enumerate(chunk)]
            _, _, tokens = self.batch_converter(data)
            tokens = tokens.to(self.device, non_blocking=True)

            # valid mask: exclude BOS and EOS and pad
            valid = (tokens != self.pad)
            valid[:, 0] = False
            nonpad = (tokens != self.pad)
            last_idx = nonpad.long().sum(dim=1) - 1  # EOS
            valid.scatter_(1, last_idx.view(-1, 1), False)

            if (not self.no_amp) and str(self.device).startswith("cuda") and self.amp_dtype in (torch.float16, torch.bfloat16):
                with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                    out = self.model(tokens, repr_layers=[self.layer], return_contacts=False)
                    reps = out["representations"][self.layer]  # (B,L,2560)
            else:
                out = self.model(tokens, repr_layers=[self.layer], return_contacts=False)
                reps = out["representations"][self.layer]

            if pool == "center":
                if reps.size(1) > ESM_CENTER_TOKEN_IDX_101:
                    vec = reps[:, ESM_CENTER_TOKEN_IDX_101, :]
                else:
                    # fallback to mean
                    v = valid.unsqueeze(-1)
                    denom = v.sum(dim=1).clamp(min=1)
                    vec = (reps * v).sum(dim=1) / denom
            else:
                v = valid.unsqueeze(-1)
                denom = v.sum(dim=1).clamp(min=1)
                vec = (reps * v).sum(dim=1) / denom

            vec = vec.detach().to("cpu").to(torch.float16).numpy()
            outs.append(vec)
        return np.concatenate(outs, axis=0)


class ESMCacheManager:
    """
    Two-level cache:
      - base cache loaded from npz (seqs + emb_mean/emb_center)
      - online cache computed during training
    """
    def __init__(
        self,
        base_npz: str,
        pool: str = "mean",
        embedder: Optional[ESMEmbedder] = None,
        online_dump_npz: str = "",
    ):
        self.pool = pool.lower().strip()
        assert self.pool in ("mean", "center")

        self.embedder = embedder
        self.online_dump_npz = online_dump_npz

        self.cache: Dict[str, np.ndarray] = {}
        self.queue: List[str] = []

        if base_npz:
            z = np.load(base_npz, allow_pickle=True)
            seqs = [str(s) for s in z["seqs"].tolist()]
            emb_key = "emb_mean" if self.pool == "mean" else "emb_center"
            emb = z[emb_key]
            if emb.dtype != np.float16:
                emb = emb.astype(np.float16)
            for s, e in zip(seqs, emb):
                self.cache[normalize_seq101(s)] = e
            print(f"[ESM] loaded base cache: {len(self.cache)} seqs from {base_npz}")

    def has(self, seq101: str) -> bool:
        return normalize_seq101(seq101) in self.cache

    def get_many(self, seqs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (embs, hit_mask):
          embs: (N,2560) float16, missing filled with zeros
          hit_mask: (N,) bool
        """
        n = len(seqs)
        embs = np.zeros((n, 2560), dtype=np.float16)
        hit = np.zeros((n,), dtype=bool)
        for i, s in enumerate(seqs):
            k = normalize_seq101(s)
            v = self.cache.get(k, None)
            if v is not None:
                embs[i] = v
                hit[i] = True
        return embs, hit

    def enqueue_missing(self, seqs: List[str]):
        for s in seqs:
            k = normalize_seq101(s)
            if k in self.cache:
                continue
            self.queue.append(k)

    def flush(self, max_new: int = 1024, batch_size: int = 8):
        """
        Compute embeddings for up to max_new queued seqs and add to cache.
        """
        if not self.queue:
            return 0
        if self.embedder is None:
            raise RuntimeError("ESMCacheManager.flush() requires an ESMEmbedder.")
        # de-duplicate while preserving order
        seen = set()
        uniq = []
        while self.queue and len(uniq) < int(max_new):
            s = self.queue.pop(0)
            if s in self.cache:
                continue
            if s in seen:
                continue
            seen.add(s)
            uniq.append(s)
        if not uniq:
            return 0
        vec = self.embedder.embed(uniq, pool=self.pool, batch_size=batch_size)  # float16
        for s, e in zip(uniq, vec):
            self.cache[s] = e.astype(np.float16, copy=False)

        if self.online_dump_npz:
            self._dump_online_npz()

        print(f"[ESM] flushed {len(uniq)} new embeddings (cache size={len(self.cache)})")
        return len(uniq)

    def inline_compute_for_missing(self, seqs_missing: List[str], batch_size: int = 8) -> Dict[str, np.ndarray]:
        """
        Compute embedding for a small list of missing seqs immediately (used when --esm-inline-max > 0).
        Returns dict seq->emb (float16).
        """
        if not seqs_missing:
            return {}
        if self.embedder is None:
            raise RuntimeError("inline_compute_for_missing requires embedder.")
        vec = self.embedder.embed(seqs_missing, pool=self.pool, batch_size=batch_size)
        out = {}
        for s, e in zip(seqs_missing, vec):
            k = normalize_seq101(s)
            self.cache[k] = e.astype(np.float16, copy=False)
            out[k] = self.cache[k]
        return out

    def _dump_online_npz(self):
        """
        Dump the *entire* cache to npz. (Potentially large; use sparingly.)
        If you need frequent persistence, switch to LMDB/SQLite.
        """
        path = Path(self.online_dump_npz)
        path.parent.mkdir(parents=True, exist_ok=True)
        seqs = np.array(list(self.cache.keys()), dtype=object)
        emb = np.stack([self.cache[s] for s in seqs.tolist()], axis=0)
        if self.pool == "mean":
            np.savez_compressed(path, seqs=seqs, emb_mean=emb, emb_center=emb)
        else:
            np.savez_compressed(path, seqs=seqs, emb_center=emb, emb_mean=emb)
        print(f"[ESM] dumped cache -> {path} (N={len(seqs)})")


# ===================== Losses & helpers =====================
def bce_loss(logits, targets):
    return nn.functional.binary_cross_entropy_with_logits(logits, targets)


def center_penalty(tokens_soft: torch.Tensor, weight: float = 1.0):
    p_c = tokens_soft[:, CENTER_POS, AA2IDX["C"]]
    return weight * (1.0 - p_c).mean()


def seq_ce_lm_loss(G: Generator, batch_tokens: torch.Tensor, bg: torch.Tensor):
    inp = batch_tokens[:, :-1]
    tgt = batch_tokens[:, 1:]
    logits = G.forward(inp, bg)
    return nn.functional.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE),
        tgt.reshape(-1),
        ignore_index=PAD_IDX
    )


@torch.no_grad()
def _batch_metrics_binary(logits: torch.Tensor, labels: torch.Tensor):
    prob = torch.sigmoid(logits)
    pred = (prob >= 0.5).float()
    acc = (pred == labels).float().mean().item()
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels.cpu().numpy(), prob.cpu().numpy())
    except Exception:
        auc = float("nan")
    return acc, auc


def pretrain_discriminator_supervised(
    D: nn.Module,
    loader: DataLoader,
    device: torch.device,
    esm_cache: Optional[ESMCacheManager],
    epochs: int = 10,
    lr: float = 2e-4,
    grad_clip: float = 1.0,
):
    if epochs <= 0:
        return
    D.train()
    opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    for ep in range(1, epochs + 1):
        total_loss, total_acc, total_auc, n_batches = 0.0, 0.0, 0.0, 0
        for batch in loader:
            tokens = batch["tokens"].to(device)
            labels = batch["label"].to(device)
            bg     = batch["bg"].to(device)
            seqs   = batch["seq101"]

            esm_vec = None
            if esm_cache is not None:
                emb_np, hit = esm_cache.get_many(seqs)
                # for supervised pretrain, real should all hit if cache built from dataset
                esm_vec = torch.from_numpy(emb_np.astype(np.float32)).to(device)

            opt.zero_grad(set_to_none=True)
            logits = D(tokens, bg, esm_vec=esm_vec, is_soft=False)

            pos = float((labels > 0.5).sum().item())
            neg = float((labels <= 0.5).sum().item())
            ratio = neg / max(1.0, pos)
            ratio = min(ratio, 20.0)  # e.g. at most 20
            pos_weight = torch.tensor(ratio, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = criterion(logits, labels)
            loss.backward()

            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(D.parameters(), grad_clip)
            opt.step()

            acc, auc = _batch_metrics_binary(logits, labels)
            total_loss += loss.item()
            total_acc  += acc
            total_auc  += 0.0 if math.isnan(auc) else auc
            n_batches  += 1

        print(f"[D-pretrain] epoch {ep:02d}/{epochs} "
              f"loss={total_loss/max(1,n_batches):.4f} "
              f"acc={total_acc/max(1,n_batches):.4f} "
              f"auc={ (total_auc/max(1,n_batches) if n_batches>0 else float('nan')) :.4f}")


def tokens_to_fasta(tokens: np.ndarray, path: Path, header_prefix: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, row in enumerate(tokens):
            seq = tokens_to_seq101(row)
            f.write(f">{header_prefix}_{i}\n{seq}\n")


# ===================== Main =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein-table", default="",
                    help="protein_level_with_reps.(csv|parquet) for bg_vec (optional if --bg-npz/--bg-npy provided)")
    ap.add_argument("--site-table",    required=True, help="site_level_merged.csv (contains PALM_* columns & sequence column)")
    ap.add_argument("--seq-col", default="", help="name of the sequence column in site-table; leave empty for auto-detection")
    ap.add_argument("--bg-npy", default="", help="precomputed bg_vec.npy (used preferentially if provided)")
    ap.add_argument("--bg-npz", default="", help="precomputed bg_vec.npz (e.g. output of build_proteome_background_embeddings.py)")
    ap.add_argument("--bins", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr-g", type=float, default=2e-4)
    ap.add_argument("--lr-d", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--lambda-lm", type=float, default=0.2)
    ap.add_argument("--lambda-center", type=float, default=0.5)
    ap.add_argument("--q-threshold", type=float, default=0.05, help="Percolator q-Value threshold (inclusive)")
    ap.add_argument("--pretrain-d-epochs", type=int, default=10, help="number of supervised pretraining epochs for the discriminator")
    ap.add_argument("--d-grad-clip", type=float, default=1.0, help="discriminator gradient clipping threshold (0 to disable)")

    # ---- ESM options ----
    ap.add_argument("--esm-npz", default="", help="offline ESM cache npz (seqs + emb_mean/emb_center)")
    ap.add_argument("--esm-pool", default="mean", choices=["mean", "center"], help="ESM pooling")
    ap.add_argument("--esm-d-emb", type=int, default=256, help="ESM projection dimension d_esm used inside D")
    ap.add_argument("--esm-dropout", type=float, default=0.0, help="ESM feature dropout (prevents overfitting/shortcuts)")
    ap.add_argument("--esm-update-every", type=int, default=20, help="flush the queue once every K steps")
    ap.add_argument("--esm-flush-max", type=int, default=2048, help="max number of new sequences to compute per flush")
    ap.add_argument("--esm-inline-max", type=int, default=64, help="compute inline immediately when current-step cache misses <= this threshold")
    ap.add_argument("--esm-batch-size", type=int, default=8, help="ESM inference batch size")
    ap.add_argument("--esm-layer", type=int, default=36)
    ap.add_argument("--esm-dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--esm-device", default="", help="cuda/cpu/cuda:0 (default auto)")
    ap.add_argument("--esm-no-amp", action="store_true", help="disable autocast for ESM")
    ap.add_argument("--esm-online-dump", default="", help="optional: periodically dump the online cache to npz (not recommended at large scale)")
    ap.add_argument("--label-col", default="", help="name of the column in site-table to use directly as the label (e.g. label_bin); takes priority over auto-inference")

    args = ap.parse_args()
    set_seed(args.seed)

    outdir = Path(args.outdir)
    (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (outdir / "samples").mkdir(exist_ok=True)

    # -------- bg_vec --------
    bg_vec = None

    def _load_bg_from_npz(npz_path: str) -> np.ndarray:
        """Load a 1D background conditioning vector from an npz.

        Supports:
          - direct 1D vectors under common keys (bg_vec/bg/background/...)
          - a 2D matrix named 'combined_emb' shaped (K,D) or (K,K), reduced by mean over axis 0
            (this matches build_proteome_background_embeddings.py outputs in this project).
        """
        z = np.load(npz_path, allow_pickle=True)

        # 1) common 1D keys
        for k in ["bg_vec", "bg", "background", "mean_bg", "emb", "embedding"]:
            if k in z.files:
                arr = z[k]
                if hasattr(arr, "ndim") and arr.ndim == 1 and arr.size > 0:
                    return arr.astype(np.float32)

        # 2) project-specific key: combined_emb (2D)
        if "combined_emb" in z.files:
            arr = z["combined_emb"]
            if hasattr(arr, "ndim") and arr.ndim == 2 and arr.size > 0:
                # reduce to 1D conditioning vector
                vec = arr.mean(axis=0)
                return vec.astype(np.float32)

        # 3) fallback: find any 1D vector
        candidates_1d = []
        candidates_2d = []
        for k in z.files:
            arr = z[k]
            if not hasattr(arr, "ndim"):
                continue
            if arr.ndim == 1 and arr.size > 0:
                candidates_1d.append((k, arr))
            elif arr.ndim == 2 and arr.size > 0:
                candidates_2d.append((k, arr))

        if len(candidates_1d) == 1:
            return candidates_1d[0][1].astype(np.float32)

        # If there is exactly one 2D candidate, reduce it.
        if len(candidates_2d) == 1:
            k, arr = candidates_2d[0]
            vec = arr.mean(axis=0)
            return vec.astype(np.float32)

        raise ValueError(f"Unable to parse bg vector from {npz_path}; available keys={z.files}")

    if args.bg_npy and Path(args.bg_npy).exists():
        bg_vec = np.load(args.bg_npy).astype(np.float32)

    elif args.bg_npz and Path(args.bg_npz).exists():
        bg_vec = _load_bg_from_npz(args.bg_npz)

    elif args.protein_table and Path(args.protein_table).exists():
        suf = Path(args.protein_table).suffix.lower()
        dfp = pd.read_parquet(args.protein_table) if suf == ".parquet" else pd.read_csv(args.protein_table)
        for c in NUMERIC_X_COLS:
            if c not in dfp.columns:
                raise ValueError(f"protein_level_with_reps is missing {c}")
        dfp[NUMERIC_X_COLS] = dfp[NUMERIC_X_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        dfp[NUMERIC_X_COLS] = dfp[NUMERIC_X_COLS].clip(lower=0.0)
        bg_vec = build_background_vector(dfp, bins=args.bins)

    else:
        raise FileNotFoundError(
            "Background vector not found: please provide --bg-npy or --bg-npz, "
            "or provide an existing --protein-table (protein_level_with_reps.*)"
        )

    bg_dim = int(bg_vec.shape[0])
    if args.bins != bg_dim:
        print(f"[Warn] args.bins={args.bins} != bg_dim={bg_dim}; override bins to {bg_dim} based on bg vector.")
        args.bins = bg_dim

    # -------- dataset --------
    ds = SiteSeqDataset(args.site_table, q_threshold=args.q_threshold, seq_col=args.seq_col, label_col=args.label_col)

    def _collate(b):
        base = collate(b)
        base["bg"] = torch.tensor(bg_vec).float().unsqueeze(0).repeat(base["tokens"].size(0), 1)
        return base

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        collate_fn=_collate, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- ESM cache --------
    esm_cache = None
    esm_embedder = None
    if args.esm_npz:
        esm_embedder = ESMEmbedder(
            layer=args.esm_layer,
            device=args.esm_device,
            dtype=args.esm_dtype,
            no_amp=args.esm_no_amp,
        )
        esm_cache = ESMCacheManager(
            base_npz=args.esm_npz,
            pool=args.esm_pool,
            embedder=esm_embedder,
            online_dump_npz=args.esm_online_dump,
        )
    else:
        print("[Warn] --esm-npz not provided: D will run without ESM features.")

    # -------- models --------
    G = Generator(bg_dim=bg_dim).to(device)
    D = ConditionalSequenceDisc(
        bg_dim=bg_dim,
        esm_dim=2560,
        d_esm=int(args.esm_d_emb),
        esm_dropout=float(args.esm_dropout),
    ).to(device)

    optG = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    # ===== supervised pretrain D =====
    print(f"[Info] Pretraining D for {args.pretrain_d_epochs} epoch(s) in supervised mode...")
    pretrain_discriminator_supervised(
        D=D,
        loader=loader,
        device=device,
        esm_cache=esm_cache,
        epochs=args.pretrain_d_epochs,
        lr=args.lr_d,
        grad_clip=args.d_grad_clip,
    )
    torch.save({"state_dict": D.state_dict()}, outdir / "checkpoints" / "D_pretrained.pt")

    logs = []
    step = 0

    for epoch in range(1, args.epochs + 1):
        for batch in loader:
            step += 1
            real_tok = batch["tokens"].to(device)   # (B,L)
            labels   = batch["label"].to(device)    # (B,)
            bg       = batch["bg"].to(device)       # (B,bg_dim)
            real_seqs = batch["seq101"]

            # optional: periodic flush (using queued fake sequences)
            if esm_cache is not None and args.esm_update_every and step % int(args.esm_update_every) == 0:
                esm_cache.flush(max_new=int(args.esm_flush_max), batch_size=int(args.esm_batch_size))

            # -------- warmup (only LM) --------
            if epoch <= args.warmup_epochs:
                G.train()
                optG.zero_grad(set_to_none=True)
                lm_loss = seq_ce_lm_loss(G, real_tok, bg)
                lm_loss.backward()
                optG.step()
                if step % 100 == 0:
                    logs.append({"step": step, "epoch": epoch, "phase": "warmup", "lm_loss": float(lm_loss.item())})
                continue

            # -------- get real ESM --------
            real_esm = None
            if esm_cache is not None:
                emb_np, hit = esm_cache.get_many(real_seqs)
                if not bool(hit.all()):
                    # if this happens, your offline cache doesn't match seq normalization
                    miss_n = int((~hit).sum())
                    print(f"[Warn] real ESM cache miss {miss_n}/{len(real_seqs)} (will inline compute)")
                    miss_seqs = [real_seqs[i] for i in range(len(real_seqs)) if not hit[i]]
                    # inline compute (real should be small misses)
                    esm_cache.inline_compute_for_missing(miss_seqs, batch_size=int(args.esm_batch_size))
                    emb_np, _ = esm_cache.get_many(real_seqs)
                real_esm = torch.from_numpy(emb_np.astype(np.float32)).to(device)

            # ======== 1) Update D ========
            D.train()
            G.eval()
            optD.zero_grad(set_to_none=True)

            # real positive / negative
            mask_pos = (labels > 0.5)
            mask_neg = ~mask_pos

            pos = float(mask_pos.sum().item())
            neg = float(mask_neg.sum().item())
            ratio = neg / max(1.0, pos)
            ratio = min(ratio, 20.0)  # e.g. at most 20
            pos_weight = torch.tensor(ratio, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            loss_rp = None
            loss_rn = None

            if mask_pos.any():
                logit_rp = D(real_tok[mask_pos], bg[mask_pos],
                             esm_vec=(real_esm[mask_pos] if real_esm is not None else None),
                             is_soft=False)
                loss_rp = criterion(logit_rp, torch.ones_like(logit_rp))

            if mask_neg.any():
                logit_rn = D(real_tok[mask_neg], bg[mask_neg],
                             esm_vec=(real_esm[mask_neg] if real_esm is not None else None),
                             is_soft=False)
                loss_rn = criterion(logit_rn, torch.zeros_like(logit_rn))

            # fake (soft for gradient path)
            with torch.no_grad():
                y_soft_detached = G.sample_gumbel_soft(bg, tau=args.tau)  # (B,L,V)

            # ---- fake ESM (hard argmax, cache + inline + queue) ----
            fake_esm = None
            if esm_cache is not None:
                y_hard = soft_to_hard_tokens(y_soft_detached).detach().cpu().numpy()  # (B,L)
                fake_seqs = [tokens_to_seq101(row) for row in y_hard]
                emb_np, hit = esm_cache.get_many(fake_seqs)

                miss_idx = np.where(~hit)[0].tolist()
                if miss_idx:
                    miss_seqs = [fake_seqs[i] for i in miss_idx]
                    # inline compute if small, else enqueue
                    if args.esm_inline_max and len(miss_seqs) <= int(args.esm_inline_max):
                        esm_cache.inline_compute_for_missing(miss_seqs, batch_size=int(args.esm_batch_size))
                    else:
                        esm_cache.enqueue_missing(miss_seqs)
                # re-fetch (some might have been inlined)
                emb_np, _ = esm_cache.get_many(fake_seqs)
                fake_esm = torch.from_numpy(emb_np.astype(np.float32)).to(device)

            logit_rf = D(y_soft_detached, bg, esm_vec=fake_esm, is_soft=True)
            loss_rf = criterion(logit_rf, torch.zeros_like(logit_rf))

            loss_D = loss_rf
            if loss_rp is not None:
                loss_D = loss_D + loss_rp
            if loss_rn is not None:
                loss_D = loss_D + loss_rn

            loss_D.backward()
            nn.utils.clip_grad_norm_(D.parameters(), args.d_grad_clip)
            optD.step()

            # ======== 2) Update G ========
            D.eval()
            G.train()
            optG.zero_grad(set_to_none=True)

            y_soft = G.sample_gumbel_soft(bg, tau=args.tau)  # (B,L,V)

            # fake ESM for G step: use cache/inline/queue similarly, but it does not backprop to G anyway
            fake_esm_g = None
            if esm_cache is not None:
                y_hard_g = soft_to_hard_tokens(y_soft).detach().cpu().numpy()
                fake_seqs_g = [tokens_to_seq101(row) for row in y_hard_g]
                emb_np, hit = esm_cache.get_many(fake_seqs_g)
                miss_idx = np.where(~hit)[0].tolist()
                if miss_idx:
                    miss_seqs = [fake_seqs_g[i] for i in miss_idx]
                    if args.esm_inline_max and len(miss_seqs) <= int(args.esm_inline_max):
                        esm_cache.inline_compute_for_missing(miss_seqs, batch_size=int(args.esm_batch_size))
                    else:
                        esm_cache.enqueue_missing(miss_seqs)
                emb_np, _ = esm_cache.get_many(fake_seqs_g)
                fake_esm_g = torch.from_numpy(emb_np.astype(np.float32)).to(device)

            logit_fake = D(y_soft, bg, esm_vec=fake_esm_g, is_soft=True)
            g_adv = bce_loss(logit_fake, torch.ones_like(logit_fake))
            lm_loss = seq_ce_lm_loss(G, real_tok, bg)
            c_pen  = center_penalty(y_soft, weight=1.0)

            loss_G = g_adv + args.lambda_lm * lm_loss + args.lambda_center * c_pen
            loss_G.backward()
            optG.step()

            if step % 100 == 0:
                logs.append({
                    "step": step, "epoch": epoch,
                    "loss_D": float(loss_D.item()),
                    "loss_G": float(loss_G.item()),
                    "adv": float(g_adv.item()),
                    "lm": float(lm_loss.item()),
                    "center": float(c_pen.item()),
                })

            # sampling & checkpoints
            if step % 1000 == 0:
                with torch.no_grad():
                    fake_tok = G.sample_discrete(bg[:16], start_token=AA2IDX.get("M", PAD_IDX), temperature=0.9, top_k=5)
                fasta_path = outdir / "samples" / f"sample_step{step}.fasta"
                tokens_to_fasta(fake_tok.cpu().numpy(), fasta_path, header_prefix=f"ep{epoch}_st{step}")
                torch.save({"state_dict": G.state_dict()}, outdir / "checkpoints" / "G.pt")
                torch.save({"state_dict": D.state_dict()}, outdir / "checkpoints" / "D.pt")

        print(f"[Epoch {epoch}] done.")

    torch.save({"state_dict": G.state_dict()}, outdir / "checkpoints" / "G_final.pt")
    torch.save({"state_dict": D.state_dict()}, outdir / "checkpoints" / "D_final.pt")
    with open(outdir / "train_loss.json", "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    print(f"[OK] training finished. Check {outdir}")


if __name__ == "__main__":
    main()
