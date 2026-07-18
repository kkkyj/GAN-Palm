#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Patched from your version: conditional_seq_gan_noesm_poslm_bgdb_balanced.py

Stability + debug hotfix for G warmup NaN:
1) FiLM: bound BOTH gamma and beta with tanh, and scale to small amplitude.
2) FiLM last layer zero-init => start from "no conditioning".
3) Separate warmup lr: --lr-g-warmup (default lr_g * 0.2).
4) NaN guard: if bg/logits/loss/grad has NaN/Inf => skip step (no opt.step), print diagnostics.
5) Debug prints every --debug-every steps: bg stats, logits stats, grad_norm.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

SEQ_LEN = 101
CENTER_POS = 50

VOCAB = list("ACDEFGHIKLMNPQRSTVWYBX")
AA2IDX = {a: i for i, a in enumerate(VOCAB)}
IDX2AA = {i: a for a, i in AA2IDX.items()}
PAD_IDX = AA2IDX.get("X", len(VOCAB) - 1)
VOCAB_SIZE = len(VOCAB)


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


def load_bg_db(bg_npz: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    z = np.load(bg_npz, allow_pickle=True)
    if "samples" not in z.files or "combined_emb" not in z.files:
        raise ValueError(f"bg npz must contain keys 'samples' and 'combined_emb'. got keys={z.files}")

    samples = z["samples"]
    emb = z["combined_emb"].astype(np.float32)
    if emb.ndim != 2:
        raise ValueError(f"combined_emb must be 2D, got shape={emb.shape}")

    sids = [str(x) for x in samples.tolist()]
    bg_map = {sid: emb[i].copy() for i, sid in enumerate(sids)}
    bg_mean = emb.mean(axis=0).astype(np.float32)
    return bg_map, bg_mean


def make_sample_id(dataset: str, sample: str) -> str:
    return f"{str(dataset)}__{str(sample)}"


class SiteSeqDataset(Dataset):
    def __init__(
        self,
        site_table: str,
        seq_col: str = "seq101",
        label_col: str = "label_bin",
        dataset_col: str = "dataset",
        sample_col: str = "sample",
    ):
        suf = Path(site_table).suffix.lower()
        df = pd.read_parquet(site_table) if suf == ".parquet" else pd.read_csv(site_table)

        miss = [c for c in [seq_col, label_col, dataset_col, sample_col] if c not in df.columns]
        if miss:
            raise ValueError(f"site table missing columns: {miss}. available={df.columns.tolist()[:50]}")

        seqs = [normalize_seq101(s) for s in df[seq_col].astype(str).tolist()]
        tokens = np.stack([encode_seq(s) for s in seqs], axis=0).astype(np.int64)

        labels = pd.to_numeric(df[label_col], errors="coerce").fillna(0.0).astype(np.float32).to_numpy()

        datasets = df[dataset_col].astype(str).tolist()
        samples = df[sample_col].astype(str).tolist()
        sample_ids = [make_sample_id(d, s) for d, s in zip(datasets, samples)]

        self.seqs = seqs
        self.tokens = tokens
        self.labels = labels
        self.datasets = datasets
        self.samples = samples
        self.sample_ids = sample_ids

    def __len__(self):
        return int(self.tokens.shape[0])

    def __getitem__(self, idx: int):
        return {
            "tokens": torch.from_numpy(self.tokens[idx]),
            "label": torch.tensor(self.labels[idx]),
            "seq101": self.seqs[idx],
            "dataset": self.datasets[idx],
            "sample": self.samples[idx],
            "sample_id": self.sample_ids[idx],
        }


def collate_with_bg(
    batch: List[dict],
    bg_map: Dict[str, np.ndarray],
    bg_fallback: np.ndarray,
    bg_clip: float = 5.0,
    bg_l2norm: bool = True,
) -> Dict[str, Any]:
    tokens = torch.stack([b["tokens"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    seqs = [b["seq101"] for b in batch]
    sample_ids = [b["sample_id"] for b in batch]
    datasets = [b.get("dataset", "") for b in batch]
    samples = [b.get("sample", "") for b in batch]

    miss = 0
    bg_rows = []
    for sid in sample_ids:
        v = bg_map.get(sid, None)
        if v is None:
            miss += 1
            v = bg_fallback
        bg_rows.append(v)

    bg = np.stack(bg_rows, axis=0).astype(np.float32)
    bg = np.nan_to_num(bg, nan=0.0, posinf=0.0, neginf=0.0)

    if bg_l2norm:
        denom = np.linalg.norm(bg, axis=1, keepdims=True) + 1e-8
        bg = bg / denom

    if bg_clip is not None and float(bg_clip) > 0:
        c = float(bg_clip)
        bg = np.clip(bg, -c, c).astype(np.float32)

    bg = torch.from_numpy(bg)

    return {
        "tokens": tokens,
        "label": labels,
        "seq101": seqs,
        "bg": bg,
        "sample_id": sample_ids,
        "dataset": datasets,
        "sample": samples,
        "_bg_miss": miss,
    }


def cycle_loader(loader: DataLoader) -> Iterator[Dict[str, Any]]:
    while True:
        for batch in loader:
            yield batch


def _tensor_stats(x: torch.Tensor) -> dict:
    x2 = x.detach()
    return {
        "dtype": str(x2.dtype),
        "min": float(x2.min().item()) if x2.numel() else float("nan"),
        "max": float(x2.max().item()) if x2.numel() else float("nan"),
        "mean": float(x2.mean().item()) if x2.numel() else float("nan"),
        "std": float(x2.std(unbiased=False).item()) if x2.numel() else float("nan"),
        "has_nan": bool(torch.isnan(x2).any().item()) if x2.numel() else False,
        "has_inf": bool(torch.isinf(x2).any().item()) if x2.numel() else False,
    }


def _global_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if torch.isnan(g).any() or torch.isinf(g).any():
            return float("nan")
        total += float(g.norm(2).item()) ** 2
    return float(total ** 0.5)


class Generator(nn.Module):
    def __init__(self, bg_dim: int, d_model=256, nhead=8, nlayers=4, dff=512, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_IDX)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, d_model) * 0.02)

        self.film = nn.Sequential(
            nn.Linear(bg_dim, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model * 2),
        )
        # --- key: zero-init last layer => start from no conditioning ---
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dff,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.out = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, tokens: torch.Tensor, bg_vec: torch.Tensor, return_film: bool = False):
        x = self.emb(tokens) + self.pos[:tokens.size(1)].unsqueeze(0)
        gamma, beta = self.film(bg_vec).chunk(2, dim=-1)

        # --- key: bound both gamma & beta, and scale small ---
        g = torch.tanh(gamma).unsqueeze(1)
        b = torch.tanh(beta).unsqueeze(1)
        x = x * (1.0 + 0.5 * g) + 0.5 * b

        h = self.enc(x)
        logits = self.out(h)
        if return_film:
            return logits, gamma, beta
        return logits

    def sample_gumbel_soft(self, bg_vec: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        B = bg_vec.size(0)
        device = bg_vec.device
        start = torch.full((B, SEQ_LEN), AA2IDX.get("M", PAD_IDX), dtype=torch.long, device=device)
        logits = self.forward(start, bg_vec)
        y = nn.functional.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)

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
    def __init__(self, bg_dim: int, emb_dim: int = 64, channels: int = 128, kernels=(3, 5, 7), dropout: float = 0.1):
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
        if is_soft:
            return torch.matmul(tokens_or_soft, self.emb.weight)
        return self.emb(tokens_or_soft)

    def forward(self, tokens_or_soft: torch.Tensor, bg_vec: torch.Tensor, is_soft: bool = False) -> torch.Tensor:
        x = self._embed(tokens_or_soft, is_soft=is_soft)  # (B,L,E)
        x = x.transpose(1, 2)  # (B,E,L)

        feats = []
        for block in self.convs:
            h = block(x)
            h = torch.amax(h, dim=-1)
            feats.append(h)
        h = torch.cat(feats, dim=1)

        if self.bg_dim > 0:
            bg2 = self.bg_ln(bg_vec)
            h = torch.cat([h, bg2], dim=1)

        logit = self.mlp(h).squeeze(-1)
        return logit


def center_penalty(tokens_soft: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    p_c = tokens_soft[:, CENTER_POS, AA2IDX["C"]]
    return weight * (1.0 - p_c).mean()


def seq_mlm_loss(G: Generator, batch_tokens: torch.Tensor, bg: torch.Tensor, mask_ratio: float = 0.15,
                 debug_return_logits: bool = False):
    x = batch_tokens.clone()
    B, L = x.shape
    device = x.device
    mask = (torch.rand((B, L), device=device) < mask_ratio)
    mask[:, CENTER_POS] = False
    x[mask] = PAD_IDX

    logits, gamma, beta = G.forward(x, bg, return_film=True)
    # if logits already contain NaN/Inf, cross entropy can become NaN
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        # keep original for debug; replace for CE
        logits_safe = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        logits_safe = logits

    tgt = batch_tokens[mask]
    pred = logits_safe[mask]

    # IMPORTANT: pred could be empty (very unlikely with big B), but handle anyway
    if tgt.numel() == 0:
        loss = nn.functional.cross_entropy(
            logits_safe.reshape(-1, VOCAB_SIZE),
            batch_tokens.reshape(-1),
            ignore_index=PAD_IDX
        )
    else:
        loss = nn.functional.cross_entropy(pred, tgt, ignore_index=PAD_IDX)

    if debug_return_logits:
        return loss, logits, gamma, beta
    return loss


def hinge_d_loss(logit_pos, logit_neg, logit_fake, w_pos=1.0, w_neg=1.0, w_fake=1.0):
    loss = 0.0
    if logit_pos is not None and logit_pos.numel() > 0:
        loss = loss + float(w_pos) * torch.relu(1.0 - logit_pos).mean()
    if logit_neg is not None and logit_neg.numel() > 0:
        loss = loss + float(w_neg) * torch.relu(1.0 + logit_neg).mean()
    loss = loss + float(w_fake) * torch.relu(1.0 + logit_fake).mean()
    return loss


def hinge_g_adv(logit_fake: torch.Tensor) -> torch.Tensor:
    return -logit_fake.mean()


def tokens_to_fasta(tokens: np.ndarray, path: Path, header_prefix: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, row in enumerate(tokens):
            f.write(f">{header_prefix}_{i}\n{tokens_to_seq101(row)}\n")


def dump_logs_jsonl(logs: list, outdir: Path, fname: str = "train_loss.jsonl"):
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / fname
    with open(path, "w", encoding="utf-8") as f:
        for d in logs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"[OK] wrote: {path}")


def save_loss_plot(logs: list, outdir: Path, smooth_window: int = 5):
    if not logs:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    steps = np.array([d.get("step", i) for i, d in enumerate(logs)], dtype=np.int64)

    def series(key: str):
        return np.array([d.get(key, np.nan) for d in logs], dtype=np.float64)

    keys = ["loss_D", "loss_G", "adv", "lm", "center"]
    for k in keys:
        y = series(k)
        if np.isnan(y).all():
            continue
        y2 = y.copy()
        if smooth_window and smooth_window > 1:
            isn = np.isnan(y2)
            if isn.any():
                idx = np.where(~isn)[0]
                y2[:idx[0]] = y2[idx[0]]
                for i in range(idx[0] + 1, len(y2)):
                    if np.isnan(y2[i]):
                        y2[i] = y2[i - 1]
            kernel = np.ones(int(smooth_window), dtype=np.float64) / float(smooth_window)
            y2 = np.convolve(y2, kernel, mode="same")
        plt.figure()
        plt.plot(steps, y2)
        plt.xlabel("step")
        plt.ylabel(k)
        plt.title(k)
        plt.tight_layout()
        plt.savefig(outdir / f"loss_{k}.png", dpi=150)
        plt.close()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--site-table", required=True)
    ap.add_argument("--seq-col", default="seq101")
    ap.add_argument("--label-col", default="label_bin")
    ap.add_argument("--dataset-col", default="dataset")
    ap.add_argument("--sample-col", default="sample")

    ap.add_argument("--bg-npz", required=True)

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr-g", type=float, default=2e-4)
    ap.add_argument("--lr-d", type=float, default=5e-5)
    ap.add_argument("--lr-g-warmup", type=float, default=None, help="If set, override G lr during warmup.")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--d-warmup-epochs", type=int, default=3)
    ap.add_argument("--tau", type=float, default=1.0)

    ap.add_argument("--lambda-adv", type=float, default=5.0)
    ap.add_argument("--lambda-lm", type=float, default=0.1)
    ap.add_argument("--lambda-center", type=float, default=0.5)

    ap.add_argument("--d-grad-clip", type=float, default=1.0)
    ap.add_argument("--pos-scale-cap", type=float, default=60.0)
    ap.add_argument("--g-grad-clip", type=float, default=1.0)

    ap.add_argument("--d-balance-posneg", action="store_true")
    ap.add_argument("--d-pos-frac", type=float, default=0.5)

    ap.add_argument("--g-adv-sample-uniform", action="store_true")

    # debug / safety
    ap.add_argument("--debug-every", type=int, default=200)
    ap.add_argument("--skip-nan-steps", action="store_true", help="If set, skip update when NaN/Inf detected.")
    ap.add_argument("--max-nan-dumps", type=int, default=5)

    # saving / best selection
    ap.add_argument("--save-every-epochs", type=int, default=1, help="Save last/best each N epochs.")
    ap.add_argument("--best-metric", type=str, default="d_plus_lm", choices=["lossG", "d_plus_lm"],
                    help="Criterion for best model. lossG=mean(loss_G); d_plus_lm=mean(loss_D)+0.1*mean(lm).")

    args = ap.parse_args()
    set_seed(args.seed)

    outdir = Path(args.outdir)
    (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (outdir / "samples").mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)

    print("[OUTDIR]", outdir.resolve())

    bg_map, bg_mean = load_bg_db(args.bg_npz)
    bg_dim = int(bg_mean.shape[0])
    bg_keys = list(bg_map.keys())
    if not bg_keys:
        raise RuntimeError("bg db empty")

    ds = SiteSeqDataset(
        args.site_table,
        seq_col=args.seq_col,
        label_col=args.label_col,
        dataset_col=args.dataset_col,
        sample_col=args.sample_col,
    )

    labels_np = ds.labels
    pos_idx = np.where(labels_np > 0.5)[0]
    neg_idx = np.where(labels_np <= 0.5)[0]
    if pos_idx.size == 0 or neg_idx.size == 0:
        raise RuntimeError(f"pos or neg missing. n_pos={pos_idx.size} n_neg={neg_idx.size}")

    class _Subset(Dataset):
        def __init__(self, base: SiteSeqDataset, indices: np.ndarray):
            self.base = base
            self.indices = indices.astype(np.int64)
        def __len__(self):
            return int(self.indices.size)
        def __getitem__(self, i: int):
            return self.base[int(self.indices[int(i)])]

    pos_ds = _Subset(ds, pos_idx)
    neg_ds = _Subset(ds, neg_idx)

    def _collate(batch):
        return collate_with_bg(batch, bg_map=bg_map, bg_fallback=bg_mean)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True, collate_fn=_collate)
    pos_loader = DataLoader(pos_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True, collate_fn=_collate)
    neg_loader = DataLoader(neg_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True, collate_fn=_collate)

    pos_iter = cycle_loader(pos_loader)
    neg_iter = cycle_loader(neg_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(bg_dim=bg_dim).to(device)
    D = ConditionalSequenceDisc(bg_dim=bg_dim).to(device)

    optG = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    # warmup lr override
    warmup_lr = args.lr_g_warmup if args.lr_g_warmup is not None else (args.lr_g * 0.2)
    nan_dumps = 0

    def _set_lr(optim: torch.optim.Optimizer, lr: float):
        for pg in optim.param_groups:
            pg["lr"] = float(lr)

    logs = []
    step = 0
    bg_miss_total = 0

    # best tracking
    best_metric = float("inf")
    best_state = {"epoch": 0, "step": 0, "metric": best_metric, "criterion": args.best_metric}

    def _save_last(tag: str = "last"):
        ckpt_dir = outdir / "checkpoints"
        torch.save(G.state_dict(), ckpt_dir / f"G_{tag}.pt")
        torch.save(D.state_dict(), ckpt_dir / f"D_{tag}.pt")

    def _save_best(metric: float, epoch: int, step: int, mean_lossG: float, mean_lossD: float, mean_lm: float):
        nonlocal best_metric, best_state
        if metric < best_metric:
            best_metric = float(metric)
            best_state = {
                "epoch": int(epoch),
                "step": int(step),
                "metric": float(metric),
                "criterion": args.best_metric,
                "mean_lossG": float(mean_lossG),
                "mean_lossD": float(mean_lossD),
                "mean_lm": float(mean_lm),
                "bg_miss_total": int(bg_miss_total),
                "args": {k: v for k, v in vars(args).items() if k != "seed"}  # optional
            }
            ckpt_dir = outdir / "checkpoints"
            torch.save(G.state_dict(), ckpt_dir / "G_best.pt")
            torch.save(D.state_dict(), ckpt_dir / "D_best.pt")
            with open(ckpt_dir / "best.json", "w", encoding="utf-8") as f:
                json.dump(best_state, f, indent=2, ensure_ascii=False)
            print(f"[BEST] updated: metric={best_metric:.6f} at epoch={epoch} step={step}")

    def build_d_batch() -> Dict[str, Any]:
        if not args.d_balance_posneg:
            return next(iter(loader))

        B = int(args.batch_size)
        pos_frac = max(0.0, min(1.0, float(args.d_pos_frac)))
        n_pos = int(round(B * pos_frac))
        n_pos = max(1, min(B - 1, n_pos))
        n_neg = B - n_pos

        pb = next(pos_iter)
        nb = next(neg_iter)

        def _take(b, n):
            out = {}
            for k, v in b.items():
                if k.startswith("_"):
                    out[k] = v
                    continue
                if torch.is_tensor(v):
                    out[k] = v[:n]
                elif isinstance(v, (list, tuple)):
                    out[k] = v[:n]
                else:
                    out[k] = v
            return out

        pb = _take(pb, n_pos)
        nb = _take(nb, n_neg)

        out = {}
        for k in pb.keys():
            if k.startswith("_"):
                continue
            if torch.is_tensor(pb[k]):
                out[k] = torch.cat([pb[k], nb[k]], dim=0)
            else:
                out[k] = list(pb[k]) + list(nb[k])

        out["_bg_miss"] = int(pb.get("_bg_miss", 0)) + int(nb.get("_bg_miss", 0))

        perm = torch.randperm(out["tokens"].size(0))
        for k in ["tokens", "label", "bg"]:
            out[k] = out[k][perm]
        for k in ["seq101", "sample_id", "dataset", "sample"]:
            out[k] = [out[k][i] for i in perm.tolist()]
        return out

    def _epoch_stats(epoch_num: int) -> Tuple[float, float, float, float]:
        """return (mean_lossG, mean_lossD, mean_lm, metric) for this epoch"""
        epoch_logs = [d for d in logs if d.get("epoch") == epoch_num]
        lossD_list = [d["loss_D"] for d in epoch_logs if d.get("loss_D") is not None]
        lossG_list = [d["loss_G"] for d in epoch_logs if d.get("loss_G") is not None]
        lm_list    = [d["lm"]     for d in epoch_logs if d.get("lm")     is not None]

        mean_lossD = float(np.mean(lossD_list)) if lossD_list else float("inf")
        mean_lossG = float(np.mean(lossG_list)) if lossG_list else float("inf")
        mean_lm    = float(np.mean(lm_list))    if lm_list    else float("inf")

        if args.best_metric == "lossG":
            metric = mean_lossG
        else:
            metric = mean_lossD + 0.1 * mean_lm

        return mean_lossG, mean_lossD, mean_lm, float(metric)

    try:
        for epoch in range(1, args.epochs + 1):
            steps_per_epoch = len(loader) if len(loader) > 0 else 1000

            for _ in range(steps_per_epoch):
                step += 1

                if epoch <= args.warmup_epochs:
                    # warmup uses smaller lr
                    _set_lr(optG, warmup_lr)

                    G.train()
                    optG.zero_grad(set_to_none=True)

                    pb = next(pos_iter)
                    bg_miss_total += int(pb.get("_bg_miss", 0))

                    pos_tok = pb["tokens"].to(device)
                    pos_bg = pb["bg"].to(device)

                    want_debug = (args.debug_every > 0 and step % args.debug_every == 0)

                    lm, logits, gamma, beta = seq_mlm_loss(G, pos_tok, pos_bg, mask_ratio=0.15, debug_return_logits=True)

                    bad = (
                        torch.isnan(pos_bg).any() or torch.isinf(pos_bg).any() or
                        torch.isnan(logits).any() or torch.isinf(logits).any() or
                        torch.isnan(lm) or torch.isinf(lm)
                    )

                    if bad:
                        if want_debug or nan_dumps < args.max_nan_dumps:
                            print(f"[NaN DETECTED @ step {step}] epoch={epoch} warmup_lr={warmup_lr} bg_miss_total={bg_miss_total}")
                            print("  bg_stats   :", _tensor_stats(pos_bg))
                            print("  logits_stats:", _tensor_stats(logits))
                            print("  gamma_stats :", _tensor_stats(gamma))
                            print("  beta_stats  :", _tensor_stats(beta))
                            print("  loss(lm)    :", float(lm.detach().cpu().item()) if lm.numel() else float("nan"))
                            nan_dumps += 1

                        optG.zero_grad(set_to_none=True)
                        if args.skip_nan_steps:
                            continue
                        else:
                            lm = torch.nan_to_num(lm, nan=0.0, posinf=0.0, neginf=0.0)

                    lm.backward()
                    if args.g_grad_clip and args.g_grad_clip > 0:
                        nn.utils.clip_grad_norm_(G.parameters(), float(args.g_grad_clip))
                    gnorm = _global_grad_norm(G)
                    if (np.isnan(gnorm) or np.isinf(gnorm)) and args.skip_nan_steps:
                        optG.zero_grad(set_to_none=True)
                        if want_debug or nan_dumps < args.max_nan_dumps:
                            print(f"[NaN GRAD @ step {step}] gnorm={gnorm} (skip step)")
                            nan_dumps += 1
                        continue

                    optG.step()

                    if want_debug:
                        print(f"[G-warmup step {step}] lm_pos={float(lm.item()):.6f} gnorm={gnorm:.4f} warmup_lr={warmup_lr:.2e} bg_miss_total={bg_miss_total}")
                        print("  bg_stats    :", _tensor_stats(pos_bg))
                        print("  logits_stats:", _tensor_stats(logits))
                        print("  beta_stats  :", _tensor_stats(beta))

                    # log warmup step (optional)
                    if args.debug_every > 0 and step % args.debug_every == 0:
                        logs.append({"step": step, "epoch": epoch,
                                     "loss_D": None, "loss_G": None,
                                     "adv": None, "lm": float(lm.item()), "center": None,
                                     "bg_miss_total": int(bg_miss_total)})
                    continue

                # after warmup restore lr
                _set_lr(optG, args.lr_g)

                # -------------------- adversarial training --------------------
                db = build_d_batch()
                bg_miss_total += int(db.get("_bg_miss", 0))

                real_tok = db["tokens"].to(device)
                real_lbl = db["label"].to(device)
                real_bg = db["bg"].to(device)

                pos_mask = (real_lbl > 0.5)
                neg_mask = ~pos_mask

                if args.g_adv_sample_uniform:
                    idx = np.random.randint(0, len(bg_keys), size=(real_bg.size(0),))
                    bg_np = np.stack([bg_map[bg_keys[i]] for i in idx], axis=0).astype(np.float32)
                    bg_np = np.nan_to_num(bg_np, nan=0.0, posinf=0.0, neginf=0.0)
                    denom = np.linalg.norm(bg_np, axis=1, keepdims=True) + 1e-8
                    bg_np = bg_np / denom
                    bg_np = np.clip(bg_np, -5.0, 5.0)
                    fake_bg = torch.from_numpy(bg_np).to(device)
                else:
                    fake_bg = real_bg

                # generate fake (soft) sequences for D training
                G.eval()
                with torch.no_grad():
                    fake_soft = G.sample_gumbel_soft(fake_bg, tau=args.tau)

                # ---- D step ----
                D.train()
                optD.zero_grad(set_to_none=True)

                logit_fake = D(fake_soft, fake_bg, is_soft=True)
                logit_pos = D(real_tok[pos_mask], real_bg[pos_mask], is_soft=False) if pos_mask.any() else None
                logit_neg = D(real_tok[neg_mask], real_bg[neg_mask], is_soft=False) if neg_mask.any() else None

                loss_D = hinge_d_loss(logit_pos, logit_neg, logit_fake)

                badD = (
                    torch.isnan(loss_D) or torch.isinf(loss_D) or
                    torch.isnan(logit_fake).any() or torch.isinf(logit_fake).any() or
                    (logit_pos is not None and (torch.isnan(logit_pos).any() or torch.isinf(logit_pos).any())) or
                    (logit_neg is not None and (torch.isnan(logit_neg).any() or torch.isinf(logit_neg).any()))
                )

                if badD:
                    if args.skip_nan_steps:
                        optD.zero_grad(set_to_none=True)
                        continue
                    loss_D = torch.nan_to_num(loss_D, nan=0.0, posinf=0.0, neginf=0.0)

                loss_D.backward()
                if args.d_grad_clip and args.d_grad_clip > 0:
                    nn.utils.clip_grad_norm_(D.parameters(), float(args.d_grad_clip))
                gnormD = _global_grad_norm(D)
                if (np.isnan(gnormD) or np.isinf(gnormD)) and args.skip_nan_steps:
                    optD.zero_grad(set_to_none=True)
                    continue

                optD.step()

                # If still in D warmup, skip G update
                if epoch <= args.warmup_epochs + args.d_warmup_epochs:
                    if args.debug_every > 0 and step % args.debug_every == 0:
                        print(f"[D-warmup step {step}] loss_D={float(loss_D.item()):.6f} gnormD={gnormD:.4f} bg_miss_total={bg_miss_total}")
                    logs.append({"step": step, "epoch": epoch, "loss_D": float(loss_D.item()), "loss_G": None,
                                 "adv": None, "lm": None, "center": None, "bg_miss_total": int(bg_miss_total)})
                    continue

                # ---- G step ----
                G.train()
                optG.zero_grad(set_to_none=True)

                fake_soft = G.sample_gumbel_soft(fake_bg, tau=args.tau)
                logit_fake_for_g = D(fake_soft, fake_bg, is_soft=True)

                adv = hinge_g_adv(logit_fake_for_g)
                center = center_penalty(fake_soft, weight=1.0)

                pb = next(pos_iter)
                pos_tok = pb["tokens"].to(device)
                pos_bg = pb["bg"].to(device)
                lm = seq_mlm_loss(G, pos_tok, pos_bg, mask_ratio=0.15)

                loss_G = args.lambda_adv * adv + args.lambda_center * center + args.lambda_lm * lm

                badG = (
                    torch.isnan(loss_G) or torch.isinf(loss_G) or
                    torch.isnan(adv) or torch.isinf(adv) or
                    torch.isnan(lm) or torch.isinf(lm) or
                    torch.isnan(fake_soft).any() or torch.isinf(fake_soft).any()
                )

                if badG:
                    if args.skip_nan_steps:
                        optG.zero_grad(set_to_none=True)
                        continue
                    loss_G = torch.nan_to_num(loss_G, nan=0.0, posinf=0.0, neginf=0.0)

                loss_G.backward()
                if args.g_grad_clip and args.g_grad_clip > 0:
                    nn.utils.clip_grad_norm_(G.parameters(), float(args.g_grad_clip))
                gnormG = _global_grad_norm(G)
                if (np.isnan(gnormG) or np.isinf(gnormG)) and args.skip_nan_steps:
                    optG.zero_grad(set_to_none=True)
                    continue
                optG.step()

                if args.debug_every > 0 and step % args.debug_every == 0:
                    print(f"[GAN step {step}] loss_D={float(loss_D.item()):.6f} loss_G={float(loss_G.item()):.6f} "
                          f"adv={float(adv.item()):.6f} lm={float(lm.item()):.6f} center={float(center.item()):.6f} "
                          f"gnormD={gnormD:.4f} gnormG={gnormG:.4f} bg_miss_total={bg_miss_total}")

                logs.append({"step": step, "epoch": epoch,
                             "loss_D": float(loss_D.item()), "loss_G": float(loss_G.item()),
                             "adv": float(adv.item()), "lm": float(lm.item()), "center": float(center.item()),
                             "bg_miss_total": int(bg_miss_total)})

            print(f"[Epoch {epoch}] done.")

            # ---- end of epoch: save last + maybe best + write logs ----
            if (epoch % int(args.save_every_epochs)) == 0:
                _save_last("last")

                mean_lossG, mean_lossD, mean_lm, metric = _epoch_stats(epoch)
                print(f"[EPOCH {epoch}] mean_lossD={mean_lossD:.6f} mean_lossG={mean_lossG:.6f} mean_lm={mean_lm:.6f} metric={metric:.6f}")

                _save_best(metric=metric, epoch=epoch, step=step,
                           mean_lossG=mean_lossG, mean_lossD=mean_lossD, mean_lm=mean_lm)

                # dump logs each epoch so directory is never empty
                try:
                    dump_logs_jsonl(logs, outdir, "train_loss.jsonl")
                except Exception as e:
                    print(f"[WARN] dump_logs_jsonl failed: {e}")

        print(f"[OK] training finished. outdir={outdir}")

    except KeyboardInterrupt:
        print("[INTERRUPT] saving last/best/logs before exit...")
        try:
            _save_last("interrupt")
        except Exception as e:
            print(f"[WARN] saving interrupt ckpt failed: {e}")

        try:
            with open(outdir / "checkpoints" / "best.json", "w", encoding="utf-8") as f:
                json.dump(best_state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[WARN] writing best.json failed: {e}")

        try:
            dump_logs_jsonl(logs, outdir, "train_loss.jsonl")
        except Exception as e:
            print(f"[WARN] dump_logs_jsonl failed: {e}")

        raise

if __name__ == "__main__":
    main()