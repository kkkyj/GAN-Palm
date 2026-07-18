#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_discriminator_posneg.py  (patched: stable CNN option + NaN guards)

- Adds --arch {cnn,gan} (default: cnn)
- Adds --no-bg (ignore bg, pure token classifier)
- Adds --subsample N (train on subset for fast debug)
- Makes imbalance handling safe:
    * If --balanced-sampler enabled, default pos_weight -> none (avoid double weighting)
- Adds isfinite checks for tokens/bg/logits/loss and logit stats printing.

Usage (recommended first run):
python train_discriminator_posneg.py \
  --gan-script code/conditional_seq_gan_noesm.py \
  --site-table data/site_sample_long.human.parquet \
  --bg-npz     data/proteome_bg_embeddings.dataset.npz \
  --outdir runs/disc_posneg_human_cnn \
  --arch cnn \
  --balanced-sampler \
  --pos-weight none \
  --lr 5e-5 --batch-size 512 --epochs 3 --device cuda
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    accuracy_score,
    f1_score,
)

import matplotlib.pyplot as plt


# ----------------------- dynamic import -----------------------
def load_module_from_path(py_path: str):
    spec = importlib.util.spec_from_file_location("gan_script", py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import gan script: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------- bg loading -----------------------
def load_bg_db(bg_npz: str) -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(bg_npz, allow_pickle=True)
    keys = set(z.files)

    if "combined_emb" in keys:
        vecs = z["combined_emb"]
    elif "vecs" in keys:
        vecs = z["vecs"]
    elif "X" in keys:
        vecs = z["X"]
    elif "emb" in keys:
        vecs = z["emb"]
    else:
        raise KeyError(f"Cannot find vectors in {bg_npz}, keys={z.files}")

    if "samples" in keys:
        ids = z["samples"].astype(str)
    elif "ids" in keys:
        ids = z["ids"].astype(str)
    elif "names" in keys:
        ids = z["names"].astype(str)
    else:
        raise KeyError(f"Cannot find id list in {bg_npz}, keys={z.files}")

    vecs = np.asarray(vecs, dtype=np.float32)
    if vecs.ndim != 2:
        raise ValueError(f"bg vectors must be 2D (N,D), got {vecs.shape}")
    return vecs, ids


# ----------------------- tokenization -----------------------
def build_tokenizer(gan_mod):
    if not hasattr(gan_mod, "AA2IDX"):
        raise AttributeError("gan script missing AA2IDX")
    aa2idx = gan_mod.AA2IDX
    unk = aa2idx.get("X", max(aa2idx.values()))
    pad = getattr(gan_mod, "PAD_IDX", aa2idx.get("X", unk))
    # vocab_size is max idx + 1
    vocab_size = max(aa2idx.values()) + 1
    return aa2idx, unk, pad, vocab_size


def tokenize_seq101(seqs: List[str], aa2idx: Dict[str, int], unk_idx: int) -> torch.LongTensor:
    arr = np.full((len(seqs), 101), unk_idx, dtype=np.int64)
    for i, s in enumerate(seqs):
        s = str(s)
        if len(s) != 101:
            raise ValueError(f"seq length != 101 at row {i}: {len(s)}")
        arr[i] = [aa2idx.get(ch, unk_idx) for ch in s]
    return torch.from_numpy(arr)


# ----------------------- dataset -----------------------
class RealPosNegDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        aa2idx: Dict[str, int],
        unk_idx: int,
        bg_vecs: np.ndarray,
        bg_ids: np.ndarray,
        seq_col: str,
        label_col: str,
        dataset_col: str,
        sample_col: str,
        use_bg: bool = True,
        strict_bg: bool = True,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.seq_col = seq_col
        self.label_col = label_col
        self.dataset_col = dataset_col
        self.sample_col = sample_col

        self.aa2idx = aa2idx
        self.unk_idx = unk_idx

        self.use_bg = bool(use_bg)

        self.bg_vecs = bg_vecs
        self.bg_ids = bg_ids
        self.id2row = {k: i for i, k in enumerate(bg_ids.tolist())}

        # precompute bg_row indices
        self.df["_bg_key"] = (
            self.df[self.dataset_col].astype(str) + "__" + self.df[self.sample_col].astype(str)
        )
        bg_rows = []
        miss = 0
        for k in self.df["_bg_key"].tolist():
            if k in self.id2row:
                bg_rows.append(self.id2row[k])
            else:
                miss += 1
                if strict_bg:
                    # mark as -1, and we'll replace with zeros in __getitem__
                    bg_rows.append(-1)
                else:
                    bg_rows.append(np.random.randint(0, self.bg_vecs.shape[0]))
        self.bg_rows = np.asarray(bg_rows, dtype=np.int64)
        self.miss = miss

        # labels
        y = self.df[self.label_col].astype(int).values
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError(f"{label_col} must be binary 0/1. got unique={np.unique(y)}")
        self.y = y.astype(np.int64)

        self.bg_dim = int(self.bg_vecs.shape[1]) if self.bg_vecs is not None else 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        seq = self.df.at[idx, self.seq_col]
        tok = tokenize_seq101([seq], self.aa2idx, self.unk_idx)[0]  # (101,)

        if self.use_bg:
            r = int(self.bg_rows[idx])
            if r >= 0:
                bg = torch.from_numpy(self.bg_vecs[r]).float()
            else:
                bg = torch.zeros(self.bg_dim, dtype=torch.float32)
        else:
            # still return a placeholder tensor to keep collate stable
            bg = torch.zeros(self.bg_dim, dtype=torch.float32)

        y = torch.tensor(self.y[idx]).float()
        return {"tokens": tok, "bg": bg, "y": y}


def collate_batch(batch):
    tokens = torch.stack([b["tokens"] for b in batch], dim=0)  # (B,101)
    bg = torch.stack([b["bg"] for b in batch], dim=0)          # (B,D)
    y = torch.stack([b["y"] for b in batch], dim=0).view(-1)   # (B,)
    return {"tokens": tokens, "bg": bg, "y": y}


# ----------------------- CNN discriminator -----------------------
class CNNDisc(nn.Module):
    """
    Stable CNN discriminator. Outputs logits (B,1).
    """
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        bg_dim: int,
        use_bg: bool = True,
        emb_dim: int = 64,
        channels: int = 128,
        kernels=(3, 5, 7),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_bg = bool(use_bg)
        self.bg_dim = int(bg_dim) if self.use_bg else 0

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=int(pad_idx))

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(emb_dim, channels, kernel_size=int(k), padding=int(k)//2),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for k in kernels
        ])

        if self.bg_dim > 0:
            self.bg_ln = nn.LayerNorm(self.bg_dim, eps=1e-5)
        else:
            self.bg_ln = None

        feat_dim = channels * len(kernels) + self.bg_dim
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, tokens: torch.LongTensor, bg: torch.Tensor, is_soft: bool = False):
        x = self.emb(tokens)          # (B,L,E)
        x = x.transpose(1, 2)         # (B,E,L)

        feats = []
        for block in self.convs:
            h = block(x)              # (B,C,L)
            h = torch.amax(h, dim=-1) # (B,C)
            feats.append(h)
        h = torch.cat(feats, dim=1)   # (B,C*nk)

        if self.bg_dim > 0:
            bg2 = self.bg_ln(bg)
            h = torch.cat([h, bg2], dim=1)

        logit = self.mlp(h)           # (B,1)
        if is_soft:
            return torch.sigmoid(logit)
        return logit


# ----------------------- eval helpers -----------------------
@torch.no_grad()
def predict_logits(model, dl, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys = []
    ls = []
    for batch in dl:
        tok = batch["tokens"].to(device)
        bg = batch["bg"].to(device)
        y = batch["y"].detach().cpu().numpy()
        logit = model(tok, bg, is_soft=False).view(-1).float()
        ys.append(y)
        ls.append(logit.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(ls)


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    p, r, t = precision_recall_curve(y_true, y_prob)
    f1 = (2 * p[:-1] * r[:-1]) / np.clip(p[:-1] + r[:-1], 1e-12, None)
    j = int(np.nanargmax(f1))
    return float(t[j]), float(f1[j])


def save_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, out_prefix: Path, title: str):
    from sklearn.metrics import roc_curve, precision_recall_curve

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC {title}")
    plt.tight_layout()
    plt.savefig(str(out_prefix) + "_roc.png", dpi=200)
    plt.close()

    p, r, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR {title}")
    plt.tight_layout()
    plt.savefig(str(out_prefix) + "_pr.png", dpi=200)
    plt.close()


def is_finite_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


# ----------------------- main -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gan-script", required=True, help="path to conditional_seq_gan_noesm.py")
    ap.add_argument("--site-table", required=True, help="parquet with seq/label/dataset/sample")
    ap.add_argument("--bg-npz", required=True, help="npz with combined_emb and samples")

    ap.add_argument("--seq-col", default="seq101")
    ap.add_argument("--label-col", default="label_bin")
    ap.add_argument("--dataset-col", default="dataset")
    ap.add_argument("--sample-col", default="sample")

    ap.add_argument("--outdir", required=True)

    # NEW:
    ap.add_argument("--arch", default="cnn", choices=["cnn", "gan"],
                    help="cnn: built-in stable CNN discriminator; gan: use ConditionalSequenceDisc from gan-script")
    ap.add_argument("--no-bg", action="store_true", help="ignore bg condition, pure token classifier")
    ap.add_argument("--subsample", type=int, default=0,
                    help="if >0, subsample TRAIN set to this many rows (keeps all val). Useful to debug quickly.")
    ap.add_argument("--strict-bg", action="store_true",
                    help="if bg_key missing, use zero bg vec (instead of random row). Recommended for debugging.")

    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=2)

    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=13)

    ap.add_argument("--balanced-sampler", action="store_true",
                    help="use WeightedRandomSampler to balance pos/neg")
    ap.add_argument("--pos-weight", default="auto",
                    help='BCE pos_weight: "auto" or float or "none". NOTE: do not combine with balanced-sampler.')

    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=200)

    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "checkpoints").mkdir(exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    gan_mod = load_module_from_path(args.gan_script)
    aa2idx, unk_idx, pad_idx, vocab_size = build_tokenizer(gan_mod)

    # load bg
    bg_vecs, bg_ids = load_bg_db(args.bg_npz)
    bg_dim = int(bg_vecs.shape[1])
    use_bg = (not args.no_bg)

    # load dataframe
    df = pd.read_parquet(args.site_table)
    need = [args.seq_col, args.label_col, args.dataset_col, args.sample_col]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {args.site_table}")
    df = df[need].dropna().reset_index(drop=True)

    # split train/val
    n = len(df)
    idx = np.arange(n)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    n_val = int(math.floor(n * args.val_frac))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_va = df.iloc[val_idx].reset_index(drop=True)

    # optional subsample train for debugging speed/stability
    if args.subsample and args.subsample > 0 and args.subsample < len(df_tr):
        rng2 = np.random.default_rng(args.seed + 1)
        keep = rng2.choice(len(df_tr), size=int(args.subsample), replace=False)
        df_tr = df_tr.iloc[keep].reset_index(drop=True)
        print(f"[INFO] subsample train -> n={len(df_tr)}")

    ds_tr = RealPosNegDataset(
        df_tr, aa2idx, unk_idx, bg_vecs, bg_ids,
        seq_col=args.seq_col, label_col=args.label_col,
        dataset_col=args.dataset_col, sample_col=args.sample_col,
        use_bg=use_bg,
        strict_bg=bool(args.strict_bg),
    )
    ds_va = RealPosNegDataset(
        df_va, aa2idx, unk_idx, bg_vecs, bg_ids,
        seq_col=args.seq_col, label_col=args.label_col,
        dataset_col=args.dataset_col, sample_col=args.sample_col,
        use_bg=use_bg,
        strict_bg=bool(args.strict_bg),
    )

    print(f"[INFO] train n={len(ds_tr)} (miss_bg={ds_tr.miss}), val n={len(ds_va)} (miss_bg={ds_va.miss})")
    print(f"[INFO] bg_dim={bg_dim}, use_bg={use_bg}, arch={args.arch}")

    # sampler for imbalance
    sampler = None
    if args.balanced_sampler:
        y_tr = ds_tr.y
        n_pos = int((y_tr == 1).sum())
        n_neg = int((y_tr == 0).sum())
        w_pos = 0.5 / max(n_pos, 1)
        w_neg = 0.5 / max(n_neg, 1)
        weights = np.where(y_tr == 1, w_pos, w_neg).astype(np.float64)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        print(f"[INFO] balanced sampler enabled: n_pos={n_pos} n_neg={n_neg} w_pos={w_pos:.3e} w_neg={w_neg:.3e}")

        # Safety: if balanced sampler is on and pos_weight=auto, silently disable it unless user explicitly set a float
        if str(args.pos_weight).lower() == "auto":
            args.pos_weight = "none"
            print("[INFO] balanced-sampler ON: forcing pos_weight=none to avoid double weighting")

    dl_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
        drop_last=True,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
        drop_last=False,
    )

    # build discriminator
    if args.arch == "cnn":
        D = CNNDisc(
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            bg_dim=bg_dim,
            use_bg=use_bg,
            emb_dim=64,
            channels=128,
            kernels=(3, 5, 7),
            dropout=0.1,
        )
    else:
        if not hasattr(gan_mod, "ConditionalSequenceDisc"):
            raise AttributeError("gan script must expose ConditionalSequenceDisc(bg_dim=...)")
        D = gan_mod.ConditionalSequenceDisc(bg_dim=bg_dim)

    device = torch.device(args.device)
    D.to(device)

    # loss
    y_tr = ds_tr.y
    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())

    pos_weight_tensor: Optional[torch.Tensor] = None
    if str(args.pos_weight).lower() == "none":
        pos_weight_tensor = None
        print("[INFO] pos_weight disabled")
    elif str(args.pos_weight).lower() == "auto":
        pw = float(n_neg / max(n_pos, 1))
        pos_weight_tensor = torch.tensor([pw], device=device)
        print(f"[INFO] pos_weight auto = {pw:.4f}")
    else:
        pw = float(args.pos_weight)
        pos_weight_tensor = torch.tensor([pw], device=device)
        print(f"[INFO] pos_weight = {pw:.4f}")

    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor) if pos_weight_tensor is not None else nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(D.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # training loop
    best_val_pr = -1.0
    best_path = outdir / "checkpoints" / "D_best.pt"
    log_path = outdir / "metrics.jsonl"

    step = 0
    history = {"train_loss": [], "val_roc": [], "val_pr": []}

    for epoch in range(1, args.epochs + 1):
        D.train()
        running = 0.0
        nb = 0

        for batch in dl_tr:
            step += 1
            tok = batch["tokens"].to(device, non_blocking=True)
            bg = batch["bg"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True).view(-1)

            # NaN guards (inputs)
            if not is_finite_tensor(bg):
                raise RuntimeError(f"[NaN] bg not finite at step {step}: min={bg.min().item()} max={bg.max().item()}")
            # tokens are int64; no isfinite check needed

            opt.zero_grad(set_to_none=True)

            logit = D(tok, bg, is_soft=False).view(-1).float()

            if not is_finite_tensor(logit):
                raise RuntimeError(f"[NaN] logit not finite at step {step}: min={logit.min().item()} max={logit.max().item()}")

            loss = bce(logit, y)

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"[NaN] loss not finite at step {step}: loss={loss.item()} "
                    f"logit(min/max)={logit.min().item():.3f}/{logit.max().item():.3f}"
                )

            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(D.parameters(), float(args.grad_clip))
            opt.step()

            running += float(loss.item())
            nb += 1

            if step % args.log_every == 0:
                with torch.no_grad():
                    mx = float(logit.max().item())
                    mn = float(logit.min().item())
                    me = float(logit.mean().item())
                print(f"[train step {step}] loss={loss.item():.4f} logit(mean/min/max)={me:.3f}/{mn:.3f}/{mx:.3f}")

        train_loss = running / max(nb, 1)

        # eval
        y_va, logit_va = predict_logits(D, dl_va, args.device)
        # stable sigmoid for numpy
        prob_va = 1.0 / (1.0 + np.exp(-np.clip(logit_va, -50, 50)))

        # metrics
        val_roc = float(roc_auc_score(y_va, prob_va)) if len(np.unique(y_va)) == 2 else float("nan")
        val_pr = float(average_precision_score(y_va, prob_va)) if len(np.unique(y_va)) == 2 else float("nan")
        thr, bestf1 = best_f1_threshold(y_va, prob_va)
        pred = (prob_va >= thr).astype(np.int32)
        val_acc = float(accuracy_score(y_va, pred))
        val_f1 = float(f1_score(y_va, pred))

        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_roc_auc": val_roc,
            "val_pr_auc": val_pr,
            "val_best_f1": float(bestf1),
            "val_best_f1_thr": float(thr),
            "val_acc_at_bestf1": val_acc,
            "val_f1_at_bestf1": val_f1,
            "n_train": len(ds_tr),
            "n_val": len(ds_va),
            "n_pos_train": n_pos,
            "n_neg_train": n_neg,
            "bg_miss_train": int(ds_tr.miss),
            "bg_miss_val": int(ds_va.miss),
            "lr": float(args.lr),
            "pos_weight": None if pos_weight_tensor is None else float(pos_weight_tensor.item()),
            "balanced_sampler": bool(args.balanced_sampler),
            "arch": args.arch,
            "use_bg": bool(use_bg),
        }

        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} "
            f"val_ROC={val_roc:.4f} val_PR={val_pr:.4f} "
            f"bestF1={bestf1:.4f} thr={thr:.4g}"
        )

        history["train_loss"].append(train_loss)
        history["val_roc"].append(val_roc)
        history["val_pr"].append(val_pr)

        # save best by PR-AUC
        if not math.isnan(val_pr) and val_pr > best_val_pr:
            best_val_pr = val_pr
            torch.save(D.state_dict(), best_path)
            print(f"[OK] saved best D to {best_path} (val_PR={best_val_pr:.4f})")

    # save last
    last_path = outdir / "checkpoints" / "D_last.pt"
    torch.save(D.state_dict(), last_path)
    print(f"[OK] saved last D to {last_path}")

    # plots
    try:
        plt.figure()
        plt.plot(history["train_loss"])
        plt.xlabel("epoch")
        plt.ylabel("train_loss")
        plt.title("Train loss")
        plt.tight_layout()
        plt.savefig(outdir / "train_loss.png", dpi=200)
        plt.close()

        plt.figure()
        plt.plot(history["val_roc"], label="val_ROC")
        plt.plot(history["val_pr"], label="val_PR")
        plt.xlabel("epoch")
        plt.ylabel("metric")
        plt.title("Validation metrics")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "val_metrics.png", dpi=200)
        plt.close()

        # final ROC/PR curves on val using best checkpoint (if exists)
        if best_path.exists():
            if args.arch == "cnn":
                D2 = CNNDisc(
                    vocab_size=vocab_size,
                    pad_idx=pad_idx,
                    bg_dim=bg_dim,
                    use_bg=use_bg,
                ).to(device)
            else:
                D2 = gan_mod.ConditionalSequenceDisc(bg_dim=bg_dim).to(device)

            D2.load_state_dict(torch.load(best_path, map_location="cpu"), strict=True)
            y_va, logit_va = predict_logits(D2, dl_va, args.device)
            prob_va = 1.0 / (1.0 + np.exp(-np.clip(logit_va, -50, 50)))
            save_roc_pr(y_va, prob_va, outdir / "val", "val pos-vs-neg (best)")
    except Exception as e:
        print(f"[WARN] plot failed: {e}")

    # summary json
    summary = {
        "best_val_pr_auc": best_val_pr,
        "best_ckpt": str(best_path) if best_path.exists() else None,
        "last_ckpt": str(last_path),
        "metrics_log": str(log_path),
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] wrote summary.json to {outdir}")


if __name__ == "__main__":
    main()