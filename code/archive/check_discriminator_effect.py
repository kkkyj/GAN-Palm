#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_discriminator_effect.py

Discriminator effect check (the "no-ESM path" version of the GAN no-ESM / with-ESM setup):
- Take real samples from the real data
- Sample fake examples with the generator from the same bg_vec
- Have the discriminator score real/fake
- Compute metrics such as ROC-AUC / PR-AUC / Acc, and report them grouped by dataset
- Check whether D "only reads bg" (correlation of score with bg_norm / bg_pcs)

Dependencies:
- Your GAN training script defines the Generator/Discriminator (or equivalent class names),
  so here we use --gan-script to dynamically import those class definitions from that script.

Usage example:
python check_discriminator_effect.py \
  --gan-script code/conditional_seq_gan_noesm.py \
  --ckpt-dir   runs/gan_noesm_human/checkpoints \
  --site-table data/site_sample_long.human.parquet \
  --bg-npz     data/proteome_bg_embeddings.dataset.npz \
  --seq-col seq101 --dataset-col dataset --sample-col sample \
  --n-real 50000 --batch-size 256 --device cuda
"""

import argparse
import importlib.util
import math
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch

import inspect

@torch.no_grad()
def sample_with_fallback(G, bg: torch.Tensor, n: int, seq_len: int, center_pos: int,
                         aa2idx: Dict[str, int], temperature: float, top_k: int) -> torch.Tensor:
    """
    Try:
      1) G.sample(bg, n=..., temperature=..., top_k=...)
      2) If no .sample: call G.forward with common signatures and sample from logits once.
    Returns: tokens [n, seq_len]
    """
    device = bg.device
    C_idx = aa2idx["C"]
    X_idx = aa2idx.get("X", aa2idx.get("B", 0))

    # (1) Native sample
    if hasattr(G, "sample") and callable(getattr(G, "sample")):
        toks = G.sample(bg, n=n, temperature=temperature, top_k=top_k)
        toks = toks[:, :seq_len]
        toks[:, center_pos] = C_idx
        return toks

    # (2) Fallback: sample from forward logits (non-autoregressive single pass)
    # Build a dummy token grid (all X). Many generators take (tokens, bg).
    dummy = torch.full((n, seq_len), X_idx, dtype=torch.long, device=device)

    # Try common forward signatures
    logits = None
    try:
        # forward(tokens, bg)
        logits = G(dummy, bg)
    except TypeError:
        try:
            # forward(bg, tokens)
            logits = G(bg, dummy)
        except TypeError:
            try:
                # forward(bg) only
                logits = G(bg)
            except TypeError as e:
                raise RuntimeError(
                    "Generator has no .sample(), and its forward() signature is not recognized. "
                    "Please paste the Generator.forward(...) definition (a few lines) from conditional_seq_gan_noesm.py."
                ) from e

    # logits expected [n, seq_len, vocab]
    if logits.dim() != 3:
        raise RuntimeError(f"Expected logits with shape [n, L, V], got {tuple(logits.shape)}")

    # temperature + top-k
    logits = logits[:, :seq_len, :] / max(1e-6, float(temperature))
    if top_k and top_k > 0:
        v, ix = torch.topk(logits, k=top_k, dim=-1)
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(-1, ix, v)
        logits = mask

    probs = torch.softmax(logits, dim=-1)
    toks = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1).reshape(n, seq_len)
    toks[:, center_pos] = C_idx
    return toks


# ---------------- metrics (no sklearn dependency) ----------------

def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # Mann–Whitney U equivalence. y_true in {0,1}
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)

    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    sum_ranks_pos = float(ranks[pos].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

def pr_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # simple step integration on precision-recall curve
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    denom = tp + fp
    precision = tp / np.maximum(denom, 1)
    recall = tp / np.maximum(tp[-1], 1)

    # integrate precision over recall (rectangle rule)
    # add (0,1) start
    recall_prev = 0.0
    ap = 0.0
    for p, r in zip(precision, recall):
        ap += float(p) * float(r - recall_prev)
        recall_prev = float(r)
    return float(ap)

def accuracy_from_scores(y_true: np.ndarray, y_score: np.ndarray, thr: float = 0.0) -> float:
    pred = (y_score >= thr).astype(np.int64)
    return float((pred == y_true.astype(np.int64)).mean())


# ---------------- helpers ----------------

def load_module_from_path(py_path: Path):
    spec = importlib.util.spec_from_file_location(py_path.stem, str(py_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

def load_bg_map(bg_npz: Path) -> Dict[str, np.ndarray]:
    z = np.load(bg_npz, allow_pickle=True)
    samples = z["samples"].astype(object)
    emb = z["combined_emb"].astype(np.float32)
    mp = {str(s): emb[i] for i, s in enumerate(samples)}
    return mp

def make_bg_vec(df: pd.DataFrame, bg_map: Dict[str, np.ndarray], dataset_col: str, sample_col: str) -> np.ndarray:
    keys = (df[dataset_col].astype(str) + "__" + df[sample_col].astype(str)).to_list()
    out = np.stack([bg_map[k] for k in keys], axis=0).astype(np.float32)
    # normalize (consistent with how you used bg previously)
    out /= (np.linalg.norm(out, axis=1, keepdims=True) + 1e-8)
    return out

def seqs_to_tokens(seqs: np.ndarray, aa2idx: Dict[str, int], center_pos: int, force_center_c: bool = True) -> np.ndarray:
    # seqs: array of strings shape [N]
    N = len(seqs)
    L = len(seqs[0])
    toks = np.empty((N, L), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = str(s).upper()
        for j, ch in enumerate(s):
            toks[i, j] = aa2idx.get(ch, aa2idx.get("X", aa2idx.get("B", 0)))
    if force_center_c:
        toks[:, center_pos] = aa2idx["C"]
    return toks

@torch.no_grad()
def batched_scores(D, tokens: torch.Tensor, bg: torch.Tensor, batch_size: int) -> np.ndarray:
    D.eval()
    outs = []
    n = tokens.size(0)
    for i in range(0, n, batch_size):
        t = tokens[i:i+batch_size]
        b = bg[i:i+batch_size]
        logit = D(t, b).view(-1)
        outs.append(logit.detach().float().cpu().numpy())
    return np.concatenate(outs, axis=0)

@torch.no_grad()
def batched_sample(G, bg: torch.Tensor, n: int, batch_size: int, temperature: float, top_k: int,
                   seq_len: int, center_pos: int, aa2idx: Dict[str, int]) -> torch.Tensor:
    G.eval()
    outs = []
    for i in range(0, n, batch_size):
        b = bg[i:i+batch_size]
        toks = sample_with_fallback(
            G, b, n=b.size(0),
            seq_len=seq_len, center_pos=center_pos,
            aa2idx=aa2idx,
            temperature=temperature, top_k=top_k
        )
        outs.append(toks.detach().cpu())
    return torch.cat(outs, dim=0)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gan-script", required=True, help="path to conditional_seq_gan_noesm.py")
    ap.add_argument("--ckpt-dir", required=True, help="directory containing G_final.pt / D_final.pt")
    ap.add_argument("--site-table", required=True)
    ap.add_argument("--bg-npz", required=True)

    ap.add_argument("--seq-col", default="seq101")
    ap.add_argument("--dataset-col", default="dataset")
    ap.add_argument("--sample-col", default="sample")

    ap.add_argument("--n-real", type=int, default=50000, help="how many real samples to draw for evaluation")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)

    ap.add_argument("--D-ckpt", default="D_final.pt")
    ap.add_argument("--G-ckpt", default="G_final.pt")

    # compatibility with your script constants
    ap.add_argument("--center-pos", type=int, default=50)
    ap.add_argument("--force-center-c", action="store_true", default=True)

    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # load gan module
    mod = load_module_from_path(Path(args.gan_script))

    # try to get vocab / aa2idx from module else fallback
    vocab = getattr(mod, "VOCAB", list("ACDEFGHIKLMNPQRSTVWYBX"))
    aa2idx = getattr(mod, "AA2IDX", {a: i for i, a in enumerate(vocab)})

    # load ckpts
    ckpt_dir = Path(args.ckpt_dir)
    d_path = ckpt_dir / args.D_ckpt
    g_path = ckpt_dir / args.G_ckpt
    d_obj = torch.load(d_path, map_location="cpu")
    g_obj = torch.load(g_path, map_location="cpu")

    def extract_state(x):
        if isinstance(x, dict) and "state_dict" in x:
            return x["state_dict"]
        if isinstance(x, dict):
            return x
        raise ValueError("Unsupported ckpt format")

    d_state = extract_state(d_obj)
    g_state = extract_state(g_obj)

    # build models: try to find the class names in the module
    # in your noesm script they are usually called Generator / ConditionalSequenceDisc (or similar)
    G_cls = getattr(mod, "Generator", None)
    D_cls = getattr(mod, "ConditionalSequenceDisc", None) or getattr(mod, "Discriminator", None)

    if G_cls is None or D_cls is None:
        raise RuntimeError("Could not find the Generator or ConditionalSequenceDisc/Discriminator class in gan-script. Please verify the class names.")

    # infer bg_dim: grab one entry from bg_npz
    bg_map = load_bg_map(Path(args.bg_npz))
    any_bg = next(iter(bg_map.values()))
    bg_dim = int(any_bg.shape[0])

    # instantiate
    # if your __init__ needs other arguments, this will raise an error; if so, paste the error and I'll fix it per your script's signature
    G = G_cls(bg_dim=bg_dim).to(device)
    D = D_cls(bg_dim=bg_dim).to(device)

    missing, unexpected = G.load_state_dict(g_state, strict=False)
    print(f"[INFO] load G: missing={len(missing)} unexpected={len(unexpected)}")
    missing, unexpected = D.load_state_dict(d_state, strict=False)
    print(f"[INFO] load D: missing={len(missing)} unexpected={len(unexpected)}")

    G.eval(); D.eval()

    # load data (only needed cols)
    p = Path(args.site_table)
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p, columns=[args.seq_col, args.dataset_col, args.sample_col])
    else:
        df = pd.read_csv(p, usecols=[args.seq_col, args.dataset_col, args.sample_col])

    # sample real
    df = df.dropna()
    if len(df) > args.n_real:
        df = df.sample(n=args.n_real, random_state=args.seed)

    # build bg + tokens
    bg_np = make_bg_vec(df, bg_map, args.dataset_col, args.sample_col)
    seqs = df[args.seq_col].astype(str).to_numpy()
    toks_np = seqs_to_tokens(seqs, aa2idx, center_pos=args.center_pos, force_center_c=args.force_center_c)

    bg = torch.tensor(bg_np, dtype=torch.float32, device=device)
    real_tokens = torch.tensor(toks_np, dtype=torch.long, device=device)

    # generate fake with same bg distribution
    fake_tokens = batched_sample(
        G, bg, n=real_tokens.size(0), batch_size=args.batch_size,
        temperature=args.temperature, top_k=args.top_k,
        seq_len=real_tokens.size(1),
        center_pos=args.center_pos,
        aa2idx=aa2idx
    ).to(device)

    # scores
    real_scores = batched_scores(D, real_tokens, bg, batch_size=args.batch_size)
    fake_scores = batched_scores(D, fake_tokens, bg, batch_size=args.batch_size)

    y = np.concatenate([np.ones_like(real_scores), np.zeros_like(fake_scores)])
    s = np.concatenate([real_scores, fake_scores])

    auc = roc_auc_score(y, s)
    ap = pr_auc_score(y, s)
    acc0 = accuracy_from_scores(y, s, thr=0.0)

    print("\n=== Overall D(real vs fake) ===")
    print(f"n_real={len(real_scores)} n_fake={len(fake_scores)}")
    print(f"ROC_AUC={auc:.4f}  PR_AUC={ap:.4f}  Acc@thr0={acc0:.4f}")
    print(f"score_mean real={real_scores.mean():.4f} fake={fake_scores.mean():.4f}")
    print(f"score_std  real={real_scores.std():.4f} fake={fake_scores.std():.4f}")

    # by dataset
    ds = df[args.dataset_col].astype(str).to_numpy()
    print("\n=== By dataset (same bg distribution within each subset) ===")
    for name in np.unique(ds):
        m = ds == name
        rs = real_scores[m]
        fs = fake_scores[m]
        yy = np.concatenate([np.ones_like(rs), np.zeros_like(fs)])
        ss = np.concatenate([rs, fs])
        auc_g = roc_auc_score(yy, ss)
        ap_g = pr_auc_score(yy, ss)
        acc_g = accuracy_from_scores(yy, ss, thr=0.0)
        print(f"[{name}] n={m.sum()} ROC_AUC={auc_g:.4f} PR_AUC={ap_g:.4f} Acc={acc_g:.4f} "
              f"mean(real)={rs.mean():.4f} mean(fake)={fs.mean():.4f}")

    # bg leakage check: correlation between score and bg norm
    bg_norm = np.linalg.norm(bg_np, axis=1)
    corr = np.corrcoef(bg_norm, real_scores)[0, 1] if np.std(bg_norm) > 0 else float("nan")
    print("\n=== Bg leakage quick check ===")
    print(f"corr(score_real, ||bg||) = {corr:.4f} (should be near 0 after normalization)")

    # save a small csv for inspection
    out = Path("disc_check_outputs.csv")
    pd.DataFrame({
        "dataset": ds,
        "score_real": real_scores,
        "score_fake": fake_scores,
    }).to_csv(out, index=False)
    print(f"\n[OK] wrote {out.resolve()}")


if __name__ == "__main__":
    main()
