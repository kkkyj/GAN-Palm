#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_gan_discriminator.py

Evaluate a conditional GAN discriminator in three ways (aligned with your training objective):

1) pos vs neg      (REAL only)       -> proves D has value as a classifier
2) pos vs fake     (REAL pos vs GEN) -> shows whether G is generating "positive-like" sequences
3) neg vs fake     (REAL neg vs GEN) -> shows whether fake collapses to neg-like sequences

Also reports the legacy metric:
4) real(all) vs fake   (REAL pos+neg vs GEN) -> often ~0.5 in your setup; keep for reference.

Key fixes for your data:
- BG alignment uses bg_key = dataset + "__" + sample (matches bg_npz["samples"] like "Hela__WT1").

Outputs:
- metrics.json
- roc/pr curves for pos-vs-neg and pos-vs-fake (plus real-vs-fake for reference)
- score_hist.png (pos/neg/fake score distributions)
- per_group.csv (optional stratify by dataset or sample; for pos-vs-neg AUC/PR)

Example:
python eval_gan_discriminator.py \
  --gan-script code/conditional_seq_gan_noesm.py \
  --ckpt-dir   runs/gan_noesm_human/checkpoints \
  --site-table data/site_sample_long.human.parquet \
  --bg-npz     data/proteome_bg_embeddings.dataset.npz \
  --seq-col seq101 --label-col label_bin --dataset-col dataset --sample-col sample \
  --n-real 50000 --batch-size 256 --device cuda \
  --temperature 1.0 --top-k 0 \
  --stratify dataset \
  --outdir results/disc_eval_noesm_human
"""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

import matplotlib.pyplot as plt


# ----------------------------- utils -----------------------------
def load_module_from_path(py_path: str):
    spec = importlib.util.spec_from_file_location("gan_script", py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import gan script: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pick_ckpt(ckpt_dir: Path, name: str, prefix: str) -> Path:
    """Pick checkpoint by explicit name, else try <prefix>_final.pt, else newest <prefix>_*.pt"""
    if name:
        p = ckpt_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p
    p = ckpt_dir / f"{prefix}_best.pt"
    if p.exists():
        return p
    cands = sorted(ckpt_dir.glob(f"{prefix}_*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No {prefix} checkpoint found under {ckpt_dir}")
    return cands[0]


def load_bg_db(bg_npz: str):
    z = np.load(bg_npz, allow_pickle=True)
    keys = set(z.files)

    # matrix
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

    if vecs.ndim != 2:
        raise ValueError(f"bg vectors must be 2D (N,D), got {vecs.shape}")

    # ids
    if "samples" in keys:
        ids = z["samples"].astype(str)
    elif "ids" in keys:
        ids = z["ids"].astype(str)
    elif "names" in keys:
        ids = z["names"].astype(str)
    else:
        raise KeyError(f"Cannot find id list in {bg_npz}, keys={z.files}")

    return vecs.astype(np.float32), ids


def tokenize_seq101(gan_mod, seqs: List[str]) -> torch.LongTensor:
    if not hasattr(gan_mod, "AA2IDX"):
        raise AttributeError("gan script missing AA2IDX for tokenization.")
    aa2idx = gan_mod.AA2IDX
    unk = aa2idx.get("X", max(aa2idx.values()))

    arr = np.full((len(seqs), 101), unk, dtype=np.int64)
    for i, s in enumerate(seqs):
        s = str(s)
        if len(s) != 101:
            raise ValueError(f"seq length != 101 at row {i}: {len(s)}")
        arr[i] = [aa2idx.get(ch, unk) for ch in s]
    return torch.from_numpy(arr)


@torch.no_grad()
def disc_scores(D, tokens: torch.LongTensor, bg: torch.FloatTensor,
                device: str, batch_size: int) -> np.ndarray:
    D.eval()
    out = []
    n = tokens.shape[0]
    for i in range(0, n, batch_size):
        t = tokens[i:i+batch_size].to(device)
        b = bg[i:i+batch_size].to(device)
        logits = D(t, b, is_soft=False).view(-1).float()
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        out.append(prob)
    return np.concatenate(out, axis=0)


@torch.no_grad()
def gen_fake_tokens(gan_mod, G, bg: torch.FloatTensor, device: str,
                    batch_size: int, temperature: float, top_k: int) -> torch.LongTensor:
    G.eval()
    toks = []
    n = bg.shape[0]
    start_token = gan_mod.AA2IDX.get("M", getattr(gan_mod, "PAD_IDX", 0))

    for i in range(0, n, batch_size):
        b = bg[i:i+batch_size].to(device)
        t = G.sample_discrete(
            b,
            start_token=start_token,
            temperature=float(temperature),
            top_k=int(top_k),
        )
        toks.append(t.detach().cpu().long())
    return torch.cat(toks, dim=0)


def auc_pr(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """AUC/PR for class1=a vs class0=b"""
    y = np.concatenate([np.ones_like(a), np.zeros_like(b)])
    s = np.concatenate([a, b])
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    p, r, t = precision_recall_curve(y_true, y_score)
    # t length = len(p)-1
    f1 = (2 * p[:-1] * r[:-1]) / np.clip(p[:-1] + r[:-1], 1e-12, None)
    j = int(np.nanargmax(f1))
    return float(t[j]), float(f1[j])


def save_roc_pr(y_true: np.ndarray, y_score: np.ndarray, out_png_prefix: Path, title_suffix: str):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC {title_suffix}")
    plt.tight_layout()
    plt.savefig(str(out_png_prefix) + "_roc.png", dpi=200)
    plt.close()

    p, r, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR {title_suffix}")
    plt.tight_layout()
    plt.savefig(str(out_png_prefix) + "_pr.png", dpi=200)
    plt.close()


def save_score_hist(pos: np.ndarray, neg: np.ndarray, fake: np.ndarray, out_png: Path):
    plt.figure()
    plt.hist(pos, bins=80, alpha=0.6, label="real_pos")
    plt.hist(neg, bins=80, alpha=0.6, label="real_neg")
    plt.hist(fake, bins=80, alpha=0.6, label="fake")
    plt.xlabel("D score (sigmoid)")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Discriminator score distribution")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gan-script", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--d-ckpt-name", default="", help="optional explicit D ckpt filename")
    ap.add_argument("--g-ckpt-name", default="", help="optional explicit G ckpt filename")

    ap.add_argument("--site-table", required=True)
    ap.add_argument("--bg-npz", required=True)

    ap.add_argument("--seq-col", default="seq101")
    ap.add_argument("--label-col", default="label_bin")
    ap.add_argument("--dataset-col", default="dataset")
    ap.add_argument("--sample-col", default="sample")

    ap.add_argument("--n-real", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)

    ap.add_argument("--stratify", default="none", choices=["none", "dataset", "sample"],
                    help="per-group metrics for pos-vs-neg (real only)")

    ap.add_argument("--auto-threshold", action="store_true",
                    help="compute best-F1 threshold for the legacy real-vs-fake split (for reference only)")

    ap.add_argument("--outdir", required=True)

    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gan_mod = load_module_from_path(args.gan_script)

    # Load data
    df = pd.read_parquet(args.site_table)
    need = [args.seq_col, args.label_col, args.dataset_col, args.sample_col]
    missing_cols = [c for c in need if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in site-table: {missing_cols}")
    df = df[need].dropna().reset_index(drop=True)

    # Subsample
    if args.n_real < len(df):
        df = df.sample(n=args.n_real, random_state=13).reset_index(drop=True)

    # Build bg_key and align bg vectors
    df["_bg_key"] = df[args.dataset_col].astype(str) + "__" + df[args.sample_col].astype(str)

    bg_vecs, bg_ids = load_bg_db(args.bg_npz)
    id2row = {k: i for i, k in enumerate(bg_ids.tolist())}
    bg_rows = []
    miss = 0
    for k in df["_bg_key"].tolist():
        if k in id2row:
            bg_rows.append(id2row[k])
        else:
            miss += 1
            bg_rows.append(np.random.randint(0, bg_vecs.shape[0]))
    print(f"[INFO] bg_key miss = {miss}/{len(df)}")
    bg = torch.from_numpy(bg_vecs[np.array(bg_rows)]).float()

    # Instantiate models
    if not hasattr(gan_mod, "Generator") or not hasattr(gan_mod, "ConditionalSequenceDisc"):
        raise AttributeError("gan script must expose Generator and ConditionalSequenceDisc")

    bg_dim = int(bg_vecs.shape[1])
    G = gan_mod.Generator(bg_dim=bg_dim)
    D = gan_mod.ConditionalSequenceDisc(bg_dim=bg_dim)

    device = torch.device(args.device)
    G.to(device)
    D.to(device)

    ckpt_dir = Path(args.ckpt_dir)
    d_ckpt = pick_ckpt(ckpt_dir, args.d_ckpt_name, prefix="D")
    g_ckpt = pick_ckpt(ckpt_dir, args.g_ckpt_name, prefix="G")

    d_state = torch.load(d_ckpt, map_location="cpu")
    if isinstance(d_state, dict) and "state_dict" in d_state:
        d_state = d_state["state_dict"]
    missing, unexpected = D.load_state_dict(d_state, strict=False)
    print(f"[INFO] load D: missing={len(missing)} unexpected={len(unexpected)} ckpt={d_ckpt.name}")

    g_state = torch.load(g_ckpt, map_location="cpu")
    if isinstance(g_state, dict) and "state_dict" in g_state:
        g_state = g_state["state_dict"]
    _m2, _u2 = G.load_state_dict(g_state, strict=False)
    print(f"[INFO] load G from {g_ckpt.name}")

    # Prepare tokens
    real_tokens = tokenize_seq101(gan_mod, df[args.seq_col].astype(str).tolist())
    fake_tokens = gen_fake_tokens(
        gan_mod, G, bg, args.device, args.batch_size, args.temperature, args.top_k
    )

    # Scores
    real_score = disc_scores(D, real_tokens, bg, args.device, args.batch_size)
    fake_score = disc_scores(D, fake_tokens, bg, args.device, args.batch_size)

    y_posneg = df[args.label_col].astype(int).values
    pos_score = real_score[y_posneg == 1]
    neg_score = real_score[y_posneg == 0]

    # Core metrics aligned with your objective
    metrics: Dict[str, float] = {}
    metrics["n_real"] = int(len(real_score))
    metrics["n_fake"] = int(len(fake_score))
    metrics["n_pos"] = int(len(pos_score))
    metrics["n_neg"] = int(len(neg_score))
    metrics["ckpt_D"] = str(d_ckpt)
    metrics["ckpt_G"] = str(g_ckpt)
    metrics["temperature"] = float(args.temperature)
    metrics["top_k"] = int(args.top_k)

    if len(pos_score) < 10 or len(neg_score) < 10:
        print("[WARN] Too few pos or neg samples for pos-vs-neg metrics.")
    else:
        roc, pr = auc_pr(pos_score, neg_score)
        metrics["pos_vs_neg_roc_auc"] = roc
        metrics["pos_vs_neg_pr_auc"] = pr
        print(f"[EVAL] pos_vs_neg  ROC={roc:.4f} PR={pr:.4f} (n_pos={len(pos_score)} n_neg={len(neg_score)})")

    if len(pos_score) >= 10:
        roc, pr = auc_pr(pos_score, fake_score)
        metrics["pos_vs_fake_roc_auc"] = roc
        metrics["pos_vs_fake_pr_auc"] = pr
        print(f"[EVAL] pos_vs_fake ROC={roc:.4f} PR={pr:.4f} (n_pos={len(pos_score)} n_fake={len(fake_score)})")

    if len(neg_score) >= 10:
        roc, pr = auc_pr(neg_score, fake_score)
        metrics["neg_vs_fake_roc_auc"] = roc
        metrics["neg_vs_fake_pr_auc"] = pr
        print(f"[EVAL] neg_vs_fake ROC={roc:.4f} PR={pr:.4f} (n_neg={len(neg_score)} n_fake={len(fake_score)})")

    # Means (very diagnostic)
    metrics["mean_D_pos"] = float(pos_score.mean()) if len(pos_score) else float("nan")
    metrics["mean_D_neg"] = float(neg_score.mean()) if len(neg_score) else float("nan")
    metrics["mean_D_fake"] = float(fake_score.mean())
    print(f"[SCORE] mean D(pos)={metrics['mean_D_pos']:.6g}  D(neg)={metrics['mean_D_neg']:.6g}  D(fake)={metrics['mean_D_fake']:.6g}")

    # Legacy metric (often ~0.5 in your setup; keep for reference)
    y_true = np.concatenate([np.ones_like(real_score), np.zeros_like(fake_score)], axis=0)
    y_score = np.concatenate([real_score, fake_score], axis=0)
    metrics["real_vs_fake_roc_auc"] = float(roc_auc_score(y_true, y_score))
    metrics["real_vs_fake_pr_auc"] = float(average_precision_score(y_true, y_score))
    if args.auto_threshold:
        thr, bestf1 = best_f1_threshold(y_true, y_score)
        metrics["real_vs_fake_best_f1_thr"] = float(thr)
        metrics["real_vs_fake_best_f1"] = float(bestf1)
        print(f"[EVAL] real_vs_fake ROC={metrics['real_vs_fake_roc_auc']:.4f} PR={metrics['real_vs_fake_pr_auc']:.4f} bestF1={bestf1:.4f} thr={thr:.6g}")
    else:
        print(f"[EVAL] real_vs_fake ROC={metrics['real_vs_fake_roc_auc']:.4f} PR={metrics['real_vs_fake_pr_auc']:.4f}")

    # Save plots
    # pos vs neg
    if len(pos_score) >= 10 and len(neg_score) >= 10:
        y = np.concatenate([np.ones_like(pos_score), np.zeros_like(neg_score)])
        s = np.concatenate([pos_score, neg_score])
        save_roc_pr(y, s, outdir / "pos_vs_neg", "(pos vs neg)")

    # pos vs fake
    if len(pos_score) >= 10:
        y = np.concatenate([np.ones_like(pos_score), np.zeros_like(fake_score)])
        s = np.concatenate([pos_score, fake_score])
        save_roc_pr(y, s, outdir / "pos_vs_fake", "(pos vs fake)")

    # real vs fake (reference)
    save_roc_pr(y_true, y_score, outdir / "real_vs_fake", "(real vs fake)")

    # score histogram
    save_score_hist(pos_score, neg_score, fake_score, outdir / "score_hist.png")

    # Per-group metrics for pos-vs-neg (real only)
    if args.stratify != "none":
        col = args.dataset_col if args.stratify == "dataset" else args.sample_col
        rows = []
        for g, sub in df.groupby(col):
            idx = sub.index.values
            rs = real_score[idx]
            yy = sub[args.label_col].astype(int).values
            ps = rs[yy == 1]
            ns = rs[yy == 0]
            if len(ps) < 30 or len(ns) < 30:
                continue
            roc, pr = auc_pr(ps, ns)
            rows.append({
                "group": str(g),
                "n_pos": int(len(ps)),
                "n_neg": int(len(ns)),
                "pos_vs_neg_roc_auc": roc,
                "pos_vs_neg_pr_auc": pr,
                "mean_pos": float(ps.mean()),
                "mean_neg": float(ns.mean()),
            })
        if rows:
            out = pd.DataFrame(rows).sort_values(["pos_vs_neg_roc_auc", "pos_vs_neg_pr_auc"], ascending=False)
            out.to_csv(outdir / "per_group.csv", index=False)
            print(f"[OK] wrote per_group.csv with {len(out)} groups")

    # Save raw scores (useful for downstream plotting)
    np.savez_compressed(
        outdir / "scores.npz",
        real_score=real_score,
        fake_score=fake_score,
        pos_score=pos_score,
        neg_score=neg_score,
        y_posneg=y_posneg,
        bg_key=df["_bg_key"].astype(str).values,
    )

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] wrote metrics.json / plots / scores.npz to {outdir}")


if __name__ == "__main__":
    main()