#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_generator_conditional_motif_from_bgdb.py

Approach C (Generator conditional motif) - construct the conditional bg_vec from proteome_bg_embeddings.dataset.npz

Inputs:
- --ckpt: trained Generator checkpoint (recommended to save either the G-only state_dict, or a dict containing state_dict)
- --bg-db: proteome_bg_embeddings.dataset.npz (contains samples, combined_emb)
- --bg-spec: one or more condition definitions, format:
    NAME:REGEX
  For example:
    WT_Hela:^Hela__WT\\d+$
    KO_Hela:^Hela__KO\\d+$
    WT_PANC1:^PANC-1__WT\\d+$
    KO_PANC1:^PANC-1__KO\\d+$

Outputs:
- tables/<NAME>.pwm.csv
- figs/delta_pwm_<A>_minus_<B>.png  (difference computed for every pair)

Note:
- By default this script generates using seq101 (length 101), and forces the center position to be C.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


SEQ_LEN = 101
CENTER = 50
VOCAB = list("ACDEFGHIKLMNPQRSTVWYBX")
AA2IDX = {a: i for i, a in enumerate(VOCAB)}
IDX2AA = {i: a for a, i in AA2IDX.items()}
PAD_IDX = AA2IDX.get("X", len(VOCAB) - 1)
AA20 = list("ACDEFGHIKLMNPQRSTVWY")
AA20_2I = {a: i for i, a in enumerate(AA20)}


# ===== The G architecture used for training must match (if your training script defines G differently, replace this with a matching version) =====
class Generator(nn.Module):
    def __init__(self, bg_dim: int, d_model=256, nhead=8, nlayers=4, dff=512, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(len(VOCAB), d_model, padding_idx=PAD_IDX)
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
        self.out = nn.Linear(d_model, len(VOCAB))

    def forward(self, tokens: torch.Tensor, bg_vec: torch.Tensor):
        x = self.emb(tokens) + self.pos[:tokens.size(1)].unsqueeze(0)
        gamma, beta = self.film(bg_vec).chunk(2, dim=-1)
        x = x * (1 + torch.tanh(gamma.unsqueeze(1))) + beta.unsqueeze(1)
        h = self.enc(x)
        return self.out(h)

    @torch.no_grad()
    def sample(self, bg_vec: torch.Tensor, n: int, temperature=1.0, top_k=0):
        device = bg_vec.device
        # any initialization token works; here we use X
        tokens = torch.full((n, SEQ_LEN), PAD_IDX, dtype=torch.long, device=device)

        logits = self.forward(tokens, bg_vec) / max(1e-6, temperature)
        if top_k > 0:
            v, ix = torch.topk(logits, k=top_k, dim=-1)
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(-1, ix, v)
            logits = mask

        probs = torch.softmax(logits, dim=-1)
        samp = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(n, SEQ_LEN)
        samp[:, CENTER] = AA2IDX["C"]
        return samp


def tokens_to_seq(tokens: np.ndarray) -> str:
    s = "".join(IDX2AA[int(t)] for t in tokens)
    s = list(s)
    s[CENTER] = "C"
    return "".join(s)


def build_pwm(seqs: List[str], win: int = 10) -> pd.DataFrame:
    mat = np.zeros((2 * win + 1, len(AA20)), dtype=np.float64)
    for s in seqs:
        for i, pos in enumerate(range(CENTER - win, CENTER + win + 1)):
            aa = s[pos]
            j = AA20_2I.get(aa)
            if j is not None:
                mat[i, j] += 1.0
    mat += 0.5
    mat /= np.maximum(mat.sum(axis=1, keepdims=True), 1e-12)
    df = pd.DataFrame(mat, columns=AA20)
    df["pos_rel"] = np.arange(-win, win + 1)
    return df


def plot_pwm_delta(pwm_a: pd.DataFrame, pwm_b: pd.DataFrame, out_png: Path, title: str):
    A = pwm_a[AA20].to_numpy()
    B = pwm_b[AA20].to_numpy()
    delta = A - B
    plt.figure(figsize=(10, 4))
    plt.imshow(delta.T, aspect="auto")
    plt.yticks(range(len(AA20)), AA20)
    plt.xticks(range(len(pwm_a)), pwm_a["pos_rel"].tolist())
    plt.axvline(len(pwm_a) // 2, linestyle="--")
    plt.colorbar(label="Δfreq")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def load_bg_db(bg_db_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(bg_db_path, allow_pickle=True)
    if "samples" not in z or "combined_emb" not in z:
        raise ValueError(f"{bg_db_path} must contain keys: samples, combined_emb")
    samples = z["samples"].astype(object)
    emb = z["combined_emb"].astype(np.float32)
    return samples, emb


def build_bg_vec_from_regex(samples: np.ndarray, emb: np.ndarray, pattern: str) -> np.ndarray:
    rgx = re.compile(pattern)
    idx = [i for i, s in enumerate(samples) if rgx.search(str(s))]
    if not idx:
        raise ValueError(f"No samples matched regex: {pattern}")
    v = emb[idx].mean(axis=0)
    # normalization is recommended; it improves FiLM stability
    v = v / (np.linalg.norm(v) + 1e-8)
    return v.astype(np.float32)


def load_ckpt_state(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        # allow the case where you saved the state_dict directly
        # typical case: torch.save(model.state_dict(), path)
        # in that case ckpt itself is the parameter dict
        return ckpt
    raise ValueError("Unsupported checkpoint format")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Generator checkpoint (G_final.pt)")
    ap.add_argument("--bg-db", required=True, help="proteome_bg_embeddings.dataset.npz")
    ap.add_argument("--bg-spec", required=True, nargs="+",
                    help='One or more: NAME:REGEX   e.g. WT_Hela:^Hela__WT\\d+$')
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--n-samples", type=int, default=10000)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--pwm-win", type=int, default=10)

    # must match training (if your G hyperparameters differed during training, change these or make the G definition match)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--nlayers", type=int, default=4)
    ap.add_argument("--dff", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)

    args = ap.parse_args()
    outdir = Path(args.outdir)
    (outdir / "figs").mkdir(parents=True, exist_ok=True)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples, emb = load_bg_db(Path(args.bg_db))

    # build condition bg vecs
    bg_vecs: Dict[str, torch.Tensor] = {}
    for spec in args.bg_spec:
        if ":" not in spec:
            raise ValueError(f"--bg-spec must be NAME:REGEX, got {spec}")
        name, pattern = spec.split(":", 1)
        v = build_bg_vec_from_regex(samples, emb, pattern)
        bg_vecs[name] = torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
        print(f"[INFO] bg {name}: matched={sum(re.search(pattern, str(s)) is not None for s in samples)} dim={v.shape[0]}")

    bg_dim = next(iter(bg_vecs.values())).shape[-1]

    # load G
    state = load_ckpt_state(Path(args.ckpt))
    G = Generator(
        bg_dim=bg_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        nlayers=args.nlayers,
        dff=args.dff,
        dropout=args.dropout,
    ).to(device)
    missing, unexpected = G.load_state_dict(state, strict=False)
    print(f"[INFO] load ckpt: missing={len(missing)} unexpected={len(unexpected)}")
    G.eval()

    pwms: Dict[str, pd.DataFrame] = {}
    for name, bg in bg_vecs.items():
        print(f"[INFO] sampling {name} n={args.n_samples} temp={args.temperature} topk={args.top_k}")
        toks = G.sample(bg.repeat(args.n_samples, 1), n=args.n_samples,
                        temperature=args.temperature, top_k=args.top_k)
        seqs = [tokens_to_seq(t.cpu().numpy()) for t in toks]
        pwm = build_pwm(seqs, win=args.pwm_win)
        pwm.to_csv(outdir / "tables" / f"{name}.pwm.csv", index=False)
        pwms[name] = pwm

    names = list(pwms.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            plot_pwm_delta(
                pwms[a], pwms[b],
                outdir / "figs" / f"delta_pwm_{a}_minus_{b}.png",
                title=f"{a} - {b} (Generator conditional motif)",
            )

    print(f"[OK] wrote: {outdir}")


if __name__ == "__main__":
    main()
