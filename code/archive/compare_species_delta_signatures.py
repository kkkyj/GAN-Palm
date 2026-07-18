#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_vec(s: str) -> np.ndarray:
    x = np.array([float(t) for t in str(s).strip().split()], dtype=np.float64)
    return x

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))

def corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def load_sigs(path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        ct = str(r["cell_type"])
        d = parse_vec(r["delta_sig"])
        j = parse_vec(r["js_sig"])
        out[ct] = (d, j)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human-sigs", type=str, required=True, help=".../tables/*delta_signatures_w15.csv")
    ap.add_argument("--mouse-sigs", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hs = load_sigs(Path(args.human_sigs))
    ms = load_sigs(Path(args.mouse_sigs))

    rows = []
    for hct, (hd, hj) in hs.items():
        for mct, (md, mj) in ms.items():
            rows.append({
                "human_celltype": hct,
                "mouse_celltype": mct,
                "cos_delta": cosine(hd, md),
                "corr_delta": corr(hd, md),
                "cos_js": cosine(hj, mj),
                "corr_js": corr(hj, mj),
            })
    res = pd.DataFrame(rows)
    res.to_csv(outdir / "human_vs_mouse_consistency.csv", index=False)

    pivot = res.pivot(index="human_celltype", columns="mouse_celltype", values="cos_delta")
    plt.figure(figsize=(max(6, 1 + 0.7 * pivot.shape[1]), max(4, 1 + 0.6 * pivot.shape[0])))
    plt.imshow(pivot.values, aspect="auto")
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    plt.title("WT→KO Δfreq signature cosine similarity (human vs mouse)")
    plt.colorbar(label="cosine")
    plt.tight_layout()
    plt.savefig(outdir / "cosine_delta_heatmap.png", dpi=200)
    plt.close()

    top = res.sort_values("cos_delta", ascending=False).head(10)
    top.to_csv(outdir / "top10_pairs_by_cosine_delta.csv", index=False)

    if len(top) > 0:
        hct = top.iloc[0]["human_celltype"]
        mct = top.iloc[0]["mouse_celltype"]
        hd, _ = hs[hct]
        md, _ = ms[mct]
        plt.figure()
        plt.scatter(hd, md, s=6)
        plt.title(f"Top pair Δ signature scatter: {hct} vs {mct}\n"
                  f"cos={cosine(hd, md):.3f}, corr={corr(hd, md):.3f}")
        plt.xlabel("human Δfreq signature")
        plt.ylabel("mouse Δfreq signature")
        plt.tight_layout()
        plt.savefig(outdir / f"top1_delta_scatter_{hct}_vs_{mct}.png", dpi=200)
        plt.close()

    print(f"[OK] wrote: {outdir}")

if __name__ == "__main__":
    main()
