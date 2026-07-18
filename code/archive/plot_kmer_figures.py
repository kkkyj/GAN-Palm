#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_kmer_figures.py

Generate a set of standard result figures from kmer_enrich_*.tsv and the raw fasta sequences:
- Top-N enriched k-mers
- Enrichment distribution
- Abundance vs enrichment
- k-mers restricted to the Cys neighborhood
- k-mers covering the central position
- pos vs fake / pos vs neg comparison

Only depends on: pandas, matplotlib
"""

import argparse
from pathlib import Path
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
AA_SET = set("ACDEFGHIKLMNPQRSTVWY")


# ------------------------
# k-mer utilities
# ------------------------
def kmer_counts_window(seqs, k, start, end):
    """Count k-mers within the [start,end) interval of the sequences"""
    c = Counter()
    for s in seqs:
        for i in range(start, end - k + 1):
            km = s[i:i+k]
            if all(a in AA_SET for a in km):
                c[km] += 1
    return c


def kmer_counts_cross_center(seqs, k, center):
    """Only count k-mers that cross the central position"""
    c = Counter()
    for s in seqs:
        for i in range(center - k + 1, center + 1):
            if i < 0 or i + k > len(s):
                continue
            km = s[i:i+k]
            if all(a in AA_SET for a in km):
                c[km] += 1
    return c


def enrich_table(cp, cb, pseudocount=1.0):
    keys = set(cp) | set(cb)
    tp = sum(cp.values())
    tb = sum(cb.values())
    rows = []
    for km in keys:
        p = (cp.get(km, 0) + pseudocount) / (tp + pseudocount * len(keys))
        b = (cb.get(km, 0) + pseudocount) / (tb + pseudocount * len(keys))
        rows.append((km, cp.get(km, 0), cb.get(km, 0), p, b, (p / b)))
    df = pd.DataFrame(rows, columns=["kmer", "count_pos", "count_bg", "freq_pos", "freq_bg", "fold"])
    df["log2_enrich"] = df["fold"].apply(lambda x: np.log2(x))
    return df.sort_values("log2_enrich", ascending=False)


# ------------------------
# plotting helpers
# ------------------------
def plot_top(df, out, title, topn=20):
    d = df.head(topn)
    plt.figure()
    plt.barh(d["kmer"], d["log2_enrich"])
    plt.gca().invert_yaxis()
    plt.xlabel("log2 enrichment")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_hist(df, out, title):
    plt.figure()
    plt.hist(df["log2_enrich"], bins=20)
    plt.xlabel("log2 enrichment")
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_scatter(df, out, title):
    plt.figure()
    plt.scatter(df["count_pos"], df["log2_enrich"])
    plt.xlabel("count_pos")
    plt.ylabel("log2 enrichment")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


# ------------------------
# main
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmer-pos-fake", required=True)
    ap.add_argument("--kmer-pos-neg", required=True)
    ap.add_argument("--pos-fasta", required=True)
    ap.add_argument("--fake-fasta", required=True)
    ap.add_argument("--neg-fasta", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--center", type=int, default=50)
    ap.add_argument("--cys-left", type=int, default=-10)
    ap.add_argument("--cys-right", type=int, default=10)
    ap.add_argument("--topn", type=int, default=20)

    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # ---------- load global k-mer tables ----------
    df_pf = pd.read_csv(args.kmer_pos_fake, sep="\t")
    df_pn = pd.read_csv(args.kmer_pos_neg, sep="\t")

    df_pf3 = df_pf[df_pf["k"] == args.k]
    df_pn3 = df_pn[df_pn["k"] == args.k]

    # ---------- global plots ----------
    plot_top(df_pf3, out / "fig_top3mer_pos_vs_fake.png",
             "Top enriched 3-mers (pos vs fake)", args.topn)
    plot_hist(df_pf3.head(args.topn), out / "fig_hist_pos_vs_fake.png",
              "Distribution of log2 enrichment (pos vs fake)")
    plot_scatter(df_pf3.head(args.topn), out / "fig_scatter_pos_vs_fake.png",
                 "Abundance vs enrichment (pos vs fake)")

    plot_top(df_pn3, out / "fig_top3mer_pos_vs_neg.png",
             "Top enriched 3-mers (pos vs neg)", args.topn)

    # ---------- load fasta ----------
    def load_fa(p):
        seqs = []
        with open(p) as f:
            for line in f:
                if not line.startswith(">"):
                    seqs.append(line.strip())
        return seqs

    pos = load_fa(args.pos_fasta)
    fake = load_fa(args.fake_fasta)
    neg = load_fa(args.neg_fasta)

    # ---------- Cys window k-mers ----------
    s = args.center + args.cys_left
    e = args.center + args.cys_right + 1

    cp = kmer_counts_window(pos, args.k, s, e)
    cf = kmer_counts_window(fake, args.k, s, e)
    cn = kmer_counts_window(neg, args.k, s, e)

    df_cys_pf = enrich_table(cp, cf)
    df_cys_pn = enrich_table(cp, cn)

    plot_top(df_cys_pf, out / "fig_cys_window_pos_vs_fake.png",
             "Cys window enriched 3-mers (pos vs fake)", args.topn)
    plot_top(df_cys_pn, out / "fig_cys_window_pos_vs_neg.png",
             "Cys window enriched 3-mers (pos vs neg)", args.topn)

    # ---------- crossing-center k-mers ----------
    cp2 = kmer_counts_cross_center(pos, args.k, args.center)
    cf2 = kmer_counts_cross_center(fake, args.k, args.center)

    df_cross = enrich_table(cp2, cf2)
    plot_top(df_cross, out / "fig_cross_center_pos_vs_fake.png",
             "Center-crossing 3-mers (pos vs fake)", args.topn)

    print(f"[OK] All figures saved to: {out}")


if __name__ == "__main__":
    main()