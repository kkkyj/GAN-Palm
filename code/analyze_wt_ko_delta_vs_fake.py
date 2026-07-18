#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WT vs KO (same fake background) delta analysis for palmitoylation site sequences.

Outputs:
  - delta_lo_heatmap_wt_minus_ko.tsv   (rows: -flank..+flank, cols: 20 AA)
  - delta_lo3_cross_center.tsv         (cross-center 3-mer ΔLO3)
  - summary.txt

Key idea (background-corrected):
  LO_WT(i,a) = log2( p_WT(i,a) / p_fake(i,a) )
  LO_KO(i,a) = log2( p_KO(i,a) / p_fake(i,a) )
  ΔLO(i,a)   = LO_WT - LO_KO

This script matches your current site table schema:
  dataset: "Hela" / "PANC-1"
  group:   "WT" / "KO"
  seq101:  101-aa sequence centered at Cys (index 50)

Example:
  python analyze_wt_ko_delta_vs_fake.py \
    --site-table data/site_sample_long.human.parquet \
    --seq-col seq101 --label-col label_bin \
    --dataset-col dataset --group-col group \
    --dataset Hela --wt-group WT --ko-group KO \
    --fake-fasta-glob "runs/gan_noesm_human/samples/sample_step*.fasta" \
    --outdir results/wt_ko_delta_vs_fake/Hela \
    --max-pos 50000 --max-fake 50000 --flank 10 --seed 13
"""

import argparse
from pathlib import Path
from collections import Counter
import glob
import numpy as np
import pandas as pd

AA = list("ACDEFGHIKLMNPQRSTVWY")
AA2I = {a: i for i, a in enumerate(AA)}


# ------------------------ IO ------------------------
def read_site_table(path: str) -> pd.DataFrame:
    """Always use pyarrow engine (you already verified parquet is OK)."""
    return pd.read_parquet(path, engine="pyarrow")


def load_fasta_seqs(fasta_path: str) -> list[str]:
    seqs: list[str] = []
    cur: list[str] = []
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
            else:
                cur.append(line)
        if cur:
            seqs.append("".join(cur))
    return seqs


def load_fake_from_fastas(glob_pattern: str, max_fake: int) -> list[str]:
    """Load fake sequences from multiple fasta files until reaching max_fake."""
    seqs: list[str] = []
    fps = sorted(glob.glob(glob_pattern))
    if not fps:
        raise FileNotFoundError(f"No fasta matched glob: {glob_pattern}")
    for fp in fps:
        seqs.extend(load_fasta_seqs(fp))
        if len(seqs) >= max_fake:
            break
    return seqs[:max_fake]


def subsample(seqs: list[str], n: int, seed: int) -> list[str]:
    if len(seqs) <= n:
        return seqs
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(seqs), size=n, replace=False)
    return [seqs[i] for i in idx]


# ------------------------ Core stats ------------------------
def seqs_to_counts(seqs: list[str], pos_idx: int) -> np.ndarray:
    """Count amino acids at a single position across sequences."""
    c = np.zeros(len(AA), dtype=np.int64)
    for s in seqs:
        if len(s) <= pos_idx:
            continue
        a = s[pos_idx]
        j = AA2I.get(a)
        if j is not None:
            c[j] += 1
    return c


def lo_vs_bg(count_fg: np.ndarray, count_bg: np.ndarray, eps: float) -> np.ndarray:
    """log2 odds: log2(p_fg / p_bg) with symmetric pseudocount eps."""
    p_fg = (count_fg + eps) / (count_fg.sum() + eps * len(AA))
    p_bg = (count_bg + eps) / (count_bg.sum() + eps * len(AA))
    return np.log2(p_fg / p_bg)


def compute_delta_lo(
    seqs_wt: list[str],
    seqs_ko: list[str],
    seqs_fake: list[str],
    center: int = 50,
    flank: int = 10,
    eps: float = 0.5,
) -> pd.DataFrame:
    """ΔLO heatmap table for positions center-flank .. center+flank."""
    positions = list(range(center - flank, center + flank + 1))
    mat = np.zeros((len(positions), len(AA)), dtype=float)

    for r, i in enumerate(positions):
        c_wt = seqs_to_counts(seqs_wt, i)
        c_ko = seqs_to_counts(seqs_ko, i)
        c_bg = seqs_to_counts(seqs_fake, i)

        lo_wt = lo_vs_bg(c_wt, c_bg, eps=eps)
        lo_ko = lo_vs_bg(c_ko, c_bg, eps=eps)
        mat[r] = lo_wt - lo_ko

    df = pd.DataFrame(mat, index=[p - center for p in positions], columns=AA)
    df.index.name = "pos_rel_to_Cys"
    return df


def extract_cross3(seqs: list[str], center: int = 50) -> list[str]:
    """Extract cross-center 3-mer (center-1:center+2)."""
    out: list[str] = []
    for s in seqs:
        if len(s) >= center + 2:
            out.append(s[center - 1 : center + 2])
    return out


def delta_lo3_cross_center(
    seqs_wt: list[str],
    seqs_ko: list[str],
    seqs_fake: list[str],
    center: int = 50,
    eps: float = 0.5,
) -> pd.DataFrame:
    """
    ΔLO3 for cross-center 3-mers:
      ΔLO3 = log2(pWT/pfake) - log2(pKO/pfake)
    """
    wt = extract_cross3(seqs_wt, center=center)
    ko = extract_cross3(seqs_ko, center=center)
    bg = extract_cross3(seqs_fake, center=center)

    c_wt, c_ko, c_bg = Counter(wt), Counter(ko), Counter(bg)
    all_k = sorted(set(c_wt) | set(c_ko) | set(c_bg))
    if not all_k:
        raise RuntimeError("No 3-mers extracted. Check sequences length / center index.")

    def lo(c_fg: Counter, c_bg_: Counter) -> dict[str, float]:
        fg_total = sum(c_fg.values())
        bg_total = sum(c_bg_.values())
        denom_fg = fg_total + eps * len(all_k)
        denom_bg = bg_total + eps * len(all_k)
        out_: dict[str, float] = {}
        for k in all_k:
            p_fg = (c_fg.get(k, 0) + eps) / denom_fg
            p_bg = (c_bg_.get(k, 0) + eps) / denom_bg
            out_[k] = float(np.log2(p_fg / p_bg))
        return out_

    lo_wt = lo(c_wt, c_bg)
    lo_ko = lo(c_ko, c_bg)
    delta = {k: lo_wt[k] - lo_ko[k] for k in all_k}

    df = pd.DataFrame(
        {
            "kmer": all_k,
            "delta_lo3": [delta[k] for k in all_k],
            "wt_n": [c_wt.get(k, 0) for k in all_k],
            "ko_n": [c_ko.get(k, 0) for k in all_k],
            "fake_n": [c_bg.get(k, 0) for k in all_k],
        }
    ).sort_values("delta_lo3", ascending=False)
    return df


# ------------------------ Main ------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-table", required=True)
    ap.add_argument("--seq-col", default="seq101")
    ap.add_argument("--label-col", default="label_bin")
    ap.add_argument("--dataset-col", default="dataset")
    ap.add_argument("--group-col", default="group")

    ap.add_argument("--dataset", required=True, help='e.g. "Hela" or "PANC-1"')
    ap.add_argument("--wt-group", default="WT")
    ap.add_argument("--ko-group", default="KO")

    ap.add_argument("--fake-fasta-glob", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--max-pos", type=int, default=50000)
    ap.add_argument("--max-fake", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=13)

    ap.add_argument("--center", type=int, default=50)
    ap.add_argument("--flank", type=int, default=10)
    ap.add_argument("--eps", type=float, default=0.5)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load site table
    df = read_site_table(args.site_table)

    # Filter positives & target dataset
    df_pos = df[df[args.label_col] == 1]
    df_pos = df_pos[df_pos[args.dataset_col].astype(str) == str(args.dataset)]

    # Split WT / KO by group_col
    grp = df_pos[args.group_col].astype(str)
    df_wt = df_pos[grp.str.upper() == args.wt_group.upper()]
    df_ko = df_pos[grp.str.upper() == args.ko_group.upper()]

    seqs_wt = df_wt[args.seq_col].astype(str).tolist()
    seqs_ko = df_ko[args.seq_col].astype(str).tolist()

    if len(seqs_wt) == 0 or len(seqs_ko) == 0:
        # Provide debug info
        debug_path = outdir / "debug_group_counts.tsv"
        (df_pos[args.group_col].astype(str).value_counts()
         .rename_axis("group")
         .reset_index(name="n")
         .to_csv(debug_path, sep="\t", index=False))
        raise RuntimeError(
            f"WT/KO positives are empty for dataset={args.dataset}. "
            f"Wrote group counts to {debug_path}."
        )

    # Balance WT/KO and cap by max_pos
    n = min(len(seqs_wt), len(seqs_ko), args.max_pos)
    seqs_wt = subsample(seqs_wt, n, args.seed)
    seqs_ko = subsample(seqs_ko, n, args.seed + 1)

    # Load fake sequences
    seqs_fake = load_fake_from_fastas(args.fake_fasta_glob, args.max_fake)
    if len(seqs_fake) == 0:
        raise RuntimeError("No fake sequences loaded from --fake-fasta-glob")
    # Cap / subsample fake (stable)
    seqs_fake = subsample(seqs_fake, min(len(seqs_fake), args.max_fake), args.seed + 2)

    # Compute ΔLO heatmap table
    delta_lo = compute_delta_lo(
        seqs_wt=seqs_wt,
        seqs_ko=seqs_ko,
        seqs_fake=seqs_fake,
        center=args.center,
        flank=args.flank,
        eps=args.eps,
    )
    delta_lo.to_csv(outdir / "delta_lo_heatmap_wt_minus_ko.tsv", sep="\t", float_format="%.6g")

    # Compute ΔLO3 cross-center
    delta_lo3 = delta_lo3_cross_center(
        seqs_wt=seqs_wt,
        seqs_ko=seqs_ko,
        seqs_fake=seqs_fake,
        center=args.center,
        eps=args.eps,
    )
    delta_lo3.to_csv(outdir / "delta_lo3_cross_center.tsv", sep="\t", index=False, float_format="%.6g")

    # Summary
    with open(outdir / "summary.txt", "w") as f:
        f.write(f"dataset={args.dataset}\n")
        f.write(f"WT_pos_n_total={len(df_wt)}\n")
        f.write(f"KO_pos_n_total={len(df_ko)}\n")
        f.write(f"WT_pos_n_used={len(seqs_wt)}\n")
        f.write(f"KO_pos_n_used={len(seqs_ko)}\n")
        f.write(f"Fake_n_used={len(seqs_fake)}\n")
        f.write(f"center={args.center}\n")
        f.write(f"flank={args.flank}\n")
        f.write(f"eps={args.eps}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"fake_glob={args.fake_fasta_glob}\n")

    print(f"[OK] wrote outputs to: {outdir}")


if __name__ == "__main__":
    main()