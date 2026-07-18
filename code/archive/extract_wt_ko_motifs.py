#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_wt_ko_motifs.py

Goal: Extract palmitoylation sequence motifs that are "enriched/specific in WT relative to KO" from site-level seq101 data (anchored at the central C).
Supports separate analysis by cell system (e.g., Hela / PANC-1).

Outputs (outdir):
- tables/
  - <tag>.<cell>.kmer_enrich.k3_w8.csv
  - <tag>.<cell>.kmer_enrich.k5_w10.csv
  - <tag>.<cell>.pwm_wt.csv / pwm_ko.csv / pwm_delta_wt_minus_ko.csv
  - <tag>.<cell>.top_positions_by_l1.csv
- figs/
  - <tag>.<cell>.pwm_delta_heatmap.png
  - <tag>.<cell>.position_l1_curve.png

Statistics:
- k-mer enrichment: build a 2x2 table from counts within the window, apply Fisher exact (two-sided) + Benjamini-Hochberg FDR
- effect size: log2((WT+pc)/(KO+pc))

Usage example (your human parquet):
python extract_wt_ko_motifs.py \
  --site-table data/site_sample_long.human.parquet \
  --outdir results/motifs_wt_vs_ko_human \
  --seq-col seq101 --geno-col group --cell-col dataset \
  --cells Hela,PANC-1 \
  --k-list 3,5 \
  --win-map 3:8,5:10 \
  --min-count 200

Notes:
- This script assumes the central position of each seq101 is the target C (pos 50).
- If you want to extract motifs using only "positive samples (true palmitoylation sites)", add --only-positive and specify --label-col label_bin
"""

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SEQ_LEN = 101
CENTER = 50
AA20 = list("ACDEFGHIKLMNPQRSTVWY")
AA2I = {a: i for i, a in enumerate(AA20)}


# ----------------- utils -----------------

def mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_any_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported file type: {path}")

def normalize_seq(seq: str) -> Optional[str]:
    if not isinstance(seq, str):
        return None
    s = seq.strip().upper()
    if len(s) != SEQ_LEN:
        return None
    return s

def infer_wtko(x: str) -> Optional[str]:
    if not isinstance(x, str):
        return None
    s = x.strip().upper()
    if s == "WT" or re.search(r"\bWT\b", s):
        return "WT"
    if s == "KO" or re.search(r"\bKO\b", s):
        return "KO"
    return None

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR control."""
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty_like(ranked)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[i] = prev
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out

def log2fc(a: float, b: float, pc: float = 1.0) -> float:
    return float(np.log2((a + pc) / (b + pc)))

# --- Fisher exact (two-sided) using hypergeometric pmf; optimized enough for k-mer counts ---
def _log_choose(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

def hypergeom_pmf(x: int, K: int, N: int, n: int) -> float:
    # P(X=x) where X ~ Hypergeom(N, K, n)
    # N population, K success states, n draws
    if x < 0 or x > K or x > n or n > N:
        return 0.0
    return math.exp(_log_choose(K, x) + _log_choose(N - K, n - x) - _log_choose(N, n))

def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """
    2x2 table:
              in_kmer   not_in_kmer
      WT         a          b
      KO         c          d
    Two-sided Fisher exact p-value (sum of probabilities <= observed).
    """
    # margins
    n1 = a + b  # WT total
    n2 = c + d  # KO total
    K = a + c   # in_kmer total
    N = n1 + n2
    n = n1

    # feasible x range
    lo = max(0, n - (N - K))
    hi = min(n, K)

    p_obs = hypergeom_pmf(a, K, N, n)
    p = 0.0
    for x in range(lo, hi + 1):
        px = hypergeom_pmf(x, K, N, n)
        if px <= p_obs + 1e-15:
            p += px
    return float(min(1.0, p))


# ----------------- motif core -----------------

def build_pwm(seqs: List[str], win: int, pseudocount: float = 0.5) -> np.ndarray:
    """
    PWM around center C: positions [-win, +win] inclusive, shape [2*win+1, 20]
    counts only AA20
    """
    L = 2 * win + 1
    counts = np.zeros((L, 20), dtype=np.float64)
    for s in seqs:
        if s is None:
            continue
        for i, pos in enumerate(range(CENTER - win, CENTER + win + 1)):
            aa = s[pos]
            j = AA2I.get(aa)
            if j is not None:
                counts[i, j] += 1.0
    counts += pseudocount
    counts /= np.maximum(counts.sum(axis=1, keepdims=True), 1e-12)
    return counts

def position_l1(delta_pwm: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(delta_pwm), axis=1)

def extract_kmer_counts(
    seqs: List[str],
    k: int,
    win: int,
    center_anchor: bool = True,
    exclude_center_c: bool = False,
) -> Dict[str, int]:
    """
    Count all k-mers within window [-win, +win] around center.
    If center_anchor=True: slide across positions within that window (standard).
    exclude_center_c=True: skip any k-mer that covers CENTER (optional).
    """
    counts: Dict[str, int] = {}
    start = CENTER - win
    end = CENTER + win  # inclusive
    for s in seqs:
        if s is None:
            continue
        for i in range(start, end - k + 2):
            j = i + k - 1
            if exclude_center_c and (i <= CENTER <= j):
                continue
            km = s[i:i + k]
            if len(km) != k:
                continue
            counts[km] = counts.get(km, 0) + 1
    return counts

def kmer_enrichment_table(
    wt_seqs: List[str],
    ko_seqs: List[str],
    k: int,
    win: int,
    min_count: int,
    pc: float = 1.0,
    exclude_center_c: bool = False,
) -> pd.DataFrame:
    """
    Return df with kmer, wt_count, ko_count, log2fc(WT/KO), fisher_p, fdr
    """
    wt_counts = extract_kmer_counts(wt_seqs, k=k, win=win, exclude_center_c=exclude_center_c)
    ko_counts = extract_kmer_counts(ko_seqs, k=k, win=win, exclude_center_c=exclude_center_c)

    wt_total = int(sum(wt_counts.values()))
    ko_total = int(sum(ko_counts.values()))

    all_km = set(wt_counts) | set(ko_counts)
    rows = []
    pvals = []

    for km in all_km:
        a = int(wt_counts.get(km, 0))
        c = int(ko_counts.get(km, 0))
        if (a + c) < min_count:
            continue
        b = wt_total - a
        d = ko_total - c
        p = fisher_exact_two_sided(a, b, c, d)
        pvals.append(p)
        rows.append([km, a, c, log2fc(a, c, pc=pc), p])

    if not rows:
        return pd.DataFrame(columns=["kmer", "wt_count", "ko_count", "log2fc_wt_vs_ko", "fisher_p", "fdr_bh"])

    df = pd.DataFrame(rows, columns=["kmer", "wt_count", "ko_count", "log2fc_wt_vs_ko", "fisher_p"])
    df["fdr_bh"] = bh_fdr(df["fisher_p"].to_numpy())
    # sort: most WT-enriched first (log2fc desc), then by fdr
    df = df.sort_values(["log2fc_wt_vs_ko", "fdr_bh"], ascending=[False, True]).reset_index(drop=True)
    return df

def plot_pwm_delta(delta: np.ndarray, win: int, title: str, out_png: Path):
    # delta: [L, 20], L=2*win+1
    plt.figure(figsize=(10, 4))
    plt.imshow(delta.T, aspect="auto")
    plt.title(title)
    plt.yticks(np.arange(len(AA20)), AA20)
    xt = np.arange(0, 2 * win + 1, 3)
    plt.xticks(xt, [str(int(x - win)) for x in xt])
    plt.axvline(win, linestyle="--")
    plt.xlabel("pos_rel (center C at 0)")
    plt.colorbar(label="Δfreq (WT - KO)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_l1_curve(l1: np.ndarray, win: int, title: str, out_png: Path):
    pos = np.arange(-win, win + 1)
    plt.figure()
    plt.plot(pos, l1)
    plt.axvline(0, linestyle="--")
    plt.title(title)
    plt.xlabel("pos_rel")
    plt.ylabel("L1(|Δfreq|) across AA20")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ----------------- main pipeline -----------------

def run_for_cell(
    df: pd.DataFrame,
    seq_col: str,
    geno_col: str,
    cell_col: str,
    label_col: Optional[str],
    only_positive: bool,
    cell: str,
    tag: str,
    outdir: Path,
    k_list: List[int],
    win_map: Dict[int, int],
    pwm_win: int,
    min_count: int,
    pc: float,
    exclude_center_c: bool,
):
    sub = df[df[cell_col].astype(str) == cell].copy()

    # optional positive-only
    if only_positive:
        if label_col is None or label_col not in sub.columns:
            raise ValueError("--only-positive requires --label-col present in table.")
        sub = sub[sub[label_col].astype(int) == 1].copy()

    # normalize genotype + seq
    sub["_geno"] = sub[geno_col].astype(str).apply(infer_wtko)
    sub["_seq"] = sub[seq_col].apply(normalize_seq)
    sub = sub[~sub["_geno"].isna() & ~sub["_seq"].isna()].copy()

    wt = sub[sub["_geno"] == "WT"]["_seq"].tolist()
    ko = sub[sub["_geno"] == "KO"]["_seq"].tolist()

    if len(wt) == 0 or len(ko) == 0:
        print(f"[WARN] skip cell={cell}: n_wt={len(wt)}, n_ko={len(ko)}")
        return

    tables_dir = outdir / "tables"
    figs_dir = outdir / "figs"
    mkdir(tables_dir)
    mkdir(figs_dir)

    # PWM (WT/KO + delta)
    pwm_wt = build_pwm(wt, win=pwm_win)
    pwm_ko = build_pwm(ko, win=pwm_win)
    delta = pwm_wt - pwm_ko
    l1 = position_l1(delta)

    pos = np.arange(-pwm_win, pwm_win + 1)
    top_idx = np.argsort(-l1)[:15]
    top_positions = pd.DataFrame({
        "pos_rel": pos[top_idx],
        "l1_abs_delta_sum": l1[top_idx],
    }).sort_values("l1_abs_delta_sum", ascending=False)
    top_positions.to_csv(tables_dir / f"{tag}.{cell}.top_positions_by_l1.csv", index=False)

    pd.DataFrame(pwm_wt, columns=[f"P_{a}" for a in AA20]).assign(pos_rel=pos).to_csv(
        tables_dir / f"{tag}.{cell}.pwm_wt.csv", index=False
    )
    pd.DataFrame(pwm_ko, columns=[f"P_{a}" for a in AA20]).assign(pos_rel=pos).to_csv(
        tables_dir / f"{tag}.{cell}.pwm_ko.csv", index=False
    )
    pd.DataFrame(delta, columns=[f"Δ_{a}" for a in AA20]).assign(pos_rel=pos).to_csv(
        tables_dir / f"{tag}.{cell}.pwm_delta_wt_minus_ko.csv", index=False
    )

    plot_pwm_delta(delta, win=pwm_win, title=f"{tag} | {cell} | PWM Δ (WT - KO) window [-{pwm_win},{pwm_win}]",
                   out_png=figs_dir / f"{tag}.{cell}.pwm_delta_heatmap.png")
    plot_l1_curve(l1, win=pwm_win, title=f"{tag} | {cell} | position L1(|Δ|) across AA",
                  out_png=figs_dir / f"{tag}.{cell}.position_l1_curve.png")

    # k-mer enrichment
    for k in k_list:
        win = win_map.get(k, None)
        if win is None:
            raise ValueError(f"win_map missing for k={k}. Use --win-map like 3:8,5:10")
        df_km = kmer_enrichment_table(
            wt_seqs=wt,
            ko_seqs=ko,
            k=k,
            win=win,
            min_count=min_count,
            pc=pc,
            exclude_center_c=exclude_center_c,
        )
        df_km.to_csv(tables_dir / f"{tag}.{cell}.kmer_enrich.k{k}_w{win}.csv", index=False)

    # small text report
    rep = outdir / "report"
    mkdir(rep)
    with open(rep / f"{tag}.{cell}.motif_summary.md", "w", encoding="utf-8") as f:
        f.write(f"# {tag} | {cell} | WT vs KO motif summary\n\n")
        f.write(f"- n_WT={len(wt)}\n- n_KO={len(ko)}\n")
        if only_positive:
            f.write("- using only label==1 (positive sites)\n")
        f.write("\n## PWM delta (WT - KO)\n")
        f.write(f"- window: [-{pwm_win}, {pwm_win}]\n")
        f.write("- top positions by L1(|Δ|) saved in tables/*.top_positions_by_l1.csv\n\n")
        f.write("## k-mer enrichment\n")
        for k in k_list:
            win = win_map[k]
            f.write(f"- k={k}, window=[-{win},{win}] -> tables/*kmer_enrich.k{k}_w{win}.csv\n")


def parse_win_map(s: str) -> Dict[int, int]:
    """
    parse like "3:8,5:10" => {3:8,5:10}
    """
    out = {}
    parts = [x.strip() for x in s.split(",") if x.strip()]
    for p in parts:
        k_str, w_str = p.split(":")
        out[int(k_str)] = int(w_str)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-table", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--tag", type=str, default="real")

    ap.add_argument("--seq-col", type=str, default="seq101")
    ap.add_argument("--geno-col", type=str, default="group")
    ap.add_argument("--cell-col", type=str, default="dataset")

    ap.add_argument("--cells", type=str, default=None, help="comma-separated; default: top unique values in cell-col")
    ap.add_argument("--label-col", type=str, default=None)
    ap.add_argument("--only-positive", action="store_true", help="use only label_col==1 (positive sites)")

    ap.add_argument("--k-list", type=str, default="3,5", help="comma-separated k values")
    ap.add_argument("--win-map", type=str, default="3:8,5:10", help="mapping k:win, e.g., 3:8,5:10")

    ap.add_argument("--pwm-win", type=int, default=15, help="PWM window half size (±pwm_win)")
    ap.add_argument("--min-count", type=int, default=200, help="min total (WT+KO) count for k-mer to test")
    ap.add_argument("--pc", type=float, default=1.0, help="pseudocount for log2FC")
    ap.add_argument("--exclude-center-c", action="store_true", help="exclude k-mers spanning the center position")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    mkdir(outdir)

    df = read_any_table(Path(args.site_table))

    # basic sanity
    for col in [args.seq_col, args.geno_col, args.cell_col]:
        if col not in df.columns:
            raise ValueError(f"Column not found: {col}. Available: {list(df.columns)[:30]} ...")

    # normalize seq + keep minimal columns to reduce memory
    use_cols = [args.seq_col, args.geno_col, args.cell_col]
    if args.only_positive:
        if args.label_col is None:
            raise ValueError("--only-positive requires --label-col")
        use_cols.append(args.label_col)
    df = df[use_cols].copy()

    # print distribution
    print(f"[INFO] seq_col={args.seq_col}, geno_col={args.geno_col}, cell_col={args.cell_col}")
    print("[INFO] group distribution (top):")
    print(df[args.geno_col].value_counts(dropna=False).head(10))
    print("[INFO] cell distribution (top):")
    print(df[args.cell_col].value_counts(dropna=False).head(10))

    # parse params
    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    win_map = parse_win_map(args.win_map)

    # choose cells
    if args.cells is None:
        cells = list(df[args.cell_col].astype(str).value_counts().index)
    else:
        cells = [x.strip() for x in args.cells.split(",") if x.strip()]

    # run per cell
    for cell in cells:
        print(f"[INFO] running cell={cell}")
        run_for_cell(
            df=df,
            seq_col=args.seq_col,
            geno_col=args.geno_col,
            cell_col=args.cell_col,
            label_col=args.label_col,
            only_positive=args.only_positive,
            cell=cell,
            tag=args.tag,
            outdir=outdir,
            k_list=k_list,
            win_map=win_map,
            pwm_win=args.pwm_win,
            min_count=args.min_count,
            pc=args.pc,
            exclude_center_c=args.exclude_center_c,
        )

    print(f"[OK] wrote outputs to: {outdir}")


if __name__ == "__main__":
    main()
