#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
interpret_wtko_single_species_noesm.py

Single-species (human or mouse or any single one) WT vs KO analysis:
- automatically selects the top3 cell/tissue with the most samples (or those you explicitly specify)
- outputs:
  - Δfreq(KO-WT) heatmap for each cell_type (center window)
  - per-position JS divergence curve for each cell_type
  - top differential position table, k-mer enrichment table
  - summary metrics table + md brief

Inputs:
- --site-table: parquet/csv/tsv, containing at least seq101, genotype(WT/KO), cell_type/tissue
- optional --gen-table: GAN-generated sequence table (also requires seq101 + genotype + cell_type)

Note:
- This script does not perform cross-species comparison; for cross-species comparison use the second script compare_species_delta_signatures.py
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


AA20 = list("ACDEFGHIKLMNPQRSTVWY")
SEQ_LEN = 101
CENTER = 50  # 0-based index


# ---------------- utils ----------------

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

def guess_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    return None

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
    if s in ("WT", "WILDTYPE", "WILD_TYPE", "CONTROL"):
        return "WT"
    if s in ("KO", "KNOCKOUT", "KNOCK_OUT", "ZDHHC20_KO"):
        return "KO"
    if re.search(r"\bWT\b", s):
        return "WT"
    if re.search(r"\bKO\b", s):
        return "KO"
    return None

def onehot_counts(seqs: List[str]) -> np.ndarray:
    counts = np.zeros((SEQ_LEN, len(AA20)), dtype=np.int64)
    aa2i = {a: i for i, a in enumerate(AA20)}
    for s in seqs:
        if s is None:
            continue
        for i, ch in enumerate(s):
            j = aa2i.get(ch, None)
            if j is not None:
                counts[i, j] += 1
    return counts

def freq_from_counts(counts: np.ndarray, pseudocount: float = 0.5) -> np.ndarray:
    x = counts.astype(np.float64) + pseudocount
    denom = x.sum(axis=1, keepdims=True)
    return x / np.maximum(denom, 1e-12)

def log_odds(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(p, 1e-12) / np.maximum(q, 1e-12))

def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(np.maximum(p, 1e-12)) - np.log(np.maximum(m, 1e-12))), axis=1)
    kl_qm = np.sum(q * (np.log(np.maximum(q, 1e-12)) - np.log(np.maximum(m, 1e-12))), axis=1)
    return 0.5 * (kl_pm + kl_qm)

def extract_kmers(seq: str, positions: List[int], k: int) -> List[str]:
    out = []
    for pos in positions:
        start = pos
        end = pos + k
        if start < 0 or end > len(seq):
            continue
        km = seq[start:end]
        if len(km) == k:
            out.append(km)
    return out

def kmer_enrichment(
    seqs_wt: List[str],
    seqs_ko: List[str],
    k: int,
    win_l: int,
    win_r: int,
    topn: int = 30,
) -> pd.DataFrame:
    positions = list(range(CENTER - win_l, CENTER + win_r - k + 2))
    wt_counts: Dict[str, int] = {}
    ko_counts: Dict[str, int] = {}

    for s in seqs_wt:
        if s is None:
            continue
        for km in extract_kmers(s, positions, k):
            wt_counts[km] = wt_counts.get(km, 0) + 1
    for s in seqs_ko:
        if s is None:
            continue
        for km in extract_kmers(s, positions, k):
            ko_counts[km] = ko_counts.get(km, 0) + 1

    all_km = set(wt_counts) | set(ko_counts)
    rows = []
    for km in all_km:
        a = ko_counts.get(km, 0)
        b = wt_counts.get(km, 0)
        l2fc = np.log2((a + 1.0) / (b + 1.0))
        rows.append((km, a, b, l2fc))

    df = pd.DataFrame(rows, columns=["kmer", "ko_count", "wt_count", "log2fc_ko_vs_wt"])
    df = df.sort_values("log2fc_ko_vs_wt", ascending=False)
    head = df.head(topn)
    tail = df.tail(topn).sort_values("log2fc_ko_vs_wt", ascending=True)
    out = pd.concat([head, tail], axis=0).drop_duplicates("kmer")
    return out.reset_index(drop=True)

def build_delta_signature(df: pd.DataFrame, seq_col: str, geno_col: str, cell_col: str, celltype: str, w: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    d = df.copy()
    d["_geno"] = d[geno_col].astype(str).apply(infer_wtko)
    d = d[~d["_geno"].isna()].copy()
    d["_seq"] = d[seq_col].apply(normalize_seq)
    d = d[~d["_seq"].isna()].copy()
    d = d[d[cell_col].astype(str) == celltype].copy()

    wt = d[d["_geno"] == "WT"]["_seq"].tolist()
    ko = d[d["_geno"] == "KO"]["_seq"].tolist()

    c_wt = onehot_counts(wt)
    c_ko = onehot_counts(ko)
    f_wt = freq_from_counts(c_wt)
    f_ko = freq_from_counts(c_ko)

    delta = f_ko - f_wt
    js = js_divergence(f_ko, f_wt)

    sl = slice(CENTER - w, CENTER + w + 1)
    delta_sig = delta[sl, :].reshape(-1)
    js_sig = js[sl]
    return delta_sig, js_sig


# ---------------- core ----------------

def analyze_one_dataset(
    df: pd.DataFrame,
    seq_col: str,
    geno_col: str,
    cell_col: str,
    outdir: Path,
    tag: str,
    celltypes: Optional[List[str]] = None,
):
    mkdir(outdir / "tables")
    mkdir(outdir / "figs")
    mkdir(outdir / "report")

    d = df.copy()
    d["_geno"] = d[geno_col].astype(str).apply(infer_wtko)
    d = d[~d["_geno"].isna()].copy()
    d["_seq"] = d[seq_col].apply(normalize_seq)
    d = d[~d["_seq"].isna()].copy()

    if celltypes is None:
        celltypes = list(d[cell_col].astype(str).value_counts().head(3).index)
    else:
        celltypes = [ct for ct in celltypes if ct in set(d[cell_col].astype(str).unique())]
        if len(celltypes) == 0:
            raise ValueError(f"None of provided celltypes exist in column {cell_col}")

    summary_rows = []

    for ct in celltypes:
        sub = d[d[cell_col].astype(str) == ct].copy()
        wt = sub[sub["_geno"] == "WT"]["_seq"].tolist()
        ko = sub[sub["_geno"] == "KO"]["_seq"].tolist()

        c_wt = onehot_counts(wt)
        c_ko = onehot_counts(ko)
        f_wt = freq_from_counts(c_wt)
        f_ko = freq_from_counts(c_ko)

        delta = f_ko - f_wt
        lod = log_odds(f_ko, f_wt)
        js = js_divergence(f_ko, f_wt)
        l1 = np.sum(np.abs(delta), axis=1)

        pos = np.arange(SEQ_LEN) - CENTER

        # tables
        delta_df = pd.DataFrame(delta, columns=[f"Δ_{a}" for a in AA20])
        delta_df.insert(0, "pos_rel", pos)
        delta_df.insert(0, "pos0", np.arange(SEQ_LEN))
        delta_df.to_csv(outdir / "tables" / f"{tag}.{ct}.delta_freq_ko_minus_wt.csv", index=False)

        lod_df = pd.DataFrame(lod, columns=[f"logodds_{a}" for a in AA20])
        lod_df.insert(0, "pos_rel", pos)
        lod_df.insert(0, "pos0", np.arange(SEQ_LEN))
        lod_df.to_csv(outdir / "tables" / f"{tag}.{ct}.logodds_ko_vs_wt.csv", index=False)

        js_df = pd.DataFrame({"pos0": np.arange(SEQ_LEN), "pos_rel": pos, "js_div": js})
        js_df.to_csv(outdir / "tables" / f"{tag}.{ct}.js_divergence.csv", index=False)

        top_idx = np.argsort(-l1)[:15]
        top_df = pd.DataFrame(
            {
                "pos0": top_idx,
                "pos_rel": top_idx - CENTER,
                "l1_abs_delta_sum": l1[top_idx],
                "js_div": js[top_idx],
            }
        ).sort_values("l1_abs_delta_sum", ascending=False)
        top_df.to_csv(outdir / "tables" / f"{tag}.{ct}.top_positions.csv", index=False)

        km3 = kmer_enrichment(wt, ko, k=3, win_l=8, win_r=8, topn=25)
        km3.to_csv(outdir / "tables" / f"{tag}.{ct}.kmer3_enrich_window8.csv", index=False)
        km5 = kmer_enrichment(wt, ko, k=5, win_l=10, win_r=10, topn=25)
        km5.to_csv(outdir / "tables" / f"{tag}.{ct}.kmer5_enrich_window10.csv", index=False)

        # figs
        plt.figure()
        plt.plot(pos, js)
        plt.axvline(0, linestyle="--")
        plt.title(f"{tag} | {ct} | JS divergence (KO vs WT)")
        plt.xlabel("Position relative to center C (pos_rel)")
        plt.ylabel("JS divergence")
        plt.tight_layout()
        plt.savefig(outdir / "figs" / f"{tag}.{ct}.js_curve.png", dpi=200)
        plt.close()

        w = 15
        sl = slice(CENTER - w, CENTER + w + 1)
        plt.figure(figsize=(10, 4))
        plt.imshow(delta[sl, :].T, aspect="auto")
        plt.title(f"{tag} | {ct} | Δfreq (KO - WT), window [-{w},{w}]")
        plt.yticks(np.arange(len(AA20)), AA20)
        xt = np.arange(0, 2 * w + 1, 3)
        plt.xticks(xt, [str(int(x - w)) for x in xt])
        plt.axvline(w, linestyle="--")
        plt.xlabel("pos_rel")
        plt.colorbar(label="Δfreq")
        plt.tight_layout()
        plt.savefig(outdir / "figs" / f"{tag}.{ct}.delta_heatmap_w{w}.png", dpi=200)
        plt.close()

        summary_rows.append(
            {
                "tag": tag,
                "cell_type": ct,
                "n_wt": len(wt),
                "n_ko": len(ko),
                "mean_js": float(np.mean(js)),
                "max_js": float(np.max(js)),
                "mean_l1_abs_delta": float(np.mean(l1)),
                "max_l1_abs_delta": float(np.max(l1)),
            }
        )

    summ = pd.DataFrame(summary_rows)
    summ.to_csv(outdir / "tables" / f"{tag}.summary_by_celltype.csv", index=False)

    # delta signatures for downstream compare
    sig_rows = []
    for ct in celltypes:
        delta_sig, js_sig = build_delta_signature(d, "_seq", "_geno", cell_col, ct, w=15)
        sig_rows.append(
            {
                "cell_type": ct,
                "delta_sig": " ".join([f"{x:.6g}" for x in delta_sig]),
                "js_sig": " ".join([f"{x:.6g}" for x in js_sig]),
            }
        )
    pd.DataFrame(sig_rows).to_csv(outdir / "tables" / f"{tag}.delta_signatures_w15.csv", index=False)

    with open(outdir / "report" / f"{tag}.summary.md", "w", encoding="utf-8") as f:
        f.write(f"# {tag} WT vs KO summary\n\n")
        f.write("## Cell types analyzed\n\n")
        f.write("\n".join([f"- {x}" for x in celltypes]) + "\n\n")
        f.write("## Key metrics (per cell type)\n\n")
        f.write(summ.to_markdown(index=False))
        f.write("\n\n")
        f.write(
            "Notes:\n"
            "- Δfreq heatmap: KO - WT\n"
            "- JS divergence curve: per-position AA-distribution distance\n"
            "- k-mer enrichment: simple log2((KO+1)/(WT+1)) within center window\n"
            "- delta_signatures_w15.csv: downstream cross-dataset/species comparison input\n"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-table", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--tag", type=str, default="real")

    ap.add_argument("--seq-col", type=str, default=None)
    ap.add_argument("--geno-col", type=str, default=None)
    ap.add_argument("--cell-col", type=str, default=None)

    ap.add_argument("--celltypes", type=str, default=None, help="comma-separated; else auto top3")

    # optional generated sequences
    ap.add_argument("--gen-table", type=str, default=None)
    ap.add_argument("--gen-tag", type=str, default="gen")
    ap.add_argument("--gen-seq-col", type=str, default="seq101")
    ap.add_argument("--gen-geno-col", type=str, default="genotype")
    ap.add_argument("--gen-cell-col", type=str, default="cell_type")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    mkdir(outdir)

    df = read_any_table(Path(args.site_table))

    seq_col = args.seq_col or guess_col(df, ["seq101", "seq", "sequence", "window101"])
    geno_col = args.geno_col or guess_col(df, ["genotype", "wtko", "condition", "group", "treatment"])
    cell_col = args.cell_col or guess_col(df, ["cell_type", "celltype", "cell", "cellline", "cell_line", "tissue", "organ", "system"])

    missing = []
    if seq_col is None: missing.append("seq_col")
    if geno_col is None: missing.append("geno_col")
    if cell_col is None: missing.append("cell_col")
    if missing:
        raise ValueError(
            f"Cannot infer required columns: {missing}. "
            "Please pass --seq-col/--geno-col/--cell-col explicitly."
        )

    celltypes = None
    if args.celltypes:
        celltypes = [x.strip() for x in args.celltypes.split(",") if x.strip()]

    # real
    analyze_one_dataset(df, seq_col, geno_col, cell_col, outdir=outdir / args.tag, tag=args.tag, celltypes=celltypes)

    # optional generated
    if args.gen_table:
        dfg = read_any_table(Path(args.gen_table))
        analyze_one_dataset(
            dfg,
            args.gen_seq_col,
            args.gen_geno_col,
            args.gen_cell_col,
            outdir=outdir / args.gen_tag,
            tag=args.gen_tag,
            celltypes=celltypes,
        )

    print(f"[OK] Done. Outputs in: {outdir}")


if __name__ == "__main__":
    main()
