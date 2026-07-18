#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_fake_bg.py
==================

Purpose
----
Treat the GAN-generated fake sequences as a "controllable background", used for downstream motif / logo analysis:
1) use fake in place of real neg for motif discovery (pos vs fake)
2) compare pos vs fake (as well as pos vs neg, neg vs fake) to highlight functional motifs
3) produce position-specific logos (especially the neighborhood of the central Cys)

Inputs
----
A) site_table (parquet/csv/tsv):
   - seq_col: the window sequence of each sample (default seq101, length L=101)
   - label_col: binary classification label (default label_bin, >pos_thr treated as pos)
   - optional dataset_col: used to group output by dataset (--by-dataset)

B) fake fasta (one or more):
   - pointed to by --samples-glob, e.g. .../samples/sample_step*.fasta

Output directory structure (outdir)
--------------------
outdir/
  run_meta.json                        # this run's parameters and sample counts
  fasta/
    pos.fasta                          # real positive samples (after subsampling)
    neg_subsample.fasta                # real negative samples (after subsampling)
    fake.fasta                         # fake background (after merging/dedup/subsampling)
  logo/
    pos_freq.tsv, fake_freq.tsv        # (L x 20) AA frequency at each position
    logodds_pos_vs_fake.tsv            # (L x 20) log2(pos/fake)
    logodds_pos_vs_neg.tsv             # (L x 20) log2(pos/neg)
    logodds_neg_vs_fake.tsv            # (L x 20) log2(neg/fake)
    *.png                              # logo or heatmap plots (full + cys zoom)
  kmer/
    kmer_enrich_pos_vs_fake.tsv        # k-mer enrichment (k configurable)
    kmer_enrich_pos_vs_neg.tsv
    kmer_enrich_neg_vs_fake.tsv
  streme/                              # optional: requires meme-suite
    pos_vs_fake/...
    pos_vs_neg/...
    neg_vs_fake/...
    status.txt

Dependencies
----
Required: numpy, pandas, matplotlib
Optional:
  - logomaker: output real sequence logos (otherwise output heatmaps)
  - meme-suite(streme): motif discovery (otherwise skipped)

Usage tips
--------
- Your data is very large, so by default pos/neg/fake are subsampled (--max-pos/--max-neg/--max-fake)
- To avoid negative-sign argument issues, cys-window is split into two integer arguments: --cys-left -10 --cys-right 10

"""

import argparse
import gzip
import math
import os
import random
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

AA20 = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA20)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() in [".tsv", ".tab"]:
        return pd.read_csv(p, sep="\t")
    return pd.read_csv(p)


def iter_fasta(path: str) -> Iterable[Tuple[str, str]]:
    op = gzip.open if path.endswith(".gz") else open
    header = None
    seq_chunks = []
    with op(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())
        if header is not None:
            yield header, "".join(seq_chunks)


def glob_paths(pattern: str) -> List[str]:
    import glob
    return sorted(glob.glob(pattern))


def load_fake_from_samples(samples_glob: str, use_latest_only: bool, dedup: bool) -> List[str]:
    paths = glob_paths(samples_glob)
    if not paths:
        raise FileNotFoundError(f"No fake fasta found by glob: {samples_glob}")

    if use_latest_only:
        paths = [max(paths, key=lambda x: Path(x).stat().st_mtime)]

    seqs: List[str] = []
    for fp in paths:
        for _, s in iter_fasta(fp):
            s = (s or "").strip().upper()
            if s:
                seqs.append(s)

    if dedup:
        seen = set()
        uniq = []
        for s in seqs:
            if s not in seen:
                uniq.append(s)
                seen.add(s)
        seqs = uniq
    return seqs


def write_fasta(seqs: Sequence[str], path: str, prefix: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for i, s in enumerate(seqs):
            f.write(f">{prefix}_{i}\n{s.strip().upper()}\n")


def clean_and_filter_seqs(seqs: Sequence[str], L: int, center_pos: int, require_center_c: bool) -> List[str]:
    out = []
    for s in seqs:
        s = (s or "").strip().upper()
        if len(s) != L:
            continue
        if any(ch not in AA_SET for ch in s):
            continue
        if require_center_c and s[center_pos] != "C":
            continue
        out.append(s)
    return out


def subsample(seqs: Sequence[str], n: int, seed: int) -> List[str]:
    if n <= 0 or n >= len(seqs):
        return list(seqs)
    rng = random.Random(seed)
    idx = list(range(len(seqs)))
    rng.shuffle(idx)
    idx = idx[:n]
    return [seqs[i] for i in idx]


def df_to_pos_neg(df: pd.DataFrame, seq_col: str, label_col: str, pos_thr: float) -> Tuple[List[str], List[str]]:
    y = pd.to_numeric(df[label_col], errors="coerce").fillna(0.0).astype(float).to_numpy()
    seqs = df[seq_col].astype(str).tolist()
    pos = [seqs[i] for i in range(len(seqs)) if y[i] > pos_thr]
    neg = [seqs[i] for i in range(len(seqs)) if y[i] <= pos_thr]
    return pos, neg


def freq_matrix(seqs: Sequence[str], alphabet: List[str]) -> np.ndarray:
    if not seqs:
        raise ValueError("Empty seq list")
    L = len(seqs[0])
    A = len(alphabet)
    idx = {ch: j for j, ch in enumerate(alphabet)}
    counts = np.zeros((L, A), dtype=np.float64)
    for s in seqs:
        for i, ch in enumerate(s):
            j = idx.get(ch, None)
            if j is not None:
                counts[i, j] += 1.0
    counts /= max(1.0, float(len(seqs)))
    return counts


def log_odds(pos_freq: np.ndarray, bg_freq: np.ndarray, pseudocount: float = 1e-4) -> np.ndarray:
    p = np.clip(pos_freq, 0.0, 1.0) + pseudocount
    b = np.clip(bg_freq, 0.0, 1.0) + pseudocount
    p = p / p.sum(axis=1, keepdims=True)
    b = b / b.sum(axis=1, keepdims=True)
    return np.log2(p / b)


def save_matrix_tsv(mat: np.ndarray, alphabet: List[str], path: str):
    df = pd.DataFrame(mat, columns=alphabet)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def plot_heatmap(mat: np.ndarray, xlabels: List[int], ylabels: List[str], title: str, out_png: str):
    import matplotlib.pyplot as plt
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(10, len(xlabels) * 0.22), 6))
    plt.imshow(mat.T, aspect="auto", interpolation="nearest")
    plt.colorbar(label="score")
    plt.yticks(range(len(ylabels)), ylabels)
    plt.xticks(range(len(xlabels)), xlabels, rotation=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_logo_if_possible(mat: np.ndarray, alphabet: List[str], title: str, out_png: str):
    try:
        import logomaker
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("logomaker not installed") from e

    df = pd.DataFrame(mat, columns=alphabet)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(10, df.shape[0] * 0.22), 3.5))
    logo = logomaker.Logo(df)
    logo.ax.set_title(title)
    logo.ax.set_ylabel("score")
    logo.ax.set_xlabel("position")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def kmer_counts(seqs: Sequence[str], k: int) -> Counter:
    c = Counter()
    for s in seqs:
        for i in range(0, len(s) - k + 1):
            km = s[i:i + k]
            if all(ch in AA_SET for ch in km):
                c[km] += 1
    return c


def kmer_enrichment(pos: Sequence[str], bg: Sequence[str], k_list: List[int], pseudocount: float = 1.0) -> pd.DataFrame:
    rows = []
    for k in k_list:
        cp = kmer_counts(pos, k)
        cb = kmer_counts(bg, k)
        tp = sum(cp.values())
        tb = sum(cb.values())
        keys = set(cp.keys()) | set(cb.keys())
        denom_p = tp + pseudocount * len(keys)
        denom_b = tb + pseudocount * len(keys)
        for km in keys:
            p = (cp.get(km, 0) + pseudocount) / denom_p
            b = (cb.get(km, 0) + pseudocount) / denom_b
            rows.append({
                "k": k,
                "kmer": km,
                "count_pos": cp.get(km, 0),
                "count_bg": cb.get(km, 0),
                "freq_pos": p,
                "freq_bg": b,
                "log2_enrich": math.log2(p / b),
            })
    df = pd.DataFrame(rows)
    df.sort_values(["k", "log2_enrich"], ascending=[True, False], inplace=True)
    return df


def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def run_streme(pos_fa: str, bg_fa: str, outdir: str, seed: int, nmotifs: int):
    streme = which("streme")
    if not streme:
        return False, "streme not found in PATH"

    Path(outdir).mkdir(parents=True, exist_ok=True)
    cmd = [streme, "--p", pos_fa, "--n", bg_fa, "--oc", outdir, "--seed", str(seed), "--nmotifs", str(nmotifs)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (Path(outdir) / "streme_cmd.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    (Path(outdir) / "streme_stdout.txt").write_text(proc.stdout, encoding="utf-8")
    ok = (proc.returncode == 0)
    return ok, ("ok" if ok else f"failed (code={proc.returncode})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-table", required=True)
    ap.add_argument("--seq-col", default="seq101")
    ap.add_argument("--label-col", default="label_bin")
    ap.add_argument("--pos-thr", type=float, default=0.5)

    ap.add_argument("--samples-glob", required=True)
    ap.add_argument("--use-latest-fake-only", action="store_true")

    ap.add_argument("--outdir", required=True)

    ap.add_argument("--L", type=int, default=101)
    ap.add_argument("--center-pos", type=int, default=50)
    ap.add_argument("--require-center-c", action="store_true", default=True)

    ap.add_argument("--max-pos", type=int, default=50000)
    ap.add_argument("--max-neg", type=int, default=200000)
    ap.add_argument("--max-fake", type=int, default=50000)

    ap.add_argument("--kmer", default="3,4,5,6")

    # Fix: use two integers to avoid "-10,10" being parsed as an argument
    ap.add_argument("--cys-left", type=int, default=-10, help="relative left bound around center (inclusive)")
    ap.add_argument("--cys-right", type=int, default=10, help="relative right bound around center (inclusive)")

    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--by-dataset", action="store_true")
    ap.add_argument("--dataset-col", default="dataset")

    ap.add_argument("--run-streme", action="store_true")
    ap.add_argument("--nmotifs", type=int, default=20)

    args = ap.parse_args()
    seed_all(args.seed)

    outdir = Path(args.outdir)
    (outdir / "fasta").mkdir(parents=True, exist_ok=True)
    (outdir / "logo").mkdir(parents=True, exist_ok=True)
    (outdir / "kmer").mkdir(parents=True, exist_ok=True)
    (outdir / "streme").mkdir(parents=True, exist_ok=True)

    # Load pos/neg
    df = read_table(args.site_table)
    if args.seq_col not in df.columns or args.label_col not in df.columns:
        raise ValueError(f"Missing columns. Have: {list(df.columns)[:50]}")

    pos_raw, neg_raw = df_to_pos_neg(df, args.seq_col, args.label_col, args.pos_thr)
    pos = clean_and_filter_seqs(pos_raw, args.L, args.center_pos, args.require_center_c)
    neg = clean_and_filter_seqs(neg_raw, args.L, args.center_pos, args.require_center_c)

    pos = subsample(pos, min(args.max_pos, len(pos)), args.seed)
    neg = subsample(neg, min(args.max_neg, len(neg)), args.seed + 7)

    # Load fake
    fake_raw = load_fake_from_samples(args.samples_glob, use_latest_only=args.use_latest_fake_only, dedup=True)
    fake = clean_and_filter_seqs(fake_raw, args.L, args.center_pos, args.require_center_c)
    fake = subsample(fake, min(args.max_fake, len(fake)), args.seed + 13)

    # Save fasta
    pos_fa = str(outdir / "fasta" / "pos.fasta")
    neg_fa = str(outdir / "fasta" / "neg_subsample.fasta")
    fake_fa = str(outdir / "fasta" / "fake.fasta")
    write_fasta(pos, pos_fa, "pos")
    write_fasta(neg, neg_fa, "neg")
    write_fasta(fake, fake_fa, "fake")

    meta = {
        "site_table": args.site_table,
        "seq_col": args.seq_col,
        "label_col": args.label_col,
        "pos_thr": args.pos_thr,
        "L": args.L,
        "center_pos": args.center_pos,
        "require_center_c": args.require_center_c,
        "n_pos_used": len(pos),
        "n_neg_used": len(neg),
        "n_fake_used": len(fake),
        "samples_glob": args.samples_glob,
        "use_latest_fake_only": args.use_latest_fake_only,
        "seed": args.seed,
        "cys_left": args.cys_left,
        "cys_right": args.cys_right,
    }
    (outdir / "run_meta.json").write_text(json_dumps(meta), encoding="utf-8")

    # Position-specific stats
    pos_freq = freq_matrix(pos, AA20)
    neg_freq = freq_matrix(neg, AA20)
    fake_freq = freq_matrix(fake, AA20)

    lo_pos_fake = log_odds(pos_freq, fake_freq, pseudocount=1e-4)
    lo_pos_neg  = log_odds(pos_freq, neg_freq,  pseudocount=1e-4)
    lo_neg_fake = log_odds(neg_freq, fake_freq, pseudocount=1e-4)

    save_matrix_tsv(pos_freq, AA20, str(outdir / "logo" / "pos_freq.tsv"))
    save_matrix_tsv(fake_freq, AA20, str(outdir / "logo" / "fake_freq.tsv"))
    save_matrix_tsv(lo_pos_fake, AA20, str(outdir / "logo" / "logodds_pos_vs_fake.tsv"))
    save_matrix_tsv(lo_pos_neg,  AA20, str(outdir / "logo" / "logodds_pos_vs_neg.tsv"))
    save_matrix_tsv(lo_neg_fake, AA20, str(outdir / "logo" / "logodds_neg_vs_fake.tsv"))

    rel = [i - args.center_pos for i in range(args.L)]

    # Full-window plots
    try:
        plot_logo_if_possible(lo_pos_fake, AA20, "Logo (log2 odds): pos vs fake", str(outdir / "logo" / "logo_logodds_pos_vs_fake.full.png"))
    except Exception:
        plot_heatmap(lo_pos_fake, rel, AA20, "Heatmap (log2 odds): pos vs fake", str(outdir / "logo" / "heatmap_logodds_pos_vs_fake.full.png"))

    # Zoom window around center Cys
    s = max(0, args.center_pos + args.cys_left)
    e = min(args.L, args.center_pos + args.cys_right + 1)
    rel_zoom = rel[s:e]
    lo_zoom = lo_pos_fake[s:e, :]

    try:
        plot_logo_if_possible(lo_zoom, AA20, f"Logo (log2 odds): pos vs fake [{args.cys_left},{args.cys_right}]", str(outdir / "logo" / "logo_logodds_pos_vs_fake.cys_zoom.png"))
    except Exception:
        plot_heatmap(lo_zoom, rel_zoom, AA20, f"Heatmap (log2 odds): pos vs fake [{args.cys_left},{args.cys_right}]", str(outdir / "logo" / "heatmap_logodds_pos_vs_fake.cys_zoom.png"))

    # k-mer enrichment
    k_list = [int(x) for x in args.kmer.split(",") if x.strip()]
    df_k1 = kmer_enrichment(pos, fake, k_list, pseudocount=1.0)
    df_k1.to_csv(outdir / "kmer" / "kmer_enrich_pos_vs_fake.tsv", sep="\t", index=False)

    df_k2 = kmer_enrichment(pos, neg, k_list, pseudocount=1.0)
    df_k2.to_csv(outdir / "kmer" / "kmer_enrich_pos_vs_neg.tsv", sep="\t", index=False)

    df_k3 = kmer_enrichment(neg, fake, k_list, pseudocount=1.0)
    df_k3.to_csv(outdir / "kmer" / "kmer_enrich_neg_vs_fake.tsv", sep="\t", index=False)

    # STREME (optional)
    if args.run_streme:
        ok1, msg1 = run_streme(pos_fa, fake_fa, str(outdir / "streme" / "pos_vs_fake"), seed=args.seed, nmotifs=args.nmotifs)
        ok2, msg2 = run_streme(pos_fa, neg_fa,  str(outdir / "streme" / "pos_vs_neg"),  seed=args.seed, nmotifs=args.nmotifs)
        ok3, msg3 = run_streme(neg_fa, fake_fa, str(outdir / "streme" / "neg_vs_fake"), seed=args.seed, nmotifs=args.nmotifs)
        (outdir / "streme" / "status.txt").write_text(
            f"pos_vs_fake: {msg1}\npos_vs_neg: {msg2}\nneg_vs_fake: {msg3}\n",
            encoding="utf-8"
        )

    # By dataset (optional, only a light version)
    if args.by_dataset and args.dataset_col in df.columns:
        base = outdir / "by_dataset"
        base.mkdir(parents=True, exist_ok=True)
        vc = df[args.dataset_col].dropna().astype(str).value_counts()
        top_ds = vc.head(20).index.tolist()

        for ds in top_ds:
            sdf = df[df[args.dataset_col].astype(str) == ds]
            p_raw, _ = df_to_pos_neg(sdf, args.seq_col, args.label_col, args.pos_thr)
            p = clean_and_filter_seqs(p_raw, args.L, args.center_pos, args.require_center_c)
            if len(p) < 200:
                continue
            p = subsample(p, min(args.max_pos, len(p)), args.seed)
            ds_dir = base / re.sub(r"[^A-Za-z0-9_.-]+", "_", ds)
            ds_dir.mkdir(parents=True, exist_ok=True)

            pf = freq_matrix(p, AA20)
            lo = log_odds(pf, fake_freq, pseudocount=1e-4)
            save_matrix_tsv(lo, AA20, str(ds_dir / "logodds_pos_vs_fake.tsv"))

            lo_zoom_ds = lo[s:e, :]
            try:
                plot_logo_if_possible(lo_zoom_ds, AA20, f"{ds}: pos vs fake [{args.cys_left},{args.cys_right}]", str(ds_dir / "logo_pos_vs_fake.cys_zoom.png"))
            except Exception:
                plot_heatmap(lo_zoom_ds, rel_zoom, AA20, f"{ds}: pos vs fake [{args.cys_left},{args.cys_right}]", str(ds_dir / "heatmap_pos_vs_fake.cys_zoom.png"))

            if args.run_streme:
                p_fa = str(ds_dir / "pos.fasta")
                write_fasta(p, p_fa, f"pos_{ds}")
                run_streme(p_fa, fake_fa, str(ds_dir / "streme_pos_vs_fake"), seed=args.seed, nmotifs=args.nmotifs)

    print("[OK] Done.")
    print(f"Outputs in: {outdir}")


def json_dumps(obj) -> str:
    import json
    return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    main()