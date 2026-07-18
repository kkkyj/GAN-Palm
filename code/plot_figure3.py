#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_figure3.py

Final main-text Figure 3:
  A. HeLa top positive / negative unique sites
  B. HeLa positional enrichment heatmap (WT - KO)
  C. HeLa short-window Shannon entropy profile:
     top WT-biased sites vs weak-shift sites
  D. Cross-dataset remodeling magnitude summary (HeLa vs PANC-1)
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_ORDER)


def eprint(*args):
    print(*args, flush=True)


def bootstrap_ci_mean(x: np.ndarray, n_boot: int = 2000, seed: int = 13, alpha: float = 0.05):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(x)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        samp = x[rng.integers(0, n, size=n)]
        means[i] = np.mean(samp)
    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)
    return float(np.mean(x)), float(lo), float(hi)


def clean_seq(seq: str) -> str:
    seq = str(seq).upper().strip()
    return "".join(ch for ch in seq if ch in AA_SET or ch == "X")


def site_key_from_df(df: pd.DataFrame) -> pd.Series:
    acc = df["Accession"].astype(str).str.strip()
    pos = df["Position"].astype(str).str.strip()
    return acc + "|" + pos


def dedup_unique_sites(df: pd.DataFrame, keep: str = "extreme") -> pd.DataFrame:
    df = df.copy()
    df["site_key"] = site_key_from_df(df)
    if keep == "extreme":
        df["_abs_delta"] = df["delta"].abs()
        df = df.sort_values(["site_key", "_abs_delta"], ascending=[True, False])
        out = df.drop_duplicates("site_key", keep="first").copy()
        out = out.drop(columns=["_abs_delta"])
        return out
    return df.drop_duplicates("site_key", keep="first").copy()


def make_short_window(seq: str, flank: int = 10) -> str:
    seq = clean_seq(seq)
    if len(seq) == 0:
        return ""
    center = len(seq) // 2
    lo = max(0, center - flank)
    hi = min(len(seq), center + flank + 1)
    return seq[lo:hi]


def filter_equal_length_seqs(seqs: List[str], min_n: int = 5) -> List[str]:
    seqs = [clean_seq(s) for s in seqs]
    seqs = [s for s in seqs if len(s) > 0]
    if len(seqs) < min_n:
        return []
    lengths = pd.Series([len(s) for s in seqs])
    L = int(lengths.mode().iloc[0])
    seqs = [s for s in seqs if len(s) == L]
    return seqs


def shannon_entropy_profile(seqs: List[str]) -> Tuple[np.ndarray, List[int]]:
    """
    Returns:
      entropy: shape (L,)
      rel_pos: positions relative to center
    """
    seqs = filter_equal_length_seqs(seqs)
    if len(seqs) == 0:
        return np.array([]), []

    L = len(seqs[0])
    aa_to_idx = {aa: i for i, aa in enumerate(AA_ORDER)}
    counts = np.zeros((20, L), dtype=float)

    for s in seqs:
        for i, aa in enumerate(s):
            if aa in aa_to_idx:
                counts[aa_to_idx[aa], i] += 1.0

    probs = counts / np.clip(counts.sum(axis=0, keepdims=True), 1e-12, None)
    probs = np.clip(probs, 1e-12, 1.0)
    entropy = -np.sum(probs * np.log2(probs), axis=0)

    center = L // 2
    rel_pos = [i - center for i in range(L)]
    return entropy, rel_pos


def load_heatmap_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    aa_cols = [c for c in df.columns if c in AA_ORDER]
    if "pos_rel_to_Cys" not in df.columns:
        raise ValueError(f"Missing pos_rel_to_Cys in {path}")
    if not aa_cols:
        raise ValueError(f"No amino-acid columns found in {path}")
    return df[["pos_rel_to_Cys"] + aa_cols].copy()


def plot_heatmap(ax, df: pd.DataFrame, title: str):
    aa_cols = [c for c in df.columns if c in AA_ORDER]
    mat = df[aa_cols].to_numpy(dtype=float).T
    positions = df["pos_rel_to_Cys"].tolist()

    vmax = np.nanpercentile(np.abs(mat), 98)
    vmax = max(vmax, 0.3)

    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_yticks(np.arange(len(aa_cols)))
    ax.set_yticklabels(aa_cols, fontsize=8)

    show_ticks, show_labels = [], []
    for i, p in enumerate(positions):
        if p in (-10, -5, 0, 5, 10):
            show_ticks.append(i)
            show_labels.append("C" if p == 0 else str(p))
    ax.set_xticks(show_ticks)
    ax.set_xticklabels(show_labels, fontsize=8)

    ax.set_xlabel("Position relative to central Cys", fontsize=9)
    ax.set_title(title, fontsize=11)
    return im


def load_sites_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["Accession", "Position", "delta", "seq101"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"{path} missing columns: {miss}")
    return df.copy()


def plot_hela_top_sites(ax, top_df: pd.DataFrame, bottom_df: pd.DataFrame, n_each: int = 6):
    top_df = dedup_unique_sites(top_df, keep="extreme").sort_values("delta", ascending=False).head(n_each).copy()
    bottom_df = dedup_unique_sites(bottom_df, keep="extreme").sort_values("delta", ascending=True).head(n_each).copy()

    show = pd.concat([top_df, bottom_df], axis=0, ignore_index=True).copy()
    show["label"] = show["Accession"].astype(str) + "\n" + show["Position"].astype(str)

    y = np.arange(len(show))
    vals = show["delta"].to_numpy(dtype=float)

    ax.barh(y, vals)
    ax.set_yticks(y)
    ax.set_yticklabels(show["label"], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0.0, ls="--", lw=1)
    ax.set_xlabel("Δ score (WT − KO)")
    ax.set_title("HeLa unique sites with the largest context-dependent shifts")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if len(top_df) > 0 and len(bottom_df) > 0:
        pos_max = float(top_df["delta"].max())
        neg_min = float(bottom_df["delta"].min())
        ax.text(
            0.98, 0.03,
            f"n={len(top_df)} / direction\nmax Δ={pos_max:.3f}\nmin Δ={neg_min:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9)
        )


def plot_entropy_panel(ax, pos_seqs: List[str], bg_seqs: List[str], title: str):
    H_pos, rel_pos1 = shannon_entropy_profile(pos_seqs)
    H_bg, rel_pos2 = shannon_entropy_profile(bg_seqs)

    if len(H_pos) == 0 or len(H_bg) == 0:
        ax.text(0.5, 0.5, "No usable sequences", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        ax.set_title(title, fontsize=11)
        return

    L = min(len(H_pos), len(H_bg))
    H_pos = H_pos[:L]
    H_bg = H_bg[:L]
    rel_pos = rel_pos1[:L]

    ax.plot(rel_pos, H_pos, marker="o", ms=3, lw=1.8, label="Top WT-biased")
    ax.plot(rel_pos, H_bg, marker="o", ms=3, lw=1.8, label="Weak-shift background")

    ax.axvline(0, ls="--", lw=1)
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_xticklabels(["-10", "-5", "C", "5", "10"])
    ax.set_xlabel("Position relative to central Cys")
    ax.set_ylabel("Shannon entropy (bits)")
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    delta = H_bg - H_pos
    strong = np.argsort(-np.abs(delta))[:3]
    txt = ", ".join([f"{rel_pos[i]}:{delta[i]:+.2f}" if rel_pos[i] != 0 else f"C:{delta[i]:+.2f}" for i in strong])
    ax.text(
        0.02, 0.02,
        f"Top Δentropy (bg-top): {txt}",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.9)
    )


def summarize_dataset_magnitude(top_df: pd.DataFrame, bottom_df: pd.DataFrame, dataset_name: str) -> dict:
    top_df = dedup_unique_sites(top_df, keep="extreme")
    bottom_df = dedup_unique_sites(bottom_df, keep="extreme")
    merged = pd.concat([top_df, bottom_df], axis=0, ignore_index=True).copy()
    merged = dedup_unique_sites(merged, keep="extreme")

    x = merged["delta"].to_numpy(dtype=float)
    absx = np.abs(x)
    mean_abs, lo, hi = bootstrap_ci_mean(absx, n_boot=2000, seed=13)
    median_abs = float(np.median(absx)) if len(absx) else np.nan
    n_unique = int(len(merged))

    return {
        "dataset": dataset_name,
        "n_unique": n_unique,
        "mean_abs_delta": mean_abs,
        "ci_lo": lo,
        "ci_hi": hi,
        "median_abs_delta": median_abs,
    }


def plot_cross_dataset_summary(ax, hela_stats: dict, panc_stats: dict):
    df = pd.DataFrame([hela_stats, panc_stats])

    x = np.arange(len(df))
    y = df["mean_abs_delta"].to_numpy(dtype=float)
    yerr_lo = y - df["ci_lo"].to_numpy(dtype=float)
    yerr_hi = df["ci_hi"].to_numpy(dtype=float) - y

    ax.bar(x, y)
    ax.errorbar(x, y, yerr=[yerr_lo, yerr_hi], fmt="none", capsize=4, lw=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(df["dataset"].tolist(), fontsize=10)
    ax.set_ylabel("Mean |Δ score| across unique shifted sites")
    ax.set_title("Cross-dataset comparison of remodeling magnitude")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, row in df.iterrows():
        ax.text(
            i, row["mean_abs_delta"],
            f"n={int(row['n_unique'])}\nmedian={row['median_abs_delta']:.3f}",
            ha="center", va="bottom", fontsize=9
        )


def load_all_inputs(fig3_dir: str, heatmap_root: str):
    fig3_dir = Path(fig3_dir)
    heatmap_root = Path(heatmap_root)

    paths = {
        "hela_top": fig3_dir / "fig3B_top_sites_Hela.csv",
        "hela_bottom": fig3_dir / "fig3B_bottom_sites_Hela.csv",
        "panc_top": fig3_dir / "fig3B_top_sites_PANC-1.csv",
        "panc_bottom": fig3_dir / "fig3B_bottom_sites_PANC-1.csv",
        "hela_heatmap": heatmap_root / "Hela" / "delta_lo_heatmap_wt_minus_ko.tsv",
        "panc_heatmap": heatmap_root / "PANC-1" / "delta_lo_heatmap_wt_minus_ko.tsv",
    }

    for _, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    data = {
        "hela_top": load_sites_csv(paths["hela_top"]),
        "hela_bottom": load_sites_csv(paths["hela_bottom"]),
        "panc_top": load_sites_csv(paths["panc_top"]),
        "panc_bottom": load_sites_csv(paths["panc_bottom"]),
        "hela_heatmap": load_heatmap_tsv(str(paths["hela_heatmap"])),
        "panc_heatmap": load_heatmap_tsv(str(paths["panc_heatmap"])),
    }
    return data


def make_figure(data: dict, out_prefix: str, hela_n_each: int = 6, hela_logo_n: int = 120, flank: int = 10):
    fig = plt.figure(figsize=(15, 11.5))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0], width_ratios=[1.15, 1.0], hspace=0.38, wspace=0.30)

    axA = fig.add_subplot(gs[0, 0])
    plot_hela_top_sites(axA, data["hela_top"], data["hela_bottom"], n_each=hela_n_each)

    axB = fig.add_subplot(gs[0, 1])
    im = plot_heatmap(axB, data["hela_heatmap"], "HeLa positional enrichment (WT − KO)")

    axC = fig.add_subplot(gs[1, 0])
    hela_top_unique = dedup_unique_sites(data["hela_top"], keep="extreme").sort_values("delta", ascending=False).copy()
    hela_all_unique = dedup_unique_sites(pd.concat([data["hela_top"], data["hela_bottom"]], axis=0, ignore_index=True), keep="extreme")

    pos_df = hela_top_unique.head(hela_logo_n).copy()
    pos_keys = set(pos_df["site_key"].tolist())

    bg_df = hela_all_unique[~hela_all_unique["site_key"].isin(pos_keys)].copy()
    bg_df["abs_delta"] = bg_df["delta"].abs()
    bg_df = bg_df.sort_values("abs_delta", ascending=True).head(max(hela_logo_n, 80)).copy()

    pos_seqs = [make_short_window(s, flank=flank) for s in pos_df["seq101"].astype(str).tolist()]
    bg_seqs = [make_short_window(s, flank=flank) for s in bg_df["seq101"].astype(str).tolist()]

    plot_entropy_panel(
        axC,
        pos_seqs=pos_seqs,
        bg_seqs=bg_seqs,
        title=f"HeLa short-window entropy profile: top WT-biased vs weak-shift sites (Cys ±{flank})",
    )

    axD = fig.add_subplot(gs[1, 1])
    hela_stats = summarize_dataset_magnitude(data["hela_top"], data["hela_bottom"], "HeLa")
    panc_stats = summarize_dataset_magnitude(data["panc_top"], data["panc_bottom"], "PANC-1")
    plot_cross_dataset_summary(axD, hela_stats, panc_stats)

    cbar = fig.colorbar(im, ax=axB, fraction=0.045, pad=0.03)
    cbar.set_label("log2 enrichment")

    for ax, lab in [(axA, "A"), (axB, "B"), (axC, "C"), (axD, "D")]:
        ax.text(-0.12, 1.08, lab, transform=ax.transAxes, fontsize=18, fontweight="bold", va="top")

    fig.suptitle(
        "Figure 3. Site-specific palmitoylation remodeling associated with ZDHHC20 perturbation",
        fontsize=16, y=0.985
    )

    out_prefix = str(out_prefix)
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)

    png = out_prefix + ".png"
    pdf = out_prefix + ".pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    summary_df = pd.DataFrame([hela_stats, panc_stats])
    summary_path = out_prefix + ".summary.csv"
    summary_df.to_csv(summary_path, index=False)

    eprint(f"[OK] wrote {png}")
    eprint(f"[OK] wrote {pdf}")
    eprint(f"[OK] wrote {summary_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig3-dir", type=str, default="results/fig3_noesm_human")
    ap.add_argument("--heatmap-root", type=str, default="results/wt_ko_delta_vs_fake")
    ap.add_argument("--out-prefix", type=str, default="results/fig3/figure3_main")
    ap.add_argument("--hela-n-each", type=int, default=6)
    ap.add_argument("--hela-logo-n", type=int, default=120)
    ap.add_argument("--flank", type=int, default=10)
    return ap.parse_args()


def main():
    args = parse_args()

    eprint(f"[INFO] fig3-dir      = {args.fig3_dir}")
    eprint(f"[INFO] heatmap-root = {args.heatmap_root}")
    eprint(f"[INFO] out-prefix   = {args.out_prefix}")

    data = load_all_inputs(args.fig3_dir, args.heatmap_root)
    make_figure(
        data=data,
        out_prefix=args.out_prefix,
        hela_n_each=args.hela_n_each,
        hela_logo_n=args.hela_logo_n,
        flank=args.flank,
    )


if __name__ == "__main__":
    main()