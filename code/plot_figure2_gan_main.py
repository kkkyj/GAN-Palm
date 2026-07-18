#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------
# Basic metrics (numpy)
# -----------------------

def rankdata_average(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sorted_a = a[order]

    i = 0
    n = len(a)
    while i < n:
        j = i + 1
        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)  # 1-based
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def auc_rank(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    y = np.concatenate([np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)])
    s = np.concatenate([pos, neg])
    r = rankdata_average(s)
    n_pos = len(pos)
    n_neg = len(neg)
    sum_r_pos = r[y == 1].sum()
    auc = (sum_r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def precision_recall_curve_np(y_true: np.ndarray, score: np.ndarray):
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)

    m = np.isfinite(score)
    y_true = y_true[m]
    score = score[m]

    order = np.argsort(-score, kind="mergesort")
    y = y_true[order]
    s = score[order]

    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    total_pos = (y == 1).sum()

    distinct = np.where(np.diff(s))[0]
    idx = np.r_[distinct, len(s) - 1]

    tp_k = tp[idx]
    fp_k = fp[idx]
    thresholds = s[idx]

    precision = tp_k / np.maximum(tp_k + fp_k, 1)
    recall = tp_k / max(total_pos, 1)

    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    thresholds = np.r_[thresholds[0] + 1e-12 if len(thresholds) else 1.0, thresholds]
    return precision, recall, thresholds


def pr_auc_np(y_true: np.ndarray, score: np.ndarray) -> float:
    p, r, _ = precision_recall_curve_np(y_true, score)
    return float(np.trapz(p, r))


# -----------------------
# Plot helpers
# -----------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_fig(fig, out_base: Path, dpi: int = 300):
    fig.savefig(str(out_base.with_suffix(".png")), dpi=dpi, bbox_inches="tight")
    fig.savefig(str(out_base.with_suffix(".pdf")), bbox_inches="tight")


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def subsample(a: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    a = np.asarray(a)
    if len(a) <= n:
        return a
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(a), size=n, replace=False)
    return a[idx]


def box_with_strip(ax, groups, labels, title, seed=0, max_points=2000):
    """
    Boxplot + jittered strip (subsampled) for skewed distributions.
    """
    ax.boxplot(groups, labels=labels, showfliers=False, widths=0.55)

    rng = np.random.default_rng(seed)
    for i, g in enumerate(groups, start=1):
        gg = g[np.isfinite(g)]
        gg = subsample(gg, max_points, seed=seed + i)
        x = rng.normal(loc=i, scale=0.06, size=len(gg))
        ax.scatter(x, gg, s=6, alpha=0.18)

    ax.set_ylabel("Discriminator score")
    ax.set_title(title, fontsize=12)
    style_axis(ax)


def mean_ci_bootstrap(a: np.ndarray, n_boot=2000, seed=0):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(a, size=len(a), replace=True)
        boots.append(np.mean(samp))
    boots = np.asarray(boots)
    return (float(np.mean(a)), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))


def plot_score_summary(ax, pos, neg, fake, seed=0):
    m_pos, lo_pos, hi_pos = mean_ci_bootstrap(pos, seed=seed + 1)
    m_neg, lo_neg, hi_neg = mean_ci_bootstrap(neg, seed=seed + 2)
    m_fake, lo_fake, hi_fake = mean_ci_bootstrap(fake, seed=seed + 3)

    labels = ["Positive", "Negative", "Generated"]
    means = np.array([m_pos, m_neg, m_fake], dtype=float)
    los = np.array([lo_pos, lo_neg, lo_fake], dtype=float)
    his = np.array([hi_pos, hi_neg, hi_fake], dtype=float)

    x = np.arange(len(labels))

    # Make CI visually obvious
    ax.errorbar(
        x, means,
        yerr=[means - los, his - means],
        fmt="o",
        capsize=8,
        elinewidth=1.6,
        markersize=6
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean discriminator score\n(bootstrap 95% CI)")
    ax.set_title("E  Score-level summary", fontsize=12)
    style_axis(ax)

    # Slight padding so CIs don't touch boundaries
    ymin = np.nanmin(los) if np.isfinite(los).any() else 0.0
    ymax = np.nanmax(his) if np.isfinite(his).any() else 1.0
    pad = max(0.03, 0.06 * (ymax - ymin))
    ax.set_ylim(ymin - pad, ymax + pad)

    return {
        "mean_pos": float(m_pos), "ci_pos": [float(lo_pos), float(hi_pos)],
        "mean_neg": float(m_neg), "ci_neg": [float(lo_neg), float(hi_neg)],
        "mean_fake": float(m_fake), "ci_fake": [float(lo_fake), float(hi_fake)],
    }


def plot_dataset_posneg(ax, df_group):
    """
    Cleaner dataset-wise ROC-AUC view:
    - y-lim zoom to [0.95, 1.0] to avoid "always 1.0" impression
    - n+ / n- placed above bars without colliding with title
    """
    dfg = df_group.copy()
    dfg = dfg.sort_values("pos_vs_neg_roc_auc", ascending=False).reset_index(drop=True)

    x = np.arange(len(dfg))
    vals = dfg["pos_vs_neg_roc_auc"].values

    ax.bar(x, vals)

    ax.set_xticks(x)
    ax.set_xticklabels(dfg["group"].tolist())
    ax.set_ylim(0.95, 1.005)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("C  Dataset-wise positive vs negative discrimination", fontsize=12, pad=10)
    style_axis(ax)

    for i, row in dfg.iterrows():
        ax.text(
            i, min(1.003, float(row["pos_vs_neg_roc_auc"]) + 0.003),
            f"n+={int(row['n_pos'])}\nn-={int(row['n_neg'])}",
            ha="center", va="bottom", fontsize=9
        )


# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser(description="Revised Figure 2 plotting (cleaner panels, no bg-correlation).")
    ap.add_argument("--result-dir", required=True, help="Directory containing scores.npz and per_group.csv")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max-points", type=int, default=2000, help="Max strip points per group")
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    scores_path = result_dir / "scores.npz"
    group_path = result_dir / "per_group.csv"

    scores = np.load(scores_path, allow_pickle=True)
    df_group = pd.read_csv(group_path)

    for k in ["pos_score", "neg_score", "real_score", "fake_score"]:
        if k not in scores.files:
            raise KeyError(f"Missing key in scores.npz: {k}")

    pos = np.asarray(scores["pos_score"], dtype=float)
    neg = np.asarray(scores["neg_score"], dtype=float)
    real = np.asarray(scores["real_score"], dtype=float)
    fake = np.asarray(scores["fake_score"], dtype=float)

    n_pos = int(np.isfinite(pos).sum())
    n_neg = int(np.isfinite(neg).sum())
    n_real = int(np.isfinite(real).sum())
    n_fake = int(np.isfinite(fake).sum())

    # Metrics (for annotation + json)
    y_posneg = np.r_[np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)]
    s_posneg = np.r_[pos, neg]
    posneg_auc = auc_rank(pos, neg)
    posneg_pr = pr_auc_np(y_posneg, s_posneg)

    y_realfake = np.r_[np.ones(len(real), dtype=int), np.zeros(len(fake), dtype=int)]
    s_realfake = np.r_[real, fake]
    realfake_auc = auc_rank(real, fake)
    # NOTE: PR for real/fake can be misleading visually; we still compute for json.
    realfake_pr = pr_auc_np(y_realfake, s_realfake)

    stats = {
        "pos_vs_neg": {
            "roc_auc": float(posneg_auc),
            "pr_auc": float(posneg_pr),
            "mean_pos": float(np.nanmean(pos)),
            "mean_neg": float(np.nanmean(neg)),
            "n_pos": n_pos,
            "n_neg": n_neg,
        },
        "real_vs_fake": {
            "roc_auc": float(realfake_auc),
            "pr_auc": float(realfake_pr),
            "mean_real": float(np.nanmean(real)),
            "mean_fake": float(np.nanmean(fake)),
            "n_real": n_real,
            "n_fake": n_fake,
        },
    }

    # -----------------------
    # Panel B: pos vs neg
    # -----------------------
    fig, ax = plt.subplots(figsize=(4.8, 4.1))
    box_with_strip(ax, [pos, neg], ["Positive", "Negative"], "B  Positive vs negative",
                   seed=args.seed, max_points=args.max_points)
    ax.text(
        0.02, 0.98,
        f"ROC-AUC={posneg_auc:.4f}\nPR-AUC={posneg_pr:.4f}\n"
        f"n+={n_pos}, n-={n_neg}",
        transform=ax.transAxes, ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, linewidth=0.6)
    )
    fig.tight_layout()
    save_fig(fig, outdir / "fig2b_pos_vs_neg", dpi=args.dpi)
    plt.close(fig)

    # -----------------------
    # Panel C: dataset-wise summary
    # -----------------------
    fig, ax = plt.subplots(figsize=(4.8, 4.1))
    plot_dataset_posneg(ax, df_group)
    fig.tight_layout()
    save_fig(fig, outdir / "fig2c_dataset_posneg", dpi=args.dpi)
    plt.close(fig)

    # -----------------------
    # Panel D: real vs generated
    # -----------------------
    fig, ax = plt.subplots(figsize=(4.8, 4.1))
    box_with_strip(ax, [real, fake], ["Real", "Generated"], "D  Real vs generated",
                   seed=args.seed + 10, max_points=args.max_points)
    # Display ROC only (PR-AUC omitted from panel to avoid over-interpretation)
    ax.text(
        0.02, 0.98,
        f"ROC-AUC={realfake_auc:.4f}\n"
        f"n_real={n_real}, n_fake={n_fake}",
        transform=ax.transAxes, ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, linewidth=0.6)
    )
    fig.tight_layout()
    save_fig(fig, outdir / "fig2d_real_vs_fake", dpi=args.dpi)
    plt.close(fig)

    # -----------------------
    # Panel E: score summary (pos/neg/fake)
    # -----------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    sum_stats = plot_score_summary(ax, pos, neg, fake, seed=args.seed + 20)
    stats["score_summary"] = sum_stats
    fig.tight_layout()
    save_fig(fig, outdir / "fig2e_score_summary", dpi=args.dpi)
    plt.close(fig)

    # -----------------------
    # Combined figure (B–E)
    # -----------------------
    fig = plt.figure(figsize=(12.8, 8.2))
    gs = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    box_with_strip(ax1, [pos, neg], ["Positive", "Negative"], "B  Positive vs negative",
                   seed=args.seed, max_points=min(args.max_points, 1800))
    ax1.text(
        0.02, 0.98,
        f"ROC-AUC={posneg_auc:.4f}\nPR-AUC={posneg_pr:.4f}\n"
        f"n+={n_pos}, n-={n_neg}",
        transform=ax1.transAxes, ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, linewidth=0.6)
    )

    ax2 = fig.add_subplot(gs[0, 1])
    plot_dataset_posneg(ax2, df_group)

    ax3 = fig.add_subplot(gs[1, 0])
    box_with_strip(ax3, [real, fake], ["Real", "Generated"], "D  Real vs generated",
                   seed=args.seed + 10, max_points=min(args.max_points, 1800))
    ax3.text(
        0.02, 0.98,
        f"ROC-AUC={realfake_auc:.4f}\n"
        f"n_real={n_real}, n_fake={n_fake}",
        transform=ax3.transAxes, ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, linewidth=0.6)
    )

    ax4 = fig.add_subplot(gs[1, 1])
    plot_score_summary(ax4, pos, neg, fake, seed=args.seed + 20)

    fig.text(0.01, 0.01, f"Input: {result_dir}", ha="left", va="bottom", fontsize=8)
    save_fig(fig, outdir / "figure2_revised_main", dpi=args.dpi)
    plt.close(fig)

    with open(outdir / "figure2_revised_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("[OK] wrote revised Figure 2 panels to:", outdir)


if __name__ == "__main__":
    main()