#!/usr/bin/env python3
"""
Figure 3 (PANC-1 version): Site-specific palmitoylation remodeling
associated with ZDHHC20 overexpression in PANC-1 cells.

Panels:
  A. Top sites with largest |delta| (Ctrl vs OE)
  B. Positional enrichment heatmap (Ctrl - OE)
  C. Shannon entropy profile (top Ctrl-biased vs background)
  D. Cross-dataset remodeling magnitude (HeLa KO vs PANC-1 OE)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json, os

DATA = "results/extracted_v10"
OUT = "results/fig3_panc1"
os.makedirs(OUT, exist_ok=True)

# ── helpers ──────────────────────────────────────────────
def bootstrap_ci_mean(x, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    means = [np.mean(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)]
    return float(np.mean(x)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def short_window(seq, flank=10):
    if len(seq) < 21:
        return seq
    c = len(seq) // 2
    return seq[c - flank : c + flank + 1]

# ── load data ────────────────────────────────────────────
print("Loading data...")

# Panel A: top/bottom sites
top_df = pd.read_csv(f"{DATA}/fig3a_panc1_top_sites.csv")
bot_df = pd.read_csv(f"{DATA}/fig3a_panc1_bottom_sites.csv")

# Dedup by site_key
def dedup(df):
    df = df.copy()
    df["site_key"] = df["Accession"].astype(str).str.strip() + "|" + df["Position"].astype(str).str.strip()
    df["_abs"] = df["delta"].abs()
    df = df.sort_values(["site_key", "_abs"], ascending=[True, False])
    df = df.drop_duplicates("site_key", keep="first").drop(columns=["_abs"])
    return df

top_u = dedup(top_df).sort_values("delta", ascending=False)
bot_u = dedup(bot_df).sort_values("delta", ascending=True)

# Panel B: heatmap
hm = pd.read_csv(f"{DATA}/fig3b_panc1_heatmap_wt_minus_ko.csv")
hm_pos = hm["pos_rel_to_Cys"].values
hm_mat = hm.drop(columns=["pos_rel_to_Cys"]).values
aa_cols = list(hm.columns[1:])

# Panel C: entropy
ent_df = pd.read_csv(f"{DATA}/fig3c_panc1_entropy.csv")

# Panel D: cross-dataset
summary_csv = pd.read_csv(f"{DATA}/../fig3/figure3_main.summary.csv")
hela_all = pd.read_csv(f"{DATA}/fig3d_hela_all_unique_sites_delta.csv")
panc_all = pd.read_csv(f"{DATA}/fig3d_panc1_all_unique_sites_delta.csv")

# ── Panel A: bar plot of top sites ───────────────────────
def plot_panel_a(ax, top, bottom, n_each=6):
    top_show = top.head(n_each).copy()
    bot_show = bottom.head(n_each).copy()

    # Build labels
    def make_label(row):
        acc = str(row["Accession"])
        pos = str(row["Position"])
        if acc == "nan" or acc == "NA":
            sw = short_window(str(row["seq101"]))
            return f"...{sw}..."
        return f"{acc}\n{pos}"

    top_labels = [make_label(r) for _, r in top_show.iterrows()]
    bot_labels = [make_label(r) for _, r in bot_show.iterrows()]

    labels = top_labels + bot_labels
    deltas = list(top_show["delta"]) + list(bot_show["delta"])
    colors = ["#d62728" if d > 0 else "#1f77b4" for d in deltas]

    y = np.arange(len(labels))
    ax.barh(y, deltas, color=colors, edgecolor="none", height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Δ score (Ctrl − OE)", fontsize=9)
    ax.set_title("PANC-1: top sites with largest\ncontext-dependent shifts", fontsize=10)
    ax.axvline(0, color="black", lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate max/min
    ax.text(0.98, 0.02, f"max Δ = {top_show['delta'].iloc[0]:.3f}\nmin Δ = {bot_show['delta'].iloc[0]:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.9))

# ── Panel B: heatmap ─────────────────────────────────────
def plot_panel_b(ax):
    vmax = np.max(np.abs(hm_mat)) * 0.8
    im = ax.imshow(hm_mat.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(np.arange(len(hm_pos)))
    ax.set_xticklabels(hm_pos.astype(int), fontsize=6, rotation=0)
    ax.set_yticks(np.arange(len(aa_cols)))
    ax.set_yticklabels(aa_cols, fontsize=7)
    ax.set_xlabel("Position relative to Cys", fontsize=9)
    ax.set_ylabel("Amino acid", fontsize=9)
    ax.set_title("PANC-1: positional enrichment\n(Ctrl − OE)", fontsize=10)

    # Mark center cysteine
    center_idx = list(hm_pos).index(0) if 0 in hm_pos else len(hm_pos) // 2
    ax.axvline(center_idx, color="black", lw=1, ls="--", alpha=0.5)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Δ enrichment", fontsize=8)

# ── Panel C: entropy ─────────────────────────────────────
def plot_panel_c(ax):
    # Compute Shannon entropy for top Ctrl-biased vs background
    # Using the kmer data: group by position relative to center
    # The kmer column has 3-mers centered at the cysteine region

    # For the entropy panel, we compute from the top/bottom site sequences
    flank = 10
    win = 2 * flank + 1

    # Top Ctrl-biased sites (positive delta)
    top_seqs = [short_window(str(s), flank) for s in top_u.head(500)["seq101"] if len(str(s)) >= win]
    # Background (weak shift)
    merged = pd.concat([top_u, bot_u], ignore_index=True)
    merged["_abs"] = merged["delta"].abs()
    bg_sites = merged.sort_values("_abs").head(500)
    bg_seqs = [short_window(str(s), flank) for s in bg_sites["seq101"] if len(str(s)) >= win]

    def positional_entropy(seqs, win_len):
        H = np.zeros(win_len)
        for pos in range(win_len):
            counts = {}
            for s in seqs:
                if pos < len(s):
                    aa = s[pos]
                    counts[aa] = counts.get(aa, 0) + 1
            total = sum(counts.values())
            if total == 0:
                H[pos] = 0
                continue
            e = 0
            for c in counts.values():
                p = c / total
                if p > 0:
                    e -= p * np.log2(p)
            H[pos] = e
        return H

    H_top = positional_entropy(top_seqs, win)
    H_bg = positional_entropy(bg_seqs, win)
    rel_pos = np.arange(-flank, flank + 1)

    ax.plot(rel_pos, H_top, "o-", color="#d62728", ms=4, lw=1.5, label=f"Top Ctrl-biased (n={len(top_seqs)})")
    ax.plot(rel_pos, H_bg, "s-", color="#7f7f7f", ms=4, lw=1.5, label=f"Background (n={len(bg_seqs)})")
    ax.set_xlabel("Position relative to Cys", fontsize=9)
    ax.set_ylabel("Shannon entropy (bits)", fontsize=9)
    ax.set_title("PANC-1: entropy profile\n(Ctrl-biased vs background)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(0, color="black", lw=0.5, ls=":", alpha=0.5)

    delta_H = H_bg - H_top
    strong = np.argsort(-np.abs(delta_H))[:3]
    txt = ", ".join([f"{rel_pos[i]}:{delta_H[i]:+.2f}" for i in strong])
    ax.text(0.02, 0.02, f"Top Δentropy: {txt}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.9))

# ── Panel D: cross-dataset comparison ────────────────────
def plot_panel_d(ax):
    hela_abs = hela_all["delta"].abs().values
    panc_abs = panc_all["delta"].abs().values

    hela_mean, hela_lo, hela_hi = bootstrap_ci_mean(hela_abs)
    panc_mean, panc_lo, panc_hi = bootstrap_ci_mean(panc_abs)

    x = [0, 1]
    means = [hela_mean, panc_mean]
    los = [hela_lo, panc_lo]
    his = [hela_hi, panc_hi]
    colors = ["#ff7f0e", "#2ca02c"]
    labels_x = ["HeLa\n(ZDHHC20 KO)", "PANC-1\n(ZDHHC20 OE)"]

    bars = ax.bar(x, means, color=colors, edgecolor="none", width=0.5, alpha=0.85)
    ax.errorbar(x, means, yerr=[[m - l for m, l in zip(means, los)],
                                 [h - m for m, h in zip(means, his)]],
                fmt="none", ecolor="black", capsize=5, lw=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, fontsize=9)
    ax.set_ylabel("Mean |Δ score|", fontsize=9)
    ax.set_title("Cross-dataset remodeling\nmagnitude", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, (m, n) in enumerate(zip(means, [len(hela_abs), len(panc_abs)])):
        ax.text(i, m + (his[i] - m) + 0.005,
                f"n={n}\nmean={m:.3f}\nmedian={np.median([hela_abs, panc_abs][i]):.3f}",
                ha="center", va="bottom", fontsize=7)

# ── Assemble figure ──────────────────────────────────────
print("Plotting...")
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])

plot_panel_a(axA, top_u, bot_u, n_each=6)
plot_panel_b(axB)
plot_panel_c(axC)
plot_panel_d(axD)

# Panel labels
for ax, letter in zip([axA, axB, axC, axD], "ABCD"):
    ax.text(-0.08, 1.08, letter, transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top", ha="right")

fig.suptitle("Figure 3 (PANC-1): Site-specific palmitoylation remodeling\nassociated with ZDHHC20 overexpression",
             fontsize=14, fontweight="bold", y=0.98)

out_png = f"{OUT}/figure3_panc1.png"
out_pdf = f"{OUT}/figure3_panc1.pdf"
fig.savefig(out_png, dpi=200, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close()

print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")

# Also save individual panels
for panel_name, plot_func in [("A", lambda: plot_panel_a(plt.gca(), top_u, bot_u, 6)),
                                ("B", lambda: plot_panel_b(plt.gca())),
                                ("C", lambda: plot_panel_c(plt.gca())),
                                ("D", lambda: plot_panel_d(plt.gca()))]:
    fig_s, ax_s = plt.subplots(figsize=(8, 6))
    if panel_name == "A":
        plot_panel_a(ax_s, top_u, bot_u, 6)
    elif panel_name == "B":
        plot_panel_b(ax_s)
    elif panel_name == "C":
        plot_panel_c(ax_s)
    elif panel_name == "D":
        plot_panel_d(ax_s)
    fig_s.tight_layout()
    fig_s.savefig(f"{OUT}/fig3{panel_name}_panc1.png", dpi=200, bbox_inches="tight")
    fig_s.savefig(f"{OUT}/fig3{panel_name}_panc1.pdf", bbox_inches="tight")
    plt.close(fig_s)
    print(f"Saved: fig3{panel_name}_panc1.png/pdf")

print("\nDone!")
