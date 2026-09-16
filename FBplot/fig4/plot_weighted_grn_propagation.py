#!/usr/bin/env python3
"""Nature-style figure for weighted GRN propagation consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", required=True)
    return p.parse_args()


def node(ax, x, y, text, face, edge):
    p = FancyBboxPatch((x - 0.10, y - 0.07), 0.20, 0.14,
                       boxstyle="round,pad=0.012,rounding_size=0.025",
                       facecolor=face, edgecolor=edge, linewidth=0.8)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=6.8)


def arrow(ax, start, end, color, label=""):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8,
                                 linewidth=0.9, color=color))
    if label:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.035,
                label, ha="center", fontsize=6.2, color=color)


def metric(block, condition, key):
    return block[condition]["spearman"][key]


def main():
    a = args()
    with open(a.input, encoding="utf-8") as f:
        d = json.load(f)
    analyses = d["analyses"]
    ks = np.array(sorted(int(k) for k in analyses))
    colors = {"real": "#3068A8", "rewired": "#7F7F7F", "weight": "#D47A27"}

    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5, "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(7.08, 2.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.28, 1.08, 1.06], wspace=0.52)

    # a. Mechanistic test
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(-0.08, 1.03, "a", weight="bold", fontsize=10, transform=ax.transAxes)
    ax.text(0.0, 0.98, "Cross-iteration GRN propagation", weight="bold", va="top", fontsize=8.2)
    node(ax, 0.12, 0.70, "Source $i$\n$|\\Delta X_i^t|$", "#E7EEF7", colors["real"])
    node(ax, 0.50, 0.70, "Target $j$\n$\\widehat{|\\Delta X_j^{t+1}|}$", "#F5EADC", colors["weight"])
    arrow(ax, (0.23, 0.70), (0.39, 0.70), colors["real"], "$W_{ij}$")
    node(ax, 0.88, 0.70, "Observed $j$\n$|\\Delta X_j^{t+1}|$", "#F1F1F1", colors["rewired"])
    arrow(ax, (0.61, 0.70), (0.77, 0.70), "#222222")
    ax.text(0.69, 0.61, "compare", ha="center", fontsize=6.0)
    ax.text(0.50, 0.43, "$\\widehat{|\\Delta X^{t+1}|}=W^\\top|\\Delta X^t|$",
            ha="center", fontsize=8.2)
    ax.text(0.50, 0.27, "Real attention GRN", ha="center", color=colors["real"], weight="bold")
    ax.text(0.50, 0.17, "vs degree-preserving rewiring", ha="center", color=colors["rewired"])
    ax.text(0.50, 0.08, "vs weight shuffling", ha="center", color=colors["weight"])

    # b. Density robustness in early cells
    ax = fig.add_subplot(gs[0, 1])
    ax.text(-0.19, 1.03, "b", weight="bold", fontsize=10, transform=ax.transAxes)
    real, rw_mean, rw_lo, rw_hi, ws_mean, ws_lo, ws_hi = ([] for _ in range(7))
    for k in ks:
        b = analyses[str(k)]["key_to_query_primary"]["early"]["windows"]["transient"]
        real.append(b["observed"]["spearman"])
        rw_mean.append(metric(b, "rewired", "null_mean")); rw_lo.append(metric(b, "rewired", "null_q05")); rw_hi.append(metric(b, "rewired", "null_q95"))
        ws_mean.append(metric(b, "weight_shuffled", "null_mean")); ws_lo.append(metric(b, "weight_shuffled", "null_q05")); ws_hi.append(metric(b, "weight_shuffled", "null_q95"))
    pos = np.arange(len(ks))
    ax.fill_between(pos, rw_lo, rw_hi, color=colors["rewired"], alpha=0.18, linewidth=0)
    ax.plot(pos, rw_mean, "--", color=colors["rewired"], lw=1, label="Rewired null")
    ax.fill_between(pos, ws_lo, ws_hi, color=colors["weight"], alpha=0.15, linewidth=0)
    ax.plot(pos, ws_mean, ":", color=colors["weight"], lw=1.2, label="Weight-shuffled")
    ax.plot(pos, real, "-o", color=colors["real"], lw=1.4, ms=4, label="Real GRN")
    ax.axhline(0, color="#B0B0B0", lw=0.7)
    ax.set_xticks(pos, [f"{k//1000}k" for k in ks])
    ax.set_xlabel("Strongest attention edges")
    ax.set_ylabel("Transient Spearman $\\rho$")
    ax.set_title("Early-state density sensitivity", loc="left", weight="bold", fontsize=8.2, pad=7)
    ax.legend(frameon=False, fontsize=6.1, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    # c. Stage specificity at top 10k
    ax = fig.add_subplot(gs[0, 2])
    ax.text(-0.20, 1.03, "c", weight="bold", fontsize=10, transform=ax.transAxes)
    stages = ["early", "middle", "late"]
    labels = ["Early", "Middle", "Late"]
    nulls, observed, pvals = [], [], []
    k = str(int(ks.max()))
    for stage in stages:
        b = analyses[k]["key_to_query_primary"][stage]["windows"]["transient"]
        nulls.append(np.asarray(b["rewired"]["spearman"]["null_values"], float))
        observed.append(b["observed"]["spearman"])
        pvals.append(b["rewired"]["spearman"]["empirical_p_greater"])
    vp = ax.violinplot(nulls, positions=np.arange(3), widths=0.72,
                       showmeans=False, showmedians=True, showextrema=False)
    for body in vp["bodies"]:
        body.set_facecolor("#B8B8B8"); body.set_edgecolor("none"); body.set_alpha(0.65)
    vp["cmedians"].set_color(colors["rewired"]); vp["cmedians"].set_linewidth(1)
    ax.scatter(np.arange(3), observed, marker="D", s=25, color=colors["real"], zorder=3,
               label="Real GRN")
    ax.axhline(0, color="#B0B0B0", lw=0.7)
    ax.set_xticks(np.arange(3), labels)
    ax.set_ylabel("Transient Spearman $\\rho$")
    ax.set_title("Stage specificity (top 10k)", loc="left", weight="bold", fontsize=8.2, pad=7)
    ax.set_ylim(-0.055, 0.125)
    for i, (o, p) in enumerate(zip(observed, pvals)):
        ax.text(i, min(0.112, max(o, np.quantile(nulls[i], .95)) + 0.010),
                f"$p$={p:.3f}", ha="center", fontsize=6.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.98, 0.98, "n=200 rewired networks\nmedian across first 8 lags",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.8, color="#666666")
    ax.text(0.98, 0.69, "Blue diamonds: real GRN", transform=ax.transAxes,
            ha="right", fontsize=5.8, color=colors["real"])

    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.20, top=0.87)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    stem = out / "weighted_grn_propagation_hESC"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
