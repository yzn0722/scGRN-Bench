#!/usr/bin/env python3
"""Publication-style summary of TFAP2A/NANOG GRN-target response enrichment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", required=True)
    return p.parse_args()


def box(ax, xy, wh, text, fc, ec, fontsize=8):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=fc, edgecolor=ec, linewidth=0.8,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, p1, p2, color):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=9,
                                 linewidth=0.9, color=color))


def main():
    args = parse_args()
    with open(args.input, encoding="utf-8") as f:
        report = json.load(f)
    rows = report["tf_finite_difference_probe"]["rows"]
    selected = {r["tf"]: r for r in rows if r["tf"] in {"TFAP2A", "NANOG"}}
    if set(selected) != {"TFAP2A", "NANOG"}:
        raise ValueError("Input must contain TFAP2A and NANOG probe results")
    genes = ["TFAP2A", "NANOG"]
    target = np.array([selected[g]["mean_target_response"] for g in genes]) * 1e3
    control = np.array([selected[g]["mean_control_response"] for g in genes]) * 1e3
    ratio = np.array([selected[g]["target_control_ratio"] for g in genes])
    ntarget = [selected[g]["n_targets"] for g in genes]

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    dark = "#222222"
    blue = "#3B6FB6"
    orange = "#D97928"
    pale_blue = "#E8EFF8"
    pale_orange = "#F8EBDD"
    grey = "#7A7A7A"
    light = "#F3F3F3"

    fig = plt.figure(figsize=(7.08, 2.55))  # 180 mm, Nature double-column width
    gs = fig.add_gridspec(1, 3, width_ratios=[1.42, 1.0, 0.88], wspace=0.58)

    # a, experimental logic
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(-0.08, 1.03, "a", weight="bold", fontsize=10, transform=ax.transAxes)
    ax.text(0.0, 0.98, "GRN-guided perturbation test", weight="bold", va="top")
    box(ax, (0.01, 0.62), (0.22, 0.17), "TF input\n$-\\delta$ / $+\\delta$", pale_orange, orange, 6.8)
    box(ax, (0.37, 0.62), (0.25, 0.17), "Iterative model\n$X_{t+1}=f(X_t)$", light, grey, 6.8)
    box(ax, (0.76, 0.62), (0.23, 0.17), "Gene-wise\nresponse $R_j$", pale_blue, blue, 6.8)
    arrow(ax, (0.24, 0.705), (0.36, 0.705), dark)
    arrow(ax, (0.63, 0.705), (0.75, 0.705), dark)
    ax.text(0.50, 0.49, "Does the response follow GRN edges?", ha="center", fontsize=7.2)
    box(ax, (0.04, 0.20), (0.34, 0.16), "GRN targets\nTF $\\rightarrow$ target", pale_blue, blue, 6.8)
    box(ax, (0.62, 0.20), (0.34, 0.16), "Expression-matched\nnon-targets", light, grey, 6.4)
    ax.text(0.50, 0.29, "vs", ha="center", va="center", weight="bold")
    ax.text(0.50, 0.08, "$R_j=|X_j^{(+\\delta)}-X_j^{(-\\delta)}|/(2\\delta)$",
            ha="center", fontsize=7.5)

    # b, response magnitude
    ax = fig.add_subplot(gs[0, 1])
    ax.text(-0.18, 1.03, "b", weight="bold", fontsize=10, transform=ax.transAxes)
    x = np.arange(len(genes))
    width = 0.34
    ax.bar(x - width / 2, target, width, color=blue, label="GRN targets")
    ax.bar(x + width / 2, control, width, color="#B8B8B8", label="Matched non-targets")
    ax.set_xticks(x, genes)
    ax.set_ylabel("Mean finite-difference\nresponse ($\\times10^{-3}$)")
    ax.set_title("Target response magnitude", loc="left", fontsize=8, weight="bold", pad=8)
    ax.set_ylim(0, max(target.max(), control.max()) * 1.30)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0)
    for xi, yi, n in zip(x, target, ntarget):
        ax.text(xi - width / 2, yi - 0.025, f"n={n:,}", ha="center", va="top",
                fontsize=6.3, color="white", weight="bold")

    # c, enrichment ratio
    ax = fig.add_subplot(gs[0, 2])
    ax.text(-0.20, 1.03, "c", weight="bold", fontsize=10, transform=ax.transAxes)
    bars = ax.bar(x, ratio, width=0.58, color=[orange, orange])
    ax.axhline(1.0, color=grey, linestyle="--", linewidth=0.9)
    ax.set_xticks(x, genes)
    ax.set_ylabel("Target / matched\nresponse ratio")
    ax.set_title("GRN-target enrichment", loc="left", fontsize=8, weight="bold", pad=8)
    ax.set_ylim(0.88, 1.18)
    for b, r in zip(bars, ratio):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.009, f"+{(r - 1) * 100:.1f}%",
                ha="center", va="bottom", fontsize=7.5, weight="bold")
    ax.text(0.98, 0.98, "Descriptive only (no CI)", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.2, color=grey)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0)

    fig.subplots_adjust(left=0.045, right=0.99, bottom=0.18, top=0.87)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / "tfap2a_nanog_grn_response"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
