#!/usr/bin/env python3
"""Create TF subsampling-stability and representative local-GRN figures."""

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch


def load_rows(path):
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)["tf_finite_difference_probe"]["rows"]
    return {row["tf"]: row for row in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True)
    parser.add_argument("--base-replicate", required=True)
    parser.add_argument("--replicate-glob", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--min-targets", type=int, default=20)
    parser.add_argument("--n-show", type=int, default=12)
    parser.add_argument("--network-tfs", nargs="+", default=["TFAP2A", "EOMES"])
    parser.add_argument("--network-targets", type=int, default=12)
    args = parser.parse_args()

    primary = load_rows(args.primary)
    eligible = [row for row in primary.values() if row["n_targets"] >= args.min_targets]
    shown_rows = sorted(
        eligible, key=lambda row: row["target_all_nontarget_ratio"], reverse=True
    )[:args.n_show]
    shown_tfs = [row["tf"] for row in shown_rows]

    replicate_paths = [args.base_replicate] + sorted(glob.glob(args.replicate_glob))
    replicate_rows = [load_rows(path) for path in replicate_paths]
    rng = np.random.default_rng(12)

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    for xi, tf in enumerate(shown_tfs):
        values = np.asarray([
            rows[tf]["target_all_nontarget_ratio"]
            for rows in replicate_rows if tf in rows
        ])
        jitter = rng.uniform(-0.13, 0.13, size=len(values))
        ax.scatter(xi + jitter, values, s=38, color="#4C83B6", alpha=0.78,
                   edgecolor="white", linewidth=0.5, zorder=3)
        ax.plot([xi - 0.20, xi + 0.20], [np.median(values)] * 2,
                color="#173F5F", lw=2.2, zorder=4)
    ax.axhline(1.0, color="#555555", ls="--", lw=1.2)
    ax.set_xticks(np.arange(len(shown_tfs)), shown_tfs, rotation=35, ha="right")
    ax.set_ylabel("Target / all non-target response ratio")
    ax.set_title("Stability across independent early-cell subsamples", loc="left", weight="bold")
    ax.text(0.995, 0.02, f"{len(replicate_rows)} subsamples; 16 cells per subsample",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#555555")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "tf_response_subsample_stability.png", dpi=300)
    fig.savefig(outdir / "tf_response_subsample_stability.pdf")
    plt.close(fig)

    selected = []
    for tf in args.network_tfs:
        row = primary[tf]
        genes = row.get("target_gene_responses", [])[:args.network_targets]
        if not genes:
            raise ValueError(f"No target_gene_responses stored for {tf}")
        selected.extend(item["response"] * 1e3 for item in genes)
    vmin, vmax = min(selected), max(selected)

    network_paths = []
    for tf in args.network_tfs:
        row = primary[tf]
        genes = row["target_gene_responses"][:args.network_targets]
        fig, ax = plt.subplots(figsize=(6.0, 5.4))
        angles = np.linspace(0, 2 * np.pi, len(genes), endpoint=False) + np.pi / 2
        radius = 1.0
        xy = np.c_[radius * np.cos(angles), radius * np.sin(angles)]
        for (tx, ty), item in zip(xy, genes):
            arrow = FancyArrowPatch(
                (0, 0), (tx * 0.88, ty * 0.88), arrowstyle="-|>",
                mutation_scale=10, lw=0.9, color="#9AA4AC", alpha=0.75,
                connectionstyle="arc3,rad=0.03", zorder=1,
            )
            ax.add_patch(arrow)
        responses = np.asarray([item["response"] * 1e3 for item in genes])
        scatter = ax.scatter(xy[:, 0], xy[:, 1], c=responses, cmap="YlOrRd",
                             vmin=vmin, vmax=vmax, s=420, edgecolor="white",
                             linewidth=1.2, zorder=3)
        ax.scatter([0], [0], s=1450, color="#285F8F", edgecolor="white",
                   linewidth=1.5, zorder=4)
        ax.text(0, 0, tf, color="white", weight="bold", ha="center", va="center",
                fontsize=9, zorder=5)
        for (tx, ty), item in zip(xy, genes):
            align = "left" if tx >= 0 else "right"
            ax.text(tx * 1.13, ty * 1.13, item["gene"], ha=align, va="center",
                    fontsize=8.2)
        ax.set_title(tf, weight="bold")
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.35, 1.35)
        ax.set_aspect("equal")
        ax.axis("off")
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Finite-difference response ($\\times 10^{-3}$)")
        fig.tight_layout()
        stem = f"representative_tf_local_network_{tf}"
        fig.savefig(outdir / f"{stem}.png", dpi=300)
        fig.savefig(outdir / f"{stem}.pdf")
        plt.close(fig)
        network_paths.append(outdir / f"{stem}.png")

    print(outdir / "tf_response_subsample_stability.png")
    for path in network_paths:
        print(path)


if __name__ == "__main__":
    main()
