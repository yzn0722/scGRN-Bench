#!/usr/bin/env python3
"""Plot observed GRN weights against weight-shuffled controls."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


KS = [2000, 4000, 6000, 8000, 10000]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attention", required=True)
    p.add_argument("--embedding", required=True)
    p.add_argument("--hidden", required=True)
    p.add_argument("--outdir", required=True)
    return p.parse_args()


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def transient(data, k):
    return data["analyses"][str(k)]["key_to_query_primary"]["early"]["windows"]["transient"]


def main():
    a = parse_args()
    panels = [
        ("attn", "#4EA3F1", load(a.attention)),
        (r"COS$_{tok}$", "#FF9A3D", load(a.embedding)),
        (r"COS$_{hid}$", "#AC99D2", load(a.hidden)),
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.linewidth": 1.0, "pdf.fonttype": 42,
                         "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.45), sharex=True, sharey=True)
    x = np.arange(len(KS))
    summary = {}
    for ax, (label, color, data) in zip(axes, panels):
        observed = []
        means, q05, q95 = [], [], []
        for k in KS:
            d = transient(data, k)
            observed.append(d["observed"]["spearman"])
            null = np.asarray(d["weight_shuffled"]["spearman"]["null_values"], float)
            means.append(null.mean())
            q05.append(np.quantile(null, .05))
            q95.append(np.quantile(null, .95))
        summary[label] = {"observed": observed, "shuffled_mean": means,
                          "shuffled_q05": q05, "shuffled_q95": q95}
        ax.fill_between(x, q05, q95, color="#BDBDBD", alpha=.35, linewidth=0,
                        label="Shuffled-weight 5th-95th percentile")
        ax.plot(x, means, "--o", color="#777777", lw=1.6, ms=4,
                label="Shuffled-weight mean")
        ax.plot(x, observed, "-o", color=color, lw=2.4, ms=7, alpha=.9,
                label="Observed weights")
        ax.axhline(0, color="#A5A5A5", lw=.8, zorder=0)
        ax.set_xticks(x, ["2k", "4k", "6k", "8k", "10k"])
        ax.set_xlabel("Top-ranked GRN edges")
        ax.text(.5, .96, label, transform=ax.transAxes, ha="center", va="top",
                fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0)
        ax.grid(False)
    axes[0].set_ylabel(r"Transient Spearman $\rho$")
    axes[0].set_ylim(-.08, .28)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [2, 1, 0]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               frameon=False, ncol=3, loc="upper center",
               bbox_to_anchor=(.5, 1.02), fontsize=9,
               handlelength=2.1, columnspacing=1.3)
    fig.subplots_adjust(left=.09, right=.985, bottom=.18, top=.79, wspace=.18)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    stem = out / "grn_three_network_weight_shuffle_control"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
