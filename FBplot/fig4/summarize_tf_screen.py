#!/usr/bin/env python3
"""Summarize systematic TF perturbation screens without post-hoc cherry-picking."""

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def bh_fdr(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    q = np.ones(len(p))
    running = 1.0
    for rank0 in range(len(p) - 1, -1, -1):
        idx = order[rank0]
        running = min(running, p[idx] * len(p) / (rank0 + 1))
        q[idx] = running
    return q


def load_rows(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["tf_finite_difference_probe"]["rows"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True, help="Primary (larger-cell) JSON")
    parser.add_argument("--replicate-glob", required=True, help="Glob covering replicate JSON files")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--min-targets", type=int, default=20)
    args = parser.parse_args()

    primary = load_rows(args.primary)
    replicate_paths = sorted(glob.glob(args.replicate_glob))
    replicates = defaultdict(list)
    for path in replicate_paths:
        for row in load_rows(path):
            replicates[row["tf"]].append(row)

    qvalues = bh_fdr([row["random_label_empirical_p"] for row in primary])
    combined = []
    for row, qvalue in zip(primary, qvalues):
        reps = replicates.get(row["tf"], [])
        rep_ratios = [x["target_all_nontarget_ratio"] for x in reps]
        item = dict(row)
        item.update(
            bh_fdr=float(qvalue),
            replicate_count=len(reps),
            replicate_mean_ratio=float(np.mean(rep_ratios)) if rep_ratios else math.nan,
            replicate_sd_ratio=float(np.std(rep_ratios)) if rep_ratios else math.nan,
            replicate_min_ratio=float(np.min(rep_ratios)) if rep_ratios else math.nan,
            replicate_positive_count=int(sum(x > 1 for x in rep_ratios)),
            passes_target_count=row["n_targets"] >= args.min_targets,
            passes_fdr=float(qvalue) < 0.05,
        )
        combined.append(item)

    combined.sort(key=lambda x: (x["bh_fdr"], x["random_label_empirical_p"]))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "tf_screen_ranked.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)

    eligible = [x for x in combined if x["passes_target_count"]]
    shown = sorted(eligible, key=lambda x: x["target_all_nontarget_ratio"], reverse=True)[:12]
    shown = list(reversed(shown))
    y = np.arange(len(shown))
    ratios = np.array([x["target_all_nontarget_ratio"] for x in shown])
    colors = ["#3B78A8"] * len(shown)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.axvline(1.0, color="#444444", lw=1, ls="--")
    ax.barh(y, ratios - 1.0, left=1.0, color=colors, height=0.66)
    ax.set_yticks(y, [f"{x['tf']}  (n={x['n_targets']})" for x in shown])
    ax.set_xlabel("Target / non-target mean response ratio")
    ax.set_title("Systematic TF perturbation screen (64 cells)", loc="left", weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    for yi, x in zip(y, shown):
        ax.text(max(1.01, x["target_all_nontarget_ratio"] + 0.01), yi,
                f"{x['target_all_nontarget_ratio']:.3f}x",
                va="center", fontsize=7.5)
    ax.set_xlim(0.95, max(ratios) + 0.42)
    fig.tight_layout()
    fig.savefig(outdir / "tf_screen_ranked.png", dpi=300)
    fig.savefig(outdir / "tf_screen_ranked.pdf")
    plt.close(fig)

    # Paired target-versus-all-non-target view for the same TFs.
    paired = list(reversed(shown))
    x = np.arange(len(paired))
    width = 0.38
    target = np.array([row["mean_target_response"] for row in paired]) * 1e3
    control = np.array([row["mean_all_nontarget_response"] for row in paired]) * 1e3
    fig, ax = plt.subplots(figsize=(11.8, 5.2))
    ax.bar(x - width / 2, target, width, color="#3F76B7", label="GRN targets")
    ax.bar(x + width / 2, control, width, color="#B8B8B8",
           label="All non-targets")
    ax.set_xticks(x, [row["tf"] for row in paired], rotation=35, ha="right")
    ax.set_ylabel("Mean finite-difference response ($\\times 10^{-3}$)")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ymax = max(float(target.max()), float(control.max()))
    ax.set_ylim(0, ymax * 1.25)
    for xi, row, tv, cv in zip(x, paired, target, control):
        ratio = row["target_all_nontarget_ratio"]
        ax.text(xi, max(tv, cv) + ymax * 0.035,
                f"n={row['n_targets']}\n{ratio:.3f}x",
                ha="center", va="bottom", fontsize=7.5, linespacing=0.9)
    fig.tight_layout()
    fig.savefig(outdir / "tf_target_vs_all_nontargets.png", dpi=300)
    fig.savefig(outdir / "tf_target_vs_all_nontargets.pdf")
    plt.close(fig)

    summary = {
        "primary": args.primary,
        "replicates": replicate_paths,
        "n_tested": len(combined),
        "n_bh_fdr_lt_0_05": sum(x["passes_fdr"] for x in combined),
        "min_targets_for_display": args.min_targets,
        "top_by_adjusted_p": combined[:10],
        "interpretation": "Exploratory candidates only when BH-FDR >= 0.05.",
    }
    with open(outdir / "tf_screen_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(csv_path)
    print(outdir / "tf_screen_ranked.png")
    print(outdir / "tf_target_vs_all_nontargets.png")
    print(f"BH-FDR < 0.05: {summary['n_bh_fdr_lt_0_05']}/{summary['n_tested']}")


if __name__ == "__main__":
    main()
