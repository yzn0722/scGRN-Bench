#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot veloBench-style Figure 2 panels (CBDir / ICCoh / Velocity Consistency).

Designed to mirror the benchmark paper layout:
  A) CBDir raincloud  — one point per boundary cell score (or per edge if raw missing)
  B) ICCoh boxplot    — one point per cell ICCoh (or per cluster mean)
  C) VC violin        — per-cell velocity_confidence_cosine distribution

Example:
  conda activate dynamo-env
  cd /mnt/10T/yzn/scGRN-Bench

  python src/GRN_inferance/dyn/plot_benchmark_fig2_style.py \\
    --dirs outputs/dynamo_fm_velocity/benchmark_metrics/dynamo_hematopoiesis_local \\
           outputs/dynamo_fm_velocity/benchmark_metrics/scgpt_zero_shot_hematopoiesis_local \\
    --labels Dynamo scGPT_zero_shot \\
    --dataset-name hematopoiesis \\
    --outdir outputs/dynamo_fm_velocity/benchmark_metrics/comparison_hematopoiesis_local/figures \\
    --prefix fig2_style
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns


PALETTE = {
    "Dynamo": "#4daf4a",
    "scGPT_zero_shot": "#377eb8",
    "scGPT_neighbor_proj": "#377eb8",
    "scGPT_dynamo_proj": "#984ea3",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="veloBench Figure-2-style metric plots")
    p.add_argument("--dirs", nargs="+", required=True, help="Benchmark result directories")
    p.add_argument("--labels", nargs="+", required=True, help="Method display labels")
    p.add_argument("--dataset-name", default="hematopoiesis", type=str)
    p.add_argument("--outdir", required=True, type=str)
    p.add_argument("--prefix", default="fig2_style", type=str)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def load_method_tables(run_dir: Path, label: str, dataset_name: str) -> dict:
    run_dir = Path(run_dir)
    out = {"method": label, "dataset": dataset_name, "run_dir": str(run_dir)}

    summary_path = run_dir / "benchmark_metrics_summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            out["summary"] = json.load(f)

    cb_raw = run_dir / "CBDir_raw_per_boundary_cell.csv"
    if cb_raw.exists():
        cb = pd.read_csv(cb_raw)
        cb["method"] = label
        cb["dataset"] = dataset_name
        out["cbdir_points"] = cb
    else:
        cb = pd.read_csv(run_dir / "CBDir_scores.csv")
        cb = cb[cb["edge"] != "MEAN"].copy()
        cb = cb.rename(columns={"cbdir_mean": "cbdir"})
        cb["method"] = label
        cb["dataset"] = dataset_name
        out["cbdir_points"] = cb

    ic_raw = run_dir / "ICCoh_raw_per_cell.csv"
    if ic_raw.exists():
        ic = pd.read_csv(ic_raw)
        ic["method"] = label
        ic["dataset"] = dataset_name
        out["iccoh_points"] = ic
    else:
        ic = pd.read_csv(run_dir / "ICCoh_scores.csv")
        ic = ic[ic["cluster"] != "MEAN"].copy()
        ic = ic.rename(columns={"iccoh_mean": "iccoh"})
        ic["method"] = label
        ic["dataset"] = dataset_name
        out["iccoh_points"] = ic

    vc = pd.read_csv(run_dir / "velocity_confidence_per_cell.csv")
    vc["method"] = label
    vc["dataset"] = dataset_name
    out["vc_points"] = vc
    return out


def method_color(label: str) -> str:
    return PALETTE.get(label, sns.color_palette("Set2")[hash(label) % 8])


def plot_raincloud(ax: plt.Axes, df: pd.DataFrame, ycol: str, order: list) -> None:
    """Half-violin + jitter + mean line (raincloud-like, no ptitprince)."""
    width = 0.35
    for i, method in enumerate(order):
        sub = df[df["method"] == method][ycol].dropna().to_numpy()
        if len(sub) == 0:
            continue
        color = method_color(method)
        x_base = i

        parts = ax.violinplot(
            [sub],
            positions=[x_base + width / 2],
            widths=width,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.35)
            verts = body.get_paths()[0].vertices
            verts[:, 0] = np.clip(verts[:, 0], x_base + width / 2, x_base + width)

        jitter = np.random.default_rng(42).uniform(-width / 2, 0, size=len(sub))
        ax.scatter(
            x_base + jitter,
            sub,
            s=10,
            alpha=0.55,
            color=color,
            edgecolors="none",
            zorder=3,
        )
        ax.hlines(
            np.mean(sub),
            x_base - width / 2,
            x_base + width / 2,
            colors="black",
            linewidth=1.2,
            zorder=4,
        )

    ax.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel("Cross-boundary direction correctness")


def plot_iccoh_box(ax: plt.Axes, df: pd.DataFrame, order: list) -> None:
    colors = [method_color(m) for m in order]
    sns.boxplot(
        data=df,
        x="method",
        y="iccoh",
        order=order,
        palette=colors,
        width=0.55,
        fliersize=0,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="method",
        y="iccoh",
        order=order,
        color="0.25",
        size=2.5,
        alpha=0.25,
        jitter=0.25,
        ax=ax,
    )
    for i, method in enumerate(order):
        sub = df[df["method"] == method]["iccoh"].dropna()
        if len(sub):
            ax.scatter(i, sub.mean(), marker="D", s=45, color="gold", edgecolor="black", zorder=5)
    ax.set_ylabel("Intra-cluster coherence")
    ax.set_xlabel("")
    ax.set_xticklabels(order, rotation=35, ha="right")


def plot_vc_violin(ax: plt.Axes, df: pd.DataFrame, order: list) -> None:
    colors = [method_color(m) for m in order]
    sns.violinplot(
        data=df,
        x="method",
        y="velocity_confidence_cosine",
        order=order,
        palette=colors,
        cut=0,
        inner=None,
        linewidth=0.8,
        ax=ax,
    )
    sns.stripplot(
        data=df.sample(min(400, len(df)), random_state=42) if len(df) > 400 else df,
        x="method",
        y="velocity_confidence_cosine",
        order=order,
        color="0.2",
        size=1.5,
        alpha=0.15,
        jitter=0.3,
        ax=ax,
    )
    for i, method in enumerate(order):
        sub = df[df["method"] == method]["velocity_confidence_cosine"].dropna()
        if len(sub):
            ax.scatter(
                i,
                sub.mean(),
                marker="D",
                s=45,
                color="gold",
                edgecolor="black",
                zorder=5,
                label="dataset mean" if i == 0 else "",
            )
    ax.set_ylabel("Velocity consistency")
    ax.set_xlabel("")
    ax.set_xticklabels(order, rotation=35, ha="right")


def main() -> None:
    args = parse_args()
    if len(args.labels) != len(args.dirs):
        raise ValueError("--labels count must match --dirs count")

    runs = [
        load_method_tables(Path(d), lab, args.dataset_name)
        for d, lab in zip(args.dirs, args.labels)
    ]
    order = args.labels

    cb_all = pd.concat([r["cbdir_points"] for r in runs], ignore_index=True)
    ic_all = pd.concat([r["iccoh_points"] for r in runs], ignore_index=True)
    vc_all = pd.concat([r["vc_points"] for r in runs], ignore_index=True)

    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)

    plot_raincloud(axes[0], cb_all, "cbdir", order)
    axes[0].set_title("A  Cross-boundary direction correctness", loc="left", fontweight="bold")

    plot_iccoh_box(axes[1], ic_all, order)
    axes[1].set_title("B  Intra-cluster coherence", loc="left", fontweight="bold")

    plot_vc_violin(axes[2], vc_all, order)
    axes[2].set_title("C  Velocity consistency", loc="left", fontweight="bold")

    diamond = mpatches.Patch(facecolor="gold", edgecolor="black", label="mean (dataset / method)")
    axes[2].legend(handles=[diamond], loc="lower right", frameon=True, fontsize=8)

    fig.suptitle(
        f"RNA velocity metrics — {args.dataset_name} (veloBench-aligned)",
        fontsize=12,
        y=1.03,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"{args.prefix}.png"
    pdf = outdir / f"{args.prefix}.pdf"
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    cb_all.to_csv(outdir / f"{args.prefix}_cbdir_points.csv", index=False)
    ic_all.to_csv(outdir / f"{args.prefix}_iccoh_points.csv", index=False)
    vc_all.to_csv(outdir / f"{args.prefix}_vc_points.csv", index=False)
    print(f"Saved -> {png}")
    print(f"Saved -> {pdf}")


if __name__ == "__main__":
    main()
