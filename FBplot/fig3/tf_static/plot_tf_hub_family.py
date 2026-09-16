#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TF hub 分层 + TF family 聚合分析（fig3/tf_static）。

1. Hub vs specialist：按 GT 出度 top20%/bottom80% 与五分位，云雨图比较 Jaccard / F1 / FP rate
2. TF family：family × model 热图；FP 靶基因 family 级 ORA

示例:
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig3
  python tf_static/plot_tf_hub_family.py --dataset hESC
  python tf_static/plot_tf_hub_family.py --dataset hESC --hub-only
  python tf_static/plot_tf_hub_family.py --dataset hESC --family-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter

FIG3 = Path(__file__).resolve().parents[1]
if str(FIG3) not in sys.path:
    sys.path.insert(0, str(FIG3))

from fig3_palette import method_color, method_label, model_color  # noqa: E402

from tf_static.collect_per_tf_metrics import assign_hub_strata, collect_per_tf_long  # noqa: E402
from tf_static.model_registry import EVL_ROOT, EXTRACTIONS, GT_DISPLAY, GT_SOURCES, INPUT_ROOT, MODELS  # noqa: E402
from tf_static.plot_tf_jaccard_raincloud_all_models import (  # noqa: E402
    EXTRACT_COLORS,
    EXTRACT_DISPLAY,
    FIG_H,
    FIG_W,
    LABEL_COLOR,
    TEXT_SIZE,
    TITLE_SIZE,
)
from tf_static.tf_family import annotate_tf_families  # noqa: E402
from tf_static.utils import classify_tf_edges, filter_prediction, load_gt_network, read_gmt, run_ora, shorten_term  # noqa: E402
from tf_static.model_registry import resolve_gt_path, resolve_pred_path  # noqa: E402

GMT_HUMAN = [
    Path("/mnt/10T/yzn/benchmark_GRN/fuji/c2.cp.v2025.1.Hs.symbols.gmt"),
    Path("/mnt/10T/yzn/benchmark_GRN/fuji/h.all.v2025.1.Hs.symbols.gmt"),
]

RNG = np.random.default_rng(42)
METRICS = [
    ("jaccard", "Per-TF Jaccard"),
    ("f1", "Per-TF F1"),
    ("fp_rate", "FP rate (1 − precision)"),
]
HUB_TIER_ORDER = ["Specialist (bottom 80%)", "Hub (top 20%)"]
HUB_SUMMARY_FIG = (8.0, 5.0)
HUB_JACCARD_YLIM = (0.0, 0.3)
HUB_LOLLIPOP_YLIM = (0.0, 0.2)


def apply_plot_style() -> None:
    sns.set_theme(style="white")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": TEXT_SIZE,
            "axes.labelsize": TEXT_SIZE,
            "xtick.labelsize": TEXT_SIZE,
            "ytick.labelsize": TEXT_SIZE,
            "legend.fontsize": TEXT_SIZE,
        }
    )


def _raincloud_at(
    ax: plt.Axes,
    values: np.ndarray,
    x: float,
    color: str,
    width: float = 0.20,
) -> None:
    if len(values) == 0:
        return
    parts = ax.violinplot(
        values,
        positions=[x],
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_edgecolor("#555555")
        pc.set_alpha(0.40)
    jitter = RNG.normal(x, width * 0.18, size=len(values))
    ax.scatter(jitter, values, s=8, alpha=0.28, color=color, edgecolors="none", rasterized=True)
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    ax.vlines(x, q1, q3, color="#333333", lw=1.5, zorder=4)
    ax.scatter([x], [med], s=24, color="white", edgecolors="#222", linewidths=0.7, zorder=5)


def plot_hub_strat_raincloud(
    df: pd.DataFrame,
    dataset: str,
    gt_source: str,
    out_dir: Path,
    stratum_col: str,
    focus_model: Optional[str],
    fig_w: float,
    fig_h: float,
) -> None:
    """3 rows (metrics) × 1 col strata; within each stratum, 6 models × 3 extractions."""
    strata = list(dict.fromkeys(df[stratum_col].dropna().astype(str)))
    if not strata:
        return

    models = list(dict.fromkeys(df["model"].tolist()))
    extractions = list(EXTRACTIONS)

    fig, axes = plt.subplots(len(METRICS), 1, figsize=(fig_w, fig_h * len(METRICS) * 0.55), sharex=True)
    if len(METRICS) == 1:
        axes = [axes]

    group_w = len(models) * 1.05
    for ax, (mcol, mlabel) in zip(axes, METRICS):
        for si, st in enumerate(strata):
            base = si * group_w
            for mi, model in enumerate(models):
                for ext in extractions:
                    sub = df[(df[stratum_col] == st) & (df["model"] == model) & (df["extraction"] == ext)]
                    if focus_model and model != focus_model:
                        continue
                    vals = sub[mcol].dropna().to_numpy()
                    xp = base + mi + {"emb500": -0.26, "att500": 0.0, "embhidden500": 0.26}[ext]
                    _raincloud_at(ax, vals, xp, EXTRACT_COLORS[ext])

            if si < len(strata) - 1:
                ax.axvline(base + len(models) - 0.5 + 0.55, color="#E8E8E8", lw=0.8)

        ax.set_ylabel(mlabel, fontsize=TEXT_SIZE, color=LABEL_COLOR)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_ylim(-0.02, 1.05 if mcol != "fp_rate" else 1.05)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # x ticks: model names at each stratum block center
    tick_x, tick_lbl = [], []
    for si, st in enumerate(strata):
        base = si * group_w
        for mi, model in enumerate(models):
            if focus_model and model != focus_model:
                continue
            tick_x.append(base + mi)
            tick_lbl.append(model if si == 0 else "")
    axes[-1].set_xticks(tick_x)
    axes[-1].set_xticklabels(tick_lbl, rotation=45, ha="right", fontsize=TEXT_SIZE - 1, color=LABEL_COLOR)

    # stratum labels on top
    for si, st in enumerate(strata):
        cx = si * group_w + (len(models) - 1) / 2
        axes[0].text(
            cx,
            1.06,
            st,
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            fontsize=TEXT_SIZE,
            fontweight="bold",
            color=LABEL_COLOR,
        )

    leg = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=EXTRACT_COLORS[e], markersize=8, label=EXTRACT_DISPLAY[e])
        for e in extractions
    ]
    axes[0].legend(handles=leg, loc="upper right", frameon=False, fontsize=TEXT_SIZE - 1, title="Extraction")
    gt_name = GT_DISPLAY.get(gt_source, gt_source)
    fig.suptitle(
        f"{dataset} — TF performance by GT out-degree ({stratum_col})\n"
        f"Gold standard: {gt_name}",
        fontsize=TITLE_SIZE,
        y=1.01,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.18, hspace=0.28)
    tag = f"{dataset}_gt-{gt_source}_hub_{stratum_col}"
    if focus_model:
        tag += f"_{focus_model}"
    stem = out_dir / tag
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {stem}.png/.pdf")


def summarize_hub_tier(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Per hub_tier × model × extraction: mean (+ sem for jaccard)."""
    sub = df[df["hub_tier"].notna()].copy()
    summ = (
        sub.groupby(["hub_tier", "model", "extraction"], as_index=False)
        .agg(
            jaccard=("jaccard", "mean"),
            f1=("f1", "mean"),
            fp_rate=("fp_rate", "mean"),
            jaccard_sem=("jaccard", "sem"),
        )
    )
    models = list(dict.fromkeys(sub["model"].tolist()))
    return summ, models


def _hub_mean(summ: pd.DataFrame, ht: str, model: str, ext: str, col: str) -> float:
    row = summ[(summ["hub_tier"] == ht) & (summ["model"] == model) & (summ["extraction"] == ext)]
    return float(row[col].iloc[0]) if len(row) else 0.0


def _hub_sem(summ: pd.DataFrame, ht: str, model: str, ext: str) -> float:
    row = summ[(summ["hub_tier"] == ht) & (summ["model"] == model) & (summ["extraction"] == ext)]
    if len(row) and pd.notna(row["jaccard_sem"].iloc[0]):
        return float(row["jaccard_sem"].iloc[0])
    return 0.0


def _hub_bar_offsets() -> tuple[float, dict[str, float]]:
    w = 0.12
    return w, {"emb500": -w, "att500": 0.0, "embhidden500": w}


def _style_hub_ax(
    ax,
    ylab: str,
    col: str,
    models: list[str],
    x: np.ndarray,
    ylim: Optional[tuple[float, float]] = None,
) -> None:
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=TEXT_SIZE - 1, color=LABEL_COLOR)
    ax.set_ylabel(ylab, fontsize=TEXT_SIZE, color=LABEL_COLOR)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        y_top = HUB_JACCARD_YLIM[1] if col == "jaccard" else 1.05
        ax.set_ylim(0, y_top)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _save_hub_fig(fig, stem: Path) -> None:
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {stem}.png/.pdf")


def plot_hub_summary_bar(
    summ: pd.DataFrame,
    models: list[str],
    dataset: str,
    gt_source: str,
    out_dir: Path,
    metric_cols: list[str],
) -> None:
    ylab_map = {m[0]: m[1] for m in METRICS}
    n_panels = len(metric_cols)
    fig_w = HUB_SUMMARY_FIG[0] if n_panels == 1 else 4.0 * n_panels
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, HUB_SUMMARY_FIG[1]), squeeze=False)
    axes = axes.ravel()
    w, offsets = _hub_bar_offsets()
    x = np.arange(len(models))

    for ax, col in zip(axes, metric_cols):
        ylab = ylab_map.get(col, col)
        for hi, ht in enumerate(HUB_TIER_ORDER):
            for ext in EXTRACTIONS:
                vals = [_hub_mean(summ, ht, m, ext, col) for m in models]
                off = (hi - 0.5) * 0.36 + offsets[ext]
                ax.bar(
                    x + off,
                    vals,
                    width=w * 0.9,
                    color=EXTRACT_COLORS[ext],
                    alpha=0.55 if hi == 0 else 0.95,
                    edgecolor="#444",
                    linewidth=0.4,
                    label=f"{ht.split()[0]} · {EXTRACT_DISPLAY[ext]}" if col == metric_cols[0] else None,
                )
        _style_hub_ax(ax, ylab, col, models, x)

    axes[0].legend(loc="upper right", fontsize=9, frameon=False, ncol=2)
    gt_name = GT_DISPLAY.get(gt_source, gt_source)
    fig.suptitle(f"{dataset} — Hub vs specialist (mean per-TF, {gt_name})", fontsize=TITLE_SIZE)
    fig.tight_layout()
    tag = "hub_binary_bar" if metric_cols == ["jaccard"] else f"hub_binary_bar_{'_'.join(metric_cols)}"
    _save_hub_fig(fig, out_dir / f"{dataset}_gt-{gt_source}_{tag}")


def plot_hub_summary_slope(
    summ: pd.DataFrame,
    models: list[str],
    dataset: str,
    gt_source: str,
    out_dir: Path,
    col: str = "jaccard",
) -> None:
    """Specialist → Hub slope per model × extraction (same means as bar chart)."""
    ylab_map = {m[0]: m[1] for m in METRICS}
    fig, ax = plt.subplots(figsize=HUB_SUMMARY_FIG)
    x = np.arange(len(models))
    dx = 0.18
    for ext in EXTRACTIONS:
        for mi, model in enumerate(models):
            y0 = _hub_mean(summ, HUB_TIER_ORDER[0], model, ext, col)
            y1 = _hub_mean(summ, HUB_TIER_ORDER[1], model, ext, col)
            ax.plot(
                [x[mi] - dx, x[mi] + dx],
                [y0, y1],
                color=EXTRACT_COLORS[ext],
                lw=2.0,
                alpha=0.85,
                zorder=2,
            )
            ax.scatter(
                [x[mi] - dx, x[mi] + dx],
                [y0, y1],
                s=52,
                color=EXTRACT_COLORS[ext],
                edgecolors="#333",
                linewidths=0.5,
                zorder=3,
                label=EXTRACT_DISPLAY[ext] if mi == 0 else None,
            )
    ax.axhline(0, color="#CCCCCC", lw=0.6, zorder=0)
    _style_hub_ax(ax, ylab_map.get(col, col), col, models, x)
    ax.text(
        x[0] - dx,
        -0.028,
        "Specialist",
        ha="center",
        va="top",
        fontsize=9,
        color="#666",
        transform=ax.get_xaxis_transform(),
    )
    ax.text(
        x[0] + dx,
        -0.028,
        "Hub",
        ha="center",
        va="top",
        fontsize=9,
        color="#666",
        transform=ax.get_xaxis_transform(),
    )
    leg = [
        plt.Line2D([0], [0], color=EXTRACT_COLORS[e], lw=2.5, label=EXTRACT_DISPLAY[e])
        for e in EXTRACTIONS
    ]
    ax.legend(handles=leg, loc="upper right", frameon=False, fontsize=9)
    gt_name = GT_DISPLAY.get(gt_source, gt_source)
    fig.suptitle(f"{dataset} — Hub vs specialist (slope, {gt_name})", fontsize=TITLE_SIZE)
    fig.tight_layout()
    _save_hub_fig(fig, out_dir / f"{dataset}_gt-{gt_source}_hub_slope_{col}")


def plot_hub_summary_dot(
    summ: pd.DataFrame,
    models: list[str],
    dataset: str,
    gt_source: str,
    out_dir: Path,
    col: str = "jaccard",
) -> None:
    """Same x positions as bar chart; markers + optional sem error bars."""
    ylab_map = {m[0]: m[1] for m in METRICS}
    fig, ax = plt.subplots(figsize=HUB_SUMMARY_FIG)
    w, offsets = _hub_bar_offsets()
    x = np.arange(len(models))
    for hi, ht in enumerate(HUB_TIER_ORDER):
        for ext in EXTRACTIONS:
            for mi, model in enumerate(models):
                y = _hub_mean(summ, ht, model, ext, col)
                off = (hi - 0.5) * 0.36 + offsets[ext]
                err = _hub_sem(summ, ht, model, ext) if col == "jaccard" else 0.0
                ax.errorbar(
                    x[mi] + off,
                    y,
                    yerr=err if err > 0 else None,
                    fmt="o",
                    markersize=7,
                    color=EXTRACT_COLORS[ext],
                    ecolor="#555555",
                    elinewidth=1.0,
                    capsize=2.5,
                    alpha=0.55 if hi == 0 else 0.95,
                    label=f"{ht.split()[0]} · {EXTRACT_DISPLAY[ext]}" if mi == 0 and hi == 0 else None,
                )
    _style_hub_ax(ax, ylab_map.get(col, col), col, models, x)
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    gt_name = GT_DISPLAY.get(gt_source, gt_source)
    fig.suptitle(f"{dataset} — Hub vs specialist (dot ± SEM, {gt_name})", fontsize=TITLE_SIZE)
    fig.tight_layout()
    _save_hub_fig(fig, out_dir / f"{dataset}_gt-{gt_source}_hub_dot_{col}")


def plot_hub_summary_heatmap(
    summ: pd.DataFrame,
    models: list[str],
    dataset: str,
    gt_source: str,
    out_dir: Path,
    col: str = "jaccard",
) -> None:
    """Rows: tier × extraction; columns: models (same cell values as bar height)."""
    row_labels = []
    mat = []
    for ht in HUB_TIER_ORDER:
        for ext in EXTRACTIONS:
            row_labels.append(f"{ht.split()[0]} · {EXTRACT_DISPLAY[ext]}")
            mat.append([_hub_mean(summ, ht, m, ext, col) for m in models])
    data = np.array(mat, dtype=float)
    vmax = HUB_JACCARD_YLIM[1] if col == "jaccard" else 1.0
    fig, ax = plt.subplots(figsize=HUB_SUMMARY_FIG)
    cmap = LinearSegmentedColormap.from_list("hub_j", ["#F7F7F7", "#8FB4DC", "#2C5F8A"])
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=TEXT_SIZE - 1, color=LABEL_COLOR)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=TEXT_SIZE - 2, color=LABEL_COLOR)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(
                j,
                i,
                f"{data[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if data[i, j] > vmax * 0.55 else "#222222",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean per-TF Jaccard", fontsize=TEXT_SIZE - 1, color=LABEL_COLOR)
    gt_name = GT_DISPLAY.get(gt_source, gt_source)
    fig.suptitle(f"{dataset} — Hub vs specialist (heatmap, {gt_name})", fontsize=TITLE_SIZE)
    fig.tight_layout()
    _save_hub_fig(fig, out_dir / f"{dataset}_gt-{gt_source}_hub_heatmap_{col}")


def plot_hub_summary_lollipop(
    summ: pd.DataFrame,
    models: list[str],
    dataset: str,
    gt_source: str,
    out_dir: Path,
    col: str = "jaccard",
) -> None:
    """Lollipop: stem 0→value at same positions as grouped bar."""
    ylab_map = {m[0]: m[1] for m in METRICS}
    fig, ax = plt.subplots(figsize=HUB_SUMMARY_FIG)
    w, offsets = _hub_bar_offsets()
    x = np.arange(len(models))
    for hi, ht in enumerate(HUB_TIER_ORDER):
        for ext in EXTRACTIONS:
            for mi, model in enumerate(models):
                y = _hub_mean(summ, ht, model, ext, col)
                xp = x[mi] + (hi - 0.5) * 0.36 + offsets[ext]
                ax.vlines(
                    xp,
                    0,
                    y,
                    color=EXTRACT_COLORS[ext],
                    lw=1.4,
                    alpha=0.45 if hi == 0 else 0.9,
                    zorder=1,
                )
                ax.scatter(
                    xp,
                    y,
                    s=58,
                    color=EXTRACT_COLORS[ext],
                    edgecolors="#333",
                    linewidths=0.4,
                    alpha=0.55 if hi == 0 else 0.95,
                    zorder=2,
                    label=f"{ht.split()[0]} · {EXTRACT_DISPLAY[ext]}" if mi == 0 and hi == 0 else None,
                )
    ylim = HUB_LOLLIPOP_YLIM if col == "jaccard" else None
    _style_hub_ax(ax, ylab_map.get(col, col), col, models, x, ylim=ylim)
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    gt_name = GT_DISPLAY.get(gt_source, gt_source)
    fig.suptitle(f"{dataset} — Hub vs specialist (lollipop, {gt_name})", fontsize=TITLE_SIZE)
    fig.tight_layout()
    _save_hub_fig(fig, out_dir / f"{dataset}_gt-{gt_source}_hub_lollipop_{col}")


def plot_hub_binary_summary(
    df: pd.DataFrame,
    dataset: str,
    gt_source: str,
    out_dir: Path,
    metrics: Optional[List[str]] = None,
    styles: Optional[List[str]] = None,
) -> None:
    """Hub vs specialist summary — bar (default) and optional alternate shapes."""
    metric_cols = metrics if metrics is not None else ["jaccard"]
    style_list = styles if styles is not None else ["bar"]
    summ, models = summarize_hub_tier(df)
    col = metric_cols[0]

    if "bar" in style_list:
        plot_hub_summary_bar(summ, models, dataset, gt_source, out_dir, metric_cols)
    if "slope" in style_list and col == "jaccard":
        plot_hub_summary_slope(summ, models, dataset, gt_source, out_dir, col)
    if "dot" in style_list and col == "jaccard":
        plot_hub_summary_dot(summ, models, dataset, gt_source, out_dir, col)
    if "heatmap" in style_list and col == "jaccard":
        plot_hub_summary_heatmap(summ, models, dataset, gt_source, out_dir, col)
    if "lollipop" in style_list and col == "jaccard":
        plot_hub_summary_lollipop(summ, models, dataset, gt_source, out_dir, col)


def plot_outdegree_jaccard_scatter(
    df: pd.DataFrame,
    dataset: str,
    gt_source: str,
    out_dir: Path,
    model: str = "scGPT",
    extraction: str = "embhidden500",
) -> None:
    sub = df[(df["model"] == model) & (df["extraction"] == extraction)].drop_duplicates("TF")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(
        sub["gt_outdegree"],
        sub["jaccard"],
        c=sub["fp_rate"],
        cmap="YlOrRd",
        s=36,
        edgecolors="#333",
        linewidths=0.3,
        alpha=0.85,
    )
    ax.set_xlabel("GT out-degree", fontsize=TEXT_SIZE, color=LABEL_COLOR)
    ax.set_ylabel("Per-TF Jaccard", fontsize=TEXT_SIZE, color=LABEL_COLOR)
    ax.set_title(f"{dataset} — {model} ({method_label(extraction)})", fontsize=TEXT_SIZE)
    fig.colorbar(sc, ax=ax, label="FP rate")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    stem = out_dir / f"{dataset}_gt-{gt_source}_outdegree_vs_jaccard_{model}_{extraction}"
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {stem}.png/.pdf")


# ---------------------------------------------------------------------------
# TF family
# ---------------------------------------------------------------------------

def plot_family_heatmap(
    df: pd.DataFrame,
    dataset: str,
    gt_source: str,
    out_dir: Path,
    min_tfs: int = 5,
) -> None:
    fam = annotate_tf_families(df["TF"].unique().tolist())
    m = df.merge(fam, on="TF", how="left")
    m["family"] = m["family"].fillna("Other")

    agg = (
        m.groupby(["family", "model", "extraction"], as_index=False)["jaccard"]
        .agg(mean_jaccard="mean", n_tf="count")
    )
    fam_counts = m.groupby("family")["TF"].nunique()
    keep = fam_counts[fam_counts >= min_tfs].index.tolist()
    agg = agg[agg["family"].isin(keep)]
    if agg.empty:
        print("  [skip] family heatmap: no family with enough TFs")
        return

    agg["col"] = agg["model"] + "\n" + agg["extraction"].map(lambda e: EXTRACT_DISPLAY.get(e, e))
    pivot = agg.pivot_table(index="family", columns="col", values="mean_jaccard", aggfunc="mean")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig_h = max(5, 0.35 * len(pivot))
    fig, ax = plt.subplots(figsize=(max(12, 0.45 * len(pivot.columns)), fig_h))
    cmap = LinearSegmentedColormap.from_list(
        "fam_j",
        [(1, 1, 1), plt.matplotlib.colors.to_rgb(model_color("scGPT"))],
        N=256,
    )
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=max(0.15, pivot.to_numpy().max()))
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9, color=LABEL_COLOR)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=TEXT_SIZE - 1, color=LABEL_COLOR)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="#222")
    ax.set_title(
        f"{dataset} — Mean per-TF Jaccard by TF family\n(GT: {GT_DISPLAY.get(gt_source, gt_source)}; families with ≥{min_tfs} TFs)",
        fontsize=TEXT_SIZE,
    )
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Mean Jaccard")
    fig.tight_layout()
    stem = out_dir / f"{dataset}_gt-{gt_source}_family_jaccard_heatmap"
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {stem}.png/.pdf")


def plot_family_box(
    df: pd.DataFrame,
    dataset: str,
    gt_source: str,
    out_dir: Path,
    model: str = "scGPT",
    extraction: str = "embhidden500",
    min_tfs: int = 8,
    top_families: int = 10,
) -> None:
    fam = annotate_tf_families(df["TF"].unique().tolist())
    m = df[(df["model"] == model) & (df["extraction"] == extraction)].merge(fam, on="TF")
    counts = m.groupby("family")["TF"].nunique()
    keep = counts[counts >= min_tfs].sort_values(ascending=False).head(top_families).index.tolist()
    m = m[m["family"].isin(keep)]

    fig, ax = plt.subplots(figsize=(10, 5))
    order = m.groupby("family")["jaccard"].median().sort_values(ascending=False).index.tolist()
    sns.boxplot(
        data=m,
        x="family",
        y="jaccard",
        order=order,
        color=model_color(model),
        width=0.55,
        fliersize=3,
        ax=ax,
    )
    sns.stripplot(data=m, x="family", y="jaccard", order=order, color="#333", alpha=0.35, size=3, ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=TEXT_SIZE - 1, color=LABEL_COLOR)
    ax.set_ylabel("Per-TF Jaccard", fontsize=TEXT_SIZE, color=LABEL_COLOR)
    ax.set_xlabel("TF family", fontsize=TEXT_SIZE, color=LABEL_COLOR)
    ax.set_title(
        f"{dataset} — {model} ({method_label(extraction)})",
        fontsize=TEXT_SIZE,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    stem = out_dir / f"{dataset}_gt-{gt_source}_family_jaccard_box_{model}_{extraction}"
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {stem}.png/.pdf")


def run_family_fp_ora(
    df: pd.DataFrame,
    dataset: str,
    gt_source: str,
    out_dir: Path,
    model: str,
    extraction: str,
    min_genes: int = 8,
) -> None:
    gt_path = resolve_gt_path(gt_source, dataset)
    gt_gene1, gt_all, gt_n, _, gt_by_tf = load_gt_network(gt_path)
    fp = resolve_pred_path(model, extraction, dataset)
    if fp is None:
        return
    pred = filter_prediction(fp, gt_gene1, gt_all, gt_n)

    fam = annotate_tf_families(df["TF"].unique().tolist())
    tf_to_fam = dict(zip(fam["TF"], fam["family"]))

    pathways: dict = {}
    for gmt in GMT_HUMAN:
        if gmt.is_file():
            pathways.update(read_gmt(gmt))
    if not pathways:
        print("  [skip] family FP ORA: no GMT")
        return
    background = gt_all

    rows = []
    for family, tfs in fam.groupby("family")["TF"]:
        fp_genes: set = set()
        for tf in tfs:
            _, fp_set, _ = classify_tf_edges(tf, pred, gt_by_tf)
            fp_genes |= fp_set
        if len(fp_genes) < min_genes:
            continue
        ora = run_ora(fp_genes, pathways, background)
        if ora.empty:
            continue
        top = ora.head(5).copy()
        top["family"] = family
        top["n_fp_genes"] = len(fp_genes)
        top["n_tfs"] = len(tfs)
        rows.append(top)

    if not rows:
        return
    full = pd.concat(rows, ignore_index=True)
    full.to_csv(out_dir / f"{dataset}_gt-{gt_source}_family_fp_ora_{model}_{extraction}.csv", index=False)

    # lollipop panel: top families by best term
    best = (
        full.sort_values("neglog10P", ascending=False)
        .groupby("family")
        .head(1)
        .sort_values("neglog10P", ascending=False)
        .head(12)
    )
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(best))))
    y = np.arange(len(best))
    labels = [f"{r.family} (n={r.n_fp_genes})" for r in best.itertuples()]
    ax.barh(y, best["neglog10P"], color=model_color("scPrint"), alpha=0.85, height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=TEXT_SIZE - 1)
    ax.set_xlabel(r"$-\log_{10} P$", fontsize=TEXT_SIZE)
    ax.set_title(
        f"{dataset} — FP target enrichment by TF family\n{model} · {method_label(extraction)} · {GT_DISPLAY.get(gt_source, gt_source)}",
        fontsize=TEXT_SIZE,
    )
    for i, r in enumerate(best.itertuples()):
        ax.text(r.neglog10P + 0.05, i, shorten_term(r.Term, 40), va="center", fontsize=8)
    ax.invert_yaxis()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    stem = out_dir / f"{dataset}_gt-{gt_source}_family_fp_ora_lollipop"
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {stem}.png/.pdf")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TF hub stratification + family analysis")
    p.add_argument("--dataset", default="hESC")
    p.add_argument("--gt-source", default="STRING", choices=list(GT_SOURCES))
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--extractions", nargs="+", default=list(EXTRACTIONS))
    p.add_argument("--evl-root", type=Path, default=EVL_ROOT)
    p.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--hub-only", action="store_true")
    p.add_argument("--family-only", action="store_true")
    p.add_argument("--focus-model", default="", help="If set, hub raincloud only for this model")
    p.add_argument("--fig-w", type=float, default=FIG_W)
    p.add_argument("--fig-h", type=float, default=FIG_H)
    p.add_argument("--min-family-tfs", type=int, default=5)
    p.add_argument(
        "--binary-bar-metrics",
        nargs="+",
        default=["jaccard"],
        choices=["jaccard", "f1", "fp_rate"],
        help="Metrics for hub_binary_bar (default: jaccard only)",
    )
    p.add_argument(
        "--hub-summary-style",
        nargs="+",
        default=["bar", "slope", "dot", "heatmap", "lollipop"],
        choices=["bar", "slope", "dot", "heatmap", "lollipop", "all"],
        help="Shapes for hub tier summary (same data as bar). 'all' = every shape.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    apply_plot_style()
    do_hub = not args.family_only
    do_family = not args.hub_only

    out_dir = args.out or (FIG3 / "tf_static" / "output" / args.dataset / f"hub_family_{args.gt_source}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting per-TF metrics: {args.dataset}, GT={args.gt_source}")
    df = collect_per_tf_long(
        args.dataset,
        gt_source=args.gt_source,
        models=list(args.models),
        extractions=list(args.extractions),
        evl_root=args.evl_root,
        input_root=args.input_root,
    )
    if df.empty:
        raise RuntimeError("No per-TF metrics collected")
    df = assign_hub_strata(df)
    df.to_csv(out_dir / f"{args.dataset}_gt-{args.gt_source}_per_tf_hub_family_long.csv", index=False)
    print(f"  {len(df)} rows, {df['TF'].nunique()} TFs")

    focus = args.focus_model.strip() or None

    if do_hub:
        styles = list(args.hub_summary_style)
        if "all" in styles:
            styles = ["bar", "slope", "dot", "heatmap", "lollipop"]
        plot_hub_binary_summary(
            df,
            args.dataset,
            args.gt_source,
            out_dir,
            metrics=list(args.binary_bar_metrics),
            styles=styles,
        )
        plot_outdegree_jaccard_scatter(df, args.dataset, args.gt_source, out_dir)
        plot_hub_strat_raincloud(
            df, args.dataset, args.gt_source, out_dir,
            "hub_tier", focus, args.fig_w, args.fig_h,
        )
        plot_hub_strat_raincloud(
            df, args.dataset, args.gt_source, out_dir,
            "degree_quintile", focus, args.fig_w * 1.4, args.fig_h,
        )

    if do_family:
        fam_tbl = annotate_tf_families(df["TF"].unique().tolist())
        fam_tbl.to_csv(out_dir / f"{args.dataset}_tf_family_annotation.csv", index=False)
        plot_family_heatmap(df, args.dataset, args.gt_source, out_dir, min_tfs=args.min_family_tfs)
        plot_family_box(df, args.dataset, args.gt_source, out_dir)
        run_family_fp_ora(df, args.dataset, args.gt_source, out_dir, "scGPT", "embhidden500")

    print("Done.")


if __name__ == "__main__":
    main()
