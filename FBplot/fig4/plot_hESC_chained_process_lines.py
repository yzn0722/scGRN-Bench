#!/usr/bin/env python3
"""hESC 链式 scGPT 过程折线图：伪时间轨迹 + 分段准确率折线。"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from fig4_palette import apply_fig4_style, model_color
except ImportError:

    def apply_fig4_style() -> None:
        plt.rcParams.update({"font.size": 11})

    def model_color(name: str, default: str = "#666") -> str:
        return default


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHAINED = SCRIPT_DIR / "error_biology" / "multistep_pt" / "hESC" / "chained_preds"
GENE_SEG = SCRIPT_DIR / "error_biology" / "multistep_pt" / "hESC" / "gene_segment_deltas.csv"
SINGLE_DIR = Path("/mnt/10T/yzn/benchmark_GRN/pre_scgpt/results_multidataset_pseudotime_227")

TRANSITIONS = ["delta_t0_t1", "delta_t1_t2", "delta_t2_t3", "delta_t3_t4"]
TRANS_LABELS = ["S0→S1", "S1→S2", "S2→S3", "S3→S4"]
PT_NODES = ["S0", "S1", "S2", "S3", "S4"]
N_SEG = 5

COLOR_TRUE = model_color("scPrint")
COLOR_CHAIN = model_color("scGPT")
COLOR_SINGLE = "#9E9E9E"
COLOR_OK = "#43A047"
COLOR_BAD = "#E53935"


def load_segment_tables(chained_dir: Path, dataset: str = "hESC") -> Dict[str, pd.DataFrame]:
    return {
        t: pd.read_csv(chained_dir / f"{dataset}_{t}_gene_result.csv").set_index("gene")
        for t in TRANSITIONS
    }


def chained_profiles(gene: str, seg_tables: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
    t0 = seg_tables["delta_t0_t1"]
    if gene not in t0.index:
        return np.full(N_SEG, np.nan), np.full(N_SEG, np.nan)
    true = [float(t0.loc[gene, "true_baseline_mean"])]
    pred = [true[0]]
    for t in TRANSITIONS:
        df = seg_tables[t]
        true.append(float(df.loc[gene, "true_late_mean"]))
        pred.append(float(df.loc[gene, "pred_late_like_mean"]))
    return np.array(true), np.array(pred)


def collect_all_profiles(
    genes: List[str], seg_tables: Dict[str, pd.DataFrame]
) -> Tuple[np.ndarray, np.ndarray]:
    true_rows, pred_rows = [], []
    for g in genes:
        ty, py = chained_profiles(g, seg_tables)
        if np.all(np.isfinite(ty)):
            true_rows.append(ty)
            pred_rows.append(py)
    if not true_rows:
        return np.full((0, N_SEG), np.nan), np.full((0, N_SEG), np.nan)
    return np.vstack(true_rows), np.vstack(pred_rows)


def plot_accuracy_lines(out: Path, seg_tables: Dict[str, pd.DataFrame], seg_sum: pd.DataFrame) -> None:
    x = np.arange(4)
    gene_acc = [seg_tables[t]["dir_correct"].mean() * 100 for t in TRANSITIONS]
    top_acc = (seg_sum.set_index("transition").loc[TRANSITIONS, "acc_top_percent"] * 100).tolist()

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x, top_acc, "o-", color=COLOR_CHAIN, lw=2.5, ms=9, label="Top 30% dynamic")
    ax.plot(x, gene_acc, "s--", color="#546E7A", lw=2, ms=8, label="All genes")
    ax.set_xticks(x)
    ax.set_xticklabels(TRANS_LABELS)
    ax.set_ylim(50, 95)
    ax.set_ylabel("Direction accuracy (%)")
    ax.set_xlabel("Chained pseudotime block")
    ax.set_title("hESC chained scGPT — accuracy along the chain", fontweight="600")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    for xi, v in zip(x, top_acc):
        ax.annotate(f"{v:.0f}%", (xi, v), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_aggregate_trajectories(out: Path, wide: pd.DataFrame, seg_tables: Dict[str, pd.DataFrame]) -> None:
    """全基因分组：四段全对 / 全错 / 仅首段错 的平均轨迹折线。"""
    dir_cols = [f"{t}_dir_correct" for t in TRANSITIONS]
    nc = wide[dir_cols].sum(axis=1)
    groups = {
        "All 4 blocks correct": wide.loc[nc == 4, "gene"].astype(str).tolist(),
        "All 4 blocks wrong": wide.loc[nc == 0, "gene"].astype(str).tolist(),
        "Wrong only S0→S1": wide.loc[
            (wide["delta_t0_t1_dir_correct"] == 0)
            & (wide["delta_t1_t2_dir_correct"] == 1)
            & (wide["delta_t2_t3_dir_correct"] == 1)
            & (wide["delta_t3_t4_dir_correct"] == 1),
            "gene",
        ].astype(str).tolist(),
    }
    x = np.arange(N_SEG)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    # 左：平均表达轨迹（归一化到 [0,1] 便于比较形状）
    ax = axes[0]
    for label, genes in groups.items():
        if not genes:
            continue
        ty, py = collect_all_profiles(genes, seg_tables)
        mean_t = np.nanmean(np.array([norm_row(r) for r in ty]), axis=0)
        mean_p = np.nanmean(np.array([norm_row(r) for r in py]), axis=0)
        color = COLOR_OK if "correct" in label else (COLOR_BAD if "wrong" in label else COLOR_CHAIN)
        ax.plot(x, mean_t, "o-", lw=2.2, ms=7, color=color, alpha=0.9, label=f"{label} (true)")
        ax.plot(x, mean_p, "s--", lw=1.8, ms=6, color=color, alpha=0.55, label=f"{label} (chained pred)")

    ax.set_xticks(x)
    ax.set_xticklabels(PT_NODES)
    ax.set_ylabel("Mean scaled expression")
    ax.set_title("A  Mean trajectory by error group", fontweight="600", loc="left")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    # 右：真值与预测偏离（pred - true）沿链的平均
    ax = axes[1]
    for label, genes in groups.items():
        if not genes:
            continue
        ty, py = collect_all_profiles(genes, seg_tables)
        gap = np.nanmean(py - ty, axis=0)
        color = COLOR_OK if "correct" in label else (COLOR_BAD if "wrong" in label else COLOR_CHAIN)
        ax.plot(x, gap, "o-", lw=2.2, ms=7, color=color, label=label)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(PT_NODES)
    ax.set_ylabel("Mean pred − true")
    ax.set_title("B  Prediction drift along chain", fontweight="600", loc="left")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("hESC chained process — aggregate pseudotime lines", fontsize=13, fontweight="700", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def norm_row(y: np.ndarray) -> np.ndarray:
    lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
    if hi - lo < 1e-9:
        return np.zeros_like(y)
    return (y - lo) / (hi - lo)


def plot_gene_grid(
    out: Path,
    genes: List[str],
    seg_tables: Dict[str, pd.DataFrame],
    titles: Dict[str, str] | None = None,
) -> None:
    n = len(genes)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)
    x = np.arange(N_SEG)

    for ax, gene in zip(axes.flat, genes):
        ty, py = chained_profiles(gene, seg_tables)
        if not np.all(np.isfinite(ty)):
            ax.set_visible(False)
            continue
        ax.plot(x, ty, "o-", color=COLOR_TRUE, lw=2.5, ms=8, label="Observed", zorder=3)
        ax.plot(x, py, "^-", color=COLOR_CHAIN, lw=2.2, ms=8, label="Chained pred")
        for i, t in enumerate(TRANSITIONS):
            df = seg_tables[t]
            if gene in df.index:
                ok = int(df.loc[gene, "dir_correct"]) > 0
                ax.annotate(
                    "✓" if ok else "✗",
                    (i + 1, py[i + 1]),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=9,
                    fontweight="bold",
                    color=COLOR_OK if ok else COLOR_BAD,
                )
        subtitle = (titles or {}).get(gene, "")
        ax.set_title(f"{gene}\n{subtitle}", fontweight="600", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(PT_NODES)
        ax.set_ylabel("Mean expression")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=7, loc="best")

    for ax in axes.flat[len(genes) :]:
        ax.set_visible(False)

    fig.suptitle(
        "hESC chained scGPT — per-gene pseudotime trajectories\n"
        "(✓/✗ = direction correct for each block)",
        fontsize=12,
        fontweight="700",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_chain_schematic_with_lines(
    out: Path,
    gene: str,
    seg_tables: Dict[str, pd.DataFrame],
) -> None:
    """单基因：上方示意链式传递，下方折线。"""
    ty, py = chained_profiles(gene, seg_tables)
    if not np.all(np.isfinite(ty)):
        return

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.9, 1.4], hspace=0.35)

    ax0 = fig.add_subplot(gs[0])
    ax0.set_xlim(-0.2, 4.2)
    ax0.set_ylim(0, 1)
    ax0.axis("off")
    ax0.set_title(f"Chain-of-thought blocks for {gene}", fontweight="600", loc="left")
    for i, lab in enumerate(TRANS_LABELS):
        ax0.text(i, 0.72, lab, ha="center", fontsize=10, fontweight="600",
                 bbox=dict(boxstyle="round", facecolor="#E3F2FD", edgecolor=COLOR_CHAIN))
        if i < 4:
            ax0.annotate("", xy=(i + 0.85, 0.45), xytext=(i + 0.15, 0.45),
                         arrowprops=dict(arrowstyle="-|>", color=COLOR_CHAIN, lw=2))
            ax0.text(i + 0.5, 0.28, "pred →\nnext block", ha="center", fontsize=8, color="#555")

    ax1 = fig.add_subplot(gs[1])
    x = np.arange(N_SEG)
    ax1.plot(x, ty, "o-", color=COLOR_TRUE, lw=3, ms=10, label="Observed (segment means)")
    ax1.plot(x, py, "^-", color=COLOR_CHAIN, lw=2.8, ms=10, label="Chained scGPT")
    ax1.fill_between(x, ty, py, alpha=0.12, color=COLOR_BAD)
    ax1.set_xticks(x)
    ax1.set_xticklabels(PT_NODES)
    ax1.set_ylabel("Mean expression")
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_title("Expression trajectory along pseudotime", fontweight="600", loc="left")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def default_gene_panels(wide: pd.DataFrame, wrong_csv: Path | None) -> Tuple[List[str], Dict[str, str]]:
    titles: Dict[str, str] = {}
    picks: List[str] = []

    if wrong_csv and wrong_csv.exists():
        w = pd.read_csv(wrong_csv)
        fixed = w[w["fixed_chained_peak"]].sort_values("abs_delta_best", ascending=False)
        stuck = w[~w["fixed_chained_peak"]].sort_values("abs_delta_best", ascending=False)
        if len(fixed):
            g = str(fixed.iloc[0]["gene"])
            picks.append(g)
            titles[g] = "Single wrong → chained fixes @ peak"
        if len(stuck):
            g = str(stuck.iloc[0]["gene"])
            picks.append(g)
            titles[g] = "Single wrong → still wrong"

    for g in ["ERCC-00113", "NRP1", "LPIN1", "CSRP2"]:
        if g not in picks:
            picks.append(g)
        if len(picks) >= 6:
            break

    nc = wide[[f"{t}_dir_correct" for t in TRANSITIONS]].sum(axis=1)
    always_ok = wide.loc[nc == 4, "gene"].astype(str).iloc[0] if (nc == 4).any() else None
    always_bad = wide.loc[nc == 0, "gene"].astype(str).iloc[0] if (nc == 0).any() else None
    for g, note in [(always_ok, "4/4 blocks correct"), (always_bad, "4/4 blocks wrong")]:
        if g and g not in picks:
            picks.append(g)
            titles[g] = note

    return picks[:6], titles


def main() -> None:
    p = argparse.ArgumentParser(description="hESC chained process line plots")
    p.add_argument("--chained-dir", type=Path, default=DEFAULT_CHAINED)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--genes", type=str, default="", help="comma-separated genes")
    p.add_argument("--schematic-gene", type=str, default="ERCC-00113")
    args = p.parse_args()

    apply_fig4_style()
    outdir = args.outdir or (args.chained_dir / "figures")
    outdir.mkdir(parents=True, exist_ok=True)

    seg_tables = load_segment_tables(args.chained_dir)
    wide = pd.read_csv(args.chained_dir / "hESC_chained_per_gene_wide.csv")
    seg_sum = pd.read_csv(args.chained_dir / "hESC_chained_segment_summary.csv")
    wrong_csv = outdir / "wrong_single_vs_chained_peak.csv"

    plot_accuracy_lines(outdir / "12_accuracy_line.png", seg_tables, seg_sum)
    plot_aggregate_trajectories(outdir / "13_aggregate_trajectory_lines.png", wide, seg_tables)

    if args.genes.strip():
        genes = [g.strip() for g in args.genes.split(",") if g.strip()]
        titles = {}
    else:
        genes, titles = default_gene_panels(wide, wrong_csv)

    plot_gene_grid(outdir / "14_gene_trajectory_lines.png", genes, seg_tables, titles)
    plot_chain_schematic_with_lines(
        outdir / f"15_chain_schematic_{args.schematic_gene}.png",
        args.schematic_gene,
        seg_tables,
    )

    print(f"Saved line plots to {outdir}")
    print(f"  Genes in grid: {', '.join(genes)}")
    print(f"  Schematic gene: {args.schematic_gene}")


if __name__ == "__main__":
    main()
