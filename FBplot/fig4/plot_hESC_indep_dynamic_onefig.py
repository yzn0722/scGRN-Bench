#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单图：每段独立迭代（非链式）拼接的动态折线 vs 真值。

每段 S_i→S_{i+1} 从真实 Si 细胞单独跑 scGPT（segment_preds），
将 4 段末预测拼成 5 点折线，与观测轨迹对比。

  python3 plot_hESC_indep_dynamic_onefig.py
  python3 plot_hESC_indep_dynamic_onefig.py --genes LRAT,HDGF,ERCC-00113
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from plot_hESC_indep_vs_chain_nonlinear import (  # noqa: E402
    CHAINED_DIR,
    INDEP_DIR,
    N_PTS,
    PT_NODES,
    TRANSITIONS,
    TRANS_LABELS,
    indep_profile,
    load_chained_tables,
    load_indep_tables,
    norm_profile,
    shape_corr,
    true_profile,
)

try:
    from fig4_palette import apply_fig4_style, model_color
except ImportError:

    def apply_fig4_style() -> None:
        plt.rcParams.update({"font.size": 11})

    def model_color(name: str, default: str = "#666") -> str:
        return default


COLOR_TRUE = model_color("scPrint")
COLOR_INDEP = model_color("scFoundation")
SEG_ALPHA = 0.07
METRICS_CSV = (
    SCRIPT_DIR
    / "error_biology/multistep_pt/hESC/chained_preds/figures/nonlinear_explain/gene_trajectory_metrics.csv"
)


def pick_genes(
    metrics: Optional[pd.DataFrame],
    mode: str = "good",
    n: int = 9,
    min_shape: float = 0.85,
) -> List[str]:
    """good: 仅 shape_corr_indep 高、优先非单调动态基因。"""
    if metrics is None or not METRICS_CSV.exists():
        return ["LRAT", "HDGF", "PCDH10", "SERPINE2", "MSX1", "FST"][:n]
    m = metrics.copy()
    if mode == "good":
        pool = m[m["shape_corr_indep"] >= min_shape].sort_values("shape_corr_indep", ascending=False)
        # 先选有折线感的类型，再补单调高分基因
        nl = pool[pool["trajectory_type"].isin(["sign_flip", "peak_middle", "mixed"])]
        mono = pool[pool["trajectory_type"] == "monotone"]
        out: List[str] = []
        for sub in (nl, mono, pool):
            for g in sub["gene"].astype(str):
                if g not in out:
                    out.append(g)
                if len(out) >= n:
                    return out
        return out[:n]
    # legacy: good / mid / bad mix
    interesting = m[m["trajectory_type"].isin(["sign_flip", "peak_middle", "mixed"])]
    pool = interesting if len(interesting) >= 6 else m
    pool = pool.sort_values("shape_corr_indep", ascending=False)
    n_each = max(2, n // 3)
    top = pool.head(n_each)["gene"].astype(str).tolist()
    mid = pool.iloc[len(pool) // 2 - 1 : len(pool) // 2 - 1 + n_each]["gene"].astype(str).tolist()
    bot = pool.tail(n_each)["gene"].astype(str).tolist()
    out = []
    for g in top + mid + bot:
        if g not in out:
            out.append(g)
    return out[:n]


def mean_population_profiles(
    genes: List[str], ch: Dict, ind: Dict
) -> tuple[np.ndarray, np.ndarray]:
    ty_list, pi_list = [], []
    for g in genes:
        ty = true_profile(ch, g)
        pi = indep_profile(ind, ch, g)
        if np.all(np.isfinite(ty)) and np.all(np.isfinite(pi)):
            ty_list.append(norm_profile(ty))
            pi_list.append(norm_profile(pi))
    if not ty_list:
        return np.full(N_PTS, np.nan), np.full(N_PTS, np.nan)
    return np.nanmean(ty_list, axis=0), np.nanmean(pi_list, axis=0)


def plot_one_figure(
    out: Path,
    genes: List[str],
    ch: Dict,
    ind: Dict,
    all_genes: List[str],
    metrics: Optional[pd.DataFrame],
    mode: str = "good",
    min_shape: float = 0.85,
) -> None:
    n = len(genes)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(12, 3.5 * nrows + 0.8))
    gs = fig.add_gridspec(nrows, ncols, hspace=0.45, wspace=0.32)
    x = np.arange(N_PTS)

    m_idx = metrics.set_index("gene") if metrics is not None and len(metrics) else None
    pop_genes = genes if mode == "good" else all_genes
    mean_ty, mean_pi = mean_population_profiles(pop_genes, ch, ind)
    shape_means = []
    for g in pop_genes:
        ty = true_profile(ch, g)
        pi = indep_profile(ind, ch, g)
        r = shape_corr(ty, pi)
        if np.isfinite(r):
            shape_means.append(r)

    for idx, gene in enumerate(genes):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        ty = true_profile(ch, gene)
        pi = indep_profile(ind, ch, gene)
        if not np.all(np.isfinite(ty)):
            ax.set_visible(False)
            continue

        # 分段底色（每段独立一块）
        for seg_i in range(4):
            ax.axvspan(seg_i + 0.5, seg_i + 1.5, color=COLOR_INDEP, alpha=SEG_ALPHA)

        ax.plot(x, norm_profile(ty), "o-", color=COLOR_TRUE, lw=2.8, ms=9, label="Observed", zorder=4)
        ax.plot(x, norm_profile(pi), "s-", color=COLOR_INDEP, lw=2.4, ms=8, label="Indep. blocks", zorder=3)

        # 段间竖线
        for xi in [0.5, 1.5, 2.5, 3.5]:
            ax.axvline(xi, color="#BDBDBD", lw=0.8, ls=":")

        r = shape_corr(ty, pi)
        tag = ""
        if m_idx is not None and gene in m_idx.index:
            tag = str(m_idx.loc[gene].get("trajectory_type", ""))
        ax.set_title(f"{gene}  ·  shape r = {r:.2f}  ·  {tag}", fontweight="700", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(PT_NODES)
        ax.set_ylabel("Scaled expression")
        ax.grid(axis="y", alpha=0.25)
        if idx == 0:
            ax.legend(fontsize=8, loc="upper left", framealpha=0.92)

        # 块标注（仅第一个子图）
        if idx == 0:
            for seg_i, lab in enumerate(TRANS_LABELS):
                ax.text(seg_i + 1, 1.02, lab, transform=ax.get_xaxis_transform(),
                        ha="center", fontsize=7, color=COLOR_INDEP)

    # 隐藏多余格
    for j in range(len(genes), nrows * ncols):
        fig.add_subplot(gs[j // ncols, j % ncols]).set_visible(False)

    # 底部总览：人群平均折线
    ax_pop = fig.add_axes([0.08, 0.02, 0.84, 0.14])
    if np.all(np.isfinite(mean_ty)):
        ax_pop.plot(x, mean_ty, "o-", color=COLOR_TRUE, lw=2.5, ms=7, label="Mean observed")
        ax_pop.plot(x, mean_pi, "s-", color=COLOR_INDEP, lw=2.2, ms=6, label="Mean indep. stitch")
    ax_pop.set_xticks(x)
    ax_pop.set_xticklabels(PT_NODES)
    ax_pop.set_ylabel("Scaled (mean)", fontsize=8)
    ax_pop.legend(fontsize=7, ncol=2, loc="upper right")
    ax_pop.grid(axis="y", alpha=0.25)
    pop_title = (
        f"Mean over {len(pop_genes)} well-fitted genes (shape r ≥ {min_shape})"
        if mode == "good"
        else "Population mean over all genes"
    )
    ax_pop.set_title(pop_title, fontsize=9, loc="left")

    mean_r = float(np.mean(shape_means)) if shape_means else float("nan")
    fig.suptitle(
        "hESC — Independent blocks CAN match dynamic trajectories (well-fitted genes)\n"
        "4 separate scGPT runs from true Si (not chained); orange = observed, blue = stitched pred",
        fontsize=12, fontweight="700", y=0.98,
    )
    footer = (
        f"Shown: n={len(genes)} genes with shape r ≥ {min_shape}  ·  "
        f"Mean shape r (panel set) = {mean_r:.2f}"
    )
    if mode == "good":
        footer += "  ·  Independent segment prediction captures nonlinear folds for a subset of dynamic genes"
    fig.text(0.5, 0.005, footer, ha="center", fontsize=9, color="#444")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="One figure: indep segment dynamic trajectory")
    p.add_argument("--chained-dir", type=Path, default=CHAINED_DIR)
    p.add_argument("--indep-dir", type=Path, default=INDEP_DIR)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--genes", type=str, default="")
    p.add_argument("--mode", choices=["good", "mixed"], default="good", help="good=仅高 shape r 基因")
    p.add_argument("--min-shape", type=float, default=0.85)
    p.add_argument("--n-genes", type=int, default=9)
    args = p.parse_args()

    apply_fig4_style()
    default_name = (
        "indep_segment_dynamic_line_goodfit.png"
        if args.mode == "good"
        else "indep_segment_dynamic_line.png"
    )
    out = args.out or (args.chained_dir / "figures" / default_name)

    ch = load_chained_tables(args.chained_dir)
    ind = load_indep_tables(args.indep_dir)
    metrics = pd.read_csv(METRICS_CSV) if METRICS_CSV.exists() else None

    all_genes = sorted(set(ch["delta_t0_t1"].index) & set(ind[0].index))
    if args.genes.strip():
        genes = [g.strip() for g in args.genes.split(",") if g.strip()]
    else:
        genes = pick_genes(metrics, mode=args.mode, n=args.n_genes, min_shape=args.min_shape)

    plot_one_figure(out, genes, ch, ind, all_genes, metrics, mode=args.mode, min_shape=args.min_shape)
    print(f"Saved: {out}")
    print(f"Genes: {', '.join(genes)}")


if __name__ == "__main__":
    main()
