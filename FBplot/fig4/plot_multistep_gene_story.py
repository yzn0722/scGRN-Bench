#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hESC 多步伪时间 — 基因案例图：三条表达轨迹 + 各 transition |Δ|。

三条线（binned 空间，与 scGPT 一致）：
  1. 真值：5 段观测均值
  2. 单步预测：20% early → 20% late 迭代后的 pred_late，中间线性插值
  3. 分段预测：每对相邻段 (Si→Si+1) 各跑一次迭代，pred 落在 Si+1

需先运行（GPU）：
  python3 run_scgpt_pt_segments.py --dataset hESC

示例：
  python3 plot_multistep_gene_story.py
  python3 plot_multistep_gene_story.py --dataset hESC --genes NRP1,MYCT1,DNMT3B
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

try:
    from fig4_palette import apply_fig4_style, model_color
except ImportError:

    def apply_fig4_style() -> None:
        plt.rcParams.update({"font.size": 11})

    def model_color(_: str, d: str = "#666") -> str:
        return d

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = SCRIPT_DIR / "error_biology" / "multistep_pt" / "hESC" / "gene_story_multistep.png"
CHIP_DIR = Path("/mnt/10T/yzn/benchmark_GRN/input_process/CHIP")
PT_ROOT = Path("/mnt/10T/yzn/benchmark_GRN/PseudoTime")
GENE_CSV = SCRIPT_DIR / "error_biology" / "multistep_pt" / "hESC" / "gene_segment_deltas.csv"
TRAJ_CSV = SCRIPT_DIR / "error_biology" / "multistep_pt" / "hESC" / "gene_trajectories.csv"
SINGLE_STEP_DIR = Path("/mnt/10T/yzn/benchmark_GRN/pre_scgpt/results_multidataset_pseudotime_227")

N_SEG = 5
SEG_COLORS = ["#E8EEF4", "#C5D9ED", "#8EBAE5", "#4EA3F1", "#1A5FA8"]
COLOR_TRUE = model_color("scPrint")
COLOR_PRED_SEG = model_color("scGPT")
COLOR_PRED_SINGLE = "#9E9E9E"
COLOR_PEAK = model_color("scFoundation")
DPI = 300

GENE_NOTES = {
    "NRP1": "Non-CHIP · angiogenesis co-receptor\nPeak change in mid-trajectory (S1→S2)",
    "MYCT1": "Non-CHIP · stem-associated TF network\nLargest |Δ| at S2→S3, not at endpoints",
    "APOBEC3G": "Non-CHIP · innate immunity\nMid-trajectory surge vs flat endpoints",
    "DNMT3B": "CHIP TF target · de novo DNMT\nMonotonic drop → single early→late captures it",
    "SOX2": "CHIP TF target · pluripotency\nEarly drop; endpoint contrast still dominant",
    "NANOG": "CHIP TF target · pluripotency\nClassic lineage TF (CHIP-regulated)",
}


def load_expr_pt(dataset: str) -> Tuple[pd.DataFrame, np.ndarray]:
    expr = pd.read_csv(CHIP_DIR / f"{dataset}_chip_matched-ExpressionData.csv", index_col=0).T
    pt_df = pd.read_csv(PT_ROOT / dataset / "PseudoTime.csv")
    pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"})
    pt = pt_df.set_index("cell")["pt"].astype(float)
    common = expr.index.intersection(pt.index)
    expr = expr.loc[common]
    pt_arr = pt.loc[common].values
    order = np.argsort(pt_arr)
    return expr.iloc[order], pt_arr[order]


def segment_masks(pt: np.ndarray, n_seg: int = N_SEG) -> List[np.ndarray]:
    edges = np.quantile(pt, np.linspace(0, 1, n_seg + 1))
    edges[-1] += 1e-9
    masks = []
    for i in range(n_seg):
        if i < n_seg - 1:
            masks.append((pt >= edges[i]) & (pt < edges[i + 1]))
        else:
            masks.append((pt >= edges[i]) & (pt <= edges[i + 1]))
    return masks


def segment_mean_profile(expr: pd.DataFrame, pt: np.ndarray, gene: str) -> Optional[np.ndarray]:
    if gene not in expr.columns:
        return None
    y = expr[gene].values.astype(float)
    masks = segment_masks(pt)
    return np.array([float(y[m].mean()) for m in masks])


def scale_profile(y: np.ndarray) -> np.ndarray:
    pmin, pmax = float(y.min()), float(y.max())
    if pmax - pmin < 1e-9:
        return np.zeros_like(y)
    return (y - pmin) / (pmax - pmin)


def load_trajectories(dataset: str) -> Optional[pd.DataFrame]:
    path = SCRIPT_DIR / "error_biology" / "multistep_pt" / dataset / "gene_trajectories.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def profiles_from_trajectory(row: pd.Series, n_seg: int = N_SEG) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    true = np.array([float(row[f"true_seg{k}"]) for k in range(n_seg)])
    ps = np.array([float(row[f"pred_single_seg{k}"]) for k in range(n_seg)])
    pm = np.array([float(row[f"pred_multistep_seg{k}"]) for k in range(n_seg)])
    return true, ps, pm


def build_single_interp(dataset: str, gene: str, n_seg: int = N_SEG) -> Optional[np.ndarray]:
    path = SINGLE_STEP_DIR / f"{dataset}_gene_result.csv"
    if not path.exists():
        return None
    gr = pd.read_csv(path).set_index("gene")
    if gene not in gr.index:
        return None
    e = float(gr.loc[gene, "true_early_mean"])
    p = float(gr.loc[gene, "pred_late_like_mean"])
    return np.array([e + (k / (n_seg - 1)) * (p - e) for k in range(n_seg)])


def draw_schematic(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    ax.set_title(
        "Three trajectories: observed · single-step pred · per-segment iterative pred",
        fontsize=13,
        fontweight="600",
        pad=12,
    )
    seg_w = 1.6
    x0 = 1.0
    labels = ["S0\nearly", "S1", "S2", "S3", "S4\nlate"]
    for i, (lab, col) in enumerate(zip(labels, SEG_COLORS)):
        x = x0 + i * seg_w
        rect = FancyBboxPatch(
            (x, 1.2), seg_w * 0.92, 0.85,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=col, edgecolor="#333333", linewidth=1,
        )
        ax.add_patch(rect)
        ax.text(x + seg_w * 0.46, 1.62, lab, ha="center", va="center", fontsize=9, color="#111")
    ax.annotate("", xy=(9.2, 1.62), xytext=(0.8, 1.62),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.8))
    ax.text(5, 2.15, "Pseudotime →", ha="center", fontsize=11, style="italic")
    ax.plot([x0, x0 + seg_w * 0.9], [0.55, 0.55], color=COLOR_PRED_SINGLE, lw=2.5)
    ax.plot([x0 + 4 * seg_w, x0 + 4 * seg_w + seg_w * 0.9], [0.55, 0.55], color=COLOR_PRED_SINGLE, lw=2.5)
    ax.annotate(
        "", xy=(x0 + 4 * seg_w + seg_w * 0.9, 0.55), xytext=(x0 + seg_w * 0.9, 0.55),
        arrowprops=dict(arrowstyle="<->", color=COLOR_PRED_SINGLE, lw=2),
    )
    ax.text(5, 0.22, "Grey: single 20%→20% pred (interpolated)", ha="center", fontsize=9, color=COLOR_PRED_SINGLE)
    mx = x0 + 2.5 * seg_w
    ax.scatter([mx], [1.62], s=280, marker="*", c=COLOR_PEAK, zorder=5, edgecolors="white", linewidths=0.8)
    ax.text(mx, 0.75, "Blue: segment-wise\niterative pred", ha="center", fontsize=9, color=COLOR_PRED_SEG, fontweight="600")


def draw_gene_panel(
    ax,
    gene: str,
    true_y: np.ndarray,
    pred_single: Optional[np.ndarray],
    pred_multistep: Optional[np.ndarray],
    gene_row: pd.Series,
    trans_cols: List[str],
    trans_labels: List[str],
) -> None:
    is_chip = bool(gene_row.get("is_chip_target", False))
    tag_color = model_color("scGPT") if is_chip else model_color("scPrint")
    tag = "CHIP TF target" if is_chip else "Non-CHIP target"

    ax2 = ax.inset_axes([0.08, 0.06, 0.88, 0.32])
    x = np.arange(N_SEG)

    # 共用 min-max（三条线可比）
    stack = [true_y]
    if pred_single is not None:
        stack.append(pred_single)
    if pred_multistep is not None:
        stack.append(pred_multistep)
    allv = np.concatenate(stack)
    vmin, vmax = float(allv.min()), float(allv.max())
    pr_true = (true_y - vmin) / (vmax - vmin + 1e-9)

    ax.fill_between(x, pr_true, alpha=0.1, color=COLOR_TRUE)
    ax.plot(x, pr_true, "o-", color=COLOR_TRUE, linewidth=2.8, markersize=9, label="Observed (binned)", zorder=4)

    if pred_single is not None:
        pr_s = (pred_single - vmin) / (vmax - vmin + 1e-9)
        ax.plot(x, pr_s, "s--", color=COLOR_PRED_SINGLE, linewidth=2, markersize=7, alpha=0.9, label="Single-step pred")

    if pred_multistep is not None:
        pr_m = (pred_multistep - vmin) / (vmax - vmin + 1e-9)
        ax.plot(x, pr_m, "^-", color=COLOR_PRED_SEG, linewidth=2.4, markersize=8, label="Per-segment pred")

    peak_seg = int(np.argmax(true_y))
    ax.axvspan(peak_seg - 0.45, peak_seg + 0.45, alpha=0.15, color=COLOR_PEAK, zorder=0)
    ax.scatter([peak_seg], [pr_true[peak_seg]], s=180, marker="*", c=COLOR_PEAK, zorder=6, edgecolors="white")

    ax.set_xticks(x)
    ax.set_xticklabels(["S0", "S1", "S2", "S3", "S4"], fontsize=9)
    ax.set_ylabel("Expr.\n(scaled)", fontsize=9)
    ax.set_xlim(-0.2, N_SEG - 0.8)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.set_title(gene, fontsize=14, fontweight="700", color=tag_color, loc="left")
    ax.text(
        0.98, 0.95, tag,
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        color="white", fontweight="600",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=tag_color, edgecolor="none"),
    )
    note = GENE_NOTES.get(gene, "")
    if note:
        ax.text(0.02, 0.04, note, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color="#444")

    dvals = [abs(float(gene_row[c])) for c in trans_cols]
    d_single = abs(float(gene_row["delta_single"]))
    best_t = str(gene_row.get("best_transition", ""))
    colors = [COLOR_PEAK if c == best_t else tag_color for c in trans_cols]
    xt = np.arange(len(trans_cols))
    ax2.bar(xt, dvals, color=colors, edgecolor="white", linewidth=0.8, alpha=0.9)
    ax2.axhline(d_single, color=COLOR_PRED_SINGLE, linestyle="--", linewidth=1.8)
    ax2.set_ylabel(r"|Δ_true|", fontsize=9)
    ax2.set_xticks(xt)
    ax2.set_xticklabels(trans_labels, fontsize=8, rotation=15)
    ax2.set_title("Per-transition |Δ_true|", fontsize=8, pad=2)

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def build_figure(dataset: str, genes: List[str], outpath: Path) -> None:
    apply_fig4_style()
    gf_path = SCRIPT_DIR / "error_biology" / "multistep_pt" / dataset / "gene_segment_deltas.csv"
    gf = pd.read_csv(gf_path)
    chip = set(pd.read_csv(CHIP_DIR / f"{dataset}_chip_matched-network.csv")["Gene2"].astype(str))
    if "is_chip_target" not in gf.columns:
        gf["is_chip_target"] = gf["gene"].astype(str).isin(chip)

    traj = load_trajectories(dataset)
    has_traj = traj is not None

    trans_cols = sorted([c for c in gf.columns if c.startswith("delta_t")])
    trans_labels = [c.replace("delta_", "").replace("_", "→") for c in trans_cols]

    expr, pt = load_expr_pt(dataset)
    n = len(genes)
    fig = plt.figure(figsize=(13, 3.2 + 3.8 * n), facecolor="white")
    gs = fig.add_gridspec(1 + n, 1, height_ratios=[1.15] + [1] * n, hspace=0.38)

    ax0 = fig.add_subplot(gs[0, 0])
    draw_schematic(ax0)
    if not has_traj:
        ax0.text(
            0.5, -0.08,
            "Run: python3 run_scgpt_pt_segments.py --dataset " + dataset,
            transform=ax0.transAxes, ha="center", fontsize=9, color="#c00",
        )

    for i, gene in enumerate(genes):
        ax = fig.add_subplot(gs[i + 1, 0])
        if gene not in gf["gene"].values:
            ax.text(0.5, 0.5, f"{gene} not found", ha="center", transform=ax.transAxes)
            continue
        row = gf.loc[gf["gene"] == gene].iloc[0]

        true_y = pred_single = pred_multistep = None
        if has_traj and gene in traj["gene"].values:
            tr = traj.loc[traj["gene"] == gene].iloc[0]
            true_y, pred_single, pred_multistep = profiles_from_trajectory(tr)
        else:
            prof = segment_mean_profile(expr, pt, gene)
            if prof is not None:
                true_y = prof
            pred_single = build_single_interp(dataset, gene)

        if true_y is None:
            ax.text(0.5, 0.5, f"No expression for {gene}", ha="center", transform=ax.transAxes)
            continue

        draw_gene_panel(ax, gene, true_y, pred_single, pred_multistep, row, trans_cols, trans_labels)

    subtitle = "binned expr · 3 trajectories" if has_traj else "observed only — run run_scgpt_pt_segments.py for preds"
    fig.suptitle(
        f"{dataset} — gene examples: {subtitle}",
        fontsize=14,
        fontweight="600",
        y=1.002,
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {outpath}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="hESC")
    p.add_argument("--genes", default="NRP1,MYCT1,DNMT3B")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    genes = [g.strip() for g in args.genes.split(",") if g.strip()]
    if args.dataset != "hESC":
        args.out = SCRIPT_DIR / "error_biology" / "multistep_pt" / args.dataset / "gene_story_multistep.png"
    build_figure(args.dataset, genes, args.out)


if __name__ == "__main__":
    main()
