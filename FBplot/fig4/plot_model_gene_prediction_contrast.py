#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig4 补充：scGPT vs scFoundation 单基因动态预测对比（可比尺度）

scGPT 输入/输出为 **binned expression**（~51 bins），不能与 CHIP 原始 log 表达直接同轴。
scFoundation 输出为 **连续 Δ**（mapped 基因），无逐基因迭代文件，用 acc_curve 插值。

本图在统一尺度上比较：
  - 真实：沿伪时间的观测进度 (E − early) / Δ_true
  - scGPT：每步预测 Δ 在 binned 空间 / Δ_true（原生 pred_delta_by_iter）
  - scFoundation：acc_curve 插值的预测 Δ / Δ_true

左列：两模型最终方向一致且均正确
右列：两模型预测方向相反

用法:
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig4
  python3 plot_model_gene_prediction_contrast.py --dataset hESC
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from fig4_palette import apply_fig4_style, model_color
except ImportError:

    def apply_fig4_style() -> None:
        plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})

    def model_color(name: str, default: str = "#808080") -> str:
        return default


from gene_result_io import load_gene_result
from plot_deg_robustness_analysis import load_model_gene_result, select_top_genes, TOP_PERCENT
from plot_hard_easy_expression_patterns import binned_pseudotime_curve, load_expression_pseudotime

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_ROOT = Path("/mnt/10T/yzn/benchmark_GRN")
DEFAULT_OUTDIR = SCRIPT_DIR / "robustness" / "supplement"
EPS = 1e-8
DPI = 600

DEFAULT_AGREE_GENES = ["POLR3G", "ESRP1", "ITGA5"]
DEFAULT_DISAGREE_GENES = ["MBOAT1", "CAMK2D", "OTX2"]


def load_gene_order(dataset: str) -> List[str]:
    gr = load_gene_result(BENCH_ROOT / f"pre_scgpt/results_multidataset_pseudotime_227/{dataset}_gene_result.csv")
    return gr["gene"].astype(str).tolist()


def get_gene_truth(dataset: str, gene: str) -> Optional[dict]:
    gr = load_gene_result(BENCH_ROOT / f"pre_scgpt/results_multidataset_pseudotime_227/{dataset}_gene_result.csv")
    row = gr[gr["gene"].astype(str) == gene]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "early": float(r["true_early_mean"]),
        "late": float(r["true_late_mean"]),
        "delta_true": float(r["delta_true"]),
        "dir_true": int(np.sign(float(r["delta_true"]))),
    }


def true_relative_progress(dataset: str, gene: str) -> Tuple[np.ndarray, np.ndarray]:
    """Observed progress toward true late state: (E − early) / Δ_true."""
    truth = get_gene_truth(dataset, gene)
    if truth is None or abs(truth["delta_true"]) < EPS:
        return np.array([]), np.array([])
    pt, expr = load_expression_pseudotime(dataset)
    if gene not in expr.index:
        return np.array([]), np.array([])
    cx, cy = binned_pseudotime_curve(pt, expr.loc[gene], n_bins=45)
    if len(cx) < 2:
        return np.array([]), np.array([])
    lo, hi = float(cx.min()), float(cx.max())
    x = (cx - lo) / (hi - lo + EPS)
    y = (cy - truth["early"]) / truth["delta_true"]
    return x, y


def scgpt_relative_delta(dataset: str, gene: str, delta_true: float) -> Tuple[np.ndarray, np.ndarray]:
    """scGPT pred_delta_by_iter (binned space) normalized by true Δ."""
    path = BENCH_ROOT / f"dyn4_results_unified/scgpt/per_dataset/{dataset}/pred_delta_by_iter.npy"
    genes = load_gene_order(dataset)
    if not path.is_file() or gene not in genes or abs(delta_true) < EPS:
        return np.array([]), np.array([])
    j = genes.index(gene)
    deltas = np.load(path).astype(float)[:, j]
    x = np.linspace(0, 1, len(deltas))
    y = deltas / delta_true
    return x, y


def scfoundation_relative_delta(dataset: str, gene: str, delta_true: float) -> Tuple[np.ndarray, np.ndarray]:
    """scFoundation: acc_curve × final pred_delta, normalized by true Δ."""
    acc_path = BENCH_ROOT / f"pre_scfoundation/scfoundation_multidataset_pseudotime_227/{dataset}/acc_curve.npy"
    scf = load_model_gene_result("scFoundation", dataset, {})
    if scf is None or not acc_path.is_file() or abs(delta_true) < EPS:
        return np.array([]), np.array([])
    row = scf[scf["gene"].astype(str) == gene]
    if row.empty:
        return np.array([]), np.array([])
    pred_delta = float(row.iloc[0]["delta_pred"])
    acc = np.load(acc_path).astype(float)
    tau = acc / acc[-1] if acc[-1] > EPS else np.linspace(0, 1, len(acc))
    x = np.linspace(0, 1, len(tau))
    y = tau * pred_delta / delta_true
    return x, y


def pick_example_genes(dataset: str, n: int = 3) -> Tuple[List[str], List[str]]:
    sg = load_model_gene_result("scGPT", dataset, {})
    sf = load_model_gene_result("scFoundation", dataset, {})
    if sg is None or sf is None:
        return DEFAULT_AGREE_GENES[:n], DEFAULT_DISAGREE_GENES[:n]

    sg = sg.set_index("gene")
    sf = sf.set_index("gene")
    top = set(select_top_genes(sg.reset_index(), TOP_PERCENT)["gene"].astype(str))

    rows = []
    for g in top:
        if g not in sg.index or g not in sf.index:
            continue
        sc_sign = np.sign(float(sg.loc[g, "delta_pred"]))
        sf_sign = np.sign(float(sf.loc[g, "delta_pred"]))
        rows.append(
            {
                "gene": g,
                "both_ok": bool(sg.loc[g, "dir_correct"] and sf.loc[g, "dir_correct"]),
                "same_sign": bool(sc_sign == sf_sign),
                "abs_true": abs(float(sg.loc[g, "delta_true"])),
            }
        )
    df = pd.DataFrame(rows)
    agree = df[df["both_ok"]].sort_values("abs_true", ascending=False)["gene"].tolist()
    disagree = df[(~df["same_sign"]) & (df["both_ok"] == False)].sort_values("abs_true", ascending=False)["gene"].tolist()
    if len(disagree) < n:
        disagree = df[~df["same_sign"]].sort_values("abs_true", ascending=False)["gene"].tolist()

    agree = agree[:n] if len(agree) >= n else DEFAULT_AGREE_GENES[:n]
    disagree = disagree[:n] if len(disagree) >= n else DEFAULT_DISAGREE_GENES[:n]
    return agree, disagree


def _dir_label(ok: bool) -> str:
    return "OK" if ok else "WRONG"


def plot_gene_panel(ax: plt.Axes, dataset: str, gene: str, panel_label: str) -> None:
    truth = get_gene_truth(dataset, gene)
    if truth is None:
        ax.set_visible(False)
        return

    d_true = truth["delta_true"]
    tx, ty = true_relative_progress(dataset, gene)
    x_sc, y_sc = scgpt_relative_delta(dataset, gene, d_true)
    x_sf, y_sf = scfoundation_relative_delta(dataset, gene, d_true)

    if len(tx) > 1:
        ax.plot(tx, ty, "-", color="#333333", lw=2.4, label="True (pseudotime)", zorder=2)
    if len(x_sc) > 0:
        ax.plot(x_sc, y_sc, "-o", ms=3.5, lw=2.0, color=model_color("scGPT"), label="scGPT (binned Δ)", zorder=3)
    if len(x_sf) > 0:
        ax.plot(x_sf, y_sf, "-s", ms=3, lw=2.0, color=model_color("scFoundation"), label="scFoundation (Δ iter)", zorder=3)

    ax.axhline(1.0, color="#888888", ls=":", lw=1.0, alpha=0.8)
    ax.axhline(0.0, color="#cccccc", ls="-", lw=0.8, alpha=0.8)

    sg = load_model_gene_result("scGPT", dataset, {}).set_index("gene")
    sf = load_model_gene_result("scFoundation", dataset, {}).set_index("gene")
    sc_ok = bool(sg.loc[gene, "dir_correct"]) if gene in sg.index else False
    sf_ok = bool(sf.loc[gene, "dir_correct"]) if gene in sf.index else False
    sc_final = float(sg.loc[gene, "delta_pred"]) / d_true if gene in sg.index else np.nan
    sf_final = float(sf.loc[gene, "delta_pred"]) / d_true if gene in sf.index else np.nan

    dir_word = "Down" if d_true < 0 else "Up"
    ax.set_title(
        f"{gene}  |  True {dir_word} (Δ_true={d_true:+.1f})\n"
        f"scGPT [{_dir_label(sc_ok)}] final Δ/Δ_true={sc_final:+.2f}   "
        f"scF [{_dir_label(sf_ok)}] final Δ/Δ_true={sf_final:+.2f}",
        fontsize=10,
        pad=6,
    )
    ax.set_ylabel(r"Progress ($\Delta / \Delta_{true}$)", fontsize=10)
    if panel_label:
        ax.text(0.02, 0.97, panel_label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def plot_contrast_figure(
    dataset: str,
    agree_genes: List[str],
    disagree_genes: List[str],
    outpath: Path,
) -> pd.DataFrame:
    apply_fig4_style()
    n = max(len(agree_genes), len(disagree_genes))
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.5 * n), dpi=DPI, sharex=True, sharey=False)

    if n == 1:
        axes = np.array([axes])

    col_titles = [
        "Consistent (scGPT & scFoundation agree, both correct)",
        "Inconsistent (opposite predicted direction)",
    ]
    for i in range(n):
        if i < len(agree_genes):
            plot_gene_panel(axes[i, 0], dataset, agree_genes[i], col_titles[0] if i == 0 else "")
        else:
            axes[i, 0].set_visible(False)
        if i < len(disagree_genes):
            plot_gene_panel(axes[i, 1], dataset, disagree_genes[i], col_titles[1] if i == 0 else "")
        else:
            axes[i, 1].set_visible(False)
        if i == n - 1:
            xlab = "Normalized progression (pseudotime or iteration 0→1)"
            axes[i, 0].set_xlabel(xlab)
            axes[i, 1].set_xlabel(xlab)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=10)

    fig.text(
        0.5,
        0.01,
        "Note: scGPT values are pred Δ in binned input space (51 bins), not raw log-expression; "
        "scFoundation trajectory is acc_curve-interpolated (no native per-gene iter). "
        "All curves divided by true Δ for comparability.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.suptitle(
        f"{dataset}: comparable prediction dynamics (scGPT binned vs scFoundation continuous)",
        fontsize=13,
        y=1.04,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

    rows = [{"dataset": dataset, "gene": g, "group": "consistent"} for g in agree_genes]
    rows += [{"dataset": dataset, "gene": g, "group": "inconsistent"} for g in disagree_genes]
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="scGPT vs scFoundation prediction contrast (comparable scale)")
    p.add_argument("--dataset", default="hESC")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--agree-genes", default=",".join(DEFAULT_AGREE_GENES))
    p.add_argument("--disagree-genes", default=",".join(DEFAULT_DISAGREE_GENES))
    p.add_argument("--auto-pick", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    (outdir / "tables").mkdir(exist_ok=True)

    if args.auto_pick:
        agree, disagree = pick_example_genes(args.dataset)
    else:
        agree = [g.strip() for g in args.agree_genes.split(",") if g.strip()]
        disagree = [g.strip() for g in args.disagree_genes.split(",") if g.strip()]

    print(f"Consistent: {agree}")
    print(f"Inconsistent: {disagree}")

    out_pdf = outdir / "figures" / f"figSX1b_scGPT_scFoundation_{args.dataset}.pdf"
    meta = plot_contrast_figure(args.dataset, agree, disagree, out_pdf)
    meta.to_csv(outdir / "tables" / f"model_prediction_contrast_genes_{args.dataset}.csv", index=False)
    print(f"\n✅ Saved: {out_pdf}")


if __name__ == "__main__":
    main()
