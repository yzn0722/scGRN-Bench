#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scGPT 三种边提取方式 — per-TF 静态 GRN 对比（fig3/tf_static）。

与 plot_tf_static_three.py（单方法深挖）配套：
  - 本脚本：**方法间**对比（emb500 / embhidden500 / att500）
  - 单方法脚本：ego 网、FP-ORA 等机制图

跨模型（scGPT vs Geneformer）见 fig2 / 3.2multi_model3.py，不在此重复。

示例:
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig3
  python tf_static/plot_tf_static_compare_methods.py --dataset hESC
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

FIG3 = Path(__file__).resolve().parents[1]
if str(FIG3) not in sys.path:
    sys.path.insert(0, str(FIG3))

from fig3_palette import method_color, method_label, model_color  # noqa: E402

from tf_static.plot_tf_static_three import (  # noqa: E402
    METHOD_PRED,
    apply_style,
    plot_hub_ego_panels,
    resolve_pred_path,
)
from tf_static.utils import (  # noqa: E402
    filter_prediction,
    load_gt_network,
    per_tf_edge_sets,
    spearman_label,
)

TEXT_SIZE = 13
TITLE_SIZE = 14
METHOD_ORDER = ["emb500", "embhidden500", "att500"]
PAIR_ORDER = [
    ("emb500", "embhidden500"),
    ("embhidden500", "att500"),
    ("emb500", "att500"),
]


def load_all_method_metrics(
    dataset: str,
    methods: List[str],
    gt_path: Path,
    pred_root: Path,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Set[str]]]:
    gt_gene1, gt_all, gt_n, _, gt_by_tf = load_gt_network(gt_path)
    per_method: Dict[str, pd.DataFrame] = {}
    for m in methods:
        pred_path = resolve_pred_path(m, pred_root, dataset)
        if not pred_path.is_file():
            raise FileNotFoundError(pred_path)
        pred_df = filter_prediction(pred_path, gt_gene1, gt_all, gt_n)
        met = per_tf_edge_sets(pred_df, gt_by_tf)
        met = met.rename(
            columns={
                c: f"{c}_{m}"
                for c in met.columns
                if c != "TF"
            }
        )
        per_method[m] = met

    wide = per_method[methods[0]][["TF"]].copy()
    for m in methods:
        wide = wide.merge(per_method[m], on="TF", how="outer")
    return wide, per_method, gt_by_tf


def method_summary_row(wide: pd.DataFrame, method: str) -> dict:
    sub = wide.dropna(subset=[f"jaccard_{method}", f"gt_outdegree_{method}"])
    if sub.empty:
        return {"method": method}
    rho, p = stats.spearmanr(sub[f"gt_outdegree_{method}"], sub[f"pred_outdegree_{method}"])
    return {
        "method": method,
        "n_tf": int(len(sub)),
        "mean_jaccard": float(sub[f"jaccard_{method}"].mean()),
        "median_jaccard": float(sub[f"jaccard_{method}"].median()),
        "mean_f1": float(sub[f"f1_{method}"].mean()),
        "mean_precision": float(sub[f"precision_{method}"].mean()),
        "mean_recall": float(sub[f"recall_{method}"].mean()),
        "outdegree_spearman": float(rho),
        "outdegree_spearman_p": float(p),
    }


def plot_method_summary_bars(summary: pd.DataFrame, dataset: str, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    colors = [method_color(m) for m in summary["method"]]
    labels = [method_label(m) for m in summary["method"]]
    x = np.arange(len(summary))

    metrics = [
        ("mean_jaccard", "Mean per-TF Jaccard", (0, 1)),
        ("mean_f1", "Mean per-TF F1", (0, 1)),
        ("outdegree_spearman", "Out-degree Spearman ρ\n(pred vs GT)", (-0.05, 1.05)),
    ]
    for ax, (col, ylab, ylim) in zip(axes, metrics):
        vals = summary[col].to_numpy()
        bars = ax.bar(x, vals, color=colors, edgecolor="#444", linewidth=0.6, width=0.62)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=TEXT_SIZE)
        ax.set_ylabel(ylab, fontsize=TEXT_SIZE - 1)
        ax.set_ylim(*ylim)
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.02 * (ylim[1] - ylim[0]),
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=TEXT_SIZE - 2,
            )
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        f"{dataset} — scGPT extraction methods (per-TF aggregate)",
        fontsize=TITLE_SIZE,
        y=1.02,
    )
    fig.tight_layout()
    stem = out_dir / f"{dataset}_compare_01_method_summary"
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {stem}.png/.pdf")


def plot_outdegree_facets(wide: pd.DataFrame, dataset: str, methods: List[str], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 4.5), sharex=True, sharey=True)
    if len(methods) == 1:
        axes = [axes]

    all_gt = []
    all_pred = []
    for m in methods:
        all_gt.extend(wide[f"gt_outdegree_{m}"].dropna().tolist())
        all_pred.extend(wide[f"pred_outdegree_{m}"].dropna().tolist())
    lim = max(all_gt + all_pred + [1]) * 1.06

    for ax, m in zip(axes, methods):
        sub = wide.dropna(subset=[f"gt_outdegree_{m}", f"pred_outdegree_{m}"])
        sc = ax.scatter(
            sub[f"gt_outdegree_{m}"],
            sub[f"pred_outdegree_{m}"],
            c=sub[f"jaccard_{m}"],
            cmap="YlGnBu",
            vmin=0,
            vmax=1,
            s=22,
            edgecolors="#333",
            linewidths=0.25,
            alpha=0.85,
        )
        ax.plot([0, lim], [0, lim], ls="--", color=model_color("STRING"), lw=1.0)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(method_label(m), fontsize=TEXT_SIZE)
        ax.text(
            0.05,
            0.95,
            spearman_label(
                sub[f"gt_outdegree_{m}"].to_numpy(),
                sub[f"pred_outdegree_{m}"].to_numpy(),
            ),
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#CCC", alpha=0.9),
        )
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Predicted out-degree")
    for ax in axes:
        ax.set_xlabel("GT out-degree (STRING)")

    fig.subplots_adjust(right=0.88, wspace=0.28)
    cbar_ax = fig.add_axes([0.90, 0.18, 0.015, 0.62])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label("Per-TF Jaccard", fontsize=TEXT_SIZE - 1)
    fig.suptitle(f"{dataset} — Out-degree agreement across methods", fontsize=TITLE_SIZE, y=1.02)
    stem = out_dir / f"{dataset}_compare_02_outdegree_facets"
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {stem}.png/.pdf")


def plot_jaccard_heatmap_methods(
    wide: pd.DataFrame,
    dataset: str,
    methods: List[str],
    out_dir: Path,
    top_n: int,
    min_gt_degree: int,
) -> None:
    gt_col = f"gt_outdegree_{methods[0]}"
    pool = wide[wide[gt_col] >= min_gt_degree].copy()
    pool = pool.sort_values(gt_col, ascending=False).head(int(top_n))
    if pool.empty:
        return

    mat = np.column_stack([pool[f"jaccard_{m}"].to_numpy() for m in methods])
    tf_labels = pool["TF"].tolist()
    col_labels = [method_label(m) for m in methods]

    fig, ax = plt.subplots(figsize=(5.2, max(5, 0.28 * len(tf_labels))))
    cmap = LinearSegmentedColormap.from_list(
        "jac_cmp",
        [(1.0, 1.0, 1.0), plt.matplotlib.colors.to_rgb(model_color("scGPT"))],
        N=256,
    )
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(col_labels, fontsize=TEXT_SIZE)
    ax.set_yticks(np.arange(len(tf_labels)))
    ax.set_yticklabels(tf_labels, fontsize=TEXT_SIZE - 2)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if v > 0.55 else "#222",
                )
    ax.set_title(f"{dataset} — Per-TF Jaccard across methods\n(top-{len(tf_labels)} GT hubs)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Jaccard")
    fig.tight_layout()
    stem = out_dir / f"{dataset}_compare_03_jaccard_tf_x_method"
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {stem}.png/.pdf")


def plot_jaccard_paired_scatter(wide: pd.DataFrame, dataset: str, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, (ma, mb) in zip(axes, PAIR_ORDER):
        ca, cb = f"jaccard_{ma}", f"jaccard_{mb}"
        sub = wide.dropna(subset=[ca, cb])
        x = sub[ca].to_numpy()
        y = sub[cb].to_numpy()
        ax.scatter(x, y, s=28, alpha=0.55, c=method_color(ma), edgecolors="#333", linewidths=0.3)
        ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(method_label(ma))
        ax.set_ylabel(method_label(mb))
        if len(sub) >= 3:
            rho, p = stats.spearmanr(x, y)
            ax.set_title(f"ρ={rho:.2f}, p={p:.2e}, n={len(sub)}", fontsize=TEXT_SIZE - 1)
        n_up = int((y > x + 1e-9).sum())
        n_dn = int((y < x - 1e-9).sum())
        ax.text(
            0.04,
            0.96,
            f"{method_label(mb)} better: {n_up}\n{method_label(ma)} better: {n_dn}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCC", alpha=0.92),
        )
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(f"{dataset} — Per-TF Jaccard concordance between methods", fontsize=TITLE_SIZE, y=1.02)
    fig.tight_layout()
    stem = out_dir / f"{dataset}_compare_04_jaccard_paired"
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {stem}.png/.pdf")


def plot_method_disagreement_bars(
    wide: pd.DataFrame,
    dataset: str,
    methods: List[str],
    out_dir: Path,
    top_n: int,
) -> None:
    jac_cols = [f"jaccard_{m}" for m in methods]
    sub = wide.dropna(subset=jac_cols).copy()
    if sub.empty:
        return
    mat = sub[jac_cols].to_numpy()
    sub["jaccard_best"] = mat.max(axis=1)
    sub["jaccard_worst"] = mat.min(axis=1)
    sub["jaccard_spread"] = sub["jaccard_best"] - sub["jaccard_worst"]
    sub["best_method"] = [methods[i] for i in mat.argmax(axis=1)]
    sub = sub.sort_values("jaccard_spread", ascending=False).head(int(top_n))

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.32 * len(sub))))
    y = np.arange(len(sub))
    width = 0.26
    for i, m in enumerate(methods):
        offset = (i - 1) * width
        ax.barh(
            y + offset,
            sub[f"jaccard_{m}"],
            height=width * 0.92,
            label=method_label(m),
            color=method_color(m),
            edgecolor="#444",
            linewidth=0.4,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(sub["TF"], fontsize=TEXT_SIZE - 1)
    ax.set_xlabel("Per-TF target-set Jaccard")
    ax.set_xlim(0, 1.05)
    ax.legend(loc="lower right", frameon=False, fontsize=TEXT_SIZE - 1)
    ax.set_title(
        f"{dataset} — TFs with largest cross-method disagreement\n"
        f"(spread = best − worst Jaccard)",
        fontsize=TITLE_SIZE - 1,
    )
    ax.invert_yaxis()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    stem = out_dir / f"{dataset}_compare_05_method_disagreement"
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {stem}.png/.pdf")
    return sub


def plot_hub_ego_method_rows(
    hub_tfs: List[str],
    dataset: str,
    methods: List[str],
    per_method_preds: Dict[str, pd.DataFrame],
    gt_by_tf: Dict[str, Set[str]],
    out_dir: Path,
    max_targets: int,
    file_stem: str | None = None,
    suptitle: str | None = None,
) -> None:
    """Each hub TF: one row × 3 method columns of ego networks."""
    n_tf = len(hub_tfs)
    n_m = len(methods)
    if n_tf == 0:
        return

    from tf_static.plot_tf_static_three import _ego_graph_for_tf
    import matplotlib.patches as mpatches
    import networkx as nx

    fig, axes = plt.subplots(
        n_tf,
        n_m,
        figsize=(4.2 * n_m, 4.0 * n_tf),
        squeeze=False,
    )

    for i, tf in enumerate(hub_tfs):
        for j, m in enumerate(methods):
            ax = axes[i, j]
            pred_df = per_method_preds[m]
            G, est = _ego_graph_for_tf(tf, pred_df, gt_by_tf, max_targets)
            tp_n = sum(1 for s in est.values() if s == "TP")
            fp_n = sum(1 for s in est.values() if s == "FP")
            fn_n = sum(1 for s in est.values() if s == "FN")

            pos = nx.spring_layout(G, seed=42, k=1.8 / max(G.number_of_nodes(), 1) ** 0.5)
            if tf in pos:
                pos[tf] = np.array([0.0, 0.0])
            ax.set_aspect("equal")
            ax.axis("off")

            from tf_static.plot_tf_static_three import EDGE_COLORS, NODE_TF, NODE_TARGET

            for u, v, data in G.edges(data=True):
                st = data.get("status", "FP")
                color = EDGE_COLORS.get(st, "#888")
                style = "dashed" if st == "FN" else "solid"
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                ax.annotate(
                    "",
                    xy=(x1, y1),
                    xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=color,
                        lw=1.5 if st == "TP" else 1.0,
                        linestyle=style,
                        alpha=0.9 if st != "FN" else 0.5,
                        shrinkA=10,
                        shrinkB=10,
                    ),
                )
            for node, data in G.nodes(data=True):
                x, y = pos[node]
                if data.get("ntype") == "tf":
                    ax.scatter([x], [y], s=400, c=NODE_TF, edgecolors="#333", linewidths=1, zorder=5)
                    ax.text(x, y, node, ha="center", va="center", fontsize=9, fontweight="bold", color="white", zorder=6)
                else:
                    ax.scatter([x], [y], s=120, c=NODE_TARGET, edgecolors="#666", linewidths=0.5, zorder=4)

            if i == 0:
                ax.set_title(method_label(m), fontsize=TEXT_SIZE, color=method_color(m), pad=8)
            if j == 0:
                ax.text(
                    -0.08,
                    0.5,
                    tf,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=TEXT_SIZE,
                    fontweight="bold",
                )
            ax.text(
                0.02,
                0.02,
                f"TP{tp_n} FP{fp_n} FN{fn_n}",
                transform=ax.transAxes,
                fontsize=8,
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85),
            )

    legend_elems = [
        mpatches.Patch(color="#4C9F70", label="TP"),
        mpatches.Patch(color=model_color("scPrint"), label="FP"),
        mpatches.Patch(color="#9AA0A6", label="FN"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        suptitle or f"{dataset} — Same TF, three extraction methods",
        fontsize=TITLE_SIZE,
        y=1.01,
    )
    fig.tight_layout(rect=[0.04, 0.04, 1, 0.98])
    stem = out_dir / (file_stem or f"{dataset}_compare_06_hub_ego_x_method")
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {stem}.png/.pdf")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare scGPT TF metrics across extraction methods")
    p.add_argument("--dataset", default="hESC")
    p.add_argument(
        "--methods",
        nargs="+",
        default=METHOD_ORDER,
        choices=list(METHOD_PRED),
    )
    p.add_argument("--gt-dir", type=Path, default=Path("/mnt/10T/yzn/benchmark_GRN/input_process/STRING"))
    p.add_argument("--pred-root", type=Path, default=Path("/mnt/10T/yzn/benchmark_GRN/evl_omipath"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--heatmap-top", type=int, default=30)
    p.add_argument("--disagree-top", type=int, default=15)
    p.add_argument("--top-hub", type=int, default=3, help="Hub TFs for cross-method ego rows")
    p.add_argument("--min-gt-degree", type=int, default=8)
    p.add_argument("--max-ego-targets", type=int, default=22)
    p.add_argument("--skip-ego", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    apply_style()
    methods = [m for m in METHOD_ORDER if m in args.methods]
    if len(methods) < 2:
        raise ValueError("Need at least 2 methods to compare")

    out_dir = args.out if args.out is not None else FIG3 / "tf_static" / "output" / args.dataset / "compare_methods"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_path = args.gt_dir / f"{args.dataset}_processed-network.csv"
    print(f"Dataset: {args.dataset}")
    print(f"Methods: {methods}")
    print(f"Out: {out_dir}")

    wide, per_method, gt_by_tf = load_all_method_metrics(
        args.dataset, methods, gt_path, args.pred_root
    )
    wide.to_csv(out_dir / f"{args.dataset}_per_tf_metrics_wide.csv", index=False)

    summary = pd.DataFrame([method_summary_row(wide, m) for m in methods])
    summary.to_csv(out_dir / f"{args.dataset}_method_summary.csv", index=False)
    print(summary.to_string(index=False))

    plot_method_summary_bars(summary, args.dataset, out_dir)
    plot_outdegree_facets(wide, args.dataset, methods, out_dir)
    plot_jaccard_heatmap_methods(
        wide, args.dataset, methods, out_dir, args.heatmap_top, args.min_gt_degree
    )
    plot_jaccard_paired_scatter(wide, args.dataset, out_dir)
    plot_method_disagreement_bars(wide, args.dataset, methods, out_dir, args.disagree_top)

    gt_col = f"gt_outdegree_{methods[0]}"
    hub_tfs = (
        wide[wide[gt_col] >= args.min_gt_degree]
        .sort_values(gt_col, ascending=False)["TF"]
        .head(int(args.top_hub))
        .tolist()
    )

    if not args.skip_ego and hub_tfs:
        gt_gene1, gt_all, gt_n, _, _ = load_gt_network(gt_path)
        pred_by_method = {}
        for m in methods:
            pred_path = resolve_pred_path(m, args.pred_root, args.dataset)
            pred_by_method[m] = filter_prediction(pred_path, gt_gene1, gt_all, gt_n)
        plot_hub_ego_method_rows(
            hub_tfs,
            args.dataset,
            methods,
            pred_by_method,
            gt_by_tf,
            out_dir,
            args.max_ego_targets,
        )

    print("Done.")


if __name__ == "__main__":
    main()
