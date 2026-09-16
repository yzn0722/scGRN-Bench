#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig2 attention head × fig4 动态方向：对比非 CHIP 靶 top30% 基因的方向错率。

思路
----
- 基线 ``scgpt_iter``：``pre_scgpt/.../gene_result.csv`` 的迭代重建方向（模型默认多头注意力）。
- ``grn_mean8`` / ``grn_max8`` / ``grn_best_head``：用 fig2 导出的 GRN，按上游 TF 的
  ``delta_pred``（scGPT 已给出的变化量）加权投票预测靶基因方向。
- ``grn_oracle_tf``（可选）：每个 TF 选用 AUPRC 最高的 head（上界，需 CHIP 标签）。

仅当预测 GRN 中存在指向该基因的 TF→target 边时才用 GRN 方向，否则退回 scGPT。

输出
----
- ``error_biology/head_compare/nonchip_direction_error_bars.png``
- ``error_biology/head_compare/nonchip_direction_error_heatmap.png``
- ``error_biology/head_compare/nonchip_coverage_vs_error_scatter.png``  (覆盖率 × 有边子集方向错)
- ``error_biology/head_compare/head_compare_metrics.csv``

示例
----
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig4

  # 六数据集（默认 scGPT gene_result + evl_omipath multihead TSV）
  python3 compare_head_dynamic_nonchip.py --all-datasets

  # 仅 hESC
  python3 compare_head_dynamic_nonchip.py --dataset hESC

  # 指定 fig2 AUPRC 汇总以自动选 best_head
  python3 compare_head_dynamic_nonchip.py --dataset hESC \\
    --auprc-summary /mnt/10T/yzn/scGRN-Bench/FBplot/fig2/plot/output/head_chip_auprc/scgpt_hESC/scgpt_hESC_auprc_summary.csv
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
FIG2_PLOT = SCRIPT_DIR.parent / "fig2" / "plot"
if str(FIG2_PLOT) not in sys.path:
    sys.path.insert(0, str(FIG2_PLOT))

try:
    from fig4_palette import apply_fig4_style, model_color
except ImportError:

    def apply_fig4_style() -> None:
        plt.rcParams.update({"font.size": 12})

    def model_color(_: str, default: str = "#808080") -> str:
        return default

try:
    import analyze_attention_heads as ath
    from eval_heads_chip_auprc import (
        chip_gene_sets_from_gt,
        filter_pred_aupr_style,
        fuse_predictions,
        load_head_pred,
        read_chip_gt,
    )
except ImportError as e:
    raise SystemExit(
        "Need fig2/plot on PYTHONPATH (analyze_attention_heads, eval_heads_chip_auprc). "
        f"Missing: {e}"
    )

DEFAULT_GENE_DIR = Path("/mnt/10T/yzn/benchmark_GRN/pre_scgpt/results_multidataset_pseudotime_227")
DEFAULT_CHIP_DIR = Path("/mnt/10T/yzn/benchmark_GRN/input_process/CHIP")
DEFAULT_HEAD_ROOT = Path("/mnt/10T/yzn/benchmark_GRN/evl_omipath/output_att500/scgpt_multihead")
DEFAULT_AUPRC_DIR = SCRIPT_DIR.parent / "fig2" / "plot" / "output" / "head_chip_auprc"
DEFAULT_OUTDIR = SCRIPT_DIR / "error_biology" / "head_compare"

TOP_PERCENT = 0.3
LOAD_MAX_EDGES = 100_000
DATASETS_DEFAULT = ["hESC", "hHep", "mDC", "mHSC-E", "mHSC-GM", "mHSC-L"]

STRATEGY_ORDER = [
    "scgpt_iter",
    "grn_mean8",
    "grn_max8",
    "grn_best_head",
    "grn_oracle_tf",
]
STRATEGY_LABELS = {
    "scgpt_iter": "scGPT iter.\n(default multi-head)",
    "grn_mean8": "GRN mean8\n(cascade Δ_pred)",
    "grn_max8": "GRN max8\n(cascade Δ_pred)",
    "grn_best_head": "GRN best head\n(cascade Δ_pred)",
    "grn_oracle_tf": "GRN oracle/TF\n(upper bound)",
}
STRATEGY_COLORS = {
    "scgpt_iter": model_color("scGPT"),
    "grn_mean8": model_color("STRING"),
    "grn_max8": "#9E9E9E",
    "grn_best_head": model_color("scFoundation"),
    "grn_oracle_tf": model_color("LangCell"),
}


# ---------------------------------------------------------------------------
# Shared with plot_dynamic_error_biology
# ---------------------------------------------------------------------------
def load_chip_tf_targets(chip_network_csv: Path) -> Set[str]:
    net = pd.read_csv(chip_network_csv)
    g2 = net.columns[1] if "Gene2" not in net.columns else "Gene2"
    return set(net[g2].astype(str).str.strip().unique())


def select_top30(df: pd.DataFrame, top_percent: float = TOP_PERCENT) -> pd.DataFrame:
    df = df.copy()
    df["abs_delta_true"] = pd.to_numeric(df["delta_true"], errors="coerce").abs()
    n_top = max(1, int(np.ceil(top_percent * len(df))))
    return df.nlargest(n_top, "abs_delta_true")


def direction_error_rate(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    return float((df["dir_correct"] == 0).mean())


# ---------------------------------------------------------------------------
# Directed GRN fusion (preserve TF → target)
# ---------------------------------------------------------------------------
def fuse_directed(
    per_head: Dict[int, pd.DataFrame],
    mode: str,
) -> pd.DataFrame:
    """按有向边 (Gene1→Gene2) 聚合多 head。"""
    acc: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for df in per_head.values():
        if df is None or df.empty:
            continue
        for row in df.itertuples(index=False):
            key = (str(row.Gene1), str(row.Gene2))
            acc[key].append(float(row.EdgeWeight))
    rows = []
    for (a, b), ws in acc.items():
        val = float(np.mean(ws)) if mode == "mean" else float(np.max(ws))
        rows.append({"Gene1": a, "Gene2": b, "EdgeWeight": val})
    return pd.DataFrame(rows)


def discover_head_files(head_root: Path, dataset: str, model_tag: str = "scgpt") -> Dict[int, Path]:
    patterns = [
        head_root / f"{model_tag}_{dataset}_head*.tsv",
        head_root / f"{model_tag}_{dataset}_head*.csv",
        head_root / f"scGPT_{dataset}_head*.tsv",
    ]
    out: Dict[int, Path] = {}
    for pat in patterns:
        for fp in sorted(pat.parent.glob(pat.name)):
            m = re.search(r"head(\d+)", fp.stem, re.I)
            if m:
                out[int(m.group(1))] = fp
    return out


def filter_pred_dynamic_style(
    pred: pd.DataFrame,
    chip_gt: pd.DataFrame,
    eval_genes: Set[str],
) -> pd.DataFrame:
    """
    动态任务用：Gene1 必须是 CHIP 中的 TF；Gene2 只需在 gene_result 评估基因集内。
    （不过滤 Gene2∈CHIP，否则非 CHIP 靶基因永远没有 GRN 入边。）
    """
    tfs, _ = chip_gene_sets_from_gt(chip_gt)
    out = pred[pred["Gene1"].isin(tfs) & pred["Gene2"].isin(eval_genes)].copy()
    return out.sort_values("EdgeWeight", ascending=False).reset_index(drop=True)


def load_per_head_grns(
    head_root: Path,
    dataset: str,
    chip_gt: Path,
    eval_genes: Set[str],
    *,
    load_max_edges: int = LOAD_MAX_EDGES,
) -> Dict[int, pd.DataFrame]:
    gt = read_chip_gt(chip_gt)
    tfs, genes = chip_gene_sets_from_gt(gt)
    universe = tfs | genes | eval_genes

    head_files = discover_head_files(head_root, dataset)
    if not head_files:
        raise FileNotFoundError(f"No head TSV under {head_root} for {dataset}")

    per_head: Dict[int, pd.DataFrame] = {}
    for h, fp in sorted(head_files.items()):
        raw = load_head_pred(fp, universe, load_max_edges)
        per_head[h] = filter_pred_dynamic_style(raw, gt, eval_genes)
    return per_head


def pick_best_head_from_summary(summary_csv: Path) -> int:
    df = pd.read_csv(summary_csv)
    sub = df[df["strategy"].str.match(r"^head\d+$", na=False)].copy()
    if sub.empty:
        raise ValueError(f"No head* rows in {summary_csv}")
    best = sub.loc[sub["AUPRC_micro"].astype(float).idxmax(), "strategy"]
    return int(str(best).replace("head", ""))


def build_oracle_grn(
    per_head: Dict[int, pd.DataFrame],
    per_tf_csv: Path,
) -> pd.DataFrame:
    """每个 TF 只保留 AUPRC 最高的 head 的边。"""
    pt = pd.read_csv(per_tf_csv)
    pt = pt[pt["strategy"].str.match(r"^head\d+$", na=False)]
    best_h: Dict[str, int] = {}
    for tf, sub in pt.groupby("TF"):
        row = sub.loc[sub["AUPRC"].astype(float).idxmax()]
        strat = str(row["strategy"])
        best_h[str(tf)] = int(strat.replace("head", ""))

    frames = []
    for h, df in per_head.items():
        if df.empty:
            continue
        tfs_use = [tf for tf, hh in best_h.items() if hh == h]
        if not tfs_use:
            continue
        sub = df[df["Gene1"].isin(tfs_use)]
        if not sub.empty:
            frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=["Gene1", "Gene2", "EdgeWeight"])
    return pd.concat(frames, ignore_index=True)


def build_incoming_index(grn: pd.DataFrame) -> Dict[str, List[Tuple[str, float]]]:
    idx: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for row in grn.itertuples(index=False):
        idx[str(row.Gene2)].append((str(row.Gene1), float(row.EdgeWeight)))
    return idx


def predict_dir_from_grn(
    genes_df: pd.DataFrame,
    incoming: Dict[str, List[Tuple[str, float]]],
    *,
    signal_col: str = "delta_pred",
) -> Tuple[pd.Series, pd.Series]:
    """
    返回 (dir_pred_grn, has_grn_edge)。
    dir 为 'Up'/'Down' 或 NaN（无边则 NaN）。
    """
    sig = genes_df.set_index("gene")[signal_col].to_dict()
    dirs = {}
    covered = {}
    for g in genes_df["gene"].astype(str):
        edges = incoming.get(g, [])
        score = 0.0
        wsum = 0.0
        for tf, w in edges:
            if tf not in sig or not np.isfinite(sig[tf]):
                continue
            score += w * float(sig[tf])
            wsum += abs(w)
        if wsum < 1e-12 or not edges:
            dirs[g] = np.nan
            covered[g] = False
        else:
            dirs[g] = "Up" if score > 0 else ("Down" if score < 0 else "Up")
            covered[g] = True
    return pd.Series(dirs), pd.Series(covered)


def apply_strategy(
    top: pd.DataFrame,
    strategy: str,
    grns: Dict[str, pd.DataFrame],
    *,
    signal_col: str = "delta_pred",
) -> pd.DataFrame:
    out = top.copy()
    if strategy == "scgpt_iter":
        out["dir_pred_new"] = out["dir_pred"]
        out["grn_covered"] = False
        out["dir_correct_new"] = out["dir_correct"]
        return out

    grn = grns.get(strategy)
    if grn is None or grn.empty:
        out["dir_pred_new"] = out["dir_pred"]
        out["grn_covered"] = False
        out["dir_correct_new"] = out["dir_correct"]
        return out

    incoming = build_incoming_index(grn)
    dir_new, covered = predict_dir_from_grn(out, incoming, signal_col=signal_col)
    out["dir_pred_new"] = out["gene"].map(dir_new)
    out["grn_covered"] = out["gene"].map(covered).fillna(False)
    # 无边：退回 scGPT
    fallback = out["dir_pred_new"].isna()
    out.loc[fallback, "dir_pred_new"] = out.loc[fallback, "dir_pred"]
    out["dir_correct_new"] = (
        out["dir_true"].astype(str).str.strip().str.lower()
        == out["dir_pred_new"].astype(str).str.strip().str.lower()
    ).astype(int)
    return out


# ---------------------------------------------------------------------------
# Metrics & plots
# ---------------------------------------------------------------------------
def summarize_nonchip(
    dataset: str,
    strategy: str,
    df: pd.DataFrame,
) -> Dict:
    non = df[~df["is_chip_target"]]
    cov = df[df["grn_covered"]]
    non_cov = non[non["grn_covered"]]
    return {
        "dataset": dataset,
        "strategy": strategy,
        "n_top30": len(df),
        "n_nonchip": len(non),
        "n_nonchip_grn_covered": len(non_cov),
        "frac_nonchip_grn_covered": len(non_cov) / max(len(non), 1),
        "dir_error_all_top30": direction_error_rate(df),
        "dir_error_nonchip": direction_error_rate(non),
        "dir_error_nonchip_covered": direction_error_rate(non_cov),
        "dir_error_chip_target": direction_error_rate(df[df["is_chip_target"]]),
    }


def plot_bars(metrics: pd.DataFrame, outpath: Path, strategies: List[str]) -> None:
    apply_fig4_style()
    sub = metrics[metrics["strategy"].isin(strategies)].copy()
    datasets = [d for d in DATASETS_DEFAULT if d in sub["dataset"].unique()]
    if not datasets:
        datasets = sorted(sub["dataset"].unique())

    n_ds = len(datasets)
    n_st = len(strategies)
    fig, axes = plt.subplots(1, 2, figsize=(6.5 + 1.1 * n_ds, 4.8), facecolor="white")

    panel_specs = [
        ("dir_error_nonchip", "Non-CHIP targets\n(top 30% dynamic genes)"),
        ("dir_error_nonchip_covered", "Non-CHIP with GRN edge\n(cascade from scGPT Δ_pred)"),
    ]

    x = np.arange(n_ds)
    w = 0.8 / max(n_st, 1)

    for ax, (col, title) in zip(axes, panel_specs):
        for j, st in enumerate(strategies):
            vals = []
            for ds in datasets:
                row = sub[(sub["dataset"] == ds) & (sub["strategy"] == st)]
                vals.append(float(row[col].iloc[0]) if len(row) else np.nan)
            offset = (j - (n_st - 1) / 2) * w
            ax.bar(
                x + offset,
                vals,
                width=w,
                label=STRATEGY_LABELS.get(st, st),
                color=STRATEGY_COLORS.get(st, "#888888"),
                edgecolor="white",
                linewidth=0.5,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=20, ha="right")
        ax.set_ylim(0, min(1.0, np.nanmax(sub[col]) * 1.25 + 0.05))
        ax.set_ylabel("Direction error rate", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.axhline(0, color="k", linewidth=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=9)
    fig.suptitle(
        "fig2 GRN head strategies vs scGPT iterative baseline",
        fontsize=14,
        y=1.08,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_heatmap(metrics: pd.DataFrame, outpath: Path, strategies: List[str]) -> None:
    apply_fig4_style()
    sub = metrics[metrics["strategy"].isin(strategies)]
    pivot = sub.pivot_table(
        index="strategy",
        columns="dataset",
        values="dir_error_nonchip",
        aggfunc="first",
    )
    ds_order = [d for d in DATASETS_DEFAULT if d in pivot.columns]
    pivot = pivot.reindex(strategies)[ds_order]

    fig, ax = plt.subplots(
        figsize=(0.9 + 1.2 * len(ds_order), 0.55 + 0.55 * len(strategies)),
        facecolor="white",
    )
    vmax = max(0.35, float(np.nanmax(pivot.values)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(len(ds_order)))
    ax.set_xticklabels(ds_order, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(strategies)))
    ax.set_yticklabels([STRATEGY_LABELS.get(s, s).replace("\n", " ") for s in strategies], fontsize=10)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(
                    j, i, f"{v:.0%}",
                    ha="center", va="center", fontsize=10,
                    color="white" if v > vmax * 0.55 else "#222",
                )
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Direction error (non-CHIP)")
    ax.set_title("Non-CHIP target direction error by GRN strategy", fontsize=13, pad=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_coverage_vs_error_scatter(
    metrics: pd.DataFrame,
    outpath: Path,
    strategies: List[str],
) -> None:
    """
    分面散点：x = 非靶 GRN 覆盖率，y = 有边非靶的方向错率。
    虚线 = 该数据集 scGPT 基线（全体非靶方向错），便于看出「覆盖-收益」权衡。
    """
    apply_fig4_style()
    grn_strats = [s for s in strategies if s != "scgpt_iter"]
    if not grn_strats:
        return

    datasets = [d for d in DATASETS_DEFAULT if d in metrics["dataset"].unique()]
    if not datasets:
        datasets = sorted(metrics["dataset"].unique())

    n = len(datasets)
    ncols = 3 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 3.8 * nrows),
        facecolor="white",
        squeeze=False,
    )

    for idx, ds in enumerate(datasets):
        ax = axes.flat[idx]
        sub_ds = metrics[metrics["dataset"] == ds]
        base_row = sub_ds[sub_ds["strategy"] == "scgpt_iter"]
        baseline = (
            float(base_row["dir_error_nonchip"].iloc[0]) if len(base_row) else np.nan
        )
        if np.isfinite(baseline):
            ax.axhline(
                baseline,
                color=model_color("scGPT"),
                linestyle="--",
                linewidth=1.4,
                alpha=0.75,
                label=f"scGPT all non-CHIP ({baseline:.0%})",
            )

        for st in grn_strats:
            row = sub_ds[sub_ds["strategy"] == st]
            if row.empty:
                continue
            r = row.iloc[0]
            cov = float(r["frac_nonchip_grn_covered"])
            n_cov = int(r["n_nonchip_grn_covered"])
            err = float(r["dir_error_nonchip_covered"])
            if not np.isfinite(err) or n_cov < 1:
                # 无覆盖：画在 x=0 处，y 用基线或略上方标注
                ax.scatter(
                    [cov],
                    [baseline if np.isfinite(baseline) else 0.5],
                    s=70,
                    c=STRATEGY_COLORS.get(st, "#888"),
                    marker="x",
                    linewidths=2,
                    alpha=0.55,
                    zorder=2,
                )
                ax.annotate(
                    f"{st.replace('grn_', '')}\nno edge",
                    (cov, baseline if np.isfinite(baseline) else 0.5),
                    fontsize=7,
                    ha="left",
                    va="bottom",
                    xytext=(4, 4),
                    textcoords="offset points",
                    color="#666666",
                )
                continue
            size = 40 + 35 * min(n_cov, 20)
            ax.scatter(
                cov,
                err,
                s=size,
                c=STRATEGY_COLORS.get(st, "#888"),
                edgecolors="white",
                linewidths=0.8,
                alpha=0.9,
                zorder=3,
                label=f"{st.replace('grn_', '')} (n={n_cov})",
            )
            ax.annotate(
                st.replace("grn_", ""),
                (cov, err),
                fontsize=8,
                ha="center",
                va="bottom",
                xytext=(0, 5),
                textcoords="offset points",
            )

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.02, min(1.0, max(0.55, baseline * 1.15 if np.isfinite(baseline) else 0.55)))
        ax.set_xlabel("GRN coverage (non-CHIP)", fontsize=10)
        ax.set_ylabel("Dir. error if covered", fontsize=10)
        ax.set_title(ds, fontsize=12, fontweight="600")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.0%}")
        )

    for j in range(len(datasets), nrows * ncols):
        axes.flat[j].set_visible(False)

    handles, labels = [], []
    for st in grn_strats:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=8,
                markerfacecolor=STRATEGY_COLORS.get(st, "#888"),
                markeredgecolor="white",
            )
        )
        labels.append(st.replace("grn_", ""))
    handles.append(
        plt.Line2D([0], [0], color=model_color("scGPT"), linestyle="--", linewidth=1.4)
    )
    labels.append("scGPT baseline (all non-CHIP)")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False, fontsize=9)
    fig.suptitle(
        "Non-CHIP: GRN coverage vs direction error (among covered genes only)\n"
        "Bubble size ∝ number of covered genes",
        fontsize=13,
        y=1.06,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare fig2 GRN heads on non-CHIP dynamic genes")
    p.add_argument("--gene-dir", type=Path, default=DEFAULT_GENE_DIR)
    p.add_argument("--chip-dir", type=Path, default=DEFAULT_CHIP_DIR)
    p.add_argument("--head-root", type=Path, default=DEFAULT_HEAD_ROOT)
    p.add_argument("--auprc-dir", type=Path, default=DEFAULT_AUPRC_DIR)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--all-datasets", action="store_true")
    p.add_argument("--top-percent", type=float, default=TOP_PERCENT)
    p.add_argument("--load-max-edges", type=int, default=LOAD_MAX_EDGES)
    p.add_argument("--no-oracle", action="store_true", help="Skip grn_oracle_tf (needs per-TF CSV)")
    p.add_argument(
        "--strategies",
        type=str,
        default="scgpt_iter,grn_mean8,grn_max8,grn_best_head",
        help="Comma-separated subset of strategies",
    )
    return p.parse_args()


def process_one_dataset(
    dataset: str,
    args: argparse.Namespace,
) -> List[Dict]:
    gene_csv = args.gene_dir / f"{dataset}_gene_result.csv"
    chip_net = args.chip_dir / f"{dataset}_chip_matched-network.csv"
    if not gene_csv.exists():
        print(f"[skip] missing {gene_csv}")
        return []
    if not chip_net.exists():
        print(f"[skip] missing {chip_net}")
        return []

    chip_targets = load_chip_tf_targets(chip_net)
    df = pd.read_csv(gene_csv)
    if "ene" in df.columns and "gene" not in df.columns:
        df = df.rename(columns={"ene": "gene"})
    top = select_top30(df, args.top_percent)
    top["is_chip_target"] = top["gene"].astype(str).str.strip().isin(chip_targets)

    eval_genes = set(top["gene"].astype(str).str.strip()) | set(df["gene"].astype(str).str.strip())
    per_head = load_per_head_grns(
        args.head_root,
        dataset,
        chip_net,
        eval_genes,
        load_max_edges=args.load_max_edges,
    )

    summary_csv = args.auprc_dir / f"scgpt_{dataset}" / f"scgpt_{dataset}_auprc_summary.csv"
    best_h = 3
    if summary_csv.exists():
        best_h = pick_best_head_from_summary(summary_csv)
        print(f"  [{dataset}] best_head from AUPRC = head{best_h}")
    else:
        print(f"  [{dataset}] no AUPRC summary; default best_head=head3")

    grns = {
        "grn_mean8": fuse_directed(per_head, "mean"),
        "grn_max8": fuse_directed(per_head, "max"),
        "grn_best_head": per_head.get(best_h, pd.DataFrame()),
    }

    if not args.no_oracle:
        per_tf = args.auprc_dir / f"scgpt_{dataset}" / f"scgpt_{dataset}_auprc_per_tf.csv"
        if per_tf.exists():
            grns["grn_oracle_tf"] = build_oracle_grn(per_head, per_tf)
        else:
            print(f"  [warn] no {per_tf}; skip grn_oracle_tf")

    strategy_list = [s.strip() for s in args.strategies.split(",") if s.strip()]
    if not args.no_oracle and "grn_oracle_tf" not in strategy_list:
        strategy_list.append("grn_oracle_tf")

    rows = []
    for st in strategy_list:
        if st == "scgpt_iter":
            evaluated = apply_strategy(top, st, {})
        else:
            evaluated = apply_strategy(top, st, grns)
        evaluated.to_csv(
            args.outdir / f"{dataset}_{st}_top30_predictions.csv",
            index=False,
        )
        rows.append(summarize_nonchip(dataset, st, evaluated))
        r = rows[-1]
        print(
            f"  {st:16s}  non-CHIP dir err={r['dir_error_nonchip']:.1%}  "
            f"(covered n={r['n_nonchip_grn_covered']}/{r['n_nonchip']}, "
            f"covered err={r['dir_error_nonchip_covered']:.1%})"
        )
    return rows


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.all_datasets:
        datasets = DATASETS_DEFAULT
    elif args.dataset:
        datasets = [args.dataset]
    else:
        datasets = DATASETS_DEFAULT

    all_rows: List[Dict] = []
    for ds in datasets:
        print(f"\n=== {ds} ===")
        all_rows.extend(process_one_dataset(ds, args))

    if not all_rows:
        raise SystemExit("No metrics collected.")

    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(args.outdir / "head_compare_metrics.csv", index=False)

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    if not args.no_oracle and "grn_oracle_tf" in metrics["strategy"].values:
        if "grn_oracle_tf" not in strategies:
            strategies.append("grn_oracle_tf")

    plot_bars(metrics, args.outdir / "nonchip_direction_error_bars.png", strategies)
    plot_heatmap(metrics, args.outdir / "nonchip_direction_error_heatmap.png", strategies)
    plot_coverage_vs_error_scatter(
        metrics,
        args.outdir / "nonchip_coverage_vs_error_scatter.png",
        strategies,
    )

    print(f"\nSaved metrics: {args.outdir / 'head_compare_metrics.csv'}")
    print(f"Saved: {args.outdir / 'nonchip_direction_error_bars.png'}")
    print(f"Saved: {args.outdir / 'nonchip_direction_error_heatmap.png'}")
    print(f"Saved: {args.outdir / 'nonchip_coverage_vs_error_scatter.png'}")


if __name__ == "__main__":
    main()
