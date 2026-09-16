#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top30% 动态基因上的误差分解与生物学解释（fig4）。

分层维度：
  - CHIP TF 靶基因 vs 非靶
  - 表达水平（mean_expr 三分位）
  - |delta_true| 强度（top30 内再三分位）

误差类型（与 scater.py 一致）：
  - well_predicted: 方向对且相对误差 <= 阈值
  - amplitude_wrong: 方向对但相对误差 > 阈值
  - strict_wrong: 方向错

输出 PNG + CSV；strict_wrong 基因做 GO/通路富集（Enrichr via gseapy）。

示例：
  python plot_dynamic_error_biology.py --dataset hESC
  python plot_dynamic_error_biology.py --all-datasets
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

try:
    from fig4_palette import apply_fig4_style, model_color
except ImportError:

    def apply_fig4_style() -> None:
        plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})

    def model_color(name: str, default: str = "#808080") -> str:
        return default


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TOP_PERCENT = 0.3
REL_ERR_THRESH = 0.5
EPS = 1e-8
N_EXPR_BINS = 3
N_DELTA_BINS = 3
MIN_GENES_ENRICH = 8
MIN_GENES_STRATUM = 3

ERROR_ORDER = ["well_predicted", "amplitude_wrong", "strict_wrong"]
ERROR_LABELS = {
    "well_predicted": "Well predicted",
    "amplitude_wrong": "Amplitude error",
    "strict_wrong": "Direction error",
}
ERROR_COLORS = {
    "well_predicted": "#4C9F70",
    "amplitude_wrong": model_color("Geneformer"),
    "strict_wrong": model_color("scPrint"),
}

DATASET_SPECIES = {
    "hESC": "human",
    "hHep": "human",
    "mDC": "mouse",
    "mHSC-E": "mouse",
    "mHSC-GM": "mouse",
    "mHSC-L": "mouse",
}

ENRICH_LIBS = {
    "human": [
        "GO_Biological_Process_2023",
        "KEGG_2021_Human",
        "Reactome_2022",
    ],
    "mouse": [
        "GO_Biological_Process_2023",
        "KEGG_2019_Mouse",
        "WikiPathways_2019_Mouse",
    ],
}

# 论文叙事相关的关键词高亮（在 GO 结果中标注）
FOCUS_KEYWORDS = [
    "differentiat",
    "development",
    "cell cycle",
    "division",
    "proliferat",
    "stem",
    "hematopoiet",
    "hepat",
    "embryo",
    "pluripot",
    "transcription",
    "signaling",
    "pathway",
    "metabol",
    "immune",
    "apoptosis",
]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GENE_DIR = Path("/mnt/10T/yzn/benchmark_GRN/pre_scgpt/results_multidataset_pseudotime_227")
DEFAULT_CHIP_DIR = Path("/mnt/10T/yzn/benchmark_GRN/input_process/CHIP")
DEFAULT_OUTDIR = SCRIPT_DIR / "error_biology"

from gene_result_io import load_gene_result  # noqa: E402

AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
TITLE_SIZE = 15
DPI = 300


# ---------------------------------------------------------------------------
# Data loading & labeling
# ---------------------------------------------------------------------------
def load_chip_tf_targets(chip_network_csv: Path) -> set[str]:
    """CHIP 有向边 Gene1(TF) -> Gene2(target)，返回靶基因集合。"""
    net = pd.read_csv(chip_network_csv)
    cols = {c.lower(): c for c in net.columns}
    g1 = cols.get("gene1", net.columns[0])
    g2 = cols.get("gene2", net.columns[1])
    targets = net[g2].astype(str).str.strip().unique()
    return set(targets)


def classify_errors(df: pd.DataFrame, rel_thresh: float = REL_ERR_THRESH) -> pd.DataFrame:
    df = df.copy()
    if "ene" in df.columns and "gene" not in df.columns:
        df = df.rename(columns={"ene": "gene"})
    df["mean_expr"] = (df["true_early_mean"] + df["true_late_mean"]) / 2.0
    df["abs_delta_true"] = df["delta_true"].abs()
    df["abs_error"] = (df["delta_true"] - df["delta_pred"]).abs()
    df["relative_error"] = df["abs_error"] / (df["abs_delta_true"] + EPS)

    df["error_type"] = "well_predicted"
    df.loc[df["dir_correct"] == 0, "error_type"] = "strict_wrong"
    mask_amp = (df["dir_correct"] == 1) & (df["relative_error"] > rel_thresh)
    df.loc[mask_amp, "error_type"] = "amplitude_wrong"
    return df


def select_top30(df: pd.DataFrame, top_percent: float = TOP_PERCENT) -> pd.DataFrame:
    n_top = max(1, int(np.ceil(top_percent * len(df))))
    return df.nlargest(n_top, "abs_delta_true").copy()


def tertile_labels(values: pd.Series, labels: Tuple[str, ...]) -> pd.Series:
    """等频三分位标签。"""
    if len(values) < N_EXPR_BINS * MIN_GENES_STRATUM:
        return pd.Series(["all"] * len(values), index=values.index)
    try:
        q = pd.qcut(values, q=len(labels), labels=labels, duplicates="drop")
        return q.astype(str)
    except ValueError:
        return pd.Series(["all"] * len(values), index=values.index)


def annotate_top30(
    df: pd.DataFrame,
    tf_targets: set[str],
) -> pd.DataFrame:
    df = df.copy()
    genes = df["gene"].astype(str).str.strip()
    df["is_tf_target"] = genes.isin(tf_targets)
    df["tf_group"] = np.where(df["is_tf_target"], "TF target", "Non-target")

    df["expr_bin"] = tertile_labels(
        df["mean_expr"],
        ("Low expr.", "Mid expr.", "High expr."),
    )
    df["delta_bin"] = tertile_labels(
        df["abs_delta_true"],
        ("Weak |Δ|", "Mid |Δ|", "Strong |Δ|"),
    )
    return df


def error_fraction_table(df: pd.DataFrame, stratum_col: str) -> pd.DataFrame:
    """每个 stratum 内三种误差的比例与计数。"""
    rows = []
    for level, sub in df.groupby(stratum_col, observed=True):
        n = len(sub)
        if n < MIN_GENES_STRATUM:
            continue
        counts = sub["error_type"].value_counts()
        for et in ERROR_ORDER:
            c = int(counts.get(et, 0))
            rows.append(
                {
                    "stratum": str(level),
                    "error_type": et,
                    "count": c,
                    "fraction": c / n,
                    "n_genes": n,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------
def run_enrichment(
    gene_list: List[str],
    species: str,
    outdir: Path,
    dataset: str,
) -> Optional[pd.DataFrame]:
    if len(gene_list) < MIN_GENES_ENRICH:
        return None
    try:
        import gseapy as gp
    except ImportError:
        print("  [warn] gseapy not installed; skip enrichment")
        return None

    libs = ENRICH_LIBS.get(species, ENRICH_LIBS["human"])
    bg = None  # Enrichr 使用库内背景
    frames = []
    for lib in libs:
        try:
            enr = gp.enrichr(
                gene_list=gene_list,
                gene_sets=lib,
                organism=species,
                outdir=None,
                cutoff=0.5,
            )
            if enr is None or enr.results is None or enr.results.empty:
                continue
            res = enr.results.copy()
            res["library"] = lib
            frames.append(res)
        except Exception as exc:
            print(f"  [warn] enrichr {lib}: {exc}")
    if not frames:
        return None

    full = pd.concat(frames, ignore_index=True)
    full["dataset"] = dataset
    if "Adjusted P-value" in full.columns:
        full["padj"] = pd.to_numeric(full["Adjusted P-value"], errors="coerce")
    elif "P-value" in full.columns:
        full["padj"] = pd.to_numeric(full["P-value"], errors="coerce")
    else:
        return None
    full = full.dropna(subset=["padj"])
    full = full.sort_values("padj")
    full.to_csv(outdir / f"{dataset}_strict_wrong_enrichment_all.csv", index=False)
    return full


def pick_top_terms(enr_df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """优先展示显著 + 含叙事关键词的 term。"""
    df = enr_df.copy()
    df["neglog10_padj"] = -np.log10(df["padj"].clip(lower=1e-300))
    term_col = "Term" if "Term" in df.columns else df.columns[0]
    df["focus"] = df[term_col].astype(str).str.lower().apply(
        lambda t: any(kw in t for kw in FOCUS_KEYWORDS)
    )
    # 每个 library 取若干，再全局排序
    picks = []
    for lib, sub in df.groupby("library"):
        sub = sub.sort_values(["focus", "neglog10_padj"], ascending=[False, False])
        picks.append(sub.head(max(4, top_n // 3)))
    out = pd.concat(picks, ignore_index=True)
    out = out.drop_duplicates(subset=[term_col]).sort_values(
        ["focus", "neglog10_padj"], ascending=[False, False]
    )
    return out.head(top_n)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_stratified_stacked(
    frac_tables: Dict[str, pd.DataFrame],
    dataset: str,
    outpath: Path,
    model_label: str = "scGPT",
) -> None:
    """三列 100% 堆叠条形图：TF / 表达 / |Δ|。"""
    apply_fig4_style()
    fig = plt.figure(figsize=(12, 4.2), facecolor="white")
    gs = GridSpec(1, 3, figure=fig, wspace=0.38)

    panels = [
        ("tf_group", "CHIP TF target status"),
        ("expr_bin", "Expression level (tertile)"),
        ("delta_bin", r"Dynamic strength $|\Delta_{true}|$ (tertile)"),
    ]

    for ax_idx, (col, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, ax_idx])
        tab = frac_tables.get(col)
        if tab is None or tab.empty:
            ax.set_visible(False)
            continue

        strata = list(dict.fromkeys(tab["stratum"].tolist()))
        x = np.arange(len(strata))
        bottom = np.zeros(len(strata))
        n_map = tab.groupby("stratum")["n_genes"].first().to_dict()

        for et in ERROR_ORDER:
            heights = []
            for s in strata:
                row = tab[(tab["stratum"] == s) & (tab["error_type"] == et)]
                heights.append(float(row["fraction"].iloc[0]) if len(row) else 0.0)
            heights = np.array(heights)
            ax.bar(
                x,
                heights,
                bottom=bottom,
                width=0.62,
                color=ERROR_COLORS[et],
                label=ERROR_LABELS[et],
                edgecolor="white",
                linewidth=0.6,
            )
            bottom += heights

        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=TICK_LABEL_SIZE)
        ax.set_xticks(x)
        ax.set_xticklabels(strata, fontsize=TICK_LABEL_SIZE, rotation=18, ha="right")
        ax.set_title(title, fontsize=TITLE_SIZE - 1, pad=8)
        if ax_idx == 0:
            ax.set_ylabel("Fraction of genes", fontsize=AXIS_LABEL_SIZE)
        for i, s in enumerate(strata):
            ax.text(
                i,
                1.02,
                f"n={n_map.get(s, 0)}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#444444",
            )

    handles = [
        mpatches.Patch(facecolor=ERROR_COLORS[e], label=ERROR_LABELS[e]) for e in ERROR_ORDER
    ]
    fig.suptitle(
        f"{dataset} — error composition in top {int(TOP_PERCENT*100)}% dynamic genes ({model_label})",
        fontsize=TITLE_SIZE + 1,
        y=1.05,
        fontweight="600",
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        fontsize=TICK_LABEL_SIZE,
    )
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_tf_comparison_all_datasets(
    summary: pd.DataFrame,
    outpath: Path,
) -> None:
    """多数据集：TF target vs non-target 的方向/幅度错误率对比（分组条形）。"""
    apply_fig4_style()
    sub = summary[summary["stratum_col"] == "tf_group"].copy()
    if sub.empty:
        return

    datasets = list(dict.fromkeys(sub["dataset"].tolist()))
    strata = ["TF target", "Non-target"]
    metrics = [
        ("strict_wrong", "Direction error", ERROR_COLORS["strict_wrong"]),
        ("amplitude_wrong", "Amplitude error", ERROR_COLORS["amplitude_wrong"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="white", sharey=True)
    x = np.arange(len(datasets))
    w = 0.34

    for ax, (metric, ylab, _) in zip(axes, metrics):
        for j, st in enumerate(strata):
            vals = []
            for ds in datasets:
                row = sub[
                    (sub["dataset"] == ds)
                    & (sub["stratum"] == st)
                    & (sub["error_type"] == metric)
                ]
                vals.append(float(row["fraction"].iloc[0]) if len(row) else 0.0)
            offset = (j - 0.5) * w
            bars = ax.bar(
                x + offset,
                vals,
                width=w,
                label=st,
                color=ERROR_COLORS[metric] if j == 0 else "#B0BEC5",
                alpha=0.92 if j == 0 else 0.55,
                edgecolor="white",
            )
            for b, v in zip(bars, vals):
                if v > 0.04:
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        v + 0.02,
                        f"{v:.0%}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=TICK_LABEL_SIZE)
        ax.set_ylim(0, min(1.0, sub["fraction"].max() * 1.35 + 0.05))
        ax.set_ylabel("Fraction of genes", fontsize=AXIS_LABEL_SIZE)
        ax.set_title(ylab, fontsize=TITLE_SIZE - 1)
        ax.legend(frameon=False, fontsize=10)

    fig.suptitle(
        f"TF-target vs non-target failures (top {int(TOP_PERCENT*100)}% dynamic genes)",
        fontsize=TITLE_SIZE,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_error_heatmap(
    summary: pd.DataFrame,
    metric: str,
    outpath: Path,
    title: str,
) -> None:
    """热图：行=分层，列=数据集，值=某类错误比例。"""
    apply_fig4_style()
    sub = summary[summary["error_type"] == metric].copy()
    if sub.empty:
        return

    sub["row_label"] = sub["stratum_col"].str.replace("_bin", "", regex=False) + ": " + sub["stratum"]
    pivot = sub.pivot_table(
        index="row_label",
        columns="dataset",
        values="fraction",
        aggfunc="first",
    )
    # 固定数据集顺序
    ds_order = [d for d in DATASET_SPECIES if d in pivot.columns]
    pivot = pivot[ds_order]

    fig, ax = plt.subplots(figsize=(1.1 + 1.4 * len(ds_order), 0.35 * len(pivot) + 2), facecolor="white")
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(0.5, pivot.values.max()))

    ax.set_xticks(np.arange(len(ds_order)))
    ax.set_xticklabels(ds_order, fontsize=TICK_LABEL_SIZE, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=TICK_LABEL_SIZE - 1)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(
                    j,
                    i,
                    f"{v:.0%}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if v > 0.35 else "#222222",
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Fraction", fontsize=11)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_enrichment_dot(
    enr_top: pd.DataFrame,
    dataset: str,
    outpath: Path,
) -> None:
    """GO/通路富集气泡图：x=-log10(padj), y=term, size=overlap ratio。"""
    if enr_top is None or enr_top.empty:
        return

    apply_fig4_style()
    term_col = "Term" if "Term" in enr_top.columns else enr_top.columns[0]
    df = enr_top.iloc[::-1].copy()  # 最高显著在上方

    # 截断过长 term
    def _shorten(s: str, n: int = 52) -> str:
        s = str(s)
        return s if len(s) <= n else s[: n - 3] + "..."

    labels = [_shorten(t) for t in df[term_col]]
    if "Overlap" in df.columns:
        # "3/120" -> ratio
        def _ratio(s):
            try:
                a, b = str(s).split("/")
                return int(a) / max(int(b), 1)
            except Exception:
                return 0.05

        sizes = df["Overlap"].map(_ratio).values * 800 + 40
    else:
        sizes = np.full(len(df), 120.0)

    colors = np.where(
        df.get("focus", False),
        model_color("scFoundation"),
        "#9E9E9E",
    )

    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(df) + 1.8), facecolor="white")
    y = np.arange(len(df))
    x = df["neglog10_padj"].values

    ax.scatter(x, y, s=sizes, c=colors, alpha=0.88, edgecolors="white", linewidths=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=TICK_LABEL_SIZE - 1)
    ax.set_xlabel(r"$-\log_{10}$(adjusted $P$)", fontsize=AXIS_LABEL_SIZE)
    ax.set_title(
        f"{dataset} — pathways enriched in direction-error genes (strict_wrong)",
        fontsize=TITLE_SIZE,
        pad=10,
    )
    ax.axvline(-np.log10(0.05), color="#888888", linestyle="--", linewidth=1, alpha=0.7)
    for xi, yi, foc in zip(x, y, df.get("focus", [False] * len(df))):
        if foc:
            ax.scatter([xi], [yi], s=sizes[yi] * 1.05, facecolors="none", edgecolors=model_color("scGPT"), linewidths=1.5)

    leg = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=model_color("scFoundation"), markersize=10, label="Differentiation / cell cycle related"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#9E9E9E", markersize=10, label="Other significant terms"),
    ]
    ax.legend(handles=leg, loc="lower right", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_updown_breakdown(df: pd.DataFrame, dataset: str, outpath: Path) -> None:
    """在 strict_wrong 中按真实方向 Up/Down 的占比（解释 recall 偏倚）。"""
    apply_fig4_style()
    sub = df[df["error_type"] == "strict_wrong"]
    if sub.empty:
        return
    counts = sub["dir_true"].astype(str).str.strip().value_counts()
    labels = [k for k in ["Up", "Down"] if k in counts.index]
    if not labels:
        return
    vals = [counts.get(l, 0) for l in labels]
    cols = [model_color("scGPT") if l == "Up" else model_color("scPrint") for l in labels]

    fig, ax = plt.subplots(figsize=(4.5, 4), facecolor="white")
    wedges, _, autotexts = ax.pie(
        vals,
        labels=labels,
        autopct="%1.0f%%",
        colors=cols,
        startangle=90,
        textprops={"fontsize": TICK_LABEL_SIZE},
    )
    for t in autotexts:
        t.set_fontsize(11)
    ax.set_title(
        f"{dataset} — true direction of direction-error genes\n(top {int(TOP_PERCENT*100)}% dynamic)",
        fontsize=TITLE_SIZE - 1,
    )
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pipeline per dataset
# ---------------------------------------------------------------------------
def process_dataset(
    dataset: str,
    gene_csv: Path,
    chip_network_csv: Path,
    outdir: Path,
    *,
    do_enrichment: bool = True,
    model_label: str = "scGPT",
    mapped_only: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    df = load_gene_result(gene_csv, mapped_only=mapped_only)
    tf_targets = load_chip_tf_targets(chip_network_csv)
    df = classify_errors(df)
    top = select_top30(df)
    top = annotate_top30(top, tf_targets)

    # 保存带标注的基因表
    top.to_csv(outdir / f"{dataset}_top30_annotated.csv", index=False)

    frac_tables = {
        "tf_group": error_fraction_table(top, "tf_group"),
        "expr_bin": error_fraction_table(top, "expr_bin"),
        "delta_bin": error_fraction_table(top, "delta_bin"),
    }
    for key, tab in frac_tables.items():
        tab["stratum_col"] = key
        tab["dataset"] = dataset
        tab.to_csv(outdir / f"{dataset}_{key}_error_fractions.csv", index=False)

    plot_stratified_stacked(
        frac_tables,
        dataset,
        outdir / f"{dataset}_error_strat_stacked.png",
        model_label=model_label,
    )
    plot_updown_breakdown(top, dataset, outdir / f"{dataset}_strict_wrong_dir_pie.png")

    enr_df = None
    if do_enrichment:
        strict_genes = top.loc[top["error_type"] == "strict_wrong", "gene"].astype(str).tolist()
        species = DATASET_SPECIES.get(dataset, "human")
        enr_df = run_enrichment(strict_genes, species, outdir, dataset)
        if enr_df is not None and not enr_df.empty:
            top_terms = pick_top_terms(enr_df, top_n=14)
            top_terms.to_csv(outdir / f"{dataset}_strict_wrong_enrichment_top.csv", index=False)
            plot_enrichment_dot(
                top_terms,
                dataset,
                outdir / f"{dataset}_go_enrichment_strict_wrong.png",
            )

    # 合并分层表
    combined = pd.concat([v for v in frac_tables.values() if not v.empty], ignore_index=True)
    return combined, enr_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dynamic-gene error stratification & GO enrichment (fig4)")
    p.add_argument("--gene-dir", type=Path, default=DEFAULT_GENE_DIR)
    p.add_argument("--chip-dir", type=Path, default=DEFAULT_CHIP_DIR)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--dataset", type=str, default=None, help="Single dataset, e.g. hESC")
    p.add_argument("--all-datasets", action="store_true", help="Process all *_gene_result.csv")
    p.add_argument("--top-percent", type=float, default=TOP_PERCENT)
    p.add_argument("--rel-err-thresh", type=float, default=REL_ERR_THRESH)
    p.add_argument("--no-enrichment", action="store_true")
    p.add_argument("--model-label", default="scGPT", help="Display name in figure titles")
    p.add_argument(
        "--mapped-only",
        action="store_true",
        help="For scFoundation: keep only vocab-mapped genes",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global TOP_PERCENT, REL_ERR_THRESH
    TOP_PERCENT = float(args.top_percent)
    REL_ERR_THRESH = float(args.rel_err_thresh)

    args.outdir.mkdir(parents=True, exist_ok=True)
    apply_fig4_style()

    gene_files = sorted(args.gene_dir.glob("*_gene_result.csv"))
    if args.dataset:
        gene_files = [f for f in gene_files if f.name.startswith(f"{args.dataset}_")]
    if not gene_files:
        raise SystemExit(f"No gene_result CSV under {args.gene_dir}")

    all_summary = []
    print(f"Output directory: {args.outdir}")

    for gf in gene_files:
        dataset = gf.name.replace("_gene_result.csv", "")
        chip_net = args.chip_dir / f"{dataset}_chip_matched-network.csv"
        if not chip_net.exists():
            print(f"[skip] {dataset}: missing {chip_net}")
            continue
        print(f"Processing {dataset} ...")
        comb, _ = process_dataset(
            dataset,
            gf,
            chip_net,
            args.outdir,
            do_enrichment=not args.no_enrichment,
            model_label=args.model_label,
            mapped_only=args.mapped_only,
        )
        all_summary.append(comb)

    if not all_summary:
        raise SystemExit("No datasets processed.")

    summary = pd.concat(all_summary, ignore_index=True)
    summary.to_csv(args.outdir / "all_datasets_error_fractions.csv", index=False)

    if len(all_summary) > 1:
        plot_tf_comparison_all_datasets(
            summary,
            args.outdir / "all_datasets_tf_target_error_bars.png",
        )
        plot_error_heatmap(
            summary,
            "strict_wrong",
            args.outdir / "heatmap_direction_error.png",
            f"Direction error rate across strata (top {int(TOP_PERCENT*100)}% dynamic genes)",
        )
        plot_error_heatmap(
            summary,
            "amplitude_wrong",
            args.outdir / "heatmap_amplitude_error.png",
            f"Amplitude error rate across strata (top {int(TOP_PERCENT*100)}% dynamic genes)",
        )

    print("Done. Key outputs:")
    for pat in [
        "*_error_strat_stacked.png",
        "*_go_enrichment_strict_wrong.png",
        "all_datasets_tf_target_error_bars.png",
        "heatmap_*.png",
    ]:
        for f in sorted(args.outdir.glob(pat)):
            print(f"  {f}")


if __name__ == "__main__":
    main()
