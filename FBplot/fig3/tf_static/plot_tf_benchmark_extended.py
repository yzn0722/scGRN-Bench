#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TF benchmark 扩展分析（按主文 20 页计划第一批顺序）：

  1. hub_forest        — Hub−Specialist Jaccard 差值 bootstrap 森林图
  2. heatmap_gt        — 四金标准 hub 差值热图
  3. recall_precision  — Hub / Specialist 层 Recall vs Precision
  4. emb_tp_summary    — scGPT emb-only vs att-only 靶的 STRING TP 比例
  5. cross_dataset     — hESC vs hHep hub 棒棒糖（可指定模型）

第二批:
  6. hub_class         — 假 hub（housekeeping）vs 功能 hub 子类 Jaccard 对比
  7. topk              — Top-k 边数敏感性（k/|E_GT| 扫描）
  8. static_dynamic    — 静态 TP 靶与 Top30% 动态基因重叠（CHIP TF 靶分层）

第三批:
  9. consensus         — 模型间 per-TF Jaccard 谱相关热图 + 可恢复 TF 计数
 10. tf_cluster         — 各模型 TF 失败模式聚类（PCA + 树状图）+ 六模型对比图
 11. recoverable        — 多模型 Jaccard≥阈值 的 TF 名单表
 12. expr_scatter       — TF 表达 vs Jaccard（若有表达矩阵）

示例:
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig3
  python tf_static/plot_tf_benchmark_extended.py --dataset hESC
  python tf_static/plot_tf_benchmark_extended.py --dataset hESC --steps forest heatmap_gt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

FIG3 = Path(__file__).resolve().parents[1]
if str(FIG3) not in sys.path:
    sys.path.insert(0, str(FIG3))

from fig3_palette import method_color, method_label, model_color  # noqa: E402

from tf_static.collect_per_tf_metrics import assign_hub_strata, collect_per_tf_long  # noqa: E402
from tf_static.model_registry import (  # noqa: E402
    EVL_ROOT,
    EXTRACTIONS,
    GT_DISPLAY,
    GT_SOURCES,
    INPUT_ROOT,
    MODELS,
    resolve_gt_path,
    resolve_pred_path,
)
from tf_static.plot_tf_hub_family import (  # noqa: E402
    HUB_JACCARD_YLIM,
    HUB_LOLLIPOP_YLIM,
    HUB_SUMMARY_FIG,
    HUB_TIER_ORDER,
    _hub_mean,
    apply_plot_style,
    summarize_hub_tier,
)
from tf_static.plot_tf_jaccard_raincloud_all_models import (  # noqa: E402
    EXTRACT_COLORS,
    EXTRACT_DISPLAY,
    LABEL_COLOR,
    TEXT_SIZE,
    TITLE_SIZE,
)
from tf_static.plot_tf_method_specific import build_per_tf_wide, filter_story_tfs  # noqa: E402
from tf_static.utils import (  # noqa: E402
    classify_tf_edges,
    filter_prediction,
    load_gt_network,
    partition_venn3_regions,
    per_tf_edge_sets,
    pred_targets_for_tf,
)

RNG = np.random.default_rng(42)
HUB_TIER_HUB = "Hub (top 20%)"
HUB_TIER_SPEC = "Specialist (bottom 80%)"
N_BOOT = 2000
BATCH1_STEPS = ("forest", "heatmap_gt", "recall_precision", "emb_tp", "cross_dataset")
BATCH2_STEPS = ("hub_class", "topk", "static_dynamic")
BATCH3_STEPS = ("consensus", "tf_cluster", "recoverable", "expr_scatter")
ALL_STEPS = BATCH1_STEPS + BATCH2_STEPS + BATCH3_STEPS

FIG4_ERROR_BIO = FIG3 / "fig4" / "error_biology"
DEFAULT_GENE_PT_DIR = Path("/mnt/10T/yzn/benchmark_GRN/pre_scgpt/results_multidataset_pseudotime_227")
HUB_CLASS_CURATED = FIG3 / "tf_static" / "data" / "hub_tf_class_curated.csv"

FALSE_HUB_PREFIXES = ("POLR2", "GTF", "TBP", "SUPT5", "XRN", "NUDT", "ERCC", "SSRP", "BTF", "TAF", "NFX")
FUNC_HUB_PREFIXES = (
    "MCM", "RAD51", "CDC", "ORC", "CDT", "GMNN", "CCNE", "PLK", "WEE", "TOP2",
    "CLSP", "CENP", "BRCA", "RPA", "ATR", "CHEK", "TRIP13", "ESCO", "DSCC", "DTL",
    "SNRPD", "PRIM", "FEN1", "TIMELESS", "EXO1", "RFC", "RMI", "WDHD", "NASP",
)
HUB_CLASS_ORDER = ("housekeeping_hub", "functional_hub", "other_hub")
HUB_CLASS_LABELS = {
    "housekeeping_hub": "Housekeeping hub",
    "functional_hub": "Functional hub (replication)",
    "other_hub": "Other hub",
}
HUB_CLASS_COLORS = {
    "housekeeping_hub": "#9E9E9E",
    "functional_hub": "#2166AC",
    "other_hub": "#F4A582",
}
DEFAULT_TOPK_FRACS = (0.25, 0.5, 0.75, 1.0, 1.25)
DEFAULT_RECOVER_THRESH = 0.05
DEFAULT_CLUSTER_K = 5
MCM_GENES = tuple(f"MCM{i}" for i in range(1, 11))


def _out_dir(base: Path, dataset: str, gt_source: str) -> Path:
    d = base / dataset / f"benchmark_extended_{gt_source}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT) -> Tuple[float, float, float]:
    """Mean(a) - Mean(b) with percentile CI."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        d = float(np.mean(a) - np.mean(b)) if len(a) and len(b) else np.nan
        return d, np.nan, np.nan
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sa = RNG.choice(a, size=len(a), replace=True)
        sb = RNG.choice(b, size=len(b), replace=True)
        diffs[i] = sa.mean() - sb.mean()
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def hub_spec_jaccard_arrays(df: pd.DataFrame, model: str, extraction: str) -> Tuple[np.ndarray, np.ndarray]:
    sub = df[(df["model"] == model) & (df["extraction"] == extraction)]
    hub = sub[sub["hub_tier"] == HUB_TIER_HUB]["jaccard"].dropna().to_numpy()
    spec = sub[sub["hub_tier"] == HUB_TIER_SPEC]["jaccard"].dropna().to_numpy()
    return hub, spec


# ---------------------------------------------------------------------------
# 1. Forest plot
# ---------------------------------------------------------------------------

def compute_forest_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in dict.fromkeys(df["model"].tolist()):
        for ext in EXTRACTIONS:
            hub, spec = hub_spec_jaccard_arrays(df, model, ext)
            diff, lo, hi = bootstrap_mean_diff(hub, spec)
            rows.append(
                {
                    "model": model,
                    "extraction": ext,
                    "label": f"{model} · {EXTRACT_DISPLAY[ext]}",
                    "mean_hub": float(np.mean(hub)) if len(hub) else np.nan,
                    "mean_spec": float(np.mean(spec)) if len(spec) else np.nan,
                    "diff_hub_minus_spec": diff,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "n_hub": len(hub),
                    "n_spec": len(spec),
                }
            )
    return pd.DataFrame(rows)


def plot_hub_forest(
    df: pd.DataFrame,
    dataset: str,
    gt_source: str,
    out_dir: Path,
) -> None:
    tab = compute_forest_table(df)
    tab.to_csv(out_dir / f"{dataset}_gt-{gt_source}_hub_forest_stats.csv", index=False)
    tab = tab.dropna(subset=["diff_hub_minus_spec"]).sort_values("diff_hub_minus_spec", ascending=True)

    fig_h = max(5.0, 0.32 * len(tab) + 1.5)
    fig, ax = plt.subplots(figsize=(8.0, fig_h))
    y = np.arange(len(tab))
    for i, row in enumerate(tab.itertuples()):
        c = EXTRACT_COLORS[row.extraction]
        lo_err = row.diff_hub_minus_spec - row.ci_lo if np.isfinite(row.ci_lo) else 0
        hi_err = row.ci_hi - row.diff_hub_minus_spec if np.isfinite(row.ci_hi) else 0
        ax.errorbar(
            row.diff_hub_minus_spec,
            i,
            xerr=[[lo_err], [hi_err]],
            fmt="o",
            markersize=7,
            color=c,
            ecolor="#444444",
            elinewidth=1.2,
            capsize=3,
            linestyle="none",
        )
    ax.axvline(0, color="#999999", lw=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(tab["label"], fontsize=TEXT_SIZE - 2)
    ax.set_xlabel("Δ mean Jaccard (Hub − Specialist)", fontsize=TEXT_SIZE, color=LABEL_COLOR)
    gt_name = GT_DISPLAY.get(gt_source, gt_source)
    ax.set_title(
        f"{dataset} — Hub vs specialist Jaccard gain ({gt_name})\n"
        f"Bootstrap 95% CI (n={N_BOOT})",
        fontsize=TITLE_SIZE - 1,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    leg = [Patch(facecolor=EXTRACT_COLORS[e], label=EXTRACT_DISPLAY[e]) for e in EXTRACTIONS]
    ax.legend(handles=leg, loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    stem = out_dir / f"{dataset}_gt-{gt_source}_hub_forest"
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white", dpi=150)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [1/5] forest → {stem}.png/.pdf")


# ---------------------------------------------------------------------------
# 2. Four GT heatmap
# ---------------------------------------------------------------------------

def collect_hub_delta_matrix(
    dataset: str,
    models: List[str],
    extractions: List[str],
    evl_root: Path,
    input_root: Path,
) -> pd.DataFrame:
    records = []
    for gt in GT_SOURCES:
        try:
            df = collect_per_tf_long(
                dataset,
                gt_source=gt,
                models=models,
                extractions=extractions,
                evl_root=evl_root,
                input_root=input_root,
            )
        except FileNotFoundError as e:
            print(f"    skip GT={gt}: {e}")
            continue
        if df.empty:
            continue
        df = assign_hub_strata(df)
        summ, model_list = summarize_hub_tier(df)
        for model in model_list:
            for ext in extractions:
                hub_m = _hub_mean(summ, HUB_TIER_HUB, model, ext, "jaccard")
                spec_m = _hub_mean(summ, HUB_TIER_SPEC, model, ext, "jaccard")
                if hub_m == 0 and spec_m == 0:
                    continue
                records.append(
                    {
                        "gt_source": gt,
                        "model": model,
                        "extraction": ext,
                        "row_label": f"{model}\n{EXTRACT_DISPLAY[ext]}",
                        "delta": float(hub_m - spec_m),
                    }
                )
    return pd.DataFrame(records)


def plot_heatmap_gt(
    dataset: str,
    models: List[str],
    extractions: List[str],
    evl_root: Path,
    input_root: Path,
    out_dir: Path,
) -> None:
    mat_df = collect_hub_delta_matrix(dataset, models, extractions, evl_root, input_root)
    if mat_df.empty:
        print("  [2/5] heatmap_gt — no data")
        return
    mat_df.to_csv(out_dir / f"{dataset}_hub_delta_by_gt.csv", index=False)

    row_order = []
    for model in models:
        for ext in extractions:
            row_order.append(f"{model}\n{EXTRACT_DISPLAY[ext]}")
    col_order = [g for g in GT_SOURCES if g in mat_df["gt_source"].unique()]
    pivot = mat_df.pivot_table(index="row_label", columns="gt_source", values="delta", aggfunc="first")
    pivot = pivot.reindex(index=[r for r in row_order if r in pivot.index], columns=col_order)

    fig, ax = plt.subplots(figsize=(max(6, len(col_order) * 1.4), max(7, len(pivot) * 0.28)))
    vmax = max(0.15, float(np.nanmax(np.abs(pivot.values))))
    cmap = LinearSegmentedColormap.from_list("div", ["#F4A582", "#FFFFFF", "#2166AC"])
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([GT_DISPLAY.get(c, c) for c in pivot.columns], fontsize=TEXT_SIZE - 1)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="#222")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Δ Jaccard (Hub − Specialist)", fontsize=TEXT_SIZE - 1)
    ax.set_title(f"{dataset} — Hub gain across gold standards", fontsize=TITLE_SIZE)
    fig.tight_layout()
    stem = out_dir / f"{dataset}_hub_delta_heatmap_gt"
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [2/5] heatmap_gt → {stem}.png/.pdf")


# ---------------------------------------------------------------------------
# 3. Recall vs Precision
# ---------------------------------------------------------------------------

def plot_recall_precision(
    df: pd.DataFrame,
    dataset: str,
    gt_source: str,
    out_dir: Path,
) -> None:
    sub = df[df["hub_tier"].notna()].copy()
    summ = (
        sub.groupby(["hub_tier", "model", "extraction"], as_index=False)[["recall", "precision"]]
        .mean()
    )
    summ.to_csv(out_dir / f"{dataset}_gt-{gt_source}_recall_precision_means.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    models = list(dict.fromkeys(summ["model"].tolist()))
    x = np.arange(len(models))
    w = 0.11

    for ax, tier, title in zip(axes, HUB_TIER_ORDER, ["Hub (top 20%)", "Specialist (bottom 80%)"]):
        for ei, ext in enumerate(EXTRACTIONS):
            recall = []
            prec = []
            for model in models:
                row = summ[(summ["hub_tier"] == tier) & (summ["model"] == model) & (summ["extraction"] == ext)]
                recall.append(float(row["recall"].iloc[0]) if len(row) else 0.0)
                prec.append(float(row["precision"].iloc[0]) if len(row) else 0.0)
            off = (ei - 1) * 0.34
            ax.bar(x + off - w / 2, recall, width=w, color=EXTRACT_COLORS[ext], alpha=0.45, edgecolor="#444", lw=0.4)
            ax.bar(
                x + off + w / 2,
                prec,
                width=w,
                color=EXTRACT_COLORS[ext],
                alpha=0.95,
                edgecolor="#444",
                lw=0.4,
                hatch="///",
                label=EXTRACT_DISPLAY[ext] if tier == HUB_TIER_HUB else None,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=TEXT_SIZE - 1)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=TEXT_SIZE)
        ax.set_ylabel("Mean per-TF", fontsize=TEXT_SIZE, color=LABEL_COLOR)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    gt_name = GT_DISPLAY.get(gt_source, gt_source)
    fig.suptitle(
        f"{dataset} — Recall vs precision by hub tier ({gt_name})\n"
        "Solid = precision; light = recall (same extraction color)",
        fontsize=TITLE_SIZE - 1,
    )
    leg = [
        Patch(facecolor="#888888", alpha=0.45, label="Recall"),
        Patch(facecolor="#888888", alpha=0.95, hatch="///", label="Precision"),
    ] + [Patch(facecolor=EXTRACT_COLORS[e], label=EXTRACT_DISPLAY[e]) for e in EXTRACTIONS]
    axes[1].legend(handles=leg, loc="upper right", fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    stem = out_dir / f"{dataset}_gt-{gt_source}_recall_precision"
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [3/5] recall_precision → {stem}.png/.pdf")


# ---------------------------------------------------------------------------
# 4. emb-only vs att-only TP rate (scGPT)
# ---------------------------------------------------------------------------

def _region_tp_rate(genes: Set[str], gt_t: Set[str]) -> Tuple[float, int, int]:
    if not genes:
        return np.nan, 0, 0
    n_tp = len(genes & gt_t)
    return n_tp / len(genes), n_tp, len(genes)


def compute_extraction_region_tp(
    dataset: str,
    model: str,
    gt_source: str,
    evl_root: Path,
    input_root: Path,
    hub_only: bool = True,
    min_spread: float = 0.0,
) -> pd.DataFrame:
    wide, gt_by_tf, preds = build_per_tf_wide(dataset, model, gt_source, evl_root, input_root)
    if min_spread > 0:
        tfs = filter_story_tfs(wide, min_spread, 0.0, "spread_only")["TF"].tolist()
    else:
        deg = wide.groupby("TF")["gt_outdegree"].first()
        q80 = deg.quantile(0.80)
        tfs = deg[deg >= q80].index.tolist() if hub_only else wide["TF"].unique().tolist()

    rows = []
    for tf in tfs:
        sets = {e: pred_targets_for_tf(preds[e], tf) for e in EXTRACTIONS}
        regions = partition_venn3_regions(sets["emb500"], sets["att500"], sets["embhidden500"])
        gt_t = gt_by_tf.get(tf, set())
        for rkey in ("emb_only", "att_only", "hid_only", "triple"):
            rate, n_tp, n_g = _region_tp_rate(regions[rkey], gt_t)
            rows.append({"TF": tf, "region": rkey, "tp_rate": rate, "n_tp": n_tp, "n_genes": n_g})
    return pd.DataFrame(rows)


def plot_emb_tp_summary(
    dataset: str,
    gt_source: str,
    out_dir: Path,
    model: str = "scGPT",
    evl_root: Path = EVL_ROOT,
    input_root: Path = INPUT_ROOT,
    hub_only: bool = True,
) -> None:
    reg = compute_extraction_region_tp(dataset, model, gt_source, evl_root, input_root, hub_only=hub_only)
    if reg.empty:
        print("  [4/5] emb_tp — no data")
        return
    reg.to_csv(out_dir / f"{dataset}_{model}_extraction_region_tp.csv", index=False)

    summ = (
        reg.groupby("region", as_index=False)
        .agg(
            mean_tp_rate=("tp_rate", "mean"),
            median_tp_rate=("tp_rate", "median"),
            n_tf=("TF", "nunique"),
            total_genes=("n_genes", "sum"),
        )
    )
    order = ["emb_only", "att_only", "hid_only", "triple"]
    summ = summ.set_index("region").reindex(order).reset_index()
    labels = {
        "emb_only": f"emb only\n({method_label('emb500')})",
        "att_only": f"att only\n({method_label('att500')})",
        "hid_only": f"hid only\n({method_label('embhidden500')})",
        "triple": "triple overlap",
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(summ))
    bars = ax.bar(
        x,
        summ["mean_tp_rate"],
        color=["#8FB4DC", "#70CDBE", "#AC99D2", "#6E6E6E"],
        edgecolor="#333",
        linewidth=0.5,
        alpha=0.9,
    )
    for i, (_, row) in enumerate(summ.iterrows()):
        if row["n_tf"] > 0 and np.isfinite(row["mean_tp_rate"]):
            ax.text(i, row["mean_tp_rate"] + 0.02, f"n={int(row['n_tf'])} TF", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(r, r) for r in summ["region"]], fontsize=TEXT_SIZE - 1)
    ax.set_ylabel("Mean fraction of STRING TP among predicted targets", fontsize=TEXT_SIZE - 1)
    ax.set_ylim(0, min(1.05, ax.get_ylim()[1] + 0.1))
    ax.set_title(
        f"{dataset} — {model} hub TF: TP rate in extraction-unique target sets\n"
        f"GT: {GT_DISPLAY.get(gt_source, gt_source)}",
        fontsize=TITLE_SIZE - 1,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    stem = out_dir / f"{dataset}_{model}_emb_region_tp_summary"
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [4/5] emb_tp → {stem}.png/.pdf")


# ---------------------------------------------------------------------------
# 5. Cross-dataset lollipop (hESC vs hHep)
# ---------------------------------------------------------------------------

def plot_cross_dataset_lollipop(
    datasets: List[str],
    gt_source: str,
    out_dir: Path,
    models: List[str],
    extractions: List[str],
    evl_root: Path,
    input_root: Path,
    focus_models: Optional[List[str]] = None,
) -> None:
    from tf_static.plot_tf_hub_family import _hub_bar_offsets, _hub_mean, _style_hub_ax

    focus = focus_models or ["scGPT"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), HUB_SUMMARY_FIG[1]), squeeze=False)
    axes = axes.ravel()
    w, offsets = _hub_bar_offsets()

    for ax, ds in zip(axes, datasets):
        try:
            df = collect_per_tf_long(ds, gt_source=gt_source, models=models, extractions=extractions,
                                     evl_root=evl_root, input_root=input_root)
        except FileNotFoundError as e:
            ax.set_title(f"{ds}\n(missing)")
            ax.axis("off")
            print(f"    cross_dataset skip {ds}: {e}")
            continue
        df = assign_hub_strata(df)
        summ, mlist = summarize_hub_tier(df)
        mlist = [m for m in mlist if m in focus] or mlist[:1]
        x = np.arange(len(mlist))
        for hi, ht in enumerate(HUB_TIER_ORDER):
            for ext in extractions:
                vals = [_hub_mean(summ, ht, m, ext, "jaccard") for m in mlist]
                off = (hi - 0.5) * 0.36 + offsets[ext]
                for mi, v in enumerate(vals):
                    ax.vlines(x[mi] + off, 0, v, color=EXTRACT_COLORS[ext], lw=1.4, alpha=0.5 if hi == 0 else 0.95)
                    ax.scatter(
                        x[mi] + off,
                        v,
                        s=50,
                        color=EXTRACT_COLORS[ext],
                        edgecolors="#333",
                        linewidths=0.4,
                        zorder=3,
                    )
        _style_hub_ax(ax, "Mean per-TF Jaccard", "jaccard", mlist, x, ylim=HUB_LOLLIPOP_YLIM)
        ax.set_title(ds, fontsize=TEXT_SIZE, fontweight="bold")

    ds_tag = "_".join(datasets)
    fig.suptitle(
        f"Hub vs specialist — {' / '.join(datasets)} ({GT_DISPLAY.get(gt_source, gt_source)})",
        fontsize=TITLE_SIZE,
    )
    fig.tight_layout()
    cross_dir = out_dir.parent / "cross_dataset"
    cross_dir.mkdir(parents=True, exist_ok=True)
    stem = cross_dir / f"{ds_tag}_gt-{gt_source}_hub_lollipop"
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [5/5] cross_dataset → {stem}.png/.pdf")


# ---------------------------------------------------------------------------
# 6. Hub subclass: housekeeping vs functional
# ---------------------------------------------------------------------------

def _load_hub_class_override() -> Dict[str, str]:
    if not HUB_CLASS_CURATED.is_file():
        return {}
    tab = pd.read_csv(HUB_CLASS_CURATED)
    if "TF" not in tab.columns or "hub_class" not in tab.columns:
        return {}
    return dict(zip(tab["TF"].astype(str).str.strip().str.upper(), tab["hub_class"].astype(str)))


def assign_hub_tf_class(tf: str, override: Dict[str, str]) -> str:
    key = str(tf).strip().upper()
    if key in override:
        return override[key]
    if any(key.startswith(p) for p in FALSE_HUB_PREFIXES):
        return "housekeeping_hub"
    if any(key.startswith(p) for p in FUNC_HUB_PREFIXES):
        return "functional_hub"
    return "other_hub"


def annotate_hub_subclass(df: pd.DataFrame) -> pd.DataFrame:
    override = _load_hub_class_override()
    out = df.copy()
    out["hub_subclass"] = out["TF"].map(lambda t: assign_hub_tf_class(t, override))
    return out


def plot_hub_class_comparison(
    df: pd.DataFrame,
    dataset: str,
    gt_source: str,
    out_dir: Path,
    model: str = "scGPT",
) -> None:
    sub = annotate_hub_subclass(df)
    sub = sub[(sub["hub_tier"] == HUB_TIER_HUB) & (sub["model"] == model)]
    if sub.empty:
        print("  [6/8] hub_class — no hub rows")
        return
    summ = (
        sub.groupby(["hub_subclass", "extraction"], as_index=False)["jaccard"]
        .mean()
        .rename(columns={"jaccard": "mean_jaccard"})
    )
    counts = sub.groupby("hub_subclass")["TF"].nunique().to_dict()
    summ.to_csv(out_dir / f"{dataset}_{model}_hub_subclass_jaccard.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(HUB_CLASS_ORDER))
    w = 0.22
    for ei, ext in enumerate(EXTRACTIONS):
        vals = []
        for hc in HUB_CLASS_ORDER:
            row = summ[(summ["hub_subclass"] == hc) & (summ["extraction"] == ext)]
            vals.append(float(row["mean_jaccard"].iloc[0]) if len(row) else 0.0)
        off = (ei - 1) * w
        ax.bar(
            x + off,
            vals,
            width=w * 0.9,
            color=EXTRACT_COLORS[ext],
            edgecolor="#333",
            linewidth=0.4,
            label=EXTRACT_DISPLAY[ext],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{HUB_CLASS_LABELS[h]}\n(n={counts.get(h, 0)})" for h in HUB_CLASS_ORDER],
        fontsize=TEXT_SIZE - 2,
    )
    ax.set_ylabel("Mean per-TF Jaccard", fontsize=TEXT_SIZE, color=LABEL_COLOR)
    ax.set_ylim(0, HUB_JACCARD_YLIM[1])
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    gt_name = GT_DISPLAY.get(gt_source, gt_source)
    ax.set_title(
        f"{dataset} — {model} hub TF subclasses ({gt_name})\n"
        "Housekeeping vs replication-associated hubs",
        fontsize=TITLE_SIZE - 1,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    stem = out_dir / f"{dataset}_{model}_hub_subclass_bar"
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [6/8] hub_class → {stem}.png/.pdf")


# ---------------------------------------------------------------------------
# 7. Top-k edge cap sensitivity
# ---------------------------------------------------------------------------

def collect_topk_metrics(
    dataset: str,
    gt_source: str,
    models: List[str],
    extractions: List[str],
    k_fracs: Tuple[float, ...],
    evl_root: Path,
    input_root: Path,
) -> pd.DataFrame:
    gt_path = resolve_gt_path(gt_source, dataset, input_root)
    gt_gene1, gt_all, gt_n, _, gt_by_tf = load_gt_network(gt_path)
    rows = []
    for model in models:
        for ext in extractions:
            fp = resolve_pred_path(model, ext, dataset, evl_root)
            if fp is None:
                continue
            for frac in k_fracs:
                cap = max(1, int(round(gt_n * frac)))
                pred = filter_prediction(fp, gt_gene1, gt_all, gt_n, edge_cap=cap)
                met = per_tf_edge_sets(pred, gt_by_tf)
                if met.empty:
                    continue
                if "gt_outdegree" not in met.columns:
                    deg = pd.DataFrame([{"TF": t, "gt_outdegree": len(v)} for t, v in gt_by_tf.items()])
                    met = met.merge(deg, on="TF", how="left")
                q80 = met["gt_outdegree"].quantile(0.80)
                met["hub_tier"] = np.where(met["gt_outdegree"] >= q80, HUB_TIER_HUB, HUB_TIER_SPEC)
                for tier in (HUB_TIER_HUB, HUB_TIER_SPEC, "all"):
                    sub = met if tier == "all" else met[met["hub_tier"] == tier]
                    if sub.empty:
                        continue
                    rows.append(
                        {
                            "model": model,
                            "extraction": ext,
                            "k_frac": frac,
                            "k_edges": cap,
                            "stratum": tier.split()[0].lower(),
                            "mean_jaccard": float(sub["jaccard"].mean()),
                            "median_jaccard": float(sub["jaccard"].median()),
                            "n_tf": len(sub),
                        }
                    )
    return pd.DataFrame(rows)


def plot_topk_sensitivity(
    dataset: str,
    gt_source: str,
    out_dir: Path,
    models: List[str],
    extractions: List[str],
    k_fracs: Tuple[float, ...],
    evl_root: Path,
    input_root: Path,
    focus_model: Optional[str] = "scGPT",
) -> None:
    tab = collect_topk_metrics(dataset, gt_source, models, extractions, k_fracs, evl_root, input_root)
    if tab.empty:
        print("  [7/8] topk — no data")
        return
    tab.to_csv(out_dir / f"{dataset}_gt-{gt_source}_topk_sensitivity.csv", index=False)

    plot_models = [focus_model] if focus_model and focus_model in tab["model"].unique() else list(models)
    fig, axes = plt.subplots(len(plot_models), 1, figsize=(8, 4 * len(plot_models)), squeeze=False)
    axes = axes.ravel()
    stratum_style = {
        "hub": (HUB_TIER_HUB.split()[0], "#2166AC", "-"),
        "specialist": ("Specialist", "#F4A582", "--"),
        "all": ("All TF", "#333333", ":"),
    }

    for ax, model in zip(axes, plot_models):
        sub_m = tab[tab["model"] == model]
        for ext in extractions:
            for sk, (label, color, ls) in stratum_style.items():
                sub = sub_m[(sub_m["extraction"] == ext) & (sub_m["stratum"] == sk)]
                if sub.empty:
                    continue
                sub = sub.sort_values("k_frac")
                ax.plot(
                    sub["k_frac"],
                    sub["mean_jaccard"],
                    marker="o",
                    lw=2,
                    ls=ls,
                    color=color,
                    alpha=0.85 if sk == "hub" else 0.55,
                    label=f"{EXTRACT_DISPLAY[ext]} · {label}" if sk == "hub" else None,
                )
        ax.axvline(1.0, color="#AAAAAA", lw=0.8, linestyle=":")
        ax.set_xlabel("Edge cap k / |E_GT|", fontsize=TEXT_SIZE - 1)
        ax.set_ylabel("Mean per-TF Jaccard", fontsize=TEXT_SIZE - 1)
        ax.set_ylim(0, HUB_JACCARD_YLIM[1])
        ax.set_title(model, fontsize=TEXT_SIZE, fontweight="bold")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].legend(loc="best", fontsize=7, frameon=False, ncol=2)
    gt_name = GT_DISPLAY.get(gt_source, gt_source)
    fig.suptitle(f"{dataset} — Top-k sensitivity ({gt_name})", fontsize=TITLE_SIZE)
    fig.tight_layout()
    stem = out_dir / f"{dataset}_gt-{gt_source}_topk_sensitivity"
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [7/8] topk → {stem}.png/.pdf")


# ---------------------------------------------------------------------------
# 8. Static TP overlap with Top30% dynamic genes
# ---------------------------------------------------------------------------

def _load_chip_tf_targets(chip_network_csv: Path) -> Set[str]:
    net = pd.read_csv(chip_network_csv)
    cols = {c.lower(): c for c in net.columns}
    g2 = cols.get("gene2", net.columns[1])
    return set(net[g2].astype(str).str.strip().str.upper())


def _load_top30_genes(dataset: str, gene_csv: Optional[Path], chip_csv: Path) -> pd.DataFrame:
    cached = FIG4_ERROR_BIO / f"{dataset}_top30_annotated.csv"
    if cached.is_file():
        return pd.read_csv(cached)
    if gene_csv is not None and gene_csv.is_file():
        df = pd.read_csv(gene_csv)
        if "gene" not in df.columns and "ene" in df.columns:
            df = df.rename(columns={"ene": "gene"})
        df["abs_delta_true"] = pd.to_numeric(df["delta_true"], errors="coerce").abs()
        n_top = max(1, int(np.ceil(0.3 * len(df))))
        top = df.nlargest(n_top, "abs_delta_true").copy()
        tf_targets = _load_chip_tf_targets(chip_csv)
        genes = top["gene"].astype(str).str.strip().str.upper()
        top["is_tf_target"] = genes.isin(tf_targets)
        top["tf_group"] = np.where(top["is_tf_target"], "TF target", "Non-target")
        return top
    raise FileNotFoundError(f"No top30 table: {gene_csv or cached}")


def _static_tp_gene_set(
    dataset: str,
    model: str,
    extraction: str,
    gt_source: str,
    evl_root: Path,
    input_root: Path,
) -> Set[str]:
    gt_path = resolve_gt_path(gt_source, dataset, input_root)
    gt_gene1, gt_all, gt_n, _, gt_by_tf = load_gt_network(gt_path)
    fp = resolve_pred_path(model, extraction, dataset, evl_root)
    if fp is None:
        return set()
    pred = filter_prediction(fp, gt_gene1, gt_all, gt_n)
    tp_genes: Set[str] = set()
    for tf in pred["Gene1"].unique():
        tp, _, _ = classify_tf_edges(tf, pred, gt_by_tf)
        tp_genes |= tp
    return tp_genes


def plot_static_dynamic_overlap(
    dataset: str,
    gt_source: str,
    out_dir: Path,
    model: str,
    extraction: str,
    evl_root: Path,
    input_root: Path,
    gene_csv: Optional[Path],
    chip_csv: Optional[Path],
) -> None:
    chip_path = chip_csv or (INPUT_ROOT / "CHIP" / f"{dataset}_chip_matched-network.csv")
    top30 = _load_top30_genes(dataset, gene_csv, chip_path)
    top30["gene"] = top30["gene"].astype(str).str.strip().str.upper()
    static_tp = _static_tp_gene_set(dataset, model, extraction, gt_source, evl_root, input_root)
    top30["in_static_tp"] = top30["gene"].isin(static_tp)

    rows = []
    for grp, sub in top30.groupby("tf_group"):
        n = len(sub)
        n_hit = int(sub["in_static_tp"].sum())
        rows.append(
            {
                "group": grp,
                "n_genes": n,
                "n_in_static_tp": n_hit,
                "frac_in_static_tp": n_hit / n if n else np.nan,
            }
        )
    summ = pd.DataFrame(rows)
    summ.to_csv(out_dir / f"{dataset}_{model}_{extraction}_static_dynamic_overlap.csv", index=False)
    top30.to_csv(out_dir / f"{dataset}_{model}_{extraction}_top30_static_flag.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    # Panel A: among top30 dynamic genes
    ax = axes[0]
    colors = ["#8FB4DC" if g == "TF target" else "#CCCCCC" for g in summ["group"]]
    ax.bar(summ["group"], summ["frac_in_static_tp"], color=colors, edgecolor="#333", linewidth=0.5)
    ax.set_ylabel("Fraction in static STRING TP targets", fontsize=TEXT_SIZE - 1)
    ax.set_ylim(0, min(1.05, summ["frac_in_static_tp"].max() + 0.15))
    ax.set_title("A  Top 30% dynamic genes", fontsize=TEXT_SIZE)
    for i, r in summ.iterrows():
        ax.text(i, r["frac_in_static_tp"] + 0.02, f"n={int(r['n_genes'])}", ha="center", fontsize=9)
    # Panel B: set sizes (intersection vs components)
    ax = axes[1]
    n_top = len(top30)
    n_hit = int(top30["in_static_tp"].sum())
    n_static = len(static_tp)
    cats = ["Top30% dynamic", "Static TP", "Overlap"]
    vals = [n_top, n_static, n_hit]
    ax.bar(cats, vals, color=["#8FB4DC", "#AC99D2", "#4C9F70"], edgecolor="#333", linewidth=0.5)
    ax.set_ylabel("Gene count", fontsize=TEXT_SIZE - 1)
    ax.set_title(
        f"B  |Top30|={n_top}, |static TP|={n_static}, |∩|={n_hit}",
        fontsize=TEXT_SIZE - 1,
    )
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, str(v), ha="center", fontsize=10)
    gt_name = GT_DISPLAY.get(gt_source, gt_source)
    fig.suptitle(
        f"{dataset} — Static ({gt_name}) vs dynamic (Top 30% |Δ|)\n"
        f"{model} · {EXTRACT_DISPLAY[extraction]}",
        fontsize=TITLE_SIZE - 1,
    )
    for ax in axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    stem = out_dir / f"{dataset}_{model}_{extraction}_static_dynamic"
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [8/8] static_dynamic → {stem}.png/.pdf")


# ---------------------------------------------------------------------------
# 9–12. Batch 3: consensus, clustering, recoverable list, expression
# ---------------------------------------------------------------------------

def _jaccard_pivot(df: pd.DataFrame, extraction: str) -> pd.DataFrame:
    sub = df[df["extraction"] == extraction].copy()
    return sub.pivot_table(index="TF", columns="model", values="jaccard", aggfunc="first")


def plot_model_consensus(
    df: pd.DataFrame,
    dataset: str,
    gt_source: str,
    out_dir: Path,
    extraction: str = "embhidden500",
) -> None:
    """Spearman correlation of per-TF Jaccard profiles between models."""
    piv = _jaccard_pivot(df, extraction)
    piv = piv.dropna(how="all")
    if piv.shape[1] < 2:
        print("  [9/12] consensus — too few models")
        return
    corr = piv.corr(method="spearman")
    corr.to_csv(out_dir / f"{dataset}_gt-{gt_source}_model_jaccard_corr_{extraction}.csv")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Spearman ρ"},
        ax=ax,
    )
    ax.set_title(
        f"{dataset} — Model consensus on per-TF Jaccard\n"
        f"{EXTRACT_DISPLAY.get(extraction, extraction)} · {GT_DISPLAY.get(gt_source, gt_source)}",
        fontsize=TITLE_SIZE - 1,
    )
    fig.tight_layout()
    stem = out_dir / f"{dataset}_gt-{gt_source}_model_consensus_{extraction}"
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [9/12] consensus → {stem}.png/.pdf")


def count_recoverable_tfs(
    df: pd.DataFrame,
    extraction: str,
    thresh: float,
) -> pd.DataFrame:
    sub = annotate_hub_subclass(df[df["extraction"] == extraction].copy())
    hits = sub[sub["jaccard"] >= thresh].groupby("TF")["model"].nunique().reset_index(name="n_models_hit")
    wide = _jaccard_pivot(df, extraction).reset_index()
    meta = sub.drop_duplicates("TF")[["TF", "gt_outdegree", "hub_tier", "hub_subclass"]]
    out = wide.merge(meta, on="TF", how="left").merge(hits, on="TF", how="left")
    out["n_models_hit"] = out["n_models_hit"].fillna(0).astype(int)
    out["recoverable"] = out["n_models_hit"] >= 2
    return out.sort_values(["n_models_hit", "TF"], ascending=[False, True])


def plot_recoverable_summary(
    df: pd.DataFrame,
    dataset: str,
    gt_source: str,
    out_dir: Path,
    extraction: str = "embhidden500",
    thresh: float = DEFAULT_RECOVER_THRESH,
) -> None:
    tab = count_recoverable_tfs(df, extraction, thresh)
    tab.to_csv(out_dir / f"{dataset}_gt-{gt_source}_recoverable_tf_{extraction}.csv", index=False)

    # bar: how many TFs hit by k models
    cnt = tab["n_models_hit"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(cnt.index.astype(str), cnt.values, color=model_color("scGPT"), edgecolor="#333", alpha=0.85)
    ax.set_xlabel(f"# models with Jaccard ≥ {thresh}", fontsize=TEXT_SIZE - 1)
    ax.set_ylabel("# TFs", fontsize=TEXT_SIZE - 1)
    ax.set_title(
        f"{dataset} — Recoverable TF tally ({EXTRACT_DISPLAY.get(extraction, extraction)})",
        fontsize=TITLE_SIZE - 1,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    stem = out_dir / f"{dataset}_gt-{gt_source}_recoverable_bar_{extraction}"
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [11/12] recoverable → {stem}.png/.pdf + CSV")


def build_tf_feature_matrix(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Per-TF features: 3 extractions + spread + hub meta."""
    sub = df[df["model"] == model].copy()
    sub = annotate_hub_subclass(sub)
    wide = sub.pivot_table(index="TF", columns="extraction", values="jaccard", aggfunc="first")
    wide.columns = [f"jac_{c}" for c in wide.columns]
    meta = sub.groupby("TF", as_index=False).agg(
        gt_outdegree=("gt_outdegree", "first"),
        hub_tier=("hub_tier", "first"),
        hub_subclass=("hub_subclass", "first"),
    )
    feat = meta.merge(wide.reset_index(), on="TF")
    jac_cols = [c for c in feat.columns if c.startswith("jac_")]
    feat["jac_spread"] = feat[jac_cols].max(axis=1) - feat[jac_cols].min(axis=1)
    feat["jac_best"] = feat[jac_cols].max(axis=1)
    return feat


def _pca_2d(X: np.ndarray) -> np.ndarray:
    xc = X - X.mean(axis=0)
    u, s, _vt = np.linalg.svd(xc, full_matrices=False)
    return u[:, :2] * s[:2]


def run_tf_clustering(
    feat: pd.DataFrame,
    n_clusters: int = DEFAULT_CLUSTER_K,
) -> Tuple[pd.DataFrame, object, np.ndarray]:
    """Ward clustering on 3-extraction Jaccard; returns feat+cluster, linkage Z, PCA coords."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist

    jac_cols = [c for c in feat.columns if c.startswith("jac_")]
    X = feat[jac_cols].fillna(0.0).to_numpy()
    dist = pdist(X, metric="euclidean")
    Z = linkage(dist, method="ward")
    feat = feat.copy()
    feat["cluster"] = fcluster(Z, t=n_clusters, criterion="maxclust")
    pcs = _pca_2d(X)
    return feat, Z, pcs


def plot_tf_cluster(
    df: pd.DataFrame,
    dataset: str,
    gt_source: str,
    out_dir: Path,
    model: str = "scGPT",
    n_clusters: int = DEFAULT_CLUSTER_K,
) -> Optional[pd.DataFrame]:
    from scipy.cluster.hierarchy import dendrogram

    feat = build_tf_feature_matrix(df, model)
    if len(feat) < n_clusters + 1:
        print(f"  [10/12] tf_cluster — {model}: too few TFs")
        return None

    feat, Z, pcs = run_tf_clustering(feat, n_clusters)
    feat.to_csv(out_dir / f"{dataset}_{model}_tf_cluster_features.csv", index=False)

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.28)
    ax_d = fig.add_subplot(gs[0, 0])
    dendrogram(Z, labels=feat["TF"].tolist(), leaf_rotation=90, ax=ax_d, color_threshold=0)
    ax_d.set_title(f"{model} — TF hierarchical clustering", fontsize=TEXT_SIZE - 1)
    ax_d.set_ylabel("Ward distance")

    ax_p = fig.add_subplot(gs[0, 1])
    cmap = plt.colormaps.get_cmap("tab10").resampled(n_clusters)
    for cid in sorted(feat["cluster"].unique()):
        m = feat["cluster"] == cid
        ax_p.scatter(
            pcs[m, 0],
            pcs[m, 1],
            s=55,
            c=[cmap(int(cid) - 1)],
            label=f"C{cid}",
            edgecolors="#333",
            linewidths=0.3,
            alpha=0.85,
        )
    ax_p.set_xlabel("PC1", fontsize=TEXT_SIZE - 1)
    ax_p.set_ylabel("PC2", fontsize=TEXT_SIZE - 1)
    ax_p.set_title("PCA on per-TF Jaccard (3 extractions)", fontsize=TEXT_SIZE - 1)
    ax_p.legend(loc="best", fontsize=8, frameon=False, ncol=2)
    for spine in ("top", "right"):
        ax_p.spines[spine].set_visible(False)

    gt_name = GT_DISPLAY.get(gt_source, gt_source)
    fig.suptitle(f"{dataset} — TF failure-mode clusters ({gt_name})", fontsize=TITLE_SIZE)
    fig.tight_layout()
    stem = out_dir / f"{dataset}_{model}_tf_cluster"
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [10/12] tf_cluster → {stem}.png/.pdf")
    return feat


def _tf_cluster_summary_row(feat: pd.DataFrame, model: str, n_clusters: int) -> Dict:
    from tf_static.plot_tf_validation import assign_semantic_type

    feat = feat.copy()
    feat["semantic_type"] = feat.apply(assign_semantic_type, axis=1)
    jac_emb = feat.get("jac_emb500", pd.Series(0.0, index=feat.index)).fillna(0.0)
    cluster_sizes = feat["cluster"].value_counts()
    largest_cluster = int(cluster_sizes.idxmax()) if len(cluster_sizes) else 1
    n_largest = int(cluster_sizes.max())
    cluster_means = feat.assign(_jac_emb=jac_emb).groupby("cluster")["_jac_emb"].mean()
    lowest_jac_cluster = int(cluster_means.idxmin()) if len(cluster_means) else largest_cluster
    sem_counts = feat["semantic_type"].value_counts()
    mcm = feat[feat["TF"].isin(MCM_GENES)]
    mcm_dual = int((mcm["semantic_type"] == "Type4_dual_emb").sum()) if len(mcm) else 0
    return {
        "model": model,
        "n_tf": len(feat),
        "n_clusters": n_clusters,
        "largest_cluster_id": largest_cluster,
        "largest_cluster_n": n_largest,
        "largest_cluster_pct": 100.0 * n_largest / max(len(feat), 1),
        "lowest_jac_cluster_id": lowest_jac_cluster,
        "lowest_jac_cluster_pct": 100.0
        * (feat["cluster"] == lowest_jac_cluster).sum()
        / max(len(feat), 1),
        "type0_all_fail_pct": 100.0 * sem_counts.get("Type0_all_fail", 0) / max(len(feat), 1),
        "type4_dual_emb_pct": 100.0 * sem_counts.get("Type4_dual_emb", 0) / max(len(feat), 1),
        "type3_emb_dom_pct": 100.0 * sem_counts.get("Type3_emb_dom", 0) / max(len(feat), 1),
        "mcm_n": len(mcm),
        "mcm_type4_dual_emb_n": mcm_dual,
        "mean_jac_emb500": float(jac_emb.mean()),
        "mean_jac_best": float(feat["jac_best"].mean()),
    }


def plot_tf_cluster_all_models(
    df: pd.DataFrame,
    dataset: str,
    gt_source: str,
    out_dir: Path,
    models: List[str],
    n_clusters: int = DEFAULT_CLUSTER_K,
) -> None:
    """Per-model dendrogram+PCA figures, plus cross-model PCA grid and summary bars."""
    from tf_static.plot_tf_validation import assign_semantic_type

    model_feats: Dict[str, pd.DataFrame] = {}
    model_pcs: Dict[str, np.ndarray] = {}
    summary_rows: List[Dict] = []
    sem_frames: List[pd.DataFrame] = []

    for model in models:
        sub = df[df["model"] == model]
        if sub["TF"].nunique() < n_clusters + 1:
            print(f"  [10/12] tf_cluster — skip {model}: too few TFs")
            continue
        feat = plot_tf_cluster(df, dataset, gt_source, out_dir, model, n_clusters)
        if feat is None:
            continue
        feat["semantic_type"] = feat.apply(assign_semantic_type, axis=1)
        model_feats[model] = feat
        jac_cols = [c for c in feat.columns if c.startswith("jac_")]
        X = feat[jac_cols].fillna(0.0).to_numpy()
        model_pcs[model] = _pca_2d(X)
        summary_rows.append(_tf_cluster_summary_row(feat, model, n_clusters))
        sem_frames.append(
            feat.groupby("semantic_type", as_index=False)
            .size()
            .assign(model=model, pct=lambda d: 100.0 * d["size"] / len(feat))
        )

    if not model_feats:
        return

    pd.DataFrame(summary_rows).to_csv(
        out_dir / f"{dataset}_all_models_tf_cluster_summary.csv", index=False
    )
    if sem_frames:
        pd.concat(sem_frames, ignore_index=True).to_csv(
            out_dir / f"{dataset}_all_models_tf_cluster_semantic.csv", index=False
        )

    plotted = [m for m in models if m in model_feats]
    n_models = len(plotted)
    if n_models == 0:
        return

    all_pcs = np.vstack([model_pcs[m] for m in plotted])
    pad = 0.08 * max(np.ptp(all_pcs[:, 0]), np.ptp(all_pcs[:, 1]), 1e-6)
    xlim = (all_pcs[:, 0].min() - pad, all_pcs[:, 0].max() + pad)
    ylim = (all_pcs[:, 1].min() - pad, all_pcs[:, 1].max() + pad)
    gt_name = GT_DISPLAY.get(gt_source, gt_source)
    cmap = plt.colormaps.get_cmap("tab10").resampled(n_clusters)

    ncols = 3
    nrows = int(np.ceil(n_models / ncols))
    fig = plt.figure(figsize=(5.2 * ncols, 4.2 * nrows + 2.8))
    gs = fig.add_gridspec(nrows + 1, ncols, height_ratios=[1.0] * nrows + [0.55], hspace=0.38, wspace=0.28)

    for i, model in enumerate(plotted):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[r, c])
        feat = model_feats[model]
        pcs = model_pcs[model]
        for cid in sorted(feat["cluster"].unique()):
            m = feat["cluster"] == cid
            ax.scatter(
                pcs[m, 0],
                pcs[m, 1],
                s=28,
                c=[cmap(int(cid) - 1)],
                label=f"C{cid}",
                edgecolors="#333",
                linewidths=0.2,
                alpha=0.8,
            )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(method_label(model), fontsize=TEXT_SIZE - 1, color=model_color(model))
        ax.set_xlabel("PC1", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if i == 0:
            ax.legend(loc="upper right", fontsize=6, frameon=False, ncol=2)

    for j in range(i + 1, nrows * ncols):
        r, c = divmod(j, ncols)
        fig.add_subplot(gs[r, c]).axis("off")

    ax_bar = fig.add_subplot(gs[nrows, :])
    summ = pd.DataFrame(summary_rows)
    x = np.arange(len(summ))
    w = 0.35
    ax_bar.bar(x - w / 2, summ["largest_cluster_pct"], width=w, color="#B2182B", label="Largest cluster %")
    ax_bar.bar(x + w / 2, summ["type4_dual_emb_pct"], width=w, color="#2166AC", label="Type4 dual-emb %")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([method_label(m) for m in summ["model"]], rotation=25, ha="right", fontsize=9)
    ax_bar.set_ylabel("% of TFs", fontsize=TEXT_SIZE - 1)
    ax_bar.set_ylim(0, min(100, summ[["largest_cluster_pct", "type4_dual_emb_pct"]].max().max() * 1.15 + 5))
    ax_bar.legend(loc="upper right", fontsize=8, frameon=False)
    ax_bar.set_title("Cross-model failure topology (k=5 Ward on 3 Jaccard features)", fontsize=TEXT_SIZE - 1)
    for spine in ("top", "right"):
        ax_bar.spines[spine].set_visible(False)

    fig.suptitle(f"{dataset} — TF failure-mode clustering across models ({gt_name})", fontsize=TITLE_SIZE)
    stem = out_dir / f"{dataset}_all_models_tf_cluster_compare"
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [10/12] tf_cluster compare → {stem}.png/.pdf + summary CSV")


def _load_tf_mean_expression(dataset: str, input_root: Path) -> pd.Series:
    path = input_root / "STRING" / f"{dataset}_processed-ExpressionData.csv"
    if not path.is_file():
        return pd.Series(dtype=float)
    expr = pd.read_csv(path, index_col=0)
    expr.index = expr.index.astype(str).str.strip().str.upper()
    return expr.mean(axis=1)


def plot_expr_jaccard_scatter(
    df: pd.DataFrame,
    dataset: str,
    gt_source: str,
    out_dir: Path,
    input_root: Path,
    model: str = "scGPT",
    extraction: str = "embhidden500",
) -> None:
    sub = df[(df["model"] == model) & (df["extraction"] == extraction)].drop_duplicates("TF")
    mean_expr = _load_tf_mean_expression(dataset, input_root)
    if mean_expr.empty:
        print("  [12/12] expr_scatter — no expression file")
        return
    sub = sub.copy()
    sub["mean_expr"] = sub["TF"].map(mean_expr)
    sub = sub.dropna(subset=["mean_expr", "jaccard"])
    if len(sub) < 10:
        print("  [12/12] expr_scatter — too few points")
        return
    sub.to_csv(out_dir / f"{dataset}_{model}_{extraction}_tf_expr_jaccard.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    sc = ax.scatter(
        sub["mean_expr"],
        sub["jaccard"],
        c=sub["gt_outdegree"],
        cmap="viridis",
        s=40,
        alpha=0.75,
        edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="GT out-degree")
    ax.set_xlabel("Mean scRNA expression (log-scale)", fontsize=TEXT_SIZE - 1)
    ax.set_ylabel("Per-TF Jaccard", fontsize=TEXT_SIZE - 1)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_ylim(0, HUB_JACCARD_YLIM[1])
    ax.set_title("A  Expression vs Jaccard", fontsize=TEXT_SIZE)

    ax = axes[1]
    try:
        sub["expr_tertile"] = pd.qcut(sub["mean_expr"], q=3, labels=["Low", "Mid", "High"], duplicates="drop")
    except ValueError:
        sub["expr_tertile"] = "all"
    summ = sub.groupby("expr_tertile", observed=True)["jaccard"].mean()
    ax.bar(summ.index.astype(str), summ.values, color=["#F4A582", "#FDDBC7", "#2166AC"][: len(summ)], edgecolor="#333")
    ax.set_ylabel("Mean per-TF Jaccard", fontsize=TEXT_SIZE - 1)
    ax.set_ylim(0, HUB_JACCARD_YLIM[1])
    ax.set_title("B  Jaccard by expression tertile", fontsize=TEXT_SIZE)

    gt_name = GT_DISPLAY.get(gt_source, gt_source)
    fig.suptitle(
        f"{dataset} — {model} · {EXTRACT_DISPLAY.get(extraction, extraction)} ({gt_name})",
        fontsize=TITLE_SIZE - 1,
    )
    fig.tight_layout()
    stem = out_dir / f"{dataset}_{model}_{extraction}_expr_jaccard"
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [12/12] expr_scatter → {stem}.png/.pdf")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TF benchmark extended analyses (batch 1–3)")
    p.add_argument("--dataset", default="hESC")
    p.add_argument("--datasets-cross", nargs="+", default=["hESC", "hHep"])
    p.add_argument("--gt-source", default="STRING")
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--extractions", nargs="+", default=list(EXTRACTIONS))
    p.add_argument("--evl-root", type=Path, default=EVL_ROOT)
    p.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--steps", nargs="+", default=list(ALL_STEPS), choices=list(ALL_STEPS) + ["all"])
    p.add_argument("--scgpt-model", default="scGPT")
    p.add_argument("--cross-models", nargs="+", default=["scGPT"])
    p.add_argument("--topk-fracs", nargs="+", type=float, default=list(DEFAULT_TOPK_FRACS))
    p.add_argument("--topk-models", nargs="+", default=["scGPT"], help="Models for top-k curves")
    p.add_argument("--gene-csv", type=Path, default=None, help="Pseudotime gene_result.csv for dynamic")
    p.add_argument("--chip-network", type=Path, default=None)
    p.add_argument(
        "--static-extraction",
        default="embhidden500",
        help="Extraction for static_dynamic overlap",
    )
    p.add_argument("--consensus-extraction", default="embhidden500")
    p.add_argument("--recover-thresh", type=float, default=DEFAULT_RECOVER_THRESH)
    p.add_argument("--cluster-k", type=int, default=DEFAULT_CLUSTER_K)
    p.add_argument(
        "--cluster-models",
        nargs="+",
        default=None,
        help="Models for tf_cluster (default: all --models). Legacy: --cluster-model runs one only.",
    )
    p.add_argument(
        "--cluster-model",
        default=None,
        help="If set, run tf_cluster for this model only (overrides --cluster-models).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    apply_plot_style()
    steps = list(ALL_STEPS) if "all" in args.steps else args.steps

    out_base = args.out or (FIG3 / "tf_static" / "output")
    out_dir = _out_dir(out_base, args.dataset, args.gt_source)

    print(f"benchmark_extended | {args.dataset} | GT={args.gt_source} | steps={steps}")

    df: Optional[pd.DataFrame] = None
    need_df = any(
        s in steps
        for s in (
            "forest",
            "recall_precision",
            "hub_class",
            "consensus",
            "tf_cluster",
            "recoverable",
            "expr_scatter",
        )
    )
    if need_df:
        print("  collecting per-TF metrics...")
        df = collect_per_tf_long(
            args.dataset,
            gt_source=args.gt_source,
            models=list(args.models),
            extractions=list(args.extractions),
            evl_root=args.evl_root,
            input_root=args.input_root,
        )
        if df.empty:
            raise RuntimeError("No metrics collected")
        df = assign_hub_strata(df)
        df.to_csv(out_dir / f"{args.dataset}_gt-{args.gt_source}_per_tf_long.csv", index=False)

    if "forest" in steps and df is not None:
        plot_hub_forest(df, args.dataset, args.gt_source, out_dir)

    if "heatmap_gt" in steps:
        plot_heatmap_gt(
            args.dataset,
            list(args.models),
            list(args.extractions),
            args.evl_root,
            args.input_root,
            out_dir,
        )

    if "recall_precision" in steps and df is not None:
        plot_recall_precision(df, args.dataset, args.gt_source, out_dir)

    if "emb_tp" in steps:
        plot_emb_tp_summary(
            args.dataset,
            args.gt_source,
            out_dir,
            model=args.scgpt_model,
            evl_root=args.evl_root,
            input_root=args.input_root,
        )

    if "cross_dataset" in steps:
        plot_cross_dataset_lollipop(
            list(args.datasets_cross),
            args.gt_source,
            out_dir,
            list(args.models),
            list(args.extractions),
            args.evl_root,
            args.input_root,
            focus_models=list(args.cross_models),
        )

    if "hub_class" in steps and df is not None:
        plot_hub_class_comparison(df, args.dataset, args.gt_source, out_dir, model=args.scgpt_model)

    if "topk" in steps:
        plot_topk_sensitivity(
            args.dataset,
            args.gt_source,
            out_dir,
            list(args.topk_models),
            list(args.extractions),
            tuple(args.topk_fracs),
            args.evl_root,
            args.input_root,
            focus_model=args.scgpt_model,
        )

    if "static_dynamic" in steps:
        gene_csv = args.gene_csv or (DEFAULT_GENE_PT_DIR / f"{args.dataset}_gene_result.csv")
        plot_static_dynamic_overlap(
            args.dataset,
            args.gt_source,
            out_dir,
            args.scgpt_model,
            args.static_extraction,
            args.evl_root,
            args.input_root,
            gene_csv,
            args.chip_network,
        )

    ext_cons = args.consensus_extraction

    if "consensus" in steps and df is not None:
        plot_model_consensus(df, args.dataset, args.gt_source, out_dir, ext_cons)

    if "recoverable" in steps and df is not None:
        plot_recoverable_summary(df, args.dataset, args.gt_source, out_dir, ext_cons, args.recover_thresh)

    if "tf_cluster" in steps and df is not None:
        if args.cluster_model:
            plot_tf_cluster(df, args.dataset, args.gt_source, out_dir, args.cluster_model, args.cluster_k)
        else:
            cluster_models = args.cluster_models or list(args.models)
            plot_tf_cluster_all_models(
                df, args.dataset, args.gt_source, out_dir, cluster_models, args.cluster_k
            )

    if "expr_scatter" in steps and df is not None:
        plot_expr_jaccard_scatter(
            df,
            args.dataset,
            args.gt_source,
            out_dir,
            args.input_root,
            args.scgpt_model,
            ext_cons,
        )

    print("Done.")


if __name__ == "__main__":
    main()
