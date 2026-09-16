#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical analysis & extra figures for per-head attention GRN exports.

Compared to analyze_attention_heads.py (topology bar charts), this script adds:

  1. Edge-weight distribution per head (violin / ECDF)
  2. Top-k edge-set Jaccard similarity between heads (heatmap)
  3. Spearman correlation of edge weights on shared edges (heatmap)
  4. Shared vs head-specific edge counts (bar)
  5. Topology metrics z-score heatmap (head × metric)
  6. PCA of heads in topology-metric space (scatter)
  7. Multi-dataset: boxplot of each metric across datasets (per head)
  8. CSV tables: pairwise overlap, weight stats, metric dispersion (CV)

Usage:
  cd FBplot/fig2/plot
  python stat_analyze_attention_heads.py --models sccello --dataset hESC \\
    --head-root /path/to/outputs/attention_heads
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import analyze_attention_heads as ath  # noqa: E402
from fig2_palette import model_color  # noqa: E402

plt.rcParams.update(ath.plt.rcParams)


def load_top_edges_weighted(
    fp: Path, g1: Set[str], gu: Set[str], top_k: int
) -> pd.DataFrame:
    """Return Gene1, Gene2, EdgeWeight (undirected canonical pairs)."""
    usecols = ["Gene1", "Gene2", "EdgeWeight"]
    try:
        df = pd.read_csv(fp, sep="\t", usecols=lambda c: c in usecols or c in ath.WEIGHT_COLS)
    except (ValueError, KeyError):
        df = pd.read_csv(fp, sep="\t")
    df = ath.normalize_edges(df)
    df = df[df["Gene1"].isin(g1) & df["Gene2"].isin(gu)].copy()
    if df.empty:
        return df
    wcol = ath.detect_weight_col(df)
    if wcol is None:
        df["EdgeWeight"] = 1.0
    else:
        df = df.rename(columns={wcol: "EdgeWeight"})
        df["EdgeWeight"] = pd.to_numeric(df["EdgeWeight"], errors="coerce").fillna(0.0)
    if top_k > 0:
        df = df.nlargest(top_k, "EdgeWeight")
    # canonical undirected pair for set ops
    a = df["Gene1"].astype(str)
    b = df["Gene2"].astype(str)
    swap = a > b
    g1c = a.where(~swap, b)
    g2c = b.where(~swap, a)
    df = df.assign(Gene1=g1c, Gene2=g2c)
    df = df.drop_duplicates(subset=["Gene1", "Gene2"])
    return df[["Gene1", "Gene2", "EdgeWeight"]]


def edge_set(df: pd.DataFrame) -> Set[Tuple[str, str]]:
    if df.empty:
        return set()
    return set(zip(df["Gene1"].astype(str), df["Gene2"].astype(str)))


def jaccard(a: Set, b: Set) -> float:
    if not a and not b:
        return np.nan
    u = len(a | b)
    return len(a & b) / u if u else np.nan


def pairwise_jaccard(edge_sets: Dict[int, Set]) -> pd.DataFrame:
    heads = sorted(edge_sets.keys())
    mat = np.eye(len(heads))
    for i, hi in enumerate(heads):
        for j, hj in enumerate(heads):
            if j > i:
                v = jaccard(edge_sets[hi], edge_sets[hj])
                mat[i, j] = mat[j, i] = v
    return pd.DataFrame(mat, index=heads, columns=heads)


def pairwise_weight_spearman(
    weighted: Dict[int, pd.DataFrame],
) -> pd.DataFrame:
    heads = sorted(weighted.keys())
    mat = np.eye(len(heads))
    for i, hi in enumerate(heads):
        for j, hj in enumerate(heads):
            if j <= i:
                continue
            a, b = weighted[hi], weighted[hj]
            merged = a.merge(b, on=["Gene1", "Gene2"], suffixes=("_a", "_b"))
            if len(merged) < 10:
                rho = np.nan
            else:
                rho, _ = stats.spearmanr(merged["EdgeWeight_a"], merged["EdgeWeight_b"])
            mat[i, j] = mat[j, i] = rho
    return pd.DataFrame(mat, index=heads, columns=heads)


def overlap_summary(edge_sets: Dict[int, Set]) -> pd.DataFrame:
    union = set().union(*edge_sets.values()) if edge_sets else set()
    rows = []
    for h, es in sorted(edge_sets.items()):
        others = set().union(*(edge_sets[k] for k in edge_sets if k != h))
        only_h = es - others
        shared_all = es.copy()
        for k, o in edge_sets.items():
            if k != h:
                shared_all &= o
        rows.append(
            {
                "Head": h,
                "n_edges": len(es),
                "n_unique_to_head": len(only_h),
                "n_shared_all_heads": len(shared_all),
                "frac_of_union": len(es) / len(union) if union else np.nan,
            }
        )
    return pd.DataFrame(rows)


def weight_summary(weighted: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for h, df in sorted(weighted.items()):
        w = df["EdgeWeight"].values
        rows.append(
            {
                "Head": h,
                "n_edges": len(w),
                "weight_mean": float(np.mean(w)),
                "weight_std": float(np.std(w)),
                "weight_median": float(np.median(w)),
                "weight_q95": float(np.quantile(w, 0.95)),
                "weight_gini": float(_gini(w)),
            }
        )
    return pd.DataFrame(rows)


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    if x.size == 0 or np.allclose(x, 0):
        return np.nan
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def topology_table(
    weighted: Dict[int, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for h, df in sorted(weighted.items()):
        edges = df[["Gene1", "Gene2"]]
        m = ath.calc_all_metrics(edges)
        row = {"Head": h, "n_edges": len(df)}
        row.update(m)
        rows.append(row)
    return pd.DataFrame(rows)


def metric_cv(topo: pd.DataFrame) -> pd.DataFrame:
    """Coefficient of variation across heads for each topology metric."""
    rows = []
    for col in ath.METRICS:
        v = topo[col].astype(float).dropna()
        if len(v) < 2:
            cv = np.nan
        else:
            cv = float(v.std() / v.mean()) if v.mean() != 0 else np.nan
        rows.append({"Metric": col, "CV_across_heads": cv, "range": float(v.max() - v.min()) if len(v) else np.nan})
    return pd.DataFrame(rows)


def kruskal_per_metric(topo: pd.DataFrame) -> pd.DataFrame:
    """Nonparametric test: do heads differ? (needs edge-level replicates — use multi-dataset mode instead.)"""
    rows = []
    for col in ath.METRICS:
        v = topo[col].astype(float).dropna()
        rows.append(
            {
                "Metric": col,
                "n_heads": len(v),
                "note": "single value per head — use --datasets for across-dataset testing",
            }
        )
    return pd.DataFrame(rows)


def analyze_one_model_dataset(
    model: str,
    dataset: str,
    head_roots: List[Path],
    ref_root: Path,
    ref_type: str,
    top_k: int,
    out_dir: Path,
) -> None:
    display = ath.MODEL_ALIASES.get(ath.model_file_prefix(model), model)
    color = model_color(display)
    tag = f"{ath.model_file_prefix(model)}_{dataset}"
    sub = out_dir / tag
    sub.mkdir(parents=True, exist_ok=True)

    files = ath.find_head_files(head_roots, model, dataset)
    if not files:
        print(f"[WARN] skip {tag}: no TSV")
        return

    g1, gu, n_ref = ath.load_reference(ref_root, ref_type, dataset)
    keep_k = min(top_k, n_ref) if top_k > 0 else n_ref

    weighted: Dict[int, pd.DataFrame] = {}
    edge_sets: Dict[int, Set] = {}
    for fp in files:
        h = ath.parse_head_number(fp)
        df = load_top_edges_weighted(fp, g1, gu, keep_k)
        weighted[h] = df
        edge_sets[h] = edge_set(df)
        print(f"  {tag} head{h}: edges={len(df)}")

    # --- CSV outputs ---
    wsum = weight_summary(weighted)
    wsum.to_csv(sub / f"{tag}_weight_stats.csv", index=False)
    ov = overlap_summary(edge_sets)
    ov.to_csv(sub / f"{tag}_edge_overlap.csv", index=False)
    jac = pairwise_jaccard(edge_sets)
    jac.to_csv(sub / f"{tag}_jaccard_matrix.csv")
    spearman = pairwise_weight_spearman(weighted)
    spearman.to_csv(sub / f"{tag}_weight_spearman.csv")
    topo = topology_table(weighted)
    topo.to_csv(sub / f"{tag}_topology_metrics.csv", index=False)
    metric_cv(topo).to_csv(sub / f"{tag}_metric_cv.csv", index=False)

    heads = sorted(weighted.keys())

    # --- Fig 1: Jaccard heatmap ---
    fig, ax = plt.subplots(figsize=(max(4, len(heads) * 0.9), max(3.5, len(heads) * 0.8)))
    sns.heatmap(
        jac,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": "Jaccard (top-k edge sets)"},
        ax=ax,
    )
    ax.set_title(f"{display} — {dataset}")
    fig.savefig(sub / f"{tag}_jaccard_heatmap.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Fig 2: Spearman heatmap ---
    fig, ax = plt.subplots(figsize=(max(4, len(heads) * 0.9), max(3.5, len(heads) * 0.8)))
    sns.heatmap(
        spearman,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"label": "Spearman ρ (shared edges)"},
        ax=ax,
    )
    ax.set_title(f"{display} — {dataset}\nEdge-weight rank correlation between heads")
    fig.savefig(sub / f"{tag}_weight_spearman_heatmap.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Fig 3: Weight violin ---
    long_w = []
    for h, df in weighted.items():
        for v in df["EdgeWeight"].values:
            long_w.append({"Head": str(h), "EdgeWeight": v})
    if long_w:
        wdf = pd.DataFrame(long_w)
        fig, ax = plt.subplots(figsize=(max(5, len(heads) * 1.2), 4))
        sns.violinplot(data=wdf, x="Head", y="EdgeWeight", color=color, inner="box", ax=ax)
        ax.set_xlabel("Attention Head")
        ax.set_ylabel("Edge weight (top-k)")
        ax.set_title(f"{display} — {dataset}")
        fig.savefig(sub / f"{tag}_weight_violin.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # --- Fig 4: Unique vs shared edges ---
    fig, ax = plt.subplots(figsize=(max(5, len(heads) * 1.2), 4))
    x = np.arange(len(ov))
    w = 0.35
    shared = ov["n_edges"] - ov["n_unique_to_head"]
    ax.bar(x - w / 2, shared, w, label="Also in ≥1 other head", color=color, alpha=0.5)
    ax.bar(x + w / 2, ov["n_unique_to_head"], w, label="Unique to this head", color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in ov["Head"]])
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Edge count")
    ax.legend(fontsize=10)
    ax.set_title(f"{display} — {dataset}\nEdge overlap decomposition (top-{keep_k})")
    fig.savefig(sub / f"{tag}_edge_overlap_bar.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Fig 5: Topology z-score heatmap ---
    z = topo.set_index("Head")[ath.METRICS].astype(float)
    z = (z - z.mean()) / z.std(ddof=0).replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(6, max(3, len(heads) * 0.5)))
    sns.heatmap(
        z.T,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        cbar_kws={"label": "z-score across heads"},
        ax=ax,
        yticklabels=ath.METRICS_DISPLAY,
    )
    ax.set_xlabel("Attention Head")
    ax.set_title(f"{display} — {dataset}\nTopology metrics (standardized across heads)")
    fig.savefig(sub / f"{tag}_topology_zscore_heatmap.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Fig 6: PCA on topology (if ≥2 heads) ---
    if len(heads) >= 2:
        X = topo.set_index("Head")[ath.METRICS].astype(float).values
        X = np.nan_to_num(X, nan=0.0)
        Xc = X - X.mean(axis=0)
        try:
            u, s, vt = np.linalg.svd(Xc, full_matrices=False)
            pc = Xc @ vt[:2].T
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(pc[:, 0], pc[:, 1], c=color, s=120, edgecolors="k", linewidths=0.5)
            for i, h in enumerate(heads):
                ax.annotate(f"H{h}", (pc[i, 0], pc[i, 1]), fontsize=11, ha="center", va="bottom")
            var = (s**2) / max((s**2).sum(), 1e-12)
            ax.set_xlabel(f"PC1 ({var[0]*100:.0f}% var)")
            ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)" if len(s) > 1 else "PC2")
            ax.set_title(f"{display} — {dataset}")
            ax.axhline(0, color="gray", lw=0.5)
            ax.axvline(0, color="gray", lw=0.5)
            fig.savefig(sub / f"{tag}_topology_pca.pdf", dpi=300, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"[WARN] PCA failed: {e}")

    # --- Fig 7: Clustered dendrogram on Jaccard distance ---
    if len(heads) >= 3:
        dist = 1 - jac.values.astype(float)
        np.fill_diagonal(dist, 0)
        condensed = squareform(dist, checks=False)
        link = hierarchy.linkage(condensed, method="average")
        fig, ax = plt.subplots(figsize=(6, 3))
        hierarchy.dendrogram(link, labels=[f"H{h}" for h in heads], ax=ax, color_threshold=0)
        ax.set_title(f"{display} — {dataset}\nHead clustering (1 − Jaccard)")
        ax.set_ylabel("Distance")
        fig.savefig(sub / f"{tag}_head_cluster_dendrogram.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"[INFO] Wrote stats & figures -> {sub}")


def multi_dataset_boxplots(
    model: str,
    datasets: List[str],
    head_roots: List[Path],
    ref_root: Path,
    ref_type: str,
    top_k: int,
    out_dir: Path,
) -> None:
    """One metric per figure: x=head, points=datasets (6 pseudo-replicates)."""
    display = ath.MODEL_ALIASES.get(ath.model_file_prefix(model), model)
    rows = []
    for ds in datasets:
        files = ath.find_head_files(head_roots, model, ds)
        if not files:
            continue
        g1, gu, n_ref = ath.load_reference(ref_root, ref_type, ds)
        keep_k = min(top_k, n_ref) if top_k > 0 else n_ref
        for fp in files:
            h = ath.parse_head_number(fp)
            df = load_top_edges_weighted(fp, g1, gu, keep_k)
            m = ath.calc_all_metrics(df[["Gene1", "Gene2"]])
            row = {"Dataset": ds, "Head": h, **m}
            rows.append(row)
    if not rows:
        return
    long = pd.DataFrame(rows)
    tag = f"{ath.model_file_prefix(model)}_multi_ds"
    sub = out_dir / tag
    sub.mkdir(parents=True, exist_ok=True)
    long.to_csv(sub / f"{tag}_topology_long.csv", index=False)

    for col, ylab in zip(ath.METRICS, ath.METRICS_DISPLAY):
        fig, ax = plt.subplots(figsize=(max(5, long["Head"].nunique() * 1.2), 4))
        sns.boxplot(
            data=long,
            x="Head",
            y=col,
            color=model_color(display),
            fliersize=2,
            ax=ax,
        )
        sns.stripplot(data=long, x="Head", y=col, color="black", alpha=0.5, size=4, ax=ax)
        ax.set_xlabel("Attention Head")
        ax.set_ylabel(ylab)
        ax.set_title(f"{display} — {len(datasets)} datasets\n{ylab} stability across heads")
        # Kruskal-Wallis across heads (values = datasets)
        groups = [g[col].dropna().values for _, g in long.groupby("Head")]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            try:
                stat, p = stats.kruskal(*groups)
                ax.text(
                    0.02,
                    0.98,
                    f"Kruskal-Wallis p={p:.3g}",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=10,
                )
            except Exception:
                pass
        fig.savefig(sub / f"{tag}_{col}_boxplot.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"[INFO] Multi-dataset boxplots -> {sub}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Statistical analysis for per-head GRN TSVs.")
    p.add_argument("--dataset", default="hESC")
    p.add_argument("--datasets", default="", help="Comma-separated; enables multi-dataset boxplots.")
    p.add_argument("--models", default="scGPT")
    p.add_argument("--head-root", action="append", default=[])
    p.add_argument("--ref-root", default="/mnt/10T/yzn/benchmark_GRN/input_process")
    p.add_argument("--ref-type", default="CHIP", choices=["STRING", "CHIP"])
    p.add_argument("--top-edges", type=int, default=10000)
    p.add_argument(
        "--output-dir",
        default=str(_SCRIPT_DIR / "output" / "head_statistics"),
    )
    return p.parse_args()


def default_head_roots() -> List[Path]:
    project = _SCRIPT_DIR.parents[3]
    return [
        project / "outputs" / "attention_heads",
        _SCRIPT_DIR.parent / "att_head",
    ]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    head_roots = [Path(r) for r in args.head_root] if args.head_root else default_head_roots()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    ref_root = Path(args.ref_root)

    if args.datasets.strip():
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        datasets = [args.dataset.strip()]

    for model in models:
        if len(datasets) > 1:
            multi_dataset_boxplots(
                model, datasets, head_roots, ref_root, args.ref_type, args.top_edges, out_dir
            )
        for ds in datasets:
            analyze_one_model_dataset(
                model, ds, head_roots, ref_root, args.ref_type, args.top_edges, out_dir
            )

    # Write analysis guide
    guide = out_dir / "README_outputs.txt"
    if not guide.exists():
        guide.write_text(
            _ANALYSIS_GUIDE,
            encoding="utf-8",
        )
    print(f"[INFO] Done. Output: {out_dir}")
    print(f"[INFO] See {guide} for figure meanings.")


_ANALYSIS_GUIDE = """\
Per (model, dataset) folder outputs
====================================

CSV
---
*_weight_stats.csv       Mean/std/median/Gini of EdgeWeight per head
*_edge_overlap.csv       Unique vs shared edge counts per head
*_jaccard_matrix.csv     Pairwise Jaccard of top-k edge sets
*_weight_spearman.csv    Spearman rho of weights on shared edges
*_topology_metrics.csv   Five graph metrics per head
*_metric_cv.csv          CV & range of each metric across heads

Figures
-------
*_jaccard_heatmap.pdf           High = heads predict similar top-k edges
*_weight_spearman_heatmap.pdf   High = same edges ranked similarly
*_weight_violin.pdf             Weight distribution shape per head
*_edge_overlap_bar.pdf          How many edges are head-specific
*_topology_zscore_heatmap.pdf   Which head is extreme on which metric
*_topology_pca.pdf              2D view of head differences (5 metrics)
*_head_cluster_dendrogram.pdf   Hierarchical grouping of heads

Multi-dataset folder (*_multi_ds)
---------------------------------
*_topology_long.csv             All datasets × heads × metrics
*_{Metric}_boxplot.pdf          Spread across 6 datasets; Kruskal-Wallis p-value

Interpretation tips
-------------------
- Low Jaccard + low Spearman → heads are genuinely different GRNs (specialization).
- High Jaccard + low Spearman → similar edge sets but different strengths.
- High Modularity z on one head → that head yields more modular GRN topology.
- Large CV on a metric → heads disagree; consider head selection or ensemble.
- scPRINT: compare slice 28-31 only when aligning to 4-head BERT models.
"""


if __name__ == "__main__":
    main()
