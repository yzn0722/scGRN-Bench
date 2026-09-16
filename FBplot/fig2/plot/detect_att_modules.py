#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unsupervised regulatory modules from per-head attention Q→K profiles (hESC / scGPT).

Idea
----
  For each gene i on active gene set (n≈908):
    Q_profile[i] = row i of A  (Gene_i as Query → all Keys), L1-normalized
    K_profile[i] = col i of A  (all Queries → Gene_i as Key), L1-normalized
  Feature = [PCA(Q), PCA(K)] → cosine kNN graph → Louvain (default) or hierarchical.

Per-head modules (recommended) + optional mean8 fused matrix.

Outputs: output/att_modules/{model}_{dataset}/

  Per-head (supplement):
    head{h}_gene_modules.csv, head{h}_modules_pca.pdf, ...

  Cross-head comparison (main text):
    {tag}_modules_pca_all_heads.pdf       2×4 PCA, large modules colored
    {tag}_best_module_8heads.pdf          ★ 8 heads, ONE figure: best module only / head
    {tag}_best_module_overlay.pdf           8 best modules on one axes (std PCA)
    {tag}_best_module_bar.pdf               best module size + hub gene per head
    {tag}_best_module_per_head.csv

Usage
-----
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig2/plot

  python detect_att_modules.py \\
    --models scgpt --dataset hESC \\
    --head-root /mnt/10T/yzn/scGRN-Bench/FBplot/fig2/att_head

  # faster: cap edges per head
  python detect_att_modules.py ... --max-edges-per-head 200000

  # hierarchical, K=5
  python detect_att_modules.py ... --method hierarchical --n-clusters 5
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import analyze_attention_heads as ath  # noqa: E402
import eval_heads_chip_auprc as chip  # noqa: E402
import validate_head_hubs_l2 as l2  # noqa: E402
from fig2_palette import model_color  # noqa: E402

plt.rcParams.update(ath.plt.rcParams)

GO_CACHE = _SCRIPT_DIR / "output" / "head_go_enrich" / "cache"
GO_LIB = "GO_Biological_Process_2023"

KEY_GENES_DEFAULT = (
    "HAND2,POU5F1,NANOG,SOX2,SOX17,EOMES,JMJD1C,SPIB,RXRG,MIA3,"
    "TCF4,STAT3,FOXH1,GATA4,ISL1"
)


# ---------------------------------------------------------------------------
# Matrix & features
# ---------------------------------------------------------------------------


def load_edges_to_matrix(
    fp: Path,
    gene_index: Dict[str, int],
    n: int,
    max_edges: int,
    agg: str,
) -> np.ndarray:
    """Build dense A[query, key] from TSV (Gene1→Gene2)."""
    usecols = ["Gene1", "Gene2", "EdgeWeight"]
    df = pd.read_csv(fp, sep="\t", usecols=lambda c: c in usecols or c in ath.WEIGHT_COLS)
    df = ath.normalize_edges(df)
    wcol = ath.detect_weight_col(df)
    if wcol and wcol != "EdgeWeight":
        df = df.rename(columns={wcol: "EdgeWeight"})
    df["EdgeWeight"] = pd.to_numeric(df["EdgeWeight"], errors="coerce").fillna(0.0)
    df = df[df["Gene1"].isin(gene_index) & df["Gene2"].isin(gene_index)]
    if max_edges > 0 and len(df) > max_edges:
        df = df.nlargest(max_edges, "EdgeWeight")

    A = np.zeros((n, n), dtype=np.float32)
    g1 = df["Gene1"].astype(str).values
    g2 = df["Gene2"].astype(str).values
    w = df["EdgeWeight"].astype(np.float32).values
    for a, b, wt in zip(g1, g2, w):
        i, j = gene_index[a], gene_index[b]
        if agg == "sum":
            A[i, j] += wt
        else:
            A[i, j] = max(A[i, j], wt)
    return A


def fuse_matrices(mats: List[np.ndarray], mode: str) -> np.ndarray:
    stack = np.stack(mats, axis=0)
    if mode == "mean":
        return stack.mean(axis=0).astype(np.float32)
    return stack.max(axis=0).astype(np.float32)


def profiles_from_A(A: np.ndarray, log1p: bool) -> Tuple[np.ndarray, np.ndarray]:
    X = np.log1p(A) if log1p else A.copy()
    Q = normalize(X, norm="l1", axis=1)
    K = normalize(X, norm="l1", axis=0)
    return Q.astype(np.float32), K.astype(np.float32)


def dual_role_features(Q: np.ndarray, K: np.ndarray, pca_dim: int, seed: int) -> np.ndarray:
    d = min(pca_dim, Q.shape[1], Q.shape[0] - 1)
    if d < 2:
        return np.hstack([Q, K])
    pca_q = PCA(n_components=d, random_state=seed)
    pca_k = PCA(n_components=d, random_state=seed + 1)
    Fq = pca_q.fit_transform(Q)
    Fk = pca_k.fit_transform(K)
    return np.hstack([Fq, Fk]).astype(np.float32)


def cosine_knn_graph(F: np.ndarray, k: int, min_sim: float) -> "nx.Graph":
    import networkx as nx

    Fn = normalize(F, norm="l2", axis=1)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(F)), metric="cosine")
    nn.fit(Fn)
    dists, idx = nn.kneighbors(Fn)

    G = nx.Graph()
    G.add_nodes_from(range(len(F)))
    for i in range(len(F)):
        for d, j in zip(dists[i], idx[i]):
            if i == j:
                continue
            sim = 1.0 - d
            if sim >= min_sim:
                w = sim
                if G.has_edge(i, j):
                    G[i][j]["weight"] = max(G[i][j]["weight"], w)
                else:
                    G.add_edge(i, j, weight=w)
    return G


def cluster_features(
    F: np.ndarray,
    method: str,
    n_clusters: int,
    k_nn: int,
    min_sim: float,
    seed: int,
) -> np.ndarray:
    """Return integer labels 0..K-1."""
    n = len(F)
    if n < 4:
        return np.zeros(n, dtype=int)

    if method == "louvain":
        import networkx as nx
        from networkx.algorithms import community

        G = cosine_knn_graph(F, k=k_nn, min_sim=min_sim)
        if G.number_of_edges() == 0:
            method = "hierarchical"

        if method == "louvain":
            comms = community.louvain_communities(G, weight="weight", seed=seed)
            labels = np.full(n, -1, dtype=int)
            for cid, nodes in enumerate(comms):
                for i in nodes:
                    labels[i] = cid
            # remap to 0..K-1
            uniq = sorted(set(labels))
            remap = {u: k for k, u in enumerate(uniq)}
            return np.array([remap[l] for l in labels], dtype=int)

    if method == "kmeans":
        k = min(n_clusters, n - 1)
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        return km.fit_predict(F)

    # hierarchical (default fallback)
    k = min(n_clusters, n - 1)
    if n_clusters <= 0:
        best_k, best_sc = 3, -1.0
        for kk in range(3, min(13, n)):
            lab = AgglomerativeClustering(n_clusters=kk).fit_predict(F)
            if len(set(lab)) < 2:
                continue
            sc = silhouette_score(F, lab, metric="cosine")
            if sc > best_sc:
                best_sc, best_k = sc, kk
        k = best_k
    agg = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    return agg.fit_predict(F)


def gene_level_stats(A: np.ndarray, genes: List[str]) -> pd.DataFrame:
    out_strength = A.sum(axis=1)
    in_strength = A.sum(axis=0)
    Q = normalize(np.log1p(A), norm="l1", axis=1)
    K = normalize(np.log1p(A), norm="l1", axis=0)

    def entropy(x):
        x = x[x > 0]
        if x.size == 0:
            return 0.0
        return float(-(x * np.log(x + 1e-12)).sum())

    rows = []
    for i, g in enumerate(genes):
        rows.append(
            {
                "gene": g,
                "out_strength": float(out_strength[i]),
                "in_strength": float(in_strength[i]),
                "Q_entropy": entropy(Q[i]),
                "K_entropy": entropy(K[i]),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# GO per module (reuse cached GMT + hypergeom)
# ---------------------------------------------------------------------------


def enrich_module_genes(
    module_genes: List[str],
    background: List[str],
    gmt: Dict[str, List[str]],
    padj_cutoff: float,
    min_overlap: int,
) -> pd.DataFrame:
    from gseapy.stats import calc_pvalues, multiple_testing_correction

    bg_set = set(background)
    query = [g for g in module_genes if g in bg_set]
    if len(query) < min_overlap:
        return pd.DataFrame()
    hg = list(calc_pvalues(query=query, gene_sets=gmt, background=bg_set))
    if not hg or len(hg[0]) == 0:
        return pd.DataFrame()
    terms, pvals, _, overlap_n, _, _ = hg
    fdrs, _ = multiple_testing_correction(ps=list(pvals), alpha=padj_cutoff, method="benjamini-hochberg")
    rows = []
    for term, padj, x in zip(terms, fdrs, overlap_n):
        if x < min_overlap or padj > padj_cutoff:
            continue
        rows.append({"term_name": term, "padj": float(padj), "overlap_size": int(x)})
    return pd.DataFrame(rows).sort_values("padj") if rows else pd.DataFrame()


def load_gmt(cache_dir: Path) -> Dict[str, List[str]]:
    pkl = cache_dir / f"{GO_LIB}.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            return pickle.load(f)
    # import from enrich script
    import enrich_head_go_gprofiler as goen

    return goen.load_go_library(GO_LIB, cache_dir, 5, 500)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def select_best_module(stats: pd.DataFrame, summary: pd.DataFrame, min_n: int = 10) -> Tuple[int, str, str, int]:
    """
    Pick the single best module per head.
    Score = n_genes + 10 × max(in_strength) in module (favors large hub-coherent modules).
    """
    if summary.empty or stats.empty:
        return 0, "", "", 0
    cand = summary[summary["n_genes"] >= max(min_n, 5)].copy()
    if cand.empty:
        cand = summary.nlargest(1, "n_genes")

    best_mid, best_hub, best_go, best_n = 0, "", "", 0
    best_score = -1.0
    for _, row in cand.iterrows():
        mid = int(row["module_id"])
        sub = stats[stats["module_id"] == mid]
        if sub.empty:
            continue
        hub_row = sub.loc[sub["in_strength"].idxmax()]
        score = float(row["n_genes"]) + 10.0 * float(hub_row["in_strength"])
        if score > best_score:
            best_score = score
            best_mid = mid
            best_hub = str(hub_row["gene"])
            best_n = int(row["n_genes"])
            best_go = str(row.get("top_GO_BP", "") or "")

    return best_mid, best_hub, best_go, best_n


def plot_best_module_8heads(
    panels: List[Tuple[str, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]],
    genes: List[str],
    display: str,
    dataset: str,
    out_pdf: Path,
    highlight: Set[str],
    min_n: int,
) -> pd.DataFrame:
    """
    ONE figure (2×4): each head shows ONLY its best module (single color per panel).
    """
    head_panels = [p for p in panels if p[0].startswith("head")]
    head_panels = sorted(head_panels, key=lambda x: int(x[0].replace("head", "")))

    rows = []
    ncol, nrow = 4, 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.8 * nrow))
    axes = np.atleast_1d(axes).flatten()
    palette = sns.color_palette("tab10", n_colors=max(8, len(head_panels)))

    for ax, (hlab, F2, labels, stats, summary), color in zip(axes, head_panels, palette):
        mid, hub, top_go, n_genes = select_best_module(stats, summary, min_n=min_n)
        mask = labels == mid
        rows.append(
            {
                "head_label": hlab,
                "best_module_id": mid,
                "n_genes": n_genes,
                "primary_in_hub": hub,
                "top_GO_BP": top_go,
            }
        )

        if not mask.any():
            ax.axis("off")
            ax.set_title(f"{hlab}\n(no module)")
            continue

        ax.scatter(F2[:, 0], F2[:, 1], c="#e0e0e0", s=10, alpha=0.35, edgecolors="none", zorder=1)
        pts = F2[mask]
        ax.scatter(pts[:, 0], pts[:, 1], c=[color], s=45, alpha=0.85, edgecolors="white", linewidths=0.25, zorder=3)

        sub = stats[mask].copy()
        top_hubs = sub.nlargest(3, "in_strength")["gene"].tolist()
        for g in top_hubs:
            i = genes.index(g)
            ax.scatter([F2[i, 0]], [F2[i, 1]], s=100, facecolors="none", edgecolors="k", linewidths=1.2, zorder=5)
            ax.annotate(g, (F2[i, 0], F2[i, 1]), fontsize=7, fontweight="bold", ha="center", va="bottom")

        go_short = (top_go[:35] + "…") if len(top_go) > 35 else top_go
        ax.set_title(f"{hlab}  |  M{mid} (n={n_genes})\nhub: {hub}", fontsize=9)
        if go_short:
            ax.set_xlabel(go_short, fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(head_panels) :]:
        ax.axis("off")

    fig.suptitle(
        f"{display} — {dataset}\nBest attention module per head (Louvain; one module / head)",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_best_module_overlay(
    panels: List[Tuple[str, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]],
    display: str,
    dataset: str,
    out_pdf: Path,
    min_n: int,
) -> None:
    """All 8 best modules on ONE scatter (per-head PCA standardized → comparable layout)."""
    head_panels = sorted(
        [p for p in panels if p[0].startswith("head")],
        key=lambda x: int(x[0].replace("head", "")),
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    palette = sns.color_palette("tab10", n_colors=8)

    for hi, (hlab, F2, labels, stats, summary) in enumerate(head_panels):
        mid, hub, _, n_genes = select_best_module(stats, summary, min_n=min_n)
        mask = labels == mid
        if not mask.any():
            continue
        pts = F2[mask].astype(float)
        pts = (pts - pts.mean(axis=0)) / (pts.std(axis=0) + 1e-9)
        ax.scatter(
            pts[:, 0], pts[:, 1], c=[palette[hi]], s=40, alpha=0.75,
            label=f"{hlab} M{mid} (n={n_genes}, hub={hub})", edgecolors="none",
        )
        im = np.argmax(stats.loc[mask, "in_strength"].values)
        ax.annotate(hub, (pts[im, 0], pts[im, 1]), fontsize=7, fontweight="bold")

    ax.legend(fontsize=7, loc="best", framealpha=0.9)
    ax.set_xlabel("PC1 (standardized within head)")
    ax.set_ylabel("PC2 (standardized within head)")
    ax.set_title(f"{display} — {dataset}\nBest module per head (overlaid, 8 colors)")
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_best_module_bar(best_df: pd.DataFrame, display: str, dataset: str, out_pdf: Path) -> None:
    if best_df.empty:
        return
    df = best_df.sort_values("head_label", key=lambda s: s.str.replace("head", "").astype(int))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(df))
    colors = sns.color_palette("tab10", n_colors=len(df))
    ax.bar(x, df["n_genes"], color=colors, edgecolor="k", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.head_label}\nM{int(r.best_module_id)}" for r in df.itertuples()], fontsize=9)
    for i, r in enumerate(df.itertuples()):
        ax.text(i, r.n_genes + 3, f"hub:\n{r.primary_in_hub}", ha="center", fontsize=7)
    ax.set_ylabel("# genes in best module")
    ax.set_title(f"{display} — {dataset}\nBest module size & primary in-hub per head")
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _module_sizes_from_labels(labels: np.ndarray) -> Dict[int, int]:
    uniq, counts = np.unique(labels, return_counts=True)
    return {int(u): int(c) for u, c in zip(uniq, counts)}


def _draw_pca_modules_on_ax(
    ax: plt.Axes,
    F2: np.ndarray,
    labels: np.ndarray,
    genes: List[str],
    min_module_plot: int,
    highlight: Set[str],
    panel: bool = False,
    show_legend: bool = True,
) -> None:
    """Gray small modules; color + legend only for modules with n >= min_module_plot."""
    sizes = _module_sizes_from_labels(labels)
    small_ids = {c for c, n in sizes.items() if n < min_module_plot}
    large_ids = sorted(c for c, n in sizes.items() if n >= min_module_plot)

    small_mask = np.isin(labels, list(small_ids))
    if small_mask.any():
        ax.scatter(
            F2[small_mask, 0], F2[small_mask, 1],
            c="#d0d0d0", s=4 if panel else 6, alpha=0.35, edgecolors="none", rasterized=True,
        )

    palette = sns.color_palette("Set2", n_colors=max(len(large_ids), 3))
    pt = 28 if panel else 55
    for idx, c in enumerate(large_ids):
        m = labels == c
        n = sizes[c]
        ax.scatter(
            F2[m, 0], F2[m, 1],
            c=[palette[idx % len(palette)]], s=pt, alpha=0.82, edgecolors="white", linewidths=0.2,
            label=f"M{c} (n={n})", rasterized=True,
        )
        cx, cy = F2[m, 0].mean(), F2[m, 1].mean()
        ax.text(cx, cy, f"M{c}", fontsize=7 if panel else 9, fontweight="bold", ha="center", va="center", color="black")

    for i, g in enumerate(genes):
        if g not in highlight:
            continue
        mid = int(labels[i])
        ax.scatter(
            [F2[i, 0]], [F2[i, 1]], s=90 if panel else 140,
            facecolors="none", edgecolors="black", linewidths=1.4, zorder=10,
        )
        ax.annotate(
            f"{g}\nM{mid}",
            (F2[i, 0], F2[i, 1]),
            fontsize=6 if panel else 8,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=11,
        )

    if show_legend and large_ids and not panel:
        ax.legend(fontsize=7, loc="best", framealpha=0.9, title=f"modules n≥{min_module_plot}")


def plot_pca_modules(
    F2: np.ndarray,
    labels: np.ndarray,
    genes: List[str],
    display: str,
    dataset: str,
    head_label: str,
    out_pdf: Path,
    highlight: Set[str],
    min_module_plot: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    _draw_pca_modules_on_ax(ax, F2, labels, genes, min_module_plot, highlight, panel=False, show_legend=True)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    n_large = sum(1 for n in _module_sizes_from_labels(labels).values() if n >= min_module_plot)
    ax.set_title(
        f"{display} — {dataset}\n{head_label}: Q/K modules (PCA)\n"
        f"colored: {n_large} modules with ≥{min_module_plot} genes; gray = smaller"
    )
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_module_sizes(
    summary: pd.DataFrame, head_label: str, display: str, dataset: str, out_pdf: Path, min_module_plot: int
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["steelblue" if n >= min_module_plot else "#cccccc" for n in summary["n_genes"]]
    ax.bar(summary["module_id"].astype(str), summary["n_genes"], color=colors, edgecolor="k", linewidth=0.3)
    ax.axhline(min_module_plot, color="crimson", ls="--", lw=0.8, label=f"n≥{min_module_plot} (shown in PCA)")
    ax.set_xlabel("Module")
    ax.set_ylabel("# genes")
    ax.set_title(f"{display} — {dataset}\n{head_label}: module sizes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _ordered_head_columns(cols: List[str]) -> List[str]:
    def key(c: str):
        if c == "mean8":
            return (2, 8)
        if c.startswith("head") and c[4:].isdigit():
            return (0, int(c[4:]))
        return (1, c)

    return sorted(cols, key=key)


def plot_keygene_across_heads(df: pd.DataFrame, display: str, dataset: str, out_pdf: Path) -> None:
    """Heatmap: key genes × heads, cell = module id (main cross-head figure)."""
    if df.empty:
        return
    mat = df.pivot(index="gene", columns="head_label", values="module_id")
    mat = mat[_ordered_head_columns(list(mat.columns))]
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * mat.shape[1] + 2), max(4, 0.38 * mat.shape[0] + 2)))
    sns.heatmap(mat, annot=True, fmt="d", cmap="tab20", ax=ax, cbar_kws={"label": "Module id"})
    ax.set_title(f"{display} — {dataset}\nKey genes: module id across heads (compare specialization)")
    ax.set_xlabel("Attention head")
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_keygene_strength_heatmap(
    df: pd.DataFrame, value_col: str, title: str, display: str, dataset: str, out_pdf: Path
) -> None:
    if df.empty:
        return
    mat = df.pivot(index="gene", columns="head_label", values=value_col)
    mat = mat[_ordered_head_columns(list(mat.columns))]
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * mat.shape[1] + 2), max(4, 0.38 * mat.shape[0] + 2)))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
    ax.set_title(f"{display} — {dataset}\n{title}")
    ax.set_xlabel("Attention head")
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pca_8panel(
    panels: List[Tuple[str, np.ndarray, np.ndarray]],
    genes: List[str],
    display: str,
    dataset: str,
    out_pdf: Path,
    key_genes: Set[str],
    min_module_plot: int,
) -> None:
    """2×4 PCA — large modules colored, small modules gray."""
    n = len(panels)
    ncol, nrow = 4, int(np.ceil(n / 4))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.8 * nrow))
    axes = np.atleast_1d(axes).flatten()
    for ax, (hlab, F2, labels) in zip(axes, panels):
        _draw_pca_modules_on_ax(ax, F2, labels, genes, min_module_plot, key_genes, panel=True, show_legend=False)
        n_large = sum(1 for nv in _module_sizes_from_labels(labels).values() if nv >= min_module_plot)
        ax.set_title(f"{hlab} ({n_large} large mod.)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(panels) :]:
        ax.axis("off")
    fig.suptitle(
        f"{display} — {dataset}\nPer-head Q/K modules (PCA): color = module n≥{min_module_plot}; "
        f"gray = smaller; ring = key genes",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_module_stats_across_heads(summaries: List[Tuple[str, pd.DataFrame]], display: str, dataset: str, out_pdf: Path) -> None:
    rows = []
    for hlab, summ in summaries:
        if summ.empty:
            continue
        n_mod = len(summ)
        n_genes = summ["n_genes"].sum()
        rows.append(
            {
                "head_label": hlab,
                "n_modules": n_mod,
                "largest_module": int(summ["n_genes"].max()),
                "n_singleton": int((summ["n_genes"] == 1).sum()),
                "mean_module_size": n_genes / max(n_mod, 1),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return
    order = _ordered_head_columns(df["head_label"].tolist())
    df = df.set_index("head_label").loc[order].reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x = np.arange(len(df))
    axes[0].bar(x, df["n_modules"], color="steelblue", edgecolor="k", linewidth=0.3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["head_label"], rotation=30, ha="right")
    axes[0].set_ylabel("# modules")
    axes[0].set_title("Module count")

    axes[1].bar(x, df["largest_module"], color="coral", edgecolor="k", linewidth=0.3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["head_label"], rotation=30, ha="right")
    axes[1].set_ylabel("# genes")
    axes[1].set_title("Largest module")

    axes[2].bar(x, df["n_singleton"], color="gray", edgecolor="k", linewidth=0.3)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(df["head_label"], rotation=30, ha="right")
    axes[2].set_ylabel("# modules")
    axes[2].set_title("Singleton modules (n=1)")

    fig.suptitle(f"{display} — {dataset}\nModule detection summary across heads", y=1.05)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_combined_head_separation(
    features_by_head: List[Tuple[str, np.ndarray]],
    genes: List[str],
    display: str,
    dataset: str,
    out_pdf: Path,
    key_genes: Set[str],
) -> None:
    """Single PCA on stacked per-head features; color = head (batch structure)."""
    rows, colors, gene_idx = [], [], []
    palette = sns.color_palette("tab10", n_colors=max(10, len(features_by_head)))
    for hi, (hlab, F) in enumerate(features_by_head):
        rows.append(F)
        colors.extend([palette[hi % len(palette)]] * len(F))
    F_all = np.vstack(rows)
    F2 = PCA(n_components=2, random_state=42).fit_transform(normalize(F_all, norm="l2", axis=1))

    fig, ax = plt.subplots(figsize=(8, 6))
    start = 0
    for hi, (hlab, F) in enumerate(features_by_head):
        n = len(F)
        sl = slice(start, start + n)
        ax.scatter(F2[sl, 0], F2[sl, 1], s=10, alpha=0.35, c=[palette[hi % len(palette)]], label=hlab, edgecolors="none")
        start += n
    for i, g in enumerate(genes):
        if g not in key_genes:
            continue
        # plot each head's copy of gene i (offset by head block)
        pts_x, pts_y = [], []
        off = 0
        for _, F in features_by_head:
            j = off + i
            pts_x.append(F2[j, 0])
            pts_y.append(F2[j, 1])
            off += len(F)
        ax.plot(pts_x, pts_y, "k-", alpha=0.4, linewidth=0.8)
        ax.scatter(pts_x, pts_y, s=60, facecolors="none", edgecolors="k", linewidths=1.0, zorder=5)
        ax.annotate(g, (pts_x[-1], pts_y[-1]), fontsize=8, fontweight="bold")
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.set_title(f"{display} — {dataset}\nStacked Q/K features PCA (color = head)\nLines link key genes across heads")
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# One head / mean8
# ---------------------------------------------------------------------------


def run_module_detection(
    A: np.ndarray,
    genes: List[str],
    gene_index: Dict[str, int],
    chip_tfs: Set[str],
    chip_targets: Set[str],
    background: List[str],
    gmt: Dict[str, List[str]],
    args: argparse.Namespace,
    head_label: str,
    out_dir: Path,
    tag: str,
    display: str,
    dataset: str,
    key_genes: Set[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    Q, K = profiles_from_A(A, args.log1p)
    F = dual_role_features(Q, K, args.pca_dim, args.seed)
    labels = cluster_features(F, args.method, args.n_clusters, args.k_nn, args.min_sim, args.seed)

    stats = gene_level_stats(A, genes)
    stats["module_id"] = labels
    stats["chip_role"] = stats["gene"].map(lambda g: l2.classify_chip_role(g, chip_tfs, chip_targets))
    stats["head"] = head_label
    stats.to_csv(out_dir / f"{tag}_{head_label}_gene_modules.csv", index=False)

    # module summary
    summ_rows = []
    go_parts = []
    for mid in sorted(set(labels)):
        sub = stats[stats["module_id"] == mid]
        glist = sub["gene"].tolist()
        top_by_in = sub.nlargest(5, "in_strength")["gene"].tolist()
        top_by_out = sub.nlargest(5, "out_strength")["gene"].tolist()
        n_tf = sum(1 for g in glist if g in chip_tfs)
        enr = enrich_module_genes(glist, background, gmt, args.padj_cutoff, args.min_overlap)
        top_go = ""
        if not enr.empty:
            top_go = str(enr.iloc[0]["term_name"])[:80]
            enr = enr.copy()
            enr["module_id"] = mid
            enr["head"] = head_label
            go_parts.append(enr)
        summ_rows.append(
            {
                "module_id": int(mid),
                "n_genes": len(glist),
                "n_CHIP_TF": n_tf,
                "frac_CHIP_TF": n_tf / max(len(glist), 1),
                "top_in_strength": ";".join(top_by_in),
                "top_out_strength": ";".join(top_by_out),
                "top_GO_BP": top_go,
            }
        )
    summary = pd.DataFrame(summ_rows)
    summary.to_csv(out_dir / f"{tag}_{head_label}_module_summary.csv", index=False)
    if go_parts:
        pd.concat(go_parts, ignore_index=True).to_csv(out_dir / f"{tag}_{head_label}_module_go.csv", index=False)

    # PCA 2D for plot
    pca2 = PCA(n_components=2, random_state=args.seed)
    F2 = pca2.fit_transform(normalize(F, norm="l2", axis=1))
    plot_pca_modules(
        F2, labels, genes, display, dataset, head_label,
        out_dir / f"{tag}_{head_label}_modules_pca.pdf", key_genes, args.min_module_plot,
    )
    plot_module_sizes(
        summary, head_label, display, dataset,
        out_dir / f"{tag}_{head_label}_module_sizes.pdf", args.min_module_plot,
    )

    print(f"    {head_label}: {len(set(labels))} modules, sizes={summary['n_genes'].tolist()}")
    return stats, summary, F2, labels, F


def run_one(model: str, args: argparse.Namespace) -> None:
    display = ath.MODEL_ALIASES.get(ath.model_file_prefix(model), model)
    tag = f"{ath.model_file_prefix(model)}_{args.dataset}"
    out_dir = Path(args.output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    chip_root = Path(args.chip_root)
    gt = chip.read_chip_gt(chip_root / f"{args.dataset}_chip_matched-network.csv")
    expr = chip.read_chip_expr(chip_root / f"{args.dataset}_chip_matched-ExpressionData.csv")
    active = sorted(chip.active_genes(expr, args.min_frac_nonzero))
    chip_tfs, chip_targets, _, _ = l2.chip_gene_roles(gt)
    genes = active
    gene_index = {g: i for i, g in enumerate(genes)}
    n = len(genes)
    background = genes

    gmt_raw = load_gmt(GO_CACHE)
    gmt = {k: [x for x in v if x in set(background)] for k, v in gmt_raw.items()}
    gmt = {k: v for k, v in gmt.items() if len(v) >= 5}

    key_genes = {g.strip() for g in args.key_genes.split(",") if g.strip()}

    print(f"\n[{tag}] genes={n}, method={args.method}, pca_dim={args.pca_dim}")

    files = ath.find_head_files([Path(p) for p in args.head_root], model, args.dataset)
    if not files:
        print("[WARN] no head TSV")
        return

    key_rows = []
    head_mats: List[np.ndarray] = []
    pca_panels: List[Tuple[str, np.ndarray, np.ndarray]] = []
    best_panels: List[Tuple[str, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]] = []
    summary_list: List[Tuple[str, pd.DataFrame]] = []
    features_by_head: List[Tuple[str, np.ndarray]] = []

    for fp in files:
        h = ath.parse_head_number(fp)
        head_label = f"head{h}"
        print(f"  loading {head_label}...", end=" ", flush=True)
        A = load_edges_to_matrix(fp, gene_index, n, args.max_edges_per_head, args.edge_agg)
        head_mats.append(A)
        stats, summary, F2, labels, F = run_module_detection(
            A, genes, gene_index, chip_tfs, chip_targets, background, gmt,
            args, head_label, out_dir, tag, display, args.dataset, key_genes,
        )
        pca_panels.append((head_label, F2, labels))
        best_panels.append((head_label, F2, labels, stats, summary))
        summary_list.append((head_label, summary))
        features_by_head.append((head_label, F))
        for g in key_genes:
            row = stats[stats["gene"] == g]
            if not row.empty:
                key_rows.append(
                    {
                        "gene": g,
                        "head_label": head_label,
                        "Head": h,
                        "module_id": int(row.iloc[0]["module_id"]),
                        "in_strength": row.iloc[0]["in_strength"],
                        "out_strength": row.iloc[0]["out_strength"],
                        "chip_role": row.iloc[0]["chip_role"],
                    }
                )
        print("done")

    if args.include_mean8 and head_mats:
        print("  mean8 fusion...", end=" ", flush=True)
        A_mean = fuse_matrices(head_mats, "mean")
        stats_m, summary_m, F2_m, labels_m, F_m = run_module_detection(
            A_mean, genes, gene_index, chip_tfs, chip_targets, background, gmt,
            args, "mean8", out_dir, tag, display, args.dataset, key_genes,
        )
        pca_panels.append(("mean8", F2_m, labels_m))
        summary_list.append(("mean8", summary_m))
        features_by_head.append(("mean8", F_m))
        for g in key_genes:
            row = stats_m[stats_m["gene"] == g]
            if not row.empty:
                key_rows.append(
                    {
                        "gene": g,
                        "head_label": "mean8",
                        "Head": -1,
                        "module_id": int(row.iloc[0]["module_id"]),
                        "in_strength": row.iloc[0]["in_strength"],
                        "out_strength": row.iloc[0]["out_strength"],
                        "chip_role": row.iloc[0]["chip_role"],
                    }
                )
        print("done")

    key_df = pd.DataFrame(key_rows)
    key_df.to_csv(out_dir / f"{tag}_keygene_module_across_heads.csv", index=False)

    # ----- Cross-head comparison figures (main text) -----
    plot_keygene_across_heads(key_df, display, args.dataset, out_dir / f"{tag}_keygene_module_heatmap.pdf")
    plot_keygene_strength_heatmap(
        key_df, "in_strength", "Key genes: in-hub strength (Gene2 mass)",
        display, args.dataset, out_dir / f"{tag}_keygene_in_strength_heatmap.pdf",
    )
    plot_keygene_strength_heatmap(
        key_df, "out_strength", "Key genes: out-strength (Gene1 mass)",
        display, args.dataset, out_dir / f"{tag}_keygene_out_strength_heatmap.pdf",
    )
    plot_pca_8panel(pca_panels, genes, display, args.dataset, out_dir / f"{tag}_modules_pca_all_heads.pdf", key_genes, args.min_module_plot)
    best_df = plot_best_module_8heads(
        best_panels, genes, display, args.dataset,
        out_dir / f"{tag}_best_module_8heads.pdf", key_genes, args.min_module_plot,
    )
    best_df.to_csv(out_dir / f"{tag}_best_module_per_head.csv", index=False)
    plot_best_module_bar(best_df, display, args.dataset, out_dir / f"{tag}_best_module_bar.pdf")
    plot_module_stats_across_heads(summary_list, display, args.dataset, out_dir / f"{tag}_module_stats_across_heads.pdf")
    plot_combined_head_separation(
        features_by_head, genes, display, args.dataset,
        out_dir / f"{tag}_heads_qk_feature_pca.pdf", key_genes,
    )

    print(f"[INFO] Cross-head figures: *_all_heads.pdf, *_heatmap.pdf, *_across_heads.pdf")
    print(f"[INFO] -> {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect attention Q/K modules per head.")
    p.add_argument("--models", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--head-root", action="append", required=True)
    p.add_argument("--chip-root", default="/mnt/10T/yzn/benchmark_GRN/input_process/CHIP")
    p.add_argument("--output-dir", default=str(_SCRIPT_DIR / "output" / "att_modules"))
    p.add_argument("--max-edges-per-head", type=int, default=0, help="0=all edges in TSV")
    p.add_argument("--edge-agg", choices=["max", "sum"], default="max")
    p.add_argument("--method", choices=["louvain", "hierarchical", "kmeans"], default="louvain")
    p.add_argument("--n-clusters", type=int, default=0, help="0=auto for hierarchical")
    p.add_argument("--pca-dim", type=int, default=50)
    p.add_argument("--k-nn", type=int, default=30)
    p.add_argument("--min-sim", type=float, default=0.25)
    p.add_argument("--log1p", action="store_true", default=True)
    p.add_argument("--no-log1p", action="store_false", dest="log1p")
    p.add_argument("--include-mean8", action="store_true", default=True)
    p.add_argument("--no-mean8", action="store_false", dest="include_mean8")
    p.add_argument("--min-frac-nonzero", type=float, default=0.05)
    p.add_argument("--padj-cutoff", type=float, default=0.05)
    p.add_argument("--min-overlap", type=int, default=3)
    p.add_argument("--key-genes", default=KEY_GENES_DEFAULT)
    p.add_argument("--min-module-plot", type=int, default=20, help="PCA: color modules with >= N genes; smaller = gray")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    return args


def main() -> None:
    args = parse_args()
    for m in args.models:
        run_one(m, args)


if __name__ == "__main__":
    main()
