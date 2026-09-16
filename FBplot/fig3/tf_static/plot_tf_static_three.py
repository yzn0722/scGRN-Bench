#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Static GRN — three TF-focused analyses (fig3/tf_static).

1. Pred vs GT out-degree correlation + per-TF top-target Jaccard heatmap
2. Hub TF ego-networks with edge-level TP / FP / FN coloring
3. Per-TF ORA on false-positive targets (hub TFs)

Example:
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig3
  python tf_static/plot_tf_static_three.py --dataset hESC
  python tf_static/plot_tf_static_three.py --dataset hESC --method embhidden500 --top-hub 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy import stats

# fig3 imports
FIG3 = Path(__file__).resolve().parents[1]
if str(FIG3) not in sys.path:
    sys.path.insert(0, str(FIG3))

from fig3_palette import method_color, method_label, model_color  # noqa: E402

from tf_static.utils import (  # noqa: E402
    classify_tf_edges,
    filter_prediction,
    load_gene_set_table,
    load_gt_network,
    per_tf_edge_sets,
    read_gmt,
    run_gene_set_enrichment,
    run_ora,
    shorten_term,
    spearman_label,
)

TEXT_SIZE = 13
TITLE_SIZE = 14

DEFAULT_PLURIPOTENCY_TFS = ["SOX2", "NANOG", "POU5F1", "KLF4", "LIN28A"]
BIO_GENE_SETS_CSV = Path(__file__).resolve().parent / "data" / "hESC_biological_gene_sets.csv"
DEFAULT_CASE_STUDIES: Tuple[Tuple[str, str], ...] = (
    ("RBBP4", "FP"),
    ("POLR2H", "TP"),
    ("SOX2", "FN"),
)

METHOD_PRED = {
    "emb500": ("emb500", "scgpt_{dataset}.tsv"),
    "embhidden500": ("embhidden500", "scGPT_{dataset}.tsv"),
    "att500": ("att500", "scgpt_{dataset}.tsv"),
}

GMT_HUMAN = [
    Path("/mnt/10T/yzn/benchmark_GRN/fuji/c2.cp.v2025.1.Hs.symbols.gmt"),
    Path("/mnt/10T/yzn/benchmark_GRN/fuji/h.all.v2025.1.Hs.symbols.gmt"),
]

EDGE_COLORS = {
    "TP": "#4C9F70",
    "FP": model_color("scPrint"),
    "FN": "#9AA0A6",
}
NODE_TF = model_color("scGPT")
NODE_TARGET = "#D8D8D8"


def apply_style() -> None:
    sns.set_theme(style="white")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "font.size": TEXT_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": TEXT_SIZE,
            "xtick.labelsize": TEXT_SIZE - 1,
            "ytick.labelsize": TEXT_SIZE - 1,
        }
    )


def resolve_pred_path(method: str, pred_root: Path, dataset: str) -> Path:
    if method not in METHOD_PRED:
        raise ValueError(f"Unknown method {method!r}, choose from {list(METHOD_PRED)}")
    subdir, pattern = METHOD_PRED[method]
    return pred_root / f"output_{subdir}" / "scgpt" / pattern.format(dataset=dataset)


def load_pathways(gmt_files: List[Path]) -> Dict[str, Set[str]]:
    merged: Dict[str, Set[str]] = {}
    for p in gmt_files:
        if not p.is_file():
            print(f"  [warn] GMT missing: {p}")
            continue
        merged.update(read_gmt(p))
    return merged


# ---------------------------------------------------------------------------
# 1) Out-degree + Jaccard heatmap
# ---------------------------------------------------------------------------

def plot_outdegree_and_jaccard(
    metrics: pd.DataFrame,
    dataset: str,
    method: str,
    out_dir: Path,
    heatmap_top: int,
    min_gt_degree: int,
) -> None:
    sub = metrics[metrics["gt_outdegree"] >= min_gt_degree].copy()
    if sub.empty:
        print("  [skip] outdegree/jaccard: no TF passes min_gt_degree")
        return

    rho, pval = stats.spearmanr(sub["gt_outdegree"], sub["pred_outdegree"])
    label = spearman_label(
        sub["gt_outdegree"].to_numpy(),
        sub["pred_outdegree"].to_numpy(),
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), gridspec_kw={"width_ratios": [1.05, 1.2]})

    # --- scatter ---
    ax = axes[0]
    sizes = 28 + 3.5 * np.sqrt(sub["hits"].clip(lower=0).to_numpy())
    sc = ax.scatter(
        sub["gt_outdegree"],
        sub["pred_outdegree"],
        c=sub["jaccard"],
        s=sizes,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        edgecolors="#333333",
        linewidths=0.35,
        alpha=0.88,
        zorder=3,
    )
    lim = max(sub["gt_outdegree"].max(), sub["pred_outdegree"].max()) * 1.08
    ax.plot([0, lim], [0, lim], ls="--", color=model_color("STRING"), lw=1.1, zorder=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("GT out-degree (STRING)")
    ax.set_ylabel("Predicted out-degree (top-|GT| edges)")
    ax.set_title(f"{dataset} — Out-degree: pred vs GT\n({method_label(method)})")
    ax.text(
        0.04,
        0.96,
        label + f"\nρ={rho:.2f}, p={pval:.2e}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=TEXT_SIZE - 1,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CCCCCC", alpha=0.92),
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Target-set Jaccard", fontsize=TEXT_SIZE - 1)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # annotate top hubs
    top_ann = sub.head(5)
    for _, r in top_ann.iterrows():
        ax.annotate(
            r["TF"],
            (r["gt_outdegree"], r["pred_outdegree"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=9,
            color="#222222",
        )

    # --- Jaccard heatmap (TF × metrics) ---
    ax2 = axes[1]
    hm = sub.head(int(heatmap_top)).copy()
    mat = hm[["jaccard", "precision", "recall"]].to_numpy()
    tf_labels = hm["TF"].tolist()
    cmap = LinearSegmentedColormap.from_list(
        "tf_metrics",
        [(1.0, 1.0, 1.0), plt.matplotlib.colors.to_rgb(model_color("scGPT"))],
        N=256,
    )
    im = ax2.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(["Jaccard", "Precision", "Recall"], rotation=0)
    ax2.set_yticks(np.arange(len(tf_labels)))
    ax2.set_yticklabels(tf_labels, fontsize=TEXT_SIZE - 2)
    ax2.set_title(f"Top-{len(hm)} hub TFs (GT out-degree ↓)\nper-TF target-set quality")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax2.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if v > 0.55 else "#222222",
            )
    cbar2 = fig.colorbar(im, ax=ax2, fraction=0.035, pad=0.02)
    cbar2.set_label("Score", fontsize=TEXT_SIZE - 1)
    fig.tight_layout()
    stem = out_dir / f"{dataset}_{method}_01_outdegree_jaccard"
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {stem}.png/.pdf")


# ---------------------------------------------------------------------------
# 2) Hub TF ego-networks
# ---------------------------------------------------------------------------

def _ego_graph_for_tf(
    tf: str,
    pred_df: pd.DataFrame,
    gt_by_tf: Dict[str, Set[str]],
    max_targets: int,
) -> Tuple[nx.DiGraph, Dict[Tuple[str, str], str]]:
    tp, fp, fn = classify_tf_edges(tf, pred_df, gt_by_tf)
    G = nx.DiGraph()
    G.add_node(tf, ntype="tf")

    # predicted edges (TP/FP), cap by |weight| if many
    sub = pred_df[pred_df["Gene1"] == tf].copy()
    if len(sub) > max_targets:
        wcol = [c for c in sub.columns if "weight" in c.lower() or c == "EdgeWeight"]
        if wcol:
            sub = sub.assign(_w=pd.to_numeric(sub[wcol[0]], errors="coerce").fillna(0).abs())
            sub = sub.sort_values("_w", ascending=False).head(max_targets)
        else:
            sub = sub.head(max_targets)
        pred_show = set(sub["Gene2"])
        tp = tp & pred_show
        fp = fp & pred_show

    edge_status: Dict[Tuple[str, str], str] = {}
    for tg in tp | fp:
        G.add_node(tg, ntype="target")
        st = "TP" if tg in tp else "FP"
        G.add_edge(tf, tg, status=st)
        edge_status[(tf, tg)] = st

    # FN: dashed grey (GT only), limit count
    fn_list = sorted(fn)
    if len(fn_list) > max(8, max_targets // 4):
        fn_list = fn_list[: max(8, max_targets // 4)]
    for tg in fn_list:
        if tg not in G:
            G.add_node(tg, ntype="target")
        if not G.has_edge(tf, tg):
            G.add_edge(tf, tg, status="FN")
            edge_status[(tf, tg)] = "FN"

    return G, edge_status


def _draw_ego_on_ax(
    ax: plt.Axes,
    tf: str,
    pred_df: pd.DataFrame,
    gt_by_tf: Dict[str, Set[str]],
    max_targets: int,
    *,
    title_size: int = TEXT_SIZE - 1,
    tf_label_size: int = 10,
    target_label_size: int = 7,
) -> Tuple[int, int, int]:
    G, est = _ego_graph_for_tf(tf, pred_df, gt_by_tf, max_targets)
    tp_n = sum(1 for s in est.values() if s == "TP")
    fp_n = sum(1 for s in est.values() if s == "FP")
    fn_n = sum(1 for s in est.values() if s == "FN")

    pos = nx.spring_layout(G, seed=42, k=1.8 / max(G.number_of_nodes(), 1) ** 0.5)
    if tf in pos:
        pos[tf] = np.array([0.0, 0.0])

    ax.set_aspect("equal")
    ax.axis("off")

    for u, v, data in G.edges(data=True):
        st = data.get("status", "FP")
        color = EDGE_COLORS.get(st, "#888888")
        style = "dashed" if st == "FN" else "solid"
        width = 1.8 if st == "TP" else 1.2 if st == "FP" else 1.0
        alpha = 0.95 if st != "FN" else 0.55
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=width,
                linestyle=style,
                alpha=alpha,
                shrinkA=12,
                shrinkB=12,
            ),
        )

    for node, data in G.nodes(data=True):
        x, y = pos[node]
        if data.get("ntype") == "tf":
            ax.scatter([x], [y], s=520, c=NODE_TF, edgecolors="#333", linewidths=1.2, zorder=5)
            ax.text(
                x,
                y,
                node,
                ha="center",
                va="center",
                fontsize=tf_label_size,
                fontweight="bold",
                color="white",
                zorder=6,
            )
        else:
            ax.scatter([x], [y], s=180, c=NODE_TARGET, edgecolors="#666", linewidths=0.6, zorder=4)
            ax.text(
                x,
                y - 0.06,
                node,
                ha="center",
                va="top",
                fontsize=target_label_size,
                color="#333333",
                zorder=4,
            )

    ax.set_title(
        f"{tf}\nTP={tp_n}  FP={fp_n}  FN={fn_n}",
        fontsize=title_size,
        pad=4,
    )
    return tp_n, fp_n, fn_n


def plot_hub_ego_panels(
    hub_tfs: List[str],
    pred_df: pd.DataFrame,
    gt_by_tf: Dict[str, Set[str]],
    dataset: str,
    method: str,
    out_dir: Path,
    max_targets: int,
) -> None:
    n = len(hub_tfs)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.5 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, tf in zip(axes_flat, hub_tfs):
        _draw_ego_on_ax(ax, tf, pred_df, gt_by_tf, max_targets)

    for ax in axes_flat[len(hub_tfs) :]:
        ax.axis("off")

    legend_elems = [
        mpatches.Patch(color=EDGE_COLORS["TP"], label="TP (pred ∩ GT)"),
        mpatches.Patch(color=EDGE_COLORS["FP"], label="FP (pred only)"),
        mpatches.Patch(color=EDGE_COLORS["FN"], label="FN (GT only, dashed)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, frameon=False, fontsize=TEXT_SIZE - 1)
    fig.suptitle(
        f"{dataset} — Hub TF ego-networks ({method_label(method)})\n"
        f"Top-{n} TFs by GT out-degree",
        fontsize=TITLE_SIZE,
        y=1.02,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.98])
    stem = out_dir / f"{dataset}_{method}_02_hub_ego_networks"
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {stem}.png/.pdf")


# ---------------------------------------------------------------------------
# 3) Per-TF ORA on TP / FP / FN targets + biological gene-set enrichment
# ---------------------------------------------------------------------------

EDGE_ORA_META = {
    "TP": {
        "label": "true-positive",
        "short": "TP",
        "color": EDGE_COLORS["TP"],
        "count_col": "n_tp",
    },
    "FP": {
        "label": "false-positive",
        "short": "FP",
        "color": model_color("scPrint"),
        "count_col": "n_fp",
    },
    "FN": {
        "label": "false-negative",
        "short": "FN",
        "color": EDGE_COLORS["FN"],
        "count_col": "n_fn",
    },
}


def _edge_targets(tf: str, pred_df: pd.DataFrame, gt_by_tf: Dict[str, Set[str]], edge_type: str) -> Set[str]:
    tp, fp, fn = classify_tf_edges(tf, pred_df, gt_by_tf)
    if edge_type == "TP":
        return tp
    if edge_type == "FP":
        return fp
    if edge_type == "FN":
        return fn
    raise ValueError(f"Unknown edge_type {edge_type!r}")


def run_per_tf_edge_ora(
    hub_tfs: List[str],
    pred_df: pd.DataFrame,
    gt_by_tf: Dict[str, Set[str]],
    pathways: Dict[str, Set[str]],
    background: Set[str],
    edge_type: str,
    min_genes: int,
) -> pd.DataFrame:
    meta = EDGE_ORA_META[edge_type]
    count_col = meta["count_col"]
    rows_all = []
    for tf in hub_tfs:
        genes = _edge_targets(tf, pred_df, gt_by_tf, edge_type)
        if len(genes) < min_genes:
            continue
        ora = run_ora(genes, pathways, background)
        if ora.empty:
            continue
        top = ora.head(8).copy()
        top["TF"] = tf
        top["edge_type"] = edge_type
        top[count_col] = len(genes)
        rows_all.append(top)
    if not rows_all:
        return pd.DataFrame()
    return pd.concat(rows_all, ignore_index=True)


def run_per_tf_fp_ora(
    hub_tfs: List[str],
    pred_df: pd.DataFrame,
    gt_by_tf: Dict[str, Set[str]],
    pathways: Dict[str, Set[str]],
    background: Set[str],
    min_fp: int,
) -> pd.DataFrame:
    return run_per_tf_edge_ora(hub_tfs, pred_df, gt_by_tf, pathways, background, "FP", min_fp)


def run_bio_gene_set_enrichment(
    hub_tfs: List[str],
    pred_df: pd.DataFrame,
    gt_by_tf: Dict[str, Set[str]],
    gene_sets: Dict[str, tuple[str, Set[str]]],
    background: Set[str],
    edge_types: List[str],
    min_genes: int,
) -> pd.DataFrame:
    rows_all = []
    for tf in hub_tfs:
        for edge_type in edge_types:
            genes = _edge_targets(tf, pred_df, gt_by_tf, edge_type)
            n_genes = len(genes)
            if n_genes < min_genes:
                continue
            enrich = run_gene_set_enrichment(genes, gene_sets, background)
            if enrich.empty:
                continue
            enrich = enrich.copy()
            enrich["TF"] = tf
            enrich["edge_type"] = edge_type
            enrich["n_query"] = n_genes
            rows_all.append(enrich)
    if not rows_all:
        return pd.DataFrame()
    return pd.concat(rows_all, ignore_index=True)


def plot_per_tf_ora_lollipops(
    ora_df: pd.DataFrame,
    dataset: str,
    method: str,
    out_dir: Path,
    terms_per_tf: int,
    edge_type: str = "FP",
) -> None:
    if ora_df.empty:
        print(f"  [skip] per-TF {edge_type} ORA: no significant terms")
        return

    meta = EDGE_ORA_META[edge_type]
    tfs = list(dict.fromkeys(ora_df["TF"].tolist()))
    n = min(6, len(tfs))
    tfs = tfs[:n]
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.2 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, tf in zip(axes_flat, tfs):
        sub = ora_df[ora_df["TF"] == tf].copy()
        sub = sub.sort_values("neglog10P", ascending=False).head(int(terms_per_tf))
        if sub.empty:
            ax.axis("off")
            continue
        sub = sub.sort_values("neglog10P", ascending=True)
        labels = [shorten_term(t, 42) for t in sub["Term"]]
        y = np.arange(len(labels))
        colors = [meta["color"] if p < 0.05 else "#B0B0B0" for p in sub["PValue"]]
        ax.hlines(y, 0, sub["neglog10P"], color="#E8E8E8", lw=2, zorder=1)
        ax.scatter(sub["neglog10P"], y, c=colors, s=64, zorder=3, edgecolors="#444", linewidths=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(r"$-\log_{10} P$", fontsize=TEXT_SIZE - 2)
        n_q = int(sub[meta["count_col"]].iloc[0]) if meta["count_col"] in sub.columns else 0
        ax.set_title(
            f"{tf}  ({meta['short']} targets n={n_q})",
            fontsize=TEXT_SIZE - 1,
        )
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for ax in axes_flat[len(tfs) :]:
        ax.axis("off")

    fig.suptitle(
        f"{dataset} — Per-TF pathway enrichment ({meta['label']} targets)\n"
        f"{method_label(method)} | C2+Hallmark GMT",
        fontsize=TITLE_SIZE,
        y=1.01,
    )
    fig.tight_layout()
    stem = out_dir / f"{dataset}_{method}_03_per_tf_{edge_type.lower()}_ora"
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {stem}.png/.pdf")


def plot_per_tf_ora_heatmap(
    ora_df: pd.DataFrame,
    dataset: str,
    method: str,
    out_dir: Path,
    edge_type: str = "FP",
) -> None:
    """Summary heatmap: hub TF × top pathway (-log10 P)."""
    if ora_df.empty:
        return
    meta = EDGE_ORA_META[edge_type]
    records = []
    for tf, sub in ora_df.groupby("TF"):
        best = sub.sort_values("neglog10P", ascending=False).head(1)
        if best.empty:
            continue
        records.append(
            {
                "TF": tf,
                "Term": shorten_term(best["Term"].iloc[0], 36),
                "neglog10P": float(best["neglog10P"].iloc[0]),
                "n_query": int(best[meta["count_col"]].iloc[0]),
            }
        )
    summ = pd.DataFrame(records).sort_values("neglog10P", ascending=False)
    if summ.empty:
        return

    fig, ax = plt.subplots(figsize=(8, max(4, 0.38 * len(summ))))
    vals = summ["neglog10P"].to_numpy().reshape(-1, 1)
    im = ax.imshow(vals, aspect="auto", cmap="OrRd", vmin=0)
    ax.set_xticks([0])
    ax.set_xticklabels(["Best term −log₁₀P"])
    ax.set_yticks(np.arange(len(summ)))
    ax.set_yticklabels(
        [f"{r.TF} ({meta['short']}={r.n_query})" for r in summ.itertuples()],
        fontsize=TEXT_SIZE - 1,
    )
    for i, r in enumerate(summ.itertuples()):
        ax.text(0, i, f"{r.neglog10P:.1f}\n{r.Term}", ha="center", va="center", fontsize=8, color="#111")
    ax.set_title(f"{dataset} — Per-TF top enriched pathway ({meta['label']} targets)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    stem = out_dir / f"{dataset}_{method}_03b_per_tf_{edge_type.lower()}_ora_summary"
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {stem}.png/.pdf")


def plot_bio_gene_set_heatmap(
    enrich_df: pd.DataFrame,
    dataset: str,
    method: str,
    out_dir: Path,
) -> None:
    if enrich_df.empty:
        print("  [skip] biological gene-set enrichment: no results")
        return

    tfs = list(dict.fromkeys(enrich_df["TF"].tolist()))
    set_ids = list(dict.fromkeys(enrich_df["set_id"].tolist()))
    set_labels = []
    for sid in set_ids:
        sub = enrich_df[enrich_df["set_id"] == sid]
        set_labels.append(sub["display_name"].iloc[0] if "display_name" in sub.columns else sid)

    edge_types = [et for et in ("TP", "FP", "FN") if et in set(enrich_df["edge_type"])]
    fig, axes = plt.subplots(1, len(edge_types), figsize=(5.2 * len(edge_types), max(4.5, 0.42 * len(tfs))))
    if len(edge_types) == 1:
        axes = [axes]

    for ax, edge_type in zip(axes, edge_types):
        meta = EDGE_ORA_META[edge_type]
        mat = np.zeros((len(tfs), len(set_ids)), dtype=float)
        annot = [["" for _ in set_ids] for _ in tfs]
        sub_all = enrich_df[enrich_df["edge_type"] == edge_type]
        for i, tf in enumerate(tfs):
            sub = sub_all[sub_all["TF"] == tf]
            for j, sid in enumerate(set_ids):
                row = sub[sub["set_id"] == sid]
                if row.empty:
                    continue
                p = float(row["neglog10P"].iloc[0])
                mat[i, j] = min(p, 12.0)
                k = int(row["Overlap"].iloc[0])
                n = int(row["n_query"].iloc[0])
                annot[i][j] = f"{k}/{n}"

        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(3.0, float(np.nanmax(mat))))
        ax.set_xticks(np.arange(len(set_ids)))
        ax.set_xticklabels([shorten_term(l, 22) for l in set_labels], rotation=35, ha="right", fontsize=9)
        ax.set_yticks(np.arange(len(tfs)))
        ax.set_yticklabels(tfs, fontsize=TEXT_SIZE - 1)
        ax.set_title(f"{meta['short']} targets", fontsize=TEXT_SIZE)
        for i in range(len(tfs)):
            for j in range(len(set_ids)):
                if mat[i, j] <= 0:
                    continue
                txt = annot[i][j]
                if txt:
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="#111")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        f"{dataset} — Biological gene-set enrichment per TF\n"
        f"{method_label(method)} | cell text = overlap/query size",
        fontsize=TITLE_SIZE,
        y=1.02,
    )
    fig.tight_layout()
    stem = out_dir / f"{dataset}_{method}_04_bio_gene_set_enrichment"
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {stem}.png/.pdf")


def select_analysis_tfs(
    metrics: pd.DataFrame,
    pred_df: pd.DataFrame,
    gt_by_tf: Dict[str, Set[str]],
    tf_list: Optional[List[str]],
    top_hub: int,
    hub_rank_by: str,
    min_gt_degree: int,
    include_pluripotency_tfs: bool,
) -> List[str]:
    if tf_list:
        chosen = [tf.strip().upper() for tf in tf_list if tf.strip()]
        seen: Set[str] = set()
        out: List[str] = []
        for tf in chosen:
            if tf not in seen:
                seen.add(tf)
                out.append(tf)
        return out

    hub_pool = metrics[metrics["gt_outdegree"] >= min_gt_degree].copy()
    if hub_rank_by == "fp_count":
        fp_counts = []
        for tf in hub_pool["TF"]:
            _, fp, _ = classify_tf_edges(tf, pred_df, gt_by_tf)
            fp_counts.append(len(fp))
        hub_pool["fp_count"] = fp_counts
        hub_pool = hub_pool.sort_values(["fp_count", "gt_outdegree"], ascending=[False, False])
    elif hub_rank_by == "low_jaccard":
        hub_pool = hub_pool.sort_values(["jaccard", "gt_outdegree"], ascending=[True, False])
    else:
        hub_pool = hub_pool.sort_values("gt_outdegree", ascending=False)

    hub_tfs = hub_pool["TF"].head(int(top_hub)).tolist()
    if include_pluripotency_tfs:
        for tf in DEFAULT_PLURIPOTENCY_TFS:
            if tf in gt_by_tf and tf not in hub_tfs:
                hub_tfs.append(tf)
    return hub_tfs


def _draw_bio_gene_set_ax(
    ax: plt.Axes,
    enrich_df: pd.DataFrame,
    tfs: List[str],
    set_ids: List[str],
    set_labels: List[str],
    edge_type: str,
    *,
    label_size: int = 8,
    annot_size: int = 6,
) -> None:
    meta = EDGE_ORA_META[edge_type]
    mat = np.zeros((len(tfs), len(set_ids)), dtype=float)
    annot = [["" for _ in set_ids] for _ in tfs]
    sub_all = enrich_df[enrich_df["edge_type"] == edge_type]
    for i, tf in enumerate(tfs):
        sub = sub_all[sub_all["TF"] == tf]
        for j, sid in enumerate(set_ids):
            row = sub[sub["set_id"] == sid]
            if row.empty:
                continue
            p = float(row["neglog10P"].iloc[0])
            mat[i, j] = min(p, 12.0)
            k = int(row["Overlap"].iloc[0])
            n = int(row["n_query"].iloc[0])
            annot[i][j] = f"{k}/{n}"

    vmax = max(3.0, float(np.nanmax(mat))) if np.any(mat > 0) else 3.0
    ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(len(set_ids)))
    ax.set_xticklabels([shorten_term(l, 18) for l in set_labels], rotation=40, ha="right", fontsize=label_size)
    ax.set_yticks(np.arange(len(tfs)))
    ax.set_yticklabels(tfs, fontsize=label_size + 1)
    ax.set_title(f"{meta['short']} targets", fontsize=TEXT_SIZE - 1, pad=6)
    for i in range(len(tfs)):
        for j in range(len(set_ids)):
            if mat[i, j] <= 0:
                continue
            txt = annot[i][j]
            if txt:
                ax.text(j, i, txt, ha="center", va="center", fontsize=annot_size, color="#111")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _draw_ora_lollipop_ax(
    ax: plt.Axes,
    ora_df: pd.DataFrame,
    tf: str,
    edge_type: str,
    terms_per_tf: int,
    *,
    label_size: int = 8,
) -> bool:
    meta = EDGE_ORA_META[edge_type]
    sub = ora_df[(ora_df["TF"] == tf) & (ora_df["edge_type"] == edge_type)].copy()
    if sub.empty:
        sub = ora_df[ora_df["TF"] == tf].copy()
    if sub.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, f"No ORA for {tf}", ha="center", va="center", transform=ax.transAxes)
        return False
    sub = sub.sort_values("neglog10P", ascending=False).head(int(terms_per_tf))
    if sub.empty:
        ax.axis("off")
        return False
    sub = sub.sort_values("neglog10P", ascending=True)
    labels = [shorten_term(t, 34) for t in sub["Term"]]
    y = np.arange(len(labels))
    colors = [meta["color"] if p < 0.05 else "#B0B0B0" for p in sub["PValue"]]
    ax.hlines(y, 0, sub["neglog10P"], color="#E8E8E8", lw=1.8, zorder=1)
    ax.scatter(sub["neglog10P"], y, c=colors, s=48, zorder=3, edgecolors="#444", linewidths=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=label_size)
    ax.set_xlabel(r"$-\log_{10} P$", fontsize=label_size)
    n_q = int(sub[meta["count_col"]].iloc[0]) if meta["count_col"] in sub.columns else 0
    ax.set_title(f"{tf} ({meta['short']}, n={n_q})", fontsize=TEXT_SIZE - 1)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return True


def _panel_label(ax: plt.Axes, label: str, y: float = 1.06) -> None:
    ax.text(
        -0.08,
        y,
        label,
        transform=ax.transAxes,
        fontsize=TITLE_SIZE + 1,
        fontweight="bold",
        va="top",
        ha="right",
    )


def plot_tf_bio_interpretability_composite(
    ego_tfs: List[str],
    pred_df: pd.DataFrame,
    gt_by_tf: Dict[str, Set[str]],
    bio_df: pd.DataFrame,
    ora_by_type: Dict[str, pd.DataFrame],
    dataset: str,
    method: str,
    out_dir: Path,
    *,
    max_targets: int = 25,
    bio_tfs: Optional[List[str]] = None,
    case_studies: Tuple[Tuple[str, str], ...] = DEFAULT_CASE_STUDIES,
    ora_terms: int = 4,
) -> None:
    """Single multi-panel figure: ego nets + bio gene sets + case-study ORA."""
    ego_tfs = ego_tfs[:6]
    if not ego_tfs:
        print("  [skip] composite: no ego TFs")
        return

    if bio_tfs is None:
        bio_tfs = list(dict.fromkeys(ego_tfs + [t for t in DEFAULT_PLURIPOTENCY_TFS[:3] if t in gt_by_tf]))
    else:
        bio_tfs = list(dict.fromkeys(bio_tfs))

    fig = plt.figure(figsize=(16.5, 20))
    gs = GridSpec(
        4,
        3,
        figure=fig,
        height_ratios=[1.0, 1.0, 1.35, 0.85],
        hspace=0.42,
        wspace=0.28,
        top=0.94,
        bottom=0.06,
        left=0.07,
        right=0.98,
    )

    # --- A: Hub ego-networks (2×3) ---
    for i, tf in enumerate(ego_tfs):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        _draw_ego_on_ax(
            ax,
            tf,
            pred_df,
            gt_by_tf,
            max_targets,
            title_size=TEXT_SIZE - 2,
            tf_label_size=9,
            target_label_size=6,
        )
        if i == 0:
            _panel_label(ax, "A")

    legend_elems = [
        mpatches.Patch(color=EDGE_COLORS["TP"], label="TP (pred ∩ GT)"),
        mpatches.Patch(color=EDGE_COLORS["FP"], label="FP (pred only)"),
        mpatches.Patch(color=EDGE_COLORS["FN"], label="FN (GT only, dashed)"),
    ]
    fig.legend(
        handles=legend_elems,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=3,
        frameon=False,
        fontsize=TEXT_SIZE - 1,
    )

    # --- B: Biological gene-set enrichment ---
    if not bio_df.empty and bio_tfs:
        set_ids = list(dict.fromkeys(bio_df["set_id"].tolist()))
        set_labels = []
        for sid in set_ids:
            sub = bio_df[bio_df["set_id"] == sid]
            set_labels.append(sub["display_name"].iloc[0] if "display_name" in sub.columns else sid)
        gs_bio = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2, :], wspace=0.18)
        for j, edge_type in enumerate(("TP", "FP", "FN")):
            ax = fig.add_subplot(gs_bio[0, j])
            _draw_bio_gene_set_ax(ax, bio_df, bio_tfs, set_ids, set_labels, edge_type)
            if j == 0:
                _panel_label(ax, "B", y=1.12)
    else:
        ax = fig.add_subplot(gs[2, :])
        ax.axis("off")
        ax.text(0.5, 0.5, "No biological gene-set enrichment", ha="center", va="center")

    # --- C: Case-study GMT ORA ---
    gs_ora = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[3, :], wspace=0.32)
    for j, (tf, edge_type) in enumerate(case_studies):
        ax = fig.add_subplot(gs_ora[0, j])
        ora_df = ora_by_type.get(edge_type, pd.DataFrame())
        if ora_df.empty:
            ax.axis("off")
            continue
        _draw_ora_lollipop_ax(ax, ora_df, tf, edge_type, ora_terms, label_size=7)
        if j == 0:
            _panel_label(ax, "C", y=1.18)

    fig.suptitle(
        f"{dataset} — TF biological interpretability ({method_label(method)})\n"
        f"A: hub ego-networks | B: curated gene-set enrichment | C: case-study pathway ORA",
        fontsize=TITLE_SIZE + 1,
        y=0.995,
    )
    stem = out_dir / f"{dataset}_{method}_05_tf_bio_interpretability_composite"
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {stem}.png/.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TF static GRN deep-dive (3 analyses)")
    p.add_argument("--dataset", default="hESC")
    p.add_argument("--method", default="embhidden500", choices=list(METHOD_PRED))
    p.add_argument("--gt-dir", type=Path, default=Path("/mnt/10T/yzn/benchmark_GRN/input_process/STRING"))
    p.add_argument("--pred-root", type=Path, default=Path("/mnt/10T/yzn/benchmark_GRN/evl_omipath"))
    p.add_argument("--out", type=Path, default=None, help="Output directory (default: tf_static/output/<dataset>)")
    p.add_argument("--heatmap-top", type=int, default=35, help="TF rows in Jaccard heatmap")
    p.add_argument("--top-hub", type=int, default=6, help="Hub TFs for ego + ORA panels")
    p.add_argument(
        "--tf-list",
        nargs="*",
        default=None,
        help="Explicit TF list for ego/ORA (overrides --top-hub ranking)",
    )
    p.add_argument(
        "--include-pluripotency-tfs",
        action="store_true",
        help="Append SOX2/NANOG/POU5F1/KLF4/LIN28A to hub TF list",
    )
    p.add_argument(
        "--hub-rank-by",
        choices=["gt_outdegree", "fp_count", "low_jaccard"],
        default="gt_outdegree",
        help="How to pick hub TFs for ego/ORA (outdegree plot always uses all TFs)",
    )
    p.add_argument("--min-gt-degree", type=int, default=5, help="Min GT out-degree for plots")
    p.add_argument("--max-ego-targets", type=int, default=25, help="Max targets drawn per ego net")
    p.add_argument("--min-fp-ora", type=int, default=5, help="Min FP genes to run ORA per TF")
    p.add_argument("--min-edge-ora", type=int, default=3, help="Min TP/FN genes to run ORA per TF")
    p.add_argument("--ora-terms", type=int, default=5, help="Terms per TF in lollipop panel")
    p.add_argument(
        "--bio-gene-sets",
        type=Path,
        default=BIO_GENE_SETS_CSV,
        help="Curated biological gene sets CSV for TP/FP/FN enrichment",
    )
    p.add_argument("--skip-ora", action="store_true")
    p.add_argument("--skip-bio-gene-sets", action="store_true")
    p.add_argument("--skip-composite", action="store_true", help="Skip combined interpretability figure")
    p.add_argument(
        "--composite-only",
        action="store_true",
        help="Only build composite from existing outputs in --out (requires prior run)",
    )
    return p.parse_args()


def _load_ora_csv(out_dir: Path, dataset: str, method: str, edge_type: str) -> pd.DataFrame:
    path = out_dir / f"{dataset}_{method}_per_tf_{edge_type.lower()}_ora.csv"
    if path.is_file():
        return pd.read_csv(path)
    return pd.DataFrame()


def build_composite_from_outputs(
    args: argparse.Namespace,
    out_dir: Path,
    hub_tfs: List[str],
    pred_df: pd.DataFrame,
    gt_by_tf: Dict[str, Set[str]],
) -> None:
    bio_path = out_dir / f"{args.dataset}_{args.method}_bio_gene_set_enrichment.csv"
    bio_df = pd.read_csv(bio_path) if bio_path.is_file() else pd.DataFrame()
    ora_by_type = {
        et: _load_ora_csv(out_dir, args.dataset, args.method, et) for et in ("TP", "FP", "FN")
    }
    ego_tfs = hub_tfs[: int(args.top_hub)]
    plot_tf_bio_interpretability_composite(
        ego_tfs,
        pred_df,
        gt_by_tf,
        bio_df,
        ora_by_type,
        args.dataset,
        args.method,
        out_dir,
        max_targets=args.max_ego_targets,
        ora_terms=args.ora_terms,
    )


def main() -> None:
    args = parse_args()
    apply_style()

    out_dir = args.out if args.out is not None else FIG3 / "tf_static" / "output" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_path = args.gt_dir / f"{args.dataset}_processed-network.csv"
    pred_path = resolve_pred_path(args.method, args.pred_root, args.dataset)
    if not gt_path.is_file():
        raise FileNotFoundError(gt_path)
    if not pred_path.is_file():
        raise FileNotFoundError(pred_path)

    print(f"Dataset: {args.dataset}  Method: {args.method}")
    print(f"GT:   {gt_path}")
    print(f"Pred: {pred_path}")
    print(f"Out:  {out_dir}")

    gt_gene1, gt_all, gt_n, _, gt_by_tf = load_gt_network(gt_path)
    pred_df = filter_prediction(pred_path, gt_gene1, gt_all, gt_n)

    hub_tfs = select_analysis_tfs(
        per_tf_edge_sets(pred_df, gt_by_tf),
        pred_df,
        gt_by_tf,
        args.tf_list,
        args.top_hub,
        args.hub_rank_by,
        args.min_gt_degree,
        args.include_pluripotency_tfs,
    )

    if args.composite_only:
        build_composite_from_outputs(args, out_dir, hub_tfs, pred_df, gt_by_tf)
        print("Done (composite only).")
        return

    metrics = per_tf_edge_sets(pred_df, gt_by_tf)
    metrics.to_csv(out_dir / f"{args.dataset}_{args.method}_per_tf_metrics.csv", index=False)
    print(f"  TFs with edges: {len(metrics)}")

    # 1) Out-degree + Jaccard
    plot_outdegree_and_jaccard(
        metrics,
        args.dataset,
        args.method,
        out_dir,
        heatmap_top=args.heatmap_top,
        min_gt_degree=args.min_gt_degree,
    )

    # Hub list (for ego + ORA)
    hub_tfs = select_analysis_tfs(
        metrics,
        pred_df,
        gt_by_tf,
        args.tf_list,
        args.top_hub,
        args.hub_rank_by,
        args.min_gt_degree,
        args.include_pluripotency_tfs,
    )
    print(f"  Analysis TFs ({len(hub_tfs)}): {hub_tfs}")

    ora_by_type: Dict[str, pd.DataFrame] = {}
    bio_df = pd.DataFrame()

    # 2) Ego networks
    plot_hub_ego_panels(
        hub_tfs,
        pred_df,
        gt_by_tf,
        args.dataset,
        args.method,
        out_dir,
        max_targets=args.max_ego_targets,
    )

    # 3) Per-TF ORA (TP / FP / FN)
    if not args.skip_ora:
        pathways = load_pathways(GMT_HUMAN)
        if not pathways:
            print("  [warn] No GMT pathways loaded; skip ORA")
        else:
            background = gt_all
            for edge_type, min_genes in (
                ("FP", args.min_fp_ora),
                ("TP", args.min_edge_ora),
                ("FN", args.min_edge_ora),
            ):
                ora_df = run_per_tf_edge_ora(
                    hub_tfs,
                    pred_df,
                    gt_by_tf,
                    pathways,
                    background,
                    edge_type,
                    min_genes,
                )
                ora_by_type[edge_type] = ora_df
                if not ora_df.empty:
                    ora_df.to_csv(
                        out_dir / f"{args.dataset}_{args.method}_per_tf_{edge_type.lower()}_ora.csv",
                        index=False,
                    )
                plot_per_tf_ora_lollipops(
                    ora_df, args.dataset, args.method, out_dir, args.ora_terms, edge_type=edge_type
                )
                plot_per_tf_ora_heatmap(ora_df, args.dataset, args.method, out_dir, edge_type=edge_type)

    # 4) Curated biological gene-set enrichment (pluripotency / Pol II / etc.)
    if not args.skip_bio_gene_sets:
        gene_sets = load_gene_set_table(args.bio_gene_sets)
        if not gene_sets:
            print(f"  [warn] No biological gene sets at {args.bio_gene_sets}; skip")
        else:
            bio_df = run_bio_gene_set_enrichment(
                hub_tfs,
                pred_df,
                gt_by_tf,
                gene_sets,
                gt_all,
                ["TP", "FP", "FN"],
                min_genes=args.min_edge_ora,
            )
            if not bio_df.empty:
                bio_df.to_csv(
                    out_dir / f"{args.dataset}_{args.method}_bio_gene_set_enrichment.csv",
                    index=False,
                )
                plot_bio_gene_set_heatmap(bio_df, args.dataset, args.method, out_dir)
            else:
                print("  [skip] biological gene-set enrichment: no TF passed min gene count")

    if not args.skip_composite:
        build_composite_from_outputs(args, out_dir, hub_tfs, pred_df, gt_by_tf)

    print("Done.")


if __name__ == "__main__":
    main()
