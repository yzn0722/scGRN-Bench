#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split enrichment visualization into two PDFs:
1) Top-N pathway lollipop plot
2) Gene-concept network plot

Usage example:
python plot_enrichment_split.py \
  --enrich-csv /path/to/enrichment_results.csv \
  --top-n 10 \
  --outdir /mnt/10T/yzn/FoundBench/FBplot/fig3/tf_vene
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from fig3_palette import model_color

TEXT_SIZE = 16


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot enrichment as separate lollipop/network PDFs")
    p.add_argument("--enrich-csv", type=str, required=True, help="Path to enrichment result CSV")
    p.add_argument("--top-n", type=int, default=10, help="Top-N pathways to plot")
    p.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parent / "tf_vene"),
        help="Output directory for PDF figures",
    )
    return p.parse_args()


def _pick_first(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in cols}
    for key in candidates:
        if key.lower() in lower_map:
            return lower_map[key.lower()]
    return None


def _split_genes(x) -> List[str]:
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    for sep in [";", ",", "/", "|"]:
        s = s.replace(sep, " ")
    genes = [g.strip() for g in s.split() if g.strip()]
    return genes


def load_enrichment(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Empty enrichment table: {csv_path}")

    term_col = _pick_first(df.columns.tolist(), ["Term", "Pathway", "Description"])
    p_col = _pick_first(
        df.columns.tolist(),
        ["Adjusted P-value", "Adjusted_P-value", "adj_p", "padj", "FDR", "P-value", "PValue"],
    )
    gene_col = _pick_first(df.columns.tolist(), ["Genes", "Lead_genes", "Gene", "geneID"])

    if term_col is None or p_col is None:
        raise ValueError(
            "Cannot find required columns. Need pathway/term + p-value columns "
            "(e.g., Term, Adjusted P-value)."
        )

    out = df.copy()
    out = out.rename(columns={term_col: "Term", p_col: "PValue"})
    out["PValue"] = pd.to_numeric(out["PValue"], errors="coerce")
    out = out[np.isfinite(out["PValue"])].copy()
    out["neglog10P"] = -np.log10(np.clip(out["PValue"], 1e-300, None))
    out = out.sort_values("PValue", ascending=True).reset_index(drop=True)
    if gene_col is not None:
        out = out.rename(columns={gene_col: "Genes"})
    else:
        out["Genes"] = ""
    return out


def plot_lollipop(df: pd.DataFrame, out_pdf: Path, top_n: int) -> None:
    d = df.head(max(1, int(top_n))).copy()
    d = d.iloc[::-1]  # top at top visually

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    y = np.arange(len(d))
    x = d["neglog10P"].to_numpy(dtype=float)

    ax.hlines(y, 0, x, color=model_color("scCello"), linewidth=2.0, alpha=0.85)
    ax.scatter(x, y, s=85, color=model_color("scGPT"), edgecolors="none", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(d["Term"].tolist(), fontsize=TEXT_SIZE)
    ax.set_xlabel(r"$-\log_{10}(P)$", fontsize=TEXT_SIZE)
    ax.set_title("Enrichment Top Pathways", fontsize=TEXT_SIZE, color="#000000")
    ax.tick_params(axis="x", labelsize=TEXT_SIZE, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", format="pdf")
    plt.close(fig)


def plot_gene_concept(df: pd.DataFrame, out_pdf: Path, top_n: int) -> None:
    d = df.head(max(1, int(top_n))).copy()

    g = nx.Graph()
    pathway_nodes = []
    gene_nodes = set()
    for _, row in d.iterrows():
        term = str(row["Term"])
        pnode = f"path::{term}"
        pathway_nodes.append(pnode)
        g.add_node(pnode, kind="pathway", label=term)
        genes = _split_genes(row.get("Genes", ""))
        for gene in genes:
            gnode = f"gene::{gene}"
            gene_nodes.add(gnode)
            g.add_node(gnode, kind="gene", label=gene)
            g.add_edge(pnode, gnode)

    if g.number_of_nodes() == 0:
        raise ValueError("No pathway/gene links found. Check Genes column in enrichment CSV.")

    # Bipartite-like layout: pathways left, genes right
    pos = {}
    p_sorted = pathway_nodes
    g_sorted = sorted(gene_nodes)
    for i, n in enumerate(p_sorted):
        pos[n] = (0.0, -i)
    for i, n in enumerate(g_sorted):
        pos[n] = (1.0, -i * (max(1, len(p_sorted)) / max(1, len(g_sorted))))

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=model_color("STRING"), width=1.0, alpha=0.6)

    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=p_sorted,
        node_size=520,
        node_color=model_color("LangCell"),
        edgecolors="none",
        ax=ax,
    )
    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=g_sorted,
        node_size=230,
        node_color=model_color("scGPT"),
        edgecolors="none",
        ax=ax,
    )

    labels = {n: g.nodes[n]["label"] for n in g.nodes}
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=TEXT_SIZE, font_color="#111111", ax=ax)

    ax.set_title("Gene-Concept Network", fontsize=TEXT_SIZE, color="#000000")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", format="pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    enrich_csv = Path(args.enrich_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_enrichment(enrich_csv)
    out_lollipop = outdir / "enrichment_lollipop_topN.pdf"
    out_network = outdir / "enrichment_gene_concept_topN.pdf"

    plot_lollipop(df, out_lollipop, args.top_n)
    plot_gene_concept(df, out_network, args.top_n)

    print(f"Saved: {out_lollipop}")
    print(f"Saved: {out_network}")


if __name__ == "__main__":
    main()

