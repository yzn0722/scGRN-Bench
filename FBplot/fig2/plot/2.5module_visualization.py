#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-edge community visualization with STRING-based gene filtering.

Features:
1) Use REAL predicted edge tables (not simulated graphs).
2) Apply STRING-based gene filtering:
   Gene1 in STRING(Gene1) and Gene2 in STRING(Gene1 ∪ Gene2).
3) Support switches for dataset(s) and model(s).
4) Plot model-wise community comparison for each dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from fig2_palette import PALETTE, model_color


ROOT = Path("/mnt/10T/yzn/benchmark_GRN")
PRED_ROOT = ROOT / "evl_omipath"
STRING_ROOT = ROOT / "input_process/STRING"
OUT_DIR = Path(__file__).resolve().parent / "output" / "community_comparison_real_edges"

ALL_DATASETS = ["hESC", "hHep", "mDC", "mHSC-E", "mHSC-GM", "mHSC-L"]
ALL_MODELS = ["Geneformer", "LangCell", "scGPT", "scFoundation", "scPRINT", "scCello"]

MODEL_FILE_STEMS: Dict[str, Sequence[str]] = {
    "Geneformer": ("geneformer", "Geneformer"),
    "LangCell": ("LangCell", "Langcell", "langcell"),
    "scGPT": ("scgpt", "scGPT"),
    "scFoundation": ("scFoundation", "scfoundation"),
    "scPRINT": ("scprint", "scPRINT", "scPrint"),
    "scCello": ("scCello", "sccello", "scChello"),
}

MODEL_DIR_NAMES: Dict[str, Sequence[str]] = {
    "Geneformer": ("geneformer",),
    "LangCell": ("langcell", "Langcell"),
    "scGPT": ("scgpt",),
    "scFoundation": ("scFoundation",),
    "scPRINT": ("scprint",),
    "scCello": ("sccello",),
}

WEIGHT_COL_CANDIDATES = ("EdgeWeight", "weight", "score", "importance", "Weight")
MAX_PLOT_NODES = 350
TOP_EDGES_FACTOR = 1.0  # use top K edges where K ~= STRING edge count
THREE_EXTRACT = ("emb500", "att500", "embhidden500")
EXTRACTION_TITLE = {
    # 命名与本文件夹保持一致：costok / attn / coshid
    "emb500": r"cos$_{tok}$",
    "att500": "attn",
    "embhidden500": r"cos$_{hid}$",
}
EXTRACT_COLORS = {
    "emb500": model_color("scGPT"),
    "att500": model_color("LangCell"),
    "embhidden500": model_color("scFoundation"),
}

# ------------------- Plot style -------------------
TITLE_SIZE = 16
TITLE_COLOR = "#000000"
SCATTER_NODE_SIZE = 180  # circles bigger
GRID_NODE_SIZE = 60      # circles bigger (multi-model panels)

plt.rcParams.update(
    {
        "font.size": TITLE_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "text.color": TITLE_COLOR,
    }
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Community visualization from REAL GRN edges")
    p.add_argument("--dataset", type=str, default="hESC", help="One dataset or comma list, or 'all'")
    p.add_argument(
        "--models",
        type=str,
        default="LangCell,scGPT,scFoundation,scPRINT,scCello,Geneformer",
        help="Comma list of models. Example: LangCell,scGPT,scFoundation",
    )
    p.add_argument(
        "--extraction",
        type=str,
        default="att500",
        choices=["emb500", "att500", "embhidden500", "emb1000", "att1000", "embhidden1000"],
        help="Prediction folder suffix: output_<extraction>",
    )
    p.add_argument("--list-datasets", action="store_true", help="List datasets and exit")
    p.add_argument("--list-models", action="store_true", help="List supported models and exit")
    p.add_argument("--max-plot-nodes", type=int, default=MAX_PLOT_NODES)
    p.add_argument(
        "--scgpt-three-scatter",
        action="store_true",
        help="Plot only scGPT as 3-extraction scatter panels (emb500/att500/embhidden500)",
    )
    return p.parse_args()


def _parse_list_arg(raw: str, all_values: Sequence[str]) -> List[str]:
    s = (raw or "").strip()
    if not s or s.lower() == "all":
        return list(all_values)
    vals = [x.strip() for x in s.split(",") if x.strip()]
    uniq: List[str] = []
    for v in vals:
        if v not in uniq:
            uniq.append(v)
    return uniq


def _normalize_edges(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Gene1" not in out.columns or "Gene2" not in out.columns:
        raise ValueError("Edge file must contain Gene1 and Gene2")
    out["Gene1"] = out["Gene1"].astype(str).str.strip().str.upper()
    out["Gene2"] = out["Gene2"].astype(str).str.strip().str.upper()
    out = out[(out["Gene1"] != "") & (out["Gene2"] != "") & (out["Gene1"] != out["Gene2"])]
    out = out.drop_duplicates(subset=["Gene1", "Gene2"])
    return out


def _weight_col(df: pd.DataFrame) -> Optional[str]:
    for c in WEIGHT_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def load_string_filter(dataset: str) -> Tuple[set, set, int]:
    fp = STRING_ROOT / f"{dataset}_processed-network.csv"
    if not fp.is_file():
        raise FileNotFoundError(f"STRING file not found: {fp}")
    s = pd.read_csv(fp)
    s = _normalize_edges(s)
    gene1_set = set(s["Gene1"].unique().tolist())
    gene_union = set(pd.unique(pd.concat([s["Gene1"], s["Gene2"]], axis=0)))
    return gene1_set, gene_union, int(len(s))


def resolve_pred_path(model: str, dataset: str, extraction: str) -> Path:
    base = PRED_ROOT / f"output_{extraction}"
    for d in MODEL_DIR_NAMES.get(model, (model,)):
        folder = base / d
        if not folder.is_dir():
            continue
        for stem in MODEL_FILE_STEMS.get(model, (model,)):
            cand = folder / f"{stem}_{dataset}.tsv"
            if cand.is_file():
                return cand
    raise FileNotFoundError(f"No prediction file for model={model}, dataset={dataset}, extraction={extraction}")


def filter_pred_by_string(df: pd.DataFrame, string_gene1: set, string_gene_union: set, keep_top_k: int) -> pd.DataFrame:
    d = _normalize_edges(df)
    d = d[d["Gene1"].isin(string_gene1) & d["Gene2"].isin(string_gene_union)].copy()
    if d.empty:
        return d
    wcol = _weight_col(d)
    if wcol is None:
        d["__w__"] = np.linspace(1.0, 0.0, num=len(d), endpoint=False)
        wcol = "__w__"
    d[wcol] = pd.to_numeric(d[wcol], errors="coerce").fillna(0.0)
    d = d.sort_values(wcol, ascending=False)
    if keep_top_k > 0:
        d = d.head(keep_top_k).copy()
    if "__w__" in d.columns:
        d = d.drop(columns=["__w__"])
    return d


def build_plot_graph(filtered_edges: pd.DataFrame, max_nodes: int) -> nx.Graph:
    g = nx.from_pandas_edgelist(filtered_edges, source="Gene1", target="Gene2")
    if g.number_of_nodes() <= max_nodes:
        return g
    deg = sorted(g.degree(), key=lambda x: x[1], reverse=True)
    keep_nodes = [n for n, _ in deg[:max_nodes]]
    return g.subgraph(keep_nodes).copy()


def detect_communities(g: nx.Graph) -> Tuple[List[set], float]:
    if g.number_of_nodes() == 0:
        return [], 0.0
    try:
        import community as community_louvain  # python-louvain

        part = community_louvain.best_partition(g, random_state=42)
        bucket: Dict[int, set] = {}
        for n, cid in part.items():
            bucket.setdefault(int(cid), set()).add(n)
        comms = list(bucket.values())
    except Exception:
        from networkx.algorithms.community import greedy_modularity_communities

        comms = [set(x) for x in greedy_modularity_communities(g)]
    from networkx.algorithms.community import modularity

    mod = float(modularity(g, comms)) if comms else 0.0
    return comms, mod


def plot_scgpt_three_scatter(dataset: str, max_plot_nodes: int) -> pd.DataFrame:
    """
    Plot scGPT as three separate single figures (one per extraction):
      emb500 / att500 / embhidden500
    """
    model = "scGPT"
    string_gene1, string_gene_union, string_edge_n = load_string_filter(dataset)
    rows = []

    # Use the same fixed palette as other figures in this folder.
    fixed_comm_colors = [
        PALETTE["scGPT"],
        PALETTE["Geneformer"],
        PALETTE["GENIE3"],
        PALETTE["LangCell"],
        PALETTE["scCello"],
        PALETTE["scFoundation"],
        PALETTE["scPrint"],
    ]
    for extraction in THREE_EXTRACT:
        fig, ax = plt.subplots(1, 1, figsize=(6.8, 6.8))
        try:
            pred_fp = resolve_pred_path(model, dataset, extraction)
            pred_df = pd.read_csv(pred_fp, sep="\t")
            keep_k = int(max(100, round(string_edge_n * TOP_EDGES_FACTOR)))
            filt = filter_pred_by_string(pred_df, string_gene1, string_gene_union, keep_top_k=keep_k)
            g = build_plot_graph(filt, max_nodes=max_plot_nodes)
            comms, mod = detect_communities(g)

            if g.number_of_nodes() > 0:
                pos = nx.spring_layout(
                    g, seed=42, k=3.2 / np.sqrt(max(2, g.number_of_nodes())), iterations=150
                )
                if g.number_of_edges() > 0:
                    nx.draw_networkx_edges(g, pos, edge_color="#BFD9F2", width=0.8, alpha=1.0, ax=ax)

                # Community-wise high-contrast scatter colors.
                for j, c in enumerate(comms):
                    if not c:
                        continue
                    xy = np.array([pos[n] for n in c])
                    ax.scatter(
                        xy[:, 0],
                        xy[:, 1],
                        s=SCATTER_NODE_SIZE,
                        c=[fixed_comm_colors[j % len(fixed_comm_colors)]],
                        alpha=0.72,
                        edgecolors="none",
                        linewidths=0.0,
                    )

            ax.set_title(
                f"scGPT | {EXTRACTION_TITLE.get(extraction, extraction)}\nModularity={mod:.3f}",
                fontsize=TITLE_SIZE,
                color=TITLE_COLOR,
            )
            ax.set_axis_off()
            rows.append(
                {
                    "Dataset": dataset,
                    "Model": model,
                    "Extraction": extraction,
                    "PredPath": str(pred_fp),
                    "STRING_Edges": string_edge_n,
                    "FilteredEdges": int(len(filt)),
                    "PlotNodes": int(g.number_of_nodes()),
                    "PlotEdges": int(g.number_of_edges()),
                    "Communities": int(len(comms)),
                    "Modularity": float(mod),
                }
            )
        except Exception as e:
            ax.set_axis_off()
            ax.set_title(
                f"scGPT | {EXTRACTION_TITLE.get(extraction, extraction)}\nERROR",
                fontsize=TITLE_SIZE,
                color=TITLE_COLOR,
            )
            ax.text(0.5, 0.5, str(e), ha="center", va="center", fontsize=12, wrap=True, transform=ax.transAxes)
            rows.append(
                {
                    "Dataset": dataset,
                    "Model": model,
                    "Extraction": extraction,
                    "PredPath": "",
                    "STRING_Edges": string_edge_n,
                    "FilteredEdges": np.nan,
                    "PlotNodes": np.nan,
                    "PlotEdges": np.nan,
                    "Communities": np.nan,
                    "Modularity": np.nan,
                    "Error": str(e),
                }
            )
        plt.tight_layout()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_pdf = OUT_DIR / f"{dataset}_scgpt_{extraction}_community_scatter.pdf"
        fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure: {out_pdf}")
    return pd.DataFrame(rows)


def plot_one_dataset(dataset: str, models: Sequence[str], extraction: str, max_plot_nodes: int) -> pd.DataFrame:
    string_gene1, string_gene_union, string_edge_n = load_string_filter(dataset)
    rows = []
    ncols = len(models)
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 5.2), squeeze=False)

    # Fixed community colors for consistent appearance across panels
    fixed_comm_colors = [
        PALETTE["scGPT"],
        PALETTE["Geneformer"],
        PALETTE["GENIE3"],
        PALETTE["LangCell"],
        PALETTE["scCello"],
        PALETTE["scFoundation"],
        PALETTE["scPrint"],
    ]

    for i, model in enumerate(models):
        ax = axes[0, i]
        try:
            pred_fp = resolve_pred_path(model, dataset, extraction)
            pred_df = pd.read_csv(pred_fp, sep="\t")
            keep_k = int(max(100, round(string_edge_n * TOP_EDGES_FACTOR)))
            filt = filter_pred_by_string(pred_df, string_gene1, string_gene_union, keep_top_k=keep_k)
            g = build_plot_graph(filt, max_nodes=max_plot_nodes)
            comms, mod = detect_communities(g)

            pos = nx.spring_layout(g, seed=42, k=2 / np.sqrt(max(2, g.number_of_nodes())), iterations=120)

            for j, c in enumerate(comms):
                if not c:
                    continue
                nx.draw_networkx_nodes(
                    g,
                    pos,
                    nodelist=list(c),
                    node_size=GRID_NODE_SIZE,
                    node_color=[fixed_comm_colors[j % len(fixed_comm_colors)]],
                    alpha=0.88,
                    ax=ax,
                )
            nx.draw_networkx_edges(g, pos, edge_color="#9A9A9A", width=0.45, alpha=0.28, ax=ax)

            ax.set_title(
                f"{model} | {EXTRACTION_TITLE.get(extraction, extraction)}\nModularity={mod:.3f}",
                fontsize=TITLE_SIZE,
                color=TITLE_COLOR,
            )
            ax.set_axis_off()
            rows.append(
                {
                    "Dataset": dataset,
                    "Model": model,
                    "Extraction": extraction,
                    "PredPath": str(pred_fp),
                    "STRING_Edges": string_edge_n,
                    "FilteredEdges": int(len(filt)),
                    "PlotNodes": int(g.number_of_nodes()),
                    "PlotEdges": int(g.number_of_edges()),
                    "Communities": int(len(comms)),
                    "Modularity": float(mod),
                }
            )
        except Exception as e:
            ax.set_axis_off()
            ax.set_title(f"{model} | {EXTRACTION_TITLE.get(extraction, extraction)}\nERROR", fontsize=TITLE_SIZE, color=TITLE_COLOR)
            ax.text(0.5, 0.5, str(e), ha="center", va="center", fontsize=12, wrap=True, transform=ax.transAxes)
            rows.append(
                {
                    "Dataset": dataset,
                    "Model": model,
                    "Extraction": extraction,
                    "PredPath": "",
                    "STRING_Edges": string_edge_n,
                    "FilteredEdges": np.nan,
                    "PlotNodes": np.nan,
                    "PlotEdges": np.nan,
                    "Communities": np.nan,
                    "Modularity": np.nan,
                    "Error": str(e),
                }
            )

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / f"{dataset}_community_real_edges_{extraction}.pdf"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"Saved figure: {out_pdf}")
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.list_datasets:
        print("Datasets:", ", ".join(ALL_DATASETS))
        return
    if args.list_models:
        print("Models:", ", ".join(ALL_MODELS))
        return

    datasets = _parse_list_arg(args.dataset, ALL_DATASETS)
    models = _parse_list_arg(args.models, ALL_MODELS)

    bad_ds = [d for d in datasets if d not in ALL_DATASETS]
    bad_models = [m for m in models if m not in ALL_MODELS]
    if bad_ds:
        raise ValueError(f"Unsupported dataset(s): {bad_ds}. Choose from: {ALL_DATASETS}")
    if bad_models:
        raise ValueError(f"Unsupported model(s): {bad_models}. Choose from: {ALL_MODELS}")
    if not models:
        raise ValueError("No valid models provided.")

    all_stats = []
    if args.scgpt_three_scatter:
        for ds in datasets:
            print(f"\n=== Dataset: {ds} | scGPT three-extraction scatter ===")
            one = plot_scgpt_three_scatter(ds, max_plot_nodes=int(args.max_plot_nodes))
            all_stats.append(one)
        stat_name = "community_real_edges_stats_scgpt_three_scatter.csv"
    else:
        for ds in datasets:
            print(f"\n=== Dataset: {ds} | extraction={args.extraction} ===")
            one = plot_one_dataset(ds, models=models, extraction=args.extraction, max_plot_nodes=int(args.max_plot_nodes))
            all_stats.append(one)
        stat_name = f"community_real_edges_stats_{args.extraction}.csv"

    stat_df = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stat_csv = OUT_DIR / stat_name
    stat_df.to_csv(stat_csv, index=False)
    print(f"Saved stats: {stat_csv}")


if __name__ == "__main__":
    main()