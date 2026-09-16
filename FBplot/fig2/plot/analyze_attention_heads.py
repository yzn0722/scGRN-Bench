#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze & plot topology metrics for per-attention-head GRN exports.

Input TSV: Gene1, Gene2, EdgeWeight  (from export_heads/)
Reference: STRING or CHIP network (gene universe + top-k edge count)

Outputs per (model, dataset):
  - {model}_{dataset}_head_metrics.csv
  - {model}_{dataset}_all_metrics_combined.pdf/png
Optional:
  - cross_model_{dataset}_{metric}_heatmap.pdf
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import greedy_modularity_communities, modularity

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from fig2_palette import model_color

METRICS = ["AvgPathLength", "Clustering", "Modularity", "AvgDegree", "Assortativity"]
METRICS_DISPLAY = [
    "Avg Path Length",
    "Clustering",
    "Modularity",
    "Avg Degree",
    "Assortativity",
]
WEIGHT_COLS = (
    "EdgeWeight",
    "edgeweight",
    "edge_weight",
    "Attention score",
    "Weight",
    "Score",
    "Importance",
)

# filename prefix aliases -> display name for colors
MODEL_ALIASES: Dict[str, str] = {
    "scgpt": "scGPT",
    "geneformer": "Geneformer",
    "langcell": "LangCell",
    "sccello": "scCello",
    "scprint": "scPrint",
}

plt.rcParams.update(
    {
        "font.sans-serif": ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)


def normalize_edges(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "Gene1" not in d.columns or "Gene2" not in d.columns:
        if d.shape[1] < 2:
            raise ValueError("No usable gene columns")
        d = d.iloc[:, :2].copy()
        d.columns = ["Gene1", "Gene2"]
    d["Gene1"] = d["Gene1"].astype(str).str.strip()
    d["Gene2"] = d["Gene2"].astype(str).str.strip()
    d = d[(d["Gene1"] != "") & (d["Gene2"] != "") & (d["Gene1"] != d["Gene2"])]
    return d.drop_duplicates(subset=["Gene1", "Gene2"])


def detect_weight_col(df: pd.DataFrame) -> Optional[str]:
    for c in WEIGHT_COLS:
        if c in df.columns:
            return c
    return None


def load_reference(
    ref_root: Path, ref_type: str, dataset: str
) -> Tuple[Set[str], Set[str], int]:
    ref_type = ref_type.upper()
    if ref_type == "STRING":
        fp = ref_root / "STRING" / f"{dataset}_processed-network.csv"
    elif ref_type == "CHIP":
        fp = ref_root / "CHIP" / f"{dataset}_chip_matched-network.csv"
    else:
        raise ValueError(f"Unknown ref_type: {ref_type}")
    if not fp.exists():
        raise FileNotFoundError(fp)
    df = normalize_edges(pd.read_csv(fp))
    g1 = set(df["Gene1"].astype(str))
    gu = set(pd.concat([df["Gene1"], df["Gene2"]], axis=0).astype(str))
    return g1, gu, int(len(df))


def parse_head_number(path: Path) -> int:
    m = re.search(r"head(\d+)", path.stem, re.I)
    return int(m.group(1)) if m else 0


def model_file_prefix(model: str) -> str:
    return model.strip().lower()


def find_head_files(
    head_roots: List[Path], model: str, dataset: str
) -> List[Path]:
    """Find {model}_{dataset}_head*.tsv under head_roots (recursive)."""
    prefix = model_file_prefix(model)
    patterns = [
        f"{prefix}_{dataset}_head*.tsv",
        f"{prefix}*{dataset}*head*.tsv",
    ]
    found: List[Path] = []
    for root in head_roots:
        if not root.exists():
            continue
        for pat in patterns:
            found.extend(root.rglob(pat))
    # de-dup + sort by head index
    uniq = sorted({p.resolve() for p in found}, key=parse_head_number)
    return uniq


def load_filtered_pred(
    fp: Path, g1: Set[str], gu: Set[str], top_k: int
) -> pd.DataFrame:
    usecols = ["Gene1", "Gene2", "EdgeWeight"]
    try:
        df = pd.read_csv(fp, sep="\t", usecols=lambda c: c in usecols or c in WEIGHT_COLS)
    except (ValueError, KeyError):
        df = pd.read_csv(fp, sep="\t")
    df = normalize_edges(df)
    df = df[df["Gene1"].isin(g1) & df["Gene2"].isin(gu)].copy()
    if df.empty:
        return df
    w = detect_weight_col(df)
    if w is not None:
        df[w] = pd.to_numeric(df[w], errors="coerce").fillna(0.0)
        df = df.nlargest(top_k, w) if top_k > 0 else df.sort_values(w, ascending=False)
    elif top_k > 0:
        df = df.head(top_k)
    return df[["Gene1", "Gene2"]].copy()


def calc_avg_path_length(g: nx.Graph) -> float:
    if g.number_of_nodes() < 2:
        return np.nan
    if nx.is_connected(g):
        g_path = g
    else:
        largest = max(nx.connected_components(g), key=len)
        g_path = g.subgraph(largest).copy()
    if g_path.number_of_nodes() < 2:
        return np.nan
    try:
        return float(nx.average_shortest_path_length(g_path))
    except Exception:
        return np.nan


def calc_clustering(g: nx.Graph) -> float:
    if g.number_of_nodes() < 3:
        return np.nan
    try:
        return float(nx.average_clustering(g))
    except Exception:
        return np.nan


def calc_modularity(g: nx.Graph) -> float:
    if g.number_of_nodes() < 3 or g.number_of_edges() < 2:
        return np.nan
    try:
        comms = list(greedy_modularity_communities(g))
        if not comms:
            return np.nan
        return float(modularity(g, comms))
    except Exception:
        return np.nan


def calc_avg_degree(g: nx.Graph) -> float:
    if g.number_of_nodes() == 0:
        return np.nan
    return float(np.mean([d for _, d in g.degree()]))


def calc_assortativity(g: nx.Graph) -> float:
    if g.number_of_nodes() < 2:
        return np.nan
    try:
        r = nx.degree_assortativity_coefficient(g)
        if r is None or not np.isfinite(r):
            return np.nan
        return float(r)
    except Exception:
        return np.nan


def calc_all_metrics(edges: pd.DataFrame) -> Dict[str, float]:
    if edges.empty:
        return {m: np.nan for m in METRICS}
    g = nx.Graph()
    g.add_edges_from(edges.itertuples(index=False, name=None))
    if g.number_of_nodes() < 3:
        return {m: np.nan for m in METRICS}
    return {
        "AvgPathLength": calc_avg_path_length(g),
        "Clustering": calc_clustering(g),
        "Modularity": calc_modularity(g),
        "AvgDegree": calc_avg_degree(g),
        "Assortativity": calc_assortativity(g),
    }


def compute_metrics_for_model(
    model: str,
    dataset: str,
    head_roots: List[Path],
    ref_root: Path,
    ref_type: str,
    top_k: int,
) -> pd.DataFrame:
    rows = []
    head_files = find_head_files(head_roots, model, dataset)
    if not head_files:
        print(f"[WARN] No head TSV for model={model} dataset={dataset}")
        return pd.DataFrame(rows)

    g1, gu, n_ref = load_reference(ref_root, ref_type, dataset)
    keep_k = n_ref if top_k <= 0 else min(top_k, n_ref)
    print(f"[INFO] {model}/{dataset}: {len(head_files)} heads, ref_edges={n_ref}, top_k={keep_k}")

    for fp in head_files:
        h = parse_head_number(fp)
        try:
            edges = load_filtered_pred(fp, g1, gu, keep_k)
            metrics = calc_all_metrics(edges)
            row = {
                "Model": MODEL_ALIASES.get(model_file_prefix(model), model),
                "Dataset": dataset,
                "Head": h,
                "File": fp.name,
                "n_edges": len(edges),
            }
            row.update(metrics)
            rows.append(row)
            print(
                f"  head{h}: n_edges={len(edges)} "
                f"Mod={metrics['Modularity']:.4f} Clust={metrics['Clustering']:.4f}"
            )
        except Exception as e:
            print(f"  head{h} ERROR: {e}")
            row = {
                "Model": MODEL_ALIASES.get(model_file_prefix(model), model),
                "Dataset": dataset,
                "Head": h,
                "File": fp.name,
                "n_edges": 0,
            }
            row.update({m: np.nan for m in METRICS})
            rows.append(row)
    return pd.DataFrame(rows)


def plot_combined_bars(df: pd.DataFrame, out_dir: Path, model: str, dataset: str) -> None:
    if df.empty:
        return
    df = df.sort_values("Head").reset_index(drop=True)
    display = MODEL_ALIASES.get(model_file_prefix(model), model)
    color = model_color(display)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=300)
    axes = axes.flatten()
    x = np.arange(len(df))

    for idx, (metric, ylab) in enumerate(zip(METRICS, METRICS_DISPLAY)):
        ax = axes[idx]
        vals = df[metric].astype(float)
        valid = vals.dropna()
        if valid.empty:
            ax.text(0.5, 0.5, f"No data\n{ylab}", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("Attention Head")
            ax.set_ylabel(ylab)
            continue
        ax.bar(x, vals, color=color, edgecolor="none", width=0.7)
        ax.set_xlabel("Attention Head", fontsize=14)
        ax.set_ylabel(ylab, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([str(h) for h in df["Head"]], fontsize=12)
        y_max, y_min = valid.max(), valid.min()
        if metric == "Assortativity":
            ax.set_ylim(-0.5, 1.0)
            ax.set_yticks(np.arange(-0.5, 1.01, 0.5))
            ax.axhline(0, color="black", linewidth=0.6, alpha=0.3)
        elif not np.isnan(y_max):
            if y_min >= 0:
                ax.set_ylim(0, y_max * 1.1)
            else:
                ax.set_ylim(y_min * 1.1, y_max * 1.1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    if len(METRICS) < 6:
        axes[5].set_visible(False)
    fig.suptitle(f"{display} — {dataset} (per attention head)", fontsize=16, y=1.02)
    plt.tight_layout()
    stem = f"{model_file_prefix(model)}_{dataset}_all_metrics_combined"
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {out_dir / stem}.pdf")


def plot_metric_heatmap(
    all_df: pd.DataFrame, metric: str, dataset: str, out_dir: Path
) -> None:
    """Rows=models, cols=head index; one heatmap for cross-model comparison."""
    sub = all_df[(all_df["Dataset"] == dataset) & all_df[metric].notna()].copy()
    if sub.empty:
        return
    pivot = sub.pivot_table(index="Model", columns="Head", values=metric, aggfunc="first")
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 0.8), max(3, pivot.shape[0] * 0.6)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Model")
    ylab = dict(zip(METRICS, METRICS_DISPLAY)).get(metric, metric)
    ax.set_title(f"{dataset} — {ylab} across models & heads")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    path = out_dir / f"cross_model_{dataset}_{metric}_heatmap.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze & plot per-head attention GRN topology.")
    p.add_argument("--dataset", default="hESC", help="Dataset name, e.g. hESC")
    p.add_argument(
        "--datasets",
        default="",
        help="Comma-separated datasets (overrides --dataset if set)",
    )
    p.add_argument(
        "--models",
        default="scgpt,geneformer,langcell,sccello,scprint",
        help="Comma-separated model prefixes matching TSV filenames",
    )
    p.add_argument(
        "--head-root",
        action="append",
        default=[],
        help="Root dir(s) with *_head*.tsv (repeatable). Default: outputs + fig2/att_head",
    )
    p.add_argument(
        "--ref-root",
        default="/mnt/10T/yzn/benchmark_GRN/input_process",
        help="Root containing STRING/ and CHIP/ reference networks",
    )
    p.add_argument(
        "--ref-type",
        default="STRING",
        choices=["STRING", "CHIP"],
        help="Reference network for gene filter & top-k",
    )
    p.add_argument(
        "--top-edges",
        type=int,
        default=10000,
        help="Top edges per head by weight (after ref gene filter); 0 = match ref network size",
    )
    p.add_argument(
        "--output-dir",
        default=str(_SCRIPT_DIR / "output" / "head_topology"),
        help="Output directory for CSV and figures",
    )
    p.add_argument(
        "--heatmap-metric",
        default="Modularity",
        choices=METRICS,
        help="Metric for cross-model heatmap",
    )
    p.add_argument("--no-plot", action="store_true", help="Only write CSV, skip figures")
    p.add_argument(
        "--no-cross-heatmap",
        action="store_true",
        help="Skip cross-model heatmap",
    )
    return p.parse_args()


def default_head_roots() -> List[Path]:
    project = _SCRIPT_DIR.parents[3]  # .../scGRN-Bench
    return [
        project / "outputs" / "attention_heads",
        _SCRIPT_DIR.parent / "att_head",
    ]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_root = Path(args.ref_root)

    head_roots = [Path(r) for r in args.head_root] if args.head_root else default_head_roots()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.datasets.strip():
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        datasets = [args.dataset.strip()]

    all_parts: List[pd.DataFrame] = []
    for ds in datasets:
        ds_parts: List[pd.DataFrame] = []
        for model in models:
            df = compute_metrics_for_model(
                model, ds, head_roots, ref_root, args.ref_type, args.top_edges
            )
            if df.empty:
                continue
            csv_path = out_dir / f"{model_file_prefix(model)}_{ds}_head_metrics.csv"
            df.to_csv(csv_path, index=False)
            print(f"[INFO] CSV: {csv_path}")
            ds_parts.append(df)
            all_parts.append(df)
            if not args.no_plot:
                plot_combined_bars(df, out_dir, model, ds)

        if not args.no_cross_heatmap and ds_parts:
            ds_df = pd.concat(ds_parts, ignore_index=True)
            plot_metric_heatmap(ds_df, args.heatmap_metric, ds, out_dir)

    if all_parts:
        long_df = pd.concat(all_parts, ignore_index=True)
        long_path = out_dir / "all_models_head_metrics_long.csv"
        long_df.to_csv(long_path, index=False)
        print(f"[INFO] Combined: {long_path}")
    else:
        print("[WARN] No head files processed. Check --head-root and --models.")


if __name__ == "__main__":
    main()
