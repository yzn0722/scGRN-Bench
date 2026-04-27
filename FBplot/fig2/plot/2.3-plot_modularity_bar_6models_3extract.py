#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot grouped bar chart of modularity:
  6 models x 3 extraction types (emb500 / att500 / embhidden500)

Computation uses REAL predicted edges with STRING-based filtering:
  Gene1 in STRING(Gene1), Gene2 in STRING(Gene1 ∪ Gene2)
and keeps top-K edges (K = #STRING edges for that dataset by default).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import greedy_modularity_communities, modularity

from fig2_palette import model_color

plt.rcParams["font.size"] = 16
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 16


ROOT = Path("/mnt/10T/yzn/benchmark_GRN")
EVL_ROOT = ROOT / "evl_omipath"
STRING_ROOT = ROOT / "input_process/STRING"

EXTRACTIONS = ("emb500", "att500", "embhidden500")
EXTRACTION_DISPLAY = {
    "emb500": r"cos$_{tok}$",
    "att500": "attn",
    "embhidden500": r"cos$_{hid}$",
}
MODELS = ("Geneformer", "LangCell", "scGPT", "scCello", "scFoundation", "scPrint")

EXTRACT_COLORS = {
    # Match fig2 degree-plot palette system (fig2_palette.py)
    "emb500": model_color("scGPT"),
    "att500": model_color("LangCell"),
    "embhidden500": model_color("scFoundation"),
}

MODEL_DIRS = {
    "Geneformer": {"emb500": "geneformer", "att500": "geneformer", "embhidden500": "geneformer"},
    "LangCell": {"emb500": "langcell", "att500": "langcell", "embhidden500": "Langcell"},
    "scGPT": {"emb500": "scgpt", "att500": "scgpt", "embhidden500": "scgpt"},
    "scCello": {"emb500": "sccello", "att500": "sccello", "embhidden500": "sccello"},
    "scFoundation": {"emb500": "scFoundation", "att500": "scFoundation", "embhidden500": "scFoundation"},
    "scPrint": {"emb500": "scprint", "att500": "scprint", "embhidden500": "scprint"},
}

MODEL_STEMS = {
    "Geneformer": ("geneformer", "Geneformer"),
    "LangCell": ("LangCell", "langcell", "Langcell"),
    "scGPT": ("scgpt", "scGPT", "scGPT2"),
    "scCello": ("scCello", "sccello"),
    "scFoundation": ("scFoundation", "scfoundation"),
    "scPrint": ("scprint", "scPrint", "scPRINT"),
}

WEIGHT_COLS = ("EdgeWeight", "edgeweight", "edge_weight", "Attention score", "Weight", "Score", "Importance")


def normalize_edges(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "Gene1" not in d.columns or "Gene2" not in d.columns:
        if d.shape[1] < 2:
            raise ValueError("Prediction file has no usable gene columns")
        d = d.iloc[:, :2].copy()
        d.columns = ["Gene1", "Gene2"]
    d["Gene1"] = d["Gene1"].astype(str).str.strip()
    d["Gene2"] = d["Gene2"].astype(str).str.strip()
    d = d[(d["Gene1"] != "") & (d["Gene2"] != "") & (d["Gene1"] != d["Gene2"])]
    d = d.drop_duplicates(subset=["Gene1", "Gene2"])
    return d


def detect_weight_col(df: pd.DataFrame) -> Optional[str]:
    for c in WEIGHT_COLS:
        if c in df.columns:
            return c
    return None


def load_string(dataset: str) -> Tuple[Set[str], Set[str], int]:
    fp = STRING_ROOT / f"{dataset}_processed-network.csv"
    if not fp.exists():
        raise FileNotFoundError(f"STRING file not found: {fp}")
    df = pd.read_csv(fp)
    df = normalize_edges(df)
    g1 = set(df["Gene1"].astype(str))
    gu = set(pd.concat([df["Gene1"], df["Gene2"]], axis=0).astype(str))
    return g1, gu, int(len(df))


def resolve_pred_path(model: str, extraction: str, dataset: str) -> Path:
    base = EVL_ROOT / f"output_{extraction}" / MODEL_DIRS[model][extraction]
    if not base.exists():
        raise FileNotFoundError(f"Base dir not found: {base}")
    for stem in MODEL_STEMS[model]:
        cand = base / f"{stem}_{dataset}.tsv"
        if cand.exists():
            return cand
    # fallback: any tsv containing dataset name
    for p in base.glob("*.tsv"):
        if dataset.lower() in p.name.lower():
            return p
    raise FileNotFoundError(f"No prediction tsv for {model}/{extraction}/{dataset}")


def load_filtered_pred(fp: Path, g1: Set[str], gu: Set[str], top_k: int) -> pd.DataFrame:
    df = pd.read_csv(fp, sep="\t")
    df = normalize_edges(df)
    df = df[df["Gene1"].isin(g1) & df["Gene2"].isin(gu)].copy()
    if df.empty:
        return df
    w = detect_weight_col(df)
    if w is not None:
        df[w] = pd.to_numeric(df[w], errors="coerce").fillna(0.0)
        df = df.sort_values(w, ascending=False).head(top_k)
    else:
        df = df.head(top_k)
    return df[["Gene1", "Gene2"]].copy()


def calc_modularity(edges: pd.DataFrame) -> float:
    if edges.empty:
        return np.nan
    g = nx.Graph()
    g.add_edges_from(edges.itertuples(index=False, name=None))
    if g.number_of_nodes() < 3 or g.number_of_edges() < 2:
        return np.nan
    comms = list(greedy_modularity_communities(g))
    if len(comms) == 0:
        return np.nan
    return float(modularity(g, comms))


def compute_table(dataset: str, top_k: int) -> pd.DataFrame:
    g1, gu, n_string_edges = load_string(dataset)
    keep_k = n_string_edges if top_k <= 0 else min(top_k, n_string_edges)
    rows: List[dict] = []
    for model in MODELS:
        for ext in EXTRACTIONS:
            try:
                fp = resolve_pred_path(model, ext, dataset)
                ed = load_filtered_pred(fp, g1, gu, keep_k)
                mod = calc_modularity(ed)
                rows.append(
                    {
                        "Dataset": dataset,
                        "Model": model,
                        "Extraction": ext,
                        "Modularity": mod,
                        "EdgesUsed": int(len(ed)),
                        "Path": str(fp),
                    }
                )
            except Exception:
                rows.append(
                    {
                        "Dataset": dataset,
                        "Model": model,
                        "Extraction": ext,
                        "Modularity": np.nan,
                        "EdgesUsed": 0,
                        "Path": "",
                    }
                )
    return pd.DataFrame(rows)


def plot_bar(df: pd.DataFrame, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    pivot = df.pivot(index="Model", columns="Extraction", values="Modularity").reindex(index=list(MODELS), columns=list(EXTRACTIONS))

    x = np.arange(len(MODELS))
    width = 0.24

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    for i, ext in enumerate(EXTRACTIONS):
        ys = pivot[ext].values.astype(float)
        bars = ax.bar(
            x + (i - 1) * width,
            ys,
            width=width,
            color=EXTRACT_COLORS.get(ext, "#999999"),
            edgecolor="none",
            linewidth=0.0,
            label=EXTRACTION_DISPLAY.get(ext, ext),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=25, fontsize=16, ha="right")
    ax.set_ylabel("Modularity", fontsize=16)
    ax.tick_params(axis="both", colors="#000000")
    ax.yaxis.label.set_color("#000000")
    #ax.set_title("Modularity Comparison Across Models and Extractions", fontsize=12, pad=10)
    ax.set_ylim(0, max(0.6, np.nanmax(pivot.values) * 1.18 if np.isfinite(np.nanmax(pivot.values)) else 0.6))
    ax.grid(False)
    ax.spines["left"].set_color("#000000")
    ax.spines["bottom"].set_color("#000000")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=16)
    for t in legend.get_texts():
        t.set_color("#000000")
    fig.tight_layout()
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grouped bar chart: modularity of 6 models x 3 extractions")
    p.add_argument("--dataset", type=str, default="hESC", help="Dataset name")
    p.add_argument("--top-edges", type=int, default=0, help="Top-K edges after filtering (0 means #STRING edges)")
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Output figure path (default: fig2/output/modularity_bar/<dataset>_modularity_6models_3extract.pdf)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    out_png = (
        Path(args.out)
        if str(args.out).strip()
        else script_dir / "output" / "modularity_bar" / f"{args.dataset}_modularity_6models_3extract.pdf"
    )
    df = compute_table(dataset=args.dataset, top_k=int(args.top_edges))
    csv_out = out_png.with_suffix(".csv")
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)
    plot_bar(df, out_png)
    print(f"Saved figure: {out_png}")
    print(f"Saved table : {csv_out}")


if __name__ == "__main__":
    main()

