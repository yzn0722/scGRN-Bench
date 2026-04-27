#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scGPT different attention heads modularity analysis - simple bar chart only
"""

import argparse
from pathlib import Path
from typing import List, Optional, Set, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import greedy_modularity_communities, modularity
from fig2_palette import model_color

# Paths
STRING_ROOT = Path("/mnt/10T/yzn/benchmark_GRN/input_process/STRING")
HEAD_ROOT = Path("/mnt/10T/yzn/FoundBench/FBplot/fig2/att_head")
WEIGHT_COLS = ("EdgeWeight", "edgeweight", "edge_weight", "Attention score", "Weight", "Score", "Importance")

# Unified plotting style (match fig2/0415)
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.size"] = 16
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16


def normalize_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize edge data"""
    d = df.copy()
    if "Gene1" not in d.columns or "Gene2" not in d.columns:
        if d.shape[1] < 2:
            raise ValueError("No usable gene columns")
        d = d.iloc[:, :2].copy()
        d.columns = ["Gene1", "Gene2"]
    
    d["Gene1"] = d["Gene1"].astype(str).str.strip()
    d["Gene2"] = d["Gene2"].astype(str).str.strip()
    d = d[(d["Gene1"] != "") & (d["Gene2"] != "") & (d["Gene1"] != d["Gene2"])]
    d = d.drop_duplicates(subset=["Gene1", "Gene2"])
    return d


def detect_weight_col(df: pd.DataFrame) -> Optional[str]:
    """Detect weight column"""
    for c in WEIGHT_COLS:
        if c in df.columns:
            return c
    return None


def load_string(dataset: str) -> Tuple[Set[str], Set[str], int]:
    """Load STRING network data"""
    fp = STRING_ROOT / f"{dataset}_processed-network.csv"
    if not fp.exists():
        raise FileNotFoundError(f"STRING file not found: {fp}")
    
    df = pd.read_csv(fp)
    df = normalize_edges(df)
    
    g1 = set(df["Gene1"].astype(str))
    gu = set(pd.concat([df["Gene1"], df["Gene2"]], axis=0).astype(str))
    return g1, gu, int(len(df))


def find_all_heads(dataset: str) -> List[Path]:
    """Find all head files for the dataset"""
    pattern = f"scgpt_{dataset}_head*.tsv"
    files = list(HEAD_ROOT.glob(pattern))
    
    # Sort by head number
    def get_head_number(path: Path) -> int:
        stem = path.stem
        if "head" in stem:
            head_part = stem.split("head")[-1]
            try:
                return int(head_part)
            except ValueError:
                return 0
        return 0
    
    return sorted(files, key=get_head_number)


def parse_head_number(file_path: Path) -> int:
    """Parse head number from filename"""
    filename = file_path.stem
    if "head" in filename:
        head_part = filename.split("head")[-1]
        try:
            return int(head_part)
        except ValueError:
            return 0
    return 0


def load_filtered_pred(fp: Path, g1: Set[str], gu: Set[str], top_k: int) -> pd.DataFrame:
    """Load and filter predicted edges"""
    df = pd.read_csv(fp, sep="\t")
    df = normalize_edges(df)
    
    # Filter genes
    df = df[df["Gene1"].isin(g1) & df["Gene2"].isin(gu)].copy()
    
    if df.empty:
        return df
    
    # Detect weight column and sort
    w = detect_weight_col(df)
    if w is not None:
        df[w] = pd.to_numeric(df[w], errors="coerce").fillna(0.0)
        df = df.sort_values(w, ascending=False).head(top_k)
    else:
        df = df.head(top_k)
    
    return df[["Gene1", "Gene2"]].copy()


def calc_modularity(edges: pd.DataFrame) -> float:
    """Calculate network modularity"""
    if edges.empty:
        return np.nan
    
    # Build network graph
    g = nx.Graph()
    g.add_edges_from(edges.itertuples(index=False, name=None))
    
    # Check if network is large enough
    if g.number_of_nodes() < 3 or g.number_of_edges() < 2:
        return np.nan
    
    # Detect communities using greedy algorithm
    comms = list(greedy_modularity_communities(g))
    
    if len(comms) == 0:
        return np.nan
    
    # Calculate modularity
    return float(modularity(g, comms))


def compute_modularity_for_heads(dataset: str, top_k: int) -> pd.DataFrame:
    """Calculate modularity for all attention heads"""
    
    rows = []
    
    # Find all head files
    head_files = find_all_heads(dataset)
    if not head_files:
        print(f"No head files found for {dataset}")
        return pd.DataFrame(rows)
    
    print(f"Found {len(head_files)} head files")
    
    # Load STRING network data
    try:
        g1, gu, n_string_edges = load_string(dataset)
        keep_k = n_string_edges if top_k <= 0 else min(top_k, n_string_edges)
        print(f"STRING edges: {n_string_edges}, keep edges: {keep_k}")
    except Exception as e:
        print(f"Failed to load STRING data: {e}")
        return pd.DataFrame(rows)
    
    for head_file in head_files:
        head_num = parse_head_number(head_file)
        print(f"Processing Head {head_num}...")
        
        try:
            # Load and filter predicted edges
            edges = load_filtered_pred(head_file, g1, gu, keep_k)
            
            if edges.empty:
                mod_val = np.nan
                n_edges = 0
                print(f"  No valid edges")
            else:
                # Calculate modularity
                mod_val = calc_modularity(edges)
                n_edges = len(edges)
                print(f"  Modularity: {mod_val:.4f}, Edges: {n_edges}")
            
            # Record result
            rows.append({
                "Head": head_num,
                "Modularity": mod_val,
                "n_edges": n_edges,
                "File": head_file.name
            })
            
        except Exception as e:
            print(f"  Error: {e}")
            rows.append({
                "Head": head_num,
                "Modularity": np.nan,
                "n_edges": 0,
                "File": head_file.name
            })
    
    print(f"Computed modularity for {len(rows)} heads")
    return pd.DataFrame(rows)


def plot_simple_bar_chart(df: pd.DataFrame, output_path: Path, dataset: str):
    """Plot simple bar chart in unified fig2 style."""
    
    if df.empty:
        print("No data to plot")
        return
    
    # Sort by head number
    df_sorted = df.sort_values("Head").reset_index(drop=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    
    bar_color = model_color("scGPT")
    # Plot bars
    x = np.arange(len(df_sorted))
    bars = ax.bar(x, df_sorted["Modularity"], color=bar_color, edgecolor="none", linewidth=0.0, width=0.7)
    
    # Set axis labels
    ax.set_xlabel("Attention Head", fontsize=16)
    ax.set_ylabel("Modularity", fontsize=16)
    
    # Set x-ticks
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}" for h in df_sorted["Head"]], fontsize=16)
    
    # Add grid
    ax.grid(False)
    
    # Adjust y-axis
    y_max = df_sorted["Modularity"].max()
    if not np.isnan(y_max):
        ax.set_ylim(0, max(0.1, y_max * 1.1))
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Bar chart saved: {output_path.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser(description="Plot modularity bar chart for scGPT attention heads")
    parser.add_argument("--dataset", type=str, default="hESC", 
                       help="Dataset name")
    parser.add_argument("--top-edges", type=int, default=10000,
                       help="Number of top edges to keep")
    parser.add_argument("--output", type=str, default="",
                       help="Output file path (default: ./output/scgpt_modularity_<dataset>.pdf)")
    
    args = parser.parse_args()
    
    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path("./output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"scgpt_modularity_{args.dataset}.pdf"
    
    # Compute modularity
    print(f"Processing dataset: {args.dataset}")
    results_df = compute_modularity_for_heads(args.dataset, args.top_edges)
    
    if not results_df.empty:
        # Plot simple bar chart
        plot_simple_bar_chart(results_df, output_path, args.dataset)
        
        # Print summary
        valid_mod = results_df["Modularity"].dropna()
        if not valid_mod.empty:
            print(f"\nModularity summary:")
            print(f"  Mean: {valid_mod.mean():.4f}")
            print(f"  Max:  {valid_mod.max():.4f}")
            print(f"  Min:  {valid_mod.min():.4f}")
            print(f"  Std:  {valid_mod.std():.4f}")
            
            # Also save data to CSV
            csv_path = output_path.with_suffix('.csv')
            results_df.to_csv(csv_path, index=False)
            print(f"Data saved: {csv_path}")
    else:
        print("No results to plot")


if __name__ == "__main__":
    main()