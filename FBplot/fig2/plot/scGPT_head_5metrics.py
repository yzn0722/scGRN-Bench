#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scGPT different attention heads topology metrics analysis - bar charts for all five metrics
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

# Paths - 修正为正确的路径
STRING_ROOT = Path("/mnt/10T/yzn/benchmark_GRN/input_process/STRING")
HEAD_ROOT = Path("/mnt/10T/yzn/scGRN-Bench/FBplot/fig2/att_head")  # 修正路径
WEIGHT_COLS = ("EdgeWeight", "edgeweight", "edge_weight", "Attention score", "Weight", "Score", "Importance")

# Metrics list
METRICS = ['AvgPathLength', 'Clustering', 'Modularity', 'AvgDegree', 'Assortativity']
METRICS_DISPLAY = ['Avg Path Length', 'Clustering', 'Modularity', 'Avg Degree', 'Assortativity']

# Unified plotting style
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12


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
    
    print(f"Looking for pattern: {pattern} in {HEAD_ROOT}")
    print(f"Found files: {files}")
    
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


def calc_avg_path_length(G: nx.Graph) -> float:
    """Calculate average shortest path length on largest connected component"""
    if G.number_of_nodes() < 2:
        return np.nan
    
    # Find largest connected component
    if nx.is_connected(G):
        G_path = G
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        G_path = G.subgraph(largest_cc).copy()
    
    if G_path.number_of_nodes() < 2:
        return np.nan
    
    try:
        return float(nx.average_shortest_path_length(G_path))
    except Exception:
        return np.nan


def calc_clustering(G: nx.Graph) -> float:
    """Calculate average clustering coefficient"""
    if G.number_of_nodes() < 3:
        return np.nan
    try:
        return float(nx.average_clustering(G))
    except Exception:
        return np.nan


def calc_modularity(G: nx.Graph) -> float:
    """Calculate network modularity using greedy communities"""
    if G.number_of_nodes() < 3 or G.number_of_edges() < 2:
        return np.nan
    
    try:
        comms = list(greedy_modularity_communities(G))
        if len(comms) == 0:
            return np.nan
        return float(modularity(G, comms))
    except Exception:
        return np.nan


def calc_avg_degree(G: nx.Graph) -> float:
    """Calculate average degree"""
    if G.number_of_nodes() == 0:
        return np.nan
    degrees = [d for _, d in G.degree()]
    return float(np.mean(degrees))


def calc_assortativity(G: nx.Graph) -> float:
    """Calculate degree assortativity coefficient"""
    if G.number_of_nodes() < 2:
        return np.nan
    try:
        r = nx.degree_assortativity_coefficient(G)
        if r is None or not np.isfinite(r):
            return np.nan
        return float(r)
    except Exception:
        return np.nan


def calc_all_metrics(edges: pd.DataFrame) -> dict:
    """Calculate all five topology metrics for a given edge set"""
    if edges.empty:
        return {m: np.nan for m in METRICS}
    
    # Build undirected graph
    G = nx.Graph()
    G.add_edges_from(edges.itertuples(index=False, name=None))
    
    if G.number_of_nodes() < 3:
        return {m: np.nan for m in METRICS}
    
    metrics = {
        'AvgPathLength': calc_avg_path_length(G),
        'Clustering': calc_clustering(G),
        'Modularity': calc_modularity(G),
        'AvgDegree': calc_avg_degree(G),
        'Assortativity': calc_assortativity(G),
    }
    return metrics


def compute_metrics_for_heads(dataset: str, top_k: int) -> pd.DataFrame:
    """Calculate all topology metrics for all attention heads"""
    
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
                print(f"  No valid edges")
                row = {"Head": head_num, "File": head_file.name}
                for m in METRICS:
                    row[m] = np.nan
                row["n_edges"] = 0
                rows.append(row)
            else:
                # Calculate all metrics
                metrics = calc_all_metrics(edges)
                print(f"  Nodes: {len(set(edges['Gene1']).union(set(edges['Gene2'])))}, Edges: {len(edges)}")
                print(f"  AvgPathLength: {metrics['AvgPathLength']:.4f}, "
                      f"Clustering: {metrics['Clustering']:.4f}, "
                      f"Modularity: {metrics['Modularity']:.4f}, "
                      f"AvgDegree: {metrics['AvgDegree']:.4f}, "
                      f"Assortativity: {metrics['Assortativity']:.4f}")
                
                row = {"Head": head_num, "File": head_file.name, "n_edges": len(edges)}
                row.update(metrics)
                rows.append(row)
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            row = {"Head": head_num, "File": head_file.name}
            for m in METRICS:
                row[m] = np.nan
            row["n_edges"] = 0
            rows.append(row)
    
    print(f"Computed metrics for {len(rows)} heads")
    return pd.DataFrame(rows)

def plot_combined_bar_charts(df: pd.DataFrame, output_dir: Path, dataset: str):
    """Plot all five metrics as subplots in a single figure (2x3 layout)"""
    
    if df.empty:
        print("No data to plot")
        return
    
    df_sorted = df.sort_values("Head").reset_index(drop=True)
    bar_color = model_color("scGPT")
    
    # Create figure with subplots (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=300)
    axes = axes.flatten()
    
    for idx, (metric, display_name) in enumerate(zip(METRICS, METRICS_DISPLAY)):
        ax = axes[idx]
        
        # Filter valid data
        valid_data = df_sorted[metric].dropna()
        if valid_data.empty:
            ax.text(0.5, 0.5, f"No data for\n{display_name}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_xlabel("Attention Head", fontsize=12)
            ax.set_ylabel(display_name, fontsize=12)
            continue
        
        # Plot bars
        x = np.arange(len(df_sorted))
        bars = ax.bar(x, df_sorted[metric], color=bar_color, edgecolor="none", linewidth=0.0, width=0.7)
        
        # Set labels
        ax.set_xlabel("Attention Head", fontsize=16)
        ax.set_ylabel(display_name, fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{h}" for h in df_sorted["Head"]], fontsize=14)
        
        # 特殊处理 Assortativity 坐标轴
        if not np.isnan(y_max):
            if metric == 'Assortativity':
                # 为 Assortativity 设置固定的 y 轴范围和刻度
                # 从 -0.5 到 1.0，间隔 0.5
                ax.set_ylim(-0.5, 1.0)
                ax.set_yticks(np.arange(-0.5, 1.1, 0.5))  # -0.5, 0.0, 0.5, 1.0
                
                # 格式化 y 轴刻度标签
                ax.set_yticklabels([f'{y:.1f}' for y in np.arange(-0.5, 1.1, 0.5)], fontsize=12)
                
                # 添加 0 参考线
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
            
        else:
            # 其他指标保持原来的逻辑
            y_max = valid_data.max()
            y_min = valid_data.min()
            if not np.isnan(y_max):
                if y_min >= 0:
                    ax.set_ylim(0, y_max * 1.1)
                else:
                    ax.set_ylim(y_min * 1.1, y_max * 1.1)
        
        # 移除顶部和右侧边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        # 添加 0 参考线（对于有正负的指标）
        if metric in ['Assortativity'] and (valid_data.min() < 0 < valid_data.max()):
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # 隐藏最后一个子图（如果有6个指标，这里不需要隐藏）
    if len(METRICS) < 6:
        axes[5].set_visible(False)
    
    plt.tight_layout()
    
    # 保存合并的图表
    combined_path = output_dir / f"scgpt_{dataset}_all_metrics_combined.pdf"
    fig.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Combined bar chart saved: {combined_path}")
# def plot_combined_bar_charts(df: pd.DataFrame, output_dir: Path, dataset: str):
#     """Plot all five metrics as subplots in a single figure (2x3 layout)"""
    
#     if df.empty:
#         print("No data to plot")
#         return
    
#     df_sorted = df.sort_values("Head").reset_index(drop=True)
#     bar_color = model_color("scGPT")
    
#     # Create figure with subplots (2 rows, 3 columns)
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=300)
#     axes = axes.flatten()
    
#     for idx, (metric, display_name) in enumerate(zip(METRICS, METRICS_DISPLAY)):
#         ax = axes[idx]
        
#         # Filter valid data
#         valid_data = df_sorted[metric].dropna()
#         if valid_data.empty:
#             ax.text(0.5, 0.5, f"No data for\n{display_name}", 
#                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
#             ax.set_xlabel("Attention Head", fontsize=12)
#             ax.set_ylabel(display_name, fontsize=12)
#             continue
        
#         # Plot bars
#         x = np.arange(len(df_sorted))
#         bars = ax.bar(x, df_sorted[metric], color=bar_color, edgecolor="none", linewidth=0.0, width=0.7)
        
#         # Set labels
#         ax.set_xlabel("Attention Head", fontsize=16)
#         ax.set_ylabel(display_name, fontsize=16)
#         ax.set_xticks(x)
#         ax.set_xticklabels([f"{h}" for h in df_sorted["Head"]], fontsize=14)
        
#         # Set y-axis limits
#         y_max = valid_data.max()
#         y_min = valid_data.min()
#         if not np.isnan(y_max):
#             if y_min >= 0:
#                 ax.set_ylim(0, y_max * 1.1)
#             else:
#                 ax.set_ylim(y_min * 1.1, y_max * 1.1)
        
#         # Remove top and right spines
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        
#         # Add value labels on top of bars (optional)
#         # for i, (bar, val) in enumerate(zip(bars, df_sorted[metric])):
#         #     if not np.isnan(val):
#         #         ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
#         #                f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
#     # Hide the last (6th) subplot if not needed
#     axes[5].set_visible(False)
    
#     # Add overall title
#     #fig.suptitle(f"scGPT - {dataset} Head-wise Topology Metrics", fontsize=16, fontweight='bold')
    
#     plt.tight_layout()
    
#     # Save combined figure
#     combined_path = output_dir / f"scgpt_{dataset}_all_metrics_combined.pdf"
#     fig.savefig(combined_path, dpi=300, bbox_inches='tight')
#     plt.close(fig)
    
#     print(f"Combined bar chart saved: {combined_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot topology metrics bar charts for scGPT attention heads")
    parser.add_argument("--dataset", type=str, default="hESC", 
                       help="Dataset name")
    parser.add_argument("--top-edges", type=int, default=10000,
                       help="Number of top edges to keep")
    parser.add_argument("--output-dir", type=str, default="./output/scgpt_heads",
                       help="Output directory for plots and data")
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute metrics
    print(f"Processing dataset: {args.dataset}")
    results_df = compute_metrics_for_heads(args.dataset, args.top_edges)
    
    if not results_df.empty:
        # Save data to CSV
        csv_path = output_dir / f"scgpt_{args.dataset}_head_metrics.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"Data saved: {csv_path}")
        
        # Print summary
        print(f"\nMetrics summary across heads:")
        for metric in METRICS:
            valid = results_df[metric].dropna()
            if not valid.empty:
                print(f"  {metric}: mean={valid.mean():.4f}, max={valid.max():.4f}, min={valid.min():.4f}, std={valid.std():.4f}")
        
        # Plot combined figure
        plot_combined_bar_charts(results_df, output_dir, args.dataset)
    else:
        print("No results to plot")


if __name__ == "__main__":
    main()