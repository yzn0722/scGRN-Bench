#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TF-level Top-N overlap (ranking consistency) - Separate raincloud and heatmap plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
import warnings
warnings.filterwarnings('ignore')

# Unified fig3 palette + labels
from fig3_palette import method_color, method_label, model_color

# Plot typography (keep ticks default; user wants horizontal labels)
TEXT_SIZE = 16


def display_method_name(name: str) -> str:
    s = str(name).strip()
    if s.startswith("scGPT-"):
        suffix = s.split("-", 1)[1]
        return method_label(suffix)
    return s


def display_method_color(name: str) -> str:
    s = str(name).strip()
    if s.startswith("scGPT-"):
        suffix = s.split("-", 1)[1]
        return method_color(suffix)
    return model_color(s)


def method_sort_key(name: str) -> tuple:
    """Keep scGPT 3-extract order stable for plots."""
    s = str(name).strip()
    if s == "scGPT-emb500":
        return (0, 0)
    if s == "scGPT-embhidden500":
        return (0, 1)
    if s == "scGPT-att500":
        return (0, 2)
    return (1, s)

# 基础函数定义
def norm_gene(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().upper()

def validate_columns(df: pd.DataFrame, required_cols: List[str], file_name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{file_name} 缺少必要列: {missing}")

def clean_gene_df(df: pd.DataFrame, gene1_col: str = "Gene1", gene2_col: str = "Gene2") -> pd.DataFrame:
    df = df.copy()
    df[gene1_col] = df[gene1_col].map(norm_gene)
    df[gene2_col] = df[gene2_col].map(norm_gene)
    df = df[(df[gene1_col] != "") & (df[gene2_col] != "")]
    return df.reset_index(drop=True)

def load_gt_network(
    gt_path: str,
) -> Tuple[Set[str], Set[str], int, Set[Tuple[str, str]], Dict[str, Set[str]]]:
    gt_df = pd.read_csv(gt_path)
    validate_columns(gt_df, ["Gene1", "Gene2"], "GT 文件")
    gt_df = clean_gene_df(gt_df, "Gene1", "Gene2")
    gt_gene1_set = set(gt_df["Gene1"].unique())
    gt_all_genes = set(gt_df["Gene1"].unique()).union(set(gt_df["Gene2"].unique()))
    gt_total_edges = int(len(gt_df))
    gt_edge_dir = set((a, b) for a, b in zip(gt_df["Gene1"], gt_df["Gene2"]))
    gt_targets_by_tf: Dict[str, Set[str]] = {}
    for tf, tgt in gt_edge_dir:
        gt_targets_by_tf.setdefault(tf, set()).add(tgt)
    return gt_gene1_set, gt_all_genes, gt_total_edges, gt_edge_dir, gt_targets_by_tf

def filter_prediction(pred_path: str, gt_gene1_set: Set[str], gt_all_genes: Set[str], gt_total_edges: int) -> pd.DataFrame:
    pred_df = pd.read_csv(pred_path, sep="\t")
    validate_columns(pred_df, ["Gene1", "Gene2", "EdgeWeight"], f"预测文件: {pred_path}")
    pred_df = clean_gene_df(pred_df, "Gene1", "Gene2")
    pred_df = pred_df.dropna(subset=["EdgeWeight"]).copy()
    pred_df = pred_df[pred_df["Gene1"].isin(gt_gene1_set) & pred_df["Gene2"].isin(gt_all_genes)].copy()
    pred_df = pred_df.sort_values("EdgeWeight", ascending=False).reset_index(drop=True)
    pred_df = pred_df.head(int(gt_total_edges)).copy()
    return pred_df

def compute_tf_metrics(
    pred_df: pd.DataFrame,
    *,
    gt_edge_dir: Set[Tuple[str, str]],
    gt_targets_by_tf: Dict[str, Set[str]],
    min_edges: int,
    score: str,
) -> pd.DataFrame:
    records = []
    for tf, group in pred_df.groupby("Gene1"):
        pred_targets = set(group["Gene2"].tolist())
        total = int(len(pred_targets))
        if total <= 0:
            continue
        gt_targets = gt_targets_by_tf.get(tf, set())
        hits = int(len(pred_targets.intersection(gt_targets)))

        precision = hits / total if total > 0 else 0.0
        if score == "f1_gt":
            recall = hits / len(gt_targets) if len(gt_targets) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        elif score == "f1_pred":
            recall = precision
            f1 = precision
        else:
            raise ValueError(f"Unknown --score {score!r}, choose from: f1_gt, f1_pred")

        records.append(
            {
                "TF": tf,
                "hits": hits,
                "pred_total": total,
                "gt_total": int(len(gt_targets)),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

    tf_df = pd.DataFrame(records)
    if tf_df.empty:
        return tf_df
    tf_df = tf_df[tf_df["pred_total"] >= int(min_edges)].copy()
    tf_df = tf_df.sort_values(by=["f1", "hits", "pred_total", "TF"], ascending=[False, False, False, True]).reset_index(drop=True)
    return tf_df

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TF Top-N overlap Jaccard heatmap (ranking consistency across methods)")
    p.add_argument("--dataset", type=str, default="hESC")
    p.add_argument("--top-n", type=int, default=100, help="Take Top-N TFs by TF-level f1")
    p.add_argument("--min-edges-per-tf", type=int, default=20, help="Only consider TFs with >= this many predicted edges")
    p.add_argument(
        "--score",
        choices=["f1_gt", "f1_pred"],
        default="f1_gt",
        help="How to rank TFs before taking Top-N. f1_gt uses GT targets (real F1). f1_pred matches legacy f1==precision.",
    )
    p.add_argument(
        "--include-genie3",
        action="store_true",
        default=False,
        help="Include GENIE3 as an extra method in the overlap heatmap.",
    )
    p.add_argument("--gt-dir", type=str, default="/mnt/10T/yzn/benchmark_GRN/input_process/STRING")
    p.add_argument("--emb500", type=str, default="/mnt/10T/yzn/benchmark_GRN/evl_omipath/output_emb500/scgpt")
    p.add_argument("--att500", type=str, default="/mnt/10T/yzn/benchmark_GRN/evl_omipath/output_att500/scgpt")
    p.add_argument("--embhidden500", type=str, default="/mnt/10T/yzn/benchmark_GRN/evl_omipath/output_embhidden500/scgpt")
    p.add_argument("--genie3", type=str, default="/mnt/10T/yzn/benchmark_GRN/evl_omipath/output_emb500/GENIE3")
    p.add_argument("--out", type=str, default="", help="Output directory path")
    return p.parse_args()

def create_jaccard_heatmap(mat: np.ndarray, methods: List[str], top_mean_f1: Dict[str, float], 
                          dataset: str, top_n: int, score: str, out_dir: Path) -> None:
    """
    创建Jaccard相似度热图，保存为单独的PDF文件
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    
    # 使用统一配色：白色 -> scGPT 主色
    cmap = LinearSegmentedColormap.from_list(
        "jaccard_blue",
        [(1.0, 1.0, 1.0), plt.matplotlib.colors.to_rgb(model_color("scGPT"))],
        N=256,
    )
    
    im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=1.0)

    # 设置坐标轴标签
    ax.set_xticks(np.arange(len(methods)))
    tick_labels = [display_method_name(m) for m in methods]
    
    ax.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=TEXT_SIZE)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(tick_labels, fontsize=TEXT_SIZE)

    # 添加数值标签
    for i in range(len(methods)):
        for j in range(len(methods)):
            val = mat[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", 
                   fontsize=TEXT_SIZE, fontweight="normal", 
                   color="white" if val > 0.55 else "black")

    # 颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Jaccard Similarity", rotation=270, labelpad=16, fontsize=TEXT_SIZE, color="#000000")
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # 网格线
    ax.set_xticks(np.arange(-0.5, len(methods), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
    ax.grid(False)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    # 移除边框
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 标题
    ax.set_title(f"{dataset} - Top-{top_n} TF Jaccard Similarity (ranked by {score})", 
                fontsize=TEXT_SIZE, fontweight='normal', pad=20, color="#000000")

    plt.tight_layout()
    
    # 保存热图
    heatmap_path = out_dir / f"heatmap_top{top_n}_{dataset}.pdf"
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"保存热图: {heatmap_path}")

def create_simple_raincloud_plot(combined_df: pd.DataFrame, methods: List[str], 
                                 dataset: str, top_n: int, score_type: str, 
                                 out_dir: Path) -> None:
    """
    创建简化的云雨图（小提琴+散点+箱线图），保存为单独的PDF文件
    """
    # 设置统一颜色
    method_colors = {m: display_method_color(m) for m in methods}
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # 绘制云雨图
    for idx, method in enumerate(methods):
        method_data = combined_df[combined_df['method'] == method]['f1'].dropna()
        if len(method_data) > 0:
            # 1. 绘制小提琴图（云）
            parts = ax.violinplot(method_data, positions=[idx], showmeans=False, 
                                  showmedians=False, showextrema=False)
            for pc in parts['bodies']:
                pc.set_facecolor(method_colors[method])
                pc.set_alpha(0.3)
                pc.set_edgecolor("none")
            
            # 2. 绘制散点图（雨）
            jitter = np.random.normal(idx, 0.08, size=len(method_data))
            ax.scatter(
                jitter,
                method_data,
                alpha=0.4,
                s=28,
                color=method_colors[method],
                edgecolor="none",
                linewidth=0.0,
            )
            
            # 3. 不绘制箱体/箱线，仅保留云+雨
    
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([display_method_name(m) for m in methods], rotation=0, ha='center', fontsize=TEXT_SIZE)
    ax.set_ylabel('F1 Score', fontsize=TEXT_SIZE, color="#000000")
    ax.tick_params(axis='y', labelsize=TEXT_SIZE)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    # ax.set_title(f'TF-level F1 Score Distribution (Ranked by {score_type})', 
    #             fontsize=12, fontweight='bold', pad=20)
    ax.grid(False)
    ax.set_ylim([-0.05, 0.6])
    
    # 移除上、右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # 保存云雨图
    raincloud_path = out_dir / f"raincloud_top{top_n}_{dataset}.pdf"
    plt.savefig(raincloud_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close(fig)
    print(f"保存云雨图: {raincloud_path}")

def main() -> None:
    args = parse_args()
    dataset = str(args.dataset).strip()
    top_n = int(args.top_n)
    min_edges = int(args.min_edges_per_tf)
    score = str(args.score)

    # 加载真实网络
    gt_path = Path(args.gt_dir) / f"{dataset}_processed-network.csv"
    print(f"加载真实网络: {gt_path}")
    gt_gene1_set, gt_all_genes, gt_total_edges, gt_edge_dir, gt_targets_by_tf = load_gt_network(str(gt_path))
    print(f"真实网络: {len(gt_gene1_set)} 个TF, {gt_total_edges} 条边")

    # 定义方法文件路径
    method_files: Dict[str, str] = {
        "scGPT-emb500": str(Path(args.emb500) / f"scgpt_{dataset}.tsv"),
        "scGPT-att500": str(Path(args.att500) / f"scgpt_{dataset}.tsv"),
        "scGPT-embhidden500": str(Path(args.embhidden500) / f"scGPT_{dataset}.tsv"),
    }
    if bool(args.include_genie3):
        method_files["GENIE3"] = str(Path(args.genie3) / f"GENIE3_{dataset}.tsv")

    # 创建输出目录
    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = Path(__file__).resolve().parent / "tf_overlap" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {out_dir}")

    # 存储结果
    top_sets: Dict[str, Set[str]] = {}
    top_lists: Dict[str, List[str]] = {}
    top_mean_f1: Dict[str, float] = {}
    all_tf_data: Dict[str, pd.DataFrame] = {}  # 保存所有TF的数据

    # 处理每个方法
    for method, fp in method_files.items():
        try:
            print(f"\n处理 {method}...")
            print(f"  预测文件: {fp}")
            
            if not Path(fp).exists():
                print(f"  文件不存在: {fp}")
                all_tf_data[method] = pd.DataFrame()
                top_lists[method] = []
                top_sets[method] = set()
                top_mean_f1[method] = float("nan")
                continue
                
            pred_df = filter_prediction(fp, gt_gene1_set, gt_all_genes, gt_total_edges)
            print(f"  过滤后预测: {len(pred_df)} 条边")
            
            tf_df = compute_tf_metrics(
                pred_df,
                gt_edge_dir=gt_edge_dir,
                gt_targets_by_tf=gt_targets_by_tf,
                min_edges=min_edges,
                score=score,
            )
            
            all_tf_data[method] = tf_df.copy()  # 保存所有数据
            
            if tf_df.empty:
                print(f"  {method}: 无有效TF数据")
                top_lists[method] = []
                top_sets[method] = set()
                top_mean_f1[method] = float("nan")
                continue
                
            top_df = tf_df.head(top_n).copy()
            top = top_df["TF"].tolist()
            top_lists[method] = top
            top_sets[method] = set(top)
            top_mean_f1[method] = float(top_df["f1"].mean()) if len(top_df) > 0 else float("nan")
            print(f"  {method}: 找到 {len(tf_df)} 个TF, Top-{top_n}平均F1={top_mean_f1[method]:.3f}")
            
        except Exception as e:
            print(f"处理 {method} 时出错: {e}")
            all_tf_data[method] = pd.DataFrame()
            top_lists[method] = []
            top_sets[method] = set()
            top_mean_f1[method] = float("nan")

    # 计算Jaccard矩阵
    methods = sorted(list(method_files.keys()), key=method_sort_key)
    mat = np.zeros((len(methods), len(methods)), dtype=np.float32)
    for i, mi in enumerate(methods):
        for j, mj in enumerate(methods):
            mat[i, j] = float(jaccard(top_sets[mi], top_sets[mj]))

    # 保存Jaccard矩阵
    jaccard_df = pd.DataFrame(mat, index=methods, columns=methods)
    jaccard_csv = out_dir / f"jaccard_top{top_n}_{dataset}.csv"
    jaccard_df.to_csv(jaccard_csv)
    print(f"\n保存Jaccard矩阵: {jaccard_csv}")
    
    # 保存每个方法的Top-N TF列表
    for method, tf_list in top_lists.items():
        if tf_list:
            tf_csv = out_dir / f"top{top_n}_TFs_{method}_{dataset}.csv"
            pd.DataFrame({"TF": tf_list}).to_csv(tf_csv, index=False)
            print(f"保存Top-N TF列表 ({method}): {tf_csv}")
    
    # 保存所有TF的完整评分数据
    all_scores_data = []
    for method, tf_df in all_tf_data.items():
        if not tf_df.empty:
            tf_df['method'] = method
            all_scores_data.append(tf_df)
    
    if all_scores_data:
        combined_scores = pd.concat(all_scores_data, ignore_index=True)
        scores_csv = out_dir / f"all_TF_scores_{dataset}.csv"
        combined_scores.to_csv(scores_csv, index=False)
        print(f"保存完整TF评分数据: {scores_csv}")

    # 1. 绘制Jaccard热图
    create_jaccard_heatmap(mat, methods, top_mean_f1, dataset, top_n, score, out_dir)

    # 2. 绘制简化的云雨图
    if all_scores_data:
        combined_df = pd.concat(all_scores_data, ignore_index=True)
        create_simple_raincloud_plot(combined_df, methods, dataset, top_n, score, out_dir)
    else:
        print("无数据可用于绘制云雨图")
    
    print(f"\n分析完成！输出文件:")
    print(f"1. 热图: {out_dir / f'heatmap_top{top_n}_{dataset}.pdf'}")
    print(f"2. 云雨图: {out_dir / f'raincloud_top{top_n}_{dataset}.pdf'}")
    print(f"3. 数据文件保存在: {out_dir}")

if __name__ == "__main__":
    main()