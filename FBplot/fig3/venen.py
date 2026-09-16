#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级TF重叠分析：功能富集 + 网络中心性 + 调控重要性
修复set.intersection错误
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

# Unified fig3 palette (consistent with fig2)
from fig3_palette import METHOD_COLORS, method_label, model_color, method_color

# Colors used by this script (method + annotations)
COLORS = {
    "scGPT-att500": method_color("att500"),
    "scGPT-emb500": method_color("emb500"),
    "scGPT-embhidden500": method_color("embhidden500"),
    "Overlap": model_color("scGPT"),
    "Unique": model_color("STRING"),
}


def display_method_name(name: str) -> str:
    s = str(name).strip()
    if s.startswith("scGPT-"):
        suffix = s.split("-", 1)[1]
        return method_label(suffix)
    return s

def read_tf_file(file_path):
    """读取CSV文件，提取TF列数据"""
    try:
        df = pd.read_csv(file_path)
        print(f"读取文件: {file_path}")
        print(f"  列名: {list(df.columns)}")
        print(f"  行数: {len(df)}")
        
        # 检查是否有'TF'列
        tf_col = None
        for col in df.columns:
            if 'TF' in col or 'tf' in col.lower():
                tf_col = col
                break
        
        if tf_col is None:
            print(f"  未找到TF列，使用第一列: {df.columns[0]}")
            tf_list = df.iloc[:, 0].dropna().unique().tolist()
        else:
            print(f"  使用TF列: {tf_col}")
            tf_list = df[tf_col].dropna().unique().tolist()
        
        # 确保TF是字符串类型
        tf_list = [str(tf).strip() for tf in tf_list if str(tf).strip()]
        print(f"  提取到 {len(tf_list)} 个TF")
        if tf_list:
            print(f"  示例TF: {tf_list[:5]}")
        
        return set(tf_list)
    
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return set()

def partition_venn3_regions(set_a, set_b, set_c):
    """
    Partition the union of three sets into the seven disjoint Venn regions.
    Returns ordered dict: keys are stable region ids, values are sorted gene lists.
    """
    a, b, c = set(set_a), set(set_b), set(set_c)
    only_a = sorted(a - b - c)
    only_b = sorted(b - a - c)
    only_c = sorted(c - a - b)
    ab_only = sorted((a & b) - c)
    ac_only = sorted((a & c) - b)
    bc_only = sorted((b & c) - a)
    abc = sorted(a & b & c)
    return {
        "A only": only_a,
        "B only": only_b,
        "C only": only_c,
        "A∩B (not C)": ab_only,
        "A∩C (not B)": ac_only,
        "B∩C (not A)": bc_only,
        "A∩B∩C": abc,
    }


def load_and_analyze_tfs(
    file_paths,
    method_names,
    *,
    venn_top_n: int = 100,
    overlap_top_n: int = 50,
):
    """
    加载 TF 排名表。
    venn_top_n: 参与韦恩圆与「并集分区」名单的上限（例如 100 = 文件中的全部候选）。
    overlap_top_n: 用于三交「核心 TF」统计的上限（例如 50 → 与既有 11 个核心 TF 表一致）。
    """
    results = {}
    
    for path, name in zip(file_paths, method_names):
        print(f"\n分析: {name}")
        
        # 读取数据
        df = pd.read_csv(path)
        
        # 找到Jaccard列
        jaccard_col = None
        for col in df.columns:
            if 'jaccard' in col.lower() or 'Jaccard' in col:
                jaccard_col = col
                break
        
        # 找到TF列
        tf_col = None
        for col in df.columns:
            if 'TF' in col or 'tf' in col.lower():
                tf_col = col
                break
        if tf_col is None:
            tf_col = df.columns[0]
        
        # 清理数据
        df = df.dropna(subset=[tf_col])
        df[tf_col] = df[tf_col].astype(str).str.strip()
        
        # 如果有Jaccard列，计算排名
        if jaccard_col and jaccard_col in df.columns:
            df = df.sort_values(jaccard_col, ascending=False)
            df['Jaccard_Rank'] = range(1, len(df) + 1)
            jaccard_dict = dict(zip(df[tf_col], df[jaccard_col]))
        else:
            df['Jaccard_Rank'] = range(1, len(df) + 1)
            jaccard_dict = {}
            for tf in df[tf_col]:
                jaccard_dict[tf] = 1.0
        
        venn_slice = int(min(venn_top_n, len(df)))
        overlap_slice = int(min(overlap_top_n, len(df)))
        top_tfs = set(df[tf_col].head(venn_slice).tolist())
        overlap_basis_tfs = set(df[tf_col].head(overlap_slice).tolist())
        
        results[name] = {
            'data': df,
            'top_tfs': top_tfs,
            'overlap_basis_tfs': overlap_basis_tfs,
            'jaccard_scores': jaccard_dict,
            'all_tfs': set(df[tf_col].tolist()),
            'tf_column': tf_col,
            'venn_top_n': venn_slice,
            'overlap_top_n': overlap_slice,
        }
        
        print(f"  前5个TF: {list(top_tfs)[:5]}")
    
    return results

def calculate_overlap_metrics(results):
    """计算多种重叠指标"""
    metrics = {}
    
    # 1. 简单重叠数量（默认用 overlap_basis_tfs，与韦恩展示用的 top_tfs 可不同）
    sets = [
        results[m].get('overlap_basis_tfs', results[m]['top_tfs'])
        for m in results
    ]
    
    # 修复这里：正确计算交集
    if sets:
        overlap_tfs = sets[0].copy()
        for s in sets[1:]:
            overlap_tfs &= s
    else:
        overlap_tfs = set()
    
    # 2. 加权重叠分数（考虑TF排名）
    weighted_overlap = {}
    all_tfs_in_overlap_basis = set()
    for s in sets:
        all_tfs_in_overlap_basis.update(s)
    
    for tf in all_tfs_in_overlap_basis:
        ranks = []
        for method in results:
            df = results[method]['data']
            tf_col = results[method]['tf_column']
            if tf in df[tf_col].values:
                rank_row = df[df[tf_col] == tf]
                if not rank_row.empty and 'Jaccard_Rank' in rank_row.columns:
                    rank = rank_row['Jaccard_Rank'].values[0]
                    ranks.append(rank)
        
        if ranks:
            # 排名越靠前（数值越小），加权分数越高
            weighted_overlap[tf] = 1.0 / np.mean(ranks)
    
    # 3. 重叠稳定性指数
    if sets:
        min_set_size = min(len(s) for s in sets)
        stability_index = len(overlap_tfs) / min_set_size if min_set_size > 0 else 0
    else:
        stability_index = 0
    
    metrics['overlap_count'] = len(overlap_tfs)
    metrics['overlap_tfs'] = sorted(list(overlap_tfs))
    metrics['weighted_overlap'] = dict(sorted(weighted_overlap.items(), 
                                              key=lambda x: x[1], reverse=True))
    metrics['stability_index'] = stability_index
    
    return metrics

def analyze_regulatory_importance(overlap_tfs, all_results):
    """分析重叠TF的调控重要性"""
    importance_data = []
    
    for tf in overlap_tfs:
        tf_data = {'TF': tf}
        
        # 收集每个方法中该TF的指标
        jaccard_scores = []
        ranks = []
        
        for method in all_results:
            df = all_results[method]['data']
            tf_col = all_results[method]['tf_column']
            
            if tf in df[tf_col].values:
                row = df[df[tf_col] == tf].iloc[0]
                
                # 获取Jaccard分数
                for col in df.columns:
                    if 'jaccard' in col.lower() or 'Jaccard' in col:
                        jaccard_score = row[col]
                        tf_data[f'{method}_Jaccard'] = jaccard_score
                        jaccard_scores.append(jaccard_score)
                        break
                else:
                    tf_data[f'{method}_Jaccard'] = 1.0
                    jaccard_scores.append(1.0)
                
                # 获取排名
                if 'Jaccard_Rank' in row:
                    rank = row['Jaccard_Rank']
                    tf_data[f'{method}_Rank'] = rank
                    ranks.append(rank)
                else:
                    tf_data[f'{method}_Rank'] = 100
                    ranks.append(100)
            else:
                tf_data[f'{method}_Jaccard'] = 0
                tf_data[f'{method}_Rank'] = 1000
                jaccard_scores.append(0)
                ranks.append(1000)
        
        # 计算平均指标
        if jaccard_scores:
            tf_data['Mean_Jaccard'] = np.mean(jaccard_scores)
        else:
            tf_data['Mean_Jaccard'] = 0
        
        if ranks:
            tf_data['Mean_Rank'] = np.mean(ranks)
            tf_data['Rank_Std'] = np.std(ranks)
        else:
            tf_data['Mean_Rank'] = 1000
            tf_data['Rank_Std'] = 0
        
        # 排名稳定性
        tf_data['Stability'] = 1.0 / (tf_data['Rank_Std'] + 1e-6)  # 避免除以0
        
        importance_data.append(tf_data)
    
    if importance_data:
        importance_df = pd.DataFrame(importance_data)
        importance_df = importance_df.sort_values('Mean_Jaccard', ascending=False)
    else:
        importance_df = pd.DataFrame(columns=['TF', 'Mean_Jaccard', 'Mean_Rank', 'Rank_Std', 'Stability'])
    
    return importance_df

def create_comprehensive_plots(all_results, overlap_metrics, importance_df, output_dir):
    """创建可视化：韦恩图 + 可选「并集分区」全景基因列表（突出三交核心 TF）+ 统计表。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from matplotlib_venn import venn3

    method_order = list(all_results.keys())
    sets = [all_results[m]['top_tfs'] for m in method_order]
    core_highlight = set(overlap_metrics.get("overlap_tfs") or [])

    # 1) 单独韦恩图（无标题）
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    if len(sets) == 3:
        venn3(
            sets,
            set_labels=[display_method_name(m) for m in method_order],
            set_colors=[method_color("att500"), method_color("emb500"), method_color("embhidden500")],
            alpha=0.7,
            ax=ax,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / 'tf_overlap_venn.pdf', bbox_inches='tight')
    plt.close()

    # 2) 全景：韦恩并集内全部 TF（按 7 区划分）；三交核心 TF（与 overlap 表一致）加粗着色
    if len(sets) == 3:
        a, b, c = sets[0], sets[1], sets[2]
        regions = partition_venn3_regions(a, b, c)
        union_n = len(a | b | c)
        triple_n = len(regions["A∩B∩C"])
        venn_n_used = int(all_results[method_order[0]].get("venn_top_n", 100))
        overlap_n_used = int(all_results[method_order[0]].get("overlap_top_n", 50))

        per_line = 7
        line_dy = 0.018
        title_dy = 0.026
        region_order = [
            "A∩B∩C",
            "A∩B (not C)",
            "A∩C (not B)",
            "B∩C (not A)",
            "A only",
            "B only",
            "C only",
        ]
        label_a = display_method_name(method_order[0])
        label_b = display_method_name(method_order[1])
        label_c = display_method_name(method_order[2])
        pretty_title = {
            "A∩B∩C": f"Triple overlap ({label_a} ∩ {label_b} ∩ {label_c})",
            "A∩B (not C)": f"{label_a} ∩ {label_b} (not {label_c})",
            "A∩C (not B)": f"{label_a} ∩ {label_c} (not {label_b})",
            "B∩C (not A)": f"{label_b} ∩ {label_c} (not {label_a})",
            "A only": f"{label_a} only",
            "B only": f"{label_b} only",
            "C only": f"{label_c} only",
        }

        n_lines = 3
        for rid in region_order:
            gct = len(regions.get(rid, []))
            n_lines += 1
            n_lines += 1 if gct == 0 else int(np.ceil(gct / per_line))
            n_lines += 1
        fig_h = float(np.clip(7.5 + 0.095 * n_lines, 10.0, 48.0))

        fig = plt.figure(figsize=(16, fig_h))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.35], wspace=0.08)
        ax_v = fig.add_subplot(gs[0, 0])
        ax_txt = fig.add_subplot(gs[0, 1])
        venn3(
            sets,
            set_labels=[display_method_name(m) for m in method_order],
            set_colors=[method_color("att500"), method_color("emb500"), method_color("embhidden500")],
            alpha=0.7,
            ax=ax_v,
        )
        ax_v.set_xticks([])
        ax_v.set_yticks([])
        for side in ["top", "right", "bottom", "left"]:
            ax_v.spines[side].set_visible(False)

        ax_txt.axis("off")
        accent = COLORS.get("Overlap", method_color("emb500"))
        dim = "#555555"
        header = (
            f"Venn union: {union_n} TFs (each method top-{venn_n_used}; "
            f"{label_a} / {label_b} / {label_c}).\n"
            f"Triple overlap (diagram): {triple_n} TF(s). "
            f"Core highlight: top-{overlap_n_used} intersection, {len(core_highlight)} TF(s) "
            f"(bold colored when listed in triple-overlap region).\n"
        )
        ax_txt.text(
            0.0,
            1.0,
            header,
            transform=ax_txt.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            color=dim,
            linespacing=1.35,
        )

        y_cursor = 0.965
        body_fs = 7.5
        title_fs = 9.0
        for rid in region_order:
            genes = regions.get(rid, [])
            title = pretty_title.get(rid, rid)
            is_core_block = rid == "A∩B∩C"
            ax_txt.text(
                0.0,
                y_cursor,
                f"{title}  (n={len(genes)})",
                transform=ax_txt.transAxes,
                va="top",
                ha="left",
                fontsize=title_fs,
                fontweight="bold",
                color=accent if is_core_block else dim,
            )
            y_cursor -= title_dy
            if not genes:
                ax_txt.text(
                    0.02,
                    y_cursor,
                    "(none)",
                    transform=ax_txt.transAxes,
                    va="top",
                    ha="left",
                    fontsize=body_fs,
                    color="#999999",
                )
                y_cursor -= 0.045
                continue
            genes_sorted = sorted(genes)
            for i in range(0, len(genes_sorted), per_line):
                chunk = genes_sorted[i : i + per_line]
                x0 = 0.02
                x_pad = 0.132
                for j, g in enumerate(chunk):
                    is_core = is_core_block and (g in core_highlight)
                    ax_txt.text(
                        x0 + j * x_pad,
                        y_cursor,
                        g,
                        transform=ax_txt.transAxes,
                        va="top",
                        ha="left",
                        fontsize=body_fs + (1.2 if is_core else 0.0),
                        fontweight="bold" if is_core else "normal",
                        color=accent if is_core else "#222222",
                    )
                y_cursor -= line_dy
            y_cursor -= 0.012

        plt.tight_layout()
        plt.savefig(output_dir / "tf_overlap_venn_union_catalog.pdf", bbox_inches="tight")
        plt.close()
    
    # NOTE: 用户要求只输出一张图，因此这里不再单独生成 tf_rank_heatmap。
    
    # 3. 保存详细统计表格
    stats_path = output_dir / 'tf_overlap_detailed_statistics.csv'
    
    summary_data = []
    for tf in overlap_metrics['overlap_tfs']:
        row = {'TF': tf}
        for method in all_results:
            df = all_results[method]['data']
            tf_col = all_results[method]['tf_column']
            
            if tf in df[tf_col].values:
                tf_row = df[df[tf_col] == tf].iloc[0]
                
                # 找到Jaccard列
                jaccard_val = 0
                for col in df.columns:
                    if 'jaccard' in col.lower() or 'Jaccard' in col:
                        jaccard_val = tf_row.get(col, 0)
                        break
                
                row[f'{method}_Jaccard'] = jaccard_val
                
                # 找到排名
                if 'Jaccard_Rank' in tf_row:
                    row[f'{method}_Rank'] = tf_row['Jaccard_Rank']
                else:
                    row[f'{method}_Rank'] = 1000
            else:
                row[f'{method}_Jaccard'] = np.nan
                row[f'{method}_Rank'] = np.nan
        
        # 计算统计量
        jaccard_values = [row.get(f'{m}_Jaccard', np.nan) for m in all_results]
        valid_jaccard = [v for v in jaccard_values if not np.isnan(v)]
        
        if valid_jaccard:
            row['Mean_Jaccard'] = np.mean(valid_jaccard)
            row['Std_Jaccard'] = np.std(valid_jaccard)
        else:
            row['Mean_Jaccard'] = np.nan
            row['Std_Jaccard'] = np.nan
        
        rank_values = [row.get(f'{m}_Rank', np.nan) for m in all_results]
        valid_ranks = [r for r in rank_values if not np.isnan(r)]
        
        if valid_ranks:
            row['Mean_Rank'] = np.mean(valid_ranks)
            row['Rank_Std'] = np.std(valid_ranks)
            row['Rank_CV'] = row['Rank_Std'] / row['Mean_Rank'] if row['Mean_Rank'] > 0 else np.nan
        else:
            row['Mean_Rank'] = np.nan
            row['Rank_Std'] = np.nan
            row['Rank_CV'] = np.nan
        
        summary_data.append(row)
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Mean_Jaccard', ascending=False)
        summary_df.to_csv(stats_path, index=False)
    else:
        summary_df = pd.DataFrame()
    
    return summary_df

def main():
    # Prefer local fig3/tf_overlap outputs (generated by plot_tf_topn_overlap_heatmap.py)
    dataset = "hESC"
    csv_top = 100
    venn_top_n = 100
    overlap_top_n = 50
    overlap_dir = Path(__file__).resolve().parent / "tf_overlap"
    method_names = ["scGPT-att500", "scGPT-emb500", "scGPT-embhidden500"]
    file_paths = [
        overlap_dir / f"tf_top{csv_top}_jaccard_{dataset}_top{csv_top}_TFs_{m}.csv"
        for m in method_names
    ]
    
    # 输出目录（Path对象，支持与文件名用 / 拼接）
    output_dir = Path(__file__).resolve().parent / "tf_vene"
    
    print("="*60)
    print("高级TF重叠分析")
    print("="*60)
    
    # 1. 加载和分析数据
    print("\nStep 1: 加载和分析TF数据...")
    all_results = load_and_analyze_tfs(
        file_paths,
        method_names,
        venn_top_n=venn_top_n,
        overlap_top_n=overlap_top_n,
    )
    
    # 2. 计算重叠指标
    print("\nStep 2: 计算重叠指标...")
    overlap_metrics = calculate_overlap_metrics(all_results)
    
    print(f"\n重叠统计:")
    print(f"  重叠TF数量: {overlap_metrics['overlap_count']}")
    print(f"  重叠稳定性指数: {overlap_metrics['stability_index']:.3f}")
    
    if overlap_metrics['overlap_tfs']:
        print(f"  前10个重叠TF:")
        for tf in overlap_metrics['overlap_tfs'][:10]:
            print(f"    - {tf}")
    
    # 3. 分析调控重要性
    print("\nStep 3: 分析调控重要性...")
    importance_df = analyze_regulatory_importance(overlap_metrics['overlap_tfs'], all_results)
    
    if not importance_df.empty:
        print("\n最重要的重叠TF（按平均Jaccard分数）:")
        print(importance_df[['TF', 'Mean_Jaccard', 'Mean_Rank', 'Rank_Std']].head(10).to_string())
    else:
        print("没有找到重叠TF")
    
    # 4. 创建可视化
    print("\nStep 4: 创建可视化...")
    summary_df = create_comprehensive_plots(all_results, overlap_metrics, importance_df, output_dir)
    
    # 5. 生成最终报告
    print("\n" + "="*60)
    print("高级TF重叠分析完成!")
    print("="*60)
    print(f"\n核心发现:")
    print(f"1. 共有 {overlap_metrics['overlap_count']} 个TF在三个方法中同时出现")
    print(f"2. 方法一致性指数: {overlap_metrics['stability_index']:.3f}")
    
    if not importance_df.empty:
        print(f"3. 最重要的调控TF: {importance_df.iloc[0]['TF']} (平均Jaccard: {importance_df.iloc[0]['Mean_Jaccard']:.3f})")
        if 'Stability' in importance_df.columns and not importance_df.empty:
            most_stable_idx = importance_df['Stability'].idxmax()
            print(f"4. 最稳定的TF: {importance_df.loc[most_stable_idx, 'TF']} "
                  f"(稳定性: {importance_df.loc[most_stable_idx, 'Stability']:.3f})")
    
    print(f"\n所有结果已保存到: {output_dir}")
    print("\n主要输出文件:")
    print(f"  - tf_overlap_venn.pdf: 单独韦恩图（无标题）")
    print(f"  - tf_overlap_venn_union_catalog.pdf: 韦恩 + 并集分区全基因列表（三交核心 TF 突出）")
    print(f"  - tf_overlap_detailed_statistics.csv: 详细统计表格")
    
    # 6. 生成简单的文本报告
    report_path = output_dir / "analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write("TF重叠分析报告\n")
        f.write("="*50 + "\n\n")
        
        f.write("1. 数据概况\n")
        f.write("-"*20 + "\n")
        for method in all_results:
            f.write(f"{method}: {len(all_results[method]['top_tfs'])} 个top TF\n")
        
        f.write("\n2. 重叠分析\n")
        f.write("-"*20 + "\n")
        f.write(f"重叠TF总数: {overlap_metrics['overlap_count']}\n")
        f.write(f"重叠稳定性指数: {overlap_metrics['stability_index']:.3f}\n")
        
        if overlap_metrics['overlap_tfs']:
            f.write("\n重叠TF列表:\n")
            for tf in overlap_metrics['overlap_tfs']:
                f.write(f"  - {tf}\n")
        
        if not importance_df.empty:
            f.write("\n3. 重要TF排名\n")
            f.write("-"*20 + "\n")
            for i, row in importance_df.head(10).iterrows():
                f.write(f"{i+1:2d}. {row['TF']:15s} Jaccard={row['Mean_Jaccard']:.3f} "
                       f"Rank={row['Mean_Rank']:.1f}±{row['Rank_Std']:.1f}\n")
    
    print(f"  - analysis_report.txt: 文本分析报告")

if __name__ == "__main__":
    main()

