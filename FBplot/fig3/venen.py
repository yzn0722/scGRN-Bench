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

def load_and_analyze_tfs(file_paths, method_names):
    """加载TF文件并进行高级分析"""
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
        
        # 取前50个TF
        top_tfs = set(df[tf_col].head(50).tolist())
        
        results[name] = {
            'data': df,
            'top_tfs': top_tfs,
            'jaccard_scores': jaccard_dict,
            'all_tfs': set(df[tf_col].tolist()),
            'tf_column': tf_col
        }
        
        print(f"  前5个TF: {list(top_tfs)[:5]}")
    
    return results

def calculate_overlap_metrics(results):
    """计算多种重叠指标"""
    metrics = {}
    
    # 1. 简单重叠数量
    sets = [results[m]['top_tfs'] for m in results]
    
    # 修复这里：正确计算交集
    if sets:
        overlap_tfs = sets[0].copy()
        for s in sets[1:]:
            overlap_tfs &= s
    else:
        overlap_tfs = set()
    
    # 2. 加权重叠分数（考虑TF排名）
    weighted_overlap = {}
    all_tfs_in_top = set()
    for s in sets:
        all_tfs_in_top.update(s)
    
    for tf in all_tfs_in_top:
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
    """创建可视化：仅输出单张韦恩图（无标题）+ 统计表。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) 仅创建单独韦恩图（不加标题）
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # 准备韦恩图数据
    from matplotlib_venn import venn3
    sets = [all_results[m]['top_tfs'] for m in all_results]

    if len(sets) == 3:
        venn3(
            sets,
            set_labels=[display_method_name(m) for m in all_results.keys()],
            set_colors=[method_color("att500"), method_color("emb500"), method_color("embhidden500")],
            alpha=0.7,
            ax=ax,
        )

    # 用户要求无标题，关闭坐标轴刻度/边框
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'tf_overlap_venn.pdf', bbox_inches='tight')
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
    top_n = 100
    overlap_dir = Path(__file__).resolve().parent / "tf_overlap"
    method_names = ["scGPT-att500", "scGPT-emb500", "scGPT-embhidden500"]
    file_paths = [
        overlap_dir / f"tf_top{top_n}_jaccard_{dataset}_top{top_n}_TFs_{m}.csv"
        for m in method_names
    ]
    
    # 输出目录（Path对象，支持与文件名用 / 拼接）
    output_dir = Path(__file__).resolve().parent / "tf_vene"
    
    print("="*60)
    print("高级TF重叠分析")
    print("="*60)
    
    # 1. 加载和分析数据
    print("\nStep 1: 加载和分析TF数据...")
    all_results = load_and_analyze_tfs(file_paths, method_names)
    
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
    print(f"  - tf_overlap_venn.png/pdf: 单独韦恩图（无标题）")
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

