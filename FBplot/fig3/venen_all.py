

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制TF出度分布图：突出显示重叠TF（按Degree排序，内部读Outdegree列）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

def plot_tf_outdegree_scatter(data_file, output_dir, top_n=50):
    """
    绘制TF Degree散点图，内部读Outdegree列，图上标签全用Degree
    """
    print("="*60)
    print("绘制TF Degree分布图（按Degree排序）")
    print("="*60)

    # 读取数据
    df = pd.read_csv(data_file)
    print(f"\n读取文件: {data_file}")
    print(f"数据行数: {len(df)}")
    print(f"列名: {list(df.columns)}")

    # 内部固定用 Outdegree 列
    if 'Outdegree' not in df.columns:
        print("错误: 数据中没有'Outdegree'列")
        return

    # 按Outdegree降序排序
    df = df.sort_values('Outdegree', ascending=False).reset_index(drop=True)
    df['Rank'] = range(1, len(df) + 1)

    # 绘图标签全部用 Degree
    x_label = 'Rank by Degree (High to Low)'
    title = 'TF Degree Distribution (Sorted by Degree)'

    # 识别重叠TF
    overlap_tfs = ['APEX1', 'HNRNPK', 'ILF2', 'NONO', 'PARP1',
                   'RBMX', 'RNPS1', 'SNRPB', 'SNRPD1', 'SSB', 'YBX1']
    df['TF_upper'] = df['TF'].str.upper()
    overlap_set = set([tf.upper() for tf in overlap_tfs])
    df['Is_Overlap'] = df['TF_upper'].isin(overlap_set)

    print(f"\n重叠TF数量: {df['Is_Overlap'].sum()}")
    print(f"重叠TF列表: {df[df['Is_Overlap']]['TF'].tolist()}")

    # 打印重叠TF的Degree排名
    print("\n重叠TF的Degree排名（按Degree从高到低）:")
    overlap_ranked = df[df['Is_Overlap']][['TF', 'Outdegree', 'Rank']].sort_values('Rank')
    for idx, row in overlap_ranked.iterrows():
        print(f"  {row['TF']:15s} Degree: {row['Outdegree']:4d} 排名: {row['Rank']}")

    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    overlap_df = df[df['Is_Overlap']]
    non_overlap_df = df[~df['Is_Overlap']]

    # 绘图y轴都是 Outdegree 数值
    ax.scatter(non_overlap_df['Rank'], non_overlap_df['Outdegree'],
               c='none', edgecolors='#4EA3F1', s=60, linewidth=1.5,
               alpha=0.7, label=f'Other TFs (n={len(non_overlap_df)})')

    ax.scatter(overlap_df['Rank'], overlap_df['Outdegree'],
               c='#4EA3F1', s=100, linewidth=2,
               alpha=1.0, label=f'Overlap TFs (n={len(overlap_df)})', zorder=5)

    # 标注TF名字
    for idx, row in overlap_df.iterrows():
        ax.annotate(row['TF'],
                   (row['Rank'], row['Outdegree']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   color='#4EA3F1', alpha=0.9,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.7))

    # 坐标轴标签用 Degree
    ax.set_xlabel(x_label, fontsize=16, fontweight='bold')
    ax.set_ylabel('Degree (Number of target genes)', fontsize=16, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    ax.set_xlim(0, len(df) + 5)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=12)

    # 统计文本也用 Degree
    stats_text = f"Statistics:\n"
    stats_text += f"Overlap TFs (n={len(overlap_df)}):\n"
    stats_text += f"  Mean Degree: {overlap_df['Outdegree'].mean():.1f}\n"
    stats_text += f"  Median Degree: {overlap_df['Outdegree'].median():.1f}\n"
    stats_text += f"  Mean Rank: {overlap_df['Rank'].mean():.1f}\n"
    stats_text += f"Other TFs (n={len(non_overlap_df)}):\n"
    stats_text += f"  Mean Degree: {non_overlap_df['Outdegree'].mean():.1f}\n"
    stats_text += f"  Median Degree: {non_overlap_df['Outdegree'].median():.1f}"

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    output_file = output_dir / "tf_degree_sorted_scatter.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ 散点图已保存到: {output_file}")

    png_file = output_dir / "tf_degree_sorted_scatter.png"
    plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ PNG格式已保存到: {png_file}")

    plt.close()

    print(f"\n📊 统计信息:")
    print(f"  重叠TF平均Degree: {overlap_df['Outdegree'].mean():.2f} ± {overlap_df['Outdegree'].std():.2f}")
    print(f"  重叠TF平均排名: {overlap_df['Rank'].mean():.1f}")
    print(f"  非重叠TF平均Degree: {non_overlap_df['Outdegree'].mean():.2f} ± {non_overlap_df['Outdegree'].std():.2f}")

    return df

def plot_log_scale_version(data_file, output_dir):
    """对数坐标图：内部Outdegree，标签显示Degree"""
    print("\n" + "="*60)
    print("绘制对数坐标版本（按Degree排序）")
    print("="*60)

    df = pd.read_csv(data_file)
    df = df.sort_values('Outdegree', ascending=False).reset_index(drop=True)
    df['Rank'] = range(1, len(df) + 1)

    overlap_tfs = ['APEX1', 'HNRNPK', 'ILF2', 'NONO', 'PARP1',
                   'RBMX', 'RNPS1', 'SNRPB', 'SNRPD1', 'SSB', 'YBX1']
    df['TF_upper'] = df['TF'].str.upper()
    overlap_set = set([tf.upper() for tf in overlap_tfs])
    df['Is_Overlap'] = df['TF_upper'].isin(overlap_set)

    overlap_df = df[df['Is_Overlap']]
    non_overlap_df = df[~df['Is_Overlap']]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.scatter(non_overlap_df['Rank'], non_overlap_df['Outdegree'],
                edgecolors='#4EA3F1', s=20, label=f'Other TFs (n={len(non_overlap_df)})')

    ax.scatter(overlap_df['Rank'], overlap_df['Outdegree'],
               c='#EB7E60', s=60, alpha=1.0, label=f'Overlap TFs (n={len(overlap_df)})')

    ax.set_yscale('log')
    ax.set_xlabel('Rank by Degree', fontsize=14)
    ax.set_ylabel('Degree (log scale)', fontsize=14)

    ax.set_xlim(0, len(df) + 5)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    ax.legend(loc='upper right', frameon=False, fontsize=15)
    #plt.tight_layout()

    output_file = output_dir / "tf_degree_sorted_scatter_log.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 对数坐标版本已保存到: {output_file}")
    plt.close()

def create_summary_table(df, output_dir):
    overlap_tfs = ['APEX1', 'HNRNPK', 'ILF2', 'NONO', 'PARP1',
                   'RBMX', 'RNPS1', 'SNRPB', 'SNRPD1', 'SSB', 'YBX1']
    df['TF_upper'] = df['TF'].str.upper()
    overlap_df = df[df['TF_upper'].isin([tf.upper() for tf in overlap_tfs])].copy()

    display_cols = ['TF', 'Outdegree', 'Rank']
    if 'Mean_Jaccard' in df.columns:
        display_cols.append('Mean_Jaccard')
    if 'Mean_Rank' in df.columns:
        display_cols.append('Mean_Rank')

    overlap_summary = overlap_df[display_cols].sort_values('Outdegree', ascending=False)
    overlap_summary.rename(columns={'Outdegree':'Degree'}, inplace=True)

    summary_file = output_dir / "overlap_tf_degree_sorted_summary.csv"
    overlap_summary.to_csv(summary_file, index=False)
    print(f"\n✅ 重叠TF Degree汇总表已保存到: {summary_file}")

    print("\n📊 重叠TF Degree汇总 (按Degree从高到低排序):")
    print("-" * 70)
    print(overlap_summary.to_string(index=False))

    return overlap_summary

def main():
    data_file = "/mnt/10T/yzn/scGRN-Bench/FBplot/fig3/tf_vene/all_tfs_with_outdegree.csv"
    output_dir = Path("/mnt/10T/yzn/scGRN-Bench/FBplot/fig3/tf_vene")

    if not Path(data_file).exists():
        print(f"错误: 文件不存在 {data_file}")
        return

    df = plot_tf_outdegree_scatter(data_file, output_dir, top_n=30)
    plot_log_scale_version(data_file, output_dir)
    summary_df = create_summary_table(df, output_dir)

    print("\n" + "="*60)
    print("绘制完成!")
    print("="*60)
    print(f"\n输出文件:")
    print(f"  - tf_degree_sorted_scatter.pdf/png: 主散点图")
    print(f"  - tf_degree_sorted_scatter_log.pdf: 对数坐标版本")
    print(f"  - overlap_tf_degree_sorted_summary.csv: 重叠TF Degree汇总表")

    print("\n🎯 重叠TF在Degree排名中的位置:")
    overlap_ranked = df[df['Is_Overlap']][['TF', 'Outdegree', 'Rank']].sort_values('Rank')
    for idx, row in overlap_ranked.iterrows():
        percentile = (1 - row['Rank']/len(df)) * 100
        print(f"  {row['TF']:15s} Degree: {row['Outdegree']:4d} 排名: {row['Rank']:4d} (前{percentile:.1f}%)")

if __name__ == "__main__":
    main()