#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制 Random 和 Weight 两个模型在 top30 数据集上的平衡准确率对比柱状图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
import glob
import os
import warnings
warnings.filterwarnings('ignore')

# 导入样式（如果有fig4_palette的话）
try:
    from fig4_palette import apply_fig4_style, model_color
    apply_fig4_style()
except ImportError:
    # 如果没有，使用默认样式
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    def model_color(model_name):
        colors = {'STRING': '#4C72B0', 'scGPT': '#DD8452'}
        return colors.get(model_name, '#4C72B0')

def calculate_metrics(file_path, use_top30=True, top_percent=0.3):
    """计算单个文件的平衡准确率"""
    df = pd.read_csv(file_path)
    
    required_cols = ['dir_true', 'dir_pred', 'delta_true']
    if not all(col in df.columns for col in required_cols):
        return None
    
    if use_top30:
        df['abs_delta_true'] = df['delta_true'].abs()
        n_total = len(df)
        n_top = int(np.ceil(top_percent * n_total))
        df_used = df.nlargest(n_top, 'abs_delta_true')
    else:
        df_used = df
    
    y_true = (df_used['dir_true'] == 'Up').astype(int)
    y_pred = (df_used['dir_pred'] == 'Up').astype(int)
    
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    return bal_acc

def extract_random_mean_std(datasets, base_path_pattern):
    """提取随机实验的均值和标准差"""
    means = []
    stds = []
    
    for dataset in datasets:
        pattern = base_path_pattern.format(dataset=dataset)
        files = sorted(glob.glob(pattern))
        
        bal_accs = []
        for file_path in files:
            bal_acc = calculate_metrics(file_path, use_top30=True)
            if bal_acc is not None:
                bal_accs.append(bal_acc)
        
        if bal_accs:
            means.append(np.mean(bal_accs))
            stds.append(np.std(bal_accs, ddof=1) if len(bal_accs) > 1 else 0.0)
        else:
            means.append(np.nan)
            stds.append(0.0)
    
    return means, stds

def extract_weight_accuracies(datasets, base_path_pattern):
    """提取权重结果的准确率"""
    accuracies = []
    
    for dataset in datasets:
        file_path = base_path_pattern.format(dataset=dataset)
        if os.path.exists(file_path):
            bal_acc = calculate_metrics(file_path, use_top30=True)
            accuracies.append(bal_acc if bal_acc is not None else np.nan)
        else:
            accuracies.append(np.nan)
    
    return accuracies

def plot_accuracy_comparison(
    datasets,
    random_means,
    random_stds,
    weight_accs,
    output_pdf = "top30_balanced_accuracy.pdf"
) -> None:
    """
    绘制平衡准确率对比柱状图
    
    Parameters:
    -----------
    datasets : list
        数据集名称列表
    random_means : list
        随机实验的均值列表
    random_stds : list
        随机实验的标准差列表
    weight_accs : list
        权重实验的准确率列表
    output_pdf : str
        输出PDF文件名
    """
    
    print("=" * 60)
    print("绘制平衡准确率对比柱状图")
    print("=" * 60)
    
    # 转换为百分比
    random_means_pct = [v * 100.0 for v in random_means]
    random_stds_pct = [v * 100.0 for v in random_stds]
    weight_accs_pct = [v * 100.0 for v in weight_accs]
    
    # 创建图形（与第二个代码保持一致：8x6, dpi=600）
    plt.figure(figsize=(8, 6), dpi=600)
    
    # 设置柱状图的位置
    x = np.arange(len(datasets))
    width = 0.30  # 柱子宽度
    
    # 统一配色
    color_random = model_color("STRING")  # Random
    color_weight = model_color("scGPT")   # Weight
    
    # 绘制随机结果柱状图（带误差线）
    bars1 = plt.bar(x - width/2, random_means_pct, width, 
                    label='Random', 
                    color=color_random, 
                    alpha=0.8,
                    edgecolor='none', 
                    linewidth=0.0)
    
    # 添加误差线
    plt.errorbar(x - width/2, random_means_pct, yerr=random_stds_pct,
                 fmt='none', ecolor='#222222', elinewidth=1.1,
                 capsize=3, capthick=1.1, zorder=5)
    
    # 绘制权重结果柱状图
    bars2 = plt.bar(x + width/2, weight_accs_pct, width,
                    label='Weight', 
                    color=color_weight, 
                    alpha=0.8,
                    edgecolor='none', 
                    linewidth=0.0)
    
    # 设置图表样式（与第二个代码保持一致）
    plt.xlabel('')
    plt.ylabel('Balanced Accuracy (%)', fontsize=16, fontweight='normal')
    plt.title("")
    
    # 设置X轴标签
    plt.xticks(x, datasets, rotation=0, ha='center', fontsize=14)
    
    # 设置Y轴范围
    plt.ylim(0, 105)
    plt.yticks(np.arange(0, 101, 20), fontsize=14)
    
    # 添加图例（位置与第二个代码一致）
    plt.legend(frameon=False, ncol=2, loc='upper center',
               bbox_to_anchor=(0.5, -0.16), fontsize=14,
               borderaxespad=0)
    
    # 不显示背景网格线
    plt.grid(False)
    
    # 移除上、右边框
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#000000')
    ax.spines['bottom'].set_color('#000000')
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(axis='x', labelsize=14, length=0, colors='#000000')
    ax.tick_params(axis='y', labelsize=14, length=0, colors='#000000')
    
    # 调整布局（给底部图例留空间）
    plt.subplots_adjust(bottom=0.24)
    
    # 保存为PDF
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    
    print(f"\n✅ 图表已保存为: {output_pdf}")
    
    # 打印统计摘要
    print(f"\n📈 统计摘要:")
    print(f"数据集数量: {len(datasets)}")
    print(f"Random 平均准确率: {np.mean(random_means):.4f} ± {np.mean(random_stds):.4f}")
    print(f"Weight 平均准确率: {np.mean(weight_accs):.4f}")
    
    # 详细对比
    print(f"\n📊 详细对比:")
    for i, ds in enumerate(datasets):
        diff = weight_accs[i] - random_means[i]
        if diff > 0:
            diff_str = f"+{diff:.4f}"
            diff_symbol = "↑"
        elif diff < 0:
            diff_str = f"{diff:.4f}"
            diff_symbol = "↓"
        else:
            diff_str = "0.0000"
            diff_symbol = "="
        
        print(f"  {ds}:")
        print(f"    Random: {random_means[i]:.4f} ± {random_stds[i]:.4f}")
        print(f"    Weight: {weight_accs[i]:.4f}")
        print(f"    差值 (Weight-Random): {diff_str} {diff_symbol}")
    
    print("\n" + "=" * 60)
    print("绘图完成！")
    print("=" * 60)

def main():
    """主函数"""
    # 数据集列表（去掉mDC）
    datasets = ['hESC', 'hHep', 'mHSC-E', 'mHSC-GM', 'mHSC-L']
    
    # 文件路径模板
    random_pattern = "/mnt/10T/yzn/FoundBench/FBplot/fig4/results_multidataset_pseudotime_227_random/{dataset}_gene_result_run*.csv"
    weight_pattern = "/mnt/10T/yzn/benchmark_GRN/pre_scgpt/results_multidataset_pseudotime_227/{dataset}_gene_result.csv"
    
    print("=" * 60)
    print("处理 top30 结果 - 平衡准确率")
    print("=" * 60)
    
    # 提取随机结果（均值和标准差）
    print("\n📊 提取随机实验结果...")
    random_means, random_stds = extract_random_mean_std(datasets, random_pattern)
    
    for ds, mean, std in zip(datasets, random_means, random_stds):
        print(f"  {ds}: {mean:.4f} ± {std:.4f}")
    
    # 提取权重结果
    print("\n📊 提取权重结果...")
    weight_accs = extract_weight_accuracies(datasets, weight_pattern)
    
    for ds, acc in zip(datasets, weight_accs):
        print(f"  {ds}: {acc:.4f}")
    
    # 输出文件名
    output_pdf = "top30_balanced_accuracy.pdf"
    
    # 绘制图表
    plot_accuracy_comparison(datasets, random_means, random_stds, weight_accs, output_pdf)

if __name__ == "__main__":
    main()