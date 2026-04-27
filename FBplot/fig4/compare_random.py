#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制 Random 和 scGPT 两个模型在各数据集上的平均准确率对比柱状图
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')
from fig4_palette import apply_fig4_style, model_color

apply_fig4_style()

def _to_final(value) -> float:
    """取一条 accuracy 曲线的 final accuracy（最后一个值）"""
    if not isinstance(value, list) or len(value) == 0:
        return float("nan")
    try:
        return float(value[-1])
    except (ValueError, TypeError):
        return float("nan")


def extract_final_accuracy(json_data: Dict) -> Dict[str, float]:
    """从扁平结构 {dataset: [curve]} 提取 final accuracy。"""
    out: Dict[str, float] = {}
    if not isinstance(json_data, dict):
        return out
    for dataset_key, value in json_data.items():
        v = _to_final(value)
        if np.isfinite(v):
            out[dataset_key] = v
    return out


def extract_random_mean_std(json_data: Dict) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    从随机多次结构提取每个数据集的 final accuracy 均值和标准差：
      {"run_1": {"hESC":[...], ...}, "run_2": {...}, ...}
    """
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    if not isinstance(json_data, dict):
        return means, stds

    run_keys = sorted(
        [k for k in json_data.keys() if re.match(r"^run_\d+$", str(k))],
        key=lambda x: int(str(x).split("_")[-1]),
    )
    per_ds: Dict[str, List[float]] = {}
    for rk in run_keys:
        block = json_data.get(rk, {})
        if not isinstance(block, dict):
            continue
        for ds, curve in block.items():
            v = _to_final(curve)
            if np.isfinite(v):
                per_ds.setdefault(ds, []).append(v)

    # Fallback: if file is flat {dataset: [curve]}, treat it as one run (std=0)
    if not per_ds:
        flat = extract_final_accuracy(json_data)
        for ds, v in flat.items():
            means[ds] = float(v)
            stds[ds] = 0.0
        return means, stds

    for ds, vals in per_ds.items():
        if len(vals) == 0:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        means[ds] = float(np.mean(arr))
        stds[ds] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return means, stds

def load_json_safely(filepath: str) -> Optional[Dict]:
    """安全加载JSON文件"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ 错误: 文件 {filepath} 不是有效的JSON格式: {e}")
        return None
    except Exception as e:
        print(f"❌ 加载文件 {filepath} 时出错: {e}")
        return None

def extract_model_name_from_path(filepath: str) -> str:
    """从文件路径提取有意义的模型名称"""
    # 提取目录名
    dirname = os.path.basename(os.path.dirname(filepath))
    filename = os.path.basename(filepath)
    
    # 优先从目录名判断
    if 'random' in dirname.lower() or 'random' in filename.lower():
        return 'Random'
    elif 'scgpt' in dirname.lower() or 'scgpt' in filename.lower():
        return 'scGPT'
    
    # 从路径中提取有意义的名称
    if 'results_multidataset_pseudotime_227_random' in filepath:
        return 'Random'
    elif 'scgpt_accuracy_curves.json' in filepath:
        return 'scGPT'
    
    # 如果都匹配不上，返回简化后的目录名
    return dirname.replace('_', ' ').title()

def plot_accuracy_comparison(
    json_path1: str,
    json_path2: str,
    output_pdf: str = "accuracy_comparison.pdf"
) -> None:
    """
    绘制两个模型的准确率对比柱状图
    
    Parameters:
    -----------
    json_path1 : str
        第一个JSON文件路径
    json_path2 : str
        第二个JSON文件路径
    output_pdf : str
        输出PDF文件名
    """
    
    print("=" * 60)
    print("开始绘制准确率对比柱状图")
    print("=" * 60)
    
    # 1. 从文件路径提取模型名称
    model1_name = extract_model_name_from_path(json_path1)
    model2_name = extract_model_name_from_path(json_path2)
    
    print(f"\n📁 文件信息:")
    print(f"  模型1 ({model1_name}): {json_path1}")
    print(f"  模型2 ({model2_name}): {json_path2}")
    
    # 2. 加载JSON数据
    print(f"\n📥 正在加载数据...")
    data1 = load_json_safely(json_path1)
    data2 = load_json_safely(json_path2)
    
    if data1 is None or data2 is None:
        print("❌ 数据加载失败，程序退出")
        return
    
    # 3. 提取准确率数据
    print(f"\n📊 提取准确率数据...")
    # 模型1默认是 random 多次结果：提取 mean±std（基于每次run的final accuracy）
    accuracies1, stds1 = extract_random_mean_std(data1)
    # 模型2默认是 scGPT 本次结果：提取 final accuracy
    accuracies2 = extract_final_accuracy(data2)
    
    print(f"  {model1_name} 提取到的数据集: {list(accuracies1.keys())}")
    print(f"  {model2_name} 提取到的数据集: {list(accuracies2.keys())}")
    
    # 4. 找出两个模型共有的数据集
    common_datasets = sorted(set(accuracies1.keys()) & set(accuracies2.keys()))
    # Remove mDC from comparison (requested)
    common_datasets = [ds for ds in common_datasets if ds != "mDC"]
    
    if not common_datasets:
        print("\n❌ 没有找到共有的数据集！")
        print("可能原因:")
        print("  1. 两个JSON文件的数据集名称不匹配")
        print("  2. 数据结构不符合预期")
        print("\n调试信息:")
        print(f"  {model1_name} 数据集: {list(accuracies1.keys())}")
        print(f"  {model2_name} 数据集: {list(accuracies2.keys())}")
        return
    
    print(f"\n✅ 找到 {len(common_datasets)} 个共有数据集: {common_datasets}")
    
    # 5. 准备绘图数据
    datasets = common_datasets
    model1_acc = [accuracies1[ds] for ds in datasets]
    model1_std = [stds1.get(ds, 0.0) for ds in datasets]
    model2_acc = [accuracies2[ds] for ds in datasets]
    
    # 6. 创建图形（按需求：仅尺寸为8x6；其余样式与 plot_accuracy_bar.py 对齐）
    plt.figure(figsize=(8, 6), dpi=600)
    
    # 设置柱状图的位置
    x = np.arange(len(datasets))  # 数据集位置
    width = 0.30  # 柱子的宽度（调窄，留出两柱间缝隙）
    
    # 统一 fig4 配色
    color1 = model_color("STRING")  # Random
    color2 = model_color("scGPT")   # scGPT
    
    # 绘制柱状图
    model1_acc_pct = [v * 100.0 for v in model1_acc]
    model1_std_pct = [v * 100.0 for v in model1_std]
    model2_acc_pct = [v * 100.0 for v in model2_acc]

    bars1 = plt.bar(x - width/2, model1_acc_pct, width, 
                    label=model1_name, color=color1, alpha=0.8,
                    edgecolor='none', linewidth=0.0)
    plt.errorbar(
        x - width/2,
        model1_acc_pct,
        yerr=model1_std_pct,
        fmt='none',
        ecolor='#222222',
        elinewidth=1.1,
        capsize=3,
        capthick=1.1,
        zorder=5,
    )
    
    bars2 = plt.bar(x + width/2, model2_acc_pct, width,
                    label=model2_name, color=color2, alpha=0.8,
                    edgecolor='none', linewidth=0.0)
    
    # 8. 设置图表样式
    plt.xlabel('')
    plt.ylabel('Direction Accuracy (%)', fontsize=16, fontweight='normal')
    
    plt.title("")
    
    # 设置X轴标签
    plt.xticks(x, datasets, rotation=0, ha='center', fontsize=14)
    
    # 设置Y轴范围和刻度（与 plot_accuracy_bar.py 一致）
    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 101, 20), fontsize=14)
    
    # 添加图例
    plt.legend(
        frameon=False,
        ncol=2,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.16),
        fontsize=14,
        borderaxespad=0,
    )
    
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
    
    # 9. 调整布局（给底部图例留空间）
    plt.subplots_adjust(bottom=0.24)
    
    # 10. 保存为PDF
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    
    print(f"\n✅ 图表已保存为: {output_pdf}")
    
    # 11. 打印统计摘要
    print(f"\n📈 统计摘要:")
    print(f"数据集数量: {len(datasets)}")
    print(f"{model1_name} 平均准确率: {np.mean(model1_acc):.4f}")
    print(f"{model2_name} 平均准确率: {np.mean(model2_acc):.4f}")
    
    # 比较每个数据集
    print(f"\n📊 详细对比:")
    for i, ds in enumerate(datasets):
        diff = model2_acc[i] - model1_acc[i]  # model2 - model1
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
        print(f"    {model1_name}: {model1_acc[i]:.4f} ± {model1_std[i]:.4f}")
        print(f"    {model2_name}: {model2_acc[i]:.4f}")
        print(f"    差值 ({model2_name}-{model1_name}): {diff_str} {diff_symbol}")
    
    print(f"\n" + "=" * 60)
    print("绘图完成！")
    print("=" * 60)

def main():
    """主函数"""
    # 你的文件路径
    json_path1 = "/mnt/10T/yzn/FoundBench/FBplot/fig4/results_multidataset_pseudotime_227_random/accuracy_curves.json"
    json_path2 = "/mnt/10T/yzn/FoundBench/FBplot/fig4/interation/scgpt_accuracy_curves.json"
    
    # 输出文件名
    output_pdf = "accuracy_comparison.pdf"
    
    # 检查文件是否存在
    if not os.path.exists(json_path1):
        print(f"❌ 错误: 文件不存在 - {json_path1}")
        return
    
    if not os.path.exists(json_path2):
        print(f"❌ 错误: 文件不存在 - {json_path2}")
        return
    
    # 绘制图表
    plot_accuracy_comparison(json_path1, json_path2, output_pdf)

if __name__ == "__main__":
    main()