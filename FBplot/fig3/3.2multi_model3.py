import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product, permutations
from datetime import datetime
from matplotlib.ticker import FormatStrFormatter

# unified colors for fig3 (consistent with fig2)
from fig3_palette import method_color, method_label, model_color

# -------------------------- 全局字体配置（全英文+适配系统字体） --------------------------
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 16,
    'axes.unicode_minus': False
})

# -------------------------- 工具函数：单个方法的精度计算 --------------------------
def calculate_method_precision(
    pred_file, 
    true_file, 
    TFEdges=True, 
    methodName="method",
    percents=None
):
    """计算单个方法在不同比例下的真实边比例"""
    if percents is None:
        percents = list(range(1, 101, 5))
    else:
        percents = sorted([p for p in percents if 0 < p <= 100])

    try:
        # 1. 预处理真实边
        trueEdgesDF = pd.read_csv(true_file, sep=',', header=0, index_col=None)
        trueEdgesDF = trueEdgesDF.loc[(trueEdgesDF['Gene1'] != trueEdgesDF['Gene2'])]
        trueEdgesDF.drop_duplicates(keep='first', inplace=True)
        if trueEdgesDF.empty:
            raise ValueError("Empty true edges file.")
        
        # TFEdges筛选
        if TFEdges:
            uniqueNodes = np.unique(trueEdgesDF.loc[:, ['Gene1','Gene2']])
            possibleEdges_TF = set(product(set(trueEdgesDF.Gene1), set(uniqueNodes)))
            possibleEdges_noSelf = set(permutations(uniqueNodes, r=2))
            valid_edges_set = possibleEdges_TF.intersection(possibleEdges_noSelf)
            valid_edges_str = {'|'.join(edge) for edge in valid_edges_set}
            trueEdgesDF['EdgeStr'] = trueEdgesDF['Gene1'] + "|" + trueEdgesDF['Gene2']
            trueEdgesDF = trueEdgesDF[trueEdgesDF['EdgeStr'].isin(valid_edges_str)]
        else:
            trueEdgesDF['EdgeStr'] = trueEdgesDF['Gene1'] + "|" + trueEdgesDF['Gene2']
        true_edges_set = set(trueEdgesDF['EdgeStr'].values)
        true_edges_count = len(true_edges_set)

        # 2. 预处理预测边
        if not os.path.exists(pred_file):
            raise FileNotFoundError(f"Prediction file not found: {pred_file}")
        predDF = pd.read_csv(pred_file, sep="\t", header=0, index_col=None)
        
        # 步骤1：原始预测边数
        original_pred_edges = len(predDF)
        
        # 步骤2：基础筛选（去自环、去重）
        predDF = predDF.loc[(predDF['Gene1'] != predDF['Gene2'])]
        predDF.drop_duplicates(keep='first', inplace=True)
        basic_filtered_edges = len(predDF)
        
        # 步骤3：TFEdges节点过滤
        if TFEdges:
            predDF['EdgeStr'] = predDF['Gene1'] + "|" + predDF['Gene2']
            predDF = predDF[predDF['EdgeStr'].isin(valid_edges_str)]
        else:
            predDF['EdgeStr'] = predDF['Gene1'] + "|" + predDF['Gene2']
        
        # 筛选后的总预测边数（未卡排名）
        filtered_pred_edges_count = len(predDF)
        
        if predDF.empty:
            raise ValueError("Empty prediction file after filtering.")
        
        # 按权重降序排序
        predDF['EdgeWeightAbs'] = predDF['EdgeWeight'].abs()
        predDF_sorted = predDF.sort_values('EdgeWeightAbs', ascending=False).reset_index(drop=True)

        # 3. 计算每个比例的精度
        precision_results = []
        for p in percents:
            n = int(np.ceil(filtered_pred_edges_count * p / 100))
            n = min(n, filtered_pred_edges_count) if filtered_pred_edges_count > 0 else 1
            if n == 0:
                n = 1
            
            top_n_edges = set(predDF_sorted.iloc[:n]['EdgeStr'].values)
            correct = len(top_n_edges.intersection(true_edges_set))
            precision = correct / n if n > 0 else 0
            
            precision_results.append({
                'Method': methodName,
                'Percent': p,
                'Precision': round(precision, 6),
                'TrueEdgesTotal': true_edges_count,
                'PredEdgesTotal': filtered_pred_edges_count,
                'OriginalPredEdges': original_pred_edges,
                'BasicFilteredEdges': basic_filtered_edges
            })

        return pd.DataFrame(precision_results), true_edges_count, valid_edges_set

    except Exception as e:
        print(f"❌ Error in {methodName}: {str(e)}")
        return None, None, None


# -------------------------- 核心函数：三种方法曲线绘制（含竖线） --------------------------
def plot_three_methods_grn_curve(
    methods,  # 格式：[(方法名, 预测文件路径), ...]
    true_file,
    TFEdges=True,
    percents=None,
    save_fig_path="three_methods_grn_curve.pdf",
    save_data_path="three_methods_grn_data.csv",
    figsize=(6, 6),
    show_title=False,
):
    """绘制scGPT三种表征方法的GRN推断质量对比曲线，添加「前真实边数」竖线"""
    text_size = 16  # align with fig2/0415/2.1-plot_leda.py
    if percents is None:
        percents = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    else:
        percents = sorted(percents)

    all_methods_data = []
    true_edges_count = None
    valid_edges_set = None

    # 1. 批量计算每个方法的精度数据
    for method_name, pred_file in methods:
        print(f"\n=== Processing {method_name} ===")
        method_data, tec, ves = calculate_method_precision(
            pred_file=pred_file,
            true_file=true_file,
            TFEdges=TFEdges,
            methodName=method_name,
            percents=percents
        )
        if method_data is not None:
            all_methods_data.append(method_data)
            if true_edges_count is None:
                true_edges_count = tec
                valid_edges_set = ves

    if not all_methods_data:
        print("❌ No valid method data to plot.")
        return

    # 2. 合并数据并保存
    merged_data = pd.concat(all_methods_data, ignore_index=True)
    if save_data_path:
        merged_data.to_csv(save_data_path, index=False, encoding='utf-8')
        print(f"\n✅ Three-method data saved to: {save_data_path}")

    # 3. 随机基准
    unique_nodes = np.unique(pd.read_csv(true_file, sep=',')[['Gene1', 'Gene2']])
    possible_edges_count = len(set(permutations(unique_nodes, r=2)))
    random_precision = true_edges_count / possible_edges_count if possible_edges_count > 0 else 0

    # 4. 绘图（publication-friendly: consistent palette + clean defaults）
    plt.figure(figsize=figsize)
    # 三种方法的样式配置（统一配色）
    method_styles = {
        "emb500": (method_color("emb500"), "o"),
        "embhidden500": (method_color("embhidden500"), "o"),
        "att500": (method_color("att500"), "o"),
    }
    reference_vertical_line_x = None

    # 绘制每个方法曲线 + 对应竖线
    for method_name in [m[0] for m in methods]:
        method_data = merged_data[merged_data['Method'] == method_name]
        if not method_data.empty:
            color, marker = method_styles.get(method_name, (model_color("STRING"), "o"))
            pred_edges_total = method_data['PredEdgesTotal'].iloc[0]
            if reference_vertical_line_x is None and pred_edges_total > 0:
                reference_vertical_line_x = min((true_edges_count / pred_edges_total) * 100, 100)
            x_pos = np.arange(1, len(method_data) + 1)
            # 绘制方法曲线
            plt.plot(
                x_pos,
                method_data['Precision'],
                linewidth=2,
                color=color,
                marker=marker,
                markersize=6,
                label=method_label(method_name),
            )

    # 画单条灰色竖线（不显示数字）
    if reference_vertical_line_x is not None and len(percents) > 0:
        plt.axvline(
            x=reference_vertical_line_x,
            color=model_color("STRING"),
            linestyle=':',
            linewidth=1.5,
            alpha=0.85,
        )

    # 随机基准线
    plt.axhline(
        y=random_precision,
        color=model_color("STRING"),
        linestyle="--",
        linewidth=2,
        label=f"Random Baseline\n(={round(random_precision, 4)})"
    )

    # 图表样式
    plt.xlabel('Predicted Edge Fraction', fontsize=text_size, fontweight='normal')
    plt.ylabel('True Edge Fraction ', fontsize=text_size, fontweight='normal')
    # 标题（可选，默认关闭）
    dataset_name = os.path.basename(true_file).split('_processed-network.csv')[0]
    if show_title:
        plt.title(
            f'scGPT Three Methods GRN Inference Quality Curve - {dataset_name}',
            fontsize=text_size,
            fontweight='normal',
            pad=20,
        )
    plt.legend(loc='upper right', fontsize=text_size, frameon=False)
    plt.grid(False)
    plt.xticks(np.arange(1, len(percents) + 1), [str(i) for i in range(1, len(percents) + 1)], rotation=0)
    plt.ylim(0, min(1.0, merged_data['Precision'].max() + 0.1))
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 保存
    plt.tight_layout(rect=[0, 0, 1, 1] if show_title else [0, 0, 1, 0.98])
    plt.savefig(Path(save_fig_path).with_suffix(".pdf"), dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"\n✅ Three-method curve saved to: {Path(save_fig_path).with_suffix('.pdf')}")


def parse_args():
    p = argparse.ArgumentParser(description="Plot scGPT three-method GRN curve for one dataset.")
    p.add_argument("--dataset", type=str, default="hESC", help="Dataset name, e.g. hESC / hHep / mDC")
    p.add_argument("--figsize", type=str, default="6,4", help="Figure size W,H (default: 6,6)")
    p.add_argument("--show-title", action="store_true", help="Show figure title (default: off)")
    p.add_argument(
        "--percents",
        type=str,
        default="1,2,3,4,5,6,7,8",
        help="Comma-separated percent values",
    )
    return p.parse_args()


# -------------------------- 主函数：单数据集命令行入口 --------------------------
if __name__ == "__main__":
    args = parse_args()
    dataset_name = str(args.dataset).strip()
    outdir = os.path.join(os.path.dirname(__file__), "curve")
    os.makedirs(outdir, exist_ok=True)

    true_file = f"/mnt/10T/yzn/benchmark_GRN/input_process/STRING/{dataset_name}_processed-network.csv"
    methods = [
        ("emb500", f"/mnt/10T/yzn/benchmark_GRN/evl_omipath/output_emb500/scgpt/scgpt_{dataset_name}.tsv"),
        ("embhidden500", f"/mnt/10T/yzn/benchmark_GRN/evl_omipath/output_embhidden500/scgpt/scGPT_{dataset_name}.tsv"),
        ("att500", f"/mnt/10T/yzn/benchmark_GRN/evl_omipath/output_att500/scgpt/scgpt_{dataset_name}.tsv"),
    ]
    w, h = (7.0, 5.0)
    custom_percents = [float(x) for x in str(args.percents).split(",") if str(x).strip()]

    plot_three_methods_grn_curve(
        methods=methods,
        true_file=true_file,
        TFEdges=True,
        percents=custom_percents,
        save_fig_path=os.path.join(outdir, f"scGPT_three_methods_grn_{dataset_name}.pdf"),
        save_data_path=os.path.join(outdir, f"scGPT_three_methods_grn_{dataset_name}.csv"),
        figsize=(w, h),
        show_title=bool(args.show_title),
    )