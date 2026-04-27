#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PT_QUANTILE参数敏感性分析 - 单数据集折线图
"""

import os
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

warnings.filterwarnings("ignore")
os.environ["KMP_WARNINGS"] = "off"

# =====================================================
# 固定配置
# =====================================================
MODEL_DIR = "/mnt/10T/yzn/benchmark_GRN/pre_scgpt/scGPT/scgpt_human"
OUTDIR = "results_pt_quantile_sensitivity"

# =====================================================
# 单数据集配置
# =====================================================
DATASET_CONFIG = {
    "hESC": {
        "expr_csv": "/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/hESC_chip_matched-ExpressionData.csv",
        "pt_csv":   "/mnt/10T/yzn/benchmark_GRN/PseudoTime/hESC/PseudoTime.csv",
        "species": "human",
    }
}

DATASET_NAME = "hESC"

# =====================================================
# 测试的参数范围
# =====================================================
# 测试PT_QUANTILE参数，从0.05到0.45
PT_QUANTILE_VALUES = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
FIXED_TOP_PERCENT = 30
FIXED_EMA_ALPHA = 0.1
GEN_ITERS = 16
BATCH_SIZE = 16
NO_LOG1P = True

# =====================================================
# scGPT
# =====================================================
import sys
sys.path.insert(0, "/mnt/10T/yzn/benchmark_GRN/pre_scgpt/scGPT")
from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab


# =====================================================
# Utils
# =====================================================
def bin_expr_to_0_50(x, do_log1p=True):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0)
    x = np.clip(x, 0, None)
    if do_log1p:
        x = np.log1p(x)
    vmax = max(np.percentile(x, 99.5), 1e-6)
    return np.clip(x / vmax * 50.0, 0, 50).astype(np.float32)


def convert_mouse_to_human_gene(gene_name):
    return gene_name.upper()


# =====================================================
# Build model
# =====================================================
def build_model(model_dir, device):
    with open(Path(model_dir) / "args.json") as f:
        cfg = json.load(f)

    vocab = GeneVocab.from_file(Path(model_dir) / "vocab.json")
    for t in ["<pad>", "<cls>", "<eoc>"]:
        if t not in vocab:
            vocab.append_token(t)

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=cfg["embsize"],
        nhead=cfg["nheads"],
        d_hid=cfg["d_hid"],
        nlayers=cfg["nlayers"],
        vocab=vocab,
        pad_value=cfg["pad_value"],
        n_input_bins=cfg.get("n_bins", 51),
        use_fast_transformer=cfg.get("fast_transformer", True),
    )

    ckpt = torch.load(Path(model_dir) / "best_model.pt", map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()

    if device.type == "cuda":
        model.half()

    return model, vocab


# =====================================================
# PT_QUANTILE敏感性分析核心函数
# =====================================================
@torch.no_grad()
def run_pt_quantile_sensitivity_for_dataset(
    name: str,
    cfg: dict,
    model,
    vocab,
    device
) -> dict:
    """运行PT_QUANTILE参数敏感性分析"""
    
    print(f"\nRunning PT_QUANTILE sensitivity analysis for {name}")
    
    # 加载数据
    expr = pd.read_csv(cfg["expr_csv"], index_col=0)
    pt_df = pd.read_csv(cfg["pt_csv"])
    pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"}).set_index("cell")

    common = expr.columns.intersection(pt_df.index)
    expr = expr[common]
    pt = pt_df.loc[common, "pt"].to_numpy()

    # 获取基因名
    genes_original = expr.index.astype(str).tolist()
    species = cfg.get("species", "human")
    if species == "mouse":
        genes = [convert_mouse_to_human_gene(g) for g in genes_original]
    else:
        genes = genes_original
    
    # 预处理
    X = expr.T.to_numpy(dtype=np.float32)
    X_bin = bin_expr_to_0_50(X, do_log1p=(not NO_LOG1P))
    
    # 准备token
    gene_ids = np.array([vocab[g] if g in vocab else vocab["<pad>"] for g in genes])
    gene_ids = np.concatenate([[vocab["<cls>"]], gene_ids])
    gene_ids_tensor = torch.tensor(gene_ids[None, :], dtype=torch.long)
    
    X_in = np.concatenate([np.zeros((X_bin.shape[0], 1)), X_bin], axis=1)
    pad_mask = gene_ids_tensor.eq(vocab["<pad>"]).expand(X_in.shape[0], -1)
    values_tensor = torch.tensor(X_in, dtype=torch.float16 if device.type == "cuda" else torch.float32)
    
    # 准备更新掩码
    update_mask_1d = np.zeros(gene_ids_tensor.shape[1], dtype=bool)
    update_mask_1d[1:] = True
    
    # 存储结果
    results = {}
    
    # 测试每个PT_QUANTILE值
    for pt_quantile in PT_QUANTILE_VALUES:
        print(f"  Testing PT_QUANTILE = {pt_quantile:.2f}")
        
        # 计算伪时间分区
        lo, hi = np.quantile(pt, [pt_quantile, 1 - pt_quantile])
        early = pt <= lo
        late = pt >= hi
        
        early_mean = X_bin[early].mean(axis=0)
        late_mean = X_bin[late].mean(axis=0)
        true_delta = late_mean - early_mean
        
        # 计算要评估的基因
        total_genes = len(true_delta)
        top_n = max(int(total_genes * FIXED_TOP_PERCENT / 100), 1)
        top_idx = np.argsort(np.abs(true_delta))[::-1][:top_n].copy()
        
        print(f"    Early cells: {early.sum()} (pt <= {lo:.3f})")
        print(f"    Late cells: {late.sum()} (pt >= {hi:.3f})")
        print(f"    Evaluating top {top_n} genes")
        
        # 运行迭代生成
        acc_curve = iterative_direction_accuracy_pt_sensitivity(
            model=model,
            gene_ids_tensor=gene_ids_tensor,
            values_tensor=values_tensor[early],
            pad_mask=pad_mask[early],
            update_mask_1d=update_mask_1d,
            early_mean=early_mean,
            true_delta=true_delta,
            top_idx=top_idx,
            ema_alpha=FIXED_EMA_ALPHA
        )
        
        results[pt_quantile] = {
            "accuracy_curve": acc_curve,
            "final_accuracy": acc_curve[-1],
            "n_early_cells": int(early.sum()),
            "n_late_cells": int(late.sum()),
            "total_cells_ratio": (early.sum() + late.sum()) / len(pt),
            "early_threshold": float(lo),
            "late_threshold": float(hi),
            "pt_range_early_late": hi - lo,
            "top_n_genes": top_n
        }
        
        print(f"    Final accuracy: {acc_curve[-1]:.4f}")
    
    return results


@torch.no_grad()
def iterative_direction_accuracy_pt_sensitivity(
    model,
    gene_ids_tensor,
    values_tensor,
    pad_mask,
    update_mask_1d,
    early_mean,
    true_delta,
    top_idx,
    ema_alpha
):
    """迭代生成函数"""
    device = next(model.parameters()).device
    vals_all = values_tensor.clone()
    update_mask = torch.tensor(update_mask_1d[None, :], device=device).bool()

    acc_curve = []

    for it in range(GEN_ITERS):
        for start in range(0, vals_all.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, vals_all.shape[0])
            bs = end - start

            vals = vals_all[start:end].to(device)
            src = gene_ids_tensor.expand(bs, -1).to(device)
            mask = pad_mask[start:end].to(device)

            freeze = mask | (~update_mask.expand(bs, -1))
            out = model(src=src, values=vals, src_key_padding_mask=mask)
            new_vals = out["mlm_output"]

            vals = torch.where(
                freeze,
                vals,
                ema_alpha * vals + (1 - ema_alpha) * new_vals
            )

            vals_all[start:end] = vals.detach().cpu()

        # 计算准确率
        pred_mean = vals_all[:, 1:].numpy().mean(axis=0)
        pred_delta = pred_mean - early_mean
        acc = float((np.sign(pred_delta[top_idx]) == np.sign(true_delta[top_idx])).mean())
        acc_curve.append(acc)

    return acc_curve


# =====================================================
# 可视化函数 - 主要生成折线图
# =====================================================
def plot_pt_quantile_sensitivity(results: dict, outdir: Path, dataset_name: str):
    """绘制PT_QUANTILE参数敏感性分析图"""
    
    # 设置Nature风格
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 16,
        'axes.linewidth': 1.0,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'xtick.major.size': 0.0,
        'ytick.major.size': 0.0,
        'xtick.minor.size': 0.0,
        'ytick.minor.size': 0.0,
        'legend.fontsize': 14,
        'legend.frameon': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })
    
    # 准备数据
    pt_quantiles = sorted(results.keys())
    
    # 1. 主折线图：最终准确率 vs PT_QUANTILE
    fig, axes = plt.subplots(2, 2, figsize=(5, 5))
    
    # 1.1 最终准确率折线图
    ax = axes[0, 0]
    final_accs = [results[q]["final_accuracy"] for q in pt_quantiles]
    
    # 创建折线图
    line = ax.plot(pt_quantiles, final_accs, 'o-', 
                   color='#3498db', linewidth=2.5, markersize=8,
                   markerfacecolor='white', markeredgewidth=2)
    
    # 添加点标签
    for q, acc in zip(pt_quantiles, final_accs):
        ax.text(q, acc + 0.01, f'{acc:.3f}', 
               ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_xlabel('PT_QUANTILE (q)', fontsize=16)
    ax.set_ylabel('Final Accuracy', fontsize=16)
    ax.set_title(f'Final Accuracy vs PT_QUANTILE\n({dataset_name})', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0.4, 1.0)
    
    # 添加基准线
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)
    
    # 标记最佳值
    best_idx = np.argmax(final_accs)
    best_q = pt_quantiles[best_idx]
    best_acc = final_accs[best_idx]
    ax.scatter(best_q, best_acc, s=200, color='gold', 
              edgecolor='black', linewidth=2, zorder=5, 
              label=f'Best: q={best_q:.2f}')
    ax.legend(loc='best')
    
    # 1.2 细胞数量变化折线图
    ax = axes[0, 1]
    early_cells = [results[q]["n_early_cells"] for q in pt_quantiles]
    late_cells = [results[q]["n_late_cells"] for q in pt_quantiles]
    total_used = [early + late for early, late in zip(early_cells, late_cells)]
    
    # 绘制三条折线
    ax.plot(pt_quantiles, early_cells, 's-', color='#e74c3c', 
            linewidth=2, markersize=6, label='Early Cells')
    ax.plot(pt_quantiles, late_cells, '^-', color='#2ecc71', 
            linewidth=2, markersize=6, label='Late Cells')
    ax.plot(pt_quantiles, total_used, 'o-', color='#9b59b6', 
            linewidth=2, markersize=6, label='Total Used')
    
    ax.set_xlabel('PT_QUANTILE (q)', fontsize=16)
    ax.set_ylabel('Number of Cells', fontsize=16)
    ax.set_title('Cell Counts vs PT_QUANTILE', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best')
    
    # 添加次坐标轴显示比例
    ax2 = ax.twinx()
    cell_ratios = [results[q]["total_cells_ratio"] for q in pt_quantiles]
    ax2.plot(pt_quantiles, cell_ratios, '--', color='#34495e', 
             linewidth=1.5, alpha=0.7, label='Used Ratio')
    ax2.set_ylabel('Fraction of Cells Used', fontsize=16, color='#34495e')
    ax2.tick_params(axis='y', labelcolor='#34495e', length=0)
    ax2.set_ylim(0, 1.0)
    
    # 1.3 伪时间阈值变化折线图
    ax = axes[1, 0]
    early_thresholds = [results[q]["early_threshold"] for q in pt_quantiles]
    late_thresholds = [results[q]["late_threshold"] for q in pt_quantiles]
    pt_ranges = [results[q]["pt_range_early_late"] for q in pt_quantiles]
    
    # 绘制三条折线
    ax.plot(pt_quantiles, early_thresholds, 's-', color='#e67e22', 
            linewidth=2, markersize=6, label='Early Threshold (lo)')
    ax.plot(pt_quantiles, late_thresholds, '^-', color='#16a085', 
            linewidth=2, markersize=6, label='Late Threshold (hi)')
    ax.plot(pt_quantiles, pt_ranges, 'o-', color='#c0392b', 
            linewidth=2, markersize=6, label='Range (hi - lo)')
    
    ax.set_xlabel('PT_QUANTILE (q)', fontsize=16)
    ax.set_ylabel('Pseudotime Threshold', fontsize=16)
    ax.set_title('Pseudotime Thresholds vs PT_QUANTILE', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best')
    
    # 1.4 收敛曲线对比（选择几个关键点）
    ax = axes[1, 1]
    
    # 选择几个代表性的PT_QUANTILE值
    key_quantiles = [0.05, 0.2, 0.35, 0.45]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    
    for q, color in zip(key_quantiles, colors):
        if q in results:
            acc_curve = results[q]["accuracy_curve"]
            ax.plot(range(1, GEN_ITERS+1), acc_curve, 'o-', 
                    color=color, linewidth=1.5, markersize=4,
                    label=f'q={q:.2f}')
    
    ax.set_xlabel('Iteration', fontsize=16)
    ax.set_ylabel('Accuracy', fontsize=16)
    ax.set_title('Convergence Curves for Key PT_QUANTILE Values', 
                fontsize=14, fontweight='bold')
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right')
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(outdir / f"pt_quantile_sensitivity_main_{dataset_name}.pdf")
    plt.close()
    
    # 2. 详细折线图：不同指标的关系
    fig, axes = plt.subplots(1, 2, figsize=(5, 5))
    
    # 2.1 准确率与细胞数量的关系
    ax = axes[0]
    
    # 创建散点图，颜色表示PT_QUANTILE
    sc = ax.scatter(total_used, final_accs, c=pt_quantiles, 
                   cmap='viridis', s=100, edgecolor='black', linewidth=1)
    
    # 添加颜色条
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('PT_QUANTILE', fontsize=14)
    
    # 添加点标签
    for q, total, acc in zip(pt_quantiles, total_used, final_accs):
        ax.annotate(f'q={q:.2f}', (total, acc), 
                   fontsize=14, ha='center', va='bottom')
    
    ax.set_xlabel('Total Cells Used (Early + Late)', fontsize=16)
    ax.set_ylabel('Final Accuracy', fontsize=16)
    ax.set_title('Accuracy vs Total Cells Used\n(Color: PT_QUANTILE)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 2.2 准确率与细胞比例的关系
    ax = axes[1]
    
    # 计算早期细胞比例
    early_ratios = [early / (early + late) if (early + late) > 0 else 0 
                    for early, late in zip(early_cells, late_cells)]
    
    sc = ax.scatter(early_ratios, final_accs, c=pt_quantiles,
                   cmap='plasma', s=100, edgecolor='black', linewidth=1)
    
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('PT_QUANTILE', fontsize=14)
    
    for q, ratio, acc in zip(pt_quantiles, early_ratios, final_accs):
        ax.annotate(f'q={q:.2f}', (ratio, acc), 
                   fontsize=14, ha='center', va='bottom')
    
    ax.set_xlabel('Early Cell Ratio (Early / Total Used)', fontsize=16)
    ax.set_ylabel('Final Accuracy', fontsize=16)
    ax.set_title('Accuracy vs Early Cell Ratio\n(Color: PT_QUANTILE)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(outdir / f"pt_quantile_relationships_{dataset_name}.pdf")
    plt.close()
    
    # 3. 汇总折线图：将所有关键指标放在一起
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # 创建主坐标轴
    ax1 = ax
    line1 = ax1.plot(pt_quantiles, final_accs, 'o-', 
                     color='#3498db', linewidth=2.5, markersize=8,
                     markerfacecolor='white', label='Final Accuracy')
    ax1.set_xlabel('PT_QUANTILE (q)', fontsize=16)
    ax1.set_ylabel('Final Accuracy', fontsize=16, color='#3498db')
    ax1.tick_params(axis='y', labelcolor='#3498db', length=0)
    ax1.set_ylim(0.4, 1.0)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 创建第二个坐标轴
    ax2 = ax1.twinx()
    line2 = ax2.plot(pt_quantiles, total_used, 's-', 
                     color='#e74c3c', linewidth=2, markersize=6,
                     markerfacecolor='white', label='Total Cells Used')
    ax2.set_ylabel('Total Cells Used', fontsize=16, color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c', length=0)
    
    # 创建第三个坐标轴
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    line3 = ax3.plot(pt_quantiles, pt_ranges, '^-', 
                     color='#2ecc71', linewidth=2, markersize=6,
                     markerfacecolor='white', label='PT Range')
    ax3.set_ylabel('Pseudotime Range (hi - lo)', fontsize=16, color='#2ecc71')
    ax3.tick_params(axis='y', labelcolor='#2ecc71', length=0)
    
    # 合并图例
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=14)
    
    ax1.set_title(f'Comprehensive PT_QUANTILE Sensitivity Analysis\n({dataset_name})', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(outdir / f"pt_quantile_comprehensive_{dataset_name}.pdf")
    plt.close()
    
    # 4. 热力图：每次迭代的准确率
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # 创建数据矩阵
    heatmap_data = np.zeros((len(pt_quantiles), GEN_ITERS))
    for i, q in enumerate(pt_quantiles):
        heatmap_data[i, :] = results[q]["accuracy_curve"]
    
    # 绘制热图
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', 
                   interpolation='nearest', vmin=0.4, vmax=1.0)
    
    ax.set_xlabel('Iteration', fontsize=16)
    ax.set_ylabel('PT_QUANTILE', fontsize=16)
    ax.set_title('Accuracy Heatmap: PT_QUANTILE × Iteration', 
                fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(pt_quantiles)))
    ax.set_yticklabels([f'{q:.2f}' for q in pt_quantiles])
    ax.set_xticks(range(0, GEN_ITERS, 2))
    ax.set_xticklabels(range(1, GEN_ITERS+1, 2))
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(outdir / f"pt_quantile_heatmap_{dataset_name}.pdf")
    plt.close()
    
    print(f"✓ Generated 4 PT_QUANTILE sensitivity analysis plots")


# =====================================================
# 主函数
# =====================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Testing PT_QUANTILE sensitivity for dataset: {DATASET_NAME}")
    print(f"PT_QUANTILE values: {PT_QUANTILE_VALUES}")
    print(f"Fixed TOP_PERCENT: {FIXED_TOP_PERCENT}")
    print(f"Fixed EMA_ALPHA: {FIXED_EMA_ALPHA}")
    
    # 构建模型
    model, vocab = build_model(MODEL_DIR, device)
    print(f"Vocab size: {len(vocab)}")
    
    # 创建输出目录
    outdir = Path(OUTDIR)
    outdir.mkdir(exist_ok=True, parents=True)
    
    # 运行PT_QUANTILE敏感性分析
    print(f"\n{'='*60}")
    print(f"Running PT_QUANTILE sensitivity analysis")
    print('='*60)
    
    results = run_pt_quantile_sensitivity_for_dataset(
        name=DATASET_NAME,
        cfg=DATASET_CONFIG[DATASET_NAME],
        model=model,
        vocab=vocab,
        device=device
    )
    
    # 保存结果
    with open(outdir / f"pt_quantile_sensitivity_{DATASET_NAME}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # 绘制图表
    print(f"\n{'='*60}")
    print("Generating sensitivity analysis plots")
    print('='*60)
    
    plot_pt_quantile_sensitivity(results, outdir, DATASET_NAME)
    
    # 生成汇总表格
    generate_summary_table(results, outdir, DATASET_NAME)
    
    # 打印简要结果
    print(f"\n{'='*60}")
    print("PT_QUANTILE SENSITIVITY ANALYSIS RESULTS")
    print('='*60)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Total experiments: {len(PT_QUANTILE_VALUES)}")
    print("\nResults summary:")
    print("-" * 80)
    print(f"{'PT_QUANTILE':<12} {'Final Acc':<10} {'Early Cells':<12} {'Late Cells':<12} {'Total Used':<12} {'Used Ratio':<12} {'PT Range':<12}")
    print("-" * 80)
    
    for q in sorted(results.keys()):
        res = results[q]
        total_used = res['n_early_cells'] + res['n_late_cells']
        used_ratio = res['total_cells_ratio']
        pt_range = res['pt_range_early_late']
        
        print(f"{q:<12.2f} {res['final_accuracy']:<10.4f} {res['n_early_cells']:<12} "
              f"{res['n_late_cells']:<12} {total_used:<12} {used_ratio:<12.3f} {pt_range:<12.3f}")
    
    # 找到最佳参数
    best_q = max(results.keys(), key=lambda x: results[x]['final_accuracy'])
    best_res = results[best_q]
    
    print("\n" + "="*80)
    print(f"RECOMMENDATION")
    print("="*80)
    print(f"Best PT_QUANTILE: {best_q:.2f}")
    print(f"Final Accuracy: {best_res['final_accuracy']:.4f}")
    print(f"Early Threshold: {best_res['early_threshold']:.3f}")
    print(f"Late Threshold: {best_res['late_threshold']:.3f}")
    print(f"Early Cells: {best_res['n_early_cells']}")
    print(f"Late Cells: {best_res['n_late_cells']}")
    print(f"Total Cells Used: {best_res['n_early_cells'] + best_res['n_late_cells']}")
    print(f"Used Ratio: {best_res['total_cells_ratio']:.3f}")
    print(f"Pseudotime Range: {best_res['pt_range_early_late']:.3f}")
    print(f"\nAll results saved to: {outdir}")
    print("="*80)


def generate_summary_table(results: dict, outdir: Path, dataset_name: str):
    """生成汇总表格"""
    
    # 准备数据
    data = []
    for q in sorted(results.keys()):
        res = results[q]
        data.append({
            'PT_QUANTILE': q,
            'Final_Accuracy': res['final_accuracy'],
            'Early_Threshold': res['early_threshold'],
            'Late_Threshold': res['late_threshold'],
            'PT_Range': res['pt_range_early_late'],
            'Early_Cells': res['n_early_cells'],
            'Late_Cells': res['n_late_cells'],
            'Total_Used_Cells': res['n_early_cells'] + res['n_late_cells'],
            'Used_Cell_Ratio': res['total_cells_ratio'],
            'Top_N_Genes': res['top_n_genes'],
            'Iter1_Accuracy': res['accuracy_curve'][0] if len(res['accuracy_curve']) > 0 else None,
            'Iter2_Accuracy': res['accuracy_curve'][1] if len(res['accuracy_curve']) > 1 else None,
            'Iter3_Accuracy': res['accuracy_curve'][2] if len(res['accuracy_curve']) > 2 else None,
            'Iter4_Accuracy': res['accuracy_curve'][3] if len(res['accuracy_curve']) > 3 else None,
            'Iter5_Accuracy': res['accuracy_curve'][4] if len(res['accuracy_curve']) > 4 else None,
            'Iter6_Accuracy': res['accuracy_curve'][5] if len(res['accuracy_curve']) > 5 else None,
            'Iter7_Accuracy': res['accuracy_curve'][6] if len(res['accuracy_curve']) > 6 else None,
            'Iter8_Accuracy': res['accuracy_curve'][7] if len(res['accuracy_curve']) > 7 else None,
        })
    
    df = pd.DataFrame(data)
    
    # 保存为CSV
    csv_path = outdir / f"pt_quantile_results_{dataset_name}.csv"
    df.to_csv(csv_path, index=False)
    
    # 保存为Excel
    excel_path = outdir / f"pt_quantile_results_{dataset_name}.xlsx"
    df.to_excel(excel_path, index=False)
    
    print(f"✓ Results table saved to: {csv_path}")
    print(f"✓ Results table saved to: {excel_path}")
    
    return df


if __name__ == "__main__":
    main()