import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from fig4_palette import apply_fig4_style, model_color

warnings.filterwarnings("ignore")
AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 14
AXIS_LINEWIDTH = 1.2
LEGEND_SIZE = 14
TITLE_SIZE = 16

apply_fig4_style()

# ===================== 配置 =====================
PT_QUANTILE = 0.2
OUTPUT_DIR = Path(__file__).resolve().parent / "pseudotime_trajectory_plots_pred"


def _soften_hex(hex_color: str, blend_with_white: float = 0.22) -> str:
    """Blend a hex color with white to get a softer tone."""
    c = str(hex_color).strip().lstrip("#")
    if len(c) != 6:
        return hex_color
    r = int(c[0:2], 16)
    g = int(c[2:4], 16)
    b = int(c[4:6], 16)
    w = max(0.0, min(1.0, float(blend_with_white)))
    rr = int(round(r * (1.0 - w) + 255 * w))
    gg = int(round(g * (1.0 - w) + 255 * w))
    bb = int(round(b * (1.0 - w) + 255 * w))
    return f"#{rr:02X}{gg:02X}{bb:02X}"


# 使用 fig4 统一配色，并做轻微柔化
COLOR_EARLY = _soften_hex(model_color("Geneformer"))
COLOR_LATE = _soften_hex(model_color("scGPT"))
COLOR_PRED_LATE = _soften_hex(model_color("scFoundation"))
COLOR_BG = _soften_hex(model_color("STRING"), blend_with_white=0.55)
PRED_BASE_DIR = "/mnt/10T/yzn/benchmark_GRN/pre_scfoundation/scfoundation_multidataset_pseudotime_227/pca_data"
# =================================================

DATASETS = {
    "hESC": {
        "expr_csv": "/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/hESC_chip_matched-ExpressionData.csv",
        "pt_csv": "/mnt/10T/yzn/benchmark_GRN/PseudoTime/hESC/PseudoTime.csv",
        "pred_npy": f"{PRED_BASE_DIR}/hESC_pca_expression.npy"
    },
    "hHep": {
        "expr_csv": "/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/hHep_chip_matched-ExpressionData.csv",
        "pt_csv": "/mnt/10T/yzn/benchmark_GRN/PseudoTime/hHep/PseudoTime.csv",
        "pred_npy": f"{PRED_BASE_DIR}/hHep_pca_expression.npy"
    },
    "mDC": {
        "expr_csv": "/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/mDC_chip_matched-ExpressionData.csv",
        "pt_csv": "/mnt/10T/yzn/benchmark_GRN/PseudoTime/mDC/PseudoTime.csv",
        "pred_npy": f"{PRED_BASE_DIR}/mDC_pca_expression.npy"
    },
    "mHSC-E": {
        "expr_csv": "/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/mHSC-E_chip_matched-ExpressionData.csv",
        "pt_csv": "/mnt/10T/yzn/benchmark_GRN/PseudoTime/mHSC-E/PseudoTime.csv",
        "pred_npy": f"{PRED_BASE_DIR}/mHSC-E_pca_expression.npy"
    },
    "mHSC-GM": {
        "expr_csv": "/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/mHSC-GM_chip_matched-ExpressionData.csv",
        "pt_csv": "/mnt/10T/yzn/benchmark_GRN/PseudoTime/mHSC-GM/PseudoTime.csv",
        "pred_npy": f"{PRED_BASE_DIR}/mHSC-GM_pca_expression.npy"
    },
    "mHSC-L": {
        "expr_csv": "/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/mHSC-L_chip_matched-ExpressionData.csv",
        "pt_csv": "/mnt/10T/yzn/benchmark_GRN/PseudoTime/mHSC-L/PseudoTime.csv",
        "pred_npy": f"{PRED_BASE_DIR}/mHSC-L_pca_expression.npy"
    },
}

# 创建输出目录
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ===================== 核心修改：创建2行3列大图 =====================
fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor="white")
axes = axes.flatten()  # 展平为一维数组，方便遍历

# 批量绘制拟时间轨迹到子图
for idx, (ds_name, ds_cfg) in enumerate(DATASETS.items()):
    print(f"\n===== 处理 {ds_name} =====")
    ax = axes[idx]
    ax.set_facecolor("white")
    
    try:
        # 1. 加载真实表达数据和拟时间数据
        expr = pd.read_csv(ds_cfg["expr_csv"], index_col=0).T  # 细胞×基因
        pt_df = pd.read_csv(ds_cfg["pt_csv"])
        pt_df = pt_df.rename(columns={pt_df.columns[0]:"cell", pt_df.columns[1]:"pt"}).set_index("cell")

        # 2. 对齐真实表达数据和拟时间数据（按细胞名）
        common_cells = expr.index.intersection(pt_df.index)
        expr_filtered = expr.loc[common_cells]  # 过滤后：细胞×基因
        pt_filtered = pt_df.loc[common_cells, "pt"].values  # 过滤后拟时间
        real_cell_num = len(common_cells)
        print(f"  真实数据过滤后：{real_cell_num}个细胞")

        # 3. 划分早晚期掩码（基于真实细胞的拟时间）
        lo, hi = np.quantile(pt_filtered, [PT_QUANTILE, 1-PT_QUANTILE])
        early_mask = pt_filtered <= lo
        late_mask = pt_filtered >= hi
        print(f"  早期细胞数：{np.sum(early_mask)} | 晚期细胞数：{np.sum(late_mask)}")

        # 4. 计算真实数据的细胞平均表达
        real_mean_expr = expr_filtered.mean(axis=1).values

        # 5. 加载预测数据
        pred_data = np.load(ds_cfg["pred_npy"])
        if pred_data.shape[1] > pred_data.shape[0]:
            pred_data = pred_data.T
        pred_cell_num = pred_data.shape[0]
        print(f"  预测数据原始维度：{pred_data.shape}（细胞×基因）")

        # 匹配细胞数
        match_num = min(pred_cell_num, real_cell_num)
        pred_data_matched = pred_data[:match_num]
        pred_mean_expr = np.zeros(real_cell_num)
        pred_mean_expr[:match_num] = pred_data_matched.mean(axis=1)
        late_mask_pred = late_mask.copy()
        late_mask_pred[match_num:] = False
        print(f"  匹配的细胞数：{match_num} | 可绘制的预测晚期点：{np.sum(late_mask_pred)}")

        # 6. 绘图
        # 所有真实细胞（灰色背景）
        ax.scatter(
            pt_filtered, real_mean_expr, s=16, alpha=0.35, c=COLOR_BG,
            edgecolors="none", linewidths=0, label="_nolegend_", zorder=1
        )
        # 真实早期细胞
        ax.scatter(
            pt_filtered[early_mask], real_mean_expr[early_mask], s=36, alpha=0.9, c=COLOR_EARLY,
            edgecolors="none", linewidths=0, label="Observed early cells", zorder=3
        )
        # 真实晚期细胞
        ax.scatter(
            pt_filtered[late_mask], real_mean_expr[late_mask], s=36, alpha=0.9, c=COLOR_LATE,
            edgecolors="none", linewidths=0, label="Observed late cells", zorder=3
        )
        # 预测晚期细胞
        if np.sum(late_mask_pred) > 0:
            ax.scatter(
                pt_filtered[late_mask_pred], pred_mean_expr[late_mask_pred], s=36, alpha=0.9, c=COLOR_PRED_LATE,
                edgecolors="none", linewidths=0, label="Predicted late cells", zorder=4
            )

        # 拟合轨迹线
        sns.regplot(
            x=pt_filtered, y=real_mean_expr, scatter=False, color="black", lowess=True,
            line_kws={"alpha": 0.8, "linewidth": 1.2}, ax=ax
        )

        # 子图样式
        ax.set_xlabel("Pseudotime", fontsize=AXIS_LABEL_SIZE)
        ax.set_ylabel("Mean Gene Expression", fontsize=AXIS_LABEL_SIZE, labelpad=10)
        ax.set_title(ds_name, fontsize=TITLE_SIZE, pad=10)
        ax.grid(False)

        # 边框设置
        for side in ["bottom", "left"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color("black")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", width=AXIS_LINEWIDTH, labelsize=TICK_LABEL_SIZE, length=0, colors="black")
        ax.tick_params(axis="y", width=AXIS_LINEWIDTH, labelsize=TICK_LABEL_SIZE, length=0, colors="black")

        # 只在第一个子图显示图例
        if idx == 0:
            ax.legend(loc="best", fontsize=LEGEND_SIZE, frameon=False)

        print(f"✅ {ds_name} 绘制完成")

    except Exception as e:
        print(f"❌ {ds_name} 失败：{str(e)}")
        ax.text(0.5, 0.5, f"{ds_name}\nError", ha="center", va="center", transform=ax.transAxes, fontsize=14)
        continue

# 调整子图间距
plt.tight_layout()

# 保存为单张PDF大图
save_path = OUTPUT_DIR / "all_datasets_pseudotime_combined.pdf"
fig.savefig(save_path, dpi=600, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"\n✅ 所有数据集组合大图已保存到：{save_path.absolute()}")