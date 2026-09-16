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

# 批量绘制拟时间轨迹（无补0版）
for ds_name, ds_cfg in DATASETS.items():
    print(f"\n===== 处理 {ds_name} =====")
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

        # 5. 加载预测数据（无补0，只取和真实细胞数匹配的部分）
        pred_data = np.load(ds_cfg["pred_npy"])
        # 确保维度是细胞×基因（只判断一次，避免重复转置）
        if pred_data.shape[1] > pred_data.shape[0]:  # 基因×细胞 → 转置
            pred_data = pred_data.T
        pred_cell_num = pred_data.shape[0]
        print(f"  预测数据原始维度：{pred_data.shape}（细胞×基因）")

        # 核心修改：无补0，只取预测数据中前N个细胞（N=真实细胞数）
        # 如果预测细胞数 < 真实细胞数 → 只绘制预测数据存在的细胞对应的晚期点
        match_num = min(pred_cell_num, real_cell_num)
        pred_data_matched = pred_data[:match_num]  # 只取前match_num个细胞
        # 计算预测数据的平均表达（仅匹配的细胞）
        pred_mean_expr = np.zeros(real_cell_num)  # 初始化和真实细胞数一致的数组
        pred_mean_expr[:match_num] = pred_data_matched.mean(axis=1)  # 只填充有预测数据的部分
        # 过滤：只保留有预测数据的晚期点（避免无预测值的位置绘图）
        late_mask_pred = late_mask.copy()
        late_mask_pred[match_num:] = False  # 超出预测细胞数的晚期点不绘制
        print(f"  匹配的细胞数：{match_num} | 可绘制的预测晚期点：{np.sum(late_mask_pred)}")

        # 6. 绘图（统一散点风格：白底、无网格、只留左/下spine、tick length=0）
        fig, ax = plt.subplots(figsize=(5, 5), facecolor="white")
        ax.set_facecolor("white")

        # 所有真实细胞（灰色背景）
        ax.scatter(
            pt_filtered,
            real_mean_expr,
            s=16,
            alpha=0.35,
            c=COLOR_BG,
            edgecolors="none",
            linewidths=0,
            label="_nolegend_",
            zorder=1,
        )
        # 真实早期细胞（橙色）
        ax.scatter(
            pt_filtered[early_mask],
            real_mean_expr[early_mask],
            s=36,
            alpha=0.9,
            c=COLOR_EARLY,
            edgecolors="none",
            linewidths=0,
            label="Observed early cells",
            zorder=3,
        )
        # 真实晚期细胞（蓝色）
        ax.scatter(
            pt_filtered[late_mask],
            real_mean_expr[late_mask],
            s=36,
            alpha=0.9,
            c=COLOR_LATE,
            edgecolors="none",
            linewidths=0,
            label="Observed late cells",
            zorder=3,
        )
        # 预测晚期细胞（玫红）→ 只绘制有预测数据的部分
        if np.sum(late_mask_pred) > 0:
            ax.scatter(
                pt_filtered[late_mask_pred],
                pred_mean_expr[late_mask_pred],
                s=36,
                alpha=0.9,
                c=COLOR_PRED_LATE,
                edgecolors="none",
                linewidths=0,
                label="Predicted late cells",
                zorder=4,
            )

        # 拟合真实数据的轨迹线
        sns.regplot(
            x=pt_filtered,
            y=real_mean_expr,
            scatter=False,
            color="black",
            lowess=True,
            line_kws={"alpha": 0.8, "linewidth": 1.2},
            ax=ax,
        )

        # 图表美化
        ax.set_xlabel("Pseudotime", fontsize=AXIS_LABEL_SIZE)
        ax.set_ylabel("Mean Gene Expression", fontsize=AXIS_LABEL_SIZE,labelpad=12)
        
        #ax.set_title(ds_name, fontsize=16, color="#000000", pad=10)#给每个图片加标题
        ax.grid(False)

        # spine/ticks: align with fig4 scatter style
        for side in ["bottom", "left"]:
            ax.spines[side].set_visible(True)
            #ax.spines[side].set_linewidth(0.8)
            ax.spines[side].set_color("black")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", width=AXIS_LINEWIDTH, labelsize=TICK_LABEL_SIZE, length=0, colors="black")
        ax.tick_params(axis="y", width=AXIS_LINEWIDTH, labelsize=TICK_LABEL_SIZE, length=0, colors="black")
        

        ax.legend(loc="best", fontsize=14, frameon=False)
        fig.tight_layout()

        # 保存图片
        save_path = OUTPUT_DIR / f"{ds_name}_pseudotime_no_pad.pdf"
        fig.savefig(save_path, dpi=600, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"✅ {ds_name} 完成 → 保存至：{save_path}")

    except Exception as e:
        print(f"❌ {ds_name} 失败：{str(e)}")
        continue

print(f"\n所有图片已保存到：{OUTPUT_DIR.absolute()}")