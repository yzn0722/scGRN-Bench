import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import glob
from scipy.stats import pearsonr  # 用这个，正确！
from fig4_palette import apply_fig4_style, model_color

apply_fig4_style()
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["savefig.edgecolor"] = "white"

AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 14
AXIS_LINEWIDTH = 1.2

BASE_DIR = "/mnt/10T/yzn/benchmark_GRN/pre_scgpt/results_multidataset_pseudotime_0326"
SCRIPT_DIR = Path(__file__).resolve().parent
OUTDIR = SCRIPT_DIR / "scater"
OUTDIR.mkdir(parents=True, exist_ok=True)

csv_files = sorted(glob.glob(os.path.join(BASE_DIR, "*_gene_result.csv")))

# 2×3 子图
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

blue_color = model_color("scGPT")
red_color = model_color("scPrint")

for idx, input_csv in enumerate(csv_files):
    dataset = os.path.basename(input_csv).replace("_gene_result.csv", "")
    df = pd.read_csv(input_csv)
    ax = axes[idx]

    correct_mask = df["dir_correct"] == 1
    wrong_mask = df["dir_correct"] == 0

    # 散点
    ax.scatter(
        df.loc[correct_mask, "delta_true"],
        df.loc[correct_mask, "delta_pred"],
        c=blue_color, s=40, alpha=0.75, edgecolors="none"
    )
    ax.scatter(
        df.loc[wrong_mask, "delta_true"],
        df.loc[wrong_mask, "delta_pred"],
        c=red_color, s=40, alpha=0.75, edgecolors="none"
    )

    # 范围
    all_vals = np.concatenate([df["delta_true"].values, df["delta_pred"].values])
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
    eps = 1e-8
    pad = 0.05 * (vmax - vmin + eps)

    # 对角线 + 十字线
    ax.plot([vmin-pad, vmax+pad], [vmin-pad, vmax+pad], "--", color="gray", lw=1, alpha=0.7)
    ax.axhline(0, linestyle=":", color="gray", lw=0.8, alpha=0.7)
    ax.axvline(0, linestyle=":", color="gray", lw=0.8, alpha=0.7)

    ax.set_xlim(vmin - pad, vmax + pad)
    ax.set_ylim(vmin - pad, vmax + pad)

    # 数据集标题
    ax.set_title(dataset, fontsize=16, pad=10)

    # 坐标轴
    if idx % 3 == 0:
        ax.set_ylabel(r"Predicted expression change $\Delta_{pred}$", fontsize=AXIS_LABEL_SIZE)
    if idx >= 3:
        ax.set_xlabel(r"Observed expression change $\Delta_{true}$", fontsize=AXIS_LABEL_SIZE)

    ax.grid(False)

    # 边框
    for side in ["bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", width=AXIS_LINEWIDTH, length=0, labelsize=TICK_LABEL_SIZE)

    # ===================== ✅ 正确：Pearson 相关系数 r =====================
    x = df["delta_true"]
    y = df["delta_pred"]
    corr, _ = pearsonr(x, y)

    ax.text(
        0.95, 0.1,
        f"$r = {corr:.3f}$",  # 这里显示 r，不是 R²
        transform=ax.transAxes,
        fontsize=14,
        ha="right"
    )

# 统一图例
legend_handles = [
    Line2D([0], [0], marker="o", linestyle="", markerfacecolor=blue_color, markeredgecolor="none", markersize=8),
    Line2D([0], [0], marker="o", linestyle="", markerfacecolor=red_color, markeredgecolor="none", markersize=8),
]
fig.legend(
    handles=legend_handles,
    labels=["Consistent", "Inconsistent"],
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),
    ncol=2,
    fontsize=16
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

out_path = OUTDIR / "all_datasets_scatter_2x3.pdf"
plt.savefig(out_path, dpi=600, facecolor="white", bbox_inches="tight")
plt.close()

print(f"\n✅ 2×3 大图已保存：{out_path}")