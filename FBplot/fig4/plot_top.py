# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import numpy as np
# # import os
# # import glob
# # from fig4_palette import apply_fig4_style, model_color, FIG4_FIGSIZE
# # from matplotlib.ticker import FuncFormatter

# # apply_fig4_style()

# # # ===================== 路径与百分比设置 =====================
# # base_path = "/mnt/10T/yzn/benchmark_GRN/pre_scgpt/results_multidataset_pseudotime_0326/"
# # percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # 百分比！

# # # 自动读取数据集，排除 mDC
# # csv_files = sorted(glob.glob(os.path.join(base_path, "*_gene_result.csv")))
# # dataset_names = [os.path.basename(f).replace("_gene_result.csv", "") for f in csv_files]
# # filtered = [(f, ds) for f, ds in zip(csv_files, dataset_names) if ds.lower() != "mdc"]
# # csv_files, dataset_names = zip(*filtered)

# # # 与 fig4 其他脚本统一：使用 fig4_palette
# # colors = [
# #     model_color("scGPT"),
# #     model_color("Geneformer"),
# #     model_color("GENIE3"),
# #     model_color("LangCell"),
# #     model_color("scCello"),
# #     model_color("scFoundation"),
# #     model_color("scPrint"),
# #     model_color("STRING"),
# # ]

# # # ===================== 按百分比计算准确率 =====================
# # results = {}
# # for file, ds_name in zip(csv_files, dataset_names):
# #     df = pd.read_csv(file)
# #     total_genes = len(df)
    
# #     # 按预测置信度从高到低排序
# #     df_sorted = df.assign(abs_pred=df["delta_pred"].abs()).sort_values("abs_pred", ascending=False).reset_index(drop=True)
    
# #     accs = []
# #     for p in percentiles:
# #         top_n = max(1, int(total_genes * p / 100))  # 按百分比取数量
# #         acc = df_sorted.head(top_n)["dir_correct"].mean()
# #         accs.append(acc)
# #     results[ds_name] = accs

# # # ===================== 绘图 =====================
# # plt.figure(figsize=FIG4_FIGSIZE)
# # for i, ds in enumerate(results.keys()):
# #     plt.plot(
# #         percentiles, results[ds],
# #         color=colors[i % len(colors)],
# #         marker='o', linestyle='-', linewidth=2.5,
# #         label=ds
# #     )

# # # 坐标轴设置（精细间隔）
# # plt.xlabel("Top Percentage of Genes (%)", fontsize=16)
# # plt.ylabel("Direction Prediction Accuracy", fontsize=16)
# # plt.xticks([10, 30, 50, 70, 90])
# # plt.ylim(0.5, 1.0)
# # plt.yticks(np.arange(0.5, 1.01, 0.1))  # 仅保留 0.5~1 的 0.1 刻度
# # plt.tick_params(axis="both", length=0, labelsize=14)

# # # y-axis tick label formatting: show 1.0 as 1
# # # plt.gca().yaxis.set_major_formatter(
# # #     FuncFormatter(lambda y, _: f"{y:.1f}".rstrip("0").rstrip("."))
# # # )
# # # no background grid
# # plt.grid(False)
# # plt.legend(frameon=False, ncol=2)
# # plt.tight_layout()

# # # 保存路径：统一到 fig4/accuracy，文件名包含数据集名称
# # script_dir = os.path.dirname(os.path.abspath(__file__))
# # out_dir = os.path.join(script_dir, "accuracy")
# # os.makedirs(out_dir, exist_ok=True)
# # dataset_tag = "_".join(results.keys()) if len(results) > 0 else "no_dataset"
# # save_path = os.path.join(out_dir, f"top_percentage_{dataset_tag}.pdf")
# # plt.savefig(save_path, bbox_inches='tight')
# # plt.close()

# # # ===================== 输出结果 =====================
# # print("✅ 绘图完成！已按百分比绘制（10%~100%）")
# # print("💾 图片保存至：", save_path)
# # print("\n📊 各数据集百分比准确率：")
# # for ds, accs in results.items():
# #     print(f"{ds:12s}: {[round(a,3) for a in accs]}")


# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import glob
# from fig4_palette import apply_fig4_style, model_color, FIG4_FIGSIZE
# from matplotlib.ticker import FuncFormatter

# apply_fig4_style()

# # ===================== 路径与百分比设置 =====================
# base_path = "/mnt/10T/yzn/benchmark_GRN/pre_scgpt/results_multidataset_pseudotime_0326/"
# percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# # 自动读取数据集，排除 mDC
# csv_files = sorted(glob.glob(os.path.join(base_path, "*_gene_result.csv")))
# dataset_names = [os.path.basename(f).replace("_gene_result.csv", "") for f in csv_files]
# filtered = [(f, ds) for f, ds in zip(csv_files, dataset_names) if ds.lower() != "mdc"]
# csv_files, dataset_names = zip(*filtered)

# # 使用 fig4_palette 颜色
# colors = [
#     model_color("scGPT"),
#     model_color("Geneformer"),
#     model_color("GENIE3"),
#     model_color("LangCell"),
#     model_color("scCello"),
#     model_color("scFoundation"),
#     model_color("scPrint"),
#     model_color("STRING"),
# ]

# # ===================== 按百分比计算准确率 =====================
# results = {}
# for file, ds_name in zip(csv_files, dataset_names):
#     df = pd.read_csv(file)
#     total_genes = len(df)
#     df_sorted = df.assign(abs_pred=df["delta_pred"].abs()).sort_values("abs_pred", ascending=False).reset_index(drop=True)
    
#     accs = []
#     for p in percentiles:
#         top_n = max(1, int(total_genes * p / 100))
#         acc = df_sorted.head(top_n)["dir_correct"].mean()
#         accs.append(acc)
#     results[ds_name] = accs

# # ===================== 美化绘图 =====================
# fig, ax = plt.subplots(figsize=(9, 6), dpi=600)

# # 使用更美观的线条样式
# line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
# markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

# for i, ds in enumerate(results.keys()):
#     ax.plot(
#         percentiles, 
#         [acc * 100 for acc in results[ds]],  # 转换为百分比显示
#         color=colors[i % len(colors)],
#         marker=markers[i % len(markers)],
#         linestyle=line_styles[i % len(line_styles)],
#         linewidth=2,
#         markersize=6,
#         markerfacecolor='white',
#         markeredgewidth=1.5,
#         label=ds,
#         alpha=0.85
#     )

# # 参考线：随机基线 (50%)
# ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)

# # 坐标轴设置
# ax.set_xlabel("Top Percentage of Genes (%)", fontsize=14, fontweight='medium')
# ax.set_ylabel("Direction Prediction Accuracy (%)", fontsize=14, fontweight='medium')
# ax.set_xticks([10, 30, 50, 70, 90])
# ax.set_xticklabels(['10%', '30%', '50%', '70%', '90%'], fontsize=11)
# ax.set_ylim(45, 100)
# ax.set_yticks(np.arange(50, 101, 10))
# ax.set_yticklabels([f'{y}%' for y in np.arange(50, 101, 10)], fontsize=11)

# # 美化细节
# ax.tick_params(axis='both', length=0, labelsize=11)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_linewidth(0.8)
# ax.spines['bottom'].set_linewidth(0.8)

# # 去掉网格线
# ax.grid(False)

# # 图例 - 放在右侧外部，一列布局
# legend = ax.legend(
#     frameon=False,
#     ncol=1,  # 一列
#     loc='center left',
#     bbox_to_anchor=(1.02, 0.5),
#     fontsize=10,
#     handlelength=2,
#     handletextpad=0.8
# )

# # 调整布局，为右侧图例留出空间
# plt.subplots_adjust(right=0.75)

# # 保存
# script_dir = os.path.dirname(os.path.abspath(__file__))
# out_dir = os.path.join(script_dir, "accuracy")
# os.makedirs(out_dir, exist_ok=True)
# dataset_tag = "_".join(results.keys()) if len(results) > 0 else "no_dataset"
# save_path = os.path.join(out_dir, f"top_percentage_{dataset_tag}_beautiful.pdf")
# plt.savefig(save_path, bbox_inches='tight', dpi=600)
# plt.close()

# # ===================== 输出结果 =====================
# print("✅ 美化版绘图完成！")
# print("💾 图片保存至：", save_path)
# print("\n📊 各数据集百分比准确率（转换为百分比显示）：")
# for ds, accs in results.items():
#     print(f"{ds:12s}: {[round(a*100, 1) for a in accs]}")





import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from fig4_palette import apply_fig4_style, model_color, FIG4_FIGSIZE
from matplotlib.ticker import FuncFormatter

apply_fig4_style()

# 对齐 plot_iter_convergence_single_dataset.py 的样式参数
AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 14
AXIS_LINEWIDTH = 1.2

# ===================== 路径与百分比设置 =====================
base_path = "/mnt/10T/yzn/benchmark_GRN/pre_scgpt/results_multidataset_pseudotime_0326/"
percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# 自动读取数据集，排除 mDC
csv_files = sorted(glob.glob(os.path.join(base_path, "*_gene_result.csv")))
dataset_names = [os.path.basename(f).replace("_gene_result.csv", "") for f in csv_files]
filtered = [(f, ds) for f, ds in zip(csv_files, dataset_names) if ds.lower() != "mdc"]
csv_files, dataset_names = zip(*filtered)

# 使用 fig4_palette 颜色
colors = [
    model_color("scGPT"),
    model_color("Geneformer"),
    model_color("GENIE3"),
    model_color("LangCell"),
    model_color("scCello"),
    model_color("scFoundation"),
    model_color("scPrint"),
    model_color("STRING"),
]

# ===================== 按百分比计算准确率 =====================
results = {}
for file, ds_name in zip(csv_files, dataset_names):
    df = pd.read_csv(file)
    total_genes = len(df)
    df_sorted = df.assign(abs_pred=df["delta_pred"].abs()).sort_values("abs_pred", ascending=False).reset_index(drop=True)
    
    accs = []
    for p in percentiles:
        top_n = max(1, int(total_genes * p / 100))
        acc = df_sorted.head(top_n)["dir_correct"].mean()
        accs.append(acc)
    results[ds_name] = accs

# ===================== 美化绘图 =====================
fig, ax = plt.subplots(figsize=(5.0, 5.0))

# 统一使用圆圈标记，只有颜色不同
for i, ds in enumerate(results.keys()):
    ax.plot(
        percentiles, 
        [acc * 100 for acc in results[ds]],  # 转换为百分比显示
        color=colors[i % len(colors)],
        marker="o",
        linestyle="-",
        linewidth=2.4,
        markersize=8,
        label=ds,
        alpha=0.85
    )



# 坐标轴设置
ax.set_xlabel("Top Percentage of Genes", fontsize=AXIS_LABEL_SIZE)
ax.set_ylabel("Direction accuracy (Top-30)", fontsize=AXIS_LABEL_SIZE)
ax.set_xticks([10, 30, 50, 70, 90])
ax.set_xticklabels(['10%', '30%', '50%', '70%', '90%'], fontsize=12)
ax.set_ylim(45, 100)
ax.set_yticks(np.arange(50, 101, 10))
ax.set_yticklabels([f"{y}" for y in np.arange(50, 101, 10)], fontsize=12)

# 美化细节
ax.grid(False)

# Keep only left/bottom spines to match fig4 style
for side in ["bottom", "left"]:
    ax.spines[side].set_visible(True)
    ax.spines[side].set_linewidth(AXIS_LINEWIDTH)
    ax.spines[side].set_color("black")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(width=AXIS_LINEWIDTH, length=0, labelsize=TICK_LABEL_SIZE, pad=2)

# legend（对齐 plot_iter_convergence_single_dataset.py）
ax.legend(
    loc="lower right",
    frameon=False,
    ncol=2,
    fontsize=14,
    handlelength=1.6,
    handletextpad=0.5,
    labelspacing=0.3,
    borderaxespad=0.3,
    columnspacing=0.8,
)

# 保存
script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, "accuracy")
os.makedirs(out_dir, exist_ok=True)
dataset_tag = "_".join(results.keys()) if len(results) > 0 else "no_dataset"
save_path = os.path.join(out_dir, f"top_percentage_{dataset_tag}_beautiful.pdf")
plt.savefig(save_path, bbox_inches='tight', dpi=600)
plt.close()

# ===================== 输出结果 =====================
print("✅ 美化版绘图完成！")
print("💾 图片保存至：", save_path)
print("\n📊 各数据集百分比准确率（转换为百分比显示）：")
for ds, accs in results.items():
    print(f"{ds:12s}: {[round(a*100, 1) for a in accs]}")