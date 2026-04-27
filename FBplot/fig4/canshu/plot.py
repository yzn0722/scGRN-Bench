import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 1. 设置样式 ----------------------
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 2. 读取数据 ----------------------
# LangCell 数据
lc_path = "/mnt/10T/yzn/FoundBench/FBplot/fig4/canshu/results_langcell_swap_sensitivity/langcell_swap_sensitivity_results_hESC.csv"
df_lc = pd.read_csv(lc_path)

# Geneformer 数据
gf_path = "/mnt/10T/yzn/FoundBench/FBplot/fig4/canshu/results_geneformer_swap_sensitivity/geneformer_swap_sensitivity_results_hESC.csv"
df_gf = pd.read_csv(gf_path)

# 查看数据形状
print(f"LangCell 数据形状: {df_lc.shape}")
print(f"Geneformer 数据形状: {df_gf.shape}")
print(f"LangCell swap_trials: {df_lc['swap_trials'].tolist()}")
print(f"Geneformer swap_trials: {df_gf['swap_trials'].tolist()}")

# ---------------------- 3. 对齐数据（取交集） ----------------------
# 找到两个数据集共有的 swap_trials
common_swap = sorted(set(df_lc['swap_trials']) & set(df_gf['swap_trials']))
print(f"共同的 swap_trials: {common_swap}")

# 筛选出共有的数据
df_lc_common = df_lc[df_lc['swap_trials'].isin(common_swap)].sort_values('swap_trials')
df_gf_common = df_gf[df_gf['swap_trials'].isin(common_swap)].sort_values('swap_trials')

# 重新提取
swap_trials = df_lc_common['swap_trials'].values
lc_acc = df_lc_common['final_accuracy'].values
lc_time = df_lc_common['total_time'].values
gf_acc = df_gf_common['final_accuracy'].values
gf_time = df_gf_common['total_time'].values

# ---------------------- 4. 创建图形（6x6） ----------------------
fig, ax1 = plt.subplots(figsize=(5, 5))

# 颜色定义
COLOR_LC = '#F1C40F'   # LangCell 黄色
COLOR_GF = '#9B59B6'   # Geneformer 紫色

# ---------------------- 5. 左轴：准确率 —— 柱状图 ----------------------
bar_width = 0.35
x_pos = np.arange(len(swap_trials))  # 柱状图位置

# LangCell 准确率柱状
bars_lc = ax1.bar(x_pos - bar_width/2, lc_acc, 
                  width=bar_width, color=COLOR_LC, alpha=0.7, label='LangCell Accuracy')

# Geneformer 准确率柱状
bars_gf = ax1.bar(x_pos + bar_width/2, gf_acc, 
                  width=bar_width, color=COLOR_GF, alpha=0.7, label='Geneformer Accuracy')

# 设置x轴为对数刻度（2的幂）
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'$2^{{{int(np.log2(x))}}}$' for x in swap_trials], fontsize=14)
ax1.set_xlabel('Swap Trials', fontsize=16)
ax1.set_ylabel('Final Accuracy', fontsize=16, color='black')
ax1.set_ylim(0, 1.1)  # 准确率范围0~1
ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
ax1.tick_params(axis='both', length=0)
ax1.tick_params(axis='both', labelsize=14)
ax1.spines['top'].set_visible(True)
ax1.spines['right'].set_visible(True)

# ---------------------- 6. 右轴：时间 —— 折线图（虚线） ----------------------
ax2 = ax1.twinx()

# LangCell 时间虚线
line_lc_time, = ax2.plot(x_pos, lc_time, 
                         color=COLOR_LC, linestyle='--', marker='o', markersize=6, 
                         linewidth=2, label='LangCell Time')

# Geneformer 时间虚线
line_gf_time, = ax2.plot(x_pos, gf_time, 
                         color=COLOR_GF, linestyle='--', marker='s', markersize=6, 
                         linewidth=2, label='Geneformer Time')

ax2.set_ylabel('Time (seconds)', fontsize=16, color='black')
max_time = max(lc_time.max(), gf_time.max())
ax2.set_ylim(0, max_time * 1.1)
ax2.tick_params(axis='both', length=0)
ax2.tick_params(axis='both', labelsize=14)
ax2.spines['top'].set_visible(True)
ax2.spines['right'].set_visible(True)

# 在时间点上添加数值标签
for i, (x, y) in enumerate(zip(x_pos, lc_time)):
    ax2.text(x, y + max_time*0.02, f'{y:.1f}', 
             ha='center', va='bottom', fontsize=14, color=COLOR_LC)

for i, (x, y) in enumerate(zip(x_pos, gf_time)):
    ax2.text(x, y + max_time*0.02, f'{y:.1f}', 
             ha='center', va='bottom', fontsize=14, color=COLOR_GF)

# ---------------------- 7. 合并图例（右下角） ----------------------
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# 创建自定义图例元素
legend_elements = [
    Patch(facecolor=COLOR_LC, alpha=0.7, label='LangCell Accuracy'),
    Patch(facecolor=COLOR_GF, alpha=0.7, label='Geneformer Accuracy'),
    Line2D([0], [0], color=COLOR_LC, linestyle='--', marker='o', 
           label='LangCell Time', linewidth=2),
    Line2D([0], [0], color=COLOR_GF, linestyle='--', marker='s', 
           label='Geneformer Time', linewidth=2)
]

# 添加图例（底部两行）
ax1.legend(
    handles=legend_elements,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.14),
    fontsize=14,
    frameon=False,
    ncol=2,
    columnspacing=1.0
)

# ---------------------- 8. 调整布局 & 保存 ----------------------
plt.subplots_adjust(bottom=0.22)
plt.tight_layout()

# 保存图形
output_path = '/mnt/10T/yzn/FoundBench/FBplot/fig4/canshu/swap_sensitivity_combined_bar_line.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n图形已保存到: {output_path}")

plt.show()