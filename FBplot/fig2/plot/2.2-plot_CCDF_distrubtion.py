
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# 基因调控网络度分布可视化 - Nature风格 CCDF版 (模型 vs STRING真实网络)
# 说明:
# - 使用互补累积分布函数 (CCDF) 替代简单的频率分布，以减少尾部噪声并更清晰展示幂律特征
# - 叠加 STRING 真实网络（*_processed-network.csv）的入/出度 CCDF 曲线用于对比
# - 关键修改：
#   1. 模型文件仅保留真实网络中存在的Gene1
#   2. 模型文件仅保留真实网络中Gene1/Gene2并集中的Gene2
#   3. 模型文件选取的边数与对应真实网络边数一致
# """

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from pathlib import Path
# import argparse
# import warnings
# warnings.filterwarnings('ignore')

# # Unified fig2 palette (fixed model colors)
# from fig2_palette import model_color

# # ==================== Nature 风格全局设置 ====================
# mpl.rcParams.update({
#     'font.family': 'DejaVu Sans',
#     'font.sans-serif': ['DejaVu Sans'],
#     'font.size': 16,
#     'axes.labelsize': 16,
#     'axes.titlesize': 16,
#     'xtick.labelsize': 16,
#     'ytick.labelsize': 16,
#     'legend.fontsize': 14,
#     'axes.linewidth': 0.8,
#     'axes.edgecolor': 'black',
#     'axes.grid': False,
#     'xtick.direction': 'out',
#     'ytick.direction': 'out',
#     'lines.linewidth': 2,
#     'lines.markersize': 5,
#     'legend.frameon': False,
#     'figure.dpi': 300,
#     'savefig.bbox': 'tight',
#     'savefig.pad_inches': 0.05
# })

# # ==================== 配置 ====================
# # 模型预测网络目录（可由 --extraction 指定）
# EVL_ROOT = Path("/mnt/10T/yzn/benchmark_GRN/evl_omipath")

# # 真实网络（STRING）目录：包含 *_processed-network.csv
# REAL_DATA_DIR = Path("/mnt/10T/yzn/benchmark_GRN/input_process/STRING")

# FIG_SIZE = (6, 6)

# MODELS = {
#     'Geneformer':   {'dir': 'geneformer',   'prefix': 'geneformer_',   'color': model_color('Geneformer'),   'marker': 'o'},
#     'GENIE3':       {'dir': 'GENIE3',       'prefix': 'GENIE3_',       'color': model_color('GENIE3'),       'marker': 's'},
#     'LangCell':     {'dir': 'langcell',     'prefix': 'LangCell_',     'color': model_color('LangCell'),     'marker': '^'},
#     'scGPT':        {'dir': 'scgpt',        'prefix': 'scgpt_',        'color': model_color('scGPT'),        'marker': 'D'},
#     'scCello':      {'dir': 'sccello',      'prefix': 'scCello_',      'color': model_color('scCello'),      'marker': 'h'},
#     'scFoundation': {'dir': 'scFoundation', 'prefix': 'scFoundation_', 'color': model_color('scFoundation'), 'marker': 'v'},
#     'scPrint':    {'dir': 'scprint',      'prefix': 'scprint_',      'color': model_color('scPrint'), 'marker': 'p'},
# }

# DATASETS = ['hESC',"hHep",'mDC','mHSC-L','mHSC-M', 'mHSC-GM']
# #DATASETS = ['hESC']

# FILTER_STRATEGY = 'top_n'
# DEFAULT_TOP_N = 10000  # 真实网络不存在时的兜底值
# DEFAULT_EXTRACTION = "emb500"
# VALID_EXTRACTIONS = ("emb500", "att500", "embhidden500")
# EXTRACTION_DISPLAY = {
#     "emb500": r"cos$_{tok}$",
#     "att500": "attn",
#     "embhidden500": r"cos$_{hid}$",
# }

# # 真实网络画图风格
# REAL_STYLE = {
#     "label": "STRING",
#     "color": model_color("STRING"),
#     "linestyle": "--",
#     "linewidth": 2.8,
#     "alpha": 0.95
# }
# TAIL_K_THRESHOLD = 10

# # ==================== 函数 ====================

# def detect_weight_column(df: pd.DataFrame):
#     """自动检测权重列名"""
#     possible_names = ['EdgeWeight', 'edgeweight', 'edge_weight', 'Attention score', 'Weight', 'Score', 'Importance']
#     for name in possible_names:
#         if name in df.columns:
#             return name
#     for col in df.columns:
#         if any(k in col.lower() for k in ['weight', 'score', 'conf', 'import']):
#             return col
#     return None

# def load_model_data(model_name: str, dataset: str, real_gene1_set: set = None, 
#                    real_gene_union_set: set = None, target_edge_count: int = DEFAULT_TOP_N):
#     """
#     加载并过滤模型预测网络
#     修改点：
#     1. 新增 real_gene1_set 参数：过滤Gene1，仅保留该集合中的基因
#     2. 新增 real_gene_union_set 参数：过滤Gene2，仅保留该集合中的基因
#     3. 新增 target_edge_count 参数：动态指定要选取的top N条边（匹配真实网络边数）
#     """
#     config = MODELS[model_name]
#     filepath = DATA_DIR / config['dir'] / f"{config['prefix']}{dataset}.tsv"
#     if not filepath.exists():
#         return None

#     try:
#         df = pd.read_csv(filepath, sep='\t')
#         if 'Gene1' not in df.columns or 'Gene2' not in df.columns:
#             return None
        
#         # 记录原始数据量
#         original_count = len(df)

#         # 关键修改1：过滤Gene1，仅保留真实网络中存在的Gene1
#         if real_gene1_set is not None and len(real_gene1_set) > 0:
#             df = df[df['Gene1'].isin(real_gene1_set)]
#             gene1_filtered_count = len(df)
#             print(f"    {model_name}: Gene1过滤后剩余 {gene1_filtered_count}/{original_count} 条边")
        
#         # 关键修改2：过滤Gene2，仅保留真实网络Gene1+Gene2并集中的基因
#         if real_gene_union_set is not None and len(real_gene_union_set) > 0:
#             df = df[df['Gene2'].isin(real_gene_union_set)]
#             gene2_filtered_count = len(df)
#             print(f"    {model_name}: Gene2过滤后剩余 {gene2_filtered_count}/{gene1_filtered_count if 'gene1_filtered_count' in locals() else original_count} 条边")

#         if len(df) == 0:
#             print(f"    {model_name} - {dataset}: 过滤后无数据")
#             return None

#         # 关键修改3：根据目标边数选取top N
#         weight_col = detect_weight_column(df)
#         if weight_col and FILTER_STRATEGY == 'top_n':
#             # 确保选取的边数不超过数据总行数
#             actual_top_n = min(target_edge_count, len(df))
#             df = df.nlargest(actual_top_n, weight_col)
#             if actual_top_n < target_edge_count:
#                 print(f"    {model_name} - {dataset}: 数据量不足，仅选取{actual_top_n}条边（目标{target_edge_count}）")

#         return df[['Gene1', 'Gene2']].dropna()
#     except Exception as e:
#         print(f"    ❌ {model_name} - {dataset}: 加载失败 {str(e)}")
#         return None

# def load_real_data(dataset: str):
#     """
#     精确加载 STRING 真实网络：
#     /mnt/10T/yzn/benchmark_GRN/input_process/STRING/{dataset}_processed-network.csv
#     自动映射列名到 Gene1/Gene2；若找不到匹配列，则退化为取前两列。
#     返回：(real_df, gene1_set, gene_union_set, edge_count)
#     """
#     fp = REAL_DATA_DIR / f"{dataset}_processed-network.csv"
#     if not fp.exists():
#         return None, set(), set(), 0

#     try:
#         df = pd.read_csv(fp)
#         cols_lower = {c.lower(): c for c in df.columns}

#         gene1_candidates = ["gene1", "source", "tf", "regulator", "from",
#                             "protein1", "preferredname_a", "node1", "a"]
#         gene2_candidates = ["gene2", "target", "to",
#                             "protein2", "preferredname_b", "node2", "b"]

#         def pick_col(cands):
#             for k in cands:
#                 if k in cols_lower:
#                     return cols_lower[k]
#             return None

#         c1 = pick_col(gene1_candidates)
#         c2 = pick_col(gene2_candidates)

#         if c1 is None or c2 is None:
#             # 退化策略：取前两列当作边表
#             if df.shape[1] < 2:
#                 return None, set(), set(), 0
#             c1, c2 = df.columns[:2]

#         real_df = df[[c1, c2]].rename(columns={c1: "Gene1", c2: "Gene2"}).dropna()
#         # 获取真实网络的Gene1集合、Gene1+Gene2并集、边数
#         gene1_set = set(real_df['Gene1'].unique())
#         gene2_set = set(real_df['Gene2'].unique())
#         gene_union_set = gene1_set.union(gene2_set)  # Gene1和Gene2的并集
#         edge_count = len(real_df)
        
#         print(f"  ✔ 真实网络 {dataset}: ")
#         print(f"     - Gene1数量={len(gene1_set)}")
#         print(f"     - Gene2数量={len(gene2_set)}")
#         print(f"     - 总节点数(Gene1+Gene2并集)={len(gene_union_set)}")
#         print(f"     - 边数={edge_count}")
#         return real_df, gene1_set, gene_union_set, edge_count
#     except Exception as e:
#         print(f"  ❌ 加载真实网络 {dataset} 失败: {str(e)}")
#         return None, set(), set(), 0

# def get_ccdf(data_series: pd.Series):
#     """
#     计算互补累积分布函数 (CCDF): P(X >= k)
#     输入: 每个节点的度（value_counts结果）
#     输出: (unique_degrees, ccdf_probs)
#     """
#     if data_series is None or len(data_series) == 0:
#         return None, None

#     data = np.sort(data_series.values)
#     n = len(data)
#     unique_degrees = np.unique(data)
#     indices = np.searchsorted(data, unique_degrees, side='left')
#     ccdf_counts = n - indices
#     ccdf_probs = ccdf_counts / n
#     return unique_degrees, ccdf_probs


# def format_label_with_tail_prob(base_label: str, degree_series: pd.Series, k_threshold: int = TAIL_K_THRESHOLD) -> str:
#     """在图例标签后追加 CCDF 尾部概率 P(K>=k)。"""
#     display_label = "scPRINT" if base_label == "scPrint" else base_label
#     if degree_series is None or len(degree_series) == 0:
#         return f"{display_label} (NA)"
#     tail_prob = float((degree_series.values >= int(k_threshold)).mean())
#     return f"{display_label} ({tail_prob:.3f})"

# def _to_undirected_degree_series(df: pd.DataFrame):
#     """Convert directed edge table to undirected simple graph degree series."""
#     if df is None or len(df) == 0:
#         return None
#     d = df[["Gene1", "Gene2"]].dropna().copy()
#     d["A"] = d["Gene1"].astype(str).str.strip()
#     d["B"] = d["Gene2"].astype(str).str.strip()
#     d = d[(d["A"] != "") & (d["B"] != "") & (d["A"] != d["B"])]
#     if d.empty:
#         return None
#     d["U"] = np.minimum(d["A"], d["B"])
#     d["V"] = np.maximum(d["A"], d["B"])
#     d = d.drop_duplicates(subset=["U", "V"])
#     deg = pd.concat([d["U"], d["V"]], axis=0).value_counts()
#     return deg


# def plot_ccdf_distribution(
#     dataset: str,
#     data_dict: dict,
#     real_df: pd.DataFrame = None,
#     undirected: bool = False,
#     extraction: str = DEFAULT_EXTRACTION,
# ):
#     """绘制 CCDF 图 (Log-Log)：模型 vs 真实网络（出度/入度分开两张图，统一圆形 marker）"""
#     if not data_dict and real_df is None:
#         return None

#     if undirected:
#         fig_deg, ax_deg = plt.subplots(1, 1, figsize=FIG_SIZE)
#         has_deg = False
#         title_text = EXTRACTION_DISPLAY.get(extraction, extraction)

#         for model_name, df in data_dict.items():
#             model_deg = _to_undirected_degree_series(df)
#             x, y = get_ccdf(model_deg) if model_deg is not None else (None, None)
#             if x is not None and len(x) > 0:
#                 has_deg = True
#                 cfg = MODELS[model_name]
#                 z = 5 if model_name == "scGPT" else 3
#                 ax_deg.loglog(
#                     x, y,
#                     marker="o",
#                     color=cfg['color'],
#                     label=format_label_with_tail_prob(model_name, model_deg),
#                     linestyle='-',
#                     alpha=0.95,
#                     linewidth=2.2,
#                     markersize=4.8,
#                     markerfacecolor=cfg["color"],
#                     markeredgecolor=model_color("STRING"),
#                     markeredgewidth=0.6,
#                     zorder=z,
#                 )

#         # Draw STRING last to keep it always on top.
#         if real_df is not None and len(real_df) > 0:
#             real_deg = _to_undirected_degree_series(real_df)
#             xr, yr = get_ccdf(real_deg) if real_deg is not None else (None, None)
#             if xr is not None and len(xr) > 0:
#                 ax_deg.loglog(
#                     xr, yr,
#                     label=format_label_with_tail_prob(REAL_STYLE["label"], real_deg),
#                     color=REAL_STYLE["color"],
#                     linestyle=REAL_STYLE["linestyle"],
#                     linewidth=REAL_STYLE["linewidth"],
#                     alpha=REAL_STYLE["alpha"],
#                     marker=None,
#                     zorder=10,
#                 )
#                 has_deg = True

#         if has_deg:
#             ax_deg.set_xlabel('Degree ($k$)')
#             ax_deg.set_ylabel(r'CCDF P($K \geq k$)')
#             ax_deg.legend(loc='best', frameon=False)
#         ax_deg.set_title(title_text, fontsize=16, pad=10)
#         ax_deg.spines['left'].set_visible(True)
#         ax_deg.spines['top'].set_visible(False)
#         ax_deg.spines['right'].set_visible(False)
#         ax_deg.spines['bottom'].set_color('black')
#         ax_deg.spines['left'].set_color('black')
#         ax_deg.tick_params(axis='both', colors='black')
#         ax_deg.xaxis.label.set_color('black')
#         ax_deg.yaxis.label.set_color('black')
#         fig_deg.tight_layout()

#         out_file_deg = OUTPUT_DIR / f"{dataset}_CCDF_degree_undirected_models_vs_STRING.pdf"
#         fig_deg.savefig(out_file_deg)
#         plt.close(fig_deg)
#         print(f"  ✓ 生成: {out_file_deg.name}")
#         return out_file_deg

#     # ========== 出度分布 (CCDF) ==========
#     fig_out, ax_out = plt.subplots(1, 1, figsize=FIG_SIZE)
#     has_out = False
#     title_text = EXTRACTION_DISPLAY.get(extraction, extraction)

#     # 先画各模型
#     for model_name, df in data_dict.items():
#         out_degrees = df['Gene1'].value_counts()
#         x, y = get_ccdf(out_degrees)

#         if x is not None and len(x) > 0:
#             has_out = True
#             cfg = MODELS[model_name]
#             z = 5 if model_name == "scGPT" else 3
#             ax_out.loglog(
#                 x, y,
#                 marker="o",
#                 color=cfg['color'],
#                 label=format_label_with_tail_prob(model_name, out_degrees),
#                 linestyle='-',
#                 alpha=0.95,
#                 linewidth=2.2,
#                 markersize=4.8,
#                 markerfacecolor=cfg["color"],
#                 markeredgecolor=model_color("STRING"),
#                 markeredgewidth=0.6,
#                 zorder=z,
#             )

#     # 再画真实网络，确保 STRING 虚线在最上层
#     if real_df is not None and len(real_df) > 0:
#         out_deg_real = real_df['Gene1'].value_counts()
#         xr, yr = get_ccdf(out_deg_real)
#         if xr is not None and len(xr) > 0:
#             ax_out.loglog(
#                 xr, yr,
#                 label=format_label_with_tail_prob(REAL_STYLE["label"], out_deg_real),
#                 color=REAL_STYLE["color"],
#                 linestyle=REAL_STYLE["linestyle"],
#                 linewidth=REAL_STYLE["linewidth"],
#                 alpha=REAL_STYLE["alpha"],
#                 marker=None,
#                 zorder=10,
#             )
#             has_out = True

#     if has_out:
#         ax_out.set_xlabel('Out-degree ($k_{out}$)')
#         ax_out.set_ylabel(r'CCDF P($K \geq k_{out}$)')
#         ax_out.set_title(title_text, fontsize=16, pad=10)
#         ax_out.legend(loc='best', frameon=False)
#     ax_out.spines['left'].set_visible(True)
#     ax_out.spines['top'].set_visible(False)
#     ax_out.spines['right'].set_visible(False)
#     ax_out.spines['bottom'].set_color('black')
#     ax_out.spines['left'].set_color('black')
#     ax_out.tick_params(axis='both', colors='black')
#     ax_out.xaxis.label.set_color('black')
#     ax_out.yaxis.label.set_color('black')

#     fig_out.tight_layout()

#     # ========== 入度分布 (CCDF) ==========
#     fig_in, ax_in = plt.subplots(1, 1, figsize=FIG_SIZE)
#     has_in = False

#     # 先画各模型
#     for model_name, df in data_dict.items():
#         in_degrees = df['Gene2'].value_counts()
#         x, y = get_ccdf(in_degrees)

#         if x is not None and len(x) > 0:
#             has_in = True
#             cfg = MODELS[model_name]
#             z = 5 if model_name == "scGPT" else 3
#             ax_in.loglog(
#                 x, y,
#                 marker="o",
#                 color=cfg['color'],
#                 label=format_label_with_tail_prob(model_name, in_degrees),
#                 linestyle='-',
#                 alpha=0.95,
#                 linewidth=2.2,
#                 markersize=4.8,
#                 markerfacecolor=cfg["color"],
#                 markeredgecolor=model_color("STRING"),
#                 markeredgewidth=0.6,
#                 zorder=z,
#             )

#     # 再画真实网络，确保 STRING 虚线在最上层
#     if real_df is not None and len(real_df) > 0:
#         in_deg_real = real_df['Gene2'].value_counts()
#         xr, yr = get_ccdf(in_deg_real)
#         if xr is not None and len(xr) > 0:
#             ax_in.loglog(
#                 xr, yr,
#                 label=format_label_with_tail_prob(REAL_STYLE["label"], in_deg_real),
#                 color=REAL_STYLE["color"],
#                 linestyle=REAL_STYLE["linestyle"],
#                 linewidth=REAL_STYLE["linewidth"],
#                 alpha=REAL_STYLE["alpha"],
#                 marker=None,
#                 zorder=10,
#             )
#             has_in = True

#     if has_in:
#         ax_in.set_xlabel('In-degree ($k_{in}$)')
#         ax_in.set_ylabel(r'CCDF P($K \geq k_{in}$)')
#         ax_in.set_title(title_text, fontsize=16, pad=10)
#         ax_in.legend(loc='best', frameon=False)
#     ax_in.spines['left'].set_visible(True)
#     ax_in.spines['top'].set_visible(False)
#     ax_in.spines['right'].set_visible(False)
#     ax_in.spines['bottom'].set_color('black')
#     ax_in.spines['left'].set_color('black')
#     ax_in.tick_params(axis='both', colors='black')
#     ax_in.xaxis.label.set_color('black')
#     ax_in.yaxis.label.set_color('black')

#     fig_in.tight_layout()

#     out_file_out = OUTPUT_DIR / f"{dataset}_CCDF_outdegree_models_vs_STRING.pdf"
#     fig_out.savefig(out_file_out)
#     plt.close(fig_out)

#     out_file_in = OUTPUT_DIR / f"{dataset}_CCDF_indegree_models_vs_STRING.pdf"
#     fig_in.savefig(out_file_in)
#     plt.close(fig_in)

#     print(f"  ✓ 生成: {out_file_out.name}")
#     print(f"  ✓ 生成: {out_file_in.name}")
#     return out_file_out

# # ==================== 主程序 ====================

# def main():
#     parser = argparse.ArgumentParser(description="Plot CCDF degree distributions for selected extraction.")
#     parser.add_argument(
#         "--extraction",
#         type=str,
#         default=DEFAULT_EXTRACTION,
#         choices=VALID_EXTRACTIONS,
#         help="Extraction subdir under evl_omipath (e.g., emb500, att500, embhidden500).",
#     )
#     parser.add_argument(
#         "--undirected",
#         action="store_true",
#         help="Treat edges as undirected and plot a single degree CCDF.",
#     )
#     args = parser.parse_args()

#     data_dir = EVL_ROOT / f"output_{args.extraction}"
#     output_dir = Path(__file__).resolve().parent / f"ccdf_{args.extraction}"
#     output_dir.mkdir(exist_ok=True)

#     global DATA_DIR, OUTPUT_DIR
#     DATA_DIR = data_dir
#     OUTPUT_DIR = output_dir

#     print(f"正在生成 CCDF 图表，输出目录: {OUTPUT_DIR}")
#     if not DATA_DIR.exists():
#         print(f"模型目录不存在: {DATA_DIR}")
#         return

#     if not REAL_DATA_DIR.exists():
#         print(f"真实网络目录不存在: {REAL_DATA_DIR}")
#         print("将使用默认边数（10000），且不过滤Gene1/Gene2。")

#     for dataset in DATASETS:
#         print(f"\n处理: {dataset} ...")

#         # 关键修改：先加载真实网络数据，获取过滤依据
#         real_df, real_gene1_set, real_gene_union_set, real_edge_count = load_real_data(dataset)
        
#         if real_df is None:
#             print(f"  ! 未找到/无法解析真实网络: {REAL_DATA_DIR}/{dataset}_processed-network.csv")
#             print(f"  ! 将使用默认配置：边数={DEFAULT_TOP_N}，不过滤Gene1/Gene2")
#             # 兜底：不过滤Gene1/Gene2，使用默认边数
#             real_gene1_set = None
#             real_gene_union_set = None
#             target_edge_count = DEFAULT_TOP_N
#         else:
#             target_edge_count = real_edge_count
#             print(f"  ✔ 目标边数: {target_edge_count} (匹配真实网络)")

#         # 加载模型数据（传入过滤条件）
#         data_dict = {}
#         for name in MODELS:
#             print(f"    处理模型: {name}")
#             df = load_model_data(
#                 model_name=name,
#                 dataset=dataset,
#                 real_gene1_set=real_gene1_set,
#                 real_gene_union_set=real_gene_union_set,
#                 target_edge_count=target_edge_count
#             )
#             if df is not None and len(df) > 0:
#                 data_dict[name] = df

#         # 画图
#         plot_ccdf_distribution(
#             dataset,
#             data_dict,
#             real_df=real_df,
#             undirected=args.undirected,
#             extraction=args.extraction,
#         )

#     print("\n完成。")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基因调控网络度分布可视化 - Nature风格 CCDF版 (模型 vs STRING真实网络)
说明:
- 使用互补累积分布函数 (CCDF) 替代简单的频率分布，以减少尾部噪声并更清晰展示幂律特征
- 叠加 STRING 真实网络（*_processed-network.csv）的入/出度 CCDF 曲线用于对比
- 关键修改：
  1. 模型文件仅保留真实网络中存在的Gene1
  2. 模型文件仅保留真实网络中Gene1/Gene2并集中的Gene2
  3. 模型文件选取的边数与对应真实网络边数一致
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

# Unified fig2 palette (fixed model colors)
from fig2_palette import model_color

# ==================== Nature 风格全局设置 ====================
mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 16,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'axes.linewidth': 0.8,
    'axes.edgecolor': 'black',
    'axes.grid': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'lines.linewidth': 2,
    'lines.markersize': 5,
    'legend.frameon': False,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05
})

# ==================== 配置 ====================
# 模型预测网络目录（可由 --extraction 指定）
EVL_ROOT = Path("/mnt/10T/yzn/benchmark_GRN/evl_omipath")

# 真实网络（STRING）目录：包含 *_processed-network.csv
REAL_DATA_DIR = Path("/mnt/10T/yzn/benchmark_GRN/input_process/STRING")

FIG_SIZE = (6, 6)

MODELS = {
    'Geneformer':   {'dir': 'geneformer',   'prefix': 'geneformer_',   'color': model_color('Geneformer'),   'marker': 'o'},
    'GENIE3':       {'dir': 'GENIE3',       'prefix': 'GENIE3_',       'color': model_color('GENIE3'),       'marker': 's'},
    'LangCell':     {'dir': 'langcell',     'prefix': 'LangCell_',     'color': model_color('LangCell'),     'marker': '^'},
    'scGPT':        {'dir': 'scgpt',        'prefix': 'scgpt_',        'color': model_color('scGPT'),        'marker': 'D'},
    'scCello':      {'dir': 'sccello',      'prefix': 'scCello_',      'color': model_color('scCello'),      'marker': 'h'},
    'scFoundation': {'dir': 'scFoundation', 'prefix': 'scFoundation_', 'color': model_color('scFoundation'), 'marker': 'v'},
    'scPrint':    {'dir': 'scprint',      'prefix': 'scprint_',      'color': model_color('scPrint'), 'marker': 'p'},
}

DATASETS = ['hESC',"hHep",'mDC','mHSC-L','mHSC-E', 'mHSC-GM']
#DATASETS = ['hESC']

FILTER_STRATEGY = 'top_n'
DEFAULT_TOP_N = 10000  # 真实网络不存在时的兜底值
DEFAULT_EXTRACTION = "emb500"
VALID_EXTRACTIONS = ("emb500", "att500", "embhidden500")
EXTRACTION_DISPLAY = {
    "emb500": r"cos$_{tok}$",
    "att500": "attn",
    "embhidden500": r"cos$_{hid}$",
}

# 真实网络画图风格
REAL_STYLE = {
    "label": "STRING",
    "color": "black",
    "linestyle": "--",
    "linewidth": 2.8,
    "alpha": 0.95
}
TAIL_K_THRESHOLD = 10

# ==================== 函数 ====================

def detect_weight_column(df: pd.DataFrame):
    """自动检测权重列名"""
    possible_names = ['EdgeWeight', 'edgeweight', 'edge_weight', 'Attention score', 'Weight', 'Score', 'Importance']
    for name in possible_names:
        if name in df.columns:
            return name
    for col in df.columns:
        if any(k in col.lower() for k in ['weight', 'score', 'conf', 'import']):
            return col
    return None

def load_model_data(model_name: str, dataset: str, real_gene1_set: set = None, 
                   real_gene_union_set: set = None, target_edge_count: int = DEFAULT_TOP_N):
    """
    加载并过滤模型预测网络
    修改点：
    1. 新增 real_gene1_set 参数：过滤Gene1，仅保留该集合中的基因
    2. 新增 real_gene_union_set 参数：过滤Gene2，仅保留该集合中的基因
    3. 新增 target_edge_count 参数：动态指定要选取的top N条边（匹配真实网络边数）
    """
    config = MODELS[model_name]
    filepath = DATA_DIR / config['dir'] / f"{config['prefix']}{dataset}.tsv"
    if not filepath.exists():
        return None

    try:
        df = pd.read_csv(filepath, sep='\t')
        if 'Gene1' not in df.columns or 'Gene2' not in df.columns:
            return None
        
        # 记录原始数据量
        original_count = len(df)

        # 关键修改1：过滤Gene1，仅保留真实网络中存在的Gene1
        if real_gene1_set is not None and len(real_gene1_set) > 0:
            df = df[df['Gene1'].isin(real_gene1_set)]
            gene1_filtered_count = len(df)
            print(f"    {model_name}: Gene1过滤后剩余 {gene1_filtered_count}/{original_count} 条边")
        
        # 关键修改2：过滤Gene2，仅保留真实网络Gene1+Gene2并集中的基因
        if real_gene_union_set is not None and len(real_gene_union_set) > 0:
            df = df[df['Gene2'].isin(real_gene_union_set)]
            gene2_filtered_count = len(df)
            print(f"    {model_name}: Gene2过滤后剩余 {gene2_filtered_count}/{gene1_filtered_count if 'gene1_filtered_count' in locals() else original_count} 条边")

        if len(df) == 0:
            print(f"    {model_name} - {dataset}: 过滤后无数据")
            return None

        # 关键修改3：根据目标边数选取top N
        weight_col = detect_weight_column(df)
        if weight_col and FILTER_STRATEGY == 'top_n':
            # 确保选取的边数不超过数据总行数
            actual_top_n = min(target_edge_count, len(df))
            df = df.nlargest(actual_top_n, weight_col)
            if actual_top_n < target_edge_count:
                print(f"    {model_name} - {dataset}: 数据量不足，仅选取{actual_top_n}条边（目标{target_edge_count}）")

        return df[['Gene1', 'Gene2']].dropna()
    except Exception as e:
        print(f"    ❌ {model_name} - {dataset}: 加载失败 {str(e)}")
        return None

def load_real_data(dataset: str):
    """
    精确加载 STRING 真实网络：
    /mnt/10T/yzn/benchmark_GRN/input_process/STRING/{dataset}_processed-network.csv
    自动映射列名到 Gene1/Gene2；若找不到匹配列，则退化为取前两列。
    返回：(real_df, gene1_set, gene_union_set, edge_count)
    """
    fp = REAL_DATA_DIR / f"{dataset}_processed-network.csv"
    if not fp.exists():
        return None, set(), set(), 0

    try:
        df = pd.read_csv(fp)
        cols_lower = {c.lower(): c for c in df.columns}

        gene1_candidates = ["gene1", "source", "tf", "regulator", "from",
                            "protein1", "preferredname_a", "node1", "a"]
        gene2_candidates = ["gene2", "target", "to",
                            "protein2", "preferredname_b", "node2", "b"]

        def pick_col(cands):
            for k in cands:
                if k in cols_lower:
                    return cols_lower[k]
            return None

        c1 = pick_col(gene1_candidates)
        c2 = pick_col(gene2_candidates)

        if c1 is None or c2 is None:
            # 退化策略：取前两列当作边表
            if df.shape[1] < 2:
                return None, set(), set(), 0
            c1, c2 = df.columns[:2]

        real_df = df[[c1, c2]].rename(columns={c1: "Gene1", c2: "Gene2"}).dropna()
        # 获取真实网络的Gene1集合、Gene1+Gene2并集、边数
        gene1_set = set(real_df['Gene1'].unique())
        gene2_set = set(real_df['Gene2'].unique())
        gene_union_set = gene1_set.union(gene2_set)  # Gene1和Gene2的并集
        edge_count = len(real_df)
        
        print(f"  ✔ 真实网络 {dataset}: ")
        print(f"     - Gene1数量={len(gene1_set)}")
        print(f"     - Gene2数量={len(gene2_set)}")
        print(f"     - 总节点数(Gene1+Gene2并集)={len(gene_union_set)}")
        print(f"     - 边数={edge_count}")
        return real_df, gene1_set, gene_union_set, edge_count
    except Exception as e:
        print(f"  ❌ 加载真实网络 {dataset} 失败: {str(e)}")
        return None, set(), set(), 0

def get_ccdf(data_series: pd.Series):
    """
    计算互补累积分布函数 (CCDF): P(X >= k)
    输入: 每个节点的度（value_counts结果）
    输出: (unique_degrees, ccdf_probs)
    """
    if data_series is None or len(data_series) == 0:
        return None, None

    data = np.sort(data_series.values)
    n = len(data)
    unique_degrees = np.unique(data)
    indices = np.searchsorted(data, unique_degrees, side='left')
    ccdf_counts = n - indices
    ccdf_probs = ccdf_counts / n
    return unique_degrees, ccdf_probs


def format_label_with_tail_prob(base_label: str, degree_series: pd.Series, k_threshold: int = TAIL_K_THRESHOLD) -> str:
    """返回模型名称，不显示括号内的概率数字"""
    display_label = "scPRINT" if base_label == "scPrint" else base_label
    return f"{display_label}"

def _to_undirected_degree_series(df: pd.DataFrame):
    """Convert directed edge table to undirected simple graph degree series."""
    if df is None or len(df) == 0:
        return None
    d = df[["Gene1", "Gene2"]].dropna().copy()
    d["A"] = d["Gene1"].astype(str).str.strip()
    d["B"] = d["Gene2"].astype(str).str.strip()
    d = d[(d["A"] != "") & (d["B"] != "") & (d["A"] != d["B"])]
    if d.empty:
        return None
    d["U"] = np.minimum(d["A"], d["B"])
    d["V"] = np.maximum(d["A"], d["B"])
    d = d.drop_duplicates(subset=["U", "V"])
    deg = pd.concat([d["U"], d["V"]], axis=0).value_counts()
    return deg


def plot_ccdf_distribution(
    dataset: str,
    data_dict: dict,
    real_df: pd.DataFrame = None,
    undirected: bool = False,
    extraction: str = DEFAULT_EXTRACTION,
):
    """绘制 CCDF 图 (Log-Log)：模型 vs 真实网络（出度/入度分开两张图，统一圆形 marker）"""
    if not data_dict and real_df is None:
        return None

    if undirected:
        fig_deg, ax_deg = plt.subplots(1, 1, figsize=FIG_SIZE)
        has_deg = False
        title_text = EXTRACTION_DISPLAY.get(extraction, extraction)

        for model_name, df in data_dict.items():
            model_deg = _to_undirected_degree_series(df)
            x, y = get_ccdf(model_deg) if model_deg is not None else (None, None)
            if x is not None and len(x) > 0:
                has_deg = True
                cfg = MODELS[model_name]
                z = 5 if model_name == "scGPT" else 3
                ax_deg.loglog(
                    x, y,
                    marker="o",
                    color=cfg['color'],
                    label=format_label_with_tail_prob(model_name, model_deg),
                    linestyle='-',
                    alpha=0.95,
                    linewidth=2.2,
                    markersize=4.8,
                    markerfacecolor=cfg["color"],
                    markeredgecolor=model_color("STRING"),
                    markeredgewidth=0.6,
                    zorder=z,
                )

        # Draw STRING last to keep it always on top.
        if real_df is not None and len(real_df) > 0:
            real_deg = _to_undirected_degree_series(real_df)
            xr, yr = get_ccdf(real_deg) if real_deg is not None else (None, None)
            if xr is not None and len(xr) > 0:
                ax_deg.loglog(
                    xr, yr,
                    label=format_label_with_tail_prob(REAL_STYLE["label"], real_deg),
                    color=REAL_STYLE["color"],
                    linestyle=REAL_STYLE["linestyle"],
                    linewidth=REAL_STYLE["linewidth"],
                    alpha=REAL_STYLE["alpha"],
                    marker=None,
                    zorder=10,
                )
                has_deg = True

        if has_deg:
            ax_deg.set_xlabel('Degree ($k$)')
            ax_deg.set_ylabel(r'CCDF P($K \geq k$)')
            ax_deg.legend(loc='best', frameon=False)
        ax_deg.set_title(title_text, fontsize=16, pad=10)
        ax_deg.spines['left'].set_visible(True)
        ax_deg.spines['top'].set_visible(False)
        ax_deg.spines['right'].set_visible(False)
        ax_deg.spines['bottom'].set_color('black')
        ax_deg.spines['left'].set_color('black')
        ax_deg.tick_params(axis='both', colors='black')
        ax_deg.xaxis.label.set_color('black')
        ax_deg.yaxis.label.set_color('black')
        fig_deg.tight_layout()

        out_file_deg = OUTPUT_DIR / f"{dataset}_CCDF_degree_undirected_models_vs_STRING.pdf"
        fig_deg.savefig(out_file_deg)
        plt.close(fig_deg)
        print(f"  ✓ 生成: {out_file_deg.name}")
        return out_file_deg

    # ========== 出度分布 (CCDF) ==========
    fig_out, ax_out = plt.subplots(1, 1, figsize=FIG_SIZE)
    has_out = False
    title_text = EXTRACTION_DISPLAY.get(extraction, extraction)

    # 先画各模型
    for model_name, df in data_dict.items():
        out_degrees = df['Gene1'].value_counts()
        x, y = get_ccdf(out_degrees)

        if x is not None and len(x) > 0:
            has_out = True
            cfg = MODELS[model_name]
            z = 5 if model_name == "scGPT" else 3
            ax_out.loglog(
                x, y,
                marker="o",
                color=cfg['color'],
                label=format_label_with_tail_prob(model_name, out_degrees),
                linestyle='-',
                alpha=0.95,
                linewidth=2.2,
                markersize=4.8,
                markerfacecolor=cfg["color"],
                markeredgecolor=model_color("STRING"),
                markeredgewidth=0.6,
                zorder=z,
            )

    # 再画真实网络，确保 STRING 虚线在最上层
    if real_df is not None and len(real_df) > 0:
        out_deg_real = real_df['Gene1'].value_counts()
        xr, yr = get_ccdf(out_deg_real)
        if xr is not None and len(xr) > 0:
            ax_out.loglog(
                xr, yr,
                label=format_label_with_tail_prob(REAL_STYLE["label"], out_deg_real),
                color=REAL_STYLE["color"],
                linestyle=REAL_STYLE["linestyle"],
                linewidth=REAL_STYLE["linewidth"],
                alpha=REAL_STYLE["alpha"],
                marker=None,
                zorder=10,
            )
            has_out = True

    if has_out:
        ax_out.set_xlabel('Out-degree ($k_{out}$)')
        ax_out.set_ylabel(r'CCDF P($K \geq k_{out}$)')
        ax_out.set_title(title_text, fontsize=16, pad=10)
        ax_out.legend(loc='best', frameon=False)
    ax_out.spines['left'].set_visible(True)
    ax_out.spines['top'].set_visible(False)
    ax_out.spines['right'].set_visible(False)
    ax_out.spines['bottom'].set_color('black')
    ax_out.spines['left'].set_color('black')
    ax_out.tick_params(axis='both', colors='black')
    ax_out.xaxis.label.set_color('black')
    ax_out.yaxis.label.set_color('black')

    fig_out.tight_layout()

    # ========== 入度分布 (CCDF) ==========
    fig_in, ax_in = plt.subplots(1, 1, figsize=FIG_SIZE)
    has_in = False

    # 先画各模型
    for model_name, df in data_dict.items():
        in_degrees = df['Gene2'].value_counts()
        x, y = get_ccdf(in_degrees)

        if x is not None and len(x) > 0:
            has_in = True
            cfg = MODELS[model_name]
            z = 5 if model_name == "scGPT" else 3
            ax_in.loglog(
                x, y,
                marker="o",
                color=cfg['color'],
                label=format_label_with_tail_prob(model_name, in_degrees),
                linestyle='-',
                alpha=0.95,
                linewidth=2.2,
                markersize=4.8,
                markerfacecolor=cfg["color"],
                markeredgecolor=model_color("STRING"),
                markeredgewidth=0.6,
                zorder=z,
            )

    # 再画真实网络，确保 STRING 虚线在最上层
    if real_df is not None and len(real_df) > 0:
        in_deg_real = real_df['Gene2'].value_counts()
        xr, yr = get_ccdf(in_deg_real)
        if xr is not None and len(xr) > 0:
            ax_in.loglog(
                xr, yr,
                label=format_label_with_tail_prob(REAL_STYLE["label"], in_deg_real),
                color=REAL_STYLE["color"],
                linestyle=REAL_STYLE["linestyle"],
                linewidth=REAL_STYLE["linewidth"],
                alpha=REAL_STYLE["alpha"],
                marker=None,
                zorder=10,
            )
            has_in = True

    if has_in:
        ax_in.set_xlabel('In-degree ($k_{in}$)')
        ax_in.set_ylabel(r'CCDF P($K \geq k_{in}$)')
        ax_in.set_title(title_text, fontsize=16, pad=10)
        ax_in.legend(loc='best', frameon=False)
    ax_in.spines['left'].set_visible(True)
    ax_in.spines['top'].set_visible(False)
    ax_in.spines['right'].set_visible(False)
    ax_in.spines['bottom'].set_color('black')
    ax_in.spines['left'].set_color('black')
    ax_in.tick_params(axis='both', colors='black')
    ax_in.xaxis.label.set_color('black')
    ax_in.yaxis.label.set_color('black')

    fig_in.tight_layout()

    out_file_out = OUTPUT_DIR / f"{dataset}_CCDF_outdegree_models_vs_STRING.pdf"
    fig_out.savefig(out_file_out)
    plt.close(fig_out)

    out_file_in = OUTPUT_DIR / f"{dataset}_CCDF_indegree_models_vs_STRING.pdf"
    fig_in.savefig(out_file_in)
    plt.close(fig_in)

    print(f"  ✓ 生成: {out_file_out.name}")
    print(f"  ✓ 生成: {out_file_in.name}")
    return out_file_out

# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(description="Plot CCDF degree distributions for selected extraction.")
    parser.add_argument(
        "--extraction",
        type=str,
        default=DEFAULT_EXTRACTION,
        choices=VALID_EXTRACTIONS,
        help="Extraction subdir under evl_omipath (e.g., emb500, att500, embhidden500).",
    )
    parser.add_argument(
        "--undirected",
        action="store_true",
        help="Treat edges as undirected and plot a single degree CCDF.",
    )
    args = parser.parse_args()

    data_dir = EVL_ROOT / f"output_{args.extraction}"
    output_dir = Path(__file__).resolve().parent / f"ccdf_{args.extraction}"
    output_dir.mkdir(exist_ok=True)

    global DATA_DIR, OUTPUT_DIR
    DATA_DIR = data_dir
    OUTPUT_DIR = output_dir

    print(f"正在生成 CCDF 图表，输出目录: {OUTPUT_DIR}")
    if not DATA_DIR.exists():
        print(f"模型目录不存在: {DATA_DIR}")
        return

    if not REAL_DATA_DIR.exists():
        print(f"真实网络目录不存在: {REAL_DATA_DIR}")
        print("将使用默认边数（10000），且不过滤Gene1/Gene2。")

    for dataset in DATASETS:
        print(f"\n处理: {dataset} ...")

        # 关键修改：先加载真实网络数据，获取过滤依据
        real_df, real_gene1_set, real_gene_union_set, real_edge_count = load_real_data(dataset)
        
        if real_df is None:
            print(f"  ! 未找到/无法解析真实网络: {REAL_DATA_DIR}/{dataset}_processed-network.csv")
            print(f"  ! 将使用默认配置：边数={DEFAULT_TOP_N}，不过滤Gene1/Gene2")
            # 兜底：不过滤Gene1/Gene2，使用默认边数
            real_gene1_set = None
            real_gene_union_set = None
            target_edge_count = DEFAULT_TOP_N
        else:
            target_edge_count = real_edge_count
            print(f"  ✔ 目标边数: {target_edge_count} (匹配真实网络)")

        # 加载模型数据（传入过滤条件）
        data_dict = {}
        for name in MODELS:
            print(f"    处理模型: {name}")
            df = load_model_data(
                model_name=name,
                dataset=dataset,
                real_gene1_set=real_gene1_set,
                real_gene_union_set=real_gene_union_set,
                target_edge_count=target_edge_count
            )
            if df is not None and len(df) > 0:
                data_dict[name] = df

        # 画图
        plot_ccdf_distribution(
            dataset,
            data_dict,
            real_df=real_df,
            undirected=args.undirected,
            extraction=args.extraction,
        )

    print("\n完成。")

if __name__ == "__main__":
    main()