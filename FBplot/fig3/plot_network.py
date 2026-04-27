import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Unified fig3 palette
from fig3_palette import model_color, method_color

# ---------------------- 路径配置 ----------------------
tf_stats_path = "/mnt/10T/yzn/benchmark_GRN/plot/gene_share/tf_advanced_analysis/tf_overlap_detailed_statistics.csv"
pred_network_path = "/mnt/10T/yzn/benchmark_GRN/evl_omipath/output_emb500/scgpt/scgpt_hESC.tsv"
string_network_path = "/mnt/10T/yzn/benchmark_GRN/input_process/STRING/hESC_processed-network.csv"

OUTPUT_DIR = Path(__file__).resolve().parent / "network"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = str(OUTPUT_DIR / "core_tf_network_only.pdf")

# 绘图风格
sns.set_style("white")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

# ---------------------- 读取数据 ----------------------
df_tf_stats = pd.read_csv(tf_stats_path)
core_tfs = df_tf_stats['TF'].tolist()

pred_df = pd.read_csv(pred_network_path, sep='\t')
string_df = pd.read_csv(string_network_path)

# 标准化列名
if 'Gene1' not in pred_df.columns or 'Gene2' not in pred_df.columns:
    pred_df = pred_df.iloc[:, :2].copy()
    pred_df.columns = ['Gene1', 'Gene2']

# ---------------------- 构建【仅核心TF】网络 ----------------------
G = nx.Graph()
for tf in core_tfs:
    G.add_node(tf)

# 只保留 TF-TF 之间的边
for _, row in pred_df.iterrows():
    g1 = str(row['Gene1'])
    g2 = str(row['Gene2'])
    if g1 in core_tfs and g2 in core_tfs:
        G.add_edge(g1, g2)

# ---------------------- 绘图：只保留一张干净的TF网络 ----------------------
plt.figure(figsize=(6, 6))

# 布局更分散
pos = nx.spring_layout(G, k=3.5, iterations=300, seed=42)

# 节点
nx.draw_networkx_nodes(
    G, pos,
    node_color=method_color("emb500"),
    node_size=500,
    alpha=0.9
)

# 边
nx.draw_networkx_edges(
    G, pos,
    edge_color="#8FB4DC",
    width=1,
    alpha=0.7
)

# 标签（只显示TF名称）
nx.draw_networkx_labels(
    G, pos,
    font_size=16,
    
    font_color="black"
)

plt.axis("off")
plt.tight_layout()
plt.savefig(output_path, bbox_inches="tight", format="pdf")
plt.close()

print(f"✅ 核心TF网络已保存：{output_path}")