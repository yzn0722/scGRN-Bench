import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import argparse
from pathlib import Path
from typing import Optional, Set
import warnings
warnings.filterwarnings('ignore')

# Unified fig2 palette (fixed model colors)
from fig2_palette import MODEL_COLORS, model_color

# ------------------- 全局设置（Nature 友好：矢量字体 + 白底） -------------------
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

# 路径配置
EVL_ROOT = Path("/mnt/10T/yzn/benchmark_GRN/evl_omipath")
REAL_DATA_DIR = Path("/mnt/10T/yzn/benchmark_GRN/input_process/STRING")
OUTPUT_DIR = Path(__file__).resolve().parent / "output_hESC_radar_topology"
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_SIZE = (6, 6)

# 6模型 × 3提取方式
EXTRACTIONS = ("emb500", "att500", "embhidden500")
EXTRACTION_DISPLAY = {
    "emb500": r"cos$_{tok}$",
    "att500": "attn",
    "embhidden500": r"cos$_{hid}$",
}
MODEL_ORDER = ("Geneformer", "LangCell", "scGPT", "scCello", "scFoundation", "scPrint")
MODEL_CONFIGS = {
    "Geneformer": {
        "color": model_color("Geneformer"),
        "paths": {
            "emb500": {"dir": "geneformer", "prefixes": ("geneformer_", "Geneformer_")},
            "att500": {"dir": "geneformer", "prefixes": ("geneformer_", "Geneformer_")},
            "embhidden500": {"dir": "geneformer", "prefixes": ("geneformer_", "Geneformer_")},
        },
    },
    "LangCell": {
        "color": model_color("LangCell"),
        "paths": {
            "emb500": {"dir": "langcell", "prefixes": ("LangCell_", "langcell_", "Langcell_")},
            "att500": {"dir": "langcell", "prefixes": ("LangCell_", "langcell_", "Langcell_")},
            "embhidden500": {"dir": "Langcell", "prefixes": ("Langcell_", "LangCell_", "langcell_")},
        },
    },
    "scGPT": {
        "color": model_color("scGPT"),
        "paths": {
            "emb500": {"dir": "scgpt", "prefixes": ("scgpt_", "scGPT_")},
            "att500": {"dir": "scgpt", "prefixes": ("scgpt_", "scGPT_")},
            "embhidden500": {"dir": "scgpt", "prefixes": ("scGPT_", "scgpt_")},
        },
    },
    "scCello": {
        "color": model_color("scCello"),
        "paths": {
            "emb500": {"dir": "sccello", "prefixes": ("scCello_", "sccello_")},
            "att500": {"dir": "sccello", "prefixes": ("scCello_", "sccello_")},
            "embhidden500": {"dir": "sccello", "prefixes": ("scCello_", "sccello_")},
        },
    },
    "scFoundation": {
        "color": model_color("scFoundation"),
        "paths": {
            "emb500": {"dir": "scFoundation", "prefixes": ("scFoundation_", "scfoundation_")},
            "att500": {"dir": "scFoundation", "prefixes": ("scFoundation_", "scfoundation_")},
            "embhidden500": {"dir": "scFoundation", "prefixes": ("scFoundation_", "scfoundation_")},
        },
    },
    "scPrint": {
        "color": model_color("scPrint"),
        "paths": {
            "emb500": {"dir": "scprint", "prefixes": ("scprint_", "scPrint_", "scPRINT_")},
            "att500": {"dir": "scprint", "prefixes": ("scprint_", "scPrint_", "scPRINT_")},
            "embhidden500": {"dir": "scprint", "prefixes": ("scprint_", "scPrint_", "scPRINT_")},
        },
    },
}

# 指标名称（5个拓扑指标）
METRICS = ['AvgPathLength', 'Clustering', 'Modularity', 'AvgDegree', 'Assortativity']
METRICS_DISPLAY = ['Avg Path Length', 'Clustering', 'Modularity', 'Avg Degree', 'Assortativity']

# 真实网络（STRING）
REAL_NETWORK = 'STRING'

# 默认分析的数据集（与输出文件名一致）
DATASET = 'hESC'

# 预测边筛选方式（与 STRING 对齐）：
#   "g1_union_topn" — Pred.Gene1 ∈ STRING 的 Gene1 集合；Pred.Gene2 ∈ STRING(Gene1∪Gene2)；
#                    按权重降序去重 (Gene1,Gene2) 后取前 N 条，N 见 STRING_EDGE_BUDGET
#   "genes" — 两端均在 STRING 节点集合 (Gene1∪Gene2)
#   "edges" — 无序基因对与 STRING 参考边表一致
EDGE_FILTER_MODE = "g1_union_topn"

# STRING 是否为有向边（Gene1→Gene2）。True：建 nx.DiGraph，边数按有向有序对计，不会把 A→B 与 B→A 合并成一条。
STRING_DIRECTED = True

# STRING 边数预算 N（用于 g1_union_topn）：
#   "directed_unique" — 唯一有序对 (Gene1, Gene2) 数量（去重后，与有向简单图一致；推荐）
#   "unique_undirected" — 唯一无向对 {min,max} 数量（会把 A→B 与 B→A 合成一条，故约为有向行数的一半）
#   "rows" — 文件行数（含可能重复行，不去重）
STRING_EDGE_BUDGET = "directed_unique"

# 配色方案（统一）
COLORS = dict(MODEL_COLORS)

# ------------------- 数据加载函数 -------------------
def load_network(model_name: str, extraction: str, dataset_name: str):
    """加载模型预测的网络"""
    cfg = MODEL_CONFIGS[model_name]
    path_cfg = cfg["paths"][extraction]
    model_dir = EVL_ROOT / f"output_{extraction}" / path_cfg["dir"]
    if not model_dir.exists():
        return None
    
    # 尝试可能的文件名模式
    possible_files = []
    for p in path_cfg["prefixes"]:
        possible_files.extend(
            [
                f"{p}{dataset_name}.tsv",
                f"{p}{dataset_name}.csv",
                f"{dataset_name}_{p[:-1]}.tsv",
            ]
        )
    
    for filename in possible_files:
        pred_path = model_dir / filename
        if pred_path.exists():
            if filename.endswith('.tsv'):
                try:
                    df_pred = pd.read_csv(pred_path, sep='\t')
                except:
                    df_pred = pd.read_csv(pred_path)
            else:
                df_pred = pd.read_csv(pred_path)
            
            # 如果只有一列，尝试按制表符分割
            if len(df_pred.columns) == 1:
                df_pred = pd.read_csv(pred_path, sep='\t', header=None)
                if len(df_pred.columns) >= 3:
                    df_pred.columns = ['Gene1', 'Gene2', 'EdgeWeight'] + [f'col{i}' for i in range(3, len(df_pred.columns))]
            
            return df_pred
    
    # 如果没找到标准格式，使用第一个找到的文件
    files = list(model_dir.glob(f"*{dataset_name}*"))
    if files:
        pred_path = files[0]
        if str(pred_path).endswith('.tsv'):
            try:
                df_pred = pd.read_csv(pred_path, sep='\t')
            except:
                df_pred = pd.read_csv(pred_path)
        else:
            df_pred = pd.read_csv(pred_path)
        return df_pred
    
    return None

def load_real_network(dataset_name):
    """加载真实网络"""
    real_path = REAL_DATA_DIR / f"{dataset_name}_processed-network.csv"
    if real_path.exists():
        return pd.read_csv(real_path)
    
    possible_patterns = [
        f"{dataset_name}_processed-network.csv",
        f"{dataset_name}_STRING.csv",
        f"{dataset_name}.csv",
    ]
    
    for pattern in possible_patterns:
        file_path = REAL_DATA_DIR / pattern
        if file_path.exists():
            return pd.read_csv(file_path)
    
    raise FileNotFoundError(f"No real network file found for {dataset_name} in {REAL_DATA_DIR}")


def _rename_edge_columns(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """识别 Gene1/Gene2 列并统一命名；无有效边则返回 None。"""
    if df is None or len(df) == 0:
        return None

    df_clean = df.copy()
    df_clean.columns = [col.strip() for col in df_clean.columns]

    gene1_col, gene2_col, weight_col = None, None, None
    for col in df_clean.columns:
        col_lower = str(col).lower()
        if any(term in col_lower for term in ['gene1', 'regulator', 'tf', 'source']):
            gene1_col = col
        elif any(term in col_lower for term in ['gene2', 'target', 'regulated']):
            gene2_col = col
        elif any(term in col_lower for term in ['edgeweight', 'importance', 'weight', 'score']):
            weight_col = col

    if gene1_col is None and len(df_clean.columns) >= 1:
        gene1_col = df_clean.columns[0]
    if gene2_col is None and len(df_clean.columns) >= 2:
        gene2_col = df_clean.columns[1]

    rename_dict = {}
    if gene1_col:
        rename_dict[gene1_col] = 'Gene1'
    if gene2_col:
        rename_dict[gene2_col] = 'Gene2'
    if weight_col:
        rename_dict[weight_col] = 'weight'
    df_clean = df_clean.rename(columns=rename_dict)

    if 'Gene1' not in df_clean.columns or 'Gene2' not in df_clean.columns:
        return None
    df_clean = df_clean.dropna(subset=['Gene1', 'Gene2'])
    if len(df_clean) == 0:
        return None
    return df_clean


def string_gene1_only(df_string: pd.DataFrame) -> Set[str]:
    """STRING 参考表中 Gene1 列出现的基因集合（大写），用于约束预测边的 Gene1。"""
    d = _rename_edge_columns(df_string)
    if d is None:
        return set()
    return set(d['Gene1'].astype(str).str.strip().str.upper().tolist())


def string_gene_universe(df_string: pd.DataFrame) -> Set[str]:
    """
    STRING 参考网络中出现的全部基因（Gene1 ∪ Gene2），大写去重，用于与预测边对齐。
    """
    d = _rename_edge_columns(df_string)
    if d is None:
        return set()
    g1 = d['Gene1'].astype(str).str.strip().str.upper()
    g2 = d['Gene2'].astype(str).str.strip().str.upper()
    return set(g1.tolist()) | set(g2.tolist())


def string_undirected_edge_pairs(df_string: pd.DataFrame) -> Set[tuple]:
    """STRING 参考网络的无序边集合 {(min,max), ...}，基因大写。"""
    d = _rename_edge_columns(df_string)
    if d is None:
        return set()
    pairs = set()
    for a, b in zip(
        d['Gene1'].astype(str).str.strip().str.upper(),
        d['Gene2'].astype(str).str.strip().str.upper(),
    ):
        if not a or not b or a == b:
            continue
        u, v = (a, b) if a <= b else (b, a)
        pairs.add((u, v))
    return pairs


def string_directed_ordered_pairs(df_string: pd.DataFrame) -> Set[tuple]:
    """STRING 有向边集合 {(Gene1, Gene2), ...}，大写；A→B 与 B→A 为两条不同边。"""
    d = _rename_edge_columns(df_string)
    if d is None:
        return set()
    out = set()
    for a, b in zip(
        d['Gene1'].astype(str).str.strip().str.upper(),
        d['Gene2'].astype(str).str.strip().str.upper(),
    ):
        if not a or not b or a == b:
            continue
        out.add((a, b))
    return out


def string_edge_budget_n(df_string: pd.DataFrame, budget: str) -> int:
    """与 STRING 对齐时截取的边数 N。"""
    d = _rename_edge_columns(df_string)
    if d is None:
        return 0
    if budget == "rows":
        return int(len(d))
    if budget == "directed_unique":
        g1 = d['Gene1'].astype(str).str.strip().str.upper()
        g2 = d['Gene2'].astype(str).str.strip().str.upper()
        return int(pd.DataFrame({"g1": g1, "g2": g2}).drop_duplicates().shape[0])
    if budget == "unique_undirected":
        return len(string_undirected_edge_pairs(df_string))
    return len(string_directed_ordered_pairs(df_string))


def filter_predicted_edges_g1_union_topn(
    df_pred: pd.DataFrame,
    df_string: pd.DataFrame,
    *,
    n_keep: int,
) -> Optional[pd.DataFrame]:
    """
    筛选规则（与用户需求一致）：
      1) Pred 的 Gene1 必须出现在 STRING 的 **Gene1** 列取值集合中；
      2) Pred 的 Gene2 必须出现在 STRING 的 **Gene1 ∪ Gene2** 基因集合中；
      3) 按边权重（降序）排序，对有序对 (Gene1, Gene2) 去重保留权重最高者，再取前 N 条，N = STRING 边数预算。

    若无 weight 列，则去重后按行序取前 N 条。
    """
    d = _rename_edge_columns(df_pred)
    if d is None or n_keep <= 0:
        return None

    g1_allowed = string_gene1_only(df_string)
    g2_allowed = string_gene_universe(df_string)
    if not g1_allowed or not g2_allowed:
        return None

    g1 = d['Gene1'].astype(str).str.strip().str.upper()
    g2 = d['Gene2'].astype(str).str.strip().str.upper()
    mask = g1.isin(g1_allowed) & g2.isin(g2_allowed)
    out = d.loc[mask].copy()
    if len(out) == 0:
        return None

    out['_g1u'] = g1[mask].values
    out['_g2u'] = g2[mask].values

    if 'weight' in out.columns:
        w = pd.to_numeric(out['weight'], errors='coerce')
    else:
        w = pd.Series(np.nan, index=out.index)
    out['_w'] = w.fillna(-np.inf)

    out = out.sort_values('_w', ascending=False)
    out = out.drop_duplicates(subset=['_g1u', '_g2u'], keep='first')
    out = out.head(int(n_keep))

    out['Gene1'] = out['_g1u'].values
    out['Gene2'] = out['_g2u'].values
    out = out.drop(columns=[c for c in ['_g1u', '_g2u', '_w'] if c in out.columns])
    if len(out) == 0:
        return None
    return out


def filter_predicted_edges_to_string(
    df_pred: pd.DataFrame,
    *,
    string_genes: Set[str],
    string_edge_pairs: Optional[Set[tuple]] = None,
    mode: str = "genes",
) -> Optional[pd.DataFrame]:
    """
    与 STRING 对齐的预测边筛选。

    - mode=\"genes\": 两端基因均须在 STRING 节点集合中。
    - mode=\"edges\": 在 \"genes\" 基础上，仅保留 (Gene1, Gene2) 的无序对与 STRING 边表一致。
    """
    d = _rename_edge_columns(df_pred)
    if d is None:
        return None
    g1 = d['Gene1'].astype(str).str.strip().str.upper()
    g2 = d['Gene2'].astype(str).str.strip().str.upper()

    if string_genes:
        U = {str(x).strip().upper() for x in string_genes}
        keep = g1.isin(U) & g2.isin(U)
    else:
        keep = pd.Series(True, index=d.index)

    if mode == "edges":
        if not string_edge_pairs:
            return None
        pair_ok = []
        for a, b in zip(g1, g2):
            if not a or not b or a == b:
                pair_ok.append(False)
                continue
            if STRING_DIRECTED:
                pair_ok.append((a, b) in string_edge_pairs)
            else:
                u, v = (a, b) if a <= b else (b, a)
                pair_ok.append((u, v) in string_edge_pairs)
        keep = keep & pd.Series(pair_ok, index=d.index)

    out = d.loc[keep].copy()
    out['Gene1'] = g1[keep].values
    out['Gene2'] = g2[keep].values
    if len(out) == 0:
        return None
    return out


def preprocess_dataframe(df, directed: Optional[bool] = None):
    """预处理数据框，构建 NetworkX 图。STRING_DIRECTED=True 时为有向简单图 DiGraph。"""
    df_clean = _rename_edge_columns(df)
    if df_clean is None:
        return None

    if directed is None:
        directed = STRING_DIRECTED

    if directed:
        G = nx.DiGraph()
        for _, row in df_clean.iterrows():
            a = str(row['Gene1']).strip().upper()
            b = str(row['Gene2']).strip().upper()
            if a and b and a != b:
                G.add_edge(a, b)
    else:
        G = nx.Graph()
        for _, row in df_clean.iterrows():
            a = str(row['Gene1']).strip().upper()
            b = str(row['Gene2']).strip().upper()
            if a and b and a != b:
                G.add_edge(a, b)

    return G

# ------------------- 指标计算函数 -------------------
def calculate_metrics(G):
    """计算网络拓扑指标（支持无向图或有向图；度序列用总度 in+out）。"""
    if G is None or G.number_of_nodes() < 5:
        return {m: 0.0 for m in METRICS}

    # 模块度在无向投影上算（Louvain / greedy modularity 标准用法）
    G_mod = G.to_undirected() if isinstance(G, nx.DiGraph) else G

    metrics = {}
    
    # 1. Average Path Length（在无向最大连通子图上）
    try:
        if G_mod.number_of_nodes() <= 1:
            metrics['AvgPathLength'] = 0.0
        else:
            if nx.is_connected(G_mod):
                G_path = G_mod
            else:
                largest_cc = max(nx.connected_components(G_mod), key=len)
                G_path = G_mod.subgraph(largest_cc).copy()
            if G_path.number_of_nodes() <= 1:
                metrics['AvgPathLength'] = 0.0
            else:
                metrics['AvgPathLength'] = float(nx.average_shortest_path_length(G_path))
    except Exception:
        metrics['AvgPathLength'] = 0.0
    
    # 2. Average Clustering Coefficient（有向图使用 NetworkX 有向聚类定义）
    try:
        metrics['Clustering'] = nx.average_clustering(G)
    except Exception:
        metrics['Clustering'] = 0.0
    
    # 3. Modularity（在无向投影上）
    try:
        # 使用Louvain算法计算模块度
        from community import community_louvain
        partition = community_louvain.best_partition(G_mod)
        metrics['Modularity'] = community_louvain.modularity(partition, G_mod)
    except Exception:
        try:
            if G_mod.number_of_nodes() > 0:
                communities = list(nx.algorithms.community.greedy_modularity_communities(G_mod))
                metrics['Modularity'] = nx.algorithms.community.modularity(G_mod, communities)
            else:
                metrics['Modularity'] = 0.0
        except Exception:
            metrics['Modularity'] = 0.0
    
    # 4. Average Degree
    try:
        degrees = [d for _, d in G.degree()]
        metrics['AvgDegree'] = np.mean(degrees) if degrees else 0.0
    except:
        metrics['AvgDegree'] = 0.0
    
    # 5. Assortativity (同配性)
    try:
        r = nx.degree_assortativity_coefficient(G)
        if r is None or not np.isfinite(r):
            metrics['Assortativity'] = 0.0
        else:
            metrics['Assortativity'] = float(r)
    except Exception:
        metrics['Assortativity'] = 0.0
    
    return metrics

# ------------------- 归一化函数 -------------------
def _finite_float(x, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def normalize_metrics(all_metrics, ref_name: str = REAL_NETWORK):
    """
    雷达图用：以 STRING（ref_name）为「参考满分」。

    - STRING：每个指标轴上恒为 1.0（最外圈 = 参考网络）。
    - 其他模型：该指标上与 STRING 的**接近度** ∈ [0,1]，
      closeness = 1 - |val_model - val_STRING| / max_j |val_j - val_STRING|，
      其中 j 取遍除 STRING 外的所有模型；若所有模型与 STRING 完全一致则均为 1。

    含义：同一轴上越靠近 STRING 的模型得分越高，而不是单纯「指标越大越好」。
    """
    if ref_name not in all_metrics:
        return normalize_metrics_minmax_fallback(all_metrics)

    ref = all_metrics[ref_name]
    others = [k for k in all_metrics.keys() if k != ref_name]
    normalized: dict = {ref_name: {m: 1.0 for m in METRICS}}

    for k in others:
        normalized[k] = {}

    for metric in METRICS:
        r = _finite_float(ref.get(metric, 0.0))

        diffs = []
        for k in others:
            v = _finite_float(all_metrics[k].get(metric, 0.0))
            diffs.append(abs(v - r))

        max_diff = max(diffs) if diffs else 0.0
        eps = 1e-12

        for k in others:
            v = _finite_float(all_metrics[k].get(metric, 0.0))
            d = abs(v - r)
            if max_diff <= eps:
                closeness = 1.0
            else:
                closeness = 1.0 - d / max_diff
            normalized[k][metric] = float(np.clip(closeness, 0.0, 1.0))

    return normalized


def normalize_metrics_zscore_similarity(all_metrics, ref_name: str = REAL_NETWORK):
    """
    Z-score similarity to reference network (STRING):
    1) For each metric, compute z-score across all available models.
    2) Compute distance to STRING in z-space: d = |z_model - z_ref|.
    3) Convert to similarity in (0,1] via s = 1 / (1 + d).
       STRING is fixed at 1.0 on all axes.
    """
    if ref_name not in all_metrics:
        return normalize_metrics_minmax_fallback(all_metrics)

    normalized: dict = {m: {} for m in all_metrics.keys()}
    for metric in METRICS:
        vals = []
        keys = []
        for k, mv in all_metrics.items():
            if metric in mv:
                vals.append(_finite_float(mv.get(metric, 0.0)))
                keys.append(k)
        if not vals:
            for k in normalized.keys():
                normalized[k][metric] = 0.0
            continue

        arr = np.array(vals, dtype=float)
        mu = float(np.mean(arr))
        sigma = float(np.std(arr))
        if sigma <= 1e-12:
            # All models are identical on this metric.
            for k in normalized.keys():
                normalized[k][metric] = 1.0
            continue

        z = {k: (v - mu) / sigma for k, v in zip(keys, arr)}
        z_ref = z.get(ref_name, 0.0)
        for k in normalized.keys():
            zk = z.get(k, z_ref)
            d = abs(zk - z_ref)
            s = 1.0 / (1.0 + d)  # in (0,1], avoids hard zero
            normalized[k][metric] = float(np.clip(s, 0.0, 1.0))
        normalized[ref_name][metric] = 1.0

    return normalized


def normalize_metrics_minmax_fallback(all_metrics):
    """按每个指标在全部网络（含 STRING）上做 min-max 到 [0,1]。"""
    normalized = {}
    for model, metrics in all_metrics.items():
        norm_metrics = {}
        for metric in METRICS:
            all_values = []
            for m in all_metrics.values():
                if metric not in m:
                    continue
                v = _finite_float(m.get(metric))
                all_values.append(v)
            if not all_values:
                norm_metrics[metric] = 0.0
                continue
            min_val = min(all_values)
            max_val = max(all_values)
            value = _finite_float(metrics.get(metric, 0.0))
            if max_val > min_val:
                norm_val = (value - min_val) / (max_val - min_val)
            else:
                norm_val = 0.5
            norm_metrics[metric] = float(np.clip(norm_val, 0, 1))
        normalized[model] = norm_metrics
    return normalized

# ------------------- 绘制雷达图 -------------------
def plot_radar_chart(normalized_metrics, model_colors, dataset_name: str, extraction: str, real_network_name=REAL_NETWORK):
    """绘制 Nature 风格雷达图：NPG 配色、首轴朝上、浅网格、PNG 输出。"""
    text_size = 16
    n_metrics = len(METRICS)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=FIG_SIZE, subplot_kw=dict(polar=True))
    # 第一个指标在正上方，顺时针
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # 径向网格：浅色同心圆
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=text_size, color='black')
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    # 外圈与刻度线
    ax.spines['polar'].set_linewidth(0.8)
    ax.spines['polar'].set_color('#333333')

    # 先画预测模型（STRING 最后画在最上层）；图例顺序与 MODELS 一致
    preferred = [k for k in MODEL_ORDER if k in normalized_metrics and k != real_network_name]
    rest = [
        m
        for m in normalized_metrics
        if m != real_network_name and m not in preferred
    ]
    model_order = preferred + rest
    for model_name in model_order:
        metrics = normalized_metrics[model_name]
        vals = np.array([metrics[m] for m in METRICS], dtype=float)
        vals_closed = np.concatenate([vals, [vals[0]]])

        if model_name in model_colors:
            color = model_colors[model_name]
        elif model_name in COLORS:
            color = COLORS[model_name]
        else:
            color = '#7F7F7F'

        lw = 2.6 if model_name == 'scGPT' else 2.0
        ax.plot(
            angles_closed,
            vals_closed,
            color=color,
            linewidth=lw,
            label=model_name,
            zorder=2,
        )
        ax.fill(angles_closed, vals_closed, color=color, alpha=0.12, zorder=1)

    # 真实网络 STRING：虚线 + 极浅填充
    if real_network_name in normalized_metrics:
        ref_color = COLORS.get('STRING', model_color('STRING'))
        rv = np.array([normalized_metrics[real_network_name][m] for m in METRICS], dtype=float)
        rv_closed = np.concatenate([rv, [rv[0]]])
        ax.plot(
            angles_closed,
            rv_closed,
            color=ref_color,
            linestyle='--',
            linewidth=2.8,
            label=real_network_name,
            zorder=4,
        )
        ax.fill(angles_closed, rv_closed, color=ref_color, alpha=0.06, zorder=3)

    # 角标签（将 Clustering 角度微调，避免被曲线遮挡）
    label_angles = angles.copy()
    if len(label_angles) > 1:
        # With clockwise theta direction, decreasing angle moves label visually right.
        label_angles[1] -= 0.12
    ax.set_xticks(label_angles)
    ax.set_xticklabels(
        METRICS_DISPLAY,
        fontsize=text_size,
        #fontweight='normal',
        color='black',
    )

    # Title: always keep (user requirement)
    title_text = EXTRACTION_DISPLAY.get(extraction, extraction)
    ax.set_title(
        title_text,
        fontsize=18,
        #fontweight='normal',
        #pad=16,
        color='black',
    )

    # Legend: only keep for hidden extraction
    if extraction == "embhidden500":
        leg = ax.legend(
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            fontsize=text_size,
            frameon=False,
            ncol=1,
            handlelength=2.4,
            borderaxespad=0.0,
        )
        for text in leg.get_texts():
            text.set_color('black')

    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')

    output_path = OUTPUT_DIR / f"{dataset_name}_{extraction}_network_topology_radar.pdf"
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none',
        format='pdf',
    )
    print(f"✅ Radar chart saved to: {output_path}")
    plt.close(fig)

# ------------------- 主函数 -------------------
def main():
    parser = argparse.ArgumentParser(description="Plot radar topology metrics with optional undirected mode.")
    parser.add_argument(
        "--undirected",
        action="store_true",
        help="Use undirected graph mode for all topology metrics (consistent with undirected CCDF).",
    )
    parser.add_argument(
        "--norm-method",
        type=str,
        default="zscore",
        choices=("linear", "zscore", "minmax"),
        help="Normalization to STRING for radar values.",
    )
    args = parser.parse_args()

    global STRING_DIRECTED, OUTPUT_DIR
    if args.undirected:
        STRING_DIRECTED = False
        OUTPUT_DIR = Path(__file__).resolve().parent / "output_hESC_radar_topology_undirected"
        OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Network Topology Radar Chart - Real Data Analysis")
    print(f"Dataset: {DATASET}")
    print("Metrics: Avg Path Length, Clustering, Modularity, Avg Degree, Assortativity")
    print(
        f"Predicted edges vs STRING: EDGE_FILTER_MODE={EDGE_FILTER_MODE!r}, "
        f"STRING_EDGE_BUDGET={STRING_EDGE_BUDGET!r}, STRING_DIRECTED={STRING_DIRECTED}"
    )
    print(f"Normalization method: {args.norm_method}")
    print(
        "(若以前用无向合并，A→B 与 B→A 会算成一条边，边数约为有向唯一对的一半)"
    )
    print("=" * 60)
    
    for extraction in EXTRACTIONS:
        extraction_label = EXTRACTION_DISPLAY.get(extraction, extraction)
        print("\n" + "-" * 60)
        print(f"Extraction: {extraction_label} ({extraction})")
        print("-" * 60)

        all_metrics = {}
        string_genes: Set[str] = set()
        string_edge_pairs: Optional[Set[tuple]] = None
        string_edge_budget_n_keep: int = 0

        # 1. 真实网络
        print(f"\nProcessing {REAL_NETWORK} (real network)...")
        try:
            df_real = load_real_network(DATASET)
            string_genes = string_gene_universe(df_real)
            string_edge_pairs = (
                string_directed_ordered_pairs(df_real)
                if STRING_DIRECTED
                else string_undirected_edge_pairs(df_real)
            )
            string_edge_budget_n_keep = string_edge_budget_n(df_real, STRING_EDGE_BUDGET)
            G_real = preprocess_dataframe(df_real)
            if G_real:
                all_metrics[REAL_NETWORK] = calculate_metrics(G_real)
        except Exception as e:
            print(f"  ✗ Error loading STRING: {e}")
            continue

        # 2. 6个模型
        for model_name in MODEL_ORDER:
            print(f"\nProcessing {model_name}...")
            try:
                df_pred = load_network(model_name, extraction, DATASET)
                if df_pred is None:
                    print("  ✗ No prediction file found")
                    continue
                raw_edges = _rename_edge_columns(df_pred)
                n_raw = len(raw_edges) if raw_edges is not None else 0
                if EDGE_FILTER_MODE == "g1_union_topn":
                    df_f = filter_predicted_edges_g1_union_topn(
                        df_pred,
                        df_real,
                        n_keep=string_edge_budget_n_keep,
                    )
                else:
                    df_f = filter_predicted_edges_to_string(
                        df_pred,
                        string_genes=string_genes,
                        string_edge_pairs=string_edge_pairs,
                        mode=EDGE_FILTER_MODE,
                    )
                n_f = len(df_f) if df_f is not None else 0
                print(f"  edges: raw={n_raw} -> filtered={n_f}")
                G_pred = preprocess_dataframe(df_f) if df_f is not None else None
                if G_pred:
                    all_metrics[model_name] = calculate_metrics(G_pred)
            except Exception as e:
                print(f"  ✗ Error: {e}")

        if len(all_metrics) < 2:
            print(f"\n❌ Not enough valid models for extraction={extraction}")
            continue

        print("\nNormalizing to STRING reference ...")
        if args.norm_method == "zscore":
            normalized_metrics = normalize_metrics_zscore_similarity(all_metrics, ref_name=REAL_NETWORK)
        elif args.norm_method == "minmax":
            normalized_metrics = normalize_metrics_minmax_fallback(all_metrics)
        else:
            normalized_metrics = normalize_metrics(all_metrics, ref_name=REAL_NETWORK)
        model_colors = {m: MODEL_CONFIGS[m]["color"] for m in MODEL_ORDER}
        plot_radar_chart(normalized_metrics, model_colors, DATASET, extraction, REAL_NETWORK)

        csv_path = OUTPUT_DIR / f"{DATASET}_{extraction}_network_topology_metrics.csv"
        pd.DataFrame.from_dict(all_metrics, orient='index').to_csv(csv_path)
        print(f"✅ Metrics saved to: {csv_path}")

if __name__ == "__main__":
    main()