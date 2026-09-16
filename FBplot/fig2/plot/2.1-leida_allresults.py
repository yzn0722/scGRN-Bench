import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 禁用图形显示
import networkx as nx
from pathlib import Path
from typing import Optional, Set
import warnings
warnings.filterwarnings('ignore')

# Unified fig2 palette (fixed model colors)
from fig2_palette import MODEL_COLORS, model_color

# ------------------- 全局设置 -------------------
# 路径配置
EVL_ROOT = Path("/mnt/10T/yzn/benchmark_GRN/evl_omipath")
REAL_DATA_DIR = Path("/mnt/10T/yzn/benchmark_GRN/input_process/STRING")
OUTPUT_DIR = Path(__file__).resolve().parent / "output_all_datasets_topology"
OUTPUT_DIR.mkdir(exist_ok=True)

# 所有数据集列表（根据你提供的数据）
DATASETS = ['hESC', 'hHep', 'mDC', 'mHSC-L', 'mHSC-E', 'mHSC-GM']

# 6模型 × 3提取方式
EXTRACTIONS = ("emb500", "att500", "embhidden500")
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

REAL_NETWORK = 'STRING'
EDGE_FILTER_MODE = "g1_union_topn"
STRING_DIRECTED = True
STRING_EDGE_BUDGET = "directed_unique"

# ------------------- 数据加载函数 -------------------
def load_network(model_name: str, extraction: str, dataset_name: str):
    """加载模型预测的网络"""
    cfg = MODEL_CONFIGS[model_name]
    path_cfg = cfg["paths"][extraction]
    model_dir = EVL_ROOT / f"output_{extraction}" / path_cfg["dir"]
    if not model_dir.exists():
        return None
    
    possible_files = []
    for p in path_cfg["prefixes"]:
        possible_files.extend([
            f"{p}{dataset_name}.tsv",
            f"{p}{dataset_name}.csv",
            f"{dataset_name}_{p[:-1]}.tsv",
        ])
    
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
            
            if len(df_pred.columns) == 1:
                df_pred = pd.read_csv(pred_path, sep='\t', header=None)
                if len(df_pred.columns) >= 3:
                    df_pred.columns = ['Gene1', 'Gene2', 'EdgeWeight'] + [f'col{i}' for i in range(3, len(df_pred.columns))]
            
            return df_pred
    
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
    """识别 Gene1/Gene2 列并统一命名"""
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
    """STRING 参考表中 Gene1 列出现的基因集合"""
    d = _rename_edge_columns(df_string)
    if d is None:
        return set()
    return set(d['Gene1'].astype(str).str.strip().str.upper().tolist())

def string_gene_universe(df_string: pd.DataFrame) -> Set[str]:
    """STRING 参考网络中出现的全部基因"""
    d = _rename_edge_columns(df_string)
    if d is None:
        return set()
    g1 = d['Gene1'].astype(str).str.strip().str.upper()
    g2 = d['Gene2'].astype(str).str.strip().str.upper()
    return set(g1.tolist()) | set(g2.tolist())

def string_directed_ordered_pairs(df_string: pd.DataFrame) -> Set[tuple]:
    """STRING 有向边集合"""
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

def string_undirected_edge_pairs(df_string: pd.DataFrame) -> Set[tuple]:
    """STRING 无向边集合"""
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

def string_edge_budget_n(df_string: pd.DataFrame, budget: str) -> int:
    """与 STRING 对齐时截取的边数 N"""
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
    """筛选预测边"""
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

def preprocess_dataframe(df, directed: Optional[bool] = None):
    """预处理数据框，构建 NetworkX 图"""
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

def calculate_metrics(G):
    """计算网络拓扑指标"""
    if G is None or G.number_of_nodes() < 5:
        return {m: np.nan for m in METRICS}

    G_mod = G.to_undirected() if isinstance(G, nx.DiGraph) else G

    metrics = {}
    
    # 1. Average Path Length
    try:
        if G_mod.number_of_nodes() <= 1:
            metrics['AvgPathLength'] = np.nan
        else:
            if nx.is_connected(G_mod):
                G_path = G_mod
            else:
                largest_cc = max(nx.connected_components(G_mod), key=len)
                G_path = G_mod.subgraph(largest_cc).copy()
            if G_path.number_of_nodes() <= 1:
                metrics['AvgPathLength'] = np.nan
            else:
                metrics['AvgPathLength'] = float(nx.average_shortest_path_length(G_path))
    except Exception:
        metrics['AvgPathLength'] = np.nan
    
    # 2. Average Clustering Coefficient
    try:
        metrics['Clustering'] = nx.average_clustering(G)
    except Exception:
        metrics['Clustering'] = np.nan
    
    # 3. Modularity
    try:
        from community import community_louvain
        partition = community_louvain.best_partition(G_mod)
        metrics['Modularity'] = community_louvain.modularity(partition, G_mod)
    except Exception:
        try:
            if G_mod.number_of_nodes() > 0:
                communities = list(nx.algorithms.community.greedy_modularity_communities(G_mod))
                metrics['Modularity'] = nx.algorithms.community.modularity(G_mod, communities)
            else:
                metrics['Modularity'] = np.nan
        except Exception:
            metrics['Modularity'] = np.nan
    
    # 4. Average Degree
    try:
        degrees = [d for _, d in G.degree()]
        metrics['AvgDegree'] = np.mean(degrees) if degrees else np.nan
    except:
        metrics['AvgDegree'] = np.nan
    
    # 5. Assortativity
    try:
        r = nx.degree_assortativity_coefficient(G)
        if r is None or not np.isfinite(r):
            metrics['Assortativity'] = np.nan
        else:
            metrics['Assortativity'] = float(r)
    except Exception:
        metrics['Assortativity'] = np.nan
    
    return metrics

# ------------------- 主函数 -------------------
def main():
    print("=" * 80)
    print("Network Topology Metrics - All Datasets Analysis")
    print(f"Datasets: {DATASETS}")
    print(f"Metrics: {METRICS}")
    print("=" * 80)
    
    # 存储所有结果
    all_results = []
    
    for dataset in DATASETS:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset}")
        print('='*60)
        
        for extraction in EXTRACTIONS:
            print(f"\n  Extraction: {extraction}")
            print(f"  {'-'*40}")
            
            # 加载真实网络
            try:
                df_real = load_real_network(dataset)
                string_genes = string_gene_universe(df_real)
                string_edge_budget_n_keep = string_edge_budget_n(df_real, STRING_EDGE_BUDGET)
                G_real = preprocess_dataframe(df_real)
                real_metrics = calculate_metrics(G_real) if G_real else {m: np.nan for m in METRICS}
            except Exception as e:
                print(f"    ✗ Error loading STRING for {dataset}: {e}")
                real_metrics = {m: np.nan for m in METRICS}
            
            # 记录STRING结果
            for metric in METRICS:
                all_results.append({
                    'Dataset': dataset,
                    'Extraction': extraction,
                    'Model': REAL_NETWORK,
                    'Metric': metric,
                    'Value': real_metrics.get(metric, np.nan),
                    'NumNodes': G_real.number_of_nodes() if G_real else np.nan,
                    'NumEdges': G_real.number_of_edges() if G_real else np.nan,
                })
            
            # 处理每个模型
            for model_name in MODEL_ORDER:
                try:
                    df_pred = load_network(model_name, extraction, dataset)
                    if df_pred is None:
                        print(f"    ✗ {model_name}: No prediction file found")
                        for metric in METRICS:
                            all_results.append({
                                'Dataset': dataset,
                                'Extraction': extraction,
                                'Model': model_name,
                                'Metric': metric,
                                'Value': np.nan,
                                'NumNodes': np.nan,
                                'NumEdges': np.nan,
                            })
                        continue
                    
                    # 筛选边
                    df_f = filter_predicted_edges_g1_union_topn(
                        df_pred, df_real, n_keep=string_edge_budget_n_keep
                    )
                    
                    if df_f is None:
                        print(f"    ✗ {model_name}: No edges after filtering")
                        for metric in METRICS:
                            all_results.append({
                                'Dataset': dataset,
                                'Extraction': extraction,
                                'Model': model_name,
                                'Metric': metric,
                                'Value': np.nan,
                                'NumNodes': np.nan,
                                'NumEdges': np.nan,
                            })
                        continue
                    
                    G_pred = preprocess_dataframe(df_f)
                    if G_pred:
                        pred_metrics = calculate_metrics(G_pred)
                        print(f"    ✓ {model_name}: nodes={G_pred.number_of_nodes()}, edges={G_pred.number_of_edges()}")
                        
                        for metric in METRICS:
                            all_results.append({
                                'Dataset': dataset,
                                'Extraction': extraction,
                                'Model': model_name,
                                'Metric': metric,
                                'Value': pred_metrics.get(metric, np.nan),
                                'NumNodes': G_pred.number_of_nodes(),
                                'NumEdges': G_pred.number_of_edges(),
                            })
                    else:
                        print(f"    ✗ {model_name}: Failed to build graph")
                        for metric in METRICS:
                            all_results.append({
                                'Dataset': dataset,
                                'Extraction': extraction,
                                'Model': model_name,
                                'Metric': metric,
                                'Value': np.nan,
                                'NumNodes': np.nan,
                                'NumEdges': np.nan,
                            })
                except Exception as e:
                    print(f"    ✗ {model_name}: Error - {e}")
                    for metric in METRICS:
                        all_results.append({
                            'Dataset': dataset,
                            'Extraction': extraction,
                            'Model': model_name,
                            'Metric': metric,
                            'Value': np.nan,
                            'NumNodes': np.nan,
                            'NumEdges': np.nan,
                        })
    
    # 保存结果到CSV
    results_df = pd.DataFrame(all_results)
    
    # 保存长格式结果
    long_csv_path = OUTPUT_DIR / "all_datasets_topology_metrics_long.csv"
    results_df.to_csv(long_csv_path, index=False)
    print(f"\n✅ Long format results saved to: {long_csv_path}")
    
    # 转换为宽格式（每个指标一列）
    wide_df = results_df.pivot_table(
        index=['Dataset', 'Extraction', 'Model'],
        columns='Metric',
        values='Value'
    ).reset_index()
    
    # 添加节点数和边数信息
    node_edge_df = results_df[['Dataset', 'Extraction', 'Model', 'NumNodes', 'NumEdges']].drop_duplicates()
    wide_df = wide_df.merge(node_edge_df, on=['Dataset', 'Extraction', 'Model'], how='left')
    
    # 重新排列列顺序
    cols = ['Dataset', 'Extraction', 'Model', 'NumNodes', 'NumEdges'] + METRICS
    wide_df = wide_df[cols]
    
    wide_csv_path = OUTPUT_DIR / "all_datasets_topology_metrics_wide.csv"
    wide_df.to_csv(wide_csv_path, index=False)
    print(f"✅ Wide format results saved to: {wide_csv_path}")
    
    # 按数据集分别保存
    for dataset in DATASETS:
        dataset_df = wide_df[wide_df['Dataset'] == dataset]
        dataset_csv_path = OUTPUT_DIR / f"{dataset}_topology_metrics.csv"
        dataset_df.to_csv(dataset_csv_path, index=False)
        print(f"✅ {dataset} results saved to: {dataset_csv_path}")
    
    # 打印摘要统计
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    summary = wide_df.groupby(['Extraction', 'Model'])[METRICS].mean()
    print(summary.round(4))
    
    summary_csv_path = OUTPUT_DIR / "summary_statistics.csv"
    summary.to_csv(summary_csv_path)
    print(f"\n✅ Summary statistics saved to: {summary_csv_path}")

if __name__ == "__main__":
    import numpy as np
    main()