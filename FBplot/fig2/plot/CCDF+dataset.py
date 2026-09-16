#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基因调控网络度分布可视化 - Nature风格 CCDF版 (模型 vs STRING真实网络)
【仅保留无向图度分布】
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
EVL_ROOT = Path("/mnt/10T/yzn/benchmark_GRN/evl_omipath")
REAL_DATA_DIR = Path("/mnt/10T/yzn/benchmark_GRN/input_process/STRING")
FIG_SIZE = (6, 6)

MODELS = {
    'Geneformer':   {'dir': 'geneformer',   'prefix': 'geneformer_',   'color': model_color('Geneformer'),   'marker': 'o'},
    'GENIE3':       {'dir': 'GENIE3',       'prefix': 'GENIE3_',       'color': model_color('GENIE3'),       'marker': 's'},
    'LangCell':     {'dir': 'langcell',     'prefix': 'LangCell_',     'color': model_color('LangCell'),     'marker': '^'},
    'scGPT':        {'dir': 'scgpt',        'prefix': 'scgpt_',        'color': model_color('scGPT'),        'marker': 'D'},
    'scCello':      {'dir': 'sccello',      'prefix': 'scCello_',      'color': model_color('scCello'),      'marker': 'h'},
    'scFoundation': {'dir': 'scFoundation', 'prefix': 'scFoundation_', 'color': model_color('scFoundation'), 'marker': 'v'},
    'scPrint':      {'dir': 'scprint',      'prefix': 'scprint_',      'color': model_color('scPrint'),      'marker': 'p'},
}

DATASETS = ['hESC', 'hHep', 'mDC', 'mHSC-L', 'mHSC-E', 'mHSC-GM']
FILTER_STRATEGY = 'top_n'
DEFAULT_TOP_N = 10000
DEFAULT_EXTRACTION = "emb500"
VALID_EXTRACTIONS = ("emb500", "att500", "embhidden500")

EXTRACTION_DISPLAY = {
    "emb500": r"cos$_{tok}$",
    "att500": "attn",
    "embhidden500": r"cos$_{hid}$",
}

REAL_STYLE = {
    "label": "STRING",
    "color": model_color("STRING"),
    "linestyle": "--",
    "linewidth": 2.8,
    "alpha": 0.95
}

TAIL_K_THRESHOLD = 10

# ==================== 函数 ====================
def detect_weight_column(df: pd.DataFrame):
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
    config = MODELS[model_name]
    filepath = DATA_DIR / config['dir'] / f"{config['prefix']}{dataset}.tsv"
    if not filepath.exists():
        return None

    try:
        df = pd.read_csv(filepath, sep='\t')
        if 'Gene1' not in df.columns or 'Gene2' not in df.columns:
            return None
        
        original_count = len(df)
        if real_gene1_set is not None and len(real_gene1_set) > 0:
            df = df[df['Gene1'].isin(real_gene1_set)]
        if real_gene_union_set is not None and len(real_gene_union_set) > 0:
            df = df[df['Gene2'].isin(real_gene_union_set)]
        if len(df) == 0:
            return None

        weight_col = detect_weight_column(df)
        if weight_col and FILTER_STRATEGY == 'top_n':
            actual_top_n = min(target_edge_count, len(df))
            df = df.nlargest(actual_top_n, weight_col)

        return df[['Gene1', 'Gene2']].dropna()
    except Exception as e:
        print(f"    ❌ {model_name} - {dataset}: 加载失败 {str(e)}")
        return None

def load_real_data(dataset: str):
    fp = REAL_DATA_DIR / f"{dataset}_processed-network.csv"
    if not fp.exists():
        return None, set(), set(), 0

    try:
        df = pd.read_csv(fp)
        cols_lower = {c.lower(): c for c in df.columns}
        gene1_candidates = ["gene1", "source", "tf", "regulator", "from", "protein1", "preferredname_a", "node1", "a"]
        gene2_candidates = ["gene2", "target", "to", "protein2", "preferredname_b", "node2", "b"]

        def pick_col(cands):
            for k in cands:
                if k in cols_lower:
                    return cols_lower[k]
            return None

        c1 = pick_col(gene1_candidates)
        c2 = pick_col(gene2_candidates)

        if c1 is None or c2 is None:
            if df.shape[1] < 2:
                return None, set(), set(), 0
            c1, c2 = df.columns[:2]

        real_df = df[[c1, c2]].rename(columns={c1: "Gene1", c2: "Gene2"}).dropna()
        gene1_set = set(real_df['Gene1'].unique())
        gene2_set = set(real_df['Gene2'].unique())
        gene_union_set = gene1_set.union(gene2_set)
        edge_count = len(real_df)
        return real_df, gene1_set, gene_union_set, edge_count
    except Exception as e:
        print(f"  ❌ 加载真实网络 {dataset} 失败: {str(e)}")
        return None, set(), set(), 0

def get_ccdf(data_series: pd.Series):
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
    display_label = "scPRINT" if base_label == "scPrint" else base_label
    if degree_series is None or len(degree_series) == 0:
        return f"{display_label} (NA)"
    tail_prob = float((degree_series.values >= int(k_threshold)).mean())
    return f"{display_label} ({tail_prob:.3f})"

def _to_undirected_degree_series(df: pd.DataFrame):
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
    extraction: str = DEFAULT_EXTRACTION,
):
    """绘制 无向图 CCDF 度分布"""
    if not data_dict and real_df is None:
        return None

    title_text = f"{dataset} | {EXTRACTION_DISPLAY.get(extraction, extraction)}"
    fig_deg, ax_deg = plt.subplots(1, 1, figsize=FIG_SIZE)
    has_deg = False

    # 画模型
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

    # 画真实网络
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

# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser(description="Plot undirected CCDF degree distributions.")
    parser.add_argument(
        "--extraction",
        type=str,
        default=DEFAULT_EXTRACTION,
        choices=VALID_EXTRACTIONS,
        help="Extraction subdir under evl_omipath (e.g., emb500, att500, embhidden500).",
    )
    args = parser.parse_args()

    data_dir = EVL_ROOT / f"output_{args.extraction}"
    output_dir = Path(__file__).resolve().parent / f"ccdf_{args.extraction}"
    output_dir.mkdir(exist_ok=True)

    global DATA_DIR, OUTPUT_DIR
    DATA_DIR = data_dir
    OUTPUT_DIR = output_dir

    print(f"正在生成无向图 CCDF 图表，输出目录: {OUTPUT_DIR}")
    if not DATA_DIR.exists():
        print(f"模型目录不存在: {DATA_DIR}")
        return

    for dataset in DATASETS:
        print(f"\n处理: {dataset} ...")
        real_df, real_gene1_set, real_gene_union_set, real_edge_count = load_real_data(dataset)
        
        if real_df is None:
            real_gene1_set = None
            real_gene_union_set = None
            target_edge_count = DEFAULT_TOP_N
        else:
            target_edge_count = real_edge_count
            print(f"  ✔ 目标边数: {target_edge_count} (匹配真实网络)")

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

        plot_ccdf_distribution(
            dataset,
            data_dict,
            real_df=real_df,
            extraction=args.extraction,
        )

    print("\n完成。")

if __name__ == "__main__":
    main()