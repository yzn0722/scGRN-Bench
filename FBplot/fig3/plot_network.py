from __future__ import annotations

from collections import defaultdict
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from fig3_palette import method_color

BASE = Path(__file__).resolve().parent

# 预测边（全图）；诱导子图仅保留端点均在「韦恩并集」内的边
pred_network_path = Path("/mnt/10T/yzn/benchmark_GRN/evl_omipath/output_emb500/scgpt/scgpt_hESC.tsv")
# 每个方法 top-K 合并为并集；与 venen.py 中 tf_top100_* 一致
overlap_dir = BASE / "tf_overlap"
dataset = "hESC"
csv_top = 100
venn_top_n = 100
method_csv_keys = ["scGPT-att500", "scGPT-emb500", "scGPT-embhidden500"]
# 交集核心 TF（11 个）：优先本地 venen 输出表
core_tf_stats_path = BASE / "tf_vene" / "tf_overlap_detailed_statistics.csv"
# 过滤后的核心子网（含 nodes 表中的 Degree）
filtered_nodes_path = BASE / "network" / "filtered_core_network_nodes.csv"
filtered_edges_path = BASE / "network" / "filtered_core_network_edges.csv"

# 预测 Gene1→Gene2：只保留两端都在 TF 池中的边（TF–TF），靶点不进图。
# "union"：韦恩并集 TF；"core"：交集 11 TF（预测 TF–TF）。
# "filtered_core"：network/filtered_core_* 中 11 个核心 TF + 表列 Degree + 仅 TF–TF 边。
PLOT_NODE_SET = "filtered_core"

# scGPT 预测在并集 TF 上往往近乎「完全图」：n 个节点的无向边理论上最多 C(n,2)=n(n-1)/2。
# 例如 n=155 时上界恰为 11935。若只要画较强作用，设下边阈值（无序对取双向重复的 max EdgeWeight）。
EDGE_WEIGHT_MIN = None  # 例如 0.08 ~ 0.15 可明显稀疏化

OUTPUT_DIR = BASE / "network"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_output_map = {
    "core": OUTPUT_DIR / "core11_tf_tf_network.pdf",
    "union": OUTPUT_DIR / "union_tf_network_core_bold.pdf",
    "filtered_core": OUTPUT_DIR / "filtered_core11_tf_network_degree.pdf",
}
output_path = _output_map.get(PLOT_NODE_SET, OUTPUT_DIR / "union_tf_network_core_bold.pdf")

sns.set_style("white")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11


def _load_union_tfs() -> set[str]:
    union: set[str] = set()
    for m in method_csv_keys:
        p = overlap_dir / f"tf_top{csv_top}_jaccard_{dataset}_top{csv_top}_TFs_{m}.csv"
        df = pd.read_csv(p)
        tf_col = next(c for c in df.columns if "TF" in c or "tf" in c.lower())
        union.update(df[tf_col].head(venn_top_n).astype(str).str.strip())
    return union


def _load_core_tfs() -> set[str]:
    if core_tf_stats_path.is_file():
        df = pd.read_csv(core_tf_stats_path)
        return set(df["TF"].astype(str).str.strip())
    return set()


def _find_weight_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        cl = str(c).lower()
        if cl in ("edgeweight", "edge_weight", "weight", "score"):
            return c
    return None


def _build_graph_tf_tf(
    pred_df: pd.DataFrame,
    tf_pool: set[str],
    *,
    edge_weight_min: float | None,
) -> nx.Graph:
    """诱导子图：端点均在 tf_pool；可选按无序对 max(EdgeWeight) 阈值去掉弱边。"""
    G = nx.Graph()
    if edge_weight_min is None:
        for _, row in pred_df.iterrows():
            g1 = str(row["Gene1"]).strip()
            g2 = str(row["Gene2"]).strip()
            if g1 not in tf_pool or g2 not in tf_pool:
                continue
            G.add_edge(g1, g2)
    else:
        wcol = _find_weight_col(pred_df)
        best: dict[tuple[str, str], float] = defaultdict(lambda: float("-inf"))
        for _, row in pred_df.iterrows():
            g1 = str(row["Gene1"]).strip()
            g2 = str(row["Gene2"]).strip()
            if g1 not in tf_pool or g2 not in tf_pool:
                continue
            a, b = sorted((g1, g2))
            w = float(row[wcol]) if wcol else 1.0
            key = (a, b)
            if w > best[key]:
                best[key] = w
        for (a, b), w in best.items():
            if w >= edge_weight_min:
                G.add_edge(a, b, weight=w)

    for n in tf_pool:
        G.add_node(n)
    return G


def _plot_filtered_core_network() -> None:
    """11 核心 TF：边来自 filtered_core_network_edges（仅 TF–TF）；度数字来自 filtered_core_network_nodes。"""
    if not filtered_nodes_path.is_file():
        raise FileNotFoundError(f"缺少节点表: {filtered_nodes_path}")
    if not filtered_edges_path.is_file():
        raise FileNotFoundError(f"缺少边表: {filtered_edges_path}")

    ndf = pd.read_csv(filtered_nodes_path)
    core_mask = ndf["IsCoreTF"].astype(str).str.lower().isin(("true", "1", "yes"))
    tdf = ndf.loc[core_mask].copy()
    tdf["Gene"] = tdf["Gene"].astype(str).str.strip()
    core_set = set(tdf["Gene"].tolist())
    degree_map = dict(zip(tdf["Gene"], tdf["Degree"].astype(int)))

    edf = pd.read_csv(filtered_edges_path)
    et = edf["EdgeType"].astype(str).str.upper()
    edf = edf.loc[et.str.contains("TF-TF")]

    G = nx.Graph()
    G.add_nodes_from(core_set)
    for _, row in edf.iterrows():
        g1 = str(row["Gene1"]).strip()
        g2 = str(row["Gene2"]).strip()
        if g1 in core_set and g2 in core_set:
            G.add_edge(g1, g2)

    pos = nx.spring_layout(G, k=1.7, iterations=220, seed=42, weight=None)

    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#8FB4DC", width=1.15, alpha=0.7)
    deg_vals = [degree_map.get(n, 0) for n in G.nodes]
    smin, smax = min(deg_vals), max(deg_vals)
    span = max(smax - smin, 1)
    sizes = [400 + 520 * (degree_map.get(n, 0) - smin) / span for n in G.nodes]
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=method_color("emb500"),
        node_size=sizes,
        edgecolors="#2E5F8A",
        linewidths=1.1,
        alpha=0.95,
    )

    ys = [p[1] for p in pos.values()]
    xs = [p[0] for p in pos.values()]
    yspan = max(ys) - min(ys) or 1.0
    xspan = max(xs) - min(xs) or 1.0
    span = max(yspan, xspan)
    # 基因名在圆心；Degree 画在圆下方（数据坐标近似偏移）
    dy_num = 0.038 * span
    for n in G.nodes:
        x, y = pos[n]
        d = degree_map.get(n, 0)
        ax.text(
            x,
            y,
            n,
            fontsize=9,
            fontweight="bold",
            color="#0A1628",
            ha="center",
            va="center",
            clip_on=True,
            zorder=10,
        )
        ax.text(
            x,
            y - dy_num,
            str(d),
            fontsize=10,
            fontweight="normal",
            color="#222222",
            ha="center",
            va="top",
            clip_on=True,
            zorder=10,
        )

    ax.axis("off")
    ax.set_aspect("equal")
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", format="pdf")
    plt.close()

    print(f"✅ 已保存（filtered_core + Degree 标注）：{output_path}")
    print(f"   节点 {G.number_of_nodes()}, TF–TF 边 {G.number_of_edges()}")
    for g in sorted(degree_map, key=lambda x: -degree_map[x]):
        print(f"   {g:8s}  Degree={degree_map[g]}")


def main():
    if PLOT_NODE_SET == "filtered_core":
        _plot_filtered_core_network()
        return

    union_tfs = _load_union_tfs()
    core_tfs = _load_core_tfs()
    if not core_tfs:
        raise FileNotFoundError(
            f"未找到核心 TF 表: {core_tf_stats_path}（请先运行 venen.py 生成）"
        )

    pred_df = pd.read_csv(pred_network_path, sep="\t")
    if "Gene1" not in pred_df.columns or "Gene2" not in pred_df.columns:
        pred_df = pred_df.iloc[:, :2].copy()
        pred_df.columns = ["Gene1", "Gene2"]

    tf_pool = core_tfs if PLOT_NODE_SET == "core" else union_tfs

    G = _build_graph_tf_tf(pred_df, tf_pool, edge_weight_min=EDGE_WEIGHT_MIN)

    # 并集模式去掉无边孤立 TF；交集模式保留全部 11 个核心节点（即使暂无 TF–TF 边）
    if PLOT_NODE_SET != "core":
        G.remove_nodes_from(list(nx.isolates(G)))

    pos = nx.spring_layout(G, k=2.8 / max(len(G) ** 0.25, 1), iterations=250, seed=42)

    fig, ax = plt.subplots(figsize=(18, 18))
    edge_color = "#B8D4EC"
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_color, width=0.6, alpha=0.55)

    non_core = [n for n in G.nodes if n not in core_tfs]
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        nodelist=non_core,
        node_color="#E8EEF3",
        edgecolors="#CCCCCC",
        linewidths=0.4,
        node_size=120,
        alpha=0.92,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        nodelist=[n for n in G.nodes if n in core_tfs],
        node_color=method_color("emb500"),
        edgecolors="#2E5F8A",
        linewidths=1.2,
        node_size=380,
        alpha=0.95,
    )

    if pos:
        ys = [p[1] for p in pos.values()]
        yspan = max(ys) - min(ys)
        dy = 0.018 * yspan if yspan > 0 else 0.02
    else:
        dy = 0.02
    for n in G.nodes:
        x, y = pos[n]
        bold = n in core_tfs
        ax.text(
            x,
            y - dy,
            n,
            fontsize=11 if bold else 7,
            fontweight="bold" if bold else "normal",
            color="#0A1628" if bold else "#333333",
            ha="center",
            va="top",
            clip_on=True,
        )

    ax.axis("off")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", format="pdf")
    plt.close()

    scope = "交集 TF（仅 TF–TF 边，无靶点）" if PLOT_NODE_SET == "core" else "并集 TF（仅 TF–TF 边，无靶点）"
    print(f"✅ {scope} 网络已保存：{output_path}")
    n_n = G.number_of_nodes()
    m_n = G.number_of_edges()
    max_simple = n_n * (n_n - 1) // 2 if n_n > 1 else 0
    dens = (m_n / max_simple) if max_simple else 0.0
    print(
        f"   节点数 {n_n}, 边数 {m_n}, 核心 TF（加粗） {len(core_tfs & set(G.nodes))}"
    )
    print(
        f"   无向简单图边数上界 C(n,2)={max_simple}, 当前密度 {dens:.3f}"
        + ("（≈1 表示预测在并集上近似完全图）" if PLOT_NODE_SET == "union" and dens > 0.95 else "")
    )
    if EDGE_WEIGHT_MIN is not None:
        print(f"   已启用 EDGE_WEIGHT_MIN={EDGE_WEIGHT_MIN}（按无序对 max 权重过滤）")


if __name__ == "__main__":
    main()
