#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""绘制 STRING hESC 参考网中「TF–TF」诱导子图（TF 名单来自 grnboost hESC_tf_list）。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

from fig3_palette import method_color, model_color

BASE = Path(__file__).resolve().parent

STRING_EDGES = Path(
    "/mnt/10T/yzn/benchmark_GRN/input_process/STRING/hESC_processed-network.csv"
)
STRING_TF_LIST = Path(
    "/mnt/10T/yzn/benchmark_GRN/input_process/STRING/grnboost_results/hESC_tf_list.txt"
)
CORE_TF_STATS = BASE / "tf_vene" / "tf_overlap_detailed_statistics.csv"

OUTPUT_DIR = BASE / "network"
OUTPUT_PATH = OUTPUT_DIR / "string_hESC_all_tf_network.pdf"

sns.set_style("white")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def load_tf_set(tf_path: Path) -> set[str]:
    return {line.strip().upper() for line in tf_path.read_text().splitlines() if line.strip()}


def build_string_tf_graph(edges_path: Path, tf_set: set[str]) -> nx.Graph:
    df = pd.read_csv(edges_path)
    g1 = df["Gene1"].astype(str).str.strip().str.upper()
    g2 = df["Gene2"].astype(str).str.strip().str.upper()
    string_nodes = set(g1) | set(g2)
    tfs = tf_set & string_nodes

    G = nx.Graph()
    G.add_nodes_from(tfs)
    for a, b in zip(g1, g2):
        if a in tfs and b in tfs:
            G.add_edge(a, b)
    return G


def main() -> None:
    tf_set = load_tf_set(STRING_TF_LIST)
    G = build_string_tf_graph(STRING_EDGES, tf_set)

    core: set[str] = set()
    if CORE_TF_STATS.is_file():
        core_df = pd.read_csv(CORE_TF_STATS)
        core = set(core_df["TF"].astype(str).str.strip().str.upper()) & set(G.nodes)

    n = G.number_of_nodes()
    m = G.number_of_edges()
    n_comp = nx.number_connected_components(G)
    max_e = n * (n - 1) // 2 if n > 1 else 0
    print(
        f"STRING TF–TF 子图: {n} 个 TF 节点, {m} 条边, {n_comp} 个连通分量, "
        f"密度 {m / max_e:.3f}（相对简单图满图 C(n,2)）"
    )

    k = 2.2 / (n**0.22) if n else 1.0
    pos = nx.spring_layout(G, k=k, iterations=120, seed=42, weight=None)

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.axis("off")

    nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color="#B0B0B0", width=0.35, alpha=0.12, style="-"
    )

    other = [v for v in G.nodes if v not in core]
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=other,
        ax=ax,
        node_size=25,
        node_color=model_color("STRING"),
        edgecolors="none",
        alpha=0.55,
    )
    if core:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=list(core),
            ax=ax,
            node_size=180,
            node_color=method_color("emb500"),
            edgecolors="#2E4A66",
            linewidths=0.8,
            alpha=0.95,
        )

    if pos and core:
        ys = [p[1] for p in pos.values()]
        yspan = max(ys) - min(ys) or 1.0
        dy = 0.014 * yspan
        for name in sorted(core):
            if name not in pos:
                continue
            x, y = pos[name]
            ax.text(
                x,
                y - dy,
                name,
                fontsize=9,
                fontweight="bold",
                color="#0A1628",
                ha="center",
                va="top",
                clip_on=True,
            )

    ax.text(
        0.02,
        0.98,
        "STRING hESC — TF–TF subgraph\n"
        f"TF list: {STRING_TF_LIST.name} ({len(tf_set)} TFs)\n"
        f"Nodes in graph: {n} | Edges: {m}\n"
        + (f"Highlighted (overlap core): {len(core)} TFs\n" if core else ""),
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CCCCCC", alpha=0.92),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"✅ 已保存 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
