#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
左侧：11 个交集 TF 在 STRING hESC 参考网中的诱导子图（仅 STRING 边、两端均在核心集）。
右侧：各 TF 在 **完整 STRING 参考网** 中的度（无向图：邻居数；无向时入度 = 出度，下称 STRING 度）。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

from fig3_palette import model_color

BASE = Path(__file__).resolve().parent
STRING_PATH = Path("/mnt/10T/yzn/benchmark_GRN/input_process/STRING/hESC_processed-network.csv")
CORE_STATS = BASE / "tf_vene" / "tf_overlap_detailed_statistics.csv"
OUTPUT_PATH = BASE / "network" / "core11_tf_network_with_outdegree.pdf"

sns.set_style("white")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11


def load_string_graph(path: Path) -> nx.Graph:
    df = pd.read_csv(path)
    g1 = df["Gene1"].astype(str).str.strip().str.upper()
    g2 = df["Gene2"].astype(str).str.strip().str.upper()
    G = nx.Graph()
    for a, b in zip(g1, g2):
        if a != b:
            G.add_edge(a, b)
    return G


def main() -> None:
    if not CORE_STATS.is_file():
        raise FileNotFoundError(f"缺少核心 TF 表: {CORE_STATS}")
    if not STRING_PATH.is_file():
        raise FileNotFoundError(f"缺少 STRING 边表: {STRING_PATH}")

    core_df = pd.read_csv(CORE_STATS)
    core_order = [str(x).strip().upper() for x in core_df["TF"].tolist()]
    core_set = set(core_order)

    G_full = load_string_graph(STRING_PATH)

    missing = [tf for tf in core_order if tf not in G_full]
    if missing:
        raise ValueError(f"以下核心 TF 不在 STRING 图中: {missing}")

    # 全 STRING 网上的度
    deg_full = {tf: int(G_full.degree(tf)) for tf in core_order}
    # 11 人诱导子图（仅画图用）
    G11 = G_full.subgraph(core_set).copy()

    rows_tbl = [(tf, deg_full[tf]) for tf in core_order]
    rows_tbl.sort(key=lambda x: x[1], reverse=True)

    fig = plt.figure(figsize=(14.0, 7.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 0.78], wspace=0.22)
    ax_net = fig.add_subplot(gs[0, 0])
    ax_tab = fig.add_subplot(gs[0, 1])
    ax_tab.axis("off")

    pos = nx.spring_layout(G11, k=1.6, iterations=220, seed=42, weight=None)
    c_str = model_color("STRING")
    nx.draw_networkx_edges(
        G11, pos, ax=ax_net, edge_color="#9A9A9A", width=1.1, alpha=0.75
    )
    nx.draw_networkx_nodes(
        G11,
        pos,
        ax=ax_net,
        node_color=c_str,
        node_size=620,
        edgecolors="#3D3D3D",
        linewidths=1.0,
        alpha=0.92,
    )
    ys_n = [p[1] for p in pos.values()]
    dy = 0.022 * ((max(ys_n) - min(ys_n)) or 1.0)
    for n in G11.nodes:
        x, y = pos[n]
        ax_net.text(
            x,
            y - dy,
            n,
            fontsize=11,
            fontweight="bold",
            color="#111111",
            ha="center",
            va="top",
            clip_on=True,
        )
    ax_net.axis("off")
    ax_net.set_aspect("equal")
    ax_net.set_title("Core 11 — STRING hESC induced subgraph", fontsize=12, pad=10)

    header = ["TF", "STRING degree\n(full hESC graph)"]
    cells = [[tf, str(d)] for tf, d in rows_tbl]
    table = ax_tab.table(
        cellText=cells,
        colLabels=header,
        loc="center",
        cellLoc="center",
        colColours=["#E8E8E8"] * 2,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.08, 2.05)

    note = (
        "STRING is undirected: degree = neighbor count;\n"
        "in-degree equals out-degree, so we report STRING degree."
    )
    ax_tab.text(
        0.5,
        0.08,
        note,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#333333",
        transform=ax_tab.transAxes,
        linespacing=1.35,
    )

    fig.suptitle(
        "hESC — 11 core TFs: STRING subgraph & STRING degree",
        fontsize=13,
        y=0.98,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(top=0.88, bottom=0.08)
    plt.savefig(OUTPUT_PATH, bbox_inches="tight", format="pdf")
    plt.close()

    print(f"✅ 已保存 {OUTPUT_PATH}")
    print(f"   STRING full |V|={G_full.number_of_nodes()} |E|={G_full.number_of_edges()}")
    print(f"   core-11 induced |E|={G11.number_of_edges()}")
    for tf, d in rows_tbl:
        print(f"   {tf:8s}  STRING degree = {d}")


if __name__ == "__main__":
    main()
