# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Run ORA enrichment from 11 TF genes and plot two PDFs:
# 1) lollipop of top pathways
# 2) gene-concept network
# """

# from __future__ import annotations

# import math
# from pathlib import Path
# from typing import Dict, List, Set

# import matplotlib.pyplot as plt
# import networkx as nx
# import numpy as np
# import pandas as pd

# from fig3_palette import model_color


# # ====== Input paths ======
# TF_CSV = Path("/mnt/10T/yzn/FoundBench/FBplot/fig3/tf_vene/tf_overlap_detailed_statistics.csv")
# GMT_FILES = [
#     Path("/mnt/10T/yzn/benchmark_GRN/fuji/c2.cp.v2025.1.Hs.symbols.gmt"),
#     Path("/mnt/10T/yzn/benchmark_GRN/fuji/h.all.v2025.1.Hs.symbols.gmt"),
# ]
# OUTDIR = Path("/mnt/10T/yzn/FoundBench/FBplot/fig3/tf_vene")
# TOP_N = 15


# def read_gmt(path: Path) -> Dict[str, Set[str]]:
#     pathways: Dict[str, Set[str]] = {}
#     with path.open("r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.rstrip("\n").split("\t")
#             if len(parts) < 3:
#                 continue
#             term = parts[0].strip()
#             genes = {g.strip().upper() for g in parts[2:] if g.strip()}
#             if term and genes:
#                 pathways[term] = genes
#     return pathways


# def hypergeom_p_over(N: int, K: int, n: int, k: int) -> float:
#     # P(X >= k), X~Hypergeom(N, K, n)
#     if k > min(K, n):
#         return 1.0
#     denom = math.comb(N, n)
#     s = 0.0
#     for i in range(k, min(K, n) + 1):
#         s += (math.comb(K, i) * math.comb(N - K, n - i)) / denom
#     return float(min(max(s, 0.0), 1.0))


# def bh_fdr(pvals: List[float]) -> List[float]:
#     m = len(pvals)
#     if m == 0:
#         return []
#     order = np.argsort(pvals)
#     q = np.ones(m, dtype=float)
#     prev = 1.0
#     for rank, idx in enumerate(order[::-1], start=1):
#         j = m - rank + 1
#         val = pvals[idx] * m / j
#         prev = min(prev, val)
#         q[idx] = prev
#     return q.tolist()


# def run_ora(query_genes: Set[str], gmt_path: Path) -> pd.DataFrame:
#     pathways = read_gmt(gmt_path)
#     bg = set().union(*pathways.values()) if pathways else set()
#     q = {g.upper() for g in query_genes} & bg
#     N = len(bg)
#     n = len(q)
#     rows = []
#     for term, gs in pathways.items():
#         overlap = q & gs
#         k = len(overlap)
#         if k == 0:
#             continue
#         K = len(gs)
#         p = hypergeom_p_over(N, K, n, k)
#         rows.append(
#             {
#                 "DB": gmt_path.name,
#                 "Term": term,
#                 "PValue": p,
#                 "Overlap": k,
#                 "SetSize": K,
#                 "QuerySize": n,
#                 "BackgroundSize": N,
#                 "Genes": ";".join(sorted(overlap)),
#             }
#         )
#     df = pd.DataFrame(rows)
#     if df.empty:
#         return df
#     df["FDR_BH"] = bh_fdr(df["PValue"].tolist())
#     df["neglog10P"] = -np.log10(np.clip(df["PValue"].to_numpy(dtype=float), 1e-300, None))
#     return df.sort_values(["FDR_BH", "PValue", "Overlap"], ascending=[True, True, False]).reset_index(drop=True)


# def plot_lollipop(df: pd.DataFrame, out_pdf: Path, top_n: int) -> None:
#     d = df.head(top_n).copy().iloc[::-1]
#     fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
#     y = np.arange(len(d))
#     x = d["neglog10P"].to_numpy(dtype=float)
#     ax.hlines(y, 0, x, color=model_color("scCello"), linewidth=2)
#     ax.scatter(x, y, s=90, color=model_color("scGPT"), edgecolors="none", zorder=3)
#     ax.set_yticks(y)
#     ax.set_yticklabels(d["Term"].tolist(), fontsize=11)
#     ax.set_xlabel(r"$-\log_{10}(P)$", fontsize=14)
#     ax.set_title("ORA Top Pathways", fontsize=14, color="#000000")
#     ax.grid(False)
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     fig.tight_layout()
#     fig.savefig(out_pdf, bbox_inches="tight", format="pdf")
#     plt.close(fig)


# def plot_gene_concept(df: pd.DataFrame, out_pdf: Path, top_n: int) -> None:
#     d = df.head(top_n).copy()
#     g = nx.Graph()
#     pathway_nodes = []
#     gene_nodes = set()
#     for _, row in d.iterrows():
#         p = f"path::{row['Term']}"
#         pathway_nodes.append(p)
#         g.add_node(p, kind="path", label=row["Term"])
#         genes = [x.strip() for x in str(row["Genes"]).split(";") if x.strip()]
#         for gene in genes:
#             n = f"gene::{gene}"
#             gene_nodes.add(n)
#             g.add_node(n, kind="gene", label=gene)
#             g.add_edge(p, n)
#     if g.number_of_nodes() == 0:
#         return

#     pos = {}
#     for i, n in enumerate(pathway_nodes):
#         pos[n] = (0.0, -i)
#     genes_sorted = sorted(gene_nodes)
#     scale = max(1.0, len(pathway_nodes) / max(1, len(genes_sorted)))
#     for i, n in enumerate(genes_sorted):
#         pos[n] = (1.0, -i * scale)

#     fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
#     nx.draw_networkx_edges(g, pos, ax=ax, edge_color=model_color("STRING"), width=1.0, alpha=0.6)
#     nx.draw_networkx_nodes(g, pos, nodelist=pathway_nodes, node_size=550, node_color=model_color("LangCell"), edgecolors="none", ax=ax)
#     nx.draw_networkx_nodes(g, pos, nodelist=genes_sorted, node_size=240, node_color=model_color("scGPT"), edgecolors="none", ax=ax)
#     labels = {n: g.nodes[n]["label"] for n in g.nodes}
#     nx.draw_networkx_labels(g, pos, labels=labels, font_size=10, font_color="#111111", ax=ax)
#     ax.set_title("Gene-Concept Network", fontsize=14, color="#000000")
#     ax.axis("off")
#     fig.tight_layout()
#     fig.savefig(out_pdf, bbox_inches="tight", format="pdf")
#     plt.close(fig)


# def main() -> None:
#     OUTDIR.mkdir(parents=True, exist_ok=True)
#     tf_df = pd.read_csv(TF_CSV)
#     if "TF" not in tf_df.columns:
#         raise ValueError(f"Missing TF column in {TF_CSV}")
#     query = {str(g).strip().upper() for g in tf_df["TF"].dropna().tolist() if str(g).strip()}
#     if not query:
#         raise ValueError("Empty TF query set.")

#     all_parts = []
#     for gmt in GMT_FILES:
#         if not gmt.is_file():
#             print(f"[WARN] GMT not found: {gmt}")
#             continue
#         part = run_ora(query, gmt)
#         if not part.empty:
#             all_parts.append(part)

#     if not all_parts:
#         raise RuntimeError("No enrichment hits found from provided GMT files.")

#     res = pd.concat(all_parts, ignore_index=True)
#     res = res.sort_values(["FDR_BH", "PValue", "Overlap"], ascending=[True, True, False]).reset_index(drop=True)

#     out_csv = OUTDIR / "tf_overlap_11genes_ora.csv"
#     out_lolli = OUTDIR / "tf_overlap_11genes_ora_lollipop.pdf"
#     out_net = OUTDIR / "tf_overlap_11genes_ora_gene_concept.pdf"

#     res.to_csv(out_csv, index=False)
#     plot_lollipop(res, out_lolli, TOP_N)
#     plot_gene_concept(res, out_net, TOP_N)

#     print(f"Saved: {out_csv}")
#     print(f"Saved: {out_lolli}")
#     print(f"Saved: {out_net}")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run ORA enrichment from 11 TF genes and plot two PDFs:
1) lollipop of top pathways (cleaned & shortened names)
2) gene-concept network
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Set

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from fig3_palette import model_color

TEXT_SIZE = 16


# ====== Input paths ======
TF_CSV = Path("/mnt/10T/yzn/FoundBench/FBplot/fig3/tf_vene/tf_overlap_detailed_statistics.csv")
GMT_FILES = [
    Path("/mnt/10T/yzn/benchmark_GRN/fuji/c2.cp.v2025.1.Hs.symbols.gmt"),
    Path("/mnt/10T/yzn/benchmark_GRN/fuji/h.all.v2025.1.Hs.symbols.gmt"),
]
OUTDIR = Path("/mnt/10T/yzn/FoundBench/FBplot/fig3/tf_vene")
TOP_N = 15


def read_gmt(path: Path) -> Dict[str, Set[str]]:
    pathways: Dict[str, Set[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            term = parts[0].strip()
            genes = {g.strip().upper() for g in parts[2:] if g.strip()}
            if term and genes:
                pathways[term] = genes
    return pathways


def hypergeom_p_over(N: int, K: int, n: int, k: int) -> float:
    if k > min(K, n):
        return 1.0
    denom = math.comb(N, n)
    s = 0.0
    for i in range(k, min(K, n) + 1):
        s += (math.comb(K, i) * math.comb(N - K, n - i)) / denom
    return float(min(max(s, 0.0), 1.0))


def bh_fdr(pvals: List[float]) -> List[float]:
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    q = np.ones(m, dtype=float)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        j = m - rank + 1
        val = pvals[idx] * m / j
        prev = min(prev, val)
        q[idx] = prev
    return q.tolist()


def run_ora(query_genes: Set[str], gmt_path: Path) -> pd.DataFrame:
    pathways = read_gmt(gmt_path)
    bg = set().union(*pathways.values()) if pathways else set()
    q = {g.upper() for g in query_genes} & bg
    N = len(bg)
    n = len(q)
    rows = []
    for term, gs in pathways.items():
        overlap = q & gs
        k = len(overlap)
        if k == 0:
            continue
        K = len(gs)
        p = hypergeom_p_over(N, K, n, k)
        rows.append(
            {
                "DB": gmt_path.name,
                "Term": term,
                "PValue": p,
                "Overlap": k,
                "SetSize": K,
                "QuerySize": n,
                "BackgroundSize": N,
                "Genes": ";".join(sorted(overlap)),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["FDR_BH"] = bh_fdr(df["PValue"].tolist())
    df["neglog10P"] = -np.log10(np.clip(df["PValue"].to_numpy(dtype=float), 1e-300, None))
    return df.sort_values(["FDR_BH", "PValue", "Overlap"], ascending=[True, True, False]).reset_index(drop=True)


# ===================== 美化通路名称（核心优化）=====================
def clean_pathway_name(name: str, max_len: int = 38) -> str:
    # 去掉数据库前缀
    name = re.sub(r"^(REACTOME_|KEGG_|WP_|HALLMARK_|BIOCARTA_)", "", name)
    # 下划线转空格
    name = name.replace("_", " ")
    # 过长自动截断
    if len(name) > max_len:
        name = name[:max_len] + "…"
    return name


def plot_lollipop(df: pd.DataFrame, out_pdf: Path, top_n: int) -> None:
    d = df.head(top_n).copy().iloc[::-1]
    # 美化名称
    d["Term_clean"] = d["Term"].apply(clean_pathway_name)
    
    # 加宽画布，更适合展示
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    y = np.arange(len(d))
    x = d["neglog10P"].to_numpy(dtype=float)
    
    ax.hlines(y, 0, x, color=model_color("scCello"), linewidth=2)
    ax.scatter(x, y, s=90, color=model_color("scGPT"), edgecolors="none", zorder=3)
    
    ax.set_yticks(y)
    # 使用清洗后的名称
    ax.set_yticklabels(d["Term_clean"].tolist(), fontsize=TEXT_SIZE)
    ax.set_xlabel(r"$-\log_{10}(P)$", fontsize=TEXT_SIZE)
    ax.tick_params(axis="x", labelsize=TEXT_SIZE)
    #ax.set_title("ORA Top Pathways", fontsize=14, color="#000000")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", format="pdf")
    plt.close(fig)


def plot_gene_concept(df: pd.DataFrame, out_pdf: Path, top_n: int) -> None:
    d = df.head(top_n).copy()
    g = nx.Graph()
    pathway_nodes = []
    gene_nodes = set()
    for _, row in d.iterrows():
        p = f"path::{row['Term']}"
        pathway_nodes.append(p)
        g.add_node(p, kind="path", label=row['Term'])
        genes = [x.strip() for x in str(row["Genes"]).split(";") if x.strip()]
        for gene in genes:
            n = f"gene::{gene}"
            gene_nodes.add(n)
            g.add_node(n, kind="gene", label=gene)
            g.add_edge(p, n)
    if g.number_of_nodes() == 0:
        return

    pos = {}
    for i, n in enumerate(pathway_nodes):
        pos[n] = (0.0, -i)
    genes_sorted = sorted(gene_nodes)
    scale = max(1.0, len(pathway_nodes) / max(1, len(genes_sorted)))
    for i, n in enumerate(genes_sorted):
        pos[n] = (1.0, -i * scale)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=model_color("STRING"), width=1.0, alpha=0.6)
    nx.draw_networkx_nodes(g, pos, nodelist=pathway_nodes, node_size=550, node_color=model_color("LangCell"), edgecolors="none", ax=ax)
    nx.draw_networkx_nodes(g, pos, nodelist=genes_sorted, node_size=240, node_color=model_color("scGPT"), edgecolors="none", ax=ax)
    labels = {n: g.nodes[n]["label"] for n in g.nodes}
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=TEXT_SIZE, font_color="#111111", ax=ax)
    ax.set_title("Gene-Concept Network", fontsize=TEXT_SIZE, color="#000000")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", format="pdf")
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tf_df = pd.read_csv(TF_CSV)
    if "TF" not in tf_df.columns:
        raise ValueError(f"Missing TF column in {TF_CSV}")
    query = {str(g).strip().upper() for g in tf_df["TF"].dropna().tolist() if str(g).strip()}
    if not query:
        raise ValueError("Empty TF query set.")

    all_parts = []
    for gmt in GMT_FILES:
        if not gmt.is_file():
            print(f"[WARN] GMT not found: {gmt}")
            continue
        part = run_ora(query, gmt)
        if not part.empty:
            all_parts.append(part)

    if not all_parts:
        raise RuntimeError("No enrichment hits found from provided GMT files.")

    res = pd.concat(all_parts, ignore_index=True)
    res = res.sort_values(["FDR_BH", "PValue", "Overlap"], ascending=[True, True, False]).reset_index(drop=True)

    out_csv = OUTDIR / "tf_overlap_11genes_ora.csv"
    out_lolli = OUTDIR / "tf_overlap_11genes_ora_lollipop.pdf"
    out_net = OUTDIR / "tf_overlap_11genes_ora_gene_concept.pdf"

    res.to_csv(out_csv, index=False)
    plot_lollipop(res, out_lolli, TOP_N)
    plot_gene_concept(res, out_net, TOP_N)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_lolli}")
    print(f"Saved: {out_net}")


if __name__ == "__main__":
    main()