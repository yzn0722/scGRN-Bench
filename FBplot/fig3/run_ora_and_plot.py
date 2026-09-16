#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run ORA enrichment from 11 TF genes and plot lollipop only (no gene-concept network)
Top 15 pathways fully forced mapped for correct naming.
"""

from __future__ import annotations
import math
import re
from pathlib import Path
from typing import Dict, List, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fig3_palette import model_color

TEXT_SIZE = 16

# ====== Input paths ======
TF_CSV = Path("/mnt/10T/yzn/scGRN-Bench/FBplot/fig3/tf_vene/tf_overlap_detailed_statistics.csv")
GMT_FILES = [
    Path("/mnt/10T/yzn/benchmark_GRN/fuji/c2.cp.v2025.1.Hs.symbols.gmt"),
    Path("/mnt/10T/yzn/benchmark_GRN/fuji/h.all.v2025.1.Hs.symbols.gmt"),
]
OUTDIR = Path("/mnt/10T/yzn/scGRN-Bench/FBplot/fig3/tf_vene")
TOP_N = 15

# ====== 基本函数 ======
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

# ====== 强制映射 simplify_pathway_name（Top 15 全部映射） ======
def simplify_pathway_name(name: str) -> str:
    # 去掉数据库前缀和下划线
    name = re.sub(r"^(REACTOME_|KEGG_|WP_|HALLMARK_|BIOCARTA_)", "", name, flags=re.IGNORECASE)
    name = name.replace("_", " ")

    # Top 15 原始 GMT 名称 -> 映射后的规范名称
    forced_mapping = {
        "SARS COV 2 MODULATES HOST TRANSLATION MACHINERY": "SARS-CoV Host Translation",
        "RESOLUTION OF ABASIC SITES AP SITES": "AP Site Resolution",
        "BASE EXCISION REPAIR": "Base Excision Repair",
        "RESOLUTION OF AP SITES VIA THE MULTIPLE NUCLEOTIDE PATCH REPLACEMENT PATHWAY": "AP Site Repair",
        "SYSTEMIC LUPUS ERYTHEMATOSUS": "Systemic Lupus Erythematosus",
        "SM PATHWAY": "Sm Complex",
        "MEDICUS REFERENCE BASE EXCISION AND STRAND CLEAVAGE BY BIFUNCTIONAL GLYCOSYLASE": "Base Excision Repair",
        "POLB DEPENDENT LONG PATCH BASE EXCISION REPAIR": "POLB Long Patch Repair",
        "METABOLISM OF RNA": "RNA Metabolism",
        "MRNA SPLICING MINOR PATHWAY": "mRNA Splicing",
        "SPLICEOSOME": "Spliceosome",
        "MYC TARGETS V1": "MYC Targets",
        "PROCESSING OF CAPPED INTRON CONTAINING PRE MRNA": "Pre-mRNA Processing",
        "MRNA SPLICING": "mRNA Splicing",
        "MRNA PROCESSING": "mRNA Processing",
        "TRANSCRIPTIONAL ACTIVITY OF SMAD2 SMAD3 SMAD4 HETEROTRIMER": "SMAD Transcription",
        "SNRNP ASSEMBLY":"snRNP Assembly"
    }

    # 全大写匹配
    return forced_mapping.get(name.upper(), name)

def process_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    df['Term_clean'] = df['Term'].apply(simplify_pathway_name)
    df_dedup = df.loc[df.groupby('Term_clean')['PValue'].idxmin()].copy()
    df_dedup = df_dedup.sort_values(['FDR_BH', 'PValue', 'Overlap'], ascending=[True, True, False]).reset_index(drop=True)
    print(f"去重前: {len(df)} 条通路，去重后: {len(df_dedup)} 条通路")
    return df_dedup

def plot_lollipop(df: pd.DataFrame, out_pdf: Path, top_n: int) -> None:
    d = df.head(top_n).copy().iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    y = np.arange(len(d))
    x = d["neglog10P"].to_numpy(dtype=float)
    ax.hlines(y, 0, x, color=model_color("scCello"), linewidth=2)
    ax.scatter(x, y, s=90, color=model_color("scGPT"), edgecolors="none", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(d["Term_clean"].tolist(), fontsize=TEXT_SIZE)
    ax.set_xlabel(r"$-\log_{10}(P)$", fontsize=18)
    ax.xaxis.label.set_size(18)
    ax.tick_params(axis="x", labelsize=TEXT_SIZE)
    ax.tick_params(axis="y", labelsize=TEXT_SIZE)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"\n棒棒糖图已保存: {out_pdf}")
    print("\n强制映射后的通路名称（Top {}）:".format(top_n))
    for idx, row in d.iterrows():
        print(f"  {row['Term_clean']}: -log10(P)={row['neglog10P']:.2f}")

def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tf_df = pd.read_csv(TF_CSV)
    if "TF" not in tf_df.columns:
        raise ValueError(f"Missing TF column in {TF_CSV}")
    query = {str(g).strip().upper() for g in tf_df["TF"].dropna().tolist() if str(g).strip()}
    print(f"查询基因数: {len(query)}")
    print(f"TF列表: {sorted(query)[:10]}...")
    if not query:
        raise ValueError("Empty TF query set.")
    all_parts = []
    for gmt in GMT_FILES:
        if not gmt.is_file():
            print(f"[WARN] GMT not found: {gmt}")
            continue
        print(f"处理GMT: {gmt.name}")
        part = run_ora(query, gmt)
        if not part.empty:
            print(f"  找到 {len(part)} 条富集通路")
            all_parts.append(part)
    if not all_parts:
        raise RuntimeError("No enrichment hits found from provided GMT files.")
    res = pd.concat(all_parts, ignore_index=True)
    res = res.sort_values(["FDR_BH", "PValue", "Overlap"], ascending=[True, True, False]).reset_index(drop=True)
    res = process_and_deduplicate(res)
    out_csv = OUTDIR / "tf_overlap_11genes_ora.csv"
    res.to_csv(out_csv, index=False)
    print(f"\nCSV已保存: {out_csv}")
    out_lolli = OUTDIR / "tf_overlap_11genes_ora_lollipop.pdf"
    plot_lollipop(res, out_lolli, TOP_N)

if __name__ == "__main__":
    main()