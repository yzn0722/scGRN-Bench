#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-attention-head GO enrichment with explicit background (thesis-ready).

Method (default: local hypergeometric + BH-FDR, no online API)
--------------------------------------------------------------
  - Library: GO Biological Process 2023 (Enrichr GMT, cached under output/.../cache/).
  - **Background** = expressed active genes in hESC (default), or CHIP universe (--background-mode chip).
  - **Query** [default topw]: from each head's top-K CHIP-aligned edges (K=|GT|),
      keep TF→target (AUPR filter), take top-N genes by summed edge weight (~100 genes).
  - Test: scipy hypergeometric; **BH-FDR**.

Gene-set modes (--gene-set-mode)
  topw       [default] AUPR edges → top --top-genes by weight (recommended)
  targets    Gene2 (in-hub targets) only, AUPR edges
  topw_raw   top --top-genes without AUPR filter (usually too diffuse, no enrichment)

Outputs: output/head_go_enrich/{model}_{dataset}/
  head{h}_GO_BP.csv                 full enrichment table
  head{h}_GO_BP_bar.pdf             one bar chart per head (thesis figure)
  {tag}_go_enrich_summary.csv       top-N terms per head (paste into thesis table)
  {tag}_go_bp_heatmap.pdf           heads × terms overview
  {tag}_go_bp_bars.pdf              8-panel overview

Install:
  pip install gseapy pandas matplotlib seaborn scipy

Run:
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig2/plot
  python enrich_head_go_gprofiler.py \\
    --models scgpt --dataset hESC \\
    --head-root /mnt/10T/yzn/scGRN-Bench/FBplot/fig2/att_head
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import analyze_attention_heads as ath  # noqa: E402
import eval_heads_chip_auprc as chip  # noqa: E402
from fig2_palette import model_color  # noqa: E402

plt.rcParams.update(ath.plt.rcParams)

GSEAPY_LIBS = {
    "GO:BP": "GO_Biological_Process_2023",
    "GO:MF": "GO_Molecular_Function_2023",
    "GO:CC": "GO_Cellular_Component_2023",
}
CACHE_DIR = _SCRIPT_DIR / "output" / "head_go_enrich" / "cache"


def _read_gmt(path: Path) -> Dict[str, List[str]]:
    gmt: Dict[str, List[str]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            gmt[parts[0]] = parts[2:]
    return gmt


def load_go_library(lib_name: str, cache_dir: Path, min_size: int, max_size: int) -> Dict[str, List[str]]:
    """Load Enrichr GMT from cache; download via curl if missing (no gseapy API)."""
    import subprocess
    import urllib.request

    cache_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = cache_dir / f"{lib_name}.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    gmt_path = cache_dir / f"Enrichr.{lib_name}.gmt"
    if not gmt_path.exists():
        url = (
            "https://maayanlab.cloud/Enrichr/geneSetLibrary"
            f"?mode=text&libraryName={lib_name}"
        )
        print(f"  [cache] downloading {lib_name} GMT (~1 min)...")
        try:
            subprocess.run(
                ["curl", "-fsSL", "--max-time", "600", "-o", str(gmt_path), url],
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            with urllib.request.urlopen(url, timeout=600) as resp:
                gmt_path.write_bytes(resp.read())

    lib = _read_gmt(gmt_path)
    lib = {k: v for k, v in lib.items() if min_size <= len(v) <= max_size}
    with open(pkl_path, "wb") as f:
        pickle.dump(lib, f)
    print(f"  [cache] {len(lib)} terms -> {pkl_path.name}")
    return lib


def filter_gmt_to_background(gmt: Dict[str, List[str]], background: set) -> Dict[str, List[str]]:
    out = {}
    for term, genes in gmt.items():
        g = [x for x in genes if x in background]
        if g:
            out[term] = g
    return out


def run_local_enrich(
    query_genes: List[str],
    background_genes: List[str],
    gmt: Dict[str, List[str]],
    padj_cutoff: float,
    min_overlap: int,
) -> pd.DataFrame:
    """Hypergeometric test + BH-FDR (gseapy.stats, fully offline)."""
    from gseapy.stats import calc_pvalues, multiple_testing_correction

    bg_set = set(background_genes)
    query_set = set(query_genes) & bg_set
    if len(query_set) < min_overlap:
        return pd.DataFrame()

    hg = list(calc_pvalues(query=list(query_set), gene_sets=gmt, background=bg_set))
    if not hg or len(hg[0]) == 0:
        return pd.DataFrame()

    terms, pvals, oddr, overlap_n, term_n, hit_genes = hg
    fdrs, _ = multiple_testing_correction(ps=list(pvals), alpha=padj_cutoff, method="benjamini-hochberg")

    rows = []
    for term, p, padj, oratio, x, m, genes in zip(terms, pvals, fdrs, oddr, overlap_n, term_n, hit_genes):
        if x < min_overlap:
            continue
        term_id = ""
        if "(GO:" in term:
            term_id = term.split("(GO:")[-1].rstrip(")").replace("GO:", "GO:")
            if not term_id.startswith("GO:"):
                term_id = "GO:" + term_id
        rows.append(
            {
                "term_name": term,
                "term_id": term_id,
                "pval": float(p),
                "padj": float(padj),
                "odds_ratio": float(oratio),
                "overlap_size": int(x),
                "term_size": int(m),
                "query_size": len(query_set),
                "background_size": len(bg_set),
                "genes": ";".join(sorted(genes)),
            }
        )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("padj")
    return df[df["padj"] <= padj_cutoff]


def topk_edge_df(edge_df: pd.DataFrame, k: int) -> pd.DataFrame:
    if edge_df.empty:
        return edge_df
    k = min(k, len(edge_df))
    return edge_df.nlargest(k, "EdgeWeight")


def genes_topw(edge_df: pd.DataFrame, k: int, n_genes: int) -> List[str]:
    """Top-N genes by summed incident EdgeWeight in head top-K subgraph."""
    from collections import defaultdict

    top = topk_edge_df(edge_df, k)
    if top.empty:
        return []
    w: Dict[str, float] = defaultdict(float)
    for row in top.itertuples(index=False):
        w[str(row.Gene1)] += float(row.EdgeWeight)
        w[str(row.Gene2)] += float(row.EdgeWeight)
    ranked = sorted(w, key=w.get, reverse=True)
    return ranked[: min(n_genes, len(ranked))]


def genes_targets(edge_df: pd.DataFrame, k: int) -> List[str]:
    top = topk_edge_df(edge_df, k)
    if top.empty:
        return []
    return sorted(set(top["Gene2"].astype(str)))


def plot_head_bar(df: pd.DataFrame, out_pdf: Path, head: int, display: str, dataset: str, color: str, n_top: int) -> None:
    sub = df.nsmallest(n_top, "padj").sort_values("padj", ascending=False)
    if sub.empty:
        return
    labels = [t[:55] + "…" if len(t) > 55 else t for t in sub["term_name"]]
    scores = -np.log10(sub["padj"].clip(lower=1e-300))
    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.35 * len(sub) + 1.5)))
    y = np.arange(len(sub))
    ax.barh(y, scores, color=color, edgecolor="k", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(r"$-\log_{10}$(BH-FDR)")
    ax.set_title(f"{display} — {dataset}  |  Head {head}\nGO:BP (top {len(sub)}, padj ≤ cutoff)")
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(summary_long: pd.DataFrame, out_pdf: Path, display: str, dataset: str, src: str) -> None:
    if summary_long.empty:
        return
    mat = summary_long.pivot(index="term_name", columns="Head", values="neglog10_padj").fillna(0.0)
    mat = mat.loc[mat.max(axis=1) > 0]
    if mat.empty:
        return
    mat = mat.loc[mat.max(axis=1).sort_values(ascending=False).head(25).index]
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * mat.shape[1] + 3), max(4, 0.28 * mat.shape[0] + 2)))
    sns.heatmap(mat, cmap="YlOrRd", ax=ax, cbar_kws={"label": r"$-\log_{10}$(BH-FDR)"}, linewidths=0.3)
    ax.set_title(f"{display} — {dataset}\nPer-head {src} enrichment")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_faceted_bars(top_long: pd.DataFrame, out_pdf: Path, display: str, dataset: str, color: str) -> None:
    if top_long.empty:
        return
    heads = sorted(top_long["Head"].unique())
    fig, axes = plt.subplots(1, len(heads), figsize=(3.0 * len(heads), 4.8))
    if len(heads) == 1:
        axes = [axes]
    for ax, h in zip(axes, heads):
        sub = top_long[top_long["Head"] == h].sort_values("padj").tail(8)
        if sub.empty:
            ax.axis("off")
            continue
        y = np.arange(len(sub))
        ax.barh(y, -np.log10(sub["padj"].clip(lower=1e-300)), color=color, edgecolor="k", linewidth=0.3)
        ax.set_yticks(y)
        ax.set_yticklabels([t[:40] + "…" if len(t) > 40 else t for t in sub["term_name"]], fontsize=7)
        ax.set_title(f"H{h}", fontsize=11)
        ax.invert_yaxis()
    fig.suptitle(f"{display} — {dataset}\nTop GO:BP terms per head", y=1.02)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_one(model: str, args: argparse.Namespace) -> None:
    display = ath.MODEL_ALIASES.get(ath.model_file_prefix(model), model)
    color = model_color(display)
    tag = f"{ath.model_file_prefix(model)}_{args.dataset}"
    out_dir = Path(args.output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    chip_root = Path(args.chip_root)
    gt = chip.read_chip_gt(chip_root / f"{args.dataset}_chip_matched-network.csv")
    expr = chip.read_chip_expr(chip_root / f"{args.dataset}_chip_matched-ExpressionData.csv")
    active = chip.active_genes(expr, args.min_frac_nonzero)
    gt = gt[gt["Gene1"].isin(active) & gt["Gene2"].isin(active)].copy()

    _, chip_genes = chip.chip_gene_sets_from_gt(gt)
    if args.background_mode == "chip":
        background = sorted(active & chip_genes)
    else:
        background = sorted(active)
    bg_set = set(background)
    k_eval = chip.resolve_top_k(len(gt), args.top_edges)
    source_key = args.sources.split(",")[0].strip()
    lib_name = GSEAPY_LIBS.get(source_key, "GO_Biological_Process_2023")

    gmt_raw = load_go_library(lib_name, CACHE_DIR, args.min_term_size, args.max_term_size)
    gmt = filter_gmt_to_background(gmt_raw, bg_set)
    print(
        f"\n[{tag}] local hypergeom, background={len(background)}, "
        f"GO terms in bg={len(gmt)}, top-K={k_eval}, mode={args.gene_set_mode}"
    )

    files = ath.find_head_files([Path(p) for p in args.head_root], model, args.dataset)
    if not files:
        print("[WARN] no head TSV")
        return

    all_rows, summary_rows, heat_rows = [], [], []
    safe = source_key.replace(":", "_")

    for fp in files:
        h = ath.parse_head_number(fp)
        raw = chip.load_head_pred(fp, active, args.load_max_edges)
        use_aupr = args.gene_set_mode in ("topw", "targets")
        edge_df = chip.filter_pred_aupr_style(raw, gt) if use_aupr else raw

        if args.gene_set_mode in ("topw", "topw_raw"):
            query = genes_topw(edge_df, k_eval, args.top_genes)
        elif args.gene_set_mode == "targets":
            query = genes_targets(edge_df, k_eval)
        else:
            raise ValueError(f"Unknown gene-set-mode: {args.gene_set_mode}")

        print(f"  head{h}: n_genes={len(query)}", end="")
        if len(query) < args.min_genes:
            print(f" -> skip (<{args.min_genes})")
            continue

        enr = run_local_enrich(query, background, gmt, args.padj_cutoff, args.min_overlap)
        if enr.empty:
            print(" -> no sig. terms")
            continue

        enr = enr.copy()
        enr["Head"] = h
        enr["n_query"] = len(set(query) & bg_set)
        enr["n_background"] = len(background)
        enr["gene_set_mode"] = args.gene_set_mode
        csv_path = out_dir / f"head{h}_{safe}.csv"
        enr.to_csv(csv_path, index=False)
        all_rows.append(enr)

        plot_head_bar(enr, out_dir / f"head{h}_{safe}_bar.pdf", h, display, args.dataset, color, args.top_terms_per_head)

        top = enr.nsmallest(args.top_terms_per_head, "padj")
        for _, r in top.iterrows():
            summary_rows.append(
                {
                    "Head": h,
                    "term_name": r["term_name"],
                    "term_id": r.get("term_id", ""),
                    "padj": r["padj"],
                    "overlap_size": r["overlap_size"],
                    "term_size": r["term_size"],
                    "n_query_genes": len(set(query) & bg_set),
                    "n_background_genes": len(background),
                    "gene_set_mode": args.gene_set_mode,
                }
            )
            heat_rows.append(
                {"Head": h, "term_name": r["term_name"], "neglog10_padj": -np.log10(max(float(r["padj"]), 1e-300))}
            )
        print(f" -> {len(enr)} sig.; top: {str(top['term_name'].iloc[0])[:55]}")

    if not all_rows:
        print("[WARN] no enrichment results")
        return

    pd.concat(all_rows, ignore_index=True).to_csv(out_dir / f"{tag}_go_enrich_all.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / f"{tag}_go_enrich_summary.csv", index=False)

    meta = {
        "model": model,
        "dataset": args.dataset,
        "background_genes": len(background),
        "top_k_edges": k_eval,
        "gene_set_mode": args.gene_set_mode,
        "go_library": lib_name,
        "method": "hypergeometric + BH-FDR (local)",
        "padj_cutoff": args.padj_cutoff,
        "min_overlap": args.min_overlap,
    }
    with open(out_dir / f"{tag}_go_enrich_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    plot_heatmap(pd.DataFrame(heat_rows), out_dir / f"{tag}_go_bp_heatmap.pdf", display, args.dataset, source_key)
    plot_faceted_bars(summary, out_dir / f"{tag}_go_bp_bars.pdf", display, args.dataset, color)
    print(f"[INFO] Done -> {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-head GO enrichment (explicit background, local).")
    p.add_argument("--models", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--head-root", action="append", required=True)
    p.add_argument("--chip-root", default="/mnt/10T/yzn/benchmark_GRN/input_process/CHIP")
    p.add_argument("--output-dir", default=str(_SCRIPT_DIR / "output" / "head_go_enrich"))
    p.add_argument("--sources", default="GO:BP", help="GO:BP | GO:MF | GO:CC")
    p.add_argument("--top-edges", type=int, default=0, help="0 = K=|CHIP GT|")
    p.add_argument("--load-max-edges", type=int, default=0)
    p.add_argument("--gene-set-mode", choices=["topw", "targets", "topw_raw"], default="topw")
    p.add_argument("--top-genes", type=int, default=100, help="for topw: N highest-weight genes")
    p.add_argument(
        "--background-mode",
        choices=["active", "chip"],
        default="active",
        help="active=expressed genes; chip=active∩CHIP universe",
    )
    p.add_argument("--min-frac-nonzero", type=float, default=0.05)
    p.add_argument("--min-genes", type=int, default=20)
    p.add_argument("--min-overlap", type=int, default=3, help="min genes overlapping a GO term")
    p.add_argument("--min-term-size", type=int, default=5)
    p.add_argument("--max-term-size", type=int, default=500)
    p.add_argument("--padj-cutoff", type=float, default=0.05)
    p.add_argument("--top-terms-per-head", type=int, default=10)
    args = p.parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    return args


def main() -> None:
    args = parse_args()
    for m in args.models:
        run_one(m, args)


if __name__ == "__main__":
    main()
