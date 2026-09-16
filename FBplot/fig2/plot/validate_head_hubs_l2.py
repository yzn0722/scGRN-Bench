#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L2 validation: attention head hub genes vs hESC expression & CHIP TF roles.

Motivation (from L1 structure roles)
------------------------------------
  - High-clustering heads (e.g. H0/H6): star-like in-hubs in top-k attention edges.
  - High-modularity head (e.g. H4): several weaker in-hubs / multi-program topology.

This script checks for each head's top in-degree hubs (Gene2 in exported TSV):
  1. Expression in hESC (mean, fraction non-zero, percentile among active genes).
  2. CHIP ontology: is the hub a TF (Gene1 in CHIP), only a target, both, or neither?

Outputs under: output/head_hub_l2/{model}_{dataset}/

Usage
-----
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig2/plot

  python validate_head_hubs_l2.py \\
    --models scgpt \\
    --dataset hESC \\
    --head-root /mnt/10T/yzn/scGRN-Bench/FBplot/fig2/att_head \\
    --chip-root /mnt/10T/yzn/benchmark_GRN/input_process/CHIP \\
    --top-edges 10000 \\
    --top-hubs 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import analyze_attention_heads as ath  # noqa: E402
from fig2_palette import model_color  # noqa: E402

import stat_analyze_attention_heads as sah  # noqa: E402

plt.rcParams.update(ath.plt.rcParams)

# Topology-based role labels (same thresholds as interpretability notes)
HIGH_CLUST = 0.65
HIGH_MOD = 0.20
LONG_PATH = 2.1


def load_directed_top_edges(fp: Path, gene_universe: Set[str], top_k: int) -> pd.DataFrame:
    """Load head TSV; keep query→key direction (Gene1→Gene2)."""
    usecols = ["Gene1", "Gene2", "EdgeWeight"]
    try:
        df = pd.read_csv(fp, sep="\t", usecols=lambda c: c in usecols or c in ath.WEIGHT_COLS)
    except (ValueError, KeyError):
        df = pd.read_csv(fp, sep="\t")
    df = ath.normalize_edges(df)
    df = df[df["Gene1"].isin(gene_universe) & df["Gene2"].isin(gene_universe)].copy()
    if df.empty:
        return df
    wcol = ath.detect_weight_col(df)
    if wcol is None:
        df["EdgeWeight"] = 1.0
    else:
        df = df.rename(columns={wcol: "EdgeWeight"})
        df["EdgeWeight"] = pd.to_numeric(df["EdgeWeight"], errors="coerce").fillna(0.0)
    if top_k > 0:
        df = df.nlargest(top_k, "EdgeWeight")
    return df[["Gene1", "Gene2", "EdgeWeight"]]


def read_chip_gt(chip_root: Path, dataset: str) -> pd.DataFrame:
    fp = chip_root / f"{dataset}_chip_matched-network.csv"
    df = pd.read_csv(fp)
    return ath.normalize_edges(df)[["Gene1", "Gene2"]]


def read_expression(chip_root: Path, dataset: str) -> pd.DataFrame:
    fp = chip_root / f"{dataset}_chip_matched-ExpressionData.csv"
    df = pd.read_csv(fp, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df.index = df.index.astype(str)
    return df


def chip_gene_roles(gt: pd.DataFrame) -> Tuple[Set[str], Set[str], Dict[str, int], Dict[str, int]]:
    """
    Returns:
      chip_tfs: genes appearing as Gene1 (TF)
      chip_targets: genes appearing as Gene2
      n_targets_per_tf: out-degree in CHIP for each TF
      n_tfs_per_target: in-degree in CHIP for each target gene
    """
    chip_tfs = set(gt["Gene1"].astype(str))
    chip_targets = set(gt["Gene2"].astype(str))
    n_targets_per_tf = gt.groupby("Gene1")["Gene2"].nunique().astype(int).to_dict()
    n_tfs_per_target = gt.groupby("Gene2")["Gene1"].nunique().astype(int).to_dict()
    return chip_tfs, chip_targets, n_targets_per_tf, n_tfs_per_target


def classify_chip_role(gene: str, chip_tfs: Set[str], chip_targets: Set[str]) -> str:
    is_tf = gene in chip_tfs
    is_tg = gene in chip_targets
    if is_tf and is_tg:
        return "TF_and_target"
    if is_tf:
        return "CHIP_TF"
    if is_tg:
        return "target_only"
    return "not_in_CHIP"


def expression_stats(expr: pd.DataFrame, genes: List[str]) -> pd.DataFrame:
    """Per-gene mean expression and fraction of non-zero cells."""
    frac = (expr.values > 0).mean(axis=1)
    mean = expr.values.mean(axis=1)
    base = pd.DataFrame(
        {
            "gene": expr.index.astype(str),
            "mean_expr": mean,
            "frac_nonzero": frac,
        }
    )
    # Percentile rank among all genes in matrix (for context)
    base["mean_expr_pct"] = base["mean_expr"].rank(pct=True)
    sub = base[base["gene"].isin(genes)].copy()
    return sub


def infer_head_structure_role(row: pd.Series) -> str:
    """Short label from topology metrics (for figure annotation)."""
    cl = float(row.get("Clustering", np.nan))
    mod = float(row.get("Modularity", np.nan))
    pl = float(row.get("AvgPathLength", np.nan))
    if mod >= HIGH_MOD and cl < HIGH_CLUST:
        return "multi-program (high Q)"
    if cl >= HIGH_CLUST:
        return "star-hub (high Clust)"
    if pl >= LONG_PATH:
        return "sparse/long-path"
    return "mixed"


def top_in_hubs(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """
    In-attention hubs: genes that appear as Gene2 (key) in top edges.
    Score = sum of EdgeWeight into Gene2 (incoming attention mass).
    """
    if df.empty:
        return pd.DataFrame(columns=["gene", "in_weight_sum", "in_edge_count", "hub_rank"])
    agg = df.groupby("Gene2", as_index=False).agg(
        in_weight_sum=("EdgeWeight", "sum"),
        in_edge_count=("Gene2", "count"),
    )
    agg = agg.rename(columns={"Gene2": "gene"})
    agg = agg.sort_values(["in_weight_sum", "in_edge_count"], ascending=False)
    agg["hub_rank"] = np.arange(1, len(agg) + 1)
    return agg.head(n)


def build_hub_table(
    head: int,
    df: pd.DataFrame,
    expr_stats_all: pd.DataFrame,
    chip_tfs: Set[str],
    chip_targets: Set[str],
    n_targets_per_tf: Dict[str, int],
    n_tfs_per_target: Dict[str, int],
    top_n: int,
    topo_row: Optional[pd.Series],
) -> pd.DataFrame:
    hubs = top_in_hubs(df, n=top_n)
    if hubs.empty:
        return hubs
    hubs["Head"] = head
    hubs = hubs.merge(expr_stats_all, on="gene", how="left")
    hubs["chip_role"] = hubs["gene"].map(lambda g: classify_chip_role(g, chip_tfs, chip_targets))
    hubs["chip_n_targets"] = hubs["gene"].map(lambda g: n_targets_per_tf.get(g, 0))
    hubs["chip_n_regulators"] = hubs["gene"].map(lambda g: n_tfs_per_target.get(g, 0))
    if topo_row is not None:
        hubs["head_modularity"] = topo_row.get("Modularity", np.nan)
        hubs["head_clustering"] = topo_row.get("Clustering", np.nan)
        hubs["head_avg_path"] = topo_row.get("AvgPathLength", np.nan)
        hubs["head_structure_role"] = infer_head_structure_role(topo_row)
    return hubs


def plot_primary_hub_expression(hub_df: pd.DataFrame, display: str, dataset: str, color: str, out_pdf: Path) -> None:
    """Bar chart: #1 in-hub per head — mean expression, colored by CHIP role."""
    prim = hub_df[hub_df["hub_rank"] == 1].copy()
    if prim.empty:
        return
    role_colors = {
        "CHIP_TF": "#d62728",
        "TF_and_target": "#ff7f0e",
        "target_only": "#2ca02c",
        "not_in_CHIP": "#7f7f7f",
    }
    prim = prim.sort_values("Head")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    x = np.arange(len(prim))
    cols = [role_colors.get(r, "#7f7f7f") for r in prim["chip_role"]]
    axes[0].bar(x, prim["mean_expr"], color=cols, edgecolor="k", linewidth=0.4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"H{int(h)}\n{g}" for h, g in zip(prim["Head"], prim["gene"])], fontsize=9)
    axes[0].set_ylabel("Mean expression (hESC)")
    axes[0].set_title("Primary in-hub (top-k incoming attention)")

    axes[1].bar(x, prim["frac_nonzero"], color=cols, edgecolor="k", linewidth=0.4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"H{int(h)}" for h in prim["Head"]])
    axes[1].set_ylabel("Fraction non-zero cells")
    axes[1].set_ylim(0, 1.05)

    # legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=role_colors[k], label=k) for k in role_colors]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=9)
    fig.suptitle(f"{display} — {dataset}\nL2: hub expression & CHIP role (primary in-hub per head)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_top_hubs_dot(hub_df: pd.DataFrame, display: str, dataset: str, out_pdf: Path) -> None:
    """Dot plot: top-N in-hubs per head, y=mean_expr, marker size=in_weight_sum."""
    if hub_df.empty:
        return
    d = hub_df.copy()
    d["label"] = d.apply(lambda r: f"H{int(r.Head)}:{r.gene}", axis=1)
    role_colors = {
        "CHIP_TF": "#d62728",
        "TF_and_target": "#ff7f0e",
        "target_only": "#2ca02c",
        "not_in_CHIP": "#7f7f7f",
    }
    d["color"] = d["chip_role"].map(role_colors).fillna("#7f7f7f")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(d))))
    sizes = 30 + 120 * (d["in_weight_sum"] / d["in_weight_sum"].max())
    ax.scatter(d["Head"], d["mean_expr"], c=d["color"], s=sizes, edgecolors="k", linewidths=0.3)
    for _, r in d.iterrows():
        ax.annotate(r["gene"], (r["Head"], r["mean_expr"]), fontsize=8, ha="center", va="bottom")
    ax.set_xlabel("Attention head")
    ax.set_ylabel("Mean expression")
    ax.set_title(f"{display} — {dataset}\nTop in-hubs (size ∝ incoming attention mass)")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_expr_vs_chip_targets(hub_df: pd.DataFrame, display: str, dataset: str, out_pdf: Path) -> None:
    """Scatter: mean expression vs CHIP out-degree (if TF); primary hubs only."""
    prim = hub_df[hub_df["hub_rank"] == 1].copy()
    fig, ax = plt.subplots(figsize=(6, 5))
    for role, sub in prim.groupby("chip_role"):
        ax.scatter(
            sub["chip_n_targets"],
            sub["mean_expr"],
            label=role,
            s=100,
            edgecolors="k",
            linewidths=0.4,
        )
        for _, r in sub.iterrows():
            ax.annotate(f"H{int(r.Head)} {r.gene}", (r.chip_n_targets, r.mean_expr), fontsize=8)
    ax.set_xlabel("CHIP out-degree (n targets, 0 if not a TF)")
    ax.set_ylabel("Mean expression in hESC")
    ax.legend(fontsize=9, title="CHIP role")
    ax.set_title(f"{display} — {dataset}\nPrimary hub: expression vs CHIP TF degree")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_structure_role_panel(hub_df: pd.DataFrame, topo: pd.DataFrame, display: str, dataset: str, color: str, out_pdf: Path) -> None:
    """
    Heatmap-style panel: heads × [Modularity, Clustering, primary hub expr, is_CHIP_TF].
    Links L1 topology role to L2 hub biology.
    """
    prim = hub_df[hub_df["hub_rank"] == 1].set_index("Head")
    heads = sorted(prim.index)
    rows = []
    for h in heads:
        tr = topo[topo["Head"] == h]
        if tr.empty:
            continue
        r = tr.iloc[0]
        p = prim.loc[h]
        rows.append(
            {
                "Head": h,
                "Modularity": r["Modularity"],
                "Clustering": r["Clustering"],
                "AvgPathLength": r["AvgPathLength"],
                "hub_gene": p["gene"],
                "hub_mean_expr": p["mean_expr"],
                "hub_frac_nonzero": p["frac_nonzero"],
                "hub_is_CHIP_TF": 1.0 if p["chip_role"] in ("CHIP_TF", "TF_and_target") else 0.0,
                "structure_role": p.get("head_structure_role", infer_head_structure_role(r)),
            }
        )
    panel = pd.DataFrame(rows).set_index("Head")
    if panel.empty:
        return

    zcols = ["Modularity", "Clustering", "AvgPathLength", "hub_mean_expr", "hub_frac_nonzero", "hub_is_CHIP_TF"]
    Z = panel[zcols].astype(float)
    Z = (Z - Z.mean()) / Z.std(ddof=0).replace(0, np.nan)

    ylabels = [f"H{int(h)} {panel.loc[h,'hub_gene']} ({panel.loc[h,'structure_role']})" for h in heads]
    fig, ax = plt.subplots(figsize=(8, max(3.5, len(heads) * 0.55)))
    sns.heatmap(
        Z,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        ax=ax,
        yticklabels=ylabels,
        cbar_kws={"label": "z-score across heads"},
    )
    ax.set_title(f"{display} — {dataset}\nL1 structure + L2 hub (expression & CHIP TF flag)")
    ax.set_xlabel("Metric")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_one(model: str, args: argparse.Namespace) -> None:
    display = ath.MODEL_ALIASES.get(ath.model_file_prefix(model), model)
    color = model_color(display)
    tag = f"{ath.model_file_prefix(model)}_{args.dataset}"
    out_dir = Path(args.output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    chip_root = Path(args.chip_root)
    gt = read_chip_gt(chip_root, args.dataset)
    expr = read_expression(chip_root, args.dataset)
    chip_tfs, chip_targets, n_targets_per_tf, n_tfs_per_target = chip_gene_roles(gt)
    active = set(expr.index)

    # Optional topology from prior stat run
    topo_path = Path(args.stat_dir) / tag / f"{tag}_topology_metrics.csv"
    topo = pd.read_csv(topo_path) if topo_path.exists() else pd.DataFrame()

    expr_all = expression_stats(expr, list(active))

    head_roots = [Path(p) for p in args.head_root]
    files = ath.find_head_files(head_roots, model, args.dataset)
    if not files:
        print(f"[WARN] skip {tag}: no head TSV")
        return

    parts = []
    for fp in files:
        h = ath.parse_head_number(fp)
        df = load_directed_top_edges(fp, active, args.top_edges)
        topo_row = topo[topo["Head"] == h].iloc[0] if not topo.empty and (topo["Head"] == h).any() else None
        tbl = build_hub_table(
            h, df, expr_all, chip_tfs, chip_targets, n_targets_per_tf, n_tfs_per_target,
            args.top_hubs, topo_row,
        )
        parts.append(tbl)
        if not tbl.empty:
            g1 = tbl.iloc[0]["gene"]
            print(
                f"  head{h} [{tbl.iloc[0].get('head_structure_role', '?')}]: "
                f"primary in-hub={g1}, expr={tbl.iloc[0]['mean_expr']:.4f}, "
                f"frac_nz={tbl.iloc[0]['frac_nonzero']:.3f}, chip_role={tbl.iloc[0]['chip_role']}"
            )

    hub_df = pd.concat(parts, ignore_index=True)
    hub_df.to_csv(out_dir / f"{tag}_hub_l2_table.csv", index=False)

    # Summary: primary hub only
    prim = hub_df[hub_df["hub_rank"] == 1].copy()
    prim.to_csv(out_dir / f"{tag}_hub_l2_primary.csv", index=False)

    # Figures
    plot_primary_hub_expression(hub_df, display, args.dataset, color, out_dir / f"{tag}_hub_expr_primary_bar.pdf")
    plot_top_hubs_dot(hub_df, display, args.dataset, out_dir / f"{tag}_hub_expr_dot.pdf")
    plot_expr_vs_chip_targets(hub_df, display, args.dataset, out_dir / f"{tag}_hub_expr_vs_chip_degree.pdf")
    if not topo.empty:
        plot_structure_role_panel(hub_df, topo, display, args.dataset, color, out_dir / f"{tag}_hub_l1_l2_panel.pdf")

    print(f"[INFO] Wrote L2 hub validation -> {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L2: validate head in-hubs vs expression & CHIP TF roles.")
    p.add_argument("--models", required=True, help="e.g. scgpt or scgpt,sccello")
    p.add_argument("--dataset", required=True)
    p.add_argument("--head-root", action="append", required=True)
    p.add_argument("--chip-root", default="/mnt/10T/yzn/benchmark_GRN/input_process/CHIP")
    p.add_argument("--stat-dir", default=str(_SCRIPT_DIR / "output" / "head_statistics"))
    p.add_argument("--output-dir", default=str(_SCRIPT_DIR / "output" / "head_hub_l2"))
    p.add_argument("--top-edges", type=int, default=10000)
    p.add_argument("--top-hubs", type=int, default=3, help="Top in-hubs per head to report")
    args = p.parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    return args


def main() -> None:
    args = parse_args()
    for model in args.models:
        run_one(model, args)


if __name__ == "__main__":
    main()
