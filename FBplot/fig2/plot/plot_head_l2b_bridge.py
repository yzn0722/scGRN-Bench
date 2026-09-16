#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L2b + L3 bridge figures: in-hub vs TF out-hub (sym_max), consistency, AUPRC, mean-wiped.

Outputs under: output/head_hub_l2/{model}_{dataset}/

Figures
-------
  {tag}_hub_in_vs_tf_out_bar.pdf      L2b: primary in-hub vs top-5 TF out (per head)
  {tag}_hub_in_out_consistency.pdf    in-hub vs top-1 TF out + match matrix
  {tag}_tf_out_vs_auprc.pdf           TF attention out-degree vs per-TF CHIP AUPRC
  {tag}_mean_wiped_vs_tf_hub.pdf      mean8 wipe rate vs TF hub/out (HAND2 highlight)

Tables
------
  {tag}_hub_in_out_compare.csv
  {tag}_tf_out_degree_by_head.csv
  {tag}_tf_out_auprc_merged.csv
  {tag}_tf_mean_wiped_stats.csv

Usage
-----
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig2/plot

  python plot_head_l2b_bridge.py \\
    --models scgpt --dataset hESC \\
    --head-root /mnt/10T/yzn/scGRN-Bench/FBplot/fig2/att_head \\
    --top-edges 0

  # If AUPRC / mean-wiped already run:
  python plot_head_l2b_bridge.py ... \\
    --auprc-dir output/head_chip_auprc \\
    --wiped-csv output/head_chip_auprc/scgpt_hESC/scgpt_hESC_mean_wiped_chip_edges.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import analyze_attention_heads as ath  # noqa: E402
import eval_heads_chip_auprc as chip  # noqa: E402
import validate_head_hubs_l2 as l2  # noqa: E402
from fig2_palette import model_color  # noqa: E402

plt.rcParams.update(ath.plt.rcParams)

ROLE_COLORS = {
    "CHIP_TF": "#d62728",
    "TF_and_target": "#ff7f0e",
    "target_only": "#2ca02c",
    "not_in_CHIP": "#7f7f7f",
}


def tf_out_weights_symmax(df: pd.DataFrame, tf_set: Set[str]) -> pd.Series:
    """Per-TF outgoing attention mass after sym_max (sum over targets)."""
    lk = chip.build_tf_lookup(df, tf_set, "sym_max")
    out: Dict[str, float] = defaultdict(float)
    for (tf, _tg), w in lk.items():
        out[tf] += float(w)
    return pd.Series(out, dtype=float)


def top_tf_out_table(df: pd.DataFrame, tf_set: Set[str], n: int = 5) -> pd.DataFrame:
    s = tf_out_weights_symmax(df, tf_set).sort_values(ascending=False)
    rows = []
    for rank, (gene, wt) in enumerate(s.head(n).items(), start=1):
        rows.append({"gene": gene, "tf_out_sum": wt, "tf_out_rank": rank})
    return pd.DataFrame(rows)


def build_in_out_compare(
    files: List[Path],
    active: Set[str],
    tf_set: Set[str],
    chip_tfs: Set[str],
    chip_targets: Set[str],
    top_edges: int,
    top_tf_n: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Primary in-hub + top-N TF out per head."""
    compare_rows = []
    tf_out_rows = []

    for fp in files:
        h = ath.parse_head_number(fp)
        df = l2.load_directed_top_edges(fp, active, top_edges)
        in_hubs = l2.top_in_hubs(df, n=1)
        tf_out = top_tf_out_table(df, tf_set, n=top_tf_n)

        in_gene = in_hubs.iloc[0]["gene"] if not in_hubs.empty else ""
        in_w = float(in_hubs.iloc[0]["in_weight_sum"]) if not in_hubs.empty else 0.0
        top_tf = tf_out.iloc[0]["gene"] if not tf_out.empty else ""
        top_tf_w = float(tf_out.iloc[0]["tf_out_sum"]) if not tf_out.empty else 0.0

        compare_rows.append(
            {
                "Head": h,
                "in_hub_gene": in_gene,
                "in_hub_weight": in_w,
                "in_hub_chip_role": l2.classify_chip_role(in_gene, chip_tfs, chip_targets),
                "top_tf_out_gene": top_tf,
                "top_tf_out_weight": top_tf_w,
                "top_tf_out_chip_role": l2.classify_chip_role(top_tf, chip_tfs, chip_targets),
                "in_out_same_gene": in_gene == top_tf and in_gene != "",
                "top5_tf_out_genes": ";".join(tf_out["gene"].astype(str).tolist()),
            }
        )

        for _, r in tf_out.iterrows():
            g = str(r["gene"])
            tf_out_rows.append(
                {
                    "Head": h,
                    "gene": g,
                    "tf_out_rank": int(r["tf_out_rank"]),
                    "tf_out_sum": float(r["tf_out_sum"]),
                    "chip_role": l2.classify_chip_role(g, chip_tfs, chip_targets),
                }
            )

    return pd.DataFrame(compare_rows), pd.DataFrame(tf_out_rows)


def load_or_compute_auprc_per_tf(
    tag: str,
    model: str,
    args: argparse.Namespace,
    gt: pd.DataFrame,
    active: Set[str],
    tf_set: Set[str],
    files: List[Path],
) -> pd.DataFrame:
    """Per-TF × head AUPRC; recompute if CSV missing."""
    auprc_dir = Path(args.auprc_dir) / tag
    per_tf_path = auprc_dir / f"{tag}_auprc_per_tf.csv"
    if per_tf_path.exists():
        df = pd.read_csv(per_tf_path)
        rows = []
        for _, r in df.iterrows():
            st = str(r["strategy"])
            if st.startswith("head") and len(st) > 4 and st[4:].isdigit():
                rows.append({"Head": int(st[4:]), "TF": r["TF"], "AUPRC": float(r["AUPRC"])})
        if rows:
            return pd.DataFrame(rows)

    print("  [INFO] computing per-TF AUPRC (no cache)...")
    parts = []
    target_pool = sorted(active)
    for fp in files:
        h = ath.parse_head_number(fp)
        raw = chip.load_head_pred(fp, active, args.load_max_edges)
        pred = chip.filter_pred_aupr_style(raw, gt)
        lk = chip.build_tf_lookup(pred, tf_set, args.pred_direction)
        _, per_tf = chip.evaluate_tf_centric(
            gt, lk, target_pool, args.neg_ratio, args.seed + h
        )
        if per_tf is None or per_tf.empty:
            continue
        per_tf = per_tf.copy()
        per_tf["Head"] = h
        parts.append(per_tf[["Head", "TF", "AUPRC"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_or_compute_wiped(tag: str, args: argparse.Namespace, gt: pd.DataFrame, active: Set[str], tf_set: Set[str], files: List[Path]) -> pd.DataFrame:
    p = Path(args.wiped_csv) if args.wiped_csv else Path(args.auprc_dir) / tag / f"{tag}_mean_wiped_chip_edges.csv"
    if p.exists():
        return pd.read_csv(p)

    print("  [INFO] computing mean-wiped table (no cache)...")
    k_eval = chip.resolve_top_k(len(gt), args.top_edges)
    per_head_df: Dict[int, pd.DataFrame] = {}
    per_head_lookup: Dict[int, Dict] = {}
    for fp in files:
        h = ath.parse_head_number(fp)
        raw = chip.load_head_pred(fp, active, args.load_max_edges)
        df = chip.filter_pred_aupr_style(raw, gt)
        per_head_df[h] = df
        per_head_lookup[h] = chip.build_tf_lookup(df, tf_set, args.pred_direction)

    fused = chip.fuse_predictions(per_head_df, "mean")
    mean_lk = chip.build_tf_lookup(fused, tf_set, args.pred_direction)
    mean_topk = set(sorted(mean_lk.items(), key=lambda x: x[1], reverse=True)[:k_eval])
    mean_topk = {e for e, _ in mean_topk}

    rows = []
    for _, r in gt.iterrows():
        tf, tg = str(r["Gene1"]), str(r["Gene2"])
        e = (tf, tg)
        in_topk = {h for h, lk in per_head_lookup.items() if e in set(sorted(lk.items(), key=lambda x: x[1], reverse=True)[:k_eval])}
        rows.append(
            {
                "TF": tf,
                "target": tg,
                "wiped_from_mean_topk": e not in mean_topk and len(in_topk) > 0,
                "heads_in_topk": ",".join(f"H{h}" for h in sorted(in_topk)),
            }
        )
    return pd.DataFrame(rows)


def tf_wipe_stats(wiped_df: pd.DataFrame) -> pd.DataFrame:
    g = wiped_df.groupby("TF", as_index=False).agg(
        n_chip_edges=("target", "count"),
        n_wiped=("wiped_from_mean_topk", "sum"),
    )
    g["frac_wiped"] = g["n_wiped"] / g["n_chip_edges"].clip(lower=1)
    return g


def merge_tf_metrics(
    compare_df: pd.DataFrame,
    tf_out_long: pd.DataFrame,
    auprc_df: pd.DataFrame,
    wipe_df: pd.DataFrame,
    expr_stats: pd.DataFrame,
    chip_tfs: Set[str],
) -> pd.DataFrame:
    """One row per CHIP TF with out-degree, AUPRC, wipe frac, in-hub flags."""
    tfs = sorted(chip_tfs)
    max_out = tf_out_long.groupby("gene")["tf_out_sum"].max().rename("tf_out_max_all_heads")
    h5_out = tf_out_long[tf_out_long["Head"] == 5].set_index("gene")["tf_out_sum"].rename("tf_out_head5")

    auprc_mean8_path = None
    # best head auprc per tf
    if not auprc_df.empty:
        best_auprc = auprc_df.loc[auprc_df.groupby("TF")["AUPRC"].idxmax()]
        best_auprc = best_auprc.set_index("TF")[["AUPRC", "Head"]].rename(
            columns={"AUPRC": "AUPRC_best_head", "Head": "best_auprc_head"}
        )
        mean_auprc = auprc_df.groupby("TF")["AUPRC"].mean().rename("AUPRC_mean_heads")
    else:
        best_auprc = pd.DataFrame()
        mean_auprc = pd.Series(dtype=float)

    prim_in = compare_df.set_index("Head")
    in_map = compare_df.set_index("in_hub_gene")["in_hub_weight"].to_dict()

    rows = []
    for tf in tfs:
        is_primary_in = tf in compare_df["in_hub_gene"].values
        in_w = in_map.get(tf, 0.0)
        expr_row = expr_stats[expr_stats["gene"] == tf]
        mean_expr = float(expr_row["mean_expr"].iloc[0]) if len(expr_row) else np.nan
        frac_nz = float(expr_row["frac_nonzero"].iloc[0]) if len(expr_row) else np.nan
        wrow = wipe_df[wipe_df["TF"] == tf]
        frac_wiped = float(wrow["frac_wiped"].iloc[0]) if len(wrow) else np.nan
        rows.append(
            {
                "TF": tf,
                "tf_out_max": max_out.get(tf, 0.0),
                "tf_out_head5": h5_out.get(tf, np.nan),
                "is_primary_in_hub_any_head": is_primary_in,
                "in_hub_weight_if_hub": in_w,
                "mean_expr": mean_expr,
                "frac_nonzero": frac_nz,
                "frac_chip_edges_wiped_mean8": frac_wiped,
                "AUPRC_best_head": best_auprc.loc[tf, "AUPRC_best_head"] if tf in best_auprc.index else np.nan,
                "best_auprc_head": int(best_auprc.loc[tf, "best_auprc_head"]) if tf in best_auprc.index else np.nan,
                "AUPRC_mean_heads": mean_auprc.get(tf, np.nan),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_in_vs_tf_out(
    compare_df: pd.DataFrame,
    tf_out_long: pd.DataFrame,
    display: str,
    dataset: str,
    out_pdf: Path,
    top_tf_n: int,
) -> None:
    """Per head: primary in-hub bar + top-N TF out bars (side by side)."""
    heads = sorted(compare_df["Head"].unique())
    fig, axes = plt.subplots(len(heads), 2, figsize=(10, 2.2 * len(heads)), sharey="row")
    if len(heads) == 1:
        axes = np.array([axes])

    for i, h in enumerate(heads):
        row = compare_df[compare_df["Head"] == h].iloc[0]
        ax_in, ax_out = axes[i, 0], axes[i, 1]

        # in-hub
        c_in = ROLE_COLORS.get(row["in_hub_chip_role"], "#7f7f7f")
        ax_in.barh([0], [row["in_hub_weight"]], color=c_in, edgecolor="k", linewidth=0.4, height=0.5)
        ax_in.set_yticks([0])
        ax_in.set_yticklabels([str(row["in_hub_gene"])], fontsize=9)
        ax_in.set_xlabel("In-hub weight (Gene2)")
        ax_in.set_title(f"H{h} — primary in-hub", fontsize=10)
        ax_in.invert_yaxis()

        # TF out top-N
        sub = tf_out_long[tf_out_long["Head"] == h].sort_values("tf_out_rank")
        y = np.arange(len(sub))
        cols = [ROLE_COLORS.get(r, "#7f7f7f") for r in sub["chip_role"]]
        ax_out.barh(y, sub["tf_out_sum"], color=cols, edgecolor="k", linewidth=0.4)
        ax_out.set_yticks(y)
        ax_out.set_yticklabels(sub["gene"].astype(str), fontsize=8)
        ax_out.set_xlabel("TF out-weight (sym_max)")
        ax_out.set_title(f"H{h} — top-{top_tf_n} TF out", fontsize=10)
        ax_out.invert_yaxis()

    handles = [Patch(facecolor=ROLE_COLORS[k], label=k) for k in ROLE_COLORS]
    fig.legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle(f"{display} — {dataset}\nL2b: primary in-hub (left) vs TF out-degree sym_max (right)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_consistency(compare_df: pd.DataFrame, display: str, dataset: str, out_pdf: Path) -> None:
    """Table + match strip: in-hub gene vs top-1 TF out."""
    heads = sorted(compare_df["Head"].unique())
    fig = plt.figure(figsize=(10, 0.55 * len(heads) + 1.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 0.5], wspace=0.08)

    ax_tbl = fig.add_subplot(gs[0, 0])
    ax_tbl.axis("off")
    cell_text = []
    for h in heads:
        r = compare_df[compare_df["Head"] == h].iloc[0]
        match = "Yes" if r["in_out_same_gene"] else "No"
        cell_text.append(
            [
                f"H{h}",
                r["in_hub_gene"],
                r["in_hub_chip_role"],
                f"{r['in_hub_weight']:.2f}",
                r["top_tf_out_gene"],
                r["top_tf_out_chip_role"],
                f"{r['top_tf_out_weight']:.2f}",
                match,
            ]
        )
    col_labels = [
        "Head",
        "In-hub",
        "In role",
        "In W",
        "Top TF out",
        "Out role",
        "Out W",
        "Match",
    ]
    tbl = ax_tbl.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    ax_tbl.set_title(f"{display} — {dataset}\nIn-hub vs top-1 TF out (sym_max)", pad=20)

    ax_hm = fig.add_subplot(gs[0, 1])
    match_mat = compare_df.set_index("Head")["in_out_same_gene"].astype(int).values.reshape(-1, 1)
    sns.heatmap(
        match_mat,
        annot=True,
        fmt="d",
        cmap="Greens",
        vmin=0,
        vmax=1,
        cbar=False,
        yticklabels=[f"H{h}" for h in heads],
        xticklabels=["Match"],
        ax=ax_hm,
    )
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tf_out_vs_auprc(
    tf_out_long: pd.DataFrame,
    auprc_df: pd.DataFrame,
    display: str,
    dataset: str,
    color: str,
    out_pdf: Path,
) -> None:
    """Scatter: TF sym_max out-weight vs CHIP AUPRC (per head, panels)."""
    if auprc_df.empty or tf_out_long.empty:
        return

    auprc_df = auprc_df.rename(columns={"TF": "gene"})
    merged = tf_out_long.merge(auprc_df, on=["Head", "gene"], how="inner")
    merged = merged.rename(columns={"gene": "TF"})

    heads = sorted(merged["Head"].unique())
    ncol = 4
    nrow = int(np.ceil(len(heads) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).flatten()

    for ax, h in zip(axes, heads):
        sub = merged[merged["Head"] == h]
        ax.scatter(sub["tf_out_sum"], sub["AUPRC"], c=color, alpha=0.75, edgecolors="k", linewidths=0.3, s=45)
        for _, r in sub.nlargest(3, "tf_out_sum").iterrows():
            ax.annotate(r["TF"], (r["tf_out_sum"], r["AUPRC"]), fontsize=7, alpha=0.85)
        if h == 5 and "HAND2" in sub["TF"].values:
            hand = sub[sub["TF"] == "HAND2"].iloc[0]
            ax.scatter([hand["tf_out_sum"]], [hand["AUPRC"]], c="gold", s=120, edgecolors="k", zorder=5, marker="*")
            ax.annotate("HAND2", (hand["tf_out_sum"], hand["AUPRC"]), fontsize=8, fontweight="bold")
        ax.set_xlabel("TF out-weight")
        ax.set_ylabel("AUPRC")
        ax.set_title(f"Head {h}")
        ax.set_ylim(-0.02, 1.02)

    for ax in axes[len(heads) :]:
        ax.axis("off")

    # global panel inset: max out vs mean AUPRC
    fig.suptitle(f"{display} — {dataset}\nTF attention out-degree (sym_max) vs CHIP AUPRC", y=1.02)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # second summary scatter: tf_out_max vs AUPRC_mean
    max_out = tf_out_long.groupby("gene")["tf_out_sum"].max().rename("tf_out_max")
    mean_auprc = auprc_df.groupby("gene")["AUPRC"].mean().rename("AUPRC_mean")
    g = pd.concat([max_out, mean_auprc], axis=1).dropna()
    g = g.rename_axis("TF").reset_index()

    fig2, ax2 = plt.subplots(figsize=(6.5, 5.5))
    ax2.scatter(g["tf_out_max"], g["AUPRC_mean"], c=color, s=55, edgecolors="k", linewidths=0.3)
    if "HAND2" in g["TF"].values:
        hand = g[g["TF"] == "HAND2"].iloc[0]
        ax2.scatter([hand["tf_out_max"]], [hand["AUPRC_mean"]], c="gold", s=150, marker="*", edgecolors="k", zorder=5)
        ax2.annotate("HAND2", (hand["tf_out_max"], hand["AUPRC_mean"]), fontsize=9, fontweight="bold")
    ax2.set_xlabel("Max TF out-weight (any head)")
    ax2.set_ylabel("Mean AUPRC across heads")
    ax2.set_title(f"{display} — {dataset}\nGlobal: attention out vs CHIP predictability")
    fig2.tight_layout()
    out2 = out_pdf.parent / out_pdf.name.replace(".pdf", "_global.pdf")
    fig2.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close(fig2)


def plot_mean_wiped_vs_hub(
    tf_metrics: pd.DataFrame,
    display: str,
    dataset: str,
    color: str,
    out_pdf: Path,
    highlight_tf: str = "HAND2",
) -> None:
    """Wipe fraction vs TF out / expression; highlight HAND2."""
    d = tf_metrics.dropna(subset=["frac_chip_edges_wiped_mean8"]).copy()
    if d.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))

    def _scatter(ax, xcol, xlabel, logx=False):
        x = d[xcol].astype(float)
        y = d["frac_chip_edges_wiped_mean8"].astype(float)
        c = d["mean_expr"].astype(float)
        sc = ax.scatter(x, y, c=c, cmap="viridis", s=50 + 3 * d["tf_out_max"], edgecolors="k", linewidths=0.3)
        plt.colorbar(sc, ax=ax, label="Mean expr")
        if highlight_tf in d["TF"].values:
            h = d[d["TF"] == highlight_tf].iloc[0]
            ax.scatter([h[xcol]], [h["frac_chip_edges_wiped_mean8"]], c="gold", s=200, marker="*", edgecolors="k", zorder=5)
            ax.annotate(highlight_tf, (h[xcol], h["frac_chip_edges_wiped_mean8"]), fontsize=9, fontweight="bold")
        if logx and x.min() > 0:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frac. CHIP edges wiped by mean8 top-K")
        ax.set_ylim(-0.02, 1.02)

    _scatter(axes[0], "tf_out_max", "Max TF out-weight (sym_max, any head)")
    _scatter(axes[1], "tf_out_head5", "TF out-weight on Head 5")
    _scatter(axes[2], "mean_expr", "Mean expression (hESC)")

    fig.suptitle(
        f"{display} — {dataset}\nMean8 fusion: CHIP edge wipe rate vs TF hub/out (size ∝ tf_out_max)",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # bar: top wipe fraction TFs
    top = d.nlargest(12, "frac_chip_edges_wiped_mean8")
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    cols = ["gold" if t == highlight_tf else color for t in top["TF"]]
    ax2.barh(np.arange(len(top)), top["frac_chip_edges_wiped_mean8"], color=cols, edgecolor="k", linewidth=0.3)
    ax2.set_yticks(np.arange(len(top)))
    ax2.set_yticklabels(top["TF"], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Fraction CHIP edges not in mean8 top-K (but in ≥1 head)")
    ax2.set_title(f"{display} — {dataset}\nTFs most affected by mean8 pooling")
    fig2.tight_layout()
    out2 = out_pdf.parent / out_pdf.name.replace(".pdf", "_tf_bar.pdf")
    fig2.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close(fig2)


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
    tf_set = set(gt["Gene1"].astype(str))
    chip_tfs, chip_targets, _, _ = l2.chip_gene_roles(gt)
    expr_stats = l2.expression_stats(expr, list(active))

    top_k_edges = chip.resolve_top_k(len(gt), args.top_edges)
    print(f"\n[{tag}] top-K edges={top_k_edges}, sym_max TF out, top_tf={args.top_tf_out}")

    files = ath.find_head_files([Path(p) for p in args.head_root], model, args.dataset)
    if not files:
        print("[WARN] no head TSV")
        return

    compare_df, tf_out_long = build_in_out_compare(
        files, active, tf_set, chip_tfs, chip_targets, top_k_edges, args.top_tf_out
    )
    compare_df.to_csv(out_dir / f"{tag}_hub_in_out_compare.csv", index=False)
    tf_out_long.to_csv(out_dir / f"{tag}_tf_out_degree_by_head.csv", index=False)

    auprc_df = load_or_compute_auprc_per_tf(tag, model, args, gt, active, tf_set, files)
    if not auprc_df.empty:
        auprc_df.to_csv(out_dir / f"{tag}_tf_auprc_by_head.csv", index=False)

    merged_auprc = tf_out_long.merge(
        auprc_df, left_on=["Head", "gene"], right_on=["Head", "TF"], how="left"
    )
    merged_auprc.to_csv(out_dir / f"{tag}_tf_out_auprc_merged.csv", index=False)

    wiped_df = load_or_compute_wiped(tag, args, gt, active, tf_set, files)
    wipe_stats = tf_wipe_stats(wiped_df)
    tf_metrics = merge_tf_metrics(compare_df, tf_out_long, auprc_df, wipe_stats, expr_stats, chip_tfs)
    tf_metrics.to_csv(out_dir / f"{tag}_tf_mean_wiped_stats.csv", index=False)

    plot_in_vs_tf_out(
        compare_df, tf_out_long, display, args.dataset,
        out_dir / f"{tag}_hub_in_vs_tf_out_bar.pdf", args.top_tf_out,
    )
    plot_consistency(compare_df, display, args.dataset, out_dir / f"{tag}_hub_in_out_consistency.pdf")
    plot_tf_out_vs_auprc(tf_out_long, auprc_df, display, args.dataset, color, out_dir / f"{tag}_tf_out_vs_auprc.pdf")
    plot_mean_wiped_vs_hub(tf_metrics, display, args.dataset, color, out_dir / f"{tag}_mean_wiped_vs_tf_hub.pdf", args.highlight_tf)

    n_match = int(compare_df["in_out_same_gene"].sum())
    print(f"  in-hub == top TF out: {n_match}/{len(compare_df)} heads")
    if highlight := args.highlight_tf:
        if highlight in tf_metrics["TF"].values:
            hr = tf_metrics[tf_metrics["TF"] == highlight].iloc[0]
            print(
                f"  {highlight}: frac_wiped={hr['frac_chip_edges_wiped_mean8']:.3f}, "
                f"tf_out_max={hr['tf_out_max']:.3f}, mean_expr={hr['mean_expr']:.3f}"
            )
    print(f"[INFO] L2b bridge figures -> {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L2b bridge: in-hub vs TF out, AUPRC, mean-wiped.")
    p.add_argument("--models", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--head-root", action="append", required=True)
    p.add_argument("--chip-root", default="/mnt/10T/yzn/benchmark_GRN/input_process/CHIP")
    p.add_argument("--output-dir", default=str(_SCRIPT_DIR / "output" / "head_hub_l2"))
    p.add_argument("--auprc-dir", default=str(_SCRIPT_DIR / "output" / "head_chip_auprc"))
    p.add_argument("--wiped-csv", default="", help="optional path to mean_wiped CSV")
    p.add_argument("--top-edges", type=int, default=0, help="0 = K=|CHIP GT|")
    p.add_argument("--load-max-edges", type=int, default=0)
    p.add_argument("--top-tf-out", type=int, default=5)
    p.add_argument("--min-frac-nonzero", type=float, default=0.05)
    p.add_argument("--pred-direction", default="sym_max")
    p.add_argument("--highlight-tf", default="HAND2")
    p.add_argument("--neg-ratio", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not args.wiped_csv:
        args.wiped_csv = None
    return args


def main() -> None:
    args = parse_args()
    for m in args.models:
        run_one(m, args)


if __name__ == "__main__":
    main()
