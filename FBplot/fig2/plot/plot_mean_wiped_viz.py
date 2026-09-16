#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize mean8 vs per-head scores for CHIP case-study edges (mean wiped).

Figures (under output/head_chip_auprc/{tag}/):
  1. case_score_bars.pdf      — featured edges: score per head + mean8
  2. case_rank_bars.pdf       — same edges: rank (log10), K=|GT| line
  3. tfap2a_score_heatmap.pdf — top TFAP2A wiped edges × (H0–H7, mean8) scores
  4. tfap2a_topk_heatmap.pdf  — binary: in top-K or not (green=in, white=out)
  5. wiped_by_best_head.pdf   — count of mean-wiped edges by best-scoring head

Usage:
  cd FBplot/fig2/plot
  python plot_mean_wiped_viz.py --models scgpt --dataset hESC \\
    --head-root /mnt/10T/yzn/scGRN-Bench/FBplot/fig2/att_head
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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

# Featured case edges from mean_wiped_case_study.txt (TF, target)
FEATURED_EDGES: List[Tuple[str, str, str]] = [
    ("JUND", "TCF4", "H3 rank #7, mean out"),
    ("EGR1", "TFAP2A", "H1 high, mean diluted"),
    ("TFAP2A", "EGR1", "H1 high, mean diluted"),
    ("EGR1", "SOX2", "H1 / H5"),
    ("EGR1", "AR", "H5 only"),
]


def load_lookups(
    model: str, args: argparse.Namespace
) -> Tuple[Dict[int, Dict], Dict, int, Dict[int, set], set]:
    """Build per-head + mean8 TF lookups (AUPR-filtered preds)."""
    chip_root = Path(args.chip_root)
    gt = chip.read_chip_gt(chip_root / f"{args.dataset}_chip_matched-network.csv")
    expr = chip.read_chip_expr(chip_root / f"{args.dataset}_chip_matched-ExpressionData.csv")
    active = chip.active_genes(expr, args.min_frac_nonzero)
    gt = gt[gt["Gene1"].isin(active) & gt["Gene2"].isin(active)].copy()
    tf_set = set(gt["Gene1"].astype(str))
    k_eval = chip.resolve_top_k(len(gt), args.top_edges)

    files = ath.find_head_files([Path(p) for p in args.head_root], model, args.dataset)
    per_head_lk: Dict[int, Dict[Tuple[str, str], float]] = {}
    per_head_topk: Dict[int, set] = {}
    per_head_df: Dict[int, pd.DataFrame] = {}

    for fp in files:
        h = ath.parse_head_number(fp)
        raw = chip.load_head_pred(fp, active, args.load_max_edges)
        df = chip.filter_pred_aupr_style(raw, gt)
        per_head_df[h] = df
        lk = chip.build_tf_lookup(df, tf_set, args.pred_direction)
        per_head_lk[h] = lk
        items = sorted(lk.items(), key=lambda x: x[1], reverse=True)
        k_eff = min(k_eval, len(items))
        per_head_topk[h] = {e for e, _ in items[:k_eff]}

    fused = chip.fuse_predictions(per_head_df, "mean")
    mean_lk = chip.build_tf_lookup(fused, tf_set, args.pred_direction)
    mean_items = sorted(mean_lk.items(), key=lambda x: x[1], reverse=True)
    mean_topk = {e for e, _ in mean_items[: min(k_eval, len(mean_items))]}

    return per_head_lk, mean_lk, k_eval, per_head_topk, mean_topk


def edge_scores_row(
    tf: str,
    target: str,
    per_head_lk: Dict[int, Dict],
    mean_lk: Dict,
    heads: List[int],
) -> pd.DataFrame:
    e = (tf, target)
    rows = []
    for h in heads:
        rows.append({"label": f"H{h}", "kind": "head", "head": h, "score": per_head_lk[h].get(e, 0.0)})
    rows.append({"label": "mean8", "kind": "mean8", "head": -1, "score": mean_lk.get(e, 0.0)})
    return pd.DataFrame(rows)


def edge_ranks_row(
    tf: str,
    target: str,
    per_head_lk: Dict[int, Dict],
    mean_lk: Dict,
    heads: List[int],
) -> pd.DataFrame:
    e = (tf, target)
    rows = []
    for h in heads:
        rows.append(
            {
                "label": f"H{h}",
                "rank": chip.global_rank(per_head_lk[h], e),
            }
        )
    rows.append({"label": "mean8", "rank": chip.global_rank(mean_lk, e)})
    return pd.DataFrame(rows)


def plot_case_score_bars(
    per_head_lk: Dict,
    mean_lk: Dict,
    out_pdf: Path,
    display: str,
    dataset: str,
    color: str,
) -> None:
    heads = sorted(per_head_lk.keys())
    n = len(FEATURED_EDGES)
    ncol, nrow = 3, 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.6 * nrow), sharey=False)
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, (tf, tg, _subtitle) in zip(axes_flat, FEATURED_EDGES):
        sub = edge_scores_row(tf, tg, per_head_lk, mean_lk, heads)
        cols = [color if k == "head" else "#555555" for k in sub["kind"]]
        bars = ax.bar(range(len(sub)), sub["score"], color=cols, edgecolor="k", linewidth=0.35)
        bars[-1].set_hatch("//")
        bars[-1].set_alpha(0.75)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["label"], rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Score (sym_max)")

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_case_rank_bars(per_head_lk: Dict, mean_lk: Dict, k_eval: int, out_pdf: Path, display: str, dataset: str, color: str) -> None:
    heads = sorted(per_head_lk.keys())
    n = len(FEATURED_EDGES)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4.2))
    if n == 1:
        axes = [axes]

    for ax, (tf, tg, subtitle) in zip(axes, FEATURED_EDGES):
        sub = edge_ranks_row(tf, tg, per_head_lk, mean_lk, heads)
        ranks = sub["rank"].astype(float).clip(lower=1)
        cols = [color] * (len(sub) - 1) + ["#555555"]
        ax.bar(range(len(sub)), np.log10(ranks), color=cols, edgecolor="k", linewidth=0.35)
        ax.axhline(np.log10(k_eval), color="crimson", ls="--", lw=1.2, label=f"Top-K={k_eval}")
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["label"], rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(r"$\log_{10}$(rank)")
        in_mean = ranks.iloc[-1] <= k_eval
        ax.set_title(f"{tf}→{tg}\nmean8 in top-K: {in_mean}", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9)
    fig.suptitle(f"{display} — {dataset}\nRank of CHIP edge (lower = better); dashed = Top-K cutoff", y=1.06, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tfap2a_heatmaps(
    wiped: pd.DataFrame,
    per_head_lk: Dict,
    mean_lk: Dict,
    k_eval: int,
    mean_topk: set,
    per_head_topk: Dict[int, set],
    out_score: Path,
    out_topk: Path,
    display: str,
    dataset: str,
    tfap2a_top: int,
) -> None:
    tfa = wiped[wiped["TF"] == "TFAP2A"].copy()
    tfa = tfa.sort_values("delta_max_minus_mean", ascending=False).head(tfap2a_top)
    if tfa.empty:
        return

    heads = sorted(per_head_lk.keys())
    edge_labels = [f"{r.TF}→{r.target}" for _, r in tfa.iterrows()]

    score_mat = []
    topk_mat = []
    for _, r in tfa.iterrows():
        e = (str(r.TF), str(r.target))
        score_row = [per_head_lk[h].get(e, 0.0) for h in heads] + [mean_lk.get(e, 0.0)]
        topk_row = [1.0 if e in per_head_topk[h] else 0.0 for h in heads] + [1.0 if e in mean_topk else 0.0]
        score_mat.append(score_row)
        topk_mat.append(topk_row)

    score_df = pd.DataFrame(score_mat, index=edge_labels, columns=[f"H{h}" for h in heads] + ["mean8"])
    topk_df = pd.DataFrame(topk_mat, index=edge_labels, columns=[f"H{h}" for h in heads] + ["mean8"])

    # Score heatmap
    fig_h = max(4, 0.28 * len(edge_labels))
    fig, ax = plt.subplots(figsize=(8, fig_h))
    sns.heatmap(score_df, annot=True, fmt=".4f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "sym_max score"})
    ax.set_title(f"{display} — {dataset}\nTFAP2A CHIP edges wiped by mean8 (top {len(tfa)} by Δ score)")
    ax.set_ylabel("CHIP true edge")
    fig.savefig(out_score, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Top-K membership (clearer story)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    sns.heatmap(
        topk_df,
        annot=True,
        fmt=".0f",
        cmap=sns.color_palette(["#f7f7f7", "#2ca02c"]),
        vmin=0,
        vmax=1,
        cbar=False,
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )
    ax.set_title(f"{display} — {dataset}\nIn top-{k_eval}? (1=yes). Green=head keeps edge, white=mean8 drops")
    ax.set_ylabel("CHIP true edge")
    fig.savefig(out_topk, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_wiped_counts(wiped: pd.DataFrame, out_pdf: Path, display: str, dataset: str, color: str) -> None:
    """Among edges in some head top-K but not mean8 top-K: count by best_head."""
    cnt = wiped["best_head"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([f"H{int(h)}" for h in cnt.index], cnt.values, color=color, edgecolor="k", linewidth=0.4)
    ax.set_ylabel(f"CHIP true edges (n={len(wiped)})")
    ax.set_xlabel("Head with highest score on this edge")
    ax.set_title(f"{display} — {dataset}\nMean-wiped CHIP edges: which head ranked them best?")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(model: str, args: argparse.Namespace) -> None:
    display = ath.MODEL_ALIASES.get(ath.model_file_prefix(model), model)
    color = model_color(display)
    tag = f"{ath.model_file_prefix(model)}_{args.dataset}"
    out_dir = Path(args.output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    wiped_path = out_dir / f"{tag}_mean_wiped_chip_edges.csv"
    if not wiped_path.exists():
        raise FileNotFoundError(f"Run case_study_mean_delta_edges.py first: {wiped_path}")
    wiped = pd.read_csv(wiped_path)

    print(f"[INFO] Building lookups for plots ({tag})...")
    per_head_lk, mean_lk, k_eval, per_head_topk, mean_topk = load_lookups(model, args)

    plot_case_score_bars(
        per_head_lk, mean_lk, out_dir / f"{tag}_case_score_bars.pdf", display, args.dataset, color
    )
    plot_case_rank_bars(
        per_head_lk, mean_lk, k_eval, out_dir / f"{tag}_case_rank_bars.pdf", display, args.dataset, color
    )
    plot_tfap2a_heatmaps(
        wiped,
        per_head_lk,
        mean_lk,
        k_eval,
        mean_topk,
        per_head_topk,
        out_dir / f"{tag}_tfap2a_wiped_score_heatmap.pdf",
        out_dir / f"{tag}_tfap2a_wiped_topk_heatmap.pdf",
        display,
        args.dataset,
        args.tfap2a_top,
    )
    plot_wiped_counts(wiped, out_dir / f"{tag}_wiped_by_best_head.pdf", display, args.dataset, color)

    print(f"[INFO] Wrote figures -> {out_dir}")
    print(f"  {tag}_case_score_bars.pdf")
    print(f"  {tag}_case_rank_bars.pdf")
    print(f"  {tag}_tfap2a_wiped_score_heatmap.pdf")
    print(f"  {tag}_tfap2a_wiped_topk_heatmap.pdf")
    print(f"  {tag}_wiped_by_best_head.pdf")


def parse_args():
    p = argparse.ArgumentParser(description="Plot mean-wiped CHIP case studies.")
    p.add_argument("--models", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--head-root", action="append", required=True)
    p.add_argument("--chip-root", default="/mnt/10T/yzn/benchmark_GRN/input_process/CHIP")
    p.add_argument("--output-dir", default=str(_SCRIPT_DIR / "output" / "head_chip_auprc"))
    p.add_argument("--top-edges", type=int, default=0)
    p.add_argument("--load-max-edges", type=int, default=0)
    p.add_argument("--pred-direction", default="sym_max")
    p.add_argument("--min-frac-nonzero", type=float, default=0.05)
    p.add_argument("--tfap2a-top", type=int, default=15, help="Rows in TFAP2A heatmap")
    args = p.parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    return args


def main():
    args = parse_args()
    for m in args.models:
        run(m, args)


if __name__ == "__main__":
    main()
