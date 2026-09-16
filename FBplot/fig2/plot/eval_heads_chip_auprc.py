#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L3 interpretability: CHIP-grounded AUPRC for per-attention-head GRN exports.

What this script does
---------------------
1. For each attention head (and optional fused GRNs: mean / max across heads),
   score how well predicted edge weights rank CHIP-confirmed TF→target pairs
   against sampled negatives (same protocol as benchmark_GRN/plot_paper/CHIP).

2. Write tables + figures under output/head_chip_auprc/{model}_{dataset}/.

This answers: "Which head is best for regulatory prediction?" and
"How much do we lose by averaging heads?"  (G = best_single - mean).

Usage (example)
---------------
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig2/plot

  python eval_heads_chip_auprc.py \\
    --models scgpt \\
    --dataset hESC \\
    --head-root /mnt/10T/yzn/scGRN-Bench/FBplot/fig2/att_head \\
    --chip-root /mnt/10T/yzn/benchmark_GRN/input_process/CHIP \\
    --top-edges 10000 \\
    --neg-mode random

See module docstring at bottom of file for output file descriptions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import analyze_attention_heads as ath  # noqa: E402
from fig2_palette import model_color  # noqa: E402

# Reuse TSV loading / head discovery from stat_analyze_attention_heads.py
import stat_analyze_attention_heads as sah  # noqa: E402


# ---------------------------------------------------------------------------
# CHIP I/O
# ---------------------------------------------------------------------------


def read_chip_gt(gt_path: Path) -> pd.DataFrame:
    """CHIP network: directed edges Gene1=TF, Gene2=target."""
    df = pd.read_csv(gt_path)
    df = ath.normalize_edges(df)
    return df[["Gene1", "Gene2"]].copy()


def read_chip_expr(expr_path: Path) -> pd.DataFrame:
    """Expression matrix: rows=genes, columns=samples (cells)."""
    df = pd.read_csv(expr_path, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df.index = df.index.astype(str)
    return df


def active_genes(expr: pd.DataFrame, min_frac_nonzero: float) -> Set[str]:
    """Genes expressed in at least min_frac_nonzero fraction of cells."""
    frac = (expr.values > 0).mean(axis=1)
    return set(expr.index[frac >= float(min_frac_nonzero)].tolist())


def gt_tf_to_targets(gt: pd.DataFrame, active: Optional[Set[str]] = None) -> Dict[str, Set[str]]:
    """Map each TF -> set of positive target genes from CHIP."""
    g = gt
    if active is not None:
        g = g[g["Gene1"].isin(active) & g["Gene2"].isin(active)]
    out: Dict[str, Set[str]] = {}
    for tf, sub in g.groupby("Gene1", sort=False):
        out[str(tf)] = set(sub["Gene2"].astype(str).tolist())
    return out


# ---------------------------------------------------------------------------
# Prediction lookup (TF, target) -> score
# ---------------------------------------------------------------------------


def build_tf_lookup(
    pred: pd.DataFrame,
    tf_set: Set[str],
    direction: str,
) -> Dict[Tuple[str, str], float]:
    """
    Build a dict keyed by (TF, target) for CHIP evaluation.

    direction (TSV: Gene1=Query, Gene2=Key; CHIP: (TF, target)):
      - forward:      TF as Query → target as Key (aligns with ChIP arrow)
      - backward:     target as Query → TF as Key (reverse attention flow)
      - key_incoming: alias of backward (TF as Key, for Key-centric narrative)
      - sym_max:      max(forward, backward) per (TF, target) [undirected]
      - as_exported / tf_gene1: legacy aliases of forward
    """
    lookup: Dict[Tuple[str, str], float] = {}
    if pred.empty:
        return lookup

    g1 = pred["Gene1"].astype(str).values
    g2 = pred["Gene2"].astype(str).values
    w = pred["EdgeWeight"].astype(float).values

    dir_norm = direction
    if dir_norm in ("as_exported", "tf_gene1"):
        dir_norm = "forward"
    if dir_norm == "key_incoming":
        dir_norm = "backward"

    for i in range(len(pred)):
        a, b, wt = g1[i], g2[i], float(w[i])
        if dir_norm == "forward":
            if a in tf_set:
                lookup[(a, b)] = max(lookup.get((a, b), 0.0), wt)
        elif dir_norm == "backward":
            if b in tf_set:
                lookup[(b, a)] = max(lookup.get((b, a), 0.0), wt)
        elif dir_norm == "sym_max":
            if a in tf_set:
                lookup[(a, b)] = max(lookup.get((a, b), 0.0), wt)
            if b in tf_set:
                lookup[(b, a)] = max(lookup.get((b, a), 0.0), wt)
        else:
            raise ValueError(f"Unknown direction: {direction}")
    return lookup


# Directions compared in the direction-sensitivity panel (key_incoming == backward at edge level).
DIRECTION_COMPARE_MODES = [
    ("forward", "Forward (TF→target)"),
    ("backward", "Backward / Key-centric"),
    ("sym_max", "Undirected (sym_max)"),
]

# Six CHIP benchmarks used in fig4 cross-dataset analyses.
DEFAULT_CHIP_DATASETS = ["hESC", "hHep", "mDC", "mHSC-E", "mHSC-GM", "mHSC-L"]


def chip_gene_sets_from_gt(gt: pd.DataFrame) -> Tuple[Set[str], Set[str]]:
    """
    Same node sets as AUPR1000 (TFEdges=True):
      TFs   = genes appearing as Gene1 in CHIP GT
      Genes = Gene1 ∪ Gene2 in CHIP GT (valid regulators and targets)
    """
    tfs = set(gt["Gene1"].astype(str))
    genes = set(gt["Gene1"].astype(str)) | set(gt["Gene2"].astype(str))
    return tfs, genes


def filter_pred_aupr_style(pred: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """
    After expression/active filtering, apply AUPR-style edge filter:
      Gene1 ∈ TFs,  Gene2 ∈ Genes  (from CHIP ground truth).
    """
    tfs, genes = chip_gene_sets_from_gt(gt)
    out = pred[pred["Gene1"].isin(tfs) & pred["Gene2"].isin(genes)].copy()
    return out.sort_values("EdgeWeight", ascending=False).reset_index(drop=True)


def filter_pred_direction_style(pred: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """
    Keep both orientations for direction comparison:
      forward rows:  TF (Query) → gene (Key)
      backward rows: gene (Query) → TF (Key)
    """
    tfs, genes = chip_gene_sets_from_gt(gt)
    fwd = pred["Gene1"].isin(tfs) & pred["Gene2"].isin(genes)
    bwd = pred["Gene1"].isin(genes) & pred["Gene2"].isin(tfs)
    out = pred[fwd | bwd].copy()
    return out.sort_values("EdgeWeight", ascending=False).reset_index(drop=True)


def load_head_pred(
    fp: Path,
    gene_universe: Set[str],
    load_max_edges: int,
) -> pd.DataFrame:
    """
    Load one head TSV; keep Gene1→Gene2.

    load_max_edges: max rows to keep after active-gene filter (0 = all rows).
    Do NOT truncate to |GT| here — that happens after filter_pred_aupr_style + top-K.
    """
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
        df["EdgeWeight"] = pd.to_numeric(df["EdgeWeight"], errors="coerce").fillna(0.0).abs()
    df = df.sort_values("EdgeWeight", ascending=False)
    if load_max_edges > 0:
        df = df.head(load_max_edges)
    return df[["Gene1", "Gene2", "EdgeWeight"]].reset_index(drop=True)


def fuse_predictions_directed(
    per_head: Dict[int, pd.DataFrame],
    mode: str = "mean",
) -> pd.DataFrame:
    """Fuse per-head tables keeping Query→Key orientation (for mean8 direction eval)."""
    if not per_head:
        return pd.DataFrame(columns=["Gene1", "Gene2", "EdgeWeight"])
    acc: Dict[Tuple[str, str], List[float]] = {}
    for df in per_head.values():
        if df.empty:
            continue
        for row in df.itertuples(index=False):
            key = (str(row.Gene1), str(row.Gene2))
            acc.setdefault(key, []).append(float(row.EdgeWeight))
    rows = []
    for (a, b), ws in acc.items():
        val = float(np.mean(ws)) if mode == "mean" else float(np.max(ws))
        rows.append({"Gene1": a, "Gene2": b, "EdgeWeight": val})
    return pd.DataFrame(rows)


def fuse_predictions(
    per_head: Dict[int, pd.DataFrame],
    mode: str,
) -> pd.DataFrame:
    """
    Combine per-head edge tables into one GRN.

    mode:
      - mean: average EdgeWeight over heads that contain the undirected pair
      - max:  take maximum weight per undirected pair
    """
    if not per_head:
        return pd.DataFrame(columns=["Gene1", "Gene2", "EdgeWeight"])

    # Accumulate list of weights per canonical undirected pair
    acc: Dict[Tuple[str, str], List[float]] = {}
    for df in per_head.values():
        if df.empty:
            continue
        for row in df.itertuples(index=False):
            a, b, wt = str(row.Gene1), str(row.Gene2), float(row.EdgeWeight)
            key = (a, b) if a <= b else (b, a)
            acc.setdefault(key, []).append(wt)

    rows = []
    for (a, b), ws in acc.items():
        if mode == "mean":
            val = float(np.mean(ws))
        elif mode == "max":
            val = float(np.max(ws))
        else:
            raise ValueError(mode)
        rows.append({"Gene1": a, "Gene2": b, "EdgeWeight": val})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# AUPRC evaluation (TF-centric, CHIP protocol)
# ---------------------------------------------------------------------------


def auprc(y: np.ndarray, s: np.ndarray) -> float:
    if len(y) < 2 or len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, s))


def scores_for_edges(edges: List[Tuple[str, str]], lookup: Dict[Tuple[str, str], float]) -> np.ndarray:
    return np.array([lookup.get(e, 0.0) for e in edges], dtype=float)


def sample_negatives_random(
    pos_targets: Set[str],
    target_pool: List[str],
    n_neg: int,
    rng: np.random.Generator,
) -> List[str]:
    pool = [g for g in target_pool if g not in pos_targets]
    if n_neg <= 0 or not pool:
        return []
    n_neg = min(n_neg, len(pool))
    return rng.choice(pool, size=n_neg, replace=False).tolist()


def precompute_tf_evaluation_edges(
    gt: pd.DataFrame,
    target_pool: List[str],
    neg_ratio: float,
    seed: int,
) -> Dict[str, Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]]:
    """Per-TF (pos_edges, neg_edges) with fixed negatives for fair direction comparison."""
    rng = np.random.default_rng(seed)
    pool_set = set(target_pool)
    tf_pos = gt_tf_to_targets(gt, active=pool_set)
    out: Dict[str, Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]] = {}
    for tf, pos_targets in tf_pos.items():
        pos_targets = {t for t in pos_targets if t in pool_set}
        if not pos_targets:
            continue
        n_neg = int(round(len(pos_targets) * float(neg_ratio)))
        if n_neg <= 0:
            continue
        neg_targets = sample_negatives_random(pos_targets, target_pool, n_neg, rng)
        pos_edges = [(tf, t) for t in sorted(pos_targets)]
        neg_edges = [(tf, t) for t in sorted(neg_targets)]
        out[tf] = (pos_edges, neg_edges)
    return out


def evaluate_on_fixed_edges(
    tf_edges: Dict[str, Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]],
    lookup: Dict[Tuple[str, str], float],
) -> Tuple[Optional[Dict[str, float]], pd.DataFrame, np.ndarray, np.ndarray]:
    """TF-centric AUPRC using pre-sampled pos/neg edges (shared across directions)."""
    per_tf_rows = []
    y_all: List[np.ndarray] = []
    s_all: List[np.ndarray] = []

    for tf, (pos_edges, neg_edges) in tf_edges.items():
        edges = pos_edges + neg_edges
        y = np.array([1] * len(pos_edges) + [0] * len(neg_edges), dtype=int)
        s = scores_for_edges(edges, lookup)
        per_tf_rows.append(
            {
                "TF": tf,
                "num_pos": len(pos_edges),
                "num_neg": len(neg_edges),
                "AUPRC": auprc(y, s),
            }
        )
        y_all.append(y)
        s_all.append(s)

    per_tf = pd.DataFrame(per_tf_rows)
    if per_tf.empty:
        return None, per_tf, np.array([]), np.array([])

    y_cat = np.concatenate(y_all)
    s_cat = np.concatenate(s_all)
    global_out = {
        "AUPRC_micro": auprc(y_cat, s_cat),
        "AUPRC_macro": float(np.nanmean(per_tf["AUPRC"].values)),
        "num_TF": int(per_tf.shape[0]),
        "num_pairs": int(len(y_cat)),
        "pos_rate": float(y_cat.mean()),
    }
    return global_out, per_tf, y_cat, s_cat


def evaluate_tf_centric(
    gt: pd.DataFrame,
    lookup: Dict[Tuple[str, str], float],
    target_pool: List[str],
    neg_ratio: float,
    seed: int,
) -> Tuple[Optional[Dict[str, float]], pd.DataFrame]:
    """
    For each TF in CHIP GT:
      - positives: GT targets (in target_pool)
      - negatives: random targets, count = neg_ratio * |positives|
      - AUPRC on concatenated edges for that TF

    Returns global metrics dict and per-TF table.
    """
    rng = np.random.default_rng(seed)
    tf_pos = gt_tf_to_targets(gt, active=set(target_pool))
    pool_set = set(target_pool)

    per_tf_rows = []
    y_all: List[np.ndarray] = []
    s_all: List[np.ndarray] = []

    for tf, pos_targets in tf_pos.items():
        pos_targets = {t for t in pos_targets if t in pool_set}
        if not pos_targets:
            continue
        n_neg = int(round(len(pos_targets) * float(neg_ratio)))
        if n_neg <= 0:
            continue
        neg_targets = sample_negatives_random(pos_targets, target_pool, n_neg, rng)
        pos_edges = [(tf, t) for t in sorted(pos_targets)]
        neg_edges = [(tf, t) for t in sorted(neg_targets)]
        edges = pos_edges + neg_edges
        y = np.array([1] * len(pos_edges) + [0] * len(neg_edges), dtype=int)
        s = scores_for_edges(edges, lookup)
        per_tf_rows.append(
            {
                "TF": tf,
                "num_pos": len(pos_edges),
                "num_neg": len(neg_edges),
                "AUPRC": auprc(y, s),
            }
        )
        y_all.append(y)
        s_all.append(s)

    per_tf = pd.DataFrame(per_tf_rows)
    if per_tf.empty:
        return None, per_tf

    y_cat = np.concatenate(y_all)
    s_cat = np.concatenate(s_all)
    global_out = {
        "AUPRC_micro": auprc(y_cat, s_cat),
        "AUPRC_macro": float(np.nanmean(per_tf["AUPRC"].values)),
        "num_TF": int(per_tf.shape[0]),
        "num_pairs": int(len(y_cat)),
        "pos_rate": float(y_cat.mean()),
    }
    return global_out, per_tf


def chip_true_edges_set(gt: pd.DataFrame) -> Set[Tuple[str, str]]:
    return set(zip(gt["Gene1"].astype(str), gt["Gene2"].astype(str)))


def global_rank(lookup: Dict[Tuple[str, str], float], edge: Tuple[str, str]) -> int:
    """1 = highest score among pairs in lookup; larger rank = weaker."""
    if edge not in lookup:
        return len(lookup) + 1
    items = sorted(lookup.items(), key=lambda x: x[1], reverse=True)
    for i, (e, _) in enumerate(items, start=1):
        if e == edge:
            return i
    return len(lookup) + 1


def resolve_top_k(n_true_edges: int, top_edges_arg: int) -> int:
    """
    Align with benchmark AUPR1000: Top-K = number of ground-truth edges.

    top_edges_arg <= 0  -> use all n_true_edges
    top_edges_arg > 0   -> min(user_k, n_true_edges)
    """
    if n_true_edges <= 0:
        return 0
    if top_edges_arg <= 0:
        return n_true_edges
    return min(int(top_edges_arg), n_true_edges)


def precision_recall_at_k(
    lookup: Dict[Tuple[str, str], float],
    gt_edges: Set[Tuple[str, str]],
    k: int,
) -> Tuple[float, float]:
    """
    Rank all scored TF-aligned pairs in lookup; take top-k (k should be |GT| for AUPR-style).

    precision = |top-k ∩ GT| / k
    recall    = |top-k ∩ GT| / |GT|
    """
    if k <= 0 or not lookup or not gt_edges:
        return np.nan, np.nan
    # Same cap as AUPR1000: K = min(n_pred_scored, n_true)
    k_eff = min(k, len(lookup))
    items = sorted(lookup.items(), key=lambda x: x[1], reverse=True)
    top = items[:k_eff]
    hits = sum(1 for e, _ in top if e in gt_edges)
    prec = hits / k_eff
    rec = hits / len(gt_edges)
    return float(prec), float(rec)


def exclusive_chip_hits(
    per_head_topk: Dict[int, Set[Tuple[str, str]]],
    gt_edges: Set[Tuple[str, str]],
) -> pd.DataFrame:
    """
    CHIP true edges that appear in exactly one head's top-k edge set
    (semantic exclusivity for interpretability).
    """
    rows = []
    for e in gt_edges:
        heads_with = [h for h, es in per_head_topk.items() if e in es]
        if len(heads_with) == 1:
            rows.append({"Gene1": e[0], "Gene2": e[1], "exclusive_head": heads_with[0]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_auprc_bar(summary: pd.DataFrame, display: str, dataset: str, color: str, out_pdf: Path) -> None:
    """Bar chart: H0–H7 + mean8; highlight best single head vs mean8."""
    order = [f"head{h}" for h in range(8)] + ["mean8", "max8"]
    sub = summary[summary["strategy"].isin(order)].copy()
    if sub.empty:
        sub = summary.sort_values("AUPRC_micro", ascending=False)
    else:
        sub["order"] = sub["strategy"].map({s: i for i, s in enumerate(order)})
        sub = sub.sort_values("order")

    labels = []
    for s in sub["strategy"]:
        if str(s).startswith("head"):
            labels.append(f"H{s.replace('head', '')}")
        elif s == "mean8":
            labels.append("Mean8")
        elif s == "max8":
            labels.append("Max8")
        else:
            labels.append(str(s))

    vals = sub["AUPRC_micro"].astype(float).values
    head_mask = sub["strategy"].str.match(r"head\d+")
    best_strategy = ""
    if head_mask.any():
        best_strategy = str(
            sub.loc[head_mask].sort_values("AUPRC_micro", ascending=False)["strategy"].iloc[0]
        )

    fig, ax = plt.subplots(figsize=(max(7, len(sub) * 0.65), 4.2))
    bar_colors = [color] * len(sub)
    for i, lab in enumerate(sub["strategy"]):
        if lab == "mean8":
            bar_colors[i] = "#B0B0B0"
        elif str(lab) == best_strategy:
            bar_colors[i] = "#EB7E60"

    bars = ax.bar(range(len(sub)), vals, color=bar_colors, edgecolor="k", linewidth=0.4)
    for i, lab in enumerate(sub["strategy"]):
        if lab == "mean8":
            bars[i].set_alpha(0.75)
            bars[i].set_hatch("//")

    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUPRC (micro, CHIP TF-centric)")
    title_extra = ""
    if "mean8" in sub["strategy"].values and head_mask.any():
        mean_v = float(sub.loc[sub["strategy"] == "mean8", "AUPRC_micro"].iloc[0])
        best_v = float(sub.loc[head_mask, "AUPRC_micro"].max())
        if best_v > mean_v:
            title_extra = f"\nBest head > Mean8 (Δ={best_v - mean_v:.3f})"
    ax.set_title(f"{display} — {dataset}\nSingle-head AUPRC ranking{title_extra}")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def pick_compare_head(
    per_head_dir_df: Dict[int, pd.DataFrame],
    tf_edges: Dict[str, Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]],
    tf_set: Set[str],
) -> int:
    """Head with highest sym_max micro-AUPRC (direction-filtered preds)."""
    best_h, best_au = 0, -1.0
    for h, df in per_head_dir_df.items():
        lk = build_tf_lookup(df, tf_set, "sym_max")
        glob, _, _, _ = evaluate_on_fixed_edges(tf_edges, lk)
        if glob is None:
            continue
        au = float(glob["AUPRC_micro"])
        if au > best_au:
            best_au, best_h = au, h
    return best_h


def incoming_weights_for_tf(
    pred: pd.DataFrame,
    tf: str,
    gene_pool: List[str],
) -> pd.Series:
    """Query→Key weights with Key=tf (backward / Key-centric view)."""
    sub = pred[(pred["Gene2"] == tf) & (pred["Gene1"].isin(gene_pool))].copy()
    if sub.empty:
        return pd.Series(dtype=float)
    agg = sub.groupby("Gene1", as_index=True)["EdgeWeight"].max()
    return agg.sort_values(ascending=False)


# PR curve colors (direction panel)
_DIR_COLORS = {
    "forward": "#4C72B0",
    "backward": "#55A868",
    "sym_max": "#DD8452",
}


def plot_direction_pr_curves(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    display: str,
    dataset: str,
    head_label: str,
    out_pdf: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    for mode, label in DIRECTION_COMPARE_MODES:
        y, s, ap = curves[mode]
        if len(y) < 2 or len(np.unique(y)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y, s)
        ax.plot(rec, prec, color=_DIR_COLORS[mode], lw=2, label=f"{label} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title(f"{display} — {dataset}\nPR curves ({head_label}, shared negatives)")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_direction_bidirectional_scatter(
    per_tf_fwd: pd.DataFrame,
    per_tf_bwd: pd.DataFrame,
    display: str,
    dataset: str,
    head_label: str,
    out_pdf: Path,
) -> None:
    merged = per_tf_fwd[["TF", "AUPRC"]].merge(
        per_tf_bwd[["TF", "AUPRC"]],
        on="TF",
        suffixes=("_forward", "_backward"),
    )
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    x = merged["AUPRC_forward"].astype(float).values
    y = merged["AUPRC_backward"].astype(float).values
    lim = max(0.05, float(np.nanmax([x.max(), y.max(), 0.5])))
    ax.scatter(x, y, s=28, c="#333333", alpha=0.75, edgecolors="none")
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Forward AUPRC (TF→target)")
    ax.set_ylabel("Backward AUPRC (target→TF)")
    ax.set_title(f"{display} — {dataset}\nPer-TF direction comparison ({head_label})")
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_direction_auprc_bar(
    summary: pd.DataFrame,
    display: str,
    dataset: str,
    head_label: str,
    out_pdf: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    labels = summary["label"].tolist()
    vals = summary["AUPRC_micro"].astype(float).values
    colors = [_DIR_COLORS.get(m, "#888888") for m in summary["direction"].tolist()]
    ax.bar(range(len(vals)), vals, color=colors, edgecolor="k", linewidth=0.4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("AUPRC (micro)")
    ax.set_ylim(0, min(1.0, max(vals) * 1.15 + 0.05) if len(vals) else 1)
    ax.set_title(f"{display} — {dataset}\nScoring direction ({head_label})")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_key_incoming_heatmap(
    pred: pd.DataFrame,
    gt: pd.DataFrame,
    tf: str,
    active: Set[str],
    display: str,
    dataset: str,
    head_label: str,
    out_pdf: Path,
    top_n: int = 40,
) -> None:
    import seaborn as sns

    weights = incoming_weights_for_tf(pred, tf, sorted(active))
    if weights.empty:
        return
    true_targets = set(gt.loc[gt["Gene1"] == tf, "Gene2"].astype(str))
    top_genes = weights.head(top_n).index.tolist()
    mat = weights.loc[top_genes].to_frame(name="incoming").T
    is_chip = [g in true_targets for g in top_genes]

    fig_w = max(7, len(top_genes) * 0.14)
    fig, ax = plt.subplots(figsize=(fig_w, 2.2))
    sns.heatmap(
        mat.astype(float),
        cmap="YlOrRd",
        cbar_kws={"label": "Attention (Query→TF Key)"},
        ax=ax,
        yticklabels=[tf],
    )
    for j, chip in enumerate(is_chip):
        if chip:
            ax.plot(j + 0.5, 1.08, marker="v", color="#C0392B", markersize=6, clip_on=False)
    ax.set_xlabel("Query gene (ranked by weight to TF Key)")
    ax.set_title(
        f"{display} — {dataset}\nKey-centric incoming ({head_label}, TF={tf})\n"
        "red ▼ = CHIP target",
        fontsize=9,
    )
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_direction_panel_2x2(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    per_tf_fwd: pd.DataFrame,
    per_tf_bwd: pd.DataFrame,
    summary: pd.DataFrame,
    pred_df: pd.DataFrame,
    gt: pd.DataFrame,
    tf: str,
    active: Set[str],
    display: str,
    dataset: str,
    head_label: str,
    out_pdf: Path,
    top_n: int,
) -> None:
    """Single 2×2 publication panel (native axes, not PDF stitch)."""
    import seaborn as sns

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    for mode, label in DIRECTION_COMPARE_MODES:
        y, s, ap = curves[mode]
        if len(y) < 2 or len(np.unique(y)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y, s)
        ax.plot(rec, prec, color=_DIR_COLORS[mode], lw=2, label=f"{label} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("(a) PR curves (shared negatives)")

    ax = axes[0, 1]
    merged = per_tf_fwd[["TF", "AUPRC"]].merge(
        per_tf_bwd[["TF", "AUPRC"]], on="TF", suffixes=("_forward", "_backward")
    )
    x = merged["AUPRC_forward"].astype(float).values
    yv = merged["AUPRC_backward"].astype(float).values
    lim = max(0.05, float(np.nanmax([x.max(), yv.max(), 0.5])))
    ax.scatter(x, yv, s=24, c="#333333", alpha=0.75)
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Forward AUPRC")
    ax.set_ylabel("Backward AUPRC")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("(b) Per-TF direction scatter")

    ax = axes[1, 0]
    labels = summary["label"].tolist()
    vals = summary["AUPRC_micro"].astype(float).values
    colors = [_DIR_COLORS.get(m, "#888") for m in summary["direction"].tolist()]
    ax.bar(range(len(vals)), vals, color=colors, edgecolor="k", linewidth=0.4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("AUPRC (micro)")
    ax.set_ylim(0, min(1.0, max(vals) * 1.12 + 0.03))
    ax.set_title("(c) Direction comparison")

    ax = axes[1, 1]
    weights = incoming_weights_for_tf(pred_df, tf, sorted(active))
    if not weights.empty:
        true_targets = set(gt.loc[gt["Gene1"] == tf, "Gene2"].astype(str))
        top_genes = weights.head(top_n).index.tolist()
        mat = weights.loc[top_genes].to_frame(name="w").T
        is_chip = [g in true_targets for g in top_genes]
        sns.heatmap(mat.astype(float), cmap="YlOrRd", cbar_kws={"label": "attn"}, ax=ax, yticklabels=[tf])
        for j, chip in enumerate(is_chip):
            if chip:
                ax.plot(j + 0.5, 1.06, marker="v", color="#C0392B", markersize=5, clip_on=False)
        ax.set_xlabel("Query gene")
        ax.set_title(f"(d) Key-centric ({tf})")
    else:
        ax.text(0.5, 0.5, f"No incoming edges for {tf}", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"(d) Key-centric ({tf})")

    fig.suptitle(f"{display} — {dataset}  |  {head_label}", y=1.01, fontsize=11)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_direction_comparison(
    tag: str,
    display: str,
    pred_df: pd.DataFrame,
    gt: pd.DataFrame,
    target_pool: List[str],
    tf_set: Set[str],
    active: Set[str],
    out_dir: Path,
    head_label: str,
    args: argparse.Namespace,
    pr_only: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[np.ndarray, np.ndarray, float]]]:
    """Compare forward / backward / sym_max on one head with shared negatives."""
    tf_edges = precompute_tf_evaluation_edges(gt, target_pool, args.neg_ratio, args.seed)
    summary_rows = []
    curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
    per_tf_by_mode: Dict[str, pd.DataFrame] = {}

    for mode, label in DIRECTION_COMPARE_MODES:
        lookup = build_tf_lookup(pred_df, tf_set, mode)
        glob, per_tf, y_cat, s_cat = evaluate_on_fixed_edges(tf_edges, lookup)
        if glob is None:
            continue
        summary_rows.append({"direction": mode, "label": label, **glob})
        curves[mode] = (y_cat, s_cat, glob["AUPRC_micro"])
        per_tf_by_mode[mode] = per_tf
        print(f"  [direction] {mode}: AUPRC_micro={glob['AUPRC_micro']:.4f}  macro={glob['AUPRC_macro']:.4f}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / f"{tag}_direction_auprc_summary.csv", index=False)

    if per_tf_by_mode:
        long_parts = []
        for mode, label in DIRECTION_COMPARE_MODES:
            if mode not in per_tf_by_mode:
                continue
            pt = per_tf_by_mode[mode].copy()
            pt["direction"] = mode
            pt["label"] = label
            long_parts.append(pt)
        pd.concat(long_parts, ignore_index=True).to_csv(
            out_dir / f"{tag}_direction_auprc_per_tf.csv", index=False
        )

    prefix = out_dir / f"{tag}_direction"
    if curves:
        plot_direction_pr_curves(curves, display, args.dataset, head_label, Path(f"{prefix}_pr_curves.pdf"))

    if not pr_only:
        if "forward" in per_tf_by_mode and "backward" in per_tf_by_mode:
            plot_direction_bidirectional_scatter(
                per_tf_by_mode["forward"],
                per_tf_by_mode["backward"],
                display,
                args.dataset,
                head_label,
                Path(f"{prefix}_bidirectional_scatter.pdf"),
            )
        if not summary.empty:
            plot_direction_auprc_bar(summary, display, args.dataset, head_label, Path(f"{prefix}_auprc_bar.pdf"))

        example_tf = args.example_tf
        if example_tf and example_tf in tf_set:
            plot_key_incoming_heatmap(
                pred_df,
                gt,
                example_tf,
                active,
                display,
                args.dataset,
                head_label,
                Path(f"{prefix}_key_heatmap_{example_tf}.pdf"),
                top_n=args.heatmap_top_genes,
            )

        if curves and "forward" in per_tf_by_mode and "backward" in per_tf_by_mode and not summary.empty:
            plot_direction_panel_2x2(
                curves=curves,
                per_tf_fwd=per_tf_by_mode["forward"],
                per_tf_bwd=per_tf_by_mode["backward"],
                summary=summary,
                pred_df=pred_df,
                gt=gt,
                tf=example_tf or "POU5F1",
                active=active,
                display=display,
                dataset=args.dataset,
                head_label=head_label,
                out_pdf=Path(f"{prefix}_panel_2x2.pdf"),
                top_n=args.heatmap_top_genes,
            )

    return summary, curves


def plot_direction_pr_multipanel(
    curves_by_dataset: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, float]]],
    display: str,
    head_label: str,
    out_pdf: Path,
) -> None:
    """2×3 PR curves (six datasets) for mean8 direction comparison."""
    datasets = [d for d in DEFAULT_CHIP_DATASETS if d in curves_by_dataset]
    if not datasets:
        return
    n = len(datasets)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()
    for ax, ds in zip(axes_flat, datasets):
        curves = curves_by_dataset[ds]
        for mode, label in DIRECTION_COMPARE_MODES:
            y, s, ap = curves[mode]
            if len(y) < 2 or len(np.unique(y)) < 2:
                continue
            prec, rec, _ = precision_recall_curve(y, s)
            ax.plot(rec, prec, color=_DIR_COLORS[mode], lw=1.8, label=f"{label.split('(')[0].strip()} AP={ap:.2f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(ds, fontsize=10)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
    for ax in axes_flat[len(datasets) :]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{display} — mean8 head average\nDirection PR (shared negatives per dataset)", y=1.02, fontsize=11)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_direction_pr_mean8_one(
    model: str,
    dataset: str,
    args: argparse.Namespace,
    out_root: Path,
) -> Tuple[str, pd.DataFrame, Dict[str, Tuple[np.ndarray, np.ndarray, float]]]:
    """Load 8 heads → directed mean8 → direction PR only."""
    display = ath.MODEL_ALIASES.get(ath.model_file_prefix(model), model)
    tag = f"{ath.model_file_prefix(model)}_{dataset}"
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    head_roots = [Path(p) for p in args.head_root]
    files = ath.find_head_files(head_roots, model, dataset)
    if not files:
        print(f"[WARN] skip {tag}: no head TSV")
        return dataset, pd.DataFrame(), {}

    chip_root = Path(args.chip_root)
    gt_path = chip_root / f"{dataset}_chip_matched-network.csv"
    expr_path = chip_root / f"{dataset}_chip_matched-ExpressionData.csv"
    gt_all = read_chip_gt(gt_path)
    expr = read_chip_expr(expr_path)
    active = active_genes(expr, args.min_frac_nonzero)
    gt = gt_all[gt_all["Gene1"].isin(active) & gt_all["Gene2"].isin(active)].copy()
    tf_set = set(gt["Gene1"].astype(str))
    target_pool = sorted(active)

    per_head_dir: Dict[int, pd.DataFrame] = {}
    for fp in files:
        h = ath.parse_head_number(fp)
        raw = load_head_pred(fp, active, args.load_max_edges)
        per_head_dir[h] = filter_pred_direction_style(raw, gt)
        print(f"  {tag} head{h}: direction rows={len(per_head_dir[h])}")

    mean8 = fuse_predictions_directed(per_head_dir, "mean")
    print(f"  {tag} mean8 directed edges={len(mean8)}")

    args.dataset = dataset
    summary, curves = run_direction_comparison(
        tag=tag,
        display=display,
        pred_df=mean8,
        gt=gt,
        target_pool=target_pool,
        tf_set=tf_set,
        active=active,
        out_dir=out_dir,
        head_label="mean8",
        args=args,
        pr_only=True,
    )
    return dataset, summary, curves


SCGPT_MULTIHEAD_ROOT = Path(
    "/mnt/10T/yzn/benchmark_GRN/evl_omipath/output_att500/scgpt_multihead"
)


def run_direction_pr_mean8_batch(args: argparse.Namespace) -> None:
    """Six datasets × scGPT mean8 → PR curves only."""
    if not args.head_root:
        args.head_root = [str(SCGPT_MULTIHEAD_ROOT)]

    if args.datasets.strip():
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        datasets = list(DEFAULT_CHIP_DATASETS)

    out_root = Path(args.output_dir) / "direction_pr_mean8"
    out_root.mkdir(parents=True, exist_ok=True)

    all_summary = []
    curves_by_ds: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, float]]] = {}

    for model in args.models:
        display = ath.MODEL_ALIASES.get(ath.model_file_prefix(model), model)
        for ds in datasets:
            print(f"\n[INFO] direction PR mean8: {model} / {ds}")
            _, summary, curves = run_direction_pr_mean8_one(model, ds, args, out_root)
            if not summary.empty:
                summary = summary.copy()
                summary["dataset"] = ds
                all_summary.append(summary)
            if curves:
                curves_by_ds[ds] = curves

        if curves_by_ds:
            combined = out_root / f"{ath.model_file_prefix(model)}_mean8_direction_pr_6datasets.pdf"
            plot_direction_pr_multipanel(curves_by_ds, display, "mean8", combined)

    if all_summary:
        pd.concat(all_summary, ignore_index=True).to_csv(
            out_root / "mean8_direction_auprc_all_datasets.csv", index=False
        )
    print(f"\n[INFO] Wrote direction PR (mean8) -> {out_root}")


def plot_tf_head_heatmap(matrix: pd.DataFrame, display: str, dataset: str, out_pdf: Path) -> None:
    """Heatmap: per-TF AUPRC across heads (subset of TFs if too many)."""
    import seaborn as sns

    mat = matrix.astype(float)
    # Show at most 40 TFs with highest variance across heads
    if mat.shape[0] > 40:
        var = mat.var(axis=1)
        mat = mat.loc[var.nlargest(40).index]
    fig_w = max(6, mat.shape[1] * 0.65)
    fig_h = max(4, mat.shape[0] * 0.22)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        mat,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        annot=False,
        cbar_kws={"label": "AUPRC"},
        ax=ax,
    )
    ax.set_xlabel("Attention head / strategy")
    ax.set_ylabel("TF")
    ax.set_title(f"{display} — {dataset}\nPer-TF AUPRC (which TF is captured by which head?)")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main analysis for one (model, dataset)
# ---------------------------------------------------------------------------


def analyze_model_dataset(args: argparse.Namespace) -> None:
    for model in args.models:
        _run_one(model, args)


def _run_one(model: str, args: argparse.Namespace) -> None:
    display = ath.MODEL_ALIASES.get(ath.model_file_prefix(model), model)
    color = model_color(display)
    tag = f"{ath.model_file_prefix(model)}_{args.dataset}"
    out_dir = Path(args.output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    head_roots = [Path(p) for p in args.head_root]
    files = ath.find_head_files(head_roots, model, args.dataset)
    if not files:
        print(f"[WARN] skip {tag}: no head TSV found under {head_roots}")
        return

    gt_path = Path(args.chip_root) / f"{args.dataset}_chip_matched-network.csv"
    expr_path = Path(args.chip_root) / f"{args.dataset}_chip_matched-ExpressionData.csv"
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)
    if not expr_path.exists():
        raise FileNotFoundError(expr_path)

    gt_all = read_chip_gt(gt_path)
    expr = read_chip_expr(expr_path)
    active = active_genes(expr, args.min_frac_nonzero)
    gt = gt_all[gt_all["Gene1"].isin(active) & gt_all["Gene2"].isin(active)].copy()
    tf_set = set(gt["Gene1"].astype(str))
    target_pool = sorted(active)
    gt_edges = chip_true_edges_set(gt)
    k_eval = resolve_top_k(len(gt_edges), args.top_edges)

    print(f"\n[INFO] {tag}: heads={len(files)}, CHIP GT edges={len(gt)}, active genes={len(active)}")
    print(f"       AUPR-style filter: Gene1∈TFs, Gene2∈Genes(GT);  Top-K={k_eval} (=|GT| if --top-edges 0)")
    print(f"       pred_direction={args.pred_direction}, load_max_edges={args.load_max_edges}")

    per_head_df: Dict[int, pd.DataFrame] = {}
    per_head_dir_df: Dict[int, pd.DataFrame] = {}
    per_head_topk_edges: Dict[int, Set[Tuple[str, str]]] = {}

    for fp in files:
        h = ath.parse_head_number(fp)
        raw = load_head_pred(fp, active, args.load_max_edges)
        df = filter_pred_aupr_style(raw, gt)
        per_head_df[h] = df
        per_head_dir_df[h] = filter_pred_direction_style(raw, gt)
        # TF-aligned scores; top-K = |CHIP GT| (same as AUPR TopK_Count)
        lk = build_tf_lookup(df, tf_set, args.pred_direction)
        top_items = sorted(lk.items(), key=lambda x: x[1], reverse=True)
        k_eff = min(k_eval, len(top_items))
        per_head_topk_edges[h] = {e for e, _ in top_items[:k_eff]}
        print(
            f"  head{h}: raw={len(raw)} -> AUPR-filtered={len(df)}, "
            f"tf_lookup={len(lk)}, top-{k_eff} pairs"
        )

    # Fused GRNs (mean4/max4 for 4-head models, mean8/max8 for 8-head, etc.)
    n_heads = len(per_head_df)
    mean_key = f"mean{n_heads}"
    max_key = f"max{n_heads}"
    fused_mean = fuse_predictions(per_head_df, "mean")
    fused_max = fuse_predictions(per_head_df, "max")

    strategies: Dict[str, pd.DataFrame] = {f"head{h}": df for h, df in per_head_df.items()}
    strategies[mean_key] = fused_mean
    strategies[max_key] = fused_max

    summary_rows = []
    per_tf_parts = []
    pr_rows = []

    for name, df in sorted(strategies.items(), key=lambda x: (not x[0].startswith("head"), x[0])):
        lookup = build_tf_lookup(df, tf_set, args.pred_direction)
        glob, per_tf = evaluate_tf_centric(
            gt=gt,
            lookup=lookup,
            target_pool=target_pool,
            neg_ratio=args.neg_ratio,
            seed=args.seed + hash(name) % 10000,
        )
        if glob is None:
            print(f"[WARN] {name}: no TF evaluated")
            continue
        prec, rec = precision_recall_at_k(lookup, gt_edges, k_eval)
        row = {"strategy": name, **glob, "precision_at_k": prec, "recall_at_k": rec}
        summary_rows.append(row)
        per_tf = per_tf.copy()
        per_tf["strategy"] = name
        per_tf_parts.append(per_tf)
        pr_rows.append({"strategy": name, "precision_at_k": prec, "recall_at_k": rec})
        print(f"  {name}: AUPRC_micro={glob['AUPRC_micro']:.4f}  macro={glob['AUPRC_macro']:.4f}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / f"{tag}_auprc_summary.csv", index=False)

    if not per_tf_parts:
        print(f"[WARN] {tag}: no strategies evaluated")
        return

    per_tf_long = pd.concat(per_tf_parts, ignore_index=True)
    per_tf_long.to_csv(out_dir / f"{tag}_auprc_per_tf.csv", index=False)

    # Wide matrix: TF × strategy (for heatmap)
    wide = per_tf_long.pivot(index="TF", columns="strategy", values="AUPRC")
    wide.to_csv(out_dir / f"{tag}_auprc_tf_by_strategy.csv")

    # Oracle routed: per-TF pick head with best AUPRC (upper bound — uses labels)
    if not per_tf_long.empty:
        best_idx = per_tf_long.groupby("TF")["AUPRC"].idxmax()
        routed = per_tf_long.loc[best_idx].copy()
        routed["strategy"] = "oracle_perTF_best_head"
        oracle_micro = float(routed["AUPRC"].mean())  # macro over TFs
        oracle_row = {
            "strategy": "oracle_perTF_best_head",
            "AUPRC_micro": np.nan,
            "AUPRC_macro": oracle_micro,
            "num_TF": int(routed.shape[0]),
            "num_pairs": np.nan,
            "pos_rate": np.nan,
            "precision_at_k": np.nan,
            "recall_at_k": np.nan,
            "note": "UPPER BOUND: picks best head per TF using CHIP — not deployable without labels",
        }
        summary_oracle = pd.concat([summary, pd.DataFrame([oracle_row])], ignore_index=True)
        summary_oracle.to_csv(out_dir / f"{tag}_auprc_summary_with_oracle.csv", index=False)

        # Gain metrics G
        mean_keys = [s for s in summary["strategy"] if str(s).startswith("mean")]
        if mean_keys:
            mk = mean_keys[0]
            mean_au = float(summary.loc[summary["strategy"] == mk, "AUPRC_micro"].iloc[0])
            head_only = summary[summary["strategy"].str.match(r"head\d+")]
            best_au = float(head_only["AUPRC_micro"].max()) if not head_only.empty else float(summary["AUPRC_micro"].max())
            best_name = str(head_only.loc[head_only["AUPRC_micro"].idxmax(), "strategy"]) if not head_only.empty else ""
            gain = pd.DataFrame(
                [
                    {
                        "metric": f"G_micro_best_minus_{mk}",
                        "value": best_au - mean_au,
                        "best_strategy": best_name,
                        f"{mk}_AUPRC_micro": mean_au,
                        "best_AUPRC_micro": best_au,
                    },
                    {
                        "metric": f"G_macro_oracle_minus_{mk}",
                        "value": oracle_micro - mean_au,
                        "best_strategy": "oracle_perTF_best_head",
                        f"{mk}_AUPRC_micro": mean_au,
                        "best_AUPRC_micro": oracle_micro,
                    },
                ]
            )
            gain.to_csv(out_dir / f"{tag}_auprc_gain.csv", index=False)
            print(f"\n[GAIN] best_single({best_name}) - {mk} = {best_au - mean_au:.4f}")
            print(f"[GAIN] oracle_perTF - {mk}   = {oracle_micro - mean_au:.4f}")

    # CHIP true edges found in only one head's top-k
    excl = exclusive_chip_hits(per_head_topk_edges, gt_edges)
    excl.to_csv(out_dir / f"{tag}_chip_exclusive_topk.csv", index=False)
    print(f"[INFO] CHIP true edges exclusive to one head (top-k): {len(excl)}")

    # Optional: merge topology from head_statistics if present
    topo_path = Path(args.stat_dir) / tag / f"{tag}_topology_metrics.csv"
    if topo_path.exists():
        topo = pd.read_csv(topo_path)
        head_only = summary[summary["strategy"].str.match(r"head\d+")].copy()
        head_only["Head"] = head_only["strategy"].str.replace("head", "", regex=False).astype(int)
        merged = head_only.merge(topo, on="Head", how="left")
        merged.to_csv(out_dir / f"{tag}_topology_vs_auprc.csv", index=False)

    # Figures
    head_cols = [c for c in wide.columns if str(c).startswith("head")]
    if head_cols:
        plot_tf_head_heatmap(wide[head_cols], display, args.dataset, out_dir / f"{tag}_auprc_tf_head_heatmap.pdf")
    plot_auprc_bar(summary, display, args.dataset, color, out_dir / f"{tag}_auprc_bar.pdf")

    if args.direction_compare:
        tf_edges = precompute_tf_evaluation_edges(gt, target_pool, args.neg_ratio, args.seed)
        if args.compare_head >= 0:
            cmp_h = args.compare_head
        else:
            cmp_h = pick_compare_head(per_head_dir_df, tf_edges, tf_set)
        cmp_df = per_head_dir_df.get(cmp_h)
        if cmp_df is not None and not cmp_df.empty:
            print(f"\n[INFO] Direction comparison on head{cmp_h} (rows={len(cmp_df)})")
            run_direction_comparison(
                tag=tag,
                display=display,
                pred_df=cmp_df,
                gt=gt,
                target_pool=target_pool,
                tf_set=tf_set,
                active=active,
                out_dir=out_dir,
                head_label=f"head{cmp_h}",
                args=args,
                pr_only=getattr(args, "pr_only", False),
            )
        else:
            print(f"[WARN] {tag}: direction_compare skipped (empty head{cmp_h})")

    print(f"[INFO] Wrote L3 CHIP AUPRC results -> {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="L3: CHIP TF-centric AUPRC for per-head attention GRN exports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc_outputs_help__,
    )
    p.add_argument("--models", default="scgpt", help="Comma-separated model keys, e.g. scgpt,sccello")
    p.add_argument("--dataset", default="hESC", help="Dataset/cell type, e.g. hESC")
    p.add_argument(
        "--datasets",
        default=",".join(DEFAULT_CHIP_DATASETS),
        help="Comma-separated datasets for --direction-pr-batch",
    )
    p.add_argument(
        "--head-root",
        action="append",
        default=[],
        help="Directory with {model}_{dataset}_head*.tsv (repeatable)",
    )
    p.add_argument(
        "--chip-root",
        default="/mnt/10T/yzn/benchmark_GRN/input_process/CHIP",
        help="Folder with {dataset}_chip_matched-network.csv and ExpressionData",
    )
    p.add_argument(
        "--output-dir",
        default=str(_SCRIPT_DIR / "output" / "head_chip_auprc"),
        help="Output root; writes subfolder {model}_{dataset}/",
    )
    p.add_argument(
        "--stat-dir",
        default=str(_SCRIPT_DIR / "output" / "head_statistics"),
        help="Optional: merge topology_metrics.csv from stat_analyze_attention_heads.py",
    )
    p.add_argument(
        "--top-edges",
        type=int,
        default=0,
        help="Top-K for precision/recall & exclusive-edge sets. 0 = |CHIP GT| (same as AUPR1000 TopK).",
    )
    p.add_argument(
        "--load-max-edges",
        type=int,
        default=0,
        help="Max prediction rows per head after active-gene filter (0=load all). Not the same as Top-K.",
    )
    p.add_argument("--neg-ratio", type=float, default=1.0, help="Negatives per positive target per TF")
    p.add_argument("--neg-mode", default="random", choices=["random"], help="Negative sampling mode")
    p.add_argument("--min-frac-nonzero", type=float, default=0.05, help="Active gene filter on expression")
    p.add_argument(
        "--pred-direction",
        default="sym_max",
        choices=["sym_max", "forward", "backward", "key_incoming", "as_exported", "tf_gene1"],
        help="Map attention edges to TF→target scores for CHIP",
    )
    p.add_argument(
        "--direction-compare",
        action="store_true",
        help="Run forward/backward/sym_max panel on one head (shared negatives)",
    )
    p.add_argument(
        "--direction-pr-batch",
        action="store_true",
        help="Six datasets, mean8 directed fusion, PR curves only (fast batch)",
    )
    p.add_argument(
        "--pr-only",
        action="store_true",
        help="With --direction-compare: skip scatter/bar/heatmap/panel",
    )
    p.add_argument(
        "--compare-head",
        type=int,
        default=-1,
        help="Head index for direction panel (-1 = auto-pick best sym_max head)",
    )
    p.add_argument(
        "--example-tf",
        default="POU5F1",
        help="TF for Key-centric heatmap in direction panel",
    )
    p.add_argument(
        "--heatmap-top-genes",
        type=int,
        default=40,
        help="Top incoming Query genes shown in Key-centric heatmap",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    return args


# Printed by argparse --help and documented here for the user
__doc_outputs_help__ = """
OUTPUT FILES (under output/head_chip_auprc/{model}_{dataset}/)
-------------------------------------------------------------
{tag}_auprc_summary.csv
  One row per strategy (head0..head7, mean8, max8).
  - AUPRC_micro: pooled AUPRC over all TF pos/neg pairs (main comparison metric)
  - AUPRC_macro: mean of per-TF AUPRC
  - precision_at_k / recall_at_k: overlap of top-k scored TF→target pairs with CHIP GT

{tag}_auprc_summary_with_oracle.csv
  Same + oracle_perTF_best_head (label upper bound — NOT deployable without GT).

{tag}_auprc_gain.csv
  G = best_single - mean8, and oracle_perTF - mean8 (how much averaging hurts).

{tag}_auprc_per_tf.csv
  Long table: TF, strategy, AUPRC, num_pos/neg — for "which TF uses which head".

{tag}_auprc_tf_by_strategy.csv
  Wide TF × strategy matrix (input to heatmap).

{tag}_chip_exclusive_topk.csv
  CHIP-confirmed edges that appear in top-k of exactly ONE head (interpretability).

{tag}_topology_vs_auprc.csv  (if head_statistics already run)
  Joins topology metrics with AUPRC per head — test role vs performance.

FIGURES
  {tag}_auprc_bar.pdf          — bar chart of AUPRC_micro by strategy
  {tag}_auprc_tf_head_heatmap.pdf — TF × head AUPRC (top variable TFs)

With --direction-compare:
  {tag}_direction_auprc_summary.csv / _per_tf.csv
  {tag}_direction_pr_curves.pdf
  {tag}_direction_bidirectional_scatter.pdf
  {tag}_direction_auprc_bar.pdf
  {tag}_direction_key_heatmap_{TF}.pdf
  {tag}_direction_panel_2x2.pdf
"""


def main() -> None:
    args = parse_args()
    if args.direction_pr_batch:
        run_direction_pr_mean8_batch(args)
        return
    if not args.head_root:
        raise SystemExit("--head-root is required unless using --direction-pr-batch")
    analyze_model_dataset(args)


if __name__ == "__main__":
    main()
