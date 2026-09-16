#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utilities for static GRN TF deep-dive (fig3/tf_static)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Gene / edge I/O (aligned with 3.1plot1_TF_3extract.py)
# ---------------------------------------------------------------------------

def norm_gene(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def detect_weight_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "EdgeWeight",
        "edgeweight",
        "edge_weight",
        "Weight",
        "Score",
        "Importance",
        "Attention score",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in ["weight", "score", "import", "att"]):
            return c
    return None


def load_gt_network(gt_path: Path) -> Tuple[Set[str], Set[str], int, Set[Tuple[str, str]], Dict[str, Set[str]]]:
    gt = pd.read_csv(gt_path)
    gt["Gene1"] = gt["Gene1"].map(norm_gene)
    gt["Gene2"] = gt["Gene2"].map(norm_gene)
    gt = gt[(gt["Gene1"] != "") & (gt["Gene2"] != "") & (gt["Gene1"] != gt["Gene2"])]
    gt_gene1 = set(gt["Gene1"].unique())
    gt_all = set(gt["Gene1"].unique()) | set(gt["Gene2"].unique())
    gt_n = int(len(gt))
    gt_edges = set(zip(gt["Gene1"], gt["Gene2"]))
    gt_by_tf: Dict[str, Set[str]] = {}
    for tf, tg in zip(gt["Gene1"], gt["Gene2"]):
        gt_by_tf.setdefault(tf, set()).add(tg)
    return gt_gene1, gt_all, gt_n, gt_edges, gt_by_tf


def filter_prediction(
    pred_path: Path,
    gt_gene1: Set[str],
    gt_all: Set[str],
    gt_n: int,
    *,
    edge_cap: Optional[int] = None,
) -> pd.DataFrame:
    """Keep top-ranked edges; default cap is |E_GT| (= gt_n). Use edge_cap for Top-k sweeps."""
    pred = pd.read_csv(pred_path, sep="\t")
    pred["Gene1"] = pred["Gene1"].map(norm_gene)
    pred["Gene2"] = pred["Gene2"].map(norm_gene)
    pred = pred[(pred["Gene1"] != "") & (pred["Gene2"] != "") & (pred["Gene1"] != pred["Gene2"])]
    wcol = detect_weight_col(pred)
    if wcol is not None:
        pred["_w"] = pd.to_numeric(pred[wcol], errors="coerce").fillna(0.0).abs()
        pred = pred.sort_values("_w", ascending=False).drop(columns=["_w"])
    pred = pred[pred["Gene1"].isin(gt_gene1) & pred["Gene2"].isin(gt_all)].copy()
    cap = int(edge_cap) if edge_cap is not None else int(gt_n)
    pred = pred.head(max(1, cap)).reset_index(drop=True)
    return pred


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def per_tf_edge_sets(
    pred_df: pd.DataFrame,
    gt_by_tf: Dict[str, Set[str]],
) -> pd.DataFrame:
    """Per-TF metrics: out-degrees, Jaccard, precision, recall."""
    rows = []
    for tf, grp in pred_df.groupby("Gene1"):
        pred_t = set(grp["Gene2"].tolist())
        gt_t = gt_by_tf.get(tf, set())
        if not pred_t and not gt_t:
            continue
        hits = pred_t & gt_t
        rows.append(
            {
                "TF": tf,
                "gt_outdegree": len(gt_t),
                "pred_outdegree": len(pred_t),
                "hits": len(hits),
                "jaccard": jaccard(pred_t, gt_t),
                "precision": len(hits) / len(pred_t) if pred_t else np.nan,
                "recall": len(hits) / len(gt_t) if gt_t else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["f1"] = 2 * df["precision"] * df["recall"] / (df["precision"] + df["recall"] + 1e-12)
    return df.sort_values(["gt_outdegree", "jaccard"], ascending=[False, False]).reset_index(drop=True)


def pred_targets_for_tf(pred_df: pd.DataFrame, tf: str) -> Set[str]:
    """Predicted target genes (Gene2) for one TF (Gene1)."""
    sub = pred_df[pred_df["Gene1"] == tf]
    return set(sub["Gene2"].tolist())


def partition_venn3_regions(
    set_a: Set[str],
    set_b: Set[str],
    set_c: Set[str],
) -> Dict[str, Set[str]]:
    """Seven disjoint regions for a 3-set Venn (A=emb, B=att, C=hid by default order)."""
    a, b, c = set(set_a), set(set_b), set(set_c)
    return {
        "emb_only": a - b - c,
        "att_only": b - a - c,
        "hid_only": c - a - b,
        "emb_att_not_hid": (a & b) - c,
        "emb_hid_not_att": (a & c) - b,
        "att_hid_not_emb": (b & c) - a,
        "triple": a & b & c,
    }


def classify_tf_edges(
    tf: str,
    pred_df: pd.DataFrame,
    gt_by_tf: Dict[str, Set[str]],
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Return (TP targets, FP targets, FN targets) for one TF."""
    sub = pred_df[pred_df["Gene1"] == tf]
    pred_t = set(sub["Gene2"].tolist())
    gt_t = gt_by_tf.get(tf, set())
    tp = pred_t & gt_t
    fp = pred_t - gt_t
    fn = gt_t - pred_t
    return tp, fp, fn


# ---------------------------------------------------------------------------
# ORA (offline GMT, same spirit as run_ora_and_plot.py)
# ---------------------------------------------------------------------------

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
    if n <= 0 or N <= 0 or k <= 0:
        return 1.0
    if k > min(K, n):
        return 0.0
    denom = math.comb(N, n)
    if denom == 0:
        return 1.0
    s = 0.0
    hi = min(K, n)
    for i in range(k, hi + 1):
        j = n - i
        if j < 0 or j > N - K:
            continue
        s += math.comb(K, i) * math.comb(N - K, j) / denom
    return float(min(max(s, 0.0), 1.0))


def bh_fdr(pvals: list[float]) -> list[float]:
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


def load_gene_set_table(path: Path) -> Dict[str, tuple[str, Set[str]]]:
    """Load curated gene sets: set_id -> (display_name, genes)."""
    if not path.is_file():
        return {}
    try:
        df = pd.read_csv(path, comment="#")
    except pd.errors.EmptyDataError:
        return {}
    if "set_id" not in df.columns or "genes" not in df.columns:
        return {}
    out: Dict[str, tuple[str, Set[str]]] = {}
    for _, row in df.iterrows():
        sid = str(row["set_id"]).strip()
        if not sid or sid == "nan":
            continue
        name = str(row.get("display_name", sid)).strip()
        raw = str(row["genes"]).strip()
        genes = {norm_gene(g) for g in raw.split(";") if norm_gene(g)}
        if genes:
            out[sid] = (name, genes)
    return out


def run_gene_set_enrichment(
    query_genes: Set[str],
    gene_sets: Dict[str, tuple[str, Set[str]]],
    background: Set[str],
) -> pd.DataFrame:
    """Hypergeometric enrichment of query genes against named gene sets."""
    pathways = {sid: genes for sid, (_, genes) in gene_sets.items()}
    df = run_ora(query_genes, pathways, background)
    if df.empty:
        return df
    id_map = {sid: name for sid, (name, _) in gene_sets.items()}
    df["set_id"] = df["Term"]
    df["display_name"] = df["set_id"].map(id_map).fillna(df["set_id"])
    return df


def run_ora(
    query_genes: Set[str],
    pathways: Dict[str, Set[str]],
    background: Optional[Set[str]] = None,
) -> pd.DataFrame:
    bg = background if background is not None else set().union(*pathways.values())
    q = {g.upper() for g in query_genes} & bg
    N = len(bg)
    n = len(q)
    if n < 3:
        return pd.DataFrame()
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
                "Term": term,
                "PValue": p,
                "Overlap": k,
                "SetSize": K,
                "QuerySize": n,
                "Genes": ";".join(sorted(overlap)),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["FDR_BH"] = bh_fdr(df["PValue"].tolist())
    df["neglog10P"] = -np.log10(np.clip(df["PValue"].to_numpy(dtype=float), 1e-300, None))
    return df.sort_values(["FDR_BH", "PValue"], ascending=[True, True]).reset_index(drop=True)


def shorten_term(term: str, max_len: int = 48) -> str:
    t = str(term)
    for prefix in ("GOBP_", "GO_BP_", "KEGG_", "REACTOME_", "HALLMARK_"):
        if t.startswith(prefix):
            t = t[len(prefix) :]
    t = t.replace("_", " ")
    if len(t) > max_len:
        return t[: max_len - 1] + "…"
    return t


def spearman_label(x: np.ndarray, y: np.ndarray) -> str:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return "ρ = n/a"
    rho, p = stats.spearmanr(x[mask], y[mask])
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    return f"Spearman ρ = {rho:.2f}{star}  (n={int(mask.sum())})"
