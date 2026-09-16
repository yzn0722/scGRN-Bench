#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expression-only GRN baselines (natural baselines for embedding cosine similarity).

Baselines (gene–gene scores from the scRNA expression matrix):
  - Pearson correlation  (abs optional)
  - Spearman correlation (abs optional)
  - Mutual information   (binned / continuous-discretized)

Each score is evaluated with FBEval-compatible AUPR, AUPR_Ratio, and EPR
against STRING / Non_CHIP / CHIP for every dataset.

Expression is taken from the same GT folder as the network
(STRING/Non_CHIP/CHIP matched matrices), so gene universes stay aligned.

Example:
  python -u FBEval/expression_baselines.py \
    --gt-root /mnt/10T/yzn/benchmark_GRN/input_process \
    --datasets hESC hHep mESC mDC mHSC-E mHSC-GM mHSC-L \
    --gt-types STRING Non_CHIP CHIP \
    --outdir outputs/expression_baselines
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score

GT_SOURCES: Dict[str, Tuple[str, str, str]] = {
    # gt_type: (subdir, network_pattern, expression_pattern)
    "STRING": (
        "STRING",
        "{dataset}_processed-network.csv",
        "{dataset}_processed-ExpressionData.csv",
    ),
    "Non_CHIP": (
        "Non_CHIP",
        "{dataset}_processed-network.csv",
        "{dataset}_processed-ExpressionData.csv",
    ),
    "CHIP": (
        "CHIP",
        "{dataset}_chip_matched-network.csv",
        "{dataset}_chip_matched-ExpressionData.csv",
    ),
    # OmniPath has networks only; use --expr-source STRING for expression.
    "omnipath": (
        "omnipath",
        "{dataset}_processed-network.csv",
        "{dataset}_processed-ExpressionData.csv",
    ),
}


def norm_gene(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def load_gt(path: Path) -> Tuple[Set[str], Set[str], Set[Tuple[str, str]]]:
    df = pd.read_csv(path)
    if not {"Gene1", "Gene2"}.issubset(df.columns):
        col_map = {c.lower(): c for c in df.columns}
        df = df.rename(columns={col_map["gene1"]: "Gene1", col_map["gene2"]: "Gene2"})
    df["Gene1"] = df["Gene1"].map(norm_gene)
    df["Gene2"] = df["Gene2"].map(norm_gene)
    df = df[(df["Gene1"] != "") & (df["Gene2"] != "") & (df["Gene1"] != df["Gene2"])]
    df = df.drop_duplicates(subset=["Gene1", "Gene2"])
    tfs = set(df["Gene1"])
    genes = set(df["Gene1"]) | set(df["Gene2"])
    edges = set(zip(df["Gene1"], df["Gene2"]))
    return tfs, genes, edges


def load_expression(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.map(norm_gene)
    df = df[~df.index.duplicated(keep="first")]
    df = df.loc[df.index != ""]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df


def corr_matrix(expr: pd.DataFrame, method: str) -> Tuple[np.ndarray, Dict[str, int]]:
    genes = list(expr.index)
    gene_to_i = {g: i for i, g in enumerate(genes)}
    x = expr.to_numpy(dtype=np.float64)
    if method == "spearman":
        ranks = np.empty_like(x)
        for i in range(x.shape[0]):
            ranks[i] = stats.rankdata(x[i], method="average")
        mat = np.corrcoef(ranks)
    elif method == "pearson":
        mat = np.corrcoef(x)
    else:
        raise ValueError(method)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return mat, gene_to_i


def digitize_rows(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Discretize each gene (row) into n_bins quantile bins (labels in 0..n_bins-1)."""
    n, _ = x.shape
    d = np.zeros(x.shape, dtype=np.int32)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n):
        # equal-frequency bins; collapse duplicate edges for low-entropy genes
        edges = np.unique(np.quantile(x[i], qs[1:-1]))
        if edges.size == 0:
            continue
        d[i] = np.clip(np.digitize(x[i], edges), 0, n_bins - 1)
    return d


def _mi_from_contingency(cont: np.ndarray) -> float:
    """MI in nats from a 2D contingency table."""
    total = cont.sum()
    if total <= 0:
        return 0.0
    pxy = cont / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    mask = pxy > 0
    return float(np.sum(pxy[mask] * np.log(pxy[mask] / (px * py)[mask])))


def _mi_batch_from_contingency(cont: np.ndarray) -> np.ndarray:
    """
    cont: (n_states_i, n_genes, n_states_j) -> MI vector length n_genes.
    """
    # (n_genes,)
    total = cont.sum(axis=(0, 2))
    total_safe = np.maximum(total, 1.0)
    pxy = cont / total_safe[None, :, None]
    px = pxy.sum(axis=2, keepdims=True)  # (si, g, 1)
    py = pxy.sum(axis=0, keepdims=True)  # (1, g, sj)
    # pxy * log(pxy / (px*py)); 0 where pxy==0
    denom = np.maximum(px * py, 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(pxy > 0, pxy * np.log(pxy / denom), 0.0)
    mi = term.sum(axis=(0, 2))
    mi = np.where(total > 0, mi, 0.0)
    return mi.astype(np.float64)


def pairwise_mi_matrix(expr: pd.DataFrame, n_bins: int = 8) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Fast binned pairwise MI for all genes (symmetric).
    For each gene i, compute contingency vs all genes in one matmul.
    """
    genes = list(expr.index)
    gene_to_i = {g: i for i, g in enumerate(genes)}
    x = expr.to_numpy(dtype=np.float64)
    d = digitize_rows(x, n_bins=n_bins)
    n, n_cells = d.shape
    n_states = int(d.max()) + 1
    eye = np.eye(n_states, dtype=np.float64)
    oh = eye[d]  # (n_genes, n_cells, n_states)
    right = oh.reshape(n_cells, n * n_states)
    mi = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        left = oh[i].T  # (n_states, n_cells)
        cont_flat = left @ right  # (n_states, n*n_states)
        cont = cont_flat.reshape(n_states, n, n_states)
        row = _mi_batch_from_contingency(cont)
        mi[i, :] = row
        if (i + 1) % 200 == 0 or i + 1 == n:
            print(f"    MI matrix progress: {i + 1}/{n} genes", flush=True)
    # numerical symmetry
    mi = 0.5 * (mi + mi.T)
    return mi, gene_to_i


def mi_score_dict(
    expr: pd.DataFrame,
    pairs: Sequence[Tuple[str, str]],
    n_bins: int = 8,
    mi_mat: Optional[np.ndarray] = None,
    gene_to_i: Optional[Dict[str, int]] = None,
) -> Dict[Tuple[str, str], float]:
    """Lookup MI for directed pairs from a precomputed (or freshly built) matrix."""
    if mi_mat is None or gene_to_i is None:
        mi_mat, gene_to_i = pairwise_mi_matrix(expr, n_bins=n_bins)
    out: Dict[Tuple[str, str], float] = {}
    for a, b in pairs:
        ia = gene_to_i.get(a)
        ib = gene_to_i.get(b)
        if ia is None or ib is None:
            continue
        out[(a, b)] = float(mi_mat[ia, ib])
    return out


def corr_score_dict(
    mat: np.ndarray,
    gene_to_i: Dict[str, int],
    pairs: Sequence[Tuple[str, str]],
    use_abs: bool,
) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    for a, b in pairs:
        ia = gene_to_i.get(a)
        ib = gene_to_i.get(b)
        if ia is None or ib is None:
            continue
        v = float(mat[ia, ib])
        out[(a, b)] = abs(v) if use_abs else v
    return out


def evaluate_aupr_epr(
    score_dict: Dict[Tuple[str, str], float],
    tfs: Set[str],
    genes: Set[str],
    true_edges: Set[Tuple[str, str]],
    missing_score: float = -1.0,
) -> Dict[str, float]:
    labels: List[int] = []
    scores: List[float] = []
    for tf in tfs:
        for g in genes:
            if tf == g:
                continue
            labels.append(1 if (tf, g) in true_edges else 0)
            scores.append(score_dict.get((tf, g), missing_score))
    y = np.asarray(labels, dtype=np.int8)
    p = np.asarray(scores, dtype=np.float64)
    n_possible = int(len(tfs) * len(genes) - len(tfs))
    n_true = len(true_edges)
    baseline = (n_true / n_possible) if n_possible > 0 else 0.0
    if y.size == 0 or y.sum() == 0:
        aupr = 0.0
    else:
        aupr = float(average_precision_score(y, p))
    ratio = (aupr / baseline) if baseline > 0 else 0.0

    order = np.argsort(-p)
    k = min(n_true, int(order.size))
    if k == 0 or baseline <= 0:
        epr = 0.0
        topk_precision = 0.0
    else:
        tp = int(y[order[:k]].sum())
        topk_precision = tp / k
        epr = topk_precision / baseline

    return {
        "AUPR": round(aupr, 6),
        "AUPR_Ratio": round(ratio, 6),
        "EPR": round(float(epr), 6),
        "TopK_Precision": round(float(topk_precision), 6),
        "Random_Baseline": round(baseline, 8),
        "True_Edges": n_true,
        "Possible_Edges": n_possible,
        "Scored_Edges": int(len(score_dict)),
    }


def resolve_net_path(gt_root: Path, gt_type: str, dataset: str) -> Path:
    subdir, net_pat, _expr_pat = GT_SOURCES[gt_type]
    return gt_root / subdir / net_pat.format(dataset=dataset)


def resolve_expr_path(gt_root: Path, expr_source: str, dataset: str) -> Path:
    """Expression always from expr_source folder (default STRING)."""
    if expr_source not in GT_SOURCES:
        raise ValueError(f"Unknown expr_source {expr_source!r}")
    subdir, _net_pat, expr_pat = GT_SOURCES[expr_source]
    return gt_root / subdir / expr_pat.format(dataset=dataset)


def run_one_dataset_gt(
    *,
    dataset: str,
    gt_type: str,
    gt_root: Path,
    methods: Sequence[str],
    use_abs_corr: bool,
    mi_bins: int,
    expr_source: str = "STRING",
) -> List[Dict]:
    net_path = resolve_net_path(gt_root, gt_type, dataset)
    expr_path = resolve_expr_path(gt_root, expr_source, dataset)
    if not net_path.is_file():
        return [{
            "Dataset": dataset,
            "GroundTruth": gt_type,
            "Baseline": "NA",
            "Status": f"Missing network: {net_path}",
            "expr_source": expr_source,
        }]
    if not expr_path.is_file():
        return [{
            "Dataset": dataset,
            "GroundTruth": gt_type,
            "Baseline": "NA",
            "Status": f"Missing expression: {expr_path}",
            "expr_source": expr_source,
        }]

    tfs, genes, true_edges = load_gt(net_path)
    expr = load_expression(expr_path)

    # Restrict to genes present in expression
    tfs_u = {g for g in tfs if g in expr.index}
    genes_u = {g for g in genes if g in expr.index}
    true_u = {(a, b) for a, b in true_edges if a in expr.index and b in expr.index}
    if not tfs_u or not genes_u or not true_u:
        return [{
            "Dataset": dataset,
            "GroundTruth": gt_type,
            "Baseline": "NA",
            "Status": "No overlap between GT genes and expression",
            "expr_source": expr_source,
            "expr_path": str(expr_path),
            "net_path": str(net_path),
        }]

    # Keep expression rows needed for scoring
    keep = sorted(tfs_u | genes_u)
    expr_sub = expr.loc[keep]

    pairs = [(tf, g) for tf in tfs_u for g in genes_u if tf != g]
    rows: List[Dict] = []

    need_pearson = any(m.startswith("pearson") for m in methods)
    need_spearman = any(m.startswith("spearman") for m in methods)
    need_mi = "mi" in methods

    pearson_mat = spearman_mat = mi_mat = None
    gene_to_i_p = gene_to_i_s = gene_to_i_m = None
    if need_pearson:
        pearson_mat, gene_to_i_p = corr_matrix(expr_sub, "pearson")
    if need_spearman:
        spearman_mat, gene_to_i_s = corr_matrix(expr_sub, "spearman")
    if need_mi:
        print(f"  computing MI matrix ({expr_sub.shape[0]} genes, bins={mi_bins}) ...", flush=True)
        mi_mat, gene_to_i_m = pairwise_mi_matrix(expr_sub, n_bins=mi_bins)

    for method in methods:
        if method in ("pearson", "pearson_abs"):
            assert pearson_mat is not None and gene_to_i_p is not None
            use_abs = use_abs_corr or method.endswith("_abs")
            scores = corr_score_dict(pearson_mat, gene_to_i_p, pairs, use_abs=use_abs)
            label = "pearson_abs" if use_abs else "pearson"
        elif method in ("spearman", "spearman_abs"):
            assert spearman_mat is not None and gene_to_i_s is not None
            use_abs = use_abs_corr or method.endswith("_abs")
            scores = corr_score_dict(spearman_mat, gene_to_i_s, pairs, use_abs=use_abs)
            label = "spearman_abs" if use_abs else "spearman"
        elif method == "mi":
            assert mi_mat is not None and gene_to_i_m is not None
            scores = mi_score_dict(
                expr_sub, pairs, n_bins=mi_bins, mi_mat=mi_mat, gene_to_i=gene_to_i_m
            )
            label = "mutual_information"
        else:
            raise ValueError(f"Unknown method: {method}")

        metrics = evaluate_aupr_epr(scores, tfs_u, genes_u, true_u)
        rows.append(
            {
                "Dataset": dataset,
                "GroundTruth": gt_type,
                "Baseline": label,
                "Status": "Success",
                "expr_source": expr_source,
                "n_expr_genes": int(expr_sub.shape[0]),
                "n_expr_cells": int(expr_sub.shape[1]),
                "n_TF_in_expr": len(tfs_u),
                "n_genes_in_expr": len(genes_u),
                "n_true_in_expr": len(true_u),
                "expr_path": str(expr_path),
                "net_path": str(net_path),
                **metrics,
            }
        )
        print(
            f"  [{dataset} | {gt_type} | {label}] "
            f"AUPR_Ratio={metrics['AUPR_Ratio']:.4f}  EPR={metrics['EPR']:.4f}  "
            f"AUPR={metrics['AUPR']:.4f}"
        )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Expression Pearson/Spearman/MI GRN baselines")
    p.add_argument(
        "--gt-root",
        type=Path,
        default=Path("/mnt/10T/yzn/benchmark_GRN/input_process"),
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["hESC", "hHep", "mESC", "mDC", "mHSC-E", "mHSC-GM", "mHSC-L"],
    )
    p.add_argument(
        "--gt-types",
        nargs="+",
        default=["STRING", "Non_CHIP", "CHIP"],
        choices=list(GT_SOURCES.keys()),
        help="Networks to evaluate against. Expression is controlled by --expr-source.",
    )
    p.add_argument(
        "--expr-source",
        default="STRING",
        choices=list(GT_SOURCES.keys()),
        help="Folder for ExpressionData (default STRING: same matrix for all GT types).",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=["pearson_abs", "spearman_abs", "mi"],
        choices=["pearson", "pearson_abs", "spearman", "spearman_abs", "mi"],
        help="Defaults use abs(corr); signed corr usually underperforms for recovery ranking.",
    )
    p.add_argument(
        "--use-abs-corr",
        action="store_true",
        help="Force abs() on pearson/spearman even if method name has no _abs",
    )
    p.add_argument("--mi-bins", type=int, default=8)
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/expression_baselines"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []
    for dataset in args.datasets:
        print(f"\n=== {dataset} ===")
        for gt_type in args.gt_types:
            print(f"-- GT={gt_type} | expr={args.expr_source}")
            rows = run_one_dataset_gt(
                dataset=dataset,
                gt_type=gt_type,
                gt_root=args.gt_root,
                methods=args.methods,
                use_abs_corr=args.use_abs_corr,
                mi_bins=args.mi_bins,
                expr_source=args.expr_source,
            )
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    long_path = args.outdir / "expression_baselines_aupr_epr.csv"
    df.to_csv(long_path, index=False)

    ok = df[df["Status"] == "Success"].copy() if "Status" in df.columns else df.copy()
    wide_paths = {}
    if not ok.empty:
        for metric in ("AUPR", "AUPR_Ratio", "EPR"):
            wide = ok.pivot_table(
                index=["Dataset", "GroundTruth"],
                columns="Baseline",
                values=metric,
                aggfunc="first",
            ).reset_index()
            path = args.outdir / f"expression_baselines_{metric}_wide.csv"
            wide.to_csv(path, index=False)
            wide_paths[metric] = str(path)

        # Paper-friendly summary: mean±std over datasets per GT × baseline
        summary = (
            ok.groupby(["GroundTruth", "Baseline"])[["AUPR", "AUPR_Ratio", "EPR"]]
            .agg(["mean", "std"])
            .reset_index()
        )
        # flatten multiindex columns
        summary.columns = [
            "_".join(c).strip("_") if isinstance(c, tuple) else c for c in summary.columns
        ]
        summary_path = args.outdir / "expression_baselines_summary_mean_std.csv"
        summary.to_csv(summary_path, index=False)
        wide_paths["summary"] = str(summary_path)

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "args": {
            "gt_root": str(args.gt_root),
            "datasets": args.datasets,
            "gt_types": args.gt_types,
            "expr_source": args.expr_source,
            "methods": args.methods,
            "use_abs_corr": args.use_abs_corr,
            "mi_bins": args.mi_bins,
        },
        "n_rows": int(len(df)),
        "n_success": int((df["Status"] == "Success").sum()) if "Status" in df.columns else len(df),
        "outputs": {"long": str(long_path), **wide_paths},
    }
    meta_path = args.outdir / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n=== Done ===")
    print(f"long table : {long_path}")
    for k, v in wide_paths.items():
        print(f"{k:10s}: {v}")
    print(f"meta       : {meta_path}")


if __name__ == "__main__":
    main()
