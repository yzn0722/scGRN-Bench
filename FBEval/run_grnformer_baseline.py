#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRNFormer (Hegde & Cheng) baseline under FBEval AUPR / AUPR_Ratio / EPR.

Purpose
-------
Direct competitor for the "transferable embedding / cross cell-type GRN" claim.
Uses the same evaluation protocol as expression_baselines.py and RegFormer
token-embedding GRN (full TF×gene candidate matrix; missing scores = -1).

This is the Bioinformatics GRN-inference GRNFormer, NOT the ACL "integrate
GRN into RNA FM" paper.

Protocols
---------
A) Transferable inference (default, fair vs embedding cosine):
   Official pretrained checkpoint → infer on each BEELINE dataset.
   Do NOT train on the target dataset's ground-truth edges.

B) Supervised / paper LOO (optional upper bound):
   Train with their pipeline on other cell types, then eval held-out.
   Report separately; do not mix with Protocol A tables.

Checklist (alignment)
---------------------
1. Same datasets: hESC hHep mESC mDC mHSC-E mHSC-GM mHSC-L
2. Same GT: STRING / Non_CHIP / CHIP under --gt-root
3. Same expression matrix source (default STRING folder) for all GT types
4. Same metrics: AUPR, AUPR_Ratio, EPR over TF×gene universe
5. Cite distinction: supervised edge model vs unsupervised emb similarity
6. Prefer Protocol A for the transferable claim comparison

Example (evaluate existing predictions):
  python -u FBEval/run_grnformer_baseline.py \
    --gt-root /mnt/10T/yzn/benchmark_GRN/input_process \
    --pred-dir outputs/grnformer/preds \
    --outdir outputs/grnformer_fbeval

Example (also call GRNFormer infer_grn.py):
  python -u FBEval/run_grnformer_baseline.py \
    --gt-root /mnt/10T/yzn/benchmark_GRN/input_process \
    --grnformer-root /path/to/GRNformer \
    --ckpt /path/to/GRNFormer.ckpt \
    --run-infer \
    --pred-dir outputs/grnformer/preds \
    --outdir outputs/grnformer_fbeval
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# Reuse path conventions from expression_baselines
GT_SOURCES: Dict[str, Tuple[str, str, str]] = {
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
    "omnipath": (
        "omnipath",
        "{dataset}_processed-network.csv",
        "{dataset}_processed-ExpressionData.csv",
    ),
}

HUMAN_DATASETS = {"hESC", "hHep"}
MOUSE_DATASETS = {"mESC", "mDC", "mHSC-E", "mHSC-GM", "mHSC-L"}


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


def load_expression_genes(path: Path) -> Set[str]:
    df = pd.read_csv(path, index_col=0, nrows=0)
    # re-read index only
    df = pd.read_csv(path, usecols=[0])
    genes = {norm_gene(g) for g in df.iloc[:, 0].tolist()}
    genes.discard("")
    return genes


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
    subdir, net_pat, _ = GT_SOURCES[gt_type]
    return gt_root / subdir / net_pat.format(dataset=dataset)


def resolve_expr_path(expr_root: Path, expr_source: str, dataset: str) -> Path:
    subdir, _, expr_pat = GT_SOURCES[expr_source]
    return expr_root / subdir / expr_pat.format(dataset=dataset)


def species_for(dataset: str) -> str:
    if dataset in HUMAN_DATASETS:
        return "human"
    if dataset in MOUSE_DATASETS:
        return "mouse"
    raise ValueError(f"Unknown dataset species mapping: {dataset}")


def default_tf_path(tf_root: Path, dataset: str) -> Path:
    sp = species_for(dataset)
    return tf_root / f"{sp}-tfs.csv"


def ensure_single_column_tf(tf_file: Path, cache_dir: Path) -> Path:
    """
    GRNFormer infer_grn / on_predict_epoch_end expects a headerless
    single-column TF CSV. BEELINE human-tfs.csv / mouse-tfs.csv are
    two-column (TF,Sources) with a header.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = (cache_dir / f"{tf_file.stem}.onecol.csv").resolve()
    peek = pd.read_csv(tf_file, nrows=5)
    cols_lower = {str(c).lower(): c for c in peek.columns}
    if "tf" in cols_lower or "tfs" in cols_lower:
        df = pd.read_csv(tf_file)
        cols_lower = {str(c).lower(): c for c in df.columns}
        col = cols_lower.get("tf") or cols_lower.get("tfs")
        series = df[col]
    elif peek.shape[1] == 1:
        series = pd.read_csv(tf_file, header=None).iloc[:, 0]
    else:
        raw = pd.read_csv(tf_file, header=None)
        series = raw.iloc[:, 0]
        if str(series.iloc[0]).strip().lower() in {"tf", "tfs"}:
            series = series.iloc[1:]
    tfs = [
        str(x).strip()
        for x in series
        if pd.notna(x) and str(x).strip() and str(x).strip().lower() not in {"tf", "tfs"}
    ]
    if not tfs:
        raise ValueError(f"No TF names parsed from {tf_file}")
    pd.Series(tfs).to_csv(out, index=False, header=False)
    return out


def load_pred_scores(path: Path) -> Dict[Tuple[str, str], float]:
    """
    Accept common GRNFormer / BEELINE edge CSV schemas:
      Gene1,Gene2,EdgeWeight | source,target,weight | TF,target,score | ...
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    g1 = cols.get("gene1") or cols.get("source") or cols.get("tf") or cols.get("regulator")
    g2 = cols.get("gene2") or cols.get("target") or cols.get("gene")
    w = (
        cols.get("edgeweight")
        or cols.get("weight")
        or cols.get("score")
        or cols.get("prob")
        or cols.get("probability")
        or cols.get("edge_weight")
    )
    if g1 is None or g2 is None:
        raise ValueError(f"Cannot find Gene1/Gene2 columns in {path}: {list(df.columns)}")
    if w is None:
        # last numeric column fallback
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise ValueError(f"No score column in {path}: {list(df.columns)}")
        w = num_cols[-1]

    out: Dict[Tuple[str, str], float] = {}
    for a, b, s in zip(df[g1], df[g2], df[w]):
        a_n, b_n = norm_gene(a), norm_gene(b)
        if not a_n or not b_n or a_n == b_n:
            continue
        s_f = float(s)
        # keep max if duplicate
        prev = out.get((a_n, b_n))
        if prev is None or s_f > prev:
            out[(a_n, b_n)] = s_f
    return out


def run_infer_one(
    *,
    grnformer_root: Path,
    ckpt: Path,
    exp_file: Path,
    tf_file: Path,
    output_file: Path,
    coexpression_threshold: float,
    max_subgraph_size: int,
    python_bin: str,
) -> None:
    grnformer_root = Path(grnformer_root).resolve()
    ckpt = Path(ckpt).resolve()
    exp_file = Path(exp_file).resolve()
    tf_file = Path(tf_file).resolve()
    output_file = Path(output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_bin,
        str(grnformer_root / "infer_grn.py"),
        "--exp_file",
        str(exp_file),
        "--tf_file",
        str(tf_file),
        "--output_file",
        str(output_file),
        "--ckpt_path",
        str(ckpt),
        "--coexpression_threshold",
        str(coexpression_threshold),
        "--max_subgraph_size",
        str(max_subgraph_size),
    ]
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(grnformer_root), check=True)


def pred_path_for(pred_dir: Path, dataset: str) -> Path:
    # Prefer dataset-level preds (expression is shared across GT types)
    candidates = [
        pred_dir / f"{dataset}_predicted-edges.csv",
        pred_dir / dataset / "predicted-edges.csv",
        pred_dir / f"{dataset}.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return candidates[0]


def evaluate_one(
    *,
    dataset: str,
    gt_type: str,
    gt_root: Path,
    expr_root: Path,
    expr_source: str,
    pred_path: Path,
) -> Dict:
    net_path = resolve_net_path(gt_root, gt_type, dataset)
    expr_path = resolve_expr_path(expr_root, expr_source, dataset)
    if not net_path.is_file():
        return {
            "Dataset": dataset,
            "GroundTruth": gt_type,
            "Baseline": "GRNFormer",
            "Status": f"Missing network: {net_path}",
        }
    if not expr_path.is_file():
        return {
            "Dataset": dataset,
            "GroundTruth": gt_type,
            "Baseline": "GRNFormer",
            "Status": f"Missing expression: {expr_path}",
        }
    if not pred_path.is_file():
        return {
            "Dataset": dataset,
            "GroundTruth": gt_type,
            "Baseline": "GRNFormer",
            "Status": f"Missing predictions: {pred_path}",
            "pred_path": str(pred_path),
        }

    tfs, genes, true_edges = load_gt(net_path)
    expr_genes = load_expression_genes(expr_path)
    tfs_u = {g for g in tfs if g in expr_genes}
    genes_u = {g for g in genes if g in expr_genes}
    true_u = {(a, b) for a, b in true_edges if a in expr_genes and b in expr_genes}
    if not tfs_u or not genes_u or not true_u:
        return {
            "Dataset": dataset,
            "GroundTruth": gt_type,
            "Baseline": "GRNFormer",
            "Status": "No overlap between GT genes and expression",
        }

    scores = load_pred_scores(pred_path)
    metrics = evaluate_aupr_epr(scores, tfs_u, genes_u, true_u)
    row = {
        "Dataset": dataset,
        "GroundTruth": gt_type,
        "Baseline": "GRNFormer",
        "Status": "Success",
        "expr_source": expr_source,
        "n_TF_in_expr": len(tfs_u),
        "n_genes_in_expr": len(genes_u),
        "n_true_in_expr": len(true_u),
        "expr_path": str(expr_path),
        "net_path": str(net_path),
        "pred_path": str(pred_path),
        **metrics,
    }
    print(
        f"  [{dataset} | {gt_type} | GRNFormer] "
        f"AUPR_Ratio={metrics['AUPR_Ratio']:.4f}  EPR={metrics['EPR']:.4f}  "
        f"AUPR={metrics['AUPR']:.4f}",
        flush=True,
    )
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRNFormer FBEval baseline (AUPR/AUPR_Ratio/EPR)")
    p.add_argument(
        "--gt-root",
        type=Path,
        default=Path("/mnt/10T/yzn/benchmark_GRN/input_process"),
        help="Root for ground-truth network folders (STRING/Non_CHIP/CHIP/omnipath).",
    )
    p.add_argument(
        "--expr-root",
        type=Path,
        default=None,
        help="Root for expression matrices (default: same as --gt-root). "
        "Use when OmniPath nets live under input_process1000 but preds used input_process STRING.",
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
    )
    p.add_argument(
        "--expr-source",
        default="STRING",
        choices=[k for k in GT_SOURCES.keys() if k != "omnipath"],
        help="Expression folder (default STRING: shared matrix across GT types).",
    )
    p.add_argument(
        "--tf-root",
        type=Path,
        default=Path("/mnt/10T/yzn/Beeline-master"),
        help="Directory containing human-tfs.csv and mouse-tfs.csv",
    )
    p.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("outputs/grnformer/preds"),
        help="Directory for GRNFormer predicted-edge CSVs",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/grnformer_fbeval"),
    )
    p.add_argument(
        "--run-infer",
        action="store_true",
        help="Call GRNFormer infer_grn.py before evaluation (Protocol A).",
    )
    p.add_argument(
        "--grnformer-root",
        type=Path,
        default=None,
        help="Clone of github.com/BioinfoMachineLearning/GRNformer",
    )
    p.add_argument("--ckpt", type=Path, default=None, help="Pretrained GRNFormer checkpoint")
    p.add_argument("--coexpression-threshold", type=float, default=0.1)
    p.add_argument("--max-subgraph-size", type=int, default=100)
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument(
        "--protocol",
        default="A",
        choices=["A", "B"],
        help="A=pretrained transfer (default); B=mark as supervised LOO (metadata only).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.pred_dir.mkdir(parents=True, exist_ok=True)
    args.outdir.mkdir(parents=True, exist_ok=True)
    expr_root = args.expr_root if args.expr_root is not None else args.gt_root

    if args.run_infer:
        if args.grnformer_root is None or args.ckpt is None:
            raise SystemExit("--run-infer requires --grnformer-root and --ckpt")
        if not (args.grnformer_root / "infer_grn.py").is_file():
            raise SystemExit(f"infer_grn.py not found under {args.grnformer_root}")
        if not args.ckpt.is_file():
            raise SystemExit(f"Checkpoint not found: {args.ckpt}")

        tf_cache = (args.pred_dir / "_tf_onecol").resolve()
        for dataset in args.datasets:
            exp_file = resolve_expr_path(expr_root, args.expr_source, dataset)
            tf_raw = default_tf_path(args.tf_root, dataset)
            out_pred = (args.pred_dir / f"{dataset}_predicted-edges.csv").resolve()
            if not exp_file.is_file():
                print(f"[skip infer] missing expression: {exp_file}", flush=True)
                continue
            if not tf_raw.is_file():
                print(f"[skip infer] missing TF list: {tf_raw}", flush=True)
                continue
            if out_pred.is_file():
                print(f"[skip infer] exists: {out_pred}", flush=True)
                continue
            tf_file = ensure_single_column_tf(tf_raw, tf_cache)
            print(f"[tf] {tf_raw} -> {tf_file}", flush=True)
            run_infer_one(
                grnformer_root=args.grnformer_root,
                ckpt=args.ckpt,
                exp_file=exp_file,
                tf_file=tf_file,
                output_file=out_pred,
                coexpression_threshold=args.coexpression_threshold,
                max_subgraph_size=args.max_subgraph_size,
                python_bin=args.python_bin,
            )

    all_rows: List[Dict] = []
    for dataset in args.datasets:
        print(f"\n=== {dataset} ===", flush=True)
        pred_path = pred_path_for(args.pred_dir, dataset)
        for gt_type in args.gt_types:
            print(f"-- GT={gt_type} | expr={args.expr_source}", flush=True)
            row = evaluate_one(
                dataset=dataset,
                gt_type=gt_type,
                gt_root=args.gt_root,
                expr_root=expr_root,
                expr_source=args.expr_source,
                pred_path=pred_path,
            )
            row["Protocol"] = args.protocol
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    long_path = args.outdir / "grnformer_aupr_epr.csv"
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
            path = args.outdir / f"grnformer_{metric}_wide.csv"
            wide.to_csv(path, index=False)
            wide_paths[metric] = str(path)

        summary = (
            ok.groupby(["GroundTruth", "Baseline"])[["AUPR", "AUPR_Ratio", "EPR"]]
            .agg(["mean", "std"])
            .reset_index()
        )
        summary.columns = [
            "_".join(c).strip("_") if isinstance(c, tuple) else c for c in summary.columns
        ]
        summary_path = args.outdir / "grnformer_summary_mean_std.csv"
        summary.to_csv(summary_path, index=False)
        wide_paths["summary"] = str(summary_path)

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "paper": "Hegde & Cheng, GRNFormer: accurate gene regulatory network inference using graph transformer",
        "protocol": args.protocol,
        "protocol_note": (
            "A: pretrained ckpt transfer, no target GT supervision "
            "(fair vs transferable embedding). "
            "B: supervised LOO / their paper setting (upper bound)."
        ),
        "args": {
            "gt_root": str(args.gt_root),
            "datasets": args.datasets,
            "gt_types": args.gt_types,
            "expr_source": args.expr_source,
            "pred_dir": str(args.pred_dir),
            "run_infer": args.run_infer,
            "grnformer_root": str(args.grnformer_root) if args.grnformer_root else None,
            "ckpt": str(args.ckpt) if args.ckpt else None,
        },
        "n_rows": int(len(df)),
        "outputs": {"long": str(long_path), **wide_paths},
    }
    meta_path = args.outdir / "grnformer_run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {long_path}", flush=True)
    print(f"Meta {meta_path}", flush=True)


if __name__ == "__main__":
    main()
