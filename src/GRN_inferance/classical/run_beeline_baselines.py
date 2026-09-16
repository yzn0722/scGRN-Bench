#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run BEELINE classical baselines PIDC (Python) and PPCOR (R), then evaluate
AUPR / AUPR_Ratio / EPR against STRING / Non_CHIP / CHIP.

Expression is taken from STRING processed matrices (same as expression baselines).

Example:
  python -u src/GRN_inferance/classical/run_beeline_baselines.py \
    --methods PIDC PPCOR \
    --datasets hESC hHep mDC mHSC-E mHSC-GM mHSC-L \
    --gt-types STRING Non_CHIP CHIP \
    --outdir outputs/beeline_baselines
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# Allow running as script from repo root
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.GRN_inferance.classical.pidc import infer_pidc, load_expression_csv  # noqa: E402

GT_SOURCES = {
    "STRING": ("STRING", "{dataset}_processed-network.csv"),
    "Non_CHIP": ("Non_CHIP", "{dataset}_processed-network.csv"),
    "CHIP": ("CHIP", "{dataset}_chip_matched-network.csv"),
    "omnipath": ("omnipath", "{dataset}_processed-network.csv"),
}


def norm_gene(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def load_gt(path: Path) -> Tuple[Set[str], Set[str], Set[Tuple[str, str]]]:
    df = pd.read_csv(path)
    df["Gene1"] = df["Gene1"].map(norm_gene)
    df["Gene2"] = df["Gene2"].map(norm_gene)
    df = df[(df["Gene1"] != "") & (df["Gene2"] != "") & (df["Gene1"] != df["Gene2"])]
    df = df.drop_duplicates(subset=["Gene1", "Gene2"])
    tfs = set(df["Gene1"])
    genes = set(df["Gene1"]) | set(df["Gene2"])
    edges = set(zip(df["Gene1"], df["Gene2"]))
    return tfs, genes, edges


def evaluate_aupr_epr(
    pred: pd.DataFrame,
    tfs: Set[str],
    genes: Set[str],
    true_edges: Set[Tuple[str, str]],
) -> Dict[str, float]:
    pred = pred.copy()
    pred["Gene1"] = pred["Gene1"].map(norm_gene)
    pred["Gene2"] = pred["Gene2"].map(norm_gene)
    pred["EdgeWeight"] = pd.to_numeric(pred["EdgeWeight"], errors="coerce").abs()
    pred = pred.dropna(subset=["EdgeWeight"])
    pred = pred[pred["Gene1"] != pred["Gene2"]]
    pred = pred.drop_duplicates(subset=["Gene1", "Gene2"], keep="first")
    score = {(a, b): float(w) for a, b, w in pred[["Gene1", "Gene2", "EdgeWeight"]].itertuples(index=False)}

    labels, scores = [], []
    for tf in tfs:
        for g in genes:
            if tf == g:
                continue
            labels.append(1 if (tf, g) in true_edges else 0)
            scores.append(score.get((tf, g), -1.0))
    y = np.asarray(labels, dtype=np.int8)
    p = np.asarray(scores, dtype=np.float64)
    n_possible = int(len(tfs) * len(genes) - len(tfs))
    n_true = len(true_edges)
    baseline = n_true / n_possible if n_possible else 0.0
    aupr = float(average_precision_score(y, p)) if y.size and y.sum() else 0.0
    ratio = aupr / baseline if baseline > 0 else 0.0
    order = np.argsort(-p)
    k = min(n_true, int(order.size))
    if k == 0 or baseline <= 0:
        epr = 0.0
    else:
        epr = (y[order[:k]].sum() / k) / baseline
    return {
        "AUPR": round(aupr, 6),
        "AUPR_Ratio": round(ratio, 6),
        "EPR": round(float(epr), 6),
        "Random_Baseline": round(baseline, 8),
        "True_Edges": n_true,
        "Possible_Edges": n_possible,
        "Pred_Edges": int(len(pred)),
    }


def ppcor_shrinkage_spearman(expr: pd.DataFrame) -> pd.DataFrame:
    """
    Shrinkage partial correlation (Spearman ranks → Ledoit-Wolf precision).
    Used when classical ppcor fails (n_genes >= n_cells).
    Returns Gene1, Gene2, corVal, pValue (pValue=nan; ranking uses |corVal|).
    """
    from sklearn.covariance import LedoitWolf

    genes = [str(g) for g in expr.index]
    # Spearman: rank each gene across cells, then LW on cells × genes
    ranks = expr.to_numpy(dtype=np.float64).copy()
    for i in range(ranks.shape[0]):
        ranks[i] = pd.Series(ranks[i]).rank(method="average").to_numpy()
    X = ranks.T  # cells × genes
    # drop near-constant genes
    std = X.std(axis=0)
    keep = std > 1e-12
    if keep.sum() < 3:
        raise RuntimeError("Too few variable genes for PPCOR shrinkage")
    genes_k = [g for g, k in zip(genes, keep) if k]
    X = X[:, keep]
    lw = LedoitWolf().fit(X)
    prec = lw.precision_
    d = np.sqrt(np.clip(np.diag(prec), 1e-12, None))
    pcor = -prec / np.outer(d, d)
    np.fill_diagonal(pcor, 1.0)
    rows = []
    n = len(genes_k)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rows.append((genes_k[i], genes_k[j], float(pcor[i, j]), np.nan))
    return pd.DataFrame(rows, columns=["Gene1", "Gene2", "corVal", "pValue"])


def parse_ppcor_outfile(raw_path: Path, p_val: float = 0.01) -> pd.DataFrame:
    """BEELINE ppcorRunner.parseOutput logic (p-value gate when available)."""
    df = pd.read_csv(raw_path, sep="\t")
    df = df[df["Gene1"].astype(str) != df["Gene2"].astype(str)].copy()
    if "pValue" in df.columns and df["pValue"].notna().any():
        part1 = df.loc[df["pValue"] <= p_val].copy()
        part1["absCorVal"] = part1["corVal"].abs()
        part1 = part1.sort_values("absCorVal", ascending=False)
        part2 = df.loc[(df["pValue"] > p_val) | df["pValue"].isna()].copy()
        rows = [(r.Gene1, r.Gene2, float(r.corVal)) for r in part1.itertuples()]
        rows += [(r.Gene1, r.Gene2, 0.0) for r in part2.itertuples()]
        out = pd.DataFrame(rows, columns=["Gene1", "Gene2", "EdgeWeight"])
    else:
        # shrinkage fallback: rank by |partial corr|
        out = df.rename(columns={"corVal": "EdgeWeight"})[["Gene1", "Gene2", "EdgeWeight"]]
    out["EdgeWeight"] = pd.to_numeric(out["EdgeWeight"], errors="coerce").abs().fillna(0.0)
    out = out.sort_values("EdgeWeight", ascending=False).reset_index(drop=True)
    return out


def run_ppcor(
    expr_csv: Path,
    out_tsv: Path,
    *,
    rscript: Path,
    p_val: float,
    work_dir: Path,
    expr_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    work_dir.mkdir(parents=True, exist_ok=True)
    raw = work_dir / f"{out_tsv.stem}_ppcor_raw.txt"
    use_shrinkage = False
    if expr_df is not None and expr_df.shape[0] >= expr_df.shape[1]:
        print(
            f"[PPCOR] genes({expr_df.shape[0]}) >= cells({expr_df.shape[1]}); "
            "using Spearman+LedoitWolf shrinkage partial correlation",
            flush=True,
        )
        use_shrinkage = True
    if not use_shrinkage:
        cmd = ["Rscript", str(rscript), str(expr_csv), str(raw)]
        print("[PPCOR]", " ".join(cmd), flush=True)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print("[PPCOR] R ppcor failed (singular); falling back to shrinkage", flush=True)
            use_shrinkage = True
    if use_shrinkage:
        if expr_df is None:
            expr_df = pd.read_csv(expr_csv, index_col=0)
        raw_df = ppcor_shrinkage_spearman(expr_df)
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(raw, sep="\t", index=False)
    pred = parse_ppcor_outfile(raw, p_val=p_val)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_tsv, sep="\t", index=False)
    return pred


def run_pidc(
    expr: pd.DataFrame,
    out_tsv: Path,
    *,
    n_bins: int,
    discretizer: str,
    gene_subset: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    pred = infer_pidc(
        expr,
        n_bins=n_bins,
        discretizer=discretizer,
        apply_context=True,
        gene_subset=gene_subset,
    )
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_tsv, sep="\t", index=False)
    return pred


def collect_union_genes(gt_root: Path, dataset: str, gt_types: Sequence[str]) -> Set[str]:
    genes: Set[str] = set()
    for gt in gt_types:
        sub, pat = GT_SOURCES[gt]
        path = gt_root / sub / pat.format(dataset=dataset)
        if not path.is_file():
            continue
        try:
            tfs, nodes, edges = load_gt(path)
        except Exception:
            continue
        if edges:
            genes |= nodes
    return genes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PIDC + PPCOR baselines and evaluate")
    p.add_argument("--gt-root", type=Path, default=Path("/mnt/10T/yzn/benchmark_GRN/input_process"))
    p.add_argument("--expr-source", default="STRING", choices=["STRING", "Non_CHIP", "CHIP"])
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["hESC", "hHep", "mDC", "mHSC-E", "mHSC-GM", "mHSC-L"],
    )
    p.add_argument("--gt-types", nargs="+", default=["STRING", "Non_CHIP", "CHIP"])
    p.add_argument("--methods", nargs="+", default=["PIDC", "PPCOR"], choices=["PIDC", "PPCOR"])
    p.add_argument("--ppcor-pval", type=float, default=0.01)
    p.add_argument("--pidc-bins", type=int, default=10)
    p.add_argument(
        "--pidc-discretizer",
        default="uniform_width",
        choices=["uniform_width", "uniform_count"],
    )
    p.add_argument(
        "--pidc-gene-mode",
        default="union_gt",
        choices=["all", "union_gt"],
        help="all=all genes in expression; union_gt=genes appearing in any requested GT",
    )
    p.add_argument(
        "--rscript",
        type=Path,
        default=Path(__file__).resolve().parent / "run_ppcor.R",
    )
    p.add_argument("--outdir", type=Path, default=Path("outputs/beeline_baselines"))
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    pred_dir = args.outdir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.outdir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for dataset in args.datasets:
        if args.expr_source == "CHIP":
            expr_path = args.gt_root / "CHIP" / f"{dataset}_chip_matched-ExpressionData.csv"
        else:
            expr_path = (
                args.gt_root
                / args.expr_source
                / f"{dataset}_processed-ExpressionData.csv"
            )
        if not expr_path.is_file():
            print(f"[skip] missing expr {expr_path}")
            continue

        print(f"\n=== {dataset} | expr={expr_path} ===", flush=True)
        expr = load_expression_csv(str(expr_path))
        gene_subset = None
        if args.pidc_gene_mode == "union_gt":
            gene_subset = sorted(collect_union_genes(args.gt_root, dataset, args.gt_types))
            gene_subset = [g for g in gene_subset if g in expr.index]
            print(f"  gene subset (union GT ∩ expr): {len(gene_subset)}", flush=True)
            if len(gene_subset) < 3:
                print("  [skip] too few overlapping genes")
                continue

        preds: Dict[str, pd.DataFrame] = {}

        if "PPCOR" in args.methods:
            out_tsv = pred_dir / f"PPCOR_{dataset}.tsv"
            if args.skip_existing and out_tsv.is_file():
                preds["PPCOR"] = pd.read_csv(out_tsv, sep="\t")
                print(f"  [PPCOR] loaded {out_tsv}")
            else:
                # PPCOR on subset for tractability / alignment
                if gene_subset is not None:
                    sub_expr = expr.loc[gene_subset]
                else:
                    sub_expr = expr
                tmp_csv = raw_dir / f"{dataset}_expr_for_ppcor.csv"
                sub_expr.to_csv(tmp_csv)
                t0 = time.time()
                preds["PPCOR"] = run_ppcor(
                    tmp_csv,
                    out_tsv,
                    rscript=args.rscript,
                    p_val=args.ppcor_pval,
                    work_dir=raw_dir,
                    expr_df=sub_expr,
                )
                print(f"  [PPCOR] done in {time.time() - t0:.1f}s  edges={len(preds['PPCOR'])}")

        if "PIDC" in args.methods:
            out_tsv = pred_dir / f"PIDC_{dataset}.tsv"
            if args.skip_existing and out_tsv.is_file():
                preds["PIDC"] = pd.read_csv(out_tsv, sep="\t")
                print(f"  [PIDC] loaded {out_tsv}")
            else:
                t0 = time.time()
                preds["PIDC"] = run_pidc(
                    expr,
                    out_tsv,
                    n_bins=args.pidc_bins,
                    discretizer=args.pidc_discretizer,
                    gene_subset=gene_subset,
                )
                print(f"  [PIDC] done in {time.time() - t0:.1f}s  edges={len(preds['PIDC'])}")

        for method, pred in preds.items():
            for gt_type in args.gt_types:
                sub, pat = GT_SOURCES[gt_type]
                net_path = args.gt_root / sub / pat.format(dataset=dataset)
                if not net_path.is_file():
                    rows.append(
                        {
                            "Method": method,
                            "Dataset": dataset,
                            "GroundTruth": gt_type,
                            "Status": f"MissingGT:{net_path}",
                        }
                    )
                    continue
                tfs, genes, edges = load_gt(net_path)
                # restrict to genes present in prediction / expression
                pred_genes = set(pred["Gene1"].map(norm_gene)) | set(pred["Gene2"].map(norm_gene))
                tfs_u = {g for g in tfs if g in pred_genes}
                genes_u = {g for g in genes if g in pred_genes}
                edges_u = {(a, b) for a, b in edges if a in pred_genes and b in pred_genes}
                if not tfs_u or not genes_u or not edges_u:
                    rows.append(
                        {
                            "Method": method,
                            "Dataset": dataset,
                            "GroundTruth": gt_type,
                            "Status": "NoOverlap",
                        }
                    )
                    continue
                metrics = evaluate_aupr_epr(pred, tfs_u, genes_u, edges_u)
                rows.append(
                    {
                        "Method": method,
                        "Dataset": dataset,
                        "GroundTruth": gt_type,
                        "Status": "Success",
                        "expr_source": args.expr_source,
                        **metrics,
                    }
                )
                print(
                    f"  [{method} | {gt_type}] AUPR_Ratio={metrics['AUPR_Ratio']:.4f} "
                    f"EPR={metrics['EPR']:.4f}",
                    flush=True,
                )

    df = pd.DataFrame(rows)
    long_path = args.outdir / "beeline_baselines_aupr_epr.csv"
    df.to_csv(long_path, index=False)

    ok = df[df.get("Status", "Success") == "Success"].copy() if "Status" in df.columns else df
    if not ok.empty:
        for metric in ("AUPR", "AUPR_Ratio", "EPR"):
            wide = ok.pivot_table(
                index=["Method", "Dataset"],
                columns="GroundTruth",
                values=metric,
                aggfunc="first",
            ).reset_index()
            wide.to_csv(args.outdir / f"beeline_baselines_{metric}_wide.csv", index=False)
        summary = (
            ok.groupby(["Method", "GroundTruth"])[["AUPR", "AUPR_Ratio", "EPR"]]
            .agg(["mean", "std"])
            .reset_index()
        )
        summary.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in summary.columns]
        summary.to_csv(args.outdir / "beeline_baselines_summary_mean_std.csv", index=False)

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "note": (
            "PIDC is a Python reimplementation of NetworkInference.jl / Chan et al.; "
            "PPCOR uses R ppcor with Spearman as in BEELINE (pVal default 0.01)."
        ),
    }
    (args.outdir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\n=== Done ===")
    print(f"results: {long_path}")


if __name__ == "__main__":
    main()
