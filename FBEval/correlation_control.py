#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correlation control: is embedding GRN score largely co-expression?

For each (model, extraction, dataset):
  1) Join embedding edge score S with expression correlation R
  2) Report Spearman ρ(S, R) globally and stratified by GT edge sets
  3) Fit S ~ R, take residual ε = S - Ŝ(R)
  4) Evaluate AUPR / AUPR_Ratio for scores {R, S, ε} on STRING / Non_CHIP / CHIP
  5) Optional incremental test: label ~ R  vs  label ~ R + S

Example:
  python -u FBEval/correlation_control.py \
    --pred-root /mnt/10T/yzn/benchmark_GRN/evl_omipath \
    --gt-root /mnt/10T/yzn/benchmark_GRN/input_process \
    --expr-root /mnt/10T/yzn/benchmark_GRN/input_process \
    --datasets hESC \
    --models scGPT \
    --extractions emb500 embhidden500 \
    --gt-types STRING Non_CHIP CHIP \
    --outdir outputs/correlation_control
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Defaults aligned with FBplot/fig3/tf_static/model_registry.py
# ---------------------------------------------------------------------------

GT_SOURCES: Dict[str, Tuple[str, str]] = {
    "STRING": ("STRING", "{dataset}_processed-network.csv"),
    "Non_CHIP": ("Non_CHIP", "{dataset}_processed-network.csv"),
    "CHIP": ("CHIP", "{dataset}_chip_matched-network.csv"),
    "omnipath": ("omnipath", "{dataset}_processed-network.csv"),
}

MODEL_DIRS: Dict[str, Dict[str, str]] = {
    "Geneformer": {"emb500": "geneformer", "att500": "geneformer", "embhidden500": "geneformer"},
    "LangCell": {"emb500": "langcell", "att500": "langcell", "embhidden500": "Langcell"},
    "scGPT": {"emb500": "scgpt", "att500": "scgpt", "embhidden500": "scgpt"},
    "scCello": {"emb500": "sccello", "att500": "sccello", "embhidden500": "sccello"},
    "scFoundation": {"emb500": "scFoundation", "att500": "scFoundation", "embhidden500": "scFoundation"},
    "scPrint": {"emb500": "scprint", "att500": "scprint", "embhidden500": "scprint"},
}

MODEL_STEMS: Dict[str, Tuple[str, ...]] = {
    "Geneformer": ("geneformer", "Geneformer"),
    "LangCell": ("LangCell", "langcell", "Langcell"),
    "scGPT": ("scgpt", "scGPT", "scGPT2"),
    "scCello": ("scCello", "sccello"),
    "scFoundation": ("scFoundation", "scfoundation"),
    "scPrint": ("scprint", "scPrint", "scPRINT"),
}

WEIGHT_COLS = (
    "EdgeWeight",
    "edgeweight",
    "edge_weight",
    "Attention score",
    "Weight",
    "Score",
    "Importance",
)


def norm_gene(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def detect_weight_col(df: pd.DataFrame) -> Optional[str]:
    for c in WEIGHT_COLS:
        if c in df.columns:
            return c
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in ("weight", "score", "import", "att")):
            return c
    return None


def resolve_gt_path(gt_root: Path, gt_type: str, dataset: str) -> Path:
    if gt_type not in GT_SOURCES:
        raise ValueError(f"Unknown gt type {gt_type!r}; choose from {list(GT_SOURCES)}")
    subdir, pattern = GT_SOURCES[gt_type]
    path = gt_root / subdir / pattern.format(dataset=dataset)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def resolve_pred_path(pred_root: Path, model: str, extraction: str, dataset: str) -> Optional[Path]:
    if model not in MODEL_DIRS or extraction not in MODEL_DIRS[model]:
        return None
    base = pred_root / f"output_{extraction}" / MODEL_DIRS[model][extraction]
    if not base.is_dir():
        return None
    for stem in MODEL_STEMS[model]:
        cand = base / f"{stem}_{dataset}.tsv"
        if cand.is_file():
            return cand
    # case-insensitive / fuzzy fallback
    hits = sorted(p for p in base.glob("*.tsv") if dataset.lower() in p.name.lower())
    return hits[0] if hits else None


def resolve_expr_path(expr_root: Path, dataset: str, prefer: str = "STRING") -> Path:
    """Prefer STRING expression; fall back to CHIP matched expression."""
    candidates = [
        expr_root / prefer / f"{dataset}_processed-ExpressionData.csv",
        expr_root / "STRING" / f"{dataset}_processed-ExpressionData.csv",
        expr_root / "CHIP" / f"{dataset}_chip_matched-ExpressionData.csv",
        expr_root / "Non_CHIP" / f"{dataset}_processed-ExpressionData.csv",
        expr_root / f"{dataset}_processed-ExpressionData.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"No expression matrix for {dataset} under {expr_root}. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def load_gt_edges(path: Path) -> Tuple[Set[str], Set[str], Set[Tuple[str, str]]]:
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


def load_pred_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    wcol = detect_weight_col(df)
    if wcol is None:
        raise ValueError(f"No edge-weight column in {path}; columns={list(df.columns)}")
    out = pd.DataFrame(
        {
            "Gene1": df["Gene1"].map(norm_gene),
            "Gene2": df["Gene2"].map(norm_gene),
            "S": pd.to_numeric(df[wcol], errors="coerce").abs(),
        }
    )
    out = out[(out["Gene1"] != "") & (out["Gene2"] != "") & (out["Gene1"] != out["Gene2"])]
    out = out.dropna(subset=["S"])
    out = out.drop_duplicates(subset=["Gene1", "Gene2"], keep="first")
    return out.reset_index(drop=True)


def load_expression(path: Path) -> pd.DataFrame:
    """Return genes × cells float matrix with uppercase gene index."""
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.map(norm_gene)
    df = df[~df.index.duplicated(keep="first")]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    # drop empty gene names / all-zero genes (optional but stabilizes corr)
    df = df.loc[df.index != ""]
    return df


def pairwise_corr_matrix(expr: pd.DataFrame, method: str = "spearman") -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Compute gene–gene correlation matrix.
    method: 'spearman' | 'pearson'
    """
    genes = list(expr.index)
    gene_to_i = {g: i for i, g in enumerate(genes)}
    x = expr.to_numpy(dtype=np.float64)
    if method == "spearman":
        # rank along cells, then Pearson of ranks
        ranks = np.empty_like(x)
        for i in range(x.shape[0]):
            ranks[i] = stats.rankdata(x[i], method="average")
        mat = np.corrcoef(ranks)
    elif method == "pearson":
        mat = np.corrcoef(x)
    else:
        raise ValueError(f"Unknown corr method: {method}")
    # numerical nans (constant genes) -> 0
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return mat, gene_to_i


def lookup_corr(
    gene1: Sequence[str],
    gene2: Sequence[str],
    corr: np.ndarray,
    gene_to_i: Dict[str, int],
) -> np.ndarray:
    out = np.full(len(gene1), np.nan, dtype=np.float64)
    for k, (a, b) in enumerate(zip(gene1, gene2)):
        ia = gene_to_i.get(a)
        ib = gene_to_i.get(b)
        if ia is None or ib is None:
            continue
        out[k] = corr[ia, ib]
    return out


def spearman_safe(a: np.ndarray, b: np.ndarray) -> Tuple[float, int]:
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(mask.sum())
    if n < 5:
        return float("nan"), n
    rho, _ = stats.spearmanr(a[mask], b[mask])
    return float(rho), n


def fit_residual(s: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """
    OLS: S = β0 + β1 R + ε. Returns (ε, β0, β1, R^2).
    Fit only on finite pairs; ε is NaN where either input is NaN.
    """
    mask = np.isfinite(s) & np.isfinite(r)
    eps = np.full_like(s, np.nan, dtype=np.float64)
    if mask.sum() < 5:
        return eps, float("nan"), float("nan"), float("nan")
    reg = LinearRegression()
    reg.fit(r[mask].reshape(-1, 1), s[mask])
    pred = reg.predict(r[mask].reshape(-1, 1))
    eps[mask] = s[mask] - pred
    r2 = float(reg.score(r[mask].reshape(-1, 1), s[mask]))
    return eps, float(reg.intercept_), float(reg.coef_[0]), r2


def evaluate_aupr_on_gt(
    score_dict: Dict[Tuple[str, str], float],
    tfs: Set[str],
    genes: Set[str],
    true_edges: Set[Tuple[str, str]],
    missing_score: float = -1.0,
) -> Dict[str, float]:
    """FBEval-compatible AUPR over all TF→gene pairs in GT node universe."""
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

    # Early precision at K=|E_GT|
    order = np.argsort(-p)
    k = min(n_true, int(order.size))
    if k == 0:
        epr = 0.0
    else:
        top = order[:k]
        tp = int(y[top].sum())
        prec = tp / k
        epr = (prec / baseline) if baseline > 0 else 0.0

    return {
        "AUPR": round(aupr, 6),
        "AUPR_Ratio": round(ratio, 6),
        "EPR": round(float(epr), 6),
        "Random_Baseline": round(baseline, 8),
        "True_Edges": n_true,
        "Possible_Edges": n_possible,
        "Scored_Edges": int(sum(1 for v in score_dict.values() if np.isfinite(v))),
    }


def incremental_metrics(
    labels: np.ndarray,
    r: np.ndarray,
    s: np.ndarray,
) -> Dict[str, float]:
    """Compare label~R vs label~R+S on finite pairs (AP + AUROC)."""
    mask = np.isfinite(r) & np.isfinite(s) & np.isfinite(labels)
    out = {
        "n_pairs": int(mask.sum()),
        "n_pos": int(labels[mask].sum()) if mask.any() else 0,
        "AP_R": float("nan"),
        "AP_S": float("nan"),
        "AP_RS": float("nan"),
        "AUROC_R": float("nan"),
        "AUROC_S": float("nan"),
        "AUROC_RS": float("nan"),
        "coef_S_in_RS": float("nan"),
    }
    if out["n_pairs"] < 50 or out["n_pos"] < 5:
        return out

    y = labels[mask].astype(int)
    rr = r[mask].reshape(-1, 1)
    ss = s[mask].reshape(-1, 1)
    xs = np.hstack([rr, ss])

    out["AP_R"] = float(average_precision_score(y, r[mask]))
    out["AP_S"] = float(average_precision_score(y, s[mask]))
    try:
        out["AUROC_R"] = float(roc_auc_score(y, r[mask]))
        out["AUROC_S"] = float(roc_auc_score(y, s[mask]))
    except ValueError:
        pass

    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(xs, y)
    proba = clf.predict_proba(xs)[:, 1]
    out["AP_RS"] = float(average_precision_score(y, proba))
    try:
        out["AUROC_RS"] = float(roc_auc_score(y, proba))
    except ValueError:
        pass
    out["coef_S_in_RS"] = float(clf.coef_[0, 1])
    return out


@dataclass
class EdgeSets:
    string: Set[Tuple[str, str]]
    non_chip: Set[Tuple[str, str]]
    chip: Set[Tuple[str, str]]

    @property
    def string_only(self) -> Set[Tuple[str, str]]:
        return self.string - self.chip

    @property
    def chip_only(self) -> Set[Tuple[str, str]]:
        return self.chip - self.string

    @property
    def intersection(self) -> Set[Tuple[str, str]]:
        return self.string & self.chip


def build_score_dict(g1: np.ndarray, g2: np.ndarray, values: np.ndarray) -> Dict[Tuple[str, str], float]:
    d: Dict[Tuple[str, str], float] = {}
    for a, b, v in zip(g1, g2, values):
        if np.isfinite(v):
            d[(str(a), str(b))] = float(v)
    return d


def run_one(
    *,
    model: str,
    extraction: str,
    dataset: str,
    pred_path: Path,
    expr_path: Path,
    gt_root: Path,
    gt_types: Sequence[str],
    corr_method: str,
    max_pairs_for_rho: int,
    do_incremental: bool,
    save_residual_tsv: Optional[Path],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    pred = load_pred_scores(pred_path)
    expr = load_expression(expr_path)
    corr, gene_to_i = pairwise_corr_matrix(expr, method=corr_method)

    pred["R"] = lookup_corr(pred["Gene1"].tolist(), pred["Gene2"].tolist(), corr, gene_to_i)
    pred = pred[np.isfinite(pred["R"])].copy()
    if pred.empty:
        raise RuntimeError(f"No overlapping gene pairs between pred and expression for {pred_path}")

    eps, b0, b1, r2 = fit_residual(pred["S"].to_numpy(), pred["R"].to_numpy())
    pred["residual"] = eps

    # Load GT edge sets used for stratification (best-effort)
    edge_sets_raw: Dict[str, Set[Tuple[str, str]]] = {}
    for gt in ("STRING", "Non_CHIP", "CHIP"):
        try:
            _, _, edges = load_gt_edges(resolve_gt_path(gt_root, gt, dataset))
            edge_sets_raw[gt] = edges
        except FileNotFoundError:
            edge_sets_raw[gt] = set()
    sets = EdgeSets(
        string=edge_sets_raw.get("STRING", set()),
        non_chip=edge_sets_raw.get("Non_CHIP", set()),
        chip=edge_sets_raw.get("CHIP", set()),
    )

    # Optional downsample for global ρ if huge
    if len(pred) > max_pairs_for_rho:
        sample = pred.sample(n=max_pairs_for_rho, random_state=0)
    else:
        sample = pred
    rho_all, n_all = spearman_safe(sample["S"].to_numpy(), sample["R"].to_numpy())

    def rho_on_edge_set(edge_set: Set[Tuple[str, str]], name: str) -> Dict:
        if not edge_set:
            return {
                "stratum": name,
                "rho_S_R": float("nan"),
                "n_pairs": 0,
            }
        keys = pred.set_index(["Gene1", "Gene2"]).index
        mask = keys.isin(edge_set)
        sub = pred.loc[mask]
        rho, n = spearman_safe(sub["S"].to_numpy(), sub["R"].to_numpy())
        return {"stratum": name, "rho_S_R": rho, "n_pairs": n}

    rho_rows = [
        {
            "Model": model,
            "Extraction": extraction,
            "Dataset": dataset,
            "stratum": "all_pred_pairs",
            "rho_S_R": rho_all,
            "n_pairs": n_all,
            "ols_beta0": b0,
            "ols_beta1": b1,
            "ols_R2": r2,
            "corr_method": corr_method,
            "pred_path": str(pred_path),
            "expr_path": str(expr_path),
        }
    ]
    for name, eset in (
        ("STRING", sets.string),
        ("Non_CHIP", sets.non_chip),
        ("CHIP", sets.chip),
        ("STRING_minus_CHIP", sets.string_only),
        ("CHIP_minus_STRING", sets.chip_only),
        ("STRING_and_CHIP", sets.intersection),
    ):
        row = rho_on_edge_set(eset, name)
        row.update(
            {
                "Model": model,
                "Extraction": extraction,
                "Dataset": dataset,
                "ols_beta0": b0,
                "ols_beta1": b1,
                "ols_R2": r2,
                "corr_method": corr_method,
                "pred_path": str(pred_path),
                "expr_path": str(expr_path),
            }
        )
        rho_rows.append(row)
    rho_df = pd.DataFrame(rho_rows)

    # Score dicts for AUPR (use all overlapping pred pairs)
    g1 = pred["Gene1"].to_numpy()
    g2 = pred["Gene2"].to_numpy()
    dict_s = build_score_dict(g1, g2, pred["S"].to_numpy())
    dict_r = build_score_dict(g1, g2, pred["R"].to_numpy())
    dict_e = build_score_dict(g1, g2, pred["residual"].to_numpy())

    aupr_rows: List[Dict] = []
    incr_rows: List[Dict] = []
    for gt_type in gt_types:
        try:
            tfs, genes, true_edges = load_gt_edges(resolve_gt_path(gt_root, gt_type, dataset))
        except FileNotFoundError as exc:
            aupr_rows.append(
                {
                    "Model": model,
                    "Extraction": extraction,
                    "Dataset": dataset,
                    "GroundTruth": gt_type,
                    "Score": "NA",
                    "Status": f"MissingGT: {exc}",
                }
            )
            continue

        for score_name, score_dict in (
            ("expr_corr_R", dict_r),
            ("embedding_S", dict_s),
            ("residual_eps", dict_e),
        ):
            metrics = evaluate_aupr_on_gt(score_dict, tfs, genes, true_edges)
            aupr_rows.append(
                {
                    "Model": model,
                    "Extraction": extraction,
                    "Dataset": dataset,
                    "GroundTruth": gt_type,
                    "Score": score_name,
                    "Status": "Success",
                    **metrics,
                }
            )

        if do_incremental:
            # Build label / R / S vectors on GT candidate space where both scores exist
            labs, rr, ss = [], [], []
            for tf in tfs:
                for g in genes:
                    if tf == g:
                        continue
                    key = (tf, g)
                    if key not in dict_r or key not in dict_s:
                        continue
                    labs.append(1 if key in true_edges else 0)
                    rr.append(dict_r[key])
                    ss.append(dict_s[key])
            inc = incremental_metrics(np.asarray(labs), np.asarray(rr), np.asarray(ss))
            incr_rows.append(
                {
                    "Model": model,
                    "Extraction": extraction,
                    "Dataset": dataset,
                    "GroundTruth": gt_type,
                    **inc,
                }
            )

    meta = {
        "model": model,
        "extraction": extraction,
        "dataset": dataset,
        "n_pred_joined": int(len(pred)),
        "n_expr_genes": int(expr.shape[0]),
        "n_expr_cells": int(expr.shape[1]),
        "ols_beta0": b0,
        "ols_beta1": b1,
        "ols_R2": r2,
        "rho_all": rho_all,
    }

    if save_residual_tsv is not None:
        # Shift to non-negative so FBEval/AUPR.py's abs(EdgeWeight) keeps ranking.
        save_residual_tsv.parent.mkdir(parents=True, exist_ok=True)
        export = pred[["Gene1", "Gene2", "residual"]].dropna(subset=["residual"]).copy()
        lo = float(export["residual"].min())
        export["EdgeWeight"] = export["residual"] - lo
        export[["Gene1", "Gene2", "EdgeWeight"]].to_csv(save_residual_tsv, sep="\t", index=False)

    return rho_df, pd.DataFrame(aupr_rows), pd.DataFrame(incr_rows), meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Correlation control: embedding score vs expression co-expression"
    )
    p.add_argument(
        "--pred-root",
        type=Path,
        default=Path("/mnt/10T/yzn/benchmark_GRN/evl_omipath"),
        help="Root containing output_emb500/, output_embhidden500/, ...",
    )
    p.add_argument(
        "--gt-root",
        type=Path,
        default=Path("/mnt/10T/yzn/benchmark_GRN/input_process"),
        help="Ground-truth root (CHIP / Non_CHIP / STRING)",
    )
    p.add_argument(
        "--expr-root",
        type=Path,
        default=Path("/mnt/10T/yzn/benchmark_GRN/input_process"),
        help="Expression root (usually same as gt-root)",
    )
    p.add_argument(
        "--expr-prefer",
        default="STRING",
        choices=["STRING", "CHIP", "Non_CHIP"],
        help="Preferred folder for expression matrices",
    )
    p.add_argument("--datasets", nargs="+", default=["hESC"])
    p.add_argument(
        "--models",
        nargs="+",
        default=["scGPT"],
        choices=list(MODEL_DIRS.keys()),
    )
    p.add_argument(
        "--extractions",
        nargs="+",
        default=["emb500", "embhidden500"],
        choices=["emb500", "embhidden500", "att500"],
    )
    p.add_argument(
        "--gt-types",
        nargs="+",
        default=["STRING", "Non_CHIP", "CHIP"],
        choices=list(GT_SOURCES.keys()),
    )
    p.add_argument(
        "--corr-method",
        default="spearman",
        choices=["spearman", "pearson"],
        help="Gene–gene expression correlation used as R",
    )
    p.add_argument(
        "--max-pairs-for-rho",
        type=int,
        default=200_000,
        help="Downsample size for global ρ(S,R) if pred is huge",
    )
    p.add_argument("--no-incremental", action="store_true", help="Skip label~R vs R+S test")
    p.add_argument(
        "--save-residual-tsv",
        action="store_true",
        help="Write residual EdgeWeight TSVs under outdir/residual_tsv/",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/correlation_control"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    all_rho: List[pd.DataFrame] = []
    all_aupr: List[pd.DataFrame] = []
    all_incr: List[pd.DataFrame] = []
    metas: List[Dict] = []
    skipped: List[Dict] = []

    for dataset in args.datasets:
        try:
            expr_path = resolve_expr_path(args.expr_root, dataset, prefer=args.expr_prefer)
        except FileNotFoundError as exc:
            skipped.append({"dataset": dataset, "reason": str(exc)})
            print(f"[skip] {exc}")
            continue

        for model in args.models:
            for extraction in args.extractions:
                pred_path = resolve_pred_path(args.pred_root, model, extraction, dataset)
                if pred_path is None:
                    msg = f"Missing pred for {model}/{extraction}/{dataset}"
                    print(f"[skip] {msg}")
                    skipped.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "extraction": extraction,
                            "reason": msg,
                        }
                    )
                    continue

                print(
                    f"[run] {model} | {extraction} | {dataset}\n"
                    f"      pred={pred_path}\n"
                    f"      expr={expr_path}"
                )
                residual_path = None
                if args.save_residual_tsv:
                    residual_path = (
                        args.outdir
                        / "residual_tsv"
                        / f"{model}_{extraction}_{dataset}_residual.tsv"
                    )

                try:
                    rho_df, aupr_df, incr_df, meta = run_one(
                        model=model,
                        extraction=extraction,
                        dataset=dataset,
                        pred_path=pred_path,
                        expr_path=expr_path,
                        gt_root=args.gt_root,
                        gt_types=args.gt_types,
                        corr_method=args.corr_method,
                        max_pairs_for_rho=args.max_pairs_for_rho,
                        do_incremental=not args.no_incremental,
                        save_residual_tsv=residual_path,
                    )
                except Exception as exc:  # keep batch going
                    print(f"[error] {model}/{extraction}/{dataset}: {exc}")
                    skipped.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "extraction": extraction,
                            "reason": str(exc),
                        }
                    )
                    continue

                all_rho.append(rho_df)
                all_aupr.append(aupr_df)
                if not incr_df.empty:
                    all_incr.append(incr_df)
                metas.append(meta)
                print(
                    f"      rho(S,R)={meta['rho_all']:.4f}  "
                    f"OLS R2={meta['ols_R2']:.4f}  "
                    f"n_joined={meta['n_pred_joined']}"
                )

    if not all_aupr:
        raise SystemExit("No successful runs. Check --pred-root / --datasets / --models.")

    rho_out = pd.concat(all_rho, ignore_index=True)
    aupr_out = pd.concat(all_aupr, ignore_index=True)
    incr_out = pd.concat(all_incr, ignore_index=True) if all_incr else pd.DataFrame()

    rho_path = args.outdir / "rho_embedding_vs_expression.csv"
    aupr_path = args.outdir / "aupr_R_S_residual_by_gt.csv"
    incr_path = args.outdir / "incremental_R_vs_RS.csv"
    meta_path = args.outdir / "run_meta.json"

    rho_out.to_csv(rho_path, index=False)
    aupr_out.to_csv(aupr_path, index=False)
    if not incr_out.empty:
        incr_out.to_csv(incr_path, index=False)

    # Wide summary: AUPR_Ratio for R / S / residual × GT
    if "Status" in aupr_out.columns:
        ok = aupr_out[aupr_out["Status"] == "Success"].copy()
    else:
        ok = aupr_out.copy()
    if not ok.empty:
        pivot = ok.pivot_table(
            index=["Model", "Extraction", "Dataset", "GroundTruth"],
            columns="Score",
            values="AUPR_Ratio",
            aggfunc="first",
        ).reset_index()
        pivot_path = args.outdir / "aupr_ratio_wide.csv"
        pivot.to_csv(pivot_path, index=False)
    else:
        pivot_path = None

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "args": {
            "pred_root": str(args.pred_root),
            "gt_root": str(args.gt_root),
            "expr_root": str(args.expr_root),
            "expr_prefer": args.expr_prefer,
            "datasets": args.datasets,
            "models": args.models,
            "extractions": args.extractions,
            "gt_types": args.gt_types,
            "corr_method": args.corr_method,
        },
        "n_success": len(metas),
        "metas": metas,
        "skipped": skipped,
        "outputs": {
            "rho": str(rho_path),
            "aupr": str(aupr_path),
            "incremental": str(incr_path) if not incr_out.empty else None,
            "aupr_ratio_wide": str(pivot_path) if pivot_path else None,
        },
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== Done ===")
    print(f"rho table     : {rho_path}")
    print(f"AUPR table    : {aupr_path}")
    if not incr_out.empty:
        print(f"incremental   : {incr_path}")
    if pivot_path:
        print(f"AUPR_Ratio wide: {pivot_path}")
    print(f"meta          : {meta_path}")


if __name__ == "__main__":
    main()
