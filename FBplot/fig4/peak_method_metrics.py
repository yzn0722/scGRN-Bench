#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared metrics: single-step wrong + peak-transition rescue (chained scGPT)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

GENE_RESULT_DIR = Path("/mnt/10T/yzn/benchmark_GRN/pre_scgpt/results_multidataset_pseudotime_227")
MULTISTEP_ROOT = Path(__file__).resolve().parent / "error_biology" / "multistep_pt"
TOP_PERCENT = 0.3
DATASETS = ["hESC", "hHep", "mDC", "mHSC-E", "mHSC-GM", "mHSC-L"]


def parse_transition(trans: str) -> Optional[int]:
    m = re.match(r"delta_t(\d+)_t(\d+)", str(trans))
    return int(m.group(1)) if m else None


def dir_sign(x: float) -> str:
    if not np.isfinite(x) or x == 0:
        return "Down"
    return "Up" if x > 0 else "Down"


def is_ercc_gene(gene: str) -> bool:
    """ERCC spike-in controls (excluded in sensitivity analysis)."""
    return str(gene).upper().startswith("ERCC")


def load_chained_trans(chained_dir: Path, dataset: str, trans: str) -> Optional[pd.DataFrame]:
    p = chained_dir / f"{dataset}_{trans}_gene_result.csv"
    return pd.read_csv(p).set_index("gene") if p.exists() else None


def early_local_mean(ch: pd.DataFrame, gene: str) -> float:
    """兼容旧列 true_early_mean 与新列 true_early_mean_local。"""
    if gene not in ch.index:
        return float("nan")
    row = ch.loc[gene]
    if "true_early_mean_local" in ch.columns:
        return float(row["true_early_mean_local"])
    if "true_early_mean" in ch.columns:
        return float(row["true_early_mean"])
    if "true_late_mean" in ch.columns and "delta_true_local" in ch.columns:
        return float(row["true_late_mean"]) - float(row["delta_true_local"])
    return float("nan")


def eval_dataset(
    dataset: str,
    multistep_root: Path = MULTISTEP_ROOT,
    gene_dir: Path = GENE_RESULT_DIR,
    top_percent: float = TOP_PERCENT,
    exclude_ercc: bool = False,
) -> Dict:
    ds_dir = multistep_root / dataset
    gf_path = ds_dir / "gene_segment_deltas.csv"
    gr_path = gene_dir / f"{dataset}_gene_result.csv"
    if not gf_path.exists() or not gr_path.exists():
        return {"dataset": dataset, "status": "missing_input"}

    gf = pd.read_csv(gf_path)
    gr = pd.read_csv(gr_path)
    if exclude_ercc:
        gf = gf[~gf["gene"].astype(str).map(is_ercc_gene)].copy()
        gr = gr[~gr["gene"].astype(str).map(is_ercc_gene)].copy()
    n_top = max(1, int(np.ceil(top_percent * len(gf))))
    top = gf.assign(_abs=gf["delta_single"].abs()).nlargest(n_top, "_abs")
    top = top.merge(
        gr[["gene", "dir_correct"]].rename(columns={"dir_correct": "dir_correct_single"}),
        on="gene",
        how="left",
    )
    top["dir_true_peak"] = top["delta_best_trans"].apply(dir_sign)

    wrong = top[top["dir_correct_single"] == 0].copy()
    chained_dir = ds_dir / "chained_preds"
    ch_s0 = load_chained_trans(chained_dir, dataset, "delta_t0_t1")

    fixed = 0
    has_chain = ch_s0 is not None
    for _, r in wrong.iterrows():
        trans = str(r["best_transition"])
        g = str(r["gene"])
        ch = load_chained_trans(chained_dir, dataset, trans)
        if ch is None or g not in ch.index:
            continue
        pred_late = float(ch.loc[g, "pred_late_like_mean"])
        early_local = early_local_mean(ch, g)
        if not np.isfinite(early_local):
            continue
        if dir_sign(pred_late - early_local) == r["dir_true_peak"]:
            fixed += 1

    n_wrong = len(wrong)
    return {
        "dataset": dataset,
        "status": "ok" if has_chain else "no_chained",
        "exclude_ercc": exclude_ercc,
        "n_top30": n_top,
        "n_wrong_single": n_wrong,
        "frac_wrong_single": n_wrong / n_top if n_top else np.nan,
        "n_fixed_peak_chained": fixed if has_chain else np.nan,
        "frac_fixed_among_wrong": fixed / n_wrong if has_chain and n_wrong else np.nan,
        "n_still_wrong": n_wrong - fixed if has_chain else np.nan,
    }
