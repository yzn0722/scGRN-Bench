#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load scGPT ``gene_result.csv`` or scFoundation ``gene_delta_compare.csv`` into a unified schema."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

EPS = 1e-8

SCGPT_DEFAULT = Path("/mnt/10T/yzn/benchmark_GRN/pre_scgpt/results_multidataset_pseudotime_227")
SCF_DEFAULT = Path("/mnt/10T/yzn/benchmark_GRN/pre_scfoundation/scfoundation_multidataset_pseudotime_227")


def _dir_label(x) -> str:
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("up", "1", "true"):
            return "Up"
        if s in ("down", "-1", "false"):
            return "Down"
    v = float(x) if x is not None and str(x) not in ("", "nan") else 0.0
    if v > 0:
        return "Up"
    return "Down"


def _dir_correct_col(df: pd.DataFrame) -> pd.Series:
    if "dir_correct" in df.columns:
        return pd.to_numeric(df["dir_correct"], errors="coerce").fillna(0).astype(int)
    if "dir_match" in df.columns:
        return df["dir_match"].astype(bool).astype(int)
    return (df["dir_true"].astype(str) == df["dir_pred"].astype(str)).astype(int)


def normalize_gene_result_df(df: pd.DataFrame, mapped_only: bool = False) -> pd.DataFrame:
    """Unified columns for plot_dynamic_error_biology."""
    df = df.copy()

    if "gene_delta_compare" in str(df.columns) or "early_mean" in df.columns:
        rename = {
            "early_mean": "true_early_mean",
            "late_mean": "true_late_mean",
            "pred_mean": "pred_late_like_mean",
            "true_delta": "delta_true",
            "pred_delta": "delta_pred",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        if "true_dir" in df.columns:
            df["dir_true"] = df["true_dir"].map(_dir_label)
        if "pred_dir" in df.columns:
            df["dir_pred"] = df["pred_dir"].map(_dir_label)
        if mapped_only and "mapped" in df.columns:
            df = df[df["mapped"].astype(bool)].copy()

    required = [
        "gene",
        "true_early_mean",
        "true_late_mean",
        "pred_late_like_mean",
        "delta_true",
        "delta_pred",
        "dir_true",
        "dir_pred",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after normalize: {missing}")

    df["dir_correct"] = _dir_correct_col(df)
    for c in ["delta_true", "delta_pred", "true_early_mean", "true_late_mean", "pred_late_like_mean"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["delta_true", "delta_pred"])
    return df


def load_gene_result(path: Union[str, Path], mapped_only: bool = False) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    return normalize_gene_result_df(df, mapped_only=mapped_only)


def gene_result_path_for_model(dataset: str, model: str, root: Path | None = None) -> Path:
    model = model.lower().replace("scfoundation", "scf").replace("scprint", "scprint")
    if model in ("scgpt", "gpt"):
        base = root or SCGPT_DEFAULT
        return base / f"{dataset}_gene_result.csv"
    if model in ("scf", "scfoundation", "scfoundation"):
        base = root or SCF_DEFAULT
        return base / dataset / "gene_delta_compare.csv"
    raise ValueError(f"Unknown model: {model}")


def export_unified_gene_results(
    model: str,
    datasets: list[str],
    out_dir: Path,
    src_root: Path | None = None,
    mapped_only: bool = True,
) -> Path:
    """Write ``{dataset}_gene_result.csv`` in scGPT column layout for downstream plotting."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        src = gene_result_path_for_model(ds, model, src_root)
        if not src.is_file():
            raise FileNotFoundError(src)
        norm = load_gene_result(src, mapped_only=mapped_only)
        norm.to_csv(out_dir / f"{ds}_gene_result.csv", index=False)
    return out_dir
