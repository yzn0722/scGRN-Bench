#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute and plot AUPR (Up vs Down) for multiple models on hESC.

Unified definition (same as scGPT script):
- keep samples with true label in {Up, Down} (or {-1, +1})
- positive class: Up (+1)
- score: delta_pred (higher => more likely Up)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fig4_palette import apply_fig4_style, model_color


MODEL_FILES: Dict[str, Path] = {
    "scGPT": Path("/mnt/10T/yzn/benchmark_GRN/pre_scgpt/results_multidataset_pseudotime_227/hESC_gene_result.csv"),
    "Geneformer": Path("/mnt/10T/yzn/benchmark_GRN/pre_geneformer_results_unified/geneformer/hESC_gene_result.csv"),
    "LangCell": Path("/mnt/10T/yzn/benchmark_GRN/pre_langcell_results_unified/langcell/hESC_gene_result.csv"),
    "scFoundation": Path("/mnt/10T/yzn/benchmark_GRN/pre_scfoundation/scfoundation_multidataset_pseudotime_227/hESC/gene_delta_compare.csv"),
    "scPrint": Path("/mnt/10T/yzn/benchmark_GRN/pre_scprint_results_unified/scprint/per_dataset/hESC/per_gene_final_changes.csv"),
}

OUT_DIR = Path(__file__).resolve().parent / "accuracy"
OUT_CSV = OUT_DIR / "hESC_aupr_multimodel.csv"
OUT_PDF = OUT_DIR / "hESC_aupr_multimodel_bar.pdf"


def average_precision_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    if y_true.size == 0 or n_pos == 0:
        return float("nan")
    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    k = np.arange(1, y.size + 1)
    return float((tp / k)[y == 1].sum() / n_pos)


def extract_true_and_score(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    # true labels
    true_col = None
    for c in ("dir_true", "true_dir", "true_sign"):
        if c in df.columns:
            true_col = c
            break
    if true_col is not None:
        s = df[true_col]
        if s.dtype == object:
            mask = s.isin(["Up", "Down"])
            y_true = (s[mask] == "Up").astype(np.int32).to_numpy()
            idx = mask.to_numpy()
        else:
            # numeric: -1 / 0 / 1
            ss = pd.to_numeric(s, errors="coerce")
            mask = ss.isin([-1, 1])
            y_true = (ss[mask] == 1).astype(np.int32).to_numpy()
            idx = mask.to_numpy()
    else:
        raise ValueError("No true label column found (dir_true / true_dir / true_sign).")

    # score
    if "delta_pred" in df.columns:
        score_all = pd.to_numeric(df["delta_pred"], errors="coerce").to_numpy()
    elif "pred_delta" in df.columns:
        score_all = pd.to_numeric(df["pred_delta"], errors="coerce").to_numpy()
    else:
        raise ValueError("No score column found (delta_pred / pred_delta).")

    y_score = score_all[idx]
    finite = np.isfinite(y_score)
    return y_true[finite], y_score[finite]


def main() -> None:
    apply_fig4_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for model, path in MODEL_FILES.items():
        if not path.is_file():
            rows.append({"Model": model, "AUPR_Up_vs_Down": np.nan, "Baseline_PosRate": np.nan, "N_Used": 0, "Path": str(path)})
            continue
        df = pd.read_csv(path)
        y_true, y_score = extract_true_and_score(df)
        ap = average_precision_binary(y_true, y_score)
        base = float(y_true.mean()) if y_true.size > 0 else float("nan")
        rows.append(
            {
                "Model": model,
                "AUPR_Up_vs_Down": ap,
                "Baseline_PosRate": base,
                "N_Used": int(y_true.size),
                "Path": str(path),
            }
        )

    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)

    # plot
    fig, ax = plt.subplots(figsize=(10, 5), dpi=600)
    x = np.arange(len(res))
    vals = res["AUPR_Up_vs_Down"].to_numpy(dtype=float)
    bases = res["Baseline_PosRate"].to_numpy(dtype=float)
    names = res["Model"].tolist()
    colors = [model_color(m) for m in names]
    ax.bar(x, vals, color=colors, edgecolor="none", linewidth=0)
    for xi, b in zip(x, bases):
        if np.isfinite(b):
            ax.hlines(b, xi - 0.38, xi + 0.38, colors=model_color("STRING"), linestyles="--", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("AUPR (Up vs Down)", fontsize=16)
    ax.tick_params(axis="x", labelsize=14, length=0)
    ax.tick_params(axis="y", labelsize=14, length=0)
    ax.set_title("hESC AUPR by Model", fontsize=16, color="#000000")
    ax.set_ylim(0, 1.0)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved table: {OUT_CSV}")
    print(f"Saved plot : {OUT_PDF}")


if __name__ == "__main__":
    main()

