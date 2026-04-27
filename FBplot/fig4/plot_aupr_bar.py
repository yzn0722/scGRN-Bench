#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot grouped bar chart of AUPR(correct) for all benchmark models (NPG style),
aligned with figure_accuracy_all_models_bar_npg.pdf style.

Here, positive class = "direction prediction is correct" (sign match);
score = abs(predicted delta) as confidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fig4_palette import apply_fig4_style, model_color

apply_fig4_style()


DATASETS = ["hESC", "hHep", "mHSC-E", "mHSC-GM", "mHSC-L"]
MODELS = ["scCello", "scPRINT", "scGPT", "Geneformer", "LangCell", "scFoundation"]

MODEL_PATHS = {
    "scGPT": "/mnt/10T/yzn/benchmark_GRN/pre_scgpt/results_multidataset_pseudotime_227/{dataset}_gene_result.csv",
    "Geneformer": "/mnt/10T/yzn/benchmark_GRN/pre_geneformer_results_unified/geneformer/{dataset}_gene_result.csv",
    "LangCell": "/mnt/10T/yzn/benchmark_GRN/pre_langcell_results_unified/langcell/{dataset}_gene_result.csv",
    "scFoundation": "/mnt/10T/yzn/benchmark_GRN/pre_scfoundation/scfoundation_multidataset_pseudotime_227/{dataset}/gene_delta_compare.csv",
    "scPRINT": "/mnt/10T/yzn/benchmark_GRN/pre_scprint_results_unified/scprint/per_dataset/{dataset}/per_gene_final_changes.csv",
    "scCello": "/mnt/10T/yzn/benchmark_GRN/pre_sccello_results_unified/sccello/{dataset}_gene_result.csv",
}


def average_precision_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    if y_true.size == 0 or n_pos == 0:
        return float("nan")
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp_cum = np.cumsum(y_sorted == 1)
    rank = np.arange(1, y_sorted.size + 1)
    return float((tp_cum / rank)[y_sorted == 1].sum() / n_pos)


def extract_correctness_and_confidence(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    y_true: 1 if direction is correct, else 0
    y_score: abs(predicted delta) as confidence
    """
    if "dir_correct" in df.columns:
        y_true_all = pd.to_numeric(df["dir_correct"], errors="coerce").to_numpy()
        y_true_all = np.where(np.isfinite(y_true_all), y_true_all, np.nan)
        y_true_all = (y_true_all > 0).astype(np.int32)
    else:
        # Build correctness from delta_true and delta_pred sign match
        if "delta_true" in df.columns:
            delta_true = pd.to_numeric(df["delta_true"], errors="coerce").to_numpy()
        elif "true_delta" in df.columns:
            delta_true = pd.to_numeric(df["true_delta"], errors="coerce").to_numpy()
        else:
            raise ValueError("No true delta column found (delta_true / true_delta).")

        if "delta_pred" in df.columns:
            delta_pred = pd.to_numeric(df["delta_pred"], errors="coerce").to_numpy()
        elif "pred_delta" in df.columns:
            delta_pred = pd.to_numeric(df["pred_delta"], errors="coerce").to_numpy()
        else:
            raise ValueError("No predicted delta column found (delta_pred / pred_delta).")

        finite2 = np.isfinite(delta_true) & np.isfinite(delta_pred)
        true_sign = np.where(delta_true[finite2] > 0, 1, -1)
        pred_sign = np.where(delta_pred[finite2] > 0, 1, -1)
        y_true_all = (true_sign == pred_sign).astype(np.int32)
        # We'll return y_score computed on the same finite2 subset below.
        df = df.loc[finite2].copy()

    if "delta_pred" in df.columns:
        delta_pred_all = pd.to_numeric(df["delta_pred"], errors="coerce").to_numpy()
    elif "pred_delta" in df.columns:
        delta_pred_all = pd.to_numeric(df["pred_delta"], errors="coerce").to_numpy()
    else:
        raise ValueError("No predicted delta column found (delta_pred / pred_delta).")

    y_score_all = np.abs(delta_pred_all.astype(np.float64, copy=False))
    finite = np.isfinite(y_score_all)
    return y_true_all[finite], y_score_all[finite]


def compute_aupr_and_baseline(model: str, dataset: str) -> Tuple[float, float]:
    path = Path(MODEL_PATHS[model].format(dataset=dataset))
    if not path.is_file():
        return float("nan"), float("nan")
    try:
        df = pd.read_csv(path)
        y_true, y_score = extract_correctness_and_confidence(df)
        ap = average_precision_binary(y_true, y_score)
        baseline = float(y_true.mean()) if y_true.size > 0 else float("nan")
        return ap, baseline
    except Exception:
        return float("nan"), float("nan")


def main() -> None:
    n_models = len(MODELS)
    n_datasets = len(DATASETS)

    values = np.full((n_models, n_datasets), np.nan, dtype=np.float64)
    baselines = np.full((n_models, n_datasets), np.nan, dtype=np.float64)
    for mi, model in enumerate(MODELS):
        for di, ds in enumerate(DATASETS):
            ap, bl = compute_aupr_and_baseline(model, ds)
            values[mi, di] = ap
            baselines[mi, di] = bl

    # Unify baseline per dataset (same dashed line for all models in a dataset).
    # For AUPR(correct), the random-score expected AP equals the positive prevalence (correct_rate).
    # Here we use the mean correct_rate across available models as the dataset-level baseline.
    dataset_baseline = np.nanmean(baselines, axis=0)

    fig, ax = plt.subplots(figsize=(10.0, 5.0), dpi=600)
    x = np.arange(n_datasets)
    slot_width = min(0.8 / n_models, 0.13)
    width = slot_width * 0.9
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * slot_width

    for mi, m in enumerate(MODELS):
        ys = values[mi] * 100.0
        ax.bar(
            x + offsets[mi],
            ys,
            width=width,
            label=m,
            color=model_color(m),
            edgecolor="none",
            linewidth=0.0,
            zorder=3,
        )

    # One baseline line per dataset (shared across models)
    bls = dataset_baseline * 100.0
    for di, b in enumerate(bls):
        if not np.isfinite(b):
            continue
        ax.hlines(
            y=b,
            xmin=x[di] - 0.43,
            xmax=x[di] + 0.43,
            colors=model_color("STRING"),
            linestyles="--",
            linewidth=1.2,
            zorder=4,
        )

    ax.set_ylabel("AUPR(correct) (%)", fontsize=16, fontweight="normal")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, rotation=0, ha="center", fontsize=14)
    ax.tick_params(axis="x", labelsize=14, length=0, colors="#000000")
    ax.tick_params(axis="y", labelsize=14, length=0, colors="#000000")
    ax.yaxis.label.set_color("#000000")
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", linestyle="-", alpha=0.3, zorder=1)

    ax.legend(
        frameon=False,
        ncol=max(1, n_models),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        fontsize=14,
        borderaxespad=0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#000000")
    ax.spines["bottom"].set_color("#000000")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    legend = ax.get_legend()
    if legend is not None:
        for t in legend.get_texts():
            t.set_color("#000000")
    plt.subplots_adjust(bottom=0.24)

    out_dir = Path(__file__).resolve().parent / "accuracy"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "figure_aupr_correct_all_models_bar_npg.pdf"
    out_csv = out_dir / "figure_aupr_correct_all_models_bar_npg.csv"

    # save matrix
    table = pd.DataFrame(values, index=MODELS, columns=DATASETS)
    table.to_csv(out_csv, index=True)

    plt.savefig(out_pdf, bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()

