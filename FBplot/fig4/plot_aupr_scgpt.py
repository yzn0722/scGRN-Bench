#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute and plot scGPT AUPR across datasets from *_gene_result.csv files.

Definition used here:
- Binary task: Up vs Down (rows with dir_true in {"Up","Down"} only)
- Positive class: Up
- Score: delta_pred (higher => more likely Up)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fig4_palette import apply_fig4_style, model_color


DATA_DIR = Path("/mnt/10T/yzn/benchmark_GRN/pre_scgpt/results_multidataset_pseudotime_227")
OUT_DIR = Path(__file__).resolve().parent / "accuracy"
OUT_PDF = OUT_DIR / "scgpt_aupr_bar.pdf"
OUT_CSV = OUT_DIR / "scgpt_aupr_by_dataset.csv"


def average_precision_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision for binary labels (1=positive, 0=negative)."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.size == 0:
        return float("nan")
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp_cum = np.cumsum(y_sorted == 1)
    rank = np.arange(1, y_sorted.size + 1)
    precision_at_k = tp_cum / rank
    ap = float(precision_at_k[y_sorted == 1].sum() / n_pos)
    return ap


def compute_aupr_for_file(csv_path: Path) -> Tuple[str, float, float, int]:
    df = pd.read_csv(csv_path)
    needed = {"dir_true", "delta_pred"}
    if not needed.issubset(df.columns):
        return csv_path.stem.replace("_gene_result", ""), float("nan"), 0

    d = df[df["dir_true"].isin(["Up", "Down"])].copy()
    d["delta_pred"] = pd.to_numeric(d["delta_pred"], errors="coerce")
    d = d[np.isfinite(d["delta_pred"])]
    if d.empty:
        return csv_path.stem.replace("_gene_result", ""), float("nan"), float("nan"), 0

    y_true = (d["dir_true"] == "Up").astype(np.int32).to_numpy()
    y_score = d["delta_pred"].to_numpy(dtype=np.float64)
    ap = average_precision_binary(y_true, y_score)
    baseline = float(y_true.mean())  # AUPR random baseline = positive class prevalence
    dataset = csv_path.stem.replace("_gene_result", "")
    return dataset, ap, baseline, int(d.shape[0])


def main() -> None:
    apply_fig4_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_files: List[Path] = sorted(DATA_DIR.glob("*_gene_result.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No *_gene_result.csv found in {DATA_DIR}")

    rows = []
    for fp in csv_files:
        ds, ap, baseline, n = compute_aupr_for_file(fp)
        rows.append(
            {
                "Dataset": ds,
                "AUPR_Up_vs_Down": ap,
                "Baseline_PosRate": baseline,
                "N_Used": n,
                "Path": str(fp),
            }
        )

    res = pd.DataFrame(rows).sort_values("Dataset").reset_index(drop=True)
    res.to_csv(OUT_CSV, index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(5, 5), dpi=600)
    x = np.arange(len(res))
    vals = res["AUPR_Up_vs_Down"].to_numpy(dtype=float)
    baselines = res["Baseline_PosRate"].to_numpy(dtype=float)
    ax.bar(x, vals, color=model_color("scGPT"), edgecolor="none", linewidth=0)
    # per-dataset dashed baseline (random classifier AP)
    for xi, b in zip(x, baselines):
        if np.isfinite(b):
            ax.hlines(
                y=b,
                xmin=xi - 0.4,
                xmax=xi + 0.4,
                colors=model_color("STRING"),
                linestyles="--",
                linewidth=1.6,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(res["Dataset"].tolist(), rotation=25, ha="right")
    ax.set_ylabel("AUPR (Up vs Down)", fontsize=16)
    ax.tick_params(axis="x", labelsize=14, length=0)
    ax.tick_params(axis="y", labelsize=14, length=0)
    ax.set_ylim(0, 1.0)
    ax.set_title("scGPT AUPR by Dataset", fontsize=16)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved AUPR table: {OUT_CSV}")
    print(f"Saved AUPR plot : {OUT_PDF}")


if __name__ == "__main__":
    main()

