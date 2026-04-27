#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot iterative direction-accuracy convergence curves for ONE dataset, overlaying multiple models.

Expected input JSON format (one per model):
  {
    "hESC": [0.68, 0.77, ...],
    "hHep": [...],
    ...
  }

This matches the files in this folder, e.g.:
  - scfoundation_accuracy_curves.json
  - scgpt_accuracy_curves.json
  - scprint_accuracy_curves.json
  - sccello_accuracy_curves.json
  - Geneformer_accuracy_curves.json

Example:
  python3 /mnt/10T/yzn/FoundBench/FBplot/fig4/plot_iter_convergence_single_dataset.py \\
    --dataset hESC \\
    --curve scFoundation=/mnt/10T/yzn/FoundBench/FBplot/fig4/scfoundation_accuracy_curves.json \\
    --curve scGPT=/mnt/10T/yzn/FoundBench/FBplot/fig4/scgpt_accuracy_curves.json \\
    --curve scPrint=/mnt/10T/yzn/FoundBench/FBplot/fig4/scprint_accuracy_curves.json \\
    --curve scCello=/mnt/10T/yzn/FoundBench/FBplot/fig4/sccello_accuracy_curves.json \\
    --curve Geneformer=/mnt/10T/yzn/FoundBench/FBplot/fig4/Geneformer_accuracy_curves.json \\
    --out /mnt/10T/yzn/FoundBench/FBplot/fig4/convergence_models_hESC.pdf
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from fig4_palette import apply_fig4_style, model_color, FIG4_FIGSIZE

AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 14
AXIS_LINEWIDTH = 1.2

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "interation"
DATASET = "hESC"  # 单数据集名，或 "all" 批量绘制全部数据集
FIGSIZE = (5.0, 5.0)
YLIM: Tuple[float, float] | None = None
LEGEND_NCOL = 2
OUT_PNG = SCRIPT_DIR / "convergence" / f"convergence_models_{DATASET}.pdf"
MAX_ITER_TO_PLOT = 11

# 路径写死在代码中，直接运行即可
CURVE_JSONS: Dict[str, Path] = {
    "Geneformer": DATA_DIR / "geneformer_accuracy_curves_6datasets.json",
    "LangCell": DATA_DIR / "Langcell_accuracy_curves.json",
    "scGPT": DATA_DIR / "scgpt_accuracy_curves.json",
    "scFoundation": DATA_DIR / "scfoundation_accuracy_curves.json",
    "scPRINT": DATA_DIR / "scprint_accuracy_curves.json",
    "scCello": DATA_DIR / "sccello_accuracy_curves.json",
}


def _load_curve_for_dataset(json_path: Path, dataset: str) -> np.ndarray:
    with json_path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    if dataset not in d:
        keys = sorted([str(k) for k in d.keys()])
        raise KeyError(f"{json_path} has no dataset={dataset!r}. Available: {keys}")
    arr = np.asarray(d[dataset], dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"Invalid curve for {dataset!r} in {json_path}: expected 1D non-empty list")
    return arr


def _collect_all_datasets(curve_jsons: Dict[str, Path]) -> list[str]:
    all_sets = set()
    for p in curve_jsons.values():
        with p.open("r", encoding="utf-8") as f:
            d = json.load(f)
        all_sets.update([str(k) for k in d.keys()])
    return sorted(all_sets)


def plot_curves(
    *,
    dataset: str,
    curves: Dict[str, np.ndarray],
    out: Path,
    figsize: Tuple[float, float],
    ylim: Tuple[float, float] | None,
    legend_ncol: int,
) -> None:
    import matplotlib.pyplot as plt
    apply_fig4_style()

    fig, ax = plt.subplots(figsize=figsize)

    max_t = 0
    for name, y in curves.items():
        x = np.arange(1, len(y) + 1, dtype=np.int32)
        # plot only first MAX_ITER_TO_PLOT iterations
        keep = x <= int(MAX_ITER_TO_PLOT)
        x = x[keep]
        y = (y[keep] * 100.0)  # 统一使用0-100尺度
        max_t = max(max_t, int(x.max()))
        ax.plot(
            x,
            y,
            marker="o",
            linestyle="-",
            linewidth=2.4,
            markersize=8,
            alpha=0.85,
            color=model_color(name),
            label=name,
        )

    ax.set_xlabel("Iteration", fontsize=16)
    ax.set_ylabel("Direction accuracy (Top-30%)", fontsize=16)
    #ax.set_title(dataset, fontsize=16, color="#000000", pad=10)

    ax.set_xlim(1, int(MAX_ITER_TO_PLOT))
    # show only odd-number ticks up to 11
    ax.set_xticks(np.arange(1, int(MAX_ITER_TO_PLOT) + 1, 2))
    if ylim is None:
        ax.set_ylim(45.0, 100.0)
    else:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    ax.set_yticks(np.arange(50, 101, 10))

    ax.grid(False)

    # Keep only left/bottom spines to match fig4 style
    for side in ["bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(AXIS_LINEWIDTH)
        ax.spines[side].set_color("black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # 坐标轴刻度统一为纯数字（不加百分号）
    ax.tick_params(width=AXIS_LINEWIDTH, length=0, labelsize=TICK_LABEL_SIZE, pad=2)

    # legend
    ax.legend(
        loc="lower right",
        frameon=False,
        ncol=max(1, int(legend_ncol)),
        fontsize=14,
        handlelength=1.6,
        handletextpad=0.5,
        labelspacing=0.3,
        borderaxespad=0.3,
        columnspacing=0.8,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", dpi=600)
    plt.close(fig)


def main() -> None:
    dataset = str(DATASET).strip()
    for name, p in CURVE_JSONS.items():
        if not p.is_file():
            raise FileNotFoundError(f"[Not found] {name}: {p}")

    datasets = _collect_all_datasets(CURVE_JSONS) if dataset.lower() == "all" else [dataset]

    for ds in datasets:
        curves: Dict[str, np.ndarray] = {}
        for name, p in CURVE_JSONS.items():
            curves[name] = _load_curve_for_dataset(p, ds)

        out = OUT_PNG.parent / f"convergence_models_{ds}.pdf"
        plot_curves(
            dataset=ds,
            curves=curves,
            out=out,
            figsize=FIGSIZE,
            ylim=YLIM,
            legend_ncol=LEGEND_NCOL,
        )
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()

