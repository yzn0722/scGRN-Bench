#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2行3列收敛曲线图 + 图例在整张图最底部（完美不重叠）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from fig4_palette import apply_fig4_style, model_color

AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 14
AXIS_LINEWIDTH = 1.2

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "interation"
FIGSIZE_MAIN = (16, 10)
YLIM = (0, 100)
OUT_PDF = SCRIPT_DIR / "convergence_all_6datasets.pdf"
MAX_ITER_TO_PLOT = 11
BALANCED_CURVES_JSON = (
    SCRIPT_DIR / "balanced_convergence_work/balanced_accuracy_curves_all_models.json"
)

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
        raise ValueError(f"Invalid curve for {dataset!r} in {json_path}")
    return arr


def _collect_all_datasets(curve_jsons: Dict[str, Path]) -> list[str]:
    all_sets = set()
    for p in curve_jsons.values():
        with p.open("r", encoding="utf-8") as f:
            d = json.load(f)
        all_sets.update([str(k) for k in d.keys()])
    return sorted(all_sets)


def plot_one_subax(ax, dataset: str, curves: Dict[str, np.ndarray], ylim):
    for name, y in curves.items():
        x = np.arange(1, len(y) + 1, dtype=np.int32)
        keep = x <= MAX_ITER_TO_PLOT
        x = x[keep]
        y = y[keep] * 100
        ax.plot(
            x, y,
            marker="o", linewidth=2.4, markersize=8,
            color=model_color(name), label=name, alpha=0.85
        )

    ax.set_title(dataset, fontsize=16, pad=8)
    ax.set_xlim(1, MAX_ITER_TO_PLOT)
    ax.set_xticks(np.arange(1, MAX_ITER_TO_PLOT+1, 2))
    ax.set_ylim(ylim)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(False)

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["bottom", "left"]:
        ax.spines[side].set_linewidth(AXIS_LINEWIDTH)


def plot_all_in_one_figure(datasets, all_curves_dict, out, figsize, ylim):
    apply_fig4_style()

    # ====================== 核心修复 ======================
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for i, ds in enumerate(datasets):
        plot_one_subax(axes[i], ds, all_curves_dict[ds], ylim)

    # -------------------- 图例放在最底部（完美不重叠） --------------------
    handles, labels = axes[0].get_legend_handles_labels()
    
    # 关键：用 tight 布局 + 外部图例，绝对不会再盖到子图
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)  # 底部留空
    
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),  # 贴最底部
        ncol=6,
        frameon=False,
        fontsize=16,
        handlelength=1.8,
        columnspacing=1.3
    )

    # 统一轴标签
    for ax in axes[3:]:
        ax.set_xlabel("Iteration", fontsize=15)
    for ax in axes[::3]:
        ax.set_ylabel("Balanced Accuracy (%)", fontsize=15)

    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=600)
    plt.close()
    print(f"✅ 大图已保存：{out}")


def main():
    with BALANCED_CURVES_JSON.open("r", encoding="utf-8") as f:
        by_model = json.load(f)
    model_order = ["Geneformer", "LangCell", "scGPT", "scFoundation", "scPRINT", "scCello"]
    datasets = sorted({ds for model in model_order for ds in by_model[model]})
    all_curves = {
        ds: {
            model: np.asarray(by_model[model][ds], dtype=np.float64)
            for model in model_order
        }
        for ds in datasets
    }
    plot_all_in_one_figure(datasets, all_curves, OUT_PDF, FIGSIZE_MAIN, YLIM)

if __name__ == "__main__":
    main()
