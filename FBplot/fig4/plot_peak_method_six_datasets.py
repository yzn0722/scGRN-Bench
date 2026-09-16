#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
六数据集：单步方向错 vs 峰值段链式纠错 — 一张总览图。

输出
----
- error_biology/peak_method_six_datasets_metrics.csv
- error_biology/peak_method_six_datasets.png

若缺链式结果，加 --run-chained（GPU，每数据集 ~数分钟）：
  python3 plot_peak_method_six_datasets.py --run-chained
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from peak_method_metrics import DATASETS, MULTISTEP_ROOT, eval_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "error_biology"

try:
    from fig4_palette import apply_fig4_style, model_color
except ImportError:

    def apply_fig4_style() -> None:
        plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    def model_color(_: str, d: str = "#666") -> str:
        return d


def run_chained_all(datasets: list[str], gen_iters: int = 16) -> None:
    py = sys.executable
    for ds in datasets:
        print(f"[run chained] {ds}")
        subprocess.run(
            [
                py,
                str(SCRIPT_DIR / "run_scgpt_chained_segments.py"),
                "--dataset",
                ds,
                "--gen-iters",
                str(gen_iters),
            ],
            check=True,
        )


def plot_six(metrics: pd.DataFrame, outpath: Path) -> None:
    apply_fig4_style()
    m = metrics[metrics["status"] == "ok"].copy()
    if m.empty:
        raise RuntimeError("No dataset with chained predictions; run with --run-chained")

    order = [d for d in DATASETS if d in m["dataset"].values]
    m = m.set_index("dataset").loc[order].reset_index()
    x = np.arange(len(m))
    w = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    color_single = "#B0BEC5"
    color_fixed = model_color("scGPT")
    color_still = model_color("scPrint")

    # 左轴：top30% 中单步方向错误率
    err_pct = m["frac_wrong_single"].values * 100
    bars0 = ax.bar(x - w / 2, err_pct, w, color=color_single, edgecolor="white", linewidth=0.8,
                   label="Single-step direction error (top 30% dynamic)")

    # 右轴：在「单步错」基因中，峰值链式纠错占比（堆叠 100%）
    ax2 = ax.twinx()
    fix_pct = m["frac_fixed_among_wrong"].fillna(0).values * 100
    still_pct = 100 - fix_pct
    bars_fix = ax2.bar(x + w / 2, fix_pct, w, color=color_fixed, edgecolor="white", linewidth=0.8,
                       label="Fixed at peak transition (chained)")
    ax2.bar(x + w / 2, still_pct, w, bottom=fix_pct, color=color_still, alpha=0.55,
            edgecolor="white", linewidth=0.8, label="Still wrong after peak-chained")

    ax.set_xticks(x)
    ax.set_xticklabels(m["dataset"], fontsize=11)
    ax.set_ylabel("Direction error rate in top 30% genes (%)", fontsize=11)
    ax2.set_ylabel("Among single-step wrong genes (%)", fontsize=11)
    ax.set_ylim(0, max(55, err_pct.max() * 1.15))
    ax2.set_ylim(0, 100)

    for i, row in m.iterrows():
        xi = x[i]
        ax.text(xi - w / 2, err_pct[i] + 1.2, f"{err_pct[i]:.0f}%", ha="center", va="bottom", fontsize=8, color="#444")
        ax2.text(xi + w / 2, fix_pct[i] / 2, f"{int(row['n_fixed_peak_chained'])}/{int(row['n_wrong_single'])}",
                 ha="center", va="center", fontsize=8, color="white", fontweight="600")
        if fix_pct[i] > 12:
            ax2.text(xi + w / 2, fix_pct[i] + 2, f"{fix_pct[i]:.0f}% fixed", ha="center", va="bottom", fontsize=7,
                     color=color_fixed)

    # 缺失链式
    miss = metrics[metrics["status"] != "ok"]
    if len(miss):
        note = "No chained run: " + ", ".join(miss["dataset"].astype(str))
        ax.text(0.5, 1.02, note, transform=ax.transAxes, ha="center", fontsize=9, color="#c0392b")

    ax.set_title(
        "Peak-transition chained scGPT rescues single-step direction errors",
        fontsize=13,
        fontweight="600",
        pad=14,
    )

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=9, ncol=2)

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_stacked_only(metrics: pd.DataFrame, outpath: Path) -> None:
    """备选：仅展示「单步错基因」中修复比例堆叠条（更直观）。"""
    apply_fig4_style()
    m = metrics[metrics["status"] == "ok"].copy()
    order = [d for d in DATASETS if d in m["dataset"].values]
    m = m.set_index("dataset").loc[order].reset_index()

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(m))
    fix = m["frac_fixed_among_wrong"].values * 100
    still = 100 - fix

    ax.bar(x, fix, color=model_color("scGPT"), edgecolor="white", label="Fixed (peak + chained)")
    ax.bar(x, still, bottom=fix, color="#E57373", alpha=0.85, edgecolor="white", label="Still wrong")

    for i, row in m.iterrows():
        ax.text(i, fix[i] / 2, f"{int(row['n_fixed_peak_chained'])}/{int(row['n_wrong_single'])}",
                ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        ax.text(i, 102, f"n={int(row['n_wrong_single'])} wrong", ha="center", fontsize=8, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(m["dataset"])
    ax.set_ylabel("Fraction of single-step wrong genes (%)")
    ax.set_ylim(0, 108)
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("Rescue rate among direction-wrong genes (peak-transition chained scGPT)", fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-chained", action="store_true", help="Run chained scGPT for datasets missing outputs")
    p.add_argument("--gen-iters", type=int, default=16)
    p.add_argument("--style", choices=["dual", "stacked", "both"], default="both")
    args = p.parse_args()

    if args.run_chained:
        run_chained_all(DATASETS, args.gen_iters)

    rows = [eval_dataset(ds) for ds in DATASETS]
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "peak_method_six_datasets_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved: {csv_path}")

    rows_no_ercc = [eval_dataset(ds, exclude_ercc=True) for ds in DATASETS]
    df_no_ercc = pd.DataFrame(rows_no_ercc)
    csv_no_ercc = OUT_DIR / "peak_method_six_datasets_metrics_no_ercc.csv"
    df_no_ercc.to_csv(csv_no_ercc, index=False)
    print("\n--- exclude ERCC ---")
    print(df_no_ercc.to_string(index=False))
    print(f"\nSaved: {csv_no_ercc}")

    ok = df[df["status"] == "ok"]
    if ok.empty:
        print("No chained results. Re-run with: python3 plot_peak_method_six_datasets.py --run-chained")
        return

    if args.style in ("dual", "both"):
        plot_six(ok, OUT_DIR / "peak_method_six_datasets.png")
        print(f"Saved: {OUT_DIR / 'peak_method_six_datasets.png'}")
    if args.style in ("stacked", "both"):
        plot_stacked_only(ok, OUT_DIR / "peak_method_six_datasets_stacked.png")
        print(f"Saved: {OUT_DIR / 'peak_method_six_datasets_stacked.png'}")


if __name__ == "__main__":
    main()
