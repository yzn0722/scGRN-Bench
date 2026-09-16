#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Six models × 3×3 Spearman heatmaps: per-TF Jaccard agreement across extractions.

Layout: 2 rows × 3 columns (top-left → bottom-right = MODELS order).

Example:
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig3
  python tf_static/plot_tf_extraction_consensus_grid.py --dataset hESC --gt-source STRING
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FIG3 = Path(__file__).resolve().parents[1]
if str(FIG3) not in sys.path:
    sys.path.insert(0, str(FIG3))

from fig3_palette import method_label, model_color  # noqa: E402

from tf_static.model_registry import EXTRACTIONS, GT_DISPLAY, MODELS  # noqa: E402
from tf_static.plot_tf_hub_family import apply_plot_style  # noqa: E402
from tf_static.plot_tf_jaccard_raincloud_all_models import EXTRACT_DISPLAY  # noqa: E402

EXT_ORDER = list(EXTRACTIONS)
EXT_LABELS = [EXTRACT_DISPLAY[e] for e in EXT_ORDER]


def _load_per_tf_long(bench_dir: Path, dataset: str, gt_source: str) -> pd.DataFrame:
    path = bench_dir / f"{dataset}_gt-{gt_source}_per_tf_long.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}\n"
            "Run: python tf_static/plot_tf_benchmark_extended.py --dataset "
            f"{dataset} --gt-source {gt_source}"
        )
    return pd.read_csv(path)


def extraction_corr_matrix(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """3×3 Spearman ρ of per-TF Jaccard across emb500 / att500 / embhidden500."""
    sub = df[(df["model"] == model) & (df["extraction"].isin(EXT_ORDER))].copy()
    piv = sub.pivot_table(index="TF", columns="extraction", values="jaccard", aggfunc="first")
    piv = piv.reindex(columns=EXT_ORDER).dropna(how="any")
    if piv.shape[1] < 2 or len(piv) < 3:
        return pd.DataFrame(index=EXT_ORDER, columns=EXT_ORDER, dtype=float)
    corr = piv.corr(method="spearman")
    corr = corr.reindex(index=EXT_ORDER, columns=EXT_ORDER)
    return corr


def plot_extraction_consensus_grid(
    df: pd.DataFrame,
    dataset: str,
    gt_source: str,
    out_dir: Path,
    models: Optional[List[str]] = None,
) -> Path:
    models = list(models or MODELS)
    gt_name = GT_DISPLAY.get(gt_source, gt_source)

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2), facecolor="white", layout="constrained")
    axes_flat = axes.ravel()

    for i, model in enumerate(models[:6]):
        ax = axes_flat[i]
        corr = extraction_corr_matrix(df, model)
        corr.to_csv(out_dir / f"{dataset}_{model}_extraction_jaccard_corr.csv")

        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.6,
            linecolor="white",
            cbar=i == 2,
            cbar_kws={"label": "Spearman ρ", "shrink": 0.85} if i == 2 else {},
            ax=ax,
            annot_kws={"size": 9},
            xticklabels=EXT_LABELS,
            yticklabels=EXT_LABELS,
        )
        ax.set_title(method_label(model), fontsize=11, fontweight="bold", color=model_color(model), pad=6)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)
        if i == 0:
            ax.text(
                -0.28,
                1.12,
                "A",
                transform=ax.transAxes,
                fontsize=13,
                fontweight="bold",
                va="top",
                ha="left",
            )

    for j in range(len(models), 6):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"{dataset} — per-TF Jaccard agreement across extractions ({gt_name})",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    stem = out_dir / f"{dataset}_six_models_extraction_consensus_2x3"
    fig.savefig(f"{stem}.png", dpi=200, facecolor="white")
    fig.savefig(f"{stem}.pdf", facecolor="white")
    plt.close(fig)
    print(f"Wrote {stem}.png / .pdf")
    return stem


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="2×3 grid: per-model 3×3 extraction Spearman heatmaps")
    p.add_argument("--dataset", default="hESC")
    p.add_argument("--gt-source", default="STRING")
    p.add_argument("--bench-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    apply_plot_style()
    bench_dir = args.bench_dir or (
        FIG3 / "tf_static" / "output" / args.dataset / f"benchmark_extended_{args.gt_source}"
    )
    bench_dir.mkdir(parents=True, exist_ok=True)
    df = _load_per_tf_long(bench_dir, args.dataset, args.gt_source)
    plot_extraction_consensus_grid(df, args.dataset, args.gt_source, bench_dir)


if __name__ == "__main__":
    main()
