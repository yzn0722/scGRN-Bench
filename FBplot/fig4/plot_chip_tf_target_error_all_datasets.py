#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
六数据集：Top30% 动态基因 — CHIP TF 靶 vs 非靶 误差组成（100% 堆叠条）。

读取 plot_dynamic_error_biology.py 产出的 all_datasets_error_fractions.csv，
或现场从 gene_result 重算。

输出
----
  error_biology/all_datasets_chip_tf_target_error_stacked.png
  error_biology/all_datasets_chip_tf_target_error_stacked.pdf
  error_biology/chip_tf_target_direction_error_summary.csv

示例
----
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig4
  python3 plot_chip_tf_target_error_all_datasets.py
  python3 plot_chip_tf_target_error_all_datasets.py --exclude-datasets mDC
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

try:
    from fig4_palette import apply_fig4_style, model_color
except ImportError:

    def apply_fig4_style() -> None:
        plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    def model_color(_: str, d: str = "#666") -> str:
        return d

from plot_dynamic_error_biology import (  # noqa: E402
    DATASET_SPECIES,
    DEFAULT_CHIP_DIR,
    DEFAULT_GENE_DIR,
    DEFAULT_OUTDIR,
    ERROR_COLORS,
    ERROR_LABELS,
    ERROR_ORDER,
    process_dataset,
)

DATASET_ORDER = ["hESC", "hHep", "mDC", "mHSC-E", "mHSC-GM", "mHSC-L"]
STRATA = ["Non-target", "TF target"]


def load_tf_group_summary(
    outdir: Path,
    gene_dir: Path,
    chip_dir: Path,
    mapped_only: bool = False,
    model_label: str = "scGPT",
) -> pd.DataFrame:
    csv_path = outdir / "all_datasets_error_fractions.csv"
    if csv_path.is_file():
        df = pd.read_csv(csv_path)
        sub = df[df["stratum_col"] == "tf_group"].copy()
        if not sub.empty and sub["dataset"].nunique() >= 4:
            return sub

    rows = []
    for gf in sorted(gene_dir.glob("*_gene_result.csv")):
        ds = gf.name.replace("_gene_result.csv", "")
        chip_net = chip_dir / f"{ds}_chip_matched-network.csv"
        if not chip_net.exists():
            continue
        comb, _ = process_dataset(
            ds, gf, chip_net, outdir, do_enrichment=False,
            model_label=model_label, mapped_only=mapped_only,
        )
        rows.append(comb)
    if not rows:
        raise FileNotFoundError("No tf_group data; run plot_dynamic_error_biology.py --all-datasets first")
    full = pd.concat(rows, ignore_index=True)
    full.to_csv(csv_path, index=False)
    return full[full["stratum_col"] == "tf_group"].copy()


def direction_error_summary(sub: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds in DATASET_ORDER:
        for st in STRATA:
            block = sub[(sub["dataset"] == ds) & (sub["stratum"] == st)]
            if block.empty:
                continue
            n = int(block["n_genes"].iloc[0])
            dir_frac = float(block.loc[block["error_type"] == "strict_wrong", "fraction"].iloc[0])
            rows.append(
                {
                    "dataset": ds,
                    "stratum": st,
                    "n_genes": n,
                    "direction_error_frac": dir_frac,
                }
            )
    out = pd.DataFrame(rows)
    wide = out.pivot(index="dataset", columns="stratum", values="direction_error_frac")
    if "Non-target" in wide.columns and "TF target" in wide.columns:
        wide["delta_non_target_minus_tf"] = wide["Non-target"] - wide["TF target"]
    return out, wide.reset_index()


def plot_all_datasets(
    sub: pd.DataFrame,
    datasets: list[str],
    out_png: Path,
    out_pdf: Path,
    model_label: str = "scGPT",
) -> None:
    apply_fig4_style()
    n = len(datasets)
    ncols = 3 if n > 3 else n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.6 * nrows), facecolor="white")
    axes = np.atleast_1d(axes).ravel()

    for ax_idx, ds in enumerate(datasets):
        ax = axes[ax_idx]
        block = sub[sub["dataset"] == ds]
        x = np.arange(len(STRATA))
        bottom = np.zeros(len(STRATA))
        n_map = block.groupby("stratum")["n_genes"].first().to_dict()

        for et in ERROR_ORDER:
            heights = []
            for st in STRATA:
                row = block[(block["stratum"] == st) & (block["error_type"] == et)]
                heights.append(float(row["fraction"].iloc[0]) if len(row) else 0.0)
            heights = np.array(heights)
            ax.bar(
                x,
                heights,
                bottom=bottom,
                width=0.58,
                color=ERROR_COLORS[et],
                edgecolor="white",
                linewidth=0.6,
            )
            bottom += heights

        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(STRATA, fontsize=9, rotation=15, ha="right")
        ax.set_title(ds, fontsize=11, fontweight="600", pad=6)
        if ax_idx % ncols == 0:
            ax.set_ylabel("Fraction of genes", fontsize=10)

        for i, st in enumerate(STRATA):
            ax.text(i, 1.03, f"n={n_map.get(st, 0)}", ha="center", va="bottom", fontsize=8, color="#555")
            dir_row = block[(block["stratum"] == st) & (block["error_type"] == "strict_wrong")]
            if len(dir_row):
                d = float(dir_row["fraction"].iloc[0])
                ax.text(i, 0.5, f"dir err\n{d:.0%}", ha="center", va="center", fontsize=7.5, color="#333", alpha=0.85)

    for j in range(len(datasets), len(axes)):
        axes[j].set_visible(False)

    handles = [mpatches.Patch(facecolor=ERROR_COLORS[e], label=ERROR_LABELS[e]) for e in ERROR_ORDER]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        f"Error composition in top 30% dynamic genes — CHIP TF target vs non-target ({model_label})",
        fontsize=12,
        fontweight="600",
        y=1.02,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.98])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--gene-dir", type=Path, default=DEFAULT_GENE_DIR)
    p.add_argument("--chip-dir", type=Path, default=DEFAULT_CHIP_DIR)
    p.add_argument("--exclude-datasets", default="", help="Comma-separated, e.g. mDC")
    p.add_argument("--model-label", default="scGPT")
    p.add_argument("--mapped-only", action="store_true")
    args = p.parse_args()

    excluded = {x.strip() for x in args.exclude_datasets.split(",") if x.strip()}
    datasets = [d for d in DATASET_ORDER if d not in excluded]

    sub = load_tf_group_summary(
        args.outdir, args.gene_dir, args.chip_dir,
        mapped_only=args.mapped_only, model_label=args.model_label,
    )
    long_sum, wide_sum = direction_error_summary(sub)
    wide_sum.to_csv(args.outdir / "chip_tf_target_direction_error_summary.csv", index=False)
    print(wide_sum.to_string(index=False))

    out_png = args.outdir / (
        "all_datasets_chip_tf_target_error_stacked.png"
        if not excluded
        else "all_datasets_chip_tf_target_error_stacked_no_mDC.png"
    )
    out_pdf = out_png.with_suffix(".pdf")
    plot_all_datasets(sub, datasets, out_png, out_pdf, model_label=args.model_label)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
