#!/usr/bin/env python3
"""Plot class-imbalance-adjusted accuracy gain for Fig. 4.

Gain is balanced_accuracy - 0.5, reported in percentage points.  Balanced
accuracy averages Up and Down recall, so unequal class prevalence does not
inflate the score.  Zero is the chance-level reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fig4_palette import apply_fig4_style, model_color


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Balanced-accuracy gain bar chart")
    p.add_argument(
        "--input",
        default=str(here / "accuracy" / "figure_balanced_accuracy_all_models_bar_npg_top30.csv"),
    )
    p.add_argument(
        "--out",
        default=str(here / "accuracy" / "figure_balanced_accuracy_gain_all_models_bar_npg.pdf"),
    )
    p.add_argument("--csv-out", default="")
    p.add_argument("--include-mdc", action="store_true")
    p.add_argument("--figwidth", type=float, default=10.0)
    p.add_argument("--figheight", type=float, default=5.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    apply_fig4_style()
    src = Path(args.input)
    table = pd.read_csv(src, index_col=0)
    table.index = table.index.astype(str)
    table = table.apply(pd.to_numeric, errors="coerce")
    if not args.include_mdc:
        table = table.drop(columns=["mDC"], errors="ignore")
    if table.empty:
        raise ValueError(f"No balanced-accuracy values found in {src}")

    gain_pp = (table - 0.5) * 100.0
    long = (
        table.rename_axis("model")
        .reset_index()
        .melt(id_vars="model", var_name="dataset", value_name="balanced_accuracy")
    )
    long["balanced_accuracy_gain_pp"] = (long["balanced_accuracy"] - 0.5) * 100.0
    csv_out = Path(args.csv_out) if args.csv_out else Path(args.out).with_suffix(".csv")
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(csv_out, index=False)

    models = list(gain_pp.index)
    datasets = list(gain_pp.columns)
    x = np.arange(len(datasets))
    slot = min(0.8 / max(len(models), 1), 0.13)
    width = slot * 0.9
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2.0) * slot

    fig, ax = plt.subplots(figsize=(args.figwidth, args.figheight), dpi=600)
    for i, model in enumerate(models):
        ax.bar(
            x + offsets[i], gain_pp.loc[model].to_numpy(dtype=float),
            width=width, label=model, color=model_color(model),
            edgecolor="none", linewidth=0.0, zorder=3,
        )
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1.0, zorder=2)
    ax.set_ylabel("Balanced accuracy gain (pp)", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=14)
    ax.tick_params(axis="both", labelsize=14, length=0)
    finite = gain_pp.to_numpy(dtype=float)
    lo = min(-5.0, float(np.nanmin(finite)) - 5.0)
    hi = max(10.0, float(np.nanmax(finite)) + 5.0)
    ax.set_ylim(np.floor(lo / 10.0) * 10.0, np.ceil(hi / 10.0) * 10.0)
    ax.grid(False)
    ax.legend(
        frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.06),
        ncol=100, fontsize=14, handlelength=1.2, handletextpad=0.3,
        columnspacing=0.4, borderaxespad=0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    plt.subplots_adjust(bottom=0.1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"Saved PDF: {out.resolve()}")
    print(f"Saved values: {csv_out.resolve()}")
    print("\nBalanced accuracy gain (percentage points):")
    print(gain_pp.round(2).to_string())


if __name__ == "__main__":
    main()
