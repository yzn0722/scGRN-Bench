#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TF 识别 / 覆盖 — 6 模型 × 3 提取（条形图，版式对齐 hub_binary_bar）。

指标（相对 STRING Gene1 = TF 金标准）：
  1. TF source precision（筛选前）：预测边中 Gene1 属于 GT TF 的比例
  2. GT TF coverage（筛选后 Top-|E|）：GT TF 中被用作调控源的比例
  3. CHIP TF coverage（若有 CHIP 网）：CHIP 实验 TF 在预测中的覆盖比例

示例:
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig3
  python tf_static/plot_tf_identification.py --dataset hESC --gt-source STRING
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

FIG3 = Path(__file__).resolve().parents[1]
if str(FIG3) not in sys.path:
    sys.path.insert(0, str(FIG3))

from fig3_palette import method_label  # noqa: E402

from tf_static.model_registry import (  # noqa: E402
    EVL_ROOT,
    EXTRACTIONS,
    GT_DISPLAY,
    GT_SOURCES,
    INPUT_ROOT,
    MODELS,
    resolve_gt_path,
    resolve_pred_path,
)
from tf_static.plot_tf_jaccard_raincloud_all_models import (  # noqa: E402
    EXTRACT_COLORS,
    EXTRACT_DISPLAY,
    FIG_H,
    FIG_W,
    LABEL_COLOR,
    TEXT_SIZE,
    TITLE_SIZE,
)
from tf_static.utils import detect_weight_col, filter_prediction, load_gt_network, norm_gene  # noqa: E402


def load_prediction_raw(pred_path: Path) -> pd.DataFrame:
    pred = pd.read_csv(pred_path, sep="\t")
    pred["Gene1"] = pred["Gene1"].map(norm_gene)
    pred["Gene2"] = pred["Gene2"].map(norm_gene)
    pred = pred[(pred["Gene1"] != "") & (pred["Gene2"] != "") & (pred["Gene1"] != pred["Gene2"])]
    return pred.drop_duplicates(subset=["Gene1", "Gene2"]).reset_index(drop=True)


def load_chip_tfs(dataset: str, input_root: Path) -> Optional[Set[str]]:
    chip_path = input_root / "CHIP" / f"{dataset}_chip_matched-network.csv"
    if not chip_path.is_file():
        return None
    chip = pd.read_csv(chip_path)
    return set(chip["Gene1"].map(norm_gene).unique())


def compute_tf_identification_row(
    model: str,
    extraction: str,
    pred_path: Path,
    gt_tfs: Set[str],
    gt_all: Set[str],
    gt_n: int,
    chip_tfs: Optional[Set[str]],
) -> dict:
    raw = load_prediction_raw(pred_path)
    n_raw_edges = len(raw)
    n_raw_gene1 = raw["Gene1"].nunique()
    prec_edges = float((raw["Gene1"].isin(gt_tfs)).mean()) if n_raw_edges else np.nan
    prec_gene1 = float(raw["Gene1"].isin(gt_tfs).mean()) if n_raw_gene1 else np.nan

    filt = filter_prediction(pred_path, gt_tfs, gt_all, gt_n)
    pred_tfs = set(filt["Gene1"].unique())
    n_gt = len(gt_tfs)
    coverage_gt = len(pred_tfs & gt_tfs) / n_gt if n_gt else np.nan

    row = {
        "model": model,
        "extraction": extraction,
        "n_raw_edges": n_raw_edges,
        "n_raw_gene1": n_raw_gene1,
        "tf_source_precision_edges": prec_edges,
        "tf_source_precision_gene1": prec_gene1,
        "n_gt_tf": n_gt,
        "n_pred_tf_after_filter": len(pred_tfs),
        "gt_tf_coverage": coverage_gt,
        "n_missing_gt_tf": len(gt_tfs - pred_tfs),
    }
    if chip_tfs:
        n_chip = len(chip_tfs)
        covered = len(pred_tfs & chip_tfs)
        row["n_chip_tf"] = n_chip
        row["chip_tf_coverage"] = covered / n_chip if n_chip else np.nan
        row["n_chip_tf_covered"] = covered
    else:
        row["n_chip_tf"] = np.nan
        row["chip_tf_coverage"] = np.nan
        row["n_chip_tf_covered"] = np.nan
    return row


def collect_tf_identification(
    dataset: str,
    gt_source: str,
    models: List[str],
    extractions: List[str],
    evl_root: Path,
    input_root: Path,
) -> pd.DataFrame:
    gt_path = resolve_gt_path(gt_source, dataset, input_root)
    gt_tfs, gt_all, gt_n, _, _ = load_gt_network(gt_path)
    chip_tfs = load_chip_tfs(dataset, input_root)

    rows = []
    for model in models:
        for ext in extractions:
            fp = resolve_pred_path(model, ext, dataset, evl_root)
            if fp is None:
                print(f"  [skip] {model}|{ext}")
                continue
            rows.append(
                compute_tf_identification_row(model, ext, fp, gt_tfs, gt_all, gt_n, chip_tfs)
            )
            r = rows[-1]
            print(
                f"  {model}|{ext}: source_prec={r['tf_source_precision_edges']:.2f}, "
                f"GT_cov={r['gt_tf_coverage']:.2f}"
                + (f", CHIP_cov={r['chip_tf_coverage']:.2f}" if chip_tfs else "")
            )
    return pd.DataFrame(rows)


def plot_tf_identification_bars(
    summ: pd.DataFrame,
    dataset: str,
    gt_source: str,
    out_dir: Path,
    has_chip: bool,
) -> None:
    """Grouped bars: 6 models, 3 extractions per metric (same layout as hub_binary_bar)."""
    models = list(dict.fromkeys(summ["model"].tolist()))
    metrics = [
        ("tf_source_precision_edges", "TF source precision\n(before filter, edge-level)"),
        ("gt_tf_coverage", "GT TF coverage\n(after Top-|E| filter)"),
    ]
    if has_chip and summ["chip_tf_coverage"].notna().any():
        metrics.append(("chip_tf_coverage", "CHIP TF coverage\n(after Top-|E| filter)"))

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 5.2))
    if len(metrics) == 1:
        axes = [axes]

    x = np.arange(len(models))
    w = 0.22
    offsets = {"emb500": -w, "att500": 0.0, "embhidden500": w}

    for ax, (col, ylab) in zip(axes, metrics):
        for ext in EXTRACTIONS:
            vals = []
            for model in models:
                row = summ[(summ["model"] == model) & (summ["extraction"] == ext)]
                vals.append(float(row[col].iloc[0]) if len(row) else 0.0)
            ax.bar(
                x + offsets[ext],
                vals,
                width=w * 0.88,
                color=EXTRACT_COLORS[ext],
                edgecolor="#444",
                linewidth=0.5,
                label=EXTRACT_DISPLAY[ext],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=TEXT_SIZE, color=LABEL_COLOR)
        ax.set_ylabel(ylab, fontsize=TEXT_SIZE, color=LABEL_COLOR)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].legend(loc="lower right", frameon=False, fontsize=TEXT_SIZE - 1, title="Extraction")
    gt_name = GT_DISPLAY.get(gt_source, gt_source)
    fig.suptitle(
        f"{dataset} — TF identification vs {gt_name}\n"
        f"(Gene1 must be GT TF after benchmark filter; coverage = fraction of reference TFs used as source)",
        fontsize=TITLE_SIZE,
        y=1.02,
    )
    fig.tight_layout()
    stem = out_dir / f"{dataset}_gt-{gt_source}_tf_identification_bar"
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {stem}.png/.pdf")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TF source precision and coverage bars")
    p.add_argument("--dataset", default="hESC")
    p.add_argument("--gt-source", default="STRING", choices=list(GT_SOURCES))
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--extractions", nargs="+", default=list(EXTRACTIONS))
    p.add_argument("--evl-root", type=Path, default=EVL_ROOT)
    p.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    plt.rcParams.update(
        {
            "font.size": TEXT_SIZE,
            "axes.labelsize": TEXT_SIZE,
            "xtick.labelsize": TEXT_SIZE,
            "ytick.labelsize": TEXT_SIZE,
            "legend.fontsize": TEXT_SIZE,
            "font.family": "DejaVu Sans",
        }
    )

    out_dir = args.out or (FIG3 / "tf_static" / "output" / args.dataset / f"hub_family_{args.gt_source}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset}  GT: {args.gt_source}")
    summ = collect_tf_identification(
        args.dataset,
        args.gt_source,
        list(args.models),
        list(args.extractions),
        args.evl_root,
        args.input_root,
    )
    if summ.empty:
        raise RuntimeError("No results")
    summ.to_csv(out_dir / f"{args.dataset}_gt-{args.gt_source}_tf_identification_summary.csv", index=False)

    chip_tfs = load_chip_tfs(args.dataset, args.input_root)
    plot_tf_identification_bars(summ, args.dataset, args.gt_source, out_dir, chip_tfs is not None)
    print("Done.")


if __name__ == "__main__":
    main()
