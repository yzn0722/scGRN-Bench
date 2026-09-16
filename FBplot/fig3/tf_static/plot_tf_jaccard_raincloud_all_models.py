#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全模型 × 三提取方式 — per-TF Jaccard 云雨大图。

横轴：6 个模型名称；每组内并排 3 种提取（仅图例标注，不写 x 刻度）。
纵轴：每个 TF 相对 STRING 的靶基因集 Jaccard。图幅 12×6。

示例:
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig3
  python tf_static/plot_tf_jaccard_raincloud_all_models.py --dataset hESC
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

FIG3 = Path(__file__).resolve().parents[1]
if str(FIG3) not in sys.path:
    sys.path.insert(0, str(FIG3))

from fig3_palette import method_color, method_label, model_color  # noqa: E402

from tf_static.model_registry import (  # noqa: E402
    EVL_ROOT,
    EXTRACTIONS,
    GT_DISPLAY,
    GT_SOURCES,
    MODELS,
    resolve_gt_path,
    resolve_pred_path,
)
from tf_static.plot_tf_static_three import apply_style  # noqa: E402
from tf_static.utils import filter_prediction, per_tf_edge_sets  # noqa: E402

TEXT_SIZE = 16
TITLE_SIZE = 16
LABEL_COLOR = "#000000"
RNG = np.random.default_rng(42)

# 与 fig2/2.3 一致：三种提取方式用固定色
EXTRACT_COLORS = {
    "emb500": model_color("scGPT"),
    "att500": model_color("LangCell"),
    "embhidden500": model_color("scFoundation"),
}
EXTRACT_DISPLAY = {
    "emb500": method_label("emb500"),
    "att500": method_label("att500"),
    "embhidden500": method_label("embhidden500"),
}

FIG_W, FIG_H = 12.0, 6.0
VIOLIN_WIDTH = 0.22
EXTRACT_OFFSET = {"emb500": -0.26, "att500": 0.0, "embhidden500": 0.26}


def column_key(model: str, extraction: str) -> str:
    return f"{model}|{extraction}"


def x_position(model_idx: int, extraction: str) -> float:
    """模型 i 居中于 x=i，三种提取在组内小幅偏移。"""
    return float(model_idx) + EXTRACT_OFFSET[extraction]


def collect_jaccard_long(
    dataset: str,
    models: List[str],
    extractions: List[str],
    evl_root: Path,
    gt_gene1,
    gt_all,
    gt_n,
    gt_by_tf,
) -> pd.DataFrame:
    rows = []
    for mi, model in enumerate(models):
        for ext in extractions:
            fp = resolve_pred_path(model, ext, dataset, evl_root)
            label = column_key(model, ext)
            if fp is None:
                print(f"  [skip] missing {label}")
                continue
            pred = filter_prediction(fp, gt_gene1, gt_all, gt_n)
            met = per_tf_edge_sets(pred, gt_by_tf)
            if met.empty:
                continue
            for _, r in met.iterrows():
                rows.append(
                    {
                        "TF": r["TF"],
                        "jaccard": float(r["jaccard"]),
                        "model": model,
                        "extraction": ext,
                        "column": label,
                        "model_idx": mi,
                        "x_pos": x_position(mi, ext),
                    }
                )
            print(f"  {label}: n_tf={len(met)}, median J={met['jaccard'].median():.3f}")
    return pd.DataFrame(rows)


def plot_raincloud_all_models(
    df: pd.DataFrame,
    dataset: str,
    models: List[str],
    extractions: List[str],
    out_path: Path,
    gt_label: str = "STRING",
    y_max: float = 1.02,
    fig_w: float = FIG_W,
    fig_h: float = FIG_H,
) -> None:
    if df.empty:
        raise ValueError("No Jaccard data to plot")

    n_models = len(models)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.tick_params(axis="both", labelsize=TEXT_SIZE, colors=LABEL_COLOR)

    for mi, model in enumerate(models):
        for ext in extractions:
            col = column_key(model, ext)
            sub = df[df["column"] == col]["jaccard"].dropna().to_numpy()
            if len(sub) == 0:
                continue
            xp = x_position(mi, ext)
            color = EXTRACT_COLORS[ext]

            parts = ax.violinplot(
                sub,
                positions=[xp],
                widths=VIOLIN_WIDTH,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_edgecolor("#555555")
                pc.set_alpha(0.40)
                pc.set_linewidth(0.5)

            jitter = RNG.normal(xp, 0.045, size=len(sub))
            ax.scatter(
                jitter,
                sub,
                s=10,
                alpha=0.30,
                color=color,
                edgecolors="none",
                zorder=2,
                rasterized=True,
            )

            q1, med, q3 = np.percentile(sub, [25, 50, 75])
            ax.vlines(xp, q1, q3, color="#333333", lw=1.6, zorder=4)
            ax.scatter([xp], [med], s=28, color="white", edgecolors="#222222", linewidths=0.8, zorder=5)

    # 模型组间分隔
    for mi in range(1, n_models):
        ax.axvline(mi - 0.5, color="#E0E0E0", lw=0.9, zorder=0)

    # 横轴：仅模型名（黑色，16pt）
    ax.set_xticks(np.arange(n_models))
    ax.set_xticklabels(models, fontsize=TEXT_SIZE, rotation=0, ha="center", color=LABEL_COLOR)
    for tick in ax.get_xticklabels():
        tick.set_color(LABEL_COLOR)
        tick.set_fontweight("normal")

    ax.set_ylabel(
        f"Per-TF target-set Jaccard\n(vs {GT_DISPLAY.get(gt_label, gt_label)})",
        fontsize=TEXT_SIZE,
        color=LABEL_COLOR,
    )
    ax.yaxis.label.set_color(LABEL_COLOR)
    ax.set_ylim(-0.02, y_max)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    # ax.set_title(
    #     f"{dataset}",
    #     fontsize=TITLE_SIZE,
    #     pad=12,
    # )

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=EXTRACT_COLORS[e],
            markersize=9,
            label=EXTRACT_DISPLAY[e],
        )
        for e in extractions
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        fontsize=TEXT_SIZE,
        # title="Edge extraction",
        title_fontsize=TEXT_SIZE,
    )
    for t in leg.get_texts():
        t.set_color(LABEL_COLOR)
    leg.get_title().set_color(LABEL_COLOR)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_xlim(-0.55, n_models - 0.45)
    ax.margins(x=0.02)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.14)
    fig.savefig(f"{out_path}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out_path}.png/.pdf")


def run_one_gt(
    args: argparse.Namespace,
    gt_source: str,
) -> None:
    from tf_static.utils import load_gt_network

    gt_path = resolve_gt_path(gt_source, args.dataset, args.input_root)
    gt_gene1, gt_all, gt_n, _, gt_by_tf = load_gt_network(gt_path)

    if args.out is not None:
        out_dir = args.out
    else:
        out_dir = FIG3 / "tf_static" / "output" / args.dataset / f"gt_{gt_source}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== GT: {gt_source} ({gt_path.name}, |E|={gt_n}) ===")
    df = collect_jaccard_long(
        args.dataset,
        list(args.models),
        list(args.extractions),
        args.evl_root,
        gt_gene1,
        gt_all,
        gt_n,
        gt_by_tf,
    )
    if df.empty:
        print(f"  [warn] no data for GT {gt_source}")
        return

    tag = f"{args.dataset}_gt-{gt_source}"
    df.to_csv(out_dir / f"{tag}_all_models_tf_jaccard_long.csv", index=False)

    stem = out_dir / f"{tag}_all_models_tf_jaccard_raincloud"
    plot_raincloud_all_models(
        df,
        args.dataset,
        list(args.models),
        list(args.extractions),
        stem,
        gt_label=gt_source,
        y_max=args.y_max,
        fig_w=args.fig_w,
        fig_h=args.fig_h,
    )

    summ = (
        df.groupby(["model", "extraction"], as_index=False)["jaccard"]
        .agg(n_tf="count", mean_jaccard="mean", median_jaccard="median")
        .sort_values(["model", "extraction"])
    )
    summ.to_csv(out_dir / f"{tag}_all_models_jaccard_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Raincloud of per-TF Jaccard: all models × 3 extractions")
    p.add_argument("--dataset", default="hESC")
    p.add_argument(
        "--gt-source",
        default="STRING",
        choices=list(GT_SOURCES),
        help="Ground-truth network type",
    )
    p.add_argument(
        "--all-gt",
        action="store_true",
        help="Run all GT sources (STRING, omnipath, Non_CHIP, CHIP)",
    )
    p.add_argument("--input-root", type=Path, default=Path("/mnt/10T/yzn/benchmark_GRN/input_process"))
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--extractions", nargs="+", default=list(EXTRACTIONS))
    p.add_argument("--evl-root", type=Path, default=EVL_ROOT)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--y-max", type=float, default=1.02)
    p.add_argument("--fig-w", type=float, default=FIG_W)
    p.add_argument("--fig-h", type=float, default=FIG_H)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    apply_style()
    plt.rcParams.update(
        {
            "font.size": TEXT_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": TEXT_SIZE,
            "xtick.labelsize": TEXT_SIZE,
            "ytick.labelsize": TEXT_SIZE,
            "legend.fontsize": TEXT_SIZE,
        }
    )

    print(f"Dataset: {args.dataset}")
    sources = list(GT_SOURCES) if args.all_gt else [args.gt_source]
    for gt in sources:
        try:
            run_one_gt(args, gt)
        except FileNotFoundError as e:
            print(f"  [skip] {gt}: {e}")
    print("Done.")


if __name__ == "__main__":
    main()
