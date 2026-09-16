#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Label-free AUPRC for multi-head GRN assembly (6 CHIP datasets).

Strategies (no CHIP edges used in assembly):
  mean8, max8, fusion (semantic unique boost), routed_symmax (per-TF routing + sym_max)

CHIP is used only for TF/gene sets and post-hoc AUPRC (sym_max, TF-centric, neg_ratio=1).

Outputs:
  output/head_chip_auprc_label_free/{model}/auprc_summary_label_free.csv
  output/head_chip_auprc_label_free/{model}/auprc_6datasets_label_free.pdf

Usage:
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig2/plot
  python eval_heads_chip_auprc_label_free.py --models scgpt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import analyze_attention_heads as ath  # noqa: E402
import eval_heads_chip_auprc as ev  # noqa: E402
import eval_heads_chip_epr as em  # noqa: E402
from fig2_palette import model_color  # noqa: E402

DEFAULT_CHIP_DATASETS = ev.DEFAULT_CHIP_DATASETS
SCGPT_MULTIHEAD_ROOT = em.SCGPT_MULTIHEAD_ROOT

MAIN_STRATEGIES = ["mean8", "max8", "fusion", "routed_symmax"]

STRATEGY_LABELS = {
    "mean8": "Mean (8 heads)",
    "max8": "Max (8 heads)",
    "fusion": "Semantic fusion",
    "routed_symmax": "Routed + sym_max",
}


def evaluate_dataset(
    model: str,
    dataset: str,
    head_roots: List[Path],
    chip_root: Path,
    pred_direction: str,
    neg_ratio: float,
    seed: int,
    fusion_alpha: float,
    export_top_m: int,
    min_frac_nonzero: float,
    load_max_edges: int,
) -> List[Dict]:
    gt_path = chip_root / f"{dataset}_chip_matched-network.csv"
    expr_path = chip_root / f"{dataset}_chip_matched-ExpressionData.csv"
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    gt_all = ev.read_chip_gt(gt_path)
    if expr_path.exists():
        expr = ev.read_chip_expr(expr_path)
        active = ev.active_genes(expr, min_frac_nonzero)
        gt = gt_all[gt_all["Gene1"].isin(active) & gt_all["Gene2"].isin(active)].copy()
    else:
        active = set(gt_all["Gene1"].astype(str)) | set(gt_all["Gene2"].astype(str))
        gt = gt_all

    tfs, _ = ev.chip_gene_sets_from_gt(gt)
    gene_universe = set(gt["Gene1"].astype(str)) | set(gt["Gene2"].astype(str))
    tf_set = set(tfs)
    target_pool = sorted(active)
    gt_edges = ev.chip_true_edges_set(gt)
    k_eval = ev.resolve_top_k(len(gt_edges), 0)

    files = ath.find_head_files(head_roots, model, dataset)
    if not files:
        raise FileNotFoundError(f"No head TSV for {model} {dataset}")

    per_head: Dict[int, pd.DataFrame] = {}
    per_head_lk: Dict[int, Dict] = {}
    for fp in files:
        h = ath.parse_head_number(fp)
        raw = ev.load_head_pred(fp, gene_universe, load_max_edges)
        per_head[h] = ev.filter_pred_aupr_style(raw, gt)
        per_head_lk[h] = em.tf_forward_lookup(per_head[h], tfs)

    top_m = em.resolve_label_free_top_m(per_head_lk, export_top_m)

    strategies = {
        "mean8": em.fuse_mean8(per_head),
        "max8": em.fuse_max8(per_head),
        "fusion": em.fuse_semantic_unique(per_head, tfs, top_m, alpha=fusion_alpha),
        "routed_symmax": em.fuse_routed_symmax(per_head, tfs)[0],
    }

    rows: List[Dict] = []
    for name, pred_df in strategies.items():
        lookup = ev.build_tf_lookup(pred_df, tf_set, pred_direction)
        glob, per_tf = ev.evaluate_tf_centric(
            gt=gt,
            lookup=lookup,
            target_pool=target_pool,
            neg_ratio=neg_ratio,
            seed=seed + hash((name, dataset)) % 10000,
        )
        if glob is None:
            continue
        prec, rec = ev.precision_recall_at_k(lookup, gt_edges, k_eval)
        rows.append(
            {
                "dataset": dataset,
                "strategy": name,
                "label_free": True,
                "pred_direction": pred_direction,
                "export_top_m": top_m,
                "fusion_alpha": fusion_alpha if name == "fusion" else np.nan,
                "precision_at_k": prec,
                "recall_at_k": rec,
                **glob,
            }
        )
        print(
            f"  {dataset} {name}: AUPRC_micro={glob['AUPRC_micro']:.4f}  "
            f"macro={glob['AUPRC_macro']:.4f}"
        )
    return rows


def plot_auprc_bar(summary: pd.DataFrame, model_display: str, out_pdf: Path) -> None:
    main = summary[summary["strategy"].isin(MAIN_STRATEGIES)].copy()
    datasets = [d for d in DEFAULT_CHIP_DATASETS if d in set(main["dataset"])]
    if not datasets:
        datasets = sorted(main["dataset"].unique())

    colors = {
        "mean8": "#B0B0B0",
        "max8": "#C8D8EB",
        "fusion": model_color(model_display, "#8FB4DC"),
        "routed_symmax": "#EB7E60",
    }

    n_strat = len(MAIN_STRATEGIES)
    x = np.arange(len(datasets))
    width = 0.19
    offsets = np.linspace(-(n_strat - 1) / 2, (n_strat - 1) / 2, n_strat) * width

    fig, ax = plt.subplots(figsize=(11, 5.2))
    for off, strat in zip(offsets, MAIN_STRATEGIES):
        vals = []
        for ds in datasets:
            sub = main[(main["dataset"] == ds) & (main["strategy"] == strat)]
            vals.append(float(sub["AUPRC_micro"].iloc[0]) if not sub.empty else 0.0)
        bars = ax.bar(
            x + off,
            vals,
            width,
            label=STRATEGY_LABELS[strat],
            color=colors[strat],
            edgecolor="k",
            linewidth=0.35,
        )
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.008,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel("AUPRC (micro, TF-centric)")
    ax.set_title(
        f"{model_display} — CHIP AUPRC across 6 datasets (label-free assembly)\n"
        f"Scoring: {summary['pred_direction'].iloc[0] if 'pred_direction' in summary.columns else 'sym_max'}; "
        "GT only for evaluation",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=7, framealpha=0.92, ncol=2)
    ymax = main["AUPRC_micro"].max()
    ax.set_ylim(0, min(1.0, ymax + 0.08) if not np.isnan(ymax) else 1.0)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label-free AUPRC: mean8 / max8 / fusion / routed_symmax.")
    p.add_argument("--models", default="scgpt")
    p.add_argument("--datasets", default=",".join(DEFAULT_CHIP_DATASETS))
    p.add_argument("--head-root", action="append", default=None)
    p.add_argument("--chip-root", default="/mnt/10T/yzn/benchmark_GRN/input_process/CHIP")
    p.add_argument(
        "--output-dir",
        default=str(_SCRIPT_DIR / "output" / "head_chip_auprc_label_free"),
    )
    p.add_argument("--pred-direction", default="sym_max")
    p.add_argument("--neg-ratio", type=float, default=1.0)
    p.add_argument("--fusion-alpha", type=float, default=1.5)
    p.add_argument("--export-top-m", type=int, default=0)
    p.add_argument("--load-max-edges", type=int, default=0)
    p.add_argument("--min-frac-nonzero", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    head_roots = [Path(r) for r in args.head_root] if args.head_root else [SCGPT_MULTIHEAD_ROOT]
    chip_root = Path(args.chip_root)
    out_root = Path(args.output_dir)

    for model in models:
        display = ath.MODEL_ALIASES.get(ath.model_file_prefix(model), model)
        tag = ath.model_file_prefix(model)
        out_dir = out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        all_rows: List[Dict] = []
        print(f"\n[{tag}] label-free AUPRC ({len(datasets)} datasets, direction={args.pred_direction})")
        for ds in datasets:
            try:
                all_rows.extend(
                    evaluate_dataset(
                        model,
                        ds,
                        head_roots,
                        chip_root,
                        args.pred_direction,
                        args.neg_ratio,
                        args.seed,
                        args.fusion_alpha,
                        args.export_top_m,
                        args.min_frac_nonzero,
                        args.load_max_edges,
                    )
                )
            except FileNotFoundError as e:
                print(f"  [WARN] skip {ds}: {e}")

        if not all_rows:
            print(f"[WARN] no results for {tag}")
            continue

        summary = pd.DataFrame(all_rows)
        csv_path = out_dir / "auprc_summary_label_free.csv"
        summary.to_csv(csv_path, index=False)

        plot_auprc_bar(summary, display, out_dir / "auprc_6datasets_label_free.pdf")
        plot_auprc_bar(summary, display, out_dir / "auprc_6datasets_label_free.png")

        # gain vs mean8 table
        wide = summary.pivot(index="dataset", columns="strategy", values="AUPRC_micro")
        if "mean8" in wide.columns:
            gain = wide.sub(wide["mean8"], axis=0)
            gain.to_csv(out_dir / "auprc_gain_vs_mean8.csv")
            print("\n  ΔAUPRC_micro vs mean8:")
            print(gain[MAIN_STRATEGIES].to_string(float_format=lambda x: f"{x:+.4f}"))

        print(f"\n[INFO] CSV -> {csv_path}")
        print(f"[INFO] PDF -> {out_dir / 'auprc_6datasets_label_free.pdf'}")


if __name__ == "__main__":
    main()
