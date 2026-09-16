#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单 TF 三提取方式预测靶集 Venn：P^emb、P^att、P^hid。

用于说明「只有 emb（cos_tok）才连上的靶基因」等跨提取差异；
emb-only 区基因会标注 STRING TP / FP。

示例:
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig3
  python tf_static/plot_tf_target_venn.py --dataset hESC --model scGPT --tf MCM5 SNRPD1
  python tf_static/plot_tf_target_venn.py --dataset hESC --from-ranked --top-case 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG3 = Path(__file__).resolve().parents[1]
if str(FIG3) not in sys.path:
    sys.path.insert(0, str(FIG3))

from fig3_palette import method_color, method_label  # noqa: E402

from tf_static.model_registry import EVL_ROOT, EXTRACTIONS, GT_DISPLAY, INPUT_ROOT  # noqa: E402
from tf_static.plot_tf_jaccard_raincloud_all_models import LABEL_COLOR, TEXT_SIZE  # noqa: E402
from tf_static.utils import partition_venn3_regions, pred_targets_for_tf  # noqa: E402

# Venn circle order: A=emb500, B=att500, C=embhidden500
VENN_KEYS = EXTRACTIONS if tuple(EXTRACTIONS) == ("emb500", "att500", "embhidden500") else ("emb500", "att500", "embhidden500")
VENN_LABELS = tuple(method_label(k) for k in VENN_KEYS)
VENN_COLORS = tuple(method_color(k) for k in VENN_KEYS)

REGION_KEYS = (
    "emb_only",
    "att_only",
    "hid_only",
    "emb_att_not_hid",
    "emb_hid_not_att",
    "att_hid_not_emb",
    "triple",
)


def _gt_tag(gene: str, gt_t: Set[str]) -> str:
    return "TP" if gene in gt_t else "FP"


def collect_venn_table(
    tf: str,
    preds: Dict[str, pd.DataFrame],
    gt_by_tf: Dict[str, Set[str]],
) -> pd.DataFrame:
    sets = {ext: pred_targets_for_tf(preds[ext], tf) for ext in VENN_KEYS}
    regions = partition_venn3_regions(sets["emb500"], sets["att500"], sets["embhidden500"])
    gt_t = gt_by_tf.get(tf, set())
    rows = []
    for rkey in REGION_KEYS:
        genes = sorted(regions[rkey])
        for g in genes:
            rows.append(
                {
                    "TF": tf,
                    "region": rkey,
                    "target": g,
                    "in_STRING": g in gt_t,
                    "status_vs_STRING": _gt_tag(g, gt_t),
                }
            )
        rows.append(
            {
                "TF": tf,
                "region": f"{rkey}__count",
                "target": "",
                "in_STRING": len([g for g in genes if g in gt_t]),
                "status_vs_STRING": f"n={len(genes)}",
            }
        )
    for ext in VENN_KEYS:
        rows.append(
            {
                "TF": tf,
                "region": f"set_size_{ext}",
                "target": "",
                "in_STRING": len(sets[ext] & gt_t),
                "status_vs_STRING": f"|P|={len(sets[ext])}",
            }
        )
    return pd.DataFrame(rows)


def _format_emb_only_caption(tf: str, regions: Dict[str, Set[str]], gt_t: Set[str], max_genes: int = 8) -> str:
    emb_only = sorted(regions["emb_only"])
    if not emb_only:
        return f"{tf}: emb-only = 0"
    parts = []
    for g in emb_only[:max_genes]:
        parts.append(f"{g}({_gt_tag(g, gt_t)})")
    more = len(emb_only) - max_genes
    tail = f" … +{more}" if more > 0 else ""
    n_tp = sum(1 for g in emb_only if g in gt_t)
    return f"{tf}: emb-only {len(emb_only)} (TP {n_tp})\n" + ", ".join(parts) + tail


def plot_single_tf_venn(
    ax,
    tf: str,
    preds: Dict[str, pd.DataFrame],
    gt_by_tf: Dict[str, Set[str]],
) -> Dict[str, Set[str]]:
    from matplotlib_venn import venn3

    sets = [pred_targets_for_tf(preds[ext], tf) for ext in VENN_KEYS]
    gt_t = gt_by_tf.get(tf, set())
    regions = partition_venn3_regions(*sets)

    v = venn3(
        sets,
        set_labels=VENN_LABELS,
        set_colors=VENN_COLORS,
        alpha=0.55,
        ax=ax,
    )
    # emb-only 区数字加粗（matplotlib_venn 子集 id: 100 = A only）
    if v.get_label_by_id("100"):
        v.get_label_by_id("100").set_fontweight("bold")
        v.get_label_by_id("100").set_fontsize(TEXT_SIZE - 1)

    n_emb = len(regions["emb_only"])
    n_emb_tp = len(regions["emb_only"] & gt_t)
    ax.set_title(
        f"{tf}\n|P| emb/att/hid = {len(sets[0])}/{len(sets[1])}/{len(sets[2])}  ·  emb-only {n_emb} (TP {n_emb_tp})",
        fontsize=TEXT_SIZE - 1,
        color=LABEL_COLOR,
    )
    ax.text(
        0.5,
        -0.22,
        _format_emb_only_caption(tf, regions, gt_t),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        color="#333333",
        wrap=True,
    )
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return regions


def plot_target_venn_panels(
    tfs: List[str],
    preds: Dict[str, pd.DataFrame],
    gt_by_tf: Dict[str, Set[str]],
    out_dir: Path,
    dataset: str,
    model: str,
    gt_source: str,
    ncol: int = 2,
) -> None:
    if not tfs:
        print("  (skip Venn: no TFs)")
        return

    n = len(tfs)
    ncol = min(int(ncol), n)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 5.0 * nrow))
    axes_flat = np.atleast_1d(axes).ravel()
    tables = []
    for i, tf in enumerate(tfs):
        plot_single_tf_venn(axes_flat[i], tf, preds, gt_by_tf)
        tables.append(collect_venn_table(tf, preds, gt_by_tf))
    for j in range(len(tfs), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        f"{dataset} — {model}: per-TF predicted target sets (GT: {GT_DISPLAY.get(gt_source, gt_source)})",
        fontsize=TEXT_SIZE,
        color=LABEL_COLOR,
        y=1.02,
    )
    fig.tight_layout()
    stem = out_dir / f"{dataset}_{model}_target_venn_x{len(tfs)}"
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white", dpi=150)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {stem}.png/.pdf")

    long = pd.concat(tables, ignore_index=True)
    long.to_csv(out_dir / f"{dataset}_{model}_target_venn_regions.csv", index=False)
    print(f"  wrote {dataset}_{model}_target_venn_regions.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-TF target-set Venn (emb / att / hid)")
    p.add_argument("--dataset", default="hESC")
    p.add_argument("--model", default="scGPT")
    p.add_argument("--gt-source", default="STRING")
    p.add_argument("--evl-root", type=Path, default=EVL_ROOT)
    p.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--tf", nargs="*", default=None, help="TF gene symbols (e.g. MCM5 SNRPD1)")
    p.add_argument("--from-ranked", action="store_true", help="Use method_specific ranked list")
    p.add_argument("--ranked-csv", type=Path, default=None)
    p.add_argument("--top-case", type=int, default=4)
    p.add_argument("--min-spread", type=float, default=0.10)
    p.add_argument("--min-emb-vs-att", type=float, default=0.05)
    p.add_argument("--story", choices=["emb_vs_att", "spread_only", "att_wins", "all"], default="emb_vs_att")
    p.add_argument("--ncol", type=int, default=2)
    return p.parse_args()


def main() -> None:
    from tf_static.plot_tf_method_specific import build_per_tf_wide, filter_story_tfs

    args = parse_args()
    plt.rcParams.update({"font.size": TEXT_SIZE, "font.family": "DejaVu Sans"})

    out_dir = args.out or (
        FIG3 / "tf_static" / "output" / args.dataset / f"method_specific_{args.model}_{args.gt_source}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    wide, gt_by_tf, preds = build_per_tf_wide(
        args.dataset, args.model, args.gt_source, args.evl_root, args.input_root
    )

    if args.tf:
        tfs = list(args.tf)
    elif args.from_ranked or args.ranked_csv:
        rc = args.ranked_csv or (out_dir / f"{args.dataset}_{args.model}_method_specific_ranked.csv")
        if rc.exists():
            ranked = pd.read_csv(rc)
        else:
            ranked = (
                wide.sort_values("jaccard_spread", ascending=False)
                if args.story == "all"
                else filter_story_tfs(wide, args.min_spread, args.min_emb_vs_att, args.story)
            )
        tfs = ranked["TF"].head(int(args.top_case)).tolist()
    else:
        ranked = filter_story_tfs(wide, args.min_spread, args.min_emb_vs_att, args.story)
        tfs = ranked["TF"].head(int(args.top_case)).tolist()

    missing = [t for t in tfs if t not in wide["TF"].values]
    if missing:
        print(f"  warning: TF not in benchmark table: {missing}")

    print(f"Venn for {len(tfs)} TF(s): {', '.join(tfs)}")
    plot_target_venn_panels(tfs, preds, gt_by_tf, out_dir, args.dataset, args.model, args.gt_source, args.ncol)


if __name__ == "__main__":
    main()
