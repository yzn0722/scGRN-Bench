#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取方式敏感的 TF（method-specific TF）— 三提取 Jaccard + case study。

流程：
  1. 每个 TF 计算 emb500 / att500 / embhidden500 的靶集 Jaccard
  2. spread = max − min；emb−att = max(emb500, embhidden500) − att500
  3. 筛选 + 出图 + 导出靶基因表 + ego 网 ×3

示例:
  python tf_static/plot_tf_method_specific.py --dataset hESC --model scGPT
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG3 = Path(__file__).resolve().parents[1]
if str(FIG3) not in sys.path:
    sys.path.insert(0, str(FIG3))

from fig3_palette import method_label  # noqa: E402

from tf_static.model_registry import EVL_ROOT, EXTRACTIONS, GT_DISPLAY, INPUT_ROOT, resolve_gt_path, resolve_pred_path  # noqa: E402
from tf_static.plot_tf_jaccard_raincloud_all_models import EXTRACT_COLORS, EXTRACT_DISPLAY, LABEL_COLOR, TEXT_SIZE  # noqa: E402
from tf_static.plot_tf_static_compare_methods import plot_hub_ego_method_rows, plot_method_disagreement_bars  # noqa: E402
from tf_static.plot_tf_target_venn import plot_target_venn_panels  # noqa: E402
from tf_static.utils import classify_tf_edges, filter_prediction, load_gt_network, per_tf_edge_sets  # noqa: E402

METHOD_ORDER = list(EXTRACTIONS)


def build_per_tf_wide(
    dataset: str,
    model: str,
    gt_source: str,
    evl_root: Path,
    input_root: Path,
) -> Tuple[pd.DataFrame, Dict[str, Set[str]], Dict[str, pd.DataFrame]]:
    gt_path = resolve_gt_path(gt_source, dataset, input_root)
    gt_gene1, gt_all, gt_n, _, gt_by_tf = load_gt_network(gt_path)

    preds: Dict[str, pd.DataFrame] = {}
    parts = []
    for ext in METHOD_ORDER:
        fp = resolve_pred_path(model, ext, dataset, evl_root)
        if fp is None:
            raise FileNotFoundError(fp)
        pred = filter_prediction(fp, gt_gene1, gt_all, gt_n)
        preds[ext] = pred
        met = per_tf_edge_sets(pred, gt_by_tf)
        met = met.rename(columns={c: f"{c}_{ext}" for c in met.columns if c != "TF"})
        parts.append(met)

    wide = parts[0]
    for p in parts[1:]:
        wide = wide.merge(p, on="TF", how="outer")

    jac_cols = [f"jaccard_{e}" for e in METHOD_ORDER]
    mat = wide[jac_cols].to_numpy(dtype=float)
    wide["jaccard_best"] = np.nanmax(mat, axis=1)
    wide["jaccard_worst"] = np.nanmin(mat, axis=1)
    wide["jaccard_spread"] = wide["jaccard_best"] - wide["jaccard_worst"]
    wide["best_extraction"] = [METHOD_ORDER[i] for i in np.nanargmax(mat, axis=1)]
    wide["worst_extraction"] = [METHOD_ORDER[i] for i in np.nanargmin(mat, axis=1)]
    wide["jaccard_emb_max"] = wide[["jaccard_emb500", "jaccard_embhidden500"]].max(axis=1)
    wide["jaccard_att"] = wide["jaccard_att500"]
    wide["emb_minus_att"] = wide["jaccard_emb_max"] - wide["jaccard_att"]
    if "gt_outdegree_emb500" in wide.columns:
        wide["gt_outdegree"] = wide["gt_outdegree_emb500"]
    return wide, gt_by_tf, preds


def filter_story_tfs(
    wide: pd.DataFrame,
    min_spread: float,
    min_emb_vs_att: float,
    story: str,
) -> pd.DataFrame:
    jac_cols = [f"jaccard_{e}" for e in METHOD_ORDER]
    sub = wide.dropna(subset=jac_cols).copy()
    sub = sub[sub["jaccard_spread"] >= min_spread]
    if story == "emb_vs_att":
        sub = sub[sub["emb_minus_att"] >= min_emb_vs_att]
        sub = sub.sort_values(["emb_minus_att", "jaccard_spread"], ascending=False)
    elif story == "att_wins":
        sub = sub[sub["best_extraction"] == "att500"]
        sub = sub.sort_values("jaccard_spread", ascending=False)
    elif story == "spread_only":
        sub = sub.sort_values("jaccard_spread", ascending=False)
    else:
        sub = sub.sort_values("jaccard_spread", ascending=False)
    return sub


def plot_dumbbell_top(
    ranked: pd.DataFrame,
    dataset: str,
    model: str,
    gt_source: str,
    out_dir: Path,
    top_n: int,
) -> None:
    sub = ranked.head(int(top_n)).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.5, max(4, 0.48 * len(sub))))
    y = np.arange(len(sub))
    for i, r in enumerate(sub.itertuples()):
        vals = [getattr(r, f"jaccard_{e}") for e in METHOD_ORDER]
        ax.plot(vals, [i, i, i], color="#CCCCCC", lw=1.8, zorder=1)
        for e, v in zip(METHOD_ORDER, vals):
            ax.scatter(v, i, s=90, color=EXTRACT_COLORS[e], zorder=3, edgecolors="#333", linewidths=0.5)
        ax.text(
            1.03,
            i,
            f"Δ={r.jaccard_spread:.2f}  emb−att={r.emb_minus_att:+.2f}  best={method_label(r.best_extraction)}",
            va="center",
            fontsize=9,
            transform=ax.get_yaxis_transform(),
        )
    ax.set_yticks(y)
    ax.set_yticklabels(sub["TF"], fontsize=TEXT_SIZE - 1)
    ax.set_xlabel("Per-TF target-set Jaccard", fontsize=TEXT_SIZE, color=LABEL_COLOR)
    ax.set_xlim(-0.02, 1.0)
    leg = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=EXTRACT_COLORS[e], markersize=9, label=EXTRACT_DISPLAY[e])
        for e in METHOD_ORDER
    ]
    ax.legend(handles=leg, loc="lower right", frameon=False, fontsize=TEXT_SIZE - 1)
    ax.set_title(
        f"{dataset} — {model}: extraction-sensitive TFs\n(GT: {GT_DISPLAY.get(gt_source, gt_source)})",
        fontsize=TEXT_SIZE,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    stem = out_dir / f"{dataset}_{model}_method_specific_dumbbell"
    fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {stem}.png/.pdf")


def export_target_tables(
    tfs: List[str],
    preds: Dict[str, pd.DataFrame],
    gt_by_tf: Dict[str, Set[str]],
    out_dir: Path,
    dataset: str,
    model: str,
) -> None:
    all_rows = []
    summaries = []
    for tf in tfs:
        rec = {"TF": tf}
        for ext in METHOD_ORDER:
            tp, fp, fn = classify_tf_edges(tf, preds[ext], gt_by_tf)
            rec[f"{ext}_jaccard"] = len(tp) / len(tp | fp | fn | gt_by_tf.get(tf, set())) if (tp or fp or gt_by_tf.get(tf)) else 0
            rec[f"{ext}_TP"] = len(tp)
            rec[f"{ext}_FP"] = len(fp)
            rec[f"{ext}_FN"] = len(fn)
            rec[f"{ext}_TP_genes"] = ";".join(sorted(tp))
            rec[f"{ext}_FP_genes"] = ";".join(sorted(fp))
            rec[f"{ext}_FN_genes"] = ";".join(sorted(fn))
            for g in sorted(tp):
                all_rows.append({"TF": tf, "extraction": ext, "target": g, "status": "TP"})
            for g in sorted(fp):
                all_rows.append({"TF": tf, "extraction": ext, "target": g, "status": "FP"})
            for g in sorted(fn):
                all_rows.append({"TF": tf, "extraction": ext, "target": g, "status": "FN"})
        summaries.append(rec)
    pd.DataFrame(summaries).to_csv(out_dir / f"{dataset}_{model}_case_study_targets_summary.csv", index=False)
    pd.DataFrame(all_rows).to_csv(out_dir / f"{dataset}_{model}_case_study_targets_long.csv", index=False)
    print(f"  wrote target CSVs for {len(tfs)} TFs")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Method-specific TF case study")
    p.add_argument("--dataset", default="hESC")
    p.add_argument("--model", default="scGPT")
    p.add_argument("--gt-source", default="STRING")
    p.add_argument("--evl-root", type=Path, default=EVL_ROOT)
    p.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--top-disagree", type=int, default=20)
    p.add_argument("--top-case", type=int, default=4)
    p.add_argument("--min-spread", type=float, default=0.10)
    p.add_argument("--min-emb-vs-att", type=float, default=0.05)
    p.add_argument("--story", choices=["emb_vs_att", "spread_only", "att_wins", "all"], default="emb_vs_att")
    p.add_argument("--max-ego-targets", type=int, default=25)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    plt.rcParams.update({"font.size": TEXT_SIZE, "font.family": "DejaVu Sans"})

    out_dir = args.out or (
        FIG3 / "tf_static" / "output" / args.dataset / f"method_specific_{args.model}_{args.gt_source}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{args.dataset} | {args.model} | GT={args.gt_source} | story={args.story}")
    wide, gt_by_tf, preds = build_per_tf_wide(
        args.dataset, args.model, args.gt_source, args.evl_root, args.input_root
    )
    wide.to_csv(out_dir / f"{args.dataset}_{args.model}_per_tf_jaccard_wide.csv", index=False)

    ranked = (
        wide.sort_values("jaccard_spread", ascending=False)
        if args.story == "all"
        else filter_story_tfs(wide, args.min_spread, args.min_emb_vs_att, args.story)
    )
    ranked.to_csv(out_dir / f"{args.dataset}_{args.model}_method_specific_ranked.csv", index=False)
    print(f"  Ranked ({args.story}): {len(ranked)} TFs")
    if len(ranked):
        cols = ["TF", "jaccard_emb500", "jaccard_att500", "jaccard_embhidden500", "jaccard_spread", "emb_minus_att", "best_extraction"]
        print(ranked[cols].head(10).to_string(index=False))

    plot_method_disagreement_bars(wide, args.dataset, METHOD_ORDER, out_dir, args.top_disagree)
    plot_dumbbell_top(ranked, args.dataset, args.model, args.gt_source, out_dir, min(args.top_case, 15))

    case_tfs = ranked["TF"].head(int(args.top_case)).tolist()
    if case_tfs:
        export_target_tables(case_tfs, preds, gt_by_tf, out_dir, args.dataset, args.model)
        plot_hub_ego_method_rows(
            case_tfs,
            args.dataset,
            METHOD_ORDER,
            preds,
            gt_by_tf,
            out_dir,
            args.max_ego_targets,
            file_stem=f"{args.dataset}_{args.model}_method_specific_ego_x3",
            suptitle=f"{args.dataset} — {args.model}: method-sensitive TF case studies",
        )
        plot_target_venn_panels(
            case_tfs,
            preds,
            gt_by_tf,
            out_dir,
            args.dataset,
            args.model,
            args.gt_source,
        )
    print("Done.")


if __name__ == "__main__":
    main()
