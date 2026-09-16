#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplement figures for multi-head attention GRN (thesis / defense).

Experiments:
  1. Single-head AUPRC ranking barplot (H0–H7 + Mean8)
  2. Head contribution after routing (TF count per head histogram)
  3. Routing ablation V1/V2/V3 (AUPRC + EPR)
  4. GO enrichment stability (top50 / top100 / top150)

Usage:
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig2/plot
  python plot_head_supplement_figures.py --models scgpt --dataset hESC
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import analyze_attention_heads as ath  # noqa: E402
import enrich_head_go_gprofiler as goen  # noqa: E402
import eval_heads_chip_auprc as ev  # noqa: E402
import eval_heads_chip_epr as em  # noqa: E402
from fig2_palette import model_color  # noqa: E402

plt.rcParams.update(ath.plt.rcParams)

DEFAULT_HEAD_ROOT = _SCRIPT_DIR.parent / "att_head"
DEFAULT_CHIP_ROOT = Path("/mnt/10T/yzn/benchmark_GRN/input_process/CHIP")

# Terms to track in GO stability (substring match on term_name)
STABILITY_TERMS = [
    "Heart Development",
    "Heart Morphogenesis",
    "Regulation Of miRNA Transcription",
    "Positive Regulation Of miRNA",
    "Endodermal Cell Fate Commitment",
]

ABLATION_LABELS = {
    "V1_out_only": "V1: out-strength only",
    "V2_hub_out": "V2: hub + out-strength",
    "V3_full": "V3: full routing",
}


def head_ids(per_head: Dict[int, pd.DataFrame]) -> List[int]:
    return sorted(per_head.keys())


def fusion_strategy_name(n_heads: int, mode: str = "mean") -> str:
    """mean8 / mean4 etc., matching eval_heads_chip_auprc naming."""
    return f"{mode}{n_heads}"


def fusion_display_name(n_heads: int, mode: str = "mean") -> str:
    return f"Mean{n_heads}" if mode == "mean" else f"Max{n_heads}"


def resolve_fusion_in_summary(summary: pd.DataFrame, n_heads: int) -> str:
    """Pick meanK key present in summary (mean8 legacy or mean4)."""
    for key in (fusion_strategy_name(n_heads), "mean8", "mean4"):
        if key in summary["strategy"].values:
            return key
    return fusion_strategy_name(n_heads)


def build_tf_to_head_v1_out_only(
    per_head: Dict[int, pd.DataFrame],
    tfs: Set[str],
) -> Dict[str, int]:
    """π(t) = argmax_h S_out(t, h) — no hub consistency, no fallback."""
    tf_out = em.tf_out_strength_by_head(per_head, tfs)
    tf_to_head: Dict[str, int] = {}
    for tf in tfs:
        best_h, best_w = -1, -1.0
        for h, d in tf_out.items():
            w = float(d.get(tf, 0.0))
            if w > best_w:
                best_w, best_h = w, h
        if best_h >= 0:
            tf_to_head[tf] = best_h
    return tf_to_head


def build_tf_to_head_v2_hub_out(
    per_head: Dict[int, pd.DataFrame],
    tfs: Set[str],
) -> Dict[str, int]:
    """Hub-coherent TFs first, else argmax out-strength — no global fallback."""
    coherent = em.head_hub_coherent_tfs(per_head, tfs)
    tf_out = em.tf_out_strength_by_head(per_head, tfs)
    tf_to_head: Dict[str, int] = dict(coherent)
    for tf in tfs:
        if tf in tf_to_head:
            continue
        best_h, best_w = -1, -1.0
        for h, d in tf_out.items():
            w = float(d.get(tf, 0.0))
            if w > best_w:
                best_w, best_h = w, h
        if best_h >= 0:
            tf_to_head[tf] = best_h
    return tf_to_head


def assemble_routed_grn(
    per_head: Dict[int, pd.DataFrame],
    tf_to_head: Dict[str, int],
    tfs: Set[str],
    direction: str = "sym_max",
) -> pd.DataFrame:
    """Per-TF edges from assigned head only (sym_max lookup)."""
    rows: List[dict] = []
    for tf, h in tf_to_head.items():
        if h not in per_head:
            continue
        lk = ev.build_tf_lookup(per_head[h], tfs, direction)
        for (g1, g2), w in lk.items():
            if g1 == tf:
                rows.append({"Gene1": g1, "Gene2": g2, "EdgeWeight": float(w)})
    if not rows:
        return pd.DataFrame(columns=["Gene1", "Gene2", "EdgeWeight"])
    return pd.DataFrame(rows).drop_duplicates(subset=["Gene1", "Gene2"], keep="first")


def load_dataset(
    model: str,
    dataset: str,
    head_roots: List[Path],
    chip_root: Path,
    load_max_edges: int,
    min_frac_nonzero: float,
) -> Tuple[
    pd.DataFrame,
    Set[str],
    List[str],
    Set[Tuple[str, str]],
    Dict[int, pd.DataFrame],
    Set[str],
    str,
]:
    gt_path = chip_root / f"{dataset}_chip_matched-network.csv"
    expr_path = chip_root / f"{dataset}_chip_matched-ExpressionData.csv"
    gt_all = ev.read_chip_gt(gt_path)
    expr = ev.read_chip_expr(expr_path)
    active = ev.active_genes(expr, min_frac_nonzero)
    gt = gt_all[gt_all["Gene1"].isin(active) & gt_all["Gene2"].isin(active)].copy()
    tfs, _ = ev.chip_gene_sets_from_gt(gt)
    tf_set = set(tfs)
    gene_universe = set(gt["Gene1"].astype(str)) | set(gt["Gene2"].astype(str))
    target_pool = sorted(active)
    gt_edges = ev.chip_true_edges_set(gt)

    files = ath.find_head_files(head_roots, model, dataset)
    if not files:
        raise FileNotFoundError(f"No head TSV for {model} {dataset}")

    per_head: Dict[int, pd.DataFrame] = {}
    for fp in files:
        h = ath.parse_head_number(fp)
        raw = ev.load_head_pred(fp, gene_universe, load_max_edges)
        per_head[h] = ev.filter_pred_aupr_style(raw, gt)

    tag = f"{ath.model_file_prefix(model)}_{dataset}"
    return gt, active, target_pool, gt_edges, per_head, tf_set, tag


def plot_single_head_auprc_bar(
    summary: pd.DataFrame,
    display: str,
    dataset: str,
    color: str,
    out_pdf: Path,
    n_heads: int = 8,
) -> pd.DataFrame:
    """Barplot: per-head + mean fusion; highlight best head vs mean."""
    mean_key = resolve_fusion_in_summary(summary, n_heads)
    order = [f"head{h}" for h in range(n_heads)] + [mean_key]
    sub = summary[summary["strategy"].isin(order)].copy()
    if sub.empty:
        head_strats = [s for s in summary["strategy"] if str(s).startswith("head")]
        sub = summary[summary["strategy"].isin(head_strats + [mean_key])].copy()
    sub["order"] = sub["strategy"].map({s: i for i, s in enumerate(order)})
    sub = sub.sort_values("order")
    if sub.empty:
        return sub

    labels = []
    for s in sub["strategy"]:
        if str(s).startswith("head"):
            labels.append(f"H{str(s).replace('head', '')}")
        elif str(s).startswith("mean"):
            labels.append(fusion_display_name(n_heads))
        else:
            labels.append(str(s))

    vals = sub["AUPRC_micro"].astype(float).values
    is_fusion = sub["strategy"] == mean_key
    head_vals = vals[~is_fusion.values]
    best_idx = int(np.argmax(head_vals)) if len(head_vals) else 0
    best_val = float(head_vals[best_idx]) if len(head_vals) else np.nan
    mean_val = (
        float(sub.loc[sub["strategy"] == mean_key, "AUPRC_micro"].iloc[0])
        if mean_key in sub["strategy"].values
        else np.nan
    )
    mean_label = fusion_display_name(n_heads)

    fig, ax = plt.subplots(figsize=(max(6, len(sub) * 0.9), 4.5))
    x = np.arange(len(sub))
    bar_colors = [color] * len(sub)
    head_i = 0
    for i, lab in enumerate(sub["strategy"]):
        if lab == mean_key:
            bar_colors[i] = "#B0B0B0"
        elif str(lab).startswith("head"):
            if head_i == best_idx:
                bar_colors[i] = "#EB7E60"
            head_i += 1

    bars = ax.bar(x, vals, color=bar_colors, edgecolor="k", linewidth=0.5)
    for i, b in enumerate(bars):
        if sub["strategy"].iloc[i] == mean_key:
            b.set_hatch("//")
            b.set_alpha(0.75)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUPRC (micro, CHIP TF-centric)")
    title_extra = ""
    if not np.isnan(mean_val) and best_val > mean_val:
        title_extra = f"\nBest head ({labels[best_idx]}) > {mean_label}: Δ={best_val - mean_val:.3f}"
    ax.set_title(f"{display} — {dataset}\nSingle-head AUPRC ranking{title_extra}", fontsize=11)
    ax.set_ylim(0, max(vals) * 1.12 + 0.02)

    for i, v in enumerate(vals):
        ax.text(i, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    table = pd.DataFrame({"Head": labels, "AUPRC": vals, "strategy": sub["strategy"].tolist()})
    return table


def plot_head_usage_histogram(
    tf_to_head: Dict[str, int],
    display: str,
    dataset: str,
    color: str,
    out_pdf: Path,
    routing_label: str,
    head_list: Optional[List[int]] = None,
) -> pd.DataFrame:
    counts = Counter(tf_to_head.values())
    if head_list is not None:
        heads = head_list
    elif counts:
        heads = sorted(set(counts.keys()) | set(range(max(counts.keys()) + 1)))
    else:
        heads = list(range(4))
    if not heads:
        heads = list(range(4))
    rows = [{"Head": f"H{h}", "head_id": h, "TF_count": counts.get(h, 0)} for h in heads]
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(max(5, len(heads) * 1.1), 4.2))
    x = np.arange(len(heads))
    vals = df["TF_count"].values
    ax.bar(x, vals, color=color, edgecolor="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"H{h}" for h in heads])
    ax.set_ylabel("Number of TFs assigned")
    ax.set_title(
        f"{display} — {dataset}\nHead usage after routing ({routing_label})\n"
        f"Total TFs routed: {sum(vals)}",
        fontsize=10,
    )
    for i, v in enumerate(vals):
        if v > 0:
            ax.text(i, v + 0.3, str(int(v)), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return df


def evaluate_ablation(
    per_head: Dict[int, pd.DataFrame],
    gt: pd.DataFrame,
    tf_set: Set[str],
    target_pool: List[str],
    gt_edges: Set[Tuple[str, str]],
    pred_direction: str,
    neg_ratio: float,
    seed: int,
    top_edges: int,
) -> pd.DataFrame:
    """AUPRC + EPR for V1 / V2 / V3 routing variants."""
    k_eval = ev.resolve_top_k(len(gt_edges), top_edges)
    builders = {
        "V1_out_only": build_tf_to_head_v1_out_only,
        "V2_hub_out": build_tf_to_head_v2_hub_out,
        "V3_full": em.build_tf_to_head_map,
    }
    rows = []
    for name, fn in builders.items():
        tf_map = fn(per_head, tf_set)
        pred = assemble_routed_grn(per_head, tf_map, tf_set, pred_direction)
        lk = ev.build_tf_lookup(pred, tf_set, pred_direction)
        glob, _ = ev.evaluate_tf_centric(
            gt=gt,
            lookup=lk,
            target_pool=target_pool,
            neg_ratio=neg_ratio,
            seed=seed + hash(name) % 10000,
        )
        epr = em.compute_epr(pred, gt)
        rows.append(
            {
                "variant": name,
                "label": ABLATION_LABELS[name],
                "n_tf_routed": len(tf_map),
                "AUPRC_micro": glob["AUPRC_micro"] if glob else np.nan,
                "AUPRC_macro": glob["AUPRC_macro"] if glob else np.nan,
                "EPR": epr["EPR"],
                "Early_Precision": epr["Early_Precision"],
                "TopK": epr["TopK"],
            }
        )
        print(
            f"  {name}: AUPRC_micro={rows[-1]['AUPRC_micro']:.4f}  "
            f"EPR={rows[-1]['EPR']:.4f}  n_tf={len(tf_map)}"
        )
    return pd.DataFrame(rows)


def plot_ablation_bars(ablation: pd.DataFrame, display: str, dataset: str, out_pdf: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(len(ablation))
    labels = [ABLATION_LABELS.get(v, v) for v in ablation["variant"]]

    colors = ["#C8D8EB", "#8FB4DC", "#EB7E60"]
    for ax, metric, ylab in zip(
        axes,
        ["AUPRC_micro", "EPR"],
        ["AUPRC (micro)", "EPR"],
    ):
        vals = ablation[metric].astype(float).values
        bars = ax.bar(x, vals, color=colors[: len(vals)], edgecolor="k", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace(": ", ":\n") for l in labels], fontsize=8)
        ax.set_ylabel(ylab)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    fig.suptitle(f"{display} — {dataset}\nRouting ablation (GT only for evaluation)", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_tf_routing_detail(
    per_head: Dict[int, pd.DataFrame],
    tf_set: Set[str],
    tf_map_full: Dict[str, int],
    summary: pd.DataFrame,
    out_dir: Path,
    tag: str,
    heads_sorted: List[int],
) -> None:
    """Per-TF routing reason + AUPRC on assigned vs best head."""
    coherent = em.head_hub_coherent_tfs(per_head, tf_set)
    per_tf_path = _SCRIPT_DIR / "output" / "head_chip_auprc" / tag / f"{tag}_auprc_per_tf.csv"
    wide = None
    if per_tf_path.exists():
        per_tf = pd.read_csv(per_tf_path)
        wide = per_tf.pivot(index="TF", columns="strategy", values="AUPRC")

    rows = []
    for tf in sorted(tf_set):
        ah = int(tf_map_full.get(tf, -1))
        reason = "hub_coherent" if tf in coherent and coherent[tf] == ah else "out_strength"
        best_h, best_au, routed_au = "", np.nan, np.nan
        if wide is not None and tf in wide.index:
            head_cols = [c for c in wide.columns if str(c).startswith("head")]
            best_col = wide.loc[tf, head_cols].astype(float).idxmax()
            best_h = int(str(best_col).replace("head", ""))
            best_au = float(wide.loc[tf, best_col])
            col = f"head{ah}"
            routed_au = float(wide.loc[tf, col]) if col in wide.columns else np.nan
        rows.append(
            {
                "TF": tf,
                "assigned_head": ah,
                "reason": reason,
                "coherent_head": coherent.get(tf, ""),
                "best_auprc_head": best_h,
                "AUPRC_routed_head": routed_au,
                "AUPRC_best_head": best_au,
                "match_best_auprc": best_h == ah if best_h != "" else False,
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / f"{tag}_exp2_tf_routing_detail.csv", index=False)


def run_go_stability(
    model: str,
    dataset: str,
    head_roots: List[Path],
    chip_root: Path,
    active: Set[str],
    gt: pd.DataFrame,
    out_dir: Path,
    top_gene_list: List[int],
    load_max_edges: int,
    padj_cutoff: float,
) -> pd.DataFrame:
    """Enrichment at top50/100/150; track key developmental terms."""
    _, chip_genes = ev.chip_gene_sets_from_gt(gt)
    background = sorted(active)
    bg_set = set(background)
    k_eval = ev.resolve_top_k(len(gt), 0)
    lib_name = goen.GSEAPY_LIBS["GO:BP"]
    gmt_raw = goen.load_go_library(lib_name, goen.CACHE_DIR, 5, 500)
    gmt = goen.filter_gmt_to_background(gmt_raw, bg_set)

    files = ath.find_head_files(head_roots, model, dataset)
    stability_rows = []

    for top_n in top_gene_list:
        for fp in files:
            h = ath.parse_head_number(fp)
            raw = ev.load_head_pred(fp, active, load_max_edges)
            edge_df = ev.filter_pred_aupr_style(raw, gt)
            query = goen.genes_topw(edge_df, k_eval, top_n)
            if len(query) < 20:
                continue
            enr = goen.run_local_enrich(query, background, gmt, padj_cutoff, 3)
            for term_key in STABILITY_TERMS:
                if enr.empty:
                    hit = False
                    padj = np.nan
                    term_full = ""
                else:
                    mask = enr["term_name"].str.contains(term_key, case=False, regex=False)
                    hit = bool(mask.any())
                    if hit:
                        row = enr.loc[mask].iloc[0]
                        padj = float(row["padj"])
                        term_full = str(row["term_name"])
                    else:
                        padj = np.nan
                        term_full = ""
                stability_rows.append(
                    {
                        "top_genes": top_n,
                        "Head": h,
                        "term_pattern": term_key,
                        "significant": hit,
                        "padj": padj,
                        "term_name": term_full,
                        "n_query_genes": len(query),
                    }
                )

    stab = pd.DataFrame(stability_rows)
    stab.to_csv(out_dir / "go_stability_topN.csv", index=False)

    # Heatmap: top_N × (head, term) — fraction of heads with sig. hit
    if not stab.empty:
        pivot = (
            stab.groupby(["top_genes", "term_pattern"])["significant"]
            .mean()
            .unstack(fill_value=0)
        )
        fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 1.2), 3.5))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels([t[:28] for t in pivot.columns], rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels([f"top{n}" for n in pivot.index])
        ax.set_title(f"{dataset} — GO term stability across top-N gene sets\n"
                     "(fraction of heads with padj≤cutoff)")
        plt.colorbar(im, ax=ax, label="Fraction heads significant")
        fig.tight_layout()
        fig.savefig(out_dir / "go_stability_heatmap.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

    return stab


def run(args: argparse.Namespace) -> None:
    head_roots = [Path(r) for r in args.head_root] if args.head_root else [DEFAULT_HEAD_ROOT]
    chip_root = Path(args.chip_root)
    display = ath.MODEL_ALIASES.get(ath.model_file_prefix(args.models), args.models)
    color = model_color(display)

    gt, active, target_pool, gt_edges, per_head, tf_set, tag = load_dataset(
        args.models,
        args.dataset,
        head_roots,
        chip_root,
        args.load_max_edges,
        args.min_frac_nonzero,
    )
    out_dir = Path(args.output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    n_heads = len(per_head)
    heads_sorted = head_ids(per_head)
    mean_key = fusion_strategy_name(n_heads)

    # --- Exp 1: single-head AUPRC bar ---
    print("\n[Exp1] Single-head AUPRC ranking")
    summary_path = Path(args.auprc_summary) if args.auprc_summary else None
    if summary_path and summary_path.exists():
        summary = pd.read_csv(summary_path)
        print(f"  Loaded existing summary: {summary_path}")
    else:
        summary_rows = []
        fused_mean = ev.fuse_predictions(per_head, "mean")
        strategies = {f"head{h}": df for h, df in per_head.items()}
        strategies[mean_key] = fused_mean
        for name, df in strategies.items():
            lk = ev.build_tf_lookup(df, tf_set, args.pred_direction)
            glob, _ = ev.evaluate_tf_centric(
                gt=gt,
                lookup=lk,
                target_pool=target_pool,
                neg_ratio=args.neg_ratio,
                seed=args.seed + hash(name) % 10000,
            )
            if glob:
                summary_rows.append({"strategy": name, **glob})
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(out_dir / f"{tag}_auprc_summary_rerun.csv", index=False)

    auprc_table = plot_single_head_auprc_bar(
        summary,
        display,
        args.dataset,
        color,
        out_dir / f"{tag}_exp1_single_head_auprc_bar.pdf",
        n_heads=n_heads,
    )
    auprc_table.to_csv(out_dir / f"{tag}_exp1_single_head_auprc_table.csv", index=False)

    mk = resolve_fusion_in_summary(
        pd.DataFrame({"strategy": auprc_table["strategy"]}), n_heads
    )
    if mk in auprc_table["strategy"].values:
        heads_only = auprc_table[~auprc_table["strategy"].str.startswith("mean")]
        best = heads_only.loc[heads_only["AUPRC"].idxmax()]
        mean_au = float(auprc_table.loc[auprc_table["strategy"] == mk, "AUPRC"].iloc[0])
        print(
            f"  Best head: {best['Head']} AUPRC={best['AUPRC']:.4f}  "
            f"{fusion_display_name(n_heads)}={mean_au:.4f}  "
            f"Δ={float(best['AUPRC']) - mean_au:+.4f}"
        )

    # --- Exp 2: head usage histogram ---
    print("\n[Exp2] Head contribution (routing assignment counts)")
    tf_map_full = em.build_tf_to_head_map(per_head, tf_set)
    usage = plot_head_usage_histogram(
        tf_map_full,
        display,
        args.dataset,
        color,
        out_dir / f"{tag}_exp2_head_usage_histogram.pdf",
        "V3 full routing",
        head_list=heads_sorted,
    )
    usage.to_csv(out_dir / f"{tag}_exp2_head_usage_counts.csv", index=False)
    pd.DataFrame([{"TF": tf, "assigned_head": h} for tf, h in sorted(tf_map_full.items())]).to_csv(
        out_dir / f"{tag}_exp2_tf_to_head.csv", index=False
    )
    _write_tf_routing_detail(per_head, tf_set, tf_map_full, summary, out_dir, tag, heads_sorted)
    print(usage.to_string(index=False))

    # --- Exp 3: ablation ---
    print("\n[Exp3] Routing ablation (V1 / V2 / V3)")
    ablation = evaluate_ablation(
        per_head,
        gt,
        tf_set,
        target_pool,
        gt_edges,
        args.pred_direction,
        args.neg_ratio,
        args.seed,
        args.top_edges,
    )
    ablation.to_csv(out_dir / f"{tag}_exp3_routing_ablation.csv", index=False)
    plot_ablation_bars(ablation, display, args.dataset, out_dir / f"{tag}_exp3_routing_ablation.pdf")

    # --- Exp 4: GO stability ---
    print("\n[Exp4] GO enrichment stability (top50 / top100 / top150)")
    top_list = [int(x) for x in args.go_top_genes.split(",") if x.strip()]
    stab = run_go_stability(
        args.models,
        args.dataset,
        head_roots,
        chip_root,
        active,
        gt,
        out_dir,
        top_list,
        args.load_max_edges,
        args.padj_cutoff,
    )
    print(stab.groupby(["top_genes", "term_pattern"])["significant"].sum().to_string())

    meta = {
        "model": args.models,
        "dataset": args.dataset,
        "n_heads": n_heads,
        "fusion_strategy": mean_key,
        "pred_direction": args.pred_direction,
        "experiments": ["exp1_auprc_bar", "exp2_head_usage", "exp3_ablation", "exp4_go_stability"],
    }
    with open(out_dir / f"{tag}_supplement_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[DONE] All supplement figures -> {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Head supplement figures (AUPRC, routing, ablation, GO).")
    p.add_argument("--models", default="scgpt")
    p.add_argument("--dataset", default="hESC")
    p.add_argument("--head-root", action="append", default=None)
    p.add_argument("--chip-root", default=str(DEFAULT_CHIP_ROOT))
    p.add_argument("--output-dir", default=str(_SCRIPT_DIR / "output" / "head_supplement"))
    p.add_argument(
        "--auprc-summary",
        default="",
        help="Optional existing *_auprc_summary.csv (skip recompute for Exp1)",
    )
    p.add_argument("--pred-direction", default="sym_max")
    p.add_argument("--neg-ratio", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-edges", type=int, default=0)
    p.add_argument("--load-max-edges", type=int, default=0)
    p.add_argument("--min-frac-nonzero", type=float, default=0.05)
    p.add_argument("--go-top-genes", default="50,100,150")
    p.add_argument("--padj-cutoff", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.auprc_summary:
        tag = f"{ath.model_file_prefix(args.models)}_{args.dataset}"
        default_sum = _SCRIPT_DIR / "output" / "head_chip_auprc" / tag / f"{tag}_auprc_summary.csv"
        if default_sum.exists():
            args.auprc_summary = str(default_sum)
    run(args)


if __name__ == "__main__":
    main()
