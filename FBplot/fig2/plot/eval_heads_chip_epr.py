#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Label-free EPR evaluation for multi-head attention GRNs.

Inference (GRN assembly) uses ONLY attention head exports — no CHIP edges,
no per-dataset EPR/AUPRC to pick heads or tune fusion.

CHIP ground truth is used ONLY for:
  - TF / gene universe (benchmark node sets from matched network metadata)
  - Post-hoc EPR evaluation (Top-K = min(|pred|, |GT|), standard FBEval protocol)

Strategies (all label-free at inference):
  mean8          — average weights across heads
  max8           — element-wise max across heads
  fusion         — semantic fusion (boost edges in exactly one head's top-M)
  routed         — per-TF pick head by max TF out-strength (hub-coherent override)

Outputs:
  output/head_chip_epr/{model}/epr_summary_label_free.csv
  output/head_chip_epr/{model}/epr_6datasets_label_free.pdf

Usage:
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig2/plot
  python eval_heads_chip_epr.py --models scgpt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import analyze_attention_heads as ath  # noqa: E402
import eval_heads_chip_auprc as ev  # noqa: E402
from fig2_palette import model_color  # noqa: E402

DEFAULT_CHIP_DATASETS = ["hESC", "hHep", "mDC", "mHSC-E", "mHSC-GM", "mHSC-L"]
SCGPT_MULTIHEAD_ROOT = Path(
    "/mnt/10T/yzn/benchmark_GRN/evl_omipath/output_att500/scgpt_multihead"
)

MAIN_STRATEGIES = ["mean8", "max8", "fusion", "routed"]

STRATEGY_LABELS = {
    "mean8": "Mean (8 heads)",
    "max8": "Max (8 heads)",
    "fusion": "Semantic fusion",
    "routed": "Per-TF routed",
}


def compute_epr(pred: pd.DataFrame, gt: pd.DataFrame) -> Dict[str, float]:
    """
    EPR per FBEval/EPR.py (TFEdges=True).
    Evaluation uses GT edge count for Top-K cap only — not used during GRN assembly.
    """
    tfs = set(gt["Gene1"].astype(str))
    genes = set(gt["Gene1"].astype(str)) | set(gt["Gene2"].astype(str))
    true_edges = set(gt["Gene1"].astype(str) + "|" + gt["Gene2"].astype(str))

    if pred.empty:
        return {"EPR": np.nan, "Early_Precision": np.nan, "Early_Recall": np.nan, "TopK": 0}

    df = pred[["Gene1", "Gene2", "EdgeWeight"]].copy()
    df["Gene1"] = df["Gene1"].astype(str).str.strip()
    df["Gene2"] = df["Gene2"].astype(str).str.strip()
    df = df[df["Gene1"] != df["Gene2"]]
    df = df[df["Gene1"].isin(tfs) & df["Gene2"].isin(genes)]
    df = df.drop_duplicates(subset=["Gene1", "Gene2"], keep="first")
    df["EdgeWeight"] = pd.to_numeric(df["EdgeWeight"], errors="coerce").abs()
    df = df.dropna(subset=["EdgeWeight"])

    n_true = len(true_edges)
    maxk = min(len(df), n_true)
    if maxk <= 0 or n_true <= 0:
        return {"EPR": np.nan, "Early_Precision": np.nan, "Early_Recall": np.nan, "TopK": 0}

    top = df.sort_values("EdgeWeight", ascending=False).head(maxk)
    pred_edges = set(top["Gene1"] + "|" + top["Gene2"])
    hits = len(pred_edges & true_edges)
    eprec = hits / len(pred_edges)
    erec = hits / n_true
    possible = len(tfs) * len(genes) - len(tfs)
    random_prec = n_true / possible if possible > 0 else 0.0
    epr = eprec / random_prec if random_prec > 0 else np.nan

    return {
        "EPR": float(epr),
        "Early_Precision": float(eprec),
        "Early_Recall": float(erec),
        "TopK": int(maxk),
        "Correct": int(hits),
    }


def undirected_key(g1: str, g2: str) -> Tuple[str, str]:
    return (g1, g2) if g1 <= g2 else (g2, g1)


def tf_forward_lookup(df: pd.DataFrame, tfs: Set[str]) -> Dict[Tuple[str, str], float]:
    lk: Dict[Tuple[str, str], float] = {}
    for row in df.itertuples(index=False):
        g1, g2, w = str(row.Gene1), str(row.Gene2), float(row.EdgeWeight)
        if g1 in tfs:
            e = (g1, g2)
            lk[e] = max(lk.get(e, 0.0), w)
    return lk


def resolve_label_free_top_m(
    per_head_lk: Dict[int, Dict[Tuple[str, str], float]],
    export_top_m: int,
) -> int:
    """
    Top-M for per-head ranking (no GT).
    export_top_m > 0: fixed M; else M = min_h |TF-edges in head h|.
    """
    if export_top_m > 0:
        return int(export_top_m)
    counts = [len(lk) for lk in per_head_lk.values() if lk]
    return int(min(counts)) if counts else 0


def per_head_topm_tf_edges(
    per_head_lk: Dict[int, Dict[Tuple[str, str], float]],
    top_m: int,
) -> Dict[int, Dict[Tuple[str, str], float]]:
    out: Dict[int, Dict[Tuple[str, str], float]] = {}
    for h, lk in per_head_lk.items():
        items = sorted(lk.items(), key=lambda x: x[1], reverse=True)
        m = min(top_m, len(items)) if top_m > 0 else len(items)
        out[h] = {e: s for e, s in items[:m]}
    return out


def to_tf_forward_df(
    scores: Dict[Tuple[str, str], float],
    tfs: Set[str],
) -> pd.DataFrame:
    rows = []
    for (a, b), w in scores.items():
        if a in tfs:
            g1, g2 = a, b
        elif b in tfs:
            g1, g2 = b, a
        else:
            continue
        rows.append({"Gene1": g1, "Gene2": g2, "EdgeWeight": w})
    return pd.DataFrame(rows)


def fuse_mean8(per_head: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    return ev.fuse_predictions(per_head, "mean")


def fuse_max8(per_head: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    return ev.fuse_predictions(per_head, "max")


def fuse_semantic_unique(
    per_head: Dict[int, pd.DataFrame],
    tfs: Set[str],
    top_m: int,
    alpha: float = 1.5,
) -> pd.DataFrame:
    """Boost edges in exactly one head's label-free top-M; shared edges = mean."""
    per_head_lk = {h: tf_forward_lookup(df, tfs) for h, df in per_head.items()}
    topm = per_head_topm_tf_edges(per_head_lk, top_m)

    edge_support: Dict[Tuple[str, str], int] = {}
    for h_edges in topm.values():
        for e in h_edges:
            edge_support[e] = edge_support.get(e, 0) + 1

    acc: Dict[Tuple[str, str], List[float]] = {}
    for h, df in per_head.items():
        for row in df.itertuples(index=False):
            key = undirected_key(str(row.Gene1), str(row.Gene2))
            acc.setdefault(key, []).append(float(row.EdgeWeight))

    scores: Dict[Tuple[str, str], float] = {}
    for key, ws in acc.items():
        a, b = key
        e_tf = (a, b) if a in tfs else (b, a) if b in tfs else None
        base = float(np.mean(ws))
        if e_tf is not None and edge_support.get(e_tf, 0) == 1:
            scores[key] = base * alpha
        else:
            scores[key] = base

    return to_tf_forward_df(scores, tfs)


def tf_out_mass_symmax(df: pd.DataFrame, tfs: Set[str]) -> Dict[str, float]:
    """Per-TF outgoing mass (sym_max), same idea as plot_head_l2b_bridge."""
    lk = ev.build_tf_lookup(df, tfs, "sym_max")
    mass: Dict[str, float] = {}
    for (tf, _), w in lk.items():
        mass[tf] = mass.get(tf, 0.0) + float(w)
    return mass


def head_hub_coherent_tfs(
    per_head: Dict[int, pd.DataFrame],
    tfs: Set[str],
) -> Dict[str, int]:
    """
    Rule B: TF t → head h if t is primary in-hub (max Gene2 mass) and
    top TF-out (sym_max mass) on that head is also t.
    """
    coherent: Dict[str, int] = {}
    for h, df in per_head.items():
        if df.empty:
            continue
        in_mass = df.groupby("Gene2")["EdgeWeight"].sum()
        if in_mass.empty:
            continue
        in_hub = str(in_mass.idxmax())
        if in_hub not in tfs:
            continue
        tf_out = tf_out_mass_symmax(df, tfs)
        if not tf_out:
            continue
        top_tf = max(tf_out, key=tf_out.get)
        if in_hub == top_tf:
            coherent[in_hub] = h
    return coherent


def tf_out_strength_by_head(
    per_head: Dict[int, pd.DataFrame],
    tfs: Set[str],
) -> Dict[int, Dict[str, float]]:
    return {h: tf_out_mass_symmax(df, tfs) for h, df in per_head.items()}


def fuse_routed_label_free(
    per_head: Dict[int, pd.DataFrame],
    tfs: Set[str],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Per-TF head routing (no GT):
      1) hub-coherent TF → that head
      2) else TF → head with max TF out-strength (sum of Gene1=TF weights)
    """
    coherent = head_hub_coherent_tfs(per_head, tfs)
    tf_out = tf_out_strength_by_head(per_head, tfs)
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

    parts: List[pd.DataFrame] = []
    for tf, h in tf_to_head.items():
        if h not in per_head:
            continue
        sub = per_head[h]
        sub = sub[sub["Gene1"] == tf]
        if not sub.empty:
            parts.append(sub[["Gene1", "Gene2", "EdgeWeight"]].copy())

    # TFs never assigned: fall back to global strongest head by total TF-out mass
    if len(tf_to_head) < len(tfs):
        total_by_head = {
            h: sum(d.values()) for h, d in tf_out.items() if d
        }
        if total_by_head:
            fallback_h = max(total_by_head, key=total_by_head.get)
            for tf in tfs:
                if tf in tf_to_head:
                    continue
                sub = per_head[fallback_h]
                sub = sub[sub["Gene1"] == tf]
                if not sub.empty:
                    parts.append(sub[["Gene1", "Gene2", "EdgeWeight"]].copy())
                    tf_to_head[tf] = fallback_h

    if not parts:
        return pd.DataFrame(columns=["Gene1", "Gene2", "EdgeWeight"]), tf_to_head
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["Gene1", "Gene2"], keep="first")
    return out, tf_to_head


def build_tf_to_head_map(
    per_head: Dict[int, pd.DataFrame],
    tfs: Set[str],
) -> Dict[str, int]:
    """Label-free routing map π(t) without assembling edges."""
    coherent = head_hub_coherent_tfs(per_head, tfs)
    tf_out = tf_out_strength_by_head(per_head, tfs)
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

    if len(tf_to_head) < len(tfs):
        total_by_head = {h: sum(d.values()) for h, d in tf_out.items() if d}
        if total_by_head:
            fallback_h = max(total_by_head, key=total_by_head.get)
            for tf in tfs:
                if tf not in tf_to_head:
                    tf_to_head[tf] = fallback_h
    return tf_to_head


def fuse_routed_symmax(
    per_head: Dict[int, pd.DataFrame],
    tfs: Set[str],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Per-TF routed head (label-free) + sym_max(TF, target) weights from that head only.
    """
    tf_to_head = build_tf_to_head_map(per_head, tfs)
    rows: List[dict] = []
    for tf, h in tf_to_head.items():
        if h not in per_head:
            continue
        lk = ev.build_tf_lookup(per_head[h], tfs, "sym_max")
        for (g1, g2), w in lk.items():
            if g1 == tf:
                rows.append({"Gene1": g1, "Gene2": g2, "EdgeWeight": float(w)})
    if not rows:
        return pd.DataFrame(columns=["Gene1", "Gene2", "EdgeWeight"]), tf_to_head
    out = pd.DataFrame(rows).drop_duplicates(subset=["Gene1", "Gene2"], keep="first")
    return out, tf_to_head


def evaluate_dataset(
    model: str,
    dataset: str,
    head_roots: List[Path],
    chip_root: Path,
    fusion_alpha: float,
    export_top_m: int,
    min_frac_nonzero: float,
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
        gt = gt_all

    tfs, _ = ev.chip_gene_sets_from_gt(gt)
    gene_universe = set(gt["Gene1"].astype(str)) | set(gt["Gene2"].astype(str))

    files = ath.find_head_files(head_roots, model, dataset)
    if not files:
        raise FileNotFoundError(f"No head TSV for {model} {dataset} under {head_roots}")

    per_head: Dict[int, pd.DataFrame] = {}
    per_head_lk: Dict[int, Dict[Tuple[str, str], float]] = {}
    for fp in files:
        h = ath.parse_head_number(fp)
        raw = ev.load_head_pred(fp, gene_universe, 0)
        per_head[h] = ev.filter_pred_aupr_style(raw, gt)
        per_head_lk[h] = tf_forward_lookup(per_head[h], tfs)

    top_m = resolve_label_free_top_m(per_head_lk, export_top_m)
    rows: List[Dict] = []

    pred_mean = fuse_mean8(per_head)
    pred_max = fuse_max8(per_head)
    pred_fuse = fuse_semantic_unique(per_head, tfs, top_m, alpha=fusion_alpha)
    pred_routed, tf_map = fuse_routed_label_free(per_head, tfs)

    results = {
        "mean8": compute_epr(pred_mean, gt),
        "max8": compute_epr(pred_max, gt),
        "fusion": compute_epr(pred_fuse, gt),
        "routed": compute_epr(pred_routed, gt),
    }

    for strategy, m in results.items():
        rows.append(
            {
                "dataset": dataset,
                "strategy": strategy,
                "label_free": True,
                "export_top_m": top_m,
                "fusion_alpha": fusion_alpha if strategy == "fusion" else np.nan,
                "n_tf_routed": len(tf_map) if strategy == "routed" else np.nan,
                **m,
            }
        )

    print(
        f"  {dataset} (top-M={top_m}, no GT in assembly): "
        f"mean8={results['mean8']['EPR']:.4f}  "
        f"max8={results['max8']['EPR']:.4f}  "
        f"fusion={results['fusion']['EPR']:.4f}  "
        f"routed={results['routed']['EPR']:.4f}"
    )
    return rows


def plot_epr_bar(
    summary: pd.DataFrame,
    model_display: str,
    out_pdf: Path,
) -> None:
    """Grouped bar: 6 datasets × 4 label-free strategies."""
    main = summary[summary["strategy"].isin(MAIN_STRATEGIES)].copy()
    datasets = [d for d in DEFAULT_CHIP_DATASETS if d in set(main["dataset"])]
    if not datasets:
        datasets = sorted(main["dataset"].unique())

    colors = {
        "mean8": "#B0B0B0",
        "max8": "#C8D8EB",
        "fusion": model_color(model_display, "#8FB4DC"),
        "routed": "#EB7E60",
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
            vals.append(float(sub["EPR"].iloc[0]) if not sub.empty else 0.0)
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
                    b.get_height() + 0.015,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel("EPR (Early Precision Ratio)")
    ax.set_title(
        f"{model_display} — CHIP EPR (label-free GRN assembly)\n"
        "GT used only for evaluation; routing/fusion use attention only",
        fontsize=10,
    )
    ax.axhline(1.0, color="#888", linestyle="--", linewidth=0.8)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.92, ncol=2)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Label-free EPR: mean8 / max8 / fusion / per-TF routed (6 CHIP datasets)."
    )
    p.add_argument("--models", default="scgpt")
    p.add_argument("--datasets", default=",".join(DEFAULT_CHIP_DATASETS))
    p.add_argument("--head-root", action="append", default=None)
    p.add_argument("--chip-root", default="/mnt/10T/yzn/benchmark_GRN/input_process/CHIP")
    p.add_argument("--output-dir", default=str(_SCRIPT_DIR / "output" / "head_chip_epr"))
    p.add_argument("--fusion-alpha", type=float, default=1.5)
    p.add_argument(
        "--export-top-m",
        type=int,
        default=0,
        help="Fixed top-M per head for fusion labeling (0=min_h |TF-edges_h|, no GT)",
    )
    p.add_argument("--min-frac-nonzero", type=float, default=0.05)
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
        print(f"\n[{tag}] label-free EPR ({len(datasets)} datasets)")
        for ds in datasets:
            try:
                all_rows.extend(
                    evaluate_dataset(
                        model,
                        ds,
                        head_roots,
                        chip_root,
                        args.fusion_alpha,
                        args.export_top_m,
                        args.min_frac_nonzero,
                    )
                )
            except FileNotFoundError as e:
                print(f"  [WARN] skip {ds}: {e}")

        if not all_rows:
            print(f"[WARN] no results for {tag}")
            continue

        summary = pd.DataFrame(all_rows)
        csv_path = out_dir / "epr_summary_label_free.csv"
        summary.to_csv(csv_path, index=False)

        plot_epr_bar(summary, display, out_dir / "epr_6datasets_label_free.pdf")
        plot_epr_bar(summary, display, out_dir / "epr_6datasets_label_free.png")

        print(f"[INFO] CSV  -> {csv_path}")
        print(f"[INFO] PDF  -> {out_dir / 'epr_6datasets_label_free.pdf'}")


if __name__ == "__main__":
    main()
