#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Case study: CHIP true edges weakened or dropped when using mean8 vs single heads.

Two operational definitions (both useful in a paper):
  A) top-k membership: edge is in head-h top-k (TF-aligned) but NOT in mean8 top-k
  B) score delta:      delta = max_head(score) - mean8(score) on CHIP true edges

Outputs:
  output/head_chip_auprc/{tag}/{tag}_mean_wiped_chip_edges.csv   (full table)
  output/head_chip_auprc/{tag}/{tag}_mean_wiped_case_study.txt   (top N narratives)

Usage:
  cd FBplot/fig2/plot
  python case_study_mean_delta_edges.py \\
    --models scgpt --dataset hESC \\
    --head-root /mnt/10T/yzn/scGRN-Bench/FBplot/fig2/att_head \\
    --top-edges 10000 --top-case 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import eval_heads_chip_auprc as chip  # noqa: E402
import analyze_attention_heads as ath  # noqa: E402


def topk_tf_pairs(lookup: Dict[Tuple[str, str], float], k: int) -> Set[Tuple[str, str]]:
    if k <= 0 or not lookup:
        return set(lookup.keys())
    items = sorted(lookup.items(), key=lambda x: x[1], reverse=True)[:k]
    return {e for e, _ in items}


def global_rank(lookup: Dict[Tuple[str, str], float], edge: Tuple[str, str]) -> int:
    """1 = highest score among all pairs in lookup."""
    if edge not in lookup:
        return len(lookup) + 1
    items = sorted(lookup.items(), key=lambda x: x[1], reverse=True)
    for i, (e, _) in enumerate(items, start=1):
        if e == edge:
            return i
    return len(lookup) + 1


def run(model: str, args: argparse.Namespace) -> None:
    tag = f"{ath.model_file_prefix(model)}_{args.dataset}"
    chip_root = Path(args.chip_root)
    out_dir = Path(args.output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = chip.read_chip_gt(chip_root / f"{args.dataset}_chip_matched-network.csv")
    expr = chip.read_chip_expr(chip_root / f"{args.dataset}_chip_matched-ExpressionData.csv")
    active = chip.active_genes(expr, args.min_frac_nonzero)
    gt = gt[gt["Gene1"].isin(active) & gt["Gene2"].isin(active)].copy()
    tf_set = set(gt["Gene1"].astype(str))
    k_eval = chip.resolve_top_k(len(gt), args.top_edges)

    files = ath.find_head_files([Path(p) for p in args.head_root], model, args.dataset)
    if not files:
        print(f"[WARN] no head files for {tag}")
        return

    print(f"[INFO] Top-K = {k_eval} (|CHIP GT|={len(gt)}); AUPR filter Gene1∈TF, Gene2∈Genes")

    per_head_lookup: Dict[int, Dict[Tuple[str, str], float]] = {}
    per_head_topk: Dict[int, Set[Tuple[str, str]]] = {}
    per_head_df: Dict[int, pd.DataFrame] = {}
    for fp in files:
        h = ath.parse_head_number(fp)
        raw = chip.load_head_pred(fp, active, args.load_max_edges)
        df = chip.filter_pred_aupr_style(raw, gt)
        per_head_df[h] = df
        lk = chip.build_tf_lookup(df, tf_set, args.pred_direction)
        per_head_lookup[h] = lk
        per_head_topk[h] = topk_tf_pairs(lk, k_eval)

    # mean8 on AUPR-filtered edges (then sym_max lookup)
    fused = chip.fuse_predictions(per_head_df, "mean")
    mean_lk = chip.build_tf_lookup(fused, tf_set, args.pred_direction)
    mean_topk = topk_tf_pairs(mean_lk, k_eval)

    rows = []
    for _, r in gt.iterrows():
        tf, tg = str(r["Gene1"]), str(r["Gene2"])
        e = (tf, tg)
        scores = {h: per_head_lookup[h].get(e, 0.0) for h in per_head_lookup}
        best_h = max(scores, key=scores.get)
        s_best = scores[best_h]
        s_mean = mean_lk.get(e, 0.0)
        delta = s_best - s_mean
        in_mean = e in mean_topk
        heads_in_topk = [h for h, tk in per_head_topk.items() if e in tk]
        wiped = bool(heads_in_topk) and not in_mean
        rows.append(
            {
                "TF": tf,
                "target": tg,
                "score_best_head": s_best,
                "best_head": best_h,
                "score_mean8": s_mean,
                "delta_max_minus_mean": delta,
                "rank_at_best_head": global_rank(per_head_lookup[best_h], e),
                "rank_at_mean8": global_rank(mean_lk, e),
                "in_topk_mean8": in_mean,
                "heads_in_topk": ",".join(f"H{h}" for h in sorted(heads_in_topk)),
                "n_heads_in_topk": len(heads_in_topk),
                "wiped_from_mean_topk": wiped,
            }
        )

    tab = pd.DataFrame(rows)

    # Sort for case studies: prefer wiped, then high delta, then good rank at best head
    case = tab[tab["wiped_from_mean_topk"]].copy()
    case = case.sort_values(
        ["delta_max_minus_mean", "rank_at_best_head"],
        ascending=[False, True],
    )
    case.to_csv(out_dir / f"{tag}_mean_wiped_chip_edges.csv", index=False)

    # Also save high-delta even if still in mean topk (soft wipe)
    tab.sort_values("delta_max_minus_mean", ascending=False).to_csv(
        out_dir / f"{tag}_chip_true_delta_scores.csv", index=False
    )

    n_wiped = int(case.shape[0])
    print(f"\n[{tag}] CHIP true edges: {len(tab)}")
    print(f"  In top-k of >=1 head but NOT in mean8 top-k: {n_wiped}")
    print(f"  (top-K={k_eval}, pred_direction={args.pred_direction})\n")

    topn = min(args.top_case, len(case))
    lines = [
        f"Case studies: CHIP edges in some head top-{k_eval} but missing from mean8 top-{k_eval}",
        f"Model={model}, dataset={args.dataset}",
        "=" * 72,
    ]
    for i in range(topn):
        r = case.iloc[i]
        lines.append(
            f"\n[{i+1}] {r.TF} -> {r.target}\n"
            f"    best: H{int(r.best_head)} score={r.score_best_head:.6g}  "
            f"rank={int(r.rank_at_best_head)} in H{int(r.best_head)} top-k universe\n"
            f"    mean8: score={r.score_mean8:.6g}  rank={int(r.rank_at_mean8)}  "
            f"in_mean8_topk={bool(r.in_topk_mean8)}\n"
            f"    delta(max-mean)={r.delta_max_minus_mean:.6g}  "
            f"also in top-k of: {r.heads_in_topk}"
        )
    if topn == 0:
        lines.append("\n(No wiped edges under this definition; try larger top-edges or check sym_max.)")

    # Picked examples for TFAP2A if present
    tfa = case[case["TF"] == "TFAP2A"].head(3)
    if not tfa.empty:
        lines.append("\n" + "-" * 72)
        lines.append("TFAP2A examples (often head1):")
        for _, r in tfa.iterrows():
            lines.append(f"  {r.TF}->{r.target}  H{int(r.best_head)}  delta={r.delta_max_minus_mean:.4g}")

    text = "\n".join(lines)
    out_txt = out_dir / f"{tag}_mean_wiped_case_study.txt"
    out_txt.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[INFO] CSV -> {out_dir / f'{tag}_mean_wiped_chip_edges.csv'}")
    print(f"[INFO] TXT -> {out_txt}")


def main() -> None:
    p = argparse.ArgumentParser(description="CHIP edges lost under mean8 vs per-head top-k.")
    p.add_argument("--models", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--head-root", action="append", required=True)
    p.add_argument("--chip-root", default="/mnt/10T/yzn/benchmark_GRN/input_process/CHIP")
    p.add_argument("--output-dir", default=str(_SCRIPT_DIR / "output" / "head_chip_auprc"))
    p.add_argument("--top-edges", type=int, default=0, help="Top-K; 0 = |CHIP GT| (AUPR convention)")
    p.add_argument("--load-max-edges", type=int, default=0, help="Max pred rows/head before AUPR filter (0=all)")
    p.add_argument("--pred-direction", default="sym_max", choices=["sym_max", "as_exported", "tf_gene1"])
    p.add_argument("--min-frac-nonzero", type=float, default=0.05)
    p.add_argument("--top-case", type=int, default=15, help="How many case lines to print")
    args = p.parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in args.models:
        run(m, args)


if __name__ == "__main__":
    main()
