#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仅评估「单步方向预测错」的基因：在 |Δ| 峰值 transition 上对比能否纠正。

对比方法（真值 = sign(delta_best_trans)）
----------------------------------------
- single：pre_scgpt gene_result 的 dir_pred（20%→20%）
- indep_peak：独立段 scGPT 在该 transition 的 dir_pred
- chained_peak_local：链式该段末 pred_late_like − mean(S_i)（局部，与真值同尺度）
- chained_peak_s0：链式 pred_late_like − mean(S0)（相对起点）

示例：
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig4
  python3 evaluate_wrong_genes_peak.py --dataset hESC
  python3 evaluate_wrong_genes_peak.py --dataset hESC --nonchip-only
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
MULTISTEP_ROOT = SCRIPT_DIR / "error_biology" / "multistep_pt"

from peak_method_metrics import early_local_mean, load_chained_trans
GENE_RESULT_DIR = Path("/mnt/10T/yzn/benchmark_GRN/pre_scgpt/results_multidataset_pseudotime_227")
CHIP_DIR = Path("/mnt/10T/yzn/benchmark_GRN/input_process/CHIP")
TOP_PERCENT = 0.3

try:
    from fig4_palette import apply_fig4_style, model_color
except ImportError:

    def apply_fig4_style() -> None:
        plt.rcParams.update({"font.size": 11})

    def model_color(_: str, d: str = "#666") -> str:
        return d


def parse_transition(trans: str) -> Optional[int]:
    m = re.match(r"delta_t(\d+)_t(\d+)", str(trans))
    return int(m.group(1)) if m else None


def dir_sign(x: float) -> str:
    if not np.isfinite(x) or x == 0:
        return "Down"
    return "Up" if x > 0 else "Down"


def load_indep_at_peak(indep_dir: Path, dataset: str, seg_i: int) -> Optional[pd.DataFrame]:
    for path in (
        indep_dir / "scgpt" / f"{dataset}_seg{seg_i}_to_{seg_i + 1}_gene_result.csv",
        indep_dir / f"{dataset}_seg{seg_i}_to_{seg_i + 1}_gene_result.csv",
    ):
        if path.exists():
            return pd.read_csv(path).set_index("gene")
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="hESC")
    p.add_argument("--n-segments", type=int, default=5)
    p.add_argument("--nonchip-only", action="store_true")
    p.add_argument("--top-percent", type=float, default=TOP_PERCENT)
    p.add_argument("--outdir", type=Path, default=MULTISTEP_ROOT)
    args = p.parse_args()

    ds = args.dataset
    ds_dir = args.outdir / ds
    gf = pd.read_csv(ds_dir / "gene_segment_deltas.csv")
    gr = pd.read_csv(GENE_RESULT_DIR / f"{ds}_gene_result.csv")
    chip = set(pd.read_csv(CHIP_DIR / f"{ds}_chip_matched-network.csv")["Gene2"].astype(str))

    n = max(1, int(np.ceil(args.top_percent * len(gf))))
    top = gf.assign(_abs=gf["delta_single"].abs()).nlargest(n, "_abs").copy()
    top = top.merge(
        gr[["gene", "dir_pred", "dir_correct", "delta_pred"]].rename(
            columns={"dir_pred": "dir_pred_single", "dir_correct": "dir_correct_single"}
        ),
        on="gene",
        how="left",
    )
    top["is_chip_target"] = top["gene"].astype(str).isin(chip)
    top["dir_true_peak"] = top["delta_best_trans"].apply(dir_sign)

    wrong = top[top["dir_correct_single"] == 0].copy()
    if args.nonchip_only:
        wrong = wrong[~wrong["is_chip_target"]].copy()

    chained_dir = ds_dir / "chained_preds"
    indep_dir = ds_dir / "segment_preds"
    ch_s0 = load_chained_trans(chained_dir, ds, "delta_t0_t1")
    if ch_s0 is None:
        raise FileNotFoundError(f"Missing chained S0→S1: {chained_dir}")

    rows = []
    for _, r in wrong.iterrows():
        g = str(r["gene"])
        trans = str(r["best_transition"])
        seg_i = parse_transition(trans)
        if seg_i is None:
            continue
        true_dir = r["dir_true_peak"]

        row = {
            "gene": g,
            "is_chip_target": bool(r["is_chip_target"]),
            "best_transition": trans,
            "peak_middle": bool(r["peak_middle"]),
            "dir_true_peak": true_dir,
            "dir_pred_single": r["dir_pred_single"],
            "delta_best_trans": float(r["delta_best_trans"]),
        }

        # 链式：pred_late_like ≈ 该段迭代末的细胞平均预测表达
        ch = load_chained_trans(chained_dir, ds, trans)
        if ch is not None and g in ch.index:
            pred_late = float(ch.loc[g, "pred_late_like_mean"])
            early_local = early_local_mean(ch, g)
            early_s0 = early_local_mean(ch_s0, g) if ch_s0 is not None else np.nan
            row["pred_late_chained"] = pred_late
            row["dir_pred_chained_local"] = dir_sign(pred_late - early_local)
            row["dir_pred_chained_s0"] = dir_sign(pred_late - early_s0) if np.isfinite(early_s0) else ""
            row["ok_chained_local"] = row["dir_pred_chained_local"] == true_dir
            row["ok_chained_s0"] = row["dir_pred_chained_s0"] == true_dir
        else:
            row["ok_chained_local"] = np.nan
            row["ok_chained_s0"] = np.nan

        ind = load_indep_at_peak(indep_dir, ds, seg_i)
        if ind is not None and g in ind.index:
            row["dir_pred_indep"] = str(ind.loc[g, "dir_pred"])
            row["ok_indep"] = row["dir_pred_indep"] == true_dir
        else:
            row["ok_indep"] = np.nan

        row["fixed_indep"] = row.get("ok_indep") is True or row.get("ok_indep") == 1
        row["fixed_chained_local"] = row.get("ok_chained_local") is True or row.get("ok_chained_local") == 1
        row["fixed_chained_s0"] = row.get("ok_chained_s0") is True or row.get("ok_chained_s0") == 1
        rows.append(row)

    out = pd.DataFrame(rows)
    out_dir = ds_dir / "wrong_genes_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "wrong_genes_peak_comparison.csv", index=False)

    n = len(out)
    def rate(col: str) -> float:
        s = out[col].dropna()
        return float(s.mean()) if len(s) else float("nan")

    summary = pd.DataFrame(
        [
            {
                "dataset": ds,
                "subset": "nonchip" if args.nonchip_only else "all_wrong",
                "n_wrong": n,
                "frac_fixed_indep_peak": rate("fixed_indep"),
                "frac_fixed_chained_local_peak": rate("fixed_chained_local"),
                "frac_fixed_chained_s0_peak": rate("fixed_chained_s0"),
                "n_fixed_indep": int(out["fixed_indep"].sum()),
                "n_fixed_chained_local": int(out["fixed_chained_local"].sum()),
                "n_fixed_chained_s0": int(out["fixed_chained_s0"].sum()),
            }
        ]
    )
    summary.to_csv(out_dir / "wrong_genes_peak_summary.csv", index=False)

    # 简单柱状图
    apply_fig4_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Indep.\n@ peak", "Chained\n(local Si)", "Chained\n(vs S0)"]
    vals = [rate("fixed_indep") * 100, rate("fixed_chained_local") * 100, rate("fixed_chained_s0") * 100]
    colors = [model_color("scPrint"), model_color("scGPT"), model_color("scFoundation")]
    ax.bar(labels, vals, color=colors, edgecolor="white")
    ax.set_ylabel("Fixed / wrong-single genes (%)")
    ax.set_ylim(0, 100)
    sub = "non-CHIP " if args.nonchip_only else ""
    ax.set_title(f"{ds} — {sub}single-step wrong → fixed at peak transition (n={n})")
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "wrong_genes_fix_rate.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\n=== {ds} | 单步方向错 → 峰值段能否纠正 ===")
    print(f"子集: {'非 CHIP' if args.nonchip_only else '全部 top30% 动态基因中 dir_correct=0'}")
    print(f"n = {n}")
    print(f"  独立段 @ 峰值 transition:  {int(out['fixed_indep'].sum())}/{n}  ({rate('fixed_indep'):.1%})")
    print(f"  链式 @ 峰值 (pred−Si):    {int(out['fixed_chained_local'].sum())}/{n}  ({rate('fixed_chained_local'):.1%})")
    print(f"  链式 @ 峰值 (pred−S0):    {int(out['fixed_chained_s0'].sum())}/{n}  ({rate('fixed_chained_s0'):.1%})")
    print(f"\nSaved: {out_dir}/wrong_genes_peak_comparison.csv")
    print(f"       {out_dir}/wrong_genes_fix_rate.png")


if __name__ == "__main__":
    main()
