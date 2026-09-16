#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对「单步 scGPT 预测错」的基因：先识别 |Δ| 峰值 transition，再用该段的 scGPT 迭代结果重评方向。

对比
----
- **single**：20% early vs 20% late 一次迭代（gene_result.csv）
- **peak_segment**：在 best_transition 对应相邻段 (Si→Si+1) 上单独跑 scGPT 迭代
- **oracle_peak**（对照）：真值在峰值段的符号（理论上界，看错误是否因“评错段”）

真值默认用峰值段：**dir_true_peak** = sign(delta_best_trans)。

依赖
----
两种分段预测（二选一，含义不同）：

1. **独立段**（每段单独 16 步，互不继承）— 旧方案
   python3 run_scgpt_pt_segments.py --dataset hESC

2. **链式段**（一次预测，每走完一段保存结果）— 推荐与你描述一致
   python3 run_scgpt_chained_segments.py --dataset hESC

评估（读 segment_preds/scgpt/ 下独立段结果）：
  python3 evaluate_peak_segment_scgpt.py --dataset hESC --run-segments

示例
----
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig4
  python3 evaluate_peak_segment_scgpt.py --dataset hESC
  python3 evaluate_peak_segment_scgpt.py --dataset hESC --subset wrong_single
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
MULTISTEP_ROOT = SCRIPT_DIR / "error_biology" / "multistep_pt"
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


def transition_to_seg_pair(trans: str) -> Optional[Tuple[int, int]]:
    m = re.match(r"delta_t(\d+)_t(\d+)", str(trans))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def dir_from_delta(x: float) -> str:
    if not np.isfinite(x) or x == 0:
        return "Down"
    return "Up" if x > 0 else "Down"


def load_segment_gene_results(seg_dir: Path, dataset: str, n_segments: int) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for i in range(n_segments - 1):
        trans = f"delta_t{i}_t{i + 1}"
        for path in (
            seg_dir / "scgpt" / f"{dataset}_seg{i}_to_{i + 1}_gene_result.csv",
            seg_dir / f"{dataset}_seg{i}_to_{i + 1}_gene_result.csv",
        ):
            if path.exists():
                out[trans] = pd.read_csv(path)
                break
    return out


def select_top_dynamic(gene_df: pd.DataFrame, top_percent: float = TOP_PERCENT) -> pd.DataFrame:
    n = max(1, int(np.ceil(top_percent * len(gene_df))))
    gene_df = gene_df.copy()
    gene_df["abs_single"] = gene_df["delta_single"].abs()
    return gene_df.nlargest(n, "abs_single")


def run_segments_subprocess(dataset: str, n_segments: int, out_root: Path, gen_iters: int) -> None:
    script = SCRIPT_DIR / "run_scgpt_pt_segments.py"
    cmd = [
        sys.executable,
        str(script),
        "--dataset",
        dataset,
        "--n-segments",
        str(n_segments),
        "--outdir",
        str(out_root),
        "--gen-iters",
        str(gen_iters),
    ]
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def evaluate(
    dataset: str,
    *,
    n_segments: int,
    seg_dir: Path,
    gene_result_path: Path,
    gene_segment_path: Path,
    outdir: Path,
    subset: str,
    nonchip_only: bool,
) -> pd.DataFrame:
    if not gene_segment_path.exists():
        raise FileNotFoundError(f"Missing {gene_segment_path}; run analyze_multistep_pseudotime.py first.")
    if not gene_result_path.exists():
        raise FileNotFoundError(f"Missing {gene_result_path}")

    gene_full = pd.read_csv(gene_segment_path)
    chip = load_chip_tf_targets(CHIP_DIR / f"{dataset}_chip_matched-network.csv")
    gene_full["is_chip_target"] = gene_full["gene"].astype(str).isin(chip)

    top = select_top_dynamic(gene_full)
    single = pd.read_csv(gene_result_path)
    single = single.rename(
        columns={
            "dir_pred": "dir_pred_single",
            "dir_correct": "dir_correct_single",
            "delta_pred": "delta_pred_single",
            "dir_true": "dir_true_single_file",
        }
    )
    top = top.merge(
        single[
            [
                "gene",
                "dir_pred_single",
                "dir_correct_single",
                "delta_pred_single",
                "true_early_mean",
                "true_late_mean",
                "pred_late_like_mean",
            ]
        ],
        on="gene",
        how="left",
    )

    top["dir_true_peak"] = top["delta_best_trans"].apply(dir_from_delta)
    top["dir_true_single"] = top["delta_single"].apply(dir_from_delta)
    top["dir_same_single_peak"] = top["dir_true_single"] == top["dir_true_peak"]

    seg_preds = load_segment_gene_results(seg_dir, dataset, n_segments)
    missing_trans = [c for c in top["best_transition"].dropna().unique() if c not in seg_preds]

    peak_dir: List[str] = []
    peak_delta_pred: List[float] = []
    peak_correct: List[float] = []
    for _, row in top.iterrows():
        trans = row["best_transition"]
        g = str(row["gene"])
        if pd.isna(trans) or trans not in seg_preds:
            peak_dir.append(np.nan)
            peak_delta_pred.append(np.nan)
            peak_correct.append(np.nan)
            continue
        gr = seg_preds[trans].set_index("gene")
        if g not in gr.index:
            peak_dir.append(np.nan)
            peak_delta_pred.append(np.nan)
            peak_correct.append(np.nan)
            continue
        peak_dir.append(str(gr.loc[g, "dir_pred"]) if "dir_pred" in gr.columns else dir_from_delta(float(gr.loc[g, "delta_pred"])))
        peak_delta_pred.append(float(gr.loc[g, "delta_pred"]))
        d_true = row["dir_true_peak"]
        peak_correct.append(1.0 if str(peak_dir[-1]).lower() == str(d_true).lower() else 0.0)

    top["dir_pred_peak_segment"] = peak_dir
    top["delta_pred_peak_segment"] = peak_delta_pred
    top["dir_correct_peak_segment"] = peak_correct

    top["wrong_single"] = top["dir_correct_single"] == 0
    top["wrong_peak"] = top["dir_correct_peak_segment"] == 0
    ok_peak = top["dir_pred_peak_segment"].notna()
    top["fixed_by_peak"] = False
    top["broken_by_peak"] = False
    top.loc[ok_peak, "fixed_by_peak"] = top.loc[ok_peak, "wrong_single"] & (
        top.loc[ok_peak, "dir_correct_peak_segment"] == 1
    )
    top.loc[ok_peak, "broken_by_peak"] = (~top.loc[ok_peak, "wrong_single"]) & (
        top.loc[ok_peak, "dir_correct_peak_segment"] == 0
    )

    work = top[~top["is_chip_target"]].copy() if nonchip_only else top.copy()
    if subset == "wrong_single":
        work = work[work["wrong_single"]].copy()
    elif subset == "peak_middle":
        work = work[work["peak_middle"]].copy()
    elif subset == "wrong_single_nonchip_peak_middle":
        work = work[work["wrong_single"] & work["peak_middle"] & ~work["is_chip_target"]].copy()

    def err_rate(df: pd.DataFrame, col: str, true_col: str = "dir_true_peak") -> float:
        m = df[col].notna() & df[true_col].notna()
        if not m.any():
            return float("nan")
        return float(
            (df.loc[m, col].astype(str).str.lower() != df.loc[m, true_col].astype(str).str.lower()).mean()
        )

    metrics = []
    for label, sub in [
        ("all_top30", work),
        ("nonchip", work[~work["is_chip_target"]]),
        ("wrong_single", work[work["wrong_single"]]),
        ("wrong_single_nonchip", work[work["wrong_single"] & ~work["is_chip_target"]]),
    ]:
        if sub.empty:
            continue
        metrics.append(
            {
                "dataset": dataset,
                "group": label,
                "n": len(sub),
                "n_with_peak_pred": int(sub["dir_pred_peak_segment"].notna().sum()),
                "dir_error_single": err_rate(sub, "dir_pred_single", "dir_true_peak"),
                "dir_error_peak_segment": err_rate(sub, "dir_pred_peak_segment", "dir_true_peak"),
                "frac_wrong_single": float(sub["wrong_single"].mean()),
                "frac_fixed_among_wrong": float(sub.loc[sub["wrong_single"], "fixed_by_peak"].mean())
                if sub["wrong_single"].any()
                else float("nan"),
                "frac_broken_among_correct": float(sub.loc[~sub["wrong_single"], "broken_by_peak"].mean())
                if (~sub["wrong_single"]).any()
                else float("nan"),
            }
        )

    metrics_df = pd.DataFrame(metrics)
    ds_out = outdir / dataset
    ds_out.mkdir(parents=True, exist_ok=True)
    top.to_csv(ds_out / "peak_segment_prediction_eval.csv", index=False)
    metrics_df.to_csv(ds_out / "peak_segment_metrics.csv", index=False)

    if missing_trans:
        (ds_out / "missing_segment_transitions.txt").write_text("\n".join(sorted(missing_trans)))

    plot_comparison(metrics_df, dataset, ds_out / "peak_segment_fix_bars.png")
    plot_wrong_gene_flow(top[~top["is_chip_target"]], dataset, ds_out / "peak_segment_wrong_gene_flow.png")

    print(f"\n=== {dataset} peak-segment scGPT evaluation ===")
    print(metrics_df.to_string(index=False))
    if missing_trans:
        print(f"\n[WARN] Missing segment scGPT runs for: {missing_trans}")
        print("  Run: python3 run_scgpt_pt_segments.py --dataset", dataset)
    print(f"\nSaved: {ds_out}")
    return top


def load_chip_tf_targets(path: Path) -> set:
    df = pd.read_csv(path)
    col = "Gene2" if "Gene2" in df.columns else df.columns[1]
    return set(df[col].astype(str))


def plot_comparison(metrics_df: pd.DataFrame, dataset: str, outpath: Path) -> None:
    apply_fig4_style()
    sub = metrics_df[metrics_df["group"].isin(["nonchip", "wrong_single_nonchip", "all_top30"])]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(sub))
    w = 0.35
    ax.bar(x - w / 2, sub["dir_error_single"] * 100, w, label="Single-step scGPT", color=model_color("scPrint"))
    ax.bar(
        x + w / 2,
        sub["dir_error_peak_segment"] * 100,
        w,
        label="Peak-segment scGPT",
        color=model_color("scGPT"),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(sub["group"], rotation=15)
    ax.set_ylabel("Direction error rate (%)")
    ax.set_title(f"{dataset} — single vs peak-segment prediction (true dir @ peak transition)")
    ax.legend(frameon=False)
    for i, row in sub.iterrows():
        ax.text(x[list(sub.index).index(i)], max(row["dir_error_single"], row["dir_error_peak_segment"]) * 100 + 2,
                f"n={int(row['n'])}", ha="center", fontsize=8)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_wrong_gene_flow(top: pd.DataFrame, dataset: str, outpath: Path) -> None:
    """单步预测错 → 峰值段重预测 的修复/仍错/新错。"""
    sub = top[~top["is_chip_target"] & top["dir_pred_peak_segment"].notna()].copy()
    if sub.empty:
        return
    wrong = sub["wrong_single"]
    fixed = sub["fixed_by_peak"].fillna(False)
    n_wrong = int(wrong.sum())
    n_fixed = int((wrong & fixed).sum())
    n_still_wrong = n_wrong - n_fixed
    n_ok = int((~wrong).sum())
    n_broken = int((~wrong & (sub["dir_correct_peak_segment"] == 0)).sum())
    n_still_ok = n_ok - n_broken

    apply_fig4_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Still wrong", "Fixed by peak seg", "Still correct", "New error"]
    vals = [n_still_wrong, n_fixed, n_still_ok, n_broken]
    colors = ["#c0392b", "#27ae60", "#2980b9", "#e67e22"]
    ax.barh(labels, vals, color=colors)
    ax.set_xlabel("Gene count (non-CHIP, has peak-segment pred)")
    ax.set_title(f"{dataset} — outcome of re-predicting at peak transition")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="hESC")
    p.add_argument("--n-segments", type=int, default=5)
    p.add_argument("--outdir", type=Path, default=MULTISTEP_ROOT)
    p.add_argument("--subset", default="all", choices=["all", "wrong_single", "peak_middle", "wrong_single_nonchip_peak_middle"])
    p.add_argument("--nonchip-only", action="store_true")
    p.add_argument("--run-segments", action="store_true", help="Call run_scgpt_pt_segments.py first (GPU)")
    p.add_argument("--gen-iters", type=int, default=16)
    args = p.parse_args()

    ds_dir = args.outdir / args.dataset
    seg_dir = ds_dir / "segment_preds"
    if args.run_segments:
        run_segments_subprocess(args.dataset, args.n_segments, args.outdir, args.gen_iters)

    evaluate(
        args.dataset,
        n_segments=args.n_segments,
        seg_dir=seg_dir,
        gene_result_path=GENE_RESULT_DIR / f"{args.dataset}_gene_result.csv",
        gene_segment_path=ds_dir / "gene_segment_deltas.csv",
        outdir=args.outdir,
        subset=args.subset,
        nonchip_only=args.nonchip_only,
    )


if __name__ == "__main__":
    main()
