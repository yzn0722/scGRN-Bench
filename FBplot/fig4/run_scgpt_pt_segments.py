#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对相邻伪时间分段 (S0→S1, …) 各跑一次 scGPT 迭代，汇总为 gene_trajectories.csv 供故事图三条线使用。

单步预测仍用既有 results_multidataset_pseudotime_227/{dataset}_gene_result.csv（20% early vs 20% late）。

示例：
  python3 run_scgpt_pt_segments.py --dataset hESC
  python3 run_scgpt_pt_segments.py --dataset hESC --skip-run   # 仅合并已有分段结果
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH = Path("/mnt/10T/yzn/benchmark_GRN")
UNIFIED = BENCH / "run_unified_multidataset_pseudotime.py"
SINGLE_STEP_DIR = BENCH / "pre_scgpt" / "results_multidataset_pseudotime_227"
CHIP_DIR = BENCH / "input_process" / "CHIP"
PT_ROOT = BENCH / "PseudoTime"

N_SEG_DEFAULT = 5


def mask_segment_bins(pt: np.ndarray, n_segments: int) -> List[np.ndarray]:
    edges = np.quantile(pt, np.linspace(0, 1, n_segments + 1))
    edges[-1] += 1e-9
    masks = []
    for i in range(n_segments):
        if i < n_segments - 1:
            masks.append((pt >= edges[i]) & (pt < edges[i + 1]))
        else:
            masks.append((pt >= edges[i]) & (pt <= edges[i + 1]))
    return masks


def bin_expr_to_0_50(x: np.ndarray, do_log1p: bool = False) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0)
    x = np.clip(x, 0, None)
    if do_log1p:
        x = np.log1p(x)
    vmax = max(np.percentile(x, 99.5), 1e-6)
    return np.clip(x / vmax * 50.0, 0, 50).astype(np.float32)


def load_pt_expr(dataset: str) -> tuple[pd.DataFrame, np.ndarray]:
    expr = pd.read_csv(CHIP_DIR / f"{dataset}_chip_matched-ExpressionData.csv", index_col=0)
    pt_df = pd.read_csv(PT_ROOT / dataset / "PseudoTime.csv")
    pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"}).set_index("cell")
    common = expr.columns.intersection(pt_df.index)
    expr = expr[common]
    pt = pt_df.loc[common, "pt"].astype(float).to_numpy()
    X = expr.T.to_numpy(dtype=np.float32)
    genes = expr.index.astype(str).tolist()
    return pd.DataFrame(X_bin=bin_expr_to_0_50(X), index=common, columns=genes), pt


def run_segment_pair(
    dataset: str,
    seg_early: int,
    seg_late: int,
    outdir: Path,
    n_segments: int,
    gen_iters: int,
    scgpt_legacy_pt: bool,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}_seg{seg_early}_to_{seg_late}"
    target = outdir / f"{stem}_gene_result.csv"
    if target.exists():
        print(f"  [skip] exists: {target.name}")
        return
    cmd = [
        sys.executable,
        str(UNIFIED),
        "--model",
        "scgpt",
        "--datasets",
        dataset,
        "--outdir",
        str(outdir),
        "--pt-mask",
        "segment_pair",
        "--n-pt-segments",
        str(n_segments),
        "--segment-early",
        str(seg_early),
        "--segment-late",
        str(seg_late),
        "--gen-iters",
        str(gen_iters),
    ]
    if scgpt_legacy_pt:
        cmd.append("--scgpt-legacy-pt")
    print(f"  [run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(BENCH))


def true_segment_means_binned(dataset: str, n_segments: int) -> pd.DataFrame:
    """每段 binned 表达均值（与 scGPT 输入空间一致）。"""
    df, pt = load_pt_expr(dataset)
    masks = mask_segment_bins(pt, n_segments)
    genes = df.columns.tolist()
    rows = {g: [] for g in genes}
    for m in masks:
        seg_mean = df.loc[m].mean(axis=0)
        for g in genes:
            rows[g].append(float(seg_mean[g]))
    out = pd.DataFrame({"gene": genes})
    for k in range(n_segments):
        out[f"true_seg{k}"] = [rows[g][k] for g in genes]
    return out


def merge_trajectories(
    dataset: str,
    seg_dir: Path,
    n_segments: int,
    single_csv: Path,
) -> pd.DataFrame:
    base = true_segment_means_binned(dataset, n_segments)
    n_trans = n_segments - 1

    # 分段迭代：pred 落在 transition 的 late 端 → seg_{i+1}
    pred_seg: Dict[str, List[float]] = {g: [np.nan] * n_segments for g in base["gene"]}
    for i in range(n_trans):
        path = seg_dir / "scgpt" / f"{dataset}_seg{i}_to_{i + 1}_gene_result.csv"
        if not path.exists():
            path = seg_dir / f"{dataset}_seg{i}_to_{i + 1}_gene_result.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing segment result: {path}")
        gr = pd.read_csv(path)
        gr = gr.set_index("gene")
        for g in base["gene"]:
            if g not in gr.index:
                continue
            if i == 0:
                pred_seg[g][0] = float(gr.loc[g, "true_early_mean"])
            pred_seg[g][i + 1] = float(gr.loc[g, "pred_late_like_mean"])

    for k in range(n_segments):
        base[f"pred_multistep_seg{k}"] = [pred_seg[g][k] for g in base["gene"]]

    # 单步：端点 + 线性插值（与 fig4 主评估一致）
    if not single_csv.exists():
        raise FileNotFoundError(f"Single-step gene_result not found: {single_csv}")
    single = pd.read_csv(single_csv).set_index("gene")
    for k in range(n_segments):
        col = f"pred_single_seg{k}"
        vals = []
        for g in base["gene"]:
            if g not in single.index:
                vals.append(np.nan)
                continue
            e = float(single.loc[g, "true_early_mean"])
            p = float(single.loc[g, "pred_late_like_mean"])
            if n_segments <= 1:
                vals.append(p)
            else:
                t = k / (n_segments - 1)
                vals.append(e + t * (p - e))
        base[col] = vals

    return base


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="hESC")
    p.add_argument("--n-segments", type=int, default=N_SEG_DEFAULT)
    p.add_argument(
        "--outdir",
        type=Path,
        default=SCRIPT_DIR / "error_biology" / "multistep_pt",
        help="Root; segment scGPT outputs → {outdir}/{dataset}/segment_preds/scgpt/",
    )
    p.add_argument("--gen-iters", type=int, default=16)
    p.add_argument("--skip-run", action="store_true", help="Only merge CSVs if segment runs exist")
    p.add_argument("--scgpt-legacy-pt", action="store_true", default=True)
    args = p.parse_args()

    ds = args.dataset
    seg_root = args.outdir / ds / "segment_preds"
    scgpt_out = seg_root  # unified writes to {outdir}/scgpt/{stem}_gene_result.csv
    n_trans = args.n_segments - 1

    if not args.skip_run:
        print(f"=== {ds}: run {n_trans} adjacent segment-pair scGPT jobs ===")
        for i in range(n_trans):
            run_segment_pair(
                ds, i, i + 1, scgpt_out, args.n_segments, args.gen_iters, args.scgpt_legacy_pt
            )

    single_csv = SINGLE_STEP_DIR / f"{ds}_gene_result.csv"
    traj = merge_trajectories(ds, scgpt_out, args.n_segments, single_csv)
    out_csv = args.outdir / ds / "gene_trajectories.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    traj.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}  ({len(traj)} genes)")


if __name__ == "__main__":
    main()
