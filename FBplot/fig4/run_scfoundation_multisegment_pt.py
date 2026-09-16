#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scFoundation 多段伪时间 — 按 PT 分段评估「局部速度」方向准确率。

与 Geneformer 多段脚本对齐（``run_geneformer_multisegment_pt.py``）：
- 伪时间均分为 n_segments（默认 7 → S0…S6，6 个 transition）
- 每段 local velocity：Δ_true_local = mean(S_{i+1}) − mean(S_i)
- scFoundation 连续表达迭代（MaeAutobin），非 rank/bin 空间
- **chained**：S0 段细胞状态连续推进；**independent**：每段从 S_i 重新初始化

输出目录：``error_biology/multistep_pt/{dataset}/scfoundation_preds/``
  - {dataset}_delta_t{i}_t{j}_gene_result.csv  （列格式与 geneformer_preds 一致）
  - {dataset}_scfoundation_segment_summary.csv
  - {dataset}_scfoundation_per_gene_wide.csv

示例（**必须 GPU** + singlecell 环境）：
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig4
  conda activate singlecell
  nvidia-smi   # 确认 GPU 可用
  PYTHONUNBUFFERED=1 python3 run_scfoundation_multisegment_pt.py --dataset hESC --n-segments 7 --gen-iters 21
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH = Path("/mnt/10T/yzn/benchmark_GRN")
CHIP_DIR = BENCH / "input_process" / "CHIP"
PT_ROOT = BENCH / "PseudoTime"
DEFAULT_OUT = SCRIPT_DIR / "error_biology" / "multistep_pt"
DATASETS = ["hESC", "hHep", "mDC", "mHSC-E", "mHSC-GM", "mHSC-L"]

DEFAULT_SCF_ROOT = Path("/mnt/10T/yzn/scFoundation-main/model")
DEFAULT_SCF_CKPT = DEFAULT_SCF_ROOT / "models" / "models.ckpt"
DEFAULT_SCF_GENE_TSV = DEFAULT_SCF_ROOT / "OS_scRNA_gene_index.19264.tsv"


def mask_segment_bins(pt: np.ndarray, n_segments: int) -> List[np.ndarray]:
    n_segments = max(2, int(n_segments))
    edges = np.quantile(pt, np.linspace(0, 1, n_segments + 1))
    edges[-1] += 1e-9
    masks = []
    for i in range(n_segments):
        if i < n_segments - 1:
            masks.append((pt >= edges[i]) & (pt < edges[i + 1]))
        else:
            masks.append((pt >= edges[i]) & (pt <= edges[i + 1]))
    return masks


def read_pt_file(path: Path) -> pd.DataFrame:
    try:
        pt_df = pd.read_csv(path, header=None)
        _ = float(pt_df.iloc[0, 1])
    except (ValueError, IndexError, TypeError):
        pt_df = pd.read_csv(path)
    pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"}).set_index("cell")
    pt_df["pt"] = pd.to_numeric(pt_df["pt"], errors="coerce")
    return pt_df.dropna(subset=["pt"])


def direction_accuracy(
    pred: np.ndarray,
    true: np.ndarray,
    top_idx: np.ndarray,
    eps: float = 0.0,
) -> Tuple[float, float]:
    if len(top_idx) == 0:
        return float("nan"), float("nan")
    td = true[top_idx]
    pdv = pred[top_idx]
    if eps > 0:
        m = np.abs(td) > eps
        if m.sum() == 0:
            return float("nan"), float("nan")
        td, pdv = td[m], pdv[m]
    true_dir = np.where(td > 0, 1, -1)
    pred_dir = np.where(pdv > 0, 1, -1)
    acc = float((pred_dir == true_dir).mean())
    inv = float((pred_dir == (-true_dir)).mean())
    return acc, inv


def pred_mean_910(vals_np: np.ndarray, map_idx: np.ndarray, n_genes: int) -> np.ndarray:
    """Mean over cells in model space → 910-gene vector."""
    mean_model = vals_np.mean(axis=0)
    out = np.full(n_genes, np.nan, dtype=np.float32)
    for j in range(n_genes):
        midx = int(map_idx[j])
        if midx >= 0:
            out[j] = mean_model[midx]
    return out


def import_scfoundation_helpers():
    if str(BENCH) not in sys.path:
        sys.path.insert(0, str(BENCH))
    from run_unified_multidataset_pseudotime import (  # noqa: WPS433
        _scf_iterative_predict_curve,
        load_scfoundation_model,
        scf_build_aligned_matrix,
        scf_read_gene_index_tsv,
    )

    return _scf_iterative_predict_curve, load_scfoundation_model, scf_build_aligned_matrix, scf_read_gene_index_tsv


def run_dataset(
    dataset: str,
    n_segments: int,
    gen_iters: int,
    batch_size: int,
    top_percent: int,
    pt_quantile: float,
    mode: str,
    scf_root: Path,
    scf_ckpt: Path,
    scf_gene_tsv: Path,
    scf_mode: str,
    scf_value_mask_prob: float,
    scf_zero_mask_prob: float,
    scf_update_scope: str,
    scf_eps_dir: float,
    update_alpha: float,
    refresh_encoder: bool,
    resample_mask: bool,
    calibrate: bool,
    calibrate_on: str,
    seed: int,
    outdir: Path,
    print_every: int,
) -> None:
    import torch

    _scf_iter, load_scf_model, scf_align, scf_read_gene = import_scfoundation_helpers()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[{dataset}] scFoundation multisegment | mode={mode} | n_segments={n_segments} | device={device}", flush=True)

    expr = pd.read_csv(CHIP_DIR / f"{dataset}_chip_matched-ExpressionData.csv", index_col=0)
    pt_df = read_pt_file(PT_ROOT / dataset / "PseudoTime.csv")
    common = expr.columns.intersection(pt_df.index)
    expr = expr[common]
    pt = pt_df.loc[common, "pt"].astype(float).to_numpy()
    genes = expr.index.astype(str).tolist()
    n_genes = len(genes)

    X = np.nan_to_num(expr.T.to_numpy(dtype=np.float32), nan=0.0)
    masks = mask_segment_bins(pt, n_segments)
    n_trans = n_segments - 1
    iters_per_seg = max(1, gen_iters // n_trans)
    seg_expr_mean = [X[m].mean(axis=0).astype(np.float32) for m in masks]
    baseline_expr = seg_expr_mean[0].copy()

    lo, hi = np.quantile(pt, [pt_quantile, 1 - pt_quantile])
    delta_single = X[pt >= hi].mean(axis=0) - X[pt <= lo].mean(axis=0)
    top_n = max(int(n_genes * top_percent / 100), 1)
    top_idx = np.argsort(np.abs(delta_single))[::-1][:top_n]

    class ScfArgs:
        pass

    args = ScfArgs()
    args.scf_root = str(scf_root)
    args.scf_ckpt = str(scf_ckpt)
    args.scf_gene_index_tsv = str(scf_gene_tsv)
    args.scf_mmf_key = "gene"
    args.scf_no_refresh_encoder = not refresh_encoder
    args.scf_no_resample_mask = not resample_mask
    args.scf_no_identity_input = False

    model, config = load_scf_model(args, device)
    G_model = int(config["seq_len"])
    gene2idx = scf_read_gene(str(scf_gene_tsv), G_model)
    X_full, present_mask_np, map_idx, mapped = scf_align(X, genes, gene2idx, G_model)
    present_mask = torch.tensor(present_mask_np, dtype=torch.bool, device=device)
    print(f"  genes={n_genes} | scFoundation mapped={mapped} ({100*mapped/max(1,n_genes):.1f}%)", flush=True)
    print(f"  iters/transition={iters_per_seg} | batch_size={batch_size}", flush=True)

    cohort_idx = np.where(masks[0])[0]
    if len(cohort_idx) == 0:
        raise ValueError(f"[{dataset}] S0 segment has no cells")
    print(f"  chained cohort: S0 cells n={len(cohort_idx)}", flush=True)

    vals_all = torch.tensor(
        X_full[cohort_idx],
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device=device,
    )
    raw_all = torch.tensor(X[cohort_idx], dtype=torch.float32, device=device)
    # raw in model space for masking
    raw_full_all = torch.tensor(X_full[cohort_idx], dtype=torch.float32, device=device)

    mapped_mask = map_idx >= 0
    eval_pool = np.where(mapped_mask)[0]
    eval_top = eval_pool[np.argsort(np.abs(delta_single[eval_pool]))[::-1][:top_n]]
    eval_model_idx = map_idx[eval_top].astype(np.int64)

    segment_rows: List[dict] = []
    snapshots: List[dict] = []

    with torch.no_grad():
        for seg_i in range(n_trans):
            trans = f"delta_t{seg_i}_t{seg_i + 1}"
            true_early = seg_expr_mean[seg_i]
            true_late = seg_expr_mean[seg_i + 1]
            delta_true_local = (true_late - true_early).astype(np.float32)
            delta_true_cum = (true_late - baseline_expr).astype(np.float32)

            if mode == "independent":
                seg_cells = np.where(masks[seg_i])[0]
                if len(seg_cells) == 0:
                    print(f"  [skip] {trans}: no cells in S{seg_i}")
                    continue
                vals_all = torch.tensor(
                    X_full[seg_cells],
                    dtype=torch.float16 if device.type == "cuda" else torch.float32,
                    device=device,
                )
                raw_full_all = torch.tensor(X_full[seg_cells], dtype=torch.float32, device=device)
                cohort_idx = seg_cells

            early_mean_eval = true_early[eval_top]
            true_delta_eval = delta_true_local[eval_top].astype(np.float32)

            best_acc = -1.0
            for local_it in range(iters_per_seg):
                for start in range(0, vals_all.shape[0], batch_size):
                    end = min(start + batch_size, vals_all.shape[0])
                    vals_batch = vals_all[start:end]
                    rawb = raw_full_all[start:end]
                    _, vals_out, _ = _scf_iter(
                        model=model,
                        config=config,
                        values_full_init=vals_batch,
                        raw_full=rawb,
                        present_mask=present_mask,
                        update_scope=scf_update_scope,
                        eval_model_idx=eval_model_idx,
                        early_mean_eval=early_mean_eval,
                        true_delta_eval=true_delta_eval,
                        n_iters=1,
                        seed0=seed + seg_i * 1000 + local_it * 17 + start,
                        mode=scf_mode,
                        value_mask_prob=scf_value_mask_prob,
                        zero_mask_prob=scf_zero_mask_prob,
                        refresh_encoder=refresh_encoder,
                        resample_mask=resample_mask,
                        update_alpha=update_alpha,
                        eps_dir=scf_eps_dir,
                        calibrate=calibrate,
                        calibrate_on=calibrate_on,
                        save_full_cells_by_iter=False,
                    )
                    vals_all[start:end] = torch.tensor(
                        vals_out,
                        dtype=vals_all.dtype,
                        device=device,
                    )

                pred_local_monitor = pred_mean_910(vals_all.detach().float().cpu().numpy(), map_idx, n_genes) - true_early
                acc_l, _ = direction_accuracy(pred_local_monitor, delta_true_local, top_idx, eps=scf_eps_dir)
                if acc_l > best_acc:
                    best_acc = acc_l
                if print_every and (
                    (local_it + 1) % print_every == 0 or (local_it + 1) == iters_per_seg
                ):
                    print(f"    {trans} iter {local_it + 1}/{iters_per_seg} | local vel. acc={acc_l:.2%}", flush=True)

            pred_mean = pred_mean_910(vals_all.detach().float().cpu().numpy(), map_idx, n_genes)
            pred_local = pred_mean - true_early
            pred_cum = pred_mean - baseline_expr

            if mode == "chained" and np.isnan(pred_mean).all():
                raise ValueError("All predictions NaN — check gene mapping")

            dir_true_l = np.where(delta_true_local > 0, "Up", "Down")
            dir_pred_l = np.where(pred_local > 0, "Up", np.where(pred_local < 0, "Down", "Flat"))
            dir_ok_l = np.array(
                [
                    (dt > 0 and dp > 0) or (dt < 0 and dp < 0) or (dt == 0 and abs(dp) < scf_eps_dir)
                    for dt, dp in zip(delta_true_local, pred_local)
                ],
                dtype=int,
            )
            # NaN pred → wrong
            dir_ok_l[np.isnan(pred_local)] = 0

            dir_true_c = np.where(delta_true_cum > 0, "Up", "Down")
            dir_pred_c = np.where(pred_cum > 0, "Up", np.where(pred_cum < 0, "Down", "Flat"))
            dir_ok_c = np.array(
                [
                    (dt > 0 and dp > 0) or (dt < 0 and dp < 0) or (dt == 0 and abs(dp) < scf_eps_dir)
                    for dt, dp in zip(delta_true_cum, pred_cum)
                ],
                dtype=int,
            )
            dir_ok_c[np.isnan(pred_cum)] = 0

            acc_local, inv_local = direction_accuracy(pred_local, delta_true_local, top_idx, eps=scf_eps_dir)
            acc_cum, inv_cum = direction_accuracy(pred_cum, delta_true_cum, top_idx, eps=scf_eps_dir)
            acc_local_all = float(np.nanmean(dir_ok_l[mapped_mask])) if mapped_mask.any() else float("nan")
            acc_cum_all = float(np.nanmean(dir_ok_c[mapped_mask])) if mapped_mask.any() else float("nan")

            segment_rows.append(
                {
                    "transition": trans,
                    "seg_from": seg_i,
                    "seg_to": seg_i + 1,
                    "mode": mode,
                    "n_cells": int(vals_all.shape[0]),
                    "iters_in_block": iters_per_seg,
                    "acc_top_local": acc_local,
                    "acc_top_cumulative": acc_cum,
                    "acc_all_local": acc_local_all,
                    "acc_all_cumulative": acc_cum_all,
                    "inv_top_local": inv_local,
                }
            )

            gr = pd.DataFrame(
                {
                    "gene": genes,
                    "true_early_mean_local": true_early,
                    "true_late_mean": true_late,
                    "true_baseline_mean": baseline_expr,
                    "delta_true_local": delta_true_local,
                    "delta_true_cumulative": delta_true_cum,
                    "delta_pred_local": pred_local.astype(np.float32),
                    "delta_pred_cumulative": pred_cum.astype(np.float32),
                    "dir_true_local": dir_true_l,
                    "dir_pred_local": dir_pred_l,
                    "dir_correct_local": dir_ok_l,
                    "dir_true_cumulative": dir_true_c,
                    "dir_pred_cumulative": dir_pred_c,
                    "dir_correct_cumulative": dir_ok_c,
                    "mapped": mapped_mask.astype(int),
                }
            )
            snapshots.append({"transition": trans, "df": gr})
            print(
                f"  {trans} done | top% local={acc_local:.1%} all={acc_local_all:.1%} | "
                f"top% cum={acc_cum:.1%}",
                flush=True,
            )

    ds_out = outdir / dataset / "scfoundation_preds"
    ds_out.mkdir(parents=True, exist_ok=True)
    for snap in snapshots:
        snap["df"].to_csv(ds_out / f"{dataset}_{snap['transition']}_gene_result.csv", index=False)
    pd.DataFrame(segment_rows).to_csv(ds_out / f"{dataset}_scfoundation_segment_summary.csv", index=False)

    wide = pd.DataFrame({"gene": genes})
    for snap in snapshots:
        t = snap["transition"]
        sub = snap["df"].set_index("gene")
        wide[f"{t}_delta_pred_local"] = wide["gene"].map(sub["delta_pred_local"])
        wide[f"{t}_dir_correct_local"] = wide["gene"].map(sub["dir_correct_local"])
        wide[f"{t}_dir_correct_cumulative"] = wide["gene"].map(sub["dir_correct_cumulative"])

    wide.to_csv(ds_out / f"{dataset}_scfoundation_per_gene_wide.csv", index=False)

    cfg = {
        "dataset": dataset,
        "mode": mode,
        "n_segments": n_segments,
        "gen_iters": gen_iters,
        "iters_per_transition": iters_per_seg,
        "top_percent": top_percent,
        "scf_root": str(scf_root),
        "scf_ckpt": str(scf_ckpt),
        "scf_mode": scf_mode,
        "mapped_genes": int(mapped),
    }
    with open(ds_out / f"{dataset}_scfoundation_run_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nSaved: {ds_out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="scFoundation multisegment PT (local velocity per block)")
    p.add_argument("--dataset", default="hESC")
    p.add_argument("--all-datasets", action="store_true")
    p.add_argument("--n-segments", type=int, default=7, help="PT bins (>=3). Match Geneformer hESC: 7")
    p.add_argument("--gen-iters", type=int, default=21, help="Total MAE iterations split across transitions")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--top-percent", type=int, default=30)
    p.add_argument("--pt-quantile", type=float, default=0.2)
    p.add_argument("--mode", choices=["chained", "independent"], default="chained")
    p.add_argument("--scf-root", type=Path, default=DEFAULT_SCF_ROOT)
    p.add_argument("--scf-ckpt", type=Path, default=DEFAULT_SCF_CKPT)
    p.add_argument("--scf-gene-tsv", type=Path, default=DEFAULT_SCF_GENE_TSV)
    p.add_argument("--scf-mode", choices=["mae", "zero"], default="mae")
    p.add_argument("--scf-value-mask-prob", type=float, default=0.3)
    p.add_argument("--scf-zero-mask-prob", type=float, default=0.0)
    p.add_argument("--scf-update-scope", choices=["present_all", "mask", "zero"], default="present_all")
    p.add_argument("--scf-eps-dir", type=float, default=1e-3)
    p.add_argument("--update-alpha", type=float, default=0.1)
    p.add_argument("--no-refresh-encoder", action="store_true")
    p.add_argument("--no-resample-mask", action="store_true")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--calibrate-on", choices=["present", "present_nonzero"], default="present")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU fallback (default: exit if CUDA unavailable)",
    )
    args = p.parse_args()

    import torch

    if not args.allow_cpu and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for scFoundation multisegment (use GPU node).\n"
            "  Check: nvidia-smi && python -c \"import torch; print(torch.cuda.is_available())\"\n"
            "  If driver mismatch: reboot or reload nvidia driver, then retry.\n"
            "  Debug only: pass --allow-cpu"
        )

    if args.n_segments < 3:
        raise ValueError("--n-segments must be >= 3")

    datasets = DATASETS if args.all_datasets else [args.dataset]
    for ds in datasets:
        run_dataset(
            ds,
            args.n_segments,
            args.gen_iters,
            args.batch_size,
            args.top_percent,
            args.pt_quantile,
            args.mode,
            args.scf_root,
            args.scf_ckpt,
            args.scf_gene_tsv,
            args.scf_mode,
            args.scf_value_mask_prob,
            args.scf_zero_mask_prob,
            args.scf_update_scope,
            args.scf_eps_dir,
            args.update_alpha,
            not args.no_refresh_encoder,
            not args.no_resample_mask,
            args.calibrate,
            args.calibrate_on,
            args.seed,
            args.outdir,
            args.print_every,
        )


if __name__ == "__main__":
    main()
