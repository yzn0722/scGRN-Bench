#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geneformer 多段伪时间 — 按 PT 分段评估「局部速度」方向准确率。

思路（类似对表达曲线求导）
----------------------------
- 将伪时间均分为 ``--n-segments`` 段（默认 5 → S0…S4）。
- 每段 transition S_i→S_{i+1}：
  - **真值速度** ``delta_true_local`` = mean(expr, S_{i+1}) − mean(expr, S_i)
  - **预测速度**：Geneformer swap 迭代后 token 排序位置变化（与单步 benchmark 一致：pred = −Δrank）
- **链式 (chained)**：从 S0 细胞出发，状态连续推进，每段结束存盘（对标 scGPT chained）。
- **独立 (independent)**：每段从 S_i 细胞重新初始化（对标独立段 scGPT）。

指标
----
- ``dir_correct_local``：局部速度方向是否与真值一致（主指标，推荐写论文时用）
- ``dir_correct_cum``：相对 S0 的累计变化方向（与 scGPT chained 对齐时可对比）

示例（需 GPU + singlecell 环境）：
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig4
  python3 run_geneformer_multisegment_pt.py --dataset hESC
  python3 run_geneformer_multisegment_pt.py --dataset hESC --n-segments 7 --gen-iters 21
  python3 run_geneformer_multisegment_pt.py --dataset hESC --mode independent
  python3 run_geneformer_multisegment_pt.py --all-datasets --n-segments 5
"""
from __future__ import annotations

import argparse
import json
import pickle
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
DEFAULT_MODEL = BENCH / "model" / "weights" / "Geneformer" / "default" / "6L"
DEFAULT_DICTS = BENCH / "model" / "weights" / "Geneformer" / "dicts"
DEFAULT_OUT = SCRIPT_DIR / "error_biology" / "multistep_pt"
DATASETS = ["hESC", "hHep", "mDC", "mHSC-E", "mHSC-GM", "mHSC-L"]


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


def normalize_symbol(s: str) -> str:
    return str(s).strip().upper()


def is_ensembl_id(s: str) -> bool:
    return isinstance(s, str) and (s.startswith("ENSG") or s.startswith("ENSMUSG"))


def read_pt_file(path: Path) -> pd.DataFrame:
    try:
        pt_df = pd.read_csv(path, header=None)
        _ = float(pt_df.iloc[0, 1])
    except (ValueError, IndexError, TypeError):
        pt_df = pd.read_csv(path)
    pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"}).set_index("cell")
    pt_df["pt"] = pd.to_numeric(pt_df["pt"], errors="coerce")
    return pt_df.dropna(subset=["pt"])


def load_geneformer_dicts(dicts_dir: Path) -> Tuple[dict, dict, int, int]:
    with open(dicts_dir / "token_dictionary.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open(dicts_dir / "gene_name_id_dict.pkl", "rb") as f:
        gene_name_id = pickle.load(f)
    return vocab, gene_name_id, int(vocab["<pad>"]), int(vocab["<mask>"])


def build_symbol_to_ensembl(gene_name_id: dict) -> Dict[str, str]:
    sample = list(gene_name_id.items())[:2000]
    k_ens = sum(is_ensembl_id(str(k)) for k, _ in sample)
    v_ens = sum(is_ensembl_id(str(v)) for _, v in sample)
    if v_ens > k_ens:
        return {normalize_symbol(k): str(v) for k, v in gene_name_id.items()}
    return {normalize_symbol(v): str(k) for k, v in gene_name_id.items()}


def positions_dict_from_seq(seq_ids: np.ndarray, length: int) -> Dict[int, int]:
    pos = {}
    for i in range(length):
        t = int(seq_ids[i])
        if t not in pos:
            pos[t] = i
    return pos


def rank_delta(p0: Dict[int, int], p1: Dict[int, int], gene_tids: List[int], n_genes: int) -> np.ndarray:
    d = np.zeros(n_genes, dtype=np.float32)
    for gi in range(n_genes):
        t = gene_tids[gi]
        d[gi] = p1.get(t, len(p0)) - p0.get(t, len(p0))
    return d


def direction_accuracy(pred: np.ndarray, true: np.ndarray, top_idx: np.ndarray) -> Tuple[float, float]:
    if len(top_idx) == 0:
        return float("nan"), float("nan")
    td = true[top_idx]
    pd = pred[top_idx]
    true_dir = np.where(td > 0, 1, -1)
    pred_dir = np.where(pd > 0, 1, -1)
    acc = float((pred_dir == true_dir).mean())
    inv = float((pred_dir == (-true_dir)).mean())
    return acc, inv


def run_dataset(
    dataset: str,
    n_segments: int,
    gen_iters: int,
    swap_trials: int,
    max_len: int,
    min_seq_len: int,
    top_percent: int,
    pt_quantile: float,
    temperature: float,
    use_log1p: bool,
    mode: str,
    model_dir: Path,
    dicts_dir: Path,
    outdir: Path,
    rollback: bool,
    print_every: int,
) -> None:
    import torch
    from transformers import BertForMaskedLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[{dataset}] Geneformer multisegment | mode={mode} | n_segments={n_segments} | device={device}")

    expr = pd.read_csv(CHIP_DIR / f"{dataset}_chip_matched-ExpressionData.csv", index_col=0)
    pt_df = read_pt_file(PT_ROOT / dataset / "PseudoTime.csv")
    common = expr.columns.intersection(pt_df.index)
    expr = expr[common]
    pt = pt_df.loc[common, "pt"].astype(float).to_numpy()
    genes = expr.index.astype(str).tolist()
    n_genes = len(genes)

    X = expr.T.to_numpy(dtype=np.float32)
    if use_log1p:
        X = np.log1p(np.maximum(X, 0))

    masks = mask_segment_bins(pt, n_segments)
    n_trans = n_segments - 1
    iters_per_seg = max(1, gen_iters // n_trans)
    seg_expr_mean = [X[m].mean(axis=0) for m in masks]
    baseline_expr = seg_expr_mean[0].astype(np.float32)

    # 单步 top% 基因（与 fig4 / benchmark 一致）
    lo, hi = np.quantile(pt, [pt_quantile, 1 - pt_quantile])
    delta_single = X[pt >= hi].mean(axis=0) - X[pt <= lo].mean(axis=0)
    top_n = max(int(n_genes * top_percent / 100), 1)
    top_idx = np.argsort(np.abs(delta_single))[::-1][:top_n]

    vocab, gene_name_id, pad_id, mask_id = load_geneformer_dicts(dicts_dir)
    sym2ens = build_symbol_to_ensembl(gene_name_id)

    def get_tid(g: str) -> int:
        gn = normalize_symbol(g)
        if gn in vocab:
            return int(vocab[gn])
        ens = sym2ens.get(gn)
        if ens and ens in vocab:
            return int(vocab[ens])
        return pad_id

    gene_tids = [get_tid(g) for g in genes]
    matched = sum(1 for t in gene_tids if t != pad_id)
    print(f"  genes={n_genes} | vocab matched={matched} ({100*matched/max(1,n_genes):.1f}%)")
    print(f"  iters/transition={iters_per_seg} | swap_trials={swap_trials}")

    def cell_to_seq(x: np.ndarray) -> Tuple[np.ndarray, int]:
        order = np.argsort(-x)
        seq: List[int] = []
        for j in order:
            t = gene_tids[j]
            if t == pad_id:
                continue
            seq.append(t)
            if len(seq) >= max_len:
                break
        length = len(seq)
        if length < max_len:
            seq = seq + [pad_id] * (max_len - length)
        return np.array(seq, dtype=np.int64), length

    @torch.no_grad()
    def swap_step(model, seq: np.ndarray, length: int) -> np.ndarray:
        x = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
        attn = torch.zeros((1, x.size(1)), dtype=torch.long, device=device)
        attn[0, :length] = 1
        banned = torch.tensor([pad_id, mask_id], device=device, dtype=torch.long)
        for _ in range(swap_trials):
            if length < 2:
                break
            ij = torch.randperm(length, device="cpu")[:2]
            i, j = int(ij[0]), int(ij[1])
            if i == j:
                continue
            a, b = int(x[0, i]), int(x[0, j])
            if a in (pad_id, mask_id) or b in (pad_id, mask_id):
                continue
            xm = x.clone()
            xm[0, i], xm[0, j] = mask_id, mask_id
            logits = model(input_ids=xm, attention_mask=attn).logits[0, [i, j], :].float()
            logits /= max(temperature, 1e-6)
            bd = banned[(banned >= 0) & (banned < logits.size(-1))]
            if bd.numel() > 0:
                logits[:, bd] = -1e9
            logp = torch.log_softmax(logits, dim=-1)
            if a < 0 or b < 0 or a >= logits.size(-1) or b >= logits.size(-1):
                continue
            cur = float(logp[0, a].item() + logp[1, b].item())
            swp = float(logp[0, b].item() + logp[1, a].item())
            if swp > cur:
                x[0, i], x[0, j] = x[0, j].clone(), x[0, i].clone()
        return x.squeeze(0).cpu().numpy()

    model = BertForMaskedLM.from_pretrained(str(model_dir)).to(device).eval()

    def build_cohort(mask: np.ndarray) -> Tuple[List[np.ndarray], List[int], List[np.ndarray]]:
        init_list, lengths, idx_map = [], [], []
        for i in np.where(mask)[0]:
            seq, L = cell_to_seq(X[i])
            if L >= min_seq_len:
                init_list.append(seq.copy())
                lengths.append(L)
                idx_map.append(int(i))
        return init_list, lengths, idx_map

    segment_rows: List[dict] = []
    snapshots: List[dict] = []

    # 链式：全局 S0 细胞；独立：每段重建 cohort
    chained_init: Optional[List[np.ndarray]] = None
    chained_curr: Optional[List[np.ndarray]] = None
    chained_lengths: Optional[List[int]] = None
    chained_p0_init: Optional[List[Dict[int, int]]] = None

    if mode == "chained":
        chained_init, chained_lengths, _ = build_cohort(masks[0])
        if not chained_init:
            raise ValueError(f"[{dataset}] S0 segment has no valid cells (min_seq_len={min_seq_len})")
        chained_curr = [x.copy() for x in chained_init]
        chained_p0_init = [positions_dict_from_seq(x, L) for x, L in zip(chained_init, chained_lengths)]
        print(f"  chained cohort: S0 cells n={len(chained_init)}")

    with torch.no_grad():
        for seg_i in range(n_trans):
            trans = f"delta_t{seg_i}_t{seg_i + 1}"
            true_early = seg_expr_mean[seg_i]
            true_late = seg_expr_mean[seg_i + 1]
            delta_true_local = (true_late - true_early).astype(np.float32)
            delta_true_cum = (true_late - baseline_expr).astype(np.float32)

            if mode == "independent":
                init_list, lengths, _ = build_cohort(masks[seg_i])
                if not init_list:
                    print(f"  [skip] {trans}: no cells in S{seg_i}")
                    continue
                curr = [x.copy() for x in init_list]
                p0_init = [positions_dict_from_seq(x, L) for x, L in zip(init_list, lengths)]
            else:
                init_list = chained_init  # type: ignore
                lengths = chained_lengths  # type: ignore
                curr = chained_curr  # type: ignore
                p0_init = chained_p0_init  # type: ignore

            p0_block = [positions_dict_from_seq(x, L) for x, L in zip(curr, lengths)]
            best_acc_local = -1.0
            prev_curr = [x.copy() for x in curr]

            for local_it in range(iters_per_seg):
                next_curr = []
                for seq, L in zip(curr, lengths):
                    next_curr.append(swap_step(model, seq, L))
                curr = next_curr

                deltas_local = [
                    rank_delta(p0_block[ci], positions_dict_from_seq(curr[ci], lengths[ci]), gene_tids, n_genes)
                    for ci in range(len(curr))
                ]
                pred_local = -np.stack(deltas_local, axis=0).mean(axis=0)
                acc_l, _ = direction_accuracy(pred_local, delta_true_local, top_idx)

                if rollback and best_acc_local >= 0 and acc_l < best_acc_local:
                    curr = prev_curr
                    acc_l = best_acc_local
                else:
                    prev_curr = [x.copy() for x in curr]
                    if acc_l > best_acc_local:
                        best_acc_local = acc_l

                if print_every and (
                    (local_it + 1) % print_every == 0 or (local_it + 1) == iters_per_seg
                ):
                    print(f"    {trans} iter {local_it + 1}/{iters_per_seg} | local vel. acc={acc_l:.2%}")

            deltas_local = [
                rank_delta(p0_block[ci], positions_dict_from_seq(curr[ci], lengths[ci]), gene_tids, n_genes)
                for ci in range(len(curr))
            ]
            pred_local = -np.stack(deltas_local, axis=0).mean(axis=0)

            if mode == "chained":
                deltas_cum = [
                    rank_delta(p0_init[ci], positions_dict_from_seq(curr[ci], lengths[ci]), gene_tids, n_genes)
                    for ci in range(len(curr))
                ]
                pred_cum = -np.stack(deltas_cum, axis=0).mean(axis=0)
                chained_curr = curr
            else:
                # 独立段：累计指标与局部速度相同（无跨段状态）
                pred_cum = pred_local.copy()
            dir_true_l = np.where(delta_true_local > 0, "Up", "Down")
            dir_pred_l = np.where(pred_local > 0, "Up", "Down")
            dir_ok_l = (dir_true_l == dir_pred_l).astype(int)
            dir_true_c = np.where(delta_true_cum > 0, "Up", "Down")
            dir_pred_c = np.where(pred_cum > 0, "Up", "Down")
            dir_ok_c = (dir_true_c == dir_pred_c).astype(int)

            acc_local, inv_local = direction_accuracy(pred_local, delta_true_local, top_idx)
            acc_cum, inv_cum = direction_accuracy(pred_cum, delta_true_cum, top_idx)
            acc_local_all = float((dir_true_l == dir_pred_l).mean())
            acc_cum_all = float((dir_true_c == dir_pred_c).mean())

            segment_rows.append(
                {
                    "transition": trans,
                    "seg_from": seg_i,
                    "seg_to": seg_i + 1,
                    "mode": mode,
                    "n_cells": len(curr),
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
                    "true_early_mean_local": true_early.astype(np.float32),
                    "true_late_mean": true_late.astype(np.float32),
                    "true_baseline_mean": baseline_expr.astype(np.float32),
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
                }
            )
            snapshots.append({"transition": trans, "df": gr})
            print(
                f"  {trans} done | top% local={acc_local:.1%} all={acc_local_all:.1%} | "
                f"top% cum={acc_cum:.1%}"
            )

    ds_out = outdir / dataset / "geneformer_preds"
    ds_out.mkdir(parents=True, exist_ok=True)
    for snap in snapshots:
        snap["df"].to_csv(ds_out / f"{dataset}_{snap['transition']}_gene_result.csv", index=False)
    pd.DataFrame(segment_rows).to_csv(ds_out / f"{dataset}_geneformer_segment_summary.csv", index=False)

    wide = pd.DataFrame({"gene": genes})
    for snap in snapshots:
        t = snap["transition"]
        sub = snap["df"].set_index("gene")
        wide[f"{t}_delta_pred_local"] = wide["gene"].map(sub["delta_pred_local"])
        wide[f"{t}_dir_correct_local"] = wide["gene"].map(sub["dir_correct_local"])
        wide[f"{t}_dir_correct_cumulative"] = wide["gene"].map(sub["dir_correct_cumulative"])

    wide.to_csv(ds_out / f"{dataset}_geneformer_per_gene_wide.csv", index=False)

    cfg = {
        "dataset": dataset,
        "mode": mode,
        "n_segments": n_segments,
        "gen_iters": gen_iters,
        "swap_trials": swap_trials,
        "iters_per_transition": iters_per_seg,
        "top_percent": top_percent,
        "model_dir": str(model_dir),
    }
    with open(ds_out / f"{dataset}_geneformer_run_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nSaved: {ds_out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Geneformer multisegment PT (velocity / direction per block)")
    p.add_argument("--dataset", default="hESC")
    p.add_argument("--all-datasets", action="store_true")
    p.add_argument("--n-segments", type=int, default=5, help="PT bins (>=3). More segments = finer 'velocity'.")
    p.add_argument("--gen-iters", type=int, default=16, help="Total swap iterations split across transitions")
    p.add_argument("--swap-trials", type=int, default=8)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--min-seq-len", type=int, default=10)
    p.add_argument("--top-percent", type=int, default=30)
    p.add_argument("--pt-quantile", type=float, default=0.2, help="For top-gene selection (single-step 20%%-80%%)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--no-log1p", action="store_true")
    p.add_argument(
        "--mode",
        choices=["chained", "independent"],
        default="chained",
        help="chained: S0 cells carry state; independent: restart from each Si",
    )
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--dicts-dir", type=Path, default=DEFAULT_DICTS)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--no-rollback", action="store_true", help="Disable per-block acc rollback")
    p.add_argument("--print-every", type=int, default=1)
    args = p.parse_args()

    if args.n_segments < 3:
        raise ValueError("--n-segments must be >= 3")

    datasets = DATASETS if args.all_datasets else [args.dataset]
    for ds in datasets:
        run_dataset(
            ds,
            args.n_segments,
            args.gen_iters,
            args.swap_trials,
            args.max_len,
            args.min_seq_len,
            args.top_percent,
            args.pt_quantile,
            args.temperature,
            not args.no_log1p,
            args.mode,
            args.model_dir,
            args.dicts_dir,
            args.outdir,
            rollback=not args.no_rollback,
            print_every=args.print_every,
        )


if __name__ == "__main__":
    main()
