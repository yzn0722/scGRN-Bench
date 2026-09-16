#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
链式分段 scGPT：一次预测过程沿 S0→S1→…→S4 推进，**每一段结束都保存结果**。

与 run_scgpt_pt_segments.py 的区别
--------------------------------
- **旧做法（独立段）**：S0→S1 单独 16 步、S1→S2 再单独 16 步……共 4 次互不继承的实验。
- **本脚本（链式）**：从 S0 细胞出发，状态在内存里连续更新；
  每走完一段（默认把总 iter 均分到 4 个 transition），记录该段的 pred_mean / dir_pred，
  再进入下一段 —— **预测过程中每一段都有一个结果**。

评估基准（与 S0 cohort 一致）
--------------------------------
- ``delta_true`` = mean(S_{i+1}) − **mean(S0)**（到第 i+1 段的累计真值变化）
- ``delta_pred`` = pred_mean − **mean(S0)**（S0 细胞推演到该段末的累计预测变化）
- ``delta_true_local`` = mean(S_{i+1}) − mean(S_i) 仅作参考，不用于 acc

示例（singlecell 环境 + GPU）：
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig4
  python3 run_scgpt_chained_segments.py --dataset hESC
  python3 run_scgpt_chained_segments.py --dataset hESC --gen-iters 16 --n-segments 5
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH = Path("/mnt/10T/yzn/benchmark_GRN")
SCGPT_REPO = BENCH / "pre_scgpt" / "scGPT"
SCGPT_MODEL = BENCH / "pre_scgpt" / "scGPT" / "scgpt_human"
CHIP_DIR = BENCH / "input_process" / "CHIP"
PT_ROOT = BENCH / "PseudoTime"
DEFAULT_OUT = SCRIPT_DIR / "error_biology" / "multistep_pt"


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


def direction_accuracy_top(
    pred_delta: np.ndarray,
    true_delta: np.ndarray,
    top_idx: np.ndarray,
) -> Tuple[float, float]:
    if len(top_idx) == 0:
        return float("nan"), float("nan")
    pd = pred_delta[top_idx]
    td = true_delta[top_idx]
    acc = float((np.sign(pd) == np.sign(td)).mean())
    inv = float((np.sign(pd) == -np.sign(td)).mean())
    return acc, inv


from chained_embedding_utils import extract_cell_embeddings, pairwise_steps, path_length, segment_centroids


def load_scgpt(device, use_fast: bool | None = None):
    if str(SCGPT_REPO) not in sys.path:
        sys.path.insert(0, str(SCGPT_REPO))
    import torch
    from scgpt.model import TransformerModel
    from scgpt.tokenizer.gene_tokenizer import GeneVocab

    if use_fast is None:
        use_fast = device.type == "cuda"

    with open(SCGPT_MODEL / "args.json") as f:
        cfg = json.load(f)
    vocab = GeneVocab.from_file(SCGPT_MODEL / "vocab.json")
    for t in ["<pad>", "<cls>", "<eoc>"]:
        if t not in vocab:
            vocab.append_token(t)
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=cfg["embsize"],
        nhead=cfg["nheads"],
        d_hid=cfg["d_hid"],
        nlayers=cfg["nlayers"],
        vocab=vocab,
        pad_value=cfg["pad_value"],
        n_input_bins=cfg.get("n_bins", 51),
        use_fast_transformer=use_fast and cfg.get("fast_transformer", True),
    )
    ckpt = torch.load(SCGPT_MODEL / "best_model.pt", map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device).eval()
    if device.type == "cuda":
        model.half()
    return model, vocab, torch, cfg


def run_chained(
    dataset: str,
    n_segments: int,
    gen_iters: int,
    batch_size: int,
    ema_alpha: float,
    top_percent: int,
    outdir: Path,
    print_every: int = 1,
    save_embeddings: bool = False,
) -> None:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{dataset}] chained segments | device={device}")

    expr = pd.read_csv(CHIP_DIR / f"{dataset}_chip_matched-ExpressionData.csv", index_col=0)
    pt_df = pd.read_csv(PT_ROOT / dataset / "PseudoTime.csv")
    pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"}).set_index("cell")
    common = expr.columns.intersection(pt_df.index)
    expr = expr[common]
    pt = pt_df.loc[common, "pt"].astype(float).to_numpy()
    genes = expr.index.astype(str).tolist()
    X_bin = bin_expr_to_0_50(expr.T.to_numpy(dtype=np.float32), do_log1p=False)

    masks = mask_segment_bins(pt, n_segments)
    n_trans = n_segments - 1
    iters_per_seg = max(1, gen_iters // n_trans)
    total_iters = iters_per_seg * n_trans
    print(f"  n_segments={n_segments} | iters_per_transition={iters_per_seg} | total_forward_blocks={total_iters}")

    # 链式起点：S0 段细胞（整条轨迹从最早段出发）
    cohort = masks[0]
    if cohort.sum() == 0:
        raise ValueError("S0 segment has no cells")
    print(f"  cohort: S0 cells n={int(cohort.sum())}")

    seg_true_means = [X_bin[m].mean(axis=0) for m in masks]
    # 链式从 S0 细胞出发：预测 Δ 与真值 Δ 均相对 S0 均值（同一基准）
    baseline_true = seg_true_means[0].astype(np.float32)
    baseline_label = "S0"

    model, vocab, torch, _ = load_scgpt(device)
    gene_ids = np.array([vocab[g] if g in vocab else vocab["<pad>"] for g in genes], dtype=np.int64)
    gene_ids = np.concatenate([[vocab["<cls>"]], gene_ids])
    gene_ids_tensor = torch.tensor(gene_ids[None, :], dtype=torch.long)
    values = np.concatenate([np.zeros((X_bin.shape[0], 1), dtype=np.float32), X_bin], axis=1)
    values_tensor = torch.tensor(values, dtype=torch.float16 if device.type == "cuda" else torch.float32)
    pad_mask = gene_ids_tensor.eq(vocab["<pad>"]).expand(values.shape[0], -1)
    update_mask = np.zeros(gene_ids_tensor.shape[1], dtype=bool)
    update_mask[1:] = True
    update_mask_t = torch.tensor(update_mask[None, :], device=device).bool()

    vals_all = values_tensor[cohort].clone()
    pad_early = pad_mask[cohort]

    # 单步基线真值（20% 端点）用于 top% 选取 — 与 fig4 一致
    lo, hi = np.quantile(pt, [0.2, 0.8])
    early_q, late_q = pt <= lo, pt >= hi
    delta_single = X_bin[late_q].mean(axis=0) - X_bin[early_q].mean(axis=0)
    top_n = max(int(len(genes) * top_percent / 100), 1)
    top_idx = np.argsort(np.abs(delta_single))[::-1][:top_n]

    global_iter = 0
    segment_rows: List[Dict] = []
    snapshots: List[Dict] = []
    chain_nodes: List[np.ndarray] = []

    def _embed_cohort() -> np.ndarray:
        arr = vals_all.numpy().astype(np.float32)
        return extract_cell_embeddings(model, vocab, torch, arr, gene_ids, batch_size=batch_size).mean(axis=0)

    true_cents = None
    if save_embeddings:
        print("  [embeddings] encoding all observed cells...")
        true_emb = extract_cell_embeddings(model, vocab, torch, values, gene_ids, batch_size=batch_size)
        seg_ids = np.zeros(len(pt), dtype=np.int64)
        for i, m in enumerate(masks):
            seg_ids[m] = i
        true_cents = segment_centroids(true_emb, seg_ids, n_segments)
        chain_nodes.append(_embed_cohort())
        print("  [embeddings] chain node S0 (start) saved")

    with torch.no_grad():
        for seg_i in range(n_trans):
            trans_name = f"delta_t{seg_i}_t{seg_i + 1}"
            true_early_local = seg_true_means[seg_i]
            true_late = seg_true_means[seg_i + 1]
            # 局部 transition 真值（仅作参考列）
            true_delta_local = true_late - true_early_local
            # 与预测对齐：相对起点 S0 的累计变化（到 S_{i+1}）
            true_delta_seg = true_late - baseline_true

            print(
                f"\n--- Transition {trans_name} (S{seg_i}→S{seg_i + 1}) | {iters_per_seg} iters | "
                f"Δ vs {baseline_label}"
            )

            for local_it in range(iters_per_seg):
                global_iter += 1
                for start in range(0, vals_all.shape[0], batch_size):
                    end = min(start + batch_size, vals_all.shape[0])
                    vals = vals_all[start:end].to(device)
                    src = gene_ids_tensor.expand(end - start, -1).to(device)
                    mask = pad_early[start:end].to(device)
                    freeze = mask | (~update_mask_t.expand(end - start, -1))
                    out = model(src=src, values=vals, src_key_padding_mask=mask)
                    new_vals = out["mlm_output"]
                    vals = torch.where(
                        freeze,
                        vals,
                        ema_alpha * vals + (1 - ema_alpha) * new_vals,
                    )
                    vals_all[start:end] = vals.detach().cpu()

                pred_mean = vals_all[:, 1:].numpy().mean(axis=0)
                pred_delta_seg = pred_mean - baseline_true
                acc, inv = direction_accuracy_top(pred_delta_seg, true_delta_seg, top_idx)
                if print_every and (local_it + 1 == iters_per_seg or (local_it + 1) % print_every == 0):
                    print(
                        f"    global_iter {global_iter:>2}/{total_iters} | "
                        f"seg_local {local_it + 1}/{iters_per_seg} | acc@{trans_name}={acc:.2%} | inv={inv:.2%}"
                    )

            pred_mean = vals_all[:, 1:].numpy().mean(axis=0)
            pred_delta_seg = pred_mean - baseline_true
            pred_late_like = baseline_true + pred_delta_seg
            dir_true = np.where(true_delta_seg > 0, "Up", "Down")
            dir_pred = np.where(pred_delta_seg > 0, "Up", "Down")
            dir_correct = (dir_true == dir_pred).astype(int)

            segment_rows.append(
                {
                    "transition": trans_name,
                    "seg_from": seg_i,
                    "seg_to": seg_i + 1,
                    "iters_in_block": iters_per_seg,
                    "global_iter_end": global_iter,
                    "acc_top_percent": direction_accuracy_top(pred_delta_seg, true_delta_seg, top_idx)[0],
                }
            )

            gr = pd.DataFrame(
                {
                    "gene": genes,
                    "true_baseline_mean": baseline_true.astype(np.float32),
                    "true_early_mean_local": true_early_local.astype(np.float32),
                    "true_late_mean": true_late.astype(np.float32),
                    "pred_late_like_mean": pred_late_like.astype(np.float32),
                    "delta_true_local": true_delta_local.astype(np.float32),
                    "delta_true": true_delta_seg.astype(np.float32),
                    "delta_pred": pred_delta_seg.astype(np.float32),
                    "dir_true": dir_true,
                    "dir_pred": dir_pred,
                    "dir_correct": dir_correct,
                }
            )
            snapshots.append({"transition": trans_name, "df": gr})
            if save_embeddings:
                chain_nodes.append(_embed_cohort())
                print(f"  [embeddings] chain node S{seg_i + 1} saved after {trans_name}")

    # 宽表：每基因在各 transition 的预测方向
    wide = pd.DataFrame({"gene": genes})
    for snap in snapshots:
        t = snap["transition"]
        sub = snap["df"].set_index("gene")
        wide[f"{t}_delta_pred"] = wide["gene"].map(sub["delta_pred"])
        wide[f"{t}_dir_pred"] = wide["gene"].map(sub["dir_pred"])
        wide[f"{t}_dir_correct"] = wide["gene"].map(sub["dir_correct"])

    ds_out = outdir / dataset / "chained_preds"
    ds_out.mkdir(parents=True, exist_ok=True)
    for snap in snapshots:
        snap["df"].to_csv(ds_out / f"{dataset}_{snap['transition']}_gene_result.csv", index=False)
    pd.DataFrame(segment_rows).to_csv(ds_out / f"{dataset}_chained_segment_summary.csv", index=False)
    wide.to_csv(ds_out / f"{dataset}_chained_per_gene_wide.csv", index=False)

    if save_embeddings and true_cents is not None and chain_nodes:
        emb_dir = ds_out / "chained_embeddings"
        emb_dir.mkdir(parents=True, exist_ok=True)
        chain_cents = np.stack(chain_nodes, axis=0).astype(np.float32)
        np.save(emb_dir / "chain_segment_centroids.npy", chain_cents)
        np.save(emb_dir / "true_segment_centroids.npy", true_cents)
        summary = {
            "dataset": dataset,
            "n_segments": n_segments,
            "true_step_l2": pairwise_steps(true_cents),
            "chain_step_l2": pairwise_steps(chain_cents),
            "true_path_length": path_length(true_cents),
            "chain_path_length": path_length(chain_cents),
            "path_length_ratio_chain_over_true": path_length(chain_cents)
            / max(path_length(true_cents), 1e-9),
        }
        with open(emb_dir / "trajectory_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Chained embeddings: {emb_dir}")

    print(f"\nSaved chained outputs: {ds_out}")
    print("  Per-segment gene_result: {dataset}_delta_t{i}_t{j}_gene_result.csv")
    print("  Wide table: chained_per_gene_wide.csv")


def main() -> None:
    p = argparse.ArgumentParser(description="Chained scGPT: one run, one result per PT segment")
    p.add_argument("--dataset", default="hESC")
    p.add_argument("--n-segments", type=int, default=5)
    p.add_argument("--gen-iters", type=int, default=16, help="Total iterations split across transitions")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--ema-alpha", type=float, default=0.1)
    p.add_argument("--top-percent", type=int, default=30)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument(
        "--save-embeddings",
        action="store_true",
        help="save true + chained CLS centroids per PT node to chained_embeddings/",
    )
    args = p.parse_args()
    run_chained(
        args.dataset,
        args.n_segments,
        args.gen_iters,
        args.batch_size,
        args.ema_alpha,
        args.top_percent,
        args.outdir,
        args.print_every,
        args.save_embeddings,
    )


if __name__ == "__main__":
    main()
