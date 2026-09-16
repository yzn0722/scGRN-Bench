#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
链式 scGPT 每 block 结束时的 CLS embedding 质心（S0 起点 + 每段末状态）。

输出目录：multistep_pt/<dataset>/chained_preds/chained_embeddings/
  - chain_segment_centroids.npy   (n_segments, dim)
  - true_segment_centroids.npy    (n_segments, dim)  全细胞按 PT 分段的质心
  - trajectory_summary.json

需 GPU + scGPT 环境。与 run_scgpt_chained_segments 使用相同 cohort / iters。

示例：
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig4
  python3 extract_chained_block_embeddings.py --dataset hESC
  python3 plot_hESC_mechanistic_interpret.py --plot-embedding-compare
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chained_embedding_utils import (
    extract_cell_embeddings,
    pairwise_steps,
    path_length,
    segment_centroids,
)

# reuse data / model helpers from chained run
from run_scgpt_chained_segments import (  # noqa: E402
    CHIP_DIR,
    DEFAULT_OUT,
    PT_ROOT,
    bin_expr_to_0_50,
    load_scgpt,
    mask_segment_bins,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def run_extract(
    dataset: str,
    n_segments: int,
    gen_iters: int,
    batch_size: int,
    ema_alpha: float,
    outdir: Path,
) -> Path:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{dataset}] extract chained block embeddings | device={device}")

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

    cohort = masks[0]
    if cohort.sum() == 0:
        raise ValueError("S0 cohort empty")

    model, vocab, torch, _ = load_scgpt(device)
    gene_ids = np.array([vocab[g] if g in vocab else vocab["<pad>"] for g in genes], dtype=np.int64)
    gene_ids = np.concatenate([[vocab["<cls>"]], gene_ids])

    values = np.concatenate([np.zeros((X_bin.shape[0], 1), dtype=np.float32), X_bin], axis=1)
    values_tensor = torch.tensor(values, dtype=torch.float16 if device.type == "cuda" else torch.float32)

    gene_ids_tensor = torch.tensor(gene_ids[None, :], dtype=torch.long)
    pad_early = gene_ids_tensor.eq(vocab["<pad>"]).expand(values.shape[0], -1)
    update_mask = np.zeros(len(gene_ids), dtype=bool)
    update_mask[1:] = True
    update_mask_t = torch.tensor(update_mask[None, :], device=device).bool()

    vals_all = values_tensor[cohort].clone()
    pad_cohort = pad_early[cohort]

    # --- true path: all cells embedded once ---
    print("  Embedding all cells (true observed states)...")
    true_emb = extract_cell_embeddings(model, vocab, torch, values, gene_ids, batch_size=batch_size)
    seg_ids = np.zeros(len(pt), dtype=np.int64)
    for i, m in enumerate(masks):
        seg_ids[m] = i
    true_cents = segment_centroids(true_emb, seg_ids, n_segments)

    # --- chain path: S0 start + after each block ---
    chain_nodes: list[np.ndarray] = []

    def embed_cohort_vals() -> np.ndarray:
        arr = vals_all.numpy().astype(np.float32)
        emb = extract_cell_embeddings(model, vocab, torch, arr, gene_ids, batch_size=batch_size)
        return emb.mean(axis=0)

    print(f"  S0 cohort n={int(cohort.sum())} | {iters_per_seg} iters/block × {n_trans} blocks")
    chain_nodes.append(embed_cohort_vals())
    print("    node S0 (chain start) saved")

    with torch.no_grad():
        for seg_i in range(n_trans):
            trans = f"delta_t{seg_i}_t{seg_i + 1}"
            for _ in range(iters_per_seg):
                for start in range(0, vals_all.shape[0], batch_size):
                    end = min(start + batch_size, vals_all.shape[0])
                    vals = vals_all[start:end].to(device)
                    src = gene_ids_tensor.expand(end - start, -1).to(device)
                    mask = pad_cohort[start:end].to(device)
                    freeze = mask | (~update_mask_t.expand(end - start, -1))
                    out = model(src=src, values=vals, src_key_padding_mask=mask)
                    new_vals = out["mlm_output"]
                    vals = torch.where(
                        freeze,
                        vals,
                        ema_alpha * vals + (1 - ema_alpha) * new_vals,
                    )
                    vals_all[start:end] = vals.detach().cpu()

            chain_nodes.append(embed_cohort_vals())
            print(f"    after {trans} → node S{seg_i + 1} saved")

    chain_cents = np.stack(chain_nodes, axis=0).astype(np.float32)

    ds_out = outdir / dataset / "chained_preds" / "chained_embeddings"
    ds_out.mkdir(parents=True, exist_ok=True)
    np.save(ds_out / "chain_segment_centroids.npy", chain_cents)
    np.save(ds_out / "true_segment_centroids.npy", true_cents)

    true_steps = pairwise_steps(true_cents)
    chain_steps = pairwise_steps(chain_cents)
    summary = {
        "dataset": dataset,
        "n_segments": n_segments,
        "n_s0_cohort": int(cohort.sum()),
        "iters_per_block": iters_per_seg,
        "embedding_dim": int(chain_cents.shape[1]),
        "true_step_l2": true_steps,
        "chain_step_l2": chain_steps,
        "true_path_length": path_length(true_cents),
        "chain_path_length": path_length(chain_cents),
        "path_length_ratio_chain_over_true": path_length(chain_cents)
        / max(path_length(true_cents), 1e-9),
        "labels": [f"S{k}" for k in range(n_segments)],
        "transitions": [f"S{i}→S{i+1}" for i in range(n_trans)],
    }
    with open(ds_out / "trajectory_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved → {ds_out}")
    print(f"  true path length (L2 sum): {summary['true_path_length']:.4f}")
    print(f"  chain path length:         {summary['chain_path_length']:.4f}")
    print(f"  ratio chain/true:          {summary['path_length_ratio_chain_over_true']:.3f}")
    return ds_out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="hESC")
    p.add_argument("--n-segments", type=int, default=5)
    p.add_argument("--gen-iters", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--ema-alpha", type=float, default=0.1)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    run_extract(
        args.dataset,
        args.n_segments,
        args.gen_iters,
        args.batch_size,
        args.ema_alpha,
        args.outdir,
    )


if __name__ == "__main__":
    main()
