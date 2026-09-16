#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast path: scGPT zero-shot per-cell velocity ONLY (for veloBench eval).

Skips early/late accuracy curves and per-time loops in scgpt_dyn.py.
Prints progress during the long per-cell MLM iteration.

Example (Dataset 12, GPU):
  conda activate singlecell
  export CUDA_VISIBLE_DEVICES=0
  cd /mnt/10T/yzn/scGRN-Bench

  python -u src/GRN_inferance/dyn/scgpt_per_cell_velocity_only.py \\
    --model-dir /mnt/10T/yzn/benchmark_GRN/model/weights/scgpt/scGPT_human \\
    --expr-csv outputs/dynamo_fm_velocity/dataset12_official/export/CHIP/dataset12_chip_matched-ExpressionData.csv \\
    --out-npz outputs/dynamo_fm_velocity/dataset12_official/scgpt_zero_shot/dataset12_official/velocity_field.npz \\
    --batch-size 128 \\
    --gen-iters 16
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
os.environ["KMP_WARNINGS"] = "off"

from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab


def parse_args():
    p = argparse.ArgumentParser(description="scGPT per-cell velocity only (fast benchmark path)")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--expr-csv", required=True, help="genes x cells CSV (CHIP export)")
    p.add_argument("--out-npz", required=True)
    p.add_argument("--gen-iters", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--ema-alpha", type=float, default=0.1)
    p.add_argument("--log1p", action="store_true")
    return p.parse_args()


def bin_expr_to_0_50(x: np.ndarray, do_log1p: bool) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0)
    x = np.clip(x, 0, None)
    if do_log1p:
        x = np.log1p(x)
    vmax = max(float(np.percentile(x, 99.5)), 1e-6)
    return np.clip(x / vmax * 50.0, 0, 50).astype(np.float32)


def build_model(model_dir: str, device: torch.device):
    with open(Path(model_dir) / "args.json") as f:
        cfg = json.load(f)

    vocab = GeneVocab.from_file(Path(model_dir) / "vocab.json")
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
        use_fast_transformer=cfg.get("fast_transformer", True),
    )

    ckpt = torch.load(Path(model_dir) / "best_model.pt", map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()
    if device.type == "cuda":
        model.half()
    return model, vocab


@torch.no_grad()
def run_iterative_per_cell(model, gene_ids_tensor, values_tensor, pad_mask, update_mask_1d, gen_iters, batch_size, ema_alpha):
    device = next(model.parameters()).device
    n_cells = values_tensor.shape[0]
    vals_all = values_tensor.clone()
    initial = vals_all[:, 1:].float().cpu().numpy()
    update_mask = torch.tensor(update_mask_1d[None, :], device=device).bool()
    n_batches_per_iter = (n_cells + batch_size - 1) // batch_size
    total_steps = gen_iters * n_batches_per_iter
    done = 0
    t0 = time.time()

    for it in range(gen_iters):
        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)
            bs = end - start
            vals = vals_all[start:end].to(device)
            src = gene_ids_tensor.expand(bs, -1).to(device)
            mask = pad_mask[start:end].to(device)
            freeze = mask | (~update_mask.expand(bs, -1))
            out = model(src=src, values=vals, src_key_padding_mask=mask)
            new_vals = out["mlm_output"]
            vals = torch.where(freeze, vals, ema_alpha * vals + (1 - ema_alpha) * new_vals)
            vals_all[start:end] = vals.detach().cpu()
            done += 1
            if done == 1 or done % max(1, n_batches_per_iter // 5) == 0 or done == total_steps:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                eta = (total_steps - done) / max(rate, 1e-6)
                print(
                    f"  [progress] iter {it+1}/{gen_iters} batch {start}-{end}/{n_cells} "
                    f"| step {done}/{total_steps} ({100*done/total_steps:.1f}%) "
                    f"| elapsed {elapsed/60:.1f}m ETA {eta/60:.1f}m",
                    flush=True,
                )

    final = vals_all[:, 1:].float().cpu().numpy()
    velocity = (final - initial) / max(gen_iters, 1)
    return np.nan_to_num(velocity.astype(np.float32), nan=0.0)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU required. Check: python -c \"import torch; print(torch.cuda.is_available())\"")
    print(f"Using device: {device}", flush=True)

    expr = pd.read_csv(args.expr_csv, index_col=0)
    genes = expr.index.astype(str).tolist()
    cells = expr.columns.astype(str).tolist()
    print(f"Loaded expr: {len(genes)} genes x {len(cells)} cells", flush=True)

    X = expr.T.to_numpy(dtype=np.float32)
    X_bin = bin_expr_to_0_50(X, do_log1p=args.log1p)

    model, vocab = build_model(args.model_dir, device)
    print(f"Vocab size: {len(vocab)}", flush=True)

    gene_ids = np.array([vocab[g] if g in vocab else vocab["<pad>"] for g in genes])
    gene_ids = np.concatenate([[vocab["<cls>"]], gene_ids])
    gene_ids_tensor = torch.tensor(gene_ids[None, :], dtype=torch.long)

    X_in = np.concatenate([np.zeros((X_bin.shape[0], 1)), X_bin], axis=1)
    pad_mask = gene_ids_tensor.eq(vocab["<pad>"]).expand(X_in.shape[0], -1)
    values_tensor = torch.tensor(X_in, dtype=torch.float16 if device.type == "cuda" else torch.float32)
    update_mask_1d = np.zeros(gene_ids_tensor.shape[1], dtype=bool)
    update_mask_1d[1:] = True

    print(
        f"Running per-cell velocity: {len(cells)} cells, {args.gen_iters} iters, batch={args.batch_size}",
        flush=True,
    )
    vel = run_iterative_per_cell(
        model, gene_ids_tensor, values_tensor, pad_mask, update_mask_1d,
        args.gen_iters, args.batch_size, args.ema_alpha,
    )

    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, vel_cell=vel, cells=np.array(cells), genes=np.array(genes))
    print(f"Saved -> {out}  shape={vel.shape}", flush=True)


if __name__ == "__main__":
    main()
