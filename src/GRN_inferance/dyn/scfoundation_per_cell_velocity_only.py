#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scFoundation zero-shot per-cell velocity for veloBench eval.

Iterative MAE autobin inference on all cells; velocity = (final - initial) / n_iters
in export gene space (same npz layout as scGPT: vel_cell, cells, genes).

Example (HumanBoneMarrow, GPU):
  conda activate singlecell
  export CUDA_VISIBLE_DEVICES=0
  cd /mnt/10T/yzn/scGRN-Bench

  python -u src/GRN_inferance/dyn/scfoundation_per_cell_velocity_only.py \\
    --expr-csv outputs/dynamo_fm_velocity/velobench_four_scgpt/HumanBoneMarrow/export/CHIP/HumanBoneMarrow_chip_matched-ExpressionData.csv \\
    --out-npz outputs/dynamo_fm_velocity/velobench_four_scfoundation/HumanBoneMarrow/scfoundation_zero_shot/HumanBoneMarrow/velocity_field.npz \\
    --gen-iters 21 --batch-size 16
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
os.environ["KMP_WARNINGS"] = "off"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

DEFAULT_SCF_ROOT = Path("/mnt/10T/yzn/scFoundation-main/model")
DEFAULT_SCF_CKPT = DEFAULT_SCF_ROOT / "models" / "models.ckpt"
DEFAULT_SCF_GENE_TSV = DEFAULT_SCF_ROOT / "OS_scRNA_gene_index.19264.tsv"

MODEL_CFG = {
    "model": "mae_autobin",
    "seq_len": 19266,
    "n_class": 100,
    "bin_alpha": 1.0,
    "bin_num": 100,
    "pad_token_id": 0,
    "mask_token_id": 1,
    "encoder": {
        "module_type": "transformer",
        "hidden_dim": 768,
        "depth": 12,
        "heads": 12,
        "dim_head": 64,
        "ff_dropout": 0.0,
        "attn_dropout": 0.0,
    },
    "decoder": {
        "module_type": "transformer",
        "hidden_dim": 512,
        "depth": 6,
        "heads": 8,
        "dim_head": 64,
        "ff_dropout": 0.0,
        "attn_dropout": 0.0,
    },
    "ppi_edge": None,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="scFoundation per-cell velocity (veloBench fast path)")
    p.add_argument("--expr-csv", required=True, type=str)
    p.add_argument("--out-npz", required=True, type=str)
    p.add_argument("--scf-root", default=str(DEFAULT_SCF_ROOT), type=str)
    p.add_argument("--scf-ckpt", default=str(DEFAULT_SCF_CKPT), type=str)
    p.add_argument("--scf-gene-tsv", default=str(DEFAULT_SCF_GENE_TSV), type=str)
    p.add_argument("--scf-mmf-key", default="gene", type=str)
    p.add_argument("--gen-iters", type=int, default=12)
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Cells per forward (try 8 on 24GB; auto-halves on OOM).",
    )
    p.add_argument("--cell-start", type=int, default=0, help="First cell index (0-based, inclusive).")
    p.add_argument("--cell-end", type=int, default=-1, help="End cell index (exclusive). -1 = all cells.")
    p.add_argument("--checkpoint-every", type=int, default=25, help="Save checkpoint every N batches (0=off).")
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint if present.")
    p.add_argument("--torch-compile", action="store_true", help="torch.compile(model) when supported.")
    p.add_argument("--ema-alpha", type=float, default=0.1, help="Update rate (same role as scGPT EMA).")
    p.add_argument("--mode", default="mae", choices=["mae", "zero"])
    p.add_argument("--value-mask-prob", type=float, default=0.3)
    p.add_argument("--zero-mask-prob", type=float, default=0.0)
    p.add_argument("--update-scope", default="present_all", choices=["mask", "zero", "present_all"])
    p.add_argument("--no-refresh-encoder", action="store_true")
    p.add_argument("--no-resample-mask", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


# gather_data pads with pad_value; must not collide with real expression (>=0) or gene index (>=0).
GATHER_PAD_EXPR = -1.0
GATHER_PAD_IDX = -1


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_path(out_npz: Path) -> Path:
    return out_npz.with_suffix(out_npz.suffix + ".ckpt.npz")


def load_checkpoint(path: Path, n_cells: int, n_genes: int) -> tuple[np.ndarray, int]:
    if not path.exists():
        return np.zeros((n_cells, n_genes), dtype=np.float32), 0
    d = np.load(path)
    completed = int(d["completed_end"])
    vel_part = d["vel_cell"].astype(np.float32)
    if vel_part.shape == (n_cells, n_genes):
        vel = vel_part
    elif vel_part.shape[0] <= n_cells and vel_part.shape[1] == n_genes:
        vel = np.zeros((n_cells, n_genes), dtype=np.float32)
        vel[: vel_part.shape[0]] = vel_part
    else:
        raise ValueError(f"Checkpoint shape {vel_part.shape} != expected {(n_cells, n_genes)}")
    completed = min(completed, n_cells)
    print(f"[INFO] Resume checkpoint: {completed}/{n_cells} cells done", flush=True)
    return vel, completed


def save_checkpoint(path: Path, vel: np.ndarray, cells: list[str], genes: list[str], completed_end: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    completed_end = min(int(completed_end), len(cells))
    np.savez_compressed(
        path,
        vel_cell=vel[:completed_end],
        cells=np.array(cells),
        genes=np.array(genes),
        completed_end=np.array(completed_end),
        total_cells=np.array(len(cells)),
    )


def load_expression_slice(csv_path: Path, cell_start: int, cell_end: int) -> tuple[list[str], list[str], np.ndarray]:
    """Load only genes x [cell_start:cell_end] from export CSV (avoids 65k-column RAM spike)."""
    header = pd.read_csv(csv_path, nrows=0, index_col=0)
    all_cells = header.columns.astype(str).tolist()
    n_total = len(all_cells)
    cell_start = max(0, int(cell_start))
    cell_end = n_total if int(cell_end) < 0 else min(int(cell_end), n_total)
    if cell_start >= cell_end:
        raise ValueError(f"Invalid cell range [{cell_start}, {cell_end})")
    cells = all_cells[cell_start:cell_end]
    expr = pd.read_csv(csv_path, index_col=0, usecols=cells)
    genes = expr.index.astype(str).tolist()
    x = np.nan_to_num(expr.T.to_numpy(dtype=np.float32), nan=0.0)
    del expr
    return genes, cells, x


def _strip_prefix(sd: dict, prefix: str) -> dict:
    return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in sd.items()}


def gather_data(data: torch.Tensor, labels: torch.Tensor, pad_value: float):
    value_nums = labels.sum(1)
    max_num = int(value_nums.max().item())
    fake_data = torch.full((data.shape[0], max_num), pad_value, device=data.device, dtype=data.dtype)
    data2 = torch.hstack([data, fake_data])
    fake_label = torch.full((labels.shape[0], max_num), 1, device=labels.device)
    none_labels = ~labels
    labels_f = labels.float()
    labels_f[none_labels] = torch.tensor(-float("Inf"), device=labels.device)
    tmp_data = torch.tensor([(i + 1) * 20000 for i in range(labels.shape[1], 0, -1)], device=labels.device)
    labels_f = labels_f + tmp_data
    labels_f = torch.hstack([labels_f, fake_label])
    idx = labels_f.topk(max_num).indices
    new_data = torch.gather(data2, 1, idx)
    padding_labels = new_data.eq(pad_value)
    return new_data, padding_labels


def load_model(ckpt_path: str, scf_root: str, mmf_key: str, device: torch.device, use_compile: bool = False):
    if scf_root not in sys.path:
        sys.path.insert(0, scf_root)
    from pretrainmodels import select_model

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if mmf_key not in ckpt:
        raise RuntimeError(f"Checkpoint keys={list(ckpt.keys())}, missing {mmf_key!r}")
    gene_block = ckpt[mmf_key]
    sd = _strip_prefix(gene_block["state_dict"], "model.")
    seq_len = int(sd["pos_emb.weight"].shape[0] - 1) if "pos_emb.weight" in sd else int(MODEL_CFG["seq_len"])
    enc_type = "performer" if any(k.startswith("encoder.performer.") for k in sd) else "transformer"
    dec_type = "performer" if any(k.startswith("decoder.performer.") for k in sd) else "transformer"

    config = dict(MODEL_CFG)
    config["seq_len"] = seq_len
    config["encoder"] = dict(config["encoder"])
    config["decoder"] = dict(config["decoder"])
    config["encoder"]["module_type"] = enc_type
    config["decoder"]["module_type"] = dec_type

    print(f"[INFO] encoder={enc_type} decoder={dec_type} seq_len={seq_len}", flush=True)
    model = select_model(config)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    if device.type == "cuda":
        model.half()
    if use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("[INFO] torch.compile enabled", flush=True)
        except Exception as exc:
            print(f"[WARN] torch.compile failed: {exc}", flush=True)
    return model, config


def read_gene_index_tsv(path: str, g_model: int) -> dict[str, int]:
    df = pd.read_csv(path, sep="\t", header=0)
    cols = {c.lower(): c for c in df.columns}
    gene_col = cols.get("gene_name", df.columns[0])
    idx_col = cols.get("index")
    if idx_col is None:
        raise RuntimeError(f"TSV missing index column: {list(df.columns)}")
    df = df[[gene_col, idx_col]].dropna()
    df[gene_col] = df[gene_col].astype(str)
    df[idx_col] = pd.to_numeric(df[idx_col], errors="coerce").astype("Int64")
    df = df.dropna()
    df = df[(df[idx_col] >= 0) & (df[idx_col] < g_model)]
    return dict(zip(df[gene_col].tolist(), df[idx_col].astype(int).tolist()))


def build_aligned_matrix(x: np.ndarray, genes: list[str], gene2idx: dict[str, int], g_model: int):
    n_cells, n_genes = x.shape
    x_full = np.zeros((n_cells, g_model), dtype=np.float32)
    present_mask = np.zeros(g_model, dtype=bool)
    map_idx = np.full(n_genes, -1, dtype=np.int64)
    found = 0
    for j, g in enumerate(genes):
        idx = gene2idx.get(g)
        if idx is None:
            continue
        idx = int(idx)
        if 0 <= idx < g_model:
            x_full[:, idx] = x[:, j]
            present_mask[idx] = True
            map_idx[j] = idx
            found += 1
    return x_full, present_mask, map_idx, found


def make_masks_present_only(
    raw_full: torch.Tensor,
    present_mask: torch.Tensor,
    mode: str,
    value_mask_prob: float,
    zero_mask_prob: float,
    seed: int,
):
    device = raw_full.device
    b, g = raw_full.shape
    present = present_mask.view(1, g).expand(b, g)
    nonzero_present = (raw_full > 0) & present

    if mode == "zero":
        encoder_visible = nonzero_present
        update_pos = (~nonzero_present) & present
    else:
        seed_all(seed)
        rnd = torch.rand((b, g), device=device)
        masked_nonzero = nonzero_present & (rnd < value_mask_prob)
        masked_zero = (
            ((~nonzero_present) & present) & (torch.rand((b, g), device=device) < zero_mask_prob)
            if zero_mask_prob > 0
            else torch.zeros((b, g), dtype=torch.bool, device=device)
        )
        update_pos = masked_nonzero | masked_zero
        encoder_visible = (~update_pos) & nonzero_present

    bad = encoder_visible.sum(dim=1) == 0
    if bad.any():
        first_present = int(torch.nonzero(present_mask, as_tuple=False).min().item())
        encoder_visible[bad] = nonzero_present[bad]
        update_pos[bad] = masked_nonzero[bad] if mode == "mae" else update_pos[bad]
        bad2 = encoder_visible.sum(dim=1) == 0
        if bad2.any():
            encoder_visible[bad2, first_present] = True
            update_pos[bad2, first_present] = False
    return encoder_visible, update_pos


def build_io(vals: torch.Tensor, encoder_visible: torch.Tensor, config: dict):
    device = vals.device
    decoder_data = vals
    decoder_data_padding = torch.full_like(decoder_data, False, dtype=torch.bool, device=device)

    encoder_data, encoder_data_padding = gather_data(decoder_data, encoder_visible, GATHER_PAD_EXPR)
    gene_ids = torch.arange(decoder_data.shape[1], device=device, dtype=torch.float32).unsqueeze(0).repeat(
        decoder_data.shape[0], 1
    )
    encoder_pos, _ = gather_data(gene_ids, encoder_visible, GATHER_PAD_EXPR)
    decoder_pos = gene_ids.long()
    encoder_pos[encoder_data_padding] = float(config["seq_len"])
    decoder_pos[decoder_data_padding] = config["seq_len"]

    n_labels = int(encoder_visible.sum().item())
    n_tokens = int((~encoder_data_padding).sum().item())
    if n_labels != n_tokens:
        raise RuntimeError(
            f"encoder pack mismatch: encoder_visible={n_labels} tokens={n_tokens} "
            "(report if this persists after pad sentinel fix)"
        )

    return encoder_data, encoder_data_padding, encoder_pos.long(), encoder_visible, decoder_data, decoder_data_padding, decoder_pos


@torch.no_grad()
def iterative_predict_batch(
    model,
    config: dict,
    values_init: torch.Tensor,
    raw_full: torch.Tensor,
    present_mask: torch.Tensor,
    n_iters: int,
    seed0: int,
    args: argparse.Namespace,
) -> torch.Tensor:
    device = next(model.parameters()).device
    vals = values_init.to(device)
    raw_full = raw_full.to(device)
    present_mask = present_mask.to(device)
    b, g = vals.shape
    resample = not args.no_resample_mask

    encoder_visible, update_pos = make_masks_present_only(
        raw_full, present_mask, args.mode, args.value_mask_prob, args.zero_mask_prob, seed0
    )

    for it in range(n_iters):
        if args.mode == "mae" and resample and it > 0:
            encoder_visible, update_pos = make_masks_present_only(
                raw_full, present_mask, args.mode, args.value_mask_prob, args.zero_mask_prob, seed0 + it + 1
            )
        enc = build_io(vals, encoder_visible, config)
        encoder_data, padding_label, enc_pos, enc_labels, dec_data, dec_pad, dec_pos = enc

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            pred = model(
                x=encoder_data,
                padding_label=padding_label,
                encoder_position_gene_ids=enc_pos,
                encoder_labels=enc_labels,
                decoder_data=dec_data,
                mask_gene_name=False,
                mask_labels=None,
                decoder_position_gene_ids=dec_pos,
                decoder_data_padding_labels=dec_pad,
            )

        del encoder_data, padding_label, enc_pos, enc_labels, dec_data, dec_pad, dec_pos

        present = present_mask.view(1, g).expand(b, g)
        if args.update_scope == "mask":
            pos = update_pos
        elif args.update_scope == "zero":
            pos = (raw_full <= 0) & present
        else:
            pos = present

        alpha = args.ema_alpha
        if alpha >= 1.0:
            vals[pos] = pred[pos]
        else:
            vals[pos] = (1.0 - alpha) * vals[pos] + alpha * pred[pos]

    return vals.detach().float()


def model_delta_to_gene_velocity(
    initial_model: np.ndarray,
    final_model: np.ndarray,
    map_idx: np.ndarray,
    n_iters: int,
) -> np.ndarray:
    delta_model = (final_model - initial_model) / max(n_iters, 1)
    n_cells = initial_model.shape[0]
    vel = np.zeros((n_cells, len(map_idx)), dtype=np.float32)
    valid = map_idx >= 0
    if valid.any():
        vel[:, valid] = delta_model[:, map_idx[valid]]
    return np.nan_to_num(vel, nan=0.0)


def process_cell_chunk(
    model,
    config: dict,
    x_full: np.ndarray,
    present_mask: torch.Tensor,
    map_idx: np.ndarray,
    start: int,
    end: int,
    args: argparse.Namespace,
) -> np.ndarray:
    device = next(model.parameters()).device
    vals0 = torch.tensor(x_full[start:end], dtype=torch.float32)
    rawb = torch.tensor(x_full[start:end], dtype=torch.float32)
    try:
        final = iterative_predict_batch(
            model, config, vals0, rawb, present_mask, args.gen_iters, args.seed + start, args
        )
        return model_delta_to_gene_velocity(
            x_full[start:end], final.cpu().numpy(), map_idx, args.gen_iters
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if isinstance(exc, RuntimeError) and "encoder pack mismatch" not in str(exc):
            raise
        if device.type == "cuda":
            torch.cuda.empty_cache()
        n = end - start
        if n <= 1:
            raise RuntimeError(
                "CUDA OOM even with batch_size=1. Close other GPU jobs or use a larger GPU."
            ) from None
        mid = start + n // 2
        print(
            f"  [WARN] {type(exc).__name__} on cells {start}-{end}, splitting into {start}-{mid} and {mid}-{end}",
            flush=True,
        )
        left = process_cell_chunk(model, config, x_full, present_mask, map_idx, start, mid, args)
        right = process_cell_chunk(model, config, x_full, present_mask, map_idx, mid, end, args)
        return np.vstack([left, right])


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU required for scFoundation velocity inference.")
    print(f"Using device: {device}", flush=True)

    cell_start = max(0, int(args.cell_start))
    cell_end = int(args.cell_end)
    genes, cells, x = load_expression_slice(Path(args.expr_csv), cell_start, cell_end)
    print(
        f"Loaded expr: {len(genes)} genes x {len(cells)} cells "
        f"(slice [{cell_start}:{cell_end if cell_end >= 0 else 'all'}])",
        flush=True,
    )

    model, config = load_model(
        args.scf_ckpt, args.scf_root, args.scf_mmf_key, device, use_compile=args.torch_compile
    )
    g_model = int(config["seq_len"])
    gene2idx = read_gene_index_tsv(args.scf_gene_tsv, g_model)
    x_full, present_mask_np, map_idx, mapped = build_aligned_matrix(x, genes, gene2idx, g_model)
    print(f"Mapped genes: {mapped}/{len(genes)} ({100 * mapped / max(1, len(genes)):.1f}%)", flush=True)

    present_mask = torch.tensor(present_mask_np, dtype=torch.bool)
    out = Path(args.out_npz)
    ckpt = checkpoint_path(out)

    vel_all, completed_end = (
        load_checkpoint(ckpt, len(cells), len(genes)) if args.resume else (np.zeros((len(cells), len(genes)), dtype=np.float32), 0)
    )

    print(
        f"Running per-cell velocity: batch_size={args.batch_size}, gen_iters={args.gen_iters}, "
        f"resume_from={completed_end}",
        flush=True,
    )

    n_batches = (len(cells) + args.batch_size - 1) // args.batch_size
    done = 0
    t0 = time.time()

    for start in range(0, len(cells), args.batch_size):
        end = min(start + args.batch_size, len(cells))
        if end <= completed_end:
            done += 1
            continue

        vel_all[start:end] = process_cell_chunk(
            model, config, x_full, present_mask, map_idx, start, end, args
        )
        done += 1
        elapsed = time.time() - t0
        rate = done / max(elapsed, 1e-6)
        eta = (n_batches - done) / max(rate, 1e-6)
        print(
            f"  [progress] batch {start}-{end}/{len(cells)} "
            f"| {done}/{n_batches} ({100 * done / n_batches:.1f}%) "
            f"| elapsed {elapsed / 60:.1f}m ETA {eta / 60:.1f}m",
            flush=True,
        )

        if args.checkpoint_every > 0 and (done % args.checkpoint_every == 0 or end == len(cells)):
            save_checkpoint(ckpt, vel_all, cells, genes, end)
            print(f"  [checkpoint] saved -> {ckpt} (completed_end={end})", flush=True)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, vel_cell=vel_all, cells=np.array(cells), genes=np.array(genes))
    print(f"Saved -> {out}  shape={vel_all.shape}", flush=True)
    if ckpt.exists():
        ckpt.unlink()
        print(f"[INFO] Removed checkpoint {ckpt}", flush=True)


if __name__ == "__main__":
    main()
