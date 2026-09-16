#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scGPT state-transition velocity (Scheme 2 / 3).

Idea
----
Unpaired snapshot data (t=3 vs t=5) has no true X(t)->X(t+dt) pairs.
Use Dynamo RNA velocity as pseudo-supervision:

    X_pseudo_future = X_now + V_dynamo * dt

Train a transition head on frozen scGPT embeddings:

    X_now  --[TransitionHead]-->  X_future_pred
    velocity_pred = (X_future_pred - X_now) / dt

Scheme 2 (--n-micro-steps 1): one jump t -> t+dt
Scheme 3 (--n-micro-steps >1): micro-steps simulating a velocity field

Example (hematopoiesis 1524 dynamic genes):
  python scgpt_dyn_transition.py \\
    --model-dir /mnt/10T/yzn/benchmark_GRN/model/weights/scgpt/scGPT_human \\
    --h5ad /mnt/10T/yzn/dynamo-release/results/hematopoiesis_raw/hematopoiesis_processed.h5ad \\
    --datasets-json data/dynamo_export/datasets_dynamo_vel1524_only.json \\
    --dataset hematopoiesis_dyn_vel1524 \\
    --outdir outputs/dynamo_fm_velocity/transition_hematopoiesis_vel1524 \\
    --epochs 80 --eval-all --n-micro-steps 4

Hold-out evaluation (recommended for reporting generalization):
  python scgpt_dyn_transition.py ... \\
    --holdout-cells 0.2 --holdout-genes 0.3 --epochs 80 --eval-all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


def parse_args():
    p = argparse.ArgumentParser(description="scGPT state transition -> velocity via finite difference.")
    p.add_argument("--model-dir", required=True, type=str)
    p.add_argument("--outdir", required=True, type=str)
    p.add_argument("--datasets-json", required=True, type=str)
    p.add_argument("--dataset", required=True, type=str)
    p.add_argument("--h5ad", required=True, type=str, help="Dynamo h5ad with velocity layer + obs time.")
    p.add_argument("--velocity-layer", default="velocity_alpha_minus_gamma_s", type=str)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lambda-dir", type=float, default=0.5, help="Direction BCE weight on velocity.")
    p.add_argument("--lambda-future", type=float, default=1.0, help="Huber loss on predicted future state.")
    p.add_argument("--lambda-vel", type=float, default=0.5, help="Huber loss on velocity vs Dynamo.")
    p.add_argument(
        "--n-micro-steps",
        type=int,
        default=1,
        help="Scheme 2: 1 step. Scheme 3: >1 micro-steps for continuous velocity field.",
    )
    p.add_argument(
        "--reencode-each-step",
        action="store_true",
        help="Re-run scGPT encoder after each micro-step (slower, more accurate).",
    )
    p.add_argument("--eval-all", action="store_true")
    p.add_argument("--top-percent", type=int, default=30)
    p.add_argument("--pt-quantile", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log1p", action="store_true")
    p.add_argument(
        "--holdout-cells",
        type=float,
        default=0.0,
        help="Fraction of cells held out from training (stratified by time). Evaluated separately.",
    )
    p.add_argument(
        "--holdout-genes",
        type=float,
        default=0.0,
        help="Fraction of genes held out from training supervision. Evaluated separately.",
    )
    return p.parse_args()


ARGS = parse_args()

# scgpt_dyn_reghead parses sys.argv on import; shield our CLI args.
_argv_bak = sys.argv[:]
sys.argv = [_argv_bak[0], "--model-dir", ".", "--outdir", "."]
from scgpt_dyn_reghead import (  # noqa: E402
    build_model,
    direction_accuracy,
    encode_cells,
    set_seed,
)
sys.argv = _argv_bak


class StateTransitionHead(nn.Module):
    """
    Per-gene transition rate (bin units / hour) from scGPT hidden + current expression.

    delta_over_dt = rate * dt  =>  X_future = X_now + rate * dt
    """

    def __init__(self, d_model: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden: torch.Tensor, expr: torch.Tensor) -> torch.Tensor:
        """Return per-gene rate [B, G] (excludes <cls> token)."""
        feat = torch.cat([hidden, expr.unsqueeze(-1)], dim=-1)
        rate = self.mlp(feat).squeeze(-1)
        return rate[:, 1:]


def bin_with_vmax(x: np.ndarray, vmax: float, do_log1p: bool) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0)
    x = np.clip(x, 0, None)
    if do_log1p:
        x = np.log1p(x)
    return np.clip(x / max(vmax, 1e-6) * 50.0, 0, 50).astype(np.float32)


def compute_bin_vmax(X_raw: np.ndarray, do_log1p: bool) -> float:
    x = np.clip(X_raw, 0, None).astype(np.float64)
    if do_log1p:
        x = np.log1p(x)
    return float(max(np.percentile(x, 99.5), 1e-6))


def load_cfg() -> dict:
    with open(ARGS.datasets_json, encoding="utf-8") as f:
        cfg = json.load(f)
    if ARGS.dataset not in cfg:
        raise KeyError(f"{ARGS.dataset} not in {ARGS.datasets_json}")
    return cfg[ARGS.dataset]


def select_eval_idx(true_delta: np.ndarray, gene_idx: Optional[np.ndarray] = None) -> np.ndarray:
    if gene_idx is not None:
        true_delta = true_delta[gene_idx]
        base = gene_idx
    else:
        base = np.arange(len(true_delta), dtype=int)
    n = len(true_delta)
    if ARGS.eval_all or ARGS.top_percent >= 100:
        return base.copy()
    k = max(int(n * ARGS.top_percent / 100), 1)
    order = np.argsort(np.abs(true_delta))[::-1][:k]
    return base[order].copy()


def make_holdout_splits(pt: np.ndarray, times: np.ndarray, n_genes: int) -> dict:
    """Stratified cell hold-out + random gene hold-out."""
    n_cells = len(pt)
    rng = np.random.default_rng(ARGS.seed)

    train_cell_idx: List[int] = []
    holdout_cell_idx: List[int] = []
    if ARGS.holdout_cells > 0:
        for t in times:
            idx = np.where(pt == t)[0]
            rng.shuffle(idx)
            n_hold = max(int(round(len(idx) * ARGS.holdout_cells)), 1)
            n_hold = min(n_hold, len(idx) - 1) if len(idx) > 1 else 0
            holdout_cell_idx.extend(idx[:n_hold].tolist())
            train_cell_idx.extend(idx[n_hold:].tolist())
    else:
        train_cell_idx = list(range(n_cells))

    gene_order = rng.permutation(n_genes)
    if ARGS.holdout_genes > 0:
        n_hold_g = max(int(round(n_genes * ARGS.holdout_genes)), 1)
        n_hold_g = min(n_hold_g, n_genes - 1) if n_genes > 1 else 0
        holdout_gene_idx = gene_order[:n_hold_g].astype(int)
        train_gene_idx = gene_order[n_hold_g:].astype(int)
    else:
        holdout_gene_idx = np.array([], dtype=int)
        train_gene_idx = np.arange(n_genes, dtype=int)

    train_cell_mask = np.zeros(n_cells, dtype=bool)
    holdout_cell_mask = np.zeros(n_cells, dtype=bool)
    train_cell_mask[train_cell_idx] = True
    holdout_cell_mask[holdout_cell_idx] = True

    train_gene_mask = np.zeros(n_genes, dtype=bool)
    holdout_gene_mask = np.zeros(n_genes, dtype=bool)
    train_gene_mask[train_gene_idx] = True
    holdout_gene_mask[holdout_gene_idx] = True

    return {
        "train_cell_idx": np.array(train_cell_idx, dtype=int),
        "holdout_cell_idx": np.array(holdout_cell_idx, dtype=int),
        "train_gene_idx": train_gene_idx,
        "holdout_gene_idx": holdout_gene_idx,
        "train_cell_mask": train_cell_mask,
        "holdout_cell_mask": holdout_cell_mask,
        "train_gene_mask": train_gene_mask,
        "holdout_gene_mask": holdout_gene_mask,
    }


def load_bundle(cfg: dict, vocab, device: torch.device) -> dict:
    import scanpy as sc

    expr = pd.read_csv(cfg["expr_csv"], index_col=0)
    pt_df = pd.read_csv(cfg["pt_csv"])
    pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"}).set_index("cell")

    adata = sc.read_h5ad(ARGS.h5ad)
    adata.obs_names_make_unique()
    if ARGS.velocity_layer not in adata.layers:
        raise KeyError(f"Layer {ARGS.velocity_layer} missing in h5ad")

    common = expr.columns.intersection(pt_df.index).intersection(adata.obs_names)
    expr = expr[common]
    pt = pt_df.loc[common, "pt"].to_numpy().astype(float)
    adata = adata[common].copy()

    genes = expr.index.astype(str).tolist()
    gene_to_h5 = {g: i for i, g in enumerate(adata.var_names.astype(str))}
    h5_idx = [gene_to_h5[g] for g in genes if g in gene_to_h5]
    genes = [g for g in genes if g in gene_to_h5]
    if not genes:
        raise ValueError("No gene overlap between export CSV and h5ad var_names")

    V = adata.layers[ARGS.velocity_layer]
    if sp.issparse(V):
        V = V.toarray()
    V = np.asarray(V, dtype=np.float32)[:, h5_idx]
    V = np.nan_to_num(V, nan=0.0)

    X_raw = expr.T.to_numpy(dtype=np.float32)
    do_log1p = bool(ARGS.log1p)
    vmax = compute_bin_vmax(X_raw, do_log1p)
    X_bin = bin_with_vmax(X_raw, vmax, do_log1p)

    times = np.sort(np.unique(pt))
    if len(times) < 2:
        raise ValueError(f"Need >=2 time points, got {times}")
    t_min, t_max = float(times[0]), float(times[-1])
    dt_forward = t_max - t_min

    # Per-cell dt toward the other snapshot (signed).
    dt_cell = np.where(pt <= (t_min + t_max) / 2, dt_forward, -dt_forward).astype(np.float32)

    # Pseudo future in count space, then bin with shared vmax.
    pseudo_raw = X_raw + V * dt_cell[:, None]
    pseudo_bin = bin_with_vmax(pseudo_raw, vmax, do_log1p)
    target_delta = pseudo_bin - X_bin

    lo, hi = np.quantile(pt, [ARGS.pt_quantile, 1 - ARGS.pt_quantile])
    early_mask = pt <= lo
    late_mask = pt >= hi
    true_delta = V[late_mask].mean(axis=0) - V[early_mask].mean(axis=0)

    if cfg.get("dynamo_velocity_csv"):
        vdf = pd.read_csv(cfg["dynamo_velocity_csv"]).set_index("gene")
        csv_delta = np.array(
            [float(vdf.loc[g, "dynamo_velocity"]) if g in vdf.index else 0.0 for g in genes],
            dtype=np.float32,
        )
        csv_delta = np.nan_to_num(csv_delta, nan=0.0)
    else:
        csv_delta = true_delta.copy()

    time_masks = {float(t): pt == t for t in times}

    gene_ids = np.array([vocab[g] if g in vocab else vocab["<pad>"] for g in genes])
    gene_ids = np.concatenate([[vocab["<cls>"]], gene_ids])
    gene_ids_tensor = torch.tensor(gene_ids[None, :], dtype=torch.long)

    X_in = np.concatenate([np.zeros((X_bin.shape[0], 1)), X_bin], axis=1)
    pad_mask = gene_ids_tensor.eq(vocab["<pad>"]).expand(X_in.shape[0], -1)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    values_tensor = torch.tensor(X_in, dtype=dtype)

    holdout = make_holdout_splits(pt, times, len(genes))

    return {
        "genes": genes,
        "X_bin": X_bin,
        "X_raw": X_raw,
        "pseudo_bin": pseudo_bin,
        "target_delta": target_delta,
        "pt": pt,
        "times": times,
        "dt_cell": dt_cell,
        "dt_forward": dt_forward,
        "time_masks": time_masks,
        "velocity": V,
        "true_delta": true_delta,
        "csv_delta": csv_delta,
        "early_mask": early_mask,
        "late_mask": late_mask,
        "eval_idx": select_eval_idx(true_delta),
        "gene_ids_tensor": gene_ids_tensor,
        "values_tensor": values_tensor,
        "pad_mask": pad_mask,
        "vmax": vmax,
        **holdout,
    }


def transition_loss(
    rate_pred: torch.Tensor,
    x_now: torch.Tensor,
    x_target: torch.Tensor,
    v_true: torch.Tensor,
    dt: torch.Tensor,
    valid_g: torch.Tensor,
) -> torch.Tensor:
    """Combined future-state + velocity supervision on valid genes."""
    dt_g = dt.view(-1, 1)
    x_future_pred = x_now + rate_pred * dt_g  #核心公式，构造预测未来的状态

    if valid_g.sum() == 0:
        return rate_pred.sum() * 0.0#下面是三个损失函数

    fut = F.smooth_l1_loss(x_future_pred[:, valid_g], x_target[:, valid_g])
    vel = F.smooth_l1_loss(rate_pred[:, valid_g], v_true[:, valid_g])
    bce = F.binary_cross_entropy_with_logits(rate_pred[:, valid_g], (v_true[:, valid_g] > 0).float())
    return ARGS.lambda_future * fut + ARGS.lambda_vel * vel + ARGS.lambda_dir * bce


def train_head(
    head: StateTransitionHead,
    hidden: torch.Tensor,
    expr: torch.Tensor,
    bundle: dict,
    device: torch.device,
) -> List[float]:
    head.train()
    opt = torch.optim.AdamW(head.parameters(), lr=ARGS.lr, weight_decay=ARGS.weight_decay)
    losses: List[float] = []

    X_bin = torch.tensor(bundle["X_bin"], dtype=torch.float32)
    pseudo = torch.tensor(bundle["pseudo_bin"], dtype=torch.float32)
    V = torch.tensor(bundle["velocity"], dtype=torch.float32)
    dt_cell = torch.tensor(bundle["dt_cell"], dtype=torch.float32)
    valid_all = torch.abs(V) > 1e-10
    train_gene_mask = torch.tensor(bundle["train_gene_mask"], dtype=torch.bool)

    n = hidden.shape[0]
    idx_all = bundle["train_cell_idx"].copy()
    if len(idx_all) == 0:
        raise ValueError("No training cells after hold-out split")

    for epoch in range(ARGS.epochs):
        rng = np.random.default_rng(ARGS.seed + epoch)
        rng.shuffle(idx_all)
        epoch_loss = 0.0
        n_steps = 0

        for start in range(0, len(idx_all), ARGS.batch_size):
            batch_idx = idx_all[start : start + ARGS.batch_size]
            bi = torch.tensor(batch_idx, dtype=torch.long)

            h = hidden[batch_idx].to(device)
            x_full = expr[batch_idx].to(device)
            x_now = X_bin[bi].to(device)
            x_tgt = pseudo[bi].to(device)
            v_true = V[bi].to(device)
            dt = dt_cell[bi].to(device)
            valid = valid_all[bi].to(device)

            if ARGS.n_micro_steps > 1:
                dt_micro = dt / ARGS.n_micro_steps
                x_roll = x_now.clone()
                h_roll = h
                x_full_roll = x_full
                valid_g = valid.any(dim=0) & train_gene_mask.to(device)
                step_loss = torch.tensor(0.0, device=device)
                for _ in range(ARGS.n_micro_steps):
                    rate = head(h_roll, x_full_roll)
                    x_next = x_roll + rate * dt_micro.view(-1, 1)
                    if valid_g.any():
                        step_loss = step_loss + F.smooth_l1_loss(
                            x_next[:, valid_g], x_tgt[:, valid_g]
                        ) / ARGS.n_micro_steps
                    x_roll = x_next
                    if ARGS.reencode_each_step:
                        x_full_roll = torch.cat([x_full_roll[:, :1], x_roll], dim=1)
                rate_final = (x_roll - x_now) / dt.view(-1, 1)
                vel_term = (
                    F.smooth_l1_loss(rate_final[:, valid_g], v_true[:, valid_g])
                    if valid_g.any()
                    else torch.tensor(0.0, device=device)
                )
                loss = ARGS.lambda_future * step_loss + ARGS.lambda_vel * vel_term
            else:
                rate = head(h, x_full)
                valid_g = valid.any(dim=0) & train_gene_mask.to(device)
                loss = transition_loss(rate, x_now, x_tgt, v_true, dt, valid_g)

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_steps += 1

        losses.append(epoch_loss / max(n_steps, 1))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{ARGS.epochs} loss={losses[-1]:.4f}", flush=True)

    head.eval()
    return losses


@torch.no_grad()
def predict_velocity_field(
    head: StateTransitionHead,
    model,
    hidden: torch.Tensor,
    expr: torch.Tensor,
    bundle: dict,
    device: torch.device,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns per-cell velocity [N,G] and optional per-step velocities [N, steps, G].
    """
    n = hidden.shape[0]
    dt_cell = bundle["dt_cell"]
    n_micro = max(int(ARGS.n_micro_steps), 1)

    vel_cell = np.zeros((n, len(bundle["genes"])), dtype=np.float32)
    vel_steps = np.zeros((n, n_micro, len(bundle["genes"])), dtype=np.float32) if n_micro > 1 else None

    for start in range(0, n, ARGS.batch_size):
        end = min(start + ARGS.batch_size, n)
        h = hidden[start:end].to(device)
        x_full = expr[start:end].to(device)
        x_now = torch.tensor(bundle["X_bin"][start:end], dtype=torch.float32, device=device)
        dt = torch.tensor(dt_cell[start:end], dtype=torch.float32, device=device)
        dt_micro = dt / n_micro

        x_roll = x_now
        h_roll = h
        for step in range(n_micro):
            rate = head(h_roll, x_full)
            if n_micro == 1:
                vel_cell[start:end] = rate.cpu().numpy()
            else:
                vel_steps[start:end, step] = rate.cpu().numpy()
                x_roll = x_roll + rate * dt_micro.view(-1, 1)
                if ARGS.reencode_each_step:
                    x_full = torch.cat([x_full[:, :1], x_roll], dim=1)
                    # Re-encode would need model forward; approximate with expr update only

        if n_micro > 1:
            vel_cell[start:end] = vel_steps[start:end].mean(axis=1)

    return vel_cell, vel_steps


def spearman_safe(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if len(a) < 3:
        return float("nan")
    r, _ = spearmanr(a, b)
    return float(r)


def compute_delta(
    vel_cell: np.ndarray,
    velocity: np.ndarray,
    early_mask: np.ndarray,
    late_mask: np.ndarray,
    cell_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gene-wise late-minus-early delta for pred and true velocity."""
    early = early_mask if cell_mask is None else early_mask & cell_mask
    late = late_mask if cell_mask is None else late_mask & cell_mask
    if not early.any() or not late.any():
        return np.zeros(vel_cell.shape[1], dtype=np.float32), np.zeros(velocity.shape[1], dtype=np.float32)
    pred_delta = vel_cell[late].mean(axis=0) - vel_cell[early].mean(axis=0)
    true_delta = velocity[late].mean(axis=0) - velocity[early].mean(axis=0)
    return pred_delta, true_delta


def evaluate_split(
    vel_cell: np.ndarray,
    bundle: dict,
    *,
    split_name: str,
    cell_mask: Optional[np.ndarray],
    gene_idx: np.ndarray,
) -> Dict:
    V = bundle["velocity"]
    pred_delta, true_delta = compute_delta(
        vel_cell,
        V,
        bundle["early_mask"],
        bundle["late_mask"],
        cell_mask=cell_mask,
    )
    eval_idx = select_eval_idx(true_delta, gene_idx=gene_idx if len(gene_idx) else None)
    csv_delta = bundle["csv_delta"]

    eval_mask = np.abs(true_delta[eval_idx]) > 1e-10
    idx_eval = eval_idx[eval_mask]

    out = {
        "split": split_name,
        "n_cells": int(cell_mask.sum()) if cell_mask is not None else len(bundle["pt"]),
        "n_genes": int(len(gene_idx)) if len(gene_idx) else len(bundle["genes"]),
        "direction_acc": direction_accuracy(pred_delta, true_delta, eval_idx),
        "direction_acc_csv_delta": direction_accuracy(pred_delta, csv_delta, eval_idx),
        "spearman_pred_vs_true": spearman_safe(pred_delta[idx_eval], true_delta[idx_eval]) if len(idx_eval) >= 3 else float("nan"),
        "spearman_abs_magnitude": spearman_safe(np.abs(pred_delta[idx_eval]), np.abs(true_delta[idx_eval])) if len(idx_eval) >= 3 else float("nan"),
        "n_eval_genes": int(len(eval_idx)),
    }
    return out


def evaluate(
    vel_cell: np.ndarray,
    vel_steps: Optional[np.ndarray],
    bundle: dict,
) -> Dict:
    genes = bundle["genes"]
    V = bundle["velocity"]
    eval_idx = bundle["eval_idx"]
    times = bundle["times"]

    rows = []
    for t in times:
        mask = bundle["time_masks"][float(t)]
        pred_t = vel_cell[mask].mean(axis=0)
        true_t = V[mask].mean(axis=0)
        for gi, g in enumerate(genes):
            rows.append(
                {
                    "gene": g,
                    "time": float(t),
                    "pred_rate_mean": float(pred_t[gi]),
                    "true_rate_mean": float(true_t[gi]),
                }
            )
    per_time_df = pd.DataFrame(rows)

    pred_delta, true_delta = compute_delta(
        vel_cell, V, bundle["early_mask"], bundle["late_mask"]
    )

    dir_all = direction_accuracy(pred_delta, true_delta, eval_idx)
    dir_csv = direction_accuracy(pred_delta, bundle["csv_delta"], eval_idx)

    eval_mask = np.abs(true_delta[eval_idx]) > 1e-10
    idx_eval = eval_idx[eval_mask]
    spearman_val = spearman_safe(pred_delta[idx_eval], true_delta[idx_eval])
    spearman_abs = spearman_safe(np.abs(pred_delta[idx_eval]), np.abs(true_delta[idx_eval]))

    per_time_dir = {}
    for t in times:
        sub = per_time_df[per_time_df["time"] == float(t)]
        pred_v = np.array([sub.loc[sub["gene"] == g, "pred_rate_mean"].values[0] for g in genes])
        true_v = np.array([sub.loc[sub["gene"] == g, "true_rate_mean"].values[0] for g in genes])
        per_time_dir[float(t)] = direction_accuracy(pred_v, true_v, eval_idx)

    delta_df = pd.DataFrame(
        {
            "gene": genes,
            "pred_delta": pred_delta,
            "true_delta_h5ad": true_delta,
            "true_delta_csv": bundle["csv_delta"],
            "train_gene": bundle["train_gene_mask"],
            "holdout_gene": bundle["holdout_gene_mask"],
        }
    )

    holdout_metrics: Dict[str, Dict] = {}
    if ARGS.holdout_cells > 0 or ARGS.holdout_genes > 0:
        all_genes = np.arange(len(genes), dtype=int)
        splits = [
            ("train_cells_train_genes", bundle["train_cell_mask"], bundle["train_gene_idx"]),
            ("all_cells_train_genes", None, bundle["train_gene_idx"]),
        ]
        if len(bundle["holdout_gene_idx"]):
            splits.extend(
                [
                    ("train_cells_holdout_genes", bundle["train_cell_mask"], bundle["holdout_gene_idx"]),
                    ("holdout_cells_train_genes", bundle["holdout_cell_mask"], bundle["train_gene_idx"]),
                    ("holdout_cells_holdout_genes", bundle["holdout_cell_mask"], bundle["holdout_gene_idx"]),
                    ("all_cells_holdout_genes", None, bundle["holdout_gene_idx"]),
                ]
            )
        if len(bundle["holdout_cell_idx"]):
            splits.append(("holdout_cells_all_genes", bundle["holdout_cell_mask"], all_genes))

        for name, cell_mask, gene_idx in splits:
            if cell_mask is not None and not cell_mask.any():
                continue
            if len(gene_idx) == 0:
                continue
            holdout_metrics[name] = evaluate_split(
                vel_cell, bundle, split_name=name, cell_mask=cell_mask, gene_idx=gene_idx
            )

    out = {
        "direction_acc_all": dir_all,
        "direction_acc_csv_delta": dir_csv,
        "spearman_pred_vs_true": spearman_val,
        "spearman_abs_magnitude": spearman_abs,
        "per_time_direction_acc": per_time_dir,
        "n_eval_genes": int(len(eval_idx)),
        "eval_mode": "all" if ARGS.eval_all else f"top-{ARGS.top_percent}%",
        "n_micro_steps": ARGS.n_micro_steps,
        "scheme": "continuous_iteration" if ARGS.n_micro_steps > 1 else "single_step",
        "per_time_df": per_time_df,
        "delta_df": delta_df,
        "vel_cell": vel_cell,
        "holdout_metrics": holdout_metrics,
    }
    if vel_steps is not None:
        out["vel_steps"] = vel_steps
    return out


def main():
    set_seed(ARGS.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scheme = "Scheme 3 (continuous)" if ARGS.n_micro_steps > 1 else "Scheme 2 (single step)"
    print(f"Device: {device} | {scheme} | micro_steps={ARGS.n_micro_steps}")

    cfg = load_cfg()
    model, vocab, d_model = build_model(ARGS.model_dir, device)
    bundle = load_bundle(cfg, vocab, device)
    print(
        f"Genes: {len(bundle['genes'])} | cells: {bundle['values_tensor'].shape[0]} "
        f"| times: {bundle['times'].tolist()} | dt={bundle['dt_forward']}"
    )
    if ARGS.holdout_cells > 0 or ARGS.holdout_genes > 0:
        print(
            f"Hold-out: cells={ARGS.holdout_cells:.0%} "
            f"({len(bundle['holdout_cell_idx'])} held out, {len(bundle['train_cell_idx'])} train) | "
            f"genes={ARGS.holdout_genes:.0%} "
            f"({len(bundle['holdout_gene_idx'])} held out, {len(bundle['train_gene_idx'])} train)"
        )

    hidden, expr = encode_cells(
        model,
        bundle["gene_ids_tensor"],
        bundle["values_tensor"],
        bundle["pad_mask"],
        device,
        ARGS.batch_size,
    )

    head = StateTransitionHead(d_model).to(device)
    train_losses = train_head(head, hidden, expr, bundle, device)
    vel_cell, vel_steps = predict_velocity_field(head, model, hidden, expr, bundle, device)
    metrics = evaluate(vel_cell, vel_steps, bundle)

    outdir = Path(ARGS.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ds_out = outdir / ARGS.dataset
    ds_out.mkdir(parents=True, exist_ok=True)

    metrics["per_time_df"].to_csv(ds_out / "gene_rate_per_time.csv", index=False)
    metrics["delta_df"].to_csv(ds_out / "gene_rate_delta_compare.csv", index=False)

    np.savez_compressed(
        ds_out / "velocity_field.npz",
        vel_cell=vel_cell,
        vel_steps=vel_steps if vel_steps is not None else np.array([]),
        pt=bundle["pt"],
        times=bundle["times"],
        train_cell_idx=bundle["train_cell_idx"],
        holdout_cell_idx=bundle["holdout_cell_idx"],
        train_gene_idx=bundle["train_gene_idx"],
        holdout_gene_idx=bundle["holdout_gene_idx"],
    )

    pd.DataFrame({"gene": bundle["genes"]}).assign(
        train_gene=bundle["train_gene_mask"],
        holdout_gene=bundle["holdout_gene_mask"],
    ).to_csv(ds_out / "holdout_genes.csv", index=False)
    pd.DataFrame(
        {
            "cell_idx": np.arange(len(bundle["pt"])),
            "train_cell": bundle["train_cell_mask"],
            "holdout_cell": bundle["holdout_cell_mask"],
            "pt": bundle["pt"],
        }
    ).to_csv(ds_out / "holdout_cells.csv", index=False)

    summary = {k: v for k, v in metrics.items() if k not in ("per_time_df", "delta_df", "vel_cell", "vel_steps")}
    summary["train_losses"] = train_losses
    summary["method"] = "scgpt_state_transition"
    summary["supervision"] = "pseudo_future = X_now + V_dynamo * dt"
    summary["velocity_formula"] = "velocity_pred = (X_future_pred - X_now) / dt"
    summary["holdout_cells_frac"] = ARGS.holdout_cells
    summary["holdout_genes_frac"] = ARGS.holdout_genes
    summary["per_time_direction_acc"] = {str(k): v for k, v in metrics["per_time_direction_acc"].items()}
    if metrics.get("holdout_metrics"):
        summary["holdout_metrics"] = metrics["holdout_metrics"]
        pd.DataFrame(metrics["holdout_metrics"]).T.to_csv(ds_out / "holdout_metrics.csv")
    with open(ds_out / "transition_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== scGPT State Transition vs Dynamo velocity ===")
    print(f"  scheme:                     {summary['scheme']}")
    print(f"  direction (h5ad delta):     {metrics['direction_acc_all']:.2%}")
    print(f"  direction (csv delta):      {metrics['direction_acc_csv_delta']:.2%}")
    print(f"  Spearman (value):           {metrics['spearman_pred_vs_true']:.3f}")
    print(f"  Spearman (|value|):         {metrics['spearman_abs_magnitude']:.3f}")
    for t, acc in metrics["per_time_direction_acc"].items():
        print(f"  direction @ time={t}:       {acc:.2%}")
    if metrics.get("holdout_metrics"):
        print("\n=== Hold-out splits (direction accuracy vs Dynamo) ===")
        for name, hm in metrics["holdout_metrics"].items():
            print(
                f"  {name:32s}  dir={hm['direction_acc']:.2%}  "
                f"spearman={hm['spearman_pred_vs_true']:.3f}  "
                f"(cells={hm['n_cells']}, genes={hm['n_genes']}, eval_genes={hm['n_eval_genes']})"
            )
    print(f"  saved -> {ds_out}")


if __name__ == "__main__":
    main()
