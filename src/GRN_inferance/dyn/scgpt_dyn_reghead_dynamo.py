#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RegVelo head trained/evaluated against Dynamo RNA velocity (per labeling time).

Frozen scGPT encoder + trainable rate head:
  rate_g = MLP(H_g, [GRN agg], x_g)

Supervision (--supervision):
  dynamo_cell      per-cell velocity from h5ad (default)
  dynamo_time_mean mean velocity per (gene, time) — fewer cells per step

Evaluation (--eval-all): all genes in the expression matrix (e.g. 1524 dynamic genes).

Example (hematopoiesis 1524 dynamic genes):
  python scgpt_dyn_reghead_dynamo.py \\
    --model-dir /mnt/10T/yzn/benchmark_GRN/model/weights/scgpt/scGPT_human \\
    --h5ad /mnt/10T/yzn/dynamo-release/results/hematopoiesis_raw/hematopoiesis_processed.h5ad \\
    --datasets-json data/dynamo_export/datasets_dynamo_vel1524_only.json \\
    --dataset hematopoiesis_dyn_vel1524 \\
    --outdir outputs/dynamo_fm_velocity/reghead_hematopoiesis_vel1524 \\
    --init-mode none --epochs 80 --eval-all
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
import torch.nn.functional as F

# Reuse RegVelo / scGPT helpers from sibling module.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from scgpt_dyn_reghead import (  # noqa: E402
    RegVeloHead,
    bin_expr_to_0_50,
    build_grn_tensors,
    build_model,
    build_regvelo_head,
    direction_accuracy,
    encode_cells,
    load_grn_edges,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser(description="RegVelo head vs Dynamo velocity (per time).")
    p.add_argument("--model-dir", required=True, type=str)
    p.add_argument("--outdir", required=True, type=str)
    p.add_argument("--datasets-json", required=True, type=str)
    p.add_argument("--dataset", required=True, type=str)
    p.add_argument(
        "--h5ad",
        required=True,
        type=str,
        help="Processed Dynamo h5ad with velocity layer + obs time.",
    )
    p.add_argument("--velocity-layer", default="velocity_alpha_minus_gamma_s", type=str)
    p.add_argument(
        "--supervision",
        choices=["dynamo_cell", "dynamo_time_mean"],
        default="dynamo_cell",
        help="dynamo_cell: per-cell velocity; dynamo_time_mean: one target vector per time.",
    )
    p.add_argument("--init-mode", default="none", choices=["none", "grn_init", "grn_frozen", "random_init"])
    p.add_argument("--grn-extraction", default="embhidden500", choices=["emb500", "embhidden500", "att500"])
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lambda-dir", type=float, default=1.0, help="Weight for direction BCE term.")
    p.add_argument("--lambda-mse", type=float, default=1.0, help="Weight for Huber/MSE on rate values.")
    p.add_argument("--eval-all", action="store_true", help="Evaluate all genes (no Top-N%% subset).")
    p.add_argument("--top-percent", type=int, default=30)
    p.add_argument("--pt-quantile", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log1p", action="store_true")
    return p.parse_args()


ARGS = parse_args()


def load_cfg() -> dict:
    with open(ARGS.datasets_json, encoding="utf-8") as f:
        cfg = json.load(f)
    if ARGS.dataset not in cfg:
        raise KeyError(f"{ARGS.dataset} not in {ARGS.datasets_json}")
    return cfg[ARGS.dataset]


def select_eval_idx(true_delta: np.ndarray) -> np.ndarray:
    n = len(true_delta)
    if ARGS.eval_all or ARGS.top_percent >= 100:
        return np.arange(n, dtype=int)
    k = max(int(n * ARGS.top_percent / 100), 1)
    return np.argsort(np.abs(true_delta))[::-1][:k].copy()


def load_bundle(cfg: dict, vocab, device: torch.device):
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
    V = np.asarray(V, dtype=np.float32)[:, h5_idx]  # cells x genes
    V = np.nan_to_num(V, nan=0.0)

    X = expr.T.to_numpy(dtype=np.float32)
    X_bin = bin_expr_to_0_50(X, do_log1p=bool(ARGS.log1p))

    lo, hi = np.quantile(pt, [ARGS.pt_quantile, 1 - ARGS.pt_quantile])
    early_mask = pt <= lo
    late_mask = pt >= hi

    # Endpoint truth for delta comparison (same as export CSV logic).
    v_early = V[early_mask].mean(axis=0)
    v_late = V[late_mask].mean(axis=0)
    true_delta = v_late - v_early

    if cfg.get("dynamo_velocity_csv"):
        vdf = pd.read_csv(cfg["dynamo_velocity_csv"]).set_index("gene")
        csv_delta = np.array(
            [float(vdf.loc[g, "dynamo_velocity"]) if g in vdf.index else 0.0 for g in genes],
            dtype=np.float32,
        )
        csv_delta = np.nan_to_num(csv_delta, nan=0.0)
    else:
        csv_delta = true_delta.copy()

    times = np.sort(np.unique(pt))
    time_masks = {float(t): pt == t for t in times}
    time_true_mean = {t: V[mask].mean(axis=0) for t, mask in time_masks.items()}

    gene_ids = np.array([vocab[g] if g in vocab else vocab["<pad>"] for g in genes])
    gene_ids = np.concatenate([[vocab["<cls>"]], gene_ids])
    gene_ids_tensor = torch.tensor(gene_ids[None, :], dtype=torch.long)

    X_in = np.concatenate([np.zeros((X_bin.shape[0], 1)), X_bin], axis=1)
    pad_mask = gene_ids_tensor.eq(vocab["<pad>"]).expand(X_in.shape[0], -1)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    values_tensor = torch.tensor(X_in, dtype=dtype)

    eval_idx = select_eval_idx(true_delta)

    return {
        "name": ARGS.dataset,
        "genes": genes,
        "X_bin": X_bin,
        "pt": pt,
        "times": times,
        "time_masks": time_masks,
        "velocity": V,
        "time_true_mean": time_true_mean,
        "true_delta": true_delta,
        "csv_delta": csv_delta,
        "early_mask": early_mask,
        "late_mask": late_mask,
        "eval_idx": eval_idx,
        "gene_ids_tensor": gene_ids_tensor,
        "values_tensor": values_tensor,
        "pad_mask": pad_mask,
    }


def rate_loss(pred: torch.Tensor, true: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Huber on values + BCE direction on valid genes."""
    if valid.sum() == 0:
        return pred.sum() * 0.0
    p = pred[valid]
    t = true[valid]
    mse = F.smooth_l1_loss(p, t)
    bce = F.binary_cross_entropy_with_logits(p, (t > 0).float())
    return ARGS.lambda_mse * mse + ARGS.lambda_dir * bce


def train_head(
    head: RegVeloHead,
    hidden: torch.Tensor,
    expr: torch.Tensor,
    bundle: dict,
    device: torch.device,
) -> List[float]:
    head.train()
    opt = torch.optim.AdamW(head.parameters(), lr=ARGS.lr, weight_decay=ARGS.weight_decay)
    losses: List[float] = []
    V = torch.tensor(bundle["velocity"], dtype=torch.float32, device=device)
    valid_all = torch.tensor(np.abs(bundle["velocity"]) > 1e-10, device=device)

    for epoch in range(ARGS.epochs):
        epoch_loss = 0.0
        n_steps = 0
        for t in bundle["times"]:
            mask = bundle["time_masks"][float(t)]
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            true_mean = torch.tensor(bundle["time_true_mean"][float(t)], dtype=torch.float32, device=device)

            rng = np.random.default_rng(ARGS.seed + epoch)
            rng.shuffle(idx)
            for start in range(0, len(idx), ARGS.batch_size):
                batch_idx = idx[start : start + ARGS.batch_size]
                h = hidden[batch_idx].to(device)
                x = expr[batch_idx].to(device)
                pred = head(h, x)

                if ARGS.supervision == "dynamo_time_mean":
                    valid = torch.abs(true_mean) > 1e-10
                    loss = rate_loss(pred.mean(dim=0), true_mean, valid)
                else:
                    true_b = V[batch_idx]
                    valid_b = valid_all[batch_idx]
                    # gene valid if any cell in batch has non-zero true velocity
                    valid_g = valid_b.any(dim=0)
                    loss = rate_loss(pred.mean(dim=0), true_b.mean(dim=0), valid_g)

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
def predict_rates(head: RegVeloHead, hidden: torch.Tensor, expr: torch.Tensor, device: torch.device) -> np.ndarray:
    """Per-cell predicted rate [n_cells, n_genes]."""
    n = hidden.shape[0]
    out = []
    for start in range(0, n, ARGS.batch_size):
        end = min(start + ARGS.batch_size, n)
        pred = head(hidden[start:end].to(device), expr[start:end].to(device))
        out.append(pred.cpu().numpy())
    return np.concatenate(out, axis=0)


def spearman_safe(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if len(a) < 3:
        return float("nan")
    r, _ = spearmanr(a, b)
    return float(r)


def evaluate(head: RegVeloHead, hidden: torch.Tensor, expr: torch.Tensor, bundle: dict, device: torch.device) -> Dict:
    pred_cell = predict_rates(head, hidden, expr, device)
    V = bundle["velocity"]
    genes = bundle["genes"]
    eval_idx = bundle["eval_idx"]

    # Per-time gene-wise mean rates.
    rows = []
    for t in bundle["times"]:
        mask = bundle["time_masks"][float(t)]
        pred_t = pred_cell[mask].mean(axis=0)
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

    # Endpoint delta from predicted rates (late - early).
    pred_early = pred_cell[bundle["early_mask"]].mean(axis=0)
    pred_late = pred_cell[bundle["late_mask"]].mean(axis=0)
    pred_delta = pred_late - pred_early
    true_delta = bundle["true_delta"]

    dir_all = direction_accuracy(pred_delta, true_delta, eval_idx)
    dir_csv = direction_accuracy(pred_delta, bundle["csv_delta"], eval_idx)

    eval_mask = np.abs(true_delta[eval_idx]) > 1e-10
    idx_eval = eval_idx[eval_mask]
    spearman_val = spearman_safe(pred_delta[idx_eval], true_delta[idx_eval])
    spearman_abs = spearman_safe(np.abs(pred_delta[idx_eval]), np.abs(true_delta[idx_eval]))

    per_time_dir = {}
    for t in bundle["times"]:
        sub = per_time_df[per_time_df["time"] == float(t)]
        gi = {g: i for i, g in enumerate(genes)}
        pred_v = np.array([sub.loc[sub["gene"] == g, "pred_rate_mean"].values[0] for g in genes])
        true_v = np.array([sub.loc[sub["gene"] == g, "true_rate_mean"].values[0] for g in genes])
        per_time_dir[float(t)] = direction_accuracy(pred_v, true_v, eval_idx)

    delta_df = pd.DataFrame(
        {
            "gene": genes,
            "pred_delta": pred_delta,
            "true_delta_h5ad": true_delta,
            "true_delta_csv": bundle["csv_delta"],
        }
    )

    return {
        "direction_acc_all": dir_all,
        "direction_acc_csv_delta": dir_csv,
        "spearman_pred_vs_true": spearman_val,
        "spearman_abs_magnitude": spearman_abs,
        "per_time_direction_acc": per_time_dir,
        "n_eval_genes": int(len(eval_idx)),
        "eval_mode": "all" if ARGS.eval_all else f"top-{ARGS.top_percent}%",
        "per_time_df": per_time_df,
        "delta_df": delta_df,
        "pred_cell_rate": pred_cell,
    }


def main():
    set_seed(ARGS.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | supervision={ARGS.supervision} | init={ARGS.init_mode}")

    cfg = load_cfg()
    model, vocab, d_model = build_model(ARGS.model_dir, device)
    bundle = load_bundle(cfg, vocab, device)
    print(f"Genes: {len(bundle['genes'])} | cells: {bundle['values_tensor'].shape[0]} | times: {bundle['times'].tolist()}")

    hidden, expr = encode_cells(
        model,
        bundle["gene_ids_tensor"],
        bundle["values_tensor"],
        bundle["pad_mask"],
        device,
        ARGS.batch_size,
    )

    grn_tensors = None
    if ARGS.init_mode != "none":
        grn_df = load_grn_edges(
            ARGS.dataset.replace("_dyn_vel1524", "").replace("_vel1524", ""),
            set(bundle["genes"]),
            expr_root=Path(cfg["expr_csv"]).parent.parent,
            extraction=ARGS.grn_extraction,
        )
        grn_tensors = build_grn_tensors(bundle["genes"], grn_df)

    head = build_regvelo_head(d_model, grn_tensors, ARGS.init_mode, device)
    train_losses = train_head(head, hidden, expr, bundle, device)
    metrics = evaluate(head, hidden, expr, bundle, device)

    outdir = Path(ARGS.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ds_out = outdir / ARGS.dataset
    ds_out.mkdir(parents=True, exist_ok=True)

    metrics["per_time_df"].to_csv(ds_out / "gene_rate_per_time.csv", index=False)
    metrics["delta_df"].to_csv(ds_out / "gene_rate_delta_compare.csv", index=False)

    summary = {k: v for k, v in metrics.items() if k not in ("per_time_df", "delta_df", "pred_cell_rate")}
    summary["train_losses"] = train_losses
    summary["supervision"] = ARGS.supervision
    summary["init_mode"] = ARGS.init_mode
    summary["per_time_direction_acc"] = {
        str(k): v for k, v in metrics["per_time_direction_acc"].items()
    }
    with open(ds_out / "reghead_dynamo_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== RegVelo vs Dynamo velocity ===")
    print(f"  direction (all eval genes): {metrics['direction_acc_all']:.2%}")
    print(f"  direction (csv delta):      {metrics['direction_acc_csv_delta']:.2%}")
    print(f"  Spearman (value):           {metrics['spearman_pred_vs_true']:.3f}")
    print(f"  Spearman (|value|):        {metrics['spearman_abs_magnitude']:.3f}")
    for t, acc in metrics["per_time_direction_acc"].items():
        print(f"  direction @ time={t}:      {acc:.2%}")
    print(f"  saved -> {ds_out}")


if __name__ == "__main__":
    main()
