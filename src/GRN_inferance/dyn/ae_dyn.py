#!/usr/bin/env python3
"""AE / DAE control for the scGPT-style pseudotime direction-accuracy protocol.

Replaces the pretrained foundation model in ``scgpt_dyn.py`` with a simple
MLP autoencoder or denoising autoencoder, keeping the same:

  - early / late pseudotime split
  - true_delta = late_mean - early_mean (or Dynamo CSV)
  - iterative reconstruction with EMA
  - direction accuracy on Top-N% |true_delta| genes
  - output tables: gene_delta_compare.csv, accuracy_curves.json, ...

Purpose: test whether directed "evolution" under iterative reconstruction is
specific to single-cell foundation models, or also appears for generic
reconstruction / denoising models.

Examples
--------
# Per-dataset AE (train on non-early cells; default)
python src/GRN_inferance/dyn/ae_dyn.py \\
  --model-type ae \\
  --expr-root data/input_process \\
  --pt-root data/PseudoTime \\
  --outdir outputs/ae_dyn/ae_self

# Denoising AE, leave-one-out train on other datasets
python src/GRN_inferance/dyn/ae_dyn.py \\
  --model-type dae \\
  --train-mode loo \\
  --expr-root data/input_process \\
  --pt-root data/PseudoTime \\
  --outdir outputs/ae_dyn/dae_loo

# Untrained random AE (architecture / noise floor)
python src/GRN_inferance/dyn/ae_dyn.py \\
  --model-type ae \\
  --train-mode untrained \\
  --expr-root data/input_process \\
  --pt-root data/PseudoTime \\
  --outdir outputs/ae_dyn/ae_untrained
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
os.environ["KMP_WARNINGS"] = "off"


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="AE/DAE pseudotime direction-accuracy control (mirrors scgpt_dyn)."
    )
    p.add_argument("--outdir", required=True, type=str)
    p.add_argument("--datasets-json", default="", type=str)
    p.add_argument("--expr-root", default="", type=str)
    p.add_argument("--pt-root", default="", type=str)
    p.add_argument("--dataset", default="", help="Run only this dataset name.")
    p.add_argument(
        "--model-type",
        choices=["ae", "dae"],
        default="ae",
        help="ae=plain autoencoder; dae=denoising autoencoder (train with noise/mask).",
    )
    p.add_argument(
        "--train-mode",
        choices=["self", "self_holdout_early", "self_mid", "loo", "untrained"],
        default="self_holdout_early",
        help=(
            "self: train on all cells of the eval dataset; "
            "self_holdout_early: train on non-early cells only (default); "
            "self_mid: train only on middle-pseudotime cells (exclude early+late; stricter control); "
            "loo: leave-one-out train on other datasets; "
            "untrained: random init, no training."
        ),
    )
    p.add_argument("--pt-quantile", type=float, default=0.2)
    p.add_argument("--top-percent", type=int, default=30)
    p.add_argument("--eval-all", action="store_true")
    p.add_argument(
        "--truth-source",
        choices=["expression", "dynamo"],
        default="expression",
    )
    p.add_argument("--gen-iters", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--ema-alpha", type=float, default=0.1)
    p.add_argument("--log1p", action="store_true")
    p.add_argument("--hidden-dims", type=str, default="512,256,128",
                   help="Comma-separated encoder hidden sizes (decoder is mirrored).")
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--noise-std", type=float, default=0.1,
                   help="Gaussian noise std (on [0,1]-scaled inputs) for DAE train/iter.")
    p.add_argument("--mask-rate", type=float, default=0.15,
                   help="Fraction of genes zeroed during DAE training/iteration.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="", help="cuda / cpu (default: auto).")
    p.add_argument("--save-ckpts", action="store_true",
                   help="Save per-dataset AE checkpoints under outdir.")
    p.add_argument(
        "--run-trajectory-probes",
        action="store_true",
        help="Repeat early/late/mid initialization probes used by the unified FM experiment.",
    )
    p.add_argument(
        "--probe-init-sources",
        default="early,late,mid",
        help="Comma-separated probe initializations: early, late, mid.",
    )
    p.add_argument("--probe-max-cells", type=int, default=0,
                   help="Maximum cells per initialization (0 keeps all).")
    p.add_argument("--probe-gene-scope", choices=["top_genes", "all_genes"],
                   default="top_genes")
    p.add_argument("--probe-conv-rel-eps", type=float, default=0.02)
    p.add_argument("--probe-late-stability-rel-eps", type=float, default=0.15)
    return p.parse_args()


ARGS = parse_args()
OUTDIR = ARGS.outdir

DATASET_SPECS = {
    "hESC": "human",
    "hHep": "human",
    "mDC": "mouse",
    "mHSC-E": "mouse",
    "mHSC-GM": "mouse",
    "mHSC-L": "mouse",
}


def build_datasets_from_roots(expr_root: str, pt_root: str):
    if not expr_root or not pt_root:
        raise ValueError("When --datasets-json is not set, both --expr-root and --pt-root are required.")
    datasets = {}
    for ds, species in DATASET_SPECS.items():
        expr_root_path = Path(expr_root)
        nested_expr = expr_root_path / "CHIP" / f"{ds}_chip_matched-ExpressionData.csv"
        flat_expr = expr_root_path / f"{ds}_chip_matched-ExpressionData.csv"
        expr_path = nested_expr if nested_expr.is_file() else flat_expr
        datasets[ds] = {
            "expr_csv": str(expr_path),
            "pt_csv": str(Path(pt_root) / ds / "PseudoTime.csv"),
            "species": species,
        }
    return datasets


def load_datasets_config(args):
    if args.datasets_json:
        with open(args.datasets_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict) or not cfg:
            raise ValueError("datasets-json must be a non-empty object.")
        if args.dataset:
            if args.dataset not in cfg:
                raise KeyError(f"Dataset '{args.dataset}' not in {list(cfg.keys())}")
            cfg = {args.dataset: cfg[args.dataset]}
        return cfg
    cfg = build_datasets_from_roots(args.expr_root, args.pt_root)
    if args.dataset:
        if args.dataset not in cfg:
            raise KeyError(f"Dataset '{args.dataset}' not in {list(cfg.keys())}")
        cfg = {args.dataset: cfg[args.dataset]}
    return cfg


DATASETS = load_datasets_config(ARGS)

PT_QUANTILE = float(ARGS.pt_quantile)
TOP_PERCENT = int(ARGS.top_percent)
EVAL_ALL = bool(ARGS.eval_all)
TRUTH_SOURCE = ARGS.truth_source
GEN_ITERS = int(ARGS.gen_iters)
BATCH_SIZE = int(ARGS.batch_size)
EMA_ALPHA = float(ARGS.ema_alpha)
NO_LOG1P = not bool(ARGS.log1p)


# =============================================================================
# Utils (aligned with scgpt_dyn.py)
# =============================================================================
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bin_expr_to_0_50(x, do_log1p=True):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0)
    x = np.clip(x, 0, None)
    if do_log1p:
        x = np.log1p(x)
    vmax = max(float(np.percentile(x, 99.5)), 1e-6)
    return np.clip(x / vmax * 50.0, 0, 50).astype(np.float32)


def select_eval_indices(true_delta: np.ndarray):
    total_genes = len(true_delta)
    if EVAL_ALL or TOP_PERCENT >= 100:
        idx = np.arange(total_genes, dtype=int)
        label = f"all ({total_genes})"
    else:
        top_n = max(int(total_genes * TOP_PERCENT / 100), 1)
        idx = np.argsort(np.abs(true_delta))[::-1][:top_n].copy()
        label = f"top-{TOP_PERCENT}% ({len(idx)})"
    return idx, label


def direction_accuracy_idx(pred: np.ndarray, true: np.ndarray, idx: np.ndarray) -> float:
    return float((np.sign(pred[idx]) == np.sign(true[idx])).mean())


def load_true_delta(genes, cfg, early_mean, late_mean):
    if TRUTH_SOURCE == "dynamo":
        vpath = cfg.get("dynamo_velocity_csv", "")
        if not vpath or not os.path.isfile(vpath):
            raise FileNotFoundError(
                f"--truth-source dynamo requires dynamo_velocity_csv; got {vpath!r}"
            )
        vdf = pd.read_csv(vpath)
        if "gene" not in vdf.columns:
            vdf = vdf.rename(columns={vdf.columns[0]: "gene"})
        vcol = "dynamo_velocity" if "dynamo_velocity" in vdf.columns else vdf.columns[1]
        vmap = vdf.set_index("gene")[vcol]
        true_delta = np.array(
            [float(vmap[g]) if g in vmap.index else 0.0 for g in genes],
            dtype=np.float32,
        )
        return np.nan_to_num(true_delta, nan=0.0), "dynamo_velocity"
    return (late_mean - early_mean).astype(np.float32), "expression_delta"


def parse_hidden_dims(s: str) -> List[int]:
    dims = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not dims:
        raise ValueError("--hidden-dims must contain at least one integer")
    return dims


# =============================================================================
# Model
# =============================================================================
class MLPAutoEncoder(nn.Module):
    """Symmetric MLP autoencoder over gene expression vectors."""

    def __init__(self, n_genes: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        self.n_genes = n_genes
        enc: List[nn.Module] = []
        prev = n_genes
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        enc.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc)

        dec: List[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        dec.append(nn.Linear(prev, n_genes))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def corrupt_batch(
    x: torch.Tensor,
    noise_std: float,
    mask_rate: float,
) -> torch.Tensor:
    """Corrupt inputs for DAE (gaussian noise + random gene masking)."""
    out = x
    if noise_std > 0:
        out = out + torch.randn_like(out) * noise_std
    if mask_rate > 0:
        mask = torch.rand_like(out) < mask_rate
        out = out.masked_fill(mask, 0.0)
    return out.clamp(0.0, 1.0)


# =============================================================================
# Data loading
# =============================================================================
def load_dataset_matrix(name: str, cfg: dict) -> dict:
    expr_path = Path(cfg["expr_csv"])
    pt_path = Path(cfg["pt_csv"])
    if not expr_path.is_file():
        raise FileNotFoundError(f"[{name}] missing expression: {expr_path}")
    if not pt_path.is_file():
        raise FileNotFoundError(f"[{name}] missing pseudotime: {pt_path}")

    expr = pd.read_csv(expr_path, index_col=0)
    pt_df = pd.read_csv(pt_path)
    pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"}).set_index("cell")

    common = expr.columns.intersection(pt_df.index)
    if len(common) == 0:
        raise ValueError(f"[{name}] no overlapping cells between expr and pt")
    expr = expr[common]
    pt = pt_df.loc[common, "pt"].to_numpy(dtype=np.float64)
    genes = expr.index.astype(str).tolist()

    X = expr.T.to_numpy(dtype=np.float32)
    X_bin = bin_expr_to_0_50(X, do_log1p=(not NO_LOG1P))
    # Scale to [0, 1] for AE training stability; keep X_bin for scgpt-aligned metrics.
    X_unit = (X_bin / 50.0).astype(np.float32)

    lo, hi = np.quantile(pt, [PT_QUANTILE, 1.0 - PT_QUANTILE])
    mid_lo, mid_hi = np.quantile(pt, [0.5 - PT_QUANTILE / 2.0, 0.5 + PT_QUANTILE / 2.0])
    early = pt <= lo
    late = pt >= hi
    mid = (pt >= mid_lo) & (pt <= mid_hi)
    return {
        "name": name,
        "cfg": cfg,
        "genes": genes,
        "cells": common.tolist(),
        "pt": pt,
        "X_bin": X_bin,
        "X_unit": X_unit,
        "early": early,
        "late": late,
        "mid": mid,
        "lo": float(lo),
        "hi": float(hi),
        "mid_lo": float(mid_lo),
        "mid_hi": float(mid_hi),
    }


def align_matrix_to_genes(X_unit: np.ndarray, src_genes: List[str], tgt_genes: List[str]) -> np.ndarray:
    """Project a matrix onto tgt gene order (missing genes -> 0)."""
    idx = {g: i for i, g in enumerate(src_genes)}
    out = np.zeros((X_unit.shape[0], len(tgt_genes)), dtype=np.float32)
    for j, g in enumerate(tgt_genes):
        i = idx.get(g)
        if i is not None:
            out[:, j] = X_unit[:, i]
    return out


def collect_train_cells(
    eval_name: str,
    eval_pack: dict,
    all_packs: Dict[str, dict],
    train_mode: str,
) -> np.ndarray:
    """Return training matrix in the *eval dataset gene space* (unit scale)."""
    tgt_genes = eval_pack["genes"]
    if train_mode == "untrained":
        return np.zeros((0, len(tgt_genes)), dtype=np.float32)

    if train_mode in ("self", "self_holdout_early", "self_mid"):
        X = eval_pack["X_unit"]
        if train_mode == "self_holdout_early":
            mask = ~eval_pack["early"]
            X = X[mask]
            if X.shape[0] == 0:
                raise ValueError(f"[{eval_name}] no non-early cells for training")
        elif train_mode == "self_mid":
            # Exclude both early and late tails so reconstruction is not
            # trivially pulled toward the late distribution.
            mask = ~(eval_pack["early"] | eval_pack["late"])
            X = X[mask]
            if X.shape[0] == 0:
                raise ValueError(f"[{eval_name}] no mid-pseudotime cells for training")
        return X

    if train_mode == "loo":
        chunks = []
        for name, pack in all_packs.items():
            if name == eval_name:
                continue
            chunks.append(align_matrix_to_genes(pack["X_unit"], pack["genes"], tgt_genes))
        if not chunks:
            raise ValueError("loo train-mode requires at least one other dataset")
        return np.concatenate(chunks, axis=0)

    raise ValueError(f"Unknown train-mode: {train_mode}")


# =============================================================================
# Train / iterate
# =============================================================================
def train_autoencoder(
    model: MLPAutoEncoder,
    X_train: np.ndarray,
    device: torch.device,
    model_type: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    noise_std: float,
    mask_rate: float,
) -> dict:
    if X_train.shape[0] == 0:
        return {"epochs": 0, "final_loss": None, "n_train": 0}

    model.train()
    ds = TensorDataset(torch.from_numpy(X_train))
    loader = DataLoader(
        ds,
        batch_size=min(batch_size, len(ds)),
        shuffle=True,
        num_workers=ARGS.num_workers,
        drop_last=False,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    last_loss = float("nan")

    for ep in range(1, epochs + 1):
        total, n = 0.0, 0
        for (xb,) in loader:
            xb = xb.to(device)
            if model_type == "dae":
                inp = corrupt_batch(xb, noise_std=noise_std, mask_rate=mask_rate)
            else:
                inp = xb
            pred = model(inp)
            loss = loss_fn(pred, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.shape[0]
            n += xb.shape[0]
        last_loss = total / max(n, 1)
        if ep == 1 or ep == epochs or ep % max(epochs // 5, 1) == 0:
            print(f"    [train] epoch {ep}/{epochs}  mse={last_loss:.6f}")

    model.eval()
    return {"epochs": epochs, "final_loss": last_loss, "n_train": int(X_train.shape[0])}


@torch.no_grad()
def run_iterative(
    model: MLPAutoEncoder,
    X_unit: np.ndarray,
    baseline_mean_bin: np.ndarray,
    true_delta: Optional[np.ndarray],
    eval_idx: Optional[np.ndarray],
    device: torch.device,
    model_type: str,
    track_curve: bool = False,
) -> Tuple[List[float], np.ndarray, np.ndarray]:
    """Iterative reconstruction on unit-scale inputs; deltas reported in bin space."""
    model.eval()
    vals = torch.from_numpy(X_unit.copy()).to(device)
    acc_curve: List[float] = []

    for _ in range(GEN_ITERS):
        for start in range(0, vals.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, vals.shape[0])
            chunk = vals[start:end]
            if model_type == "dae":
                inp = corrupt_batch(chunk, noise_std=ARGS.noise_std, mask_rate=ARGS.mask_rate)
            else:
                inp = chunk
            recon = model(inp).clamp(0.0, 1.0)
            updated = EMA_ALPHA * chunk + (1.0 - EMA_ALPHA) * recon
            vals[start:end] = updated

        post_mean_bin = vals.mean(dim=0).float().cpu().numpy() * 50.0
        pred_delta = post_mean_bin - baseline_mean_bin
        if track_curve and true_delta is not None and eval_idx is not None:
            acc_curve.append(direction_accuracy_idx(pred_delta, true_delta, eval_idx))

    post_mean_bin = vals.mean(dim=0).float().cpu().numpy() * 50.0
    final_pred_delta = post_mean_bin - baseline_mean_bin
    if track_curve and not acc_curve:
        # GEN_ITERS==0 edge case
        acc_curve = [direction_accuracy_idx(final_pred_delta, true_delta, eval_idx)]
    return acc_curve, post_mean_bin, final_pred_delta


@torch.no_grad()
def run_iterative_per_cell(
    model: MLPAutoEncoder,
    X_unit: np.ndarray,
    device: torch.device,
    model_type: str,
) -> np.ndarray:
    """Per-cell velocity in bin space: (final - initial) / n_iters."""
    model.eval()
    vals = torch.from_numpy(X_unit.copy()).to(device)
    initial = vals.float().cpu().numpy() * 50.0

    for _ in range(GEN_ITERS):
        for start in range(0, vals.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, vals.shape[0])
            chunk = vals[start:end]
            if model_type == "dae":
                inp = corrupt_batch(chunk, noise_std=ARGS.noise_std, mask_rate=ARGS.mask_rate)
            else:
                inp = chunk
            recon = model(inp).clamp(0.0, 1.0)
            vals[start:end] = EMA_ALPHA * chunk + (1.0 - EMA_ALPHA) * recon

    final = vals.float().cpu().numpy() * 50.0
    velocity = (final - initial) / max(GEN_ITERS, 1)
    return np.nan_to_num(velocity.astype(np.float32), nan=0.0)


@torch.no_grad()
def run_trajectory_probe(
    model: MLPAutoEncoder,
    pack: dict,
    source: str,
    eval_idx: np.ndarray,
    true_delta: np.ndarray,
    device: torch.device,
) -> Tuple[List[float], np.ndarray, dict, dict]:
    """Track centroid geometry under the same repeated reconstruction operator."""
    if source not in ("early", "late", "mid"):
        raise ValueError(f"Unknown probe source: {source}")
    mask = pack[source]
    rows = np.flatnonzero(mask)
    if ARGS.probe_max_cells > 0 and len(rows) > ARGS.probe_max_cells:
        rng = np.random.default_rng(ARGS.seed + {"early": 11, "late": 23, "mid": 37}[source])
        rows = np.sort(rng.choice(rows, size=ARGS.probe_max_cells, replace=False))
    if len(rows) == 0:
        raise ValueError(f"No cells available for {source} probe")

    vals = torch.from_numpy(pack["X_unit"][rows].copy()).to(device)
    early_mean = pack["X_bin"][pack["early"]].mean(axis=0)
    late_mean = pack["X_bin"][pack["late"]].mean(axis=0)
    global_mean = pack["X_bin"].mean(axis=0)
    means = [vals.mean(dim=0).float().cpu().numpy() * 50.0]
    curve: List[float] = []
    model.eval()

    for _ in range(GEN_ITERS):
        for start in range(0, vals.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, vals.shape[0])
            chunk = vals[start:end]
            inp = corrupt_batch(chunk, ARGS.noise_std, ARGS.mask_rate) if ARGS.model_type == "dae" else chunk
            recon = model(inp).clamp(0.0, 1.0)
            vals[start:end] = EMA_ALPHA * chunk + (1.0 - EMA_ALPHA) * recon
        mean_bin = vals.mean(dim=0).float().cpu().numpy() * 50.0
        means.append(mean_bin)
        curve.append(direction_accuracy_idx(mean_bin - early_mean, true_delta, eval_idx))

    traj = np.asarray(means, dtype=np.float32)
    gene_idx = eval_idx if ARGS.probe_gene_scope == "top_genes" else np.arange(len(true_delta))
    t = traj[:, gene_idx].astype(np.float64)
    e = early_mean[gene_idx].astype(np.float64)
    l = late_mean[gene_idx].astype(np.float64)
    g = global_mean[gene_idx].astype(np.float64)
    axis = l - e
    axis_norm = float(np.linalg.norm(axis))
    denom = max(axis_norm, 1e-12)
    steps = np.linalg.norm(np.diff(t, axis=0), axis=1)
    tail_n = min(3, len(steps))
    tail_mean = float(steps[-tail_n:].mean()) if tail_n else 0.0
    displacement = float(np.linalg.norm(t[-1] - t[0]))
    final_late = float(np.linalg.norm(t[-1] - l))
    final_global = float(np.linalg.norm(t[-1] - g))
    proj_init = float(np.dot(t[0] - e, axis) / denom)
    proj_final = float(np.dot(t[-1] - e, axis) / denom)
    displacement_rel = displacement / denom

    supports = {
        "convergence_and_near_late": None,
        "late_reverse_stable": None,
        "mid_follows_trajectory_not_global": None,
    }
    converged = tail_mean / denom <= ARGS.probe_conv_rel_eps
    if source == "early":
        supports["convergence_and_near_late"] = bool(converged and final_late < final_global)
    elif source == "late":
        supports["late_reverse_stable"] = bool(displacement_rel <= ARGS.probe_late_stability_rel_eps)
    else:
        supports["mid_follows_trajectory_not_global"] = bool(final_late < final_global)

    inv = float((np.sign((traj[-1] - early_mean)[eval_idx]) == -np.sign(true_delta[eval_idx])).mean())
    diag = {
        "dataset": pack["name"], "model": ARGS.model_type, "init_source": source,
        "n_genes": len(pack["genes"]), "n_cells": int(pack["X_bin"].shape[0]),
        "n_early": int(pack["early"].sum()), "n_mid": int(pack["mid"].sum()),
        "n_late": int(pack["late"].sum()), "n_init_cells": int(len(rows)),
        "pt_lo": pack["lo"], "pt_hi": pack["hi"],
        "mid_pt_lo_value": pack["mid_lo"], "mid_pt_hi_value": pack["mid_hi"],
        "vocab_match_rate": 100.0, "top_vocab_match_rate": 100.0,
        "eval_percent": TOP_PERCENT, "eval_genes_count": int(len(eval_idx)),
        "final_acc_inv_truth": inv,
    }
    summary = {
        "dataset": pack["name"], "model": ARGS.model_type, "init_source": source,
        "gene_scope": ARGS.probe_gene_scope, "n_iters_including_t0": int(len(traj)),
        "gen_iters": GEN_ITERS, "axis_norm_late_minus_early": axis_norm,
        "tail_step_norm_mean": tail_mean, "tail_step_norm_rel": tail_mean / denom,
        "converged": bool(converged), "dist_final_late": final_late,
        "dist_final_global": final_global,
        "closer_to_late_than_global": bool(final_late < final_global),
        "proj_init": proj_init, "proj_final": proj_final,
        "moved_toward_late_on_axis": bool(proj_final > proj_init),
        "displacement_from_init": displacement,
        "displacement_from_init_rel": displacement_rel,
        "late_stability_rel_eps": ARGS.probe_late_stability_rel_eps,
        "conv_rel_eps": ARGS.probe_conv_rel_eps, "mid_collapse_stats": {},
        "supports": supports, "diagnostics": diag,
    }
    return curve, traj, diag, summary


# =============================================================================
# Run one dataset
# =============================================================================
def run_dataset(
    name: str,
    pack: dict,
    all_packs: Dict[str, dict],
    device: torch.device,
) -> Tuple[List[float], dict, dict]:
    print(f"\n{'=' * 60}")
    print(f"Running {name}  model={ARGS.model_type}  train={ARGS.train_mode}")
    print(f"{'=' * 60}")

    genes = pack["genes"]
    X_bin = pack["X_bin"]
    X_unit = pack["X_unit"]
    pt = pack["pt"]
    early = pack["early"]
    late = pack["late"]
    cfg = pack["cfg"]

    print(f"  [INFO] genes={len(genes)}  cells={X_bin.shape[0]}")
    print(f"  [INFO] Early (pt<={pack['lo']:.3f}): {int(early.sum())}")
    print(f"  [INFO] Late  (pt>={pack['hi']:.3f}): {int(late.sum())}")

    early_mean = X_bin[early].mean(axis=0)
    late_mean = X_bin[late].mean(axis=0)
    true_delta, truth_label = load_true_delta(genes, cfg, early_mean, late_mean)
    eval_idx, eval_label = select_eval_indices(true_delta)
    print(f"  [INFO] Truth: {truth_label}; eval: {eval_label}")

    # ---- train ----
    hidden = parse_hidden_dims(ARGS.hidden_dims)
    model = MLPAutoEncoder(len(genes), hidden, ARGS.latent_dim).to(device)
    X_train = collect_train_cells(name, pack, all_packs, ARGS.train_mode)
    print(f"  [INFO] Train cells: {X_train.shape[0]}  (gene dim={len(genes)})")

    if ARGS.train_mode == "untrained":
        train_info = {"epochs": 0, "final_loss": None, "n_train": 0, "mode": "untrained"}
        print("  [INFO] Untrained random weights (no fitting).")
    else:
        train_info = train_autoencoder(
            model,
            X_train,
            device=device,
            model_type=ARGS.model_type,
            epochs=ARGS.epochs,
            lr=ARGS.lr,
            weight_decay=ARGS.weight_decay,
            batch_size=ARGS.batch_size,
            noise_std=ARGS.noise_std,
            mask_rate=ARGS.mask_rate,
        )
        train_info["mode"] = ARGS.train_mode

    # ---- iterative eval on early cells (same as scgpt_dyn) ----
    acc_curve, post_early, pred_delta = run_iterative(
        model,
        X_unit[early],
        early_mean,
        true_delta=true_delta,
        eval_idx=eval_idx,
        device=device,
        model_type=ARGS.model_type,
        track_curve=True,
    )
    print(f"  [RESULT] Final accuracy ({eval_label}, {truth_label}): {acc_curve[-1]:.2%}")

    out_ds = Path(OUTDIR) / name
    out_ds.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"gene": genes, "pred_delta": pred_delta, "true_delta": true_delta}
    ).to_csv(out_ds / "gene_delta_compare.csv", index=False)

    # Optional late pass for cross early→late style delta
    _, post_late, _ = run_iterative(
        model,
        X_unit[late],
        late_mean,
        true_delta=None,
        eval_idx=None,
        device=device,
        model_type=ARGS.model_type,
        track_curve=False,
    )
    pred_delta_cross = post_late - post_early
    pd.DataFrame(
        {
            "gene": genes,
            "pred_delta_early_ref": pred_delta,
            "pred_delta_cross": pred_delta_cross,
            "true_delta": true_delta,
        }
    ).to_csv(out_ds / "gene_delta_compare_extended.csv", index=False)

    # Per-cell unsupervised velocity field (no Dynamo required).
    # Primary unsup metric: mean velocity on early cells (same spirit as
    # scgpt_dyn early-ref iterative delta). late-early velocity difference is
    # also saved, but for attractor-style AE it often anti-aligns with true_delta.
    vel_pred = run_iterative_per_cell(model, X_unit, device, ARGS.model_type)
    np.savez_compressed(
        out_ds / "velocity_field.npz",
        vel_cell=vel_pred,
        pt=pt,
        early=early.astype(np.bool_),
        late=late.astype(np.bool_),
    )
    pred_delta_unsup_early = vel_pred[early].mean(axis=0)
    pred_delta_unsup_cross = vel_pred[late].mean(axis=0) - vel_pred[early].mean(axis=0)
    pd.DataFrame(
        {
            "gene": genes,
            "pred_delta_unsup_early": pred_delta_unsup_early,
            "pred_delta_unsup_cross": pred_delta_unsup_cross,
            "true_delta": true_delta,
        }
    ).to_csv(out_ds / "gene_delta_unsup.csv", index=False)
    unsup_dir_acc = direction_accuracy_idx(pred_delta_unsup_early, true_delta, eval_idx)

    meta = {
        "model_type": ARGS.model_type,
        "train_mode": ARGS.train_mode,
        "train": train_info,
        "hidden_dims": hidden,
        "latent_dim": ARGS.latent_dim,
        "gen_iters": GEN_ITERS,
        "ema_alpha": EMA_ALPHA,
        "noise_std": ARGS.noise_std,
        "mask_rate": ARGS.mask_rate,
        "final_direction_acc": acc_curve[-1],
        "unsup_direction_acc": unsup_dir_acc,
        "truth_source": truth_label,
        "eval_mode": eval_label,
    }
    with open(out_ds / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if ARGS.save_ckpts:
        torch.save(
            {"state_dict": model.state_dict(), "n_genes": len(genes), "genes": genes, "meta": meta},
            out_ds / "model.pt",
        )

    diagnostics = {
        "n_genes": len(genes),
        "n_cells": int(X_bin.shape[0]),
        "n_early": int(early.sum()),
        "n_late": int(late.sum()),
        "n_train": int(train_info.get("n_train") or 0),
        "train_final_loss": train_info.get("final_loss"),
        "eval_mode": eval_label,
        "eval_genes_count": int(len(eval_idx)),
        "truth_source": truth_label,
        "model_type": ARGS.model_type,
        "train_mode": ARGS.train_mode,
        "final_direction_acc": float(acc_curve[-1]),
        "unsup_direction_acc": float(unsup_dir_acc),
        # Keep keys used by scgpt_dyn plotting helpers
        "vocab_match_rate": 100.0,
        "eval_vocab_match_rate": 100.0,
    }
    probes = {"curves": {}, "diagnostics": {}, "summary": {}}
    if ARGS.run_trajectory_probes:
        sources = [x.strip() for x in ARGS.probe_init_sources.split(",") if x.strip()]
        for source in sources:
            # Reset the stochastic DAE stream per source for reproducibility.
            set_seed(ARGS.seed + {"early": 11, "late": 23, "mid": 37}.get(source, 0))
            pcurve, traj, pdiag, psummary = run_trajectory_probe(
                model, pack, source, eval_idx, true_delta, device
            )
            key = f"{name}/{source}"
            probes["curves"][key] = pcurve
            probes["diagnostics"][key] = pdiag
            probes["summary"][source] = psummary
            np.save(out_ds / f"trajectory_{source}_mean_by_iter.npy", traj)
            pd.DataFrame(traj, columns=genes).assign(iteration=np.arange(len(traj))).to_csv(
                out_ds / f"trajectory_{source}_mean_by_iter.csv", index=False
            )
            print(
                f"  [PROBE] {source}: final_acc={pcurve[-1]:.2%} "
                f"proj={psummary['proj_init']:.3f}->{psummary['proj_final']:.3f} "
                f"converged={psummary['converged']}"
            )
        # Match the unified runner's convenient root-level direction table.
        pd.DataFrame({
            "gene": genes, "true_early_mean": early_mean, "true_late_mean": late_mean,
            "pred_late_like_mean": post_early, "delta_true": true_delta,
            "delta_pred": pred_delta,
            "dir_true": np.where(true_delta > 0, "Up", "Down"),
            "dir_pred": np.where(pred_delta > 0, "Up", "Down"),
            "dir_correct": (np.where(true_delta > 0, 1, -1) == np.where(pred_delta > 0, 1, -1)).astype(int),
        }).to_csv(Path(OUTDIR) / f"{name}_gene_result.csv", index=False)
    return acc_curve, diagnostics, probes


# =============================================================================
# Plotting (lightweight; same spirit as scgpt_dyn)
# =============================================================================
def plot_results(all_curves: dict, diagnostics: dict, outdir: Path) -> None:
    import matplotlib.pyplot as plt

    eval_mode = next(iter(diagnostics.values())).get("eval_mode", f"top-{TOP_PERCENT}%")
    ylabel = f"Direction accuracy ({eval_mode})"
    colors = {
        "hESC": "#E64B35",
        "hHep": "#E64B35",
        "mDC": "#4DBBD5",
        "mHSC-E": "#00A087",
        "mHSC-GM": "#3C5488",
        "mHSC-L": "#F39B7F",
    }

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for name, acc in all_curves.items():
        ax.plot(
            range(1, len(acc) + 1),
            acc,
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            color=colors.get(name, "#666666"),
            label=name,
        )
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.35, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "fig1_convergence.pdf")
    fig.savefig(outdir / "fig1_convergence.png", dpi=300)
    plt.close(fig)

    names = list(all_curves.keys())
    final_acc = [all_curves[n][-1] for n in names]
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    bars = ax.bar(
        range(len(names)),
        final_acc,
        color=[colors.get(n, "#666666") for n in names],
        width=0.65,
        edgecolor="black",
        linewidth=0.5,
    )
    for bar, v in zip(bars, final_acc):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{v:.1%}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylim(0, 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / "fig2_final_accuracy.pdf")
    fig.savefig(outdir / "fig2_final_accuracy.png", dpi=300)
    plt.close(fig)
    print(f"\n[OK] Figures saved under {outdir}")


# =============================================================================
# Main
# =============================================================================
def main():
    set_seed(ARGS.seed)
    if ARGS.device:
        device = torch.device(ARGS.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(
        f"Config: model={ARGS.model_type} train={ARGS.train_mode} "
        f"iters={GEN_ITERS} ema={EMA_ALPHA} epochs={ARGS.epochs} "
        f"truth={TRUTH_SOURCE}"
    )

    # Preload all datasets (needed for loo; cheap relative to training)
    all_packs: Dict[str, dict] = {}
    for name, cfg in DATASETS.items():
        all_packs[name] = load_dataset_matrix(name, cfg)

    # For loo we also need other datasets even if --dataset filters eval set.
    if ARGS.train_mode == "loo":
        full_cfg = (
            json.loads(Path(ARGS.datasets_json).read_text(encoding="utf-8"))
            if ARGS.datasets_json
            else build_datasets_from_roots(ARGS.expr_root, ARGS.pt_root)
        )
        for name, cfg in full_cfg.items():
            if name not in all_packs:
                print(f"  [INFO] Loading auxiliary dataset for loo: {name}")
                all_packs[name] = load_dataset_matrix(name, cfg)

    all_curves = {}
    all_diagnostics = {}
    trajectory_summary = {}
    for name in DATASETS:
        acc, diag, probes = run_dataset(name, all_packs[name], all_packs, device)
        if ARGS.run_trajectory_probes:
            all_curves.update(probes["curves"])
            all_diagnostics.update(probes["diagnostics"])
        else:
            all_curves[name] = acc
            all_diagnostics[name] = diag
        if probes["summary"]:
            trajectory_summary[name] = probes["summary"]

    with open(outdir / "accuracy_curves.json", "w", encoding="utf-8") as f:
        json.dump(all_curves, f, indent=2)
    with open(outdir / "diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(all_diagnostics, f, indent=2)
    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(ARGS), f, indent=2)
    if trajectory_summary:
        with open(outdir / "trajectory_probes_summary.json", "w", encoding="utf-8") as f:
            json.dump(trajectory_summary, f, indent=2)

    plot_results(all_curves, all_diagnostics, outdir)

    eval_mode = next(iter(all_diagnostics.values())).get("eval_mode", "")
    print("\n" + "=" * 80)
    print(f"SUMMARY ({eval_mode}, model={ARGS.model_type}, train={ARGS.train_mode})")
    print("=" * 80)
    if ARGS.run_trajectory_probes:
        print(f"{'Dataset/source':<24} {'Genes':<8} {'Eval':<8} {'InitN':<8} {'Acc':<10} {'InvAcc':<10}")
        print("-" * 80)
        for name, curve in all_curves.items():
            d = all_diagnostics[name]
            print(
                f"{name:<24} {d['n_genes']:<8} {d['eval_genes_count']:<8} "
                f"{d['n_init_cells']:<8} {curve[-1]:<10.2%} {d['final_acc_inv_truth']:<10.2%}"
            )
    else:
        print(f"{'Dataset':<16} {'Genes':<8} {'Eval':<8} {'TrainN':<8} {'Acc':<10} {'UnsupAcc':<10}")
        print("-" * 80)
        for name, curve in all_curves.items():
            d = all_diagnostics[name]
            print(
                f"{name:<16} {d['n_genes']:<8} {d['eval_genes_count']:<8} "
                f"{d['n_train']:<8} {curve[-1]:<10.2%} {d['unsup_direction_acc']:<10.2%}"
            )
    print("=" * 80)
    print(
        "Interpret vs scgpt_dyn: if AE/DAE accuracy ≈ chance (0.5) while FM >> 0.5, "
        "directed evolution is more consistent with foundation-model priors."
    )


if __name__ == "__main__":
    main()
