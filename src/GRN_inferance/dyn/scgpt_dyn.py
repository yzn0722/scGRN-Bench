


import os
import re
import json
import warnings
from pathlib import Path

import argparse
from typing import Optional
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
os.environ["KMP_WARNINGS"] = "off"

# =====================================================
# CLI ()
# =====================================================
def parse_args():
    p = argparse.ArgumentParser(description="scGPT pseudotime direction-accuracy benchmark (dynamic).")
    p.add_argument("--model-dir", required=True, type=str, help="scGPT model dir containing args.json/vocab.json/best_model.pt.")
    p.add_argument("--outdir", required=True, type=str, help="Output directory.")
    p.add_argument("--datasets-json", default="", type=str, help="Datasets JSON path. If empty, build from --expr-root and --pt-root.")
    p.add_argument("--expr-root", default="", type=str, help="Expression root directory (CHIP/*.csv).")
    p.add_argument("--pt-root", default="", type=str, help="Pseudotime root directory (<dataset>/PseudoTime.csv).")
    p.add_argument("--pt-quantile", type=float, default=0.2)
    p.add_argument("--top-percent", type=int, default=30, help="Evaluate top N%% by |true_delta|; ignored if --eval-all.")
    p.add_argument(
        "--eval-all",
        action="store_true",
        help="Evaluate all genes in the expression matrix (no Top-N%% subset).",
    )
    p.add_argument(
        "--truth-source",
        choices=["expression", "dynamo"],
        default="expression",
        help="Gold-standard delta: expression (late-early) or dynamo_velocity_csv from datasets json.",
    )
    p.add_argument("--dataset", default="", help="Run only this dataset name from datasets-json.")
    p.add_argument("--gen-iters", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--ema-alpha", type=float, default=0.1)
    p.add_argument("--log1p", action="store_true", help="Use log1p in binning (default off to match previous NO_LOG1P=True).")
    p.add_argument(
        "--h5ad",
        default="",
        help="Optional processed Dynamo h5ad for per-time velocity metrics (like RegVelo script).",
    )
    p.add_argument("--velocity-layer", default="velocity_alpha_minus_gamma_s", type=str)
    return p.parse_args()


ARGS = parse_args()

MODEL_DIR = ARGS.model_dir
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
        datasets[ds] = {
            "expr_csv": str(Path(expr_root) / "CHIP" / f"{ds}_chip_matched-ExpressionData.csv"),
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
    return build_datasets_from_roots(args.expr_root, args.pt_root)


DATASETS = load_datasets_config(ARGS)

# =====================================================
# (:TOP_PERCENTTOPK)
# =====================================================
PT_QUANTILE = float(ARGS.pt_quantile)
TOP_PERCENT = int(ARGS.top_percent)
EVAL_ALL = bool(ARGS.eval_all)
TRUTH_SOURCE = ARGS.truth_source
GEN_ITERS = int(ARGS.gen_iters)
BATCH_SIZE = int(ARGS.batch_size)
EMA_ALPHA = float(ARGS.ema_alpha)
NO_LOG1P = (not bool(ARGS.log1p))
H5AD_PATH = ARGS.h5ad
VELOCITY_LAYER = ARGS.velocity_layer

# =====================================================
# scGPT
# =====================================================
import sys
from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab


# =====================================================
# Utils
# =====================================================
def bin_expr_to_0_50(x, do_log1p=True):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0)
    x = np.clip(x, 0, None)
    if do_log1p:
        x = np.log1p(x)
    vmax = max(np.percentile(x, 99.5), 1e-6)
    return np.clip(x / vmax * 50.0, 0, 50).astype(np.float32)


def convert_mouse_to_human_gene(gene_name):
    """
    :
    :  (e.g., Gapdh)
    :  (e.g., GAPDH)
    """
    return gene_name.upper()


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


def load_h5ad_velocity(genes, cells, h5ad_path: str, layer: str) -> Optional[np.ndarray]:
    """Return velocity matrix [n_cells, n_genes] aligned to genes/cells order."""
    if not h5ad_path or not os.path.isfile(h5ad_path):
        return None
    import scanpy as sc
    import scipy.sparse as sp

    adata = sc.read_h5ad(h5ad_path)
    adata.obs_names_make_unique()
    if layer not in adata.layers:
        raise KeyError(f"Layer {layer} missing in {h5ad_path}")

    common = [c for c in cells if c in adata.obs_names]
    if len(common) != len(cells):
        print(f"  [WARN] h5ad cell overlap {len(common)}/{len(cells)}")

    gene_to_idx = {g: i for i, g in enumerate(adata.var_names.astype(str))}
    gi = [gene_to_idx[g] for g in genes if g in gene_to_idx]
    if len(gi) != len(genes):
        print(f"  [WARN] h5ad gene overlap {len(gi)}/{len(genes)}")

    sub = adata[common, gi]
    V = sub.layers[layer]
    if sp.issparse(V):
        V = V.toarray()
    return np.nan_to_num(np.asarray(V, dtype=np.float32), nan=0.0)


def direction_accuracy_idx(pred: np.ndarray, true: np.ndarray, idx: np.ndarray) -> float:
    return float((np.sign(pred[idx]) == np.sign(true[idx])).mean())


def spearman_safe(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if len(a) < 3:
        return float("nan")
    r, _ = spearmanr(a, b)
    return float(r)


def load_true_delta(genes, cfg, early_mean, late_mean):
    if TRUTH_SOURCE == "dynamo":
        vpath = cfg.get("dynamo_velocity_csv", "")
        if not vpath or not os.path.isfile(vpath):
            raise FileNotFoundError(
                f"--truth-source dynamo requires dynamo_velocity_csv in datasets json; got {vpath!r}"
            )
        vdf = pd.read_csv(vpath)
        if "gene" not in vdf.columns:
            vdf = vdf.rename(columns={vdf.columns[0]: "gene"})
        vcol = "dynamo_velocity" if "dynamo_velocity" in vdf.columns else vdf.columns[1]
        vmap = vdf.set_index("gene")[vcol]
        true_delta = np.array([float(vmap[g]) if g in vmap.index else 0.0 for g in genes], dtype=np.float32)
        true_delta = np.nan_to_num(true_delta, nan=0.0)
        return true_delta, "dynamo_velocity"
    return (late_mean - early_mean).astype(np.float32), "expression_delta"


# =====================================================
# Build model
# =====================================================
def build_model(model_dir, device):
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


# =====================================================
# Iterative generation + accuracy curve
# =====================================================
@torch.no_grad()
def run_iterative_per_cell(
    model,
    gene_ids_tensor,
    values_tensor,
    pad_mask,
    update_mask_1d,
) -> np.ndarray:
    """Run MLM iteration per cell; return gene-space velocity (final - initial) / n_iters."""
    device = next(model.parameters()).device
    vals_all = values_tensor.clone()
    initial = vals_all[:, 1:].float().cpu().numpy()
    update_mask = torch.tensor(update_mask_1d[None, :], device=device).bool()

    for _ in range(GEN_ITERS):
        for start in range(0, vals_all.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, vals_all.shape[0])
            bs = end - start

            vals = vals_all[start:end].to(device)
            src = gene_ids_tensor.expand(bs, -1).to(device)
            mask = pad_mask[start:end].to(device)

            freeze = mask | (~update_mask.expand(bs, -1))
            out = model(src=src, values=vals, src_key_padding_mask=mask)
            new_vals = out["mlm_output"]

            vals = torch.where(
                freeze,
                vals,
                EMA_ALPHA * vals + (1 - EMA_ALPHA) * new_vals,
            )

            vals_all[start:end] = vals.detach().cpu()

    final = vals_all[:, 1:].float().cpu().numpy()
    velocity = (final - initial) / max(GEN_ITERS, 1)
    return np.nan_to_num(velocity.astype(np.float32), nan=0.0)


def compute_unsup_velocity_metrics(
    vel_pred: np.ndarray,
    vel_true: np.ndarray,
    pt: np.ndarray,
    early: np.ndarray,
    late: np.ndarray,
    eval_idx: np.ndarray,
) -> dict:
    """Quantitative comparison: unsupervised scGPT per-cell velocity vs Dynamo."""
    n_cells, n_genes = vel_pred.shape
    times = np.sort(np.unique(pt))

    # Gene-level early/late delta (same as transition head eval).
    pred_delta = vel_pred[late].mean(axis=0) - vel_pred[early].mean(axis=0)
    true_delta = vel_true[late].mean(axis=0) - vel_true[early].mean(axis=0)

    eval_mask = np.abs(true_delta[eval_idx]) > 1e-10
    idx_eval = eval_idx[eval_mask]

    per_cell_dir = []
    per_cell_cos = []
    for i in range(n_cells):
        mask = np.abs(vel_true[i, eval_idx]) > 1e-10
        if not mask.any():
            continue
        gi = eval_idx[mask]
        per_cell_dir.append(float((np.sign(vel_pred[i, gi]) == np.sign(vel_true[i, gi])).mean()))
        vn = np.linalg.norm(vel_true[i, gi])
        pn = np.linalg.norm(vel_pred[i, gi])
        if vn > 1e-10 and pn > 1e-10:
            per_cell_cos.append(float(np.dot(vel_pred[i, gi], vel_true[i, gi]) / (vn * pn)))

    per_time_dir = {}
    for t in times:
        mask = pt == t
        if mask.sum() == 0:
            continue
        pred_t = vel_pred[mask].mean(axis=0)
        true_t = vel_true[mask].mean(axis=0)
        per_time_dir[float(t)] = direction_accuracy_idx(pred_t, true_t, eval_idx)

    # Flatten all (cell, gene) pairs on dynamic genes.
    flat_pred = vel_pred[:, eval_idx].reshape(-1)
    flat_true = vel_true[:, eval_idx].reshape(-1)
    flat_mask = np.abs(flat_true) > 1e-10
    flat_dir = float((np.sign(flat_pred[flat_mask]) == np.sign(flat_true[flat_mask])).mean()) if flat_mask.any() else float("nan")

    return {
        "direction_acc_gene_delta": direction_accuracy_idx(pred_delta, true_delta, eval_idx),
        "direction_acc_per_cell_mean": float(np.mean(per_cell_dir)) if per_cell_dir else float("nan"),
        "direction_acc_flat_pairs": flat_dir,
        "spearman_gene_delta": spearman_safe(pred_delta[idx_eval], true_delta[idx_eval]) if len(idx_eval) >= 3 else float("nan"),
        "spearman_abs_gene_delta": spearman_safe(np.abs(pred_delta[idx_eval]), np.abs(true_delta[idx_eval])) if len(idx_eval) >= 3 else float("nan"),
        "cosine_sim_per_cell_mean": float(np.mean(per_cell_cos)) if per_cell_cos else float("nan"),
        "n_cells": int(n_cells),
        "n_genes": int(n_genes),
        "n_eval_genes": int(len(eval_idx)),
        "n_eval_genes_nonzero": int(len(idx_eval)),
        "per_time_direction_acc": per_time_dir,
        "pred_delta": pred_delta,
        "true_delta": true_delta,
    }


@torch.no_grad()
def run_iterative(
    model,
    gene_ids_tensor,
    values_tensor,
    pad_mask,
    update_mask_1d,
    baseline_mean,
    true_delta=None,
    eval_idx=None,
    track_curve=False,
):
    """Run MLM iteration; return optional acc curve, post-iteration mean, pred_delta."""
    device = next(model.parameters()).device
    vals_all = values_tensor.clone()
    update_mask = torch.tensor(update_mask_1d[None, :], device=device).bool()
    acc_curve = []

    for it in range(GEN_ITERS):
        for start in range(0, vals_all.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, vals_all.shape[0])
            bs = end - start

            vals = vals_all[start:end].to(device)
            src = gene_ids_tensor.expand(bs, -1).to(device)
            mask = pad_mask[start:end].to(device)

            freeze = mask | (~update_mask.expand(bs, -1))
            out = model(src=src, values=vals, src_key_padding_mask=mask)
            new_vals = out["mlm_output"]

            vals = torch.where(
                freeze,
                vals,
                EMA_ALPHA * vals + (1 - EMA_ALPHA) * new_vals,
            )

            vals_all[start:end] = vals.detach().cpu()

        post_mean = vals_all[:, 1:].numpy().mean(axis=0)
        pred_delta = post_mean - baseline_mean
        if track_curve and true_delta is not None and eval_idx is not None:
            acc_curve.append(direction_accuracy_idx(pred_delta, true_delta, eval_idx))

    post_mean = vals_all[:, 1:].numpy().mean(axis=0)
    final_pred_delta = post_mean - baseline_mean
    return acc_curve, post_mean, final_pred_delta


def compute_dynamo_style_metrics(
    genes,
    pt,
    early,
    late,
    eval_idx,
    pred_delta_cross,
    true_delta_h5ad,
    csv_delta,
    per_time_df,
):
    """Metrics aligned with scgpt_dyn_reghead_dynamo.py output."""
    dir_all = direction_accuracy_idx(pred_delta_cross, true_delta_h5ad, eval_idx)
    dir_csv = direction_accuracy_idx(pred_delta_cross, csv_delta, eval_idx)

    eval_mask = np.abs(true_delta_h5ad[eval_idx]) > 1e-10
    idx_eval = eval_idx[eval_mask]
    spearman_val = spearman_safe(pred_delta_cross[idx_eval], true_delta_h5ad[idx_eval])
    spearman_abs = spearman_safe(
        np.abs(pred_delta_cross[idx_eval]),
        np.abs(true_delta_h5ad[idx_eval]),
    )

    per_time_dir = {}
    for t in sorted(per_time_df["time"].unique()):
        sub = per_time_df[per_time_df["time"] == t]
        gi = {g: i for i, g in enumerate(genes)}
        pred_v = np.array([sub.loc[sub["gene"] == g, "pred_rate_mean"].values[0] for g in genes])
        true_v = np.array([sub.loc[sub["gene"] == g, "true_rate_mean"].values[0] for g in genes])
        per_time_dir[float(t)] = direction_accuracy_idx(pred_v, true_v, eval_idx)

    delta_df = pd.DataFrame(
        {
            "gene": genes,
            "pred_delta": pred_delta_cross,
            "true_delta_h5ad": true_delta_h5ad,
            "true_delta_csv": csv_delta,
        }
    )

    return {
        "direction_acc_all": dir_all,
        "direction_acc_csv_delta": dir_csv,
        "spearman_pred_vs_true": spearman_val,
        "spearman_abs_magnitude": spearman_abs,
        "per_time_direction_acc": per_time_dir,
        "per_time_df": per_time_df,
        "delta_df": delta_df,
    }


def run_per_time_iterative(
    model,
    gene_ids_tensor,
    genes,
    X_bin,
    pad_mask_full,
    update_mask_1d,
    pt,
    times,
    velocity=None,
):
    """
    For each discrete time t, run iteration on cells at that time.
    pred_rate_mean = post_iter_mean - initial_mean (expression-bin proxy, not kinetic rate).
    """
    rows = []
    post_means = {}
    for t in times:
        mask = pt == t
        if mask.sum() == 0:
            continue
        baseline = X_bin[mask].mean(axis=0)
        _, post_mean, pred_change = run_iterative(
            model,
            gene_ids_tensor,
            torch.tensor(
                np.concatenate([np.zeros((mask.sum(), 1)), X_bin[mask]], axis=1),
                dtype=torch.float16 if next(model.parameters()).device.type == "cuda" else torch.float32,
            ),
            pad_mask_full[mask],
            update_mask_1d,
            baseline,
        )
        post_means[float(t)] = post_mean
        true_t = velocity[mask].mean(axis=0) if velocity is not None else np.zeros(len(baseline), dtype=np.float32)
        for gi, g in enumerate(genes):
            rows.append(
                {
                    "gene": g,
                    "time": float(t),
                    "pred_rate_mean": float(pred_change[gi]),
                    "true_rate_mean": float(true_t[gi]),
                }
            )
    return pd.DataFrame(rows), post_means


# =====================================================
# Run one dataset
# =====================================================
def run_dataset(name, cfg, model, vocab, device):
    print(f"\n{'='*60}")
    print(f"Running {name}")
    print(f"{'='*60}")

    expr = pd.read_csv(cfg["expr_csv"], index_col=0)
    pt_df = pd.read_csv(cfg["pt_csv"])
    pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"}).set_index("cell")

    common = expr.columns.intersection(pt_df.index)
    expr = expr[common]
    pt = pt_df.loc[common, "pt"].to_numpy()

    # 
    genes_original = expr.index.astype(str).tolist()
    
    # ,
    species = cfg.get("species", "human")
    if species == "mouse":
        genes = [convert_mouse_to_human_gene(g) for g in genes_original]
        print(f"  [INFO] Mouse data detected, converting gene names to uppercase")
    else:
        genes = genes_original
    
    # vocab
    matched = sum(1 for g in genes if g in vocab)
    match_rate = matched / len(genes) * 100
    print(f"  [INFO] Total genes: {len(genes)}")
    print(f"  [INFO] Vocab matched: {matched} ({match_rate:.1f}%)")
    print(f"  [INFO] Total cells: {len(common)}")

    X = expr.T.to_numpy(dtype=np.float32)
    X_bin = bin_expr_to_0_50(X, do_log1p=(not NO_LOG1P))

    lo, hi = np.quantile(pt, [PT_QUANTILE, 1 - PT_QUANTILE])
    early = pt <= lo
    late = pt >= hi
    print(f"  [INFO] Early cells (pt <= {lo:.3f}): {early.sum()}")
    print(f"  [INFO] Late cells (pt >= {hi:.3f}): {late.sum()}")

    early_mean = X_bin[early].mean(axis=0)
    late_mean = X_bin[late].mean(axis=0)
    true_delta, truth_label = load_true_delta(genes, cfg, early_mean, late_mean)
    print(f"  [INFO] Truth source: {truth_label}")

    eval_idx, eval_label = select_eval_indices(true_delta)
    print(f"  [INFO] Evaluation genes: {eval_label}")
    print(f"         - Total genes in dataset: {len(true_delta)}")

    eval_genes = [genes[i] for i in eval_idx]
    eval_matched = sum(1 for g in eval_genes if g in vocab)
    eval_match_rate = eval_matched / len(eval_genes) * 100 if eval_genes else 0.0
    print(f"  [INFO] Eval-set vocab matched: {eval_matched} ({eval_match_rate:.1f}%)")

    gene_ids = np.array([vocab[g] if g in vocab else vocab["<pad>"] for g in genes])
    gene_ids = np.concatenate([[vocab["<cls>"]], gene_ids])
    gene_ids_tensor = torch.tensor(gene_ids[None, :], dtype=torch.long)

    X_in = np.concatenate([np.zeros((X_bin.shape[0], 1)), X_bin], axis=1)
    pad_mask = gene_ids_tensor.eq(vocab["<pad>"]).expand(X_in.shape[0], -1)
    values_tensor = torch.tensor(X_in, dtype=torch.float16 if device.type == "cuda" else torch.float32)

    # ()
    update_mask_1d = np.zeros(gene_ids_tensor.shape[1], dtype=bool)
    update_mask_1d[1:] = True  # <cls> token,
    print(f"  [INFO] Iteration covers ALL {len(genes)} genes; eval on {eval_label}")

    acc_curve, post_early, pred_delta_early_ref = run_iterative(
        model,
        gene_ids_tensor,
        values_tensor[early],
        pad_mask[early],
        update_mask_1d,
        early_mean,
        true_delta=true_delta,
        eval_idx=eval_idx,
        track_curve=True,
    )

    print(f"  [RESULT] Final accuracy ({eval_label}, {truth_label}): {acc_curve[-1]:.2%}")

    out_ds = Path(OUTDIR) / name
    out_ds.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "gene": genes,
            "pred_delta": pred_delta_early_ref,
            "true_delta": true_delta,
        }
    ).to_csv(out_ds / "gene_delta_compare.csv", index=False)
    print(f"  [OK] Saved per-gene predictions -> {out_ds / 'gene_delta_compare.csv'}")

    dynamo_metrics = None
    unsup_velocity = None
    h5ad_path = H5AD_PATH or cfg.get("h5ad", "")
    if h5ad_path and os.path.isfile(h5ad_path):
        print(f"  [INFO] Computing RegVelo-style metrics from h5ad: {h5ad_path}")
        velocity = load_h5ad_velocity(genes, common.tolist(), h5ad_path, VELOCITY_LAYER)
        if velocity is not None:
            times = np.sort(np.unique(pt))
            _, post_late, _ = run_iterative(
                model,
                gene_ids_tensor,
                values_tensor[late],
                pad_mask[late],
                update_mask_1d,
                late_mean,
            )
            pred_delta_cross = post_late - post_early
            true_delta_h5ad = velocity[late].mean(axis=0) - velocity[early].mean(axis=0)
            csv_delta = true_delta.copy()

            per_time_df, _ = run_per_time_iterative(
                model,
                gene_ids_tensor,
                genes,
                X_bin,
                pad_mask,
                update_mask_1d,
                pt,
                times,
                velocity=velocity,
            )
            dynamo_metrics = compute_dynamo_style_metrics(
                genes,
                pt,
                early,
                late,
                eval_idx,
                pred_delta_cross,
                true_delta_h5ad,
                csv_delta,
                per_time_df,
            )
            per_time_df.to_csv(out_ds / "gene_rate_per_time.csv", index=False)
            dynamo_metrics["delta_df"].to_csv(out_ds / "gene_rate_delta_compare.csv", index=False)

            print(f"  [INFO] Per-cell unsupervised velocity (all {len(common)} cells, {GEN_ITERS} MLM iters)...")
            vel_pred = run_iterative_per_cell(
                model,
                gene_ids_tensor,
                values_tensor,
                pad_mask,
                update_mask_1d,
            )
            unsup_velocity = compute_unsup_velocity_metrics(
                vel_pred, velocity, pt, early, late, eval_idx
            )

            np.savez_compressed(
                out_ds / "velocity_field.npz",
                vel_cell=vel_pred,
                vel_dynamo=velocity,
                pt=pt,
                times=times,
            )
            pd.DataFrame(
                {
                    "gene": genes,
                    "pred_delta_unsup": unsup_velocity["pred_delta"],
                    "true_delta_dynamo": unsup_velocity["true_delta"],
                }
            ).to_csv(out_ds / "gene_delta_unsup_vs_dynamo.csv", index=False)

            summary = {
                k: v
                for k, v in dynamo_metrics.items()
                if k not in ("per_time_df", "delta_df")
            }
            summary["method"] = "scgpt_iterative"
            summary["pred_delta_definition"] = "post_iter_late_mean - post_iter_early_mean (bin space)"
            summary["pred_rate_mean_definition"] = "post_iter_mean - initial_mean per time group (bin space)"
            summary["per_time_direction_acc"] = {
                str(k): v for k, v in dynamo_metrics["per_time_direction_acc"].items()
            }
            with open(out_ds / "scgpt_iter_dynamo_metrics.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            unsup_summary = {
                k: v
                for k, v in unsup_velocity.items()
                if k not in ("pred_delta", "true_delta")
            }
            unsup_summary["method"] = "scgpt_iterative_per_cell"
            unsup_summary["velocity_definition"] = f"(X_final - X_initial) / {GEN_ITERS} after MLM iteration"
            unsup_summary["supervision"] = "none (Dynamo used for evaluation only)"
            unsup_summary["per_time_direction_acc"] = {
                str(k): v for k, v in unsup_velocity["per_time_direction_acc"].items()
            }
            with open(out_ds / "scgpt_unsup_velocity_metrics.json", "w", encoding="utf-8") as f:
                json.dump(unsup_summary, f, indent=2)

            print(f"  [METRICS] direction (h5ad delta): {dynamo_metrics['direction_acc_all']:.2%}")
            print(f"  [METRICS] direction (csv delta):  {dynamo_metrics['direction_acc_csv_delta']:.2%}")
            print(f"  [METRICS] Spearman (value):       {dynamo_metrics['spearman_pred_vs_true']:.3f}")
            print(f"  [METRICS] Spearman (|value|):     {dynamo_metrics['spearman_abs_magnitude']:.3f}")
            for t, acc in dynamo_metrics["per_time_direction_acc"].items():
                print(f"  [METRICS] direction @ time={t}:    {acc:.2%}")
            print(f"  [UNSUP] per-cell velocity field -> {out_ds / 'velocity_field.npz'}")
            print(f"  [UNSUP] direction (gene delta):   {unsup_velocity['direction_acc_gene_delta']:.2%}")
            print(f"  [UNSUP] direction (per-cell mean): {unsup_velocity['direction_acc_per_cell_mean']:.2%}")
            print(f"  [UNSUP] direction (flat pairs):    {unsup_velocity['direction_acc_flat_pairs']:.2%}")
            print(f"  [UNSUP] cosine (per-cell mean):   {unsup_velocity['cosine_sim_per_cell_mean']:.3f}")
            print(f"  [UNSUP] Spearman (gene delta):    {unsup_velocity['spearman_gene_delta']:.3f}")
            for t, acc in unsup_velocity["per_time_direction_acc"].items():
                print(f"  [UNSUP] direction @ time={t}:     {acc:.2%}")
            print(f"  [OK] Saved RegVelo-style tables -> {out_ds}")

    diagnostics = {
        "n_genes": len(genes),
        "n_cells": len(common),
        "vocab_match_rate": match_rate,
        "eval_vocab_match_rate": eval_match_rate,
        "n_early": int(early.sum()),
        "n_late": int(late.sum()),
        "iter_genes_count": len(genes),
        "eval_mode": eval_label,
        "eval_genes_count": len(eval_idx),
        "truth_source": truth_label,
    }
    if dynamo_metrics is not None:
        diagnostics["dynamo_style_metrics"] = {
            k: v
            for k, v in dynamo_metrics.items()
            if k not in ("per_time_df", "delta_df")
        }
        diagnostics["dynamo_style_metrics"]["per_time_direction_acc"] = {
            str(k): v for k, v in dynamo_metrics["per_time_direction_acc"].items()
        }
    if unsup_velocity is not None:
        diagnostics["unsup_per_cell_velocity"] = {
            k: v
            for k, v in unsup_velocity.items()
            if k not in ("pred_delta", "true_delta")
        }
        diagnostics["unsup_per_cell_velocity"]["per_time_direction_acc"] = {
            str(k): v for k, v in unsup_velocity["per_time_direction_acc"].items()
        }

    return acc_curve, diagnostics


# =====================================================
# Nature-style plotting
# =====================================================
def plot_nature_style(all_curves, diagnostics, outdir):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    eval_mode = next(iter(diagnostics.values())).get("eval_mode", f"top-{TOP_PERCENT}%")
    acc_ylabel = f"Direction accuracy ({eval_mode})"
    
    # Nature
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 8,
        'axes.linewidth': 0.8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'legend.fontsize': 8,
        'legend.frameon': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })
    
    # Nature
    colors = {
        'hHep': '#E64B35',      # 
        'mDC': '#4DBBD5',       # 
        'mHSC-E': '#00A087',    # 
        'mHSC-GM': '#3C5488',   # 
    }
    
    # ==================== Figure 1:  ====================
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    
    for name, acc in all_curves.items():
        ax.plot(range(1, GEN_ITERS + 1), acc, 
                marker='o', markersize=4, linewidth=1.5,
                color=colors.get(name, '#666666'),
                label=name)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel(acc_ylabel)
    ax.set_xlim(0.5, GEN_ITERS + 0.5)
    ax.set_ylim(0.4, 1.0)
    ax.set_xticks(range(1, GEN_ITERS + 1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right', ncol=2)
    
    plt.tight_layout()
    plt.savefig(outdir / "fig1_convergence.pdf")
    plt.savefig(outdir / "fig1_convergence.png", dpi=300)
    plt.close()
    
    # ==================== Figure 2:  ====================
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    
    names = list(all_curves.keys())
    final_acc = [all_curves[n][-1] for n in names]
    bar_colors = [colors.get(n, '#666666') for n in names]
    
    bars = ax.bar(range(len(names)), final_acc, color=bar_colors, width=0.6, edgecolor='black', linewidth=0.5)
    
    # 
    for i, (bar, v) in enumerate(zip(bars, final_acc)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{v:.1%}', ha='center', va='bottom', fontsize=7)
    
    ax.set_ylabel(acc_ylabel)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(outdir / "fig2_final_accuracy.pdf")
    plt.savefig(outdir / "fig2_final_accuracy.png", dpi=300)
    plt.close()
    
    # ==================== Figure 3:  - Vocab vs  ====================
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    
    for name in names:
        match_rate = diagnostics[name].get("eval_vocab_match_rate", diagnostics[name].get("top_vocab_match_rate", 0))
        acc = all_curves[name][-1]
        ax.scatter(match_rate, acc, s=60, c=colors.get(name, '#666666'), 
                   edgecolor='black', linewidth=0.5, label=name, zorder=3)
    
    ax.set_xlabel("Eval-set vocab match rate (%)")
    ax.set_ylabel(acc_ylabel)
    ax.set_xlim(0, 105)
    ax.set_ylim(0.4, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(outdir / "fig3_vocab_vs_accuracy.pdf")
    plt.savefig(outdir / "fig3_vocab_vs_accuracy.png", dpi=300)
    plt.close()
    
    # ==================== Figure 4:  () ====================
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
    
    # Panel A: 
    ax = axes[0]
    for name, acc in all_curves.items():
        ax.plot(range(1, GEN_ITERS + 1), acc, 
                marker='o', markersize=4, linewidth=1.5,
                color=colors.get(name, '#666666'),
                label=name)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(acc_ylabel)
    ax.set_xlim(0.5, GEN_ITERS + 0.5)
    ax.set_ylim(0.4, 1.0)
    ax.set_xticks(range(1, GEN_ITERS + 1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right', fontsize=7)
    ax.text(-0.15, 1.05, 'a', transform=ax.transAxes, fontsize=12, fontweight='bold')
    
    # Panel B: 
    ax = axes[1]
    bars = ax.bar(range(len(names)), final_acc, color=bar_colors, width=0.6, edgecolor='black', linewidth=0.5)
    for i, (bar, v) in enumerate(zip(bars, final_acc)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{v:.1%}', ha='center', va='bottom', fontsize=7)
    ax.set_ylabel(acc_ylabel)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.15, 1.05, 'b', transform=ax.transAxes, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(outdir / "fig_combined.pdf")
    plt.savefig(outdir / "fig_combined.png", dpi=300)
    plt.close()
    
    print(f"\n[OK] Nature-style figures saved to: {outdir}")


# =====================================================
# Main
# =====================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model, vocab = build_model(MODEL_DIR, device)
    print(f"Vocab size: {len(vocab)}")
    eval_label = "all genes" if EVAL_ALL or TOP_PERCENT >= 100 else f"top-{TOP_PERCENT}%"
    print(f"Run mode: iterate all input genes; evaluate {eval_label}; truth={TRUTH_SOURCE}")

    all_curves = {}
    all_diagnostics = {}

    for name, cfg in DATASETS.items():
        acc_curve, diag = run_dataset(name, cfg, model, vocab, device)
        all_curves[name] = acc_curve
        all_diagnostics[name] = diag

    outdir = Path(OUTDIR)
    outdir.mkdir(exist_ok=True, parents=True)

    # Save results
    with open(outdir / "accuracy_curves.json", "w") as f:
        json.dump(all_curves, f, indent=2)
    
    with open(outdir / "diagnostics.json", "w") as f:
        json.dump(all_diagnostics, f, indent=2)

    # Nature
    plot_nature_style(all_curves, all_diagnostics, outdir)
    
    # ()
    eval_mode = next(iter(all_diagnostics.values())).get("eval_mode", eval_label) if all_diagnostics else eval_label
    truth = next(iter(all_diagnostics.values())).get("truth_source", TRUTH_SOURCE) if all_diagnostics else TRUTH_SOURCE
    print("\n" + "="*80)
    print(f"SUMMARY ({eval_mode}, truth={truth})")
    print("="*80)
    print(f"{'Dataset':<20} {'Genes':<8} {'Eval':<8} {'Vocab%':<10} {'EvalVocab%':<12} {'Accuracy':<10}")
    print("-"*80)
    for name in all_curves:
        d = all_diagnostics[name]
        acc = all_curves[name][-1]
        ev_vocab = d.get("eval_vocab_match_rate", d.get("top_vocab_match_rate", 0))
        print(f"{name:<20} {d['n_genes']:<8} {d['eval_genes_count']:<8} {d['vocab_match_rate']:<10.1f} {ev_vocab:<12.1f} {acc:<10.2%}")
    print("="*80)


if __name__ == "__main__":
    main()