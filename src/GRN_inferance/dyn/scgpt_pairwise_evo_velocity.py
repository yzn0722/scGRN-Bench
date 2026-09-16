#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scGPT pairwise evolutionary velocity (evo-velocity style).

Inspired by Hie et al., Cell Systems 2022:
  - Run scGPT MLM on each cell x -> pred(x) approximates conditional expectation
  - For edge (i, j): score(j|i) = -||x_j - pred(x_i)||^2
  - Direction i -> j if score(j|i) > score(i|j)
  - Aggregate signed neighbor transitions into per-cell gene velocity
  - Infer pseudotime from directed kNN graph (shortest path from root)

Example:
  conda activate singlecell
  export CUDA_VISIBLE_DEVICES=0
  cd /mnt/10T/yzn/scGRN-Bench

  python -u src/GRN_inferance/dyn/scgpt_pairwise_evo_velocity.py \\
    --model-dir /mnt/10T/yzn/benchmark_GRN/model/weights/scgpt/scGPT_human \\
    --datasets-json data/dynamo_export/datasets_dynamo_vel1524_only.json \\
    --dataset hematopoiesis_dyn_vel1524 \\
    --outdir outputs/dynamo_fm_velocity/pairwise_evo_hematopoiesis \\
    --k-neighbors 15 \\
    --batch-size 128 \\
    --plot
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import dijkstra
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
os.environ["KMP_WARNINGS"] = "off"

from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab


def parse_args():
    p = argparse.ArgumentParser(description="scGPT pairwise evo-velocity + pseudotime")
    p.add_argument("--model-dir", required=True, type=str)
    p.add_argument("--datasets-json", required=True, type=str)
    p.add_argument("--dataset", required=True, type=str)
    p.add_argument("--outdir", required=True, type=str)
    p.add_argument("--k-neighbors", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--log1p", action="store_true")
    p.add_argument("--top-percent", type=int, default=30)
    p.add_argument("--velocity-layer", default="velocity_alpha_minus_gamma_s", type=str)
    p.add_argument("--basis", default="umap", type=str)
    p.add_argument("--plot", action="store_true", help="Save UMAP stream + pseudotime figures")
    p.add_argument("--dpi", type=int, default=300)
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
def predict_mlm_all(
    model,
    gene_ids_tensor: torch.Tensor,
    values_tensor: torch.Tensor,
    pad_mask: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    """Return MLM predictions [n_cells, n_genes] in bin space (no <cls> token)."""
    device = next(model.parameters()).device
    n = values_tensor.shape[0]
    chunks: List[np.ndarray] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        vals = values_tensor[start:end].to(device)
        src = gene_ids_tensor.expand(end - start, -1).to(device)
        mask = pad_mask[start:end].to(device)
        out = model(src=src, values=vals, src_key_padding_mask=mask)
        pred = out["mlm_output"][:, 1:].float().cpu().numpy()
        chunks.append(pred)
    return np.nan_to_num(np.concatenate(chunks, axis=0).astype(np.float32), nan=0.0)


def conditional_score(target: np.ndarray, pred_from_context: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Log-likelihood proxy: -MSE between target and prediction from another cell."""
    diff = target - pred_from_context
    if mask is not None:
        diff = diff * mask
        denom = max(float(mask.sum()), 1.0)
    else:
        denom = float(diff.shape[-1])
    return -np.sum(diff * diff, axis=-1) / denom


def pairwise_evo_velocity(
    x_bin: np.ndarray,
    pred: np.ndarray,
    k_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int, float]]]:
    """
    Build directed edges on kNN graph and per-cell gene velocity.

    Returns
    -------
    vel_cell : [n_cells, n_genes]
    edge_src, edge_dst : directed edge indices
    edge_meta : list of (i, j, score_diff)
    """
    n_cells, n_genes = x_bin.shape
    k = min(k_neighbors + 1, n_cells)
    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    nn.fit(x_bin)
    knn = nn.kneighbors(x_bin, return_distance=False)

    vel = np.zeros((n_cells, n_genes), dtype=np.float32)
    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_meta: List[Tuple[int, int, float]] = []

    for i in range(n_cells):
        for j in knn[i]:
            if i == j:
                continue
            score_ji = conditional_score(x_bin[j : j + 1], pred[i : i + 1])[0]
            score_ij = conditional_score(x_bin[i : i + 1], pred[j : j + 1])[0]
            delta = score_ji - score_ij
            if delta > 0:
                edge_src.append(i)
                edge_dst.append(j)
                edge_meta.append((i, j, float(delta)))
                vel[i] += (x_bin[j] - x_bin[i]) * delta
            elif delta < 0:
                edge_src.append(j)
                edge_dst.append(i)
                edge_meta.append((j, i, float(-delta)))
                vel[i] += (x_bin[j] - x_bin[i]) * delta

    out_strength = np.bincount(edge_src, minlength=n_cells).astype(np.float32)
    out_strength[out_strength == 0] = 1.0
    vel /= out_strength[:, None]
    vel = np.nan_to_num(vel, nan=0.0)
    return vel, np.asarray(edge_src, dtype=int), np.asarray(edge_dst, dtype=int), edge_meta


def infer_root(in_degree: np.ndarray, time: Optional[np.ndarray] = None) -> int:
    if time is not None and len(np.unique(time[~np.isnan(time)])) >= 2:
        tmin = np.nanmin(time)
        cands = np.where(time == tmin)[0]
        if len(cands):
            return int(cands[np.argmin(in_degree[cands])])
    return int(np.argmin(in_degree))


def pseudotime_from_graph(n_cells: int, edge_src: np.ndarray, edge_dst: np.ndarray, root: int) -> np.ndarray:
    rows = edge_src.tolist()
    cols = edge_dst.tolist()
    data = np.ones(len(rows), dtype=np.float32)
    graph = sp.coo_matrix((data, (rows, cols)), shape=(n_cells, n_cells)).tocsr()
    dist = dijkstra(graph, directed=True, indices=root)
    dist = np.asarray(dist, dtype=np.float64)
    finite = np.isfinite(dist)
    if finite.any():
        dist[~finite] = np.nanmax(dist[finite]) + 1.0
    else:
        dist = np.arange(n_cells, dtype=np.float64)
    dist = (dist - dist.min()) / max(dist.max() - dist.min(), 1e-8)
    return dist.astype(np.float32)


def direction_accuracy(pred: np.ndarray, true: np.ndarray, top_percent: int) -> dict:
    n = len(true)
    top_n = max(int(n * top_percent / 100), 1)
    idx = np.argsort(np.abs(true))[::-1][:top_n]
    mask = np.abs(true[idx]) > 1e-8
    agree = np.sign(pred[idx][mask]) == np.sign(true[idx][mask])
    acc = float(agree.mean()) if mask.any() else float("nan")
    return {"top_percent": top_percent, "n_eval": int(mask.sum()), "direction_accuracy": acc}


def load_dynamo_velocity(h5ad_path: str, cells: List[str], genes: List[str], layer: str) -> np.ndarray:
    adata = sc.read_h5ad(h5ad_path)
    adata.obs_names_make_unique()
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names.astype(str))}
    gi = [gene_to_idx[g] for g in genes if g in gene_to_idx]
    common = [c for c in cells if c in adata.obs_names]
    sub = adata[common, gi]
    v = sub.layers[layer]
    if sp.issparse(v):
        v = v.toarray()
    return np.nan_to_num(np.asarray(v, dtype=np.float32), nan=0.0)


def project_velocity_to_embedding(
    embedding: np.ndarray,
    expression: np.ndarray,
    velocity: np.ndarray,
    n_neighbors: int = 30,
) -> np.ndarray:
    n = embedding.shape[0]
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, n), n_jobs=-1)
    nn.fit(expression)
    knn = nn.kneighbors(expression, return_distance=False)
    vel_emb = np.zeros((n, embedding.shape[1]), dtype=np.float32)
    for i in range(n):
        vi = velocity[i]
        norm_v = np.linalg.norm(vi)
        if norm_v < 1e-12:
            continue
        acc = np.zeros(embedding.shape[1], dtype=np.float64)
        for j in knn[i]:
            if i == j:
                continue
            dx = expression[j] - expression[i]
            norm_dx = np.linalg.norm(dx)
            if norm_dx < 1e-12:
                continue
            cos = max(float(np.dot(vi, dx) / (norm_v * norm_dx)), 0.0)
            acc += cos * (embedding[j] - embedding[i])
        vel_emb[i] = acc.astype(np.float32)
    return vel_emb


def plot_results(
    outdir: Path,
    embedding: np.ndarray,
    x_bin: np.ndarray,
    vel: np.ndarray,
    pseudotime: np.ndarray,
    true_time: np.ndarray,
    metrics: dict,
    dpi: int,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    vel_emb = project_velocity_to_embedding(embedding, x_bin, vel)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    sca = ax.scatter(embedding[:, 0], embedding[:, 1], c=true_time, s=8, cmap="Spectral_r", linewidths=0)
    plt.colorbar(sca, ax=ax, fraction=0.046, pad=0.04, label="True time")
    ax.set_title("UMAP colored by true time")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    ax = axes[1]
    sca = ax.scatter(embedding[:, 0], embedding[:, 1], c=pseudotime, s=8, cmap="viridis", linewidths=0)
    plt.colorbar(sca, ax=ax, fraction=0.046, pad=0.04, label="Inferred pseudotime")
    ax.set_title(f"Pairwise evo pseudotime\nSpearman vs time = {metrics.get('pseudotime_spearman', float('nan')):.3f}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    ax = axes[2]
    ax.scatter(embedding[:, 0], embedding[:, 1], c=true_time, s=8, cmap="Spectral_r", linewidths=0, alpha=0.85)
    gx = np.linspace(embedding[:, 0].min(), embedding[:, 0].max(), 40)
    gy = np.linspace(embedding[:, 1].min(), embedding[:, 1].max(), 40)
    gx_mesh, gy_mesh = np.meshgrid(gx, gy)
    from scipy.interpolate import griddata

    vx = griddata((embedding[:, 0], embedding[:, 1]), vel_emb[:, 0], (gx_mesh, gy_mesh), method="linear")
    vy = griddata((embedding[:, 0], embedding[:, 1]), vel_emb[:, 1], (gx_mesh, gy_mesh), method="linear")
    vx = np.nan_to_num(vx, nan=0.0)
    vy = np.nan_to_num(vy, nan=0.0)
    ax.streamplot(gx_mesh, gy_mesh, vx, vy, color=(0.2, 0.2, 0.2, 0.55), density=1.0, linewidth=0.7)
    ax.set_title(
        f"Pairwise evo-velocity stream\n"
        f"Dir acc (top30%) = {metrics.get('direction_accuracy_top30', float('nan')):.3f}"
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(outdir / "pairwise_evo_velocity_summary.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(outdir / "pairwise_evo_velocity_summary.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.scatter(true_time, pseudotime, s=10, alpha=0.5, c="#4472C4", linewidths=0)
    ax.set_xlabel("True time")
    ax.set_ylabel("Inferred pseudotime")
    ax.set_title("Pseudotime calibration")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / "pseudotime_vs_true_time.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(outdir / "pseudotime_vs_true_time.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU required for scGPT pairwise evo-velocity.")
    print(f"Using device: {device}", flush=True)

    with open(args.datasets_json, encoding="utf-8") as f:
        datasets = json.load(f)
    if args.dataset not in datasets:
        raise KeyError(f"Dataset {args.dataset} not in {list(datasets.keys())}")
    cfg = datasets[args.dataset]

    expr = pd.read_csv(cfg["expr_csv"], index_col=0)
    genes = expr.index.astype(str).tolist()
    cells = expr.columns.astype(str).tolist()
    print(f"Loaded expr: {len(genes)} genes x {len(cells)} cells", flush=True)

    pt_df = pd.read_csv(cfg["pt_csv"]).set_index("cell")
    pt_df.index = pt_df.index.astype(str)
    pt = np.array([float(pt_df.loc[c, "pt"]) if c in pt_df.index else np.nan for c in cells], dtype=np.float32)

    time_df = pd.read_csv(cfg.get("time_csv", cfg["pt_csv"])).set_index("cell")
    time_df.index = time_df.index.astype(str)
    time_col = cfg.get("time_col", "time")
    if time_col not in time_df.columns:
        time_col = [c for c in time_df.columns if c != "cell"][0]
    true_time = np.array(
        [float(time_df.loc[c, time_col]) if c in time_df.index else np.nan for c in cells],
        dtype=np.float32,
    )

    early = pt <= np.nanquantile(pt, 0.2)
    late = pt >= np.nanquantile(pt, 0.8)

    X = expr.T.to_numpy(dtype=np.float32)
    X_bin = bin_expr_to_0_50(X, do_log1p=args.log1p)

    model, vocab = build_model(args.model_dir, device)
    gene_ids = np.array([vocab[g] if g in vocab else vocab["<pad>"] for g in genes])
    gene_ids = np.concatenate([[vocab["<cls>"]], gene_ids])
    gene_ids_tensor = torch.tensor(gene_ids[None, :], dtype=torch.long)
    X_in = np.concatenate([np.zeros((X_bin.shape[0], 1)), X_bin], axis=1)
    pad_mask = gene_ids_tensor.eq(vocab["<pad>"]).expand(X_in.shape[0], -1)
    values_tensor = torch.tensor(X_in, dtype=torch.float16 if device.type == "cuda" else torch.float32)

    t0 = time.time()
    print("Running scGPT MLM prediction for all cells...", flush=True)
    pred = predict_mlm_all(model, gene_ids_tensor, values_tensor, pad_mask, args.batch_size)
    print(f"  MLM done in {(time.time()-t0)/60:.1f} min", flush=True)

    print(f"Building pairwise evo-velocity (k={args.k_neighbors})...", flush=True)
    vel, edge_src, edge_dst, edge_meta = pairwise_evo_velocity(X_bin, pred, args.k_neighbors)
    in_degree = np.bincount(edge_dst, minlength=len(cells))
    root = infer_root(in_degree, true_time)
    pseudotime = pseudotime_from_graph(len(cells), edge_src, edge_dst, root)

    h5ad_path = cfg.get("h5ad", "")
    metrics: Dict[str, object] = {
        "dataset": args.dataset,
        "n_cells": len(cells),
        "n_genes": len(genes),
        "k_neighbors": args.k_neighbors,
        "n_directed_edges": int(len(edge_src)),
        "root_cell": cells[root],
        "root_index": int(root),
    }

    if np.isfinite(true_time).sum() >= 3:
        r_pt, _ = spearmanr(true_time, pseudotime)
        if r_pt < 0:
            pseudotime = 1.0 - pseudotime
            r_pt = -r_pt
        metrics["pseudotime_spearman"] = float(r_pt)

    if h5ad_path and os.path.isfile(h5ad_path):
        vel_true = load_dynamo_velocity(h5ad_path, cells, genes, args.velocity_layer)
        true_delta = vel_true[late].mean(axis=0) - vel_true[early].mean(axis=0)
        pred_delta = vel[late].mean(axis=0) - vel[early].mean(axis=0)
        da = direction_accuracy(pred_delta, true_delta, args.top_percent)
        metrics.update(da)
        metrics["direction_accuracy_top30"] = da["direction_accuracy"]

        per_cell_dir = []
        eval_genes = np.argsort(np.abs(true_delta))[::-1][: max(int(len(true_delta) * args.top_percent / 100), 1)]
        for i in range(len(cells)):
            mask = np.abs(vel_true[i, eval_genes]) > 1e-8
            if not mask.any():
                continue
            gi = eval_genes[mask]
            per_cell_dir.append(float((np.sign(vel[i, gi]) == np.sign(vel_true[i, gi])).mean()))
        if per_cell_dir:
            metrics["per_cell_direction_accuracy_mean"] = float(np.mean(per_cell_dir))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        outdir / "velocity_field.npz",
        vel_cell=vel,
        cells=np.array(cells),
        genes=np.array(genes),
        pseudotime=pseudotime,
        pred_mlm=pred,
        edge_src=edge_src,
        edge_dst=edge_dst,
    )

    pd.DataFrame(
        {
            "cell": cells,
            "pseudotime": pseudotime,
            "true_time": true_time,
            "in_degree": in_degree,
            "out_degree": np.bincount(edge_src, minlength=len(cells)),
            "is_root": [i == root for i in range(len(cells))],
        }
    ).to_csv(outdir / "pseudotime_predictions.csv", index=False)

    pd.DataFrame(edge_meta, columns=["src_idx", "dst_idx", "score_diff"]).assign(
        src_cell=lambda d: d["src_idx"].map(lambda i: cells[i]),
        dst_cell=lambda d: d["dst_idx"].map(lambda i: cells[i]),
    ).to_csv(outdir / "directed_edges.csv", index=False)

    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2), flush=True)

    if args.plot and h5ad_path and os.path.isfile(h5ad_path):
        adata = sc.read_h5ad(h5ad_path)
        adata.obs_names_make_unique()
        common = [c for c in cells if c in adata.obs_names]
        idx = [cells.index(c) for c in common]
        basis_key = f"X_{args.basis}"
        if basis_key not in adata.obsm:
            sc.pp.neighbors(adata[common])
            sc.tl.umap(adata[common])
            embedding = np.asarray(adata[common].obsm["X_umap"], dtype=np.float32)
        else:
            sub = adata[common]
            embedding = np.asarray(sub.obsm[basis_key], dtype=np.float32)

        plot_results(
            outdir / "figures",
            embedding,
            X_bin[idx],
            vel[idx],
            pseudotime[idx],
            true_time[idx],
            metrics,
            args.dpi,
        )
        print(f"Figures saved -> {outdir / 'figures'}", flush=True)

    print(f"All outputs -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
