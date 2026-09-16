#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Velocity stream plot on UMAP (Dynamo vs scGPT transition).

Projection modes (--projection):
  unified_neighbor   both methods project gene-space velocity with the same
                     neighbor algorithm (default; fair comparison)
  dynamo_precomputed Dynamo uses h5ad obsm['velocity_umap']; scGPT uses neighbor
  dynamo_precomputed_flip  same as above but flips Dynamo UMAP velocity sign

Example:
  python plot_velocity_stream.py \\
    --h5ad .../hematopoiesis_processed.h5ad \\
    --velocity-npz .../velocity_field.npz \\
    --datasets-json data/dynamo_export/datasets_dynamo_vel1524_only.json \\
    --dataset hematopoiesis_dyn_vel1524 \\
    --scgpt-expr-space bins \\
    --color-key cell_type \\
    --projection unified_neighbor \\
    --outdir outputs/.../figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.interpolate import griddata
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors


def parse_args():
    p = argparse.ArgumentParser(description="Velocity stream plot on UMAP.")
    p.add_argument("--h5ad", required=True, type=str)
    p.add_argument("--velocity-npz", default="", type=str, help="scGPT velocity_field.npz (optional).")
    p.add_argument("--datasets-json", default="", type=str)
    p.add_argument("--cell-type-csv", default="", type=str, help="Optional CellType.csv from enrich_dynamo_export.py")
    p.add_argument("--dataset", default="", type=str)
    p.add_argument("--velocity-layer", default="velocity_alpha_minus_gamma_s", type=str)
    p.add_argument("--basis", default="umap", type=str)
    p.add_argument("--color-key", default="time", type=str, help="obs column for cell coloring.")
    p.add_argument("--outdir", required=True, type=str)
    p.add_argument("--prefix", default="stream", type=str)
    p.add_argument("--grid-size", type=int, default=50)
    p.add_argument("--n-neighbors", type=int, default=30)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--point-size", type=float, default=8.0)
    p.add_argument("--stream-density", type=float, default=1.2)
    p.add_argument("--stream-lw", type=float, default=0.8)
    p.add_argument("--stream-alpha", type=float, default=0.55)
    p.add_argument(
        "--projection",
        default="unified_neighbor",
        choices=["unified_neighbor", "dynamo_precomputed", "dynamo_precomputed_flip"],
        help=(
            "How to obtain UMAP velocity arrows. "
            "unified_neighbor: same neighbor projection for Dynamo and scGPT (recommended). "
            "dynamo_precomputed*: Dynamo uses h5ad velocity_umap (legacy / may look reversed)."
        ),
    )
    p.add_argument(
        "--scgpt-expr-space",
        default="bins",
        choices=["bins", "M_s", "X", "raw_csv"],
        help=(
            "Expression matrix for neighbor projection of scGPT gene-space velocity. "
            "Unsupervised/supervised scGPT velocities are computed in scGPT bin space (0-50); "
            "use 'bins' (default) to match. Dynamo panel always uses M_s."
        ),
    )
    p.add_argument(
        "--auto-flip-sign",
        action="store_true",
        default=True,
        help="Flip projected scGPT arrows when time-alignment cosine is negative (default: on).",
    )
    p.add_argument(
        "--no-auto-flip-sign",
        action="store_false",
        dest="auto_flip_sign",
        help="Disable automatic sign correction for scGPT velocity on UMAP.",
    )
    return p.parse_args()


ARGS = parse_args()


def load_cells_and_genes(cfg: Optional[dict] = None) -> Tuple[Optional[pd.Index], Optional[list], Optional[dict]]:
    if cfg is None:
        if not ARGS.datasets_json or not ARGS.dataset:
            return None, None, None
        with open(ARGS.datasets_json, encoding="utf-8") as f:
            cfg = json.load(f)[ARGS.dataset]
    expr = pd.read_csv(cfg["expr_csv"], index_col=0)
    pt_df = pd.read_csv(cfg["pt_csv"])
    pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"}).set_index("cell")
    common = expr.columns.intersection(pt_df.index)
    genes = expr.index.astype(str).tolist()
    return common, genes, cfg


def bin_expr_to_0_50(x: np.ndarray, do_log1p: bool = False) -> np.ndarray:
    """Match scgpt_dyn.py / scgpt_dyn_transition.py binning."""
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0)
    x = np.clip(x, 0, None)
    if do_log1p:
        x = np.log1p(x)
    vmax = max(float(np.percentile(x, 99.5)), 1e-6)
    return np.clip(x / vmax * 50.0, 0, 50).astype(np.float32)


def load_expression_matrix(
    adata: sc.AnnData,
    cfg: Optional[dict],
    space: str,
) -> Tuple[np.ndarray, str]:
    """Return [n_cells, n_genes] expression for velocity projection."""
    if space == "M_s":
        if sp.issparse(adata.layers.get("M_s", None)):
            return adata.layers["M_s"].toarray().astype(np.float32), "M_s"
        if "M_s" in adata.layers:
            return np.asarray(adata.layers["M_s"], dtype=np.float32), "M_s"
        space = "X"

    if space == "raw_csv":
        if not cfg or not cfg.get("expr_csv"):
            raise ValueError("--scgpt-expr-space raw_csv requires expr_csv in datasets json.")
        expr_df = pd.read_csv(cfg["expr_csv"], index_col=0)
        expr_df = expr_df.reindex(index=[g for g in adata.var_names.astype(str)])
        expr_df = expr_df[adata.obs_names.astype(str)]
        return expr_df.values.T.astype(np.float32), "raw_csv"

    if space == "bins":
        if cfg and cfg.get("expr_csv"):
            expr_df = pd.read_csv(cfg["expr_csv"], index_col=0)
            expr_df = expr_df.reindex(index=[g for g in adata.var_names.astype(str)])
            expr_df = expr_df[adata.obs_names.astype(str)]
            raw = expr_df.values.T.astype(np.float32)
        else:
            raw = np.asarray(
                adata.X if not sp.issparse(adata.X) else adata.X.toarray(),
                dtype=np.float32,
            )
        return bin_expr_to_0_50(raw, do_log1p=False), "scGPT bins (0-50)"

    # X / fallback
    if sp.issparse(adata.X):
        return adata.X.toarray().astype(np.float32), "X"
    return np.asarray(adata.X, dtype=np.float32), "X"


def maybe_flip_velocity_sign(
    embedding: np.ndarray,
    velocity_emb: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, bool, float]:
    cos = time_alignment_cos(embedding, velocity_emb, labels)
    if ARGS.auto_flip_sign and not np.isnan(cos) and cos < 0:
        return (-velocity_emb).astype(np.float32), True, cos
    return velocity_emb, False, cos


def align_adata(adata: sc.AnnData, cells: pd.Index, genes: list) -> sc.AnnData:
    adata = adata[cells].copy()
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names.astype(str))}
    gi = [gene_to_idx[g] for g in genes if g in gene_to_idx]
    genes_ok = [g for g in genes if g in gene_to_idx]
    return adata[:, gi].copy(), genes_ok


def get_velocity_matrix(adata: sc.AnnData, layer: str) -> np.ndarray:
    V = adata.layers[layer]
    if sp.issparse(V):
        V = V.toarray()
    return np.nan_to_num(np.asarray(V, dtype=np.float32), nan=0.0)


def load_scgpt_velocity(cells: pd.Index, genes: list) -> np.ndarray:
    d = np.load(ARGS.velocity_npz)
    vel = d["vel_cell"].astype(np.float32)
    if vel.shape[0] != len(cells):
        raise ValueError(f"npz cells {vel.shape[0]} != export cells {len(cells)}")
    if vel.shape[1] != len(genes):
        raise ValueError(f"npz genes {vel.shape[1]} != export genes {len(genes)}")
    return vel


def embed_velocity(
    embedding: np.ndarray,
    expression: np.ndarray,
    velocity_gene: np.ndarray,
    adata: sc.AnnData,
    method: str,
    basis: str,
    n_neighbors: int,
) -> Tuple[np.ndarray, str]:
    """Return UMAP velocity vectors and a short label describing the method."""
    if method == "unified_neighbor":
        return (
            project_velocity_to_embedding(embedding, expression, velocity_gene, n_neighbors),
            "neighbor projection",
        )

    vel_key = f"velocity_{basis}"
    if vel_key not in adata.obsm:
        print(f"  [WARN] {vel_key} missing; falling back to neighbor projection.")
        return (
            project_velocity_to_embedding(embedding, expression, velocity_gene, n_neighbors),
            "neighbor projection (fallback)",
        )

    vel = np.asarray(adata.obsm[vel_key], dtype=np.float32).copy()
    if method == "dynamo_precomputed_flip":
        vel = -vel
        label = "Dynamo precomputed UMAP (sign flipped)"
    else:
        label = "Dynamo precomputed UMAP"
    return vel, label


def time_alignment_cos(
    embedding: np.ndarray,
    velocity_emb: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Cosine between mean velocity and centroid direction early->late."""
    uniq = np.sort(np.unique(labels))
    if len(uniq) < 2:
        return float("nan")
    c0 = embedding[labels == uniq[0]].mean(axis=0)
    c1 = embedding[labels == uniq[-1]].mean(axis=0)
    direction = c1 - c0
    mean_v = velocity_emb.mean(axis=0)
    nd = np.linalg.norm(direction)
    nv = np.linalg.norm(mean_v)
    if nd < 1e-8 or nv < 1e-8:
        return float("nan")
    return float(np.dot(mean_v, direction) / (nd * nv))


def project_velocity_to_embedding(
    embedding: np.ndarray,
    expression: np.ndarray,
    velocity: np.ndarray,
    n_neighbors: int = 30,
) -> np.ndarray:
    """scVelo-style projection: gene-space velocity -> embedding arrows."""
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
            cos = np.dot(vi, dx) / (norm_v * norm_dx)
            cos = max(float(cos), 0.0)
            acc += cos * (embedding[j] - embedding[i])
        vel_emb[i] = acc.astype(np.float32)
    return vel_emb


def build_velocity_grid(
    embedding: np.ndarray,
    velocity_emb: np.ndarray,
    grid_size: int = 50,
    smooth: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y = embedding[:, 0], embedding[:, 1]
    pad = 0.02
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    dx = (xmax - xmin) * pad
    dy = (ymax - ymin) * pad
    gx = np.linspace(xmin - dx, xmax + dx, grid_size)
    gy = np.linspace(ymin - dy, ymax + dy, grid_size)
    gx_mesh, gy_mesh = np.meshgrid(gx, gy)

    vx = griddata((x, y), velocity_emb[:, 0], (gx_mesh, gy_mesh), method="linear")
    vy = griddata((x, y), velocity_emb[:, 1], (gx_mesh, gy_mesh), method="linear")
    vx = np.nan_to_num(vx, nan=0.0)
    vy = np.nan_to_num(vy, nan=0.0)

    if smooth > 0:
        try:
            from scipy.ndimage import gaussian_filter

            vx = gaussian_filter(vx, sigma=smooth)
            vy = gaussian_filter(vy, sigma=smooth)
        except Exception:
            pass

    speed = np.sqrt(vx ** 2 + vy ** 2)
    cutoff = np.percentile(speed[speed > 0], 5) if np.any(speed > 0) else 0.0
    mask = speed < cutoff
    vx[mask] = 0.0
    vy[mask] = 0.0
    return gx_mesh, gy_mesh, vx, vy


def velocity_confidence(velocity_emb: np.ndarray) -> float:
    speed = np.linalg.norm(velocity_emb, axis=1)
    valid = speed > 1e-8
    if valid.sum() < 3:
        return float("nan")
    dirs = velocity_emb[valid] / speed[valid][:, None]
    nn = NearestNeighbors(n_neighbors=min(15, valid.sum()), n_jobs=-1)
    nn.fit(velocity_emb[valid])
    idx = nn.kneighbors(velocity_emb[valid], return_distance=False)
    conf = []
    for i, neigh in enumerate(idx):
        d0 = dirs[i]
        cos = [np.dot(d0, dirs[j]) for j in neigh[1:]]
        if cos:
            conf.append(np.mean(cos))
    return float(np.mean(conf))


def time_correlation(embedding: np.ndarray, velocity_emb: np.ndarray, time: np.ndarray) -> float:
    """Correlation between velocity flow and time gradient on embedding."""
    nn = NearestNeighbors(n_neighbors=min(30, len(time)), n_jobs=-1)
    nn.fit(embedding)
    knn = nn.kneighbors(embedding, return_distance=False)
    time_grad = np.zeros(len(time), dtype=np.float64)
    for i, neigh in enumerate(knn):
        dt = time[neigh] - time[i]
        de = embedding[neigh] - embedding[i]
        den = np.linalg.norm(de, axis=1)
        den[den == 0] = 1.0
        time_grad[i] = np.mean(dt / den)
    speed = np.linalg.norm(velocity_emb, axis=1)
    valid = speed > 1e-8
    if valid.sum() < 3:
        return float("nan")
    flow = []
    align = []
    for i in np.where(valid)[0]:
        flow.append(speed[i])
        tg = time_grad[i]
        if tg == 0:
            continue
        align.append(np.sign(velocity_emb[i].dot(embedding[knn[i][1]] - embedding[i])) * tg)
    if len(align) < 3:
        return float("nan")
    r, _ = spearmanr(time.astype(float), time_grad)
    return float(r)


def cross_boundary_correlation(
    embedding: np.ndarray,
    velocity_emb: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """CBC between adjacent label groups in embedding space."""
    uniq = np.sort(np.unique(labels))
    out = {}
    for a, b in zip(uniq[:-1], uniq[1:]):
        ma = labels == a
        mb = labels == b
        ca = embedding[ma].mean(axis=0)
        cb = embedding[mb].mean(axis=0)
        direction = cb - ca
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            continue
        direction = direction / norm
        va = velocity_emb[ma]
        sa = np.linalg.norm(va, axis=1)
        valid = sa > 1e-8
        if valid.sum() < 3:
            continue
        cos = (va[valid] @ direction) / sa[valid]
        out[f"{a}->{b}"] = float(np.mean(cos))
    return out


def plot_stream_panel(
    ax: plt.Axes,
    embedding: np.ndarray,
    color_values: np.ndarray,
    velocity_emb: np.ndarray,
    title: str,
    color_label: str = "Latent time",
    grid_size: int = 50,
) -> Dict[str, float]:
    gx, gy, vx, vy = build_velocity_grid(embedding, velocity_emb, grid_size=grid_size)

    sca = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=color_values,
        s=ARGS.point_size,
        cmap="Spectral_r",
        linewidths=0,
        alpha=0.95,
        rasterized=True,
        zorder=1,
    )
    ax.streamplot(
        gx,
        gy,
        vx,
        vy,
        color=(0.33, 0.33, 0.33, ARGS.stream_alpha),
        density=ARGS.stream_density,
        linewidth=ARGS.stream_lw,
        arrowsize=0.8,
        maxlength=0.12,
        zorder=2,
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cbar = plt.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(color_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    metrics = {
        "velocity_conf": velocity_confidence(velocity_emb),
        "time_corr": time_correlation(embedding, velocity_emb, color_values),
        "time_align_cos": time_alignment_cos(embedding, velocity_emb, color_values),
    }
    metrics.update({f"CBC_{k}": v for k, v in cross_boundary_correlation(embedding, velocity_emb, color_values).items()})

    txt = (
        f"Velocity conf.: {metrics['velocity_conf']:.2f}\n"
        f"Time corr.: {metrics['time_corr']:.2f}\n"
        f"Time align cos: {metrics['time_align_cos']:.2f}"
    )
    ax.text(
        0.02,
        0.02,
        txt,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="none"),
    )
    return metrics


def main():
    outdir = Path(ARGS.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(ARGS.h5ad)
    adata.obs_names_make_unique()
    cells, genes, cfg = load_cells_and_genes()
    if cells is not None:
        adata, genes = align_adata(adata, cells, genes)

    basis_key = f"X_{ARGS.basis}"
    if basis_key not in adata.obsm:
        raise KeyError(f"{basis_key} not in adata.obsm")
    embedding = np.asarray(adata.obsm[basis_key], dtype=np.float32)

    color_key = ARGS.color_key
    if color_key == "cell_type" and cfg and cfg.get("cell_type_csv"):
        ct_df = pd.read_csv(cfg["cell_type_csv"]).set_index("cell")
        ct_df.index = ct_df.index.astype(str)
        color_values = np.array([ct_df.loc[c, "cell_type"] if c in ct_df.index else "Unknown" for c in adata.obs_names])
        color_values = pd.Categorical(color_values).codes.astype(float)
        color_label = "Cell type"
    elif color_key == "time" and cfg and cfg.get("time_csv"):
        tm_df = pd.read_csv(cfg["time_csv"]).set_index("cell")
        tm_df.index = tm_df.index.astype(str)
        color_values = np.array([float(tm_df.loc[c, "time"]) if c in tm_df.index else np.nan for c in adata.obs_names])
        color_label = "Time"
    else:
        if color_key not in adata.obs.columns:
            raise KeyError(f"{color_key} not in adata.obs")
        color_values = adata.obs[color_key].to_numpy()
        if not np.issubdtype(color_values.dtype, np.number):
            color_values = pd.Categorical(color_values).codes.astype(float)
        color_label = color_key.replace("_", " ").title()

    expr = np.asarray(adata.X if not sp.issparse(adata.X) else adata.X.toarray(), dtype=np.float32)
    if sp.issparse(adata.layers.get("M_s", None)):
        expr = adata.layers["M_s"].toarray()
    elif "M_s" in adata.layers:
        expr = np.asarray(adata.layers["M_s"], dtype=np.float32)
    expr_dynamo = expr

    # Dynamo velocity (gene space) — project with M_s expression
    v_dynamo = get_velocity_matrix(adata, ARGS.velocity_layer)
    vel_dynamo_emb, dynamo_proj_label = embed_velocity(
        embedding,
        expr_dynamo,
        v_dynamo,
        adata,
        ARGS.projection if ARGS.projection.startswith("dynamo_precomputed") else "unified_neighbor",
        ARGS.basis,
        ARGS.n_neighbors,
    )
    panel_specs = [(f"Dynamo RNA velocity\n({dynamo_proj_label})", vel_dynamo_emb, "dynamo")]
    projection_labels = {"dynamo": dynamo_proj_label}
    metrics_all = {"projection_mode": ARGS.projection}

    # scGPT velocity — project in bin space (matches velocity units from MLM / transition head)
    if ARGS.velocity_npz:
        v_scgpt = load_scgpt_velocity(cells if cells is not None else adata.obs_names, genes)
        expr_scgpt, scgpt_expr_label = load_expression_matrix(adata, cfg, ARGS.scgpt_expr_space)
        vel_scgpt_emb = project_velocity_to_embedding(
            embedding, expr_scgpt, v_scgpt, ARGS.n_neighbors
        )
        vel_scgpt_emb, flipped, cos_before = maybe_flip_velocity_sign(
            embedding, vel_scgpt_emb, color_values.astype(float)
        )
        scgpt_proj_label = f"neighbor on {scgpt_expr_label}"
        if flipped:
            scgpt_proj_label += ", sign corrected"
        panel_specs.append(
            (f"scGPT velocity (zero-shot)\n({scgpt_proj_label})", vel_scgpt_emb, "scgpt")
        )
        projection_labels["scgpt"] = scgpt_proj_label
        metrics_all["scgpt_projection"] = {
            "expr_space": scgpt_expr_label,
            "sign_flipped": flipped,
            "time_align_cos_before_flip": cos_before,
        }
    metrics_all["projection_labels"] = projection_labels

    ncols = len(panel_specs)
    fig, axes = plt.subplots(1, ncols, figsize=(5.8 * ncols, 5.2), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    for ax, (title, vel_emb, key) in zip(axes, panel_specs):
        metrics_all[key] = plot_stream_panel(
            ax,
            embedding,
            color_values.astype(float),
            vel_emb,
            title=title,
            color_label=color_label,
            grid_size=ARGS.grid_size,
        )

    fig.savefig(outdir / f"{ARGS.prefix}_combined.pdf", dpi=ARGS.dpi, bbox_inches="tight")
    fig.savefig(outdir / f"{ARGS.prefix}_combined.png", dpi=ARGS.dpi, bbox_inches="tight")
    plt.close(fig)

    for title, vel_emb, key in panel_specs:
        fig, ax = plt.subplots(figsize=(5.8, 5.2), constrained_layout=True)
        plot_stream_panel(
            ax,
            embedding,
            color_values.astype(float),
            vel_emb,
            title=title,
            color_label=color_label,
            grid_size=ARGS.grid_size,
        )
        slug = key.replace(" ", "_")
        fig.savefig(outdir / f"{ARGS.prefix}_{slug}.pdf", dpi=ARGS.dpi, bbox_inches="tight")
        fig.savefig(outdir / f"{ARGS.prefix}_{slug}.png", dpi=ARGS.dpi, bbox_inches="tight")
        plt.close(fig)

    with open(outdir / f"{ARGS.prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, indent=2)

    print(f"[OK] Saved stream plots -> {outdir}  (projection={ARGS.projection})")
    skip = {"projection_mode", "projection_labels", "scgpt_projection"}
    for k, m in metrics_all.items():
        if k in skip or not isinstance(m, dict) or "velocity_conf" not in m:
            continue
        print(
            f"  {k}: conf={m['velocity_conf']:.3f}, "
            f"time_corr={m['time_corr']:.3f}, "
            f"time_align_cos={m['time_align_cos']:.3f}"
        )


if __name__ == "__main__":
    main()
