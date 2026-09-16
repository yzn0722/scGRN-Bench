#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
veloBench-aligned velocity metrics: CBDir, ICCoh, Velocity Consistency.

Uses ``unitvelo.evaluate()`` (veloAE / veloBench reference) plus the modified
``velocity_confidence`` from veloBench (cosine, scope_key=clusters).

Example (Dynamo velocity on hematopoiesis):
  conda activate dynamo-env
  cd /mnt/10T/yzn/scGRN-Bench

  python src/GRN_inferance/dyn/eval_velocity_benchmark_metrics.py \\
    --h5ad /mnt/10T/yzn/dynamo-release/results/hematopoiesis_raw/hematopoiesis_processed.h5ad \\
    --datasets-json data/dynamo_export/datasets_dynamo_vel1524_only.json \\
    --dataset hematopoiesis_dyn_vel1524 \\
    --velocity-source dynamo \\
    --edge-preset hematopoiesis_local \\
    --outdir outputs/dynamo_fm_velocity/benchmark_metrics/dynamo_hematopoiesis_local

Example (scGPT zero-shot, requires velocity_field.npz from scgpt_dyn.py):
  python src/GRN_inferance/dyn/eval_velocity_benchmark_metrics.py \\
    --h5ad /mnt/10T/yzn/dynamo-release/results/hematopoiesis_raw/hematopoiesis_processed.h5ad \\
    --datasets-json data/dynamo_export/datasets_dynamo_vel1524_only.json \\
    --dataset hematopoiesis_dyn_vel1524 \\
    --velocity-source scgpt \\
    --velocity-npz outputs/dynamo_fm_velocity/scgpt_hematopoiesis_vel1524_zero_shot/hematopoiesis_dyn_vel1524/velocity_field.npz \\
    --edge-preset hematopoiesis_local \\
    --outdir outputs/dynamo_fm_velocity/benchmark_metrics/scgpt_zero_shot_hematopoiesis_local
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

from sklearn.metrics.pairwise import cosine_similarity

try:
    from unitvelo.eval_utils import inner_cluster_coh, keep_type
except ImportError as exc:
    raise ImportError(
        "unitvelo is required. Install in dynamo-env: pip install unitvelo"
    ) from exc

from scvelo.core import l2_norm


def cross_boundary_correctness_fixed(
    adata: sc.AnnData,
    k_cluster: str,
    k_velocity: str,
    cluster_edges: Sequence[Tuple[str, str]],
    return_raw: bool = False,
    x_emb: str = "X_umap",
) -> Tuple[dict, float]:
    """veloBench CBDir with unitvelo bugfix (x_emb array vs string compare)."""
    scores: Dict[Tuple[str, str], float] = {}
    all_scores: Dict[Tuple[str, str], list] = {}

    x_points_all = adata.obsm[x_emb]
    preferred = f"{k_velocity}_umap"
    if preferred in adata.obsm:
        v_emb_all = adata.obsm[preferred]
    else:
        umap_keys = [k for k in adata.obsm if k.startswith(f"{k_velocity}_")]
        if not umap_keys:
            raise KeyError(
                f"No obsm key '{preferred}' or '{k_velocity}_*'. Have: {list(adata.obsm.keys())}"
            )
        v_emb_all = adata.obsm[umap_keys[0]]

    for u, v in cluster_edges:
        sel = adata.obs[k_cluster].astype(str) == u
        if not np.any(sel):
            scores[(u, v)] = float("nan")
            all_scores[(u, v)] = []
            continue
        nbs = adata.uns["neighbors"]["indices"][sel]
        boundary_nodes = map(lambda nodes: keep_type(adata, nodes, v, k_cluster), nbs)
        x_points = x_points_all[sel]
        x_velocities = v_emb_all[sel]

        type_score = []
        for x_pos, x_vel, nodes in zip(x_points, x_velocities, boundary_nodes):
            if len(nodes) == 0:
                continue
            position_dif = x_points_all[nodes] - x_pos
            dir_scores = cosine_similarity(position_dif, x_vel.reshape(1, -1)).flatten()
            type_score.append(float(np.mean(dir_scores)))

        scores[(u, v)] = float(np.mean(type_score)) if type_score else float("nan")
        all_scores[(u, v)] = type_score

    if return_raw:
        return all_scores
    return scores, float(np.nanmean(list(scores.values())))


def evaluate_velobench(
    adata: sc.AnnData,
    cluster_edges: Sequence[Tuple[str, str]],
    k_cluster: str,
    k_velocity: str,
    x_emb: str = "X_umap",
    verbose: bool = True,
) -> dict:
    crs_bdr_crc = cross_boundary_correctness_fixed(
        adata, k_cluster, k_velocity, cluster_edges, return_raw=True, x_emb=x_emb
    )
    ic_coh = inner_cluster_coh(adata, k_cluster, k_velocity, return_raw=True)

    cb_mean = float(np.nanmean([np.mean(v) if len(v) else np.nan for v in crs_bdr_crc.values()]))
    ic_mean = float(np.nanmean([np.mean(v) if len(v) else np.nan for v in ic_coh.values()]))

    if verbose:
        print("# Cross-Boundary Direction Correctness (A->B)")
        for edge, vals in crs_bdr_crc.items():
            print(f"  {edge[0]} -> {edge[1]}: {np.mean(vals) if len(vals) else float('nan'):.4f}")
        print(f"Total Mean: {cb_mean:.4f}")
        print("# In-cluster Coherence")
        for cluster, vals in ic_coh.items():
            print(f"  {cluster}: {np.mean(vals) if len(vals) else float('nan'):.4f}")
        print(f"Total Mean: {ic_mean:.4f}")

    return {
        "Cross-Boundary Direction Correctness (A->B)": crs_bdr_crc,
        "In-cluster Coherence": ic_coh,
    }


DEFAULT_EDGES_JSON = (
    Path(__file__).resolve().parents[3]
    / "data/dynamo_export/hematopoiesis_dyn_vel1524/benchmark_cluster_edges.json"
)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_data_path(path_str: str, root: Optional[Path] = None) -> Path:
    """Resolve export paths relative to scGRN-Bench root (or cwd fallback)."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    root = root or PROJECT_ROOT
    for base in (Path.cwd(), root):
        candidate = base / p
        if candidate.exists():
            return candidate
    return root / p


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="veloBench-aligned CBDir / ICCoh / VC metrics")
    p.add_argument("--h5ad", required=True, type=str)
    p.add_argument("--datasets-json", default="", type=str)
    p.add_argument("--dataset", default="", type=str)
    p.add_argument(
        "--velocity-source",
        required=True,
        choices=["dynamo", "scgpt", "scfoundation", "layer"],
        help="Where to read per-cell gene-space velocity.",
    )
    p.add_argument(
        "--velocity-layer",
        default="velocity_alpha_minus_gamma_s",
        type=str,
        help="Dynamo / layer source key in adata.layers.",
    )
    p.add_argument("--velocity-npz", default="", type=str, help="scGPT velocity_field.npz")
    p.add_argument("--vkey", default="velocity", type=str, help="Layer/obsm prefix for unitvelo.evaluate")
    p.add_argument("--cluster-col", default="clusters", type=str, help="obs column for clusters")
    p.add_argument(
        "--x-emb",
        default="X_umap",
        type=str,
        help="Low-dim embedding for CBDir (e.g. X_umap, X_tsne).",
    )
    p.add_argument(
        "--edge-preset",
        default="",
        type=str,
        help=f"Key in benchmark_cluster_edges.json (default file: {DEFAULT_EDGES_JSON})",
    )
    p.add_argument("--edges-json", default="", type=str, help="JSON file with edge presets or raw edge list")
    p.add_argument("--ground-truth-pickle", default="", type=str, help="veloBench-style pickle of (A,B) tuples")
    p.add_argument(
        "--projection",
        default="auto",
        choices=["auto", "dynamo", "precomputed", "neighbor"],
        help="How to obtain UMAP velocity for CBDir.",
    )
    p.add_argument("--n-neighbors", type=int, default=30)
    p.add_argument("--n-pcs", type=int, default=30)
    p.add_argument(
        "--expr-space",
        default="auto",
        choices=["auto", "M_s", "M_t", "bins", "X"],
        help="Expression space for neighbor projection (scGPT: bins).",
    )
    p.add_argument("--method-name", default="", type=str, help="Label in output JSON")
    p.add_argument("--outdir", required=True, type=str)
    p.add_argument("--save-h5ad", action="store_true", help="Write AnnData_Forscore.h5ad")
    return p.parse_args()


def load_dataset_cfg(datasets_json: str, dataset: str) -> Optional[dict]:
    if not datasets_json or not dataset:
        return None
    with open(datasets_json, encoding="utf-8") as f:
        return json.load(f)[dataset]


def load_export_cells_genes(cfg: dict) -> Tuple[pd.Index, List[str]]:
    root = Path(cfg.get("_root", PROJECT_ROOT))
    expr_path = resolve_data_path(cfg["expr_csv"], root)
    pt_path = resolve_data_path(cfg["pt_csv"], root)
    expr = pd.read_csv(expr_path, index_col=0)
    pt_df = pd.read_csv(pt_path)
    pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"}).set_index("cell")
    common = expr.columns.astype(str).intersection(pt_df.index.astype(str))
    genes = expr.index.astype(str).tolist()
    return common, genes


def bin_expr_to_0_50(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0)
    x = np.clip(x, 0, None)
    vmax = max(float(np.percentile(x, 99.5)), 1e-6)
    return np.clip(x / vmax * 50.0, 0, 50).astype(np.float32)


def layer_to_array(adata: sc.AnnData, key: str) -> np.ndarray:
    if key not in adata.layers:
        raise KeyError(f"Layer '{key}' not in adata.layers. Available: {list(adata.layers.keys())[:20]}")
    mat = adata.layers[key]
    if sp.issparse(mat):
        mat = mat.toarray()
    return np.nan_to_num(np.asarray(mat, dtype=np.float32), nan=0.0)


def align_adata_subset(adata: sc.AnnData, cells: Sequence[str], genes: Sequence[str]) -> sc.AnnData:
    cells = [c for c in cells if c in adata.obs_names]
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names.astype(str))}
    gi = [gene_to_idx[g] for g in genes if g in gene_to_idx]
    genes_ok = [g for g in genes if g in gene_to_idx]
    return adata[cells, gi].copy(), genes_ok


def ensure_clusters(adata: sc.AnnData, cluster_col: str, cfg: Optional[dict]) -> str:
    if cluster_col in adata.obs.columns:
        adata.obs["clusters"] = adata.obs[cluster_col].astype(str)
        return "clusters"
    if cfg and cfg.get("cell_type_csv"):
        ct_path = resolve_data_path(cfg["cell_type_csv"], Path(cfg.get("_root", PROJECT_ROOT)))
        ct_df = pd.read_csv(ct_path).set_index("cell")
        ct_df.index = ct_df.index.astype(str)
        adata.obs["clusters"] = [
            str(ct_df.loc[c, "cell_type"]) if c in ct_df.index else "Unknown"
            for c in adata.obs_names.astype(str)
        ]
        return "clusters"
    raise KeyError(
        f"Cluster column '{cluster_col}' missing and no cell_type_csv in dataset config."
    )


def ensure_neighbors(adata: sc.AnnData, n_neighbors: int, n_pcs: int) -> None:
    need = (
        "neighbors" not in adata.uns
        or "indices" not in adata.uns["neighbors"]
        or adata.uns["neighbors"]["indices"].shape[0] != adata.n_obs
    )
    if need:
        if "X_pca" not in adata.obsm:
            sc.pp.pca(adata, n_comps=min(n_pcs, adata.n_vars - 1, adata.n_obs - 1))
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    params = adata.uns.setdefault("neighbors", {}).setdefault("params", {})
    params["n_neighbors"] = n_neighbors
    params["n_pcs"] = n_pcs


def get_expression_matrix(adata: sc.AnnData, cfg: Optional[dict], space: str) -> Tuple[np.ndarray, str]:
    if space == "bins":
        if cfg and cfg.get("expr_csv"):
            expr_path = resolve_data_path(cfg["expr_csv"], Path(cfg.get("_root", PROJECT_ROOT)))
            expr_df = pd.read_csv(expr_path, index_col=0)
            expr_df = expr_df.reindex(index=adata.var_names.astype(str))
            expr_df = expr_df[adata.obs_names.astype(str)]
            return expr_df.values.T.astype(np.float32), "export_csv_bins"
        raw = layer_to_array(adata, "M_s") if "M_s" in adata.layers else np.asarray(adata.X)
        return bin_expr_to_0_50(raw), "M_s_bins"

    if space in adata.layers:
        return layer_to_array(adata, space), space
    if sp.issparse(adata.X):
        return adata.X.toarray().astype(np.float32), "X"
    return np.asarray(adata.X, dtype=np.float32), "X"


def project_velocity_neighbor(
    embedding: np.ndarray,
    expression: np.ndarray,
    velocity: np.ndarray,
    n_neighbors: int,
) -> np.ndarray:
    n = embedding.shape[0]
    k = min(n_neighbors, n)
    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
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


def project_velocity_dynamo(
    adata: sc.AnnData,
    velocity: np.ndarray,
    expression: np.ndarray,
    basis: str = "umap",
    method: str = "cosine",
    velocity_key: str = "velocity_umap",
) -> np.ndarray:
    import dynamo as dyn

    work = adata.copy()
    work.layers["_eval_expr_tmp"] = expression.astype(np.float32)
    work.layers["_eval_velocity_tmp"] = velocity.astype(np.float32)
    work.var["use_for_transition"] = True
    dyn.tl.cell_velocities(
        work,
        enforce=True,
        X=work.layers["_eval_expr_tmp"],
        V=work.layers["_eval_velocity_tmp"],
        method=method,
        basis=basis,
        add_velocity_key=velocity_key,
    )
    if velocity_key not in work.obsm:
        raise KeyError(f"Dynamo cell_velocities did not create obsm['{velocity_key}']")
    return np.asarray(work.obsm[velocity_key], dtype=np.float32)


def maybe_flip_by_time(embedding: np.ndarray, velocity_emb: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, float]:
    uniq = np.sort(np.unique(labels[~pd.isna(labels)]))
    if len(uniq) < 2:
        return velocity_emb, float("nan")
    c0 = embedding[labels == uniq[0]].mean(axis=0)
    c1 = embedding[labels == uniq[-1]].mean(axis=0)
    direction = c1 - c0
    mean_v = velocity_emb.mean(axis=0)
    nd, nv = np.linalg.norm(direction), np.linalg.norm(mean_v)
    if nd < 1e-8 or nv < 1e-8:
        return velocity_emb, float("nan")
    cos = float(np.dot(mean_v, direction) / (nd * nv))
    if cos < 0:
        return (-velocity_emb).astype(np.float32), cos
    return velocity_emb, cos


def load_cluster_edges(args: argparse.Namespace) -> Tuple[List[Tuple[str, str]], str]:
    if args.ground_truth_pickle:
        with open(args.ground_truth_pickle, "rb") as f:
            edges = pickle.load(f)
        return [(str(a), str(b)) for a, b in edges], f"pickle:{args.ground_truth_pickle}"

    edges_path = Path(args.edges_json) if args.edges_json else DEFAULT_EDGES_JSON
    if not edges_path.exists():
        raise FileNotFoundError(f"Edges config not found: {edges_path}")

    with open(edges_path, encoding="utf-8") as f:
        payload = json.load(f)

    if args.edge_preset:
        if args.edge_preset not in payload:
            raise KeyError(f"Preset '{args.edge_preset}' not in {edges_path}")
        edges = payload[args.edge_preset]["edges"]
        desc = payload[args.edge_preset].get("description", args.edge_preset)
        return [(str(a), str(b)) for a, b in edges], desc

    if isinstance(payload, list):
        return [(str(a), str(b)) for a, b in payload], "custom_list"

    raise ValueError("Provide --edge-preset, --ground-truth-pickle, or an edges JSON list.")


def filter_valid_edges(
    edges: Sequence[Tuple[str, str]],
    clusters: Sequence[str],
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    present = set(map(str, clusters))
    valid, skipped = [], []
    for a, b in edges:
        if a in present and b in present:
            valid.append((a, b))
        else:
            skipped.append((a, b))
    return valid, skipped


def load_gene_velocity(
    adata: sc.AnnData,
    args: argparse.Namespace,
    cfg: Optional[dict],
    genes_ok: List[str],
) -> Tuple[np.ndarray, str]:
    if args.velocity_source == "layer":
        vel = layer_to_array(adata, args.velocity_layer)
        return vel, f"layer:{args.velocity_layer}"

    if args.velocity_source == "dynamo":
        vel = layer_to_array(adata, args.velocity_layer)
        return vel, f"dynamo:{args.velocity_layer}"

    if args.velocity_source in ("scgpt", "scfoundation"):
        if not args.velocity_npz:
            raise ValueError(f"--velocity-npz required for --velocity-source {args.velocity_source}")
        d = np.load(args.velocity_npz)
        vel = d["vel_cell"].astype(np.float32)
        if vel.shape[0] != adata.n_obs:
            raise ValueError(f"npz cells {vel.shape[0]} != adata {adata.n_obs}")
        if vel.shape[1] != adata.n_vars:
            raise ValueError(f"npz genes {vel.shape[1]} != adata {adata.n_vars}")
        return vel, f"{args.velocity_source}:velocity_field.npz"

    raise ValueError(args.velocity_source)


def setup_umap_velocity(
    adata: sc.AnnData,
    velocity: np.ndarray,
    args: argparse.Namespace,
    cfg: Optional[dict],
) -> Tuple[np.ndarray, str]:
    emb_key = args.x_emb
    basis = emb_key.removeprefix("X_")
    if emb_key not in adata.obsm:
        sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, n_pcs=args.n_pcs)
        if basis == "umap":
            sc.tl.umap(adata)
        elif basis == "tsne":
            sc.tl.tsne(adata)
        else:
            raise KeyError(f"Embedding {emb_key!r} missing and basis {basis!r} is unsupported.")

    projection = args.projection
    if projection == "auto":
        if args.velocity_source == "dynamo":
            projection = "precomputed" if f"velocity_{basis}" in adata.obsm else "dynamo"
        else:
            projection = "neighbor"

    if projection == "precomputed":
        key = f"velocity_{basis}"
        if key not in adata.obsm:
            raise KeyError(f"--projection precomputed requires obsm['{key}']")
        return np.asarray(adata.obsm[key], dtype=np.float32), f"precomputed:{key}"

    if projection == "dynamo":
        if args.velocity_source in ("scgpt", "scfoundation"):
            expr_space = "bins"
            expr, expr_label = get_expression_matrix(adata, cfg, expr_space)
        else:
            expr_space = "M_t" if "M_t" in adata.layers else "M_s"
            expr, expr_label = get_expression_matrix(adata, cfg, expr_space)
        vel_key = f"{args.vkey}_{basis}"
        vel_umap = project_velocity_dynamo(
            adata,
            velocity,
            expression=expr,
            basis=basis,
            velocity_key=vel_key,
        )
        return vel_umap, f"dynamo_cell_velocities:{expr_label}->{vel_key}"

    expr_space = args.expr_space
    if expr_space == "auto":
        expr_space = "bins" if args.velocity_source in ("scgpt", "scfoundation") else "M_s"
    expr, expr_label = get_expression_matrix(adata, cfg, expr_space)
    vel_umap = project_velocity_neighbor(
        adata.obsm[emb_key],
        expr,
        velocity,
        args.n_neighbors,
    )

    time_labels = None
    if "time" in adata.obs.columns:
        time_labels = adata.obs["time"].to_numpy()
    elif cfg and cfg.get("time_csv"):
        tm_path = resolve_data_path(cfg["time_csv"], Path(cfg.get("_root", PROJECT_ROOT)))
        tm_df = pd.read_csv(tm_path).set_index("cell")
        tm_df.index = tm_df.index.astype(str)
        time_labels = np.array(
            [float(tm_df.loc[c, "time"]) if c in tm_df.index else np.nan for c in adata.obs_names.astype(str)]
        )

    if time_labels is not None:
        vel_umap, align_cos = maybe_flip_by_time(adata.obsm[emb_key], vel_umap, time_labels)
        return vel_umap, f"neighbor:{expr_label};time_align_cos={align_cos:.4f}"

    return vel_umap, f"neighbor:{expr_label}"


def velocity_confidence_velobench(
    adata: sc.AnnData,
    vkey: str,
    method: str = "cosine",
    scope_key: str = "clusters",
) -> np.ndarray:
    """veloBench variant: cosine confidence within scope_key cluster."""
    if vkey not in adata.layers:
        raise ValueError(f"Layer '{vkey}' missing for velocity_confidence")

    mat = adata.layers[vkey]
    if sp.issparse(mat):
        V = mat.toarray().astype(np.float64)
    else:
        V = np.asarray(mat, dtype=np.float64)
    V = np.nan_to_num(V, nan=0.0)
    tmp_filter = np.isfinite(np.sum(V, axis=0))
    V = V[:, tmp_filter]

    V_norm = l2_norm(V, axis=1)
    if method == "cosine":
        denom = V_norm[:, None]
        denom[denom < 1e-12] = 1.0
        V = V / denom

    scope = adata.obs[scope_key].astype(str).to_numpy()
    cache: Dict[str, np.ndarray] = {}
    R = np.zeros(adata.n_obs, dtype=np.float64)
    for i in range(adata.n_obs):
        label = scope[i]
        if label not in cache:
            cache[label] = V[scope == label].mean(axis=0)
        R[i] = float(np.inner(V[i], cache[label]))
    adata.obs[f"{vkey}_length"] = V_norm.round(2)
    adata.obs[f"velocity_confidence_{method}"] = R
    return R


def metrics_to_tables(metrics: dict, edge_desc: str) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    cb = metrics["Cross-Boundary Direction Correctness (A->B)"]
    ic = metrics["In-cluster Coherence"]

    cb_rows = []
    for edge, scores in cb.items():
        cb_rows.append(
            {
                "edge": f"{edge[0]}->{edge[1]}",
                "source": edge[0],
                "target": edge[1],
                "cbdir_mean": float(np.mean(scores)) if len(scores) else float("nan"),
                "n_boundary_cells_scored": len(scores),
            }
        )
    cb_df = pd.DataFrame(cb_rows)
    if not cb_df.empty:
        cb_df.loc[len(cb_df)] = {
            "edge": "MEAN",
            "source": "",
            "target": "",
            "cbdir_mean": float(cb_df["cbdir_mean"].mean()),
            "n_boundary_cells_scored": int(cb_df["n_boundary_cells_scored"].sum()),
        }

    ic_rows = []
    for cluster, scores in ic.items():
        ic_rows.append(
            {
                "cluster": cluster,
                "iccoh_mean": float(np.mean(scores)) if len(scores) else float("nan"),
                "n_cells_scored": len(scores),
            }
        )
    ic_df = pd.DataFrame(ic_rows)
    if not ic_df.empty:
        ic_df.loc[len(ic_df)] = {
            "cluster": "MEAN",
            "iccoh_mean": float(ic_df["iccoh_mean"].mean()),
            "n_cells_scored": int(ic_df["n_cells_scored"].sum()),
        }

    summary = {
        "edge_preset_description": edge_desc,
        "cbdir_mean": float(cb_df.loc[cb_df["edge"] == "MEAN", "cbdir_mean"].iloc[0])
        if not cb_df.empty
        else float("nan"),
        "iccoh_mean": float(ic_df.loc[ic_df["cluster"] == "MEAN", "iccoh_mean"].iloc[0])
        if not ic_df.empty
        else float("nan"),
    }
    return cb_df, ic_df, summary


def run_evaluation(args: argparse.Namespace) -> dict:
    cfg = load_dataset_cfg(args.datasets_json, args.dataset)
    if cfg is not None:
        cfg = dict(cfg)
        cfg["_root"] = str(PROJECT_ROOT)

    adata = sc.read_h5ad(args.h5ad)
    genes_ok: Optional[List[str]] = None
    if cfg is not None:
        cells, genes = load_export_cells_genes(cfg)
        adata, genes_ok = align_adata_subset(adata, cells, genes)

    cluster_key = ensure_clusters(adata, args.cluster_col, cfg)
    ensure_neighbors(adata, args.n_neighbors, args.n_pcs)

    emb_key = args.x_emb
    basis = emb_key.removeprefix("X_")
    if emb_key not in adata.obsm:
        if basis == "umap":
            sc.tl.umap(adata)
        elif basis == "tsne":
            sc.tl.tsne(adata)

    velocity, vel_source = load_gene_velocity(adata, args, cfg, genes_ok or [])
    vel_emb, proj_desc = setup_umap_velocity(adata, velocity, args, cfg)

    vkey = args.vkey
    adata.layers[vkey] = velocity.astype(np.float32)
    adata.obsm[f"{vkey}_{basis}"] = vel_emb.astype(np.float32)

    all_edges, edge_desc = load_cluster_edges(args)
    valid_edges, skipped_edges = filter_valid_edges(all_edges, adata.obs[cluster_key].astype(str).unique())

    if not valid_edges:
        raise ValueError(
            f"No valid cluster edges. Requested={all_edges}, skipped={skipped_edges}, "
            f"clusters={sorted(adata.obs[cluster_key].unique())}"
        )

    metrics = evaluate_velobench(
        adata, valid_edges, cluster_key, vkey, x_emb=emb_key, verbose=True
    )
    vc = velocity_confidence_velobench(adata, vkey=vkey, method="cosine", scope_key=cluster_key)
    vc_mean = float(np.nanmean(vc))

    cb_df, ic_df, summary = metrics_to_tables(metrics, edge_desc)
    summary.update(
        {
            "method": args.method_name or args.velocity_source,
            "velocity_source": vel_source,
            "x_emb": emb_key,
            "embedding_projection": proj_desc,
            "umap_projection": proj_desc,
            "vkey": vkey,
            "cluster_key": cluster_key,
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "n_neighbors": args.n_neighbors,
            "velocity_consistency_mean": vc_mean,
            "cluster_edges_requested": [f"{a}->{b}" for a, b in all_edges],
            "cluster_edges_evaluated": [f"{a}->{b}" for a, b in valid_edges],
            "cluster_edges_skipped": [f"{a}->{b}" for a, b in skipped_edges],
        }
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cb_df.to_csv(outdir / "CBDir_scores.csv", index=False)
    ic_df.to_csv(outdir / "ICCoh_scores.csv", index=False)
    pd.DataFrame({"cell": adata.obs_names.astype(str), "velocity_confidence_cosine": vc}).to_csv(
        outdir / "velocity_confidence_per_cell.csv",
        index=False,
    )

    cb_raw_rows = []
    for (src, tgt), scores in metrics["Cross-Boundary Direction Correctness (A->B)"].items():
        for sc_val in scores:
            cb_raw_rows.append(
                {"edge": f"{src}->{tgt}", "source": src, "target": tgt, "cbdir": float(sc_val)}
            )
    if cb_raw_rows:
        pd.DataFrame(cb_raw_rows).to_csv(outdir / "CBDir_raw_per_boundary_cell.csv", index=False)

    ic_raw_rows = []
    cluster_arr = adata.obs[cluster_key].astype(str).to_numpy()
    for cluster, scores in metrics["In-cluster Coherence"].items():
        sel = cluster_arr == str(cluster)
        cells = adata.obs_names.astype(str).to_numpy()[sel]
        for cell, sc_val in zip(cells, scores):
            ic_raw_rows.append({"cluster": cluster, "cell": cell, "iccoh": float(sc_val)})
    if ic_raw_rows:
        pd.DataFrame(ic_raw_rows).to_csv(outdir / "ICCoh_raw_per_cell.csv", index=False)

    with open(outdir / "benchmark_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.save_h5ad:
        adata.write_h5ad(outdir / "AnnData_Forscore.h5ad")

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {outdir}")
    return summary


def main() -> None:
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
