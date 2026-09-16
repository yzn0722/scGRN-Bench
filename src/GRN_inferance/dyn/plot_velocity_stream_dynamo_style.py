#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamo-native velocity stream plots (fair comparison with scGPT).

Uses Dynamo's ``cell_velocities`` (cosine kernel) + ``streamline_plot`` pipeline,
matching the official hematopoiesis LAP tutorial.

Requires: conda env with dynamo + scanpy (e.g. dynamo-env after pip install scanpy).

Example:
  conda activate dynamo-env
  cd /mnt/10T/yzn/scGRN-Bench
  python src/GRN_inferance/dyn/plot_velocity_stream_dynamo_style.py \\
    --h5ad /mnt/10T/yzn/dynamo-release/results/hematopoiesis_raw/hematopoiesis_processed.h5ad \\
    --velocity-npz outputs/dynamo_fm_velocity/scgpt_hematopoiesis_vel1524_zero_shot/hematopoiesis_dyn_vel1524/velocity_field.npz \\
    --datasets-json data/dynamo_export/datasets_dynamo_vel1524_only.json \\
    --dataset hematopoiesis_dyn_vel1524 \\
    --color-key cell_type \\
    --outdir outputs/dynamo_fm_velocity/scgpt_hematopoiesis_vel1524_zero_shot/hematopoiesis_dyn_vel1524/figures \\
    --prefix stream_dynamo_style
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore")

import dynamo as dyn


def parse_args():
    p = argparse.ArgumentParser(description="Dynamo-style velocity stream plot (Dynamo vs scGPT).")
    p.add_argument("--h5ad", required=True, type=str)
    p.add_argument("--velocity-npz", default="", type=str, help="scGPT velocity_field.npz")
    p.add_argument("--datasets-json", default="", type=str)
    p.add_argument("--dataset", default="", type=str)
    p.add_argument("--velocity-layer", default="velocity_alpha_minus_gamma_s", type=str)
    p.add_argument("--dynamo-expr-layer", default="M_t", type=str, help="Expression layer for Dynamo projection.")
    p.add_argument("--basis", default="umap", type=str)
    p.add_argument("--color-key", default="cell_type", type=str)
    p.add_argument("--projection-method", default="cosine", choices=["cosine", "pearson", "kmc"])
    p.add_argument("--recompute-dynamo", action="store_true", help="Re-run cell_velocities for Dynamo panel.")
    p.add_argument("--inverse-dynamo", action="store_true", help="Flip Dynamo streamline arrows.")
    p.add_argument("--inverse-scgpt", action="store_true", help="Flip scGPT streamline arrows.")
    p.add_argument("--frontier", action="store_true", default=True)
    p.add_argument("--no-frontier", action="store_false", dest="frontier")
    p.add_argument("--outdir", required=True, type=str)
    p.add_argument("--prefix", default="stream_dynamo_style", type=str)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


ARGS = parse_args()


def bin_expr_to_0_50(x: np.ndarray, do_log1p: bool = False) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0)
    x = np.clip(x, 0, None)
    if do_log1p:
        x = np.log1p(x)
    vmax = max(float(np.percentile(x, 99.5)), 1e-6)
    return np.clip(x / vmax * 50.0, 0, 50).astype(np.float32)


def load_cells_and_genes(cfg: dict) -> Tuple[pd.Index, list]:
    expr = pd.read_csv(cfg["expr_csv"], index_col=0)
    pt_df = pd.read_csv(cfg["pt_csv"])
    pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"}).set_index("cell")
    common = expr.columns.intersection(pt_df.index)
    genes = expr.index.astype(str).tolist()
    return common.astype(str), genes


def align_adata(adata: sc.AnnData, cells: pd.Index, genes: list) -> Tuple[sc.AnnData, list]:
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names.astype(str))}
    gi = [gene_to_idx[g] for g in genes if g in gene_to_idx]
    genes_ok = [g for g in genes if g in gene_to_idx]
    return adata[cells, gi].copy(), genes_ok


def resolve_color(adata: sc.AnnData, cfg: dict, color_key: str) -> str:
    if color_key == "cell_type" and cfg.get("cell_type_csv"):
        ct_df = pd.read_csv(cfg["cell_type_csv"]).set_index("cell")
        ct_df.index = ct_df.index.astype(str)
        adata.obs["plot_color"] = [
            ct_df.loc[c, "cell_type"] if c in ct_df.index else "Unknown" for c in adata.obs_names
        ]
        return "plot_color"
    if color_key == "time" and cfg.get("time_csv"):
        tm_df = pd.read_csv(cfg["time_csv"]).set_index("cell")
        tm_df.index = tm_df.index.astype(str)
        adata.obs["plot_color"] = [
            float(tm_df.loc[c, "time"]) if c in tm_df.index else np.nan for c in adata.obs_names
        ]
        return "plot_color"
    if color_key not in adata.obs.columns:
        raise KeyError(f"{color_key} not in adata.obs")
    return color_key


def ensure_dynamo_velocity(adata: sc.AnnData) -> None:
    vel_key = f"velocity_{ARGS.basis}"
    if not ARGS.recompute_dynamo and vel_key in adata.obsm:
        print(f"  [INFO] Using existing obsm['{vel_key}']")
        return
    print(
        f"  [INFO] cell_velocities: X={ARGS.dynamo_expr_layer}, "
        f"V={ARGS.velocity_layer}, method={ARGS.projection_method}"
    )
    dyn.tl.cell_velocities(
        adata,
        enforce=True,
        X=adata.layers[ARGS.dynamo_expr_layer],
        V=adata.layers[ARGS.velocity_layer],
        method=ARGS.projection_method,
        basis=ARGS.basis,
    )


def project_scgpt_velocity(adata: sc.AnnData, cfg: dict, npz_path: str) -> None:
    d = np.load(npz_path)
    vel = d["vel_cell"].astype(np.float32)
    if vel.shape[0] != adata.n_obs:
        raise ValueError(f"npz cells {vel.shape[0]} != adata {adata.n_obs}")
    if vel.shape[1] != adata.n_vars:
        raise ValueError(f"npz genes {vel.shape[1]} != adata {adata.n_vars}")

    expr_df = pd.read_csv(cfg["expr_csv"], index_col=0)
    expr_df = expr_df.reindex(index=adata.var_names.astype(str))
    expr_df = expr_df[adata.obs_names.astype(str)]
    bins = bin_expr_to_0_50(expr_df.values.T.astype(np.float32), do_log1p=False)

    adata.layers["scgpt_bins"] = bins
    adata.layers["velocity_scgpt"] = vel
    adata.var["use_for_transition"] = True

    print(
        f"  [INFO] scGPT cell_velocities: X=scgpt_bins, V=velocity_scgpt, "
        f"method={ARGS.projection_method}"
    )
    dyn.tl.cell_velocities(
        adata,
        enforce=True,
        X=adata.layers["scgpt_bins"],
        V=adata.layers["velocity_scgpt"],
        method=ARGS.projection_method,
        basis=ARGS.basis,
        add_transition_key="scgpt_cosine_transition_matrix",
        add_velocity_key=f"velocity_scgpt_{ARGS.basis}",
    )


def time_alignment_cos(embedding: np.ndarray, velocity_emb: np.ndarray, labels: np.ndarray) -> float:
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


def plot_panel(
    adata: sc.AnnData,
    ax: plt.Axes,
    color: str,
    title: str,
    vector: str = "velocity",
    inverse: bool = False,
) -> dict:
    axes = dyn.pl.streamline_plot(
        adata,
        basis=ARGS.basis,
        color=color,
        vector=vector,
        inverse=inverse,
        frontier=ARGS.frontier,
        cut_off_velocity=True,
        method="gaussian",
        ax=ax,
        save_show_or_return="return",
        show_legend="on data",
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    emb = adata.obsm[f"X_{ARGS.basis}"]
    vel_emb = adata.obsm[f"{vector}_{ARGS.basis}"]
    labels = adata.obs[color].to_numpy()
    if not np.issubdtype(labels.dtype, np.number):
        labels = pd.Categorical(labels).codes.astype(float)
    return {
        "time_align_cos": time_alignment_cos(emb, vel_emb, labels.astype(float)),
        "vector_key": f"{vector}_{ARGS.basis}",
    }


def main():
    dyn.configuration.set_figure_params("dynamo", background="white")

    if not ARGS.datasets_json or not ARGS.dataset:
        raise ValueError("--datasets-json and --dataset are required.")

    with open(ARGS.datasets_json, encoding="utf-8") as f:
        cfg = json.load(f)[ARGS.dataset]

    outdir = Path(ARGS.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(ARGS.h5ad)
    adata.obs_names_make_unique()
    cells, genes = load_cells_and_genes(cfg)
    adata, genes = align_adata(adata, cells, genes)
    color = resolve_color(adata, cfg, ARGS.color_key)

    if ARGS.velocity_layer not in adata.layers:
        raise KeyError(f"Layer {ARGS.velocity_layer} missing in h5ad")
    if ARGS.dynamo_expr_layer not in adata.layers:
        raise KeyError(f"Layer {ARGS.dynamo_expr_layer} missing in h5ad")

    ensure_dynamo_velocity(adata)

    has_scgpt = bool(ARGS.velocity_npz)
    if has_scgpt:
        project_scgpt_velocity(adata, cfg, ARGS.velocity_npz)

    ncols = 2 if has_scgpt else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5.0), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    metrics = {"projection": "dynamo cell_velocities + streamline_plot", "method": ARGS.projection_method}

    metrics["dynamo"] = plot_panel(
        adata,
        axes[0],
        color,
        f"Dynamo RNA velocity\n({ARGS.dynamo_expr_layer} → cosine → UMAP)",
        vector="velocity",
        inverse=ARGS.inverse_dynamo,
    )

    if has_scgpt:
        metrics["scgpt"] = plot_panel(
            adata,
            axes[1],
            color,
            "scGPT velocity (zero-shot)\n(scGPT bins → cosine → UMAP)",
            vector="velocity_scgpt",
            inverse=ARGS.inverse_scgpt,
        )

    fig.savefig(outdir / f"{ARGS.prefix}_combined.png", dpi=ARGS.dpi, bbox_inches="tight")
    fig.savefig(outdir / f"{ARGS.prefix}_combined.pdf", bbox_inches="tight")
    plt.close(fig)

    for name, ax, vector, inverse, subtitle in [
        ("dynamo", axes[0], "velocity", ARGS.inverse_dynamo, "Dynamo RNA velocity"),
        *(
            [("scgpt", axes[1], "velocity_scgpt", ARGS.inverse_scgpt, "scGPT velocity (zero-shot)")]
            if has_scgpt
            else []
        ),
    ]:
        fig, ax_single = plt.subplots(figsize=(5.5, 5.0), constrained_layout=True)
        plot_panel(adata, ax_single, color, subtitle, vector=vector, inverse=inverse)
        fig.savefig(outdir / f"{ARGS.prefix}_{name}.png", dpi=ARGS.dpi, bbox_inches="tight")
        fig.savefig(outdir / f"{ARGS.prefix}_{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    with open(outdir / f"{ARGS.prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] Saved Dynamo-style stream plots -> {outdir}")
    for k, m in metrics.items():
        if k in ("projection", "method"):
            continue
        print(f"  {k}: time_align_cos={m['time_align_cos']:.3f} ({m['vector_key']})")


if __name__ == "__main__":
    main()
