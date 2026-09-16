#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scGPT zero-shot velocity stream plots (Dynamo streamline_plot style).

Matches hematopoiesis reference:
  outputs/.../stream_dynamo_style_combined.png

Example:
  conda activate dynamo-env
  cd /mnt/10T/yzn/scGRN-Bench
  python -u src/GRN_inferance/dyn/plot_velobench_scgpt_streams.py
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore")

import dynamo as dyn

ROOT = Path(__file__).resolve().parents[3]
VELO_ROOT = ROOT / "outputs/dynamo_fm_velocity/velobench_four_scgpt"
H5AD_DIR = Path("/mnt/10T/yzn/veloBench/results/Dynamo/CB_IC")

DATASETS: dict[str, dict[str, Any]] = {
    "HumanBoneMarrow": {
        "velobench_id": "Dataset4",
        "cluster_col": "clusters",
        "basis": "umap",
        "title": "Human bone marrow",
    },
    "HumanBrain": {
        "velobench_id": "Dataset9",
        "cluster_col": "cluster",
        "basis": "umap",
        "title": "Developing human brain",
    },
    "PBMC68k": {
        "velobench_id": "Dataset11",
        "cluster_col": "celltype",
        "basis": "tsne",
        "title": "PBMC-68k",
    },
    "HumanHSPC": {
        "velobench_id": "Dataset12",
        "cluster_col": "leiden",
        "basis": "umap",
        "title": "Human HSPC",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dynamo-style scGPT stream plots for veloBench four datasets")
    p.add_argument("--dataset", nargs="*", choices=list(DATASETS.keys()), default=list(DATASETS.keys()))
    p.add_argument("--out-root", default=str(VELO_ROOT), type=str)
    p.add_argument("--max-cells", type=int, default=0, help="Subsample for plot (0=all; PBMC default 8000).")
    p.add_argument("--projection-method", default="cosine", choices=["cosine", "pearson", "kmc"])
    p.add_argument("--inverse", action="store_true", help="Flip streamline arrow direction.")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def bin_expr_to_0_50(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0)
    x = np.clip(x, 0, None)
    vmax = max(float(np.percentile(x, 99.5)), 1e-6)
    return np.clip(x / vmax * 50.0, 0, 50).astype(np.float32)


def align_adata(adata: sc.AnnData, cells: pd.Index, genes: list) -> sc.AnnData:
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names.astype(str))}
    gi = [gene_to_idx[g] for g in genes if g in gene_to_idx]
    return adata[cells, gi].copy()


def subsample_stratified(adata: sc.AnnData, cluster_col: str, max_cells: int, seed: int = 0) -> sc.AnnData:
    if max_cells <= 0 or adata.n_obs <= max_cells:
        return adata
    rng = np.random.default_rng(seed)
    labels = adata.obs[cluster_col].astype(str)
    groups = labels.groupby(labels).groups
    per_group = max(1, max_cells // len(groups))
    picked: list[str] = []
    for _, idx_labels in groups.items():
        idx_labels = list(idx_labels)
        if len(idx_labels) <= per_group:
            picked.extend([str(i) for i in idx_labels])
        else:
            sel = rng.choice(idx_labels, size=per_group, replace=False)
            picked.extend([str(s) for s in sel])
    if len(picked) > max_cells:
        picked = [str(x) for x in rng.choice(np.array(picked, dtype=object), size=max_cells, replace=False)]
    return adata[picked].copy()


def project_scgpt_velocity(adata: sc.AnnData, cfg: dict, npz_path: Path, basis: str, method: str) -> None:
    d = np.load(npz_path)
    vel = d["vel_cell"].astype(np.float32)
    npz_cells = d["cells"].astype(str)
    cell_to_row = {c: i for i, c in enumerate(npz_cells)}
    rows = [cell_to_row[c] for c in adata.obs_names.astype(str)]
    vel = vel[rows]

    expr_df = pd.read_csv(ROOT / cfg["expr_csv"], index_col=0)
    expr_df = expr_df.reindex(index=adata.var_names.astype(str))
    expr_df = expr_df[adata.obs_names.astype(str)]
    bins = bin_expr_to_0_50(expr_df.values.T.astype(np.float32))

    adata.layers["scgpt_bins"] = bins
    adata.layers["velocity_scgpt"] = vel
    adata.var["use_for_transition"] = True

    emb_key = f"X_{basis}"
    if emb_key not in adata.obsm:
        raise KeyError(f"Missing embedding {emb_key} in adata.obsm")

    dyn.tl.cell_velocities(
        adata,
        enforce=True,
        X=adata.layers["scgpt_bins"],
        V=adata.layers["velocity_scgpt"],
        method=method,
        basis=basis,
        add_transition_key="scgpt_cosine_transition_matrix",
        add_velocity_key=f"velocity_scgpt_{basis}",
    )


def plot_stream_panel(
    adata: sc.AnnData,
    ax: plt.Axes,
    color: str,
    basis: str,
    inverse: bool = False,
) -> None:
    dyn.pl.streamline_plot(
        adata,
        basis=basis,
        color=color,
        vector="velocity_scgpt",
        inverse=inverse,
        frontier=True,
        cut_off_velocity=True,
        method="gaussian",
        ax=ax,
        save_show_or_return="return",
        show_legend="on data",
    )
    emb = basis.upper()
    ax.set_xlabel(f"{emb}_1")
    ax.set_ylabel(f"{emb}_2")


def plot_dataset(name: str, cfg: dict[str, Any], args: argparse.Namespace) -> Path:
    ds_root = Path(args.out_root) / name
    export_json = ds_root / "export" / f"datasets_{name}.json"
    with open(export_json, encoding="utf-8") as f:
        ds_cfg = json.load(f)[name]

    h5ad_path = Path(ds_cfg.get("h5ad") or H5AD_DIR / f"{name}_AnnData_Forscore.h5ad")
    npz_path = ds_root / "scgpt_zero_shot" / name / "velocity_field.npz"
    basis = cfg["basis"]
    cluster_col = cfg["cluster_col"]
    outdir = ds_root / "figures_scgpt_stream"
    outdir.mkdir(parents=True, exist_ok=True)

    expr_df = pd.read_csv(ROOT / ds_cfg["expr_csv"], index_col=0)
    pt_df = pd.read_csv(ROOT / ds_cfg["pt_csv"])
    pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"}).set_index("cell")
    common = expr_df.columns.astype(str).intersection(pt_df.index.astype(str))
    genes = expr_df.index.astype(str).tolist()

    adata = sc.read_h5ad(h5ad_path)
    adata.obs_names_make_unique()
    adata = align_adata(adata, common, genes)

    if cluster_col not in adata.obs.columns:
        raise KeyError(f"{cluster_col} not in adata.obs")

    n_before = adata.n_obs
    max_cells = args.max_cells
    if name == "PBMC68k" and max_cells <= 0:
        max_cells = 8000
    adata = subsample_stratified(adata, cluster_col, max_cells)

    project_scgpt_velocity(adata, ds_cfg, npz_path, basis, args.projection_method)

    prefix = f"stream_dynamo_style_{name}_{basis}"
    fig, ax = plt.subplots(figsize=(5.5, 5.0), constrained_layout=True)
    plot_stream_panel(adata, ax, cluster_col, basis, inverse=args.inverse)
    png_path = outdir / f"{prefix}.png"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    try:
        fig.savefig(outdir / f"{prefix}.pdf", bbox_inches="tight")
    except ValueError as exc:
        print(f"[WARN] {name}: PDF skipped ({exc})")
    plt.close(fig)

    meta = {
        "dataset": name,
        "style": "dynamo streamline_plot",
        "basis": basis,
        "n_cells_plotted": int(adata.n_obs),
        "cluster_col": cluster_col,
        "npz": str(npz_path),
    }
    with open(outdir / f"{prefix}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] {name} -> {png_path}")
    return png_path


def plot_combined(panel_paths: list[tuple[str, Path]], out_path: Path, dpi: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.0), constrained_layout=True)
    labels = ["A", "B", "C", "D"]
    for ax, (title, img_path), lab in zip(axes.flat, panel_paths, labels):
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.text(-0.02, 1.02, lab, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    try:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    except ValueError as exc:
        print(f"[WARN] combined PDF skipped ({exc})")
    plt.close(fig)
    print(f"[OK] combined -> {png_path}")


def main() -> None:
    args = parse_args()
    dyn.configuration.set_figure_params("dynamo", background="white")

    panel_paths: list[tuple[str, Path]] = []
    for name in args.dataset:
        try:
            png = plot_dataset(name, DATASETS[name], args)
            panel_paths.append((DATASETS[name]["title"], png))
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}")

    if len(panel_paths) == 4:
        combined = Path(args.out_root) / "figures_scgpt_stream" / "stream_dynamo_style_velobench_four_panel"
        plot_combined(panel_paths, combined, args.dpi)


if __name__ == "__main__":
    main()
