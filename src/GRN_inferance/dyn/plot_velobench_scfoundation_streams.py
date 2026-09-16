#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scFoundation zero-shot velocity stream plots (Dynamo streamline_plot style).

Example:
  conda activate dynamo-env
  cd /mnt/10T/yzn/scGRN-Bench
  python -u src/GRN_inferance/dyn/plot_velobench_scfoundation_streams.py
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
VELO_ROOT = ROOT / "outputs/dynamo_fm_velocity/velobench_four_scfoundation"
REUSE_EXPORT = ROOT / "outputs/dynamo_fm_velocity/velobench_four_scgpt"
H5AD_DIR = Path("/mnt/10T/yzn/veloBench/results/Dynamo/CB_IC")

DATASETS: dict[str, dict[str, Any]] = {
    "HumanBoneMarrow": {"cluster_col": "clusters", "basis": "umap", "title": "Human bone marrow"},
    "HumanBrain": {"cluster_col": "cluster", "basis": "umap", "title": "Developing human brain"},
    "PBMC68k": {"cluster_col": "celltype", "basis": "tsne", "title": "PBMC-68k"},
    "HumanHSPC": {"cluster_col": "leiden", "basis": "umap", "title": "Human HSPC"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dynamo-style scFoundation stream plots for veloBench")
    p.add_argument("--dataset", nargs="*", choices=list(DATASETS.keys()), default=list(DATASETS.keys()))
    p.add_argument("--out-root", default=str(VELO_ROOT), type=str)
    p.add_argument("--reuse-export-from", default=str(REUSE_EXPORT), type=str)
    p.add_argument("--max-cells", type=int, default=0)
    p.add_argument("--projection-method", default="cosine", choices=["cosine", "pearson", "kmc"])
    p.add_argument("--inverse", action="store_true")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def resolve_export_json(name: str, args: argparse.Namespace) -> Path:
    local = Path(args.out_root) / name / "export" / f"datasets_{name}.json"
    if local.exists():
        return local
    shared = Path(args.reuse_export_from) / name / "export" / f"datasets_{name}.json"
    if shared.exists():
        return shared
    raise FileNotFoundError(f"No export json for {name}")


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


def project_scfoundation_velocity(adata: sc.AnnData, cfg: dict, npz_path: Path, basis: str, method: str) -> None:
    d = np.load(npz_path)
    vel = d["vel_cell"].astype(np.float32)
    npz_cells = d["cells"].astype(str)
    cell_to_row = {c: i for i, c in enumerate(npz_cells)}
    rows = [cell_to_row[c] for c in adata.obs_names.astype(str)]
    vel = vel[rows]

    expr_path = cfg["expr_csv"]
    expr_df = pd.read_csv(ROOT / expr_path if not Path(expr_path).is_absolute() else expr_path, index_col=0)
    expr_df = expr_df.reindex(index=adata.var_names.astype(str))
    expr_df = expr_df[adata.obs_names.astype(str)]
    bins = bin_expr_to_0_50(expr_df.values.T.astype(np.float32))

    adata.layers["scf_bins"] = bins
    adata.layers["velocity_scfoundation"] = vel
    adata.var["use_for_transition"] = True

    if f"X_{basis}" not in adata.obsm:
        raise KeyError(f"Missing embedding X_{basis}")

    dyn.tl.cell_velocities(
        adata,
        enforce=True,
        X=adata.layers["scf_bins"],
        V=adata.layers["velocity_scfoundation"],
        method=method,
        basis=basis,
        add_transition_key="scf_cosine_transition_matrix",
        add_velocity_key=f"velocity_scfoundation_{basis}",
    )


def plot_stream_panel(adata: sc.AnnData, ax: plt.Axes, color: str, basis: str, inverse: bool = False) -> None:
    dyn.pl.streamline_plot(
        adata,
        basis=basis,
        color=color,
        vector="velocity_scfoundation",
        inverse=inverse,
        frontier=True,
        cut_off_velocity=True,
        method="gaussian",
        ax=ax,
        save_show_or_return="return",
        show_legend="on data",
    )
    ax.set_xlabel(f"{basis.upper()}_1")
    ax.set_ylabel(f"{basis.upper()}_2")


def plot_dataset(name: str, cfg: dict[str, Any], args: argparse.Namespace) -> Path:
    export_json = resolve_export_json(name, args)
    with open(export_json, encoding="utf-8") as f:
        ds_cfg = json.load(f)[name]

    ds_root = Path(args.out_root) / name
    h5ad_path = Path(ds_cfg.get("h5ad") or H5AD_DIR / f"{name}_AnnData_Forscore.h5ad")
    npz_path = ds_root / "scfoundation_zero_shot" / name / "velocity_field.npz"
    basis = cfg["basis"]
    cluster_col = cfg["cluster_col"]
    outdir = ds_root / "figures_scfoundation_stream"
    outdir.mkdir(parents=True, exist_ok=True)

    expr_path = ds_cfg["expr_csv"]
    expr_df = pd.read_csv(ROOT / expr_path if not Path(expr_path).is_absolute() else expr_path, index_col=0)
    pt_path = ds_cfg["pt_csv"]
    pt_df = pd.read_csv(ROOT / pt_path if not Path(pt_path).is_absolute() else pt_path)
    pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"}).set_index("cell")
    common = expr_df.columns.astype(str).intersection(pt_df.index.astype(str))
    genes = expr_df.index.astype(str).tolist()

    adata = sc.read_h5ad(h5ad_path)
    adata.obs_names_make_unique()
    adata = align_adata(adata, common, genes)

    max_cells = args.max_cells
    if name == "PBMC68k" and max_cells <= 0:
        max_cells = 8000
    adata = subsample_stratified(adata, cluster_col, max_cells)

    project_scfoundation_velocity(adata, ds_cfg, npz_path, basis, args.projection_method)

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
    print(f"[OK] {name} -> {png_path}")
    return png_path


def plot_combined(panel_paths: list[tuple[str, Path]], out_path: Path, dpi: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.0), constrained_layout=True)
    for ax, (title, img_path), lab in zip(axes.flat, panel_paths, ["A", "B", "C", "D"]):
        ax.imshow(plt.imread(img_path))
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.text(-0.02, 1.02, lab, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] combined -> {png_path}")


def main() -> None:
    args = parse_args()
    dyn.configuration.set_figure_params("dynamo", background="white")
    panel_paths: list[tuple[str, Path]] = []
    for name in args.dataset:
        try:
            panel_paths.append((DATASETS[name]["title"], plot_dataset(name, DATASETS[name], args)))
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}")
    if len(panel_paths) == 4:
        plot_combined(
            panel_paths,
            Path(args.out_root) / "figures_scfoundation_stream" / "stream_dynamo_style_velobench_four_panel",
            args.dpi,
        )


if __name__ == "__main__":
    main()
