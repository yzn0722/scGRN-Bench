#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run veloBench Dataset 12 official CBDir evaluation (Dynamo + optional scGPT export).

Steps:
  1. Dynamo recipe_monocle + velocity (veloBench notebook style)
  2. unitvelo CBDir / ICCoh / VC with official 4 edges
  3. Export expression + pseudotime for scGPT zero-shot
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dataset12 benchmark pipeline")
    p.add_argument(
        "--h5ad",
        default="/mnt/10T/yzn/veloBench/data/3423-MV-2_adata_postpro.h5ad",
        type=str,
    )
    p.add_argument(
        "--out-root",
        default=str(ROOT / "outputs/dynamo_fm_velocity/dataset12_official"),
        type=str,
    )
    p.add_argument("--n-neighbors", type=int, default=30)
    p.add_argument("--skip-dynamo", action="store_true")
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    return p.parse_args()


def run_dynamo_velobench(adata: sc.AnnData, n_neighbors: int) -> sc.AnnData:
    import dynamo as dyn

    adata = adata.copy()
    print("[Dynamo] recipe_monocle ...")
    dyn.pp.recipe_monocle(adata, n_top_genes=1000, fg_kwargs={"shared_count": 20})
    print("[Dynamo] dynamics (stochastic) ...")
    dyn.tl.dynamics(adata, model="stochastic")
    print(f"[Dynamo] neighbors n={n_neighbors} ...")
    dyn.tl.neighbors(adata, n_neighbors=n_neighbors)
    print("[Dynamo] reduceDimension ...")
    dyn.tl.reduceDimension(adata, n_pca_components=30)
    print("[Dynamo] cell_velocities (pearson) ...")
    dyn.tl.cell_velocities(
        adata,
        method="pearson",
        other_kernels_dict={"transform": "sqrt"},
    )
    if "velocity_umap" in adata.obsm:
        adata.obsm["velocity_S_umap"] = adata.obsm["velocity_umap"].copy()
    return adata


def export_for_scgpt(adata: sc.AnnData, export_dir: Path) -> dict:
    export_dir.mkdir(parents=True, exist_ok=True)
    chip_dir = export_dir / "CHIP"
    chip_dir.mkdir(exist_ok=True)

    expr_layer = "Ms" if "Ms" in adata.layers else None
    if expr_layer is None:
        X = adata.X
    else:
        X = adata.layers[expr_layer]
    if sp.issparse(X):
        X = X.toarray()
    expr_df = pd.DataFrame(
        np.asarray(X, dtype=np.float32).T,
        index=adata.var_names.astype(str),
        columns=adata.obs_names.astype(str),
    )
    expr_path = chip_dir / "dataset12_chip_matched-ExpressionData.csv"
    expr_df.to_csv(expr_path)

    pt = adata.obs["root_prediction"].astype(float).to_numpy()
    pt = (pt - pt.min()) / max(pt.max() - pt.min(), 1e-8)
    pt_df = pd.DataFrame({"cell": adata.obs_names.astype(str), "pt": pt})
    pt_path = export_dir / "PseudoTime.csv"
    pt_df.to_csv(pt_path, index=False)

    ct_df = pd.DataFrame(
        {"cell": adata.obs_names.astype(str), "cell_type": adata.obs["leiden"].astype(str)}
    )
    ct_path = export_dir / "CellType.csv"
    ct_df.to_csv(ct_path, index=False)

    def rel(p: Path) -> str:
        p = p.resolve()
        try:
            return str(p.relative_to(ROOT.resolve()))
        except ValueError:
            return str(p)

    cfg = {
        "dataset12_official": {
            "expr_csv": rel(expr_path),
            "pt_csv": rel(pt_path),
            "cell_type_csv": rel(ct_path),
            "h5ad": "",
            "species": "human",
            "cell_type_col": "leiden",
            "n_genes": int(adata.n_vars),
        }
    }
    cfg_path = export_dir / "datasets_dataset12.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return cfg


def run_eval(h5ad: Path, velocity_source: str, velocity_layer: str, velocity_npz: str, outdir: Path, method: str):
    cmd = [
        sys.executable,
        str(ROOT / "src/GRN_inferance/dyn/eval_velocity_benchmark_metrics.py"),
        "--h5ad",
        str(h5ad),
        "--velocity-source",
        velocity_source,
        "--velocity-layer",
        velocity_layer,
        "--cluster-col",
        "leiden",
        "--edge-preset",
        "dataset12_official",
        "--edges-json",
        str(ROOT / "data/dynamo_export/hematopoiesis_dyn_vel1524/benchmark_cluster_edges.json"),
        "--n-neighbors",
        "30",
        "--method-name",
        method,
        "--outdir",
        str(outdir),
        "--save-h5ad",
    ]
    if velocity_source == "dynamo":
        cmd.extend(["--projection", "precomputed"])
    elif velocity_source == "scgpt":
        cmd.extend(["--velocity-npz", velocity_npz, "--projection", "neighbor", "--expr-space", "bins"])
    print("[eval]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    raw_path = Path(args.h5ad)
    adata = sc.read_h5ad(raw_path)
    print(f"Loaded {raw_path.name}: {adata.n_obs} x {adata.n_vars}")

    dynamo_h5ad = out_root / "dataset12_dynamo_processed.h5ad"
    export_dir = out_root / "export"

    if not args.skip_dynamo:
        adata_dyn = run_dynamo_velobench(adata, args.n_neighbors)
        adata_dyn.write_h5ad(dynamo_h5ad)
        print(f"Saved Dynamo processed -> {dynamo_h5ad}")
    else:
        adata_dyn = sc.read_h5ad(dynamo_h5ad)

    if not args.skip_export:
        cfg = export_for_scgpt(adata_dyn, export_dir)
        print(f"Exported scGPT inputs -> {export_dir}")

    if not args.skip_eval:
        run_eval(
            dynamo_h5ad,
            velocity_source="dynamo",
            velocity_layer="velocity_S",
            velocity_npz="",
            outdir=out_root / "metrics_dynamo",
            method="Dynamo_dataset12",
        )


if __name__ == "__main__":
    main()
