#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scGPT zero-shot RNA velocity on veloBench local four CBDir datasets.

Pipeline per dataset:
  1. Export expression from Dynamo-processed AnnData (veloBench Forscore h5ad)
  2. scGPT per-cell iterative MLM velocity -> velocity_field.npz  (GPU)
  3. veloBench CBDir / ICCoh via eval_velocity_benchmark_metrics.py

Example (all four datasets):
  conda activate singlecell
  export CUDA_VISIBLE_DEVICES=0
  cd /mnt/10T/yzn/scGRN-Bench

  python -u src/GRN_inferance/dyn/run_velobench_scgpt_zero_shot.py

Example (single dataset, skip GPU if npz exists):
  python -u src/GRN_inferance/dyn/run_velobench_scgpt_zero_shot.py \\
    --dataset HumanBrain --skip-scgpt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[3]
VELO_H5AD_DIR = Path("/mnt/10T/yzn/veloBench/results/Dynamo/CB_IC")
DEFAULT_MODEL = Path("/mnt/10T/yzn/benchmark_GRN/model/weights/scgpt/scGPT_human")
DEFAULT_OUT = ROOT / "outputs/dynamo_fm_velocity/velobench_four_scgpt"
EDGES_JSON = ROOT / "data/dynamo_export/velobench_cbdir_edges.json"

DATASETS: dict[str, dict[str, Any]] = {
    "HumanBoneMarrow": {
        "velobench_id": "Dataset4",
        "edge_preset": "dataset4",
        "cluster_col": "clusters",
        "x_emb": "X_umap",
        "expr_layer": "M_s",
        "pt_obs": "palantir_pseudotime",
    },
    "HumanBrain": {
        "velobench_id": "Dataset9",
        "edge_preset": "dataset9",
        "cluster_col": "cluster",
        "x_emb": "X_umap",
        "expr_layer": "Ms",
        "pt_obsm": "X_umap",
        "pt_dim": 0,
    },
    "PBMC68k": {
        "velobench_id": "Dataset11",
        "edge_preset": "dataset11",
        "cluster_col": "celltype",
        "x_emb": "X_tsne",
        "expr_layer": "M_s",
        "pt_obsm": "X_tsne",
        "pt_dim": 0,
    },
    "HumanHSPC": {
        "velobench_id": "Dataset12",
        "edge_preset": "dataset12",
        "cluster_col": "leiden",
        "x_emb": "X_umap",
        "expr_layer": "Ms",
        "pt_obs": "root_prediction",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="scGPT zero-shot on veloBench four datasets")
    p.add_argument(
        "--dataset",
        nargs="*",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
        help="Subset to run (default: all four CBDir datasets).",
    )
    p.add_argument("--h5ad-dir", default=str(VELO_H5AD_DIR), type=str)
    p.add_argument("--out-root", default=str(DEFAULT_OUT), type=str)
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL), type=str)
    p.add_argument("--gen-iters", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--skip-scgpt", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--scgpt-env", default="singlecell", type=str, help="Conda env for GPU scGPT")
    p.add_argument("--eval-env", default="dynamo-env", type=str, help="Conda env for CBDir/ICCoh eval")
    return p.parse_args()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pick_expression(adata: sc.AnnData, preferred: str) -> tuple[np.ndarray, list[str]]:
    gene_mask = None
    if "use_for_dynamics" in adata.var.columns:
        gene_mask = adata.var["use_for_dynamics"].astype(bool).to_numpy()
    elif "highly_variable" in adata.var.columns:
        gene_mask = adata.var["highly_variable"].astype(bool).to_numpy()
    if gene_mask is not None and int(gene_mask.sum()) > 0:
        adata = adata[:, gene_mask].copy()
        log(f"  Using {adata.n_vars} dynamics/HVG genes for scGPT export")

    for layer in (preferred, "Ms", "M_s", "spliced"):
        if layer in adata.layers:
            mat = adata.layers[layer]
            if sp.issparse(mat):
                mat = mat.toarray()
            genes = adata.var_names.astype(str).tolist()
            return np.asarray(mat, dtype=np.float32), genes
    raise KeyError(f"No expression layer in adata.layers: {list(adata.layers.keys())[:15]}")


def pick_pseudotime(adata: sc.AnnData, cfg: dict) -> pd.Series:
    if cfg.get("pt_obs") and cfg["pt_obs"] in adata.obs.columns:
        pt = adata.obs[cfg["pt_obs"]].astype(float)
    elif cfg.get("pt_obsm") and cfg["pt_obsm"] in adata.obsm:
        dim = int(cfg.get("pt_dim", 0))
        pt = pd.Series(adata.obsm[cfg["pt_obsm"]][:, dim], index=adata.obs_names)
    else:
        if "X_umap" in adata.obsm:
            pt = pd.Series(adata.obsm["X_umap"][:, 0], index=adata.obs_names)
        else:
            pt = pd.Series(np.linspace(0, 1, adata.n_obs), index=adata.obs_names)
        log(f"  [WARN] No pseudotime column; using UMAP dim0 proxy")
    pt = pt.astype(float)
    lo, hi = pt.min(), pt.max()
    if hi > lo:
        pt = (pt - lo) / (hi - lo)
    return pt


def export_dataset(name: str, cfg: dict, h5ad_path: Path, export_dir: Path) -> dict:
    export_dir.mkdir(parents=True, exist_ok=True)
    chip_dir = export_dir / "CHIP"
    chip_dir.mkdir(exist_ok=True)

    adata = sc.read_h5ad(h5ad_path)
    adata.obs_names_make_unique()
    log(f"  Loaded {h5ad_path.name}: {adata.n_obs} x {adata.n_vars}")

    X, genes = pick_expression(adata, cfg["expr_layer"])
    cells = adata.obs_names.astype(str).tolist()
    expr_df = pd.DataFrame(X.T, index=genes, columns=cells)
    expr_path = chip_dir / f"{name}_chip_matched-ExpressionData.csv"
    expr_df.to_csv(expr_path)

    pt = pick_pseudotime(adata, cfg)
    pt_path = export_dir / "PseudoTime.csv"
    pd.DataFrame({"cell": cells, "pt": pt.values}).to_csv(pt_path, index=False)

    ct_path = export_dir / "CellType.csv"
    pd.DataFrame(
        {"cell": cells, "cell_type": adata.obs[cfg["cluster_col"]].astype(str).values}
    ).to_csv(ct_path, index=False)

    ds_cfg = {
        "expr_csv": str(expr_path.relative_to(ROOT)),
        "pt_csv": str(pt_path.relative_to(ROOT)),
        "cell_type_csv": str(ct_path.relative_to(ROOT)),
        "h5ad": str(h5ad_path),
        "species": "human",
        "cell_type_col": cfg["cluster_col"],
        "n_genes": int(adata.n_vars),
        "n_cells": int(adata.n_obs),
        "velobench_id": cfg["velobench_id"],
        "edge_preset": cfg["edge_preset"],
        "x_emb": cfg["x_emb"],
    }
    with open(export_dir / f"datasets_{name}.json", "w", encoding="utf-8") as f:
        json.dump({name: ds_cfg}, f, indent=2)
    return ds_cfg


def run_scgpt(
    model_dir: Path,
    expr_csv: Path,
    out_npz: Path,
    gen_iters: int,
    batch_size: int,
    scgpt_env: str,
) -> None:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    script = ROOT / "src/GRN_inferance/dyn/scgpt_per_cell_velocity_only.py"
    cmd = [
        "conda",
        "run",
        "-n",
        scgpt_env,
        "python",
        "-u",
        str(script),
        "--model-dir",
        str(model_dir),
        "--expr-csv",
        str(expr_csv),
        "--out-npz",
        str(out_npz),
        "--gen-iters",
        str(gen_iters),
        "--batch-size",
        str(batch_size),
    ]
    log(f"  [scGPT] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def run_eval(
    h5ad: Path,
    datasets_json: Path,
    dataset: str,
    velocity_npz: Path,
    cluster_col: str,
    x_emb: str,
    edge_preset: str,
    outdir: Path,
    eval_env: str,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    script = ROOT / "src/GRN_inferance/dyn/eval_velocity_benchmark_metrics.py"
    cmd = [
        "conda",
        "run",
        "-n",
        eval_env,
        "python",
        "-u",
        str(script),
        "--h5ad",
        str(h5ad),
        "--datasets-json",
        str(datasets_json),
        "--dataset",
        dataset,
        "--velocity-source",
        "scgpt",
        "--velocity-npz",
        str(velocity_npz),
        "--cluster-col",
        cluster_col,
        "--x-emb",
        x_emb,
        "--edge-preset",
        edge_preset,
        "--edges-json",
        str(EDGES_JSON),
        "--projection",
        "neighbor",
        "--expr-space",
        "bins",
        "--n-neighbors",
        "30",
        "--method-name",
        f"scGPT_zero_shot_{dataset}",
        "--outdir",
        str(outdir),
        "--save-h5ad",
    ]
    log(f"  [eval] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def update_master_json(out_root: Path, names: list[str]) -> None:
    master = out_root / "datasets_velobench_four.json"
    cfg: dict = {}
    if master.exists():
        with open(master, encoding="utf-8") as f:
            cfg = json.load(f)
    for name in names:
        ds_json = out_root / name / "export" / f"datasets_{name}.json"
        if ds_json.exists():
            with open(ds_json, encoding="utf-8") as f:
                cfg.update(json.load(f))
    with open(master, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    log(f"Updated master config -> {master}")


def run_one(name: str, cfg: dict, args: argparse.Namespace) -> dict:
    out_root = Path(args.out_root)
    ds_root = out_root / name
    h5ad_path = Path(args.h5ad_dir) / f"{name}_AnnData_Forscore.h5ad"
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Missing Dynamo Forscore h5ad: {h5ad_path}")

    export_dir = ds_root / "export"
    expr_csv = export_dir / "CHIP" / f"{name}_chip_matched-ExpressionData.csv"
    npz_path = ds_root / "scgpt_zero_shot" / name / "velocity_field.npz"
    metrics_dir = ds_root / "metrics_scgpt"
    ds_json = export_dir / f"datasets_{name}.json"

    log(f"\n{'=' * 60}\n{name} ({cfg['velobench_id']})\n{'=' * 60}")

    if not args.skip_export:
        export_dataset(name, cfg, h5ad_path, export_dir)
    elif not expr_csv.exists():
        raise FileNotFoundError(f"Export missing: {expr_csv} (run without --skip-export)")

    if not args.skip_scgpt:
        if npz_path.exists():
            log(f"  [scGPT] Removing stale npz -> {npz_path}")
            npz_path.unlink()
        run_scgpt(
            Path(args.model_dir),
            expr_csv,
            npz_path,
            args.gen_iters,
            args.batch_size,
            args.scgpt_env,
        )
    elif not npz_path.exists():
        raise FileNotFoundError(f"velocity_field.npz missing: {npz_path}")

    summary = {"dataset": name, "status": "exported+scgpt"}
    if not args.skip_eval:
        run_eval(
            h5ad_path,
            ds_json,
            name,
            npz_path,
            cfg["cluster_col"],
            cfg["x_emb"],
            cfg["edge_preset"],
            metrics_dir,
            args.eval_env,
        )
        with open(metrics_dir / "benchmark_metrics_summary.json", encoding="utf-8") as f:
            summary = json.load(f)
            summary["dataset"] = name

    return summary


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    failures = []
    for name in args.dataset:
        try:
            summaries.append(run_one(name, DATASETS[name], args))
        except subprocess.CalledProcessError as exc:
            log(f"FAILED {name}: exit code {exc.returncode}")
            failures.append(name)
        except Exception as exc:
            log(f"FAILED {name}: {exc}")
            failures.append(name)

    update_master_json(out_root, args.dataset)

    if summaries:
        rows = []
        for s in summaries:
            rows.append(
                {
                    "dataset": s.get("dataset", ""),
                    "cbdir_mean": s.get("cbdir_mean"),
                    "iccoh_mean": s.get("iccoh_mean"),
                    "n_cells": s.get("n_cells"),
                    "n_genes": s.get("n_genes"),
                }
            )
        status_path = out_root / "scgpt_velobench_status.csv"
        pd.DataFrame(rows).to_csv(status_path, index=False)
        log(f"\nStatus table -> {status_path}")
        print(pd.DataFrame(rows).to_string(index=False))

    if failures:
        log(f"\nFailed datasets: {', '.join(failures)}")
        sys.exit(1)

    log("\nAll done.")


if __name__ == "__main__":
    main()
