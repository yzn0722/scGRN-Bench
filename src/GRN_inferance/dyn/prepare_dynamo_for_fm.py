#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export Dynamo / AnnData (h5ad) to scGRN-Bench dynamic-benchmark layout.

Outputs per dataset:
  <out_root>/<name>/CHIP/<name>_chip_matched-ExpressionData.csv   (genes x cells)
  <out_root>/<name>/PseudoTime.csv                                (cell, pt)
  <out_root>/<name>/dynamo_velocity_truth.csv  (optional, genes x 1 velocity)

Then run foundation models, e.g.:
  python scgpt_dyn.py --model-dir ... --outdir ... \\
    --datasets-json <out_root>/datasets_dynamo.json

Example:
  python prepare_dynamo_for_fm.py \\
    --h5ad /mnt/10T/yzn/dynamo-release/data/zebrafish.h5ad \\
    --name zebrafish --species zebrafish --pt-source umap_1

  python prepare_dynamo_for_fm.py \\
    --h5ad /mnt/10T/yzn/dynamo-release/results/zebrafish/zebrafish_processed.h5ad \\
    --name zebrafish_dyn --species zebrafish --pt-source latent_time \\
    --velocity-layer velocity_S
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Prepare Dynamo h5ad for FM velocity benchmark")
    p.add_argument("--h5ad", required=True, help="Input AnnData h5ad")
    p.add_argument("--name", required=True, help="Dataset name (e.g. zebrafish, hematopoiesis)")
    p.add_argument(
        "--out-root",
        default="/mnt/10T/yzn/scGRN-Bench/data/dynamo_export",
        help="Root output directory",
    )
    p.add_argument(
        "--species",
        choices=["human", "mouse", "zebrafish"],
        default="human",
        help="For datasets.json species field (gene mapping in scGPT)",
    )
    p.add_argument(
        "--layer",
        default="spliced",
        help="Expression layer (spliced, unspliced, or use X)",
    )
    p.add_argument(
        "--pt-source",
        default="auto",
        help="Pseudotime column in obs, or: umap_1, auto",
    )
    p.add_argument(
        "--velocity-layer",
        default="",
        help="If set (e.g. velocity_S), export dynamo RNA velocity per gene as truth",
    )
    p.add_argument(
        "--gene-transform",
        choices=["none", "upper", "mouse_to_human"],
        default="none",
        help="Gene symbol transform before export",
    )
    p.add_argument(
        "--write-datasets-json",
        action="store_true",
        default=True,
        help="Append/update datasets_dynamo.json under out-root",
    )
    return p.parse_args()


def transform_genes(genes, mode: str):
    genes = [str(g) for g in genes]
    if mode == "upper":
        return [g.upper() for g in genes]
    if mode == "mouse_to_human":
        return [g.upper() for g in genes]
    return genes


def pick_pseudotime(obs: pd.DataFrame, pt_source: str) -> pd.Series:
    candidates = [
        "pseudotime_fp",
        "latent_time",
        "palantir_pseudotime",
        "dpt_pseudotime",
        "pseudotime",
        "time",
        "PT",
        "pt",
    ]
    if pt_source != "auto":
        if pt_source not in obs.columns:
            raise KeyError(f"pt-source '{pt_source}' not in obs: {list(obs.columns)}")
        return obs[pt_source].astype(float)

    for c in candidates:
        if c in obs.columns:
            print(f"[INFO] Using obs['{c}'] as pseudotime")
            return obs[c].astype(float)

    if "umap_1" in obs.columns:
        print("[WARN] No pseudotime column; using normalized umap_1 as proxy")
        u = obs["umap_1"].astype(float).to_numpy()
        u = (u - u.min()) / (u.max() - u.min() + 1e-9)
        return pd.Series(u, index=obs.index)

    raise ValueError(
        f"No pseudotime found in obs. Columns: {list(obs.columns)}. "
        "Run Dynamo first (dyn.tl.dynamics + reduceDimension) or pass --pt-source."
    )


def export_velocity_truth(
    adata,
    layer: str,
    pt: pd.Series,
    gene_transform: str,
    quantile: float = 0.2,
) -> Optional[pd.DataFrame]:
    if layer not in adata.layers:
        print(f"[WARN] velocity layer '{layer}' missing; skip dynamo_velocity_truth.csv")
        return None

    V = adata.layers[layer]
    if hasattr(V, "toarray"):
        V = V.toarray()
    V = np.asarray(V, dtype=np.float32)

    lo, hi = np.quantile(pt.to_numpy(), [quantile, 1 - quantile])
    early = pt <= lo
    late = pt >= hi
    # gene-wise mean velocity in late minus early (direction along trajectory)
    v_early = V[early].mean(axis=0)
    v_late = V[late].mean(axis=0)
    delta_v = v_late - v_early

    return pd.DataFrame(
        {"gene": transform_genes(adata.var_names, gene_transform), "dynamo_velocity": delta_v}
    )


def main():
    global args
    args = parse_args()

    out_ds = Path(args.out_root) / args.name
    chip_dir = out_ds / "CHIP"
    chip_dir.mkdir(parents=True, exist_ok=True)

    import scanpy as sc

    adata = sc.read_h5ad(args.h5ad)
    adata.obs_names_make_unique()

    if args.layer == "X" or args.layer.lower() == "none":
        X = adata.X
    elif args.layer in adata.layers:
        X = adata.layers[args.layer]
    else:
        raise KeyError(f"Layer '{args.layer}' not found. Available: {list(adata.layers.keys())}")

    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    # AnnData stores (cells, genes); scGRN-Bench CHIP CSV expects (genes, cells).
    if X.shape != (adata.n_obs, adata.n_vars):
        raise ValueError(f"Unexpected X shape {X.shape}, expected {(adata.n_obs, adata.n_vars)}")
    X = X.T

    genes = transform_genes(adata.var_names, args.gene_transform)
    cells = adata.obs_names.astype(str).tolist()

    expr = pd.DataFrame(X, index=genes, columns=cells)
    expr_path = chip_dir / f"{args.name}_chip_matched-ExpressionData.csv"
    expr.to_csv(expr_path)
    print(f"[OK] Expression -> {expr_path}  shape={expr.shape}")

    pt = pick_pseudotime(adata.obs, args.pt_source)
    pt_df = pd.DataFrame({"cell": pt.index.astype(str), "pt": pt.values})
    pt_path = out_ds / "PseudoTime.csv"
    pt_df.to_csv(pt_path, index=False)
    print(f"[OK] Pseudotime -> {pt_path}")

    if args.velocity_layer:
        vdf = export_velocity_truth(adata, args.velocity_layer, pt, args.gene_transform)
        if vdf is not None:
            vpath = out_ds / "dynamo_velocity_truth.csv"
            vdf.to_csv(vpath, index=False)
            print(f"[OK] Dynamo velocity truth -> {vpath}")

    if args.write_datasets_json:
        json_path = Path(args.out_root) / "datasets_dynamo.json"
        cfg = {}
        if json_path.exists():
            with open(json_path, encoding="utf-8") as f:
                cfg = json.load(f)
        cfg[args.name] = {
            "expr_csv": str(expr_path),
            "pt_csv": str(pt_path),
            "species": args.species,
        }
        if args.velocity_layer:
            cfg[args.name]["dynamo_velocity_csv"] = str(out_ds / "dynamo_velocity_truth.csv")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"[OK] datasets_dynamo.json updated -> {json_path}")


if __name__ == "__main__":
    main()
