#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enrich existing dynamo_export folder with cell metadata + dataset config.

Exports (aligned to CHIP cell order):
  CellType.csv   (cell, cell_type)
  Time.csv       (cell, time)  [optional duplicate of obs time]

Updates datasets json with h5ad path, GRN proxy, driver TF labels.

Example:
  python enrich_dynamo_export.py \\
    --h5ad /mnt/10T/yzn/dynamo-release/results/hematopoiesis_raw/hematopoiesis_processed.h5ad \\
    --export-dir data/dynamo_export/hematopoiesis_dyn_vel1524 \\
    --dataset-key hematopoiesis_dyn_vel1524 \\
    --datasets-json data/dynamo_export/datasets_dynamo_vel1524_full.json \\
    --grn-proxy-tsv /mnt/10T/yzn/benchmark_GRN/evl_omipath/output_embhidden500/scgpt/scGPT_hESC.tsv \\
    --build-filtered-grn
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, Set

import pandas as pd


DEFAULT_H5AD = "/mnt/10T/yzn/dynamo-release/results/hematopoiesis_raw/hematopoiesis_processed.h5ad"
DEFAULT_GRN_PROXY = "/mnt/10T/yzn/benchmark_GRN/evl_omipath/output_embhidden500/scgpt/scGPT_hESC.tsv"
DEFAULT_DRIVERS = "/mnt/10T/yzn/scGRN-Bench/data/dynamo_export/driver_tfs_hematopoiesis.json"


def parse_args():
    p = argparse.ArgumentParser(description="Export cell metadata and enrich datasets json.")
    p.add_argument("--h5ad", default=DEFAULT_H5AD, type=str)
    p.add_argument("--export-dir", required=True, type=str, help="Existing dynamo_export/<name>/ folder.")
    p.add_argument("--dataset-key", required=True, type=str, help="Key in datasets json.")
    p.add_argument("--datasets-json", required=True, type=str, help="Output datasets json path.")
    p.add_argument("--cell-type-col", default="cell_type", type=str)
    p.add_argument("--time-col", default="time", type=str)
    p.add_argument("--species", default="human", type=str)
    p.add_argument("--grn-proxy-tsv", default=DEFAULT_GRN_PROXY, type=str)
    p.add_argument("--grn-proxy-source", default="hESC_embhidden500", type=str)
    p.add_argument("--driver-tfs-json", default=DEFAULT_DRIVERS, type=str)
    p.add_argument(
        "--build-filtered-grn",
        action="store_true",
        help="Write grn/<dataset>_grn_vel1524.tsv filtered to export genes.",
    )
    p.add_argument("--grn-topk-per-target", type=int, default=50)
    return p.parse_args()


def chip_cells(export_dir: Path, dataset_key: str) -> pd.Index:
    chip_dir = export_dir / "CHIP"
    csvs = list(chip_dir.glob("*_chip_matched-ExpressionData.csv"))
    if not csvs:
        raise FileNotFoundError(f"No expression CSV under {chip_dir}")
    expr = pd.read_csv(csvs[0], index_col=0, nrows=1)
    return expr.columns.astype(str)


def export_metadata(h5ad_path: str, cell_list: pd.Index, cell_type_col: str, time_col: str, out_dir: Path):
    import scanpy as sc

    adata = sc.read_h5ad(h5ad_path)
    adata.obs_names_make_unique()
    cells = pd.Index([str(c) for c in cell_list if str(c) in adata.obs_names])
    if len(cells) == 0:
        raise ValueError("No overlapping cells between CHIP export and h5ad.")

    obs = adata.obs.loc[cells]
    if cell_type_col not in obs.columns:
        raise KeyError(f"{cell_type_col} not in adata.obs: {list(obs.columns)}")
    if time_col not in obs.columns:
        raise KeyError(f"{time_col} not in adata.obs: {list(obs.columns)}")

    ct = pd.DataFrame({"cell": cells.astype(str), "cell_type": obs[cell_type_col].astype(str).values})
    ct_path = out_dir / "CellType.csv"
    ct.to_csv(ct_path, index=False)

    tm = pd.DataFrame({"cell": cells.astype(str), "time": obs[time_col].astype(float).values})
    tm_path = out_dir / "Time.csv"
    tm.to_csv(tm_path, index=False)

    print(f"[OK] CellType -> {ct_path}  ({ct['cell_type'].nunique()} types)")
    print(f"[OK] Time     -> {tm_path}  ({tm['time'].nunique()} timepoints)")
    print(ct["cell_type"].value_counts().to_string())
    return ct_path, tm_path


def filter_grn_to_genes(grn_tsv: Path, genes: Set[str], out_path: Path, topk_per_target: int = 50) -> Path:
    df = pd.read_csv(grn_tsv, sep="\t")
    if "Gene1" not in df.columns or "Gene2" not in df.columns:
        raise ValueError(f"GRN TSV needs Gene1/Gene2: {grn_tsv}")
    wcol = "EdgeWeight" if "EdgeWeight" in df.columns else df.columns[-1]
    df = df[df["Gene1"].astype(str).isin(genes) & df["Gene2"].astype(str).isin(genes)].copy()
    df[wcol] = pd.to_numeric(df[wcol], errors="coerce").fillna(0.0)
    df = df[df[wcol] > 0]
    df = df.sort_values(wcol, ascending=False)
    df = df.groupby("Gene2", sort=False).head(topk_per_target).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"[OK] Filtered GRN -> {out_path}  edges={len(df)} genes={len(genes)}")
    return out_path


def main():
    args = parse_args()
    export_dir = Path(args.export_dir)
    if not export_dir.is_dir():
        raise FileNotFoundError(export_dir)

    cells = chip_cells(export_dir, args.dataset_key)
    ct_path, tm_path = export_metadata(
        args.h5ad, cells, args.cell_type_col, args.time_col, export_dir
    )

    expr_csv = next((export_dir / "CHIP").glob("*_chip_matched-ExpressionData.csv"))
    pt_csv = export_dir / "PseudoTime.csv"
    vel_csv = export_dir / "dynamo_velocity_truth.csv"
    genes = set(pd.read_csv(expr_csv, index_col=0).index.astype(str))
    n_genes = len(genes)

    cfg_entry = {
        "expr_csv": str(expr_csv),
        "pt_csv": str(pt_csv),
        "cell_type_csv": str(ct_path),
        "time_csv": str(tm_path),
        "h5ad": str(args.h5ad),
        "species": args.species,
        "cell_type_col": args.cell_type_col,
        "time_col": args.time_col,
        "grn_proxy_tsv": str(args.grn_proxy_tsv),
        "grn_proxy_source": args.grn_proxy_source,
        "driver_tfs_json": str(args.driver_tfs_json),
    }
    if vel_csv.is_file():
        cfg_entry["dynamo_velocity_csv"] = str(vel_csv)
    cfg_entry["n_dynamic_genes"] = n_genes

    filtered_grn = None
    if args.build_filtered_grn:
        grn_out = export_dir / "grn" / f"{args.dataset_key}_grn_vel1524.tsv"
        filtered_grn = filter_grn_to_genes(
            Path(args.grn_proxy_tsv), genes, grn_out, args.grn_topk_per_target
        )
        cfg_entry["grn_filtered_tsv"] = str(filtered_grn)

    json_path = Path(args.datasets_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    all_cfg = {}
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            all_cfg = json.load(f)
    all_cfg[args.dataset_key] = cfg_entry
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_cfg, f, indent=2)
    print(f"[OK] datasets json -> {json_path}")


if __name__ == "__main__":
    main()
