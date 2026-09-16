#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scFoundation zero-shot RNA velocity on veloBench four CBDir datasets.

Single-GPU workflow (default, uses GPU 0):
  conda activate singlecell
  cd /mnt/10T/yzn/scGRN-Bench

  python -u src/GRN_inferance/dyn/run_velobench_scfoundation_zero_shot.py \\
    --dataset HumanBrain HumanHSPC PBMC68k --skip-export --resume

One dataset:
  python -u src/GRN_inferance/dyn/run_velobench_scfoundation_zero_shot.py \\
    --dataset HumanBoneMarrow --skip-export --resume
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
VELO_H5AD_DIR = Path("/mnt/10T/yzn/veloBench/results/Dynamo/CB_IC")
DEFAULT_SCF_ROOT = Path("/mnt/10T/yzn/scFoundation-main/model")
DEFAULT_SCF_CKPT = DEFAULT_SCF_ROOT / "models" / "models.ckpt"
DEFAULT_SCF_GENE_TSV = DEFAULT_SCF_ROOT / "OS_scRNA_gene_index.19264.tsv"
DEFAULT_OUT = ROOT / "outputs/dynamo_fm_velocity/velobench_four_scfoundation"
DEFAULT_REUSE_EXPORT = ROOT / "outputs/dynamo_fm_velocity/velobench_four_scgpt"
EDGES_JSON = ROOT / "data/dynamo_export/velobench_cbdir_edges.json"
AUTO_SHARD_CELL_THRESHOLD = 10000
AUTO_SHARD_CELLS_PER = 8000

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
    p = argparse.ArgumentParser(description="scFoundation zero-shot on veloBench four datasets")
    p.add_argument(
        "--dataset",
        nargs="*",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
    )
    p.add_argument("--h5ad-dir", default=str(VELO_H5AD_DIR), type=str)
    p.add_argument("--out-root", default=str(DEFAULT_OUT), type=str)
    p.add_argument("--reuse-export-from", default=str(DEFAULT_REUSE_EXPORT), type=str)
    p.add_argument("--scf-root", default=str(DEFAULT_SCF_ROOT), type=str)
    p.add_argument("--scf-ckpt", default=str(DEFAULT_SCF_CKPT), type=str)
    p.add_argument("--scf-gene-tsv", default=str(DEFAULT_SCF_GENE_TSV), type=str)
    p.add_argument("--gen-iters", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="CUDA device index (default 0). Sets CUDA_VISIBLE_DEVICES for inference.",
    )
    p.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Manual cell shards for large datasets (use with --shard-id). 0 = auto for >10k cells.",
    )
    p.add_argument("--shard-id", type=int, default=0, help="Shard index in [0, num-shards).")
    p.add_argument(
        "--auto-shard-cells",
        type=int,
        default=AUTO_SHARD_CELLS_PER,
        help=f"Cells per shard when auto-sharding (default {AUTO_SHARD_CELLS_PER}).",
    )
    p.add_argument(
        "--no-auto-shard",
        action="store_true",
        help="Disable auto-sharding for large datasets (not recommended for PBMC68k).",
    )
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint / skip finished shards.")
    p.add_argument("--torch-compile", action="store_true")
    p.add_argument("--resample-mask", action="store_true", help="Re-enable per-iter MAE mask resampling (slower).")
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--skip-scfoundation", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--merge-only", action="store_true", help="Only merge shard npz into velocity_field.npz")
    p.add_argument("--scf-env", default="singlecell", type=str)
    p.add_argument("--eval-env", default="dynamo-env", type=str)
    return p.parse_args()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def require_cuda_gpu() -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required in the scFoundation env (singlecell).") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        raise RuntimeError(
            "CUDA GPU required for scFoundation inference. "
            "Check: conda run -n singlecell python -c "
            "\"import torch; print(torch.cuda.is_available(), torch.cuda.device_count())\""
        )


def count_cells(expr_csv: Path) -> int:
    return int(pd.read_csv(expr_csv, index_col=0, nrows=0).shape[1])


def resolve_export(name: str, args: argparse.Namespace) -> tuple[Path, Path]:
    local = Path(args.out_root) / name / "export"
    local_json = local / f"datasets_{name}.json"
    if local_json.exists():
        return local, local_json
    if args.reuse_export_from:
        shared = Path(args.reuse_export_from) / name / "export"
        shared_json = shared / f"datasets_{name}.json"
        if shared_json.exists():
            log(f"  [export] Reusing export from {shared}")
            return shared, shared_json
    return local, local_json


def export_dataset(name: str, cfg: dict, h5ad_path: Path, export_dir: Path) -> None:
    import sys as _sys

    dyn_dir = ROOT / "src/GRN_inferance/dyn"
    if str(dyn_dir) not in _sys.path:
        _sys.path.insert(0, str(dyn_dir))
    from run_velobench_scgpt_zero_shot import export_dataset as scgpt_export

    scgpt_export(name, cfg, h5ad_path, export_dir)


def shard_ranges(n_cells: int, num_shards: int) -> list[tuple[int, int]]:
    num_shards = max(1, num_shards)
    size = (n_cells + num_shards - 1) // num_shards
    return [(i * size, min((i + 1) * size, n_cells)) for i in range(num_shards)]


def shard_npz_path(final_npz: Path, shard_id: int) -> Path:
    return final_npz.parent / f"{final_npz.stem}.shard{shard_id:02d}.npz"


def monolithic_checkpoint_path(final_npz: Path) -> Path:
    return final_npz.with_suffix(final_npz.suffix + ".ckpt.npz")


def resolve_num_shards(n_cells: int, final_npz: Path, args: argparse.Namespace) -> int:
    if args.num_shards > 1:
        return args.num_shards
    if args.num_shards == 0 or (not args.no_auto_shard and n_cells > AUTO_SHARD_CELL_THRESHOLD):
        if args.resume and monolithic_checkpoint_path(final_npz).exists():
            log(
                f"  [auto-shard] monolithic checkpoint exists -> continue single job "
                f"({monolithic_checkpoint_path(final_npz)})"
            )
            return 1
        n = max(2, (n_cells + args.auto_shard_cells - 1) // args.auto_shard_cells)
        log(f"  [auto-shard] {n_cells} cells -> {n} shards (~{args.auto_shard_cells} cells each)")
        return n
    return 1


def build_scfoundation_cmd(
    expr_csv: Path,
    out_npz: Path,
    args: argparse.Namespace,
    cell_start: int,
    cell_end: int,
) -> tuple[list[str], dict[str, str]]:
    script = ROOT / "src/GRN_inferance/dyn/scfoundation_per_cell_velocity_only.py"
    cmd = [
        "conda",
        "run",
        "-n",
        args.scf_env,
        "--no-capture-output",
        "python",
        "-u",
        str(script),
        "--expr-csv",
        str(expr_csv),
        "--out-npz",
        str(out_npz),
        "--scf-root",
        str(args.scf_root),
        "--scf-ckpt",
        str(args.scf_ckpt),
        "--scf-gene-tsv",
        str(args.scf_gene_tsv),
        "--gen-iters",
        str(args.gen_iters),
        "--batch-size",
        str(args.batch_size),
        "--cell-start",
        str(cell_start),
        "--cell-end",
        str(cell_end),
    ]
    if not args.resample_mask:
        cmd.append("--no-resample-mask")
    if args.resume:
        cmd.append("--resume")
    if args.torch_compile:
        cmd.append("--torch-compile")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    return cmd, env


def run_cmd(cmd: list[str], env: dict[str, str]) -> None:
    log(f"  [scFoundation] CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '?')} {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)


def merge_shards(final_npz: Path, shard_paths: list[Path]) -> None:
    parts_vel, parts_cells, genes_ref = [], [], None
    for sp in sorted(shard_paths):
        if not sp.exists():
            raise FileNotFoundError(f"Missing shard: {sp}")
        d = np.load(sp)
        parts_vel.append(d["vel_cell"].astype(np.float32))
        parts_cells.extend(d["cells"].astype(str).tolist())
        genes = d["genes"].astype(str).tolist()
        if genes_ref is None:
            genes_ref = genes
        elif genes_ref != genes:
            raise ValueError(f"Gene list mismatch in {sp}")
    vel = np.vstack(parts_vel)
    final_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(final_npz, vel_cell=vel, cells=np.array(parts_cells), genes=np.array(genes_ref))
    log(f"  [merge] {len(shard_paths)} shards -> {final_npz} shape={vel.shape}")


def run_scfoundation_sharded(
    name: str,
    expr_csv: Path,
    final_npz: Path,
    args: argparse.Namespace,
) -> None:
    n_cells = count_cells(expr_csv)
    num_shards = resolve_num_shards(n_cells, final_npz, args)

    if args.num_shards > 1:
        shard_ids = [args.shard_id]
        if args.shard_id < 0 or args.shard_id >= num_shards:
            raise ValueError(f"shard-id must be in [0, {num_shards})")
    else:
        shard_ids = list(range(num_shards))

    ranges = shard_ranges(n_cells, num_shards)
    jobs: list[tuple[int, int, int, Path]] = []
    for sid in shard_ids:
        start, end = ranges[sid]
        if start >= end:
            continue
        out_npz = shard_npz_path(final_npz, sid) if num_shards > 1 else final_npz
        if args.resume and out_npz.exists() and num_shards > 1:
            log(f"  [skip] shard {sid} exists -> {out_npz}")
            continue
        jobs.append((sid, start, end, out_npz))

    if not jobs and num_shards > 1:
        log("  [scFoundation] all shards present, merging only")
    else:
        for sid, start, end, out_npz in jobs:
            cmd, env = build_scfoundation_cmd(expr_csv, out_npz, args, start, end)
            run_cmd(cmd, env)

    if num_shards > 1:
        shard_paths = [shard_npz_path(final_npz, i) for i in range(num_shards)]
        merge_shards(final_npz, shard_paths)


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
        "scfoundation",
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
        f"scFoundation_zero_shot_{dataset}",
        "--outdir",
        str(outdir),
        "--save-h5ad",
    ]
    log(f"  [eval] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def run_one(name: str, cfg: dict, args: argparse.Namespace) -> dict:
    out_root = Path(args.out_root)
    ds_root = out_root / name
    h5ad_path = Path(args.h5ad_dir) / f"{name}_AnnData_Forscore.h5ad"
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Missing h5ad: {h5ad_path}")

    log(f"\n{'=' * 60}\n{name} ({cfg['velobench_id']})\n{'=' * 60}")

    if args.skip_export:
        export_dir, ds_json = resolve_export(name, args)
        if not ds_json.exists():
            raise FileNotFoundError(f"Export missing: {ds_json}")
    else:
        export_dir = ds_root / "export"
        if not (export_dir / f"datasets_{name}.json").exists():
            export_dataset(name, cfg, h5ad_path, export_dir)
        ds_json = export_dir / f"datasets_{name}.json"

    expr_csv = export_dir / "CHIP" / f"{name}_chip_matched-ExpressionData.csv"
    if not expr_csv.exists():
        raise FileNotFoundError(f"Expression CSV missing: {expr_csv}")

    npz_path = ds_root / "scfoundation_zero_shot" / name / "velocity_field.npz"
    metrics_dir = ds_root / "metrics_scfoundation"

    if args.merge_only:
        n_shards = resolve_num_shards(count_cells(expr_csv), npz_path, args)
        merge_shards(npz_path, [shard_npz_path(npz_path, i) for i in range(n_shards)])
        return {"dataset": name, "status": "merged"}

    if not args.skip_scfoundation:
        if npz_path.exists() and not args.resume:
            log(f"  [scFoundation] Removing stale npz -> {npz_path}")
            npz_path.unlink()
        run_scfoundation_sharded(name, expr_csv, npz_path, args)
    elif not npz_path.exists():
        raise FileNotFoundError(f"velocity_field.npz missing: {npz_path}")

    summary = {"dataset": name, "status": "exported+scfoundation"}
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
    if not args.skip_scfoundation and not args.merge_only:
        require_cuda_gpu()
        log(f"  [gpu] single-GPU mode: CUDA_VISIBLE_DEVICES={args.gpu_id}")
    Path(args.out_root).mkdir(parents=True, exist_ok=True)

    summaries, failures = [], []
    for name in args.dataset:
        try:
            summaries.append(run_one(name, DATASETS[name], args))
        except subprocess.CalledProcessError as exc:
            log(f"FAILED {name}: exit code {exc.returncode}")
            failures.append(name)
        except Exception as exc:
            log(f"FAILED {name}: {exc}")
            failures.append(name)

    if summaries:
        rows = [
            {
                "dataset": s.get("dataset", ""),
                "cbdir_mean": s.get("cbdir_mean"),
                "iccoh_mean": s.get("iccoh_mean"),
                "n_cells": s.get("n_cells"),
                "n_genes": s.get("n_genes"),
            }
            for s in summaries
        ]
        status_path = Path(args.out_root) / "scfoundation_velobench_status.csv"
        pd.DataFrame(rows).to_csv(status_path, index=False)
        log(f"\nStatus table -> {status_path}")
        print(pd.DataFrame(rows).to_string(index=False))

    if failures:
        log(f"\nFailed datasets: {', '.join(failures)}")
        sys.exit(1)

    log("\nAll done.")


if __name__ == "__main__":
    main()
