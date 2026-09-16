#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按模型批量生成与 scGPT 相同的 fig4 动态误差图（误差分层 + CHIP TF 靶对比）。

已支持
----
- scGPT: ``pre_scgpt/.../{dataset}_gene_result.csv``
- scFoundation: ``pre_scfoundation/.../{dataset}/gene_delta_compare.csv``

输出目录默认 ``error_biology/{model}/``。

示例
----
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig4

  # scFoundation — 与 scGPT 同版式六数据集图
  python3 plot_model_dynamic_figures.py --model scFoundation

  # scGPT（默认路径）
  python3 plot_model_dynamic_figures.py --model scGPT

  # 跳过 GO 富集（更快）
  python3 plot_model_dynamic_figures.py --model scFoundation --no-enrichment
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from gene_result_io import SCF_DEFAULT, SCGPT_DEFAULT, export_unified_gene_results

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS = ["hESC", "hHep", "mDC", "mHSC-E", "mHSC-GM", "mHSC-L"]

MODEL_CONFIG = {
    "scgpt": {
        "label": "scGPT",
        "gene_dir": SCGPT_DEFAULT,
        "glob": "*_gene_result.csv",
        "mapped_only": False,
    },
    "scfoundation": {
        "label": "scFoundation",
        "gene_dir": SCF_DEFAULT,
        "glob": "*/gene_delta_compare.csv",
        "mapped_only": True,
    },
}


def resolve_model(name: str) -> dict:
    key = name.lower().replace("-", "").replace("_", "")
    if key in ("scgpt", "gpt"):
        return MODEL_CONFIG["scgpt"]
    if key in ("scfoundation", "scf", "foundation"):
        return MODEL_CONFIG["scfoundation"]
    raise ValueError(f"Unknown model {name!r}; use scGPT or scFoundation")


def main() -> None:
    p = argparse.ArgumentParser(description="Model-parallel dynamic error biology figures")
    p.add_argument("--model", default="scFoundation")
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--no-enrichment", action="store_true")
    p.add_argument("--exclude-datasets", default="", help="e.g. mDC")
    args = p.parse_args()

    cfg = resolve_model(args.model)
    outdir = args.outdir or (SCRIPT_DIR / "error_biology" / cfg["label"].lower())
    outdir.mkdir(parents=True, exist_ok=True)

    excluded = {x.strip() for x in args.exclude_datasets.split(",") if x.strip()}
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip() and d.strip() not in excluded]

    # Unified gene_result copies (optional convenience)
    unified_dir = outdir / "gene_results"
    export_unified_gene_results(
        args.model,
        datasets,
        unified_dir,
        src_root=cfg["gene_dir"],
        mapped_only=cfg["mapped_only"],
    )
    print(f"Unified gene_result -> {unified_dir}")

    # 1) Per-dataset stacked + pie + GO
    cmd1 = [
        sys.executable,
        str(SCRIPT_DIR / "plot_dynamic_error_biology.py"),
        "--gene-dir",
        str(unified_dir),
        "--outdir",
        str(outdir),
        "--model-label",
        cfg["label"],
    ]
    if cfg["mapped_only"]:
        cmd1.append("--mapped-only")
    if args.no_enrichment:
        cmd1.append("--no-enrichment")
    for ds in datasets:
        subprocess.run(cmd1 + ["--dataset", ds], check=True)

    # 2) Six-dataset CHIP TF target panel
    cmd2 = [
        sys.executable,
        str(SCRIPT_DIR / "plot_chip_tf_target_error_all_datasets.py"),
        "--gene-dir",
        str(unified_dir),
        "--outdir",
        str(outdir),
        "--model-label",
        cfg["label"],
    ]
    if cfg["mapped_only"]:
        cmd2.append("--mapped-only")
    if excluded:
        cmd2 += ["--exclude-datasets", ",".join(sorted(excluded))]
    subprocess.run(cmd2, check=True)

    print(f"\nDone. Outputs under: {outdir}")
    print("  *_error_strat_stacked.png")
    print("  all_datasets_chip_tf_target_error_stacked.png")


if __name__ == "__main__":
    main()
