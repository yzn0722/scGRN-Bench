#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geneformer (4-head) attention GRN analysis — all outputs under one folder.

Output layout:
  output/geneformer_head_analysis/
    geneformer_hESC/
      supplement/          # Exp1–4 (AUPRC bar, routing, ablation, GO stability)
      chip_auprc/          # full CHIP AUPRC tables + bar
      chip_epr/            # label-free EPR (mean4/max4/fusion/routed)
      topology/            # 5 graph metrics per head
      statistics/          # Jaccard, Spearman, overlap, PCA
      go_enrich/           # per-head GO:BP
      hub_l2/              # in-hub vs TF-out bridge
    README.txt

Usage:
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig2/plot
  python run_geneformer_head_analysis.py
  python run_geneformer_head_analysis.py --dataset hESC --skip-modules
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
OUT_ROOT = _SCRIPT_DIR / "output" / "geneformer_head_analysis"
HEAD_ROOT = _SCRIPT_DIR.parent / "att_head" / "geneformer"
CHIP_ROOT = Path("/mnt/10T/yzn/benchmark_GRN/input_process/CHIP")
REF_ROOT = Path("/mnt/10T/yzn/benchmark_GRN/input_process")


def run_cmd(cmd: list[str], desc: str) -> None:
    print(f"\n{'='*60}\n[{desc}]\n  {' '.join(cmd)}\n{'='*60}")
    r = subprocess.run(cmd, cwd=_SCRIPT_DIR)
    if r.returncode != 0:
        raise RuntimeError(f"Failed: {desc} (exit {r.returncode})")


def write_readme(tag: str, out_root: Path) -> None:
    readme = out_root / "README.txt"
    readme.write_text(
        f"""Geneformer multi-head GRN analysis (hESC)
==========================================
Model: Geneformer 6L, 4 attention heads (H0–H3)
Dataset tag: {tag}

Subfolders under {tag}/:
  supplement/     — single-head AUPRC, routing TF counts, ablation, GO stability
  chip_auprc/     — CHIP TF-centric AUPRC (heads + mean4/max4)
  chip_epr/       — label-free EPR
  topology/       — graph metrics (STRING universe)
  statistics/     — head redundancy (Jaccard, Spearman, …)
  go_enrich/      — GO:BP per head (top-100 genes)
  hub_l2/         — in-hub vs TF-out-strength

Head TSV source: {HEAD_ROOT}

Regenerate:
  python run_geneformer_head_analysis.py --dataset hESC
""",
        encoding="utf-8",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Run full Geneformer head analysis pipeline.")
    p.add_argument("--dataset", default="hESC")
    p.add_argument("--model", default="geneformer")
    p.add_argument("--head-root", default=str(HEAD_ROOT))
    p.add_argument("--chip-root", default=str(CHIP_ROOT))
    p.add_argument("--ref-root", default=str(REF_ROOT))
    p.add_argument("--ref-type", default="STRING", choices=["STRING", "CHIP"])
    p.add_argument("--skip-topology", action="store_true")
    p.add_argument("--skip-statistics", action="store_true")
    p.add_argument("--skip-go", action="store_true")
    p.add_argument("--skip-modules", action="store_true", help="Skip detect_att_modules (slow)")
    p.add_argument("--skip-hub-l2", action="store_true")
    args = p.parse_args()

    tag = f"{args.model}_{args.dataset}"
    base = OUT_ROOT / tag
    py = sys.executable
    hr = args.head_root

    base.mkdir(parents=True, exist_ok=True)

    # 1) Topology
    if not args.skip_topology:
        run_cmd(
            [
                py,
                "analyze_attention_heads.py",
                "--models",
                args.model,
                "--datasets",
                args.dataset,
                "--head-root",
                hr,
                "--ref-root",
                args.ref_root,
                "--ref-type",
                args.ref_type,
                "--output-dir",
                str(base / "topology"),
            ],
            "topology metrics",
        )

    # 2) Head statistics
    if not args.skip_statistics:
        run_cmd(
            [
                py,
                "stat_analyze_attention_heads.py",
                "--models",
                args.model,
                "--dataset",
                args.dataset,
                "--head-root",
                hr,
                "--ref-root",
                args.ref_root,
                "--ref-type",
                args.ref_type,
                "--output-dir",
                str(base / "statistics"),
            ],
            "head statistics",
        )

    # 3) CHIP AUPRC
    auprc_dir = base / "chip_auprc"
    run_cmd(
        [
            py,
            "eval_heads_chip_auprc.py",
            "--models",
            args.model,
            "--dataset",
            args.dataset,
            "--head-root",
            hr,
            "--chip-root",
            args.chip_root,
            "--output-dir",
            str(auprc_dir),
            "--stat-dir",
            str(base / "statistics"),
        ],
        "CHIP AUPRC",
    )

    # 4) Supplement figures (routing, ablation, GO stability)
    summary_csv = auprc_dir / tag / f"{tag}_auprc_summary.csv"
    run_cmd(
        [
            py,
            "plot_head_supplement_figures.py",
            "--models",
            args.model,
            "--dataset",
            args.dataset,
            "--head-root",
            hr,
            "--chip-root",
            args.chip_root,
            "--output-dir",
            str(base / "supplement"),
            "--auprc-summary",
            str(summary_csv),
        ],
        "supplement figures",
    )

    # 5) Label-free EPR
    run_cmd(
        [
            py,
            "eval_heads_chip_epr.py",
            "--models",
            args.model,
            "--datasets",
            args.dataset,
            "--head-root",
            hr,
            "--chip-root",
            args.chip_root,
            "--output-dir",
            str(base / "chip_epr"),
        ],
        "CHIP EPR",
    )

    # 6) GO enrichment
    if not args.skip_go:
        run_cmd(
            [
                py,
                "enrich_head_go_gprofiler.py",
                "--models",
                args.model,
                "--dataset",
                args.dataset,
                "--head-root",
                hr,
                "--chip-root",
                args.chip_root,
                "--output-dir",
                str(base / "go_enrich"),
            ],
            "GO enrichment",
        )

    # 7) Hub L2 bridge
    if not args.skip_hub_l2:
        run_cmd(
            [
                py,
                "plot_head_l2b_bridge.py",
                "--models",
                args.model,
                "--dataset",
                args.dataset,
                "--head-root",
                hr,
                "--chip-root",
                args.chip_root,
                "--output-dir",
                str(base / "hub_l2"),
            ],
            "hub in/out bridge",
        )

    # 8) Optional modules
    if not args.skip_modules:
        run_cmd(
            [
                py,
                "detect_att_modules.py",
                "--models",
                args.model,
                "--dataset",
                args.dataset,
                "--head-root",
                hr,
                "--chip-root",
                args.chip_root,
                "--output-dir",
                str(base / "att_modules"),
                "--max-edges-per-head",
                "200000",
            ],
            "attention modules",
        )

    write_readme(tag, OUT_ROOT)
    print(f"\n[DONE] Geneformer head analysis -> {base}")


if __name__ == "__main__":
    main()
