#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge veloBench metric runs into summary / per-edge comparison tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge benchmark metric result directories")
    p.add_argument(
        "--dirs",
        nargs="+",
        required=True,
        help="Result directories containing benchmark_metrics_summary.json",
    )
    p.add_argument("--outdir", required=True, type=str)
    p.add_argument("--label", nargs="*", default=[], help="Optional method labels (same order as --dirs)")
    return p.parse_args()


def load_run(run_dir: Path, label: str) -> dict:
    summary_path = run_dir / "benchmark_metrics_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    summary["run_dir"] = str(run_dir)
    summary["label"] = label or summary.get("method", run_dir.name)
    return summary


def main() -> None:
    args = parse_args()
    labels = args.label if args.label else [Path(d).name for d in args.dirs]
    if len(labels) != len(args.dirs):
        raise ValueError("--label count must match --dirs count")

    runs = [load_run(Path(d), lab) for d, lab in zip(args.dirs, labels)]

    summary_rows = []
    edge_frames = []
    ic_frames = []

    for run in runs:
        run_dir = Path(run["run_dir"])
        label = run["label"]
        summary_rows.append(
            {
                "method": label,
                "cbdir_mean": run.get("cbdir_mean"),
                "iccoh_mean": run.get("iccoh_mean"),
                "velocity_consistency_mean": run.get("velocity_consistency_mean"),
                "velocity_source": run.get("velocity_source"),
                "umap_projection": run.get("umap_projection"),
                "n_cells": run.get("n_cells"),
                "n_genes": run.get("n_genes"),
                "edge_preset": run.get("edge_preset_description"),
                "run_dir": run_dir,
            }
        )

        cb_path = run_dir / "CBDir_scores.csv"
        if cb_path.exists():
            cb = pd.read_csv(cb_path)
            cb = cb[cb["edge"] != "MEAN"].copy()
            cb["method"] = label
            edge_frames.append(cb)

        ic_path = run_dir / "ICCoh_scores.csv"
        if ic_path.exists():
            ic = pd.read_csv(ic_path)
            ic = ic[ic["cluster"] != "MEAN"].copy()
            ic["method"] = label
            ic_frames.append(ic)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "comparison_summary.csv", index=False)

    if edge_frames:
        edges = pd.concat(edge_frames, ignore_index=True)
        pivot = edges.pivot_table(index="edge", columns="method", values="cbdir_mean", aggfunc="first")
        pivot = pivot.reset_index()
        pivot.to_csv(outdir / "comparison_cbdir_per_edge.csv", index=False)

    if ic_frames:
        ic_all = pd.concat(ic_frames, ignore_index=True)
        pivot_ic = ic_all.pivot_table(index="cluster", columns="method", values="iccoh_mean", aggfunc="first")
        pivot_ic = pivot_ic.reset_index()
        pivot_ic.to_csv(outdir / "comparison_iccoh_per_cluster.csv", index=False)

    with open(outdir / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump({"runs": runs}, f, indent=2, ensure_ascii=False)

    print(summary_df.to_string(index=False))
    print(f"\nSaved -> {outdir}")


if __name__ == "__main__":
    main()
