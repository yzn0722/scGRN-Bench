#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare foundation-model pred_delta (from scgpt_dyn output) vs Dynamo velocity truth.

Requires:
  - dynamo_velocity_truth.csv from prepare_dynamo_for_fm.py
  - gene_delta_compare.csv or per_gene_final_changes.csv from FM run (if available)

Minimal use: pass pred_delta and true arrays via numpy exports from your run.

Example after scGPT run:
  python evaluate_fm_vs_dynamo_velocity.py \\
    --dynamo-velocity-csv data/dynamo_export/zebrafish_dyn/dynamo_velocity_truth.csv \\
    --fm-gene-delta data/dynamo_export/results/scgpt/zebrafish/gene_delta_compare.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="FM vs Dynamo velocity direction agreement")
    p.add_argument("--dynamo-velocity-csv", required=True)
    p.add_argument("--fm-gene-delta", required=True, help="CSV with gene + pred_delta columns")
    p.add_argument("--top-percent", type=int, default=30)
    p.add_argument("--out-json", default="", help="Save summary JSON")
    return p.parse_args()


def direction_accuracy(pred: np.ndarray, true: np.ndarray, top_percent: int) -> dict:
    n = len(true)
    top_n = max(int(n * top_percent / 100), 1)
    idx = np.argsort(np.abs(true))[::-1][:top_n]
    agree = np.sign(pred[idx]) == np.sign(true[idx])
    # exclude near-zero true
    mask = np.abs(true[idx]) > 1e-8
    acc = float(agree[mask].mean()) if mask.any() else float("nan")
    return {
        "top_percent": top_percent,
        "n_eval": int(mask.sum()),
        "direction_accuracy": acc,
    }


def main():
    args = parse_args()
    dyn = pd.read_csv(args.dynamo_velocity_csv)
    fm = pd.read_csv(args.fm_gene_delta)

    dyn = dyn.rename(columns={dyn.columns[0]: "gene"})
    if "gene" not in fm.columns:
        fm = fm.rename(columns={fm.columns[0]: "gene"})

    pred_col = None
    for c in ("pred_delta", "Pred_Delta", "final_pred_delta", "delta_pred"):
        if c in fm.columns:
            pred_col = c
            break
    if pred_col is None:
        raise KeyError(f"No pred_delta column in {list(fm.columns)}")

    merged = dyn.merge(fm[["gene", pred_col]], on="gene", how="inner")
    print(f"Matched genes: {len(merged)} / dynamo {len(dyn)} / fm {len(fm)}")

    pred = merged[pred_col].to_numpy(dtype=float)
    true = merged["dynamo_velocity"].to_numpy(dtype=float)
    pred = np.nan_to_num(pred, nan=0.0)
    true = np.nan_to_num(true, nan=0.0)

    res = direction_accuracy(pred, true, args.top_percent)
    res["n_genes_matched"] = len(merged)
    print(json.dumps(res, indent=2))

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"Saved -> {args.out_json}")


if __name__ == "__main__":
    main()
