#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect per-TF static metrics for all models × extractions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

from tf_static.model_registry import EXTRACTIONS, MODELS, resolve_gt_path, resolve_pred_path
from tf_static.plot_tf_jaccard_raincloud_all_models import column_key
from tf_static.utils import classify_tf_edges, filter_prediction, load_gt_network, per_tf_edge_sets


def collect_per_tf_long(
    dataset: str,
    gt_source: str = "STRING",
    models: List[str] | None = None,
    extractions: List[str] | None = None,
    evl_root: Path | None = None,
    input_root: Path | None = None,
) -> pd.DataFrame:
    from tf_static.model_registry import EVL_ROOT, INPUT_ROOT

    models = list(models or MODELS)
    extractions = list(extractions or EXTRACTIONS)
    evl_root = evl_root or EVL_ROOT
    input_root = input_root or INPUT_ROOT

    gt_path = resolve_gt_path(gt_source, dataset, input_root)
    gt_gene1, gt_all, gt_n, _, gt_by_tf = load_gt_network(gt_path)

    # GT out-degree (shared across methods)
    gt_deg = pd.DataFrame(
        [{"TF": tf, "gt_outdegree": len(tg)} for tf, tg in gt_by_tf.items()]
    )

    rows = []
    for model in models:
        for ext in extractions:
            fp = resolve_pred_path(model, ext, dataset, evl_root)
            if fp is None:
                continue
            pred = filter_prediction(fp, gt_gene1, gt_all, gt_n)
            met = per_tf_edge_sets(pred, gt_by_tf)
            if met.empty:
                continue
            for _, r in met.iterrows():
                tf = r["TF"]
                tp, fp_set, fn = classify_tf_edges(tf, pred, gt_by_tf)
                pred_n = int(r["pred_outdegree"])
                fp_rate = len(fp_set) / pred_n if pred_n > 0 else float("nan")
                fn_rate = len(fn) / int(r["gt_outdegree"]) if int(r["gt_outdegree"]) > 0 else float("nan")
                rows.append(
                    {
                        "TF": tf,
                        "model": model,
                        "extraction": ext,
                        "column": column_key(model, ext),
                        "jaccard": float(r["jaccard"]),
                        "f1": float(r["f1"]),
                        "precision": float(r["precision"]),
                        "recall": float(r["recall"]),
                        "pred_outdegree": pred_n,
                        "gt_outdegree": int(r["gt_outdegree"]),
                        "hits": int(r["hits"]),
                        "n_fp": len(fp_set),
                        "n_fn": len(fn),
                        "fp_rate": fp_rate,
                        "fn_rate": fn_rate,
                        "gt_source": gt_source,
                        "dataset": dataset,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.merge(gt_deg, on="TF", how="left", suffixes=("", "_gt"))
    if "gt_outdegree_gt" in df.columns:
        df["gt_outdegree"] = df["gt_outdegree"].fillna(df["gt_outdegree_gt"]).astype(int)
        df = df.drop(columns=["gt_outdegree_gt"])
    return df


def assign_hub_strata(df: pd.DataFrame) -> pd.DataFrame:
    """Add hub_tier (top20/bottom80) and degree_quintile (Q1–Q5) per TF from GT out-degree."""
    if df.empty:
        return df
    deg = df.groupby("TF")["gt_outdegree"].first().reset_index()
    q80 = deg["gt_outdegree"].quantile(0.80)
    deg["hub_tier"] = deg["gt_outdegree"].apply(
        lambda x: "Hub (top 20%)" if x >= q80 else "Specialist (bottom 80%)"
    )
    try:
        deg["degree_quintile"] = pd.qcut(
            deg["gt_outdegree"],
            q=5,
            labels=["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"],
            duplicates="drop",
        )
    except ValueError:
        deg["degree_quintile"] = "Q_all"
    return df.merge(deg[["TF", "hub_tier", "degree_quintile"]], on="TF", how="left")
