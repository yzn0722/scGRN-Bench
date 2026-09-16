#!/usr/bin/env python3
"""Compare pairwise evo-velocity vs per-cell MLM velocity metrics."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


def direction_accuracy(pred, true, top_percent=30):
    n = len(true)
    top_n = max(int(n * top_percent / 100), 1)
    idx = np.argsort(np.abs(true))[::-1][:top_n]
    mask = np.abs(true[idx]) > 1e-8
    agree = np.sign(pred[idx][mask]) == np.sign(true[idx][mask])
    return float(agree.mean()) if mask.any() else float("nan")


def eval_velocity(vel, cells, genes, h5ad_path, layer, pt_csv):
    adata = sc.read_h5ad(h5ad_path)
    adata.obs_names_make_unique()
    pt_df = pd.read_csv(pt_csv).set_index("cell")
    pt_df.index = pt_df.index.astype(str)
    pt = np.array([float(pt_df.loc[c, "pt"]) if c in pt_df.index else np.nan for c in cells])
    early = pt <= np.nanquantile(pt, 0.2)
    late = pt >= np.nanquantile(pt, 0.8)

    gene_to_idx = {g: i for i, g in enumerate(adata.var_names.astype(str))}
    gi = [gene_to_idx[g] for g in genes if g in gene_to_idx]
    common = [c for c in cells if c in adata.obs_names]
    sub = adata[common, gi]
    v = sub.layers[layer]
    if sp.issparse(v):
        v = v.toarray()
    vel_true = np.nan_to_num(np.asarray(v, dtype=np.float32), nan=0.0)

    idx = [cells.index(c) for c in common]
    vel = vel[idx]
    true_delta = vel_true[late].mean(axis=0) - vel_true[early].mean(axis=0)
    pred_delta = vel[late].mean(axis=0) - vel[early].mean(axis=0)
    return {
        "direction_accuracy_top30": direction_accuracy(pred_delta, true_delta, 30),
        "n_eval_genes": int((np.abs(true_delta) > 1e-8).sum()),
    }


def main():
    outdir = Path("outputs/dynamo_fm_velocity/pairwise_evo_hematopoiesis")
    cfg_path = Path("data/dynamo_export/datasets_dynamo_vel1524_only.json")
    cfg = json.loads(cfg_path.read_text())["hematopoiesis_dyn_vel1524"]

    pairwise = np.load(outdir / "velocity_field.npz", allow_pickle=True)
    percell = np.load(outdir / "percell_mlm_velocity_field.npz", allow_pickle=True)

    cells = pairwise["cells"].astype(str).tolist()
    genes = pairwise["genes"].astype(str).tolist()

    m_pair = json.loads((outdir / "metrics.json").read_text())
    m_per = eval_velocity(
        percell["vel_cell"],
        percell["cells"].astype(str).tolist(),
        percell["genes"].astype(str).tolist(),
        cfg["h5ad"],
        "velocity_alpha_minus_gamma_s",
        cfg["pt_csv"],
    )

    comparison = {
        "pairwise_evo_velocity": {
            "direction_accuracy_top30": m_pair.get("direction_accuracy_top30"),
            "pseudotime_spearman": m_pair.get("pseudotime_spearman"),
            "n_directed_edges": m_pair.get("n_directed_edges"),
        },
        "percell_mlm_velocity": m_per,
    }
    (outdir / "comparison_metrics.json").write_text(json.dumps(comparison, indent=2))
    print(json.dumps(comparison, indent=2))

    labels = ["Pairwise evo-velocity", "Per-cell MLM velocity"]
    vals = [
        comparison["pairwise_evo_velocity"]["direction_accuracy_top30"],
        comparison["percell_mlm_velocity"]["direction_accuracy_top30"],
    ]

    fig, ax = plt.subplots(figsize=(4.5, 4))
    bars = ax.bar(labels, vals, color=["#4472C4", "#ED7D31"], width=0.55)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Direction accuracy (Top 30% genes)")
    ax.set_title("scGPT velocity methods vs Dynamo")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig_dir = outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "method_comparison.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "method_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison figure -> {fig_dir / 'method_comparison.png'}")


if __name__ == "__main__":
    main()
