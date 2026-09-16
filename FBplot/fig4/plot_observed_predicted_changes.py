#!/usr/bin/env python3
"""Plot observed versus GRN-propagated next-iteration changes for three networks."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", required=True)
    p.add_argument("--expression-csv", required=True)
    p.add_argument("--vocab-json", required=True)
    p.add_argument("--attention-grn", required=True)
    p.add_argument("--embedding-grn", required=True)
    p.add_argument("--hidden-grn", required=True)
    p.add_argument("--attention-report", required=True)
    p.add_argument("--embedding-report", required=True)
    p.add_argument("--hidden-report", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--top-k", type=int, default=5000)
    p.add_argument("--transient-iters", type=int, default=8)
    p.add_argument("--max-points", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260904)
    return p.parse_args()


def load_network(path, allowed, gene_to_idx, top_k):
    frame = pd.read_csv(path, sep="\t", usecols=["Gene1", "Gene2", "EdgeWeight"])
    frame["Gene1"] = frame["Gene1"].astype(str).str.strip().str.upper()
    frame["Gene2"] = frame["Gene2"].astype(str).str.strip().str.upper()
    frame["EdgeWeight"] = pd.to_numeric(frame["EdgeWeight"], errors="coerce")
    frame = frame[
        frame.Gene1.isin(allowed)
        & frame.Gene2.isin(allowed)
        & np.isfinite(frame.EdgeWeight)
        & (frame.EdgeWeight > 0)
        & (frame.Gene1 != frame.Gene2)
    ]
    frame = (
        frame.sort_values("EdgeWeight", ascending=False)
        .drop_duplicates(["Gene1", "Gene2"])
        .head(top_k)
    )
    if len(frame) < top_k:
        raise ValueError(f"{path}: only {len(frame)} valid edges")

    # Match key_to_query_primary in run_weighted_grn_propagation.py.
    src = frame.Gene2.map(gene_to_idx).to_numpy(np.int32)
    dst = frame.Gene1.map(gene_to_idx).to_numpy(np.int32)
    weight = frame.EdgeWeight.to_numpy(np.float64)
    denom = np.bincount(dst, weights=np.abs(weight), minlength=len(gene_to_idx))
    weight = weight / np.maximum(denom[dst], EPS)
    valid = np.bincount(dst, minlength=len(gene_to_idx)) > 0
    return src, dst, weight, valid


def propagated_points(states, src, dst, weight, valid, n_lags):
    delta = np.abs(np.diff(states, axis=0))
    xs, ys = [], []
    for t in range(min(n_lags, len(delta) - 1)):
        predicted = np.zeros(states.shape[1], dtype=np.float64)
        np.add.at(predicted, dst, delta[t, src] * weight)
        observed = delta[t + 1]
        x = observed[valid]
        y = predicted[valid]
        # Percentile ranks directly visualize the Spearman association used in
        # the network-density panel, without confounding network-specific scale.
        xs.append(rankdata(x, method="average") / (len(x) + 1.0))
        ys.append(rankdata(y, method="average") / (len(y) + 1.0))
    return np.concatenate(xs), np.concatenate(ys)


def reported_metrics(path, top_k):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    block = data["analyses"][str(top_k)]["key_to_query_primary"]["early"]
    return block["windows"]["transient"]["observed"]


def main():
    a = parse_args()
    states = np.asarray(np.load(a.trajectory), dtype=np.float64)
    genes = [
        str(x).strip().upper()
        for x in pd.read_csv(a.expression_csv, index_col=0, usecols=[0]).index
    ]
    if states.shape[1] != len(genes):
        raise ValueError("Trajectory and expression gene dimensions differ")
    with open(a.vocab_json, encoding="utf-8") as handle:
        vocab = {str(x).strip() for x in json.load(handle)}
    allowed = set(genes) & vocab
    gene_to_idx = {gene: i for i, gene in enumerate(genes)}

    specs = [
        ("Attention", a.attention_grn, a.attention_report, "#3A70B2"),
        ("Embedding", a.embedding_grn, a.embedding_report, "#D27A2C"),
        ("Hidden embedding", a.hidden_grn, a.hidden_report, "#3D8B68"),
    ]
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.2,
        "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.65), sharex=True, sharey=True)
    rng = np.random.default_rng(a.seed)
    saved = {}
    for ax, (name, grn, report, color) in zip(axes, specs):
        src, dst, weight, valid = load_network(grn, allowed, gene_to_idx, a.top_k)
        x, y = propagated_points(states, src, dst, weight, valid, a.transient_iters)
        if len(x) > a.max_points:
            take = rng.choice(len(x), a.max_points, replace=False)
            xp, yp = x[take], y[take]
        else:
            xp, yp = x, y
        ax.scatter(xp, yp, s=5, color=color, alpha=0.14, edgecolors="none", rasterized=True)
        ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#555555")
        metrics = reported_metrics(report, a.top_k)
        ax.text(0.05, 0.94,
                f"median $\\rho$ = {metrics['spearman']:.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.2,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.82))
        ax.set_title(name, weight="bold", color=color, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        saved[name] = {"observed_normalized": x, "predicted_normalized": y}

    axes[0].set_ylabel("Predicted next-iteration change\n(within-iteration percentile rank)")
    axes[1].set_xlabel("Observed next-iteration change (within-iteration percentile rank)")
    fig.suptitle("Observed versus network-predicted expression changes (top 5k)",
                 x=0.075, y=1.02, ha="left", weight="bold", fontsize=10)
    fig.tight_layout()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / "observed_vs_predicted_top5k"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(outdir / "observed_vs_predicted_top5k_data.npz", **{
        name.lower().replace(" ", "_") + "_" + key: value
        for name, values in saved.items() for key, value in values.items()
    })
    print(stem)


if __name__ == "__main__":
    main()
