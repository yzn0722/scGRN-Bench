#!/usr/bin/env python3
"""Temporal-order and attention direction/weight controls for GRN propagation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_weighted_grn_propagation import (
    corr,
    load_grn,
    normalize_incoming,
    propagate,
    score_trajectory,
    summarize_window,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory-dir", required=True)
    p.add_argument("--expression-csv", required=True)
    p.add_argument("--vocab-json", required=True)
    p.add_argument("--attention", required=True)
    p.add_argument("--embedding", required=True)
    p.add_argument("--hidden", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--top-k", type=int, default=10000)
    p.add_argument("--n-null", type=int, default=200)
    p.add_argument("--transient-iters", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260910)
    return p.parse_args()


def empirical_p(obs, null):
    null = np.asarray(null, float)
    return float((1 + np.sum(null >= obs)) / (len(null) + 1))


def temporal_null(states, src, dst, weight, n_null, n_lags, rng):
    delta = np.abs(np.diff(np.asarray(states, float), axis=0))
    n_lags = min(n_lags, len(delta) - 1)
    valid = np.bincount(dst, minlength=states.shape[1]) > 0
    pred = [propagate(delta[t], src, dst, weight, states.shape[1]) for t in range(n_lags)]
    values = []
    for _ in range(n_null):
        # Permute the next-iteration changes across transient lags. Use a
        # derangement so no null pair retains its original next-lag match.
        perm = rng.permutation(n_lags)
        while np.any(perm == np.arange(n_lags)):
            perm = rng.permutation(n_lags)
        rhos = [corr(pred[t][valid], delta[int(perm[t]) + 1, valid], "spearman")
                for t in range(n_lags)]
        values.append(float(np.nanmedian(rhos)))
    return values


def observed_rho(states, src, dst, raw_weight, n_lags):
    weight = normalize_incoming(dst, raw_weight, states.shape[1])
    rows = score_trajectory(states, src, dst, weight, .10)
    return summarize_window(rows, 0, min(n_lags, len(rows)))["spearman"], weight


def main():
    a = parse_args()
    genes = pd.read_csv(a.expression_csv, index_col=0, usecols=[0]).index.astype(str)
    genes = [g.strip().upper() for g in genes]
    with open(a.vocab_json, encoding="utf-8") as f:
        allowed = set(genes) & {str(g).strip() for g in json.load(f)}
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    states = np.asarray(np.load(Path(a.trajectory_dir) / "early_mean_trajectory.npy"), float)
    networks = {
        "attn": a.attention,
        "tok": a.embedding,
        "hid": a.hidden,
    }
    colors = {"attn": "#4EA3F1", "tok": "#FF9A3D", "hid": "#AC99D2"}
    labels = {"attn": "attn", "tok": r"COS$_{tok}$", "hid": r"COS$_{hid}$"}
    result = {"top_k": a.top_k, "n_null": a.n_null, "temporal": {}, "attention": {}}
    attention_edges = None
    for i, (name, path) in enumerate(networks.items()):
        edge = load_grn(path, allowed, gene_to_idx, a.top_k).head(a.top_k)
        stored_src = edge.Gene1.map(gene_to_idx).to_numpy(np.int32)
        stored_dst = edge.Gene2.map(gene_to_idx).to_numpy(np.int32)
        raw_weight = edge.EdgeWeight.to_numpy(float)
        # Primary convention used by the existing analysis.
        src, dst = stored_dst, stored_src
        obs, weight = observed_rho(states, src, dst, raw_weight, a.transient_iters)
        null = temporal_null(states, src, dst, weight, a.n_null, a.transient_iters,
                             np.random.default_rng(a.seed + i * 1009))
        result["temporal"][name] = {"observed": obs, "null": null,
                                     "empirical_p": empirical_p(obs, null)}
        if name == "attn":
            attention_edges = stored_src, stored_dst, raw_weight

    stored_src, stored_dst, raw_weight = attention_edges
    density_ks = [2000, 4000, 6000, 8000, 10000]
    direction_result = {}
    for k in density_ks:
        # Primary direction follows the convention used in the main analysis.
        src0, dst0, w0 = stored_dst[:k], stored_src[:k], raw_weight[:k]
        observed, _ = observed_rho(states, src0, dst0, w0, a.transient_iters)
        rng = np.random.default_rng(a.seed + 5001 + k)
        null = []
        for _ in range(a.n_null):
            flip = rng.random(k) < .5
            src = np.where(flip, dst0, src0)
            dst = np.where(flip, src0, dst0)
            value, _ = observed_rho(states, src, dst, w0, a.transient_iters)
            null.append(value)
        direction_result[str(k)] = {"observed": observed, "random_direction_null": null}
    result["attention_random_direction"] = direction_result

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "grn_temporal_direction_controls.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 16,
                         "axes.linewidth": .8})
    reference_size = (510.503 / 72.0, 335.753 / 72.0)
    fig, ax = plt.subplots(figsize=reference_size)
    names = ("attn", "tok", "hid")
    x = np.arange(3)
    width = .25
    gap = .08
    observed_values = [result["temporal"][n]["observed"] for n in names]
    nulls = [np.asarray(result["temporal"][n]["null"], float) for n in names]
    null_means = np.array([v.mean() for v in nulls])
    null_q05 = np.array([np.quantile(v, .05) for v in nulls])
    null_q95 = np.array([np.quantile(v, .95) for v in nulls])
    ax.bar(x - width / 2 - gap / 2, observed_values, width=width,
           color=[colors[n] for n in names], alpha=.80, label="Observed order")
    ax.bar(x + width / 2 + gap / 2, null_means, width=width, color="#BDBDBD", alpha=.80,
           yerr=np.vstack([null_means - null_q05, null_q95 - null_means]),
           error_kw={"ecolor": "#222222", "elinewidth": 1.1, "capsize": 3,
                     "capthick": 1.1},
           label="Temporally permuted")
    ax.set_xticks(x, [labels[n] for n in names])
    ax.set_ylabel(r"Transient Spearman $\rho$", fontsize=16, fontweight="normal")
    ax.legend(frameon=False, fontsize=16, loc="lower center",
              bbox_to_anchor=(.5, -.38), ncol=2,
              handlelength=1.6, handletextpad=.5, labelspacing=.3,
              borderaxespad=0, columnspacing=.8)
    ax.axhline(0, color="#A5A5A5", lw=.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_linewidth(.8)
    ax.spines["bottom"].set_linewidth(.8)
    ax.tick_params(length=0, labelsize=14, colors="#000000")
    ax.set_ylim(-.02, .30)
    ax.grid(False)
    fig.subplots_adjust(left=.18, right=.98, bottom=.36, top=.97)
    stem = out / "grn_temporal_order_control"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=400)
    plt.close(fig)
    print(stem)

    fig, ax = plt.subplots(figsize=(6, 6))
    x = np.arange(len(density_ks))
    obs = np.array([direction_result[str(k)]["observed"] for k in density_ks])
    nulls = [np.asarray(direction_result[str(k)]["random_direction_null"], float)
             for k in density_ks]
    means = np.array([v.mean() for v in nulls])
    q05 = np.array([np.quantile(v, .05) for v in nulls])
    q95 = np.array([np.quantile(v, .95) for v in nulls])
    ax.fill_between(x, q05, q95, color="#BDBDBD", alpha=.35, linewidth=0,
                    label="Random-direction 5th-95th percentile")
    ax.plot(x, means, "--o", color="#777777", lw=1.8, ms=5,
            label="Random-direction mean")
    ax.plot(x, obs, "-o", color=colors["attn"], lw=2.5, ms=8, alpha=.9,
            label="Observed direction")
    ax.axhline(0, color="#A5A5A5", lw=.8, zorder=0)
    ax.set_xticks(x, ["2k", "4k", "6k", "8k", "10k"])
    ax.set_xlabel("Number of top-ranked GRN edges", fontsize=15)
    ax.set_ylabel(r"Transient Spearman $\rho$", fontsize=15)
    ax.text(-.13, 1.02, "b", transform=ax.transAxes, fontsize=16, weight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0, labelsize=12)
    ax.set_ylim(-.08, .16)
    ax.grid(False)
    fig.subplots_adjust(left=.17, right=.97, bottom=.14, top=.96)
    stem = out / "grn_attention_random_direction_control"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
