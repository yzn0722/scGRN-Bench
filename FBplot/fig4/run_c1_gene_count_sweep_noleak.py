#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leakage-free gene-count sweep for GRN binding claim.

Mask selection NEVER uses true_delta / late / top-|Δ| genes.
Only network membership, undirected degree, and random order of the same nodes.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_c1_fair_null import degree_matched_random_fair  # noqa: E402
from run_grn_perturbation_probes import (  # noqa: E402
    ScgptBundle,
    gene_update_mask,
    neighborhood_from_edges,
)
from run_unified_multidataset_pseudotime import (  # noqa: E402
    direction_accuracy_top_genes,
    set_seed,
)


def undirected_degrees(edges: pd.DataFrame, gene_to_idx: dict) -> Dict[str, int]:
    deg: Dict[str, int] = defaultdict(int)
    for a, b in zip(edges["Gene1"], edges["Gene2"]):
        if a in gene_to_idx and b in gene_to_idx:
            deg[a] += 1
            deg[b] += 1
    return dict(deg)


def sweep_sizes(n_min: int, n_max: int, step: int) -> List[int]:
    sizes = list(range(n_min, n_max + 1, step))
    if sizes[-1] != n_max:
        sizes.append(n_max)
    return sorted(set(sizes))


def parse_args():
    p = argparse.ArgumentParser(description="Leakage-free GRN binding gene-count sweep")
    p.add_argument("--dataset", default="hESC")
    p.add_argument("--network", choices=["chip", "string"], default="chip")
    p.add_argument("--outdir", default="outputs/gene_count_sweep_noleak")
    p.add_argument("--expr-root", default="/mnt/10T/yzn/benchmark_GRN/input_process")
    p.add_argument("--pt-root", default="/mnt/10T/yzn/benchmark_GRN/PseudoTime")
    p.add_argument("--chip-network", default="")
    p.add_argument("--string-network", default="")
    p.add_argument("--grn-tsv", default="")
    p.add_argument("--grn-source", default="chip")
    p.add_argument("--scgpt-model-dir", required=True)
    p.add_argument("--scgpt-repo-dir", required=True)
    p.add_argument("--scgpt-bin-log1p", action="store_true", default=False)
    p.add_argument("--scgpt-legacy-pt", action="store_true", default=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pt-quantile", type=float, default=0.2)
    p.add_argument("--top-percent", type=int, default=30)
    p.add_argument("--gen-iters", type=int, default=16)
    # 64 early cells → one GPU batch (old default 16 caused 4× H2D/D2H per iter)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--ema-alpha", type=float, default=0.1)
    p.add_argument("--max-early-cells", type=int, default=64)
    p.add_argument("--print-every", type=int, default=0)
    p.add_argument("--n-min", type=int, default=100)
    p.add_argument("--n-step", type=int, default=50)
    p.add_argument("--n-null", type=int, default=10)
    p.add_argument("--orders", default="net_degree,net_degree_asc,net_random")
    p.add_argument("--append", action="store_true",
                   help="Keep CSV rows for orders not in --orders")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Skip (order,draw,n) already in CSV (default on)")
    p.add_argument("--no-resume", action="store_false", dest="resume")
    p.add_argument("--c1-intersect-top-dynamic", action="store_true", default=False)
    p.add_argument("--top-grn-edges", type=int, default=5000)
    return p.parse_args()


def prepare_gpu_cache(bundle: ScgptBundle) -> None:
    """Pin early-cell state + gene ids on device once (avoids per-batch CPU↔GPU)."""
    if hasattr(bundle, "X_bin"):
        del bundle.X_bin
    bundle.chip_adj = bundle.string_adj = bundle.pred_adj = None
    idx = bundle.init_idx
    device = bundle.device
    bundle._vals0 = bundle.values_tensor[idx].to(device, non_blocking=False).contiguous()
    bundle._pad0 = bundle.pad_mask[idx].to(device, non_blocking=False).contiguous()
    bundle._src0 = bundle.gene_ids_tensor.to(device).contiguous()  # [1, T]
    # free full-cell CPU tensors
    del bundle.values_tensor
    bundle.values_tensor = bundle._vals0  # keep attribute for safety
    bundle.pad_mask = bundle._pad0
    bundle.init_idx = np.arange(bundle._vals0.shape[0], dtype=np.int64)
    gc.collect()


def run_iteration_fast(bundle: ScgptBundle, update_mask_genes: np.ndarray, n_iters: int) -> dict:
    """Same math as run_iteration, but stay on GPU and only return scalars."""
    args = bundle.args
    vals = bundle._vals0.clone()
    pad = bundle._pad0
    src_row = bundle._src0
    update_mask_t = torch.as_tensor(update_mask_genes, device=bundle.device).bool()[None, :]
    n_cells = vals.shape[0]
    bs = max(1, int(args.batch_size))
    final_acc = 0.0
    pred_mean = None

    with torch.no_grad():
        for _ in range(n_iters):
            for start in range(0, n_cells, bs):
                end = min(start + bs, n_cells)
                b = end - start
                v = vals[start:end]
                src = src_row.expand(b, -1)
                mask = pad[start:end]
                freeze = mask | (~update_mask_t.expand(b, -1))
                out = bundle.model(src=src, values=v, src_key_padding_mask=mask)
                new_v = out["mlm_output"]
                vals[start:end] = torch.where(
                    freeze, v, args.ema_alpha * v + (1.0 - args.ema_alpha) * new_v
                )
            pred_mean = vals[:, 1:].float().mean(dim=0).detach().cpu().numpy()
            pred_delta = pred_mean - bundle.early_mean
            final_acc, _ = direction_accuracy_top_genes(
                pred_delta, bundle.true_delta, bundle.top_idx
            )

    top = bundle.top_idx
    axis = bundle.late_mean - bundle.early_mean
    axis_u = axis / (float(np.linalg.norm(axis)) + 1e-12)
    x0 = bundle.init_mean
    xT = pred_mean

    def _sl(v):
        return v[top]

    return {
        "final_acc": float(final_acc),
        "proj_init": float(np.dot(_sl(x0) - _sl(bundle.early_mean), _sl(axis_u))),
        "proj_final": float(np.dot(_sl(xT) - _sl(bundle.early_mean), _sl(axis_u))),
    }


def plot_noleak(df, out_png, dataset, net_name, n_net, baseline_acc):
    fig, ax = plt.subplots(figsize=(8.2, 4.9), dpi=160)
    tag = net_name.upper()

    def _line(order, style, color, label, band=False):
        sub = df[df["order"] == order]
        if sub.empty:
            return
        if band:
            g = sub.groupby("n_update_genes")["final_acc"]
            ns = g.mean().index.to_numpy()
            mu = g.mean().to_numpy() * 100
            lo = g.min().to_numpy() * 100
            hi = g.max().to_numpy() * 100
            ax.fill_between(ns, lo, hi, color=color, alpha=0.18, linewidth=0)
            ax.plot(ns, mu, style, ms=5, lw=2, color=color, label=label)
        else:
            sub = sub.sort_values("n_update_genes").drop_duplicates("n_update_genes", keep="last")
            ax.plot(sub["n_update_genes"], sub["final_acc"] * 100, style, ms=5, lw=2, color=color, label=label)

    _line("net_degree", "o-", "#d95f02", f"{tag} high→low degree")
    _line("net_degree_asc", "D-", "#e7298a", f"{tag} low→high degree")
    _line("net_random", "s-", "#7570b3", f"Random {tag} order (same genes)", band=True)

    ax.axhline(50, color="0.75", ls="--", lw=1)
    ax.axhline(baseline_acc * 100, color="0.45", ls=":", lw=1,
               label=f"baseline all genes ({baseline_acc*100:.0f}%)")
    ax.axvline(n_net, color="0.6", ls="--", lw=1, alpha=0.8)
    ax.text(n_net, 47, f" all {tag}={n_net}", fontsize=8, color="0.4", va="bottom")
    ax.set_xlabel("Number of updatable genes")
    ax.set_ylabel("Direction accuracy (top 30% dynamic genes, %)")
    ax.set_title(dataset)
    ax.set_xlim(left=90)
    ax.set_ylim(45, 95)
    ax.legend(frameon=False, fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    net_name = args.network
    outdir = Path(args.outdir) / args.dataset / net_name
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "noleak_sweep.csv"

    t0 = time.time()
    print(f"[start] loading model/data for {args.dataset}/{net_name} …", flush=True)
    bundle = ScgptBundle(args)
    if net_name == "chip":
        edges = bundle.chip_edges
    else:
        if bundle.string_edges is None:
            raise SystemExit("STRING edges not loaded")
        edges = bundle.string_edges

    deg = undirected_degrees(edges, bundle.gene_to_idx)
    net_nodes, _ = neighborhood_from_edges(bundle, edges, intersect_top_dynamic=False)
    n_net = len(net_nodes)
    prepare_gpu_cache(bundle)
    print(f"[ready] model+data on {bundle.device} in {time.time()-t0:.1f}s "
          f"(early_cells={bundle._vals0.shape[0]}, batch={args.batch_size})", flush=True)

    net_by_degree = sorted(net_nodes, key=lambda g: (-deg.get(g, 0), g))
    net_by_degree_asc = sorted(net_nodes, key=lambda g: (deg.get(g, 0), g))
    want = [o.strip() for o in args.orders.split(",") if o.strip()]
    random_perms = []
    if "net_random" in want:
        for k in range(args.n_null):
            rng_k = np.random.default_rng(args.seed * 30011 + k)
            random_perms.append(list(rng_k.permutation(net_nodes)))

    sizes = [n for n in sweep_sizes(args.n_min, n_net, args.n_step) if n <= n_net]
    n_per_size = (
        (1 if "net_degree" in want else 0)
        + (1 if "net_degree_asc" in want else 0)
        + (args.n_null if "net_random" in want else 0)
        + (args.n_null if "random_matched" in want else 0)
        + (args.n_null if "random_matched_asc" in want else 0)
    )
    print(
        f"[noleak/{net_name}] nodes={n_net}, sizes={len(sizes)}, "
        f"~{1 + len(sizes)*n_per_size} evals × {args.gen_iters} iters "
        f"(orders={want}, n_null={args.n_null})",
        flush=True,
    )

    rows: List[dict] = []
    done: Set[Tuple[str, str, int]] = set()
    if csv_path.is_file() and (args.resume or args.append):
        existing = pd.read_csv(csv_path)
        if args.append:
            rows = existing[~existing["order"].isin(want)].to_dict("records")
            keep = existing[existing["order"].isin(want + ["baseline_all"])]
        else:
            keep = existing
            rows = keep.to_dict("records")
        for r in keep.itertuples(index=False):
            done.add((str(r.order), str(r.draw), int(r.n_update_genes)))
        print(f"  resume: {len(done)} rows already done", flush=True)

    def already(order: str, draw: str, n: int) -> bool:
        return (order, draw, n) in done

    def run_mask(genes: Sequence[str], order: str, draw: str, mode: str) -> dict:
        umask = gene_update_mask(bundle, genes)
        umask[0] = False
        r = run_iteration_fast(bundle, umask, args.gen_iters)
        return {
            "order": order,
            "draw": draw,
            "n_update_genes": len(genes),
            "final_acc": float(r["final_acc"]),
            "proj_gain": float(r["proj_final"] - r["proj_init"]),
            "sample_mode": mode,
        }

    # baseline
    if already("baseline_all", "obs", len(bundle.genes)) or any(
        r.get("order") == "baseline_all" for r in rows
    ):
        baseline_acc = float(next(r["final_acc"] for r in rows if r["order"] == "baseline_all"))
        print(f"===== baseline_all (reuse) acc={baseline_acc:.4f} =====", flush=True)
    else:
        print("===== baseline_all =====", flush=True)
        t1 = time.time()
        base_mask = np.ones(bundle._src0.shape[1], dtype=bool)
        base_mask[0] = False
        base = run_iteration_fast(bundle, base_mask, args.gen_iters)
        baseline_acc = float(base["final_acc"])
        rows.append({
            "order": "baseline_all", "draw": "obs",
            "n_update_genes": len(bundle.genes),
            "final_acc": baseline_acc,
            "proj_gain": base["proj_final"] - base["proj_init"],
            "sample_mode": "all_genes",
        })
        done.add(("baseline_all", "obs", len(bundle.genes)))
        print(f"  baseline_acc={baseline_acc:.4f} ({time.time()-t1:.1f}s)", flush=True)

    for n in sizes:
        print(f"\n===== n={n} =====", flush=True)
        t_n = time.time()
        net_pref = net_by_degree[:n]
        net_pref_asc = net_by_degree_asc[:n]

        if "net_degree" in want and not already("net_degree", "obs", n):
            row = run_mask(net_pref, "net_degree", "obs", f"{net_name}_high_to_low_degree")
            rows.append(row)
            done.add(("net_degree", "obs", n))
            print(f"  net_degree (high→low) acc={row['final_acc']:.4f}", flush=True)

        if "net_degree_asc" in want and not already("net_degree_asc", "obs", n):
            row = run_mask(net_pref_asc, "net_degree_asc", "obs", f"{net_name}_low_to_high_degree")
            rows.append(row)
            done.add(("net_degree_asc", "obs", n))
            print(f"  net_degree_asc (low→high) acc={row['final_acc']:.4f}", flush=True)

        if "net_random" in want:
            null_accs = []
            for k, perm in enumerate(random_perms):
                draw = f"rand_{k}"
                if already("net_random", draw, n):
                    prev = next(
                        r for r in rows
                        if r["order"] == "net_random" and r["draw"] == draw and int(r["n_update_genes"]) == n
                    )
                    null_accs.append(float(prev["final_acc"]))
                    continue
                row = run_mask(perm[:n], "net_random", draw, f"{net_name}_random_shuffle_no_delta")
                rows.append(row)
                done.add(("net_random", draw, n))
                null_accs.append(row["final_acc"])
            if null_accs:
                print(
                    f"  net_random mean={np.mean(null_accs):.4f} "
                    f"[{min(null_accs):.4f},{max(null_accs):.4f}]",
                    flush=True,
                )

        if "random_matched" in want:
            for k in range(args.n_null):
                draw = f"null_{k}"
                if already("random_matched", draw, n):
                    continue
                rng_k = np.random.default_rng(args.seed * 10007 + n * 97 + k)
                samp, mode = degree_matched_random_fair(bundle.genes, net_pref, deg, rng_k)
                rows.append(run_mask(samp, "random_matched", draw, mode))
                done.add(("random_matched", draw, n))

        if "random_matched_asc" in want:
            for k in range(args.n_null):
                draw = f"null_{k}"
                if already("random_matched_asc", draw, n):
                    continue
                rng_k = np.random.default_rng(args.seed * 20011 + n * 89 + k)
                samp, mode = degree_matched_random_fair(bundle.genes, net_pref_asc, deg, rng_k)
                rows.append(run_mask(samp, "random_matched_asc", draw, mode))
                done.add(("random_matched_asc", draw, n))

        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"  size done in {time.time()-t_n:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    tab = []
    for n in sizes:
        row_out = {"n": n}
        for key, order in [("net_degree_acc", "net_degree"), ("net_degree_asc_acc", "net_degree_asc")]:
            sub = df[(df.order == order) & (df.n_update_genes == n)]
            row_out[key] = float(sub["final_acc"].iloc[0]) if len(sub) else float("nan")
        sub_r = df[(df.order == "net_random") & (df.n_update_genes == n)]["final_acc"]
        row_out["net_random_mean"] = float(sub_r.mean()) if len(sub_r) else float("nan")
        row_out["net_random_min"] = float(sub_r.min()) if len(sub_r) else float("nan")
        row_out["net_random_max"] = float(sub_r.max()) if len(sub_r) else float("nan")
        tab.append(row_out)
    tab_df = pd.DataFrame(tab)
    tab_df.to_csv(outdir / "noleak_summary.csv", index=False)

    with open(outdir / "noleak_meta.json", "w") as f:
        json.dump({
            "protocol": "leakage_free",
            "network": net_name,
            "n_network_nodes": n_net,
            "n_random_draws": args.n_null,
            "baseline_acc": baseline_acc,
            "orders_run": want,
        }, f, indent=2)

    plot_noleak(df, outdir / "noleak_sweep_acc.png", args.dataset, net_name, n_net, baseline_acc)
    plot_noleak(df, outdir / "noleak_sweep_acc.pdf", args.dataset, net_name, n_net, baseline_acc)
    print(tab_df.to_string(index=False), flush=True)
    print(f"\nDone in {time.time()-t0:.1f}s. Outputs: {outdir}", flush=True)


if __name__ == "__main__":
    main()
