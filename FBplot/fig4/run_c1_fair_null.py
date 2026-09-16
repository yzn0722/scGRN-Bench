#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fair multi-draw null for C1: is direction signal bound to GRN gene sets?

Anti-cheat rules
----------------
1. Sample only from the analysis universe (genes in the expression matrix).
2. Same k as the observed GRN set.
3. Degree-histogram matched (log2 degree buckets from the same network).
4. Prefer excluding the observed set when pool is large enough; otherwise
   size-matched draws from the full universe (documented). Never reconstruct
   the observed set by identity matching.
5. Empirical p from B independent draws on final_acc and proj_gain.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_grn_perturbation_probes import (  # noqa: E402
    ScgptBundle,
    chip_tf_nontf_gene_sets,
    gene_update_mask,
    neighborhood_from_edges,
    run_iteration,
    string_tf_nontf_gene_sets,
)
from run_unified_multidataset_pseudotime import set_seed  # noqa: E402


def degree_matched_random_fair(
    genes: Sequence[str],
    keep: Sequence[str],
    degrees: Dict[str, int],
    rng: np.random.Generator,
) -> Tuple[List[str], str]:
    """Histogram degree-matched sample of size |keep| (never identity-reconstruct)."""
    keep_list = list(keep)
    k = len(keep_list)
    keep_set = set(keep_list)
    pool_excl = [g for g in genes if g not in keep_set]

    if len(pool_excl) >= k:
        pool = pool_excl
        mode = "exclude_observed_degree_matched"
    else:
        pool = list(genes)
        mode = "full_universe_size_matched_degree_matched"

    def bucket(g: str) -> int:
        return int(np.floor(np.log2(degrees.get(g, 0) + 1)))

    need: Dict[int, int] = defaultdict(int)
    for g in keep_list:
        need[bucket(g)] += 1

    by_b: Dict[int, List[str]] = defaultdict(list)
    for g in pool:
        by_b[bucket(g)].append(g)
    for b in by_b:
        rng.shuffle(by_b[b])

    chosen: List[str] = []
    for b, n in list(need.items()):
        take = min(n, len(by_b.get(b, [])))
        if take:
            chosen.extend(by_b[b][:take])
            by_b[b] = by_b[b][take:]
            need[b] -= take

    remaining = sum(need.values())
    if remaining:
        leftovers: List[str] = []
        for b in sorted(by_b.keys()):
            leftovers.extend(by_b[b])
        rng.shuffle(leftovers)
        if len(leftovers) < remaining:
            extra = [g for g in genes if g not in set(chosen) and g not in leftovers]
            rng.shuffle(extra)
            leftovers.extend(extra)
            mode = mode + "+topup_from_universe"
        chosen.extend(leftovers[:remaining])

    chosen = list(dict.fromkeys(chosen))
    if len(chosen) < k:
        extra = [g for g in pool if g not in set(chosen)]
        if len(extra) < (k - len(chosen)):
            extra = [g for g in genes if g not in set(chosen)]
            mode = mode + "+topup_from_universe"
        chosen.extend(list(rng.choice(extra, size=k - len(chosen), replace=False)))
    return chosen[:k], mode


def top_dynamic_intersect(bundle: ScgptBundle, nodes: Sequence[str]) -> List[str]:
    top = {bundle.genes[i] for i in bundle.top_idx}
    return sorted(set(nodes) & top)


def build_observed_sets(bundle: ScgptBundle) -> Dict[str, Tuple[List[str], Dict[str, int]]]:
    chip_nb, chip_deg = neighborhood_from_edges(bundle, bundle.chip_edges, False)
    string_nb, string_deg = neighborhood_from_edges(bundle, bundle.string_edges, False)
    chip_sets, chip_degs = chip_tf_nontf_gene_sets(bundle)
    string_sets, string_degs = string_tf_nontf_gene_sets(bundle)
    chip_top = top_dynamic_intersect(bundle, chip_nb)
    string_top = top_dynamic_intersect(bundle, string_nb)
    return {
        "chip_all_nodes": (chip_nb, chip_deg),
        "string_neighborhood": (string_nb, string_deg),
        # Legacy protocol behind prior 0.85-vs-0.52 claim (CHIP ∩ top30% |Δ|)
        "chip_neighborhood_topdyn": (chip_top, chip_deg),
        "string_neighborhood_topdyn": (string_top, string_deg),
        "chip_tf": (chip_sets["chip_tf"], chip_degs["chip_tf"]),
        "chip_nontf": (chip_sets["chip_nontf"], chip_degs["chip_nontf"]),
        "string_tf": (string_sets["string_tf"], string_degs["string_tf"]),
        "string_nontf": (string_sets["string_nontf"], string_degs["string_nontf"]),
        "string_all_nodes": (string_sets["string_all_nodes"], string_degs["string_all_nodes"]),
    }


def emp_p(obs: float, null: np.ndarray, alternative: str = "greater") -> float:
    null = np.asarray(null, dtype=float)
    b = len(null)
    if alternative == "greater":
        extreme = int(np.sum(null >= obs))
    else:
        extreme = int(np.sum(null <= obs))
    return float((1 + extreme) / (b + 1))


def parse_args():
    p = argparse.ArgumentParser(description="Fair multi-draw C1 null for GRN binding")
    p.add_argument("--dataset", default="hESC")
    p.add_argument("--outdir", default="outputs/grn_perturbation_probes_fair_null")
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
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--ema-alpha", type=float, default=0.1)
    p.add_argument("--max-early-cells", type=int, default=64)
    p.add_argument("--print-every", type=int, default=0)
    p.add_argument("--n-null", type=int, default=30)
    p.add_argument(
        "--sets",
        default=(
            "chip_neighborhood_topdyn,string_neighborhood_topdyn,"
            "string_tf,string_nontf,chip_tf,chip_nontf,string_neighborhood"
        ),
    )
    p.add_argument("--c1-intersect-top-dynamic", action="store_true", default=False)
    p.add_argument("--top-grn-edges", type=int, default=5000)
    return p.parse_args()


def _checkpoint(outdir: Path, rows: list, summary: dict) -> None:
    pd.DataFrame(rows).to_csv(outdir / "fair_null_draws.csv", index=False)
    with open(outdir / "fair_null_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    args = parse_args()
    set_seed(args.seed)
    outdir = Path(args.outdir) / args.dataset / "c1_fair_null"
    outdir.mkdir(parents=True, exist_ok=True)

    bundle = ScgptBundle(args)
    observed = build_observed_sets(bundle)
    want = [s.strip() for s in args.sets.split(",") if s.strip()]
    for s in want:
        if s not in observed:
            raise SystemExit(f"Unknown set {s}. Choose from {list(observed)}")

    print("===== baseline_all =====", flush=True)
    base_mask = np.ones(bundle.gene_ids_tensor.shape[1], dtype=bool)
    base_mask[0] = False
    base = run_iteration(bundle, base_mask, args.gen_iters, label="baseline_all")
    rows = [{
        "set": "baseline_all",
        "draw": "obs",
        "n_update": base["n_update_genes"],
        "final_acc": base["final_acc"],
        "proj_gain": base["proj_final"] - base["proj_init"],
        "sample_mode": "all_genes",
        "overlap_with_obs": 1.0,
    }]
    summary = {
        "baseline_final_acc": base["final_acc"],
        "baseline_proj_gain": base["proj_final"] - base["proj_init"],
        "n_null": args.n_null,
        "set_sizes": {k: len(v[0]) for k, v in observed.items() if k in want},
        "sets": {},
    }
    _checkpoint(outdir, rows, summary)
    print(f"  baseline final_acc={base['final_acc']:.4f}", flush=True)
    print(f"  set sizes: {summary['set_sizes']}", flush=True)

    rng_master = np.random.default_rng(args.seed)

    for name in want:
        genes, deg = observed[name]
        print(f"\n===== observed {name} n={len(genes)} =====", flush=True)
        omask = gene_update_mask(bundle, genes)
        omask[0] = False
        obs_r = run_iteration(bundle, omask, args.gen_iters, label=name)
        obs_acc = float(obs_r["final_acc"])
        obs_proj = float(obs_r["proj_final"] - obs_r["proj_init"])
        rows.append({
            "set": name,
            "draw": "obs",
            "n_update": obs_r["n_update_genes"],
            "final_acc": obs_acc,
            "proj_gain": obs_proj,
            "sample_mode": "observed",
            "overlap_with_obs": 1.0,
        })
        print(f"  obs_acc={obs_acc:.4f} obs_proj={obs_proj:.2f}", flush=True)

        null_acc: List[float] = []
        null_proj: List[float] = []
        modes: List[str] = []
        overlaps: List[float] = []
        for b in range(args.n_null):
            rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
            rand_genes, mode = degree_matched_random_fair(bundle.genes, genes, deg, rng)
            modes.append(mode)
            ov = len(set(rand_genes) & set(genes)) / max(len(genes), 1)
            overlaps.append(ov)
            rmask = gene_update_mask(bundle, rand_genes)
            rmask[0] = False
            rr = run_iteration(bundle, rmask, args.gen_iters, label=f"{name}_null{b}")
            null_acc.append(float(rr["final_acc"]))
            null_proj.append(float(rr["proj_final"] - rr["proj_init"]))
            rows.append({
                "set": name,
                "draw": f"null_{b}",
                "n_update": rr["n_update_genes"],
                "final_acc": null_acc[-1],
                "proj_gain": null_proj[-1],
                "sample_mode": mode,
                "overlap_with_obs": ov,
            })
            if (b + 1) % 5 == 0 or b == 0:
                print(
                    f"  null {b+1}/{args.n_null}: acc={null_acc[-1]:.3f} "
                    f"(obs={obs_acc:.3f}) mode={mode} overlap={ov:.2f}",
                    flush=True,
                )
                _checkpoint(outdir, rows, summary)

        na = np.asarray(null_acc)
        npj = np.asarray(null_proj)
        set_sum = {
            "n_obs_genes": len(genes),
            "obs_final_acc": obs_acc,
            "obs_proj_gain": obs_proj,
            "null_acc_mean": float(na.mean()),
            "null_acc_std": float(na.std(ddof=1)) if len(na) > 1 else 0.0,
            "null_acc_q05": float(np.quantile(na, 0.05)),
            "null_acc_q50": float(np.quantile(na, 0.50)),
            "null_acc_q95": float(np.quantile(na, 0.95)),
            "null_proj_mean": float(npj.mean()),
            "null_proj_std": float(npj.std(ddof=1)) if len(npj) > 1 else 0.0,
            "p_acc_greater": emp_p(obs_acc, na, "greater"),
            "p_acc_less": emp_p(obs_acc, na, "less"),
            "p_proj_greater": emp_p(obs_proj, npj, "greater"),
            "p_proj_less": emp_p(obs_proj, npj, "less"),
            "sample_mode": modes[0] if modes else "",
            "mean_overlap_with_obs": float(np.mean(overlaps)) if overlaps else 0.0,
            "supports_binding_acc": bool(emp_p(obs_acc, na, "greater") < 0.05),
            "supports_binding_proj": bool(emp_p(obs_proj, npj, "greater") < 0.05),
        }
        summary["sets"][name] = set_sum
        _checkpoint(outdir, rows, summary)
        print(
            f"  RESULT {name}: obs_acc={obs_acc:.3f} vs null "
            f"{set_sum['null_acc_mean']:.3f} "
            f"[{set_sum['null_acc_q05']:.3f},{set_sum['null_acc_q95']:.3f}] "
            f"p_greater={set_sum['p_acc_greater']:.4f} p_less={set_sum['p_acc_less']:.4f}",
            flush=True,
        )

    tab = []
    for name, s in summary["sets"].items():
        tab.append({
            "set": name,
            "n": s["n_obs_genes"],
            "obs_acc": s["obs_final_acc"],
            "null_acc_mean": s["null_acc_mean"],
            "null_acc_q05": s["null_acc_q05"],
            "null_acc_q95": s["null_acc_q95"],
            "p_acc_greater": s["p_acc_greater"],
            "p_acc_less": s["p_acc_less"],
            "obs_proj": s["obs_proj_gain"],
            "null_proj_mean": s["null_proj_mean"],
            "p_proj_greater": s["p_proj_greater"],
            "sample_mode": s["sample_mode"],
            "mean_overlap": s["mean_overlap_with_obs"],
            "supports_binding": s["supports_binding_acc"] or s["supports_binding_proj"],
        })
    pd.DataFrame(tab).to_csv(outdir / "fair_null_table.csv", index=False)
    print("\nDone.", flush=True)
    print(f"Outputs: {outdir}", flush=True)
    print(pd.DataFrame(tab).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
