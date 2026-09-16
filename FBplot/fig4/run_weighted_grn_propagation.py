#!/usr/bin/env python3
"""Test whether iterative expression changes propagate through weighted GRNs.

For unsigned attention weights, this evaluates
    predicted |dX(t+1)| = W.T @ |dX(t)|
against observed |dX(t+1)|. Real networks are compared with directed
configuration-model rewiring (exact in/out stub counts) and weight shuffling.

No model is loaded. The script only reads saved mean trajectories and a GRN TSV.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy import stats


EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectory-dir", required=True)
    p.add_argument("--expression-csv", required=True, help="Expression CSV supplying trajectory gene order.")
    p.add_argument("--grn-tsv", required=True)
    p.add_argument("--network-name", default="network")
    p.add_argument("--vocab-json", default="", help="Optional scGPT vocab; excludes OOV/frozen genes.")
    p.add_argument("--outdir", required=True)
    p.add_argument("--top-k", default="1000,5000,10000")
    p.add_argument("--groups", default="early,middle,late")
    p.add_argument("--n-null", type=int, default=200)
    p.add_argument("--transient-iters", type=int, default=8)
    p.add_argument("--top-fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=20260903)
    p.add_argument("--memory-limit-gb", type=float, default=2.0)
    p.add_argument("--execute", action="store_true", help="Run after preflight; default is read-only preflight.")
    p.add_argument("--primary-only", action="store_true", help="Run key-to-query orientation only.")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def finite(x) -> bool:
    return x is not None and np.isfinite(x)


def corr(x, y, kind):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) <= EPS or np.ptp(y) <= EPS:
        return float("nan")
    if kind == "pearson":
        return float(stats.pearsonr(x, y).statistic)
    return float(stats.spearmanr(x, y).statistic)


def score_vectors(pred, obs, top_fraction=0.10):
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    ok = np.isfinite(pred) & np.isfinite(obs)
    pred, obs = pred[ok], obs[ok]
    if len(pred) < 3:
        return {"pearson": None, "spearman": None, "ols_r2": None, "top_overlap": None}
    pearson = corr(pred, obs, "pearson")
    spearman = corr(pred, obs, "spearman")
    # OLS with intercept: explanatory R2 equals Pearson^2 for one predictor.
    r2 = pearson * pearson if np.isfinite(pearson) else float("nan")
    n_top = max(1, int(math.ceil(len(pred) * top_fraction)))
    ip = set(np.argpartition(pred, -n_top)[-n_top:].tolist())
    io = set(np.argpartition(obs, -n_top)[-n_top:].tolist())
    overlap = len(ip & io) / n_top
    return {
        "pearson": pearson,
        "spearman": spearman,
        "ols_r2": r2,
        "top_overlap": float(overlap),
    }


def normalize_incoming(dst, weight, n_genes):
    denom = np.bincount(dst, weights=np.abs(weight), minlength=n_genes)
    return weight / np.maximum(denom[dst], EPS)


def propagate(source_change, src, dst, weight, n_genes):
    out = np.zeros(n_genes, dtype=np.float64)
    np.add.at(out, dst, source_change[src] * weight)
    return out


def score_trajectory(states, src, dst, weight, top_fraction):
    delta = np.abs(np.diff(np.asarray(states, float), axis=0))
    valid = np.bincount(dst, minlength=states.shape[1]) > 0
    per_lag = []
    for t in range(len(delta) - 1):
        pred = propagate(delta[t], src, dst, weight, states.shape[1])
        s = score_vectors(pred[valid], delta[t + 1, valid], top_fraction)
        s["lag"] = t + 1
        per_lag.append(s)
    return per_lag


def summarize_window(per_lag, start, stop):
    rows = per_lag[start:stop]
    result = {}
    for metric in ("pearson", "spearman", "ols_r2", "top_overlap"):
        x = [r[metric] for r in rows if finite(r[metric])]
        result[metric] = float(np.median(x)) if x else None
    result["n_lags"] = len(rows)
    return result


def configuration_rewire(src, dst, weight, rng, candidates=8):
    """Fast directed configuration null preserving all source/target stubs.

    Parallel edges are retained as separate stubs and summed during propagation.
    Among several target permutations, keep the one with fewest self-loops.
    """
    best, best_loops = None, len(dst) + 1
    for _ in range(max(1, candidates)):
        candidate = rng.permutation(dst)
        loops = int(np.sum(src == candidate))
        if loops < best_loops:
            best, best_loops = candidate, loops
    pairs = np.column_stack([src, best])
    n_unique = len(np.unique(pairs, axis=0))
    diagnostics = {
        "changed_target_stubs": int(np.sum(best != dst)),
        "self_loops": best_loops,
        "parallel_edge_stubs": int(len(dst) - n_unique),
    }
    return src.copy(), best, weight.copy(), diagnostics


def empirical_p(obs, null):
    x = np.asarray([v for v in null if finite(v)], float)
    if not finite(obs) or len(x) == 0:
        return None
    return float((1 + np.sum(x >= obs)) / (len(x) + 1))


def compare_nulls(states, src, dst, raw_weight, args, rng):
    n = states.shape[1]
    real_w = normalize_incoming(dst, raw_weight, n)
    real_lags = score_trajectory(states, src, dst, real_w, args.top_fraction)
    windows = {
        "transient": (0, min(args.transient_iters, len(real_lags))),
        "all": (0, len(real_lags)),
    }
    result = {"per_lag": real_lags, "windows": {}}
    null_store = {
        w: {c: {m: [] for m in ("pearson", "spearman", "ols_r2", "top_overlap")}
            for c in ("rewired", "weight_shuffled")}
        for w in windows
    }
    rewire_diags = []
    for _ in range(args.n_null):
        rs, rd, rw, diag = configuration_rewire(src, dst, raw_weight, rng)
        rewire_diags.append(diag)
        rw = normalize_incoming(rd, rw, n)
        rewired_lags = score_trajectory(states, rs, rd, rw, args.top_fraction)

        shuffled_raw = rng.permutation(raw_weight)
        shuffled_w = normalize_incoming(dst, shuffled_raw, n)
        shuffled_lags = score_trajectory(states, src, dst, shuffled_w, args.top_fraction)
        for window, (lo, hi) in windows.items():
            for condition, rows in (("rewired", rewired_lags), ("weight_shuffled", shuffled_lags)):
                summary = summarize_window(rows, lo, hi)
                for metric in null_store[window][condition]:
                    null_store[window][condition][metric].append(summary[metric])

    for window, (lo, hi) in windows.items():
        observed = summarize_window(real_lags, lo, hi)
        packed = {"observed": observed}
        for condition in ("rewired", "weight_shuffled"):
            packed[condition] = {}
            for metric, values in null_store[window][condition].items():
                vals = np.asarray([v for v in values if finite(v)], float)
                obs = observed[metric]
                packed[condition][metric] = {
                    "null_mean": float(vals.mean()) if len(vals) else None,
                    "null_q05": float(np.quantile(vals, 0.05)) if len(vals) else None,
                    "null_q95": float(np.quantile(vals, 0.95)) if len(vals) else None,
                    "real_minus_null_mean": float(obs - vals.mean()) if finite(obs) and len(vals) else None,
                    "empirical_p_greater": empirical_p(obs, vals),
                    "null_values": vals.tolist(),
                }
        result["windows"][window] = packed
    result["rewire_diagnostics"] = {
        "null_model": "directed configuration stubs; exact in/out degrees",
        "changed_target_stubs_median": float(np.median([x["changed_target_stubs"] for x in rewire_diags])) if rewire_diags else 0,
        "self_loops_median": float(np.median([x["self_loops"] for x in rewire_diags])) if rewire_diags else 0,
        "parallel_edge_stubs_median": float(np.median([x["parallel_edge_stubs"] for x in rewire_diags])) if rewire_diags else 0,
    }
    return result


def load_inputs(args):
    groups = [x.strip() for x in args.groups.split(",") if x.strip()]
    paths = {g: Path(args.trajectory_dir) / f"{g}_mean_trajectory.npy" for g in groups}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    for p in (Path(args.expression_csv), Path(args.grn_tsv)):
        if not p.is_file():
            missing.append(str(p))
    if args.vocab_json and not Path(args.vocab_json).is_file():
        missing.append(args.vocab_json)
    if missing:
        raise FileNotFoundError("Missing inputs: " + ", ".join(missing))
    genes = pd.read_csv(args.expression_csv, index_col=0, usecols=[0]).index.astype(str)
    genes = [g.strip().upper() for g in genes]
    allowed = set(genes)
    if args.vocab_json:
        with open(args.vocab_json) as f:
            # Match ScgptBundle exactly: expression symbols are upper-cased, but
            # vocabulary membership itself is case-sensitive.
            allowed &= {str(g).strip() for g in json.load(f)}
    states = {g: np.load(p, mmap_mode="r") for g, p in paths.items()}
    shape = next(iter(states.values())).shape
    if any(x.shape != shape for x in states.values()):
        raise ValueError("trajectory arrays have different shapes")
    if shape[1] != len(genes):
        raise ValueError(f"trajectory has {shape[1]} genes but expression CSV has {len(genes)}")
    return genes, allowed, states


def load_grn(path, allowed, gene_to_idx, max_k):
    if Path(path).stat().st_size > 512 * 1024**2:
        raise MemoryError("GRN file >512 MiB is blocked")
    df = pd.read_csv(path, sep="\t", usecols=["Gene1", "Gene2", "EdgeWeight"])
    df["Gene1"] = df.Gene1.astype(str).str.strip().str.upper()
    df["Gene2"] = df.Gene2.astype(str).str.strip().str.upper()
    df["EdgeWeight"] = pd.to_numeric(df.EdgeWeight, errors="coerce")
    df = df[df.Gene1.isin(allowed) & df.Gene2.isin(allowed)]
    # Primary unsigned analysis: retain positive similarities/attention only.
    # Negative cosine similarity is not interpreted as transcriptional repression.
    df = df[np.isfinite(df.EdgeWeight) & (df.EdgeWeight > 0) & (df.Gene1 != df.Gene2)]
    df = df.sort_values("EdgeWeight", ascending=False).drop_duplicates(["Gene1", "Gene2"]).head(max_k)
    return df.reset_index(drop=True)


def preflight(args):
    top_k = sorted({int(x) for x in args.top_k.split(",") if x.strip()})
    if not top_k or min(top_k) < 10 or max(top_k) > 50000:
        raise ValueError("top-k values must be between 10 and 50000")
    if not 1 <= args.n_null <= 1000:
        raise ValueError("n-null must be 1..1000")
    if not 0 < args.top_fraction <= 0.5:
        raise ValueError("top-fraction must be in (0,0.5]")
    genes, allowed, states = load_inputs(args)
    traj_bytes = sum(Path(args.trajectory_dir, f"{g}_mean_trajectory.npy").stat().st_size
                     for g in states)
    grn_bytes = Path(args.grn_tsv).stat().st_size
    conservative_peak = 8 * grn_bytes + 4 * traj_bytes + max(top_k) * 64
    if conservative_peak > args.memory_limit_gb * 1024**3:
        raise MemoryError("conservative memory estimate exceeds memory-limit-gb")
    report = {
        "mode": "execute" if args.execute else "preflight_only",
        "trajectory_shape": list(next(iter(states.values())).shape),
        "n_genes": len(genes),
        "n_allowed_genes": len(allowed),
        "top_k": top_k,
        "n_null": args.n_null,
        "estimated_peak_gib": conservative_peak / 1024**3,
        "model_loaded": False,
        "signed_direction_accuracy": "not computed: attention weights are unsigned",
    }
    print(json.dumps(report, indent=2))
    return report, top_k, genes, allowed, states


def run(args):
    pre, top_ks, genes, allowed, states = preflight(args)
    if not args.execute:
        print("Preflight only. Add --execute to run.")
        return
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    df_all = load_grn(args.grn_tsv, allowed, gene_to_idx, max(top_ks))
    if len(df_all) < max(top_ks):
        raise ValueError(f"only {len(df_all)} valid edges available, fewer than max top-k")
    max_edge = df_all.head(max(top_ks))
    pair_weight = {(a, b): float(w) for a, b, w in max_edge[["Gene1", "Gene2", "EdgeWeight"]].itertuples(index=False, name=None)}
    reverse_pairs = [(w, pair_weight[(b, a)]) for (a, b), w in pair_weight.items() if (b, a) in pair_weight]
    reverse_fraction = len(reverse_pairs) / max(len(pair_weight), 1)
    reverse_weight_corr = corr(
        np.asarray([x[0] for x in reverse_pairs]), np.asarray([x[1] for x in reverse_pairs]), "pearson"
    ) if len(reverse_pairs) >= 3 else None
    report = {
        "preflight": pre,
        "network_name": args.network_name,
        "grn": str(args.grn_tsv),
        "weight_policy": "positive weights only; incoming normalization",
        "top_max_reverse_edge_fraction": reverse_fraction,
        "top_max_reverse_weight_pearson": reverse_weight_corr,
        "is_effectively_symmetric": bool(reverse_fraction > 0.95 and finite(reverse_weight_corr) and reverse_weight_corr > 0.99),
        "analyses": {},
    }
    curve_rows = []
    started = time.time()
    for k in top_ks:
        edge = df_all.head(k)
        stored_src = edge.Gene1.map(gene_to_idx).to_numpy(np.int32)
        stored_dst = edge.Gene2.map(gene_to_idx).to_numpy(np.int32)
        weight = edge.EdgeWeight.to_numpy(np.float64)
        report["analyses"][str(k)] = {}
        orientations = {
            "key_to_query_primary": (stored_dst, stored_src),
            "as_stored_sensitivity": (stored_src, stored_dst),
        }
        if args.primary_only:
            orientations.pop("as_stored_sensitivity")
        for orientation, (src, dst) in orientations.items():
            report["analyses"][str(k)][orientation] = {}
            for gi, (group, state_mm) in enumerate(states.items()):
                state = np.asarray(state_mm, dtype=np.float64)
                rng = np.random.default_rng(args.seed + k * 31 + gi * 1009 + (0 if orientation.startswith("key") else 1))
                result = compare_nulls(state, src, dst, weight, args, rng)
                report["analyses"][str(k)][orientation][group] = result
                for row in result["per_lag"]:
                    curve_rows.append({"top_k": k, "orientation": orientation, "group": group, **row})
                s = result["windows"]["transient"]["observed"]
                p = result["windows"]["transient"]["rewired"]["spearman"]["empirical_p_greater"]
                print(f"k={k} {orientation} {group}: transient rho={s['spearman']:.4g}, p_rewired={p}", flush=True)
    report["elapsed_seconds"] = time.time() - started
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "weighted_grn_propagation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    pd.DataFrame(curve_rows).to_csv(outdir / "weighted_grn_propagation_curves.csv", index=False)
    print(f"Wrote {outdir / 'weighted_grn_propagation.json'}")


def self_test():
    states = np.array([[1, 0, 0, 0], [2, 0, 0, 0], [2, 3, 0, 0], [2, 3, 4, 0]], float)
    src = np.array([0, 1, 2], dtype=np.int32)
    dst = np.array([1, 2, 3], dtype=np.int32)
    w = normalize_incoming(dst, np.ones(3), 4)
    rows = score_trajectory(states, src, dst, w, 0.25)
    assert len(rows) == 2 and rows[0]["spearman"] > 0.9
    rs, rd, _, diag = configuration_rewire(src, dst, w, np.random.default_rng(1))
    assert sorted(np.bincount(src, minlength=4)) == sorted(np.bincount(rs, minlength=4))
    assert sorted(np.bincount(dst, minlength=4)) == sorted(np.bincount(rd, minlength=4))
    assert diag["changed_target_stubs"] >= 0
    print("self-test: OK")


if __name__ == "__main__":
    a = parse_args()
    if a.self_test:
        self_test()
    else:
        run(a)
