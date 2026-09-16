#!/usr/bin/env python3
"""Resource-guarded validation of scGPT iterative dynamics and GRN coupling.

This script addresses four questions in one reproducible protocol:
1. Does a long iteration converge, and is its fixed point close to late cells?
2. Are observed late cells approximately stationary under the same operator?
3. Do middle-pseudotime cells continue along the early-to-late axis without
   collapsing to one common mean state?
4. Are successive changes and TF perturbation responses more coherent on the
   observed directed GRN than on degree-preserving rewired controls?

The default invocation is a read-only preflight. Pass --execute explicitly to
load scGPT and run inference. Outputs contain group-mean trajectories rather
than all cell-by-iteration tensors, keeping memory bounded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="hESC")
    p.add_argument("--outdir", default="outputs/dynamic_grn_validation")
    p.add_argument("--expr-root", default="/mnt/10T/yzn/benchmark_GRN/input_process")
    p.add_argument("--pt-root", default="/mnt/10T/yzn/benchmark_GRN/PseudoTime")
    p.add_argument("--chip-network", default="")
    p.add_argument("--string-network", default="", help="Optional STRING network path (kept for bundle compatibility).")
    p.add_argument("--grn-tsv", default="", help="Optional predicted GRN; CHIP is always the primary graph.")
    p.add_argument("--scgpt-model-dir", required=True)
    p.add_argument("--scgpt-repo-dir", required=True)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    p.add_argument("--execute", action="store_true", help="Actually load the model and run; otherwise preflight only.")
    p.add_argument("--self-test", action="store_true", help="Run dependency-light metric tests and exit.")

    p.add_argument("--gen-iters", type=int, default=32)
    p.add_argument("--max-cells", type=int, default=16, help="Maximum cells per early/middle/late start group.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--ema-alpha", type=float, default=0.1)
    p.add_argument("--pt-quantile", type=float, default=0.2)
    p.add_argument("--top-percent", type=int, default=30)
    p.add_argument("--convergence-tol", type=float, default=1e-4)
    p.add_argument("--convergence-patience", type=int, default=4)
    p.add_argument("--n-rewired", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--n-tf-probes", type=int, default=3)
    p.add_argument("--probe-iters", type=int, default=4)
    p.add_argument("--perturb-delta", type=float, default=1.0)
    p.add_argument("--n-random-target-sets", type=int, default=1000,
                   help="Random label sets for the pure-random target enrichment sensitivity test.")
    p.add_argument("--max-forward-passes", type=int, default=5000)
    p.add_argument("--memory-limit-gb", type=float, default=4.0)
    p.add_argument("--cpu-threads", type=int, default=2, help="PyTorch CPU thread cap for this process.")

    # Compatibility fields consumed by ScgptBundle.
    p.add_argument("--scgpt-bin-log1p", action="store_true", default=False)
    p.add_argument("--scgpt-legacy-pt", action="store_true", default=True)
    p.add_argument("--top-grn-edges", type=int, default=5000)
    p.add_argument("--print-every", type=int, default=4)
    p.set_defaults(grn_source="chip", max_early_cells=16)
    return p.parse_args()


def _json_float(x: float):
    return float(x) if np.isfinite(x) else None


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) <= EPS or np.ptp(y[ok]) <= EPS:
        return float("nan")
    return float(stats.spearmanr(x[ok], y[ok]).statistic)


def degree_preserving_rewire(edges: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Preserve source out-stubs and target in-stubs; reject loops/duplicates."""
    src = edges["Gene1"].astype(str).to_numpy()
    tgt = edges["Gene2"].astype(str).to_numpy()
    if len(src) == 0:
        return edges[["Gene1", "Gene2"]].copy()
    best: List[Tuple[str, str]] = []
    for _ in range(12):
        candidate = list(zip(src.tolist(), rng.permutation(tgt).tolist()))
        candidate = list(dict.fromkeys((a, b) for a, b in candidate if a != b))
        if len(candidate) > len(best):
            best = candidate
        if len(best) >= 0.95 * len(src):
            break
    return pd.DataFrame(best, columns=["Gene1", "Gene2"])


def trajectory_metrics(
    states: np.ndarray,
    cell_variance: np.ndarray,
    early_mean: np.ndarray,
    late_mean: np.ndarray,
    global_mean: np.ndarray,
    tol: float,
    patience: int,
) -> dict:
    """Metrics for states shaped (T+1, genes)."""
    axis = late_mean - early_mean
    axis_norm2 = float(np.dot(axis, axis)) + EPS
    steps = np.diff(states, axis=0)
    step_norm = np.linalg.norm(steps, axis=1)
    state_norm = np.linalg.norm(states[:-1], axis=1)
    relative_step = step_norm / np.maximum(state_norm, EPS)
    dist_late = np.linalg.norm(states - late_mean[None, :], axis=1)
    dist_global = np.linalg.norm(states - global_mean[None, :], axis=1)
    projection = ((states - early_mean[None, :]) @ axis) / axis_norm2
    converged_at = None
    for i in range(max(0, len(relative_step) - patience + 1)):
        if np.all(relative_step[i : i + patience] <= tol):
            converged_at = i + 1
            break
    return {
        "step_norm": step_norm.tolist(),
        "relative_step": relative_step.tolist(),
        "distance_to_late": dist_late.tolist(),
        "distance_to_global": dist_global.tolist(),
        "early_to_late_projection": projection.tolist(),
        "cell_variance": np.asarray(cell_variance, dtype=float).tolist(),
        "converged_at": converged_at,
        "final_relative_step": _json_float(relative_step[-1]),
        "final_distance_to_late": _json_float(dist_late[-1]),
        "final_distance_to_global": _json_float(dist_global[-1]),
        "final_projection": _json_float(projection[-1]),
        "variance_retention": _json_float(cell_variance[-1] / max(cell_variance[0], EPS)),
    }


def edge_lag_score(states: np.ndarray, edges: pd.DataFrame, gene_to_idx: Dict[str, int]) -> dict:
    """TF change at t versus target change at t+1 across directed edges."""
    delta = np.diff(states, axis=0)
    src_idx, dst_idx = [], []
    for a, b in zip(edges["Gene1"], edges["Gene2"]):
        ia, ib = gene_to_idx.get(str(a).upper()), gene_to_idx.get(str(b).upper())
        if ia is not None and ib is not None and ia != ib:
            src_idx.append(ia)
            dst_idx.append(ib)
    if len(src_idx) < 3 or len(delta) < 2:
        return {"n_edges": len(src_idx), "per_lag": [], "median_abs_spearman": None}
    src_idx = np.asarray(src_idx)
    dst_idx = np.asarray(dst_idx)
    per_lag = []
    for t in range(len(delta) - 1):
        rho = safe_spearman(np.abs(delta[t, src_idx]), np.abs(delta[t + 1, dst_idx]))
        per_lag.append(rho)
    finite = np.asarray([x for x in per_lag if np.isfinite(x)])
    return {
        "n_edges": int(len(src_idx)),
        "per_lag": [_json_float(x) for x in per_lag],
        "median_abs_spearman": _json_float(float(np.median(finite))) if len(finite) else None,
    }


def grn_null_test(
    states: np.ndarray,
    edges: pd.DataFrame,
    gene_to_idx: Dict[str, int],
    n_rewired: int,
    rng: np.random.Generator,
) -> dict:
    observed = edge_lag_score(states, edges, gene_to_idx)
    obs = observed["median_abs_spearman"]
    null = []
    for _ in range(n_rewired):
        rewired = degree_preserving_rewire(edges, rng)
        score = edge_lag_score(states, rewired, gene_to_idx)["median_abs_spearman"]
        if score is not None:
            null.append(float(score))
    p = None
    if obs is not None and null:
        p = float((1 + np.sum(np.asarray(null) >= obs)) / (len(null) + 1))
    return {
        "observed": observed,
        "rewired_n": len(null),
        "rewired_mean": _json_float(float(np.mean(null))) if null else None,
        "rewired_q95": _json_float(float(np.quantile(null, 0.95))) if null else None,
        "empirical_p_greater": p,
        "supports_grn_lag_coupling": bool(obs is not None and p is not None and p < 0.05),
    }


def estimate_forward_passes(args: argparse.Namespace) -> int:
    batches = math.ceil(args.max_cells / args.batch_size)
    main = 3 * args.gen_iters * batches
    probes = 2 * args.n_tf_probes * args.probe_iters * batches
    return int(main + probes)


def resolve_input_paths(args: argparse.Namespace) -> Dict[str, Path]:
    chip = Path(args.chip_network) if args.chip_network else (
        Path(args.expr_root) / "CHIP" / f"{args.dataset}_chip_matched-network.csv"
    )
    return {
        "expression": Path(args.expr_root) / "CHIP" / f"{args.dataset}_chip_matched-ExpressionData.csv",
        "pseudotime": Path(args.pt_root) / args.dataset / "PseudoTime.csv",
        "chip_grn": chip,
        "model": Path(args.scgpt_model_dir),
        "scgpt_repo": Path(args.scgpt_repo_dir),
    }


def preflight(args: argparse.Namespace) -> dict:
    if args.gen_iters < 2 or args.max_cells < 1 or args.batch_size < 1:
        raise ValueError("gen-iters >= 2, max-cells >= 1 and batch-size >= 1 are required")
    if not 0.0 <= args.ema_alpha <= 1.0:
        raise ValueError("ema-alpha must be in [0, 1]")
    if args.max_cells > 256:
        raise ValueError("max-cells > 256 is blocked; edit the guard only after profiling")
    if args.gen_iters > 256:
        raise ValueError("gen-iters > 256 is blocked; use staged runs after profiling")
    if not 1 <= args.cpu_threads <= 8:
        raise ValueError("cpu-threads must be between 1 and 8")
    if not 100 <= args.n_random_target_sets <= 100000:
        raise ValueError("n-random-target-sets must be between 100 and 100000")
    paths = resolve_input_paths(args)
    missing = [str(p) for p in paths.values() if not p.exists()]
    forward = estimate_forward_passes(args)
    if forward > args.max_forward_passes:
        raise RuntimeError(f"planned forward passes {forward} exceed cap {args.max_forward_passes}")
    n_cells = n_genes = None
    data_bytes = model_bytes = peak_bytes = None
    if not missing:
        with paths["expression"].open("r", encoding="utf-8-sig", newline="") as f:
            n_cells = max(len(next(csv.reader(f))) - 1, 0)
            n_genes = sum(1 for _ in f)
        # X_bin + values_tensor + a conservative temporary/copy allowance.
        data_bytes = int(n_cells * (n_genes + 1) * 4 * 4)
        checkpoint = paths["model"] / "best_model.pt"
        model_bytes = checkpoint.stat().st_size if checkpoint.is_file() else 0
        # torch.load temporarily holds both checkpoint and instantiated weights.
        peak_bytes = int(data_bytes + 3 * model_bytes)
        if peak_bytes > args.memory_limit_gb * (1024**3):
            raise MemoryError(
                f"estimated peak {peak_bytes / 1024**3:.2f} GiB exceeds "
                f"memory-limit-gb={args.memory_limit_gb}"
            )
    report = {
        "mode": "execute" if args.execute else "preflight_only",
        "paths": {k: str(v) for k, v in paths.items()},
        "missing": missing,
        "estimated_forward_passes": forward,
        "dataset_shape_cells_by_genes": [n_cells, n_genes],
        "estimated_data_gib": None if data_bytes is None else data_bytes / 1024**3,
        "checkpoint_gib": None if model_bytes is None else model_bytes / 1024**3,
        "estimated_peak_gib": None if peak_bytes is None else peak_bytes / 1024**3,
        "limits": {
            "max_cells_per_group": args.max_cells,
            "iterations": args.gen_iters,
            "batch_size": args.batch_size,
            "memory_limit_gb": args.memory_limit_gb,
            "cpu_threads": args.cpu_threads,
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if missing:
        raise FileNotFoundError("preflight found missing inputs")
    return report


def select_indices(mask: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if len(idx) > limit:
        idx = np.sort(rng.choice(idx, size=limit, replace=False))
    return idx


def run_group(bundle, idx: np.ndarray, args: argparse.Namespace, label: str):
    import torch

    if len(idx) == 0:
        raise ValueError(f"empty start group: {label}")
    vals_all = bundle.values_tensor[idx].clone()
    pad = bundle.pad_mask[idx]
    update = (~bundle.gene_ids_tensor.eq(bundle.vocab["<pad>"])).clone()
    update[:, 0] = False
    mapped = update[0, 1:].cpu().numpy()
    states = [vals_all[:, 1:].float().numpy().mean(axis=0)]
    variances = [float(np.mean(np.var(vals_all[:, 1:].float().numpy()[:, mapped], axis=0)))]
    with torch.no_grad():
        for it in range(args.gen_iters):
            for start in range(0, len(idx), args.batch_size):
                end = min(start + args.batch_size, len(idx))
                vals = vals_all[start:end].to(bundle.device)
                src = bundle.gene_ids_tensor.expand(end - start, -1).to(bundle.device)
                mask = pad[start:end].to(bundle.device)
                freeze = mask | (~update.expand(end - start, -1).to(bundle.device))
                pred = bundle.model(src=src, values=vals, src_key_padding_mask=mask)["mlm_output"]
                vals = torch.where(freeze, vals, args.ema_alpha * vals + (1.0 - args.ema_alpha) * pred)
                vals_all[start:end] = vals.detach().cpu()
            arr = vals_all[:, 1:].float().numpy()
            states.append(arr.mean(axis=0))
            variances.append(float(np.mean(np.var(arr[:, mapped], axis=0))))
            if args.print_every and ((it + 1) % args.print_every == 0 or it == 0):
                print(f"[{label}] {it + 1}/{args.gen_iters}", flush=True)
    return np.asarray(states, dtype=np.float32), np.asarray(variances), vals_all


def choose_probe_tfs(edges: pd.DataFrame, gene_to_idx: Dict[str, int], n: int) -> List[str]:
    e = edges[edges.Gene1.isin(gene_to_idx) & edges.Gene2.isin(gene_to_idx)]
    degree = e.groupby("Gene1").size().sort_values(ascending=False)
    return degree.index.astype(str).tolist()[:n]


def tf_response_probe(bundle, early_idx: np.ndarray, args, rng) -> dict:
    if args.n_tf_probes <= 0:
        return {"enabled": False}
    edges = bundle.chip_edges.copy()
    edges["Gene1"] = edges["Gene1"].astype(str).str.upper()
    edges["Gene2"] = edges["Gene2"].astype(str).str.upper()
    pad_id = bundle.vocab["<pad>"]
    mapped_idx = np.flatnonzero(bundle.gene_ids_tensor[0, 1:].cpu().numpy() != pad_id)
    mapped_map = {bundle.genes[i]: i for i in mapped_idx}
    tfs = choose_probe_tfs(edges, mapped_map, args.n_tf_probes)
    base_expr = bundle.values_tensor[early_idx, 1:].float().numpy().mean(axis=0)
    rows = []
    original_iters = args.gen_iters
    args.gen_iters = args.probe_iters
    try:
        for tf in tfs:
            gi = bundle.gene_to_idx[tf]
            finals = []
            for sign in (-1.0, 1.0):
                vals0 = bundle.values_tensor[early_idx].clone()
                vals0[:, gi + 1] = (vals0[:, gi + 1].float() + sign * args.perturb_delta).clamp(0, 50).to(vals0.dtype)
                # Inline the same bounded update, but start from the perturbed tensor.
                import torch
                pad = bundle.pad_mask[early_idx]
                update = (~bundle.gene_ids_tensor.eq(bundle.vocab["<pad>"])).clone()
                update[:, 0] = False
                with torch.no_grad():
                    for _ in range(args.probe_iters):
                        for start in range(0, len(early_idx), args.batch_size):
                            end = min(start + args.batch_size, len(early_idx))
                            vals = vals0[start:end].to(bundle.device)
                            src = bundle.gene_ids_tensor.expand(end - start, -1).to(bundle.device)
                            mask = pad[start:end].to(bundle.device)
                            freeze = mask | (~update.expand(end - start, -1).to(bundle.device))
                            pred = bundle.model(src=src, values=vals, src_key_padding_mask=mask)["mlm_output"]
                            vals = torch.where(freeze, vals, args.ema_alpha * vals + (1 - args.ema_alpha) * pred)
                            vals0[start:end] = vals.detach().cpu()
                finals.append(vals0[:, 1:].float().numpy().mean(axis=0))
            response = np.abs(finals[1] - finals[0]) / (2 * args.perturb_delta)
            targets = sorted(set(edges.loc[edges.Gene1 == tf, "Gene2"]) & set(mapped_map))
            target_idx = np.asarray([bundle.gene_to_idx[g] for g in targets], dtype=int)
            excluded = set(target_idx.tolist()) | {gi}
            pool = np.asarray([i for i in mapped_idx if i not in excluded], dtype=int)
            if len(target_idx) == 0 or len(pool) == 0:
                continue
            # Greedy expression matching gives a stricter control than uniform sampling.
            available = pool.tolist()
            control = []
            for ti in target_idx:
                if not available:
                    break
                pos = int(np.argmin(np.abs(base_expr[np.asarray(available)] - base_expr[ti])))
                control.append(available.pop(pos))
            control_idx = np.asarray(control, dtype=int)
            target_effect = float(np.mean(response[target_idx]))
            control_effect = float(np.mean(response[control_idx]))
            all_nontarget_effect = float(np.mean(response[pool]))
            target_gene_responses = sorted(
                (
                    {"gene": gene, "response": float(response[bundle.gene_to_idx[gene]])}
                    for gene in targets
                ),
                key=lambda item: item["response"],
                reverse=True,
            )
            # Pure-random label null: draw sets of the same size as the real
            # target set from all mapped genes except the perturbed TF. This is
            # valid even when the real target set is larger than the non-target pool.
            eligible = np.asarray([i for i in mapped_idx if i != gi], dtype=int)
            n_pick = min(len(target_idx), len(eligible))
            null_means = np.asarray([
                float(np.mean(response[rng.choice(eligible, size=n_pick, replace=False)]))
                for _ in range(args.n_random_target_sets)
            ])
            random_p = float((1 + np.sum(null_means >= target_effect)) / (len(null_means) + 1))
            rows.append({
                "tf": tf,
                "n_targets": int(len(target_idx)),
                "n_nontargets": int(len(pool)),
                "mean_target_response": target_effect,
                "mean_control_response": control_effect,
                "target_control_ratio": target_effect / max(control_effect, EPS),
                "mean_all_nontarget_response": all_nontarget_effect,
                "target_all_nontarget_ratio": target_effect / max(all_nontarget_effect, EPS),
                "target_gene_responses": target_gene_responses,
                "random_label_null_mean": float(np.mean(null_means)),
                "random_label_null_q05": float(np.quantile(null_means, 0.05)),
                "random_label_null_q95": float(np.quantile(null_means, 0.95)),
                "random_label_empirical_p": random_p,
            })
    finally:
        args.gen_iters = original_iters
    ratios = [r["target_control_ratio"] for r in rows]
    return {
        "enabled": True,
        "rows": rows,
        "median_target_control_ratio": _json_float(float(np.median(ratios))) if ratios else None,
        "supports_target_specific_response": bool(ratios and np.median(ratios) > 1.0),
        "note": "Exploratory finite-difference evidence; use more TFs and matched controls for a final claim.",
    }


def execute(args: argparse.Namespace, preflight_report: dict) -> None:
    import torch

    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    sys.path.insert(0, str(HERE))
    from run_grn_perturbation_probes import ScgptBundle, set_seed

    set_seed(args.seed)
    args.max_early_cells = args.max_cells
    bundle = ScgptBundle(args)
    rng = np.random.default_rng(args.seed)
    estimated_tensor_bytes = 3 * args.max_cells * (len(bundle.genes) + 1) * 4
    if estimated_tensor_bytes > args.memory_limit_gb * (1024**3):
        raise MemoryError("estimated state tensors exceed memory-limit-gb")
    if bundle.device.type == "cuda":
        # torch 1.13 requires an explicit integer index; newer versions also accept it.
        free_b, total_b = torch.cuda.mem_get_info(torch.cuda.current_device())
        if free_b < max(2 * estimated_tensor_bytes, 512 * 1024**2):
            raise MemoryError(f"insufficient free GPU memory: {free_b / 1024**3:.2f} GiB")

    groups = {
        "early": select_indices(bundle.early, args.max_cells, rng),
        "middle": select_indices(bundle.mid, args.max_cells, rng),
        "late": select_indices(bundle.late, args.max_cells, rng),
    }
    outdir = Path(args.outdir) / args.dataset
    outdir.mkdir(parents=True, exist_ok=True)
    trajectories, metrics = {}, {}
    pad_id = bundle.vocab["<pad>"]
    mapped_idx = np.flatnonzero(bundle.gene_ids_tensor[0, 1:].cpu().numpy() != pad_id)
    eval_gene_to_idx = {bundle.genes[old_i]: new_i for new_i, old_i in enumerate(mapped_idx)}
    for name, idx in groups.items():
        states, variance, _ = run_group(bundle, idx, args, name)
        trajectories[name] = states
        metrics[name] = trajectory_metrics(
            states[:, mapped_idx], variance, bundle.early_mean[mapped_idx],
            bundle.late_mean[mapped_idx], bundle.global_mean[mapped_idx],
            args.convergence_tol, args.convergence_patience,
        )
        np.save(outdir / f"{name}_mean_trajectory.npy", states)

    chip_edges = bundle.chip_edges.copy()
    chip_edges["Gene1"] = chip_edges.Gene1.astype(str).str.upper()
    chip_edges["Gene2"] = chip_edges.Gene2.astype(str).str.upper()
    grn_tests = {
        name: grn_null_test(states[:, mapped_idx], chip_edges, eval_gene_to_idx, args.n_rewired, rng)
        for name, states in trajectories.items()
    }
    tf_probe = tf_response_probe(bundle, groups["early"], args, rng)

    late_initial_move = metrics["late"]["step_norm"][0]
    early_initial_move = metrics["early"]["step_norm"][0]
    report = {
        "preflight": preflight_report,
        "dataset": args.dataset,
        "n_genes": len(bundle.genes),
        "n_mapped_genes": int(len(mapped_idx)),
        "group_sizes": {k: int(len(v)) for k, v in groups.items()},
        "metrics": metrics,
        "reviewer_checks": {
            "long_run_converged": metrics["early"]["converged_at"] is not None,
            "fixed_point_closer_to_late_than_global": metrics["early"]["final_distance_to_late"] < metrics["early"]["final_distance_to_global"],
            "late_relative_initial_movement": late_initial_move / max(early_initial_move, EPS),
            "late_is_approximately_stationary": late_initial_move < early_initial_move,
            "middle_progresses_toward_late": metrics["middle"]["final_projection"] > metrics["middle"]["early_to_late_projection"][0],
            "middle_variance_retention": metrics["middle"]["variance_retention"],
        },
        "grn_lag_tests": grn_tests,
        "tf_finite_difference_probe": tf_probe,
        "interpretation_guard": "GRN association requires real-edge scores above rewired controls and target-specific TF responses; convergence alone is not GRN evidence.",
    }
    with open(outdir / "dynamic_grn_validation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Wrote {outdir / 'dynamic_grn_validation.json'}")


def self_test() -> None:
    genes = ["A", "B", "C", "D"]
    g2i = {g: i for i, g in enumerate(genes)}
    states = np.asarray([
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.1, 2.0, 0.0, 1.0],
        [1.11, 2.2, 3.0, 1.0],
    ])
    edges = pd.DataFrame({"Gene1": ["A", "B", "C"], "Gene2": ["B", "C", "D"]})
    score = edge_lag_score(states, edges, g2i)
    assert score["n_edges"] == 3
    m = trajectory_metrics(states, np.ones(4), states[0], states[-1], states.mean(0), 1e-8, 2)
    assert len(m["step_norm"]) == 3
    rewired = degree_preserving_rewire(edges, np.random.default_rng(0))
    assert set(rewired.columns) == {"Gene1", "Gene2"}
    print("self-test: OK")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    report = preflight(args)
    if not args.execute:
        print("Preflight only. Add --execute after reviewing the estimate.")
        return
    execute(args, report)


if __name__ == "__main__":
    main()
