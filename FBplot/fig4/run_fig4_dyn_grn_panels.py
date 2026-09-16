#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4 redesign: dynamics organized by GRN structure (not gene coverage).

Panel A — matched enrichment of dynamic genes in CHIP nodes / TFs / targets
           + abs_delta ~ log1p(degree) + expression covariates
Panel B — CHIP TF→target pseudotime-lag coherence vs rewired / matched-random edges
Panel C — held-out target prediction: update S only, score on T (T ∩ S = ∅)
           real GRN regulators vs expression/degree-matched random vs rewired S

Gene-count sweep is NOT used as the main claim (coverage confound).

Examples
--------
  # A + B only (fast, no GPU)
  python run_fig4_dyn_grn_panels.py --dataset hESC --panels AB

  # Full A+B+C
  python run_fig4_dyn_grn_panels.py --dataset hESC --panels ABC \\
    --scgpt-model-dir /mnt/10T/yzn/benchmark_GRN/pre_scgpt/scGPT/scgpt_human \\
    --scgpt-repo-dir /mnt/10T/yzn/benchmark_GRN/pre_scgpt/scGPT
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from dyn_grn_common import (  # noqa: E402
    covariate_matrix,
    direction_accuracy_on_idx,
    edge_lag_scores,
    emp_p,
    enrichment_fold,
    expression_matched_random_edges,
    hypergeom_p,
    load_dyn_grn_data,
    matched_null_sets,
    ols_partial_spearman,
    rewire_directed_degree_preserving,
    select_regulators_and_heldout_targets,
)
from fig4_palette import apply_fig4_style, model_color  # noqa: E402


# ---------------------------------------------------------------------------
# Panel A
# ---------------------------------------------------------------------------

def run_panel_a(data, rng: np.random.Generator, n_null: int = 2000) -> dict:
    cov = covariate_matrix(data)
    dyn_idx = np.where(data.top_dyn_mask)[0]
    all_idx = np.arange(len(data.genes))
    pool = all_idx

    tests = {
        "in_CHIP_node": data.is_node,
        "is_CHIP_TF": data.is_tf,
        "is_CHIP_target": data.is_target,
    }
    rows = []
    for name, mask in tests.items():
        obs_overlap = int(mask[dyn_idx].sum())
        n_set = int(len(dyn_idx))
        n_bg_in = int(mask.sum())
        n_bg = int(len(data.genes))
        fold = enrichment_fold(obs_overlap, n_set, n_bg_in, n_bg)
        p_hyp_g = hypergeom_p(obs_overlap, n_set, n_bg_in, n_bg, "greater")
        p_hyp_l = hypergeom_p(obs_overlap, n_set, n_bg_in, n_bg, "less")

        # matched null: fraction of null dyn-sets overlapping the category
        # Equivalent: sample matched gene sets of size |dyn|, count membership
        nulls = matched_null_sets(dyn_idx, pool, cov, rng, n_null=n_null)
        null_ov = [int(mask[n].sum()) for n in nulls]
        p_gt = emp_p(obs_overlap, null_ov, "greater")
        p_lt = emp_p(obs_overlap, null_ov, "less")
        rows.append(
            {
                "test": name,
                "n_dynamic": n_set,
                "n_category": n_bg_in,
                "overlap": obs_overlap,
                "fold_vs_background": fold,
                "hypergeom_p_greater": p_hyp_g,
                "hypergeom_p_less": p_hyp_l,
                "matched_null_mean_overlap": float(np.mean(null_ov)),
                "matched_null_q05": float(np.quantile(null_ov, 0.05)),
                "matched_null_q95": float(np.quantile(null_ov, 0.95)),
                "matched_p_greater": p_gt,
                "matched_p_less": p_lt,
                "direction": (
                    "enrichment"
                    if p_gt < 0.05
                    else ("depletion" if p_lt < 0.05 else "ns")
                ),
            }
        )

    # STRING naive coverage (for transparency; not the main claim)
    string_row = None
    if data.string_edges is not None:
        s_nodes = set(data.string_edges.Gene1) | set(data.string_edges.Gene2)
        in_string = np.array([g in s_nodes for g in data.genes], dtype=bool)
        ov = int(in_string[dyn_idx].sum())
        string_row = {
            "test": "in_STRING_node",
            "n_dynamic": int(len(dyn_idx)),
            "n_category": int(in_string.sum()),
            "overlap": ov,
            "fold_vs_background": enrichment_fold(
                ov, len(dyn_idx), int(in_string.sum()), len(data.genes)
            ),
            "hypergeom_p_greater": hypergeom_p(
                ov, len(dyn_idx), int(in_string.sum()), len(data.genes), "greater"
            ),
            "hypergeom_p_less": hypergeom_p(
                ov, len(dyn_idx), int(in_string.sum()), len(data.genes), "less"
            ),
            "note": "naive; covariates not controlled — do not treat as final biology",
        }

    # Topology: abs_delta ~ log1p(undirected degree) + covariates
    # Covariates for residualization exclude degree itself
    cov_no_deg = np.column_stack(
        [
            data.mean_expr,
            data.detection_rate,
            np.log1p(data.expr_var),
            data.is_tf.astype(np.float64),
        ]
    )
    reg = ols_partial_spearman(
        y=data.abs_delta,
        x=np.log1p(data.undirected_degree.astype(float)),
        covariates=cov_no_deg,
        rng=rng,
        n_perm=n_null,
    )
    # also out-degree among nodes with out_degree>0 or all genes
    reg_out = ols_partial_spearman(
        y=data.abs_delta,
        x=np.log1p(data.out_degree.astype(float)),
        covariates=cov_no_deg,
        rng=rng,
        n_perm=n_null,
    )

    # Dynamic–dynamic connectivity: fraction of dyn-dyn undirected pairs that are edges
    dyn_set = set(dyn_idx.tolist())
    und_pairs = set()
    for a, b in zip(data.chip_edges.Gene1, data.chip_edges.Gene2):
        ia, ib = data.gene_to_idx[a], data.gene_to_idx[b]
        u, v = (ia, ib) if ia < ib else (ib, ia)
        und_pairs.add((u, v))
    dyn_list = list(dyn_idx)
    n_possible = len(dyn_list) * (len(dyn_list) - 1) // 2
    n_dd = sum(
        1
        for i in range(len(dyn_list))
        for j in range(i + 1, len(dyn_list))
        if (min(dyn_list[i], dyn_list[j]), max(dyn_list[i], dyn_list[j])) in und_pairs
    )
    obs_rate = n_dd / n_possible if n_possible else float("nan")
    null_rates = []
    null_sets = matched_null_sets(dyn_idx, pool, cov, rng, n_null=min(n_null, 500))
    for ns in null_sets:
        nl = list(ns)
        npairs = len(nl) * (len(nl) - 1) // 2
        if npairs == 0:
            null_rates.append(0.0)
            continue
        nd = sum(
            1
            for i in range(len(nl))
            for j in range(i + 1, len(nl))
            if (min(nl[i], nl[j]), max(nl[i], nl[j])) in und_pairs
        )
        null_rates.append(nd / npairs)

    conn = {
        "n_dyn_dyn_edges": int(n_dd),
        "n_possible_pairs": int(n_possible),
        "edge_rate": float(obs_rate),
        "matched_null_mean_rate": float(np.mean(null_rates)) if null_rates else float("nan"),
        "matched_p_greater": emp_p(obs_rate, null_rates, "greater") if null_rates else float("nan"),
    }

    # TF proximity: min hop to any TF among dynamic vs matched null (undirected CHIP)
    # BFS from TFs
    adj = {i: set() for i in range(len(data.genes))}
    for a, b in zip(data.chip_edges.Gene1, data.chip_edges.Gene2):
        ia, ib = data.gene_to_idx[a], data.gene_to_idx[b]
        adj[ia].add(ib)
        adj[ib].add(ia)
    from collections import deque

    dist = np.full(len(data.genes), 999, dtype=int)
    q = deque()
    for i in np.where(data.is_tf)[0]:
        dist[i] = 0
        q.append(i)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                q.append(v)
    # among non-TF genes that are reachable
    non_tf = np.where(~data.is_tf & (dist < 999))[0]
    dyn_nt = [i for i in dyn_idx if i in set(non_tf.tolist())]
    obs_dist = float(np.mean(dist[dyn_nt])) if dyn_nt else float("nan")
    null_dmeans = []
    if dyn_nt:
        cov_nt = cov
        nulls_d = matched_null_sets(dyn_nt, non_tf, cov_nt, rng, n_null=min(n_null, 500))
        for ns in nulls_d:
            null_dmeans.append(float(np.mean(dist[ns])))
    prox = {
        "mean_dist_to_TF_dynamic": obs_dist,
        "matched_null_mean_dist": float(np.mean(null_dmeans)) if null_dmeans else float("nan"),
        "matched_p_closer": emp_p(obs_dist, null_dmeans, "less") if null_dmeans else float("nan"),
        "n_dynamic_nonTF_reachable": int(len(dyn_nt)),
    }

    return {
        "enrichment": rows,
        "string_naive": string_row,
        "degree_partial_spearman": reg,
        "outdegree_partial_spearman": reg_out,
        "dyn_dyn_connectivity": conn,
        "tf_proximity": prox,
        "n_genes": len(data.genes),
        "n_dynamic": int(len(dyn_idx)),
        "n_chip_edges": int(len(data.chip_edges)),
        "n_tf": int(data.is_tf.sum()),
        "n_target": int(data.is_target.sum()),
    }


# ---------------------------------------------------------------------------
# Panel B
# ---------------------------------------------------------------------------

def run_panel_b(
    data,
    rng: np.random.Generator,
    n_null: int = 200,
) -> dict:
    real = edge_lag_scores(data, data.chip_edges)
    # strip arrays for JSON
    real_summary = {k: v for k, v in real.items() if k not in ("xs", "ys")}

    rewired_sign, rewired_rho, rewired_prod = [], [], []
    for i in range(n_null):
        re = rewire_directed_degree_preserving(data.chip_edges, np.random.default_rng(rng.integers(1e9)))
        sc = edge_lag_scores(data, re)
        rewired_sign.append(sc["sign_agree"])
        rewired_rho.append(sc["spearman"])
        rewired_prod.append(sc["mean_product"])

    rand_sign, rand_rho, rand_prod = [], [], []
    for i in range(n_null):
        re = expression_matched_random_edges(
            data, n_edges=len(data.chip_edges), rng=np.random.default_rng(rng.integers(1e9))
        )
        sc = edge_lag_scores(data, re)
        rand_sign.append(sc["sign_agree"])
        rand_rho.append(sc["spearman"])
        rand_prod.append(sc["mean_product"])

    # completely random gene pairs (no expression matching)
    unif_sign, unif_rho = [], []
    genes = data.genes
    for i in range(n_null):
        rr = np.random.default_rng(rng.integers(1e9))
        ia = rr.integers(0, len(genes), size=len(data.chip_edges))
        ib = rr.integers(0, len(genes), size=len(data.chip_edges))
        mask = ia != ib
        ed = pd.DataFrame({"Gene1": [genes[i] for i in ia[mask]], "Gene2": [genes[j] for j in ib[mask]]})
        sc = edge_lag_scores(data, ed)
        unif_sign.append(sc["sign_agree"])
        unif_rho.append(sc["spearman"])

    def pack(obs_key, null_list, alt="greater"):
        return {
            "obs": real_summary[obs_key],
            "null_mean": float(np.nanmean(null_list)),
            "null_q05": float(np.nanquantile(null_list, 0.05)),
            "null_q95": float(np.nanquantile(null_list, 0.95)),
            "p": emp_p(real_summary[obs_key], null_list, alt),
        }

    return {
        "real": real_summary,
        "vs_rewired": {
            "sign_agree": pack("sign_agree", rewired_sign),
            "spearman": pack("spearman", rewired_rho),
            "mean_product": pack("mean_product", rewired_prod),
        },
        "vs_expr_matched_random": {
            "sign_agree": pack("sign_agree", rand_sign),
            "spearman": pack("spearman", rand_rho),
            "mean_product": pack("mean_product", rand_prod),
        },
        "vs_uniform_random": {
            "sign_agree": pack("sign_agree", unif_sign),
            "spearman": pack("spearman", unif_rho),
        },
        "n_null": n_null,
        "null_draws": {
            "rewired_sign": rewired_sign,
            "rewired_rho": rewired_rho,
            "expr_matched_sign": rand_sign,
            "expr_matched_rho": rand_rho,
            "uniform_sign": unif_sign,
            "uniform_rho": unif_rho,
        },
    }


# ---------------------------------------------------------------------------
# Panel C
# ---------------------------------------------------------------------------

def _build_scgpt_args(args):
    """Namespace compatible with ScgptBundle."""
    class A:
        pass

    a = A()
    for k, v in vars(args).items():
        setattr(a, k, v)
    # defaults expected by ScgptBundle
    a.chip_network = getattr(args, "chip_network", "") or ""
    a.string_network = getattr(args, "string_network", "") or ""
    a.grn_tsv = getattr(args, "grn_tsv", "") or ""
    a.grn_source = "chip"
    a.c1_intersect_top_dynamic = False
    a.top_grn_edges = 5000
    a.scgpt_bin_log1p = False
    a.scgpt_legacy_pt = True
    a.print_every = 0
    a.ema_alpha = args.ema_alpha
    a.batch_size = args.batch_size
    a.max_early_cells = args.max_early_cells
    a.pt_quantile = 0.2  # early/late for true_delta in bundle (eval uses same)
    a.top_percent = args.top_percent
    return a


def run_panel_c(
    data,
    args,
    rng: np.random.Generator,
    outdir: Path,
) -> dict:
    """
    Held-out target prediction without coverage confound.

    Protocol:
      1. Start from early cells.
      2. Clamp S to late_mean (teacher-force regulators as context).
      3. Update mask = T only (held-out targets); S stays frozen.
      4. Score direction accuracy ONLY on T (T ∩ S = ∅).

    Controls (same T):
      - matched_random: covariate-matched clamp genes
      - nonparent_matched: matched genes with no CHIP edge into T
      - rewired_GRN: parents of T under config-model rewiring (exclude real S)
      - baseline_noclamp: update T only, no clamp
    """
    from run_grn_perturbation_probes import ScgptBundle, run_iteration
    from run_unified_multidataset_pseudotime import set_seed

    set_seed(args.seed)
    ba = _build_scgpt_args(args)
    bundle = ScgptBundle(ba)

    S, T, edges_st = select_regulators_and_heldout_targets(
        data,
        k_reg=args.k_reg,
        min_targets_per_tf=args.min_targets_per_tf,
        require_dyn_targets=True,
    )
    if len(T) < 10:
        S, T, edges_st = select_regulators_and_heldout_targets(
            data,
            k_reg=args.k_reg,
            min_targets_per_tf=args.min_targets_per_tf,
            require_dyn_targets=False,
        )
    T_idx = [data.gene_to_idx[g] for g in T]
    S_idx = [data.gene_to_idx[g] for g in S]
    assert len(set(S) & set(T)) == 0

    S_ok = [g for g in S if g in bundle.gene_to_idx]
    T_bundle = [bundle.gene_to_idx[g] for g in T if g in bundle.gene_to_idx]
    T_set = set(T)

    def values_with_clamp(clamp_genes: Sequence[str]) -> "torch.Tensor":
        vals = bundle.values_tensor[bundle.init_idx].clone()
        for g in clamp_genes:
            gi = bundle.gene_to_idx.get(g)
            if gi is None:
                continue
            # token index = gene index + 1 (CLS)
            vals[:, gi + 1] = float(bundle.late_mean[gi])
        return vals

    def update_mask_targets_only(target_genes: Sequence[str]) -> np.ndarray:
        """Only held-out targets may update; everything else (incl. clamped S) frozen."""
        umask = np.zeros(bundle.gene_ids_tensor.shape[1], dtype=bool)
        for g in target_genes:
            gi = bundle.gene_to_idx.get(g)
            if gi is not None:
                umask[gi + 1] = True
        return umask

    # --- hard T: genes baseline fails (direction wrong with no clamp) ---
    if getattr(args, "hard_T", True):
        base_delta = run_iteration(
            bundle,
            update_mask_targets_only(T),
            args.gen_iters,
            values_init=values_with_clamp([]),
            label="baseline_for_hardT",
        )["pred_delta_final"]
        td = bundle.true_delta
        wrong = []
        for gi in T_bundle:
            true_dir = 1 if td[gi] > 0 else -1
            pred_dir = 1 if base_delta[gi] > 0 else -1
            if true_dir != pred_dir:
                wrong.append(bundle.genes[gi])
        if len(wrong) >= 15:
            T = wrong
            T_bundle = [bundle.gene_to_idx[g] for g in T]
            T_set = set(T)
            print(f"  [hard T] restricted to {len(T)} genes baseline gets wrong", flush=True)
        else:
            print(
                f"  [hard T] only {len(wrong)} wrong under baseline — keep full T={len(T_bundle)}",
                flush=True,
            )

    def score_run(clamp_genes: Sequence[str], label: str) -> dict:
        vals0 = values_with_clamp(clamp_genes)
        umask = update_mask_targets_only(T)
        r = run_iteration(
            bundle, umask, args.gen_iters, values_init=vals0, label=label
        )
        acc_T = direction_accuracy_on_idx(
            r["pred_delta_final"], bundle.true_delta, T_bundle
        )
        S_b = [bundle.gene_to_idx[g] for g in clamp_genes if g in bundle.gene_to_idx]
        acc_S = direction_accuracy_on_idx(
            r["pred_delta_final"], bundle.true_delta, S_b
        )
        return {
            "label": label,
            "n_clamp": len(clamp_genes),
            "acc_heldout_T": acc_T,
            "acc_on_clamped_S": acc_S,
            "acc_top30_dynamic": float(r["final_acc"]),
            "n_T": len(T_bundle),
        }

    rows = []
    print(
        f"[C] clamp |S|={len(S_ok)} to late; update ONLY |T|={len(T_bundle)}; score T "
        f"(S∩T=∅)",
        flush=True,
    )

    base = score_run([], "baseline_noclamp_update_T")
    rows.append({**base, "draw": "obs", "condition": "baseline_noclamp"})
    print(f"  baseline_noclamp (update T only) acc_T={base['acc_heldout_T']:.3f}", flush=True)

    obs = score_run(S_ok, "real_GRN_regulators")
    rows.append({**obs, "draw": "obs", "condition": "real_GRN"})
    print(
        f"  real  acc_T={obs['acc_heldout_T']:.3f} "
        f"acc_S(clamped)={obs['acc_on_clamped_S']:.3f}",
        flush=True,
    )

    cov = covariate_matrix(data)
    null_S_sets = matched_null_sets(
        S_idx, np.arange(len(data.genes)), cov, rng, n_null=args.n_null_c
    )
    matched_acc = []
    for i, ns in enumerate(null_S_sets):
        genes = [data.genes[j] for j in ns]
        # keep T out of clamp set
        genes = [g for g in genes if g not in T_set][: args.k_reg]
        if len(genes) < args.k_reg:
            extra = [
                data.genes[j]
                for j in rng.permutation(len(data.genes))
                if data.genes[j] not in T_set and data.genes[j] not in genes
            ]
            genes = (genes + extra)[: args.k_reg]
        rr = score_run(genes, f"matched_rand_{i}")
        rows.append({**rr, "draw": f"null_{i}", "condition": "matched_random"})
        matched_acc.append(rr["acc_heldout_T"])
        if (i + 1) % 5 == 0 or i == 0:
            print(
                f"  matched {i+1}/{args.n_null_c} acc_T={rr['acc_heldout_T']:.3f}",
                flush=True,
            )

    # Structural null: non-parents of T (no CHIP edge into any t∈T), matched to S
    real_parents = set()
    for a, b in zip(data.chip_edges.Gene1, data.chip_edges.Gene2):
        if b in T_set:
            real_parents.add(a)
    nonparent_idx = [
        i
        for i, g in enumerate(data.genes)
        if g not in T_set and g not in real_parents
    ]
    nonparent_acc: List[float] = []
    if len(nonparent_idx) >= args.k_reg:
        np_sets = matched_null_sets(
            S_idx, nonparent_idx, cov, rng, n_null=args.n_null_c
        )
        for i, ns in enumerate(np_sets):
            genes = [data.genes[j] for j in ns]
            rr = score_run(genes, f"nonparent_{i}")
            rows.append({**rr, "draw": f"nonparent_{i}", "condition": "nonparent_matched"})
            nonparent_acc.append(rr["acc_heldout_T"])
            if (i + 1) % 5 == 0 or i == 0:
                print(
                    f"  nonparent {i+1}/{args.n_null_c} acc_T={rr['acc_heldout_T']:.3f}",
                    flush=True,
                )
    else:
        print("  [warn] too few non-parents for structural null", flush=True)

    # Rewired parents of T, excluding real S when possible.
    # NOTE: global top-out-degree is invariant under config-model — do not use it.
    rewired_acc: List[float] = []
    S_real_set = set(S_ok)
    for i in range(args.n_null_c):
        re = rewire_directed_degree_preserving(
            data.chip_edges, np.random.default_rng(rng.integers(1e9))
        )
        into_count: Dict[str, int] = {}
        for a, b in zip(re.Gene1, re.Gene2):
            if b in T_set and a not in T_set:
                into_count[a] = into_count.get(a, 0) + 1
        ranked = sorted(into_count.keys(), key=lambda g: (-into_count[g], g))
        S_rew = [g for g in ranked if g not in S_real_set][: args.k_reg]
        if len(S_rew) < args.k_reg:
            for g in ranked:
                if g not in S_rew:
                    S_rew.append(g)
                if len(S_rew) >= args.k_reg:
                    break
        if len(S_rew) < args.k_reg and nonparent_idx:
            extra = [
                data.genes[j]
                for j in rng.permutation(nonparent_idx)
                if data.genes[j] not in S_rew
            ]
            S_rew = (S_rew + extra)[: args.k_reg]
        rr = score_run(S_rew, f"rewired_{i}")
        ov = len(set(S_rew) & S_real_set) / max(len(S_ok), 1)
        rows.append(
            {
                **rr,
                "draw": f"rewired_{i}",
                "condition": "rewired_GRN",
                "overlap_with_real_S": ov,
            }
        )
        rewired_acc.append(rr["acc_heldout_T"])
        if (i + 1) % 5 == 0 or i == 0:
            print(
                f"  rewired {i+1}/{args.n_null_c} acc_T={rr['acc_heldout_T']:.3f} "
                f"overlap_S={ov:.2f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "panel_c_draws.csv", index=False)
    edges_st.to_csv(outdir / "panel_c_S_to_T_edges.csv", index=False)
    pd.Series(S_ok, name="regulator").to_csv(outdir / "panel_c_S.csv", index=False)
    pd.Series(T, name="heldout_target").to_csv(outdir / "panel_c_T.csv", index=False)

    summary = {
        "protocol": "clamp_S_to_late_update_only_T_score_T",
        "hard_T": bool(getattr(args, "hard_T", True)),
        "k_reg": args.k_reg,
        "n_S": len(S_ok),
        "n_T": len(T_bundle),
        "n_edges_S_to_T": int(len(edges_st)),
        "baseline_noclamp_acc_T": float(base["acc_heldout_T"]),
        "real_acc_T": obs["acc_heldout_T"],
        "real_acc_S_clamped": obs["acc_on_clamped_S"],
        "matched_mean_acc_T": float(np.mean(matched_acc)),
        "matched_q05": float(np.quantile(matched_acc, 0.05)),
        "matched_q95": float(np.quantile(matched_acc, 0.95)),
        "p_real_gt_matched": emp_p(obs["acc_heldout_T"], matched_acc, "greater"),
        "rewired_mean_acc_T": float(np.mean(rewired_acc)) if rewired_acc else float("nan"),
        "rewired_q05": float(np.quantile(rewired_acc, 0.05)) if rewired_acc else float("nan"),
        "rewired_q95": float(np.quantile(rewired_acc, 0.95)) if rewired_acc else float("nan"),
        "p_real_gt_rewired": emp_p(obs["acc_heldout_T"], rewired_acc, "greater")
        if rewired_acc
        else float("nan"),
        "nonparent_mean_acc_T": float(np.mean(nonparent_acc)) if nonparent_acc else float("nan"),
        "nonparent_q05": float(np.quantile(nonparent_acc, 0.05)) if nonparent_acc else float("nan"),
        "nonparent_q95": float(np.quantile(nonparent_acc, 0.95)) if nonparent_acc else float("nan"),
        "p_real_gt_nonparent": emp_p(obs["acc_heldout_T"], nonparent_acc, "greater")
        if nonparent_acc
        else float("nan"),
        "supports_claim": bool(
            emp_p(obs["acc_heldout_T"], matched_acc, "greater") < 0.05
            and (
                (rewired_acc and emp_p(obs["acc_heldout_T"], rewired_acc, "greater") < 0.05)
                or (
                    nonparent_acc
                    and emp_p(obs["acc_heldout_T"], nonparent_acc, "greater") < 0.05
                )
            )
        ),
    }
    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_figure(
    panel_a: Optional[dict],
    panel_b: Optional[dict],
    panel_c: Optional[dict],
    out_png: Path,
    dataset: str,
) -> None:
    apply_fig4_style()
    n_panels = sum(x is not None for x in (panel_a, panel_b, panel_c))
    fig, axes = plt.subplots(1, max(n_panels, 1), figsize=(4.2 * max(n_panels, 1), 4.0), dpi=200)
    if n_panels == 1:
        axes = [axes]
    ax_i = 0
    col_real = model_color("scGPT")
    col_null = "#9AA0A6"
    col_rew = "#EB7E60"

    if panel_a is not None:
        ax = axes[ax_i]
        ax_i += 1
        enr = panel_a["enrichment"]
        labels = []
        obs_frac = []
        null_mu = []
        null_lo = []
        null_hi = []
        for r in enr:
            labels.append(
                {
                    "in_CHIP_node": "CHIP node",
                    "is_CHIP_TF": "CHIP TF",
                    "is_CHIP_target": "CHIP target",
                }[r["test"]]
            )
            obs_frac.append(r["overlap"] / r["n_dynamic"])
            null_mu.append(r["matched_null_mean_overlap"] / r["n_dynamic"])
            null_lo.append(r["matched_null_q05"] / r["n_dynamic"])
            null_hi.append(r["matched_null_q95"] / r["n_dynamic"])
        x = np.arange(len(labels))
        ax.bar(x - 0.18, obs_frac, 0.36, color=col_real, label="Top-dynamic")
        ax.bar(x + 0.18, null_mu, 0.36, color=col_null, label="Matched null")
        ax.errorbar(
            x + 0.18,
            null_mu,
            yerr=[
                np.array(null_mu) - np.array(null_lo),
                np.array(null_hi) - np.array(null_mu),
            ],
            fmt="none",
            ecolor="0.3",
            capsize=3,
            lw=1,
        )
        # annotate fold / p
        for i, r in enumerate(enr):
            tag = r["direction"]
            ax.text(i, max(obs_frac[i], null_mu[i]) + 0.03, tag, ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Fraction of top-dynamic genes")
        ax.set_title("A  Matched enrichment", loc="left", fontsize=11, fontweight="600")
        ax.set_ylim(0, min(1.05, max(obs_frac + null_hi) + 0.15))
        ax.legend(frameon=False, fontsize=8)
        # small text for degree correlation
        sp = panel_a["degree_partial_spearman"]
        ax.text(
            0.02,
            0.98,
            f"|Δ|~degree ρ={sp['spearman_partial']:.2f}\nperm p={sp['spearman_perm_p']:.3g}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color="0.25",
        )

    if panel_b is not None:
        ax = axes[ax_i]
        ax_i += 1
        # Primary metric: Spearman(ΔTF early→mid, Δtarget mid→late)
        # (binary sign-agree is ~chance here; continuous coherence carries the signal)
        draws = panel_b["null_draws"]
        series = [
            ("Rewired", draws["rewired_rho"], col_rew),
            ("Expr-matched", draws["expr_matched_rho"], col_null),
            ("Uniform", draws["uniform_rho"], "#B0B0B0"),
        ]
        positions = []
        for i, (lab, vals, color) in enumerate(series):
            positions.append(i)
            vp = ax.violinplot(vals, positions=[i], showmeans=False, showextrema=False, widths=0.7)
            for b in vp["bodies"]:
                b.set_facecolor(color)
                b.set_alpha(0.55)
            ax.plot([i], [np.mean(vals)], "o", color="0.2", ms=4)
        real_r = panel_b["real"]["spearman"]
        ax.axhline(real_r, color=col_real, lw=2, label=f"Real CHIP (ρ={real_r:.3f})")
        ax.axhline(0, color="0.8", ls=":", lw=1)
        ax.set_xticks(positions)
        ax.set_xticklabels([s[0] for s in series], fontsize=9)
        ax.set_ylabel("Spearman (ΔTF early→mid,\nΔtarget mid→late)")
        ax.set_title("B  Edge lag coherence", loc="left", fontsize=11, fontweight="600")
        p_rew = panel_b["vs_rewired"]["spearman"]["p"]
        p_em = panel_b["vs_expr_matched_random"]["spearman"]["p"]
        sa = panel_b["real"]["sign_agree"]
        ax.text(
            0.02,
            0.98,
            f"vs rewired p={p_rew:.3g}\nvs expr-matched p={p_em:.3g}\n"
            f"sign-agree={sa:.3f} (≈chance)",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color="0.25",
        )
        ax.legend(frameon=False, fontsize=8, loc="lower right")

    if panel_c is not None:
        ax = axes[ax_i]
        ax_i += 1
        names = ["Real GRN", "Matched\nrandom", "Non-parent", "Rewired\nparents"]
        keys = [
            ("real_acc_T", None, None),
            ("matched_mean_acc_T", "matched_q05", "matched_q95"),
            ("nonparent_mean_acc_T", "nonparent_q05", "nonparent_q95"),
            ("rewired_mean_acc_T", "rewired_q05", "rewired_q95"),
        ]
        means, lo, hi, colors, keep_names = [], [], [], [], []
        palette = [col_real, col_null, "#7AC3DF", col_rew]
        for (mean_k, lo_k, hi_k), name, color in zip(keys, names, palette):
            if mean_k not in panel_c or panel_c[mean_k] != panel_c[mean_k]:
                continue  # NaN skip
            m = float(panel_c[mean_k])
            means.append(m)
            lo.append(float(panel_c[lo_k]) if lo_k and lo_k in panel_c else m)
            hi.append(float(panel_c[hi_k]) if hi_k and hi_k in panel_c else m)
            colors.append(color)
            keep_names.append(name)
        yerr_lo = np.clip(np.array(means) - np.array(lo), 0, None)
        yerr_hi = np.clip(np.array(hi) - np.array(means), 0, None)
        x = np.arange(len(means))
        ax.bar(x, np.array(means) * 100, color=colors, width=0.65)
        ax.errorbar(
            x,
            np.array(means) * 100,
            yerr=np.vstack([yerr_lo, yerr_hi]) * 100,
            fmt="none",
            ecolor="0.25",
            capsize=3,
        )
        if "baseline_noclamp_acc_T" in panel_c:
            ax.axhline(
                panel_c["baseline_noclamp_acc_T"] * 100,
                color="0.45",
                ls=":",
                lw=1.2,
                label="no clamp",
            )
        ax.axhline(50, color="0.75", ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(keep_names, fontsize=8)
        ax.set_ylabel("Direction accuracy on held-out T (%)")
        ax.set_title("C  Held-out target prediction", loc="left", fontsize=11, fontweight="600")
        ax.text(
            0.02,
            0.98,
            f"p vs matched={panel_c.get('p_real_gt_matched', float('nan')):.3g}\n"
            f"p vs nonparent={panel_c.get('p_real_gt_nonparent', float('nan')):.3g}\n"
            f"p vs rewired={panel_c.get('p_real_gt_rewired', float('nan')):.3g}\n"
            f"|S|={panel_c['n_S']}, |T|={panel_c['n_T']}",
            transform=ax.transAxes,
            va="top",
            fontsize=7.5,
            color="0.25",
        )
        ax.legend(frameon=False, fontsize=8, loc="lower right")

    fig.suptitle(
        f"{dataset}: dynamics organized by GRN (not update-mask coverage)",
        fontsize=12,
        fontweight="700",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Fig4 dynamics↔GRN panels A/B/C")
    p.add_argument("--dataset", default="hESC")
    p.add_argument("--panels", default="AB", help="A, B, C, AB, ABC, ...")
    p.add_argument("--outdir", default="outputs/dyn_grn_panels")
    p.add_argument("--expr-root", default="/mnt/10T/yzn/benchmark_GRN/input_process")
    p.add_argument("--pt-root", default="/mnt/10T/yzn/benchmark_GRN/PseudoTime")
    p.add_argument("--top-percent", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-null-a", type=int, default=2000)
    p.add_argument("--n-null-b", type=int, default=200)
    p.add_argument("--n-null-c", type=int, default=20)
    p.add_argument("--k-reg", type=int, default=20)
    p.add_argument("--min-targets-per-tf", type=int, default=3)
    p.add_argument(
        "--hard-T",
        action="store_true",
        default=True,
        help="Restrict T to genes baseline (no clamp) gets wrong",
    )
    p.add_argument("--no-hard-T", action="store_false", dest="hard_T")
    # Panel C / scGPT
    p.add_argument("--scgpt-model-dir", default="")
    p.add_argument("--scgpt-repo-dir", default="")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--gen-iters", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--ema-alpha", type=float, default=0.1)
    p.add_argument("--max-early-cells", type=int, default=64)
    p.add_argument("--chip-network", default="")
    p.add_argument("--string-network", default="")
    p.add_argument("--grn-tsv", default="")
    return p.parse_args()


def main():
    args = parse_args()
    panels = set(args.panels.upper())
    outdir = Path(args.outdir) / args.dataset
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"[load] {args.dataset}", flush=True)
    data = load_dyn_grn_data(
        dataset=args.dataset,
        expr_root=args.expr_root,
        pt_root=args.pt_root,
        top_percent=args.top_percent,
    )
    print(
        f"  genes={len(data.genes)} dyn={data.top_dyn_mask.sum()} "
        f"CHIP edges={len(data.chip_edges)} TFs={data.is_tf.sum()} "
        f"early/mid/late={data.early.sum()}/{data.mid.sum()}/{data.late.sum()}",
        flush=True,
    )

    panel_a = panel_b = panel_c = None

    if "A" in panels:
        print("\n===== Panel A =====", flush=True)
        panel_a = run_panel_a(data, rng, n_null=args.n_null_a)
        pd.DataFrame(panel_a["enrichment"]).to_csv(outdir / "panel_a_enrichment.csv", index=False)
        with open(outdir / "panel_a_summary.json", "w") as f:
            dump = {k: v for k, v in panel_a.items() if k != "enrichment"}
            json.dump(dump, f, indent=2, default=float)
        for r in panel_a["enrichment"]:
            print(
                f"  {r['test']}: overlap={r['overlap']}/{r['n_dynamic']} "
                f"fold={r['fold_vs_background']:.3f} "
                f"matched_p_gt={r['matched_p_greater']:.4g} "
                f"matched_p_lt={r['matched_p_less']:.4g} → {r['direction']}",
                flush=True,
            )
        if panel_a["string_naive"]:
            s = panel_a["string_naive"]
            print(
                f"  [naive STRING] overlap={s['overlap']}/{s['n_dynamic']} "
                f"fold={s['fold_vs_background']:.3f} "
                f"hyp_p_less={s['hypergeom_p_less']:.4g}",
                flush=True,
            )
        sp = panel_a["degree_partial_spearman"]
        print(
            f"  |Δ|~log1p(degree) partial Spearman ρ={sp['spearman_partial']:.3f} "
            f"perm_p={sp['spearman_perm_p']:.4g}",
            flush=True,
        )
        print(f"  dyn–dyn connectivity: {panel_a['dyn_dyn_connectivity']}", flush=True)
        print(f"  TF proximity: {panel_a['tf_proximity']}", flush=True)

    if "B" in panels:
        print("\n===== Panel B =====", flush=True)
        panel_b = run_panel_b(data, rng, n_null=args.n_null_b)
        # save without huge draws in main json — keep draws separately
        draws = panel_b.pop("null_draws")
        with open(outdir / "panel_b_summary.json", "w") as f:
            json.dump(panel_b, f, indent=2, default=float)
        pd.DataFrame(draws).to_csv(outdir / "panel_b_null_draws.csv", index=False)
        panel_b["null_draws"] = draws  # restore for plot
        print(f"  real sign_agree={panel_b['real']['sign_agree']:.4f} "
              f"spearman={panel_b['real']['spearman']:.4f} n={panel_b['real']['n_edges']}", flush=True)
        print(f"  vs rewired: {panel_b['vs_rewired']['sign_agree']}", flush=True)
        print(f"  vs expr-matched: {panel_b['vs_expr_matched_random']['sign_agree']}", flush=True)

    if "C" in panels:
        print("\n===== Panel C =====", flush=True)
        if not args.scgpt_model_dir or not args.scgpt_repo_dir:
            raise SystemExit("Panel C requires --scgpt-model-dir and --scgpt-repo-dir")
        panel_c = run_panel_c(data, args, rng, outdir)
        with open(outdir / "panel_c_summary.json", "w") as f:
            json.dump(panel_c, f, indent=2, default=float)
        print(json.dumps(panel_c, indent=2), flush=True)

    # Reload siblings for compose if this run skipped them
    if panel_a is None and (outdir / "panel_a_summary.json").is_file():
        with open(outdir / "panel_a_summary.json") as f:
            panel_a = json.load(f)
        enr = pd.read_csv(outdir / "panel_a_enrichment.csv")
        panel_a["enrichment"] = enr.to_dict("records")
    if panel_b is None and (outdir / "panel_b_summary.json").is_file():
        with open(outdir / "panel_b_summary.json") as f:
            panel_b = json.load(f)
        if (outdir / "panel_b_null_draws.csv").is_file():
            panel_b["null_draws"] = pd.read_csv(outdir / "panel_b_null_draws.csv").to_dict("list")
    if panel_c is None and (outdir / "panel_c_summary.json").is_file():
        with open(outdir / "panel_c_summary.json") as f:
            panel_c = json.load(f)

    plot_figure(panel_a, panel_b, panel_c, outdir / "Fig4_dyn_grn_panels.png", args.dataset)
    meta = {
        "dataset": args.dataset,
        "panels": args.panels,
        "design_note": (
            "Panel C separates S (update) from T (held-out score). "
            "Gene-count sweep is coverage sensitivity only, not the main claim."
        ),
    }
    with open(outdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone → {outdir}", flush=True)


if __name__ == "__main__":
    main()
