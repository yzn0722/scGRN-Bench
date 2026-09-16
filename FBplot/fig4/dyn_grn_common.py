#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for Fig4 dynamics↔GRN panels (A/B/C).

Design goals
------------
- Prefer CHIP directed TF→target over STRING (PPI/functional).
- Always control expression / detection / variance / degree / TF status
  when claiming enrichment or lag coherence.
- Keep regulator set S and evaluation set T disjoint for prediction probes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def _norm(g: str) -> str:
    return str(g).strip().upper()


@dataclass
class DynGrnData:
    dataset: str
    genes: List[str]
    gene_to_idx: Dict[str, int]
    X: np.ndarray  # cells × genes (raw or as provided)
    pt: np.ndarray
    early: np.ndarray
    mid: np.ndarray
    late: np.ndarray
    early_mean: np.ndarray
    mid_mean: np.ndarray
    late_mean: np.ndarray
    true_delta: np.ndarray  # late - early
    abs_delta: np.ndarray
    mean_expr: np.ndarray
    detection_rate: np.ndarray
    expr_var: np.ndarray
    chip_edges: pd.DataFrame  # Gene1→Gene2, both in matrix
    string_edges: Optional[pd.DataFrame]
    out_degree: np.ndarray  # CHIP directed
    in_degree: np.ndarray
    undirected_degree: np.ndarray
    is_tf: np.ndarray
    is_target: np.ndarray
    is_node: np.ndarray
    top_dyn_mask: np.ndarray
    top_dyn_idx: np.ndarray


def split_pt_tertiles(
    pt: np.ndarray,
    early_q: float = 0.33,
    late_q: float = 0.67,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Early / middle / late by pseudotime quantiles (disjoint)."""
    pt = np.asarray(pt, dtype=np.float64)
    lo, hi = np.quantile(pt, [early_q, late_q])
    early = pt <= lo
    late = pt >= hi
    mid = (~early) & (~late)
    return early, mid, late


def load_edges(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep)
    cols = {c.lower(): c for c in df.columns}
    g1 = cols.get("gene1") or cols.get("tf") or df.columns[0]
    g2 = cols.get("gene2") or cols.get("target") or df.columns[1]
    out = pd.DataFrame({"Gene1": df[g1].map(_norm), "Gene2": df[g2].map(_norm)})
    out = out[(out.Gene1 != "") & (out.Gene2 != "") & (out.Gene1 != out.Gene2)]
    return out.drop_duplicates().reset_index(drop=True)


def load_dyn_grn_data(
    dataset: str = "hESC",
    expr_root: str | Path = "/mnt/10T/yzn/benchmark_GRN/input_process",
    pt_root: str | Path = "/mnt/10T/yzn/benchmark_GRN/PseudoTime",
    top_percent: float = 30.0,
    early_q: float = 0.33,
    late_q: float = 0.67,
) -> DynGrnData:
    expr_root = Path(expr_root)
    pt_root = Path(pt_root)
    expr = pd.read_csv(expr_root / "CHIP" / f"{dataset}_chip_matched-ExpressionData.csv", index_col=0)
    pt_df = pd.read_csv(pt_root / dataset / "PseudoTime.csv")
    pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"}).set_index("cell")
    common = expr.columns.intersection(pt_df.index)
    if len(common) == 0:
        raise ValueError(f"No overlapping cells for {dataset}")
    expr = expr[common]
    pt = pt_df.loc[common, "pt"].to_numpy(dtype=np.float64)

    genes = [_norm(g) for g in expr.index.astype(str)]
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    X = expr.T.to_numpy(dtype=np.float64)  # cells × genes

    early, mid, late = split_pt_tertiles(pt, early_q=early_q, late_q=late_q)
    early_mean = X[early].mean(axis=0)
    mid_mean = X[mid].mean(axis=0)
    late_mean = X[late].mean(axis=0)
    true_delta = late_mean - early_mean
    abs_delta = np.abs(true_delta)

    mean_expr = X.mean(axis=0)
    detection_rate = (X > 0).mean(axis=0)
    expr_var = X.var(axis=0)

    chip = load_edges(expr_root / "CHIP" / f"{dataset}_chip_matched-network.csv")
    chip = chip[chip.Gene1.isin(gene_to_idx) & chip.Gene2.isin(gene_to_idx)].reset_index(drop=True)

    string_path = expr_root / "STRING" / f"{dataset}_processed-network.csv"
    string_edges = None
    if string_path.is_file():
        se = load_edges(string_path)
        string_edges = se[se.Gene1.isin(gene_to_idx) & se.Gene2.isin(gene_to_idx)].reset_index(drop=True)

    n = len(genes)
    out_d = np.zeros(n, dtype=np.int32)
    in_d = np.zeros(n, dtype=np.int32)
    und_d = np.zeros(n, dtype=np.int32)
    is_tf = np.zeros(n, dtype=bool)
    is_target = np.zeros(n, dtype=bool)
    seen_und: Set[Tuple[int, int]] = set()
    for a, b in zip(chip.Gene1, chip.Gene2):
        ia, ib = gene_to_idx[a], gene_to_idx[b]
        out_d[ia] += 1
        in_d[ib] += 1
        is_tf[ia] = True
        is_target[ib] = True
        u, v = (ia, ib) if ia < ib else (ib, ia)
        if (u, v) not in seen_und:
            seen_und.add((u, v))
            und_d[ia] += 1
            und_d[ib] += 1
    is_node = is_tf | is_target

    top_n = max(int(n * top_percent / 100.0), 1)
    top_dyn_idx = np.argsort(abs_delta)[::-1][:top_n]
    top_dyn_mask = np.zeros(n, dtype=bool)
    top_dyn_mask[top_dyn_idx] = True

    return DynGrnData(
        dataset=dataset,
        genes=genes,
        gene_to_idx=gene_to_idx,
        X=X,
        pt=pt,
        early=early,
        mid=mid,
        late=late,
        early_mean=early_mean,
        mid_mean=mid_mean,
        late_mean=late_mean,
        true_delta=true_delta,
        abs_delta=abs_delta,
        mean_expr=mean_expr,
        detection_rate=detection_rate,
        expr_var=expr_var,
        chip_edges=chip,
        string_edges=string_edges,
        out_degree=out_d,
        in_degree=in_d,
        undirected_degree=und_d,
        is_tf=is_tf,
        is_target=is_target,
        is_node=is_node,
        top_dyn_mask=top_dyn_mask,
        top_dyn_idx=top_dyn_idx,
    )


def covariate_matrix(data: DynGrnData, degree: Optional[np.ndarray] = None) -> np.ndarray:
    """Columns: log1p(degree), mean_expr, detection_rate, log1p(expr_var), is_tf."""
    deg = data.undirected_degree if degree is None else degree
    return np.column_stack(
        [
            np.log1p(deg.astype(np.float64)),
            data.mean_expr,
            data.detection_rate,
            np.log1p(data.expr_var),
            data.is_tf.astype(np.float64),
        ]
    )


def _bucket_keys(cov: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Discretize continuous covariates into quantile bins for matching."""
    keys = []
    for j in range(cov.shape[1] - 1):  # last col is binary is_tf
        col = cov[:, j]
        try:
            bins = pd.qcut(col, q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            bins = np.zeros(len(col), dtype=int)
        keys.append(np.asarray(bins, dtype=int))
    keys.append(cov[:, -1].astype(int))
    # pack into a single int key
    out = np.zeros(cov.shape[0], dtype=np.int64)
    for k in keys:
        out = out * (int(k.max()) + 2) + (k + 1)
    return out


def matched_null_sets(
    observed_idx: Sequence[int],
    pool_idx: Sequence[int],
    cov: np.ndarray,
    rng: np.random.Generator,
    n_null: int = 1000,
    n_bins: int = 5,
) -> List[np.ndarray]:
    """
    Sample null gene sets of size |observed|, matching covariate buckets
    (expr / detection / variance / degree / TF) within pool.
    """
    obs = np.asarray(list(observed_idx), dtype=int)
    pool = np.asarray(list(pool_idx), dtype=int)
    k = len(obs)
    if k == 0:
        return [np.array([], dtype=int) for _ in range(n_null)]

    keys = _bucket_keys(cov, n_bins=n_bins)
    need: Dict[int, int] = defaultdict(int)
    for i in obs:
        need[int(keys[i])] += 1

    by_b: Dict[int, List[int]] = defaultdict(list)
    obs_set = set(obs.tolist())
    for i in pool:
        if i in obs_set:
            continue
        by_b[int(keys[i])].append(int(i))

    nulls: List[np.ndarray] = []
    for _ in range(n_null):
        chosen: List[int] = []
        local = {b: list(v) for b, v in by_b.items()}
        for b in local:
            rng.shuffle(local[b])
        leftover_need = 0
        for b, n in need.items():
            take = min(n, len(local.get(b, [])))
            if take:
                chosen.extend(local[b][:take])
                local[b] = local[b][take:]
            leftover_need += n - take
        if leftover_need:
            rest = [i for xs in local.values() for i in xs if i not in set(chosen)]
            # also allow pool including non-obs if still short
            if len(rest) < leftover_need:
                rest = [int(i) for i in pool if i not in set(chosen)]
            rng.shuffle(rest)
            chosen.extend(rest[:leftover_need])
        chosen = list(dict.fromkeys(chosen))[:k]
        if len(chosen) < k:
            extra = [int(i) for i in pool if i not in set(chosen)]
            rng.shuffle(extra)
            chosen.extend(extra[: k - len(chosen)])
        nulls.append(np.asarray(chosen[:k], dtype=int))
    return nulls


def emp_p(obs: float, null: Sequence[float], alternative: str = "greater") -> float:
    null_arr = np.asarray(null, dtype=float)
    b = len(null_arr)
    if b == 0:
        return float("nan")
    if alternative == "greater":
        extreme = int(np.sum(null_arr >= obs))
    elif alternative == "less":
        extreme = int(np.sum(null_arr <= obs))
    else:
        extreme = int(np.sum(np.abs(null_arr) >= abs(obs)))
    return float((1 + extreme) / (b + 1))


def enrichment_fold(n_overlap: int, n_set: int, n_bg_in: int, n_bg: int) -> float:
    if n_set == 0 or n_bg == 0:
        return float("nan")
    expected = n_set * (n_bg_in / n_bg)
    return float(n_overlap / expected) if expected > 0 else float("nan")


def hypergeom_p(n_overlap: int, n_set: int, n_bg_in: int, n_bg: int, alternative: str = "greater") -> float:
    # P(X >= k) or P(X <= k)
    M, n, N = n_bg, n_bg_in, n_set
    k = n_overlap
    if alternative == "greater":
        return float(stats.hypergeom.sf(k - 1, M, n, N))
    return float(stats.hypergeom.cdf(k, M, n, N))


def ols_partial_spearman(
    y: np.ndarray,
    x: np.ndarray,
    covariates: np.ndarray,
    rng: np.random.Generator,
    n_perm: int = 2000,
) -> dict:
    """
    Residualize y and x on covariates, report Spearman(rho) and permutation p
    (permute residualized x).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    C = np.asarray(covariates, dtype=float)
    # add intercept
    design = np.column_stack([np.ones(len(y)), C])

    def _resid(v):
        beta, *_ = np.linalg.lstsq(design, v, rcond=None)
        return v - design @ beta

    yr = _resid(y)
    xr = _resid(x)
    rho, p_para = stats.spearmanr(yr, xr)
    null = []
    for _ in range(n_perm):
        xp = rng.permutation(xr)
        r, _ = stats.spearmanr(yr, xp)
        null.append(r)
    p_perm = emp_p(float(rho), null, alternative="two-sided")
    # also simple OLS coef of x after covariates
    design_x = np.column_stack([design, x])
    beta, *_ = np.linalg.lstsq(design_x, y, rcond=None)
    return {
        "spearman_partial": float(rho),
        "spearman_parametric_p": float(p_para),
        "spearman_perm_p": float(p_perm),
        "ols_coef_x": float(beta[-1]),
        "n": int(len(y)),
    }


def rewire_directed_degree_preserving(
    edges: pd.DataFrame,
    rng: np.random.Generator,
    n_swaps: Optional[int] = None,
) -> pd.DataFrame:
    """
    Configuration-model style rewiring for directed edges, preserving
    out-degree of Gene1 and in-degree of Gene2 (approx via stub matching).
    """
    src = edges["Gene1"].tolist()
    tgt = edges["Gene2"].tolist()
    m = len(src)
    if m == 0:
        return edges.copy()
    # stub matching: shuffle targets
    new_tgt = list(tgt)
    rng.shuffle(new_tgt)
    # remove self-loops / duplicates with limited retries
    seen: Set[Tuple[str, str]] = set()
    out_src: List[str] = []
    out_tgt: List[str] = []
    for a, b in zip(src, new_tgt):
        if a == b:
            continue
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        out_src.append(a)
        out_tgt.append(b)
    # if we lost too many edges, top up with random stubs
    if len(out_src) < int(0.9 * m):
        stubs_out = list(src)
        stubs_in = list(tgt)
        rng.shuffle(stubs_out)
        rng.shuffle(stubs_in)
        for a, b in zip(stubs_out, stubs_in):
            if a == b or (a, b) in seen:
                continue
            seen.add((a, b))
            out_src.append(a)
            out_tgt.append(b)
            if len(out_src) >= m:
                break
    return pd.DataFrame({"Gene1": out_src[:m], "Gene2": out_tgt[:m]})


def edge_lag_scores(
    data: DynGrnData,
    edges: pd.DataFrame,
) -> dict:
    """
    For each TF→target edge:
      dTF = mid(TF) - early(TF)
      dTarget = late(target) - mid(target)
    Returns sign-agreement rate and Spearman across edges.
    """
    d_tf = data.mid_mean - data.early_mean
    d_tg = data.late_mean - data.mid_mean
    xs, ys = [], []
    for a, b in zip(edges["Gene1"], edges["Gene2"]):
        ia = data.gene_to_idx.get(a)
        ib = data.gene_to_idx.get(b)
        if ia is None or ib is None:
            continue
        xs.append(d_tf[ia])
        ys.append(d_tg[ib])
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) == 0:
        return {
            "n_edges": 0,
            "sign_agree": float("nan"),
            "spearman": float("nan"),
            "mean_product": float("nan"),
        }
    sign_agree = float(np.mean(np.sign(xs) == np.sign(ys)))
    # ignore zeros in sign: treat 0 as not agreeing with nonzero opposite... already ==
    rho, _ = stats.spearmanr(xs, ys)
    return {
        "n_edges": int(len(xs)),
        "sign_agree": sign_agree,
        "spearman": float(rho) if np.isfinite(rho) else float("nan"),
        "mean_product": float(np.mean(xs * ys)),
        "xs": xs,
        "ys": ys,
    }


def expression_matched_random_edges(
    data: DynGrnData,
    n_edges: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Random directed pairs matched on TF/target mean expression buckets."""
    # candidate TFs / targets
    tf_idx = np.where(data.is_tf)[0]
    tg_idx = np.where(data.is_target)[0]
    if len(tf_idx) == 0:
        tf_idx = np.arange(len(data.genes))
    if len(tg_idx) == 0:
        tg_idx = np.arange(len(data.genes))

    # bucket by mean expression
    try:
        tf_bins = pd.qcut(data.mean_expr[tf_idx], q=5, labels=False, duplicates="drop")
    except ValueError:
        tf_bins = np.zeros(len(tf_idx), dtype=int)
    try:
        tg_bins = pd.qcut(data.mean_expr[tg_idx], q=5, labels=False, duplicates="drop")
    except ValueError:
        tg_bins = np.zeros(len(tg_idx), dtype=int)

    by_tf = defaultdict(list)
    for i, b in zip(tf_idx, tf_bins):
        by_tf[int(b)].append(int(i))
    by_tg = defaultdict(list)
    for i, b in zip(tg_idx, tg_bins):
        by_tg[int(b)].append(int(i))

    src, tgt = [], []
    seen: Set[Tuple[str, str]] = set()
    tries = 0
    while len(src) < n_edges and tries < n_edges * 50:
        tries += 1
        b = int(rng.choice(list(by_tf.keys())))
        if b not in by_tg or not by_tf[b] or not by_tg[b]:
            continue
        ia = int(rng.choice(by_tf[b]))
        ib = int(rng.choice(by_tg[b]))
        if ia == ib:
            continue
        a, bb = data.genes[ia], data.genes[ib]
        if (a, bb) in seen:
            continue
        seen.add((a, bb))
        src.append(a)
        tgt.append(bb)
    return pd.DataFrame({"Gene1": src, "Gene2": tgt})


def select_regulators_and_heldout_targets(
    data: DynGrnData,
    k_reg: int = 20,
    min_targets_per_tf: int = 3,
    require_dyn_targets: bool = True,
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    S = top-K CHIP TFs by out-degree (among those with enough in-matrix targets).
    T = CHIP targets of S that are not in S (optionally restricted to top-dynamic).
    Returns (S, T, edges_S_to_T).
    """
    # out-neighbors
    outs: Dict[str, Set[str]] = defaultdict(set)
    for a, b in zip(data.chip_edges.Gene1, data.chip_edges.Gene2):
        outs[a].add(b)

    tf_scores = []
    for g, nbrs in outs.items():
        if len(nbrs) >= min_targets_per_tf:
            tf_scores.append((data.out_degree[data.gene_to_idx[g]], g))
    tf_scores.sort(reverse=True)
    S = [g for _, g in tf_scores[:k_reg]]
    S_set = set(S)

    T_set: Set[str] = set()
    edge_rows = []
    for a in S:
        for b in outs[a]:
            if b in S_set:
                continue
            if require_dyn_targets and not data.top_dyn_mask[data.gene_to_idx[b]]:
                continue
            T_set.add(b)
            edge_rows.append((a, b))
    T = sorted(T_set)
    edges = pd.DataFrame(edge_rows, columns=["Gene1", "Gene2"])
    return S, T, edges


def direction_accuracy_on_idx(
    pred_delta: np.ndarray,
    true_delta: np.ndarray,
    idx: Sequence[int],
) -> float:
    idx = np.asarray(list(idx), dtype=int)
    if len(idx) == 0:
        return float("nan")
    td = true_delta[idx]
    pd = pred_delta[idx]
    true_dir = np.where(td > 0, 1, -1)
    pred_dir = np.where(pd > 0, 1, -1)
    return float((pred_dir == true_dir).mean())
