#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python reimplementation of PIDC (Chan et al., Cell Systems 2017)
matching NetworkInference.jl / BEELINE:

  1) Discretize expression (uniform_width by default)
  2) Pairwise MI + specific information
  3) PUC scores over gene triples (proportional unique contribution)
  4) Context weighting via Gamma CDF (PIDC) → undirected edge weights

Reference:
  https://github.com/Tchanders/NetworkInference.jl
  BEELINE Algorithms/PIDC/runPIDC.jl
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import gamma as gamma_dist


def _safe_log(x: np.ndarray, base: float) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    mask = x > 0
    out[:] = 0.0
    out[mask] = np.log(x[mask]) / np.log(base)
    return out


def discretize_uniform_width(values: np.ndarray, n_bins: int) -> Tuple[np.ndarray, int]:
    """Map continuous values to bin ids in 0..n_bins-1 (Julia uses 1..n; we use 0-based)."""
    v = np.asarray(values, dtype=np.float64)
    vmin, vmax = float(v.min()), float(v.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return np.zeros(v.shape[0], dtype=np.int32), 1
    # equal-width edges
    edges = np.linspace(vmin, vmax, n_bins + 1)
    # digitize: rightmost edge inclusive
    ids = np.digitize(v, edges[1:-1], right=False).astype(np.int32)
    ids = np.clip(ids, 0, n_bins - 1)
    return ids, int(n_bins)


def discretize_uniform_count(values: np.ndarray, n_bins: int) -> Tuple[np.ndarray, int]:
    """Equal-frequency bins; fall back to uniform_width on failure."""
    v = np.asarray(values, dtype=np.float64)
    if np.unique(v).size < 2:
        return np.zeros(v.shape[0], dtype=np.int32), 1
    try:
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(v, qs[1:-1]))
        if edges.size == 0:
            return discretize_uniform_width(v, n_bins)
        ids = np.digitize(v, edges, right=False).astype(np.int32)
        ids = np.clip(ids, 0, n_bins - 1)
        return ids, int(n_bins)
    except Exception:
        return discretize_uniform_width(v, n_bins)


def joint_counts_2d(a: np.ndarray, b: np.ndarray, na: int, nb: int) -> np.ndarray:
    """(na, nb) contingency from 0-based bin ids."""
    return np.bincount(a * nb + b, minlength=na * nb).reshape(na, nb).astype(np.float64)


def ml_probs(freq: np.ndarray) -> np.ndarray:
    total = freq.sum()
    if total <= 0:
        return np.zeros_like(freq, dtype=np.float64)
    return freq / total


def mutual_information(p_xy: np.ndarray, base: float = 2.0) -> float:
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(p_xy > 0, p_xy / (p_x * p_y), 1.0)
        term = np.where(p_xy > 0, p_xy * _safe_log(ratio, base), 0.0)
    return float(np.sum(term))


def specific_information(
    p_xz: np.ndarray,
    p_x: np.ndarray,
    p_z: np.ndarray,
    sum_over_source: bool,
    base: float = 2.0,
) -> np.ndarray:
    """
    SI(z) = sum_x p(x|z) log( p(x,z) / (p(x)p(z)) ).
    If sum_over_source: p_xz shaped (n_source, n_target), sum axis 0 → length n_target.
    Else: p_xz shaped (n_target, n_source) with sum axis 1 → length n_target
          (matches Julia dim_sum=2 path after argument swap).
    """
    # Broadcast carefully
    if sum_over_source:
        # p_x: (ns,1), p_z: (1,nt)
        px = p_x.reshape(-1, 1)
        pz = p_z.reshape(1, -1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(p_xz > 0, p_xz / (px * pz), 1.0)
            cond = np.where(pz > 0, p_xz / pz, 0.0)
            term = np.where(p_xz > 0, cond * _safe_log(ratio, base), 0.0)
        si = np.sum(term, axis=0)
    else:
        # Julia: dims=2 on (n1,n2) with p_x=p2, p_z=p1 → SI about dim1
        px = p_x.reshape(1, -1)
        pz = p_z.reshape(-1, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(p_xz > 0, p_xz / (px * pz), 1.0)
            cond = np.where(pz > 0, p_xz / pz, 0.0)
            term = np.where(p_xz > 0, cond * _safe_log(ratio, base), 0.0)
        si = np.sum(term, axis=1)
    return np.nan_to_num(si, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)


@njit(cache=True)
def _accumulate_puc(
    mi: np.ndarray,
    si_flat: np.ndarray,
    si_offsets: np.ndarray,
    si_lengths: np.ndarray,
    target_probs: np.ndarray,
    tp_offsets: np.ndarray,
    tp_lengths: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Triple loop PUC accumulation (NetworkInference.jl get_puc_scores).
    si for pair (i,j) stored at si_offsets[i,j] with length = bins of target j.
    """
    puc = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # target k, sources i,j
                red = 0.0
                lk = tp_lengths[k]
                ok = tp_offsets[k]
                oi_k = si_offsets[i, k]
                oj_k = si_offsets[j, k]
                for t in range(lk):
                    s1 = si_flat[oi_k + t]
                    s2 = si_flat[oj_k + t]
                    m = s1 if s1 < s2 else s2
                    red += target_probs[ok + t] * m
                mi_ik = mi[i, k]
                mi_jk = mi[j, k]
                if mi_ik > 0.0:
                    score = (mi_ik - red) / mi_ik
                    if score > 0.0 and np.isfinite(score):
                        puc[i, k] += score
                        puc[k, i] += score
                if mi_jk > 0.0:
                    score = (mi_jk - red) / mi_jk
                    if score > 0.0 and np.isfinite(score):
                        puc[j, k] += score
                        puc[k, j] += score

                # target j, sources i,k
                red = 0.0
                lj = tp_lengths[j]
                oj = tp_offsets[j]
                oi_j = si_offsets[i, j]
                ok_j = si_offsets[k, j]
                for t in range(lj):
                    s1 = si_flat[oi_j + t]
                    s2 = si_flat[ok_j + t]
                    m = s1 if s1 < s2 else s2
                    red += target_probs[oj + t] * m
                mi_ij = mi[i, j]
                mi_kj = mi[k, j]
                if mi_ij > 0.0:
                    score = (mi_ij - red) / mi_ij
                    if score > 0.0 and np.isfinite(score):
                        puc[i, j] += score
                        puc[j, i] += score
                if mi_kj > 0.0:
                    score = (mi_kj - red) / mi_kj
                    if score > 0.0 and np.isfinite(score):
                        puc[k, j] += score
                        puc[j, k] += score

                # target i, sources j,k
                red = 0.0
                li = tp_lengths[i]
                oi = tp_offsets[i]
                oj_i = si_offsets[j, i]
                ok_i = si_offsets[k, i]
                for t in range(li):
                    s1 = si_flat[oj_i + t]
                    s2 = si_flat[ok_i + t]
                    m = s1 if s1 < s2 else s2
                    red += target_probs[oi + t] * m
                mi_ji = mi[j, i]
                mi_ki = mi[k, i]
                if mi_ji > 0.0:
                    score = (mi_ji - red) / mi_ji
                    if score > 0.0 and np.isfinite(score):
                        puc[j, i] += score
                        puc[i, j] += score
                if mi_ki > 0.0:
                    score = (mi_ki - red) / mi_ki
                    if score > 0.0 and np.isfinite(score):
                        puc[k, i] += score
                        puc[i, k] += score
    return puc


def apply_pidc_context(scores: np.ndarray, genes: Sequence[str]) -> np.ndarray:
    """
    Gamma-CDF context as in NetworkInference.jl PIDCNetworkInference.

    Fits one Gamma per gene on that gene's score profile (cached), then
    weight(i,j) = CDF_i(score) + CDF_j(score). Equivalent to refitting
    inside the nested loop, but far faster.
    """
    n = scores.shape[0]
    weights = np.zeros_like(scores, dtype=np.float64)

    # Cache per-gene Gamma params (shape, loc, scale) or None → CLR fallback stats
    gamma_params = []
    clr_stats = []  # (mean, var) of each gene's score profile
    for i in range(n):
        others = np.concatenate([scores[:i, i], scores[i + 1 :, i]])
        others = others[np.isfinite(others)]
        mu = float(np.mean(others)) if others.size else 0.0
        var = float(np.var(others)) if others.size else 0.0
        clr_stats.append((mu, var))
        pos = others[others > 0]
        params = None
        if pos.size >= 2:
            try:
                shape, loc, scale = gamma_dist.fit(pos, floc=0)
                if np.isfinite(shape) and np.isfinite(scale) and scale > 0:
                    params = (float(shape), float(loc), float(scale))
            except Exception:
                params = None
        gamma_params.append(params)

    for i in range(n):
        for j in range(i + 1, n):
            score = float(scores[i, j])
            pi, pj = gamma_params[i], gamma_params[j]
            if pi is not None and pj is not None:
                w = float(
                    gamma_dist.cdf(score, pi[0], loc=pi[1], scale=pi[2])
                    + gamma_dist.cdf(score, pj[0], loc=pj[1], scale=pj[2])
                )
            else:
                def clr_one(sc, mu, var):
                    diff = sc - mu
                    if var == 0 or diff < 0:
                        return 0.0
                    return (diff * diff) / var

                mu_i, var_i = clr_stats[i]
                mu_j, var_j = clr_stats[j]
                w = float(np.sqrt(clr_one(score, mu_i, var_i) + clr_one(score, mu_j, var_j)))
            weights[i, j] = w
            weights[j, i] = w
    return weights


def infer_pidc(
    expr: pd.DataFrame,
    *,
    n_bins: int = 10,
    discretizer: str = "uniform_width",
    base: float = 2.0,
    apply_context: bool = True,
    gene_subset: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Run PIDC on genes × cells expression DataFrame.

    Returns undirected edges in BEELINE format (both directions), columns:
    Gene1, Gene2, EdgeWeight.
    """
    if gene_subset is not None:
        keep = [g for g in gene_subset if g in expr.index]
        expr = expr.loc[keep]
    genes = [str(g) for g in expr.index]
    n = len(genes)
    if n < 3:
        raise ValueError(f"PIDC needs ≥3 genes, got {n}")

    x = expr.to_numpy(dtype=np.float64)
    bins = np.zeros((n, x.shape[1]), dtype=np.int32)
    nbin = np.zeros(n, dtype=np.int32)
    probs: List[np.ndarray] = []

    disc = discretize_uniform_width if discretizer == "uniform_width" else discretize_uniform_count
    for i in range(n):
        ids, nb = disc(x[i], n_bins)
        bins[i] = ids
        nbin[i] = nb
        freq = np.bincount(ids, minlength=nb).astype(np.float64)
        probs.append(ml_probs(freq))

    # Pairwise MI + SI
    mi = np.zeros((n, n), dtype=np.float64)
    # Flatten SI storage: for each (i,j), SI that i provides about states of j
    si_lengths = nbin.copy()  # length of SI vector for target j is nbin[j]
    # offset matrix
    # Total size: sum_j n * nbin[j]
    total_si = int(n * np.sum(nbin))
    si_flat = np.zeros(total_si, dtype=np.float64)
    si_offsets = np.zeros((n, n), dtype=np.int64)

    cursor = 0
    # We'll fill offsets as we go: for fixed target j, sources i occupy contiguous? 
    # Simpler: offset[i,j] = cursor assigned when computing pair.
    # Pre-assign: for each (i,j) i!=j, need space nbin[j]
    for i in range(n):
        for j in range(n):
            if i == j:
                si_offsets[i, j] = 0
                continue
            si_offsets[i, j] = cursor
            cursor += int(nbin[j])
    if cursor > si_flat.size:
        si_flat = np.zeros(cursor, dtype=np.float64)

    print(f"[PIDC] genes={n} cells={x.shape[1]} bins={n_bins} discretizer={discretizer}", flush=True)
    print("[PIDC] pairwise MI/SI ...", flush=True)
    for i in range(n):
        for j in range(i + 1, n):
            na, nb = int(nbin[i]), int(nbin[j])
            cnt = joint_counts_2d(bins[i], bins[j], na, nb)
            p_xy = ml_probs(cnt)
            p_x = p_xy.sum(axis=1, keepdims=True)
            p_y = p_xy.sum(axis=0, keepdims=True)
            m = mutual_information(p_xy, base=base)
            mi[i, j] = m
            mi[j, i] = m
            # SI that i provides about j
            si_ij = specific_information(p_xy, p_x, p_y, sum_over_source=True, base=base)
            # SI that j provides about i
            si_ji = specific_information(p_xy, p_y, p_x, sum_over_source=False, base=base)
            off_ij = int(si_offsets[i, j])
            off_ji = int(si_offsets[j, i])
            si_flat[off_ij : off_ij + nb] = si_ij[:nb]
            si_flat[off_ji : off_ji + na] = si_ji[:na]
        if (i + 1) % 50 == 0 or i + 1 == n:
            print(f"  pairwise {i + 1}/{n}", flush=True)

    # Target probability flat storage
    tp_lengths = nbin.astype(np.int64)
    tp_offsets = np.zeros(n, dtype=np.int64)
    c = 0
    for i in range(n):
        tp_offsets[i] = c
        c += int(nbin[i])
    target_probs = np.zeros(c, dtype=np.float64)
    for i in range(n):
        o = int(tp_offsets[i])
        target_probs[o : o + int(nbin[i])] = probs[i]

    print("[PIDC] PUC triple accumulation (numba) ...", flush=True)
    puc = _accumulate_puc(
        mi,
        si_flat,
        si_offsets,
        nbin.astype(np.int64),
        target_probs,
        tp_offsets,
        tp_lengths,
        n,
    )

    if apply_context:
        print("[PIDC] Gamma context weighting ...", flush=True)
        weights = apply_pidc_context(puc, genes)
    else:
        weights = puc

    # Undirected export both directions (BEELINE write_network_file)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            w = float(weights[i, j])
            if not np.isfinite(w):
                w = 0.0
            rows.append((genes[i], genes[j], w))
            rows.append((genes[j], genes[i], w))
    out = pd.DataFrame(rows, columns=["Gene1", "Gene2", "EdgeWeight"])
    out = out.sort_values("EdgeWeight", ascending=False).reset_index(drop=True)
    return out


def load_expression_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.map(lambda x: str(x).strip().upper())
    df = df[~df.index.duplicated(keep="first")]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df
