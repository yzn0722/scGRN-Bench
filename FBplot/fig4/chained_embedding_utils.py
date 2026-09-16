#!/usr/bin/env python3
"""Shared scGPT CLS embedding extraction for chained / validation scripts."""
from __future__ import annotations

from typing import List

import numpy as np


def extract_cell_embeddings(
    model,
    vocab,
    torch,
    values: np.ndarray,
    gene_ids: np.ndarray,
    batch_size: int = 32,
) -> np.ndarray:
    """
    CLS token embedding for each row in ``values`` (n_cells, 1+n_genes), L2-normalized.
    """
    device = next(model.parameters()).device
    gene_ids_t = torch.tensor(gene_ids[None, :], dtype=torch.long)
    pad_mask = gene_ids_t.eq(vocab["<pad>"]).expand(values.shape[0], -1)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    values_t = torch.tensor(values, dtype=dtype)

    with torch.no_grad():
        emb = model.encode_batch(
            gene_ids_t.expand(values.shape[0], -1),
            values_t,
            src_key_padding_mask=pad_mask,
            batch_size=batch_size,
            time_step=0,
            return_np=True,
        )
    emb = emb.astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return emb / norms


def segment_centroids(emb: np.ndarray, seg_ids: np.ndarray, n_seg: int) -> np.ndarray:
    cents = np.full((n_seg, emb.shape[1]), np.nan, dtype=np.float32)
    for s in range(n_seg):
        m = seg_ids == s
        if m.sum() > 0:
            cents[s] = emb[m].mean(axis=0)
    return cents


def pairwise_steps(centroids: np.ndarray) -> List[float]:
    steps = []
    for i in range(len(centroids) - 1):
        if np.all(np.isfinite(centroids[i])) and np.all(np.isfinite(centroids[i + 1])):
            steps.append(float(np.linalg.norm(centroids[i + 1] - centroids[i])))
        else:
            steps.append(float("nan"))
    return steps


def path_length(centroids: np.ndarray) -> float:
    return float(sum(pairwise_steps(centroids)))
