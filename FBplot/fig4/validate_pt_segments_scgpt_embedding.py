#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤 1 补充：用 scGPT 细胞嵌入检验伪时间「中间段」是否有独立生物学状态。

思路
----
1. 与 fig4 一致：表达 bin 到 [0,50]、全基因序列 + <cls>（同 run_unified scGPT）。
2. 对每细胞提取 CLS embedding（单次前向，非迭代预测）。
3. 按 PT 等频分 K 段（默认 5），在嵌入空间上：
   - UMAP/PCA 着色：伪时间段 / 连续 PT
   - 段质心两两距离、Silhouette(段标签)
   - 「中间段」S1–S2 vs 首尾 S0+S4 的分离度（质心距、PERMANOVA 置换检验）

若中间段嵌入与首尾明显分离 → 支持「中间态」不仅是表达均值上的尖峰。

依赖：torch, scGPT（与 benchmark 相同环境）；可选 umap-learn。

示例（需 GPU + 含 torch 的 Python）：
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig4
  python validate_pt_segments_scgpt_embedding.py --dataset hESC
  python validate_pt_segments_scgpt_embedding.py --dataset hESC --n-segments 5 --no-umap
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH = Path("/mnt/10T/yzn/benchmark_GRN")
SCGPT_REPO = BENCH / "pre_scgpt" / "scGPT"
SCGPT_MODEL = BENCH / "pre_scgpt" / "scGPT" / "scgpt_human"
CHIP_DIR = BENCH / "input_process" / "CHIP"
PT_ROOT = BENCH / "PseudoTime"
DEFAULT_OUT = SCRIPT_DIR / "error_biology" / "multistep_pt"

try:
    from fig4_palette import apply_fig4_style, model_color
except ImportError:

    def apply_fig4_style() -> None:
        pass

    def model_color(_: str, d: str = "#333") -> str:
        return d


def mask_segment_bins(pt: np.ndarray, n_segments: int) -> List[np.ndarray]:
    edges = np.quantile(pt, np.linspace(0, 1, n_segments + 1))
    edges[-1] += 1e-9
    masks = []
    for i in range(n_segments):
        if i < n_segments - 1:
            masks.append((pt >= edges[i]) & (pt < edges[i + 1]))
        else:
            masks.append((pt >= edges[i]) & (pt <= edges[i + 1]))
    return masks


def bin_expr_to_0_50(x: np.ndarray, do_log1p: bool = False) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0)
    x = np.clip(x, 0, None)
    if do_log1p:
        x = np.log1p(x)
    vmax = max(np.percentile(x, 99.5), 1e-6)
    return np.clip(x / vmax * 50.0, 0, 50).astype(np.float32)


def load_expr_pt(dataset: str, legacy_pt: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    expr = pd.read_csv(CHIP_DIR / f"{dataset}_chip_matched-ExpressionData.csv", index_col=0)
    pt_df = pd.read_csv(PT_ROOT / dataset / "PseudoTime.csv")
    if legacy_pt:
        pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"}).set_index("cell")
    else:
        pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"}).set_index("cell")
    common = expr.columns.intersection(pt_df.index)
    expr = expr[common]
    cells = list(common)
    pt = pt_df.loc[common, "pt"].astype(float).to_numpy()
    genes = expr.index.astype(str).tolist()
    X = expr.T.to_numpy(dtype=np.float32)
    return X, pt, genes, cells


def load_scgpt(device):
    if str(SCGPT_REPO) not in sys.path:
        sys.path.insert(0, str(SCGPT_REPO))
    import torch
    from scgpt.model import TransformerModel
    from scgpt.tokenizer.gene_tokenizer import GeneVocab

    with open(SCGPT_MODEL / "args.json") as f:
        cfg = json.load(f)
    vocab = GeneVocab.from_file(SCGPT_MODEL / "vocab.json")
    for t in ["<pad>", "<cls>", "<eoc>"]:
        if t not in vocab:
            vocab.append_token(t)
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=cfg["embsize"],
        nhead=cfg["nheads"],
        d_hid=cfg["d_hid"],
        nlayers=cfg["nlayers"],
        vocab=vocab,
        pad_value=cfg["pad_value"],
        n_input_bins=cfg.get("n_bins", 51),
        use_fast_transformer=cfg.get("fast_transformer", True),
    )
    ckpt = torch.load(SCGPT_MODEL / "best_model.pt", map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device).eval()
    if device.type == "cuda":
        model.half()
    return model, vocab, torch


def extract_cell_embeddings(
    model,
    vocab,
    torch,
    X_bin: np.ndarray,
    genes: List[str],
    batch_size: int = 32,
) -> np.ndarray:
    """CLS token embedding, shape [n_cells, d_model]."""
    device = next(model.parameters()).device
    gene_ids = np.array([vocab[g] if g in vocab else vocab["<pad>"] for g in genes], dtype=np.int64)
    gene_ids = np.concatenate([[vocab["<cls>"]], gene_ids])
    gene_ids_t = torch.tensor(gene_ids[None, :], dtype=torch.long)
    values = np.concatenate([np.zeros((X_bin.shape[0], 1), dtype=np.float32), X_bin], axis=1)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    values_t = torch.tensor(values, dtype=dtype)
    pad_mask = gene_ids_t.eq(vocab["<pad>"]).expand(values.shape[0], -1)

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


def pairwise_centroid_distances(emb: np.ndarray, seg_ids: np.ndarray, n_seg: int) -> pd.DataFrame:
    centroids = {}
    for s in range(n_seg):
        m = seg_ids == s
        if m.sum() == 0:
            continue
        centroids[s] = emb[m].mean(axis=0)
    rows = []
    for i in centroids:
        for j in centroids:
            if i >= j:
                continue
            d = float(np.linalg.norm(centroids[i] - centroids[j]))
            rows.append({"seg_i": i, "seg_j": j, "centroid_l2": d})
    return pd.DataFrame(rows)


def permanova_one_way(
    emb: np.ndarray, labels: np.ndarray, n_perm: int = 999, seed: int = 0
) -> Dict[str, float]:
    """简单置换版 PERMANOVA：组间/组内平方距离比 F。"""
    rng = np.random.default_rng(seed)
    n = emb.shape[0]
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return {"F": np.nan, "p_perm": np.nan, "R2": np.nan}

    def ss_within_between(z: np.ndarray, lab: np.ndarray) -> Tuple[float, float]:
        d = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=2)
        grand = z.mean(axis=0)
        ss_tot = np.sum((z - grand) ** 2)
        ss_within = 0.0
        ss_between = 0.0
        for g in np.unique(lab):
            idx = lab == g
            ng = idx.sum()
            if ng == 0:
                continue
            cg = z[idx].mean(axis=0)
            ss_within += np.sum((z[idx] - cg) ** 2)
            ss_between += ng * np.sum((cg - grand) ** 2)
        return ss_between, ss_within

    sb, sw = ss_within_between(emb, labels)
    F_obs = (sb / max(len(uniq) - 1, 1)) / (sw / max(n - len(uniq), 1))
    R2 = sb / max(sb + sw, 1e-9)
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(labels)
        sb_p, sw_p = ss_within_between(emb, perm)
        F_p = (sb_p / max(len(uniq) - 1, 1)) / (sw_p / max(n - len(uniq), 1))
        if F_p >= F_obs:
            cnt += 1
    p = (cnt + 1) / (n_perm + 1)
    return {"F": float(F_obs), "p_perm": float(p), "R2": float(R2)}


def reduce_2d(emb: np.ndarray, method: str = "umap") -> np.ndarray:
    if method == "umap":
        try:
            import umap

            return umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(emb)
        except ImportError:
            method = "pca"
    from sklearn.decomposition import PCA

    return PCA(n_components=2, random_state=0).fit_transform(emb)


def plot_embedding_panel(
    xy: np.ndarray,
    pt: np.ndarray,
    seg_ids: np.ndarray,
    n_seg: int,
    dataset: str,
    outpath: Path,
) -> None:
    import matplotlib.pyplot as plt

    apply_fig4_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sc = axes[0].scatter(xy[:, 0], xy[:, 1], c=pt, cmap="viridis", s=8, alpha=0.75, linewidths=0)
    plt.colorbar(sc, ax=axes[0], label="Pseudotime")
    axes[0].set_title(f"{dataset} — scGPT CLS embedding (by PT)")
    axes[0].set_xlabel("dim 1")
    axes[0].set_ylabel("dim 2")

    cmap = plt.cm.get_cmap("tab10", n_seg)
    for s in range(n_seg):
        m = seg_ids == s
        axes[1].scatter(
            xy[m, 0], xy[m, 1], c=[cmap(s)], s=10, alpha=0.7, label=f"S{s}", linewidths=0
        )
    axes[1].legend(fontsize=8, markerscale=2, frameon=False)
    axes[1].set_title(f"Equal-frequency PT segments (K={n_seg})")
    axes[1].set_xlabel("dim 1")
    axes[1].set_ylabel("dim 2")
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(dataset: str, n_segments: int, outdir: Path, batch_size: int, n_perm: int, use_umap: bool) -> None:
    try:
        import torch
    except ImportError as e:
        raise SystemExit(
            "需要安装 PyTorch 及 scGPT 环境（与 run_unified_multidataset_pseudotime.py 相同）。\n"
            f"原始错误: {e}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{dataset}] device={device}")

    X, pt, genes, cells = load_expr_pt(dataset)
    X_bin = bin_expr_to_0_50(X, do_log1p=False)
    model, vocab, _ = load_scgpt(device)
    emb = extract_cell_embeddings(model, vocab, torch, X_bin, genes, batch_size=batch_size)

    masks = mask_segment_bins(pt, n_segments)
    seg_ids = np.zeros(len(pt), dtype=np.int64)
    for i, m in enumerate(masks):
        seg_ids[m] = i

    ds_dir = outdir / dataset / "embedding_validation"
    ds_dir.mkdir(parents=True, exist_ok=True)
    np.save(ds_dir / "cell_embeddings.npy", emb)
    meta = pd.DataFrame({"cell": cells, "pt": pt, "segment": seg_ids})
    meta.to_csv(ds_dir / "cell_embedding_meta.csv", index=False)

    from sklearn.metrics import silhouette_score

    sil = silhouette_score(emb, seg_ids) if len(np.unique(seg_ids)) > 1 else np.nan
    dist_df = pairwise_centroid_distances(emb, seg_ids, n_segments)
    dist_df.to_csv(ds_dir / "segment_centroid_distances.csv", index=False)

    # 中间段 vs 首尾
    mid_mask = (seg_ids > 0) & (seg_ids < n_segments - 1)
    end_mask = (seg_ids == 0) | (seg_ids == n_segments - 1)
    labels_3 = np.where(mid_mask, 1, np.where(end_mask, 0, -1))
    valid = labels_3 >= 0
    perm_mid = permanova_one_way(emb[valid], labels_3[valid], n_perm=n_perm)
    perm_seg = permanova_one_way(emb, seg_ids, n_perm=n_perm)

    # 中间质心 vs 首尾质心距离比
    c_mid = emb[mid_mask].mean(axis=0) if mid_mask.any() else np.full(emb.shape[1], np.nan)
    c_early = emb[seg_ids == 0].mean(axis=0)
    c_late = emb[seg_ids == n_segments - 1].mean(axis=0)
    d_mid_early = float(np.linalg.norm(c_mid - c_early))
    d_mid_late = float(np.linalg.norm(c_mid - c_late))
    d_early_late = float(np.linalg.norm(c_early - c_late))

    summary = {
        "dataset": dataset,
        "n_cells": int(len(cells)),
        "n_segments": n_segments,
        "embedding_dim": int(emb.shape[1]),
        "silhouette_segment_labels": float(sil),
        "permanova_all_segments": perm_seg,
        "permanova_middle_vs_endpoints": perm_mid,
        "centroid_l2_mid_to_early": d_mid_early,
        "centroid_l2_mid_to_late": d_mid_late,
        "centroid_l2_early_to_late": d_early_late,
        "interpretation_hints": [
            "silhouette > 0.1 且 permanova p<0.05：段标签在嵌入空间有一定结构",
            "d(mid,early)+d(mid,late) 明显大于 d(early,late) 或 mid 与两端都远：中间态可能独立",
            "若嵌入沿 PT 单调、中间无分离：表达峰值可能为过渡态/噪声，需结合基因案例",
        ],
    }
    with open(ds_dir / "embedding_validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    method = "umap" if use_umap else "pca"
    xy = reduce_2d(emb, method=method)
    meta["umap1"] = xy[:, 0]
    meta["umap2"] = xy[:, 1]
    meta.to_csv(ds_dir / "cell_embedding_meta.csv", index=False)
    plot_embedding_panel(xy, pt, seg_ids, n_segments, dataset, ds_dir / f"scgpt_embedding_{method}.png")
    print(f"Saved: {ds_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Validate PT middle segments via scGPT cell embeddings")
    p.add_argument("--dataset", default="hESC")
    p.add_argument("--n-segments", type=int, default=5)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-perm", type=int, default=499)
    p.add_argument("--no-umap", action="store_true", help="Use PCA only (no umap-learn)")
    args = p.parse_args()
    run(
        args.dataset,
        args.n_segments,
        args.outdir,
        args.batch_size,
        args.n_perm,
        use_umap=not args.no_umap,
    )


if __name__ == "__main__":
    main()
