#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K={3,5,7} embedding 敏感性：复用已缓存 scGPT CLS embedding，仅重算分段指标。

输出
----
  error_biology/embedding_k_sensitivity/{dataset}_K{K}/embedding_validation_summary.json
  error_biology/embedding_k_sensitivity/embedding_k_sensitivity_summary.csv
  error_biology/embedding_k_sensitivity/embedding_k_sensitivity.png

示例
----
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig4

  # hESC 有缓存时（秒级）
  python3 sweep_embedding_k_sensitivity.py --datasets hESC --k-list 3,5,7

  # mHSC-L 需首次提 embedding（CPU/GPU，~数分钟）
  /mnt/10T/yzn/anconda3/envs/singlecell/bin/python3 sweep_embedding_k_sensitivity.py \\
    --datasets mHSC-L --k-list 3,5,7 --extract-if-missing

  # 一次跑两个数据集
  /mnt/10T/yzn/anconda3/envs/singlecell/bin/python3 sweep_embedding_k_sensitivity.py \\
    --datasets hESC,mHSC-L --k-list 3,5,7 --extract-if-missing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = SCRIPT_DIR / "error_biology" / "embedding_k_sensitivity"
MULTISTEP_ROOT = SCRIPT_DIR / "error_biology" / "multistep_pt"
BENCH = Path("/mnt/10T/yzn/benchmark_GRN")
CHIP_DIR = BENCH / "input_process" / "CHIP"
PT_ROOT = BENCH / "PseudoTime"
K_DEFAULT = [3, 5, 7]


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


def permanova_one_way(emb: np.ndarray, labels: np.ndarray, n_perm: int = 99, seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = emb.shape[0]
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return {"F": float("nan"), "p_perm": float("nan"), "R2": float("nan")}

    def ss_between_within(z: np.ndarray, lab: np.ndarray) -> tuple[float, float]:
        grand = z.mean(axis=0)
        ss_within = 0.0
        ss_between = 0.0
        for g in np.unique(lab):
            idx = lab == g
            ng = int(idx.sum())
            if ng == 0:
                continue
            cg = z[idx].mean(axis=0)
            ss_within += np.sum((z[idx] - cg) ** 2)
            ss_between += ng * float(np.sum((cg - grand) ** 2))
        return ss_between, ss_within

    sb, sw = ss_between_within(emb, labels)
    F_obs = (sb / max(len(uniq) - 1, 1)) / (sw / max(n - len(uniq), 1))
    R2 = sb / max(sb + sw, 1e-9)
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(labels)
        sb_p, sw_p = ss_between_within(emb, perm)
        F_p = (sb_p / max(len(uniq) - 1, 1)) / (sw_p / max(n - len(uniq), 1))
        if F_p >= F_obs:
            cnt += 1
    return {"F": float(F_obs), "p_perm": float((cnt + 1) / (n_perm + 1)), "R2": float(R2)}


def pairwise_centroid_distances(emb: np.ndarray, seg_ids: np.ndarray, n_seg: int) -> pd.DataFrame:
    centroids = {s: emb[seg_ids == s].mean(axis=0) for s in range(n_seg) if (seg_ids == s).any()}
    rows = []
    for i in centroids:
        for j in centroids:
            if i < j:
                rows.append({"seg_i": i, "seg_j": j, "centroid_l2": float(np.linalg.norm(centroids[i] - centroids[j]))})
    return pd.DataFrame(rows)

try:
    from fig4_palette import apply_fig4_style, model_color
except ImportError:

    def apply_fig4_style() -> None:
        plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    def model_color(name: str, d: str = "#666") -> str:
        return d


def load_or_extract_embeddings(
    dataset: str,
    extract_if_missing: bool,
    batch_size: int,
    legacy_root: Path,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (emb, pt, cells). Prefer legacy K=5 cache, else extract."""
    legacy_dir = legacy_root / dataset / "embedding_validation"
    emb_path = legacy_dir / "cell_embeddings.npy"
    meta_path = legacy_dir / "cell_embedding_meta.csv"

    if emb_path.is_file() and meta_path.is_file():
        emb = np.load(emb_path)
        meta = pd.read_csv(meta_path)
        return emb, meta["pt"].to_numpy(dtype=float), meta["cell"].astype(str).tolist()

    if not extract_if_missing:
        raise FileNotFoundError(
            f"No cached embeddings for {dataset} at {legacy_dir}. "
            "Re-run with --extract-if-missing (needs torch + scGPT)."
        )

    sys.path.insert(0, str(SCRIPT_DIR))
    from validate_pt_segments_scgpt_embedding import (  # noqa: E402
        SCGPT_MODEL,
        SCGPT_REPO,
        bin_expr_to_0_50,
        extract_cell_embeddings,
        load_expr_pt,
    )

    try:
        import json
        import torch
        from scgpt.model import TransformerModel
        from scgpt.tokenizer.gene_tokenizer import GeneVocab
    except ImportError as e:
        raise SystemExit(f"torch + scGPT required for extraction: {e}")

    if str(SCGPT_REPO) not in sys.path:
        sys.path.insert(0, str(SCGPT_REPO))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{dataset}] extracting embeddings on {device} ...")
    X, pt, genes, cells = load_expr_pt(dataset)
    X_bin = bin_expr_to_0_50(X, do_log1p=False)

    with open(SCGPT_MODEL / "args.json") as f:
        cfg = json.load(f)
    vocab = GeneVocab.from_file(SCGPT_MODEL / "vocab.json")
    for t in ["<pad>", "<cls>", "<eoc>"]:
        if t not in vocab:
            vocab.append_token(t)
    use_fast = cfg.get("fast_transformer", True) and device.type == "cuda"
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=cfg["embsize"],
        nhead=cfg["nheads"],
        d_hid=cfg["d_hid"],
        nlayers=cfg["nlayers"],
        vocab=vocab,
        pad_value=cfg["pad_value"],
        n_input_bins=cfg.get("n_bins", 51),
        use_fast_transformer=use_fast,
    )
    ckpt = torch.load(SCGPT_MODEL / "best_model.pt", map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device).eval()
    if device.type == "cuda":
        model.half()
    emb = extract_cell_embeddings(model, vocab, torch, X_bin, genes, batch_size=batch_size)

    legacy_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, emb)
    pd.DataFrame({"cell": cells, "pt": pt, "segment": 0}).to_csv(meta_path, index=False)
    print(f"  cached: {emb_path}")
    return emb, pt, cells


def eval_one_k(
    emb: np.ndarray,
    pt: np.ndarray,
    n_segments: int,
    n_perm: int,
) -> Dict:
    from sklearn.metrics import silhouette_score

    masks = mask_segment_bins(pt, n_segments)
    seg_ids = np.zeros(len(pt), dtype=np.int64)
    for i, m in enumerate(masks):
        seg_ids[m] = i

    counts = [int(m.sum()) for m in masks]
    sil = float(silhouette_score(emb, seg_ids)) if len(np.unique(seg_ids)) > 1 else float("nan")

    mid_mask = (seg_ids > 0) & (seg_ids < n_segments - 1)
    end_mask = (seg_ids == 0) | (seg_ids == n_segments - 1)
    labels_3 = np.where(mid_mask, 1, np.where(end_mask, 0, -1))
    valid = labels_3 >= 0
    perm_mid = permanova_one_way(emb[valid], labels_3[valid], n_perm=n_perm)
    perm_seg = permanova_one_way(emb, seg_ids, n_perm=n_perm)

    c_mid = emb[mid_mask].mean(axis=0) if mid_mask.any() else np.full(emb.shape[1], np.nan)
    c_early = emb[seg_ids == 0].mean(axis=0)
    c_late = emb[seg_ids == n_segments - 1].mean(axis=0)
    d_me = float(np.linalg.norm(c_mid - c_early))
    d_ml = float(np.linalg.norm(c_mid - c_late))
    d_el = float(np.linalg.norm(c_early - c_late))

    dist_df = pairwise_centroid_distances(emb, seg_ids, n_segments)
    mean_adjacent = float("nan")
    if len(dist_df):
        adj = []
        for s in range(n_segments - 1):
            row = dist_df[(dist_df["seg_i"] == s) & (dist_df["seg_j"] == s + 1)]
            if len(row):
                adj.append(row["centroid_l2"].iloc[0])
        if adj:
            mean_adjacent = float(np.mean(adj))

    return {
        "n_segments": n_segments,
        "n_cells": int(len(pt)),
        "min_cells_per_segment": int(min(counts)),
        "max_cells_per_segment": int(max(counts)),
        "silhouette": sil,
        "permanova_all_R2": perm_seg["R2"],
        "permanova_all_p": perm_seg["p_perm"],
        "permanova_mid_vs_end_R2": perm_mid["R2"],
        "permanova_mid_vs_end_p": perm_mid["p_perm"],
        "centroid_l2_mid_to_early": d_me,
        "centroid_l2_mid_to_late": d_ml,
        "centroid_l2_early_to_late": d_el,
        "centroid_l2_mean_adjacent": mean_adjacent,
        "mid_endpoint_sum_over_el": (d_me + d_ml) / d_el if d_el > 1e-9 else float("nan"),
    }


def run_dataset(
    dataset: str,
    k_list: List[int],
    out_root: Path,
    extract_if_missing: bool,
    n_perm: int,
    batch_size: int,
    use_umap: bool,
    legacy_root: Path,
    skip_panels: bool,
) -> List[Dict]:
    emb, pt, cells = load_or_extract_embeddings(
        dataset, extract_if_missing, batch_size, legacy_root
    )
    rows: List[Dict] = []
    for k in k_list:
        print(f"  [{dataset}] K={k} ...")
        metrics = eval_one_k(emb, pt, k, n_perm=n_perm)
        metrics["dataset"] = dataset
        ds_k_dir = out_root / dataset / f"K{k}"
        ds_k_dir.mkdir(parents=True, exist_ok=True)
        with open(ds_k_dir / "embedding_validation_summary.json", "w") as f:
            json.dump(metrics, f, indent=2)

        rows.append(metrics)
    return rows


def plot_sensitivity(summary: pd.DataFrame, outpath: Path) -> None:
    apply_fig4_style()
    datasets = summary["dataset"].unique().tolist()
    metrics = [
        ("silhouette", "Silhouette (segment labels)", (0, 0.35)),
        ("permanova_all_R2", "PERMANOVA R² (all segments)", (0, 0.35)),
        ("permanova_mid_vs_end_R2", "PERMANOVA R² (middle vs endpoints)", (0, 0.08)),
        ("mid_endpoint_sum_over_el", "(d_mid→early + d_mid→late) / d_early→late", (0.8, 1.4)),
    ]

    fig, axes = plt.subplots(len(metrics), len(datasets), figsize=(4.2 * len(datasets), 3.2 * len(metrics)), squeeze=False)
    colors = {3: model_color("Geneformer"), 5: model_color("scGPT"), 7: model_color("scFoundation")}

    for col, ds in enumerate(datasets):
        sub = summary[summary["dataset"] == ds].sort_values("n_segments")
        x = np.arange(len(sub))
        for row, (col_name, ylabel, ylim) in enumerate(metrics):
            ax = axes[row, col]
            vals = sub[col_name].values
            ks = sub["n_segments"].astype(int).values
            bars = ax.bar(
                x,
                vals,
                color=[colors.get(int(k), "#888") for k in ks],
                edgecolor="white",
                linewidth=0.8,
            )
            for i, (xi, v, k) in enumerate(zip(x, vals, ks)):
                ax.text(xi, v + 0.01 * (ylim[1] - ylim[0]), f"K={k}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels([f"K={k}\nmin n={int(m)}" for k, m in zip(ks, sub["min_cells_per_segment"])], fontsize=8)
            ax.set_ylabel(ylabel, fontsize=9)
            if ylim[1] > ylim[0]:
                ax.set_ylim(ylim)
            if row == 0:
                ax.set_title(ds, fontweight="600", fontsize=11)
            ax.axhline(0, color="#ccc", lw=0.5)

    fig.suptitle(
        "PT segmentation sensitivity in scGPT CLS embedding space",
        fontsize=12,
        fontweight="600",
        y=1.01,
    )
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="K-sensitivity for PT segment embedding validation")
    p.add_argument("--datasets", default="hESC,mHSC-L")
    p.add_argument("--k-list", default="3,5,7")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--legacy-emb-root", type=Path, default=MULTISTEP_ROOT)
    p.add_argument("--extract-if-missing", action="store_true")
    p.add_argument("--n-perm", type=int, default=199, help="PERMANOVA permutations (lower=faster)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--no-umap", action="store_true")
    p.add_argument("--skip-panels", action="store_true", help="Skip per-K UMAP/PCA panels (faster)")
    args = p.parse_args()

    k_list = [int(x) for x in args.k_list.split(",")]
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]

    all_rows: List[Dict] = []
    for ds in datasets:
        print(f"\n=== {ds} ===")
        rows = run_dataset(
            ds,
            k_list,
            args.outdir,
            args.extract_if_missing,
            args.n_perm,
            args.batch_size,
            use_umap=not args.no_umap,
            legacy_root=args.legacy_emb_root,
            skip_panels=args.skip_panels,
        )
        all_rows.extend(rows)

    summary = pd.DataFrame(all_rows)
    csv_path = args.outdir / "embedding_k_sensitivity_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\n{summary.to_string(index=False)}")
    print(f"\nSaved: {csv_path}")

    plot_sensitivity(summary, args.outdir / "embedding_k_sensitivity.png")
    print(f"Saved: {args.outdir / 'embedding_k_sensitivity.png'}")


if __name__ == "__main__":
    main()
