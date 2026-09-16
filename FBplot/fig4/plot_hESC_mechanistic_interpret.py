#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hESC 链式 scGPT 机制向可解释分析（残差/曲率/平滑 + 嵌入空间轨迹）

Tier A：从 chained / independent CSV 计算每段 |Δ_pred|/|Δ_true|、曲率、峰段平滑
Tier B：真值细胞 CLS embedding 的伪时间质心路径（已有 embedding_validation/）

输出：chained_preds/figures/mechanistic_interpret/

示例：
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig4
  python3 plot_hESC_mechanistic_interpret.py
  python3 plot_hESC_mechanistic_interpret.py --genes CALB1,HDGF,PRTG,HESX1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

try:
    from fig4_palette import apply_fig4_style, model_color
except ImportError:

    def apply_fig4_style() -> None:
        plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12})

    def model_color(name: str, default: str = "#666") -> str:
        return default


from plot_hESC_indep_vs_chain_nonlinear import (  # noqa: E402
    N_PTS,
    PT_NODES,
    TRANSITIONS,
    TRANS_LABELS,
    TRAJ_COLORS,
    build_metrics_table,
    chain_profile,
    classify_trajectory,
    curvature,
    indep_profile,
    load_chained_tables,
    load_indep_tables,
    norm_profile,
    peak_segment_index,
    true_profile,
)

SCRIPT_DIR = Path(__file__).resolve().parent
HESC = SCRIPT_DIR / "error_biology" / "multistep_pt" / "hESC"
CHAINED_DIR = HESC / "chained_preds"
CHAINED_EMB_DIR = CHAINED_DIR / "chained_embeddings"
INDEP_DIR = HESC / "segment_preds" / "scgpt"
EMB_DIR = HESC / "embedding_validation"
GENE_SEG = HESC / "gene_segment_deltas.csv"
DEFAULT_EXAMPLES = HESC / "chained_preds" / "figures" / "nonlinear_explain" / "example_genes.csv"

COLOR_TRUE = model_color("scPrint")
COLOR_INDEP = model_color("scFoundation")
COLOR_CHAIN = model_color("scGPT")
COLOR_OK = "#2E7D32"
COLOR_BAD = "#C62828"
COLOR_PEAK = model_color("scFoundation")
MARKER_KW = dict(markersize=10, markeredgewidth=1.4, markeredgecolor="white")


def segment_local_deltas(
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """节点序列 → 段内 Δ（长度 n_seg）。"""
    d = np.diff(y)
    return d, np.arange(len(d))


def per_segment_metrics(
    ty: np.ndarray,
    py_chain: np.ndarray,
    py_indep: np.ndarray,
) -> pd.DataFrame:
    """每段：真值/预测 Δ、幅度比、残差、方向是否一致。"""
    dt = np.diff(ty)
    dc = np.diff(py_chain)
    di = np.diff(py_indep)
    rows = []
    for i, lab in enumerate(TRANS_LABELS):
        t, c, ind = float(dt[i]), float(dc[i]), float(di[i])
        rows.append(
            {
                "segment": lab,
                "seg_idx": i,
                "delta_true": t,
                "delta_chain": c,
                "delta_indep": ind,
                "abs_true": abs(t),
                "abs_chain": abs(c),
                "abs_indep": abs(ind),
                "ratio_chain": abs(c) / (abs(t) + 1e-9),
                "ratio_indep": abs(ind) / (abs(t) + 1e-9),
                "residual_chain": c - t,
                "residual_indep": ind - t,
                "dir_ok_chain": int(np.sign(c) == np.sign(t)) if t != 0 else int(abs(c) < 1e-6),
                "dir_ok_indep": int(np.sign(ind) == np.sign(t)) if t != 0 else int(abs(ind) < 1e-6),
            }
        )
    return pd.DataFrame(rows)


def load_example_genes(path: Path) -> List[str]:
    if not path.exists():
        return ["CALB1", "HDGF", "PRTG", "HESX1"]
    df = pd.read_csv(path)
    if "gene" in df.columns:
        col = "gene"
    else:
        col = df.columns[-1]
    priority = ["peak_middle_bad", "sign_flip_good", "peak_middle_good", "sign_flip_bad"]
    genes = []
    for key in priority:
        row = df[df.iloc[:, 0].astype(str) == key] if df.shape[1] > 1 else pd.DataFrame()
        if len(row):
            genes.append(str(row.iloc[0][col]))
    if not genes:
        genes = df[col].astype(str).tolist()[:4]
    return list(dict.fromkeys(genes))[:4]


def plot_fig01_case_trajectories(
    out: Path,
    genes: List[str],
    ch: Dict[str, pd.DataFrame],
    ind: Dict[int, pd.DataFrame],
    metrics: pd.DataFrame,
) -> None:
    """2×2 典型案例：观测 / 独立 / 链式 + 真峰段高亮。"""
    n = len(genes)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 5.2 * nrows), squeeze=False)
    x = np.arange(N_PTS)
    m_idx = metrics.set_index("gene") if len(metrics) else pd.DataFrame()

    for ax, gene in zip(axes.flat, genes):
        ty = true_profile(ch, gene)
        pi = indep_profile(ind, ch, gene)
        pc = chain_profile(ch, gene)
        if not np.all(np.isfinite(ty)):
            ax.set_visible(False)
            continue
        peak_i = peak_segment_index(np.diff(ty))
        ax.axvspan(peak_i, peak_i + 1, color=COLOR_PEAK, alpha=0.22, zorder=0)
        ax.plot(x, ty, "o-", color=COLOR_TRUE, lw=3, label="Observed", zorder=4, **MARKER_KW)
        ax.plot(
            x, pi, "s--", color=COLOR_INDEP, lw=2.4, ms=9,
            label="Independent", zorder=3,
        )
        ax.plot(
            x, pc, "^-", color=COLOR_CHAIN, lw=2.4, ms=9,
            label="Chained", zorder=3,
        )
        traj = m_idx.loc[gene, "trajectory_type"] if gene in m_idx.index else "?"
        sc = m_idx.loc[gene, "shape_corr_chain"] if gene in m_idx.index else np.nan
        si = m_idx.loc[gene, "shape_corr_indep"] if gene in m_idx.index else np.nan
        ax.set_title(
            f"{gene}  ({traj})\nshape r: indep {si:.2f}  |  chain {sc:.2f}",
            fontweight="700",
            fontsize=12,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(PT_NODES, fontsize=11)
        ax.set_ylabel("Mean expression", fontsize=11)
        ax.legend(fontsize=9, loc="best", framealpha=0.95)
        ax.grid(axis="y", alpha=0.3)
        ax.text(
            0.02, 0.98, f"Peak |v| @ {TRANS_LABELS[peak_i]}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )

    for ax in axes.flat[len(genes) :]:
        ax.set_visible(False)

    fig.suptitle(
        "A  Representative genes — true peak segment (gold) vs predictions",
        fontsize=14, fontweight="700", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_fig02_smoothing_and_curvature(
    out: Path,
    gene: str,
    seg_df: pd.DataFrame,
    peak_idx: int,
) -> None:
    """单基因三指标：|Δ| 幅度比、残差、离散曲率。"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    x = np.arange(len(TRANSITIONS))
    w = 0.36

    ax = axes[0]
    ax.bar(x - w / 2, seg_df["ratio_indep"], w, label="Independent", color=COLOR_INDEP, alpha=0.9)
    ax.bar(x + w / 2, seg_df["ratio_chain"], w, label="Chained", color=COLOR_CHAIN, alpha=0.9)
    ax.axhline(1.0, color="#78909C", ls="--", lw=1)
    ax.axvspan(peak_idx - 0.45, peak_idx + 0.45, color=COLOR_PEAK, alpha=0.15)
    ax.set_xticks(x)
    ax.set_xticklabels(TRANS_LABELS, rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("|Δ_pred| / |Δ_true|")
    ax.set_title("Magnitude ratio (<1 ⇒ smoothing)", fontweight="600")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.bar(x - w / 2, seg_df["residual_indep"], w, label="Independent", color=COLOR_INDEP, alpha=0.85)
    ax.bar(x + w / 2, seg_df["residual_chain"], w, label="Chained", color=COLOR_CHAIN, alpha=0.85)
    ax.axhline(0, color="#666", lw=0.8)
    ax.axvspan(peak_idx - 0.45, peak_idx + 0.45, color=COLOR_PEAK, alpha=0.15)
    ax.set_xticks(x)
    ax.set_xticklabels(TRANS_LABELS, rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("Δ_pred − Δ_true")
    ax.set_title("Per-block velocity residual", fontweight="600")
    ax.legend(fontsize=9)

    ax = axes[2]
    ty_c = seg_df["delta_true"].values
    # 曲率用相邻段 Δ 的二阶差分（3 点）
    ct = curvature(np.concatenate([[0], ty_c]))  # pad for length
    cc = curvature(np.concatenate([[0], seg_df["delta_chain"].values]))
    ci = curvature(np.concatenate([[0], seg_df["delta_indep"].values]))
    xc = np.arange(len(ct))
    ax.plot(xc, np.abs(ct), "o-", color=COLOR_TRUE, lw=2.5, label="|curv| true", ms=8)
    ax.plot(xc, np.abs(cc), "^-", color=COLOR_CHAIN, lw=2.2, label="|curv| chain", ms=8)
    ax.plot(xc, np.abs(ci), "s--", color=COLOR_INDEP, lw=2.2, label="|curv| indep", ms=8)
    ax.set_xticks(xc)
    ax.set_xticklabels([f"k{i}" for i in xc])
    ax.set_ylabel("|second difference|")
    ax.set_title("Curvature proxy along blocks", fontweight="600")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle(
        f"B  Mechanistic diagnostics — {gene}  (gold = true peak block)",
        fontsize=13, fontweight="700", y=1.05,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_fig03_aggregate_smoothing(
    out: Path,
    metrics: pd.DataFrame,
    ch: Dict[str, pd.DataFrame],
    ind: Dict[int, pd.DataFrame],
) -> None:
    """按轨迹类型：峰段 vs 非峰段的 |Δ| 比值（链式更平滑？）。"""
    records = []
    genes = metrics["gene"].astype(str).tolist()
    for g in genes:
        ty = true_profile(ch, g)
        pc = chain_profile(ch, g)
        pi = indep_profile(ind, ch, g)
        if not np.all(np.isfinite(ty)) or not np.all(np.isfinite(pc)):
            continue
        seg = per_segment_metrics(ty, pc, pi)
        peak_i = peak_segment_index(np.diff(ty))
        traj = metrics.loc[metrics["gene"] == g, "trajectory_type"].iloc[0]
        for _, r in seg.iterrows():
            is_peak = int(r["seg_idx"] == peak_i)
            records.append(
                {
                    "gene": g,
                    "trajectory_type": traj,
                    "is_peak_seg": is_peak,
                    "ratio_chain": r["ratio_chain"],
                    "ratio_indep": r["ratio_indep"],
                    "abs_true": r["abs_true"],
                }
            )
    df = pd.DataFrame(records)
    if df.empty:
        return

    types = ["peak_middle", "sign_flip", "monotone"]
    types = [t for t in types if t in df["trajectory_type"].values]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    for ax, col, title in zip(
        axes,
        ["ratio_chain", "ratio_indep"],
        ["Chained: |Δ_pred|/|Δ_true|", "Independent: |Δ_pred|/|Δ_true|"],
    ):
        data_peak, data_non = [], []
        labels = []
        for t in types:
            sub = df[df["trajectory_type"] == t]
            data_peak.append(sub[sub["is_peak_seg"] == 1][col].values)
            data_non.append(sub[sub["is_peak_seg"] == 0][col].values)
            labels.append(t.replace("_", "\n"))
        xp = np.arange(len(types))
        bp = ax.boxplot(
            data_peak,
            positions=xp - 0.2,
            widths=0.32,
            patch_artist=True,
            showfliers=False,
        )
        bn = ax.boxplot(
            data_non,
            positions=xp + 0.2,
            widths=0.32,
            patch_artist=True,
            showfliers=False,
        )
        for box in bp["boxes"]:
            box.set_facecolor(COLOR_PEAK)
            box.set_alpha(0.75)
        for box in bn["boxes"]:
            box.set_facecolor("#B0BEC5")
            box.set_alpha(0.75)
        ax.axhline(1.0, color="#546E7A", ls="--", lw=1)
        ax.set_xticks(xp)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel(col.replace("_", " "))
        ax.set_title(title, fontweight="600")
        ax.legend(
            handles=[
                mpatches.Patch(facecolor=COLOR_PEAK, alpha=0.75, label="True peak block"),
                mpatches.Patch(facecolor="#B0BEC5", alpha=0.75, label="Other blocks"),
            ],
            fontsize=9,
            loc="upper right",
        )
        ax.set_ylim(0, min(3.0, ax.get_ylim()[1]))

    fig.suptitle(
        "C  Peak-block smoothing — ratio < 1 means predicted change smaller than truth",
        fontsize=13, fontweight="700", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    df.to_csv(out.with_suffix(".csv"), index=False)


def plot_fig04_embedding_path(out: Path, emb_dir: Path) -> None:
    """真值细胞 embedding：各伪时间段的质心路径（Tier B 轻量版）。"""
    meta_path = emb_dir / "cell_embedding_meta.csv"
    emb_path = emb_dir / "cell_embeddings.npy"
    if not meta_path.exists() or not emb_path.exists():
        return
    meta = pd.read_csv(meta_path)
    emb = np.load(emb_path)
    if "segment" not in meta.columns:
        return

    n_seg = int(meta["segment"].max()) + 1
    cents = []
    for s in range(n_seg):
        m = meta["segment"].values == s
        if m.sum() == 0:
            continue
        cents.append(emb[m].mean(axis=0))
    cents = np.array(cents)

    xy = meta[["umap1", "umap2"]].values if {"umap1", "umap2"}.issubset(meta.columns) else None
    if xy is None:
        from sklearn.decomposition import PCA
        xy = PCA(2, random_state=0).fit_transform(emb)

    cent_xy = []
    for s in range(n_seg):
        m = meta["segment"].values == s
        cent_xy.append(xy[m].mean(axis=0))
    cent_xy = np.array(cent_xy)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    ax = axes[0]
    sc = ax.scatter(
        xy[:, 0], xy[:, 1], c=meta["segment"], cmap="viridis",
        s=8, alpha=0.35, linewidths=0,
    )
    ax.plot(cent_xy[:, 0], cent_xy[:, 1], "o-", color=COLOR_BAD, lw=3, zorder=5, **MARKER_KW)
    for i, lab in enumerate(PT_NODES[:n_seg]):
        ax.annotate(lab, cent_xy[i], textcoords="offset points", xytext=(6, 6), fontsize=11, fontweight="700")
    plt.colorbar(sc, ax=ax, label="PT segment", fraction=0.046)
    ax.set_title("Observed cells — segment centroids in UMAP", fontweight="600")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    ax = axes[1]
    # 段间 L2 步长 vs 沿 PT 单调路径
    steps = [float(np.linalg.norm(cents[i + 1] - cents[i])) for i in range(len(cents) - 1)]
    ax.bar(np.arange(len(steps)), steps, color=COLOR_TRUE, edgecolor="white", alpha=0.9)
    ax.set_xticks(np.arange(len(steps)))
    ax.set_xticklabels(TRANS_LABELS[: len(steps)], rotation=20, ha="right")
    ax.set_ylabel("Centroid L2 distance in CLS space")
    ax.set_title("Embedding step size between segments", fontweight="600")
    if len(steps) >= 2:
        ax.axvspan(0.5, 1.5, color=COLOR_PEAK, alpha=0.2, label="Often peak expr. block")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    summ_path = emb_dir / "embedding_validation_summary.json"
    note = ""
    if summ_path.exists():
        summ = json.loads(summ_path.read_text())
        note = (
            f"PERMANOVA R²={summ.get('permanova_all_segments', {}).get('R2', 0):.2f}  "
            f"(p={summ.get('permanova_all_segments', {}).get('p_perm', 1):.3f})"
        )
    fig.suptitle(
        f"D  True-cell scGPT embedding geometry along pseudotime  {note}",
        fontsize=13, fontweight="700", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_fig07_chained_vs_true_embedding(
    out: Path,
    emb_dir: Path,
    chained_emb_dir: Path,
) -> bool:
    """
    真细胞 UMAP 背景 + 真值段质心路径 vs 链式 S0 cohort 逐 block 质心路径。
    需要 chained_embeddings/chain_segment_centroids.npy（extract 或 run --save-embeddings）。
    """
    meta_path = emb_dir / "cell_embedding_meta.csv"
    emb_path = emb_dir / "cell_embeddings.npy"
    chain_path = chained_emb_dir / "chain_segment_centroids.npy"
    true_path = chained_emb_dir / "true_segment_centroids.npy"
    if not all(p.exists() for p in (meta_path, emb_path, chain_path, true_path)):
        print(f"  Skip fig07: missing files under {emb_dir} or {chained_emb_dir}")
        return False

    meta = pd.read_csv(meta_path)
    cell_emb = np.load(emb_path)
    chain_c = np.load(chain_path)
    true_c = np.load(true_path)
    n_seg = chain_c.shape[0]

    summ = {}
    summ_path = chained_emb_dir / "trajectory_summary.json"
    if summ_path.exists():
        summ = json.loads(summ_path.read_text())

    # Shared 2D: PCA on all cells + both centroid paths
    from sklearn.decomposition import PCA

    pca = PCA(2, random_state=0)
    pca.fit(cell_emb)
    xy_cells = pca.transform(cell_emb)
    xy_true = pca.transform(true_c)
    xy_chain = pca.transform(chain_c)

    true_steps = pairwise_steps(true_c)
    chain_steps = pairwise_steps(chain_c)
    if not true_steps:
        true_steps = summ.get("true_step_l2", [])
        chain_steps = summ.get("chain_step_l2", [])

    fig = plt.figure(figsize=(13, 5.5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    sc = ax.scatter(
        xy_cells[:, 0], xy_cells[:, 1],
        c=meta["segment"], cmap="viridis", s=10, alpha=0.3, linewidths=0,
    )
    ax.plot(
        xy_true[:, 0], xy_true[:, 1], "o-",
        color=COLOR_TRUE, lw=3.5, label="True segment centroids", zorder=5, **MARKER_KW,
    )
    ax.plot(
        xy_chain[:, 0], xy_chain[:, 1], "s--",
        color=COLOR_CHAIN, lw=3, ms=10, label="Chained block centroids (S0 cohort)",
        zorder=6,
    )
    for i in range(n_seg):
        ax.annotate(f"S{i}", xy_true[i], textcoords="offset points", xytext=(7, 5),
                    fontsize=10, fontweight="700", color=COLOR_TRUE)
        ax.annotate(f"S{i}", xy_chain[i], textcoords="offset points", xytext=(7, -12),
                    fontsize=10, fontweight="700", color=COLOR_CHAIN)
    plt.colorbar(sc, ax=ax, label="PT segment", fraction=0.046)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("A  Embedding paths (PCA of observed cells)", fontweight="600", loc="left")
    ax.legend(fontsize=9, loc="best", framealpha=0.95)
    ax.grid(alpha=0.2)

    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(n_seg - 1)
    w = 0.36
    ax.bar(x - w / 2, true_steps, w, label="True centroids", color=COLOR_TRUE, alpha=0.9)
    ax.bar(x + w / 2, chain_steps, w, label="Chained centroids", color=COLOR_CHAIN, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(TRANS_LABELS[: len(x)], rotation=18, ha="right")
    ax.set_ylabel("L2 step in CLS space (512-d)")
    ax.set_title("B  Step size between consecutive nodes", fontweight="600", loc="left")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    if len(x) >= 2:
        ax.axvspan(0.5, 1.5, color=COLOR_PEAK, alpha=0.15)

    ratio = summ.get("path_length_ratio_chain_over_true")
    note = f"path length chain/true = {ratio:.2f}" if ratio is not None else ""
    fig.suptitle(
        f"True vs chained scGPT embedding trajectory along pseudotime  ({note})",
        fontsize=13, fontweight="700", y=1.02,
    )
    fig.text(
        0.5, -0.02,
        "Chained path: S0 cells updated block-wise; centroid = mean CLS after each block. "
        "Shorter/smoother chain path ⇒ representation linearization.",
        ha="center", fontsize=9, color="#546E7A",
    )
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def pairwise_steps(centroids: np.ndarray) -> List[float]:
    steps = []
    for i in range(len(centroids) - 1):
        steps.append(float(np.linalg.norm(centroids[i + 1] - centroids[i])))
    return steps


def plot_fig05_combined_story(
    out: Path,
    gene: str,
    ch: Dict[str, pd.DataFrame],
    ind: Dict[int, pd.DataFrame],
    seg_df: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    """CALB1 / peak_middle_bad 一页故事板。"""
    ty = true_profile(ch, gene)
    pi = indep_profile(ind, ch, gene)
    pc = chain_profile(ch, gene)
    peak_i = peak_segment_index(np.diff(ty))
    x = np.arange(N_PTS)

    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.1, 1.0, 0.95], hspace=0.42, wspace=0.28)

    # A trajectory
    ax = fig.add_subplot(gs[0, :])
    ax.axvspan(peak_i, peak_i + 1, color=COLOR_PEAK, alpha=0.2)
    ax.plot(x, ty, "o-", color=COLOR_TRUE, lw=3, label="Observed", **MARKER_KW)
    ax.plot(x, pi, "s--", color=COLOR_INDEP, lw=2.5, ms=9, label="Independent")
    ax.plot(x, pc, "^-", color=COLOR_CHAIN, lw=2.5, ms=9, label="Chained")
    r = metrics[metrics["gene"] == gene].iloc[0] if gene in metrics["gene"].values else None
    subtitle = ""
    if r is not None:
        subtitle = (
            f"{r['trajectory_type']} | peak_hit chain={int(r['peak_hit_chain'])} "
            f"| shape r chain={r['shape_corr_chain']:.2f}"
        )
    ax.set_title(f"Case study: {gene}  —  {subtitle}", fontweight="700", fontsize=13, loc="left")
    ax.set_xticks(x)
    ax.set_xticklabels(PT_NODES)
    ax.set_ylabel("Mean expression")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)

    # B ratio
    ax = fig.add_subplot(gs[1, 0])
    xi = np.arange(4)
    ax.bar(xi - 0.2, seg_df["ratio_indep"], 0.38, label="Indep", color=COLOR_INDEP)
    ax.bar(xi + 0.2, seg_df["ratio_chain"], 0.38, label="Chain", color=COLOR_CHAIN)
    ax.axhline(1, color="#888", ls="--")
    ax.axvspan(peak_i - 0.45, peak_i + 0.45, color=COLOR_PEAK, alpha=0.12)
    ax.set_xticks(xi)
    ax.set_xticklabels(TRANS_LABELS, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("|Δ_pred|/|Δ_true|")
    ax.set_title("Peak block often shows strong smoothing (chain)", fontweight="600", loc="left")
    ax.legend(fontsize=9)

    # C schematic blocks
    ax = fig.add_subplot(gs[1, 1])
    ax.set_xlim(-0.2, 4.2)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Chained error mechanism (hypothesis)", fontweight="600", loc="left")
    for i, lab in enumerate(TRANS_LABELS):
        ok = seg_df.iloc[i]["dir_ok_chain"]
        fc = "#E8F5E9" if ok else "#FFEBEE"
        ec = COLOR_OK if ok else COLOR_BAD
        ax.text(i + 0.5, 0.7, lab, ha="center", fontsize=9, fontweight="600",
                bbox=dict(boxstyle="round", facecolor=fc, edgecolor=ec))
        ax.text(i + 0.5, 0.35, "✓" if ok else "✗", ha="center", fontsize=14, color=ec, fontweight="bold")
        if i < 3:
            ax.annotate("", xy=(i + 0.9, 0.52), xytext=(i + 0.1, 0.52),
                        arrowprops=dict(arrowstyle="-|>", color=COLOR_CHAIN, lw=2))
    ax.text(2.0, 0.08, "state carries → next block", ha="center", fontsize=9, color="#555")

    # D text insight
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    pk = seg_df.iloc[peak_i]
    lines = [
        "Interpretation checklist:",
        f"• True peak block: {TRANS_LABELS[peak_i]}  (|Δ_true|={pk['abs_true']:.3f})",
        f"• Chained ratio at peak: {pk['ratio_chain']:.2f}  (<<1 ⇒ underestimates velocity magnitude)",
        f"• Independent ratio at peak: {pk['ratio_indep']:.2f}",
        f"• Chained places expression peak at S{peak_segment_index(np.diff(pc))} (pred) vs S{peak_i} (true) → peak_hit=0 typ.",
        "• Conclusion: chained updates linearize trajectory in expression space; not a lack of biological signal.",
    ]
    ax.text(0.02, 0.95, "\n".join(lines), va="top", fontsize=11, family="monospace",
            bbox=dict(boxstyle="round", facecolor="#FAFAFA", edgecolor="#CFD8DC"))

    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_fig06_multi_gene_smoothing_heatmap(
    out: Path,
    genes: List[str],
    ch: Dict[str, pd.DataFrame],
    ind: Dict[int, pd.DataFrame],
) -> None:
    """多基因 × 段：链式 |Δ| 比值热图。"""
    mat = []
    for g in genes:
        ty = true_profile(ch, g)
        pc = chain_profile(ch, g)
        pi = indep_profile(ind, ch, g)
        if not np.all(np.isfinite(ty)):
            continue
        seg = per_segment_metrics(ty, pc, pi)
        mat.append(seg["ratio_chain"].values)
    if not mat:
        return
    mat = np.array(mat)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(genes))))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=2.0)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(TRANS_LABELS, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(genes)))
    ax.set_yticklabels(genes)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if mat[i, j] > 1.0 else "#333")
    plt.colorbar(im, ax=ax, label="|Δ_chain|/|Δ_true|")
    ax.set_title("E  Chained magnitude ratio per gene × block (<1 = smoothing)", fontweight="700")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--genes", default=None, help="comma-separated; default from example_genes.csv")
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--story-gene", default="CALB1", help="full story panel gene")
    args = p.parse_args()

    apply_fig4_style()
    outdir = args.outdir or (CHAINED_DIR / "figures" / "mechanistic_interpret")
    outdir.mkdir(parents=True, exist_ok=True)

    ch = load_chained_tables(CHAINED_DIR)
    ind = load_indep_tables(INDEP_DIR)
    gf = pd.read_csv(GENE_SEG).set_index("gene") if GENE_SEG.exists() else pd.DataFrame()

    genes = [g.strip() for g in args.genes.split(",")] if args.genes else load_example_genes(DEFAULT_EXAMPLES)
    all_genes = list(ch["delta_t0_t1"].index.astype(str))
    metrics = build_metrics_table(all_genes, ch, ind, gf)
    metrics.to_csv(outdir / "gene_mechanistic_metrics.csv", index=False)

    plot_fig01_case_trajectories(outdir / "01_case_trajectories_2x2.png", genes, ch, ind, metrics)
    plot_fig03_aggregate_smoothing(outdir / "03_aggregate_peak_smoothing.png", metrics, ch, ind)
    plot_fig04_embedding_path(outdir / "04_embedding_segment_path.png", EMB_DIR)
    plot_fig07_chained_vs_true_embedding(
        outdir / "07_chained_vs_true_embedding.png", EMB_DIR, CHAINED_EMB_DIR,
    )
    plot_fig06_multi_gene_smoothing_heatmap(outdir / "06_chain_ratio_heatmap.png", genes, ch, ind)

    seg_tables = {}
    for g in genes:
        ty = true_profile(ch, g)
        pc = chain_profile(ch, g)
        pi = indep_profile(ind, ch, g)
        if not np.all(np.isfinite(ty)):
            continue
        seg_df = per_segment_metrics(ty, pc, pi)
        peak_i = peak_segment_index(np.diff(ty))
        seg_df.to_csv(outdir / f"segment_metrics_{g}.csv", index=False)
        plot_fig02_smoothing_and_curvature(outdir / f"02_mechanism_{g}.png", g, seg_df, peak_i)

    story = args.story_gene
    if story in ch["delta_t0_t1"].index.astype(str):
        ty = true_profile(ch, story)
        pc = chain_profile(ch, story)
        pi = indep_profile(ind, ch, story)
        seg_df = per_segment_metrics(ty, pc, pi)
        plot_fig05_combined_story(outdir / f"05_storyboard_{story}.png", story, ch, ind, seg_df, metrics)

    summary = {
        "n_genes": len(metrics),
        "mean_shape_corr_chain": float(metrics["shape_corr_chain"].mean()),
        "mean_shape_corr_indep": float(metrics["shape_corr_indep"].mean()),
        "peak_hit_chain_pct": float(metrics["peak_hit_chain"].mean() * 100),
        "peak_hit_indep_pct": float(metrics["peak_hit_indep"].mean() * 100),
        "example_genes": genes,
        "story_gene": story,
    }
    by_type = metrics.groupby("trajectory_type").agg(
        n=("gene", "count"),
        shape_chain=("shape_corr_chain", "mean"),
        shape_indep=("shape_corr_indep", "mean"),
        peak_hit_chain=("peak_hit_chain", "mean"),
    )
    by_type.to_csv(outdir / "summary_by_trajectory_type.csv")
    with open(outdir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Mechanistic interpretability plots → {outdir} ===")
    for f in sorted(outdir.glob("*.png")):
        print(f"  {f.name}")
    print("\nGlobal:")
    print(f"  shape r  chain={summary['mean_shape_corr_chain']:.3f}  indep={summary['mean_shape_corr_indep']:.3f}")
    print(f"  peak hit chain={summary['peak_hit_chain_pct']:.1f}%  indep={summary['peak_hit_indep_pct']:.1f}%")


if __name__ == "__main__":
    main()
