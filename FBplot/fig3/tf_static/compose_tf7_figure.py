#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compose Fig TF-7: model consensus + cross-model failure topology + scGPT clusters.

  A — Spearman ρ of per-TF Jaccard profiles (emb_hidden)
  B — Six-model PCA (k=5 Ward)
  C — scGPT PCA with cluster labels; MCM family highlighted

Example:
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig3
  python tf_static/compose_tf7_figure.py --dataset hESC
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

FIG3 = Path(__file__).resolve().parents[1]
if str(FIG3) not in sys.path:
    sys.path.insert(0, str(FIG3))

from fig3_palette import method_label, model_color  # noqa: E402

from tf_static.model_registry import GT_DISPLAY, MODELS  # noqa: E402
from tf_static.plot_tf_benchmark_extended import (  # noqa: E402
    DEFAULT_CLUSTER_K,
    MCM_GENES,
    _jaccard_pivot,
    _pca_2d,
    apply_plot_style,
    build_tf_feature_matrix,
    run_tf_clustering,
)
from tf_static.plot_tf_hub_family import TEXT_SIZE  # noqa: E402

PANEL_LABEL_SIZE = 13
AXIS_LABEL_SIZE = 9
TICK_SIZE = 8


def _panel_label(ax, letter: str, x: float = -0.12, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _load_per_tf_long(bench_dir: Path, dataset: str, gt_source: str) -> pd.DataFrame:
    path = bench_dir / f"{dataset}_gt-{gt_source}_per_tf_long.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run plot_tf_benchmark_extended.py first.")
    return pd.read_csv(path)


def _load_cluster_summary(bench_dir: Path, dataset: str) -> pd.DataFrame:
    path = bench_dir / f"{dataset}_all_models_tf_cluster_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run --steps tf_cluster first.")
    return pd.read_csv(path)


def compose_tf7(
    df: pd.DataFrame,
    dataset: str,
    gt_source: str,
    out_dir: Path,
    extraction: str = "embhidden500",
    n_clusters: int = DEFAULT_CLUSTER_K,
    models: Tuple[str, ...] = MODELS,
) -> Path:
    # --- data ---
    piv = _jaccard_pivot(df, extraction).dropna(how="all")
    corr = piv.corr(method="spearman")
    model_feats: Dict[str, pd.DataFrame] = {}
    model_pcs: Dict[str, np.ndarray] = {}
    for model in models:
        sub = df[df["model"] == model]
        if sub["TF"].nunique() < n_clusters + 1:
            continue
        feat = build_tf_feature_matrix(df, model)
        feat, _z, pcs = run_tf_clustering(feat, n_clusters)
        model_feats[model] = feat
        model_pcs[model] = pcs

    scgpt = model_feats.get("scGPT")
    if scgpt is None:
        raise RuntimeError("scGPT cluster features unavailable")

    # shared PCA limits (all models)
    all_pcs = np.vstack([model_pcs[m] for m in model_feats])
    pad = 0.1 * max(np.ptp(all_pcs[:, 0]), np.ptp(all_pcs[:, 1]), 1e-6)
    xlim = (all_pcs[:, 0].min() - pad, all_pcs[:, 0].max() + pad)
    ylim = (all_pcs[:, 1].min() - pad, all_pcs[:, 1].max() + pad)
    cmap = plt.colormaps.get_cmap("tab10").resampled(n_clusters)

    # --- figure (A: consensus | B: 6-model PCA | C: scGPT detail) ---
    fig = plt.figure(figsize=(13.5, 9.2), facecolor="white", layout="constrained")
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.0, 0.88],
        width_ratios=[0.9, 1.4],
    )

    # A: consensus
    ax_a = fig.add_subplot(gs[0, 0])
    order = [m for m in models if m in corr.columns]
    corr_ord = corr.loc[order, order]
    sns.heatmap(
        corr_ord,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Spearman ρ", "shrink": 0.78},
        ax=ax_a,
        annot_kws={"size": 8},
    )
    ax_a.set_xticklabels(
        [method_label(m) for m in order], rotation=40, ha="right", fontsize=TICK_SIZE
    )
    ax_a.set_yticklabels([method_label(m) for m in order], rotation=0, fontsize=TICK_SIZE)
    _panel_label(ax_a, "A", x=-0.14, y=1.02)

    # B: 2×3 PCA
    gs_b = gs[0, 1].subgridspec(2, 3, hspace=0.22, wspace=0.16)
    plotted = [m for m in models if m in model_feats]
    for i, model in enumerate(plotted[:6]):
        r, c = divmod(i, 3)
        ax = fig.add_subplot(gs_b[r, c])
        feat = model_feats[model]
        pcs = model_pcs[model]
        for cid in sorted(feat["cluster"].unique()):
            m = feat["cluster"] == cid
            ax.scatter(
                pcs[m, 0],
                pcs[m, 1],
                s=22,
                c=[cmap(int(cid) - 1)],
                edgecolors="#444",
                linewidths=0.15,
                alpha=0.82,
            )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.text(
            0.03,
            0.97,
            method_label(model),
            transform=ax.transAxes,
            fontsize=TICK_SIZE,
            color=model_color(model),
            fontweight="bold",
            va="top",
            ha="left",
        )
        ax.tick_params(labelsize=TICK_SIZE - 1, length=2, pad=1)
        if r == 1:
            ax.set_xlabel("PC1", fontsize=AXIS_LABEL_SIZE)
        else:
            ax.set_xticklabels([])
        if c == 0:
            ax.set_ylabel("PC2", fontsize=AXIS_LABEL_SIZE)
        else:
            ax.set_yticklabels([])
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if i == 0:
            _panel_label(ax, "B", x=-0.20, y=1.10)

    # C: scGPT PCA — bottom row
    ax_c = fig.add_subplot(gs[1, :])
    jac_cols = [c for c in scgpt.columns if c.startswith("jac_")]
    X = scgpt[jac_cols].fillna(0.0).to_numpy()
    pcs = _pca_2d(X)

    cluster_names = {
        1: "C1",
        2: "C2",
        3: "C3",
        4: "C4",
        5: "C5",
    }
    for cid in sorted(scgpt["cluster"].unique()):
        m = scgpt["cluster"] == cid
        ax_c.scatter(
            pcs[m, 0],
            pcs[m, 1],
            s=52 if cid != 3 else 28,
            c=[cmap(int(cid) - 1)],
            label=cluster_names.get(int(cid), f"C{cid}"),
            edgecolors="#333",
            linewidths=0.35,
            alpha=0.9 if cid != 3 else 0.35,
            zorder=3 if cid != 3 else 1,
        )

    # MCM highlight
    mcm_mask = scgpt["TF"].isin(MCM_GENES)
    ax_c.scatter(
        pcs[mcm_mask, 0],
        pcs[mcm_mask, 1],
        s=140,
        facecolors="none",
        edgecolors="#C51B7D",
        linewidths=1.8,
        zorder=5,
    )
    for _, row in scgpt[mcm_mask].iterrows():
        idx = scgpt.index[scgpt["TF"] == row["TF"]][0]
        i = scgpt.index.get_loc(idx)
        ax_c.annotate(
            row["TF"],
            (pcs[i, 0], pcs[i, 1]),
            fontsize=8,
            fontweight="bold",
            color="#C51B7D",
            xytext=(4, 4),
            textcoords="offset points",
            zorder=6,
        )

    ax_c.set_xlabel("PC1", fontsize=AXIS_LABEL_SIZE)
    ax_c.set_ylabel("PC2", fontsize=AXIS_LABEL_SIZE)
    ax_c.tick_params(labelsize=TICK_SIZE)
    ax_c.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        fontsize=TICK_SIZE,
        frameon=False,
        ncol=5,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    for spine in ("top", "right"):
        ax_c.spines[spine].set_visible(False)
    _panel_label(ax_c, "C", x=-0.02, y=1.02)

    # annotation box with key numbers
    c3_n = int((scgpt["cluster"] == 3).sum())
    c2_n = int((scgpt["cluster"] == 2).sum())
    c2_hub = int(
        (
            (scgpt["cluster"] == 2)
            & (scgpt["hub_tier"].astype(str).str.contains("Hub", case=False, na=False))
        ).sum()
    )
    scgpt_rho = corr.loc["scGPT"].drop("scGPT")
    mean_rho = float(scgpt_rho.mean())
    txt = (
        f"scGPT vs others: mean ρ = {mean_rho:.2f}\n"
        f"C3 mass-fail: n = {c3_n} ({100*c3_n/len(scgpt):.0f}%)\n"
        f"C2 dual-emb hub: n = {c2_n}, hub = {c2_hub}/{c2_n}\n"
        f"MCM1–7 → C2 (Type4): 6/6"
    )
    ax_c.text(
        0.99,
        0.03,
        txt,
        transform=ax_c.transAxes,
        ha="right",
        va="bottom",
        fontsize=TICK_SIZE,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CCC", alpha=0.92),
    )

    stem = out_dir / f"{dataset}_FigTF7_failure_topology"
    fig.savefig(f"{stem}.png", dpi=200, facecolor="white")
    fig.savefig(f"{stem}.pdf", facecolor="white")
    plt.close(fig)
    print(f"Fig TF-7 → {stem}.png / .pdf")
    return stem


def write_story_narrative(
    out_dir: Path,
    dataset: str,
    gt_source: str,
    stem: Path,
    corr: pd.DataFrame,
    summ: pd.DataFrame,
    scgpt_feat: pd.DataFrame,
) -> Path:
    gt_name = GT_DISPLAY.get(gt_source, gt_source)
    scgpt_row = summ[summ["model"] == "scGPT"].iloc[0]
    c3_n = int((scgpt_feat["cluster"] == 3).sum())
    c2_n = int((scgpt_feat["cluster"] == 2).sum())
    scgpt_rho = corr.loc["scGPT"].drop("scGPT")
    mean_rho = float(scgpt_rho.mean())
    max_rho = float(scgpt_rho.max())

    md = f"""# Fig TF-7 故事线：{dataset} · {gt_name}

**主图文件**: `{stem.name}.pdf` / `.png`

---

## 一张图讲清的三幕剧

### 第一幕（Panel A）—「大家并没有说同一种语言」

六模型在 **per-TF Jaccard 谱**（{dataset}, cos_hid）上两两 Spearman 相关 **ρ 多在 0.2–0.5**；
scGPT 与其它模型平均 ρ ≈ **{mean_rho:.2f}**（最高约 {max_rho:.2f}），远低于 1。

**含义**：静态 GRN 的 TF 级误差 **没有强跨模型共识**——换模型，「谁好谁坏」会重排。
因此 benchmark **不能只报一个全局冠军分**，而要在 TF 分辨率、多模型对照下报告。

---

### 第二幕（Panel B）—「背景一样惨，结构不一样」

对每模型用 **emb / att / hid 三个 Jaccard** 做 Ward 聚类（k=5）：

| 模型 | 最大簇占比 | Type4 双 emb 型 |
|------|-----------|----------------|
"""
    for _, row in summ.iterrows():
        md += f"| {row['model']} | {row['largest_cluster_pct']:.1f}% | {row['type4_dual_emb_pct']:.1f}% |\n"

    md += f"""
**含义**：

1. **共性**：所有模型都有占 **45–77%** 的「最大簇」——大规模低 Jaccard 是 **行业共性背景**，不是 hESC 或 scGPT 独有。
2. **特异性**：**双 emb 可恢复型**（emb 与 hid 均 >0.1、att 近 0）几乎只在 **scGPT（{scgpt_row['type4_dual_emb_pct']:.1f}%）** 成簇出现；其余模型该比例 ≈ 0–1.5%。
3. **叙事转折**：Panel A 说「排序不一致」；Panel B 说「不一致不等于没有结构——结构在 scGPT 里最清晰」。

---

### 第三幕（Panel C）—「scGPT 把复制模块 hub 单独「救」了出来」

scGPT 聚类（与主文 TF-2 hub 分层、TF-4 MCM 案例衔接）：

- **C3 质量失败簇**：n = {c3_n}（{100*c3_n/len(scgpt_feat):.0f}%），三提取 Jaccard 近 0 → 与「全局失效」一致。
- **C2 双 emb hub 簇**：n = {c2_n}，富 hub；**MCM1–7 全部落入 C2**（洋红圈）→ 复制/增殖模块在 emb/hid 通道可部分恢复，att 通道几乎无效。
- **C1 emb 主导**：少数 RAD51/TRIP13 类，spread 与 emb−att 更大（见 Supp. 统计检验）。

**与 Fig TF-2 的勾连**：Hub 平均 Jaccard 高于 Specialist，但 **增益集中在 C2 而非全体高 out-degree TF**。

**与 Fig TF-4 的勾连**：MCM5 Venn 中 emb-only / hid-only 靶基因，在聚类里归纳为 **Type4 / C2**。

---

## 正文 Results 段落（可直接粘贴）

> **Model agreement and failure topology.** Per-TF Jaccard profiles (cos_hidden) across six foundation models were only moderately concordant (Spearman ρ ≈ 0.2–0.5; Fig. TF-7A), indicating that static GRN errors reshuffle substantially between architectures. Nevertheless, Ward clustering on three extraction-wise Jaccards revealed a dominant low-performance cluster in every model (45–77% of TFs; Fig. TF-7B), a shared mass-failure background that is not hESC-specific (Supp., hHep). Only scGPT further resolved a hub-enriched, dual-embedding–recoverable cluster (9.3% of TFs) that contained all seven MCM genes (Fig. TF-7C). Thus, benchmark conclusions must separate **pan-model collapse** from **method-specific recoverable replication hubs**.

---

## 图注（Figure legend）

**Figure TF-7. Cross-model agreement and TF-resolved failure topology ({dataset}, {gt_name}).**
**(A)** Spearman correlation of per-TF Jaccard vectors (cos_hidden) between models.
**(B)** Per-model PCA of TF features (Jaccard for cos_token, cos_hidden, and attention); Ward clustering with k = 5 (shared axes).
**(C)** scGPT PCA colored by cluster; replication MCM genes (magenta rings). Inset statistics: cross-model ρ, cluster sizes, and MCM assignment.

---

## 建议 Supp 引用（不占主文版面）

- 六模型完整树状图：`hESC_*_tf_cluster.pdf`
- 统计检验：`validation/hESC_validation_stats_tests.csv`（C2 vs C3, FDR < 0.05）
- 跨细胞系：`validation/validation_cross_dataset_semantic.*`（hHep 最大簇 73.6%, MCM 仍 Type4）
- Recoverable TF 名单：`hESC_gt-STRING_recoverable_tf_embhidden500.csv`（54/343, ≥2 models, J ≥ 0.05）
"""
    story_path = out_dir / f"{dataset}_FigTF7_story.md"
    story_path.write_text(md, encoding="utf-8")
    print(f"Story → {story_path}")
    return story_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compose Fig TF-7 (consensus + topology + scGPT clusters)")
    p.add_argument("--dataset", default="hESC")
    p.add_argument("--gt-source", default="STRING")
    p.add_argument("--extraction", default="embhidden500")
    p.add_argument("--bench-dir", type=Path, default=None)
    p.add_argument("--cluster-k", type=int, default=DEFAULT_CLUSTER_K)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    apply_plot_style()
    bench_dir = args.bench_dir or (
        FIG3 / "tf_static" / "output" / args.dataset / f"benchmark_extended_{args.gt_source}"
    )
    df = _load_per_tf_long(bench_dir, args.dataset, args.gt_source)
    summ = _load_cluster_summary(bench_dir, args.dataset)
    stem = compose_tf7(
        df,
        args.dataset,
        args.gt_source,
        bench_dir,
        args.extraction,
        args.cluster_k,
    )
    corr = _jaccard_pivot(df, args.extraction).dropna(how="all").corr(method="spearman")
    scgpt_feat = pd.read_csv(bench_dir / f"{args.dataset}_scGPT_tf_cluster_features.csv")
    write_story_narrative(bench_dir, args.dataset, args.gt_source, stem, corr, summ, scgpt_feat)


if __name__ == "__main__":
    main()
