#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compose Fig4 multi-panel figures aligned with the three-act story.

Outputs (under --outdir):
  Fig4_main_act123.{pdf,png}           — 12-panel main figure (Acts I–III)
  Fig4_supp_S1_benchmark.{pdf,png}     — accuracy / convergence gallery
  Fig4_supp_S2_error_biology.{pdf,png} — error stratification gallery (6 datasets)
  Fig4_supp_S3_hESC_mechanism.{pdf,png} — hESC multistep / ERCC / case genes
  Fig4_supp_S4_methods.{pdf,png}       — K-sensitivity / overlap / reghead
  Fig4_storyboard.md                   — panel → story mapping + file paths

Example:
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig4
  python3 compose_fig4_figure.py
  python3 compose_fig4_figure.py --outdir output/compose --dpi 300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

SCRIPT_DIR = Path(__file__).resolve().parent
FIG3_OVERLAP = (
    SCRIPT_DIR.parent
    / "fig3/tf_static/output/hESC/benchmark_extended_STRING/hESC_scGPT_embhidden500_static_dynamic_overlap.csv"
)

MODEL_JSON = {
    "scGPT": SCRIPT_DIR / "interation/scgpt_accuracy_curves.json",
    "Geneformer": SCRIPT_DIR / "interation/geneformer_accuracy_curves_6datasets.json",
    "LangCell": SCRIPT_DIR / "interation/Langcell_accuracy_curves.json",
    "scCello": SCRIPT_DIR / "interation/sccello_accuracy_curves.json",
    "scFoundation": SCRIPT_DIR / "interation/scfoundation_accuracy_curves.json",
    "scPrint": SCRIPT_DIR / "interation/scprint_accuracy_curves.json",
}

MODEL_ORDER = ["scFoundation", "LangCell", "scPrint", "scGPT", "Geneformer", "scCello"]
MAIN_DATASETS = ["hESC", "hHep", "mHSC-E", "mHSC-GM", "mHSC-L"]
ALL_DATASETS = ["hESC", "hHep", "mDC", "mHSC-E", "mHSC-GM", "mHSC-L"]

ERROR_COLORS = {
    "well_predicted": "#4C9F70",
    "amplitude_wrong": "#F9A825",
    "strict_wrong": "#EB7E60",
}
ERROR_LABELS = {
    "well_predicted": "Well predicted",
    "amplitude_wrong": "Amplitude error",
    "strict_wrong": "Direction error",
}
TRAJ_COLORS = {
    "peak_middle": "#EB7E60",
    "sign_flip": "#AC99D2",
    "monotone": "#4EA3F1",
    "mixed": "#70CDBE",
}

try:
    from fig4_palette import apply_fig4_style, model_color
except ImportError:

    def apply_fig4_style() -> None:
        plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    def model_color(name: str, default: str = "#808080") -> str:
        return default


def _panel_label(ax, letter: str, x: float = -0.10, y: float = 1.05) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")


def load_final_accuracy_matrix(
    datasets: Sequence[str],
) -> Tuple[pd.DataFrame, List[str]]:
    rows = []
    present_models = []
    for model, path in MODEL_JSON.items():
        if not path.is_file():
            continue
        with open(path) as f:
            curves = json.load(f)
        present_models.append(model)
        for ds in datasets:
            acc = np.nan
            if ds in curves and curves[ds]:
                acc = float(curves[ds][-1])
            rows.append({"model": model, "dataset": ds, "accuracy": acc})
    df = pd.DataFrame(rows)
    order = [m for m in MODEL_ORDER if m in present_models]
    return df, order


def plot_panel_accuracy_bar(ax, df: pd.DataFrame, models: List[str], datasets: List[str]) -> None:
    x = np.arange(len(datasets))
    width = 0.13
    offsets = np.linspace(-(len(models) - 1) / 2, (len(models) - 1) / 2, len(models)) * width
    for mi, m in enumerate(models):
        sub = df[df["model"] == m].set_index("dataset").reindex(datasets)
        ys = sub["accuracy"].to_numpy(dtype=float) * 100
        ax.bar(x + offsets[mi], ys, width=width, color=model_color(m), label=m, edgecolor="none")
    ax.axhline(50, color="#888", ls="--", lw=0.9, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Direction accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Six models × five datasets (Top 30%)", fontsize=10)


def plot_panel_accuracy_heatmap(ax, df: pd.DataFrame, models: List[str], datasets: List[str]) -> None:
    mat = np.full((len(models), len(datasets)), np.nan)
    for i, m in enumerate(models):
        for j, ds in enumerate(datasets):
            v = df[(df["model"] == m) & (df["dataset"] == ds)]["accuracy"]
            if len(v):
                mat[i, j] = float(v.iloc[0])
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0.45, vmax=1.0)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(datasets)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]*100:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if mat[i, j] < 0.72 else "black")
    ax.set_title("Accuracy heatmap", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Accuracy")


def plot_panel_dataset_difficulty(ax, df: pd.DataFrame, models: List[str], datasets: List[str]) -> None:
    means = []
    stds = []
    for ds in datasets:
        v = df[df["dataset"] == ds]["accuracy"].dropna()
        means.append(v.mean() * 100)
        stds.append(v.std() * 100 if len(v) > 1 else 0)
    x = np.arange(len(datasets))
    ax.bar(x, means, yerr=stds, color="#8FB4DC", ecolor="#555", capsize=3, edgecolor="none")
    ax.axhline(50, color="#888", ls="--", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel("Mean acc. across models (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Dataset difficulty (cross-model mean)", fontsize=10)


def plot_panel_peak_rescue(ax, peak_csv: Path, ercc_csv: Path) -> None:
    peak = pd.read_csv(peak_csv)
    ercc = pd.read_csv(ercc_csv)
    ds = peak["dataset"].tolist()
    x = np.arange(len(ds))
    w = 0.35
    ax.bar(x - w / 2, peak["frac_wrong_single"] * 100, w, label="Single-step wrong (%)", color="#EB7E60")
    ax.bar(x + w / 2, peak["frac_fixed_among_wrong"] * 100, w, label="Peak-chain rescue (%)", color="#4C9F70")
    ax.set_xticks(x)
    ax.set_xticklabels(ds, rotation=25, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.set_title("Peak rescue (with ERCC)", fontsize=10)
    hesc_ercc = ercc[ercc["dataset"] == "hESC"]
    if len(hesc_ercc):
        r = float(hesc_ercc["frac_fixed_among_wrong"].iloc[0]) * 100
        ax.annotate(f"hESC no-ERCC rescue ≈ {r:.1f}%", xy=(0.02, 0.95), xycoords="axes fraction",
                    fontsize=7, color="#333")


def plot_panel_chip_tf_error(ax, chip_csv: Path, datasets: List[str]) -> None:
    df = pd.read_csv(chip_csv).set_index("dataset").reindex(datasets)
    x = np.arange(len(datasets))
    w = 0.35
    ax.bar(x - w / 2, df["Non-target"] * 100, w, label="Non-target", color="#AC99D2")
    ax.bar(x + w / 2, df["TF target"] * 100, w, label="CHIP TF target", color="#4EA3F1")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=25, ha="right")
    ax.set_ylabel("Direction error (%)")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("CHIP target vs non-target", fontsize=10)


def plot_panel_error_stacked_all(ax, err_csv: Path) -> None:
    df = pd.read_csv(err_csv)
    agg = df.groupby(["dataset", "error_type"], as_index=False)["count"].sum()
    totals = agg.groupby("dataset")["count"].transform("sum")
    agg["fraction"] = agg["count"] / totals
    datasets = [d for d in ALL_DATASETS if d in agg["dataset"].unique()]
    types = ["well_predicted", "amplitude_wrong", "strict_wrong"]
    x = np.arange(len(datasets))
    bottom = np.zeros(len(datasets))
    for et in types:
        vals = []
        for ds in datasets:
            sub = agg[(agg["dataset"] == ds) & (agg["error_type"] == et)]
            vals.append(float(sub["fraction"].iloc[0]) if len(sub) else 0.0)
        vals = np.array(vals)
        ax.bar(x, vals * 100, bottom=bottom * 100, label=ERROR_LABELS[et], color=ERROR_COLORS[et], width=0.7)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=25, ha="right")
    ax.set_ylabel("Fraction (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    ax.set_title("Error composition (all genes, pooled strata)", fontsize=10)


def plot_panel_static_dynamic_overlap(ax, overlap_csv: Path) -> None:
    df = pd.read_csv(overlap_csv)
    labels = df["group"].tolist()
    fracs = df["frac_in_static_tp"].to_numpy() * 100
    ns = df["n_genes"].tolist()
    colors = ["#EB7E60", "#4EA3F1"]
    bars = ax.bar(labels, fracs, color=colors, width=0.55, edgecolor="white")
    for b, n, f in zip(bars, ns, fracs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"n={n}\n{f:.1f}%",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Static STRING TP overlap (%)")
    ax.set_ylim(0, max(fracs.max() + 15, 10))
    ax.set_title("Static vs dynamic (hESC Top30%)", fontsize=10)


def plot_panel_trajectory_types(ax, traj_csv: Path) -> None:
    df = pd.read_csv(traj_csv)
    order = ["peak_middle", "sign_flip", "monotone", "mixed"]
    df = df.set_index("trajectory_type").reindex(order).reset_index()
    colors = [TRAJ_COLORS.get(t, "#999") for t in df["trajectory_type"]]
    ax.bar(df["trajectory_type"], df["n"], color=colors, edgecolor="white")
    nonlinear = df[df["trajectory_type"].isin(["peak_middle", "sign_flip"])]["n"].sum()
    total = df["n"].sum()
    ax.set_title(f"Trajectory types (hESC n={total}; nonlinear={nonlinear/total:.0%})", fontsize=10)
    ax.set_ylabel("Gene count")
    ax.tick_params(axis="x", rotation=25)


def plot_panel_shape_chain_vs_indep(ax, run_json: Path, traj_csv: Path) -> None:
    run = json.loads(run_json.read_text())
    traj = pd.read_csv(traj_csv)
    labels = ["Chain", "Independent"]
    vals = [run["mean_shape_corr_chain"], run["mean_shape_corr_indep"]]
    ax.bar(labels, vals, color=["#4EA3F1", "#FF9A3D"], width=0.5)
    ax.set_ylabel("Mean shape correlation")
    ax.set_ylim(0, max(vals) * 1.4)
    ax.set_title("Direction rescue ≠ shape gain", fontsize=10)
    ax.text(0.5, 0.92, f"peak hit chain {run['peak_hit_chain_pct']:.1f}% vs indep {run['peak_hit_indep_pct']:.1f}%",
            transform=ax.transAxes, ha="center", fontsize=7)


def plot_panel_k_sensitivity(ax, k_csv: Path) -> None:
    df = pd.read_csv(k_csv)
    for ds, color in [("hESC", "#4EA3F1"), ("mHSC-L", "#EB7E60")]:
        sub = df[df["dataset"] == ds].sort_values("n_segments")
        ax.plot(sub["n_segments"], sub["silhouette"], "o-", label=f"{ds} silhouette", color=color, lw=1.8)
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xlabel("K segments")
    ax.set_ylabel("Silhouette")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Embedding segment stability", fontsize=10)


def plot_panel_hesc_segment_acc(ax, seg_csv: Path) -> None:
    if not seg_csv.is_file():
        ax.text(0.5, 0.5, "segment summary missing", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
    df = pd.read_csv(seg_csv)
    if "segment" in df.columns and "top30_acc" in df.columns:
        ax.plot(df["segment"], df["top30_acc"] * 100, "o-", color="#4EA3F1", lw=2)
        ax.set_ylabel("Top30% acc (%)")
        ax.set_xlabel("Segment transition")
    ax.set_title("hESC chained segment accuracy", fontsize=10)


def compose_main_figure(outdir: Path, dpi: int) -> Path:
    apply_fig4_style()
    acc_df, models = load_final_accuracy_matrix(MAIN_DATASETS)

    peak_csv = SCRIPT_DIR / "error_biology/peak_method_six_datasets_metrics.csv"
    ercc_csv = SCRIPT_DIR / "error_biology/peak_method_six_datasets_metrics_no_ercc.csv"
    chip_csv = SCRIPT_DIR / "error_biology/chip_tf_target_direction_error_summary.csv"
    err_csv = SCRIPT_DIR / "error_biology/all_datasets_error_fractions.csv"
    traj_csv = SCRIPT_DIR / "error_biology/multistep_pt/hESC/chained_preds/figures/mechanistic_interpret/summary_by_trajectory_type.csv"
    run_json = SCRIPT_DIR / "error_biology/multistep_pt/hESC/chained_preds/figures/mechanistic_interpret/run_summary.json"
    k_csv = SCRIPT_DIR / "error_biology/embedding_k_sensitivity/embedding_k_sensitivity_summary.csv"
    seg_csv = SCRIPT_DIR / "error_biology/multistep_pt/hESC/chained_preds/hESC_chained_segment_summary.csv"
    overlap_csv = FIG3_OVERLAP

    fig = plt.figure(figsize=(16, 14), facecolor="white")
    gs = GridSpec(4, 3, figure=fig, hspace=0.42, wspace=0.32,
                  height_ratios=[1, 1, 1, 0.95])

    # Act I — benchmark
    ax_a = fig.add_subplot(gs[0, 0])
    plot_panel_accuracy_bar(ax_a, acc_df, models, MAIN_DATASETS)
    _panel_label(ax_a, "A")

    ax_b = fig.add_subplot(gs[0, 1])
    plot_panel_accuracy_heatmap(ax_b, acc_df, models, MAIN_DATASETS)
    _panel_label(ax_b, "B")

    ax_c = fig.add_subplot(gs[0, 2])
    plot_panel_dataset_difficulty(ax_c, acc_df, models, MAIN_DATASETS)
    _panel_label(ax_c, "C")

    # Act II — errors & rescue
    ax_d = fig.add_subplot(gs[1, 0])
    plot_panel_peak_rescue(ax_d, peak_csv, ercc_csv)
    _panel_label(ax_d, "D")

    ax_e = fig.add_subplot(gs[1, 1])
    plot_panel_chip_tf_error(ax_e, chip_csv, ALL_DATASETS)
    _panel_label(ax_e, "E")

    ax_f = fig.add_subplot(gs[1, 2])
    plot_panel_error_stacked_all(ax_f, err_csv)
    _panel_label(ax_f, "F")

    # Act III — mechanism & complementarity
    ax_g = fig.add_subplot(gs[2, 0])
    if overlap_csv.is_file():
        plot_panel_static_dynamic_overlap(ax_g, overlap_csv)
    _panel_label(ax_g, "G")

    ax_h = fig.add_subplot(gs[2, 1])
    if traj_csv.is_file():
        plot_panel_trajectory_types(ax_h, traj_csv)
    _panel_label(ax_h, "H")

    ax_i = fig.add_subplot(gs[2, 2])
    if run_json.is_file() and traj_csv.is_file():
        plot_panel_shape_chain_vs_indep(ax_i, run_json, traj_csv)
    _panel_label(ax_i, "I")

    ax_j = fig.add_subplot(gs[3, 0])
    if k_csv.is_file():
        plot_panel_k_sensitivity(ax_j, k_csv)
    _panel_label(ax_j, "J")

    ax_k = fig.add_subplot(gs[3, 1])
    plot_panel_hesc_segment_acc(ax_k, seg_csv)
    _panel_label(ax_k, "K")

    ax_l = fig.add_subplot(gs[3, 2])
    ax_l.axis("off")
    story = (
        "Act I (A–C): No global dynamic champion; dataset difficulty varies.\n"
        "Act II (D–F): Peak rescue is hESC-specific & ERCC-driven; CHIP targets differ.\n"
        "Act III (G–L): Static–dynamic overlap 2.98%; nonlinear trajectories dominate;\n"
        "chain improves local direction but not shape; segment embedding sensitivity."
    )
    ax_l.text(0.02, 0.98, story, va="top", ha="left", fontsize=9, family="monospace",
              bbox=dict(boxstyle="round", facecolor="#f7f7f7", edgecolor="#ccc"))
    _panel_label(ax_l, "L", x=-0.02, y=1.02)

    fig.suptitle("Figure 4 — Dynamic direction benchmark (composed)", fontsize=14, fontweight="bold", y=0.995)
    out = outdir / "Fig4_main_act123"
    fig.savefig(f"{out}.pdf", bbox_inches="tight", dpi=dpi)
    fig.savefig(f"{out}.png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return out


def compose_image_gallery(
    title: str,
    images: List[Tuple[str, Path]],
    out_path: Path,
    ncols: int,
    dpi: int,
    figsize: Tuple[float, float],
) -> Optional[Path]:
    valid = [(lbl, p) for lbl, p in images if p.is_file()]
    if not valid:
        print(f"[skip] No images for {out_path.name}")
        return None
    n = len(valid)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor="white")
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    for ax, (lbl, path) in zip(axes, valid):
        try:
            img = mpimg.imread(path)
        except Exception:
            ax.text(0.5, 0.5, f"Cannot load\n{path.name}", ha="center", va="center", fontsize=7)
            ax.set_title(lbl, fontsize=8)
            ax.axis("off")
            continue
        ax.imshow(img)
        ax.set_title(lbl, fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight", dpi=dpi)
    fig.savefig(f"{out_path}.png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return out_path


def compose_supp_galleries(outdir: Path, dpi: int) -> None:
    eb = SCRIPT_DIR / "error_biology"
    ms = eb / "multistep_pt/hESC/chained_preds/figures"

    s1 = [
        ("Accuracy bar", SCRIPT_DIR / "accuracy/figure_accuracy_all_models_bar_npg.png"),
        ("Convergence 6ds", SCRIPT_DIR / "convergence_all_6datasets.pdf"),
    ]
    if (SCRIPT_DIR / "accuracy/top_percentage_hESC_hHep_mHSC-E_mHSC-GM_mHSC-L_beautiful.pdf").is_file():
        s1.append(("Top% sweep", SCRIPT_DIR / "accuracy/top_percentage_hESC_hHep_mHSC-E_mHSC-GM_mHSC-L_beautiful.pdf"))
    compose_image_gallery("Supp S1 — Benchmark extensions", s1, outdir / "Fig4_supp_S1_benchmark", 2, dpi, (12, 10))

    s2 = []
    for ds in ALL_DATASETS:
        s2.append((f"{ds} error strat", eb / f"{ds}_error_strat_stacked.png"))
        s2.append((f"{ds} dir pie", eb / f"{ds}_strict_wrong_dir_pie.png"))
    s2.extend([
        ("CHIP TF 6ds", eb / "all_datasets_chip_tf_target_error_stacked_no_mDC.png"),
        ("Direction heatmap", eb / "heatmap_direction_error.png"),
        ("Amplitude heatmap", eb / "heatmap_amplitude_error.png"),
        ("Peak rescue", eb / "peak_method_six_datasets_stacked.png"),
    ])
    compose_image_gallery("Supp S2 — Error biology (6 datasets)", s2, outdir / "Fig4_supp_S2_error_biology", 4, dpi, (18, 28))

    s3 = [
        ("Segment acc", ms / "01_segment_accuracy.png"),
        ("Delta scatter", ms / "02_delta_scatter_by_segment.png"),
        ("Problem overview", ms / "04_problem_overview.png"),
        ("ERCC rescue", ms / "06_ERCC_rescue.png"),
        ("Spotlight genes", ms / "07_spotlight_genes.png"),
        ("Acc top30 vs all", ms / "08_acc_top30_vs_allgenes.png"),
        ("Error persistence", ms / "09_error_persistence.png"),
        ("Flip breakdown", ms / "11_flip_breakdown.png"),
        ("CALB1 mechanism", ms / "mechanistic_interpret/02_mechanism_CALB1.png"),
        ("HDGF mechanism", ms / "mechanistic_interpret/02_mechanism_HDGF.png"),
        ("SOX2 case", ms / "mechanistic_interpret/03_case_SOX2.png") if (ms / "mechanistic_interpret/03_case_SOX2.png").is_file() else ("Case 2x2", ms / "mechanistic_interpret/01_case_trajectories_2x2.png"),
        ("Nonlinear explain", ms / "nonlinear_explain/nonlinear_summary.png") if (ms / "nonlinear_explain/nonlinear_summary.png").is_file() else ("Aggregate traj", ms / "13_aggregate_trajectory_lines.png"),
    ]
    compose_image_gallery("Supp S3 — hESC multistep mechanism", s3, outdir / "Fig4_supp_S3_hESC_mechanism", 3, dpi, (16, 22))

    emb = eb / "embedding_k_sensitivity"
    s4 = [
        ("K sensitivity summary", emb / "embedding_k_sensitivity.png"),
        ("hESC K3 PCA", emb / "hESC/K3/scgpt_embedding_pca.png"),
        ("hESC K5 PCA", emb / "hESC/K5/scgpt_embedding_pca.png"),
        ("hESC K7 PCA", emb / "hESC/K7/scgpt_embedding_pca.png"),
        ("Static–dynamic overlap fig3", FIG3_OVERLAP.parent / "hESC_static_dynamic_overlap_bar.png") if (FIG3_OVERLAP.parent / "hESC_static_dynamic_overlap_bar.png").is_file() else ("TF7 topology", SCRIPT_DIR.parent / "fig3/tf_static/output/hESC/benchmark_extended_STRING/hESC_FigTF7_failure_topology.png"),
        ("Reghead smoke", SCRIPT_DIR / "reghead_mvp/smoke/reghead_summary.csv"),
    ]
    # filter to images only for gallery
    s4_img = [(a, b) for a, b in s4 if str(b).endswith((".png", ".pdf", ".jpg"))]
    compose_image_gallery("Supp S4 — Methods & sensitivity", s4_img, outdir / "Fig4_supp_S4_methods", 3, dpi, (14, 12))


def write_storyboard(outdir: Path, main_path: Path) -> Path:
    md = f"""# Fig4 组图故事线 & 文件索引

> 由 `compose_fig4_figure.py` 生成。主图：**{main_path.name}.pdf**

---

## 三幕剧结构

### 第一幕 — Benchmark：六模型 × 数据集难度

| Panel | 内容 | 故事要点 |
|-------|------|----------|
| **A** | 六模型 × 五数据集 direction accuracy 柱状图 | 无全局冠军；排序随 dataset 重排 |
| **B** | 模型×数据集 accuracy 热图 | scFoundation→hESC/hHep；scGPT→mHSC-E |
| **C** | 跨模型平均 accuracy（dataset difficulty） | mHSC-L / scCello-hESC 是 stress test |

**已有单图**：`accuracy/figure_accuracy_all_models_bar_npg.pdf`

---

### 第二幕 — 误差生物学：错在哪、能否 rescue

| Panel | 内容 | 故事要点 |
|-------|------|----------|
| **D** | 六数据集单步错率 + peak-chain rescue | hESC 32.5% rescue；剔除 ERCC 后 ≈3.2% |
| **E** | CHIP TF 靶 vs 非靶 direction error | 非靶错误率系统性更高（多数 dataset） |
| **F** | 六数据集误差三分（well / amplitude / direction） | 方向错 ≠ 幅度错；需分层报告 |

**已有单图**：
- `error_biology/peak_method_six_datasets_stacked.png`
- `error_biology/all_datasets_chip_tf_target_error_stacked_no_mDC.png`
- `error_biology/{{dataset}}_error_strat_stacked.png`（6 数据集）

---

### 第三幕 — 机制：静动互补、非线性、形状 vs 方向

| Panel | 内容 | 故事要点 |
|-------|------|----------|
| **G** | 静态 STRING TP ∩ 动态 CHIP 靶（hESC） | **仅 2.98% overlap** — 与 fig3 互补 |
| **H** | 轨迹分型（peak_middle / sign_flip / …） | 非线性占 ~60% |
| **I** | 链式 vs 独立分段 shape correlation | 方向 rescue ≠ 形状改善 |
| **J** | embedding K 分段 silhouette（hESC vs mHSC-L） | mHSC-L 分段不稳定 |
| **K** | hESC 链式分段 direction accuracy | S0→S1 最弱 |
| **L** | 故事摘要文本框 | — |

**已有单图**：
- `multistep_pt/hESC/chained_preds/figures/mechanistic_interpret/`
- `embedding_k_sensitivity/hESC/K{{3,5,7}}/scgpt_embedding_pca.png`

---

## Supplement 组图（图越多越好）

| 文件 | 内容 |
|------|------|
| `Fig4_supp_S1_benchmark.pdf` | accuracy / convergence / top% sweep |
| `Fig4_supp_S2_error_biology.pdf` | 6 数据集 error strat + pie + heatmap + peak |
| `Fig4_supp_S3_hESC_mechanism.pdf` | multistep / ERCC / 案例基因 / 非线性 |
| `Fig4_supp_S4_methods.pdf` | K 敏感性 PCA / overlap / fig3 勾连 |

---

## 额外已有资产（可引 Supp）

| 主题 | 路径 |
|------|------|
| 收敛曲线 | `convergence/convergence_models_{{dataset}}.pdf` |
| GO 富集 | `error_biology/{{dataset}}_go_enrichment_strict_wrong.png` |
| Geneformer 多段 | `multistep_pt/hESC/geneformer_preds/` |
| PT quantile 敏感性 | `canshu/results_pt_quantile_sensitivity/` |
| Swap 敏感性 | `canshu/results_geneformer_swap_sensitivity/` |
| RegVelo-head 阴性 | `reghead_mvp/smoke/reghead_summary.csv` |

---

## 复现命令

```bash
cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig4
python3 compose_fig4_figure.py --outdir output/compose --dpi 300
```

---

## 正文 Figure legend（草稿）

**Figure 4. Dynamic gene-direction benchmark across foundation models.**
**(A–C)** Direction accuracy on top 30% dynamic genes (five datasets; mDC excluded from main panels). No single model wins across all datasets.
**(D–F)** Error decomposition and peak-transition rescue; hESC rescue is largely ERCC-driven after control removal.
**(G–L)** Complementarity with static GRN inference (2.98% overlap), nonlinear trajectory dominance, and dissociation between direction accuracy and trajectory shape under chained prediction.
See Supp. Fig. S4-1–S4-4 for per-dataset panels and case studies.
"""
    path = outdir / "Fig4_storyboard.md"
    path.write_text(md, encoding="utf-8")
    return path


def parse_args():
    p = argparse.ArgumentParser(description="Compose Fig4 multi-panel figures + storyboard.")
    p.add_argument("--outdir", default=str(SCRIPT_DIR / "output/compose"), type=str)
    p.add_argument("--dpi", default=300, type=int)
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    main_path = compose_main_figure(outdir, args.dpi)
    compose_supp_galleries(outdir, args.dpi)
    sb = write_storyboard(outdir, main_path)
    print(f"[OK] Main figure: {main_path}.pdf / .png")
    print(f"[OK] Storyboard:  {sb}")
    print(f"[OK] Supp galleries in {outdir}")


if __name__ == "__main__":
    main()
