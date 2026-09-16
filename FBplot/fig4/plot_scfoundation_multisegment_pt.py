#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scFoundation 多段伪时间结果可视化（读取 scfoundation_preds/ 下 CSV，无需 GPU）。

示例：
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig4
  python3 run_scfoundation_multisegment_pt.py --dataset hESC
  python3 plot_scfoundation_multisegment_pt.py --dataset hESC
  python3 plot_scfoundation_multisegment_pt.py --dataset hESC --compare-geneformer
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from fig4_palette import apply_fig4_style, model_color
except ImportError:

    def apply_fig4_style() -> None:
        plt.rcParams.update({"font.size": 11})

    def model_color(name: str, default: str = "#666") -> str:
        return default


SCRIPT_DIR = Path(__file__).resolve().parent
MULTISTEP_ROOT = SCRIPT_DIR / "error_biology" / "multistep_pt"
COLOR_SCF = model_color("scFoundation")
COLOR_GF = COLOR_SCF  # reuse plot helpers
COLOR_SCGPT = model_color("scGPT")
COLOR_TRUE = model_color("scPrint")
COLOR_OK = "#43A047"
COLOR_BAD = "#E53935"
COLOR_CUM = "#7E57C2"
MARKER_KW = dict(markersize=9, markeredgewidth=1.2, markeredgecolor="white")


def discover_transitions(pred_dir: Path, dataset: str) -> List[str]:
    pat = re.compile(rf"^{re.escape(dataset)}_(delta_t\d+_t\d+)_gene_result\.csv$")
    trans = []
    for p in sorted(pred_dir.glob(f"{dataset}_delta_t*_t*_gene_result.csv")):
        m = pat.match(p.name)
        if m:
            trans.append(m.group(1))
    return trans


def load_tables(pred_dir: Path, dataset: str, transitions: List[str]) -> Dict[str, pd.DataFrame]:
    return {
        t: pd.read_csv(pred_dir / f"{dataset}_{t}_gene_result.csv").set_index("gene")
        for t in transitions
    }


def seg_labels(transitions: List[str]) -> List[str]:
    labels = []
    for t in transitions:
        m = re.match(r"delta_t(\d+)_t(\d+)", t)
        if m:
            labels.append(f"S{m.group(1)}→S{m.group(2)}")
        else:
            labels.append(t)
    return labels


def pt_node_labels(n_trans: int) -> List[str]:
    return [f"S{k}" for k in range(n_trans + 1)]


def plot_accuracy_lines(
    out: Path,
    seg_sum: pd.DataFrame,
    labels: List[str],
    dataset: str,
) -> None:
    """沿伪时间 transition 的方向准确率折线（主图，更直观）。"""
    x = np.arange(len(labels))
    local_top = (seg_sum["acc_top_local"] * 100).to_numpy()
    local_all = (seg_sum["acc_all_local"] * 100).to_numpy()
    cum_top = (
        (seg_sum["acc_top_cumulative"] * 100).to_numpy()
        if "acc_top_cumulative" in seg_sum.columns
        else None
    )
    is_chained = seg_sum.get("mode", pd.Series(["chained"])).iloc[0] == "chained"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        x, local_top, "o-", color=COLOR_SCF, linewidth=2.8, label="Top 30% · local velocity",
        **MARKER_KW,
    )
    ax.plot(
        x, local_all, "s--", color="#F9A825", linewidth=2, alpha=0.9, label="All genes · local velocity",
        markersize=7,
    )
    if cum_top is not None and is_chained:
        ax.plot(
            x, cum_top, "^-", color=COLOR_CUM, linewidth=2.4, label="Top 30% · cumulative vs S0",
            markersize=8, markeredgewidth=1, markeredgecolor="white",
        )
    ax.fill_between(x, local_top, alpha=0.12, color=COLOR_SCF)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(40, 100)
    ax.set_ylabel("Direction accuracy (%)")
    ax.set_xlabel("Pseudotime transition (equal-frequency bins)")
    ax.set_title(
        f"{dataset} — scFoundation direction accuracy along pseudotime",
        fontweight="600",
    )
    ax.axhline(50, color="#B0BEC5", ls=":", lw=1, zorder=0)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.35)
    for i, v in enumerate(local_top):
        ax.annotate(f"{v:.0f}%", (x[i], v), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9, color=COLOR_SCF, fontweight="600")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_true_velocity_and_accuracy(
    out: Path,
    tables: Dict[str, pd.DataFrame],
    seg_sum: pd.DataFrame,
    transitions: List[str],
    labels: List[str],
    dataset: str,
) -> None:
    """双轴折线：真值速度强度（|Δ_local| 中位数）+ 预测方向准确率。"""
    x = np.arange(len(transitions))
    med_abs = [tables[t]["delta_true_local"].abs().median() for t in transitions]
    acc = (seg_sum["acc_top_local"] * 100).to_numpy()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ln1 = ax1.plot(
        x, med_abs, "o-", color=COLOR_TRUE, linewidth=2.6, label="Median |true Δ| (velocity)",
        **MARKER_KW,
    )
    ln2 = ax2.plot(
        x, acc, "s-", color=COLOR_SCF, linewidth=2.6, label="Top 30% direction accuracy",
        markersize=8, markeredgewidth=1, markeredgecolor="white",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("Median |Δ| (log1p expr.)", color=COLOR_TRUE)
    ax2.set_ylabel("Accuracy (%)", color=COLOR_SCF)
    ax2.set_ylim(40, 100)
    ax1.set_xlabel("Pseudotime transition")
    ax1.set_title(
        f"{dataset} — biological velocity vs scFoundation accuracy",
        fontweight="600",
    )
    ax1.grid(axis="y", alpha=0.3)
    lines = ln1 + ln2
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_observed_trajectory_lines(
    out: Path,
    tables: Dict[str, pd.DataFrame],
    transitions: List[str],
    dataset: str,
    n_genes: int = 6,
) -> None:
    """Top 动态基因：观测表达沿 S0→S1→… 的折线（归一化，展示「速度」形状）。"""
    t0 = tables[transitions[0]]
    n_nodes = len(transitions) + 1
    # 每基因在 S0..Sn 的观测均值
    genes = t0.index.astype(str)
    peak_scores = np.zeros(len(genes))
    for t in transitions:
        df = tables[t]
        peak_scores = np.maximum(peak_scores, df["delta_true_local"].abs().to_numpy())

    top_idx = np.argsort(peak_scores)[::-1][:n_genes]
    top_genes = [genes[i] for i in top_idx]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), squeeze=False)
    nodes = np.arange(n_nodes)

    for ax, gene in zip(axes.flat, top_genes):
        y = [float(tables[transitions[0]].loc[gene, "true_baseline_mean"])]
        for t in transitions:
            y.append(float(tables[t].loc[gene, "true_late_mean"]))
        y = np.array(y, dtype=float)
        yn = (y - y.min()) / (y.max() - y.min() + 1e-9)

        ax.plot(nodes, yn, "o-", color=COLOR_TRUE, linewidth=2.5, markersize=8)
        ax.fill_between(nodes, yn, alpha=0.15, color=COLOR_TRUE)
        # 标出 |Δ| 最大段
        d_locals = [
            abs(float(tables[t].loc[gene, "delta_true_local"])) for t in transitions
        ]
        peak_seg = int(np.argmax(d_locals))
        ax.axvspan(peak_seg, peak_seg + 1, alpha=0.2, color=model_color("scFoundation"))
        ax.set_xticks(nodes)
        ax.set_xticklabels(pt_node_labels(len(transitions)), fontsize=9)
        ax.set_ylabel("Scaled expr.", fontsize=9)
        ax.set_title(gene, fontweight="600", fontsize=11)
        ax.grid(alpha=0.25)

    fig.suptitle(
        f"{dataset} — observed expression along pseudotime (gold band = peak |velocity|)",
        fontsize=13, fontweight="700", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_per_gene_accuracy_trajectory(
    out: Path,
    wide: pd.DataFrame,
    transitions: List[str],
    labels: List[str],
    dataset: str,
) -> None:
    """聚合折线：随伪时间推进，累计有多少比例基因「至今全对」。"""
    cols = [f"{t}_dir_correct_local" for t in transitions]
    mat = wide[cols].astype(float).values
    n = len(wide)
    x = np.arange(len(transitions))

    cum_ok_frac = []
    for j in range(len(transitions)):
        cum_ok_frac.append((mat[:, : j + 1].all(axis=1)).mean() * 100)
    never_ok = (mat.sum(axis=1) == 0).mean() * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, cum_ok_frac, "o-", color=COLOR_OK, linewidth=2.8, label="Genes correct in ALL segments so far",
            **MARKER_KW)
    err_rate = [(mat[:, j] == 0).mean() * 100 for j in range(len(transitions))]
    ax.plot(x, err_rate, "s-", color=COLOR_BAD, linewidth=2.2, label="% genes wrong in this segment",
            markersize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Gene fraction (%)")
    ax.set_xlabel("Pseudotime transition")
    ax.set_title(
        f"{dataset} — error accumulation along chained scFoundation blocks",
        fontweight="600",
    )
    ax.legend(loc="center right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.text(0.02, 0.04, f"Never correct in any block: {never_ok:.1f}% of genes",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#FFEBEE"))
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_compare_scgpt_lines(
    out: Path,
    gf_sum: pd.DataFrame,
    scgpt_sum: pd.DataFrame,
    dataset: str,
) -> None:
    """Geneformer vs scGPT 沿伪时间准确率折线（按 seg_from 对齐）。"""
    gf = gf_sum.copy()
    sc = scgpt_sum.copy()
    sc_cols = ["transition", "acc_top_percent"]
    if "seg_from" in sc.columns and "seg_from" in gf.columns:
        sc_cols.append("seg_from")
        on_cols = ["transition", "seg_from"]
    else:
        on_cols = ["transition"]
    merged = gf.merge(sc[sc_cols], on=on_cols, how="left")
    if merged.empty:
        # fallback: 按顺序对齐较短长度
        n = min(len(gf), len(sc))
        merged = pd.DataFrame({
            "label": seg_labels(gf["transition"].tolist()[:n]),
            "acc_gf": (gf["acc_top_local"].iloc[:n] * 100).tolist(),
            "acc_sc": (sc["acc_top_percent"].iloc[:n] * 100).tolist(),
        })
        labels = merged["label"].tolist()
        x = np.arange(n)
    else:
        labels = [f"S{int(r.seg_from)}→S{int(r.seg_to)}" for r in merged.itertuples()]
        x = np.arange(len(merged))
        merged["acc_gf"] = merged["acc_top_local"] * 100
        merged["acc_sc"] = merged["acc_top_percent"] * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, merged["acc_gf"], "o-", color=COLOR_GF, linewidth=2.8,
            label="Geneformer · local velocity", **MARKER_KW)
    sc_vals = merged["acc_sc"].to_numpy()
    mask = np.isfinite(sc_vals)
    if mask.any():
        ax.plot(
            x[mask], sc_vals[mask], "s-", color=COLOR_SCGPT, linewidth=2.8,
            label="scGPT chained · vs S0", markersize=8, markeredgewidth=1, markeredgecolor="white",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(40, 100)
    ax.set_ylabel("Top 30% direction accuracy (%)")
    ax.set_xlabel("Pseudotime transition")
    ax.set_title(f"{dataset} — Geneformer vs scGPT along pseudotime", fontweight="600")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_velocity_accuracy(
    out: Path,
    seg_sum: pd.DataFrame,
    transitions: List[str],
    labels: List[str],
) -> None:
    """分段局部速度（local）方向准确率 — 主图。"""
    local_top = (seg_sum["acc_top_local"] * 100).tolist()
    local_all = (seg_sum["acc_all_local"] * 100).tolist()
    if "acc_top_cumulative" in seg_sum.columns:
        cum_top = (seg_sum["acc_top_cumulative"] * 100).tolist()
    else:
        cum_top = None

    x = np.arange(len(transitions))
    w = 0.28
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - w, local_top, w, label="Top 30% dynamic (local velocity)", color=COLOR_SCF, edgecolor="white")
    ax.bar(x, local_all, w, label="All genes (local velocity)", color="#FFE082", edgecolor="#F9A825")
    if cum_top is not None and seg_sum.get("mode", pd.Series(["chained"])).iloc[0] == "chained":
        ax.bar(x + w, cum_top, w, label="Top 30% (cumulative vs S0)", color="#B39DDB", edgecolor="#7E57C2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Direction accuracy (%)")
    ax.set_title("scFoundation — per-segment velocity direction accuracy", fontweight="600")
    ax.legend(loc="lower right", fontsize=9)
    for i, v in enumerate(local_top):
        ax.text(i - w, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_true_vs_pred_velocity(
    out: Path,
    tables: Dict[str, pd.DataFrame],
    transitions: List[str],
    labels: List[str],
) -> None:
    n = len(transitions)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.8 * ncol, 3.6 * nrow), squeeze=False)
    for idx, (t, lab) in enumerate(zip(transitions, labels)):
        ax = axes.flat[idx]
        df = tables[t]
        x = df["delta_true_local"].astype(float)
        y = df["delta_pred_local"].astype(float)
        ok = df["dir_correct_local"].astype(int) > 0
        ax.scatter(x[~ok], y[~ok], c=COLOR_BAD, s=6, alpha=0.35, linewidths=0)
        ax.scatter(x[ok], y[ok], c=COLOR_OK, s=6, alpha=0.35, linewidths=0)
        lim = max(float(np.abs(x).max()), float(np.abs(y).max()), 1e-3)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.7, alpha=0.35)
        acc = float(ok.mean()) * 100
        ax.set_title(f"{lab}  acc={acc:.0f}%", fontsize=10, fontweight="600")
        ax.set_xlabel("True Δ (velocity)")
        ax.set_ylabel("Pred Δ (velocity)")
    for j in range(len(transitions), nrow * ncol):
        axes.flat[j].set_visible(False)
    fig.suptitle("scFoundation local velocity: true vs predicted Δ", fontsize=12, fontweight="700", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_heatmap_wide(out: Path, wide: pd.DataFrame, transitions: List[str], labels: List[str]) -> None:
    cols = [f"{t}_dir_correct_local" for t in transitions]
    err = wide[cols].eq(0).sum(axis=1)
    show = wide.loc[err.nlargest(min(30, len(wide))).index]
    mat = show[cols].astype(float).values
    genes = show["gene"].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(6.5, 9))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(genes)))
    ax.set_yticklabels(genes, fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, "✓" if mat[i, j] > 0.5 else "✗", ha="center", va="center", fontsize=7,
                    color="white" if mat[i, j] < 0.5 else "#1B5E20", fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax.set_title("Genes with most segment errors (local velocity)", fontweight="600")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def geneformer_true_profile(
    gene: str,
    tables: Dict[str, pd.DataFrame],
    transitions: List[str],
) -> np.ndarray:
    """观测表达沿 S0…Sn（log1p 段均值）。"""
    t0 = tables[transitions[0]]
    if gene not in t0.index:
        return np.full(len(transitions) + 1, np.nan)
    y = [float(t0.loc[gene, "true_baseline_mean"])]
    for t in transitions:
        y.append(float(tables[t].loc[gene, "true_late_mean"]))
    return np.array(y, dtype=float)


def pick_best_gene(
    wide: pd.DataFrame,
    tables: Dict[str, pd.DataFrame],
    transitions: List[str],
) -> str:
    """六段 local 方向全对 + peak |Δ_true| 最大（动态基因里表现最好）。"""
    dir_cols = [f"{t}_dir_correct_local" for t in transitions]
    ok = wide[dir_cols].sum(axis=1) == len(transitions)
    cands = wide.loc[ok, "gene"].astype(str).tolist()
    if not cands:
        raise ValueError("No gene with all segments locally correct")
    best_g, best_peak = cands[0], -1.0
    for g in cands:
        peak = max(
            abs(float(tables[t].loc[g, "delta_true_local"]))
            for t in transitions
            if g in tables[t].index
        )
        if peak > best_peak:
            best_peak, best_g = peak, g
    return best_g


def plot_single_gene_example(
    out: Path,
    gene: str,
    tables: Dict[str, pd.DataFrame],
    transitions: List[str],
    labels: List[str],
    dataset: str,
) -> None:
    """
    单基因教学图：上=伪时间段示意；中=该基因真值表达折线；下=每段 local Δ_true 与方向预测。
    对照 04_velocity_vs_accuracy_line：汇总图每个点是全基因统计，这里是其中一个具体基因。
    """
    n_seg = len(transitions)
    nodes = pt_node_labels(n_seg)
    x_nodes = np.arange(n_seg + 1)
    true_y = geneformer_true_profile(gene, tables, transitions)
    if not np.all(np.isfinite(true_y)):
        raise ValueError(f"Gene {gene} missing from segment tables")

    d_true = []
    d_pred = []
    dirs_ok = []
    for t in transitions:
        r = tables[t].loc[gene]
        d_true.append(float(r["delta_true_local"]))
        d_pred.append(float(r["delta_pred_local"]))
        dirs_ok.append(int(r["dir_correct_local"]) > 0)

    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.55, 1.35, 1.0], hspace=0.42)

    # --- A: schematic ---
    ax0 = fig.add_subplot(gs[0])
    ax0.set_xlim(-0.15, n_seg + 0.15)
    ax0.set_ylim(0, 1)
    ax0.axis("off")
    ax0.set_title(
        f"A  One gene across pseudotime blocks — {gene}  (6/6 segments ✓)",
        fontweight="600",
        loc="left",
        fontsize=12,
    )
    for i, lab in enumerate(labels):
        ok = dirs_ok[i]
        fc = "#E8F5E9" if ok else "#FFEBEE"
        ec = COLOR_OK if ok else COLOR_BAD
        ax0.text(
            i + 0.5, 0.72, lab, ha="center", fontsize=9, fontweight="600",
            bbox=dict(boxstyle="round", facecolor=fc, edgecolor=ec, linewidth=1.2),
        )
        mark = "✓" if ok else "✗"
        ax0.text(i + 0.5, 0.38, mark, ha="center", fontsize=14, fontweight="bold", color=ec)
        if i < n_seg:
            ax0.annotate(
                "", xy=(i + 0.92, 0.55), xytext=(i + 0.08, 0.55),
                arrowprops=dict(arrowstyle="-|>", color="#78909C", lw=1.5),
            )
    for i, nd in enumerate(nodes):
        ax0.text(i, 0.08, nd, ha="center", fontsize=9, color="#455A64")

    # --- B: true expression trajectory (what one gene contributes to biology) ---
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(
        x_nodes, true_y, "o-", color=COLOR_TRUE, linewidth=3, markersize=11,
        markeredgewidth=1.2, markeredgecolor="white",
        label="Observed mean expr. (log1p)",
    )
    ax1.fill_between(x_nodes, true_y, alpha=0.12, color=COLOR_TRUE)
    peak_seg = int(np.argmax(np.abs(d_true)))
    ax1.axvspan(peak_seg, peak_seg + 1, alpha=0.22, color=model_color("scFoundation"), label="Peak |velocity| segment")
    for i, t in enumerate(transitions):
        dt = d_true[i]
        ax1.annotate(
            f"Δ={dt:+.2f}",
            (i + 1, true_y[i + 1]),
            textcoords="offset points",
            xytext=(0, 10 if dt >= 0 else -18),
            ha="center",
            fontsize=8,
            color=COLOR_OK if dirs_ok[i] else COLOR_BAD,
        )
    ax1.set_xticks(x_nodes)
    ax1.set_xticklabels(nodes)
    ax1.set_ylabel("Mean expression (log1p)")
    ax1.set_title(
        "B  This gene's expression at each pseudotime node (not a population median)",
        fontweight="600",
        loc="left",
        fontsize=11,
    )
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # --- C: per-segment local velocity (true Δ) + pred direction ---
    ax2 = fig.add_subplot(gs[2])
    x_seg = np.arange(n_seg)
    w = 0.55
    colors = [COLOR_OK if ok else COLOR_BAD for ok in dirs_ok]
    ax2.bar(x_seg, d_true, width=w, color=colors, edgecolor="white", alpha=0.85, label="True local Δ")
    ax2.bar(
        x_seg + 0.08,
        d_pred,
        width=w * 0.55,
        color=COLOR_SCF,
        edgecolor="white",
        alpha=0.55,
        label="scFoundation pred Δ_local",
    )
    ax2.axhline(0, color="#90A4AE", lw=0.8)
    for i, (dt, dp, ok, lab) in enumerate(zip(d_true, d_pred, dirs_ok, labels)):
        pred_dir = "Up" if dp > 0 else ("Down" if dp < 0 else "flat")
        true_dir = "Up" if dt > 0 else ("Down" if dt < 0 else "flat")
        y_top = max(abs(dt), 0.05) * 1.15 * (1 if dt >= 0 else -1)
        ax2.text(
            i, y_top,
            f"pred {pred_dir}\n(true {true_dir}) {'✓' if ok else '✗'}",
            ha="center",
            va="bottom" if dt >= 0 else "top",
            fontsize=8,
            fontweight="600",
            color=COLOR_OK if ok else COLOR_BAD,
        )
    ax2.set_xticks(x_seg)
    ax2.set_xticklabels(labels, rotation=18, ha="right")
    ax2.set_ylabel("Local velocity Δ (log1p)")
    ax2.set_title(
        "C  Per-block true vs scFoundation predicted Δ_local (continuous expression space)",
        fontweight="600",
        loc="left",
        fontsize=11,
    )
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"{dataset} — example gene {gene}: scFoundation segment dynamics",
        fontsize=13,
        fontweight="700",
        y=0.98,
    )
    fig.text(
        0.5, 0.01,
        "Fig 04 purple point at S1→S2 = median |Δ_true| over ALL genes; "
        "yellow point = Top30% direction accuracy. scFoundation outputs continuous Δ_local.",
        ha="center",
        fontsize=9,
        color="#546E7A",
    )
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_compare_geneformer(
    out: Path,
    scf_sum: pd.DataFrame,
    gf_sum: pd.DataFrame,
    labels: List[str],
) -> None:
    """与 Geneformer 分段准确率对比（若存在 geneformer_preds）。"""
    n = len(labels)
    x = np.arange(n)
    w = 0.35
    scf_acc = (scf_sum["acc_top_local"] * 100).tolist()
    gf_acc = []
    for t in scf_sum["transition"]:
        row = gf_sum[gf_sum["transition"] == t]
        gf_acc.append(float(row.iloc[0]["acc_top_local"]) * 100 if len(row) else np.nan)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - w / 2, scf_acc, w, label="scFoundation (local velocity)", color=COLOR_SCF)
    ax.bar(x + w / 2, gf_acc, w, label="Geneformer (local velocity)", color=model_color("Geneformer"))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Top 30% direction accuracy (%)")
    ax.set_title("scFoundation vs Geneformer — per segment", fontweight="600")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_compare_geneformer_lines(
    out: Path,
    scf_sum: pd.DataFrame,
    gf_sum: pd.DataFrame,
    dataset: str,
) -> None:
    """scFoundation vs Geneformer 沿伪时间准确率折线。"""
    merged = scf_sum.merge(
        gf_sum[["transition", "acc_top_local"]].rename(columns={"acc_top_local": "acc_gf"}),
        on="transition",
        how="left",
    )
    labels = seg_labels(merged["transition"].tolist())
    x = np.arange(len(merged))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        x, merged["acc_top_local"] * 100, "o-", color=COLOR_SCF, linewidth=2.8,
        label="scFoundation · local velocity", **MARKER_KW,
    )
    gf_vals = merged["acc_gf"].to_numpy() * 100
    mask = np.isfinite(gf_vals)
    if mask.any():
        ax.plot(
            x[mask], gf_vals[mask], "s-", color=model_color("Geneformer"), linewidth=2.8,
            label="Geneformer · local velocity", markersize=8, markeredgewidth=1, markeredgecolor="white",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(40, 100)
    ax.set_ylabel("Top 30% direction accuracy (%)")
    ax.set_xlabel("Pseudotime transition")
    ax.set_title(f"{dataset} — scFoundation vs Geneformer along pseudotime", fontweight="600")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_compare_scgpt(
    out: Path,
    gf_sum: pd.DataFrame,
    scgpt_sum: pd.DataFrame,
    labels: List[str],
) -> None:
    """与 scGPT chained 分段准确率对比（若存在 chained_preds）。"""
    n = len(labels)
    x = np.arange(n)
    w = 0.35
    gf_acc = (gf_sum["acc_top_local"] * 100).tolist()
    sc_acc = []
    for t in gf_sum["transition"]:
        row = scgpt_sum[scgpt_sum["transition"] == t]
        if len(row):
            sc_acc.append(float(row.iloc[0]["acc_top_percent"]) * 100)
        else:
            sc_acc.append(np.nan)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - w / 2, gf_acc, w, label="Geneformer (local velocity)", color=COLOR_GF)
    ax.bar(x + w / 2, sc_acc, w, label="scGPT chained (vs S0)", color=COLOR_SCGPT)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Top 30% direction accuracy (%)")
    ax.set_title("Geneformer velocity vs scGPT chained — per segment", fontweight="600")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="hESC")
    p.add_argument(
        "--pred-dir",
        type=Path,
        default=None,
        help="default: multistep_pt/<dataset>/scfoundation_preds",
    )
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--compare-geneformer", action="store_true")
    p.add_argument(
        "--style",
        choices=["line", "bar", "both"],
        default="line",
        help="line=折线图（默认）; bar=柱状图; both=都输出",
    )
    p.add_argument(
        "--example-gene",
        default=None,
        help="单基因教学图；默认 auto=六段全对且 |Δ| 最大",
    )
    args = p.parse_args()

    apply_fig4_style()
    pred_dir = args.pred_dir or (MULTISTEP_ROOT / args.dataset / "scfoundation_preds")
    if not pred_dir.exists():
        raise FileNotFoundError(
            f"Missing {pred_dir}. Run:\n"
            f"  python3 run_scfoundation_multisegment_pt.py --dataset {args.dataset}"
        )

    outdir = args.outdir or (pred_dir / "figures")
    outdir.mkdir(parents=True, exist_ok=True)

    transitions = discover_transitions(pred_dir, args.dataset)
    if not transitions:
        raise FileNotFoundError(f"No gene_result CSV in {pred_dir}")

    labels = seg_labels(transitions)
    tables = load_tables(pred_dir, args.dataset, transitions)
    seg_sum = pd.read_csv(pred_dir / f"{args.dataset}_scfoundation_segment_summary.csv")
    wide_path = pred_dir / f"{args.dataset}_scfoundation_per_gene_wide.csv"
    wide = pd.read_csv(wide_path) if wide_path.exists() else None

    if args.style in ("line", "both"):
        plot_accuracy_lines(outdir / "01_velocity_accuracy_line.png", seg_sum, labels, args.dataset)
        plot_true_velocity_and_accuracy(
            outdir / "04_velocity_vs_accuracy_line.png",
            tables, seg_sum, transitions, labels, args.dataset,
        )
        plot_observed_trajectory_lines(
            outdir / "05_top_genes_observed_trajectory.png",
            tables, transitions, args.dataset,
        )
        if wide is not None:
            plot_per_gene_accuracy_trajectory(
                outdir / "06_error_accumulation_line.png",
                wide, transitions, labels, args.dataset,
            )
    if args.style in ("bar", "both"):
        plot_velocity_accuracy(outdir / "01_velocity_accuracy_bar.png", seg_sum, transitions, labels)

    plot_true_vs_pred_velocity(outdir / "02_velocity_scatter_segments.png", tables, transitions, labels)
    if wide is not None:
        plot_heatmap_wide(outdir / "03_velocity_error_heatmap.png", wide, transitions, labels)

    if wide is not None:
        gene = args.example_gene
        if gene is None or str(gene).lower() == "auto":
            gene = pick_best_gene(wide, tables, transitions)
        plot_single_gene_example(
            outdir / f"08_example_gene_{gene}.png",
            gene,
            tables,
            transitions,
            labels,
            args.dataset,
        )
        print(f"Example gene panel: {gene}")

    gf_sum_path = MULTISTEP_ROOT / args.dataset / "geneformer_preds" / f"{args.dataset}_geneformer_segment_summary.csv"
    if args.compare_geneformer and gf_sum_path.exists():
        gf_sum = pd.read_csv(gf_sum_path)
        plot_compare_geneformer_lines(outdir / "07_vs_geneformer_line.png", seg_sum, gf_sum, args.dataset)
        if args.style in ("bar", "both"):
            plot_compare_geneformer(outdir / "07_vs_geneformer_bar.png", seg_sum, gf_sum, labels)

    print(f"\n=== {args.dataset} scFoundation multisegment plots ===")
    print(f"Input:  {pred_dir}")
    print(f"Output: {outdir}")
    for f in sorted(outdir.glob("*.png")):
        print(f"  - {f.name}")
    print("\nSegment summary (local velocity, top 30%):")
    for _, r in seg_sum.iterrows():
        print(f"  {r['transition']}: {100*r['acc_top_local']:.1f}%")


if __name__ == "__main__":
    main()
