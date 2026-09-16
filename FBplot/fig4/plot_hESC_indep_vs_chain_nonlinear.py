#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hESC：独立分段 vs 链式 — 模型能否跟上基因表达的非线性轨迹？

三种预测轨迹（5 个伪时间点 S0…S4）
------------------------------------
1. Observed     : 各段真值 bin-mean 拼接
2. Independent  : 每段从**真实** Si 细胞单独跑 scGPT（segment_preds/scgpt/）
3. Chained      : S0 细胞连续迭代 16 步，误差沿链传递（chained_preds/）

输出目录默认：chained_preds/figures/nonlinear_explain/

示例：
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig4
  python3 plot_hESC_indep_vs_chain_nonlinear.py
"""
from __future__ import annotations

import argparse
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
        plt.rcParams.update({"font.size": 11})

    def model_color(name: str, default: str = "#666") -> str:
        return default


SCRIPT_DIR = Path(__file__).resolve().parent
HESC = SCRIPT_DIR / "error_biology" / "multistep_pt" / "hESC"
CHAINED_DIR = HESC / "chained_preds"
INDEP_DIR = HESC / "segment_preds" / "scgpt"
GENE_SEG = HESC / "gene_segment_deltas.csv"

TRANSITIONS = ["delta_t0_t1", "delta_t1_t2", "delta_t2_t3", "delta_t3_t4"]
TRANS_LABELS = ["S0→S1", "S1→S2", "S2→S3", "S3→S4"]
PT_NODES = ["S0", "S1", "S2", "S3", "S4"]
N_PTS = 5

COLOR_TRUE = model_color("scPrint")
COLOR_INDEP = model_color("scFoundation")
COLOR_CHAIN = model_color("scGPT")
COLOR_OK = "#43A047"
COLOR_BAD = "#E53935"

TRAJ_COLORS = {
    "monotone": "#5C6BC0",
    "peak_middle": "#FF9A3D",
    "sign_flip": "#AB47BC",
    "mixed": "#90A4AE",
}


# ---------------------------------------------------------------------------
# Data loading & trajectories
# ---------------------------------------------------------------------------

def load_chained_tables(chained_dir: Path, dataset: str = "hESC") -> Dict[str, pd.DataFrame]:
    return {
        t: pd.read_csv(chained_dir / f"{dataset}_{t}_gene_result.csv").set_index("gene")
        for t in TRANSITIONS
    }


def load_indep_tables(indep_dir: Path, dataset: str = "hESC") -> Dict[int, pd.DataFrame]:
    out = {}
    for i in range(4):
        p = indep_dir / f"{dataset}_seg{i}_to_{i + 1}_gene_result.csv"
        if p.exists():
            out[i] = pd.read_csv(p).set_index("gene")
    return out


def true_profile(ch: Dict[str, pd.DataFrame], gene: str) -> np.ndarray:
    t0 = ch["delta_t0_t1"]
    if gene not in t0.index:
        return np.full(N_PTS, np.nan)
    y = [float(t0.loc[gene, "true_baseline_mean"])]
    for t in TRANSITIONS:
        y.append(float(ch[t].loc[gene, "true_late_mean"]))
    return np.array(y)


def chain_profile(ch: Dict[str, pd.DataFrame], gene: str) -> np.ndarray:
    """链式：每段末 pred 相对 S0 基准的累计表达（与 run 脚本一致）。"""
    t0 = ch["delta_t0_t1"]
    if gene not in t0.index:
        return np.full(N_PTS, np.nan)
    y = [float(t0.loc[gene, "true_baseline_mean"])]
    for t in TRANSITIONS:
        y.append(float(ch[t].loc[gene, "pred_late_like_mean"]))
    return np.array(y)


def indep_profile(ind: Dict[int, pd.DataFrame], ch: Dict[str, pd.DataFrame], gene: str) -> np.ndarray:
    """独立分段：每段从真实 Si 出发，取该段 pred_late。"""
    t0 = ch["delta_t0_t1"]
    if gene not in t0.index or 0 not in ind:
        return np.full(N_PTS, np.nan)
    y = [float(ind[0].loc[gene, "true_early_mean"])]
    for i in range(4):
        if i not in ind or gene not in ind[i].index:
            return np.full(N_PTS, np.nan)
        y.append(float(ind[i].loc[gene, "pred_late_like_mean"]))
    return np.array(y)


def norm_profile(y: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(y)):
        return y
    lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
    if hi - lo < 1e-9:
        return np.zeros_like(y)
    return (y - lo) / (hi - lo)


def curvature(y: np.ndarray) -> np.ndarray:
    """二阶差分（曲率代理），长度 N_PTS-2。"""
    y = np.asarray(y, dtype=float)
    if len(y) < 3 or not np.all(np.isfinite(y)):
        return np.array([np.nan])
    return np.diff(y, n=2)


def shape_corr(true_y: np.ndarray, pred_y: np.ndarray) -> float:
    a, b = norm_profile(true_y), norm_profile(pred_y)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return np.nan
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def curv_mae(true_y: np.ndarray, pred_y: np.ndarray) -> float:
    ct, cp = curvature(true_y), curvature(pred_y)
    if not np.all(np.isfinite(ct)) or not np.all(np.isfinite(cp)):
        return np.nan
    n = min(len(ct), len(cp))
    return float(np.mean(np.abs(ct[:n] - cp[:n])))


def peak_segment_index(deltas: np.ndarray) -> int:
    return int(np.argmax(np.abs(deltas)))


def classify_trajectory(gene: str, gf: pd.DataFrame, ch: Dict[str, pd.DataFrame]) -> str:
    deltas = []
    for t in TRANSITIONS:
        if gene in ch[t].index:
            deltas.append(float(ch[t].loc[gene, "delta_true_local"]))
    if len(deltas) < 4:
        return "mixed"
    signs = [np.sign(d) for d in deltas if d != 0]
    if not signs:
        return "mixed"
    n_flip = sum(1 for i in range(len(signs) - 1) if signs[i] != signs[i + 1])
    if n_flip == 0:
        return "monotone"
    if n_flip >= 2:
        return "sign_flip"
    if gene in gf.index and bool(gf.loc[gene, "peak_middle"]):
        return "peak_middle"
    return "mixed"


def nonlinearity_score(true_y: np.ndarray) -> float:
    """0=近线性, 1=强非线性：基于归一化轨迹曲率能量。"""
    c = curvature(true_y)
    if not np.all(np.isfinite(c)):
        return np.nan
    y = norm_profile(true_y)
    rng = float(np.nanmax(y) - np.nanmin(y)) + 1e-9
    return float(np.sum(c ** 2) / (rng ** 2 + 1e-9))


def build_metrics_table(
    genes: List[str],
    ch: Dict[str, pd.DataFrame],
    ind: Dict[int, pd.DataFrame],
    gf: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for g in genes:
        ty = true_profile(ch, g)
        py_c = chain_profile(ch, g)
        py_i = indep_profile(ind, ch, g)
        if not np.all(np.isfinite(ty)):
            continue
        d_true = np.diff(ty)
        traj = classify_trajectory(g, gf, ch)
        nl = nonlinearity_score(ty)
        row = {
            "gene": g,
            "trajectory_type": traj,
            "nonlinearity_score": nl,
            "shape_corr_chain": shape_corr(ty, py_c),
            "shape_corr_indep": shape_corr(ty, py_i),
            "curv_mae_chain": curv_mae(ty, py_c),
            "curv_mae_indep": curv_mae(ty, py_i),
            "peak_seg_true": peak_segment_index(d_true),
            "peak_seg_chain": peak_segment_index(np.diff(py_c)),
            "peak_seg_indep": peak_segment_index(np.diff(py_i)),
        }
        for i, t in enumerate(TRANSITIONS):
            if g in ch[t].index:
                row[f"dir_ok_chain_{t}"] = int(ch[t].loc[g, "dir_correct"])
            if i in ind and g in ind[i].index:
                row[f"dir_ok_indep_{t}"] = int(ind[i].loc[g, "dir_correct"])
        rows.append(row)
    df = pd.DataFrame(rows)
    chain_cols = [c for c in df.columns if c.startswith("dir_ok_chain_")]
    indep_cols = [c for c in df.columns if c.startswith("dir_ok_indep_")]
    if chain_cols:
        df["dir_acc_chain_mean"] = df[chain_cols].mean(axis=1)
    if indep_cols:
        df["dir_acc_indep_mean"] = df[indep_cols].mean(axis=1)
    df["peak_hit_chain"] = (df["peak_seg_true"] == df["peak_seg_chain"]).astype(int)
    df["peak_hit_indep"] = (df["peak_seg_true"] == df["peak_seg_indep"]).astype(int)
    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_concept_diagram(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    ax.set_title(
        "Three ways to read pseudotime gene dynamics (hESC)",
        fontweight="700",
        fontsize=13,
        loc="left",
    )

    modes = [
        (0.3, "Observed", "Bin-mean expression\nat S0…S4", COLOR_TRUE),
        (3.5, "Independent\nsegments", "4 separate runs:\nstart from true Si cells", COLOR_INDEP),
        (6.7, "Chained\n16-step", "One run from S0:\npred feeds next block", COLOR_CHAIN),
    ]
    for x0, title, desc, col in modes:
        rect = mpatches.FancyBboxPatch(
            (x0, 1.0), 2.6, 1.6,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=col, alpha=0.15, edgecolor=col, linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(x0 + 1.3, 2.15, title, ha="center", fontweight="700", fontsize=11, color=col)
        ax.text(x0 + 1.3, 1.45, desc, ha="center", fontsize=9, color="#333")

    for i, lab in enumerate(PT_NODES):
        ax.text(0.8 + i * 2.15, 0.35, lab, ha="center", fontsize=10, fontweight="600")

    ax.annotate(
        "Nonlinear biology = trajectory shape\n(curvature, peak timing, sign changes)",
        xy=(5, 2.85), fontsize=10, ha="center",
        bbox=dict(boxstyle="round", facecolor="#FFFDE7", edgecolor="#F9A825"),
    )
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_metrics_by_type(out: Path, metrics: pd.DataFrame) -> None:
    types_order = ["monotone", "peak_middle", "sign_flip", "mixed"]
    present = [t for t in types_order if t in metrics["trajectory_type"].values]
    if not present:
        return

    agg = metrics.groupby("trajectory_type").agg(
        n=("gene", "count"),
        shape_chain=("shape_corr_chain", "mean"),
        shape_indep=("shape_corr_indep", "mean"),
        curv_chain=("curv_mae_chain", "mean"),
        curv_indep=("curv_mae_indep", "mean"),
        dir_chain=("dir_acc_chain_mean", "mean"),
        dir_indep=("dir_acc_indep_mean", "mean"),
        peak_chain=("peak_hit_chain", "mean"),
        peak_indep=("peak_hit_indep", "mean"),
    ).reindex(present)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    x = np.arange(len(present))
    w = 0.35

    ax = axes[0]
    ax.bar(x - w / 2, agg["shape_chain"] * 100, w, label="Chained", color=COLOR_CHAIN, alpha=0.9)
    ax.bar(x + w / 2, agg["shape_indep"] * 100, w, label="Independent", color=COLOR_INDEP, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(n={int(agg.loc[t,'n'])})" for t in present], fontsize=9)
    ax.set_ylabel("Mean shape correlation (%)")
    ax.set_ylim(0, 100)
    ax.set_title("A  Trajectory shape match", fontweight="600", loc="left")
    ax.legend(fontsize=8)
    ax.axhline(50, color="#999", ls=":", lw=0.8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x - w / 2, agg["curv_chain"], w, label="Chained", color=COLOR_CHAIN, alpha=0.9)
    ax.bar(x + w / 2, agg["curv_indep"], w, label="Independent", color=COLOR_INDEP, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([t for t in present], fontsize=9)
    ax.set_ylabel("Mean |curvature error|")
    ax.set_title("B  Curvature (2nd diff.) error", fontweight="600", loc="left")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    ax.bar(x - w / 2, agg["peak_chain"] * 100, w, label="Chained", color=COLOR_CHAIN, alpha=0.9)
    ax.bar(x + w / 2, agg["peak_indep"] * 100, w, label="Independent", color=COLOR_INDEP, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([t for t in present], fontsize=9)
    ax.set_ylabel("Peak |Δ| in correct segment (%)")
    ax.set_ylim(0, 100)
    ax.set_title("C  Peak timing accuracy", fontweight="600", loc="left")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Can scGPT capture nonlinear change? — by true trajectory type",
        fontsize=12, fontweight="700", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_nonlinearity_scatter(out: Path, metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    m = metrics.dropna(subset=["nonlinearity_score", "shape_corr_chain", "shape_corr_indep"])
    sc = ax.scatter(
        m["nonlinearity_score"], m["shape_corr_chain"],
        c=COLOR_CHAIN, s=18, alpha=0.45, label="Chained shape r",
        edgecolors="none",
    )
    ax.scatter(
        m["nonlinearity_score"], m["shape_corr_indep"],
        c=COLOR_INDEP, s=18, alpha=0.35, label="Independent shape r",
        edgecolors="none",
    )
    ax.set_xlabel("True trajectory nonlinearity score")
    ax.set_ylabel("Shape correlation (pred vs true)")
    ax.set_title(
        "Higher nonlinearity → harder to match trajectory shape",
        fontweight="600",
    )
    ax.axhline(0, color="#CCC", lw=0.8)
    ax.legend(markerscale=2)
    ax.grid(alpha=0.25)
    # trend lines
    for col, color in [("shape_corr_chain", COLOR_CHAIN), ("shape_corr_indep", COLOR_INDEP)]:
        sub = m.dropna(subset=["nonlinearity_score", col])
        if len(sub) > 10:
            z = np.polyfit(sub["nonlinearity_score"], sub[col], 1)
            xs = np.linspace(sub["nonlinearity_score"].min(), sub["nonlinearity_score"].max(), 50)
            ax.plot(xs, np.poly1d(z)(xs), color=color, lw=2, alpha=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pick_example_genes(metrics: pd.DataFrame, ch: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    """每种轨迹类型 + 链式最差/独立最好对比各选一个。"""
    picks: Dict[str, str] = {}
    for t in ["monotone", "peak_middle", "sign_flip"]:
        sub = metrics[metrics["trajectory_type"] == t].dropna(subset=["shape_corr_chain"])
        if sub.empty:
            continue
        picks[f"{t}_good"] = str(sub.nlargest(1, "shape_corr_indep").iloc[0]["gene"])
        picks[f"{t}_bad"] = str(sub.nsmallest(1, "shape_corr_chain").iloc[0]["gene"])

    diff = metrics.dropna(subset=["shape_corr_indep", "shape_corr_chain"]).copy()
    diff["gain"] = diff["shape_corr_indep"] - diff["shape_corr_chain"]
    if len(diff):
        picks["indep_wins"] = str(diff.nlargest(1, "gain").iloc[0]["gene"])
        picks["chain_ok"] = str(diff.nlargest(1, "shape_corr_chain").iloc[0]["gene"])
    for g in ["ERCC-00113", "LPIN1", "CSRP2"]:
        if g in ch["delta_t0_t1"].index and g not in picks.values():
            picks[f"default_{g}"] = g
    return picks


def plot_example_trajectories(
    out: Path,
    examples: Dict[str, str],
    ch: Dict[str, pd.DataFrame],
    ind: Dict[int, pd.DataFrame],
    metrics: pd.DataFrame,
) -> None:
    genes = list(dict.fromkeys(examples.values()))[:6]
    n = len(genes)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.6 * nrows), squeeze=False)
    x = np.arange(N_PTS)
    m_idx = metrics.set_index("gene") if len(metrics) else pd.DataFrame()

    for ax, gene in zip(axes.flat, genes):
        ty = true_profile(ch, gene)
        pi = indep_profile(ind, ch, gene)
        pc = chain_profile(ch, gene)
        if not np.all(np.isfinite(ty)):
            ax.set_visible(False)
            continue
        ax.plot(x, ty, "o-", color=COLOR_TRUE, lw=2.5, ms=8, label="Observed", zorder=4)
        ax.plot(x, pi, "s--", color=COLOR_INDEP, lw=2, ms=7, label="Independent")
        ax.plot(x, pc, "^-", color=COLOR_CHAIN, lw=2, ms=7, label="Chained")
        note = ""
        if gene in m_idx.index:
            r = m_idx.loc[gene]
            note = (
                f"{r.get('trajectory_type','?')} | nl={r.get('nonlinearity_score',0):.2f}\n"
                f"shape r: indep {r.get('shape_corr_indep',0):.2f} / chain {r.get('shape_corr_chain',0):.2f}"
            )
        ax.set_title(gene, fontweight="700", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(PT_NODES)
        ax.set_ylabel("Mean expression")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=6, loc="best")
        if note:
            ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=7, va="bottom",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    for ax in axes.flat[len(genes):]:
        ax.set_visible(False)

    fig.suptitle(
        "Example genes — observed vs independent vs chained trajectories",
        fontsize=12, fontweight="700", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_summary_panel(
    out: Path,
    metrics: pd.DataFrame,
    ch: Dict[str, pd.DataFrame],
    ind: Dict[int, pd.DataFrame],
) -> None:
    """一张总览：文字结论 + 关键数字 + 小折线示意。"""
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)

    # A: 文字解读
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    m = metrics.dropna(subset=["shape_corr_chain", "shape_corr_indep"])
    sc = m["shape_corr_chain"].mean()
    si = m["shape_corr_indep"].mean()
    dc = m["dir_acc_chain_mean"].mean() * 100 if "dir_acc_chain_mean" in m else np.nan
    di = m["dir_acc_indep_mean"].mean() * 100 if "dir_acc_indep_mean" in m else np.nan
    pk_c = m["peak_hit_chain"].mean() * 100
    pk_i = m["peak_hit_indep"].mean() * 100
    nl_hi = m[m["nonlinearity_score"] > m["nonlinearity_score"].median()]
    nl_lo = m[m["nonlinearity_score"] <= m["nonlinearity_score"].median()]
    r_hi = nl_hi["shape_corr_indep"].mean() - nl_hi["shape_corr_chain"].mean()
    text = (
        "How to read this figure\n"
        "─────────────────────────\n"
        "• Observed: true bin-mean along S0→S4.\n"
        "• Independent: each block restarts from\n"
        "  real Si cells — tests local dynamics.\n"
        "• Chained: one continuous run from S0;\n"
        "  errors accumulate.\n\n"
        "Key findings (all genes, n=%d)\n"
        "─────────────────────────\n"
        "• Mean shape correlation:\n"
        "    Independent %.2f  |  Chained %.2f\n"
        "• Mean direction acc (4 blocks):\n"
        "    Independent %.1f%%  |  Chained %.1f%%\n"
        "• Peak |Δ| segment hit rate:\n"
        "    Independent %.1f%%  |  Chained %.1f%%\n\n"
        "Nonlinear genes (upper half nl. score):\n"
        "  indep − chain shape r ≈ %+.2f\n\n"
        "Interpretation:\n"
        "  Flat chained lines ≈ model pushes\n"
        "  expression toward S0+global Up offset;\n"
        "  independent segments test whether\n"
        "  local blocks match true curvature.\n"
        "  Both struggle on sign-flip / peak-mid\n"
        "  genes → limited nonlinear capture."
    ) % (len(m), si, sc, di, dc, pk_i, pk_c, r_hi)
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", fontsize=9.5,
            family="monospace", linespacing=1.35)
    ax.set_title("A  Guide & summary", fontweight="600", loc="left")

    # B: 形状相关 链式 vs 独立
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(m["shape_corr_chain"], m["shape_corr_indep"], s=12, alpha=0.35, c="#546E7A")
    lim = [0, 1]
    ax.plot(lim, lim, "k--", lw=1, alpha=0.5)
    ax.set_xlabel("Chained shape correlation")
    ax.set_ylabel("Independent shape correlation")
    ax.set_title("B  Independent vs chained shape match", fontweight="600", loc="left")
    above = (m["shape_corr_indep"] > m["shape_corr_chain"]).mean() * 100
    ax.text(0.05, 0.95, f"{above:.0f}% genes: indep > chain", transform=ax.transAxes,
            va="top", fontsize=10, bbox=dict(boxstyle="round", facecolor="#E8F5E9"))
    ax.grid(alpha=0.25)

    # C: 按类型的形状相关
    ax = fig.add_subplot(gs[1, 0])
    types = ["monotone", "peak_middle", "sign_flip"]
    xs = np.arange(len(types))
    for j, (col, lab, c) in enumerate([
        ("shape_corr_chain", "Chained", COLOR_CHAIN),
        ("shape_corr_indep", "Independent", COLOR_INDEP),
    ]):
        vals = [metrics[metrics["trajectory_type"] == t][col].mean() for t in types]
        ax.bar(xs + (j - 0.5) * 0.38, vals, 0.36, label=lab, color=c, alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(types)
    ax.set_ylabel("Mean shape correlation")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title("C  Nonlinear types need shape, not only direction", fontweight="600", loc="left")
    ax.grid(axis="y", alpha=0.3)

    # D: 示意折线（聚合）
    ax = fig.add_subplot(gs[1, 1])
    for t, col in [("peak_middle", COLOR_INDEP), ("sign_flip", COLOR_BAD)]:
        sub = metrics[metrics["trajectory_type"] == t]["gene"].head(80).tolist()
        if not sub:
            continue
        ty_m, pi_m, pc_m = [], [], []
        for g in sub:
            ty = true_profile(ch, g)
            if np.all(np.isfinite(ty)):
                ty_m.append(norm_profile(ty))
                pi_m.append(norm_profile(indep_profile(ind, ch, g)))
                pc_m.append(norm_profile(chain_profile(ch, g)))
        if ty_m:
            ax.plot(x := np.arange(N_PTS), np.nanmean(ty_m, axis=0), "o-", color=COLOR_TRUE,
                    lw=2.5, label=f"True ({t})")
            ax.plot(x, np.nanmean(pi_m, axis=0), "s--", color=col, lw=2, label=f"Indep ({t})")
            ax.plot(x, np.nanmean(pc_m, axis=0), "^:", color=COLOR_CHAIN, lw=2, label=f"Chain ({t})")
    ax.set_xticks(np.arange(N_PTS))
    ax.set_xticklabels(PT_NODES)
    ax.set_ylabel("Mean scaled expr.")
    ax.set_title("D  Average trajectories by type", fontweight="600", loc="left")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "hESC scGPT — independent segments vs chained inference (nonlinear dynamics)",
        fontsize=13, fontweight="700", y=0.98,
    )
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_curvature_lines(out: Path, metrics: pd.DataFrame, ch: Dict, ind: Dict) -> None:
    """高非线性基因：曲率真值 vs 预测折线。"""
    sub = metrics.nlargest(12, "nonlinearity_score")
    fig, axes = plt.subplots(3, 4, figsize=(14, 8), squeeze=False)
    for ax, (_, r) in zip(axes.flat, sub.iterrows()):
        g = str(r["gene"])
        ty = true_profile(ch, g)
        ct, ci, cc = curvature(ty), curvature(indep_profile(ind, ch, g)), curvature(chain_profile(ch, g))
        xp = np.arange(len(ct))
        ax.plot(xp, ct, "o-", color=COLOR_TRUE, lw=2, label="True κ")
        ax.plot(xp[: len(ci)], ci[: len(xp)], "s--", color=COLOR_INDEP, lw=1.8, label="Indep κ")
        ax.plot(xp[: len(cc)], cc[: len(xp)], "^-", color=COLOR_CHAIN, lw=1.8, label="Chain κ")
        ax.axhline(0, color="#999", lw=0.6)
        ax.set_title(g, fontsize=9, fontweight="600")
        ax.set_xticks(xp)
        ax.set_xticklabels(["S1", "S2", "S3"][: len(xp)], fontsize=7)
        ax.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=6, loc="upper right")
    fig.suptitle("Curvature (2nd difference) — highly nonlinear genes", fontsize=12, fontweight="700")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--chained-dir", type=Path, default=CHAINED_DIR)
    p.add_argument("--indep-dir", type=Path, default=INDEP_DIR)
    p.add_argument("--outdir", type=Path, default=None)
    args = p.parse_args()

    apply_fig4_style()
    outdir = args.outdir or (args.chained_dir / "figures" / "nonlinear_explain")
    outdir.mkdir(parents=True, exist_ok=True)

    ch = load_chained_tables(args.chained_dir)
    ind = load_indep_tables(args.indep_dir)
    if len(ind) < 4:
        raise FileNotFoundError(f"Need 4 independent segment CSVs under {args.indep_dir}")

    gf = pd.read_csv(GENE_SEG).set_index("gene") if GENE_SEG.exists() else pd.DataFrame()
    genes = sorted(set(ch["delta_t0_t1"].index) & set(ind[0].index))
    metrics = build_metrics_table(genes, ch, ind, gf)
    metrics.to_csv(outdir / "gene_trajectory_metrics.csv", index=False)

    summary = metrics.groupby("trajectory_type").agg(
        n=("gene", "count"),
        mean_nl=("nonlinearity_score", "mean"),
        shape_chain=("shape_corr_chain", "mean"),
        shape_indep=("shape_corr_indep", "mean"),
        curv_chain=("curv_mae_chain", "mean"),
        curv_indep=("curv_mae_indep", "mean"),
        peak_chain=("peak_hit_chain", "mean"),
        peak_indep=("peak_hit_indep", "mean"),
    )
    summary.to_csv(outdir / "summary_by_trajectory_type.csv")

    plot_concept_diagram(outdir / "01_concept_three_modes.png")
    plot_metrics_by_type(outdir / "02_metrics_by_trajectory_type.png", metrics)
    plot_nonlinearity_scatter(outdir / "03_nonlinearity_vs_shape_corr.png", metrics)
    examples = pick_example_genes(metrics, ch)
    pd.Series(examples, name="gene").to_csv(outdir / "example_genes.csv", header=True)
    plot_example_trajectories(outdir / "04_example_trajectories.png", examples, ch, ind, metrics)
    plot_curvature_lines(outdir / "05_curvature_high_nl_genes.png", metrics, ch, ind)
    plot_summary_panel(outdir / "06_summary_panel.png", metrics, ch, ind)

    print("\n=== hESC independent vs chained (nonlinear explain) ===")
    print(f"Genes: {len(metrics)}")
    print(f"Output: {outdir}\n")
    print("Trajectory type counts:")
    print(metrics["trajectory_type"].value_counts().to_string())
    print("\nGlobal means:")
    print(f"  shape_corr  indep={metrics['shape_corr_indep'].mean():.3f}  chain={metrics['shape_corr_chain'].mean():.3f}")
    print(f"  curv_mae    indep={metrics['curv_mae_indep'].mean():.3f}  chain={metrics['curv_mae_chain'].mean():.3f}")
    print(f"  peak_hit%   indep={100*metrics['peak_hit_indep'].mean():.1f}  chain={100*metrics['peak_hit_chain'].mean():.1f}")
    print("\nFigures:")
    for f in sorted(outdir.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
