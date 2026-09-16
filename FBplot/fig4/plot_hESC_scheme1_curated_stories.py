#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案1：独立分段 + 5点轨迹拼接 — 筛选有故事基因并出图。

故事类型
--------
1. error_propagation   独立形状好 + 链式差 → 误差传播主导
2. nonlinear_fail      高非线性 + 独立/链式形状都差 + 曲率错 → 非线性未学到
3. peak_mid_rescue     单步方向错 + 中段峰 + 独立分段在峰值段更对 → 峰值在中段

指标：shape r, DTW(归一化), 曲率 MAE, 峰值段命中；可选单步 5 点插值轨迹。

  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig4
  python3 plot_hESC_scheme1_curated_stories.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from plot_hESC_indep_vs_chain_nonlinear import (  # noqa: E402
    CHAINED_DIR,
    GENE_SEG,
    INDEP_DIR,
    N_PTS,
    PT_NODES,
    TRANSITIONS,
    TRANS_LABELS,
    build_metrics_table,
    chain_profile,
    classify_trajectory,
    curv_mae,
    indep_profile,
    load_chained_tables,
    load_indep_tables,
    norm_profile,
    shape_corr,
    true_profile,
)

try:
    from fig4_palette import apply_fig4_style, model_color
except ImportError:

    def apply_fig4_style() -> None:
        plt.rcParams.update({"font.size": 11})

    def model_color(name: str, default: str = "#666") -> str:
        return default


SINGLE_DIR = Path("/mnt/10T/yzn/benchmark_GRN/pre_scgpt/results_multidataset_pseudotime_227")
OUT_SUB = "scheme1_curated"

COLOR_TRUE = model_color("scPrint")
COLOR_INDEP = model_color("scFoundation")
COLOR_CHAIN = model_color("scGPT")
COLOR_SINGLE = "#9E9E9E"

STORY_STYLE = {
    "error_propagation": {"color": "#1565C0", "title": "① Error propagation\n(indep OK → chain collapses)"},
    "nonlinear_fail": {"color": "#C62828", "title": "② Nonlinear not captured\n(both indep & chain fail)"},
    "peak_mid_rescue": {"color": "#2E7D32", "title": "③ Peak in middle\n(single wrong, indep better @ peak)"},
}


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """归一化轨迹上的 DTW（越小越好）。"""
    a = norm_profile(np.asarray(a, dtype=float))
    b = norm_profile(np.asarray(b, dtype=float))
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return np.nan
    n, m = len(a), len(b)
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            d[i, j] = cost + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return float(d[i, j])


def dtw_similarity(true_y: np.ndarray, pred_y: np.ndarray) -> float:
    """映射到 (0,1]，越大越好。"""
    dist = dtw_distance(true_y, pred_y)
    if not np.isfinite(dist):
        return np.nan
    return float(1.0 / (1.0 + dist))


def single_profile(gene: str, gr: pd.DataFrame, true_y: np.ndarray) -> np.ndarray:
    if gene not in gr.index or not np.all(np.isfinite(true_y)):
        return np.full(N_PTS, np.nan)
    e = float(gr.loc[gene, "true_early_mean"])
    p = float(gr.loc[gene, "pred_late_like_mean"])
    return np.array([e + (k / (N_PTS - 1)) * (p - e) for k in range(N_PTS)])


def enrich_metrics(
    metrics: pd.DataFrame,
    ch: Dict[str, pd.DataFrame],
    ind: Dict[int, pd.DataFrame],
    gr: Optional[pd.DataFrame],
    gf: pd.DataFrame,
) -> pd.DataFrame:
    m = metrics.copy()
    m["gain_shape"] = m["shape_corr_indep"] - m["shape_corr_chain"]
    rows = []
    for _, r in m.iterrows():
        g = str(r["gene"])
        ty = true_profile(ch, g)
        pc, pi = chain_profile(ch, g), indep_profile(ind, ch, g)
        ps = single_profile(g, gr, ty) if gr is not None else np.full(N_PTS, np.nan)
        row = dict(r)
        row["dtw_sim_chain"] = dtw_similarity(ty, pc)
        row["dtw_sim_indep"] = dtw_similarity(ty, pi)
        row["shape_corr_single"] = shape_corr(ty, ps) if np.all(np.isfinite(ps)) else np.nan
        row["dtw_sim_single"] = dtw_similarity(ty, ps) if np.all(np.isfinite(ps)) else np.nan
        if g in gf.index:
            row["peak_middle"] = bool(gf.loc[g, "peak_middle"])
            row["best_transition"] = str(gf.loc[g, "best_transition"])
            row["abs_delta_best"] = float(gf.loc[g, "abs_delta_best"])
            row["is_chip_target"] = bool(gf.loc[g, "is_chip_target"])
        else:
            row["peak_middle"] = False
            row["best_transition"] = ""
            row["abs_delta_best"] = np.nan
            row["is_chip_target"] = False
        if gr is not None and g in gr.index:
            row["single_dir_correct"] = int(gr.loc[g, "dir_correct"])
        else:
            row["single_dir_correct"] = np.nan
        # 峰值段方向（独立 / 链式 / 单步）
        bt = row.get("best_transition", "")
        if bt in TRANSITIONS:
            seg_i = TRANSITIONS.index(bt)
            col_c = f"dir_ok_chain_{bt}"
            col_i = f"dir_ok_indep_{bt}"
            row["dir_ok_at_peak_chain"] = int(r[col_c]) if col_c in r.index and pd.notna(r[col_c]) else np.nan
            row["dir_ok_at_peak_indep"] = int(r[col_i]) if col_i in r.index and pd.notna(r[col_i]) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _top(df: pd.DataFrame, n: int, col: str, ascending: bool = False) -> List[str]:
    if df.empty:
        return []
    return df.sort_values(col, ascending=ascending).head(n)["gene"].astype(str).tolist()


def screen_story_genes(m: pd.DataFrame, per_story: int = 2) -> Dict[str, List[str]]:
    m = m.copy()
    nl_med = m["nonlinearity_score"].median()
    nl_q80 = m["nonlinearity_score"].quantile(0.80)
    dyn = m["abs_delta_best"].notna() & (m["abs_delta_best"] >= m["abs_delta_best"].quantile(0.5))

    # ① 误差传播
    prop = m[
        dyn
        & (m["gain_shape"] >= 0.30)
        & (m["shape_corr_indep"] >= 0.50)
        & (m["shape_corr_chain"] <= 0.30)
        & (m["nonlinearity_score"] >= nl_med)
    ]
    prop_genes = _top(prop, per_story, "gain_shape")

    # ② 非线性未学到
    fail = m[
        (m["nonlinearity_score"] >= nl_q80)
        & (m["shape_corr_indep"] < 0.20)
        & (m["shape_corr_chain"] < 0.25)
        & (m["trajectory_type"].isin(["sign_flip", "peak_middle"]))
    ]
    fail_genes = _top(fail, per_story, "nonlinearity_score")

    # ③ 中段峰 + 单步错 + 独立在峰值段方向对或形状明显优于链式
    peak = m[
        m["peak_middle"].astype(bool)
        & (m["single_dir_correct"] == 0)
        & dyn
        & (
            (m["dir_ok_at_peak_indep"] == 1)
            | (m["gain_shape"] >= 0.15)
            | (m["shape_corr_indep"] - m["shape_corr_single"].fillna(-1) >= 0.10)
        )
    ]
    peak = peak.copy()
    peak["peak_score"] = peak["abs_delta_best"].fillna(0) * (
        peak["dir_ok_at_peak_indep"].fillna(0) + peak["gain_shape"].clip(lower=0)
    )
    peak_genes = _top(peak, per_story, "peak_score")

    # 回退：类别不足时放宽
    if len(prop_genes) < per_story:
        extra = _top(
            m[(m["gain_shape"] >= 0.20)].drop(index=prop_genes, errors="ignore"),
            per_story - len(prop_genes),
            "gain_shape",
        )
        prop_genes.extend([g for g in extra if g not in prop_genes])
    if len(fail_genes) < per_story:
        extra = _top(
            m[m["nonlinearity_score"] >= nl_q80]
            .drop(index=fail_genes, errors="ignore")
            .nsmallest(per_story * 3, "shape_corr_indep"),
            per_story - len(fail_genes),
            "nonlinearity_score",
        )
        fail_genes.extend([g for g in extra if g not in fail_genes])
    if len(peak_genes) < per_story:
        ercc = m[m["gene"].astype(str).str.startswith("ERCC")].sort_values(
            "abs_delta_best", ascending=False
        )
        for g in _top(ercc, 3, "abs_delta_best"):
            if g not in peak_genes:
                peak_genes.append(g)
            if len(peak_genes) >= per_story:
                break
        for g in ["ERCC-00113", "ERCC-00108", "NRP1", "CALB1"]:
            if g in m["gene"].values and g not in peak_genes:
                peak_genes.append(g)
            if len(peak_genes) >= per_story:
                break

    return {
        "error_propagation": prop_genes[:per_story],
        "nonlinear_fail": fail_genes[:per_story],
        "peak_mid_rescue": peak_genes[:per_story],
    }


def assign_story_labels(m: pd.DataFrame, picks: Dict[str, List[str]]) -> pd.DataFrame:
    inv = {}
    for story, genes in picks.items():
        for g in genes:
            inv.setdefault(g, []).append(story)
    m = m.copy()
    m["story_curated"] = m["gene"].map(lambda g: ";".join(inv.get(str(g), [])) or "")
    m["is_curated"] = m["story_curated"].str.len() > 0
    return m


def plot_story_trajectories(
    out: Path,
    picks: Dict[str, List[str]],
    ch: Dict,
    ind: Dict,
    gr: Optional[pd.DataFrame],
    metrics: pd.DataFrame,
) -> None:
    """3 行 × 2 列：每格 4 条轨迹 + 指标注释。"""
    m_idx = metrics.set_index("gene")
    fig, axes = plt.subplots(3, 2, figsize=(11, 10.5), squeeze=False)
    x = np.arange(N_PTS)

    for row, (story, genes) in enumerate(picks.items()):
        sty = STORY_STYLE[story]
        for col, gene in enumerate(genes[:2]):
            ax = axes[row, col]
            ty = true_profile(ch, gene)
            pi, pc = indep_profile(ind, ch, gene), chain_profile(ch, gene)
            ps = single_profile(gene, gr, ty) if gr is not None else np.full(N_PTS, np.nan)
            if not np.all(np.isfinite(ty)):
                ax.set_visible(False)
                continue

            def draw_line(y, **kw):
                if np.all(np.isfinite(y)):
                    ax.plot(x, norm_profile(y), **kw)

            draw_line(ty, marker="o", linestyle="-", color=COLOR_TRUE, lw=2.6, ms=8, label="Observed", zorder=4)
            draw_line(pi, marker="s", linestyle="--", color=COLOR_INDEP, lw=2, ms=6, label="Independent")
            draw_line(pc, marker="^", linestyle="-", color=COLOR_CHAIN, lw=2, ms=6, label="Chained")
            if np.all(np.isfinite(ps)):
                draw_line(ps, marker="d", linestyle=":", color=COLOR_SINGLE, lw=1.8, ms=5, label="Single S0↔S4")

            # 峰值段竖线
            if gene in m_idx.index:
                bt = str(m_idx.loc[gene].get("best_transition", ""))
                if bt in TRANSITIONS:
                    peak_i = TRANSITIONS.index(bt) + 1
                    ax.axvline(peak_i, color=sty["color"], ls=":", lw=2, alpha=0.7)

            ax.set_xticks(x)
            ax.set_xticklabels(PT_NODES)
            ax.set_ylabel("Scaled expression")
            ax.set_title(gene, fontweight="700", fontsize=11)
            ax.legend(fontsize=6, loc="best", framealpha=0.9)
            ax.grid(axis="y", alpha=0.25)

            if gene in m_idx.index:
                r = m_idx.loc[gene]
                txt = (
                    f"shape r  indep {r.get('shape_corr_indep', 0):.2f}  "
                    f"chain {r.get('shape_corr_chain', 0):.2f}  "
                    f"single {r.get('shape_corr_single', float('nan')):.2f}\n"
                    f"DTW sim  indep {r.get('dtw_sim_indep', 0):.2f}  "
                    f"chain {r.get('dtw_sim_chain', 0):.2f}\n"
                    f"peak hit  indep {int(r.get('peak_hit_indep', 0))}  "
                    f"chain {int(r.get('peak_hit_chain', 0))}  "
                    f"| curv err indep {r.get('curv_mae_indep', 0):.1f}"
                )
                ax.text(
                    0.02, 0.02, txt, transform=ax.transAxes, fontsize=6.5, va="bottom",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.92),
                )

        axes[row, 0].text(
            -0.35, 0.5, sty["title"], transform=axes[row, 0].transAxes,
            fontsize=9, fontweight="700", color=sty["color"], va="center", ha="right",
        )

    fig.suptitle(
        "Scheme 1 — curated genes: independent stitch vs chained (normalized trajectories)",
        fontsize=12, fontweight="700", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_metrics_comparison(out: Path, picks: Dict[str, List[str]], metrics: pd.DataFrame) -> None:
    """精选基因 × 4 指标 × 3 模式（indep/chain/single）。"""
    genes = []
    colors_g = []
    for story, gs in picks.items():
        for g in gs:
            genes.append(g)
            colors_g.append(STORY_STYLE[story]["color"])

    if not genes:
        return
    sub = metrics.set_index("gene").loc[genes]
    metric_defs = [
        ("shape_corr_indep", "shape_corr_chain", "shape_corr_single", "Shape correlation"),
        ("dtw_sim_indep", "dtw_sim_chain", "dtw_sim_single", "DTW similarity"),
        ("peak_hit_indep", "peak_hit_chain", None, "Peak segment hit (0/1)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    x = np.arange(len(genes))
    w = 0.25

    for ax, (col_i, col_c, col_s, title) in zip(axes, metric_defs):
        vals_i = sub[col_i].fillna(0).values
        vals_c = sub[col_c].fillna(0).values
        ax.bar(x - w, vals_i, w, label="Independent", color=COLOR_INDEP, alpha=0.9)
        ax.bar(x, vals_c, w, label="Chained", color=COLOR_CHAIN, alpha=0.9)
        if col_s and col_s in sub.columns:
            vals_s = sub[col_s].fillna(0).values
            ax.bar(x + w, vals_s, w, label="Single", color=COLOR_SINGLE, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(genes, rotation=35, ha="right", fontsize=8)
        ax.set_title(title, fontweight="600")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        for i, c in enumerate(colors_g):
            ax.axvspan(i - 0.5, i + 0.5, color=c, alpha=0.06)

    axes[0].legend(fontsize=7, ncol=3, loc="upper right")
    fig.suptitle("Curated genes — shape & DTW & peak timing", fontsize=12, fontweight="700", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_screening_landscape(out: Path, m: pd.DataFrame, picks: Dict[str, List[str]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    base = m[~m["is_curated"]]
    ax.scatter(
        base["gain_shape"], base["nonlinearity_score"],
        c="#B0BEC5", s=14, alpha=0.35, label="Other genes",
    )
    for story, genes in picks.items():
        sub = m[m["gene"].isin(genes)]
        ax.scatter(
            sub["gain_shape"], sub["nonlinearity_score"],
            c=STORY_STYLE[story]["color"], s=120, edgecolors="white", linewidths=1.5,
            label=STORY_STYLE[story]["title"].split("\n")[0], zorder=5,
        )
        for _, r in sub.iterrows():
            ax.annotate(
                str(r["gene"]), (r["gain_shape"], r["nonlinearity_score"]),
                fontsize=8, fontweight="600", xytext=(4, 4), textcoords="offset points",
            )
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlabel("Shape gain: independent − chained")
    ax.set_ylabel("True trajectory nonlinearity score")
    ax.set_title("Gene screening for Scheme 1 stories", fontweight="600")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_population_summary(out: Path, m: pd.DataFrame) -> None:
    """全队列：三故事候选数量 + 平均指标对比。"""
    m = m.copy()
    m["story_prop"] = (
        (m["gain_shape"] >= 0.30)
        & (m["shape_corr_indep"] >= 0.5)
        & (m["shape_corr_chain"] <= 0.3)
    )
    m["story_fail"] = (
        (m["nonlinearity_score"] >= m["nonlinearity_score"].quantile(0.8))
        & (m["shape_corr_indep"] < 0.2)
    )
    m["story_peak"] = m["peak_middle"].astype(bool) & (m["single_dir_correct"] == 0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    counts = [m["story_prop"].sum(), m["story_fail"].sum(), m["story_peak"].sum()]
    labels = ["Propagation\n(indep≫chain)", "Nonlinear fail\n(both low shape)", "Peak-mid\n(single wrong)"]
    colors = [STORY_STYLE[k]["color"] for k in ["error_propagation", "nonlinear_fail", "peak_mid_rescue"]]
    axes[0].bar(labels, counts, color=colors, edgecolor="white", width=0.6)
    axes[0].set_ylabel("Gene count (relaxed criteria)")
    axes[0].set_title("A  How many genes fit each narrative?", fontweight="600", loc="left")

    pop = pd.DataFrame({
        "All genes": [
            m["shape_corr_indep"].mean(),
            m["shape_corr_chain"].mean(),
            m["dtw_sim_indep"].mean(),
            m["dtw_sim_chain"].mean(),
            m["peak_hit_indep"].mean(),
            m["peak_hit_chain"].mean(),
        ],
        "Propagation-like": [
            m.loc[m["story_prop"], "shape_corr_indep"].mean(),
            m.loc[m["story_prop"], "shape_corr_chain"].mean(),
            m.loc[m["story_prop"], "dtw_sim_indep"].mean(),
            m.loc[m["story_prop"], "dtw_sim_chain"].mean(),
            m.loc[m["story_prop"], "peak_hit_indep"].mean(),
            m.loc[m["story_prop"], "peak_hit_chain"].mean(),
        ],
    }, index=["Shape r indep", "Shape r chain", "DTW sim indep", "DTW sim chain", "Peak hit indep", "Peak hit chain"])

    x = np.arange(len(pop))
    w = 0.35
    axes[1].bar(x - w / 2, pop["All genes"], w, label="All genes", color="#90A4AE")
    axes[1].bar(x + w / 2, pop["Propagation-like"], w, label="Propagation-like", color=COLOR_INDEP)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(pop.index, rotation=25, ha="right", fontsize=8)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("B  Population metrics (shape > acc for nonlinearity)", fontweight="600", loc="left")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Scheme 1 — population context", fontsize=12, fontweight="700", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--chained-dir", type=Path, default=CHAINED_DIR)
    p.add_argument("--indep-dir", type=Path, default=INDEP_DIR)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--per-story", type=int, default=2, help="genes per story row")
    args = p.parse_args()

    apply_fig4_style()
    outdir = args.outdir or (args.chained_dir / "figures" / OUT_SUB)
    outdir.mkdir(parents=True, exist_ok=True)

    ch = load_chained_tables(args.chained_dir)
    ind = load_indep_tables(args.indep_dir)
    gf = pd.read_csv(GENE_SEG).set_index("gene") if GENE_SEG.exists() else pd.DataFrame()
    gr = None
    sp = SINGLE_DIR / "hESC_gene_result.csv"
    if sp.exists():
        gr = pd.read_csv(sp).set_index("gene")

    genes = sorted(set(ch["delta_t0_t1"].index) & set(ind[0].index))
    metrics = build_metrics_table(genes, ch, ind, gf)
    metrics = enrich_metrics(metrics, ch, ind, gr, gf)

    picks = screen_story_genes(metrics, per_story=args.per_story)
    metrics = assign_story_labels(metrics, picks)
    metrics.to_csv(outdir / "scheme1_gene_metrics_full.csv", index=False)

    curated_rows = []
    for story, gs in picks.items():
        for g in gs:
            curated_rows.append({"story": story, "gene": g})
    pd.DataFrame(curated_rows).to_csv(outdir / "scheme1_curated_genes.csv", index=False)

    plot_story_trajectories(outdir / "01_curated_trajectories_3x2.png", picks, ch, ind, gr, metrics)
    plot_metrics_comparison(outdir / "02_curated_metrics_bars.png", picks, metrics)
    plot_screening_landscape(outdir / "03_screening_landscape.png", metrics, picks)
    plot_population_summary(outdir / "04_population_summary.png", metrics)

    print("\n=== Scheme 1 curated stories ===")
    print(f"Output: {outdir}\n")
    for story, gs in picks.items():
        print(f"  {story}: {', '.join(gs)}")
    sub = metrics[metrics["gene"].isin(sum(picks.values(), []))].set_index("gene")
    print("\nCurated metrics:")
    cols = [
        "shape_corr_indep", "shape_corr_chain", "gain_shape",
        "dtw_sim_indep", "dtw_sim_chain", "peak_hit_indep", "peak_hit_chain",
    ]
    print(sub[cols].round(3).to_string())
    print("\nFigures:")
    for f in sorted(outdir.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
