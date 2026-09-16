#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hESC 链式 scGPT 分段预测 — 形象说明「单步错、峰值段可救」等问题。

输入（默认）：
  error_biology/multistep_pt/hESC/chained_preds/
  error_biology/multistep_pt/hESC/gene_segment_deltas.csv
  pre_scgpt .../hESC_gene_result.csv

输出（默认写入 chained_preds/figures/）：
  segment_accuracy.png, delta_scatter_by_segment.png, direction_heatmap.png, ...

示例：
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig4
  python3 plot_hESC_chained_problem_story.py
  python3 plot_hESC_chained_problem_story.py --chained-dir error_biology/multistep_pt/hESC/chained_preds
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

try:
    from fig4_palette import apply_fig4_style, model_color
except ImportError:

    def apply_fig4_style() -> None:
        plt.rcParams.update({"font.size": 11})

    def model_color(_: str, d: str = "#666") -> str:
        return d


SCRIPT_DIR = Path(__file__).resolve().parent
HESC_DIR = SCRIPT_DIR / "error_biology" / "multistep_pt" / "hESC"
DEFAULT_CHAINED = HESC_DIR / "chained_preds"
GENE_SEG = HESC_DIR / "gene_segment_deltas.csv"
SINGLE_DIR = Path("/mnt/10T/yzn/benchmark_GRN/pre_scgpt/results_multidataset_pseudotime_227")

TRANSITIONS = ["delta_t0_t1", "delta_t1_t2", "delta_t2_t3", "delta_t3_t4"]
TRANS_LABELS = ["S0→S1", "S1→S2", "S2→S3", "S3→S4"]
PT_NODES = ["S0", "S1", "S2", "S3", "S4"]
N_SEG = 5
TOP_PERCENT = 0.3

COLOR_TRUE = model_color("scPrint")
COLOR_CHAIN = model_color("scGPT")
COLOR_SINGLE = "#9E9E9E"
COLOR_PEAK = model_color("scFoundation")
COLOR_OK = "#43A047"
COLOR_BAD = "#E53935"
SEG_BG = ["#EEF4FA", "#D4E6F5", "#9ECAE8", "#6BAED6", "#2171B5"]


def dir_sign(x: float) -> str:
    if not np.isfinite(x) or x == 0:
        return "Down"
    return "Up" if x > 0 else "Down"


def early_local(ch: pd.DataFrame, gene: str) -> float:
    if gene not in ch.index:
        return float("nan")
    r = ch.loc[gene]
    if "true_early_mean_local" in ch.columns:
        return float(r["true_early_mean_local"])
    if "true_early_mean" in ch.columns:
        return float(r["true_early_mean"])
    return float(r["true_late_mean"]) - float(r["delta_true_local"])


def load_segment_tables(chained_dir: Path, dataset: str = "hESC") -> Dict[str, pd.DataFrame]:
    out = {}
    for t in TRANSITIONS:
        p = chained_dir / f"{dataset}_{t}_gene_result.csv"
        if p.exists():
            out[t] = pd.read_csv(p).set_index("gene")
    return out


def build_analytics(
    gf: pd.DataFrame,
    gr: pd.DataFrame,
    wide: pd.DataFrame,
    seg_tables: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_top = max(1, int(np.ceil(TOP_PERCENT * len(gf))))
    top = gf.assign(_abs=gf["delta_single"].abs()).nlargest(n_top, "_abs")
    top = top.merge(
        gr[["gene", "dir_pred", "dir_correct"]].rename(
            columns={"dir_pred": "dir_pred_single", "dir_correct": "dir_correct_single"}
        ),
        on="gene",
        how="left",
    )
    top["dir_true_peak"] = top["delta_best_trans"].apply(dir_sign)

    wrong = top[top["dir_correct_single"] == 0].copy()
    ch_s0 = seg_tables.get("delta_t0_t1")
    rows = []
    for _, r in wrong.iterrows():
        g = str(r["gene"])
        trans = str(r["best_transition"])
        ch = seg_tables.get(trans)
        if ch is None or g not in ch.index:
            continue
        pred_late = float(ch.loc[g, "pred_late_like_mean"])
        el = early_local(ch, g)
        fixed = dir_sign(pred_late - el) == r["dir_true_peak"] if np.isfinite(el) else False
        seg_acc = {}
        for t in TRANSITIONS:
            col = f"{t}_dir_correct"
            seg_acc[t] = int(wide.loc[wide["gene"] == g, col].iloc[0]) if col in wide.columns else np.nan
        rows.append(
            {
                "gene": g,
                "is_chip_target": bool(r["is_chip_target"]),
                "peak_middle": bool(r["peak_middle"]),
                "best_transition": trans,
                "dir_true_peak": r["dir_true_peak"],
                "dir_pred_single": r["dir_pred_single"],
                "abs_delta_best": abs(float(r["delta_best_trans"])),
                "fixed_chained_peak": fixed,
                **{f"ok_{t}": seg_acc[t] for t in TRANSITIONS},
            }
        )
    wrong_eval = pd.DataFrame(rows)

    return top, wrong, wrong_eval, compute_seg_summary(seg_tables)


def compute_seg_summary(seg_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for t in TRANSITIONS:
        if t not in seg_tables:
            continue
        df = seg_tables[t]
        rows.append(
            {
                "transition": t,
                "label": TRANS_LABELS[TRANSITIONS.index(t)],
                "acc_all": float(df["dir_correct"].mean()),
            }
        )
    return pd.DataFrame(rows)


def chained_profiles(gene: str, seg_tables: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
    """5 个伪时间点：真值轨迹 + 链式预测轨迹。"""
    t0 = seg_tables["delta_t0_t1"]
    if gene not in t0.index:
        return np.full(N_SEG, np.nan), np.full(N_SEG, np.nan)
    true = [float(t0.loc[gene, "true_baseline_mean"])]
    pred = [true[0]]
    for t in TRANSITIONS:
        df = seg_tables[t]
        true.append(float(df.loc[gene, "true_late_mean"]))
        pred.append(float(df.loc[gene, "pred_late_like_mean"]))
    return np.array(true), np.array(pred)


def single_interp(gene: str, gr: pd.DataFrame, n: int = N_SEG) -> Optional[np.ndarray]:
    if gene not in gr.index:
        return None
    e = float(gr.loc[gene, "true_early_mean"])
    p = float(gr.loc[gene, "pred_late_like_mean"])
    return np.array([e + (k / (n - 1)) * (p - e) for k in range(n)])


def norm_profile(y: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(y)):
        return y
    lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
    if hi - lo < 1e-9:
        return np.zeros_like(y)
    return (y - lo) / (hi - lo)


def pick_spotlight_genes(wrong_eval: pd.DataFrame) -> List[str]:
    """自动选 3 个有故事性的基因。"""
    if wrong_eval.empty:
        return ["ERCC-00113", "NRP1", "SOX2"]
    w = wrong_eval.copy()
    picks: List[str] = []

    rescued = w[w["fixed_chained_peak"]].sort_values("abs_delta_best", ascending=False)
    if len(rescued):
        nonchip = rescued[~rescued["is_chip_target"]]
        picks.append(str((nonchip if len(nonchip) else rescued).iloc[0]["gene"]))

    stuck_mid = w[(~w["fixed_chained_peak"]) & w["peak_middle"]]
    if len(stuck_mid):
        picks.append(str(stuck_mid.sort_values("abs_delta_best", ascending=False).iloc[0]["gene"]))

    stuck_end = w[(~w["fixed_chained_peak"]) & (~w["peak_middle"])]
    if len(stuck_end):
        picks.append(str(stuck_end.iloc[0]["gene"]))

    for g in ["ERCC-00113", "CALB1", "SOX2", "NRP1"]:
        if g not in picks and g in w["gene"].astype(str).values:
            picks.append(g)
        if len(picks) >= 3:
            break
    return picks[:3]


def plot_overview(
    out_path: Path,
    seg_summary: pd.DataFrame,
    top: pd.DataFrame,
    wrong_eval: pd.DataFrame,
    wide: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

    # --- A: 伪时间示意 + 分段准确率 ---
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.set_title("A  Chain-of-thought scGPT along pseudotime", fontweight="600", loc="left")

    seg_w = 1.55
    x0 = 0.6
    for i, (lab, col) in enumerate(zip(PT_NODES, SEG_BG)):
        x = x0 + i * seg_w
        rect = FancyBboxPatch(
            (x, 2.0), seg_w * 0.88, 0.9,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=col, edgecolor="#333", linewidth=0.8,
        )
        ax.add_patch(rect)
        ax.text(x + seg_w * 0.44, 2.45, lab, ha="center", va="center", fontsize=10, fontweight="600")
        if i < 4 and not seg_summary.empty:
            acc_row = seg_summary[seg_summary["transition"] == TRANSITIONS[i]]
            if len(acc_row):
                acc = float(acc_row.iloc[0]["acc_all"]) * 100
                ax.text(
                    x + seg_w * 0.44, 1.55,
                    f"{TRANS_LABELS[i]}\n{acc:.0f}% dir.",
                    ha="center", va="center", fontsize=9,
                    color="#1A1A1A",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#CCC", alpha=0.95),
                )
    ax.annotate("", xy=(9.0, 2.45), xytext=(0.5, 2.45),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=2))
    ax.text(4.8, 3.35, "developmental pseudotime →", ha="center", fontsize=11, style="italic")
    ax.text(
        0.05, 0.35,
        "One 16-step run, split into 4 blocks.\n"
        "Each block predicts Δ from segment start Si,\n"
        "then feeds pred into the next block.",
        fontsize=9, va="bottom", color="#444",
    )

    # --- B: 问题漏斗 ---
    ax = fig.add_subplot(gs[0, 1])
    n_top = len(top)
    n_wrong = len(wrong_eval) if len(wrong_eval) else int((top["dir_correct_single"] == 0).sum())
    n_fixed = int(wrong_eval["fixed_chained_peak"].sum()) if len(wrong_eval) else 0
    n_mid_wrong = int(wrong_eval["peak_middle"].sum()) if len(wrong_eval) else 0

    stages = [
        f"Top 30% dynamic\n(n={n_top})",
        f"Single-step\nwrong direction\n(n={n_wrong})",
        f"Chained fixes\nat |Δ| peak\n(n={n_fixed})",
        f"Still wrong\n(n={n_wrong - n_fixed})",
    ]
    vals = [n_top, n_wrong, n_fixed, n_wrong - n_fixed]
    colors = ["#B0BEC5", COLOR_BAD, COLOR_OK, "#FF8A65"]
    ypos = np.arange(len(stages))[::-1]
    bars = ax.barh(ypos, vals, color=colors, edgecolor="white", height=0.62)
    ax.set_yticks(ypos)
    ax.set_yticklabels(stages)
    ax.set_xlabel("Gene count")
    ax.set_title("B  Where does single-step fail?", fontweight="600", loc="left")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + max(n_top * 0.02, 2), bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=11, fontweight="600")
    pct = 100 * n_fixed / n_wrong if n_wrong else 0
    ax.text(0.98, 0.06, f"{pct:.0f}% of wrong-single\nrescued at peak segment",
            transform=ax.transAxes, ha="right", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="#E8F5E9", edgecolor=COLOR_OK))
    ax.text(0.98, 0.92, f"{n_mid_wrong}/{n_wrong} peak in\nmiddle segments",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#FFF8E1", edgecolor=COLOR_PEAK))

    # --- C: 错基因 × 四段方向热图 ---
    ax = fig.add_subplot(gs[1, 0])
    if len(wrong_eval):
        show = wrong_eval.nlargest(min(28, len(wrong_eval)), "abs_delta_best")
        mat = show[[f"ok_{t}" for t in TRANSITIONS]].astype(float).values
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_yticks(np.arange(len(show)))
        ax.set_yticklabels(show["gene"].astype(str), fontsize=7)
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(TRANS_LABELS, rotation=25, ha="right")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, "✓" if mat[i, j] > 0.5 else "✗",
                        ha="center", va="center", fontsize=8,
                        color="white" if mat[i, j] < 0.5 else "#1B5E20", fontweight="bold")
        # 峰值列高亮
        peak_col = [TRANSITIONS.index(t) if t in TRANSITIONS else -1 for t in show["best_transition"]]
        for i, j in enumerate(peak_col):
            if j >= 0:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=COLOR_PEAK, lw=2.5))
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="dir. correct")
    ax.set_title("C  Single-wrong genes: chained direction per segment\n(gold box = |Δ| peak)", fontweight="600", loc="left")

    # --- D: 峰值在中间 vs 两端 ---
    ax = fig.add_subplot(gs[1, 1])
    if len(wrong_eval):
        mid = int(wrong_eval["peak_middle"].sum())
        end = len(wrong_eval) - mid
        fixed_mid = int(wrong_eval[wrong_eval["peak_middle"]]["fixed_chained_peak"].sum())
        fixed_end = int(wrong_eval[~wrong_eval["peak_middle"]]["fixed_chained_peak"].sum())
        x = np.arange(2)
        w = 0.35
        ax.bar(x - w / 2, [mid, end], width=w, color="#90CAF9", label="Wrong-single count")
        ax.bar(x + w / 2, [fixed_mid, fixed_end], width=w, color=COLOR_OK, label="Fixed @ peak (chained)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Peak in middle\n(S1–S3, n={mid})", f"Peak at ends\n(S0↔S4, n={end})"])
        ax.set_ylabel("Genes")
        ax.legend(loc="upper right", fontsize=9)
        for i, (a, b) in enumerate([(mid, fixed_mid), (end, fixed_end)]):
            if a:
                ax.text(i - w / 2, a + 0.8, str(a), ha="center", fontsize=10)
                ax.text(i + w / 2, b + 0.8, f"{b}\n({100*b/a:.0f}%)", ha="center", fontsize=9)
    ax.set_title("D  Rescue rate depends on where dynamics peak", fontweight="600", loc="left")

    fig.suptitle(
        "hESC — chained segment scGPT explains mid-trajectory errors",
        fontsize=15, fontweight="700", y=0.98,
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_spotlight_genes(
    out_path: Path,
    genes: List[str],
    seg_tables: Dict[str, pd.DataFrame],
    gr: pd.DataFrame,
    gf: pd.DataFrame,
    wrong_eval: pd.DataFrame,
) -> None:
    gf_idx = gf.set_index("gene")
    w_idx = wrong_eval.set_index("gene") if len(wrong_eval) and "gene" in wrong_eval.columns else pd.DataFrame()

    titles = {
        "rescued": "Peak mid-trajectory · single wrong → chained correct",
        "stuck_mid": "Peak in middle · still wrong after chaining",
        "stuck_end": "Peak at trajectory end · single & chained disagree with biology",
    }
    roles = ["rescued", "stuck_mid", "stuck_end"]

    fig, axes = plt.subplots(len(genes), 1, figsize=(11, 3.6 * len(genes)), squeeze=False)
    x = np.arange(N_SEG)

    for ax, gene, role in zip(axes[:, 0], genes, roles[: len(genes)]):
        true_y, pred_chain = chained_profiles(gene, seg_tables)
        pred_single = single_interp(gene, gr.set_index("gene"))
        if not np.all(np.isfinite(true_y)):
            ax.set_visible(False)
            continue

        stack = [true_y]
        if pred_single is not None:
            stack.append(pred_single)
        stack.append(pred_chain)
        allv = np.concatenate(stack)
        vmin, vmax = float(np.nanmin(allv)), float(np.nanmax(allv))
        scale = lambda a: (a - vmin) / (vmax - vmin + 1e-9)

        ax.fill_between(x, scale(true_y), alpha=0.12, color=COLOR_TRUE)
        ax.plot(x, scale(true_y), "o-", color=COLOR_TRUE, lw=2.8, ms=10, label="Observed (binned mean)", zorder=4)
        if pred_single is not None:
            ax.plot(x, scale(pred_single), "s--", color=COLOR_SINGLE, lw=2, ms=7, alpha=0.85, label="Single 20%→20%")
        ax.plot(x, scale(pred_chain), "^-", color=COLOR_CHAIN, lw=2.6, ms=9, label="Chained scGPT (per segment)")

        if gene in gf_idx.index:
            trans = str(gf_idx.loc[gene, "best_transition"])
            peak_i = TRANSITIONS.index(trans) + 1 if trans in TRANSITIONS else 2
            ax.scatter([peak_i], [scale(true_y)[peak_i]], s=320, marker="*", c=COLOR_PEAK,
                       edgecolors="white", linewidths=1, zorder=6, label="|Δ| peak")
            mid = bool(gf_idx.loc[gene, "peak_middle"])
            chip = bool(gf_idx.loc[gene, "is_chip_target"])
        else:
            peak_i, mid, chip = 2, False, False

        fixed = bool(w_idx.loc[gene, "fixed_chained_peak"]) if gene in w_idx.index else False
        status = "FIXED at peak" if fixed else "still WRONG"
        tag = "CHIP TF target" if chip else "Non-CHIP"
        note = titles.get(role, "")
        ax.set_xticks(x)
        ax.set_xticklabels(PT_NODES)
        ax.set_ylabel("Scaled expression")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
        ax.set_title(f"{gene}  ·  {tag}  ·  {status}\n{note}", fontweight="600", loc="left", fontsize=11)
        ax.grid(axis="y", alpha=0.25)

        # 箭头标注单步错方向
        if gene in w_idx.index:
            d_single = str(w_idx.loc[gene, "dir_pred_single"])
            d_true = str(w_idx.loc[gene, "dir_true_peak"])
            ax.text(
                0.99, 0.04,
                f"Single pred: {d_single}  ·  True @ peak: {d_true}\n"
                f"{'Mid' if mid else 'End'} segment peak ({TRANS_LABELS[peak_i - 1] if peak_i > 0 else '?'})",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="#DDD"),
            )

    fig.suptitle("hESC gene spotlights — why endpoint-only prediction fails", fontsize=14, fontweight="700", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_peak_delta_scatter(
    out_path: Path,
    wrong_eval: pd.DataFrame,
    seg_tables: Dict[str, pd.DataFrame],
) -> None:
    if wrong_eval.empty:
        return
    xs, ys, colors, sizes = [], [], [], []
    for _, r in wrong_eval.iterrows():
        g = str(r["gene"])
        ch = seg_tables.get(str(r["best_transition"]))
        if ch is None or g not in ch.index:
            continue
        xs.append(float(ch.loc[g, "delta_true_local"]))
        ys.append(float(ch.loc[g, "delta_pred"]))
        colors.append(COLOR_OK if r["fixed_chained_peak"] else COLOR_BAD)
        sizes.append(30 + 80 * min(r["abs_delta_best"] / 6.0, 1.0))

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(xs, ys, c=colors, s=sizes, alpha=0.75, edgecolors="white", linewidths=0.6)
    lim = max(max(map(abs, xs + ys), default=1), 1)
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, alpha=0.4)
    ax.axhline(0, color="#999", lw=0.8)
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlabel("True Δ at peak segment (local Si→Sj)")
    ax.set_ylabel("Chained predicted Δ at same segment")
    ax.set_title("hESC — single-wrong genes at their |Δ| peak transition", fontweight="600")
    ax.legend(
        handles=[
            mpatches.Patch(color=COLOR_OK, label="Direction fixed"),
            mpatches.Patch(color=COLOR_BAD, label="Still wrong"),
        ],
        loc="upper left",
    )
    n_fix = int(wrong_eval["fixed_chained_peak"].sum())
    ax.text(
        0.98, 0.02, f"{n_fix}/{len(wrong_eval)} above diagonal quadrant match true sign",
        transform=ax.transAxes, ha="right",
        bbox=dict(boxstyle="round", facecolor="#F5F5F5"),
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_segment_accuracy(out_path: Path, seg_summary: pd.DataFrame) -> None:
    """仅用 chained_preds 内 segment_summary + 各段 gene_result。"""
    if seg_summary.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = seg_summary["label"].tolist() if "label" in seg_summary.columns else TRANS_LABELS
    acc = (seg_summary["acc_all"] * 100).tolist()
    bars = ax.bar(labels, acc, color=COLOR_CHAIN, edgecolor="white", width=0.65)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Direction accuracy (%)")
    ax.set_title("hESC chained scGPT — per-segment direction accuracy", fontweight="600")
    for b, v in zip(bars, acc):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=11, fontweight="600")
    ax.axhline(76, color=COLOR_SINGLE, ls="--", lw=1.2, alpha=0.7, label="S0→S1 (lowest)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_delta_scatter_by_segment(out_path: Path, seg_tables: Dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, t, lab in zip(axes.flat, TRANSITIONS, TRANS_LABELS):
        df = seg_tables[t]
        x = df["delta_true_local"].astype(float)
        y = df["delta_pred"].astype(float)
        ok = df["dir_correct"].astype(int) > 0
        ax.scatter(x[~ok], y[~ok], c=COLOR_BAD, s=8, alpha=0.35, linewidths=0)
        ax.scatter(x[ok], y[ok], c=COLOR_OK, s=8, alpha=0.35, linewidths=0)
        lim = max(float(np.abs(x).max()), float(np.abs(y).max()), 1)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.35)
        ax.axhline(0, color="#AAA", lw=0.6)
        ax.axvline(0, color="#AAA", lw=0.6)
        acc = float(ok.mean()) * 100
        ax.set_title(f"{lab}  (acc {acc:.0f}%)", fontweight="600")
        ax.set_xlabel("True Δ (local)")
        ax.set_ylabel("Pred Δ")
    fig.suptitle("hESC chained — true vs predicted Δ per pseudotime block", fontsize=13, fontweight="700", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_direction_heatmap_wide(out_path: Path, wide: pd.DataFrame, seg_tables: Dict[str, pd.DataFrame]) -> None:
    """按各段 |delta_true| 选 top 动态基因，展示四段方向对错。"""
    scores = np.zeros(len(wide))
    for t in TRANSITIONS:
        df = seg_tables[t]
        g2d = df["delta_true"].abs().to_dict()
        for i, g in enumerate(wide["gene"].astype(str)):
            scores[i] = max(scores[i], g2d.get(g, 0))
    wide = wide.assign(_score=scores).nlargest(35, "_score")
    mat = wide[[f"{t}_dir_correct" for t in TRANSITIONS]].astype(float).values
    genes = wide["gene"].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(6.5, 10))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(TRANS_LABELS)
    ax.set_yticks(np.arange(len(genes)))
    ax.set_yticklabels(genes, fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, "✓" if mat[i, j] > 0.5 else "✗", ha="center", va="center", fontsize=7,
                    color="white" if mat[i, j] < 0.5 else "#1B5E20", fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax.set_title("Top dynamic genes — chained direction correct per segment", fontweight="600")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_ercc_story(
    out_path: Path,
    wrong_eval: pd.DataFrame,
) -> None:
    """ERCC spike-in：非 CHIP、多在 S1→S2 峰值，链式常能纠正单步方向。"""
    ercc = wrong_eval[wrong_eval["gene"].astype(str).str.startswith("ERCC")].copy()
    if ercc.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ercc = ercc.sort_values("abs_delta_best", ascending=True)
    y = np.arange(len(ercc))
    colors = [COLOR_OK if f else COLOR_BAD for f in ercc["fixed_chained_peak"]]
    ax.barh(y, ercc["abs_delta_best"], color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(ercc["gene"].astype(str), fontsize=8)
    ax.set_xlabel("|Δ| at peak segment (true)")
    ax.set_title(
        f"hESC ERCC controls (n={len(ercc)}) — spike-in dynamics peak in mid-trajectory\n"
        "Green = chained fixes single-step direction at peak",
        fontweight="600",
    )
    n_fix = int(ercc["fixed_chained_peak"].sum())
    ax.text(0.98, 0.95, f"{n_fix}/{len(ercc)} fixed", transform=ax.transAxes,
            ha="right", va="top", fontsize=12, fontweight="600",
            bbox=dict(boxstyle="round", facecolor="#E8F5E9"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="hESC chained scGPT problem-story figures")
    p.add_argument("--chained-dir", type=Path, default=DEFAULT_CHAINED)
    p.add_argument("--outdir", type=Path, default=None, help="default: <chained-dir>/figures")
    p.add_argument("--genes", type=str, default="", help="comma-separated spotlight genes")
    args = p.parse_args()

    apply_fig4_style()
    outdir = args.outdir or (args.chained_dir / "figures")
    outdir.mkdir(parents=True, exist_ok=True)

    wide = pd.read_csv(args.chained_dir / "hESC_chained_per_gene_wide.csv")
    seg_tables = load_segment_tables(args.chained_dir, "hESC")
    if len(seg_tables) < 4:
        raise FileNotFoundError(f"Need 4 segment gene_result CSVs under {args.chained_dir}")

    seg_summary = compute_seg_summary(seg_tables)

    # --- 仅 chained_preds 即可生成的图 ---
    plot_segment_accuracy(outdir / "01_segment_accuracy.png", seg_summary)
    plot_delta_scatter_by_segment(outdir / "02_delta_scatter_by_segment.png", seg_tables)
    plot_direction_heatmap_wide(outdir / "03_direction_heatmap_top35.png", wide, seg_tables)
    seg_summary.to_csv(outdir / "segment_accuracy.csv", index=False)

    has_extra = GENE_SEG.exists() and (SINGLE_DIR / "hESC_gene_result.csv").exists()
    if has_extra:
        gf = pd.read_csv(GENE_SEG)
        gr = pd.read_csv(SINGLE_DIR / "hESC_gene_result.csv")
        top, wrong, wrong_eval, seg_summary = build_analytics(gf, gr, wide, seg_tables)
        wrong_eval.to_csv(outdir / "wrong_single_vs_chained_peak.csv", index=False)
        plot_overview(outdir / "04_problem_overview.png", seg_summary, top, wrong_eval, wide)
        plot_peak_delta_scatter(outdir / "05_peak_delta_wrong_single.png", wrong_eval, seg_tables)
        plot_ercc_story(outdir / "06_ERCC_rescue.png", wrong_eval)
        if args.genes.strip():
            spotlight = [g.strip() for g in args.genes.split(",") if g.strip()]
        else:
            spotlight = pick_spotlight_genes(wrong_eval)
        plot_spotlight_genes(
            outdir / "07_spotlight_genes.png",
            spotlight,
            seg_tables,
            gr,
            gf,
            wrong_eval,
        )
        n_wrong = len(wrong_eval)
        n_fix = int(wrong_eval["fixed_chained_peak"].sum()) if n_wrong else 0
        print("\n=== hESC chained_preds figures ===")
        print(f"Data: {args.chained_dir}")
        print(f"Top 30% dynamic: {len(top)} | single wrong: {n_wrong} | chained fix @ peak: {n_fix}")
        print(f"Spotlight: {', '.join(spotlight)}")
    else:
        # 无单步对照时，按链式四段全错基因选 spotlight
        err_cnt = wide[[f"{t}_dir_correct" for t in TRANSITIONS]].eq(0).sum(axis=1)
        spotlight = wide.loc[err_cnt.nlargest(3).index, "gene"].astype(str).tolist()
        plot_spotlight_genes(
            outdir / "07_spotlight_genes.png",
            spotlight,
            seg_tables,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
        print("\n=== hESC chained_preds figures (chained only) ===")
        print(f"Data: {args.chained_dir}")
        print(f"Spotlight (most segment errors): {', '.join(spotlight)}")

    print(f"\nSaved under: {outdir}/")
    for f in sorted(outdir.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
