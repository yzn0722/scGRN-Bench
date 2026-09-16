#!/usr/bin/env python3
"""hESC chained_preds 专用分析图（仅依赖 chained_preds/ 内 CSV）。"""
from __future__ import annotations

import argparse
from pathlib import Path

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
DEFAULT_CHAINED = SCRIPT_DIR / "error_biology" / "multistep_pt" / "hESC" / "chained_preds"

TRANSITIONS = ["delta_t0_t1", "delta_t1_t2", "delta_t2_t3", "delta_t3_t4"]
TRANS_LABELS = ["S0→S1", "S1→S2", "S2→S3", "S3→S4"]
COLOR_CHAIN = model_color("scGPT")
COLOR_OK = "#43A047"
COLOR_BAD = "#E53935"
COLOR_UP = "#4EA3F1"
COLOR_DOWN = "#FF9A3D"


def load(chained_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    wide = pd.read_csv(chained_dir / "hESC_chained_per_gene_wide.csv")
    seg_sum = pd.read_csv(chained_dir / "hESC_chained_segment_summary.csv")
    tables = {
        t: pd.read_csv(chained_dir / f"hESC_{t}_gene_result.csv")
        for t in TRANSITIONS
    }
    return wide, seg_sum, tables


def plot_acc_dual(out: Path, seg_sum: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    """官方 top% acc vs 全基因方向准确率。"""
    gene_acc = [tables[t]["dir_correct"].mean() * 100 for t in TRANSITIONS]
    top_acc = (seg_sum.set_index("transition").loc[TRANSITIONS, "acc_top_percent"] * 100).tolist()

    x = np.arange(4)
    w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    b1 = ax.bar(x - w / 2, top_acc, w, label="Top 30% dynamic (official)", color=COLOR_CHAIN, alpha=0.85)
    b2 = ax.bar(x + w / 2, gene_acc, w, label="All genes (dir. correct)", color="#90A4AE", edgecolor="#546E7A")
    ax.set_xticks(x)
    ax.set_xticklabels(TRANS_LABELS)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Direction accuracy (%)")
    ax.set_title("hESC chained scGPT — segment accuracy (two metrics)", fontweight="600")
    ax.legend(loc="lower right", fontsize=9)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.0f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_error_persistence(out: Path, wide: pd.DataFrame) -> None:
    """四段中答对段数分布 + 转移矩阵。"""
    dir_cols = [f"{t}_dir_correct" for t in TRANSITIONS]
    wide = wide.copy()
    wide["n_correct"] = wide[dir_cols].sum(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1, 1.15]})

    # 左：n_correct 分布
    ax = axes[0]
    counts = wide["n_correct"].value_counts().sort_index()
    labels = [f"{k}/4 correct" for k in counts.index]
    colors = [COLOR_OK if k == 4 else (COLOR_BAD if k == 0 else "#FFB74D") for k in counts.index]
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", width=0.7)
    ax.set_ylabel("Genes (n=910)")
    ax.set_title("A  Direction hits across 4 chained blocks", fontweight="600", loc="left")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 8, str(v), ha="center", fontweight="600")
    ax.text(0.98, 0.92, f"Always correct: {int((wide['n_correct']==4).sum())}\n"
            f"Always wrong: {int((wide['n_correct']==0).sum())}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#F5F5F5"))

    # 右：段间转移 (correct -> correct / wrong)
    ax = axes[1]
    trans = np.zeros((2, 2))
    for i in range(3):
        a = wide[dir_cols[i]].astype(int)
        b = wide[dir_cols[i + 1]].astype(int)
        trans[0, 0] += ((a == 1) & (b == 1)).sum()
        trans[0, 1] += ((a == 1) & (b == 0)).sum()
        trans[1, 0] += ((a == 0) & (b == 1)).sum()
        trans[1, 1] += ((a == 0) & (b == 0)).sum()
    trans /= 3  # 平均到每对相邻段
    im = ax.imshow(trans, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Next: wrong", "Next: correct"])
    ax.set_yticklabels(["Prev: wrong", "Prev: correct"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{trans[i, j]:.0f}", ha="center", va="center", fontsize=14, fontweight="600",
                    color="white" if trans[i, j] > trans.max() * 0.55 else "#333")
    ax.set_title("B  Avg. gene flow between adjacent blocks", fontweight="600", loc="left")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="gene count (mean over 3 pairs)")

    fig.suptitle("hESC chained — error persistence along pseudotime", fontsize=13, fontweight="700", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_pred_bias(out: Path, tables: dict[str, pd.DataFrame]) -> None:
    """模型强烈偏向预测 Up。"""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    up_frac = []
    true_up = []
    for t in TRANSITIONS:
        df = tables[t]
        up_frac.append((df["dir_pred"] == "Up").mean() * 100)
        true_up.append((df["dir_true"] == "Up").mean() * 100)
    x = np.arange(4)
    w = 0.35
    ax.bar(x - w / 2, true_up, w, label="True Up fraction", color=COLOR_OK, alpha=0.8)
    ax.bar(x + w / 2, up_frac, w, label="Pred Up fraction", color=COLOR_CHAIN)
    ax.axhline(50, color="#999", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(TRANS_LABELS)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% genes labeled Up")
    ax.set_title("hESC chained — prediction bias toward Up", fontweight="600")
    ax.legend()
    ax.text(0.02, 0.95, "754/910 genes predicted Up in every block\n"
            "→ most Down→Up flips drive low S0→S1 accuracy",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#FFF3E0", edgecolor=COLOR_DOWN))
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_flip_breakdown(out: Path, tables: dict[str, pd.DataFrame]) -> None:
    """各段方向错误类型：Down→Up vs Up→Down。"""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    down2up, up2down = [], []
    for t in TRANSITIONS:
        wrong = tables[t][tables[t]["dir_correct"] == 0]
        down2up.append(((wrong["dir_true"] == "Down") & (wrong["dir_pred"] == "Up")).sum())
        up2down.append(((wrong["dir_true"] == "Up") & (wrong["dir_pred"] == "Down")).sum())
    x = np.arange(4)
    w = 0.4
    ax.bar(x, down2up, w, label="True Down → Pred Up", color=COLOR_BAD)
    ax.bar(x, up2down, w, bottom=down2up, label="True Up → Pred Down", color="#7986CB")
    ax.set_xticks(x)
    ax.set_xticklabels(TRANS_LABELS)
    ax.set_ylabel("Wrong-direction genes")
    ax.set_title("hESC chained — direction error composition", fontweight="600")
    ax.legend(loc="upper right")
    totals = [a + b for a, b in zip(down2up, up2down)]
    for i, (d, t) in enumerate(zip(down2up, totals)):
        if t:
            ax.text(i, t + 5, str(t), ha="center", fontweight="600")
            ax.text(i, d / 2, f"{100*d/t:.0f}%", ha="center", va="center", fontsize=9, color="white")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_summary(out: Path, wide: pd.DataFrame, seg_sum: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    lines = ["# hESC chained_preds analysis summary\n"]
    lines.append(f"- Genes: {len(wide)}\n")
    lines.append("- Per-segment direction accuracy (all genes):\n")
    for t, lab in zip(TRANSITIONS, TRANS_LABELS):
        acc = tables[t]["dir_correct"].mean()
        lines.append(f"  - {lab}: {acc*100:.1f}%\n")
    lines.append("- Official top-30% accuracy:\n")
    for _, r in seg_sum.iterrows():
        lines.append(f"  - {r['transition']}: {r['acc_top_percent']*100:.1f}%\n")
    dir_cols = [f"{t}_dir_correct" for t in TRANSITIONS]
    nc = wide[dir_cols].sum(axis=1)
    lines.append(f"- All 4 segments correct: {(nc==4).sum()} genes\n")
    lines.append(f"- All 4 segments wrong: {(nc==0).sum()} genes\n")
    c0 = wide["delta_t0_t1_dir_correct"]
    lines.append(f"- Wrong@S0→S1 fixed by later block: {((c0==0) & (wide[dir_cols[-1]]==1)).sum()}\n")
    out.write_text("".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--chained-dir", type=Path, default=DEFAULT_CHAINED)
    p.add_argument("--outdir", type=Path, default=None)
    args = p.parse_args()
    apply_fig4_style()
    outdir = args.outdir or (args.chained_dir / "figures")
    outdir.mkdir(parents=True, exist_ok=True)

    wide, seg_sum, tables = load(args.chained_dir)
    plot_acc_dual(outdir / "08_acc_top30_vs_allgenes.png", seg_sum, tables)
    plot_error_persistence(outdir / "09_error_persistence.png", wide)
    plot_pred_bias(outdir / "10_pred_up_bias.png", tables)
    plot_flip_breakdown(outdir / "11_flip_breakdown.png", tables)
    write_summary(outdir / "ANALYSIS_SUMMARY.md", wide, seg_sum, tables)
    print(f"Saved chained analysis figures to {outdir}")


if __name__ == "__main__":
    main()
