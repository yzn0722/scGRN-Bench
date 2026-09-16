#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案3（简化版）：显式非线性参照指标 — 无需重跑 scGPT。

对每条真值 5 点轨迹 (S0…S4) 计算三类非线性特征，并检验 scGPT 预测轨迹的偏差：
  1. 单调性是否与真值一致
  2. 峰值 |Δ| 是否落在中间段（非首尾）
  3. 中段斜率是否显著大于两端（mid/end slope ratio）

预测模式：Independent（真实 Si 拼接）、Chained、Single（S0↔S4 线性插值）

数据来源（已有 CSV，不必重跑模型）：
  chained_preds/hESC_delta_t*_gene_result.csv
  segment_preds/scgpt/hESC_seg*_gene_result.csv
  pre_scgpt/.../hESC_gene_result.csv

若要做「完整版」二次拟合参照，见输出目录 SCHEME3_rerun_guide.md。

  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig4
  python3 plot_hESC_scheme3_nonlinear_indicators.py
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
    INDEP_DIR,
    N_PTS,
    PT_NODES,
    TRANSITIONS,
    TRANS_LABELS,
    chain_profile,
    indep_profile,
    load_chained_tables,
    load_indep_tables,
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
OUT_SUB = "scheme3_nonlinear_ref"
MID_SLOPE_RATIO_THR = 1.5  # 中段斜率 / 端点斜率 > 此阈值 → 「中段主导」

COLOR_INDEP = model_color("scFoundation")
COLOR_CHAIN = model_color("scGPT")
COLOR_SINGLE = "#9E9E9E"
MODES = ["indep", "chain", "single"]
MODE_LABELS = {"indep": "Independent", "chain": "Chained", "single": "Single S0↔S4"}
MODE_COLORS = {"indep": COLOR_INDEP, "chain": COLOR_CHAIN, "single": COLOR_SINGLE}


def single_profile(gene: str, gr: pd.DataFrame, true_y: np.ndarray) -> np.ndarray:
    if gene not in gr.index or not np.all(np.isfinite(true_y)):
        return np.full(N_PTS, np.nan)
    e = float(gr.loc[gene, "true_early_mean"])
    p = float(gr.loc[gene, "pred_late_like_mean"])
    return np.array([e + (k / (N_PTS - 1)) * (p - e) for k in range(N_PTS)])


# ---------------------------------------------------------------------------
# 三类非线性参照指标（仅依赖 5 点表达轨迹）
# ---------------------------------------------------------------------------

def segment_slopes(y: np.ndarray) -> np.ndarray:
    """4 段斜率 Δ_i = y_{i+1} - y_i。"""
    y = np.asarray(y, dtype=float)
    if len(y) < 2 or not np.all(np.isfinite(y)):
        return np.full(4, np.nan)
    return np.diff(y)


def monotonicity_class(y: np.ndarray) -> str:
    """
    up     : 全程非降
    down   : 全程非增
    mixed  : 存在换向（非线性）
    flat   : 几乎不变
    """
    d = segment_slopes(y)
    if not np.all(np.isfinite(d)):
        return "unknown"
    pos, neg = (d > 1e-8).sum(), (d < -1e-8).sum()
    if pos > 0 and neg > 0:
        return "mixed"
    if pos > 0 and neg == 0:
        return "up"
    if neg > 0 and pos == 0:
        return "down"
    return "flat"


def peak_segment_index(slopes: np.ndarray) -> int:
    """|斜率|最大的段索引 0..3。"""
    if not np.all(np.isfinite(slopes)) or len(slopes) == 0:
        return -1
    return int(np.argmax(np.abs(slopes)))


def peak_not_at_ends(slopes: np.ndarray) -> bool:
    """峰值在中间两段 S1→S2 或 S2→S3（索引 1 或 2）。"""
    k = peak_segment_index(slopes)
    return k in (1, 2)


def mid_vs_end_slope_ratio(slopes: np.ndarray) -> float:
    """中段 |斜率| 均值 / 端点 |斜率| 均值。"""
    if not np.all(np.isfinite(slopes)) or len(slopes) < 4:
        return np.nan
    end = np.mean(np.abs(slopes[[0, 3]]))
    mid = np.mean(np.abs(slopes[[1, 2]]))
    return float(mid / (end + 1e-9))


def mid_slope_dominant(slopes: np.ndarray, thr: float = MID_SLOPE_RATIO_THR) -> bool:
    r = mid_vs_end_slope_ratio(slopes)
    return bool(np.isfinite(r) and r >= thr)


def extract_nl_features(y: np.ndarray, thr: float = MID_SLOPE_RATIO_THR) -> Dict:
    sl = segment_slopes(y)
    return {
        "mono_class": monotonicity_class(y),
        "peak_seg": peak_segment_index(sl),
        "peak_mid": peak_not_at_ends(sl),
        "mid_end_ratio": mid_vs_end_slope_ratio(sl),
        "mid_dominant": mid_slope_dominant(sl, thr),
    }


def compare_to_true(true_y: np.ndarray, pred_y: np.ndarray, thr: float = MID_SLOPE_RATIO_THR) -> Dict:
    """预测相对真值在三类指标上的偏差（0=一致，1=不一致）。"""
    t = extract_nl_features(true_y, thr)
    p = extract_nl_features(pred_y, thr)
    return {
        "mono_match": int(t["mono_class"] == p["mono_class"]),
        "mono_dev": int(t["mono_class"] != p["mono_class"]),
        "peak_seg_match": int(t["peak_seg"] == p["peak_seg"]),
        "peak_mid_match": int(t["peak_mid"] == p["peak_mid"]),
        "peak_mid_dev": int(t["peak_mid"] != p["peak_mid"]),
        "mid_dom_match": int(t["mid_dominant"] == p["mid_dominant"]),
        "mid_dom_dev": int(t["mid_dominant"] != p["mid_dominant"]),
        "mid_ratio_gap": abs(t["mid_end_ratio"] - p["mid_end_ratio"])
        if np.isfinite(t["mid_end_ratio"]) and np.isfinite(p["mid_end_ratio"])
        else np.nan,
        "true_mono": t["mono_class"],
        "pred_mono": p["mono_class"],
        "true_peak_mid": int(t["peak_mid"]),
        "pred_peak_mid": int(p["peak_mid"]),
        "true_mid_dom": int(t["mid_dominant"]),
        "pred_mid_dom": int(p["mid_dominant"]),
        "true_mid_ratio": t["mid_end_ratio"],
        "pred_mid_ratio": p["mid_end_ratio"],
    }


def build_full_table(
    genes: List[str],
    ch: Dict,
    ind: Dict,
    gr: Optional[pd.DataFrame],
    thr: float,
) -> pd.DataFrame:
    rows = []
    for g in genes:
        ty = true_profile(ch, g)
        if not np.all(np.isfinite(ty)):
            continue
        tf = extract_nl_features(ty, thr)
        base = {"gene": g, **{f"true_{k}": v for k, v in tf.items()}}
        preds = {
            "indep": indep_profile(ind, ch, g),
            "chain": chain_profile(ch, g),
            "single": single_profile(g, gr, ty) if gr is not None else np.full(N_PTS, np.nan),
        }
        for mode, py in preds.items():
            if not np.all(np.isfinite(py)):
                continue
            cmp = compare_to_true(ty, py, thr)
            rows.append({**base, "mode": mode, **cmp})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_concept(out: Path, thr: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.set_title(
        "Scheme 3 (simplified): explicit nonlinear reference checks on 5-point trajectories",
        fontweight="700", fontsize=12, loc="left",
    )
    items = [
        (0.4, "① Monotonicity", "True vs pred:\nup / down / mixed / flat\nmust match class"),
        (3.5, "② Peak not at ends", "argmax |Δ| on S1→S2\nor S2→S3 (not S0→S1 / S3→S4)"),
        (6.4, f"③ Mid > ends slope", f"mean|Δ|_{'{mid}'}/mean|Δ|_{'{end}'}\n≥ {thr} → mid-dominant"),
    ]
    for x0, title, desc in items:
        ax.text(x0, 2.0, title, fontsize=11, fontweight="700")
        ax.text(x0, 1.2, desc, fontsize=9, color="#444")
    ax.text(5, 0.35, "No parametric fit — auditable rules on bin-mean pseudotime profiles", ha="center", fontsize=9, style="italic")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_agreement_rates(out: Path, df: pd.DataFrame) -> None:
    """三指标 × 三模式 的一致率。"""
    metrics = [
        ("mono_match", "Monotonicity class match"),
        ("peak_mid_match", "Peak-not-at-ends match"),
        ("mid_dom_match", "Mid-slope-dominant match"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, (col, title) in zip(axes, metrics):
        rates = []
        for mode in MODES:
            sub = df[df["mode"] == mode]
            rates.append(100 * sub[col].mean() if len(sub) else 0)
        bars = ax.bar([MODE_LABELS[m] for m in MODES], rates,
                      color=[MODE_COLORS[m] for m in MODES], edgecolor="white", width=0.65)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Agreement with true (%)")
        ax.set_title(title, fontweight="600", fontsize=10)
        ax.axhline(50, color="#999", ls=":", lw=0.8)
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, rates):
            ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%", ha="center", fontsize=9, fontweight="600")
    fig.suptitle("scGPT trajectory vs simplified nonlinear reference (hESC, n genes per mode)", fontsize=12, fontweight="700", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_stratified_by_true_nl(out: Path, df: pd.DataFrame) -> None:
    """仅在真值满足某非线性条件时，看预测一致率。"""
    base = df[df["mode"] == "indep"]
    strata = [
        (lambda b: b["true_mono"] == "mixed", "True: non-monotone (mixed)"),
        (lambda b: b["true_peak_mid"] == 1, "True: peak in middle segments"),
        (lambda b: b["true_mid_dom"] == 1, f"True: mid/end slope ≥ {MID_SLOPE_RATIO_THR}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x = np.arange(len(MODES))
    w = 0.25
    for ax, (filt, title) in zip(axes, strata):
        sub_true = base[filt(base)]
        genes = sub_true["gene"].astype(str).tolist()
        if not genes:
            ax.set_title(title + "\n(n=0)", fontsize=9)
            continue
        for j, (col, lab) in enumerate([
            ("mono_match", "Mono"),
            ("peak_mid_match", "Peak"),
            ("mid_dom_match", "Mid slope"),
        ]):
            vals = []
            for mode in MODES:
                s = df[(df["mode"] == mode) & (df["gene"].isin(genes))]
                vals.append(100 * s[col].mean() if len(s) else 0)
            ax.bar(x + (j - 1) * w, vals, w, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels([MODE_LABELS[m] for m in MODES], fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_title(f"{title}\n(n={len(genes)} genes)", fontweight="600", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Agreement rates conditional on true nonlinear phenotype", fontsize=12, fontweight="700", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_mid_ratio_scatter(out: Path, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, mode in zip(axes, MODES):
        sub = df[df["mode"] == mode].dropna(subset=["true_mid_ratio", "pred_mid_ratio"])
        ax.scatter(sub["true_mid_ratio"], sub["pred_mid_ratio"], s=12, alpha=0.4, c=MODE_COLORS[mode])
        lim = max(sub["true_mid_ratio"].max(), sub["pred_mid_ratio"].max(), 2)
        ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.4)
        ax.axhline(MID_SLOPE_RATIO_THR, color="#F9A825", ls=":", lw=1, label=f"thr={MID_SLOPE_RATIO_THR}")
        ax.axvline(MID_SLOPE_RATIO_THR, color="#F9A825", ls=":", lw=1)
        ax.set_xlabel("True mid/end slope ratio")
        ax.set_ylabel("Pred mid/end slope ratio")
        ax.set_title(MODE_LABELS[mode], fontweight="600")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    fig.suptitle("Mid-segment slope dominance: true vs predicted", fontsize=12, fontweight="700", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_deviation_heatmap(out: Path, df: pd.DataFrame) -> None:
    """各模式平均偏差率（dev=1 的比例）。"""
    dev_cols = ["mono_dev", "peak_mid_dev", "mid_dom_dev"]
    labels = ["Monotonicity", "Peak location", "Mid-slope dom."]
    mat = []
    for mode in MODES:
        sub = df[df["mode"] == mode]
        mat.append([100 * sub[c].mean() for c in dev_cols])
    mat = np.array(mat)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels([MODE_LABELS[m] for m in MODES])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}%", ha="center", va="center", fontsize=11, fontweight="600")
    plt.colorbar(im, ax=ax, label="Mismatch rate (%)")
    ax.set_title("Nonlinear indicator mismatch (lower is better)", fontweight="600")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pick_example_genes(df: pd.DataFrame, n: int = 3) -> List[str]:
    """真值三类非线性都显著，但链式三项都错。"""
    base = df[df["mode"] == "indep"]
    t = base[
        (base["true_mono"] == "mixed")
        & (base["true_peak_mid"] == 1)
        & (base["true_mid_dom"] == 1)
    ]
    if t.empty:
        t = base[base["true_peak_mid"] == 1]
    c = df[df["mode"] == "chain"].set_index("gene")
    scores = []
    for g in t["gene"].astype(str):
        if g not in c.index:
            continue
        r = c.loc[g]
        score = r["mono_dev"] + r["peak_mid_dev"] + r["mid_dom_dev"]
        scores.append((g, score, r["mid_ratio_gap"]))
    scores.sort(key=lambda x: (-x[1], x[2]))
    return [x[0] for x in scores[:n]]


def plot_examples(out: Path, genes: List[str], ch, ind, gr) -> None:
    from plot_hESC_indep_vs_chain_nonlinear import norm_profile

    n = len(genes)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.2 * n), squeeze=False)
    x = np.arange(N_PTS)
    for row, gene in enumerate(genes):
        ty = true_profile(ch, gene)
        sl_t = segment_slopes(ty)
        preds = [
            ("Independent", indep_profile(ind, ch, gene), COLOR_INDEP),
            ("Chained", chain_profile(ch, gene), COLOR_CHAIN),
            ("Single", single_profile(gene, gr, ty) if gr is not None else np.full(N_PTS, np.nan), COLOR_SINGLE),
        ]
        for col, (name, py, pcol) in enumerate(preds):
            ax = axes[row, col]
            if not np.all(np.isfinite(ty)):
                ax.set_visible(False)
                continue
            ax.plot(x, norm_profile(ty), "o-", color=model_color("scPrint"), lw=2.5, ms=7, label="True")
            if np.all(np.isfinite(py)):
                ax.plot(x, norm_profile(py), "s--", color=pcol, lw=2, ms=6, label="Pred")
            ft = extract_nl_features(ty)
            fp = extract_nl_features(py) if np.all(np.isfinite(py)) else {}
            ax.set_title(f"{gene} — {name}", fontsize=9, fontweight="600")
            ax.set_xticks(x)
            ax.set_xticklabels(PT_NODES, fontsize=7)
            ax.grid(alpha=0.25)
            txt = (
                f"T: mono={ft['mono_class']} peak@S{ft['peak_seg']+1}→S{ft['peak_seg']+2} "
                f"mid/end={ft['mid_end_ratio']:.1f}\n"
            )
            if fp:
                txt += (
                    f"P: mono={fp.get('mono_class','?')} peak@S{fp.get('peak_seg',-1)+1} "
                    f"mid/end={fp.get('mid_end_ratio', float('nan')):.1f}"
                )
            ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=6, va="bottom",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    fig.suptitle("Examples: true nonlinear pattern vs scGPT failure on all 3 checks", fontsize=11, fontweight="700", y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_rerun_guide(out: Path, thr: float) -> None:
    text = f"""# Scheme 3 — 何时需要重跑 scGPT？

## 本脚本（简化版，已完成）

- **输入**：已有 `chained_preds` + `segment_preds` + 单步 CSV
- **参照**：三条可审计规则（单调类、峰值段、mid/end 斜率比 ≥ {thr}）
- **输出**：一致率 / 偏差率 / 分层图 — **不必重跑模型**

## 完整版（可选，给审稿人加强版）

若审稿人要求「参数化非线性参照」，可在 scGPT 跑法上扩展：

1. **保存 5 点轨迹**（已有 gene_result 即 S0…S4 bin-mean）
2. **对真值拟合二次多项式** `y(t)=a+bt+ct²`, t∈{{0,0.25,0.5,0.75,1}}
   - 记录 `c` 符号与大小 → 曲率参照
3. **对 scGPT 轨迹**不算拟合，只算到二次参照的 RMSE / 曲率差
4. **重跑仅当**需要：
   - 每步迭代中间 snapshot（16 个时间点而不仅是 4 段末）
   - 用**真实 Si 细胞**重新 chain（对照当前 S0 链）
   - 换 `gen_iters` / `n_segments` 做敏感性

### 重跑命令（链式）

```bash
cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig4
python3 run_scgpt_chained_segments.py --dataset hESC --gen-iters 16 --n-segments 5
```

### 重跑命令（独立分段，方案1 对照）

```bash
python3 run_scgpt_pt_segments.py --dataset hESC
```

### 建议写作表述

> We defined three auditable nonlinear phenotypes on pseudotime-binned
> expression profiles (monotonicity class, interior peak segment, and
> mid-vs-end slope ratio). scGPT predictions were scored by phenotype
> agreement rather than endpoint accuracy alone.

简化版已足以回应「非线性」是否为随口一提；完整二次拟合可作为 supplement。
"""
    out.write_text(text, encoding="utf-8")


def write_summary_csv(out: Path, df: pd.DataFrame) -> None:
    rows = []
    for mode in MODES:
        sub = df[df["mode"] == mode]
        rows.append({
            "mode": mode,
            "n": len(sub),
            "mono_match_pct": 100 * sub["mono_match"].mean(),
            "peak_mid_match_pct": 100 * sub["peak_mid_match"].mean(),
            "mid_dom_match_pct": 100 * sub["mid_dom_match"].mean(),
            "mono_mismatch_pct": 100 * sub["mono_dev"].mean(),
            "peak_mid_mismatch_pct": 100 * sub["peak_mid_dev"].mean(),
            "mid_dom_mismatch_pct": 100 * sub["mid_dom_dev"].mean(),
            "mean_mid_ratio_gap": sub["mid_ratio_gap"].mean(),
        })
    pd.DataFrame(rows).to_csv(out, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Scheme 3 simplified nonlinear reference")
    p.add_argument("--chained-dir", type=Path, default=CHAINED_DIR)
    p.add_argument("--indep-dir", type=Path, default=INDEP_DIR)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--mid-ratio-thr", type=float, default=MID_SLOPE_RATIO_THR)
    args = p.parse_args()

    apply_fig4_style()
    outdir = args.outdir or (args.chained_dir / "figures" / OUT_SUB)
    outdir.mkdir(parents=True, exist_ok=True)

    ch = load_chained_tables(args.chained_dir)
    ind = load_indep_tables(args.indep_dir)
    gr = None
    sp = SINGLE_DIR / "hESC_gene_result.csv"
    if sp.exists():
        gr = pd.read_csv(sp).set_index("gene")

    genes = sorted(set(ch["delta_t0_t1"].index) & set(ind[0].index))
    df = build_full_table(genes, ch, ind, gr, args.mid_ratio_thr)
    df.to_csv(outdir / "scheme3_nl_indicators_per_gene.csv", index=False)
    write_summary_csv(outdir / "scheme3_summary_by_mode.csv", df)

    plot_concept(outdir / "01_concept_three_checks.png", args.mid_ratio_thr)
    plot_agreement_rates(outdir / "02_agreement_three_modes.png", df)
    plot_stratified_by_true_nl(outdir / "03_stratified_true_nl.png", df)
    plot_mid_ratio_scatter(outdir / "04_mid_ratio_scatter.png", df)
    plot_deviation_heatmap(outdir / "05_mismatch_heatmap.png", df)

    examples = pick_example_genes(df, n=3)
    pd.Series(examples, name="gene").to_csv(outdir / "scheme3_example_genes.csv", index=True)
    if examples:
        plot_examples(outdir / "06_example_failures.png", examples, ch, ind, gr)

    write_rerun_guide(outdir / "SCHEME3_rerun_guide.md", args.mid_ratio_thr)

    print("\n=== Scheme 3 (simplified nonlinear reference) ===")
    print(f"Output: {outdir}")
    print(f"Mid/end slope threshold: {args.mid_ratio_thr}")
    summ = pd.read_csv(outdir / "scheme3_summary_by_mode.csv")
    print(summ.to_string(index=False))
    print(f"\nExample genes: {', '.join(examples)}")
    print("\nFigures:")
    for f in sorted(outdir.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
