#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多步伪时间（分段 early→late）分析 — fig4 扩展。

动机
----
单步 early(20%) vs late(20%) 把整条发育轨迹压成一步，transition 期瞬时基因会被平均掉；
非 CHIP 靶的高动态信号可能集中在中间某段。本脚本：

1. 沿 PT 分 K 段，计算每段及相邻段 transition 的 true Δ；
2. 比较「单步 end-to-end」vs「基因在其最强 transition 段」的信号与方向；
3. 用分段 TF Δ + GRN（可选）做非 CHIP 靶方向预测，对比单步 scGPT gene_result。

无需 GPU；若要每段重跑 scGPT 迭代，见文末 ``--print-scgpt-commands``。

示例
----
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig4

  python3 analyze_multistep_pseudotime.py --dataset hESC
  python3 analyze_multistep_pseudotime.py --all-datasets --n-segments 5
  python3 analyze_multistep_pseudotime.py --dataset hESC --use-grn --head-root \\
    /mnt/10T/yzn/benchmark_GRN/evl_omipath/output_att500/scgpt_multihead
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
FIG2_PLOT = SCRIPT_DIR.parent / "fig2" / "plot"
if str(FIG2_PLOT) not in sys.path:
    sys.path.insert(0, str(FIG2_PLOT))

try:
    from fig4_palette import apply_fig4_style, model_color
except ImportError:

    def apply_fig4_style() -> None:
        plt.rcParams.update({"font.size": 12})

    def model_color(_: str, d: str = "#808") -> str:
        return d

DEFAULT_CHIP_DIR = Path("/mnt/10T/yzn/benchmark_GRN/input_process/CHIP")
DEFAULT_PT_ROOT = Path("/mnt/10T/yzn/benchmark_GRN/PseudoTime")
DEFAULT_GENE_DIR = Path("/mnt/10T/yzn/benchmark_GRN/pre_scgpt/results_multidataset_pseudotime_227")
DEFAULT_HEAD_ROOT = Path("/mnt/10T/yzn/benchmark_GRN/evl_omipath/output_att500/scgpt_multihead")
DEFAULT_OUTDIR = SCRIPT_DIR / "error_biology" / "multistep_pt"

DATASETS = ["hESC", "hHep", "mDC", "mHSC-E", "mHSC-GM", "mHSC-L"]
TOP_PERCENT = 0.3
PT_QUANTILE_END = 0.2  # 单步基线：两端各 20%
LOAD_MAX_EDGES = 80_000
DPI = 300


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def dataset_paths(chip_dir: Path, pt_root: Path, dataset: str) -> Dict[str, Path]:
    return {
        "expr": chip_dir / f"{dataset}_chip_matched-ExpressionData.csv",
        "pt": pt_root / dataset / "PseudoTime.csv",
        "chip": chip_dir / f"{dataset}_chip_matched-network.csv",
        "gene_result": DEFAULT_GENE_DIR / f"{dataset}_gene_result.csv",
    }


def load_chip_tf_targets(chip_net: Path) -> Set[str]:
    df = pd.read_csv(chip_net)
    g2 = "Gene2" if "Gene2" in df.columns else df.columns[1]
    return set(df[g2].astype(str).str.strip())


def read_expression(expr_csv: Path) -> pd.DataFrame:
    """CHIP matched 表达文件为 genes×cells，统一转为 cells×genes。"""
    expr = pd.read_csv(expr_csv, index_col=0)
    expr = expr.T
    expr.index = expr.index.astype(str).str.strip()
    expr.columns = expr.columns.astype(str).str.strip()
    return expr


def read_pseudotime(pt_csv: Path) -> pd.Series:
    pt_df = pd.read_csv(pt_csv)
    if pt_df.shape[1] >= 2 and pt_df.columns[0] in ("", "Unnamed: 0", "cell", "Cell"):
        first = pt_df.columns[0]
        second = pt_df.columns[1]
        if str(second).lower() in ("pt", "pseudotime", "pseudotime_value"):
            pt_df = pt_df.rename(columns={first: "cell", second: "pt"})
        else:
            pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"})
    elif pt_df.shape[1] >= 2:
        pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"})
    pt_df["cell"] = pt_df["cell"].astype(str).str.strip()
    pt_df["pt"] = pd.to_numeric(pt_df["pt"], errors="coerce")
    return pt_df.set_index("cell")["pt"].dropna()


def align_expr_pt(expr: pd.DataFrame, pt: pd.Series) -> Tuple[pd.DataFrame, np.ndarray]:
    common = expr.index.intersection(pt.index)
    if len(common) == 0:
        raise ValueError("No overlapping cells between expression and pseudotime")
    expr = expr.loc[common]
    pt_arr = pt.loc[common].values.astype(np.float64)
    order = np.argsort(pt_arr)
    return expr.iloc[order], pt_arr[order]


# ---------------------------------------------------------------------------
# 分段
# ---------------------------------------------------------------------------
def mask_endpoints(pt: np.ndarray, q: float = PT_QUANTILE_END) -> Tuple[np.ndarray, np.ndarray]:
    lo, hi = np.quantile(pt, [q, 1.0 - q])
    return pt <= lo, pt >= hi


def mask_segment_bins(pt: np.ndarray, n_segments: int) -> List[np.ndarray]:
    """按 PT 分位把细胞分成 n_segments 段（等细胞数）。"""
    n_segments = max(2, int(n_segments))
    edges = np.quantile(pt, np.linspace(0, 1, n_segments + 1))
    edges[-1] += 1e-9
    masks = []
    for i in range(n_segments):
        if i < n_segments - 1:
            m = (pt >= edges[i]) & (pt < edges[i + 1])
        else:
            m = (pt >= edges[i]) & (pt <= edges[i + 1])
        masks.append(m)
    return masks


def segment_mean_expr(expr: pd.DataFrame, mask: np.ndarray) -> pd.Series:
    if mask.sum() == 0:
        return pd.Series(dtype=np.float64)
    return expr.loc[mask].mean(axis=0)


def compute_transition_deltas(
    expr: pd.DataFrame,
    pt: np.ndarray,
    n_segments: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    返回
      transitions: 每行一个 transition (seg_i -> seg_{i+1})，列含 delta 向量展平为长表
      gene_table:  每基因一行，含单步 delta 与各 transition delta、argmax 段
    """
    masks = mask_segment_bins(pt, n_segments)
    seg_means = [segment_mean_expr(expr, m) for m in masks]
    genes = seg_means[0].index.astype(str).tolist()

    # 单步基线（与 fig4 一致）
    early_m, late_m = mask_endpoints(pt)
    mean_early = segment_mean_expr(expr, early_m)
    mean_late = segment_mean_expr(expr, late_m)
    delta_single = mean_late - mean_early

    trans_records = []
    delta_cols: Dict[str, pd.Series] = {}
    for i in range(len(seg_means) - 1):
        d = seg_means[i + 1] - seg_means[i]
        name = f"delta_t{i}_t{i+1}"
        delta_cols[name] = d
        trans_records.append(
            {
                "transition": f"S{i}→S{i+1}",
                "seg_from": i,
                "seg_to": i + 1,
                "n_cells_from": int(masks[i].sum()),
                "n_cells_to": int(masks[i + 1].sum()),
                "mean_abs_delta": float(d.abs().mean()),
            }
        )
    trans_summary = pd.DataFrame(trans_records)

    gene_df = pd.DataFrame({"gene": genes})
    gene_df["delta_single"] = gene_df["gene"].map(delta_single)
    for name, ser in delta_cols.items():
        gene_df[name] = gene_df["gene"].map(ser)

    trans_cols = list(delta_cols.keys())
    abs_mat = gene_df[trans_cols].abs()
    gene_df["best_transition"] = abs_mat.idxmax(axis=1)
    idx = np.arange(len(gene_df))
    col_idx = [trans_cols.index(c) for c in gene_df["best_transition"]]
    raw = gene_df[trans_cols].to_numpy()[idx, col_idx]
    gene_df["delta_best_trans"] = raw

    gene_df["abs_delta_single"] = gene_df["delta_single"].abs()
    gene_df["abs_delta_best"] = gene_df["delta_best_trans"].abs()
    gene_df["best_is_single"] = gene_df["best_transition"].isna() | (
        gene_df["abs_delta_best"] <= gene_df["abs_delta_single"] * 1.001
    )
    # 若最强 transition 不是单步对应的 S0→S_{K-1} 代理：标记 middle-heavy
    first_last = trans_cols[0] if trans_cols else None
    last_trans = trans_cols[-1] if trans_cols else None
    gene_df["peak_middle"] = ~gene_df["best_transition"].isin([first_last, last_trans])

    return trans_summary, gene_df


# ---------------------------------------------------------------------------
# GRN 方向（分段 TF Δ）
# ---------------------------------------------------------------------------
def load_grn_mean8(
    head_root: Path,
    dataset: str,
    chip_net: Path,
    eval_genes: Set[str],
) -> pd.DataFrame:
    try:
        from eval_heads_chip_auprc import (
            chip_gene_sets_from_gt,
            load_head_pred,
            read_chip_gt,
        )
    except ImportError as e:
        raise SystemExit(f"Need fig2 eval_heads_chip_auprc: {e}")

    gt = read_chip_gt(chip_net)
    tfs, genes = chip_gene_sets_from_gt(gt)
    universe = tfs | genes | eval_genes
    per_head: Dict[int, pd.DataFrame] = {}
    for fp in sorted(head_root.glob(f"scgpt_{dataset}_head*.tsv")):
        m = re.search(r"head(\d+)", fp.stem, re.I)
        if not m:
            continue
        h = int(m.group(1))
        raw = load_head_pred(fp, universe, LOAD_MAX_EDGES)
        # Gene1∈TF；Gene2 可在评估基因集（含非 CHIP 靶）
        per_head[h] = raw[
            raw["Gene1"].isin(tfs) & raw["Gene2"].isin(eval_genes)
        ].copy()

    acc: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for df in per_head.values():
        if df.empty:
            continue
        for row in df.itertuples(index=False):
            acc[(str(row.Gene1), str(row.Gene2))].append(float(row.EdgeWeight))
    rows = [
        {"Gene1": a, "Gene2": b, "EdgeWeight": float(np.mean(ws))}
        for (a, b), ws in acc.items()
    ]
    return pd.DataFrame(rows)


def grn_predict_direction(
    genes: List[str],
    tf_delta: Dict[str, float],
    grn: pd.DataFrame,
) -> pd.Series:
    incoming: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for row in grn.itertuples(index=False):
        incoming[str(row.Gene2)].append((str(row.Gene1), float(row.EdgeWeight)))

    out = {}
    for g in genes:
        score, wsum = 0.0, 0.0
        for tf, w in incoming.get(g, []):
            if tf not in tf_delta or not np.isfinite(tf_delta[tf]):
                continue
            score += w * tf_delta[tf]
            wsum += abs(w)
        if wsum < 1e-12:
            out[g] = np.nan
        else:
            out[g] = "Up" if score > 0 else "Down"
    return pd.Series(out)


def load_scgpt_gene_result(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "ene" in df.columns and "gene" not in df.columns:
        df = df.rename(columns={"ene": "gene"})
    return df


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
def select_top30(gene_df: pd.DataFrame, col: str = "abs_delta_single") -> pd.DataFrame:
    n = max(1, int(np.ceil(TOP_PERCENT * len(gene_df))))
    return gene_df.nlargest(n, col).copy()


def direction_error(df: pd.DataFrame, pred_col: str, true_col: str = "dir_true") -> float:
    if df.empty:
        return float("nan")
    ok = df[pred_col].notna() & df[true_col].notna()
    if not ok.any():
        return float("nan")
    sub = df.loc[ok]
    return float((sub[pred_col].astype(str).str.lower() != sub[true_col].astype(str).str.lower()).mean())


def dir_from_delta(s: pd.Series) -> pd.Series:
    return s.apply(lambda x: "Up" if x > 0 else ("Down" if x < 0 else "Down"))


def evaluate_dataset(
    dataset: str,
    paths: Dict[str, Path],
    *,
    n_segments: int,
    use_grn: bool,
    head_root: Path,
    outdir: Path,
) -> pd.DataFrame:
    expr = read_expression(paths["expr"])
    pt = read_pseudotime(paths["pt"])
    expr, pt = align_expr_pt(expr, pt)

    chip_targets = load_chip_tf_targets(paths["chip"])
    trans_sum, gene_full = compute_transition_deltas(expr, pt, n_segments)
    gene_full["is_chip_target"] = gene_full["gene"].isin(chip_targets)

    top = select_top30(gene_full)
    eval_genes = set(gene_full["gene"].astype(str))

    # 方向真值
    top["dir_true_single"] = dir_from_delta(top["delta_single"])
    top["dir_true_best_trans"] = dir_from_delta(top["delta_best_trans"])

    rows = []

    # --- 信号分析：峰值是否在中间段 ---
    for label, sub in [("all_top30", top), ("nonchip", top[~top["is_chip_target"]]), ("chip", top[top["is_chip_target"]])]:
        if sub.empty:
            continue
        rows.append(
            {
                "dataset": dataset,
                "metric": f"frac_peak_not_single_{label}",
                "value": float((~sub["best_is_single"]).mean()),
                "n": len(sub),
            }
        )
        rows.append(
            {
                "dataset": dataset,
                "metric": f"frac_peak_middle_{label}",
                "value": float(sub["peak_middle"].mean()),
                "n": len(sub),
            }
        )
        rows.append(
            {
                "dataset": dataset,
                "metric": f"mean_gain_abs_{label}",
                "value": float((sub["abs_delta_best"] - sub["abs_delta_single"]).mean()),
                "n": len(sub),
            }
        )

    # --- scGPT 单步 ---
    gr = load_scgpt_gene_result(paths["gene_result"])
    if gr is not None:
        gr = gr.rename(columns={"dir_pred": "dir_pred_scgpt", "dir_correct": "dir_correct_scgpt"})
        top = top.merge(gr[["gene", "dir_pred_scgpt", "delta_pred", "dir_correct_scgpt"]], on="gene", how="left")
        top["dir_true"] = top["dir_true_single"]
        err = direction_error(top[~top["is_chip_target"]], "dir_pred_scgpt")
        rows.append(
            {
                "dataset": dataset,
                "metric": "dir_error_scgpt_single_nonchip",
                "value": err,
                "n": int((~top["is_chip_target"]).sum()),
            }
        )

    # --- 分段 GRN：每个 transition 用该段 TF 的 true Δ，再对非靶投票 ---
    if use_grn and head_root.exists():
        try:
            grn = load_grn_mean8(head_root, dataset, paths["chip"], eval_genes)
        except FileNotFoundError as e:
            print(f"  [warn] GRN skip: {e}")
            grn = pd.DataFrame()

        if not grn.empty:
            trans_cols = [c for c in top.columns if c.startswith("delta_t")]
            tfs = set(grn["Gene1"].astype(str))

            # 汇总：每基因只在 best_transition 段用 GRN 预测
            preds = []
            for _, r in top.iterrows():
                tc = r["best_transition"]
                if pd.isna(tc):
                    preds.append(np.nan)
                    continue
                tf_delta = {}
                for tf in tfs:
                    row = gene_full.loc[gene_full["gene"] == tf, tc]
                    if len(row) and np.isfinite(row.iloc[0]):
                        tf_delta[tf] = float(row.iloc[0])
                p = grn_predict_direction([str(r["gene"])], tf_delta, grn).iloc[0]
                preds.append(p)
            top["dir_pred_grn_peak_segment"] = preds
            sub = top[(~top["is_chip_target"]) & top["dir_pred_grn_peak_segment"].notna()]
            if len(sub):
                err = direction_error(sub, "dir_pred_grn_peak_segment", "dir_true_best_trans")
                rows.append(
                    {
                        "dataset": dataset,
                        "metric": "dir_error_grn_peak_segment_nonchip",
                        "value": err,
                        "n": len(sub),
                    }
                )
                # 与单步 scGPT 比（同一批有 GRN 边的非靶）
                if "dir_pred_scgpt" in top.columns:
                    sub2 = sub[sub["dir_pred_scgpt"].notna()]
                    if len(sub2):
                        rows.append(
                            {
                                "dataset": dataset,
                                "metric": "dir_error_scgpt_single_nonchip_at_grn_peak",
                                "value": direction_error(sub2, "dir_pred_scgpt", "dir_true_best_trans"),
                                "n": len(sub2),
                            }
                        )

    # 保存表
    ds_dir = outdir / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)
    trans_sum.to_csv(ds_dir / "transition_summary.csv", index=False)
    gene_full.to_csv(ds_dir / "gene_segment_deltas.csv", index=False)
    top.to_csv(ds_dir / "top30_multistep_annotated.csv", index=False)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
def plot_peak_transition_fraction(metrics: pd.DataFrame, outpath: Path) -> None:
    apply_fig4_style()
    sub = metrics[metrics["metric"].str.startswith("frac_peak_not_single_")].copy()
    sub["group"] = sub["metric"].str.replace("frac_peak_not_single_", "")
    datasets = [d for d in DATASETS if d in sub["dataset"].unique()]
    groups = ["nonchip", "chip", "all_top30"]
    groups = [g for g in groups if g in sub["group"].values]

    x = np.arange(len(datasets))
    w = 0.25
    fig, ax = plt.subplots(figsize=(6 + len(datasets) * 0.3, 4.5), facecolor="white")
    colors = {
        "nonchip": model_color("scPrint"),
        "chip": model_color("scGPT"),
        "all_top30": model_color("STRING"),
    }
    for j, g in enumerate(groups):
        vals = []
        for ds in datasets:
            row = sub[(sub["dataset"] == ds) & (sub["group"] == g)]
            vals.append(float(row["value"].iloc[0]) if len(row) else 0)
        ax.bar(x + (j - 1) * w, vals, width=w, label=g, color=colors.get(g, "#888"), edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of genes")
    ax.set_title(
        "Top30% dynamic genes: peak |Δ| not in single early→late step\n"
        "(stronger signal in an intermediate PT segment)",
        fontsize=12,
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_direction_comparison(metrics: pd.DataFrame, outpath: Path) -> None:
    apply_fig4_style()
    want = [
        ("dir_error_scgpt_single_nonchip", "scGPT single-step"),
        ("dir_error_grn_peak_segment_nonchip", "GRN @ peak segment"),
        ("dir_error_scgpt_single_nonchip_at_grn_peak", "scGPT @ same genes"),
    ]
    datasets = [d for d in DATASETS if d in metrics["dataset"].unique()]
    fig, ax = plt.subplots(figsize=(5 + len(datasets) * 0.5, 4.2), facecolor="white")
    x = np.arange(len(datasets))
    w = 0.26
    for j, (metric, lab) in enumerate(want):
        subm = metrics[metrics["metric"] == metric]
        vals = []
        for ds in datasets:
            row = subm[subm["dataset"] == ds]
            vals.append(float(row["value"].iloc[0]) if len(row) else np.nan)
        ax.bar(x + (j - 1) * w, vals, width=w, label=lab, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel("Direction error (non-CHIP)")
    ax.set_ylim(0, 1)
    ax.set_title("Non-CHIP direction error: single-step vs peak-segment GRN", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_transition_heatmap(gene_full: pd.DataFrame, dataset: str, outpath: Path) -> None:
    """非靶 vs 靶：各 transition 平均 |Δ|。"""
    apply_fig4_style()
    trans_cols = [c for c in gene_full.columns if c.startswith("delta_t")]
    if not trans_cols:
        return
    top = select_top30(gene_full)
    labels = [c.replace("delta_", "").replace("_", "→") for c in trans_cols]
    mat = []
    for name, chip_flag in [("CHIP target", True), ("Non-CHIP", False)]:
        sub = top[top["is_chip_target"] == chip_flag]
        if sub.empty:
            mat.append([0] * len(trans_cols))
        else:
            mat.append([float(sub[c].abs().mean()) for c in trans_cols])
    mat = np.array(mat)
    fig, ax = plt.subplots(figsize=(1.2 + len(trans_cols) * 0.9, 3.2), facecolor="white")
    im = ax.imshow(mat, aspect="auto", cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["CHIP target", "Non-CHIP"])
    for i in range(2):
        for j in range(len(labels)):
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=9)
    ax.set_title(f"{dataset} — mean |Δ| per transition (top30%)", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _transition_labels(gene_full: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cols = sorted([c for c in gene_full.columns if c.startswith("delta_t")])
    labels = [c.replace("delta_", "").replace("_", "→") for c in cols]
    return labels, cols


def plot_transition_meandelta_lines(
    gene_full: pd.DataFrame,
    dataset: str,
    outpath: Path,
) -> None:
    """沿 PT 各 transition 的平均 |Δ|：非靶 vs CHIP 靶（折线）。"""
    apply_fig4_style()
    labels, trans_cols = _transition_labels(gene_full)
    if not trans_cols:
        return

    top = select_top30(gene_full)
    fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor="white")

    for lab, is_chip, color, marker in [
        ("Non-CHIP target", False, model_color("scPrint"), "o"),
        ("CHIP TF target", True, model_color("scGPT"), "s"),
    ]:
        sub = top[top["is_chip_target"] == is_chip]
        if sub.empty:
            continue
        y = [float(sub[c].abs().mean()) for c in trans_cols]
        ax.plot(
            labels, y,
            marker=marker, markersize=9, linewidth=2.4, color=color,
            label=f"{lab} (n={len(sub)})", alpha=0.9,
        )
        ax.axhline(
            float(sub["abs_delta_single"].mean()),
            color=color, linestyle="--", linewidth=1.2, alpha=0.45,
        )

    ax.set_xlabel("Pseudotime transition (equal-frequency segments)", fontsize=12)
    ax.set_ylabel(r"Mean $|\Delta_{true}|$ (top 30% dynamic genes)", fontsize=12)
    ax.set_title(
        f"{dataset} — dynamic strength along PT\n"
        r"Dashed = mean $|\Delta_{single}|$ (20% early vs 20% late)",
        fontsize=12,
    )
    ax.legend(frameon=False, loc="best", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_peak_location_lines(
    gene_full: pd.DataFrame,
    dataset: str,
    outpath: Path,
) -> None:
    """每个 transition 上 |Δ| 峰值占比；标注「峰值≠单步」比例。"""
    apply_fig4_style()
    labels, trans_cols = _transition_labels(gene_full)
    if not trans_cols:
        return

    top = select_top30(gene_full)
    fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor="white")

    for lab, is_chip, color, marker in [
        ("Non-CHIP target", False, model_color("scPrint"), "o"),
        ("CHIP TF target", True, model_color("scGPT"), "s"),
    ]:
        sub = top[top["is_chip_target"] == is_chip]
        if sub.empty:
            continue
        abs_mat = sub[trans_cols].abs()
        best_col = abs_mat.idxmax(axis=1)
        frac = best_col.value_counts(normalize=True).reindex(trans_cols, fill_value=0.0)
        y = [float(frac[c]) for c in trans_cols]
        ax.plot(
            labels, y,
            marker=marker, markersize=9, linewidth=2.4, color=color,
            label=f"{lab} (n={len(sub)})",
        )

    for lab, is_chip, color, ypos in [
        ("Non-CHIP", False, model_color("scPrint"), 0.88),
        ("CHIP", True, model_color("scGPT"), 0.98),
    ]:
        sub = top[top["is_chip_target"] == is_chip]
        if sub.empty:
            continue
        frac_ns = float((~sub["best_is_single"]).mean())
        ax.text(
            0.98, ypos, f"{lab}: {frac_ns:.0%} peak ≠ single step",
            transform=ax.transAxes, ha="right", va="top", fontsize=10, color=color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.85),
        )

    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Segment where |Δ| peaks (among transitions)", fontsize=12)
    ax.set_ylabel("Fraction of genes in group", fontsize=12)
    ax.set_title(
        f"{dataset} — where top30% genes show strongest change\n"
        "(non-targets peak more in middle transitions)",
        fontsize=12,
    )
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_peak_not_single_lines(metrics: pd.DataFrame, outpath: Path) -> None:
    """六数据集：非靶 vs 靶「峰值不在首尾单步」折线。"""
    apply_fig4_style()
    sub = metrics[metrics["metric"].str.startswith("frac_peak_not_single_")].copy()
    sub["group"] = sub["metric"].str.replace("frac_peak_not_single_", "")
    datasets = [d for d in DATASETS if d in sub["dataset"].unique()]

    fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor="white")
    for g, lab, color, marker in [
        ("nonchip", "Non-CHIP target", model_color("scPrint"), "o"),
        ("chip", "CHIP TF target", model_color("scGPT"), "s"),
    ]:
        rows = sub[sub["group"] == g].set_index("dataset").reindex(datasets)
        y = rows["value"].astype(float).tolist()
        n = rows["n"].astype(int).tolist()
        ax.plot(datasets, y, marker=marker, markersize=9, linewidth=2.4, color=color, label=lab)
        for xi, yi, ni in zip(datasets, y, n):
            if np.isfinite(yi):
                ax.annotate(
                    f"n={ni}", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=color,
                )

    ax.set_ylim(0, max(0.5, float(sub.loc[sub["group"] == "nonchip", "value"].max()) * 1.2))
    ax.set_ylabel("Fraction with peak ≠ single early→late step", fontsize=12)
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_title(
        "Top30% dynamic genes: multi-step hypothesis across datasets\n"
        "(hESC non-targets ~42%)",
        fontsize=12,
    )
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_multistep_lines_from_saved(outdir: Path) -> None:
    metrics_path = outdir / "multistep_metrics.csv"
    if metrics_path.exists():
        plot_peak_not_single_lines(pd.read_csv(metrics_path), outdir / "peak_not_single_lines.png")
    for ds_dir in sorted(outdir.iterdir()):
        if not ds_dir.is_dir():
            continue
        gf = ds_dir / "gene_segment_deltas.csv"
        if not gf.exists():
            continue
        ds = ds_dir.name
        gene_full = pd.read_csv(gf)
        if "is_chip_target" not in gene_full.columns:
            chip = load_chip_tf_targets(DEFAULT_CHIP_DIR / f"{ds}_chip_matched-network.csv")
            gene_full["is_chip_target"] = gene_full["gene"].astype(str).isin(chip)
        plot_transition_meandelta_lines(gene_full, ds, ds_dir / "transition_meandelta_lines.png")
        plot_peak_location_lines(gene_full, ds, ds_dir / "peak_location_lines.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def print_scgpt_commands(datasets: List[str], n_segments: int, outdir: Path) -> None:
    print("\n# --- 若要对每个 PT 段重跑 scGPT 迭代（需改 run_unified_multidataset_pseudotime.py）---")
    print("# 思路：对每对 (seg_i, seg_{i+1}) 用该段细胞作 early/late，分别输出 gene_result")
    for ds in datasets:
        print(f"\n# {ds}: {n_segments} segments → {n_segments-1} transitions")
        for i in range(n_segments - 1):
            print(
                f"# python run_unified_multidataset_pseudotime.py --model scgpt "
                f"--datasets {ds} --outdir {outdir}/{ds}/seg{i}_to_{i+1} "
                f"# + 需在代码中传入 segment_masks[{i}], [{i+1}] 替代 quantile early/late"
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-step pseudotime analysis (fig4)")
    p.add_argument("--chip-dir", type=Path, default=DEFAULT_CHIP_DIR)
    p.add_argument("--pt-root", type=Path, default=DEFAULT_PT_ROOT)
    p.add_argument("--gene-dir", type=Path, default=DEFAULT_GENE_DIR)
    p.add_argument("--head-root", type=Path, default=DEFAULT_HEAD_ROOT)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--all-datasets", action="store_true")
    p.add_argument("--n-segments", type=int, default=5, help="PT bins (default 5 → 4 transitions)")
    p.add_argument("--use-grn", action="store_true", help="GRN mean8 + segment TF delta")
    p.add_argument("--print-scgpt-commands", action="store_true")
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Only redraw line plots from saved CSV under --outdir",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global DEFAULT_GENE_DIR
    DEFAULT_GENE_DIR = args.gene_dir

    args.outdir.mkdir(parents=True, exist_ok=True)
    datasets = DATASETS if args.all_datasets or not args.dataset else [args.dataset]

    if args.print_scgpt_commands:
        print_scgpt_commands(datasets, args.n_segments, args.outdir)
        return

    if args.plot_only:
        plot_multistep_lines_from_saved(args.outdir)
        print(f"Line plots saved under {args.outdir}")
        return

    all_metrics = []
    for ds in datasets:
        print(f"\n=== {ds} (n_segments={args.n_segments}) ===")
        paths = dataset_paths(args.chip_dir, args.pt_root, ds)
        paths["gene_result"] = args.gene_dir / f"{ds}_gene_result.csv"
        try:
            m = evaluate_dataset(
                ds,
                paths,
                n_segments=args.n_segments,
                use_grn=args.use_grn,
                head_root=args.head_root,
                outdir=args.outdir,
            )
            all_metrics.append(m)
            gene_full = pd.read_csv(args.outdir / ds / "gene_segment_deltas.csv")
            plot_transition_heatmap(gene_full, ds, args.outdir / ds / "transition_absdelta_heatmap.png")
            plot_transition_meandelta_lines(
                gene_full, ds, args.outdir / ds / "transition_meandelta_lines.png"
            )
            plot_peak_location_lines(
                gene_full, ds, args.outdir / ds / "peak_location_lines.png"
            )
            for _, r in m.iterrows():
                print(f"  {r['metric']:45s} {r['value']:.3f}  (n={r['n']})")
        except Exception as e:
            print(f"  [error] {ds}: {e}")

    if not all_metrics:
        raise SystemExit("No results.")
    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(args.outdir / "multistep_metrics.csv", index=False)
    plot_peak_transition_fraction(metrics, args.outdir / "peak_not_single_fraction.png")
    plot_peak_not_single_lines(metrics, args.outdir / "peak_not_single_lines.png")
    if args.use_grn:
        plot_direction_comparison(metrics, args.outdir / "nonchip_direction_single_vs_multistep.png")

    print(f"\nSaved: {args.outdir / 'multistep_metrics.csv'}")
    print(f"Saved: {args.outdir / 'peak_not_single_fraction.png'}")
    print(f"Saved: {args.outdir / 'peak_not_single_lines.png'}")
    print(f"Per-dataset: .../transition_meandelta_lines.png, .../peak_location_lines.png")
    if args.use_grn:
        print(f"Saved: {args.outdir / 'nonchip_direction_single_vs_multistep.png'}")


if __name__ == "__main__":
    main()
