#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 fig4/.../results_multidataset_pseudotime_227_random 读取 gene_result CSV（默认仅 run1，与 pre_scgpt 单次 CSV 对齐），
计算 balanced accuracy 并绘图。可选合并 benchmark_GRN 下各模型结果。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

try:
    from fig4_palette import apply_fig4_style, model_color

    apply_fig4_style()
except Exception:

    def model_color(model_name: str, default: str = "#7f7f7f") -> str:
        colors = {
            "scCello": "#1f77b4",
            "scPRINT": "#ff7f0e",
            "scGPT": "#2ca02c",
            "Geneformer": "#d62728",
            "LangCell": "#9467bd",
            "scFoundation": "#8c564b",
            "Random": "#4EA3F1",
            "Results": "#4EA3F1",
        }
        return colors.get(str(model_name), default)


TOP_PERCENT = 0.3
_GENE_CSV_RE = re.compile(r"^(.+)_gene_result(?:_run\d+)?\.csv$", re.IGNORECASE)


def balanced_accuracy_gene_result_csv(
    csv_path: Path,
    *,
    top_percent: float = TOP_PERCENT,
    use_top: bool = True,
) -> Optional[float]:
    """scGPT 风格 gene_result：dir_true/dir_pred/delta_true，与 weight_balance.calculate_balance 一致（balanced accuracy）。"""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    req = ["dir_true", "dir_pred", "delta_true"]
    if not all(c in df.columns for c in req):
        return None
    if use_top:
        df = df.copy()
        df["abs_delta_true"] = pd.to_numeric(df["delta_true"], errors="coerce").abs().fillna(0.0)
        n_top = max(1, int(np.ceil(float(top_percent) * len(df))))
        df = df.nlargest(n_top, "abs_delta_true")
    y_true = (df["dir_true"].astype(str).str.strip() == "Up").astype(int)
    y_pred = (df["dir_pred"].astype(str).str.strip() == "Up").astype(int)
    return float(balanced_accuracy_score(y_true, y_pred))


def balanced_accuracy_gene_delta_compare(csv_path: Path) -> Optional[float]:
    """scFoundation 等：true_dir / pred_dir（数值），可选 gene_used / mapped 过滤。"""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if "true_dir" not in df.columns or "pred_dir" not in df.columns:
        return None
    if "gene_used" in df.columns:
        df = df[df["gene_used"] == True].copy()
    elif "mapped" in df.columns:
        df = df[df["mapped"] == True].copy()
    if "gene" in df.columns:
        df = df[~df["gene"].astype(str).str.contains("ERCC", na=False)]
    df = df[pd.to_numeric(df["pred_dir"], errors="coerce").notna()]
    if len(df) == 0:
        return None
    y_true = pd.to_numeric(df["true_dir"], errors="coerce")
    y_pred = pd.to_numeric(df["pred_dir"], errors="coerce")
    mask = y_true.notna() & y_pred.notna()
    y_true = y_true[mask].astype(int)
    y_pred = y_pred[mask].astype(int)
    if len(y_true) == 0:
        return None
    try:
        return float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        return None


def load_from_local_results_dir(
    dir_path: Path,
    *,
    top_percent: float,
    random_run: str = "1",
) -> Dict[str, float]:
    """random_run=all：所有 *_gene_result*.csv 按数据集取 BA 再均值；否则仅 *_gene_result_run{N}.csv（默认 N=1）。"""
    per_ds: Dict[str, List[float]] = {}
    if random_run.lower() == "all":
        files = sorted(dir_path.glob("*_gene_result*.csv"))
    else:
        files = sorted(dir_path.glob(f"*_gene_result_run{int(random_run)}.csv"))
    for fp in files:
        m = _GENE_CSV_RE.match(fp.name)
        if not m:
            continue
        ds = m.group(1)
        ba = balanced_accuracy_gene_result_csv(fp, top_percent=top_percent)
        if ba is not None:
            per_ds.setdefault(ds, []).append(ba)
    return {ds: float(np.mean(vals)) for ds, vals in per_ds.items() if vals}


def _label_for_local_dir(d: Path) -> str:
    name = d.name.lower()
    if "random" in name:
        return "Random"
    return d.name


def discover_under_fig4(script_dir: Path, extra_dirs: List[Path]) -> List[Tuple[str, Path]]:
    """默认包含 script_dir/results_multidataset_pseudotime_227_random；另扫脚本目录下含 gene_result CSV 的一级子目录。"""
    roots: List[Path] = []
    default_r = script_dir / "results_multidataset_pseudotime_227_random"
    if default_r.is_dir():
        roots.append(default_r)
    for p in extra_dirs:
        if p.is_dir():
            roots.append(p.resolve())
    seen = set()
    unique_roots: List[Path] = []
    for r in roots:
        rp = r.resolve()
        if rp not in seen:
            seen.add(rp)
            unique_roots.append(r)
    # 其它同级目录：含 *_gene_result*.csv
    for child in sorted(script_dir.iterdir()):
        if not child.is_dir():
            continue
        cr = child.resolve()
        if cr in seen:
            continue
        if any(child.glob("*_gene_result*.csv")):
            unique_roots.append(child)
            seen.add(cr)
    out: List[Tuple[str, Path]] = []
    for r in unique_roots:
        out.append((_label_for_local_dir(r), r))
    return out


def scan_benchmark_grn(root: Path) -> Dict[str, Dict[str, float]]:
    """从 benchmark_GRN 根目录扫描各模型单次结果（无 run 聚合）。"""
    patterns: List[Tuple[str, str, str]] = [
        ("scGPT", "pre_scgpt/results_multidataset_pseudotime_227/*_gene_result.csv", "gene_result"),
        ("scCello", "pre_sccello_results_unified/sccello/*_gene_result.csv", "gene_result"),
        ("LangCell", "pre_langcell_results_unified/langcell/*_gene_result.csv", "gene_result"),
        ("Geneformer", "pre_geneformer_results_unified/geneformer/*_gene_result.csv", "gene_result"),
        ("scPRINT", "pre_scprint/results_multidataset_pseudotime_227/*_gene_result.csv", "gene_result"),
        (
            "scFoundation",
            "pre_scfoundation/scfoundation_multidataset_pseudotime_227/*/gene_delta_compare.csv",
            "delta_compare",
        ),
    ]
    results: Dict[str, Dict[str, float]] = {}
    root = root.resolve()
    for model, pattern, kind in patterns:
        model_dict: Dict[str, float] = {}
        for fp in sorted(root.glob(pattern)):
            if kind == "gene_result":
                ds = fp.stem.replace("_gene_result", "")
                ba = balanced_accuracy_gene_result_csv(fp)
            else:
                ds = fp.parent.name
                ba = balanced_accuracy_gene_delta_compare(fp)
            if ba is not None:
                model_dict[ds] = ba
        if model_dict:
            results[model] = model_dict
    return results


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Balanced accuracy bar chart from local fig4 results and/or benchmark_GRN")
    p.add_argument(
        "--also-benchmark",
        action="store_true",
        help="除 fig4 本地 Random 等目录外，再从 --benchmark-root 合并各模型结果",
    )
    p.add_argument(
        "--benchmark-only",
        action="store_true",
        help="只从 --benchmark-root 读取（pre_scgpt / pre_scfoundation / pre_sccello / pre_geneformer / pre_langcell / pre_scprint），不读 fig4 本地结果",
    )
    p.add_argument(
        "--extra-local-dir",
        action="append",
        default=[],
        help="额外本地结果目录（可多次指定）",
    )
    p.add_argument(
        "--benchmark-root",
        type=str,
        default="/mnt/10T/yzn/benchmark_GRN",
        help="与 --also-benchmark 联用的 benchmark_GRN 根路径",
    )
    p.add_argument(
        "--random-run",
        type=str,
        default="1",
        help='本地 Random 目录：仅使用该 run 编号的 CSV（默认 1，即 fig4/.../hESC_gene_result_run1.csv）；填 all 则所有 run 取均值',
    )
    p.add_argument("--top-percent", type=float, default=TOP_PERCENT, help="gene_result CSV 取 |delta_true| top 比例")
    p.add_argument("--out", type=str, default="", help="输出 PDF 路径")
    p.add_argument("--figwidth", type=float, default=10.0)
    p.add_argument("--figheight", type=float, default=5.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    extra = [Path(x) for x in args.extra_local_dir]

    results_data: Dict[str, Dict[str, float]] = {}

    if not args.benchmark_only:
        for label, dpath in discover_under_fig4(script_dir, extra):
            acc = load_from_local_results_dir(
                dpath, top_percent=args.top_percent, random_run=args.random_run
            )
            if acc:
                if label in results_data:
                    merged = {**results_data[label], **acc}
                    results_data[label] = merged
                else:
                    results_data[label] = acc
                print(
                    f"本地结果 [{label}] ← {dpath} (random-run={args.random_run}): {sorted(acc.keys())}"
                )

    if args.also_benchmark or args.benchmark_only:
        br = Path(args.benchmark_root)
        if br.is_dir():
            bench = scan_benchmark_grn(br)
            for model, acc in bench.items():
                results_data[model] = acc
                print(f"benchmark [{model}]: {sorted(acc.keys())}")
        else:
            print(f"[WARN] benchmark 根目录不存在，跳过: {br}")

    if not results_data:
        raise SystemExit(
            "未找到任何结果 CSV。使用默认请保留 fig4/results_multidataset_pseudotime_227_random；"
            "或运行 python balance.py --benchmark-only --benchmark-root /mnt/10T/yzn/benchmark_GRN"
        )

    all_datasets: set[str] = set()
    for acc in results_data.values():
        all_datasets.update(acc.keys())
    datasets = sorted(ds for ds in all_datasets if ds != "mDC")
    if "mDC" in all_datasets:
        datasets.append("mDC")

    preferred = ["Random", "Results", "scCello", "scPRINT", "scGPT", "Geneformer", "LangCell", "scFoundation"]
    methods_order = [m for m in preferred if m in results_data]
    methods_order += sorted(m for m in results_data if m not in methods_order)

    n_methods = len(methods_order)
    n_datasets = len(datasets)
    values = np.full((n_methods, n_datasets), np.nan)
    for mi, model in enumerate(methods_order):
        for di, ds in enumerate(datasets):
            if ds in results_data[model]:
                values[mi, di] = results_data[model][ds] * 100.0

    print("\nBalanced Accuracy (%)")
    print(f"{'Model':<14}", *(f"{ds:<10}" for ds in datasets), sep="")
    for mi, model in enumerate(methods_order):
        row = [model]
        for di in range(n_datasets):
            v = values[mi, di]
            row.append(f"{v:.1f}" if np.isfinite(v) else "—")
        print(f"{row[0]:<14}", *([f"{x:<10}" for x in row[1:]]), sep="")

    fig, ax = plt.subplots(figsize=(args.figwidth, args.figheight), dpi=600)
    x = np.arange(n_datasets)
    slot_width = min(0.8 / max(n_methods, 1), 0.13)
    width = slot_width * 0.9
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2) * slot_width

    for mi, model in enumerate(methods_order):
        ax.bar(
            x + offsets[mi],
            values[mi],
            width=width,
            label=model,
            color=model_color(model),
            edgecolor="none",
            linewidth=0.0,
            zorder=3,
        )

    ax.axhline(50, color="#888888", linestyle="--", linewidth=1.0, alpha=0.7, zorder=2)
    ax.set_ylabel("Balanced Accuracy (%)", fontsize=16, fontweight="normal")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=0, ha="center", fontsize=14)
    ax.tick_params(axis="x", labelsize=14, length=0)
    ax.tick_params(axis="y", labelsize=14, length=0)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(False)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=min(n_methods, 6),
        fontsize=14,
        handlelength=1.2,
        handletextpad=0.3,
        columnspacing=0.4,
        borderaxespad=0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    plt.subplots_adjust(bottom=0.15)

    out = Path(args.out) if args.out else script_dir / "accuracy" / "figure_balanced_accuracy_0507.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"\n✅ 已保存: {out.resolve()}")


if __name__ == "__main__":
    main()
