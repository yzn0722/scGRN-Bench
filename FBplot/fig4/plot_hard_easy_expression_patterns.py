#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig4 补充：难例 vs 易例基因表达模式可视化 + 对照富集

图 SX-1  难例/易例代表基因伪时间表达曲线（3×2 多面板）
图 SX-2  单调性 |Spearman ρ| 与振荡次数分布（小提琴 + 散点 + MWU p 值）
图 SX-3  难例 vs 易例 GO/KEGG 富集对比（水平条图）

依赖:
  - robustness/tables/gene_consensus_hard_easy.csv（先运行 plot_deg_robustness_analysis.py）
  - benchmark_GRN 表达矩阵 + PseudoTime

用法:
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig4
  python3 plot_hard_easy_expression_patterns.py
  python3 plot_hard_easy_expression_patterns.py --dataset hESC --no-enrichment
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks, savgol_filter

warnings.filterwarnings("ignore")

try:
    from fig4_palette import apply_fig4_style, model_color
except ImportError:

    def apply_fig4_style() -> None:
        plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})

    def model_color(name: str, default: str = "#808080") -> str:
        return default


SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_ROOT = Path("/mnt/10T/yzn/benchmark_GRN")
CHIP_DIR = BENCH_ROOT / "input_process" / "CHIP"
PT_ROOT = BENCH_ROOT / "PseudoTime"
DEFAULT_GENE_TABLE = SCRIPT_DIR / "robustness" / "tables" / "gene_consensus_hard_easy.csv"
DEFAULT_OUTDIR = SCRIPT_DIR / "robustness" / "supplement"

PT_QUANTILE = 0.2
DPI = 600
N_EXAMPLE = 3
MIN_CELLS = 20

DATASET_SPECIES = {
    "hESC": "human",
    "hHep": "human",
    "mDC": "mouse",
    "mHSC-E": "mouse",
    "mHSC-GM": "mouse",
    "mHSC-L": "mouse",
}

ENRICH_LIBS = {
    "human": ["GO_Biological_Process_2023", "KEGG_2021_Human"],
    "mouse": ["GO_Biological_Process_2023", "KEGG_2019_Mouse"],
}

CELL_CYCLE_KW = re.compile(
    r"cell.?cycle|mitos|spindle|chromosome|kinesin|anaphase|metaphase|"
    r"cyclin|cdk|mcm|pcna|plk|aurora|checkpoint|separase|cohesin",
    re.I,
)
RIBO_KW = re.compile(r"ribosom|rpl|rps|translation|srp\b", re.I)


def paths_for_dataset(dataset: str) -> Tuple[Path, Path]:
    expr = CHIP_DIR / f"{dataset}_chip_matched-ExpressionData.csv"
    pt = PT_ROOT / dataset / "PseudoTime.csv"
    return expr, pt


def load_expression_pseudotime(dataset: str) -> Tuple[pd.Series, pd.DataFrame]:
    expr_path, pt_path = paths_for_dataset(dataset)
    expr = pd.read_csv(expr_path, index_col=0)
    pt_df = pd.read_csv(pt_path, index_col=0)
    pt_col = pt_df.columns[0]
    pt = pt_df[pt_col].astype(float)
    common = expr.columns.intersection(pt.index)
    if len(common) < MIN_CELLS:
        raise ValueError(f"{dataset}: only {len(common)} cells overlap expr/pt")
    expr = expr[common]
    pt = pt.loc[common]
    return pt, expr


def analyze_gene_expression_pattern(
    pt: pd.Series,
    expr_row: pd.Series,
    gene: str,
    pt_quantile: float = PT_QUANTILE,
) -> dict:
    """Monotonicity, oscillation count, early/late direction (no AnnData)."""
    pt_v = pt.values.astype(float)
    ex_v = expr_row.values.astype(float)
    order = np.argsort(pt_v)
    pt_s = pt_v[order]
    ex_s = ex_v[order]

    if len(ex_s) >= 3 and np.std(ex_s) > 0 and np.std(pt_s) > 0:
        rho, p_mono = stats.spearmanr(pt_s, ex_s)
    else:
        rho, p_mono = np.nan, np.nan

    n_osc = 0
    if len(ex_s) > 10 and np.std(ex_s) > 0:
        cx, cy = binned_pseudotime_curve(
            pd.Series(pt_s, index=range(len(pt_s))),
            pd.Series(ex_s, index=range(len(ex_s))),
            n_bins=min(40, max(12, len(ex_s) // 15)),
        )
        if len(cy) >= 7 and np.std(cy) > 0:
            win = min(9, (len(cy) // 2) * 2 + 1)
            try:
                smooth = savgol_filter(cy, window_length=win, polyorder=min(3, win - 2))
                prom = max(np.std(smooth) * 0.35, 1e-6)
                peaks, _ = find_peaks(smooth, prominence=prom)
                valleys, _ = find_peaks(-smooth, prominence=prom)
                n_osc = int(len(peaks) + len(valleys))
            except Exception:
                n_osc = 0

    lo, hi = np.quantile(pt_s, [pt_quantile, 1 - pt_quantile])
    early = ex_s[pt_s <= lo]
    late = ex_s[pt_s >= hi]
    early_mean = float(np.mean(early)) if len(early) else np.nan
    late_mean = float(np.mean(late)) if len(late) else np.nan
    direction = int(np.sign(late_mean - early_mean)) if np.isfinite(early_mean) and np.isfinite(late_mean) else 0

    return {
        "gene": gene,
        "monotonicity": float(rho) if np.isfinite(rho) else np.nan,
        "abs_monotonicity": abs(float(rho)) if np.isfinite(rho) else np.nan,
        "p_monotonicity": float(p_mono) if np.isfinite(p_mono) else np.nan,
        "n_oscillations": n_osc,
        "direction": direction,
        "early_expr": early_mean,
        "late_expr": late_mean,
        "delta_expr": late_mean - early_mean if np.isfinite(early_mean) and np.isfinite(late_mean) else np.nan,
    }


def binned_pseudotime_curve(pt: pd.Series, expr_row: pd.Series, n_bins: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    pt_v = pt.values.astype(float)
    ex_v = expr_row.values.astype(float)
    edges = np.quantile(pt_v, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    centers, means = [], []
    for i in range(n_bins):
        m = (pt_v >= edges[i]) & (pt_v < edges[i + 1])
        if m.sum() == 0:
            continue
        centers.append(float(np.mean(pt_v[m])))
        means.append(float(np.mean(ex_v[m])))
    return np.array(centers), np.array(means)


def compute_pattern_table(
    gene_table: pd.DataFrame,
    datasets: Optional[List[str]] = None,
) -> pd.DataFrame:
    rows = []
    subsets = gene_table[gene_table["category"].isin(["hard", "easy"])]
    if datasets:
        subsets = subsets[subsets["dataset"].isin(datasets)]

    cache: Dict[str, Tuple[pd.Series, pd.DataFrame]] = {}
    for _, row in subsets.iterrows():
        ds, gene, cat = row["dataset"], str(row["gene"]).strip(), row["category"]
        if ds not in cache:
            try:
                cache[ds] = load_expression_pseudotime(ds)
            except Exception as exc:
                print(f"  [warn] skip {ds}: {exc}")
                continue
        pt, expr = cache[ds]
        if gene not in expr.index:
            continue
        feat = analyze_gene_expression_pattern(pt, expr.loc[gene], gene)
        feat.update(
            {
                "dataset": ds,
                "category": cat,
                "n_wrong": int(row.get("n_wrong", 0)),
                "frac_wrong": float(row.get("frac_wrong", 0)),
            }
        )
        rows.append(feat)
    return pd.DataFrame(rows)


def pick_example_genes(
    pattern_df: pd.DataFrame,
    dataset: str,
    n: int = N_EXAMPLE,
) -> Tuple[List[dict], List[dict]]:
    sub = pattern_df[pattern_df["dataset"] == dataset]
    hard = sub[sub["category"] == "hard"].copy()
    easy = sub[sub["category"] == "easy"].copy()

    def score_hard(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["cc_hint"] = df["gene"].astype(str).apply(lambda g: bool(CELL_CYCLE_KW.search(g)))
        df["score"] = df["n_oscillations"] * 2 + (1 - df["abs_monotonicity"].fillna(0)) + df["cc_hint"].astype(int) * 3
        prefer = df[df["n_oscillations"] >= 1]
        return (prefer if len(prefer) >= n else df).sort_values("score", ascending=False)

    def score_easy(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ribo_hint"] = df["gene"].astype(str).apply(lambda g: bool(RIBO_KW.search(g)))
        df["score"] = df["abs_monotonicity"].fillna(0) * 2 + df["ribo_hint"].astype(int) * 2 - df["n_oscillations"]
        return df.sort_values("score", ascending=False)

    hard_pick = score_hard(hard).head(n).to_dict("records")
    easy_pick = score_easy(easy).head(n).to_dict("records")
    return hard_pick, easy_pick


def plot_sx1_expression_panels(
    pattern_df: pd.DataFrame,
    dataset: str,
    outpath: Path,
    n_examples: int = N_EXAMPLE,
) -> None:
    apply_fig4_style()
    hard_ex, easy_ex = pick_example_genes(pattern_df, dataset, n_examples)
    if not hard_ex or not easy_ex:
        print(f"  [warn] skip SX-1: insufficient examples for {dataset}")
        return

    pt, expr = load_expression_pseudotime(dataset)
    fig, axes = plt.subplots(n_examples, 2, figsize=(10, 3.2 * n_examples), dpi=DPI, sharex=False)
    if n_examples == 1:
        axes = np.array([axes])

    col_titles = ["Hard", "Easy"]
    for j, (examples, color) in enumerate([(hard_ex, model_color("scPrint")), (easy_ex, model_color("scGPT"))]):
        for i, rec in enumerate(examples[:n_examples]):
            ax = axes[i, j]
            gene = rec["gene"]
            if gene not in expr.index:
                ax.set_visible(False)
                continue
            cx, cy = binned_pseudotime_curve(pt, expr.loc[gene], n_bins=45)
            ax.plot(cx, cy, "-", color=color, lw=2.2)
            ax.fill_between(cx, cy, alpha=0.12, color=color)
            ax.set_ylabel("Expression", fontsize=11)
            dir_lbl = "Up" if rec["direction"] > 0 else ("Down" if rec["direction"] < 0 else "Flat")
            ax.set_title(
                f"{gene}  |  ρ={rec['monotonicity']:+.2f}, oscillations={rec['n_oscillations']}, {dir_lbl}",
                fontsize=11,
                pad=6,
            )
            if i == n_examples - 1:
                ax.set_xlabel("Pseudotime", fontsize=11)

    fig.tight_layout()
    for j, title in enumerate(col_titles):
        pos = axes[0, j].get_position()
        fig.text(
            (pos.x0 + pos.x1) / 2,
            pos.y1 + 0.015,
            title,
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_sx2_monotonicity_violin(pattern_df: pd.DataFrame, outpath: Path) -> None:
    apply_fig4_style()
    df = pattern_df[pattern_df["category"].isin(["hard", "easy"])].copy()
    if df.empty:
        print("  [warn] skip SX-2: empty pattern table")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), dpi=DPI)
    metrics = [
        ("abs_monotonicity", r"|Spearman $\rho$| (monotonicity)"),
        ("n_oscillations", "Oscillation count (peaks + valleys)"),
    ]
    palette = {"hard": model_color("scPrint"), "easy": model_color("scGPT")}

    for ax, (col, ylabel) in zip(axes, metrics):
        sns.violinplot(
            data=df,
            x="category",
            y=col,
            hue="category",
            order=["hard", "easy"],
            palette=palette,
            ax=ax,
            inner=None,
            cut=0,
            legend=False,
        )
        sns.stripplot(
            data=df,
            x="category",
            y=col,
            order=["hard", "easy"],
            color="#333333",
            alpha=0.35,
            size=3,
            jitter=0.18,
            ax=ax,
        )
        h = df.loc[df["category"] == "hard", col].dropna()
        e = df.loc[df["category"] == "easy", col].dropna()
        if len(h) >= 3 and len(e) >= 3:
            _, p = stats.mannwhitneyu(h, e, alternative="two-sided")
            ax.set_title(f"{ylabel}\nMWU p = {p:.3g}")
        else:
            ax.set_title(ylabel)
        ax.set_xlabel("")
        ax.set_xticklabels(["Hard cases", "Easy cases"])

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def run_enrichment(genes: List[str], species: str, tag: str, outdir: Path) -> Optional[pd.DataFrame]:
    if len(genes) < 8:
        print(f"  [warn] skip enrichment {tag}: only {len(genes)} genes")
        return None
    try:
        import gseapy as gp
    except ImportError:
        print("  [warn] gseapy not installed")
        return None

    frames = []
    for lib in ENRICH_LIBS.get(species, ENRICH_LIBS["human"]):
        try:
            enr = gp.enrichr(gene_list=genes, gene_sets=lib, organism=species, outdir=None, cutoff=0.5)
            if enr is None or enr.results is None or enr.results.empty:
                continue
            res = enr.results.copy()
            res["library"] = lib
            frames.append(res)
        except Exception as exc:
            print(f"  [warn] enrichr {lib} ({tag}): {exc}")
    if not frames:
        return None
    full = pd.concat(frames, ignore_index=True)
    full["group"] = tag
    if "Adjusted P-value" in full.columns:
        full["padj"] = pd.to_numeric(full["Adjusted P-value"], errors="coerce")
    else:
        full["padj"] = pd.to_numeric(full.get("P-value", np.nan), errors="coerce")
    full = full.dropna(subset=["padj"]).sort_values("padj")
    full.to_csv(outdir / f"{tag}_enrichment.csv", index=False)
    return full


def pick_top_terms(enr_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    term_col = "Term" if "Term" in enr_df.columns else enr_df.columns[0]
    return enr_df.drop_duplicates(subset=[term_col]).nsmallest(top_n, "padj")


def plot_sx3_enrichment_contrast(
    hard_enr: Optional[pd.DataFrame],
    easy_enr: Optional[pd.DataFrame],
    outpath: Path,
    top_n: int = 10,
) -> None:
    apply_fig4_style()
    if hard_enr is None and easy_enr is None:
        print("  [warn] skip SX-3: no enrichment results")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, top_n * 0.45)), dpi=DPI, sharey=False)
    term_col = "Term"

    for ax, enr, title, color in [
        (axes[0], hard_enr, "Hard cases", model_color("scPrint")),
        (axes[1], easy_enr, "Easy cases", model_color("scGPT")),
    ]:
        if enr is None or enr.empty:
            ax.set_visible(False)
            continue
        top = pick_top_terms(enr, top_n)
        top = top.copy()
        top["neglog10_padj"] = -np.log10(top["padj"].clip(lower=1e-300))
        y = np.arange(len(top))
        ax.barh(y, top["neglog10_padj"], color=color, edgecolor="none")
        ax.set_yticks(y)
        ax.set_yticklabels([str(t)[:72] for t in top[term_col]], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(r"$-\log_{10}$(adj. p-value)")
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Contrasting pathway enrichment: hard vs easy genes", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def summarize_pattern_stats(pattern_df: pd.DataFrame, outpath: Path) -> None:
    rows = []
    for cat in ["hard", "easy"]:
        sub = pattern_df[pattern_df["category"] == cat]
        rows.append(
            {
                "category": cat,
                "n_genes": len(sub),
                "mean_abs_monotonicity": sub["abs_monotonicity"].mean(),
                "median_abs_monotonicity": sub["abs_monotonicity"].median(),
                "mean_n_oscillations": sub["n_oscillations"].mean(),
                "median_n_oscillations": sub["n_oscillations"].median(),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(outpath, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hard/easy gene expression pattern supplement")
    p.add_argument("--gene-table", type=Path, default=DEFAULT_GENE_TABLE)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--dataset", default="hESC", help="Dataset for SX-1 exemplar curves")
    p.add_argument("--datasets", default="", help="Comma-separated; empty = all in gene table")
    p.add_argument("--n-examples", type=int, default=N_EXAMPLE)
    p.add_argument("--no-enrichment", action="store_true")
    p.add_argument("--skip-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    (outdir / "tables").mkdir(exist_ok=True)

    if not args.gene_table.is_file():
        raise SystemExit(
            f"Gene table not found: {args.gene_table}\n"
            "Run: python3 plot_deg_robustness_analysis.py first."
        )

    gene_table = pd.read_csv(args.gene_table)
    ds_filter = [d.strip() for d in args.datasets.split(",") if d.strip()] or None

    print("[1] Computing expression pattern features …")
    pattern_df = compute_pattern_table(gene_table, ds_filter)
    pattern_df.to_csv(outdir / "tables" / "hard_easy_expression_patterns.csv", index=False)
    summarize_pattern_stats(pattern_df, outdir / "tables" / "hard_easy_pattern_summary.csv")

    hard_genes = pattern_df.loc[pattern_df["category"] == "hard", "gene"].unique().tolist()
    easy_genes = pattern_df.loc[pattern_df["category"] == "easy", "gene"].unique().tolist()
    print(f"  patterns: {len(pattern_df)} gene×dataset rows | hard={len(hard_genes)} easy={len(easy_genes)}")

    hard_enr = easy_enr = None
    if not args.no_enrichment:
        print("[2] GO/KEGG enrichment (pooled unique genes) …")
        # Use human for mixed datasets; split if needed
        hard_enr = run_enrichment(hard_genes, "human", "hard_genes", outdir / "tables")
        easy_enr = run_enrichment(easy_genes, "human", "easy_genes", outdir / "tables")

    if args.skip_plots:
        print(f"Tables saved under {outdir / 'tables'}")
        return

    print("[3] Plotting SX-1 / SX-2 / SX-3 …")
    figdir = outdir / "figures"
    plot_sx1_expression_panels(
        pattern_df,
        args.dataset,
        figdir / f"figSX1_expression_patterns_{args.dataset}.pdf",
        n_examples=args.n_examples,
    )
    plot_sx2_monotonicity_violin(pattern_df, figdir / "figSX2_monotonicity_violin.pdf")
    plot_sx3_enrichment_contrast(hard_enr, easy_enr, figdir / "figSX3_enrichment_contrast.pdf")

    # Quick stats print
    h = pattern_df[pattern_df["category"] == "hard"]
    e = pattern_df[pattern_df["category"] == "easy"]
    if len(h) and len(e):
        _, p_rho = stats.mannwhitneyu(h["abs_monotonicity"].dropna(), e["abs_monotonicity"].dropna())
        _, p_osc = stats.mannwhitneyu(h["n_oscillations"].dropna(), e["n_oscillations"].dropna())
        print(f"  |ρ|: hard mean={h['abs_monotonicity'].mean():.3f}, easy mean={e['abs_monotonicity'].mean():.3f}, p={p_rho:.3g}")
        print(f"  oscillations: hard mean={h['n_oscillations'].mean():.2f}, easy mean={e['n_oscillations'].mean():.2f}, p={p_osc:.3g}")

    print(f"\n✅ Done. Figures: {figdir}")


if __name__ == "__main__":
    main()
