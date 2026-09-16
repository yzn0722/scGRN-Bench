#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig4 补充分析：差异表达基因动态预测鲁棒性

子节1 — 表达变化幅度分层（图1-A/B/C）
子节2 — 上调/下调不对称性（图2-A/B/C）
子节3 — 跨模型一致性 & 难例基因生物学（图3-A/B/C/D）

用法:
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig4
  python3 plot_deg_robustness_analysis.py
  python3 plot_deg_robustness_analysis.py --exclude-datasets mDC --no-enrichment
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

warnings.filterwarnings("ignore")

try:
    from fig4_palette import apply_fig4_style, model_color
except ImportError:

    def apply_fig4_style() -> None:
        plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})

    def model_color(name: str, default: str = "#808080") -> str:
        return default


from gene_result_io import load_gene_result, normalize_gene_result_df

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_ROOT = Path("/mnt/10T/yzn/benchmark_GRN")
HGNC_PATH = Path("/mnt/10T/yzn/scFoundation-main/SCAD/data/processing/HGNC_symbol_all_genes.tsv")
CHIP_DIR = BENCH_ROOT / "input_process" / "CHIP"
STRING_DIR = BENCH_ROOT / "input_process" / "STRING"
DEFAULT_OUTDIR = SCRIPT_DIR / "robustness"

TOP_PERCENT = 0.3
CONSERVATIVE_RATIO = 0.5
HARD_GENE_FRAC = 0.5  # >=50% models wrong => hard
EPS = 1e-8
DPI = 600

DATASET_SPECIES = {
    "hESC": "human",
    "hHep": "human",
    "mDC": "mouse",
    "mHSC-E": "mouse",
    "mHSC-GM": "mouse",
    "mHSC-L": "mouse",
}

ALL_DATASETS = ["hESC", "hHep", "mDC", "mHSC-E", "mHSC-GM", "mHSC-L"]
MODEL_ORDER = ["scFoundation", "LangCell", "scPrint", "scGPT", "Geneformer", "scCello"]
MAGNITUDE_LABELS = ("Low |Δ|", "Mid |Δ|", "High |Δ|")

MODEL_PATHS: Dict[str, Tuple[str, str]] = {
    "scGPT": ("gene_result", str(BENCH_ROOT / "pre_scgpt/results_multidataset_pseudotime_227/{ds}_gene_result.csv")),
    "scFoundation": (
        "delta_compare",
        str(BENCH_ROOT / "pre_scfoundation/scfoundation_multidataset_pseudotime_227/{ds}/gene_delta_compare.csv"),
    ),
    "Geneformer": ("gene_result", str(BENCH_ROOT / "pre_geneformer_results_unified/geneformer/{ds}_gene_result.csv")),
    "LangCell": ("gene_result", str(BENCH_ROOT / "pre_langcell_results_unified/langcell/{ds}_gene_result.csv")),
    "scCello": ("gene_result", str(BENCH_ROOT / "pre_sccello_results_unified/sccello/{ds}_gene_result.csv")),
    "scPrint": (
        "scprint",
        str(BENCH_ROOT / "pre_scprint_results_unified/scprint/per_dataset/{ds}/per_gene_final_changes.csv"),
    ),
}

# Per-iteration prediction sources for tradeoff / Pareto analysis (Fig 2-A1/A2)
MODEL_ITER_SOURCES: Dict[str, Tuple[str, str]] = {
    "scGPT": (
        "native",
        str(BENCH_ROOT / "dyn4_results_unified/scgpt/per_dataset/{ds}/pred_delta_by_iter.npy"),
    ),
    "Geneformer": (
        "rank",
        str(BENCH_ROOT / "pre_geneformer_results_unified/geneformer/per_dataset/{ds}/mean_rank_delta_by_iter.npy"),
    ),
    "LangCell": (
        "rank",
        str(BENCH_ROOT / "pre_langcell_results_unified/langcell/per_dataset/{ds}/mean_rank_delta_by_iter.npy"),
    ),
    "scCello": (
        "rank",
        str(BENCH_ROOT / "pre_sccello_results_unified/sccello/per_dataset/{ds}/mean_rank_delta_by_iter.npy"),
    ),
    "scPrint": (
        "scprint_iter",
        str(BENCH_ROOT / "pre_scprint_results_unified/scprint/per_dataset/{ds}/pred_delta_by_iter.npy"),
    ),
    "scFoundation": (
        "acc_interp",
        str(BENCH_ROOT / "pre_scfoundation/scfoundation_multidataset_pseudotime_227/{ds}/acc_curve.npy"),
    ),
}

MODEL_MARKERS = {
    "scFoundation": "s",
    "LangCell": "D",
    "scPrint": "v",
    "scGPT": "o",
    "Geneformer": "^",
    "scCello": "X",
}

N_TAU_GRID = 16

ENRICH_LIBS = {
    "human": ["GO_Biological_Process_2023", "KEGG_2021_Human"],
    "mouse": ["GO_Biological_Process_2023", "KEGG_2019_Mouse"],
}

GENE_TYPE_COLORS = {
    "TF": "#FF9A3D",
    "Kinase": "#4EA3F1",
    "Surface protein": "#AC99D2",
    "Other": "#B0B0B0",
}

DATASET_MARKERS = {"hESC": "o", "hHep": "s", "mDC": "D", "mHSC-E": "^", "mHSC-GM": "v", "mHSC-L": "P"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_hgnc_maps() -> Tuple[Dict[str, str], Set[str], Set[str]]:
    """Ensembl→symbol, kinase set, surface-protein set from HGNC names."""
    ens_to_sym: Dict[str, str] = {}
    kinases: Set[str] = set()
    surface: Set[str] = set()
    if not HGNC_PATH.is_file():
        print(f"[warn] HGNC file missing: {HGNC_PATH}")
        return ens_to_sym, kinases, surface

    df = pd.read_csv(HGNC_PATH, sep="\t", low_memory=False)
    for _, row in df.iterrows():
        sym = str(row.get("Approved symbol", "")).strip()
        ens = str(row.get("Ensembl gene ID", "")).strip()
        name = str(row.get("Approved name", "")).lower()
        if ens and ens != "nan" and sym and sym != "nan":
            ens_to_sym[ens] = sym
        if not sym or sym == "nan":
            continue
        if "kinase" in name:
            kinases.add(sym.upper())
        if any(kw in name for kw in ("receptor", "transmembrane", "cell surface", "integrin")):
            surface.add(sym.upper())
    return ens_to_sym, kinases, surface


def load_tf_set(dataset: str) -> Set[str]:
    tfs: Set[str] = set()
    tf_list = STRING_DIR / "grnboost_results" / f"{dataset}_tf_list.txt"
    if tf_list.is_file():
        tfs.update(line.strip().upper() for line in tf_list.read_text().splitlines() if line.strip())
    chip_net = CHIP_DIR / f"{dataset}_chip_matched-network.csv"
    if chip_net.is_file():
        net = pd.read_csv(chip_net)
        g1 = net.columns[0]
        tfs.update(net[g1].astype(str).str.strip().str.upper())
    return tfs


def load_scprint_gene_result(path: Path, ens_to_sym: Dict[str, str]) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw = raw.copy()
    raw["gene"] = raw["gene"].astype(str).str.strip()
    raw["gene"] = raw["gene"].map(lambda g: ens_to_sym.get(g, g))
    raw = raw.rename(columns={"true_delta": "delta_true", "pred_delta": "delta_pred"})
    raw["delta_true"] = pd.to_numeric(raw["delta_true"], errors="coerce")
    raw["delta_pred"] = pd.to_numeric(raw["delta_pred"], errors="coerce")
    raw["dir_true"] = np.where(raw["delta_true"] > 0, "Up", "Down")
    raw["dir_pred"] = np.where(raw["delta_pred"] > 0, "Up", "Down")
    raw["dir_correct"] = (raw["dir_true"] == raw["dir_pred"]).astype(int)
    raw["true_early_mean"] = np.nan
    raw["true_late_mean"] = np.nan
    raw["pred_late_like_mean"] = np.nan
    raw = raw.dropna(subset=["delta_true", "delta_pred"])
    return normalize_gene_result_df(raw, mapped_only=False)


def load_model_gene_result(model: str, dataset: str, ens_to_sym: Dict[str, str]) -> Optional[pd.DataFrame]:
    kind, pattern = MODEL_PATHS[model]
    path = Path(pattern.format(ds=dataset))
    if not path.is_file():
        print(f"  [warn] missing {model}/{dataset}: {path}")
        return None
    if kind == "scprint":
        return load_scprint_gene_result(path, ens_to_sym)
    mapped_only = kind == "delta_compare"
    return load_gene_result(path, mapped_only=mapped_only)


def select_top_genes(df: pd.DataFrame, top_percent: float = TOP_PERCENT) -> pd.DataFrame:
    out = df.copy()
    out["abs_delta_true"] = out["delta_true"].abs()
    n_top = max(1, int(np.ceil(top_percent * len(out))))
    out = out.nlargest(n_top, "abs_delta_true").copy()
    return out


def assign_magnitude_tier(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    try:
        out["magnitude_tier"] = pd.qcut(
            out["abs_delta_true"],
            q=3,
            labels=MAGNITUDE_LABELS,
            duplicates="drop",
        ).astype(str)
    except ValueError:
        out["magnitude_tier"] = "Mid |Δ|"
    return out


def is_conservative(row: pd.Series, ratio: float = CONSERVATIVE_RATIO) -> bool:
    denom = abs(float(row["delta_true"])) + EPS
    return abs(float(row["delta_pred"])) / denom < ratio


def direction_accuracy(sub: pd.DataFrame) -> float:
    if sub.empty:
        return np.nan
    return float(sub["dir_correct"].mean())


def updown_accuracy_arrays(
    true_delta: np.ndarray,
    pred_delta: np.ndarray,
    top_percent: float = TOP_PERCENT,
) -> Tuple[float, float, int, int]:
    """Return (up_acc, down_acc, n_up, n_down) on top-percent |true_delta| genes."""
    n_top = max(1, int(np.ceil(top_percent * len(true_delta))))
    idx = np.argsort(np.abs(true_delta))[-n_top:]
    td = true_delta[idx]
    pd = pred_delta[idx]
    true_dir = np.where(td > 0, 1, -1)
    pred_dir = np.where(pd > 0, 1, -1)
    ok = pred_dir == true_dir
    up = td > 0
    down = ~up
    n_up = int(up.sum())
    n_down = int(down.sum())
    up_acc = float(ok[up].mean()) if n_up > 0 else np.nan
    down_acc = float(ok[down].mean()) if n_down > 0 else np.nan
    return up_acc, down_acc, n_up, n_down


def _row_synthetic(row: pd.Series) -> bool:
    v = row.get("synthetic", False)
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        return False
    return bool(v)


def _load_scprint_iter_arrays(
    dataset: str,
    ens_to_sym: Dict[str, str],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """scPRINT per_gene row order matches pred_delta_by_iter columns."""
    pg_path = BENCH_ROOT / f"pre_scprint_results_unified/scprint/per_dataset/{dataset}/per_gene_final_changes.csv"
    it_path = BENCH_ROOT / f"pre_scprint_results_unified/scprint/per_dataset/{dataset}/pred_delta_by_iter.npy"
    if not pg_path.is_file() or not it_path.is_file():
        return None
    pg = pd.read_csv(pg_path)
    iters = np.load(it_path)
    pg["gene"] = pg["gene"].astype(str).str.strip().map(lambda g: ens_to_sym.get(g, g))
    td = pd.to_numeric(pg["true_delta"], errors="coerce").values
    mask = np.isfinite(td)
    td = td[mask]
    iters = iters[:, mask]
    return td, iters


def load_iter_prediction_trajectory(
    model: str,
    dataset: str,
    ens_to_sym: Dict[str, str],
    top_percent: float = TOP_PERCENT,
) -> Optional[pd.DataFrame]:
    """
    Per-iteration (down_acc, up_acc) for one model×dataset.
    Uses native per-iter arrays when saved; scFoundation falls back to
    acc_curve-interpolated endpoints (zero-pred start → final pred).
    """
    if model not in MODEL_ITER_SOURCES:
        return None
    kind, pattern = MODEL_ITER_SOURCES[model]
    rows: List[dict] = []

    if kind == "native":
        path = Path(pattern.format(ds=dataset))
        gr = load_model_gene_result("scGPT", dataset, ens_to_sym)
        if not path.is_file() or gr is None:
            return None
        iters = np.load(path)
        td = gr["delta_true"].values.astype(float)
        n = min(len(td), iters.shape[1])
        td, iters = td[:n], iters[:, :n]
        for it in range(iters.shape[0]):
            up_acc, down_acc, n_up, n_down = updown_accuracy_arrays(td, iters[it], top_percent)
            rows.append(
                {
                    "iter": it,
                    "up_accuracy": up_acc,
                    "down_accuracy": down_acc,
                    "n_up_top30": n_up,
                    "n_down_top30": n_down,
                }
            )

    elif kind == "rank":
        path = Path(pattern.format(ds=dataset))
        gr = load_model_gene_result(model, dataset, ens_to_sym)
        if not path.is_file() or gr is None:
            return None
        rank_delta = np.load(path)
        td = gr["delta_true"].values.astype(float)
        n = min(len(td), rank_delta.shape[1])
        td, rank_delta = td[:n], rank_delta[:, :n]
        for it in range(rank_delta.shape[0]):
            pred_delta = -rank_delta[it]
            up_acc, down_acc, n_up, n_down = updown_accuracy_arrays(td, pred_delta, top_percent)
            rows.append(
                {
                    "iter": it,
                    "up_accuracy": up_acc,
                    "down_accuracy": down_acc,
                    "n_up_top30": n_up,
                    "n_down_top30": n_down,
                }
            )

    elif kind == "scprint_iter":
        loaded = _load_scprint_iter_arrays(dataset, ens_to_sym)
        if loaded is None:
            return None
        td, iters = loaded
        for it in range(iters.shape[0]):
            up_acc, down_acc, n_up, n_down = updown_accuracy_arrays(td, iters[it], top_percent)
            rows.append(
                {
                    "iter": it,
                    "up_accuracy": up_acc,
                    "down_accuracy": down_acc,
                    "n_up_top30": n_up,
                    "n_down_top30": n_down,
                }
            )

    elif kind == "acc_interp":
        acc_path = Path(pattern.format(ds=dataset))
        gr = load_model_gene_result(model, dataset, ens_to_sym)
        if not acc_path.is_file() or gr is None:
            return None
        acc = np.load(acc_path).astype(float)
        td = gr["delta_true"].values.astype(float)
        final_pred = gr["delta_pred"].values.astype(float)
        zero_pred = np.zeros_like(final_pred)
        up0, down0, n_up, n_down = updown_accuracy_arrays(td, zero_pred, top_percent)
        upT, downT, _, _ = updown_accuracy_arrays(td, final_pred, top_percent)
        for it, acc_val in enumerate(acc):
            tau = float(acc_val / acc[-1]) if acc[-1] > 0 else float(it / max(len(acc) - 1, 1))
            up_i = up0 + tau * (upT - up0) if np.isfinite(up0) and np.isfinite(upT) else np.nan
            down_i = down0 + tau * (downT - down0) if np.isfinite(down0) and np.isfinite(downT) else np.nan
            rows.append(
                {
                    "iter": it,
                    "up_accuracy": up_i,
                    "down_accuracy": down_i,
                    "n_up_top30": n_up,
                    "n_down_top30": n_down,
                    "synthetic": True,
                }
            )
    else:
        return None

    if not rows:
        return None
    out = pd.DataFrame(rows)
    out["model"] = model
    out["dataset"] = dataset
    out["n_iters"] = len(out)
    return out


def compute_tradeoff_trajectories(
    datasets: List[str],
    models: List[str],
    ens_to_sym: Dict[str, str],
    top_percent: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Long-form iter trajectories + dataset endpoints (start/end)."""
    traj_rows: List[pd.DataFrame] = []
    for ds in datasets:
        for model in models:
            tdf = load_iter_prediction_trajectory(model, ds, ens_to_sym, top_percent)
            if tdf is not None and not tdf.empty:
                traj_rows.append(tdf)
    if not traj_rows:
        return pd.DataFrame(), pd.DataFrame()

    traj = pd.concat(traj_rows, ignore_index=True)
    endpoints = []
    for (model, ds), sub in traj.groupby(["model", "dataset"]):
        sub = sub.sort_values("iter")
        for label, row in [("start", sub.iloc[0]), ("end", sub.iloc[-1])]:
            endpoints.append(
                {
                    "model": model,
                    "dataset": ds,
                    "phase": label,
                    "iter": int(row["iter"]),
                    "up_accuracy": row["up_accuracy"],
                    "down_accuracy": row["down_accuracy"],
                    "synthetic": _row_synthetic(row),
                }
            )
    return traj, pd.DataFrame(endpoints)


def _resample_trajectory_to_tau(
    sub: pd.DataFrame, n_grid: int = N_TAU_GRID
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Resample one trajectory; return None if insufficient finite points."""
    sub = sub.sort_values("iter")
    down_v = sub["down_accuracy"].astype(float).values
    up_v = sub["up_accuracy"].astype(float).values
    finite = np.isfinite(down_v) & np.isfinite(up_v)
    if finite.sum() < 2:
        return None
    down_v = down_v[finite]
    up_v = up_v[finite]
    n = len(down_v)
    tau_orig = np.linspace(0, 1, n)
    tau_new = np.linspace(0, 1, n_grid)
    down = np.interp(tau_new, tau_orig, down_v)
    up = np.interp(tau_new, tau_orig, up_v)
    return tau_new, down, up


def average_model_pareto_curves(traj: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    """Average resampled (down, up) across datasets per model (nan-aware)."""
    rows = []
    for model in models:
        parts = []
        for _, sub in traj[traj["model"] == model].groupby("dataset"):
            res = _resample_trajectory_to_tau(sub)
            if res is None:
                continue
            _, down, up = res
            parts.append(np.stack([down, up], axis=1))
        if not parts:
            continue
        mean_curve = np.nanmean(np.stack(parts, axis=0), axis=0)
        tau = np.linspace(0, 1, N_TAU_GRID)
        for i, t in enumerate(tau):
            rows.append(
                {
                    "model": model,
                    "tau": t,
                    "down_accuracy": mean_curve[i, 0],
                    "up_accuracy": mean_curve[i, 1],
                    "n_datasets": len(parts),
                }
            )
    return pd.DataFrame(rows)


def fit_tradeoff_curvature(pareto_df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    """Quadratic fit up = a*down^2 + b*down + c (x=down acc, y=up acc); a<0 convex."""
    rows = []
    for model in models:
        sub = pareto_df[pareto_df["model"] == model].sort_values("tau")
        mask = np.isfinite(sub["down_accuracy"]) & np.isfinite(sub["up_accuracy"])
        sub = sub[mask]
        if len(sub) < 3:
            continue
        # Fit on percentage scale so quadratic coefficient a is interpretable (≈ ±0.1–0.3)
        x = sub["down_accuracy"].values.astype(float) * 100.0
        y = sub["up_accuracy"].values.astype(float) * 100.0
        if np.unique(np.round(x, 4)).size < 3:
            continue
        try:
            coef = np.polyfit(x, y, 2)
            a, b, c = float(coef[0]), float(coef[1]), float(coef[2])
            y_hat = np.polyval(coef, x)
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        except (np.linalg.LinAlgError, ValueError):
            a, b, c, r2 = np.nan, np.nan, np.nan, np.nan
        shape = "convex" if np.isfinite(a) and a < -0.01 else ("concave" if np.isfinite(a) and a > 0.01 else "near-linear")
        rows.append(
            {
                "model": model,
                "curvature_a": a,
                "linear_b": b,
                "intercept_c": c,
                "r2": r2,
                "shape": shape,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("curvature_a", na_position="last")
    return out


def _finite_endpoints(endpoints_df: pd.DataFrame) -> pd.DataFrame:
    m = endpoints_df["up_accuracy"].map(np.isfinite) & endpoints_df["down_accuracy"].map(np.isfinite)
    return endpoints_df[m].copy()


def classify_gene_type(gene: str, tf_set: Set[str], kinases: Set[str], surface: Set[str]) -> str:
    g = str(gene).strip().upper()
    if g in tf_set:
        return "TF"
    if g in kinases:
        return "Kinase"
    if g in surface or re.match(r"^CD[0-9]+$", g):
        return "Surface protein"
    return "Other"


def build_network_centrality(dataset: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    import networkx as nx

    net_path = STRING_DIR / f"{dataset}_processed-network.csv"
    if not net_path.is_file():
        return {}, {}
    edges = pd.read_csv(net_path)
    g1, g2 = edges.columns[0], edges.columns[1]
    G = nx.Graph()
    for a, b in zip(edges[g1].astype(str), edges[g2].astype(str)):
        G.add_edge(a.strip().upper(), b.strip().upper())
    deg = nx.degree_centrality(G)
    btw = nx.betweenness_centrality(G)
    return deg, btw


# ---------------------------------------------------------------------------
# Metric tables
# ---------------------------------------------------------------------------
def compute_magnitude_metrics(
    datasets: List[str],
    models: List[str],
    ens_to_sym: Dict[str, str],
    top_percent: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-model×tier accuracy (1-A) and per-dataset×model conservative fraction on low tier (1-B)."""
    acc_rows: List[dict] = []
    cons_rows: List[dict] = []

    for ds in datasets:
        for model in models:
            df = load_model_gene_result(model, ds, ens_to_sym)
            if df is None:
                continue
            top = assign_magnitude_tier(select_top_genes(df, top_percent))
            for tier in MAGNITUDE_LABELS:
                sub = top[top["magnitude_tier"] == tier]
                acc_rows.append(
                    {
                        "dataset": ds,
                        "model": model,
                        "magnitude_tier": tier,
                        "n_genes": len(sub),
                        "direction_accuracy": direction_accuracy(sub),
                    }
                )
            low = top[top["magnitude_tier"] == MAGNITUDE_LABELS[0]]
            if len(low):
                cons_frac = float(low.apply(is_conservative, axis=1).mean())
                cons_rows.append(
                    {
                        "dataset": ds,
                        "model": model,
                        "conservative_fraction": cons_frac,
                        "n_low_genes": len(low),
                    }
                )

    return pd.DataFrame(acc_rows), pd.DataFrame(cons_rows)


def compute_asymmetry_metrics(
    datasets: List[str],
    models: List[str],
    ens_to_sym: Dict[str, str],
    top_percent: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Up/down accuracy and preference index per model×dataset."""
    rows: List[dict] = []
    imbalance_rows: List[dict] = []

    for ds in datasets:
        ref = load_model_gene_result("scGPT", ds, ens_to_sym)
        if ref is None:
            continue
        ref_top = select_top_genes(ref, top_percent)
        n_up = int((ref_top["dir_true"] == "Up").sum())
        n_down = int((ref_top["dir_true"] == "Down").sum())
        imbalance_rows.append(
            {
                "dataset": ds,
                "n_up": n_up,
                "n_down": n_down,
                "up_fraction": n_up / max(len(ref_top), 1),
            }
        )

        for model in models:
            df = load_model_gene_result(model, ds, ens_to_sym)
            if df is None:
                continue
            top = select_top_genes(df, top_percent)
            up_sub = top[top["dir_true"] == "Up"]
            down_sub = top[top["dir_true"] == "Down"]
            up_acc = direction_accuracy(up_sub)
            down_acc = direction_accuracy(down_sub)
            rows.append(
                {
                    "dataset": ds,
                    "model": model,
                    "up_accuracy": up_acc,
                    "down_accuracy": down_acc,
                    "overall_accuracy": direction_accuracy(top),
                    "preference_index": up_acc - down_acc if np.isfinite(up_acc) and np.isfinite(down_acc) else np.nan,
                    "n_up": len(up_sub),
                    "n_down": len(down_sub),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(imbalance_rows)


def compute_consistency_tables(
    datasets: List[str],
    models: List[str],
    ens_to_sym: Dict[str, str],
    top_percent: float,
    hard_frac: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Jaccard matrix, per-gene consensus, hard/easy gene lists."""
    jacc_rows: List[dict] = []
    gene_rows: List[dict] = []

    for ds in datasets:
        per_model: Dict[str, pd.DataFrame] = {}
        for model in models:
            df = load_model_gene_result(model, ds, ens_to_sym)
            if df is None:
                continue
            per_model[model] = select_top_genes(df, top_percent).set_index("gene")

        if len(per_model) < 2:
            continue

        common_genes = set.intersection(*(set(d.index) for d in per_model.values()))
        if not common_genes:
            continue

        for g in common_genes:
            n_wrong = sum(int(per_model[m].loc[g, "dir_correct"] == 0) for m in per_model)
            n_models = len(per_model)
            gene_rows.append(
                {
                    "dataset": ds,
                    "gene": g,
                    "n_models": n_models,
                    "n_wrong": n_wrong,
                    "frac_wrong": n_wrong / n_models,
                    "category": "hard" if n_wrong / n_models >= hard_frac else ("easy" if n_wrong == 0 else "mixed"),
                }
            )

        model_list = [m for m in models if m in per_model]
        for i, m1 in enumerate(model_list):
            for m2 in model_list[i:]:
                c1 = {g for g in common_genes if per_model[m1].loc[g, "dir_correct"] == 1}
                c2 = {g for g in common_genes if per_model[m2].loc[g, "dir_correct"] == 1}
                w1 = {g for g in common_genes if per_model[m1].loc[g, "dir_correct"] == 0}
                w2 = {g for g in common_genes if per_model[m2].loc[g, "dir_correct"] == 0}
                j_correct = len(c1 & c2) / max(len(c1 | c2), 1)
                j_wrong = len(w1 & w2) / max(len(w1 | w2), 1)
                jacc_rows.append(
                    {
                        "dataset": ds,
                        "model_a": m1,
                        "model_b": m2,
                        "jaccard_correct": j_correct,
                        "jaccard_wrong": j_wrong,
                    }
                )

    gene_df = pd.DataFrame(gene_rows)
    jacc_df = pd.DataFrame(jacc_rows)

    if jacc_df.empty:
        jacc_matrix = pd.DataFrame()
    else:
        pooled = (
            jacc_df.groupby(["model_a", "model_b"], as_index=False)[["jaccard_correct", "jaccard_wrong"]]
            .mean()
        )
        jacc_matrix = pooled

    return jacc_matrix, gene_df, jacc_df


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------
def run_go_enrichment(genes: List[str], species: str, outdir: Path, tag: str) -> Optional[pd.DataFrame]:
    if len(genes) < 8:
        return None
    try:
        import gseapy as gp
    except ImportError:
        print("  [warn] gseapy not installed; skip GO enrichment")
        return None

    libs = ENRICH_LIBS.get(species, ENRICH_LIBS["human"])
    frames = []
    for lib in libs:
        try:
            enr = gp.enrichr(gene_list=genes, gene_sets=lib, organism=species, outdir=None, cutoff=0.5)
            if enr is None or enr.results is None or enr.results.empty:
                continue
            res = enr.results.copy()
            res["library"] = lib
            frames.append(res)
        except Exception as exc:
            print(f"  [warn] enrichr {lib}: {exc}")

    if not frames:
        return None
    full = pd.concat(frames, ignore_index=True)
    if "Adjusted P-value" in full.columns:
        full["padj"] = pd.to_numeric(full["Adjusted P-value"], errors="coerce")
    else:
        full["padj"] = pd.to_numeric(full.get("P-value", np.nan), errors="coerce")
    full = full.dropna(subset=["padj"]).sort_values("padj")
    full.to_csv(outdir / f"{tag}_go_enrichment.csv", index=False)
    return full


# ---------------------------------------------------------------------------
# Plotting — Subsection 1
# ---------------------------------------------------------------------------
def plot_fig1a_magnitude_accuracy(acc_df: pd.DataFrame, models: List[str], outpath: Path) -> None:
    apply_fig4_style()
    agg = (
        acc_df.groupby(["model", "magnitude_tier"], as_index=False)["direction_accuracy"]
        .mean()
        .rename(columns={"direction_accuracy": "acc"})
    )

    fig, ax = plt.subplots(figsize=(9, 5), dpi=DPI)
    x = np.arange(len(MAGNITUDE_LABELS))
    width = 0.12
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * width

    for mi, model in enumerate(models):
        sub = agg[agg["model"] == model].set_index("magnitude_tier")
        ys = [sub.loc[t, "acc"] * 100 if t in sub.index else np.nan for t in MAGNITUDE_LABELS]
        ax.bar(x + offsets[mi], ys, width=width * 0.92, label=model, color=model_color(model), edgecolor="none")

    ax.axhline(50, color="#888888", ls="--", lw=1.0, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(MAGNITUDE_LABELS)
    ax.set_ylabel("Direction accuracy (%)")
    ax.set_xlabel("Dynamic magnitude tier (|Δ_true| tertile within top 30%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14), fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_fig1b_conservative_heatmap(cons_df: pd.DataFrame, models: List[str], datasets: List[str], outpath: Path) -> None:
    apply_fig4_style()
    pivot = cons_df.pivot(index="dataset", columns="model", values="conservative_fraction")
    pivot = pivot.reindex(index=datasets, columns=[m for m in models if m in pivot.columns])

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=DPI)
    cmap = LinearSegmentedColormap.from_list("cons", ["#F7FBFF", "#FF9A3D"])
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Conservative prediction fraction"},
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Dataset")
    ax.set_title(f"Low-|Δ| genes: fraction with |Δ_pred|/|Δ_true| < {CONSERVATIVE_RATIO}")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_fig1c_conservative_boxplot(cons_df: pd.DataFrame, outpath: Path) -> None:
    apply_fig4_style()
    fig, ax = plt.subplots(figsize=(4.5, 5), dpi=DPI)
    data = [cons_df["conservative_fraction"].dropna().values]
    bp = ax.boxplot(data, widths=0.45, patch_artist=True, showfliers=False)
    bp["boxes"][0].set_facecolor(model_color("scGPT"))
    bp["boxes"][0].set_alpha(0.35)
    jitter_x = 1 + np.random.default_rng(42).normal(0, 0.06, size=len(data[0]))
    ax.scatter(jitter_x, data[0], s=28, alpha=0.75, color=model_color("scFoundation"), edgecolors="white", linewidths=0.4)
    ax.set_xticks([1])
    ax.set_xticklabels(["Low |Δ| tier"])
    ax.set_ylabel("Conservative prediction fraction")
    ax.set_title("Across datasets & models")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plotting — Subsection 2 (Fig 2-A1/A2/A3 tradeoff panel)
# ---------------------------------------------------------------------------
def _style_updown_axes(ax: plt.Axes, title: str = "") -> None:
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.plot([0, 105], [0, 105], color="#CCCCCC", ls="--", lw=0.9, zorder=0)
    ax.set_xlabel("Down-regulation accuracy (%)")
    ax.set_ylabel("Up-regulation accuracy (%)")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=13, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_fig2a1_pareto_frontier(
    pareto_df: pd.DataFrame,
    endpoints_df: pd.DataFrame,
    models: List[str],
    outpath: Path,
) -> None:
    apply_fig4_style()
    fig, ax = plt.subplots(figsize=(7.2, 6.5), dpi=DPI)
    ep = _finite_endpoints(endpoints_df)

    for model in models:
        sub = pareto_df[pareto_df["model"] == model].sort_values("tau")
        if sub.empty:
            continue
        x = sub["down_accuracy"].values * 100
        y = sub["up_accuracy"].values * 100
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 2:
            continue
        ax.plot(x[ok], y[ok], "-", color=model_color(model), lw=2.4, label=model, zorder=2)

        ep_m = ep[ep["model"] == model]
        if ep_m.empty:
            continue
        starts = ep_m[ep_m["phase"] == "start"]
        ends = ep_m[ep_m["phase"] == "end"]
        ax.scatter(
            starts["down_accuracy"] * 100,
            starts["up_accuracy"] * 100,
            s=42,
            facecolors="none",
            edgecolors=model_color(model),
            linewidths=1.2,
            zorder=3,
            alpha=0.65,
        )
        for _, row in ends.iterrows():
            ds = row["dataset"]
            ax.scatter(
                row["down_accuracy"] * 100,
                row["up_accuracy"] * 100,
                s=58,
                marker=MODEL_MARKERS.get(model, "o"),
                color=model_color(model),
                edgecolors="white",
                linewidths=0.6,
                zorder=4,
            )
            ax.scatter(
                [],
                [],
                marker=DATASET_MARKERS.get(ds, "o"),
                color=model_color(model),
                label=f"{model} end ({ds})" if False else None,
            )

    _style_updown_axes(ax, "Pareto frontier (mean trajectory across datasets)")
    ax.legend(frameon=False, fontsize=10, loc="lower right", title="Model mean curve")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_fig2a2_drift_arrows(
    endpoints_df: pd.DataFrame,
    models: List[str],
    outpath: Path,
) -> None:
    apply_fig4_style()
    fig, ax = plt.subplots(figsize=(7.2, 6.5), dpi=DPI)
    ep = _finite_endpoints(endpoints_df)

    for _, grp in ep.groupby(["model", "dataset"]):
        start = grp[grp["phase"] == "start"]
        end = grp[grp["phase"] == "end"]
        if start.empty or end.empty:
            continue
        model = str(start.iloc[0]["model"])
        ds = str(start.iloc[0]["dataset"])
        x0, y0 = float(start.iloc[0]["down_accuracy"]) * 100, float(start.iloc[0]["up_accuracy"]) * 100
        x1, y1 = float(end.iloc[0]["down_accuracy"]) * 100, float(end.iloc[0]["up_accuracy"]) * 100
        if not all(np.isfinite([x0, y0, x1, y1])):
            continue
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=model_color(model),
                lw=1.8,
                shrinkA=3,
                shrinkB=3,
                alpha=0.75,
            ),
            zorder=2,
        )
        ax.scatter([x0], [y0], s=32, facecolors="none", edgecolors=model_color(model), linewidths=1.0, zorder=3)
        ax.scatter(
            [x1],
            [y1],
            s=48,
            marker=MODEL_MARKERS.get(model, "o"),
            color=model_color(model),
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
        )

    _style_updown_axes(ax, "Preference drift (iteration 0 → final)")
    model_handles = [
        plt.Line2D([0], [0], color=model_color(m), lw=2.2, label=m)
        for m in models
        if m in set(ep["model"])
    ]
    ds_handles = [
        plt.Line2D([0], [0], marker=DATASET_MARKERS.get(d, "o"), color="#666666", ls="", ms=7, label=d)
        for d in ALL_DATASETS
        if d in set(ep["dataset"])
    ]
    ax.legend(handles=model_handles + ds_handles, frameon=False, fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_fig2a3_curvature_bar(curvature_df: pd.DataFrame, models: List[str], outpath: Path) -> None:
    apply_fig4_style()
    if curvature_df.empty:
        print("  [warn] skip fig2-A3: empty curvature table")
        return

    df = curvature_df[np.isfinite(curvature_df["curvature_a"])].copy()
    order = [m for m in models if m in set(df["model"])]
    df = df.set_index("model").reindex(order).dropna(subset=["curvature_a"])
    if df.empty:
        print("  [warn] skip fig2-A3: no finite curvature coefficients")
        return

    fig, ax = plt.subplots(figsize=(7.5, max(3.8, 0.5 * len(df))), dpi=DPI)
    y = np.arange(len(df))
    vals = df["curvature_a"].values
    colors = [model_color(m) for m in df.index]
    ax.barh(y, vals, color=colors, edgecolor="none", height=0.62)
    ax.axvline(0, color="#888888", lw=1.0)
    ax.set_yticks(y)
    labels = [f"{m}  ({df.loc[m, 'shape']}, a={df.loc[m, 'curvature_a']:+.3f})" for m in df.index]
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(r"Quadratic coefficient $a$  ($y = ax^2 + bx + c$; $x$=down, $y$=up)")
    ax.set_title("Tradeoff curvature (more negative $a$ → more convex / Pareto-efficient)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_fig2a_combined_panel(
    pareto_df: pd.DataFrame,
    endpoints_df: pd.DataFrame,
    curvature_df: pd.DataFrame,
    models: List[str],
    outpath: Path,
) -> None:
    """Single-row 3-panel figure for Fig 2-A (paper layout)."""
    apply_fig4_style()
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), dpi=DPI)
    ep = _finite_endpoints(endpoints_df)

    ax = axes[0]
    for model in models:
        sub = pareto_df[pareto_df["model"] == model].sort_values("tau")
        x = sub["down_accuracy"].values * 100
        y = sub["up_accuracy"].values * 100
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 2:
            ax.plot(x[ok], y[ok], "-", color=model_color(model), lw=2.2, label=model)
        ep_m = ep[ep["model"] == model]
        ax.scatter(
            ep_m[ep_m["phase"] == "start"]["down_accuracy"] * 100,
            ep_m[ep_m["phase"] == "start"]["up_accuracy"] * 100,
            s=28,
            facecolors="none",
            edgecolors=model_color(model),
            linewidths=1.0,
            zorder=3,
        )
        ax.scatter(
            ep_m[ep_m["phase"] == "end"]["down_accuracy"] * 100,
            ep_m[ep_m["phase"] == "end"]["up_accuracy"] * 100,
            s=40,
            marker=MODEL_MARKERS.get(model, "o"),
            color=model_color(model),
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
        )
    _style_updown_axes(ax, "A1  Pareto frontier")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    for _, grp in ep.groupby(["model", "dataset"]):
        start = grp[grp["phase"] == "start"]
        end = grp[grp["phase"] == "end"]
        if start.empty or end.empty:
            continue
        model = str(start.iloc[0]["model"])
        x0, y0 = float(start.iloc[0]["down_accuracy"]) * 100, float(start.iloc[0]["up_accuracy"]) * 100
        x1, y1 = float(end.iloc[0]["down_accuracy"]) * 100, float(end.iloc[0]["up_accuracy"]) * 100
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=model_color(model), lw=1.5, alpha=0.75),
        )
        ax.scatter([x1], [y1], s=34, marker=MODEL_MARKERS.get(model, "o"), color=model_color(model), edgecolors="white", linewidths=0.4)
    _style_updown_axes(ax, "A2  Preference drift")

    ax = axes[2]
    cur = curvature_df[np.isfinite(curvature_df["curvature_a"])].copy()
    order = [m for m in models if m in set(cur["model"])]
    cur = cur.set_index("model").reindex(order).dropna(subset=["curvature_a"])
    if not cur.empty:
        y = np.arange(len(cur))
        ax.barh(y, cur["curvature_a"].values, color=[model_color(m) for m in cur.index], height=0.62)
        ax.axvline(0, color="#888888", lw=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels([str(m) for m in cur.index], fontsize=10)
        ax.set_xlabel(r"$a$ (quadratic term)")
        ax.set_title("A3  Curvature")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_fig2a_updown_scatter(asym_df: pd.DataFrame, outpath: Path) -> None:
    """Legacy final-iteration scatter (kept for backward compatibility)."""
    apply_fig4_style()
    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=DPI)

    for _, row in asym_df.iterrows():
        ds = row["dataset"]
        model = row["model"]
        x = row["down_accuracy"] * 100
        y = row["up_accuracy"] * 100
        overall = row["overall_accuracy"] * 100
        ax.scatter(
            x,
            y,
            s=40 + overall * 1.2,
            marker=DATASET_MARKERS.get(ds, "o"),
            color=model_color(model),
            alpha=0.78,
            edgecolors="white",
            linewidths=0.5,
        )

    _style_updown_axes(ax, "Final-iteration up/down accuracy")
    model_handles = [mpatches.Patch(color=model_color(m), label=m) for m in MODEL_ORDER if m in set(asym_df["model"])]
    ds_handles = [
        plt.Line2D([0], [0], marker=DATASET_MARKERS.get(d, "o"), color="gray", ls="", ms=8, label=d)
        for d in ALL_DATASETS
        if d in set(asym_df["dataset"])
    ]
    ax.legend(handles=model_handles + ds_handles, frameon=False, fontsize=9, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_fig2b_preference_boxplot(asym_df: pd.DataFrame, models: List[str], outpath: Path) -> None:
    apply_fig4_style()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=DPI)
    data = [asym_df.loc[asym_df["model"] == m, "preference_index"].dropna().values for m in models]
    positions = np.arange(1, len(models) + 1)
    bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True, showfliers=False)
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(model_color(m))
        patch.set_alpha(0.35)

    rng = np.random.default_rng(1)
    for i, (m, vals) in enumerate(zip(models, data)):
        if len(vals):
            ax.scatter(
                positions[i] + rng.normal(0, 0.07, len(vals)),
                vals,
                s=22,
                color=model_color(m),
                alpha=0.85,
                edgecolors="white",
                linewidths=0.3,
                zorder=3,
            )

    ax.axhline(0, color="#888888", ls="--", lw=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Preference index (up acc − down acc)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_fig2c_direction_imbalance(imb_df: pd.DataFrame, outpath: Path) -> None:
    apply_fig4_style()
    row = imb_df[imb_df["dataset"] == "mDC"]
    if row.empty:
        print("  [warn] skip fig2-C: no mDC imbalance data")
        return
    row = row.iloc[0]
    fig, ax = plt.subplots(figsize=(5, 1.8), dpi=DPI)
    fracs = [row["up_fraction"], 1 - row["up_fraction"]]
    labels = [f"Up ({int(row['n_up'])})", f"Down ({int(row['n_down'])})"]
    colors = ["#EB7E60", "#4EA3F1"]
    ax.barh([0], [fracs[0]], color=colors[0], height=0.5, label=labels[0])
    ax.barh([0], [fracs[1]], left=[fracs[0]], color=colors[1], height=0.5, label=labels[1])
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Fraction of top-30% dynamic genes")
    ax.set_title("mDC: up/down gene imbalance (scGPT reference)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plotting — Subsection 3
# ---------------------------------------------------------------------------
def plot_fig3a_jaccard_heatmap(jacc_matrix: pd.DataFrame, models: List[str], outpath: Path) -> None:
    apply_fig4_style()
    if jacc_matrix.empty:
        print("  [warn] skip fig3-A: empty Jaccard matrix")
        return

    mat = pd.DataFrame(np.nan, index=models, columns=models)
    for _, row in jacc_matrix.iterrows():
        a, b = row["model_a"], row["model_b"]
        v = row["jaccard_correct"]
        mat.loc[a, b] = v
        mat.loc[b, a] = v
    np.fill_diagonal(mat.values, 1.0)

    mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=DPI)
    sns.heatmap(
        mat.astype(float),
        ax=ax,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Jaccard (correct predictions)"},
    )
    ax.set_title("Cross-model prediction agreement")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_fig3b_go_bar(enr_df: Optional[pd.DataFrame], outpath: Path, top_n: int = 12) -> None:
    apply_fig4_style()
    if enr_df is None or enr_df.empty:
        print("  [warn] skip fig3-B: no enrichment results")
        return

    term_col = "Term" if "Term" in enr_df.columns else enr_df.columns[0]
    df = enr_df.copy()
    df["neglog10_padj"] = -np.log10(df["padj"].clip(lower=1e-300))
    df = df.nsmallest(top_n, "padj")

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(df))), dpi=DPI)
    y = np.arange(len(df))
    colors = ["#FF9A3D" if p < 0.05 else "#AC99D2" for p in df["padj"]]
    ax.barh(y, df["neglog10_padj"], color=colors, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels([str(t)[:70] for t in df[term_col]], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(r"$-\log_{10}$(adjusted p-value)")
    ax.set_title("Hard-case genes: GO/KEGG enrichment")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_fig3c_centrality_violin(
    gene_df: pd.DataFrame,
    datasets: List[str],
    kinases: Set[str],
    outpath: Path,
) -> None:
    apply_fig4_style()
    hard = gene_df[gene_df["category"] == "hard"].copy()
    easy = gene_df[gene_df["category"] == "easy"].copy()
    if hard.empty or easy.empty:
        print("  [warn] skip fig3-C: insufficient hard/easy genes")
        return

    records: List[dict] = []
    tf_cache: Dict[str, Set[str]] = {}
    for ds in datasets:
        if ds not in tf_cache:
            tf_cache[ds] = load_tf_set(ds)
        deg, btw = build_network_centrality(ds)
        if not deg:
            continue
        for label, sub in [("Hard cases", hard), ("Easy cases", easy)]:
            sub_ds = sub[sub["dataset"] == ds]
            for g in sub_ds["gene"]:
                gu = str(g).strip().upper()
                if gu in deg:
                    records.append({"group": label, "metric": "Degree centrality", "value": deg[gu]})
                    records.append({"group": label, "metric": "Betweenness centrality", "value": btw[gu]})

    if not records:
        print("  [warn] skip fig3-C: no network centrality matched")
        return

    cdf = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), dpi=DPI, sharey=False)
    palette = {"Hard cases": model_color("scPrint"), "Easy cases": model_color("scGPT")}

    for ax, metric in zip(axes, ["Degree centrality", "Betweenness centrality"]):
        sub = cdf[cdf["metric"] == metric]
        sns.violinplot(data=sub, x="group", y="value", hue="group", ax=ax, palette=palette, cut=0, inner=None, legend=False)
        sns.boxplot(
            data=sub,
            x="group",
            y="value",
            ax=ax,
            width=0.15,
            showcaps=True,
            boxprops={"facecolor": "white", "zorder": 3},
            whiskerprops={"linewidth": 1},
            medianprops={"color": "black", "linewidth": 1.2},
            showfliers=False,
        )
        h = sub.loc[sub["group"] == "Hard cases", "value"]
        e = sub.loc[sub["group"] == "Easy cases", "value"]
        if len(h) >= 3 and len(e) >= 3:
            _, p = stats.mannwhitneyu(h, e, alternative="two-sided")
            ax.set_title(f"{metric}\np = {p:.3g}")
        else:
            ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel(metric)

    fig.suptitle("STRING network centrality: hard vs easy genes", y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_fig3d_gene_type_stacked(
    gene_df: pd.DataFrame,
    datasets: List[str],
    kinases: Set[str],
    surface: Set[str],
    outpath: Path,
) -> None:
    apply_fig4_style()
    groups = [
        ("Hard cases", gene_df[gene_df["category"] == "hard"]),
        ("Easy cases", gene_df[gene_df["category"] == "easy"]),
        ("All top-30%", gene_df),
    ]
    type_order = ["TF", "Kinase", "Surface protein", "Other"]
    comp_rows: List[dict] = []

    tf_union: Set[str] = set()
    for ds in datasets:
        tf_union |= load_tf_set(ds)

    for gname, sub in groups:
        if sub.empty:
            continue
        types = [classify_gene_type(g, tf_union, kinases, surface) for g in sub["gene"]]
        vc = pd.Series(types).value_counts()
        n = len(types)
        for t in type_order:
            comp_rows.append({"group": gname, "gene_type": t, "fraction": vc.get(t, 0) / n})

    comp = pd.DataFrame(comp_rows)
    if comp.empty:
        print("  [warn] skip fig3-D: empty composition table")
        return

    fig, ax = plt.subplots(figsize=(6, 4), dpi=DPI)
    groups_order = [g for g, _ in groups if not comp[comp["group"] == g].empty]
    y = np.arange(len(groups_order))
    left = np.zeros(len(groups_order))

    for t in type_order:
        heights = []
        for g in groups_order:
            row = comp[(comp["group"] == g) & (comp["gene_type"] == t)]
            heights.append(float(row["fraction"].iloc[0]) if len(row) else 0.0)
        heights = np.array(heights)
        ax.barh(y, heights, left=left, height=0.55, color=GENE_TYPE_COLORS[t], label=t, edgecolor="white", linewidth=0.5)
        left += heights

    ax.set_yticks(y)
    ax.set_yticklabels(groups_order)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction")
    ax.set_title("Gene type composition")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fig4 DEG dynamic prediction robustness analysis")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--datasets", default=",".join(ALL_DATASETS))
    p.add_argument("--exclude-datasets", default="mDC", help="Excluded from main panels (default: mDC)")
    p.add_argument("--models", default=",".join(MODEL_ORDER))
    p.add_argument("--top-percent", type=float, default=TOP_PERCENT)
    p.add_argument("--conservative-ratio", type=float, default=CONSERVATIVE_RATIO)
    p.add_argument("--hard-gene-frac", type=float, default=HARD_GENE_FRAC)
    p.add_argument("--no-enrichment", action="store_true")
    p.add_argument("--skip-plots", action="store_true", help="Only write CSV tables")
    return p.parse_args()


def main() -> None:
    global CONSERVATIVE_RATIO
    args = parse_args()
    CONSERVATIVE_RATIO = args.conservative_ratio

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(exist_ok=True)
    (outdir / "tables").mkdir(exist_ok=True)

    all_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    excluded = {d.strip() for d in args.exclude_datasets.split(",") if d.strip()}
    main_datasets = [d for d in all_datasets if d not in excluded]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in MODEL_PATHS:
            raise ValueError(f"Unknown model: {m}")

    ens_to_sym, kinases, surface = load_hgnc_maps()
    print(f"HGNC maps: {len(ens_to_sym)} Ensembl IDs, {len(kinases)} kinases, {len(surface)} surface-related")

    # --- Subsection 1 ---
    print("\n[1] Magnitude stratification …")
    acc_df, cons_df = compute_magnitude_metrics(main_datasets, models, ens_to_sym, args.top_percent)
    acc_df.to_csv(outdir / "tables" / "magnitude_direction_accuracy.csv", index=False)
    cons_df.to_csv(outdir / "tables" / "low_magnitude_conservative_fraction.csv", index=False)

    # --- Subsection 2 ---
    print("[2] Up/down asymmetry & tradeoff trajectories …")
    asym_df, imb_df = compute_asymmetry_metrics(all_datasets, models, ens_to_sym, args.top_percent)
    asym_df.to_csv(outdir / "tables" / "updown_asymmetry.csv", index=False)
    imb_df.to_csv(outdir / "tables" / "direction_imbalance.csv", index=False)

    # Pareto curves: default exclude mDC (top30% often has no up genes → NaN up accuracy)
    tradeoff_datasets = main_datasets if main_datasets else all_datasets
    traj_df, endpoints_df = compute_tradeoff_trajectories(tradeoff_datasets, models, ens_to_sym, args.top_percent)
    traj_df.to_csv(outdir / "tables" / "updown_tradeoff_trajectories.csv", index=False)
    endpoints_df.to_csv(outdir / "tables" / "updown_tradeoff_endpoints.csv", index=False)
    pareto_df = average_model_pareto_curves(traj_df, models)
    pareto_df.to_csv(outdir / "tables" / "updown_pareto_mean_curves.csv", index=False)
    curvature_df = fit_tradeoff_curvature(pareto_df, models)
    curvature_df.to_csv(outdir / "tables" / "updown_tradeoff_curvature.csv", index=False)

    # --- Subsection 3 ---
    print("[3] Cross-model consistency …")
    jacc_matrix, gene_df, jacc_raw = compute_consistency_tables(
        main_datasets, models, ens_to_sym, args.top_percent, args.hard_gene_frac
    )
    jacc_matrix.to_csv(outdir / "tables" / "jaccard_correct_pooled.csv", index=False)
    jacc_raw.to_csv(outdir / "tables" / "jaccard_by_dataset.csv", index=False)
    gene_df.to_csv(outdir / "tables" / "gene_consensus_hard_easy.csv", index=False)

    hard_genes = gene_df[gene_df["category"] == "hard"]["gene"].unique().tolist()
    enr_df = None
    if not args.no_enrichment and hard_genes:
        species = "human" if all(DATASET_SPECIES.get(d) == "human" for d in main_datasets) else "human"
        print(f"  GO enrichment on {len(hard_genes)} hard genes …")
        enr_df = run_go_enrichment(hard_genes, species, outdir / "tables", "hard_genes")

    if args.skip_plots:
        print(f"\nTables saved under {outdir / 'tables'}")
        return

    figdir = outdir / "figures"
    print("\n[plots] Generating figures …")
    plot_fig1a_magnitude_accuracy(acc_df, models, figdir / "fig1A_magnitude_accuracy_bar.pdf")
    plot_fig1b_conservative_heatmap(cons_df, models, main_datasets, figdir / "fig1B_conservative_heatmap.pdf")
    plot_fig1c_conservative_boxplot(cons_df, figdir / "fig1C_conservative_boxplot.pdf")
    plot_fig2a1_pareto_frontier(pareto_df, endpoints_df, models, figdir / "fig2A1_pareto_frontier.pdf")
    plot_fig2a2_drift_arrows(endpoints_df, models, figdir / "fig2A2_drift_arrows.pdf")
    plot_fig2a3_curvature_bar(curvature_df, models, figdir / "fig2A3_curvature_bar.pdf")
    plot_fig2a_combined_panel(pareto_df, endpoints_df, curvature_df, models, figdir / "fig2A_tradeoff_combined.pdf")
    plot_fig2a_updown_scatter(asym_df, figdir / "fig2A_final_scatter_legacy.pdf")
    plot_fig2b_preference_boxplot(asym_df, models, figdir / "fig2B_preference_index_boxplot.pdf")
    if "mDC" in all_datasets:
        plot_fig2c_direction_imbalance(imb_df, figdir / "fig2C_mDC_direction_imbalance.pdf")
    plot_fig3a_jaccard_heatmap(jacc_matrix, models, figdir / "fig3A_jaccard_heatmap.pdf")
    plot_fig3b_go_bar(enr_df, figdir / "fig3B_hard_genes_go_bar.pdf")
    plot_fig3c_centrality_violin(gene_df, main_datasets, kinases, figdir / "fig3C_centrality_violin.pdf")
    plot_fig3d_gene_type_stacked(gene_df, main_datasets, kinases, surface, figdir / "fig3D_gene_type_stacked.pdf")

    print(f"\n✅ Done. Outputs:")
    print(f"   Tables:  {outdir / 'tables'}")
    print(f"   Figures: {figdir}")
    print(f"   Hard genes: {len(hard_genes)} | Easy: {(gene_df['category']=='easy').sum()}")


if __name__ == "__main__":
    main()
