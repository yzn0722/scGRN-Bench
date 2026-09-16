#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig4 子节二：迭代轨迹动力学特性分析

维度1 收敛性 (fig2-B1, B2)
维度2 路径效率 (fig2-B3, B4)
维度3 插值合理性 (fig2-B5, B6)
维度4 连续值 vs 排序类 (fig2-B7)

用法:
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig4
  python3 plot_iter_trajectory_dynamics.py
  python3 plot_iter_trajectory_dynamics.py --example-dataset mHSC-L --example-model scFoundation
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

try:
    from fig4_palette import apply_fig4_style, model_color
except ImportError:

    def apply_fig4_style() -> None:
        plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})

    def model_color(name: str, default: str = "#808080") -> str:
        return default


from gene_result_io import load_gene_result

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_ROOT = Path("/mnt/10T/yzn/benchmark_GRN")
CHIP_DIR = BENCH_ROOT / "input_process" / "CHIP"
PT_ROOT = BENCH_ROOT / "PseudoTime"
DEFAULT_OUTDIR = SCRIPT_DIR / "robustness" / "trajectory_dynamics"

MODEL_ORDER = ["scFoundation", "LangCell", "scPrint", "scGPT", "Geneformer", "scCello"]
CONTINUOUS_GROUP = ["scFoundation", "scGPT", "scPrint"]
RANK_GROUP = ["Geneformer", "LangCell", "scCello"]
ALL_DATASETS = ["hESC", "hHep", "mDC", "mHSC-E", "mHSC-GM", "mHSC-L"]
MAIN_DATASETS = ["hESC", "hHep", "mHSC-E", "mHSC-GM", "mHSC-L"]

STABLE_THRESH = 0.99
STABLE_STREAK = 3
PT_QUANTILE = 0.2
DPI = 600
EPS = 1e-8

MODEL_TRAJ_CONFIG: Dict[str, dict] = {
    "scGPT": {
        "kind": "preds_full",
        "path": BENCH_ROOT / "dyn4_results_unified/scgpt/per_dataset/{ds}/preds_full_by_iter.npy",
    },
    "Geneformer": {
        "kind": "rank_map",
        "path": BENCH_ROOT / "pre_geneformer_results_unified/geneformer/per_dataset/{ds}/mean_rank_delta_by_iter.npy",
    },
    "LangCell": {
        "kind": "rank_map",
        "path": BENCH_ROOT / "pre_langcell_results_unified/langcell/per_dataset/{ds}/mean_rank_delta_by_iter.npy",
    },
    "scCello": {
        "kind": "rank_map",
        "path": BENCH_ROOT / "pre_sccello_results_unified/sccello/per_dataset/{ds}/mean_rank_delta_by_iter.npy",
    },
    "scPrint": {
        "kind": "pred_delta_cumsum",
        "path": BENCH_ROOT / "pre_scprint_results_unified/scprint/per_dataset/{ds}/pred_delta_by_iter.npy",
    },
    "scFoundation": {
        "kind": "synthetic_acc",
        "acc_path": BENCH_ROOT / "pre_scfoundation/scfoundation_multidataset_pseudotime_227/{ds}/acc_curve.npy",
    },
}


def load_early_late_vectors(dataset: str, n_genes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Use scGPT gene_result as reference gene order (910 CHIP genes)."""
    gr = load_gene_result(
        BENCH_ROOT / f"pre_scgpt/results_multidataset_pseudotime_227/{dataset}_gene_result.csv"
    )
    early = gr["true_early_mean"].values.astype(float)
    late = gr["true_late_mean"].values.astype(float)
    n = min(n_genes, len(early))
    return early[:n], late[:n]


def build_rank_mapped_trajectory(rank_seq: np.ndarray, early: np.ndarray) -> np.ndarray:
    """
    Map cumulative rank displacements to expression space along the model's own final Δ.
    rank_seq[t] is cumulative (not incremental); pred_delta at t ≈ -rank_seq[t].
    """
    n = min(len(early), rank_seq.shape[1])
    early = early[:n]
    final_disp = -rank_seq[-1, :n]
    target = final_disp
    denom = float(np.dot(final_disp, final_disp)) + EPS
    X = [early.copy()]
    for t in range(rank_seq.shape[0]):
        r = -rank_seq[t, :n]
        alpha = float(np.dot(r, final_disp) / denom)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        X.append(early + alpha * target)
    return np.vstack(X)


def build_state_trajectory(model: str, dataset: str) -> Optional[np.ndarray]:
    """
    Build X_seq shape (T+1, G): state at iteration 0 = early mean, then iterative updates.
    """
    cfg = MODEL_TRAJ_CONFIG[model]
    kind = cfg["kind"]

    if kind == "preds_full":
        path = Path(str(cfg["path"]).format(ds=dataset))
        if not path.is_file():
            return None
        preds = np.load(path).astype(float)  # (T, n_early, G)
        states = preds.mean(axis=1)
        early, _ = load_early_late_vectors(dataset, states.shape[1])
        X = np.vstack([early.reshape(1, -1), states])
        return X

    if kind == "pred_delta_cumsum":
        path = Path(str(cfg["path"]).format(ds=dataset))
        if not path.is_file():
            return None
        deltas = np.load(path).astype(float)
        early, _ = load_early_late_vectors(dataset, deltas.shape[1])
        X = [early.copy()]
        cur = early.copy()
        for t in range(deltas.shape[0]):
            cur = cur + deltas[t]
            X.append(cur.copy())
        return np.vstack(X)

    if kind == "rank_map":
        path = Path(str(cfg["path"]).format(ds=dataset))
        if not path.is_file():
            return None
        rank_seq = np.load(path).astype(float)
        early, _ = load_early_late_vectors(dataset, rank_seq.shape[1])
        return build_rank_mapped_trajectory(rank_seq, early)

    if kind == "synthetic_acc":
        acc_path = Path(str(cfg["acc_path"]).format(ds=dataset))
        scf = load_gene_result(
            BENCH_ROOT / f"pre_scfoundation/scfoundation_multidataset_pseudotime_227/{dataset}/gene_delta_compare.csv",
            mapped_only=True,
        )
        if not acc_path.is_file() or scf is None:
            return None
        acc = np.load(acc_path).astype(float)
        early, late = load_early_late_vectors(dataset, 10_000)
        pred_map = dict(zip(scf["gene"].astype(str), scf["pred_late_like_mean"].values.astype(float)))
        pred_late = np.array([pred_map.get(g, np.nan) for g in load_gene_result(
            BENCH_ROOT / f"pre_scgpt/results_multidataset_pseudotime_227/{dataset}_gene_result.csv"
        )["gene"].astype(str)], dtype=float)
        pred_late = np.nan_to_num(pred_late, nan=early)
        delta_final = pred_late - early
        X = [early.copy()]
        for a in acc:
            tau = float(a / acc[-1]) if acc[-1] > 0 else 0.0
            X.append(early + tau * delta_final)
        return np.vstack(X)

    return None


def compute_convergence_metrics(X_seq: np.ndarray) -> dict:
    T = len(X_seq) - 1
    step_changes = [float(np.linalg.norm(X_seq[t + 1] - X_seq[t])) for t in range(T)]
    conv = [float(cosine_similarity(X_seq[t : t + 1], X_seq[-1:])[0, 0]) for t in range(T + 1)]

    stable_steps = T
    for t in range(max(0, T - STABLE_STREAK + 1)):
        if all(conv[t + k] >= STABLE_THRESH for k in range(STABLE_STREAK)):
            stable_steps = t
            break

    step_sims = []
    for t in range(T):
        step_sims.append(float(cosine_similarity(X_seq[t : t + 1], X_seq[t + 1 : t + 2])[0, 0]))

    return {
        "step_changes": step_changes,
        "step_similarities": step_sims,
        "convergence_curve": conv,
        "stable_steps": stable_steps,
        "total_variation": float(np.sum(step_changes)),
        "final_similarity": conv[-1],
    }


def direction_consistency_curve(X_seq: np.ndarray, X_early: np.ndarray, X_late: np.ndarray) -> List[float]:
    final_dir = X_late - X_early
    fn = np.linalg.norm(final_dir) + EPS
    out = []
    for t in range(len(X_seq) - 1):
        step = X_seq[t + 1] - X_seq[t]
        out.append(float(np.dot(step, final_dir) / (np.linalg.norm(step) + EPS) / fn))
    return out


def compute_path_efficiency(X_seq: np.ndarray, X_early: np.ndarray, X_late: np.ndarray) -> dict:
    actual = float(sum(np.linalg.norm(X_seq[t + 1] - X_seq[t]) for t in range(len(X_seq) - 1)))
    direct = float(np.linalg.norm(X_late - X_early))
    efficiency = min(1.0, direct / actual) if actual > EPS else 1.0
    final_dir = X_late - X_early
    fn = np.linalg.norm(final_dir) + EPS
    dirs = []
    for t in range(len(X_seq) - 1):
        step = X_seq[t + 1] - X_seq[t]
        dirs.append(float(np.dot(step, final_dir) / (np.linalg.norm(step) + EPS) / fn))
    return {
        "efficiency": efficiency,
        "redundancy": actual - direct,
        "actual_length": actual,
        "direct_distance": direct,
        "direction_consistency": float(np.mean(dirs)) if dirs else np.nan,
    }


def characterize_geometry(X_seq: np.ndarray) -> dict:
    step_changes = [float(np.linalg.norm(X_seq[t + 1] - X_seq[t])) for t in range(len(X_seq) - 1)]
    smoothness = np.nan
    if len(step_changes) > 2:
        smoothness = float(np.corrcoef(step_changes[:-1], step_changes[1:])[0, 1])

    reversals = 0
    dirs = []
    for t in range(len(X_seq) - 1):
        v = X_seq[t + 1] - X_seq[t]
        n = np.linalg.norm(v)
        dirs.append(v / n if n > EPS else v)
    for t in range(len(dirs) - 1):
        if float(np.dot(dirs[t], dirs[t + 1])) < -0.5:
            reversals += 1

    accs = []
    for t in range(len(X_seq) - 2):
        v1 = X_seq[t + 1] - X_seq[t]
        v2 = X_seq[t + 2] - X_seq[t + 1]
        accs.append(v2 - v1)
    mean_curvature = float(np.mean([np.linalg.norm(a) for a in accs])) if accs else 0.0

    return {
        "smoothness": smoothness,
        "reversal_count": reversals,
        "mean_curvature": mean_curvature,
        "max_jump": float(max(step_changes)) if step_changes else 0.0,
    }


def load_reference_genes(dataset: str) -> List[str]:
    gr = load_gene_result(
        BENCH_ROOT / f"pre_scgpt/results_multidataset_pseudotime_227/{dataset}_gene_result.csv"
    )
    return gr["gene"].astype(str).tolist()


def load_intermediate_cells(dataset: str, genes: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    expr_path = CHIP_DIR / f"{dataset}_chip_matched-ExpressionData.csv"
    pt_path = PT_ROOT / dataset / "PseudoTime.csv"
    expr = pd.read_csv(expr_path, index_col=0)
    if genes is not None:
        expr = expr.reindex(genes).fillna(0.0)
    pt = pd.read_csv(pt_path, index_col=0).iloc[:, 0].astype(float)
    common = expr.columns.intersection(pt.index)
    expr = expr[common]
    pt = pt.loc[common]
    lo, hi = np.quantile(pt.values, [PT_QUANTILE, 1 - PT_QUANTILE])
    mid = (pt.values > lo) & (pt.values < hi)
    cells = expr.columns[mid]
    X = expr[cells].T.values.astype(float)
    return X, pt.loc[cells].values.astype(float)


def evaluate_interpolation(X_seq: np.ndarray, dataset: str) -> dict:
    genes = load_reference_genes(dataset)[: X_seq.shape[1]]
    cells, pt = load_intermediate_cells(dataset, genes=genes)
    if len(cells) < 10:
        return {"alignment_corr": np.nan, "interpolation_error": np.nan, "mean_match_distance": np.nan}

    n_steps = len(X_seq)
    expected_pt = np.linspace(0, 1, n_steps)
    matched_pt = []
    min_dists = []

    for x in X_seq:
        d = cdist([x], cells, metric="cosine")[0]
        j = int(np.argmin(d))
        min_dists.append(float(d[j]))
        matched_pt.append(float(pt[j]))

    matched_pt = np.array(matched_pt)
    if np.std(expected_pt) > 0 and np.std(matched_pt) > 0:
        alignment_corr = float(np.corrcoef(expected_pt, matched_pt)[0, 1])
    else:
        alignment_corr = np.nan

    # Interpolation error at true cell pseudotimes
    from scipy.interpolate import interp1d

    pred_at_cells = np.zeros_like(cells)
    for g in range(cells.shape[1]):
        f = interp1d(expected_pt, X_seq[:, g], kind="linear", fill_value="extrapolate", assume_sorted=True)
        pred_at_cells[:, g] = f(pt)

    diag = cdist(pred_at_cells, cells, metric="cosine").diagonal()
    return {
        "alignment_corr": alignment_corr,
        "interpolation_error": float(np.mean(diag)),
        "mean_match_distance": float(np.mean(min_dists)),
    }


def evaluate_interpolation_by_stage(X_seq: np.ndarray, dataset: str) -> dict:
    """Early / mid / late third of intermediate cells."""
    genes = load_reference_genes(dataset)[: X_seq.shape[1]]
    cells, pt = load_intermediate_cells(dataset, genes=genes)
    out = {}
    for label, lo, hi in [("early", 0.0, 0.33), ("mid", 0.33, 0.66), ("late", 0.66, 1.01)]:
        mask = (pt >= lo) & (pt < hi)
        if mask.sum() < 5:
            out[f"interp_error_{label}"] = np.nan
            continue
        sub_cells = cells[mask]
        sub_pt = pt[mask]
        from scipy.interpolate import interp1d

        expected_pt = np.linspace(0, 1, len(X_seq))
        pred = np.zeros_like(sub_cells)
        for g in range(sub_cells.shape[1]):
            f = interp1d(expected_pt, X_seq[:, g], kind="linear", fill_value="extrapolate", assume_sorted=True)
            pred[:, g] = f(sub_pt)
        out[f"interp_error_{label}"] = float(np.mean(cdist(pred, sub_cells, metric="cosine").diagonal()))
    return out


def analyze_all(datasets: List[str], models: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    dir_rows = []
    for ds in datasets:
        early, late = load_early_late_vectors(ds, 10_000)
        for model in models:
            X = build_state_trajectory(model, ds)
            if X is None or X.shape[0] < 3:
                continue
            n = min(X.shape[1], len(early))
            X, e, l = X[:, :n], early[:n], late[:n]
            conv = compute_convergence_metrics(X)
            path = compute_path_efficiency(X, e, l)
            geom = characterize_geometry(X)
            interp = evaluate_interpolation(X, ds)
            interp_stg = evaluate_interpolation_by_stage(X, ds)

            dir_curve = direction_consistency_curve(X, e, l)
            for t, sim in enumerate(conv["convergence_curve"]):
                dir_rows.append(
                    {
                        "dataset": ds,
                        "model": model,
                        "iter": t,
                        "conv_to_final": sim,
                        "direction_consistency": dir_curve[t - 1] if 0 < t <= len(dir_curve) else np.nan,
                    }
                )

            traj_kind = MODEL_TRAJ_CONFIG.get(model, {}).get("kind", "")
            rows.append(
                {
                    "dataset": ds,
                    "model": model,
                    "traj_kind": traj_kind,
                    "n_iters": len(X) - 1,
                    "stable_steps": conv["stable_steps"],
                    "final_similarity": conv["final_similarity"],
                    "total_variation": conv["total_variation"],
                    "path_efficiency": path["efficiency"],
                    "direction_consistency": path["direction_consistency"],
                    "smoothness": geom["smoothness"],
                    "reversal_count": geom["reversal_count"],
                    "mean_curvature": geom["mean_curvature"],
                    "max_jump": geom["max_jump"],
                    "alignment_corr": interp["alignment_corr"],
                    "interpolation_error": interp["interpolation_error"],
                    **interp_stg,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(dir_rows), pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_b1_convergence_curves(dir_df: pd.DataFrame, models: List[str], outpath: Path) -> None:
    apply_fig4_style()
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=DPI)
    for model in models:
        sub = dir_df[dir_df["model"] == model]
        if sub.empty:
            continue
        mean_curve = sub.groupby("iter")["conv_to_final"].mean()
        ax.plot(mean_curve.index, mean_curve.values, "-o", ms=4, lw=2.2, label=model, color=model_color(model))
    ax.axhline(STABLE_THRESH, color="#888888", ls="--", lw=1.0, alpha=0.7)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cosine similarity to final state")
    ax.set_ylim(0.75, 1.02)
    ax.legend(frameon=False, ncol=2, fontsize=10)
    ax.set_title("Convergence to final predicted state")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_b2_stable_steps(metrics_df: pd.DataFrame, models: List[str], datasets: List[str], outpath: Path) -> None:
    apply_fig4_style()
    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)
    sub = metrics_df[metrics_df["dataset"].isin(datasets)]
    data = []
    positions = []
    labels = []
    width = 0.12
    for mi, m in enumerate(models):
        vals = [sub[(sub["model"] == m) & (sub["dataset"] == d)]["stable_steps"].values for d in datasets]
        vals = [v[0] if len(v) else np.nan for v in vals]
        pos = np.arange(len(datasets)) + (mi - (len(models) - 1) / 2) * width
        ax.bar(pos, vals, width=width * 0.9, color=model_color(m), label=m, edgecolor="none")
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Iterations to reach stability")
    ax.set_title(f"Convergence speed (sim ≥ {STABLE_THRESH} for {STABLE_STREAK} steps)")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_b3_path_efficiency(metrics_df: pd.DataFrame, models: List[str], outpath: Path) -> None:
    apply_fig4_style()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=DPI)
    order = [m for m in models if m in set(metrics_df["model"])]
    sns.boxplot(data=metrics_df, x="model", y="path_efficiency", order=order, hue="model", palette={m: model_color(m) for m in order}, ax=ax, legend=False, width=0.55)
    sns.stripplot(data=metrics_df, x="model", y="path_efficiency", order=order, color="#333333", alpha=0.45, size=4, jitter=0.2, ax=ax)
    ax.set_ylabel("Path efficiency (direct / path length)")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    ax.set_title("Trajectory path efficiency")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_b4_direction_consistency(dir_df: pd.DataFrame, models: List[str], outpath: Path) -> None:
    apply_fig4_style()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=DPI)
    for model in models:
        sub = dir_df[(dir_df["model"] == model) & (dir_df["iter"] > 0)]
        if sub.empty or "direction_consistency" not in sub.columns:
            continue
        mean_c = sub.groupby("iter")["direction_consistency"].mean()
        ax.plot(mean_c.index, mean_c.values, "-o", ms=4, lw=2, label=model, color=model_color(model))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Step direction vs early→late axis")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Direction consistency along iteration")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_b5_alignment_scatter(metrics_df: pd.DataFrame, dataset: str, model: str, outpath: Path) -> None:
    apply_fig4_style()
    X = build_state_trajectory(model, dataset)
    if X is None:
        return
    genes = load_reference_genes(dataset)[: X.shape[1]]
    cells, pt = load_intermediate_cells(dataset, genes=genes)
    n_steps = len(X)
    expected = np.linspace(0, 1, n_steps)
    matched = []
    for x in X:
        j = int(np.argmin(cdist([x], cells, metric="cosine")[0]))
        matched.append(pt[j])
    matched = np.array(matched)
    r = np.corrcoef(expected, matched)[0, 1] if np.std(matched) > 0 else np.nan

    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=DPI)
    ax.scatter(expected, matched, s=55, color=model_color(model), edgecolors="white", linewidths=0.5)
    lims = [0, 1]
    ax.plot(lims, lims, "--", color="#888888", lw=1.0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Normalized iteration position (t / T)")
    ax.set_ylabel("Pseudotime of best-matching real cell")
    ax.set_title(f"{model} @ {dataset}\nalignment r = {r:.3f}")
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_b6_interp_stages(metrics_df: pd.DataFrame, models: List[str], outpath: Path) -> None:
    apply_fig4_style()
    stages = ["early", "mid", "late"]
    cols = [f"interp_error_{s}" for s in stages]
    plot_df = metrics_df[["model"] + cols].copy()
    m = plot_df.groupby("model")[cols].mean().reindex([m for m in models if m in plot_df["model"].values])
    if m.empty:
        return

    x = np.arange(len(m))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)
    for i, (stage, col) in enumerate(zip(stages, cols)):
        ax.bar(x + (i - 1) * width, m[col].values, width=width, label=stage.capitalize(), edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(m.index, rotation=15, ha="right")
    ax.set_ylabel("Interpolation error (cosine distance)")
    ax.set_title("Interpolation error by pseudotime stage")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_b7_radar(metrics_df: pd.DataFrame, outpath: Path) -> None:
    apply_fig4_style()
    df = metrics_df.copy()
    df["model_group"] = df["model"].apply(lambda m: "Continuous" if m in CONTINUOUS_GROUP else "Rank-based")

    def norm_series(s: pd.Series, higher_better: bool) -> pd.Series:
        lo, hi = s.min(), s.max()
        if hi - lo < EPS:
            return pd.Series(0.5, index=s.index)
        z = (s - lo) / (hi - lo)
        return z if higher_better else 1 - z

    dims = {
        "smoothness": ("smoothness", True),
        "low_oscillation": ("reversal_count", False),
        "low_curvature": ("mean_curvature", False),
        "fast_convergence": ("stable_steps", False),
        "path_efficiency": ("path_efficiency", True),
        "interpolation": ("interpolation_error", False),
    }

    labels = list(dims.keys())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True), dpi=DPI)
    for grp, color in [("Continuous", model_color("scGPT")), ("Rank-based", model_color("Geneformer"))]:
        sub = df[df["model_group"] == grp]
        if sub.empty:
            continue
        vals = []
        for _, (col, hb) in dims.items():
            vals.append(float(norm_series(sub[col], hb).mean()))
        vals += vals[:1]
        ax.plot(angles, vals, "-", lw=2.2, label=grp, color=color)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.25, 1.1))
    ax.set_title("Trajectory geometry: continuous vs rank-based models", y=1.08)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_combined_panel(metrics_df: pd.DataFrame, dir_df: pd.DataFrame, models: List[str], outpath: Path) -> None:
    apply_fig4_style()
    fig = plt.figure(figsize=(16, 10), dpi=DPI)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    for model in models:
        sub = dir_df[dir_df["model"] == model].groupby("iter")["conv_to_final"].mean()
        ax.plot(sub.index, sub.values, "-", lw=2, label=model, color=model_color(model))
    ax.axhline(STABLE_THRESH, ls="--", color="#888")
    ax.set_title("B1 Convergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Similarity to final")

    ax = fig.add_subplot(gs[0, 1])
    sns.boxplot(data=metrics_df, x="model", y="path_efficiency", order=models, hue="model", palette={m: model_color(m) for m in models}, ax=ax, legend=False, width=0.55)
    ax.set_title("B3 Path efficiency")
    ax.set_xlabel("")

    ax = fig.add_subplot(gs[1, 0])
    m = metrics_df.groupby("model")[["interp_error_early", "interp_error_mid", "interp_error_late"]].mean().reindex(models)
    x = np.arange(len(models))
    w = 0.25
    for i, c in enumerate(["interp_error_early", "interp_error_mid", "interp_error_late"]):
        ax.bar(x + (i - 1) * w, m[c].values, width=w, label=c.split("_")[-1])
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_title("B6 Interpolation error by stage")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    summ = metrics_df.groupby("model")[["stable_steps", "path_efficiency", "alignment_corr"]].mean()
    ax.table(cellText=np.round(summ.values, 3), rowLabels=summ.index, colLabels=summ.columns, loc="center")
    ax.set_title("Summary metrics", pad=20)

    fig.suptitle("Iterative trajectory dynamics", fontsize=15, y=1.01)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Iterative trajectory dynamics (Fig4 subsection B)")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--datasets", default=",".join(MAIN_DATASETS))
    p.add_argument("--models", default=",".join(MODEL_ORDER))
    p.add_argument("--example-dataset", default=None, help="B5 case study; default = highest alignment_corr")
    p.add_argument("--example-model", default=None)
    p.add_argument("--skip-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    print("[1] Computing trajectory metrics …")
    metrics_df, dir_df, _ = analyze_all(datasets, models)
    metrics_df.to_csv(outdir / "tables" / "trajectory_dynamics_metrics.csv", index=False)
    dir_df.to_csv(outdir / "tables" / "convergence_curves.csv", index=False)

    print(metrics_df.groupby("model")[["stable_steps", "path_efficiency", "alignment_corr", "reversal_count"]].mean().round(3))

    if args.skip_plots:
        return

    ex_ds, ex_model = args.example_dataset, args.example_model
    if ex_ds is None or ex_model is None:
        finite = metrics_df[np.isfinite(metrics_df["alignment_corr"])]
        if not finite.empty:
            best = finite.loc[finite["alignment_corr"].idxmax()]
            ex_ds = ex_ds or str(best["dataset"])
            ex_model = ex_model or str(best["model"])
        else:
            ex_ds, ex_model = "mHSC-L", "scFoundation"

    print("[2] Plotting …")
    figdir = outdir / "figures"
    plot_b1_convergence_curves(dir_df, models, figdir / "fig2B1_convergence_curves.pdf")
    plot_b2_stable_steps(metrics_df, models, datasets, figdir / "fig2B2_stable_steps.pdf")
    plot_b3_path_efficiency(metrics_df, models, figdir / "fig2B3_path_efficiency.pdf")
    plot_b4_direction_consistency(dir_df, models, figdir / "fig2B4_direction_consistency.pdf")
    plot_b5_alignment_scatter(metrics_df, ex_ds, ex_model, figdir / f"fig2B5_pt_alignment_{ex_model}_{ex_ds}.pdf")
    plot_b6_interp_stages(metrics_df, models, figdir / "fig2B6_interpolation_stages.pdf")
    plot_b7_radar(metrics_df, figdir / "fig2B7_radar_continuous_vs_rank.pdf")
    plot_combined_panel(metrics_df, dir_df, models, figdir / "fig2B_trajectory_dynamics_combined.pdf")

    print(f"\n✅ Done. Output: {outdir}")


if __name__ == "__main__":
    main()
