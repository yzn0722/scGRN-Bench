#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

FIG4 = Path("/mnt/10T/yzn/scGRN-Bench/FBplot/fig4")
WORK = FIG4 / "balanced_convergence_work"
OUT = FIG4 / "convergence"
sys.path.insert(0, str(FIG4))

from fig4_palette import apply_fig4_style, model_color  # noqa: E402

DATASETS = ["hESC", "hHep", "mDC", "mHSC-E", "mHSC-GM", "mHSC-L"]
MAX_ITER = 11

SAVED_MODELS = {
    "scGPT": (
        Path("/mnt/10T/yzn/benchmark_GRN/dyn4_results_unified/scgpt"),
        "pred_delta_by_iter.npy",
        1.0,
        FIG4 / "interation/scgpt_accuracy_curves.json",
    ),
    "scPRINT": (
        Path("/mnt/10T/yzn/benchmark_GRN/pre_scprint_results_unified/scprint"),
        "pred_delta_by_iter.npy",
        1.0,
        FIG4 / "interation/scprint_accuracy_curves.json",
    ),
    "scCello": (
        Path("/mnt/10T/yzn/benchmark_GRN/pre_sccello_results_unified/sccello"),
        "mean_rank_delta_by_iter.npy",
        -1.0,
        FIG4 / "interation/sccello_accuracy_curves.json",
    ),
}

RERUN_JSONS = {
    "Geneformer": WORK / "geneformer_balanced_accuracy_curves.json",
    "LangCell": WORK / "langcell_balanced_accuracy_curves.json",
    "scFoundation": WORK / "scfoundation_balanced_accuracy_curves.json",
}


def scores(pred_delta: np.ndarray, true_delta: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
    truth = np.where(true_delta[idx] > 0, 1, -1)
    pred = np.where(pred_delta[idx] > 0, 1, -1)
    acc = float(np.mean(pred == truth))
    pos = truth == 1
    neg = truth == -1
    # Match sklearn and the existing balance.py implementation. If only one
    # truth class is present (mDC top-30%), this reduces to that class's recall.
    ba = float(balanced_accuracy_score(truth, pred))
    return acc, ba


def load_saved_model(name: str, root: Path, array_name: str, sign: float, ref_json: Path):
    reference = json.loads(ref_json.read_text(encoding="utf-8"))
    curves: dict[str, list[float]] = {}
    for ds in DATASETS:
        ds_dir = root / "per_dataset" / ds
        frame = pd.read_csv(ds_dir / "per_gene_final_changes.csv")
        true_delta = pd.to_numeric(frame["true_delta"], errors="raise").to_numpy(float)
        idx = np.flatnonzero(frame["in_top_eval"].astype(bool).to_numpy())
        pred_by_iter = np.load(ds_dir / array_name).astype(float) * sign
        acc_curve, ba_curve = [], []
        for row in pred_by_iter:
            acc, ba = scores(row, true_delta, idx)
            acc_curve.append(acc)
            ba_curve.append(ba)
        ref = np.asarray(reference[ds], dtype=float)
        got = np.asarray(acc_curve[: len(ref)], dtype=float)
        max_diff = float(np.max(np.abs(got - ref)))
        if max_diff > 1e-7:
            raise RuntimeError(f"{name}/{ds}: saved arrays do not reproduce source accuracy (max diff={max_diff})")
        print(f"validated {name:8s} {ds:7s}: max accuracy diff={max_diff:.3g}")
        curves[ds] = ba_curve
    return curves


def collect_curves() -> dict[str, dict[str, list[float]]]:
    result = {}
    for name, args in SAVED_MODELS.items():
        result[name] = load_saved_model(name, *args)
    for name, path in RERUN_JSONS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Waiting for rerun output: {path}")
        result[name] = json.loads(path.read_text(encoding="utf-8"))
    return result


def plot(curves: dict[str, dict[str, list[float]]]):
    import matplotlib.pyplot as plt

    apply_fig4_style()
    order = ["Geneformer", "LangCell", "scGPT", "scFoundation", "scPRINT", "scCello"]
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        fig, ax = plt.subplots(figsize=(5.0, 5.0))
        for name in order:
            y = np.asarray(curves[name][ds], dtype=float)[:MAX_ITER] * 100.0
            x = np.arange(1, len(y) + 1)
            ax.plot(x, y, marker="o", linestyle="-", linewidth=2.4, markersize=8,
                    alpha=0.85, color=model_color(name), label=name)
        ax.set_xlabel("Iteration", fontsize=16)
        ax.set_ylabel("Balanced Accuracy(%)", fontsize=16)
        ax.set_xlim(1, MAX_ITER)
        ax.set_xticks(np.arange(1, MAX_ITER + 1, 2))
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.grid(False)
        for side in ["bottom", "left"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(1.2)
            ax.spines[side].set_color("black")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="lower right", frameon=False, ncol=2, fontsize=14,
                  handlelength=1.6, handletextpad=0.5, labelspacing=0.3,
                  borderaxespad=0.3, columnspacing=0.8)
        path = OUT / f"convergence_models_{ds}_balanced_accuracy.pdf"
        fig.savefig(path, bbox_inches="tight", dpi=600)
        plt.close(fig)
        print(f"saved {path}")


def main():
    curves = collect_curves()
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "balanced_accuracy_curves_all_models.json").write_text(
        json.dumps(curves, indent=2), encoding="utf-8"
    )
    plot(curves)


if __name__ == "__main__":
    main()
