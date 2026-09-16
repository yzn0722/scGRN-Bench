import argparse
import glob
import os
from typing import Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from fig4_palette import apply_fig4_style, model_color
from matplotlib.ticker import FuncFormatter

apply_fig4_style()

# 对齐 plot_iter_convergence_single_dataset.py 的样式参数
AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 14
AXIS_LINEWIDTH = 1.2

DEFAULT_BASE_PATH = "/mnt/10T/yzn/benchmark_GRN/pre_scgpt/results_multidataset_pseudotime_227"
percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def parse_exclude_datasets(s: str) -> Set[str]:
    """Comma-separated names to skip. Empty or 'none' => exclude nothing."""
    raw = (s or "").strip()
    if not raw or raw.lower() == "none":
        return set()
    return {p.strip() for p in raw.split(",") if p.strip()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Top-percentage accuracy curves per dataset (scGPT gene results).")
    p.add_argument(
        "--base-path",
        type=str,
        default=DEFAULT_BASE_PATH,
        help="Directory containing *_gene_result.csv",
    )
    p.add_argument(
        "--exclude-datasets",
        type=str,
        default="mDC",
        help="Comma-separated dataset IDs to omit. Default: mDC. Use empty or 'none' to keep all datasets.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_path = args.base_path
    excluded = parse_exclude_datasets(args.exclude_datasets)

    csv_files = sorted(glob.glob(os.path.join(base_path, "*_gene_result.csv")))
    dataset_names = [os.path.basename(f).replace("_gene_result.csv", "") for f in csv_files]
    filtered = [(f, ds) for f, ds in zip(csv_files, dataset_names) if ds not in excluded]
    if not filtered:
        raise SystemExit("No datasets left after --exclude-datasets; check paths and exclusions.")
    csv_files, dataset_names = zip(*filtered)

    # 使用 fig4_palette 颜色
    colors = [
        model_color("scGPT"),
        model_color("Geneformer"),
        model_color("GENIE3"),
        model_color("LangCell"),
        model_color("scCello"),
        model_color("scFoundation"),
        model_color("scPrint"),
        model_color("STRING"),
    ]

    # ===================== 按百分比计算准确率 =====================
    results = {}
    for file, ds_name in zip(csv_files, dataset_names):
        df = pd.read_csv(file)
        total_genes = len(df)

        # 按【真实变化绝对值】排序
        df_sorted = df.assign(abs_true=df["delta_true"].abs()).sort_values("abs_true", ascending=False).reset_index(drop=True)

        accs = []
        for p in percentiles:
            top_n = max(1, int(np.ceil(total_genes * p / 100)))
            subset = df_sorted.head(top_n)
            acc = balanced_accuracy_score(subset["dir_true"], subset["dir_pred"])
            accs.append(acc)
        results[ds_name] = accs

    # ===================== 美化绘图 =====================
    fig, ax = plt.subplots(figsize=(5.0, 5.0))

    for i, ds in enumerate(results.keys()):
        ax.plot(
            percentiles,
            [acc * 100 for acc in results[ds]],
            color=colors[i % len(colors)],
            marker="o",
            linestyle="-",
            linewidth=2.4,
            markersize=8,
            label=ds,
            alpha=0.85,
        )

    ax.set_xlabel("Top percentage of genes", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Balanced Accuracy(%)", fontsize=AXIS_LABEL_SIZE)
    ax.set_xticks([10, 30, 50, 70, 90])
    ax.set_xticklabels(["10%", "30%", "50%", "70%", "90%"], fontsize=14)
    # 未画 mDC 时纵轴聚焦高准确率区间
    if "mDC" not in results:
        ax.set_ylim(50, 100)
        yticks = np.arange(50, 101, 10)
    else:
        ax.set_ylim(20, 100)
        yticks = np.arange(0, 101, 20)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y}" for y in yticks], fontsize=14)

    ax.grid(False)

    for side in ["bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(AXIS_LINEWIDTH)
        ax.spines[side].set_color("black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="lower left",
        frameon=False,
        ncol=2,
        fontsize=14,
        handlelength=1.6,
        handletextpad=0.5,
        labelspacing=0.3,
        borderaxespad=0.3,
        columnspacing=0.8,
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "accuracy")
    os.makedirs(out_dir, exist_ok=True)
    dataset_tag = "_".join(results.keys()) if len(results) > 0 else "no_dataset"
    save_path = os.path.join(out_dir, f"top_percentage_{dataset_tag}_balanced_accuracy_beautiful.pdf")
    plt.savefig(save_path, bbox_inches="tight", dpi=600)
    plt.close()

    print("✅ 美化版绘图完成！")
    print("💾 图片保存至：", save_path)
    if excluded:
        print(f"🚫 Excluded datasets: {', '.join(sorted(excluded))}")
    print("\n📊 各数据集百分比准确率（转换为百分比显示）：")
    for ds, accs in results.items():
        print(f"{ds:12s}: {[round(a * 100, 1) for a in accs]}")


if __name__ == "__main__":
    main()
