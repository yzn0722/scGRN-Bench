# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Plot direction accuracy (final iteration) for all benchmark models as a grouped bar chart.

# Reads `accuracy_curves.json` from each model output directory. Missing files or datasets
# show as gaps (NaN) when using --skip-missing.

# Default paths match common locations under benchmark_GRN; override with CLI flags.

# Usage:
#   python plot_accuracy_four_models_bar.py
#   python plot_accuracy_four_models_bar.py --skip-missing --out figure_accuracy_all_models.png
#   python plot_accuracy_four_models_bar.py --color-scheme scheme2
#   python plot_accuracy_four_models_bar.py --color-scheme scheme3
#   python plot_accuracy_four_models_bar.py --color-scheme scheme4
# """

# from __future__ import annotations

# import argparse
# import json
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple, Union

# import matplotlib.pyplot as plt
# import numpy as np

# AXIS_LABEL_SIZE = 14
# TICK_LABEL_SIZE = 12
# AXIS_LINEWIDTH = 1.2


# # 四种配色方案定义
# COLOR_SCHEMES = {
    

#     "scheme4": {  # 
#         "scCello": "#7AC3DF",
#         "scGPT": "#8FB4DC",
        
#         "Geneformer": "#FFDD8E",
#         "LangCell": "#AC99D2",
        
#         "scPRINT": "#EB7E60",
#         "scFoundation": "#F5AA61",
        
#     }
    
# }


# def get_model_color(model_name: str, scheme: str = "scheme1") -> str:
#     """根据配色方案和模型名称返回对应的HEX颜色码"""
#     scheme_colors = COLOR_SCHEMES.get(scheme, COLOR_SCHEMES["scheme1"])
#     return scheme_colors.get(model_name, "#666666")  # 默认灰色


# def load_final_accuracies(path: Union[str, Path]) -> Dict[str, float]:
#     path = Path(path)
#     with open(path, "r") as f:
#         curves: dict = json.load(f)
#     out: Dict[str, float] = {}
#     for ds, arr in curves.items():
#         if not arr:
#             continue
#         out[str(ds)] = float(arr[-1])
#     return out


# def build_default_models(root: Path) -> List[Tuple[str, Path]]:
#     """(display_name, accuracy_curves.json path)."""
#     return [
#         ("Geneformer", root / "pre_geneformer_results_unified/geneformer/accuracy_curves.json"),
#         ("LangCell", root / "pre_langcell_results_unified/langcell/accuracy_curves.json"),
#         ("scGPT", root / "pre_scgpt/results_multidataset_pseudotime_227/accuracy_curves.json"),
#         (
#             "scFoundation",
#             root / "pre_scfoundation/scfoundation_multidataset_pseudotime_227/accuracy_curves.json",
#         ),
#         ("scPRINT", root / "pre_scprint_results_unified/scprint/accuracy_curves.json"),
#         ("scCello", root / "pre_sccello_results_unified/sccello/accuracy_curves.json"),
#     ]


# def parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser(description="Grouped bar chart: final direction accuracy per model")
#     p.add_argument("--root", type=str, default="/mnt/10T/yzn/benchmark_GRN", help="Repo root")
#     p.add_argument(
#         "--out",
#         type=str,
#         default="",
#         help="Output PNG (default: fig4/accuracy/figure_accuracy_all_models_bar.png)",
#     )
#     p.add_argument(
#         "--skip-missing",
#         action="store_true",
#         default=False,
#         help="Skip missing JSON files instead of raising",
#     )
#     p.add_argument("--figwidth", type=float, default=10.0)
#     p.add_argument("--figheight", type=float, default=5.0)
#     p.add_argument("--label-fontsize", type=float, default=4.5)
    
#     # Override arguments for each model
#     for key, _ in [
#         ("geneformer", "pre_geneformer_results_unified/geneformer/accuracy_curves.json"),
#         ("langcell", "pre_langcell_results_unified/langcell/accuracy_curves.json"),
#         ("scgpt", "pre_scgpt/results_multidataset_pseudotime_227/accuracy_curves.json"),
#         (
#             "scf",
#             "pre_scfoundation/scfoundation_multidataset_pseudotime_227/accuracy_curves.json",
#         ),
#         ("scprint", "pre_scprint_results_unified/scprint/accuracy_curves.json"),
#         ("sccello", "pre_sccello_results_unified/sccello/accuracy_curves.json"),
#     ]:
#         p.add_argument(f"--{key}-json", type=str, default="", help=f"Override path for {key}")
    
#     p.add_argument(
#         "--color-scheme",
#         type=str,
#         default="scheme1",
#         choices=["scheme1", "scheme2", "scheme3", "scheme4"],
#         help="Color scheme to use: scheme1 (academic), scheme2 (morandi), scheme3 (retro), scheme4 (blue-red)",
#     )
#     return p.parse_args()


# def main() -> None:
#     args = parse_args()
#     root = Path(args.root)

#     models = build_default_models(root)
#     overrides = {
#         "Geneformer": args.geneformer_json,
#         "LangCell": args.langcell_json,
#         "scGPT": args.scgpt_json,
#         "scFoundation": args.scf_json,
#         "scPRINT": args.scprint_json,
#         "scCello": args.sccello_json,
#     }
#     resolved: List[Tuple[str, Path]] = []
#     for name, default_path in models:
#         ov = (overrides.get(name) or "").strip()
#         resolved.append((name, Path(ov) if ov else default_path))

#     data: Dict[str, Dict[str, float]] = {}
#     methods_order: List[str] = []
#     for name, p in resolved:
#         if not p.is_file():
#             if args.skip_missing:
#                 print(f"[WARN] skip missing: {name} -> {p}")
#                 continue
#             raise FileNotFoundError(f"[Not found] {name}: {p}")
#         data[name] = load_final_accuracies(p)
#         methods_order.append(name)

#     if not methods_order:
#         raise RuntimeError("No model JSON loaded. Check paths or use defaults.")

#     all_datasets: set[str] = set()
#     for d in data.values():
#         all_datasets.update(d.keys())
#     datasets = sorted(all_datasets)
#     if not datasets:
#         raise RuntimeError("No dataset keys found in accuracy_curves.json files.")

#     values = np.full((len(methods_order), len(datasets)), np.nan, dtype=np.float64)
#     for mi, m in enumerate(methods_order):
#         for di, ds in enumerate(datasets):
#             if ds in data[m]:
#                 values[mi, di] = data[m][ds]

#     fig, ax = plt.subplots(figsize=(args.figwidth, args.figheight), dpi=300)
#     x = np.arange(len(datasets))
#     n_methods = len(methods_order)
#     width = min(0.8 / n_methods, 0.14)
#     offsets = (np.arange(n_methods) - (n_methods - 1) / 2) * width

#     for mi, m in enumerate(methods_order):
#         ys = values[mi]
#         bars = ax.bar(
#             x + offsets[mi],
#             ys,
#             width=width,
#             label=m,
#             color=get_model_color(m, args.color_scheme),
#             edgecolor="black",
#             linewidth=0.35,
#         )
#         for b in bars:
#             h = b.get_height()
#             if np.isnan(h):
#                 continue
#             ax.text(
#                 b.get_x() + b.get_width() / 2.0,
#                 h + 0.01,
#                 f"{h:.1%}",
#                 ha="center",
#                 va="bottom",
#                 fontsize=args.label_fontsize,
#             )

#     ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
#     ax.set_ylabel("Direction accuracy (final iteration)", fontsize=AXIS_LABEL_SIZE)
#     ax.set_xticks(x)
#     ax.set_xticklabels(datasets, rotation=35, ha="right", fontsize=TICK_LABEL_SIZE)
#     ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE, width=AXIS_LINEWIDTH)
#     ax.set_ylim(0, 1.05)
#     ncol = 3 if n_methods > 4 else 2
#     ax.legend(
#         frameon=False,
#         ncol=ncol,
#         loc="upper center",
#         bbox_to_anchor=(0.5, 1.18),
#         fontsize=7,
#     )
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.spines["left"].set_linewidth(AXIS_LINEWIDTH)
#     ax.spines["bottom"].set_linewidth(AXIS_LINEWIDTH)
#     plt.tight_layout()

#     script_dir = Path(__file__).resolve().parent
#     default_out = script_dir / "accuracy" / "figure_accuracy_all_models_bar.png"
#     out_png = Path(args.out) if args.out else default_out
#     out_png.parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(out_png, bbox_inches="tight")
#     plt.close(fig)
#     print(f"Saved: {out_png.resolve()}")
#     print(f"Models plotted ({n_methods}): {', '.join(methods_order)}")
#     print(f"Color scheme used: {args.color_scheme}")


# if __name__ == "__main__":
#     main()





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot direction accuracy (final iteration) for all benchmark models as a grouped bar chart.
顶会NPG格式 | 固定配色 | 右侧单列图例 | 专业排版
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from fig4_palette import apply_fig4_style, model_color, FIG4_FIGSIZE

apply_fig4_style()

# ====================== 工具函数 ======================
def load_final_accuracies(path: Union[str, Path]) -> Dict[str, float]:
    """加载accuracy_curves.json，提取每个数据集的最终迭代准确率"""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        curves: dict = json.load(f)
    out: Dict[str, float] = {}
    for ds, arr in curves.items():
        if not arr:
            continue
        out[str(ds)] = float(arr[-1])
    return out


def build_default_models(data_dir: Path) -> List[Tuple[str, Path]]:
    """(模型显示名, accuracy_curves.json路径)，默认读取 fig4/interation 下数据"""
    return [
        ("scCello", data_dir / "sccello_accuracy_curves.json"),
        ("scPRINT", data_dir / "scprint_accuracy_curves.json"),
        ("scGPT", data_dir / "scgpt_accuracy_curves.json"),
        ("Geneformer", data_dir / "geneformer_accuracy_curves_6datasets.json"),
        ("LangCell", data_dir / "Langcell_accuracy_curves.json"),
        ("scFoundation", data_dir / "scfoundation_accuracy_curves.json"),
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grouped bar chart: final direction accuracy per model (NPG format)")
    script_dir = Path(__file__).resolve().parent
    p.add_argument(
        "--data-dir",
        type=str,
        default=str(script_dir / "interation"),
        help="Directory containing *_accuracy_curves*.json files",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Output PDF (default: fig4/accuracy/figure_accuracy_all_models_bar_npg.pdf)",
    )
    p.add_argument(
        "--skip-missing",
        action="store_true",
        default=False,
        help="Skip missing JSON files instead of raising",
    )
    p.add_argument("--figwidth", type=float, default=5.0, help="Figure width (inches)")
    p.add_argument("--figheight", type=float, default=5.0, help="Figure height (inches)")
    
    # 模型路径覆盖参数
    for key, _ in [
        ("geneformer", "geneformer_accuracy_curves_6datasets.json"),
        ("langcell", "Langcell_accuracy_curves.json"),
        ("scgpt", "scgpt_accuracy_curves.json"),
        ("scfoundation", "scfoundation_accuracy_curves.json"),
        ("scprint", "scprint_accuracy_curves.json"),
        ("sccello", "sccello_accuracy_curves.json"),
    ]:
        p.add_argument(f"--{key}-json", type=str, default="", help=f"Override path for {key}")
    
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    # 加载模型路径
    models = build_default_models(data_dir)
    overrides = {
        "Geneformer": args.geneformer_json,
        "LangCell": args.langcell_json,
        "scGPT": args.scgpt_json,
        "scFoundation": args.scfoundation_json,
        "scPRINT": args.scprint_json,
        "scCello": args.sccello_json,
    }
    resolved: List[Tuple[str, Path]] = []
    for name, default_path in models:
        ov = (overrides.get(name) or "").strip()
        resolved.append((name, Path(ov) if ov else default_path))

    # 加载数据
    data: Dict[str, Dict[str, float]] = {}
    methods_order: List[str] = []
    for name, p in resolved:
        if not p.is_file():
            if args.skip_missing:
                print(f"[WARN] skip missing: {name} -> {p}")
                continue
            raise FileNotFoundError(f"[Not found] {name}: {p}")
        data[name] = load_final_accuracies(p)
        methods_order.append(name)

    if not methods_order:
        raise RuntimeError("No model JSON loaded. Check paths or use defaults.")

    # 统一数据集顺序
    all_datasets: set[str] = set()
    for d in data.values():
        all_datasets.update(d.keys())
    datasets = sorted(all_datasets)
    if not datasets:
        raise RuntimeError("No dataset keys found in accuracy_curves.json files.")

    # 构建数值矩阵
    n_methods = len(methods_order)
    n_datasets = len(datasets)
    values = np.full((n_methods, n_datasets), np.nan, dtype=np.float64)
    for mi, m in enumerate(methods_order):
        for di, ds in enumerate(datasets):
            if ds in data[m]:
                values[mi, di] = data[m][ds]

    # ====================== 绘图（NPG顶会格式） ======================
    fig, ax = plt.subplots(figsize=(args.figwidth, args.figheight), dpi=600)
    
    # 柱状图位置计算
    x = np.arange(n_datasets)
    slot_width = min(0.8 / n_methods, 0.13)
    width = slot_width * 0.9  # 柱子之间留一点缝隙
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2) * slot_width

    # 绘制分组柱状图
    for mi, m in enumerate(methods_order):
        ys = values[mi] * 100  # 转百分比（0-100）
        ax.bar(
            x + offsets[mi],
            ys,
            width=width,
            label=m,
            color=model_color(m),
            edgecolor="none",
            linewidth=0.0,
            zorder=3
        )

    # ====================== 图表样式优化（NPG标准） ======================
    # 基准线（50%随机水平）
    ax.axhline(50, color="#888888", linestyle="--", linewidth=1.0, alpha=0.7, zorder=2)
    
    # 坐标轴设置
    ax.set_ylabel("Direction Accuracy (%)", fontsize=16, fontweight="normal")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=0, ha="center", fontsize=14)
    ax.tick_params(axis="x", labelsize=14, length=0)
    ax.tick_params(axis="y", labelsize=14, length=0)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))  # 0,20,40,60,80,100刻度
    
    # 网格线（仅y轴，浅色）
    ax.grid(axis="y", linestyle="-", alpha=0.3, zorder=1)
    
    # 底部单行图例
    ax.legend(
        frameon=False,
        ncol=max(1, n_methods),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        fontsize=14,
        borderaxespad=0
    )
    
    # 隐藏上/右边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    # 调整布局，给底部图例留出空间
    plt.subplots_adjust(bottom=0.24)

    # 保存图片
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir / "accuracy" / "figure_accuracy_all_models_bar_npg.pdf"
    out_png = Path(args.out) if args.out else default_out
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight", dpi=600)
    plt.close(fig)

    # 打印日志
    print(f"✅ Saved: {out_png.resolve()}")
    print(f"📊 Models plotted ({n_methods}): {', '.join(methods_order)}")
    print(f"🎨 Fixed color scheme applied (matches your reference chart)")
    print(f"📐 NPG top-conference format: right-side single-column legend")


if __name__ == "__main__":
    main()