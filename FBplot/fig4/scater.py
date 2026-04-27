# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path
# import glob
# from fig4_palette import apply_fig4_style, model_color

# apply_fig4_style()
# plt.rcParams["figure.facecolor"] = "white"
# plt.rcParams["axes.facecolor"] = "white"
# plt.rcParams["savefig.facecolor"] = "white"
# plt.rcParams["savefig.edgecolor"] = "white"


# BASE_DIR = "/mnt/10T/yzn/benchmark_GRN/pre_scgpt/results_multidataset_pseudotime_0326"
# SCRIPT_DIR = Path(__file__).resolve().parent
# OUTDIR = SCRIPT_DIR / "accuracy"
# OUTDIR.mkdir(parents=True, exist_ok=True)

# csv_files = sorted(glob.glob(os.path.join(BASE_DIR, "*_gene_result.csv")))
# if not csv_files:
#     raise FileNotFoundError(f"No *_gene_result.csv found under {BASE_DIR}")

# for input_csv in csv_files:
#     dataset = os.path.basename(input_csv).replace("_gene_result.csv", "")
#     df = pd.read_csv(input_csv)

#     if "ene" in df.columns and "gene" not in df.columns:
#         df = df.rename(columns={"ene": "gene"})

#     eps = 1e-8
#     df["mean_expr"] = (df["true_early_mean"] + df["true_late_mean"]) / 2.0
#     df["dynamic_strength"] = df["delta_true"].abs()
#     df["abs_error"] = (df["delta_true"] - df["delta_pred"]).abs()
#     df["relative_error"] = df["abs_error"] / (df["dynamic_strength"] + eps)
#     df["recovery_rate"] = df["delta_pred"] / (df["delta_true"] + eps)

#     rel_err_thresh = 0.5
#     df["error_type"] = "well_predicted"
#     df.loc[df["dir_correct"] == 0, "error_type"] = "strict_wrong"
#     df.loc[(df["dir_correct"] == 1) & (df["relative_error"] > rel_err_thresh), "error_type"] = "amplitude_wrong"

#     df["dir_group"] = df["dir_correct"].map({0: "wrong", 1: "correct"})

#     # ---------- plot ----------
#     fig, ax = plt.subplots(figsize=(5, 5), facecolor="white")
#     ax.set_facecolor("white")

#     # keep fig4 palette consistent
#     colors = df["dir_correct"].map({1: model_color("scGPT"), 0: model_color("scPrint")})
#     ax.scatter(df["delta_true"], df["delta_pred"], c=colors, alpha=0.75, s=60, edgecolors="none", linewidths=0)

#     all_vals = np.concatenate([df["delta_true"].values, df["delta_pred"].values])
#     vmin = np.nanmin(all_vals)
#     vmax = np.nanmax(all_vals)
#     pad = 0.05 * (vmax - vmin + eps)

#     ax.plot([vmin - pad, vmax + pad], [vmin - pad, vmax + pad], linestyle="--")
#     ax.axhline(0, linestyle=":")
#     ax.axvline(0, linestyle=":")
#     ax.set_xlabel("delta_true", fontsize=16)
#     ax.set_ylabel("delta_pred", fontsize=16)
#     #ax.set_title(dataset, fontsize=16)
#     ax.tick_params(axis="both", length=0, labelsize=14)

#     plt.tight_layout()
#     out_path = OUTDIR / f"{dataset}_scatter_4.4.pdf"
#     plt.savefig(
#         out_path,
#         dpi=600,
#         facecolor="white",
#         bbox_inches="tight"
#     )
#     plt.close()
#     print(f"Saved: {out_path}")


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from fig4_palette import apply_fig4_style, model_color

apply_fig4_style()
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["savefig.edgecolor"] = "white"

AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 14
AXIS_LINEWIDTH = 1.2


BASE_DIR = "/mnt/10T/yzn/benchmark_GRN/pre_scgpt/results_multidataset_pseudotime_0326"
SCRIPT_DIR = Path(__file__).resolve().parent
OUTDIR = SCRIPT_DIR / "accuracy"
OUTDIR.mkdir(parents=True, exist_ok=True)

csv_files = sorted(glob.glob(os.path.join(BASE_DIR, "*_gene_result.csv")))
if not csv_files:
    raise FileNotFoundError(f"No *_gene_result.csv found under {BASE_DIR}")

for input_csv in csv_files:
    dataset = os.path.basename(input_csv).replace("_gene_result.csv", "")
    df = pd.read_csv(input_csv)

    if "ene" in df.columns and "gene" not in df.columns:
        df = df.rename(columns={"ene": "gene"})

    eps = 1e-8
    df["mean_expr"] = (df["true_early_mean"] + df["true_late_mean"]) / 2.0
    df["dynamic_strength"] = df["delta_true"].abs()
    df["abs_error"] = (df["delta_true"] - df["delta_pred"]).abs()
    df["relative_error"] = df["abs_error"] / (df["dynamic_strength"] + eps)
    df["recovery_rate"] = df["delta_pred"] / (df["delta_true"] + eps)

    rel_err_thresh = 0.5
    df["error_type"] = "well_predicted"
    df.loc[df["dir_correct"] == 0, "error_type"] = "strict_wrong"
    df.loc[(df["dir_correct"] == 1) & (df["relative_error"] > rel_err_thresh), "error_type"] = "amplitude_wrong"

    df["dir_group"] = df["dir_correct"].map({0: "wrong", 1: "correct"})

    # ---------- plot ----------
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="white")
    ax.set_facecolor("white")

    # keep fig4 palette consistent
    colors = df["dir_correct"].map({1: model_color("scGPT"), 0: model_color("scPrint")})
    ax.scatter(df["delta_true"], df["delta_pred"], c=colors, alpha=0.75, s=60, edgecolors="none", linewidths=0)

    all_vals = np.concatenate([df["delta_true"].values, df["delta_pred"].values])
    vmin = np.nanmin(all_vals)
    vmax = np.nanmax(all_vals)
    pad = 0.05 * (vmax - vmin + eps)

    # 绘制对角线 (y=x)
    ax.plot([vmin - pad, vmax + pad], [vmin - pad, vmax + pad], linestyle="--", color='gray', linewidth=1, alpha=0.7)
    
    # 绘制十字形：水平线 (y=0) 和垂直线 (x=0)
    ax.axhline(0, linestyle=":", color='gray', linewidth=0.8, alpha=0.7)
    ax.axvline(0, linestyle=":", color='gray', linewidth=0.8, alpha=0.7)
    
    # 设置坐标轴范围
    ax.set_xlim(vmin - pad, vmax + pad)
    ax.set_ylim(vmin - pad, vmax + pad)
    
    # 使用数学符号Δ (delta)
    ax.set_xlabel(r"$\Delta_{true}$", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(r"$\Delta_{pred}$", fontsize=AXIS_LABEL_SIZE)
    ax.set_title(dataset, fontsize=16, color="#000000", pad=10)
    ax.grid(False)

    # spine/ticks: align with fig4 scatter style
    for side in ["bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", width=AXIS_LINEWIDTH, length=0, labelsize=TICK_LABEL_SIZE, colors="black")
    ax.tick_params(axis="y", width=AXIS_LINEWIDTH, length=0, labelsize=TICK_LABEL_SIZE, colors="black")

    plt.tight_layout()
    out_path = OUTDIR / f"{dataset}_scatter_4.4.pdf"
    plt.savefig(
        out_path,
        dpi=600,
        facecolor="white",
        bbox_inches="tight"
    )
    plt.close()
    print(f"Saved: {out_path}")