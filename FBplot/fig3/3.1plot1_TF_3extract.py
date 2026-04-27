#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scGPT 三种提取方式 边召回率箱线图（单图 PNG）。

筛选规则与之前保持一致：
1) Pred.Gene1 ∈ STRING.Gene1
2) Pred.Gene2 ∈ (STRING.Gene1 ∪ STRING.Gene2)
3) 按 |EdgeWeight| 降序，保留 top-N，N = STRING 边数
4) 计算边召回率 recall = |Pred ∩ GT| / |GT|
"""

from pathlib import Path
from typing import Set, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

from fig3_palette import method_color, method_label


# -------------------- 配置 --------------------
DATASET = "hESC"
GT_PATH = Path(f"/mnt/10T/yzn/benchmark_GRN/input_process/STRING/{DATASET}_processed-network.csv")
PRED_FILES = {
    "emb500": Path(f"/mnt/10T/yzn/benchmark_GRN/evl_omipath/output_emb500/scgpt/scgpt_{DATASET}.tsv"),
    "embhidden500": Path(f"/mnt/10T/yzn/benchmark_GRN/evl_omipath/output_embhidden500/scgpt/scGPT_{DATASET}.tsv"),
    "att500": Path(f"/mnt/10T/yzn/benchmark_GRN/evl_omipath/output_att500/scgpt/scgpt_{DATASET}.tsv"),
}
OUT_EDGE_BAR_PDF = Path(__file__).resolve().parent / f"scgpt_three_extract_edge_recall_bar_{DATASET}.pdf"
OUT_TF_BOX_PDF = Path(__file__).resolve().parent / f"scgpt_three_extract_tf_recall_box_{DATASET}.pdf"
OUT_DIR = Path(__file__).resolve().parent / "tf_recall"

METHOD_ORDER = ["emb500", "embhidden500", "att500"]
TEXT_SIZE = 16  # align with fig2/0415/2.1-plot_leda.py

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = TEXT_SIZE
plt.rcParams["axes.titlesize"] = TEXT_SIZE
plt.rcParams["axes.labelsize"] = TEXT_SIZE
plt.rcParams["legend.fontsize"] = TEXT_SIZE
plt.rcParams["xtick.labelsize"] = TEXT_SIZE
plt.rcParams["ytick.labelsize"] = TEXT_SIZE


def normalize_edges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Gene1" not in df.columns or "Gene2" not in df.columns:
        raise ValueError("Missing Gene1/Gene2 columns")
    df["Gene1"] = df["Gene1"].astype(str).str.strip().str.upper()
    df["Gene2"] = df["Gene2"].astype(str).str.strip().str.upper()
    df = df[(df["Gene1"] != "") & (df["Gene2"] != "") & (df["Gene1"] != df["Gene2"])]
    return df


def detect_weight_col(df: pd.DataFrame) -> Optional[str]:
    candidates = ["EdgeWeight", "edgeweight", "edge_weight", "Weight", "Score", "Importance", "Attention score"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in ["weight", "score", "import", "att"]):
            return c
    return None


def load_gt(gt_path: Path):
    gt = pd.read_csv(gt_path)
    gt = normalize_edges(gt)
    gt_gene1 = set(gt["Gene1"].unique())
    gt_union = set(gt["Gene1"].unique()).union(set(gt["Gene2"].unique()))
    gt_n = int(len(gt))
    gt_edge_set = set(zip(gt["Gene1"], gt["Gene2"]))
    gt_targets_by_tf = {}
    for tf, tg in zip(gt["Gene1"], gt["Gene2"]):
        gt_targets_by_tf.setdefault(tf, set()).add(tg)
    return gt_gene1, gt_union, gt_n, gt_edge_set, gt_targets_by_tf


def filter_pred(pred_path: Path, gt_gene1: Set[str], gt_union: Set[str], n_keep: Optional[int] = None) -> pd.DataFrame:
    pred = pd.read_csv(pred_path, sep="\t")
    pred = normalize_edges(pred)
    pred = pred[pred["Gene1"].isin(gt_gene1) & pred["Gene2"].isin(gt_union)].copy()
    wcol = detect_weight_col(pred)
    if wcol is not None:
        pred["_w"] = pd.to_numeric(pred[wcol], errors="coerce").fillna(0.0).abs()
        pred = pred.sort_values("_w", ascending=False).drop(columns=["_w"])
    if n_keep is not None:
        pred = pred.head(int(n_keep)).copy()
    return pred


def edge_recall_one_n(pred_top_n: pd.DataFrame, gt_edge_set: Set[tuple], method_name: str) -> pd.DataFrame:
    """只计算一次：N条边（N=|GT|）的全局边召回率。"""
    if pred_top_n.empty or len(gt_edge_set) == 0:
        return pd.DataFrame([{"Method": method_name, "EdgeRecall": 0.0}])
    pred_edges = set(zip(pred_top_n["Gene1"], pred_top_n["Gene2"]))
    hits = len(pred_edges.intersection(gt_edge_set))
    recall = hits / len(gt_edge_set)
    return pd.DataFrame([{"Method": method_name, "EdgeRecall": float(recall)}])


def tf_recall_all(pred_top_n: pd.DataFrame, gt_targets_by_tf: dict, method_name: str) -> pd.DataFrame:
    """使用全部TF（可计算召回的TF）生成召回率分布，用于箱线图（含离群点）。"""
    rows = []
    if pred_top_n.empty:
        return pd.DataFrame(rows)
    for tf, g in pred_top_n.groupby("Gene1"):
        gt_targets = gt_targets_by_tf.get(tf, set())
        if len(gt_targets) == 0:
            continue
        pred_targets = set(g["Gene2"].tolist())
        hits = len(pred_targets.intersection(gt_targets))
        recall = hits / len(gt_targets)
        rows.append({"Method": method_name, "TF": tf, "TFRecall": float(recall)})
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_edge_pdf = OUT_DIR / OUT_EDGE_BAR_PDF.name
    out_tf_pdf = OUT_DIR / OUT_TF_BOX_PDF.name
    gt_gene1, gt_union, gt_n, gt_edge_set, gt_targets_by_tf = load_gt(GT_PATH)

    all_edge_rows = []
    all_tf_rows = []
    for m in METHOD_ORDER:
        fp = PRED_FILES[m]
        # 保持与之前一致：先按规则筛选并按权重排序，再卡到 STRING 边数
        pred_f = filter_pred(fp, gt_gene1, gt_union, n_keep=gt_n)
        all_edge_rows.append(edge_recall_one_n(pred_f, gt_edge_set, m))
        all_tf_rows.append(tf_recall_all(pred_f, gt_targets_by_tf, m))

    edge_df = pd.concat(all_edge_rows, ignore_index=True)
    tf_df = pd.concat(all_tf_rows, ignore_index=True)
    edge_df["Method"] = pd.Categorical(edge_df["Method"], categories=METHOD_ORDER, ordered=True)
    tf_df["Method"] = pd.Categorical(tf_df["Method"], categories=METHOD_ORDER, ordered=True)

    sns.set_theme(style="white")
    # 1) Edge recall bar (single panel)
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    sns.barplot(
        ax=ax1,
        data=edge_df,
        x="Method",
        y="EdgeRecall",
        order=METHOD_ORDER,
        palette=[method_color("emb500"), method_color("embhidden500"), method_color("att500")],
        edgecolor="#555555",
        linewidth=0.8,
    )
    ax1.set_xticklabels([method_label(m) for m in METHOD_ORDER], fontsize=TEXT_SIZE)
    ax1.set_xlabel("")
    ax1.set_ylabel("Edge Recall", fontsize=TEXT_SIZE, fontweight="normal")
    ax1.tick_params(
        axis="both",
        labelsize=TEXT_SIZE,
        colors="#000000",
        direction="out",
        length=6,
        width=1.0,
        bottom=True,
        left=True,
        top=False,
        right=False,
    )
    ax1.yaxis.label.set_color("#000000")
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax1.grid(False)
    ax1.spines["left"].set_color("#000000")
    ax1.spines["bottom"].set_color("#000000")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.xaxis.set_ticks_position("bottom")
    ax1.yaxis.set_ticks_position("left")
    fig1.tight_layout()
    fig1.savefig(out_edge_pdf, bbox_inches="tight", facecolor="white", format="pdf")
    plt.close(fig1)
    print(f"Saved PDF: {out_edge_pdf}")

    # 2) TF recall box (single panel)
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
    sns.boxplot(
        ax=ax2,
        data=tf_df,
        x="Method",
        y="TFRecall",
        order=METHOD_ORDER,
        palette=[method_color("emb500"), method_color("embhidden500"), method_color("att500")],
        linewidth=1.2,
        width=0.7,
        showfliers=True,
        fliersize=3.5,
    )
    ax2.set_xticklabels([method_label(m) for m in METHOD_ORDER], fontsize=TEXT_SIZE)
    ax2.set_xlabel("")
    ax2.set_ylabel("TF Recall (all TFs)", fontsize=TEXT_SIZE, fontweight="normal")
    ax2.tick_params(
        axis="both",
        labelsize=TEXT_SIZE,
        colors="#000000",
        direction="out",
        length=6,
        width=1.0,
        bottom=True,
        left=True,
        top=False,
        right=False,
    )
    ax2.yaxis.label.set_color("#000000")
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax2.grid(False)
    ax2.spines["left"].set_color("#000000")
    ax2.spines["bottom"].set_color("#000000")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.xaxis.set_ticks_position("bottom")
    ax2.yaxis.set_ticks_position("left")
    fig2.tight_layout()
    fig2.savefig(out_tf_pdf, bbox_inches="tight", facecolor="white", format="pdf")
    plt.close(fig2)
    print(f"Saved PDF: {out_tf_pdf}")


if __name__ == "__main__":
    main()