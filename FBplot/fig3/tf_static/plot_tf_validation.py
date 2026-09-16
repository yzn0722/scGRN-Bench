#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TF 规律三类验证（hESC 聚类结果之后）：

  1. cross_dataset  — hESC vs hHep 重复聚类 + 语义分型占比 + MCM 家族
  2. cross_gt       — C1/C2（及 C3 对照）TF 在 STRING vs CHIP 上的 Jaccard
  3. stats_tests    — C2 vs C3、C1 vs C2：Mann–Whitney + FDR

示例:
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig3
  python tf_static/plot_tf_validation.py --dataset hESC --datasets-compare hESC hHep
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

FIG3 = Path(__file__).resolve().parents[1]
if str(FIG3) not in sys.path:
    sys.path.insert(0, str(FIG3))

from fig3_palette import method_label, model_color  # noqa: E402

from tf_static.collect_per_tf_metrics import assign_hub_strata, collect_per_tf_long  # noqa: E402
from tf_static.model_registry import EVL_ROOT, EXTRACTIONS, GT_DISPLAY, GT_SOURCES, INPUT_ROOT, MODELS  # noqa: E402
from tf_static.plot_tf_benchmark_extended import (  # noqa: E402
    DEFAULT_CLUSTER_K,
    HUB_TIER_HUB,
    annotate_hub_subclass,
    build_tf_feature_matrix,
)
from tf_static.plot_tf_hub_family import HUB_JACCARD_YLIM, apply_plot_style  # noqa: E402
from tf_static.plot_tf_jaccard_raincloud_all_models import EXTRACT_DISPLAY, LABEL_COLOR, TEXT_SIZE, TITLE_SIZE  # noqa: E402

MCM_GENES = tuple(f"MCM{i}" for i in range(1, 11))


def _jac_triplet(row: pd.Series) -> Tuple[float, float, float]:
    e = float(row.get("jac_emb500", 0) or 0)
    a = float(row.get("jac_att500", 0) or 0)
    h = float(row.get("jac_embhidden500", 0) or 0)
    if pd.isna(row.get("jac_emb500")):
        e = 0.0
    if pd.isna(row.get("jac_att500")):
        a = 0.0
    if pd.isna(row.get("jac_embhidden500")):
        h = 0.0
    return e, a, h


def assign_semantic_type(row: pd.Series) -> str:
    """Rule-based failure type (comparable across datasets)."""
    e, a, h = _jac_triplet(row)
    best = max(e, a, h)
    spread = max(e, a, h) - min(e, a, h)
    if best < 0.01:
        return "Type0_all_fail"
    if e > 0.10 and h > 0.10 and a < 0.05:
        return "Type4_dual_emb"
    if e > 0.12 and a < 0.05 and spread >= 0.15:
        return "Type3_emb_dom"
    if e > 0.05 and h < 0.05 and a < 0.05:
        return "Type2_tok_only"
    if best < 0.05:
        return "Type1_weak"
    return "Other_mixed"


def cluster_features(
    df: pd.DataFrame,
    model: str = "scGPT",
    n_clusters: int = DEFAULT_CLUSTER_K,
) -> pd.DataFrame:
    feat = build_tf_feature_matrix(df, model)
    jac_cols = [c for c in feat.columns if c.startswith("jac_")]
    X = feat[jac_cols].fillna(0.0).to_numpy()
    if len(feat) < n_clusters + 1:
        feat["cluster"] = 1
    else:
        Z = linkage(pdist(X, metric="euclidean"), method="ward")
        feat["cluster"] = fcluster(Z, t=n_clusters, criterion="maxclust")
    feat["semantic_type"] = feat.apply(assign_semantic_type, axis=1)
    feat["is_hub"] = feat["hub_tier"] == HUB_TIER_HUB
    feat["emb_minus_att"] = feat["jac_emb500"].fillna(0) - feat["jac_att500"].fillna(0)
    return feat


def summarize_cluster_profile(feat: pd.DataFrame) -> pd.DataFrame:
    """Per cluster: size, mean jaccards (for matching C3-like mass-fail cluster)."""
    jac_cols = ["jac_emb500", "jac_att500", "jac_embhidden500"]
    rows = []
    for cid, sub in feat.groupby("cluster"):
        row = {
            "cluster": int(cid),
            "n_tf": len(sub),
            "pct": 100.0 * len(sub) / len(feat),
            "pct_hub": 100.0 * sub["is_hub"].mean(),
            "mean_best": sub["jac_best"].mean(),
        }
        for c in jac_cols:
            row[f"mean_{c}"] = sub[c].fillna(0).mean()
        rows.append(row)
    prof = pd.DataFrame(rows).sort_values("n_tf", ascending=False)
    prof["rank_by_size"] = np.arange(1, len(prof) + 1)
    # label largest low-jaccard cluster as "C3-like"
    prof["c3_like"] = (prof["mean_jac_emb500"] < 0.02) & (prof["n_tf"] == prof["n_tf"].max())
    return prof


def run_cross_dataset(
    datasets: List[str],
    gt_source: str,
    model: str,
    out_dir: Path,
    evl_root: Path,
    input_root: Path,
) -> pd.DataFrame:
    rows = []
    cluster_tables = []
    for ds in datasets:
        print(f"  cross_dataset: collecting {ds}...")
        df = collect_per_tf_long(ds, gt_source, [model], list(EXTRACTIONS), evl_root, input_root)
        if df.empty:
            continue
        df = assign_hub_strata(df)
        feat = cluster_features(df, model)
        feat["dataset"] = ds
        feat.to_csv(out_dir / f"{ds}_{model}_tf_cluster_validation.csv", index=False)
        prof = summarize_cluster_profile(feat)
        prof["dataset"] = ds
        cluster_tables.append(prof)

        sem = feat["semantic_type"].value_counts()
        for st, n in sem.items():
            rows.append({"dataset": ds, "metric": st, "n_tf": int(n), "pct": 100 * n / len(feat)})

        mcm = feat[feat["TF"].isin(MCM_GENES)]
        for _, r in mcm.iterrows():
            rows.append(
                {
                    "dataset": ds,
                    "metric": "MCM_gene",
                    "TF": r["TF"],
                    "cluster": int(r["cluster"]),
                    "semantic_type": r["semantic_type"],
                    "jac_emb500": r["jac_emb500"],
                    "jac_embhidden500": r["jac_embhidden500"],
                    "jac_att500": r["jac_att500"],
                }
            )

        # largest cluster id and its %
        top_c = prof.iloc[0]
        rows.append(
            {
                "dataset": ds,
                "metric": "largest_cluster",
                "cluster_id": int(top_c["cluster"]),
                "n_tf": int(top_c["n_tf"]),
                "pct": float(top_c["pct"]),
                "mean_jac_emb500": float(top_c["mean_jac_emb500"]),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "validation_cross_dataset_summary.csv", index=False)
    if cluster_tables:
        pd.concat(cluster_tables, ignore_index=True).to_csv(
            out_dir / "validation_cross_dataset_cluster_profiles.csv", index=False
        )

    # Figure: semantic type proportions
    sem_wide = summary[summary["metric"].str.startswith("Type")].copy()
    if not sem_wide.empty:
        pivot = sem_wide.pivot_table(index="metric", columns="dataset", values="pct", aggfunc="first")
        order = [
            "Type0_all_fail",
            "Type1_weak",
            "Type2_tok_only",
            "Type3_emb_dom",
            "Type4_dual_emb",
            "Other_mixed",
        ]
        pivot = pivot.reindex([m for m in order if m in pivot.index])

        fig, ax = plt.subplots(figsize=(max(6, len(datasets) * 2.5), 5))
        x = np.arange(len(pivot.columns))
        w = 0.12
        for i, st in enumerate(pivot.index):
            vals = pivot.loc[st].values
            ax.bar(x + (i - len(pivot) / 2) * w, vals, width=w, label=st.replace("_", " "))
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.columns)
        ax.set_ylabel("% of TFs", fontsize=TEXT_SIZE - 1)
        ax.set_title(
            f"Semantic failure types ({model}, {GT_DISPLAY.get(gt_source, gt_source)})",
            fontsize=TITLE_SIZE - 1,
        )
        ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        stem = out_dir / "validation_cross_dataset_semantic"
        fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
        fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  saved {stem}.png/.pdf")

    return summary


def collect_gt_for_tfs(
    tfs: List[str],
    dataset: str,
    model: str,
    gt_sources: List[str],
    evl_root: Path,
    input_root: Path,
) -> pd.DataFrame:
    rows = []
    for gt in gt_sources:
        try:
            df = collect_per_tf_long(dataset, gt, [model], list(EXTRACTIONS), evl_root, input_root)
        except FileNotFoundError:
            continue
        sub = df[df["TF"].isin(tfs)]
        for _, r in sub.iterrows():
            rows.append(
                {
                    "TF": r["TF"],
                    "gt_source": gt,
                    "extraction": r["extraction"],
                    "jaccard": r["jaccard"],
                }
            )
    return pd.DataFrame(rows)


def run_cross_gt(
    feat_hesc: pd.DataFrame,
    dataset: str,
    model: str,
    out_dir: Path,
    evl_root: Path,
    input_root: Path,
) -> pd.DataFrame:
    groups = {
        "C1_emb_dom": feat_hesc[feat_hesc["cluster"] == 1]["TF"].tolist(),
        "C2_dual_emb": feat_hesc[feat_hesc["cluster"] == 2]["TF"].tolist(),
        "C3_all_fail": feat_hesc[feat_hesc["cluster"] == 3]["TF"].tolist(),
    }
    gt_list = ["STRING", "CHIP"]
    rows = []
    coverage_rows = []
    for gname, tfs in groups.items():
        if not tfs:
            continue
        long = collect_gt_for_tfs(tfs, dataset, model, gt_list, evl_root, input_root)
        if long.empty:
            continue
        for gt in gt_list:
            gt_tfs = set(long[long["gt_source"] == gt]["TF"].unique())
            coverage_rows.append(
                {
                    "cluster_group": gname,
                    "gt_source": gt,
                    "n_tf_list": len(tfs),
                    "n_tf_in_gt": len(set(tfs) & gt_tfs),
                }
            )
        summ = (
            long.groupby(["gt_source", "extraction"], as_index=False)["jaccard"]
            .mean()
            .assign(cluster_group=gname, n_tf=len(tfs))
        )
        rows.append(summ)
    if coverage_rows:
        pd.DataFrame(coverage_rows).to_csv(out_dir / "validation_C1C2C3_gt_coverage.csv", index=False)
        per_tf = (
            long.groupby(["TF", "gt_source"], as_index=False)["jaccard"]
            .mean()
            .assign(cluster_group=gname)
        )
        per_tf.to_csv(out_dir / f"validation_{gname}_per_tf_gt.csv", index=False)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_csv(out_dir / "validation_C1C2C3_cross_gt.csv", index=False)

    if not out.empty:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for ax, gname in zip(axes, ["C1_emb_dom", "C2_dual_emb", "C3_all_fail"]):
            sub = out[out["cluster_group"] == gname]
            if sub.empty:
                ax.axis("off")
                continue
            pivot = sub.pivot_table(index="extraction", columns="gt_source", values="jaccard")
            x = np.arange(len(pivot.index))
            w = 0.35
            for j, gt in enumerate(pivot.columns):
                ax.bar(x + (j - 0.5) * w, pivot[gt].values, width=w, label=GT_DISPLAY.get(gt, gt))
            ax.set_xticks(x)
            ax.set_xticklabels([EXTRACT_DISPLAY.get(e, e) for e in pivot.index], fontsize=9)
            ax.set_ylim(0, min(0.45, HUB_JACCARD_YLIM[1] + 0.15))
            ax.set_title(gname.replace("_", "\n"), fontsize=TEXT_SIZE - 1)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        axes[0].set_ylabel("Mean per-TF Jaccard", fontsize=TEXT_SIZE - 1)
        axes[-1].legend(loc="upper right", fontsize=8, frameon=False)
        fig.suptitle(
            f"{dataset} — C1/C2/C3 TF lists: STRING vs CHIP ({model})",
            fontsize=TITLE_SIZE - 1,
        )
        fig.tight_layout()
        stem = out_dir / "validation_C1C2C3_cross_gt"
        fig.savefig(f"{stem}.png", bbox_inches="tight", facecolor="white")
        fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  saved {stem}.png/.pdf")

    return out


def _mann_whitney(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return {"statistic": np.nan, "pvalue": np.nan, "n_a": len(a), "n_b": len(b)}
    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"statistic": float(res.statistic), "pvalue": float(res.pvalue), "n_a": len(a), "n_b": len(b)}


def _fisher_hub_prop(n_hub_a: int, n_a: int, n_hub_b: int, n_b: int) -> float:
    table = [[n_hub_a, n_a - n_hub_a], [n_hub_b, n_b - n_hub_b]]
    try:
        return float(stats.fisher_exact(table)[1])
    except ValueError:
        return np.nan


def bh_fdr(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        j = m - rank + 1
        val = p[idx] * m / j
        prev = min(prev, val)
        q[idx] = prev
    return q.tolist()


def run_stats_tests(feat: pd.DataFrame, out_dir: Path, dataset: str) -> pd.DataFrame:
    c1 = feat[feat["cluster"] == 1]
    c2 = feat[feat["cluster"] == 2]
    c3 = feat[feat["cluster"] == 3]

    tests = []

    # C2 vs C3
    for var, col in [
        ("jac_best", "jac_best"),
        ("jac_spread", "jac_spread"),
        ("jac_emb500", "jac_emb500"),
        ("jac_embhidden500", "jac_embhidden500"),
        ("emb_minus_att", "emb_minus_att"),
    ]:
        mw = _mann_whitney(c2[col].to_numpy(), c3[col].to_numpy())
        tests.append(
            {
                "comparison": "C2_vs_C3",
                "variable": var,
                "test": "mannwhitney",
                **mw,
                "mean_C2": float(c2[col].mean()),
                "mean_C3": float(c3[col].mean()),
            }
        )
    fh = _fisher_hub_prop(int(c2["is_hub"].sum()), len(c2), int(c3["is_hub"].sum()), len(c3))
    tests.append(
        {
            "comparison": "C2_vs_C3",
            "variable": "hub_fraction",
            "test": "fisher_exact",
            "pvalue": fh,
            "mean_C2": 100 * c2["is_hub"].mean(),
            "mean_C3": 100 * c3["is_hub"].mean(),
            "n_a": len(c2),
            "n_b": len(c3),
        }
    )

    # C1 vs C2
    for var, col in [
        ("jac_spread", "jac_spread"),
        ("emb_minus_att", "emb_minus_att"),
        ("jac_best", "jac_best"),
    ]:
        mw = _mann_whitney(c1[col].to_numpy(), c2[col].to_numpy())
        tests.append(
            {
                "comparison": "C1_vs_C2",
                "variable": var,
                "test": "mannwhitney",
                **mw,
                "mean_C1": float(c1[col].mean()),
                "mean_C2": float(c2[col].mean()),
            }
        )

    tab = pd.DataFrame(tests)
    mask = tab["pvalue"].notna()
    tab.loc[mask, "fdr_bh"] = bh_fdr(tab.loc[mask, "pvalue"].tolist())
    tab.to_csv(out_dir / f"{dataset}_validation_stats_tests.csv", index=False)

    # short text summary
    sig = tab[tab["fdr_bh"] < 0.05] if "fdr_bh" in tab.columns else tab[tab["pvalue"] < 0.05]
    lines = ["Significant tests (FDR<0.05):"]
    for _, r in sig.iterrows():
        lines.append(
            f"  {r['comparison']} {r['variable']}: p={r['pvalue']:.2e} FDR={r.get('fdr_bh', np.nan):.2e}"
        )
    (out_dir / f"{dataset}_validation_stats_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"  stats: {len(sig)} significant at FDR<0.05")
    return tab


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate TF failure-mode patterns")
    p.add_argument("--dataset", default="hESC", help="Primary dataset (C1/C2 lists from here)")
    p.add_argument("--datasets-compare", nargs="+", default=["hESC", "hHep"])
    p.add_argument("--gt-source", default="STRING")
    p.add_argument("--model", default="scGPT")
    p.add_argument("--evl-root", type=Path, default=EVL_ROOT)
    p.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--steps",
        nargs="+",
        default=["cross_dataset", "cross_gt", "stats_tests"],
        choices=["cross_dataset", "cross_gt", "stats_tests", "all"],
    )
    p.add_argument(
        "--cluster-csv",
        type=Path,
        default=None,
        help="Existing hESC cluster CSV; if missing, recompute",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    apply_plot_style()
    steps = ["cross_dataset", "cross_gt", "stats_tests"] if "all" in args.steps else args.steps

    out_dir = args.out or (
        FIG3 / "tf_static" / "output" / args.dataset / f"benchmark_extended_{args.gt_source}" / "validation"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"validation | primary={args.dataset} | steps={steps}")

    cluster_csv = args.cluster_csv or (
        FIG3
        / "tf_static"
        / "output"
        / args.dataset
        / f"benchmark_extended_{args.gt_source}"
        / f"{args.dataset}_{args.model}_tf_cluster_features.csv"
    )

    if cluster_csv.is_file():
        feat_hesc = pd.read_csv(cluster_csv)
        if "emb_minus_att" not in feat_hesc.columns:
            feat_hesc["emb_minus_att"] = feat_hesc["jac_emb500"].fillna(0) - feat_hesc["jac_att500"].fillna(0)
        if "is_hub" not in feat_hesc.columns:
            feat_hesc["is_hub"] = feat_hesc["hub_tier"] == HUB_TIER_HUB
    else:
        df = collect_per_tf_long(
            args.dataset, args.gt_source, [args.model], list(EXTRACTIONS), args.evl_root, args.input_root
        )
        df = assign_hub_strata(df)
        feat_hesc = cluster_features(df, args.model)
        feat_hesc.to_csv(cluster_csv, index=False)

    if "cross_dataset" in steps:
        print("[1/3] cross_dataset")
        run_cross_dataset(
            list(args.datasets_compare),
            args.gt_source,
            args.model,
            out_dir,
            args.evl_root,
            args.input_root,
        )

    if "cross_gt" in steps:
        print("[2/3] cross_gt")
        run_cross_gt(feat_hesc, args.dataset, args.model, out_dir, args.evl_root, args.input_root)

    if "stats_tests" in steps:
        print("[3/3] stats_tests")
        run_stats_tests(feat_hesc, out_dir, args.dataset)

    print(f"Done. → {out_dir}")


if __name__ == "__main__":
    main()
