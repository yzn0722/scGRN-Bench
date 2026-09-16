#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compose leakage-free STRING (or CHIP) gene-count sweep panels for 6 datasets.

Reads existing outputs:
  outputs/gene_count_sweep_noleak/{dataset}/{network}/noleak_sweep.csv

Example
-------
  python plot_noleak_six_datasets.py --network string
  python plot_noleak_six_datasets.py --network string --outdir outputs/gene_count_sweep_noleak
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASETS = ["hESC", "hHep", "mDC", "mHSC-E", "mHSC-GM", "mHSC-L"]


def load_sweep(root: Path, dataset: str, network: str) -> Optional[Tuple[pd.DataFrame, float, int]]:
    d = root / dataset / network
    csv_path = d / "noleak_sweep.csv"
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    base = df[df["order"] == "baseline_all"]
    if len(base):
        baseline = float(base["final_acc"].iloc[0])
    else:
        baseline = float("nan")
    meta = d / "noleak_meta.json"
    n_net = None
    if meta.is_file():
        import json

        n_net = int(json.load(open(meta)).get("n_network_nodes", 0)) or None
    if n_net is None:
        # max n among network orders
        sub = df[df["order"].isin(["net_degree", "net_degree_asc", "net_random"])]
        n_net = int(sub["n_update_genes"].max()) if len(sub) else 0
    return df, baseline, n_net


def plot_one_ax(
    ax,
    df: pd.DataFrame,
    dataset: str,
    net_name: str,
    n_net: int,
    baseline_acc: float,
    show_legend: bool = False,
) -> None:
    tag = net_name.upper()

    def _line(order: str, style: str, color: str, label: str, band: bool = False) -> None:
        sub = df[df["order"] == order]
        if sub.empty:
            return
        if band:
            g = sub.groupby("n_update_genes")["final_acc"]
            ns = g.mean().index.to_numpy()
            mu = g.mean().to_numpy() * 100
            lo = g.min().to_numpy() * 100
            hi = g.max().to_numpy() * 100
            ax.fill_between(ns, lo, hi, color=color, alpha=0.18, linewidth=0)
            ax.plot(ns, mu, style, ms=4, lw=1.8, color=color, label=label)
        else:
            sub = sub.sort_values("n_update_genes")
            # one draw per size for degree orders
            sub = sub.drop_duplicates("n_update_genes", keep="last")
            ax.plot(
                sub["n_update_genes"],
                sub["final_acc"] * 100,
                style,
                ms=4,
                lw=1.8,
                color=color,
                label=label,
            )

    _line("net_degree", "o-", "#d95f02", f"{tag} high→low degree")
    _line("net_degree_asc", "D-", "#e7298a", f"{tag} low→high degree")
    _line("net_random", "s-", "#7570b3", f"Random {tag} order", band=True)

    ax.axhline(50, color="0.75", ls="--", lw=1)
    if np.isfinite(baseline_acc):
        ax.axhline(
            baseline_acc * 100,
            color="0.45",
            ls=":",
            lw=1,
            label=f"baseline all ({baseline_acc*100:.0f}%)",
        )
    if n_net > 0:
        ax.axvline(n_net, color="0.6", ls="--", lw=1, alpha=0.8)
        ymin, ymax = ax.get_ylim()
        ax.text(n_net, 47, f" {tag}={n_net}", fontsize=7, color="0.4", va="bottom")

    ax.set_title(dataset, fontweight="600", loc="left", fontsize=11)
    ax.set_xlim(left=90)
    ax.set_ylim(45, 95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_legend:
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")


def main():
    p = argparse.ArgumentParser(description="6-dataset noleak sweep figure")
    p.add_argument("--outdir", default="outputs/gene_count_sweep_noleak")
    p.add_argument("--network", choices=["string", "chip"], default="string")
    p.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset IDs",
    )
    p.add_argument(
        "--out-name",
        default="",
        help="Output stem under outdir (default: noleak_sweep_acc_6datasets_{network})",
    )
    args = p.parse_args()

    root = Path(args.outdir)
    if not root.is_absolute():
        root = SCRIPT_DIR / root
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    loaded = []
    missing = []
    for ds in datasets:
        item = load_sweep(root, ds, args.network)
        if item is None:
            missing.append(ds)
        else:
            loaded.append((ds, *item))

    if missing:
        print(f"[warn] missing {args.network} sweep for: {', '.join(missing)}")
    if not loaded:
        raise SystemExit("No datasets with noleak_sweep.csv found — run sweeps first.")

    n = len(loaded)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 3.6 * nrows),
        dpi=160,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for i, (ds, df, baseline, n_net) in enumerate(loaded):
        plot_one_ax(
            axes[i],
            df,
            ds,
            args.network,
            n_net,
            baseline,
            show_legend=(i == 0),
        )
        if i // ncols == nrows - 1:
            axes[i].set_xlabel("Number of updatable genes")
        if i % ncols == 0:
            axes[i].set_ylabel("Direction accuracy\n(top 30% dynamic, %)")

    for j in range(len(loaded), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Leakage-free gene-count sweep — {args.network.upper()} "
        f"({len(loaded)} datasets)",
        fontsize=13,
        fontweight="700",
        y=1.01,
    )
    fig.tight_layout()

    stem = args.out_name or f"noleak_sweep_acc_6datasets_{args.network}"
    # put combined figure under network folder if hESC exists, else outdir root
    out_dir = root / "string" if args.network == "string" else root / args.network
    # Prefer a shared location next to per-dataset folders
    out_dir = root
    out_png = out_dir / f"{stem}.png"
    out_pdf = out_dir / f"{stem}.pdf"
    # Also copy into string/ for discoverability next to hESC/string
    side = root / "hESC" / args.network
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    if side.is_dir():
        fig.savefig(side / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(side / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Plotted {len(loaded)} datasets → {out_png}")
    if missing:
        print(f"Still need to run: {', '.join(missing)}")


if __name__ == "__main__":
    main()
