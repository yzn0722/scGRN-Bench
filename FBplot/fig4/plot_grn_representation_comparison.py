#!/usr/bin/env python3
"""Compare attention, token-embedding and hidden-embedding propagation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from fig4_palette import apply_fig4_style, FIG4_FIGSIZE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attention", required=True)
    p.add_argument("--embedding", required=True)
    p.add_argument("--hidden", required=True)
    p.add_argument("--outdir", required=True)
    return p.parse_args()


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def block(data, k):
    return data["analyses"][str(k)]["key_to_query_primary"]["early"]["windows"]["transient"]


def box(ax, x, y, text, fc, ec):
    p = FancyBboxPatch((x - .13, y - .08), .26, .16,
                       boxstyle="round,pad=.012,rounding_size=.025",
                       facecolor=fc, edgecolor=ec, linewidth=.9)
    ax.add_patch(p); ax.text(x, y, text, ha="center", va="center", fontsize=6.5)


def main():
    a = parse_args()
    data = {"Attention": load(a.attention), "Embedding": load(a.embedding), "Hidden embedding": load(a.hidden)}
    colors = {"Attention": "#4EA3F1", "Embedding": "#FF9A3D", "Hidden embedding": "#AC99D2"}
    ks = [1000, 5000, 10000]
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.2,
                         "axes.linewidth": .8, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(7.08, 2.48))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.05, 1.18, 1.16, .78], wspace=.57)

    ax = fig.add_subplot(gs[0, 0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(-.10, 1.03, "a", weight="bold", fontsize=10, transform=ax.transAxes)
    ax.text(0, .98, "Network representations", weight="bold", va="top", fontsize=8)
    box(ax, .23, .74, "Attention\n(directed)", "#E7EEF7", colors["Attention"])
    box(ax, .77, .74, "Token embedding\n(symmetric)", "#F6EBDD", colors["Embedding"])
    box(ax, .50, .43, "Hidden embedding\n(symmetric)", "#E4F0EA", colors["Hidden embedding"])
    ax.add_patch(FancyArrowPatch((.50,.33),(.50,.20),arrowstyle="-|>",mutation_scale=8,color="#333",lw=.9))
    ax.text(.50,.12,"Predict next-iteration\nchange magnitude",ha="center",va="center",fontsize=6.5)

    ax = fig.add_subplot(gs[0, 1])
    ax.text(-.19, 1.03, "b", weight="bold", fontsize=10, transform=ax.transAxes)
    x = np.arange(3)
    for name, d in data.items():
        y = [block(d, k)["observed"]["spearman"] for k in ks]
        ax.plot(x, y, "-o", color=colors[name], lw=1.4, ms=4, label=name)
    ax.axhline(0,color="#AAA",lw=.7)
    ax.set_xticks(x,["1k","5k","10k"]); ax.set_xlabel("Strongest network edges")
    ax.set_ylabel("Transient Spearman $\\rho$")
    ax.set_title("Network-density sensitivity",loc="left",weight="bold",fontsize=8,pad=7)
    ax.legend(frameon=False,fontsize=5.8,loc="lower right")
    ax.spines[["top","right"]].set_visible(False)

    ax = fig.add_subplot(gs[0, 2])
    ax.text(-.18, 1.03, "c", weight="bold", fontsize=10, transform=ax.transAxes)
    names = list(data)
    nulls=[]; obs=[]; pvals=[]
    for name in names:
        b=block(data[name],5000); nulls.append(np.asarray(b["rewired"]["spearman"]["null_values"])); obs.append(b["observed"]["spearman"]); pvals.append(b["rewired"]["spearman"]["empirical_p_greater"])
    vp=ax.violinplot(nulls,positions=np.arange(3),widths=.72,showmedians=True,showextrema=False)
    for body in vp["bodies"]: body.set_facecolor("#C2C2C2"); body.set_edgecolor("none"); body.set_alpha(.65)
    vp["cmedians"].set_color("#777"); vp["cmedians"].set_linewidth(1)
    for i,(name,y,p) in enumerate(zip(names,obs,pvals)):
        ax.scatter(i,y,s=27,marker="D",color=colors[name],zorder=3)
        ax.text(i,y+.014,f"$p$={p:.3f}",ha="center",fontsize=6.2)
    ax.axhline(0,color="#AAA",lw=.7)
    ax.set_xticks(np.arange(3),["Attention","Embedding","Hidden\nembedding"],rotation=18,ha="right")
    ax.set_ylabel("Transient Spearman $\\rho$")
    ax.set_title("Top-5k vs rewired null",loc="left",weight="bold",fontsize=8,pad=7)
    ax.spines[["top","right"]].set_visible(False)
    ax.text(.98,.98,"n=200 null networks",transform=ax.transAxes,ha="right",va="top",fontsize=5.7,color="#666")

    ax = fig.add_subplot(gs[0, 3])
    ax.text(-.27, 1.03, "d", weight="bold", fontsize=10, transform=ax.transAxes)
    r2=np.array([block(data[n],5000)["observed"]["ols_r2"] for n in names])*100
    bars=ax.bar(np.arange(3),r2,color=[colors[n] for n in names],width=.68)
    ax.set_xticks(np.arange(3),["Attn","Embed","Hidden"],rotation=30,ha="right")
    ax.set_ylabel("Median OLS $R^2$ (%)")
    ax.set_title("Explained variation\n(top 5k)",loc="left",weight="bold",fontsize=8,pad=7)
    for b,v in zip(bars,r2): ax.text(b.get_x()+b.get_width()/2,v+.35,f"{v:.1f}%",ha="center",fontsize=6.3)
    ax.set_ylim(0,max(r2)*1.25); ax.spines[["top","right"]].set_visible(False); ax.tick_params(axis="x",length=0)

    fig.subplots_adjust(left=.055,right=.99,bottom=.22,top=.87)
    out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True); stem=out/"grn_representation_propagation_comparison"
    fig.savefig(stem.with_suffix(".pdf"),bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"),bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"),dpi=600,bbox_inches="tight")
    plt.close(fig); print(stem)

    # Standalone panel b
    apply_fig4_style()
    display_names = {
        "Attention": "attn",
        "Embedding": r"COS$_{tok}$",
        "Hidden embedding": r"COS$_{hid}$",
    }
    fig_b, ax = plt.subplots(figsize=FIG4_FIGSIZE)
    x = np.arange(3)
    for name, d in data.items():
        y = [block(d, k)["observed"]["spearman"] for k in ks]
        ax.plot(x, y, "-o", color=colors[name], lw=2.4, ms=8,
                label=display_names[name], alpha=0.85)
    ax.axhline(0, color="#AAA", lw=1.0)
    ax.set_xticks(x, ["1k", "5k", "10k"])
    ax.set_xlabel("Strongest network edges", fontsize=16)
    ax.set_ylabel("Transient Spearman $\\rho$", fontsize=16)
    ax.legend(frameon=False, fontsize=14, loc="lower right",
              handlelength=1.6, handletextpad=0.5, labelspacing=0.3,
              borderaxespad=0.3, columnspacing=0.8)
    for side in ["bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.2)
        ax.spines[side].set_color("black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0, labelsize=14)
    ax.grid(False)
    fig_b.tight_layout()
    stem_b = out / "grn_representation_density_sensitivity"
    fig_b.savefig(stem_b.with_suffix(".pdf"), bbox_inches="tight", dpi=600)
    fig_b.savefig(stem_b.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig_b)
    print(stem_b)

    # Standalone panel d — match convergence_models style
    apply_fig4_style()
    fig_d, ax = plt.subplots(figsize=FIG4_FIGSIZE)
    r2 = np.array([block(data[n], 5000)["observed"]["ols_r2"] for n in names]) * 100
    bars = ax.bar(np.arange(3), r2, color=[colors[n] for n in names], width=.68)
    ax.set_xticks(np.arange(3), ["attn", r"COS$_{tok}$", r"COS$_{hid}$"])
    ax.set_ylabel("Median OLS $R^2$ (%)", fontsize=16)
    for b, v in zip(bars, r2):
        ax.text(b.get_x() + b.get_width() / 2, v + max(r2) * 0.03,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=14)
    ax.set_ylim(0, max(r2) * 1.25)
    for side in ["bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.2)
        ax.spines[side].set_color("black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0, labelsize=14)
    ax.grid(False)
    fig_d.tight_layout()
    stem_d = out / "grn_representation_explained_variation"
    fig_d.savefig(stem_d.with_suffix(".pdf"), bbox_inches="tight", dpi=600)
    fig_d.savefig(stem_d.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig_d)
    print(stem_d)


if __name__ == "__main__": main()
