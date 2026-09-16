from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


KS = [2000, 4000, 6000, 8000, 10000]
COLORS = {"attn": "#4EA3F1", "tok": "#FF9A3D", "hid": "#AC99D2"}
LABELS = {"attn": "attn", "tok": r"COS$_{tok}$", "hid": r"COS$_{hid}$"}


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--attention", required=True)
    p.add_argument("--embedding", required=True)
    p.add_argument("--hidden", required=True)
    p.add_argument("--outdir", required=True)
    return p.parse_args()


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def transient(data, k):
    return data["analyses"][str(k)]["key_to_query_primary"]["early"]["windows"]["transient"]


def main():
    a = args()
    data = {
        "attn": load(a.attention),
        "tok": load(a.embedding),
        "hid": load(a.hidden),
    }
    observed = {name: [transient(d, k)["observed"]["spearman"] for k in KS]
                for name, d in data.items()}
    pooled = []
    for k in KS:
        values = []
        for d in data.values():
            values.extend(transient(d, k)["rewired"]["spearman"]["null_values"])
        pooled.append(np.asarray(values, float))
    null_mean = np.array([v.mean() for v in pooled])
    null_q05 = np.array([np.quantile(v, .05) for v in pooled])
    null_q95 = np.array([np.quantile(v, .95) for v in pooled])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.linewidth": 1.2, "pdf.fonttype": 42,
                         "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6, 6))
    x = np.arange(len(KS))
    ax.fill_between(x, null_q05, null_q95, color="#B8B8B8", alpha=.35,
                    linewidth=0, label="Pooled rewired 5th-95th percentile", zorder=1)
    ax.plot(x, null_mean, "--o", color="#777777", lw=1.8, ms=5,
            label="Pooled rewired mean", zorder=2)
    for name in ("attn", "tok", "hid"):
        ax.plot(x, observed[name], "-o", color=COLORS[name], lw=2.5, ms=8,
                alpha=.9, label=LABELS[name], zorder=3)
    ax.axhline(0, color="#AAAAAA", lw=.9, zorder=0)
    ax.set_xticks(x, ["2k", "4k", "6k", "8k", "10k"])
    ax.set_xlabel("Number of top-ranked GRN edges", fontsize=15)
    ax.set_ylabel(r"Transient Spearman $\rho$", fontsize=15)
    ax.set_ylim(-.08, .28)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0, labelsize=12)
    handles, labels = ax.get_legend_handles_labels()
    order = [2, 3, 4, 1, 0]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              frameon=False, ncol=2, loc="lower center",
              bbox_to_anchor=(.5, 1.02), fontsize=10,
              handlelength=2, columnspacing=1.2)
    fig.subplots_adjust(left=.18, right=.97, bottom=.14, top=.72)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    stem = out / "grn_representation_combined_to10k_with_pooled_null"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
