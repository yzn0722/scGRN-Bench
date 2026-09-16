from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np


X = np.arange(5)
XLABELS = ["1k", "5k", "10k", "20k", "30k"]

OBSERVED = {
    "attn": {
        "color": "#4EA3F1",
        "values": [-0.0439733538, 0.0533248487, 0.0891385699, 0.0749321230, 0.0422636298],
    },
    r"COS$_{tok}$": {
        "color": "#FF9A3D",
        "values": [0.1388477806, 0.1414005656, 0.1447418305, 0.1907019138, 0.2180762328],
    },
    r"COS$_{hid}$": {
        "color": "#AC99D2",
        "values": [0.2004276884, 0.2411240527, 0.2099518950, 0.1779605604, 0.1564664486],
    },
}

# Pooled over 200 degree-preserving rewired networks for each of the three
# representations (600 null values per network density).
NULL_MEAN = [-0.0070387382, 0.0019594718, -0.0018345895, -0.0020817666, -0.0022611812]
NULL_Q05 = [-0.0598648645, -0.0364388123, -0.0331181454, -0.0361943746, -0.0326809357]
NULL_Q95 = [0.0445681649, 0.0415135239, 0.0301016365, 0.0320256395, 0.0271108727]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(Path(__file__).resolve().parents[1] / "output" / "pdf"))
    args = parser.parse_args()
    outdir = Path(args.outdir)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.linewidth": 0.8,
        }
    )

    reference_size = (510.503 / 72.0, 335.753 / 72.0)
    fig, ax = plt.subplots(figsize=reference_size)
    q05 = np.asarray(NULL_Q05)
    q95 = np.asarray(NULL_Q95)
    null_mean = np.asarray(NULL_MEAN)

    ax.fill_between(
        X,
        q05,
        q95,
        color="#B8B8B8",
        alpha=0.35,
        linewidth=0,
        label="Pooled rewired 5th-95th percentile",
        zorder=1,
    )
    ax.plot(
        X,
        null_mean,
        "--o",
        color="#777777",
        linewidth=2.4,
        markersize=8,
        label="Pooled rewired mean",
        zorder=2,
    )
    for name, values in OBSERVED.items():
        ax.plot(
            X,
            values["values"],
            "-o",
            color=values["color"],
            linewidth=2.4,
            markersize=8,
            alpha=0.85,
            label=name,
            zorder=3,
        )

    ax.axhline(0, color="#AAAAAA", linewidth=0.9, zorder=0)
    ax.set_xticks(X, XLABELS)
    ax.set_xlabel("Number of top-ranked GRN edges", fontsize=16,
                  fontweight="normal", labelpad=0)
    ax.set_ylabel(r"Transient Spearman $\rho$", fontsize=16, fontweight="normal")
    ax.set_ylim(-0.08, 0.28)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_linewidth(.8)
    ax.spines["bottom"].set_linewidth(.8)
    ax.tick_params(length=0, labelsize=14, colors="#000000")
    ax.grid(False)

    handles, labels = ax.get_legend_handles_labels()
    order = [2, 3, 4, 1, 0]
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.55),
        fontsize=16,
        handlelength=1.6,
        handletextpad=.5,
        columnspacing=.8,
        labelspacing=.3,
        borderaxespad=0,
    )

    fig.subplots_adjust(left=.18, right=.98, bottom=.36, top=.97)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / "grn_representation_combined_to30k_with_pooled_null"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=400)
    plt.close(fig)


if __name__ == "__main__":
    main()
